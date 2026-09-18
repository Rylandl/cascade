from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade.control import aerobatic_reference_controller, build_gain_schedule
from cascade.env import (
    EpisodeConfig,
    ObservationSpec,
    SensorBlockConfig,
    SensorObservation,
    SensorPipelineConfig,
    cascade_policy,
    observation,
    observation_cascade_policy,
    observation_layout,
    onboard_observation,
    orbit_task,
    reset,
    rollout_policy,
    scheduled_tracking_task,
    sensor_noise,
    tracking_task,
    waypoint_task,
)
from cascade.math import quaternion_from_euler
from cascade.reference import aerobatic_reference


@pytest.fixture(scope="module")
def flight():
    model = aerobatic_reference()
    task = tracking_task(12.0, 50.0, 0.1)
    reference = task.reference(model)
    config = EpisodeConfig(
        horizon_steps=12,
        observation=onboard_observation(),
        reset_position_std_m=0,
        reset_velocity_std_m_s=0,
        reset_attitude_std_rad=0,
        reset_rate_std_rad_s=0,
    )
    state, _ = reset(model, config, task, reference, jax.random.PRNGKey(0))
    rigid = state.aircraft.rigid_body._replace(
        position=jnp.array([23.0, 12.0, -48.0]),
        attitude=quaternion_from_euler(0.12, -0.08, 0.24),
        velocity=jnp.array([11.7, 1.0, -0.1]),
        angular_velocity=jnp.array([0.02, -0.03, 0.04]),
    )
    return (
        model,
        task,
        reference,
        config,
        state._replace(aircraft=state.aircraft._replace(rigid_body=rigid)),
    )


def assert_tree_equal(first, second):
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("spec", [ObservationSpec(), onboard_observation()])
@pytest.mark.parametrize("kind", ["tracking", "scheduled", "waypoint", "orbit"])
def test_ideal_observations_match_privileged_actions_for_missions(flight, spec, kind):
    model, task, reference, config, state = flight
    config = replace(config, observation=spec)
    tasks = {
        "tracking": task,
        "scheduled": scheduled_tracking_task([0, 2], [12, 14], [50, 52], [0.1, 0.3]),
        "waypoint": waypoint_task([0, 10], [[0, 0, -50], [120, 12, -52]], 12.0),
        "orbit": orbit_task([0, -100, -50], 100, 12.0),
    }
    task = tasks[kind]
    state = state._replace(step=jnp.asarray(20), time_s=jnp.asarray(0.5))
    obs = observation(model, task, reference, state, spec)
    controller = aerobatic_reference_controller()
    observed, memory = observation_cascade_policy(controller, model, config, task, reference)
    privileged, old_memory = cascade_policy(controller, model, config, task, reference)
    action, next_memory = jax.jit(observed)(memory._replace(step=state.step), obs, None)
    expected, expected_memory = privileged(old_memory, obs, state)
    np.testing.assert_allclose(action, jnp.clip(expected, -1, 1), atol=2e-6)
    np.testing.assert_allclose(
        next_memory.cascade.rate.integral, expected_memory.rate.integral, atol=2e-7
    )
    assert bool(next_memory.measurement_valid)


def test_hidden_state_cannot_affect_actions_and_jit_vmap_matches(flight):
    model, task, reference, config, state = flight
    policy, memory = observation_cascade_policy(
        aerobatic_reference_controller(), model, config, task, reference
    )
    obs = observation(model, task, reference, state, config.observation)
    expected = policy(memory, obs, None)
    assert_tree_equal(expected, policy(memory, obs, object()))
    poisoned = jax.tree.map(lambda x: jnp.full_like(x, 999), state)
    assert_tree_equal(expected, policy(memory, obs, poisoned))
    inputs = jnp.stack([obs, obs.at[0].multiply(0.9), obs.at[1].add(0.2)])
    memories = jax.tree.map(lambda x: jnp.stack([x, x, x]), memory)
    actions, _ = jax.jit(jax.vmap(lambda m, o: policy(m, o, None)))(memories, inputs)
    for index in range(3):
        np.testing.assert_allclose(actions[index], policy(memory, inputs[index])[0], atol=1e-6)
    assert not np.allclose(actions[0], actions[1])
    assert not np.allclose(actions[0], actions[2])


@pytest.mark.parametrize(
    "block,value",
    [
        ("airspeed", 0),
        ("gravity", 0),
        ("heading", 0),
        ("rates", np.nan),
        ("airspeed", 1e38),
        ("airspeed", 1e20),
        ("position_error", 1e38),
    ],
)
def test_invalid_measurement_holds_action_and_integrators_but_advances_clock(flight, block, value):
    model, task, reference, config, state = flight
    policy, memory = observation_cascade_policy(
        aerobatic_reference_controller(), model, config, task, reference
    )
    obs = observation(model, task, reference, state, config.observation)
    _, memory = policy(memory, obs)
    bad = obs.at[getattr(observation_layout(model, config.observation), block)].set(value)
    action, held = jax.jit(policy)(memory, bad)
    assert_tree_equal(held.cascade, memory.cascade)
    np.testing.assert_array_equal(action, memory.action)
    assert int(held.step) == int(memory.step) + 1
    assert not bool(held.measurement_valid)
    _, recovered = policy(held, obs)
    assert bool(recovered.measurement_valid)


def test_explicit_sensor_ages_validity_and_unused_blocks(flight):
    model, task, reference, config, state = flight
    policy, memory = observation_cascade_policy(
        aerobatic_reference_controller(), model, config, task, reference, max_sensor_age_s=0.1
    )
    obs = observation(model, task, reference, state, config.observation)
    reading = SensorObservation(obs, jnp.zeros_like(obs), jnp.ones(obs.shape, bool))
    expected = policy(memory, obs)
    assert_tree_equal(expected, policy(memory, reading))
    readings = jax.tree.map(lambda x: jnp.stack([x, x]), reading)
    actions, _ = jax.jit(jax.vmap(lambda value: policy(memory, value)))(readings)
    np.testing.assert_allclose(actions, jnp.stack([expected[0], expected[0]]), atol=1e-6)
    layout = observation_layout(model, config.observation)
    for ages, valid in [
        (reading.age_s.at[layout.rates].set(0.11), reading.valid),
        (reading.age_s, reading.valid.at[layout.rates].set(False)),
        (reading.age_s.at[layout.rates].set(-0.01), reading.valid),
        (reading.age_s.at[layout.rates].set(jnp.inf), reading.valid),
    ]:
        action, held = jax.jit(policy)(memory, SensorObservation(obs, ages, valid))
        np.testing.assert_array_equal(action, memory.action)
        assert not bool(held.measurement_valid)
    # Unused accelerometer readings cannot influence a rate/attitude controller.
    unused = reading._replace(
        values=obs.at[layout.specific_force].set(jnp.nan),
        age_s=reading.age_s.at[layout.specific_force].set(jnp.inf),
        valid=reading.valid.at[layout.specific_force].set(False),
    )
    assert_tree_equal(expected, policy(memory, unused))


def test_acquisition_time_restores_delayed_airspeed_normalization(flight):
    model, _, reference, config, state = flight
    task = scheduled_tracking_task([0, 2], [12, 20], 50.0, 0.1)
    state = state._replace(step=jnp.asarray(40), time_s=jnp.asarray(1.0))
    policy, memory = observation_cascade_policy(
        aerobatic_reference_controller(), model, config, task, reference
    )
    memory = memory._replace(step=state.step)
    current = observation(model, task, reference, state, config.observation)
    layout = observation_layout(model, config.observation)
    delayed = current.at[layout.airspeed].multiply(16.0 / 14.4)  # acquired at t=0.6s
    metadata = SensorObservation(
        delayed, jnp.zeros_like(current).at[layout.airspeed].set(0.4), jnp.ones(current.shape, bool)
    )
    corrected = jax.jit(policy)(memory, metadata)[0]
    expected = policy(memory, current)[0]
    np.testing.assert_allclose(corrected, expected, atol=1e-6)
    assert not np.allclose(policy(memory, delayed)[0], expected)


def test_gain_schedule_uses_measured_speed_and_actions_stay_bounded(flight):
    model, task, reference, config, state = flight
    config = replace(config, channel_scale=0.1)
    low = aerobatic_reference_controller()
    high = low._replace(rate=low.rate._replace(kp=low.rate.kp * 1.5))
    schedule = build_gain_schedule([10, 20], [low, high])
    observed, memory = observation_cascade_policy(schedule, model, config, task, reference)
    privileged, privileged_memory = cascade_policy(schedule, model, config, task, reference)
    obs = observation(model, task, reference, state, config.observation)
    action, _ = observed(memory, obs)
    expected, _ = privileged(privileged_memory, obs, state)
    np.testing.assert_allclose(action, jnp.clip(expected, -1, 1), atol=2e-6)
    assert np.max(np.abs(action)) <= 1.0


def test_sensor_pipeline_changes_closed_loop_actions(flight):
    model, task, reference, config, _ = flight
    controller = aerobatic_reference_controller()
    noisy = sensor_noise(rate_std=0.01, heading_std=0.01)
    sensed = replace(
        config,
        sensors=SensorPipelineConfig(
            heading=SensorBlockConfig(sample_period_steps=3, delay_steps=1),
            rates=SensorBlockConfig(dropout_probability=0.1),
        ),
    )

    def run(settings, noise):
        state, _ = reset(model, settings, task, reference, jax.random.PRNGKey(4), noise)
        policy, memory = observation_cascade_policy(controller, model, settings, task, reference)
        return jax.jit(
            lambda: rollout_policy(model, settings, task, reference, state, policy, memory, noise)
        )()[1]

    clean, affected = run(config, None), run(sensed, noisy)
    assert not np.allclose(clean[1], affected[1])
    assert np.all(np.isfinite(affected[1]))
    assert np.max(np.abs(affected[1])) <= 1
    assert not np.any(affected[3][:-1])
    assert bool(affected[3][-1])  # the configured horizon terminates the episode


def test_host_validation_and_metadata_shape_errors(flight):
    model, task, reference, config, state = flight
    controller = aerobatic_reference_controller()
    for maximum in (0, -1, np.inf, True):
        with pytest.raises(ValueError, match="max_sensor_age_s"):
            observation_cascade_policy(
                controller, model, config, task, reference, max_sensor_age_s=maximum
            )
    with pytest.raises(ValueError, match="requires observation blocks"):
        observation_cascade_policy(
            controller,
            model,
            replace(config, observation=ObservationSpec(gravity=False)),
            task,
            reference,
        )
    policy, memory = observation_cascade_policy(controller, model, config, task, reference)
    obs = observation(model, task, reference, state, config.observation)
    with pytest.raises(ValueError, match="floating vector"):
        policy(memory, obs[:-1])
    with pytest.raises(ValueError, match="match observation shape"):
        policy(memory, SensorObservation(obs, jnp.zeros(1), jnp.ones(obs.shape, bool)))
    with pytest.raises(ValueError, match="validity must be boolean"):
        policy(memory, SensorObservation(obs, jnp.zeros_like(obs), jnp.ones_like(obs)))
