from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade import aerobatic_reference_spec
from cascade.control import aerobatic_reference_controller
from cascade.env import (
    EpisodeConfig,
    ObservationSpec,
    SensorBlockConfig,
    SensorObservation,
    SensorPipelineConfig,
    control_to_action,
    observation_cascade_policy,
    reset,
    rollout_policy,
    sensor_noise,
    sensor_observation,
    sensor_policy,
    step,
    tracking_task,
)
from cascade.env.baselines import SensorObservation as LegacySensorObservation
from cascade.env.policies import SensorObservation as PublicSensorObservation
from cascade.experiments import Experiment, Policy, Scenario, run_experiment


@pytest.fixture(scope="module")
def flight():
    spec = aerobatic_reference_spec()
    model = spec.to_model()
    task = tracking_task(12.0, 50.0)
    reference = task.reference(model)
    config = EpisodeConfig(
        horizon_steps=6,
        observation=ObservationSpec(**{name: name == "rates" for name in ObservationSpec._fields}),
        reset_position_std_m=0.0,
        reset_velocity_std_m_s=0.0,
        reset_attitude_std_rad=0.0,
        reset_rate_std_rad_s=0.0,
    )
    return spec, model, task, reference, config


def packet_policy(memory, reading):
    """Encode delivered values/age/validity into actions for temporal assertions."""
    assert isinstance(reading, SensorObservation)
    action = jnp.stack(
        (
            0.1 * jnp.tanh(reading.values[0]),
            jnp.where(reading.valid[0], reading.age_s[0], -1.0),
            reading.valid[0].astype(reading.values.dtype),
            jnp.asarray(0.0),
        )
    )
    return action, memory + 1


def assert_tree_close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_type_alias_and_manual_snapshot_match_delivered_observation(flight):
    _, model, task, reference, config = flight
    assert SensorObservation is LegacySensorObservation is PublicSensorObservation
    config = replace(config, observation_delay_steps=2)
    state, obs = reset(
        model, config, task, reference, jax.random.PRNGKey(2), sensor_noise(rate_std=0.1)
    )
    action = control_to_action(config, reference.control)
    for tick in range(4):
        reading = jax.jit(sensor_observation)(state)
        np.testing.assert_array_equal(reading.values, obs)
        np.testing.assert_array_equal(reading.valid, True)
        np.testing.assert_allclose(reading.age_s, min(tick, 2) / config.control_frequency_hz)
        state, obs, _, _, info = step(model, config, task, reference, state, action)
        np.testing.assert_array_equal(sensor_observation(state).age_s, info["sensor_age_s"])


def test_adapter_preserves_supplied_values_and_excludes_hidden_state(flight):
    _, model, task, reference, config = flight
    state, obs = reset(model, config, task, reference, jax.random.PRNGKey(0))
    supplied = obs + 0.7  # deliberately different from the state's observation buffer
    adapted = sensor_policy(packet_policy)
    expected = packet_policy(
        jnp.asarray(0), SensorObservation(supplied, state.sensor_age_s, state.sensor_valid)
    )
    assert_tree_close(jax.jit(adapted)(jnp.asarray(0), supplied, state), expected)
    gradient = jax.jit(jax.grad(lambda values: jnp.sum(adapted(0, values, state)[0])))(supplied)
    np.testing.assert_allclose(
        gradient, [0.1 * (1.0 - np.tanh(float(supplied[0])) ** 2), 0.0, 0.0], atol=1e-7
    )
    poisoned = jax.tree.map(lambda value: jnp.full_like(value, 987), state)._replace(
        sensor_age_s=state.sensor_age_s, sensor_valid=state.sensor_valid
    )
    assert_tree_close(adapted(jnp.asarray(0), supplied, poisoned), expected)
    # The adapter does not even require aircraft, clocks, or an observation buffer.
    public_metadata = SimpleNamespace(
        sensor_age_s=state.sensor_age_s, sensor_valid=state.sensor_valid
    )
    assert_tree_close(adapted(jnp.asarray(0), supplied, public_metadata), expected)


@pytest.mark.parametrize(
    "pipeline,whole_delay,expected_ages,expected_valid",
    [
        (None, 0, [0, 0, 0, 0, 0, 0], [True] * 6),
        (None, 2, [0, 1, 2, 2, 2, 2], [True] * 6),
        (
            SensorPipelineConfig(rates=SensorBlockConfig(delay_steps=1, sample_period_steps=2)),
            1,
            [-1, -1, 2, 3, 2, 3],
            [False, False, True, True, True, True],
        ),
        (
            SensorPipelineConfig(rates=SensorBlockConfig(dropout_probability=1.0)),
            2,
            [-1] * 6,
            [False] * 6,
        ),
    ],
)
def test_rollout_delivers_current_packet_including_all_delays(
    flight, pipeline, whole_delay, expected_ages, expected_valid
):
    _, model, task, reference, config = flight
    config = replace(config, sensors=pipeline, observation_delay_steps=whole_delay)
    noise = sensor_noise(rate_std=0.03)
    state, obs = reset(model, config, task, reference, jax.random.PRNGKey(8), noise)
    initial = sensor_observation(state)
    np.testing.assert_array_equal(initial.values, obs)
    if not expected_valid[0]:
        assert np.all(np.isinf(initial.age_s)) and not np.any(initial.valid)
    _, (observations, actions, _, _) = jax.jit(
        lambda: rollout_policy(
            model,
            config,
            task,
            reference,
            state,
            sensor_policy(packet_policy),
            jnp.asarray(0),
            noise,
        )
    )()
    expected = np.where(np.asarray(expected_ages) < 0, -1, np.asarray(expected_ages) / 40.0)
    np.testing.assert_allclose(actions[:, 1], expected, atol=1e-7)
    np.testing.assert_array_equal(actions[:, 2], expected_valid)
    np.testing.assert_allclose(actions[:, 0], 0.1 * np.tanh(observations[:, 0]), atol=1e-7)


def test_snapshot_and_adapted_rollout_support_jit_vmap(flight):
    _, model, task, reference, config = flight
    config = replace(
        config,
        observation_delay_steps=1,
        sensors=SensorPipelineConfig(
            rates=SensorBlockConfig(sample_period_steps=2, dropout_probability=0.3)
        ),
    )
    noise = sensor_noise(rate_std=0.01)
    keys = jax.random.split(jax.random.PRNGKey(19), 3)
    states, _ = jax.vmap(lambda key: reset(model, config, task, reference, key, noise))(keys)
    assert_tree_close(
        jax.jit(sensor_observation)(states), jax.jit(jax.vmap(sensor_observation))(states)
    )

    def run(state):
        return rollout_policy(
            model,
            config,
            task,
            reference,
            state,
            sensor_policy(packet_policy),
            jnp.asarray(0),
            noise,
        )[1]

    actual = jax.jit(jax.vmap(run))(states)
    singles = [run(jax.tree.map(lambda leaf, i=index: leaf[i], states)) for index in range(3)]
    assert_tree_close(actual, jax.tree.map(lambda *values: jnp.stack(values), *singles))


def test_adapter_enables_missing_sensor_rejection_in_observed_cascade(flight):
    _, model, task, reference, config = flight
    config = replace(
        config,
        observation=ObservationSpec(),
        sensors=SensorPipelineConfig(rates=SensorBlockConfig(dropout_probability=1.0)),
    )
    state, obs = reset(model, config, task, reference, jax.random.PRNGKey(1))
    policy, memory = observation_cascade_policy(
        aerobatic_reference_controller(), model, config, task, reference
    )
    action, held = jax.jit(sensor_policy(policy))(memory, obs, state)
    np.testing.assert_array_equal(action, memory.action)
    assert not bool(held.measurement_valid)
    _, raw = policy(memory, obs)
    assert bool(raw.measurement_valid)  # a raw zero gyro reading cannot identify missing data


def test_experiment_factory_needs_no_custom_runner_for_sensor_policy(flight, tmp_path):
    spec, _, task, _, config = flight
    config = replace(
        config,
        observation_delay_steps=1,
        sensors=SensorPipelineConfig(rates=SensorBlockConfig(delay_steps=1, sample_period_steps=2)),
    )

    def factory(scenario, model, reference):
        return sensor_policy(packet_policy), jnp.asarray(0)

    experiment = Experiment("sensor-policy", (Scenario("timed", spec, task, config, seeds=(17,)),))
    result = run_experiment(
        experiment, [Policy("sensor-aware", factory)], tmp_path / "run", reports=False
    )
    (row,) = result["episodes"]
    assert row["finite"] and row["steps"] == config.horizon_steps
    with np.load(tmp_path / "run" / row["diagnostics"]) as diagnostics:
        np.testing.assert_allclose(
            diagnostics["commanded_action"][:, 1], [-1, -1, 0.05, 0.075, 0.05, 0.075], atol=1e-7
        )


def test_sensor_policy_rejects_noncallable():
    with pytest.raises(TypeError, match="callable"):
        sensor_policy(None)
