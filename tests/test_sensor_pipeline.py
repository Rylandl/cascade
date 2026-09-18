from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.env.episode import EpisodeConfig, current_environment, reset, step
from cascade.env.missions import scheduled_tracking_task
from cascade.env.sensors import (
    ObservationSpec,
    SensorBlockConfig,
    SensorPipelineConfig,
    block_sizes,
    initialize_sensor_pipeline,
    sensor_noise,
    step_sensor_pipeline,
)
from cascade.env.tasks import ReferenceFlight, task_at, tracking_task
from cascade.env.weather import weather_condition


def selected(*names):
    return ObservationSpec(**{name: name in names for name in ObservationSpec._fields})


@pytest.fixture(scope="module")
def model():
    return cascade.aerobatic_reference()


def scan_readings(model, pipeline, spec, key, count=20, dt=0.1):
    size = sum(size for name, size in block_sizes(model).items() if getattr(spec, name))
    keys = jax.random.split(key, count + 1)
    initial = initialize_sensor_pipeline(model, pipeline, spec, jnp.zeros(size), keys[0])

    def advance(state, inputs):
        index, key = inputs
        state = step_sensor_pipeline(
            model, pipeline, spec, state, jnp.full((size,), index.astype(float)), key, dt
        )
        return state, state

    _, history = jax.lax.scan(advance, initial, (jnp.arange(1, count + 1), keys[1:]))
    return initial, history


def test_block_sampling_holds_last_value_and_keeps_elements_synchronous(model):
    spec = selected("airspeed", "rates")
    pipeline = SensorPipelineConfig(rates=SensorBlockConfig(sample_period_steps=3))
    _, history = jax.jit(lambda key: scan_readings(model, pipeline, spec, key, count=7))(
        jax.random.PRNGKey(1)
    )
    np.testing.assert_array_equal(history.reading[:, 0], np.arange(1, 8))
    expected = np.array([0, 0, 3, 3, 3, 6, 6])
    np.testing.assert_array_equal(
        history.reading[:, 1:], np.broadcast_to(expected[:, None], (7, 3))
    )
    np.testing.assert_array_equal(history.sample_step[:, 1], expected)
    np.testing.assert_array_equal(
        history.sampled[:, 1], [False, False, True, False, False, True, False]
    )


def test_transport_delay_has_no_valid_sample_until_delivery(model):
    spec = selected("rates")
    pipeline = SensorPipelineConfig(rates=SensorBlockConfig(delay_steps=2))
    initial, history = scan_readings(model, pipeline, spec, jax.random.PRNGKey(2), count=5)
    assert not np.any(initial.valid)
    np.testing.assert_array_equal(history.valid[:, 0], [False, True, True, True, True])
    np.testing.assert_array_equal(history.sample_step[:, 0], [-1, 0, 1, 2, 3])
    np.testing.assert_array_equal(history.reading[:, 0], [0, 0, 1, 2, 3])


def test_dropout_discards_acquisitions_and_bias_walk_continues(model):
    spec = selected("rates")
    pipeline = SensorPipelineConfig(
        rates=SensorBlockConfig(sample_period_steps=4, dropout_probability=1.0, bias_walk_std=0.3)
    )
    _, history = scan_readings(model, pipeline, spec, jax.random.PRNGKey(8), count=8)
    assert not np.any(history.valid)
    assert np.all(history.reading == 0.0)
    np.testing.assert_array_equal(history.sampled, history.dropped)
    # Diffusion proceeds even through three unsampled ticks and every dropped acquisition.
    assert np.all(np.linalg.norm(np.diff(history.bias_walk, axis=0), axis=1) > 0.0)


def test_jitter_dropout_are_reproducible_vmappable_and_never_reorder_readings(model):
    spec = selected("rates")
    pipeline = SensorPipelineConfig(
        rates=SensorBlockConfig(
            sample_period_steps=2,
            sample_jitter_steps=1,
            delay_steps=3,
            delay_jitter_steps=3,
            dropout_probability=0.3,
        )
    )
    run = jax.jit(jax.vmap(lambda key: scan_readings(model, pipeline, spec, key, count=40)[1]))
    keys = jax.random.split(jax.random.PRNGKey(9), 8)
    first, repeated = run(keys), run(keys)
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(repeated), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first.sampled[0], first.sampled[1])
    assert np.any(first.dropped) and np.any(first.sampled & ~first.dropped)
    assert np.all(np.diff(first.sample_step, axis=1) >= 0)
    # Synthetic readings identify acquisition time, so stale/delayed packets are detectable.
    np.testing.assert_array_equal(
        np.asarray(first.reading)[first.valid], np.asarray(first.sample_step)[first.valid]
    )
    for world in range(8):
        ticks = np.flatnonzero(first.sampled[world, :, 0]) + 1
        gaps = np.diff(np.concatenate(([0], ticks)))
        assert np.all((gaps >= 1) & (gaps <= 3))


def test_bias_random_walk_has_physical_time_scaling(model):
    spec = selected("rates")
    pipeline = SensorPipelineConfig(rates=SensorBlockConfig(bias_walk_std=0.2))
    keys = jax.random.split(jax.random.PRNGKey(33), 512)
    run = jax.jit(
        jax.vmap(lambda key: scan_readings(model, pipeline, spec, key, count=10, dt=0.1)[1])
    )
    history = run(keys)
    final_bias = np.asarray(history.bias_walk[:, -1])
    assert abs(float(final_bias.mean())) < 0.02
    # Ten 0.1-second increments have variance sigma^2 * one second.
    assert float(final_bias.var()) == pytest.approx(0.2**2, rel=0.15)


@pytest.fixture
def episode(model):
    config = EpisodeConfig(
        simulation_frequency_hz=40.0,
        control_frequency_hz=40.0,
        horizon_steps=10,
        reset_position_std_m=0.0,
        reset_velocity_std_m_s=0.0,
        reset_attitude_std_rad=0.0,
        reset_rate_std_rad_s=0.0,
    )
    reference = ReferenceFlight(
        cascade.zero_state(model, altitude=50.0, forward_speed=12.0),
        cascade.zero_control(model),
        cascade.standard_environment(),
    )
    return config, tracking_task(12.0, 50.0), reference


def test_zero_feature_pipeline_matches_legacy_noise_and_delay(model, episode):
    config, task, reference = episode
    config = replace(config, observation_delay_steps=2)
    enabled = replace(config, sensors=SensorPipelineConfig())
    noise = sensor_noise(rate_std=0.01, rate_bias_std=0.02, air_std=0.01)
    key = jax.random.PRNGKey(1)
    plain, plain_obs = reset(model, config, task, reference, key, noise)
    richer, richer_obs = reset(model, enabled, task, reference, key, noise)
    np.testing.assert_array_equal(plain_obs, richer_obs)
    action = jnp.zeros(model.n_propellers + model.n_control_channels)
    for _ in range(5):
        plain, plain_obs, plain_reward, plain_done, _ = step(
            model, config, task, reference, plain, action, noise
        )
        richer, richer_obs, rich_reward, rich_done, _ = step(
            model, enabled, task, reference, richer, action, noise
        )
        np.testing.assert_array_equal(plain_obs, richer_obs)
        np.testing.assert_array_equal(plain.key, richer.key)
        np.testing.assert_array_equal(plain_reward, rich_reward)
        np.testing.assert_array_equal(plain_done, rich_done)
        np.testing.assert_array_equal(plain.sensor_age_s, richer.sensor_age_s)


def test_episode_age_and_validity_include_block_and_whole_observation_delay(model, episode):
    config, task, reference = episode
    config = replace(
        config,
        observation=selected("rates"),
        observation_delay_steps=2,
        sensors=SensorPipelineConfig(rates=SensorBlockConfig(delay_steps=2)),
    )
    initial, obs = reset(model, config, task, reference, jax.random.PRNGKey(2))
    assert not np.any(initial.sensor_valid) and np.all(np.isinf(initial.sensor_age_s))
    np.testing.assert_array_equal(obs, 0.0)
    action = jnp.zeros(model.n_propellers + model.n_control_channels)

    def run(state):
        def advance(state, _):
            state, reading, _, _, info = step(model, config, task, reference, state, action)
            return state, (reading, info["sensor_age_s"], info["sensor_valid"])

        return jax.lax.scan(advance, state, None, length=6)

    final, (_, age, valid) = jax.jit(run)(initial)
    np.testing.assert_array_equal(valid[:, 0], [False, False, False, True, True, True])
    np.testing.assert_allclose(age[3:], 4 / config.control_frequency_hz, atol=1e-7)
    assert np.all(np.asarray(age)[np.asarray(valid)] >= 0.0)
    assert float(final.time_s) == pytest.approx(6 / config.control_frequency_hz)
    np.testing.assert_array_equal(final.sensor_age_s, age[-1])


def test_mission_reward_uses_returned_time_and_environment_despite_sensor_delay(model, episode):
    config, _, reference = episode
    config = replace(
        config,
        observation=selected("airspeed"),
        sensors=SensorPipelineConfig(airspeed=SensorBlockConfig(delay_steps=1)),
    )
    task = scheduled_tracking_task([0.0, 1.0], [12.0, 15.0], [50.0, 52.0])
    weather = weather_condition(
        gust_amplitude_m_s=4.0,
        gust_duration_s=2 / config.control_frequency_hz,
        gust_direction_ned=(1.0, 0.0, 0.0),
    )
    initial, _ = reset(model, config, task, reference, jax.random.PRNGKey(31), weather=weather)
    action = jnp.zeros(model.n_propellers + model.n_control_channels)
    advanced, _, reward, _, info = jax.jit(
        lambda state: step(model, config, task, reference, state, action, weather=weather)
    )(initial)
    assert float(jnp.linalg.norm(advanced.wind_ned - initial.wind_ned)) > 3.0
    target = task_at(task, advanced.time_s, advanced.aircraft.rigid_body)
    expected = target.cost(
        advanced.aircraft.rigid_body, current_environment(reference, advanced), action
    )
    np.testing.assert_allclose(info["cost"], expected, atol=1e-6)
    np.testing.assert_allclose(reward, jnp.exp(-expected), atol=1e-6)
    np.testing.assert_allclose(info["sensor_age_s"], 1 / config.control_frequency_hz)
    assert np.all(info["sensor_valid"]) and np.all(info["sensor_fresh"])


def test_episode_sensor_state_supports_jit_vmap(model, episode):
    config, task, reference = episode
    config = replace(
        config,
        sensors=SensorPipelineConfig(
            rates=SensorBlockConfig(
                sample_period_steps=2, bias_walk_std=0.01, delay_steps=1, dropout_probability=0.2
            )
        ),
    )
    action = jnp.zeros(model.n_propellers + model.n_control_channels)

    def one_world(key):
        state, _ = reset(model, config, task, reference, key)
        state, reading, _, _, info = step(model, config, task, reference, state, action)
        return state.sensor_state, reading, info["sensor_age_s"], info["sensor_valid"]

    keys = jax.random.split(jax.random.PRNGKey(19), 3)
    batched = jax.jit(jax.vmap(one_world))(keys)
    scalar = [one_world(key) for key in keys]
    expected = jax.tree.map(lambda *values: jnp.stack(values), *scalar)
    for actual, wanted in zip(jax.tree.leaves(batched), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_period_steps": 0},
        {"sample_period_steps": True},
        {"delay_steps": 1.5},
        {"sample_jitter_steps": 1},
        {"delay_jitter_steps": 1},
        {"delay_steps": -1},
        {"dropout_probability": -0.1},
        {"dropout_probability": 1.1},
        {"dropout_probability": float("nan")},
        {"bias_walk_std": -0.1},
        {"bias_walk_std": float("inf")},
    ],
)
def test_sensor_block_host_validation(kwargs):
    with pytest.raises(ValueError):
        SensorBlockConfig(**kwargs)


def test_pipeline_and_existing_noise_host_validation():
    with pytest.raises(ValueError, match="SensorBlockConfig"):
        SensorPipelineConfig(rates=None)
    with pytest.raises(ValueError, match="SensorPipelineConfig"):
        EpisodeConfig(sensors={})
    for value in (-1.0, float("nan"), float("inf"), [0.1, 0.2]):
        with pytest.raises(ValueError, match="nonnegative scalar"):
            sensor_noise(rate_std=value)
