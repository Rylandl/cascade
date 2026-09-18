import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import pytest

import cascade
from cascade.env import EpisodeConfig, hover_task, tracking_task
from cascade.reference import aerobatic_reference, tailsitter_reference

ADAPTER = Path(cascade.__file__).resolve().parent / "integrations" / "gymnasium.py"


@pytest.fixture(scope="module", params=[False, True], ids=["without-gymnasium", "with-gymnasium"])
def shim(request):
    installed = request.param
    if installed:
        pytest.importorskip("gymnasium")
    spec = importlib.util.spec_from_file_location(f"cascade_gym_adapter_{installed}", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {} if installed else {"gymnasium": None}):
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(shim):
    return shim.CascadeEnv(aerobatic_reference(), tracking_task(12.0, 50.0, 0.0))


def test_seed_repeats_the_reset_sequence_and_unseeded_resets_advance(env):
    first, _ = env.reset(seed=7)
    second, _ = env.reset()
    repeated_first, _ = env.reset(seed=7)
    repeated_second, _ = env.reset()
    np.testing.assert_array_equal(first, repeated_first)
    np.testing.assert_array_equal(second, repeated_second)
    assert not np.array_equal(first, second)


def test_numpy_step_contract_and_action_validation(env):
    action = np.zeros(env.model.n_propellers + env.model.n_control_channels, dtype=np.float32)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)
    initial, reset_info = env.reset(seed=3)
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation.shape == initial.shape
    assert observation.dtype == initial.dtype == np.float32
    assert isinstance(reset_info, dict) and isinstance(info, dict)
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    for invalid in (action[:-1], action + np.nan, action + 2.0):
        with pytest.raises(ValueError, match="action"):
            env.step(invalid)


def test_time_limit_is_truncation_not_termination(shim):
    env = shim.CascadeEnv(
        aerobatic_reference(),
        tracking_task(12.0, 50.0, 0.0),
        EpisodeConfig(horizon_steps=1),
    )
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
    assert truncated and not terminated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4, dtype=np.float32))


def test_hover_task_uses_its_hover_reference(shim):
    task = hover_task(position_ned=(0.0, 0.0, -50.0))
    env = shim.CascadeEnv(tailsitter_reference(), task, EpisodeConfig(upright_limit_rad=np.pi))
    observation, _ = env.reset(seed=2)
    assert np.all(np.isfinite(observation))
    np.testing.assert_allclose(env.reference.state.rigid_body.velocity, 0.0, atol=1e-6)


def test_float64_simulation_preserves_float32_gym_boundary(shim):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        env = shim.CascadeEnv(aerobatic_reference(), tracking_task(12.0, 50.0, 0.0))
        initial, _ = env.reset(seed=3)
        observation, *_ = env.step(np.zeros(4, dtype=np.float32))
        assert env._state.aircraft.rigid_body.position.dtype == np.float64
        assert initial.dtype == observation.dtype == np.float32
        if shim.gymnasium is not None:
            assert env.observation_space.contains(initial)
            assert env.observation_space.contains(observation)
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.filterwarnings("ignore:.*Box observation space.*:UserWarning")
def test_optional_gymnasium_env_checker(shim, env):
    if shim.gymnasium is None:
        pytest.skip("Gymnasium checker only applies to the installed optional dependency")
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)


def test_mission_and_episode_option_forwarding(shim):
    from cascade.env.faults import fault_schedule
    from cascade.env.missions import scheduled_tracking_task
    from cascade.env.weather import weather_condition

    model = aerobatic_reference()
    task = scheduled_tracking_task([0.0, 0.2], [12.0, 14.0], [50.0, 52.0])
    wind = weather_condition(3.0, 0.0, turbulence_wind_20ft_m_s=0.0)
    fault = fault_schedule(model, motor_out={0: 0.0})
    env = shim.CascadeEnv(model, task, weather=wind)
    initial, _ = env.reset(seed=3, options={"faults": fault})
    initial_rpm = np.asarray(env._state.aircraft.actuators.propeller_speed)
    observation, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert observation.shape == initial.shape
    assert info["time_s"] == pytest.approx(0.025)
    assert np.all(np.asarray(env._state.aircraft.actuators.propeller_speed) < initial_rpm)
    assert np.linalg.norm(np.asarray(env._state.wind_ned)) > 0
    env.reset(seed=3, options={"weather": None})
    assert env._episode_options["faults"] is None
    np.testing.assert_allclose(env._state.wind_ned, env.reference.environment.wind)
    env.reset(seed=3)
    assert env._episode_options["weather"] is wind
    env.close()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4, dtype=np.float32))


@pytest.mark.parametrize("options", [{"unknown": 1}, [1]])
def test_unknown_reset_options_are_rejected(env, options):
    with pytest.raises(ValueError, match="options"):
        env.reset(options=options)


@pytest.mark.parametrize("seed", [-1, True, 0.5])
def test_invalid_seed_is_rejected(env, seed):
    with pytest.raises(ValueError, match="seed"):
        env.reset(seed=seed)


def test_sensor_pipeline_metadata_is_exposed_as_numpy(shim):
    from cascade.env import observation_layout
    from cascade.env.sensors import SensorBlockConfig, SensorPipelineConfig

    model = aerobatic_reference()
    config = EpisodeConfig(
        sensors=SensorPipelineConfig(airspeed=SensorBlockConfig(dropout_probability=1.0))
    )
    env = shim.CascadeEnv(model, tracking_task(12.0, 50.0), config)
    env.reset(seed=9)
    obs, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    airspeed = observation_layout(model).airspeed
    assert isinstance(info["sensor_valid"], np.ndarray)
    assert not np.any(info["sensor_valid"][airspeed])
    assert np.all(info["sensor_dropped"][airspeed])
    np.testing.assert_array_equal(obs[airspeed], 0.0)
