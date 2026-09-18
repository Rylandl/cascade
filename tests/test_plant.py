import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade.analysis import StraightFlightCondition, trim_straight_flight
from cascade.canonical import rigid_body_to_canonical
from cascade.initialization import (
    control_from_array,
    control_to_array,
    equilibrate_internal_state,
    standard_environment,
    zero_state,
)
from cascade.integration import repeat_control, rollout
from cascade.plant import Plant, PlantConfig
from cascade.reference import aerobatic_reference_spec

LEVEL_12_M_S = np.array([0.0, 0.0, 20.0, 12.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"simulation_frequency_hz": 400.0}, "positive integer"),
        ({"control_frequency_hz": 40.0}, "positive integer"),
        ({"simulation_frequency_hz": float("inf")}, "positive integer"),
        ({"control_frequency_hz": float("nan")}, "positive integer"),
        ({"control_frequency_hz": 0}, "positive integer"),
        ({"simulation_frequency_hz": -400}, "positive integer"),
        ({"control_frequency_hz": True}, "positive integer"),
        ({"simulation_frequency_hz": 20, "control_frequency_hz": 40}, "multiple"),
        ({"density_kg_m3": 0.0}, "density"),
        ({"density_kg_m3": float("nan")}, "density"),
        ({"gravity_m_s2": float("inf")}, "gravity"),
        ({"step": None}, "step"),
    ],
)
def test_plant_config_rejects_invalid_host_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PlantConfig(**kwargs)


def test_plant_config_produces_integer_substeps_and_accepts_zero_gravity():
    config = PlantConfig(
        simulation_frequency_hz=np.int64(200),
        control_frequency_hz=np.int64(40),
        gravity_m_s2=0.0,
    )
    assert type(config.simulation_steps_per_control) is int
    assert config.simulation_steps_per_control == 5
    assert config.sample_period_s == pytest.approx(0.025)


def test_plant_reports_shapes_names_and_timing():
    plant = Plant(aerobatic_reference_spec())

    first = plant.reset(LEVEL_12_M_S)
    second = plant.step(np.array([0.5, 0.0, 0.0, 0.0]))

    assert plant.control_names == ("nose_propeller", "aileron", "elevator", "rudder")
    assert first.time_s == 0.0
    assert second.time_s == pytest.approx(plant.sample_period_s)
    assert second.state.shape == (13,)
    assert second.commanded_control.shape == (4,)
    assert second.applied_control.shape == (4,)
    assert second.surface_deflection_rad.shape == (4,)
    assert second.propeller_speed_rad_s.shape == (1,)
    assert np.all(np.isfinite(second.state))
    # Canonical altitude is positive up and the aircraft is still moving north.
    assert second.state[2] > 19.0
    assert second.state[3] > 10.0


def test_plant_rejects_bad_inputs_and_stepping_before_reset():
    plant = Plant(aerobatic_reference_spec())
    with pytest.raises(RuntimeError, match="reset"):
        plant.step(np.zeros(4))
    with pytest.raises(ValueError, match="canonical state"):
        plant.reset(np.zeros(12))
    plant.reset(LEVEL_12_M_S)
    with pytest.raises(ValueError, match="command"):
        plant.step(np.array([0.5, np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError, match="multiple"):
        PlantConfig(simulation_frequency_hz=100, control_frequency_hz=40)


def test_reset_rejects_zero_quaternion_without_mutating_the_plant():
    plant = Plant(aerobatic_reference_spec())
    before = plant.reset(LEVEL_12_M_S, wind_nwu=np.array([1.0, 2.0, 3.0]))
    invalid = LEVEL_12_M_S.copy()
    invalid[6:10] = 0.0
    with pytest.raises(ValueError, match="quaternion must be nonzero"):
        plant.reset(invalid, wind_nwu=np.array([4.0, 5.0, 6.0]))
    after = plant.snapshot()
    np.testing.assert_array_equal(after.state, before.state)
    np.testing.assert_array_equal(after.wind_nwu_m_s, before.wind_nwu_m_s)


@pytest.mark.parametrize("scale", [2.0, 1e-300, 1e300])
def test_reset_normalizes_finite_nonunit_quaternions_before_precision_conversion(scale):
    plant = Plant(aerobatic_reference_spec())
    canonical = LEVEL_12_M_S.copy()
    unit = np.array([0.5, -0.5, 0.5, -0.5])
    canonical[6:10] = scale * unit
    original = canonical.copy()
    sample = plant.reset(canonical)
    np.testing.assert_allclose(sample.state[6:10], unit, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(canonical, original)
    assert np.isfinite(sample.state).all()


@pytest.fixture
def float32_precision():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("field", ["position", "velocity", "rates", "control", "wind"])
def test_reset_rejects_precision_overflow_atomically(float32_precision, field):
    plant = Plant(aerobatic_reference_spec())
    plant.reset(LEVEL_12_M_S, wind_nwu=np.array([1.0, 2.0, 3.0]))
    before = plant.step(np.array([0.3, 0.0, 0.0, 0.0]))
    canonical = LEVEL_12_M_S.copy()
    control = np.zeros(4)
    wind = np.array([4.0, 5.0, 6.0])
    if field in ("position", "velocity", "rates"):
        canonical[{"position": 0, "velocity": 3, "rates": 10}[field]] = 1e100
    elif field == "control":
        control[1] = 1e100
    else:
        wind[0] = 1e100
    with pytest.raises(ValueError, match="working precision"):
        plant.reset(canonical, applied_control=control, wind_nwu=wind)
    after = plant.snapshot()
    np.testing.assert_array_equal(after.state, before.state)
    np.testing.assert_array_equal(after.wind_nwu_m_s, before.wind_nwu_m_s)
    np.testing.assert_array_equal(after.commanded_control, before.commanded_control)
    assert after.time_s == before.time_s


@pytest.mark.parametrize("field", ["command", "wind"])
def test_step_rejects_precision_overflow_atomically(float32_precision, field):
    plant = Plant(aerobatic_reference_spec())
    before = plant.reset(LEVEL_12_M_S, wind_nwu=np.array([1.0, 2.0, 3.0]))
    command, wind = np.zeros(4), np.array([4.0, 5.0, 6.0])
    if field == "command":
        command[1] = 1e100
    else:
        wind[0] = 1e100
    with pytest.raises(ValueError, match="working precision"):
        plant.step(command, wind_nwu=wind)
    after = plant.snapshot()
    np.testing.assert_array_equal(after.state, before.state)
    np.testing.assert_array_equal(after.wind_nwu_m_s, before.wind_nwu_m_s)
    assert after.time_s == before.time_s


@pytest.mark.parametrize("field", ["density_kg_m3", "gravity_m_s2"])
def test_plant_rejects_environment_precision_overflow(float32_precision, field):
    with pytest.raises(ValueError, match="working precision"):
        Plant(aerobatic_reference_spec(), PlantConfig(**{field: 1e100}))


def test_reset_accepts_large_finite_positions_in_float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        plant = Plant(aerobatic_reference_spec())
        canonical = LEVEL_12_M_S.copy()
        canonical[0] = 1e100
        sample = plant.reset(canonical)
        assert sample.state[0] == 1e100
        assert np.isfinite(sample.state).all()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_nonfinite_integration_result_does_not_change_the_plant(float32_precision):
    plant = Plant(aerobatic_reference_spec())
    canonical = LEVEL_12_M_S.copy()
    # Initial velocity and dynamic pressure are representable, but an explicit step under
    # these extreme loads overflows. The host boundary must reject that result atomically.
    canonical[3] = 1e18
    before = plant.reset(canonical)
    with pytest.raises(ValueError, match="step produced nonfinite state"):
        plant.step(np.zeros(4), wind_nwu=np.array([1.0, 0.0, 0.0]))
    after = plant.snapshot()
    np.testing.assert_array_equal(after.state, before.state)
    np.testing.assert_array_equal(after.wind_nwu_m_s, before.wind_nwu_m_s)
    assert after.time_s == before.time_s


def test_nonfinite_reset_equilibrium_does_not_change_the_plant(float32_precision):
    plant = Plant(aerobatic_reference_spec())
    before = plant.reset(LEVEL_12_M_S)
    canonical = LEVEL_12_M_S.copy()
    canonical[3] = 1e20  # Representable velocity whose squared speed overflows float32.
    with pytest.raises(ValueError, match="reset produced nonfinite state"):
        plant.reset(canonical, wind_nwu=np.array([1.0, 0.0, 0.0]))
    after = plant.snapshot()
    np.testing.assert_array_equal(after.state, before.state)
    np.testing.assert_array_equal(after.wind_nwu_m_s, before.wind_nwu_m_s)


def test_reset_equilibrates_actuators_to_the_applied_control():
    plant = Plant(aerobatic_reference_spec())
    applied = np.array([0.5, 0.1, -0.2, 0.05])

    sample = plant.reset(LEVEL_12_M_S, applied_control=applied)

    assert np.allclose(sample.applied_control, applied, atol=1e-5)
    assert np.allclose(sample.commanded_control, applied)


def test_commanded_and_applied_controls_lag_then_converge():
    plant = Plant(aerobatic_reference_spec())
    plant.reset(LEVEL_12_M_S)
    command = np.array([0.5, 0.3, 0.0, 0.0])

    first = plant.step(command)
    assert 0.0 < first.applied_control[1] < 0.3
    assert 0.0 < first.applied_control[0] < 0.5

    for _ in range(30):
        last = plant.step(command)
    assert np.allclose(last.applied_control, command, atol=1e-2)


def test_wind_is_reported_and_changes_the_response():
    calm = Plant(aerobatic_reference_spec())
    windy = Plant(aerobatic_reference_spec())
    calm.reset(LEVEL_12_M_S)
    wind = np.array([3.0, 0.0, 0.0])
    reported = windy.reset(LEVEL_12_M_S, wind_nwu=wind)
    command = np.array([0.5, 0.0, 0.0, 0.0])

    assert np.allclose(reported.wind_nwu_m_s, wind)
    for _ in range(8):
        calm_sample = calm.step(command)
        windy_sample = windy.step(command)
    assert not np.allclose(calm_sample.state, windy_sample.state, atol=1e-3)


def test_plant_step_matches_a_direct_rollout():
    spec = aerobatic_reference_spec()
    config = PlantConfig(simulation_frequency_hz=200, control_frequency_hz=40)
    plant = Plant(spec, config)
    command = np.array([0.6, 0.1, -0.1, 0.0])
    plant.reset(LEVEL_12_M_S)
    sample = plant.step(command)

    model = plant.model
    environment = standard_environment()
    control = control_from_array(model, jnp.asarray(command))
    state = zero_state(model, altitude=20.0, forward_speed=12.0)
    state = equilibrate_internal_state(
        model, state, control_from_array(model, jnp.zeros(4)), environment
    )
    final, _ = rollout(model, state, repeat_control(control, 5), environment, 1.0 / 200.0)

    expected = np.asarray(rigid_body_to_canonical(final.rigid_body))
    assert np.allclose(sample.state, expected, atol=1e-5)


def test_holding_a_trim_command_keeps_the_trim_for_one_second():
    spec = aerobatic_reference_spec()
    plant = Plant(spec)
    condition = StraightFlightCondition(airspeed_m_s=12.0, altitude_m=20.0)
    trim = trim_straight_flight(plant.model, condition)
    assert trim.success, trim.message
    canonical = np.asarray(rigid_body_to_canonical(trim.state.rigid_body))
    command = np.asarray(control_to_array(trim.control))

    plant.reset(canonical, applied_control=command)
    for _ in range(40):
        sample = plant.step(command)

    airspeed = np.linalg.norm(sample.state[3:6])
    assert abs(airspeed - 12.0) < 0.2
    assert abs(sample.state[2] - 20.0) < 0.5
