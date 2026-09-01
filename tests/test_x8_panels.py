import jax.numpy as jnp
import numpy as np
import pytest

from cascade.analysis import StraightFlightCondition, aerodynamic_sweep, trim_straight_flight
from cascade.reference import skywalker_x8, skywalker_x8_panels, skywalker_x8_panels_spec
from cascade.state import ControlInput

# Coarse validation grid; scripts/fit_x8_panels.py fits against a finer one and prints the full
# residual table. Observed max abs errors here are C_L ~0.028, C_m ~0.015; tolerances below
# leave headroom above what is actually achieved.
ALPHA_DEG = np.array([-6.0, -2.0, 0.0, 2.0, 6.0, 9.0])
BETA_DEG = np.array([-6.0, 0.0, 6.0])
CL_TOLERANCE = 0.05
CM_TOLERANCE = 0.02


def _wind_axis_lift(force_coefficient_body: jnp.ndarray, alpha_rad: jnp.ndarray) -> jnp.ndarray:
    axial, normal = force_coefficient_body[..., 0], force_coefficient_body[..., 2]
    return axial * jnp.sin(alpha_rad) - normal * jnp.cos(alpha_rad)


def _coarse_grid_sweeps():
    alpha = jnp.deg2rad(jnp.asarray(ALPHA_DEG))
    beta = jnp.deg2rad(jnp.asarray(BETA_DEG))
    mesh = jnp.meshgrid(alpha, beta, indexing="ij")
    grid_alpha, grid_beta = (value.reshape(-1) for value in mesh)
    size = grid_alpha.shape[0]
    control = ControlInput(propeller=jnp.zeros((size, 1)), channel=jnp.zeros((size, 2)))
    panel_sweep = aerodynamic_sweep(
        skywalker_x8_panels(),
        grid_alpha,
        airspeed_m_s=18.0,
        sideslip_rad=grid_beta,
        control=control,
    )
    target_sweep = aerodynamic_sweep(
        skywalker_x8(), grid_alpha, airspeed_m_s=18.0, sideslip_rad=grid_beta, control=control
    )
    return grid_alpha, panel_sweep, target_sweep


def test_x8_panels_spec_loads_and_validates():
    spec = skywalker_x8_panels_spec()
    model = spec.to_model()

    assert spec.name == "skywalker-x8-panels"
    assert spec.body is None
    assert spec.control_channels == ("aileron", "elevator")
    assert model.n_surfaces == 7
    areas = np.asarray(model.surfaces.area)
    assert np.all(areas > 0.0)
    assert areas.sum() == pytest.approx(0.75, abs=0.01)


def test_x8_panels_static_coefficients_match_the_published_body_block():
    alpha, panel_sweep, target_sweep = _coarse_grid_sweeps()

    panel_lift = _wind_axis_lift(panel_sweep.force_coefficient_body, alpha)
    target_lift = _wind_axis_lift(target_sweep.force_coefficient_body, alpha)
    assert float(jnp.max(jnp.abs(panel_lift - target_lift))) < CL_TOLERANCE

    panel_pitch = panel_sweep.moment_coefficient_body[..., 1]
    target_pitch = target_sweep.moment_coefficient_body[..., 1]
    assert float(jnp.max(jnp.abs(panel_pitch - target_pitch))) < CM_TOLERANCE


def test_x8_panels_trims_level_at_cruise():
    condition = StraightFlightCondition(airspeed_m_s=18.0, altitude_m=100.0)
    result = trim_straight_flight(skywalker_x8_panels(), condition)

    assert result.success, result.message
    assert np.deg2rad(-2.0) < result.angle_of_attack_rad < np.deg2rad(8.0)
    assert 0.2 < float(result.control.propeller[0]) < 0.8


def test_x8_panels_roll_moment_is_monotonic_in_aileron():
    model = skywalker_x8_panels()
    alpha = jnp.deg2rad(jnp.array(3.0))
    ailerons = jnp.linspace(-0.3, 0.3, 7)
    rolls = []
    for aileron in ailerons:
        control = ControlInput(propeller=jnp.zeros((1,)), channel=jnp.array([aileron, 0.0]))
        sweep = aerodynamic_sweep(model, alpha, airspeed_m_s=18.0, control=control)
        rolls.append(float(sweep.moment_coefficient_body[0]))

    assert all(later > earlier for earlier, later in zip(rolls, rolls[1:], strict=False))


def test_x8_panels_are_finite_at_extreme_angles():
    model = skywalker_x8_panels()
    for alpha_deg in (90.0, -90.0, 180.0):
        sweep = aerodynamic_sweep(model, jnp.deg2rad(jnp.array(alpha_deg)), airspeed_m_s=18.0)
        assert jnp.all(jnp.isfinite(sweep.force_coefficient_body))
        assert jnp.all(jnp.isfinite(sweep.moment_coefficient_body))
