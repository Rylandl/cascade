from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from cascade.analysis import StraightFlightCondition, trim_straight_flight
from cascade.dynamics import evaluate_dynamics
from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.reference import skywalker_x8, skywalker_x8_spec
from cascade.spec import LongitudinalCoefficientSpec
from cascade.state import ActuatorState

# Gryte et al. 2018, Table V, wind-tunnel column.
CL0, CLA, CLDE = 0.0867, 4.02, 0.278
CD0, CDA1, CDA2, CDDE, CDB2, CDB1 = 0.0197, 0.0791, 1.06, 0.0633, 0.148, -0.00584
CM0, CMA, CMDE = 0.0302, -0.126, -0.206
FLIGHT_TUNED_PITCH = LongitudinalCoefficientSpec(
    zero=0.02275, alpha_rad=-0.4629, q=-1.3, elevator_rad=-0.2292
)


def coefficients_at(model, alpha, beta=0.0, elevator=0.0, aileron=0.0, speed=18.0):
    """Body-block coefficients [C_L, C_D, C_Y, C_l, C_m, C_n] at a still-air flow condition."""

    state = zero_state(model)
    velocity = speed * jnp.array(
        [jnp.cos(alpha) * jnp.cos(beta), jnp.sin(beta), jnp.sin(alpha) * jnp.cos(beta)]
    )
    # Elevons: left = aileron + elevator, right = elevator - aileron.
    state = state._replace(
        rigid_body=state.rigid_body._replace(velocity=velocity),
        actuators=ActuatorState(
            surface_deflection=jnp.array([aileron + elevator, elevator - aileron]),
            propeller_speed=state.actuators.propeller_speed,
        ),
    )
    result = evaluate_dynamics(model, state, zero_control(model), standard_environment())
    return result.aerodynamics.body.coefficients


def test_x8_spec_loads_and_describes_a_flying_wing():
    spec = skywalker_x8_spec()
    model = spec.to_model()

    assert spec.name == "skywalker-x8"
    assert spec.control_channels == ("aileron", "elevator")
    assert model.n_surfaces == 2
    assert model.n_propellers == 1
    assert float(model.reference_span) == pytest.approx(2.1)
    assert jnp.all(model.surfaces.area == 0.0)


def test_x8_reproduces_the_published_polynomials_with_the_stall_blend_disabled():
    model = skywalker_x8()
    model = model._replace(body=model.body._replace(stall_angle=jnp.asarray(10.0)))
    for alpha in np.deg2rad([-10.0, -3.0, 0.0, 4.0, 8.0, 12.0]):
        for elevator in (0.0, 0.1):
            lift, drag, _, _, pitch, _ = coefficients_at(model, alpha, elevator=elevator)
            assert jnp.allclose(lift, CL0 + CLA * alpha + CLDE * elevator, atol=2e-5)
            assert jnp.allclose(
                drag, CD0 + CDA1 * alpha + CDA2 * alpha**2 + CDDE * elevator**2, atol=2e-5
            )
            assert jnp.allclose(pitch, CM0 + CMA * alpha + CMDE * elevator, atol=2e-5)


def test_x8_stays_within_one_percent_of_the_polynomial_in_the_linear_range_and_is_finite_beyond():
    model = skywalker_x8()
    for alpha in np.deg2rad([0.0, 4.0, 8.0]):
        lift, drag, _, _, _, _ = coefficients_at(model, alpha)
        assert jnp.allclose(lift, CL0 + CLA * alpha, rtol=1e-2)
    for alpha in np.deg2rad([30.0, 90.0, 150.0, 180.0, -120.0]):
        values = coefficients_at(model, alpha)
        assert jnp.all(jnp.isfinite(values))
        assert jnp.abs(values[0]) < 1.5
    assert jnp.abs(coefficients_at(model, jnp.pi / 2.0)[0]) < 0.05


def test_x8_control_derivatives_have_the_published_signs():
    model = skywalker_x8()
    alpha = np.deg2rad(3.0)
    plus_aileron = coefficients_at(model, alpha, aileron=0.1)
    minus_aileron = coefficients_at(model, alpha, aileron=-0.1)
    plus_elevator = coefficients_at(model, alpha, elevator=0.1)
    minus_elevator = coefficients_at(model, alpha, elevator=-0.1)

    assert plus_aileron[3] > minus_aileron[3]  # C_l delta_a > 0
    assert plus_elevator[4] < minus_elevator[4]  # C_m delta_e < 0
    assert plus_elevator[0] > minus_elevator[0]  # C_L delta_e > 0
    slipped = coefficients_at(model, alpha, beta=0.1)
    assert slipped[2] < 0.0  # C_Y beta < 0
    assert slipped[3] < 0.0  # C_l beta < 0
    assert slipped[5] > 0.0  # C_n beta > 0


def test_x8_trims_level_at_cruise_for_both_pitch_variants():
    spec = skywalker_x8_spec()
    tuned = replace(spec, body=replace(spec.body, pitch=FLIGHT_TUNED_PITCH))
    condition = StraightFlightCondition(airspeed_m_s=18.0, altitude_m=100.0)

    for variant in (spec, tuned):
        result = trim_straight_flight(variant.to_model(), condition)
        assert result.success, result.message
        assert np.deg2rad(-2.0) < result.angle_of_attack_rad < np.deg2rad(6.0)
        assert 0.2 < float(result.control.propeller[0]) < 0.8
        assert abs(float(result.control.channel[1])) < 0.3
