import jax.numpy as jnp
import numpy as np

from cascade.analysis import StraightFlightCondition, continue_trims, trim_straight_flight
from cascade.dynamics import evaluate_dynamics
from cascade.initialization import standard_environment
from cascade.reference import aerobatic_reference


def test_reference_aircraft_trims_in_conventional_flight():
    model = aerobatic_reference()
    environment = standard_environment()
    result = trim_straight_flight(model, StraightFlightCondition(airspeed_m_s=12.0), environment)

    assert result.success, result.message
    assert result.acceleration_norm < 2e-5
    assert result.angular_acceleration_norm < 2e-5
    assert np.deg2rad(2.0) < result.angle_of_attack_rad < np.deg2rad(8.0)
    assert jnp.all((result.control.propeller >= 0.0) & (result.control.propeller <= 1.0))
    assert jnp.all(jnp.abs(result.control.channel) <= 1.0)

    dynamics = evaluate_dynamics(model, result.state, result.control, environment)
    assert jnp.linalg.norm(dynamics.derivative.actuators.surface_deflection) < 1e-7
    assert jnp.linalg.norm(dynamics.derivative.actuators.propeller_speed) < 1e-4
    assert jnp.linalg.norm(dynamics.derivative.aero.separation) < 1e-6


def test_continuation_tracks_conventional_and_high_alpha_branches():
    model = aerobatic_reference()
    conventional = continue_trims(
        model, (StraightFlightCondition(speed) for speed in (14.0, 12.0, 10.0))
    )
    high_alpha_seed = jnp.array([0.0, np.deg2rad(75.0), 0.7, 0.0, -0.85, 0.0])
    high_alpha = continue_trims(
        model,
        (StraightFlightCondition(speed) for speed in (4.0, 5.0, 6.0)),
        initial_decision=high_alpha_seed,
        residual_tolerance=1e-3,
    )

    assert all(result.success for result in conventional)
    assert all(result.success for result in high_alpha)
    assert [result.angle_of_attack_rad for result in conventional] == sorted(
        result.angle_of_attack_rad for result in conventional
    )
    assert min(result.angle_of_attack_rad for result in high_alpha) > np.deg2rad(50.0)
    assert max(result.angle_of_attack_rad for result in high_alpha) > np.deg2rad(70.0)
