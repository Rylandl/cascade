import jax.numpy as jnp
import numpy as np

from cascade.analysis import StraightFlightCondition, aerodynamic_sweep, trim_straight_flight
from cascade.dynamics import evaluate_dynamics
from cascade.initialization import equilibrate_internal_state, standard_environment, zero_state
from cascade.math import quaternion_from_euler
from cascade.reference import tailsitter_reference, tailsitter_reference_spec
from cascade.state import ControlInput

GRAVITY = 9.80665


def hover_state(model, throttle, aileron=0.0, elevator=0.0):
    """Nose-up, motionless state with actuators and separation at their equilibria."""

    state = zero_state(model, altitude=1.5)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            attitude=quaternion_from_euler(0.0, jnp.pi / 2.0, 0.0)
        )
    )
    control = ControlInput(
        propeller=jnp.array([throttle, throttle]), channel=jnp.array([aileron, elevator])
    )
    environment = standard_environment()
    state = equilibrate_internal_state(model, state, control, environment)
    return state, control, environment


def hover_throttle(model):
    """Throttle at which static thrust from both motors equals weight.

    Static thrust per motor is ``rho n^2 D^4 C_T0`` with ``n = throttle * n_max`` in rev/s.
    """

    propellers = model.propellers
    thrust_per_speed_squared = (
        1.225 * float(propellers.thrust_map[0, 1, 0]) * float(propellers.diameter[0]) ** 4
    )
    revolutions = np.sqrt(float(model.mass) * GRAVITY / (2.0 * thrust_per_speed_squared))
    maximum = float(model.actuators.propeller_speed_max[0]) / (2.0 * np.pi)
    return float(revolutions / maximum)


def test_spec_loads_with_two_washed_elevons_and_counter_rotating_motors():
    spec = tailsitter_reference_spec()
    model = spec.to_model()

    assert spec.control_channels == ("aileron", "elevator")
    assert model.n_propellers == 2
    assert model.n_surfaces == 6
    assert float(model.propellers.spin_direction[0]) == -float(model.propellers.spin_direction[1])
    # Each motor washes its own inboard panel strongly, nothing outboard or on the other side.
    assert float(model.propellers.slipstream_map[0, 0]) > 1.0
    assert float(model.propellers.slipstream_map[0, 1]) == 0.0
    assert float(model.propellers.slipstream_map[0, 2]) == 0.0
    assert float(model.propellers.slipstream_map[1, 1]) > 1.0


def test_hover_throttle_is_below_full_and_balances_weight():
    model = tailsitter_reference()
    throttle = hover_throttle(model)
    assert 0.6 < throttle < 0.9

    state, control, environment = hover_state(model, throttle)
    result = evaluate_dynamics(model, state, control, environment)

    thrust = float(jnp.sum(result.propulsion.thrust))
    assert abs(thrust - float(model.mass) * GRAVITY) < 0.03 * float(model.mass) * GRAVITY
    # Nose-up hover: world acceleration is small compared with gravity, and the counter-rotating
    # motors leave no net reaction torque about the thrust axis.
    acceleration = result.derivative.rigid_body.velocity
    assert float(jnp.linalg.norm(acceleration)) < 0.25 * GRAVITY
    assert abs(float(result.propulsion.moment_body[0])) < 1e-6


def test_propwash_gives_elevon_authority_at_zero_airspeed():
    model = tailsitter_reference()
    throttle = hover_throttle(model)
    neutral = evaluate_dynamics(model, *hover_state(model, throttle))
    nose_up = evaluate_dynamics(model, *hover_state(model, throttle, elevator=-0.5))
    right_roll = evaluate_dynamics(model, *hover_state(model, throttle, aileron=0.5))

    # The washed wing halves see real dynamic pressure although the aircraft is not moving.
    assert float(jnp.min(neutral.aerodynamics.air.dynamic_pressure[:2])) > 5.0
    pitch_authority = float(
        nose_up.aerodynamics.moment_body[1] - neutral.aerodynamics.moment_body[1]
    )
    roll_authority = float(
        right_roll.aerodynamics.moment_body[0] - neutral.aerodynamics.moment_body[0]
    )
    # Trailing-edge-up elevons in the propwash pitch the nose up (positive body pitch moment);
    # right-roll aileron gives a positive roll moment. Both must be strong enough to matter:
    # angular accelerations of several rad/s^2 from half deflection.
    assert pitch_authority > 0.0
    assert roll_authority > 0.0
    assert pitch_authority / float(model.inertia[1, 1]) > 3.0
    assert roll_authority / float(model.inertia[0, 0]) > 3.0


def test_differential_thrust_yaws_in_body_axes():
    model = tailsitter_reference()
    throttle = hover_throttle(model)
    state, _, environment = hover_state(model, throttle)
    control = ControlInput(
        propeller=jnp.array([throttle + 0.1, throttle - 0.1]), channel=jnp.zeros(2)
    )
    state = equilibrate_internal_state(model, state, control, environment)

    result = evaluate_dynamics(model, state, control, environment)

    # Left motor stronger: moment r x F about the CG points to +z (nose right).
    assert float(result.propulsion.moment_body[2]) > 0.0


def test_full_envelope_is_finite_and_cruise_trims_below_stall():
    model = tailsitter_reference()
    alpha = jnp.deg2rad(jnp.linspace(-180.0, 180.0, 73))
    sweep = aerodynamic_sweep(model, alpha, airspeed_m_s=7.0)
    assert jnp.all(jnp.isfinite(sweep.force_coefficient_body))
    assert jnp.all(jnp.isfinite(sweep.moment_coefficient_body))

    trim = trim_straight_flight(model, StraightFlightCondition(airspeed_m_s=7.0, altitude_m=1.5))
    assert trim.success, trim.message
    assert 0.0 < trim.angle_of_attack_rad < float(model.surfaces.stall_angle[0])
    assert jnp.all((trim.control.propeller > 0.05) & (trim.control.propeller < 0.95))


def test_transition_corridor_has_a_continuous_thrust_borne_branch():
    model = tailsitter_reference()
    speeds = (0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)
    seed = jnp.array([0.0, np.deg2rad(70.0), 0.0, 0.77, 0.77, 0.0, 0.0])

    thrust_borne = cascade_continue(model, speeds, seed)
    conventional = cascade_continue(model, (9.0, 8.0, 7.0), None)

    assert all(result.success for result in thrust_borne)
    assert all(result.success for result in conventional)
    pitches = [float(result.decision[1]) for result in thrust_borne]
    assert all(np.deg2rad(35.0) < pitch < np.deg2rad(75.0) for pitch in pitches)
    # The two branches coexist at cruise with very different incidence.
    cruise_thrust_borne = thrust_borne[-1]  # 7 m/s
    cruise_conventional = conventional[-1]
    assert cruise_thrust_borne.angle_of_attack_rad > np.deg2rad(25.0)
    assert cruise_conventional.angle_of_attack_rad < np.deg2rad(10.0)
    # Near hover the aircraft hangs on its motors: throttle close to the hover value.
    assert abs(float(thrust_borne[0].control.propeller[0]) - hover_throttle(model)) < 0.05


def cascade_continue(model, speeds, seed):
    from cascade.analysis import continue_trims

    return continue_trims(
        model,
        (StraightFlightCondition(speed, altitude_m=1.5) for speed in speeds),
        initial_decision=seed,
        residual_tolerance=1e-3,
    )
