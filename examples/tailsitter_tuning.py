"""Tune the tailsitter's transition schedule by gradient through the whole round trip.

The round trip (hover, accelerate to cruise, cruise, decelerate, hover) is one differentiable
program, so the schedule's acceleration, deceleration, and cruise tilt can be adjusted by
gradient descent on a cost measured over the entire flight. After compilation each
value-and-gradient of the 16 s flight costs a fraction of a second.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.control import GuidanceSetpoint, tailsitter_reference_controller
from cascade.control.vtol import (
    initial_transition_state,
    speed_profile_schedule,
    transition_rollout,
    trapezoid_speed_profile,
)
from cascade.math import quaternion_from_euler

DT = 0.01
STEPS = 1600
CRUISE = 8.0
LOWER = jnp.array([1.0, 0.5, 0.5])
UPPER = jnp.array([6.0, 4.0, 1.3])
STEP_SCALE = jnp.array([1.0, 1.0, 0.3])


def hover_start(model, environment):
    state = cascade.zero_state(model, altitude=1.5)
    state = state._replace(
        rigid_body=state.rigid_body._replace(attitude=quaternion_from_euler(0.0, jnp.pi / 2.0, 0.0))
    )
    control = cascade.ControlInput(propeller=jnp.array([0.78, 0.78]), channel=jnp.zeros(2))
    return cascade.equilibrate_internal_state(model, state, control, environment)


def round_trip_cost(theta, model, controller, state, environment):
    """Altitude error, final position and speed, and elevon effort over the round trip."""

    acceleration, deceleration, tilt_at_cruise = theta
    speed = trapezoid_speed_profile(
        STEPS,
        DT,
        hold_steps=200,
        cruise_speed_m_s=jnp.asarray(CRUISE),
        acceleration_m_s2=acceleration,
        cruise_steps=300,
        deceleration_m_s2=deceleration,
    )
    hover = speed_profile_schedule(
        speed,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.asarray(CRUISE),
        tilt_at_cruise_rad=tilt_at_cruise,
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.maximum(speed, 6.5),
        altitude_m=jnp.full(STEPS, 1.5),
        heading_rad=jnp.zeros(STEPS),
    )
    (final, _), (trajectory, controls, _) = transition_rollout(
        model, controller, state, initial_transition_state(state), hover, forward, environment, DT
    )
    altitude_error = jnp.mean(jnp.square(-trajectory.rigid_body.position[:, 2] - 1.5))
    final_error = jnp.sum(jnp.square(final.rigid_body.position - hover.position_ned[-1]))
    final_speed = jnp.sum(jnp.square(final.rigid_body.velocity))
    effort = jnp.mean(jnp.square(controls.channel))
    total = altitude_error + final_error + final_speed + 10.0 * effort
    return total, (altitude_error, final_error, final_speed, effort)


def descent_step(theta, gradient, learning_rate=0.05):
    """A bounded, sign-normalised step: the three parameters have different units."""

    step = learning_rate * gradient * STEP_SCALE / (jnp.abs(gradient) * STEP_SCALE + 1e-3)
    return jnp.clip(theta - step, LOWER, UPPER)


def main() -> None:
    spec = cascade.tailsitter_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    controller = tailsitter_reference_controller(spec)
    state = hover_start(model, environment)
    value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda theta: round_trip_cost(theta, model, controller, state, environment),
            has_aux=True,
        )
    )
    theta = jnp.array([3.5, 2.0, 1.0])
    print("iter  accel  decel  tilt |   cost   altitude  final_pos  final_speed  effort | time")
    start = time.time()
    for iteration in range(12):
        (value, terms), gradient = value_and_grad(theta)
        altitude, final_position, final_speed, effort = (float(term) for term in terms)
        accel, decel, tilt = (float(x) for x in theta)
        print(
            f"{iteration:4d} {accel:6.2f} {decel:6.2f} {tilt:5.2f} | "
            f"{float(value):6.3f} {altitude:9.3f} {final_position:10.3f} {final_speed:11.3f} "
            f"{effort:7.4f} | {time.time() - start:4.0f} s"
        )
        theta = descent_step(theta, gradient)
    print(f"\ntuned schedule: {np.round(np.asarray(theta), 3)} (acceleration, deceleration, tilt)")


if __name__ == "__main__":
    main()
