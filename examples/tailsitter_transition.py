"""Hover, transition to cruise, and cruise on the tailsitter fixture under closed-loop control."""

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.control import GuidanceSetpoint
from cascade.math import quaternion_from_euler, quaternion_rotate
from cascade.vtol import (
    initial_transition_state,
    tailsitter_reference_controller,
    transition_rollout,
    velocity_ramp_schedule,
)

DT = 0.005
STEPS = 2000


def hover_start(model, environment):
    state = cascade.zero_state(model, altitude=1.5)
    state = state._replace(
        rigid_body=state.rigid_body._replace(attitude=quaternion_from_euler(0.0, jnp.pi / 2.0, 0.0))
    )
    control = cascade.ControlInput(propeller=jnp.array([0.78, 0.78]), channel=jnp.zeros(2))
    return cascade.equilibrate_internal_state(model, state, control, environment)


def fly(model, environment, controller, state, acceleration_m_s2):
    hover = velocity_ramp_schedule(
        STEPS,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.array(7.0),
        acceleration_m_s2=acceleration_m_s2,
        hold_steps=400,
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.full(STEPS, 7.0),
        altitude_m=jnp.full(STEPS, 1.5),
        heading_rad=jnp.zeros(STEPS),
    )
    return transition_rollout(
        model, controller, state, initial_transition_state(state), hover, forward, environment, DT
    )


def main() -> None:
    spec = cascade.tailsitter_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    controller = tailsitter_reference_controller(spec)
    state = hover_start(model, environment)
    flight = jax.jit(lambda a: fly(model, environment, controller, state, a))
    (_, _), (trajectory, controls, weight) = flight(jnp.asarray(3.5))
    position = np.asarray(trajectory.rigid_body.position)
    speed = np.linalg.norm(np.asarray(trajectory.rigid_body.velocity), axis=1)
    x_body = np.asarray(
        jax.vmap(lambda q: quaternion_rotate(q, jnp.array([1.0, 0.0, 0.0])))(
            trajectory.rigid_body.attitude
        )
    )
    tilt = np.degrees(np.arccos(np.clip(-x_body[:, 2], -1.0, 1.0)))
    print("time   north   alt   speed   tilt  weight  throttle  elevator")
    for second in range(0, 10):
        k = min(int(second / DT), STEPS - 1)
        print(
            f"{second:4d} {position[k, 0]:7.2f} {-position[k, 2]:5.2f} {speed[k]:6.2f} "
            f"{tilt[k]:6.1f} {float(weight[k]):6.2f} {float(controls.propeller[k, 0]):9.2f} "
            f"{float(controls.channel[k, 1]):+9.2f}"
        )

    def final_speed_error(acceleration):
        (final, _), _ = fly(model, environment, controller, state, acceleration)
        return jnp.square(jnp.linalg.norm(final.rigid_body.velocity) - 7.0)

    gradient = jax.jit(jax.grad(final_speed_error))(jnp.asarray(3.5))
    print(f"\nd(final speed error)/d(ramp acceleration) = {float(gradient):+.4f}")


if __name__ == "__main__":
    main()
