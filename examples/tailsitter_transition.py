"""Hover, transition to cruise, cruise, and transition back to hover on the tailsitter fixture."""

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.control import GuidanceSetpoint
from cascade.math import quaternion_from_euler, quaternion_rotate
from cascade.vtol import (
    initial_transition_state,
    speed_profile_schedule,
    tailsitter_reference_controller,
    transition_rollout,
    trapezoid_speed_profile,
)

DT = 0.005
STEPS = 3200
CRUISE = 8.0


def hover_start(model, environment):
    state = cascade.zero_state(model, altitude=1.5)
    state = state._replace(
        rigid_body=state.rigid_body._replace(attitude=quaternion_from_euler(0.0, jnp.pi / 2.0, 0.0))
    )
    control = cascade.ControlInput(propeller=jnp.array([0.78, 0.78]), channel=jnp.zeros(2))
    return cascade.equilibrate_internal_state(model, state, control, environment)


def fly(model, environment, controller, state, deceleration_m_s2):
    speed = trapezoid_speed_profile(
        STEPS,
        DT,
        hold_steps=400,
        cruise_speed_m_s=jnp.asarray(CRUISE),
        acceleration_m_s2=jnp.asarray(3.5),
        cruise_steps=600,
        deceleration_m_s2=deceleration_m_s2,
    )
    hover = speed_profile_schedule(
        speed,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.asarray(CRUISE),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.maximum(speed, 6.5),
        altitude_m=jnp.full(STEPS, 1.5),
        heading_rad=jnp.zeros(STEPS),
    )
    return speed, transition_rollout(
        model, controller, state, initial_transition_state(state), hover, forward, environment, DT
    )


def main() -> None:
    spec = cascade.tailsitter_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    controller = tailsitter_reference_controller(spec)
    state = hover_start(model, environment)
    flight = jax.jit(lambda d: fly(model, environment, controller, state, d))
    commanded, ((_, _), (trajectory, controls, weight)) = flight(jnp.asarray(2.0))
    position = np.asarray(trajectory.rigid_body.position)
    speed = np.linalg.norm(np.asarray(trajectory.rigid_body.velocity), axis=1)
    x_body = np.asarray(
        jax.vmap(lambda q: quaternion_rotate(q, jnp.array([1.0, 0.0, 0.0])))(
            trajectory.rigid_body.attitude
        )
    )
    tilt = np.degrees(np.arccos(np.clip(-x_body[:, 2], -1.0, 1.0)))
    print("time  command   north   alt   speed   tilt  weight  throttle  elevator")
    for second in range(0, int(STEPS * DT)):
        k = min(int(second / DT), STEPS - 1)
        print(
            f"{second:4d} {float(commanded[k]):8.1f} {position[k, 0]:7.2f} {-position[k, 2]:5.2f} "
            f"{speed[k]:6.2f} {tilt[k]:6.1f} {float(weight[k]):6.2f} "
            f"{float(controls.propeller[k, 0]):9.2f} {float(controls.channel[k, 1]):+9.2f}"
        )

    def final_position_error(deceleration):
        _, ((final, _), _) = fly(model, environment, controller, state, deceleration)
        target = (
            jnp.array([0.0, 0.0, -1.5])
            + jnp.array([1.0, 0.0, 0.0])
            * jnp.sum(
                trapezoid_speed_profile(
                    STEPS,
                    DT,
                    hold_steps=400,
                    cruise_speed_m_s=jnp.asarray(CRUISE),
                    acceleration_m_s2=jnp.asarray(3.5),
                    cruise_steps=600,
                    deceleration_m_s2=deceleration,
                )
            )
            * DT
        )
        return jnp.sum(jnp.square(final.rigid_body.position - target))

    gradient = jax.jit(jax.grad(final_position_error))(jnp.asarray(2.0))
    print(f"\nd(final position error^2)/d(deceleration) = {float(gradient):+.4f}")


if __name__ == "__main__":
    main()
