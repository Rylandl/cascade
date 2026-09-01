"""Batched, differentiable high-angle-of-attack rollout smoke example."""

import jax
import jax.numpy as jnp

from cascade import (
    ControlInput,
    aerobatic_reference,
    equilibrate_internal_state,
    repeat_control,
    rollout,
    standard_environment,
    zero_state,
)

MODEL = aerobatic_reference()
ENVIRONMENT = standard_environment()


def final_altitude(elevator: jax.Array) -> jax.Array:
    angle_of_attack = 35.0 * jnp.pi / 180.0
    speed = 12.0
    state = zero_state(MODEL, altitude=20.0)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            velocity=jnp.array(
                [speed * jnp.cos(angle_of_attack), 0.0, speed * jnp.sin(angle_of_attack)]
            )
        )
    )
    control = ControlInput(propeller=jnp.array([0.65]), channel=jnp.stack((0.0, elevator, 0.0)))
    state = equilibrate_internal_state(MODEL, state, control, ENVIRONMENT)
    final, _ = rollout(MODEL, state, repeat_control(control, 100), ENVIRONMENT, 0.01)
    return -final.rigid_body.position[2]


if __name__ == "__main__":
    elevator = jnp.array(0.1)
    altitude = jax.jit(final_altitude)(elevator)
    sensitivity = jax.jit(jax.grad(final_altitude))(elevator)
    print(f"altitude after 1 s: {altitude:.3f} m")
    print(f"d(altitude)/d(elevator): {sensitivity:.3f} m / normalized command")
