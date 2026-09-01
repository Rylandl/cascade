import jax
import jax.numpy as jnp

from cascade.initialization import (
    equilibrate_internal_state,
    standard_environment,
    zero_control,
    zero_state,
)
from cascade.integration import repeat_control, rk4_step, rollout
from cascade.math import matvec, quaternion_rotate
from cascade.reference import aerobatic_reference
from cascade.state import ControlInput, Environment


def test_rk4_preserves_unit_quaternion():
    model = aerobatic_reference()
    state = zero_state(model, altitude=20.0, forward_speed=12.0)
    state = state._replace(
        rigid_body=state.rigid_body._replace(angular_velocity=jnp.array([4.0, -2.0, 3.0]))
    )
    next_state = rk4_step(model, state, zero_control(model), standard_environment(), 0.01)
    assert jnp.allclose(jnp.linalg.norm(next_state.rigid_body.attitude), 1.0, atol=1e-6)


def test_rollout_is_jittable_and_time_major():
    model = aerobatic_reference()
    state = zero_state(model, batch_shape=(4,), altitude=20.0, forward_speed=12.0)
    control = zero_control(model, batch_shape=(4,))
    controls = repeat_control(control, steps=8)
    final, trajectory = jax.jit(rollout, static_argnames=("step",))(
        model, state, controls, standard_environment(batch_shape=(4,)), 0.01
    )

    assert final.rigid_body.position.shape == (4, 3)
    assert trajectory.rigid_body.position.shape == (8, 4, 3)
    assert jnp.all(jnp.isfinite(trajectory.rigid_body.position))


def test_gradient_through_high_alpha_rollout_is_finite_and_nonzero():
    model = aerobatic_reference()
    alpha = 35.0 * jnp.pi / 180.0
    state = zero_state(model, altitude=20.0)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            velocity=jnp.array([12.0 * jnp.cos(alpha), 0.0, 12.0 * jnp.sin(alpha)])
        )
    )
    environment = standard_environment()

    def final_pitch_rate(elevator):
        control = ControlInput(propeller=jnp.array([0.65]), channel=jnp.stack((0.0, elevator, 0.0)))
        final, _ = rollout(model, state, repeat_control(control, 20), environment, 0.01)
        return final.rigid_body.angular_velocity[1]

    gradient = jax.jit(jax.grad(final_pitch_rate))(jnp.array(0.1))
    assert jnp.isfinite(gradient)
    assert jnp.abs(gradient) > 1e-5


def test_equilibration_initializes_post_stall_and_actuator_states():
    model = aerobatic_reference()
    alpha = 40.0 * jnp.pi / 180.0
    state = zero_state(model)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            velocity=jnp.array([12.0 * jnp.cos(alpha), 0.0, 12.0 * jnp.sin(alpha)])
        )
    )
    control = ControlInput(propeller=jnp.array([0.5]), channel=jnp.array([0.0, 0.2, 0.0]))
    state = equilibrate_internal_state(model, state, control, standard_environment())

    assert jnp.all(state.aero.separation[:3] > 0.99)
    assert state.actuators.propeller_speed[0] > 0.0
    assert state.actuators.surface_deflection[2] != 0.0


def test_torque_free_tumble_conserves_angular_momentum_and_energy():
    model = aerobatic_reference()
    vacuum = Environment(density=jnp.array(0.0), wind=jnp.zeros(3), gravity=jnp.zeros(3))
    state = zero_state(model)
    # Spin mostly about the intermediate inertia axis so the body tumbles instead of spinning
    # steadily. This exercises the quaternion kinematics and Euler's equation together.
    state = state._replace(
        rigid_body=state.rigid_body._replace(angular_velocity=jnp.array([0.1, 6.0, 0.1]))
    )
    controls = repeat_control(zero_control(model), steps=10_000)

    final, _ = jax.jit(rollout)(model, state, controls, vacuum, 0.001)

    def angular_momentum_world(value):
        body = matvec(model.inertia, value.rigid_body.angular_velocity)
        return quaternion_rotate(value.rigid_body.attitude, body)

    def rotational_energy(value):
        rate = value.rigid_body.angular_velocity
        return 0.5 * rate @ matvec(model.inertia, rate)

    initial_momentum = angular_momentum_world(state)
    momentum_error = jnp.linalg.norm(angular_momentum_world(final) - initial_momentum)
    assert momentum_error / jnp.linalg.norm(initial_momentum) < 1e-5
    energy_error = jnp.abs(rotational_energy(final) - rotational_energy(state))
    assert energy_error / rotational_energy(state) < 1e-5
    assert jnp.allclose(jnp.linalg.norm(final.rigid_body.attitude), 1.0, atol=1e-6)
