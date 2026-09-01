import jax
import jax.numpy as jnp
import numpy as np

from cascade.initialization import standard_environment, zero_state
from cascade.math import quaternion_from_euler, quaternion_from_matrix, quaternion_rotate
from cascade.reference import tailsitter_reference
from cascade.vtol import (
    HoverSetpoint,
    default_hover_gains,
    hover_guidance,
    hover_throttle,
    thrust_direction_attitude,
)


def test_quaternion_from_matrix_inverts_rotations_in_every_quadrant():
    angles = ((0.1, 0.2, 0.3), (3.0, -1.2, 2.5), (-2.8, 1.4, -3.0), (0.0, 1.5, 0.0))
    for roll, pitch, yaw in angles:
        quaternion = quaternion_from_euler(jnp.array(roll), jnp.array(pitch), jnp.array(yaw))
        columns = [quaternion_rotate(quaternion, jnp.eye(3)[i]) for i in range(3)]
        axes = jnp.stack(columns, axis=-1)
        recovered = quaternion_from_matrix(axes)
        assert jnp.allclose(jnp.abs(jnp.sum(recovered * quaternion)), 1.0, atol=1e-5)


def test_thrust_direction_attitude_points_body_x_along_the_thrust_axis():
    direction = jnp.array([0.3, 0.0, -1.0])
    attitude = thrust_direction_attitude(direction, jnp.array(0.0))
    x_body = quaternion_rotate(attitude, jnp.array([1.0, 0.0, 0.0]))
    z_body = quaternion_rotate(attitude, jnp.array([0.0, 0.0, 1.0]))
    assert jnp.allclose(x_body, direction / jnp.linalg.norm(direction), atol=1e-5)
    # The belly faces the azimuth (north here) as far as the thrust axis allows.
    assert z_body[0] > 0.9
    assert jnp.allclose(jnp.dot(z_body, x_body), 0.0, atol=1e-5)


def test_hover_guidance_commands_vertical_attitude_and_hover_throttle_at_rest():
    model = tailsitter_reference()
    environment = standard_environment()
    state = zero_state(model, altitude=1.5)
    setpoint = HoverSetpoint(
        position_ned=jnp.array([0.0, 0.0, -1.5]),
        velocity_ned=jnp.zeros(3),
        azimuth_rad=jnp.array(0.0),
    )

    attitude, throttle = hover_guidance(model, default_hover_gains(), setpoint, state, environment)

    x_body = quaternion_rotate(attitude, jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(x_body, jnp.array([0.0, 0.0, -1.0]), atol=1e-5)
    assert throttle.shape == (2,)
    assert jnp.allclose(throttle, throttle[0])
    assert 0.7 < float(throttle[0]) < 0.85
    # Static thrust at that throttle carries the weight.
    weight = float(model.mass) * 9.80665
    propellers = model.propellers
    static = float(propellers.thrust_map[0, 1, 0]) * 1.225 * float(propellers.diameter[0]) ** 4
    speed_max = float(model.actuators.propeller_speed_max[0])
    revolutions = float(throttle[0]) * speed_max / (2.0 * np.pi)
    assert abs(2.0 * static * revolutions**2 - weight) < 0.02 * weight


def test_hover_guidance_tilts_toward_a_position_error_within_the_limit():
    model = tailsitter_reference()
    environment = standard_environment()
    state = zero_state(model, altitude=1.5)
    gains = default_hover_gains()
    setpoint = HoverSetpoint(
        position_ned=jnp.array([5.0, 0.0, -1.5]),
        velocity_ned=jnp.zeros(3),
        azimuth_rad=jnp.array(0.0),
    )

    attitude, throttle = hover_guidance(model, gains, setpoint, state, environment)

    x_body = quaternion_rotate(attitude, jnp.array([1.0, 0.0, 0.0]))
    assert x_body[0] > 0.3
    tilt = float(jnp.arccos(-x_body[2]))
    assert tilt <= float(gains.tilt_limit) + 1e-4
    assert float(throttle[0]) > 0.78


def test_hover_guidance_is_batched_and_differentiable():
    model = tailsitter_reference()
    environment = standard_environment(batch_shape=(3,))
    state = zero_state(model, batch_shape=(3,), altitude=1.5)
    setpoint = HoverSetpoint(
        position_ned=jnp.array([[0.0, 0.0, -1.5], [1.0, 0.0, -1.5], [0.0, -1.0, -2.0]]),
        velocity_ned=jnp.zeros((3, 3)),
        azimuth_rad=jnp.zeros(3),
    )
    attitude, throttle = jax.jit(hover_guidance)(
        model, default_hover_gains(), setpoint, state, environment
    )
    assert attitude.shape == (3, 4)
    assert throttle.shape == (3, 2)

    def cost(position_kp):
        gains = default_hover_gains()._replace(position_kp=position_kp)
        _, throttle = hover_guidance(model, gains, setpoint, state, environment)
        return jnp.sum(throttle)

    gradient = jax.grad(cost)(jnp.asarray(2.0))
    assert jnp.isfinite(gradient)
    assert hover_throttle(model, jnp.asarray(0.0), jnp.asarray(1.225)).shape == (2,)
