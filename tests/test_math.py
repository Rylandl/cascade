import jax.numpy as jnp

from cascade.math import (
    quaternion_multiply,
    quaternion_rotate,
    quaternion_rotate_inverse,
    safe_norm,
)


def test_quaternion_rotation_and_inverse():
    half_angle = jnp.pi / 4.0
    attitude = jnp.array([0.0, 0.0, jnp.sin(half_angle), jnp.cos(half_angle)])
    vector = jnp.array([1.0, 0.0, 0.0])

    rotated = quaternion_rotate(attitude, vector)
    restored = quaternion_rotate_inverse(attitude, rotated)

    assert jnp.allclose(rotated, jnp.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert jnp.allclose(restored, vector, atol=1e-6)


def test_quaternion_identity_product():
    quaternion = jnp.array([0.1, -0.2, 0.3, 0.92736185])
    identity = jnp.array([0.0, 0.0, 0.0, 1.0])
    assert jnp.allclose(quaternion_multiply(identity, quaternion), quaternion)
    assert jnp.allclose(quaternion_multiply(quaternion, identity), quaternion)


def test_safe_norm_has_finite_zero_gradient():
    value = jnp.zeros(3)
    assert jnp.isfinite(safe_norm(value))
