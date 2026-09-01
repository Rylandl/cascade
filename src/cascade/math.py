from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def safe_norm(value: Array, axis: int = -1, keepdims: bool = False, eps: float = 1e-8) -> Array:
    """Differentiable norm with a finite derivative at zero."""

    squared = jnp.sum(jnp.square(value), axis=axis, keepdims=keepdims)
    return jnp.sqrt(squared + eps**2)


def smooth_abs(value: Array, eps: float = 1e-6) -> Array:
    """Smooth approximation of absolute value."""

    return jnp.sqrt(jnp.square(value) + eps**2)


def normalize(value: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    return value / safe_norm(value, axis=axis, keepdims=True, eps=eps)


def quaternion_conjugate(quaternion: Array) -> Array:
    xyz, w = quaternion[..., :3], quaternion[..., 3:]
    return jnp.concatenate((-xyz, w), axis=-1)


def quaternion_multiply(left: Array, right: Array) -> Array:
    """Hamilton product for scalar-last ``xyzw`` quaternions."""

    left_xyz, left_w = left[..., :3], left[..., 3:]
    right_xyz, right_w = right[..., :3], right[..., 3:]
    xyz = left_w * right_xyz + right_w * left_xyz + jnp.cross(left_xyz, right_xyz, axis=-1)
    w = left_w * right_w - jnp.sum(left_xyz * right_xyz, axis=-1, keepdims=True)
    return jnp.concatenate((xyz, w), axis=-1)


def quaternion_rotate(quaternion: Array, vector: Array) -> Array:
    """Rotate vectors from body to world without constructing a matrix."""

    q_xyz = quaternion[..., :3]
    q_w = quaternion[..., 3:]
    twice_cross = 2.0 * jnp.cross(q_xyz, vector, axis=-1)
    return vector + q_w * twice_cross + jnp.cross(q_xyz, twice_cross, axis=-1)


def quaternion_rotate_inverse(quaternion: Array, vector: Array) -> Array:
    """Rotate vectors from world to body."""

    return quaternion_rotate(quaternion_conjugate(quaternion), vector)


def quaternion_derivative(quaternion: Array, angular_velocity_body: Array) -> Array:
    """Quaternion derivative for a body-to-world attitude and body-frame angular velocity."""

    pure_angular_velocity = jnp.concatenate(
        (angular_velocity_body, jnp.zeros_like(angular_velocity_body[..., :1])), axis=-1
    )
    return 0.5 * quaternion_multiply(quaternion, pure_angular_velocity)


def quaternion_from_euler(roll: Array, pitch: Array, yaw: Array) -> Array:
    """Body-to-world quaternion from aerospace ZYX yaw-pitch-roll angles."""

    half_roll, half_pitch, half_yaw = roll / 2.0, pitch / 2.0, yaw / 2.0
    cr, sr = jnp.cos(half_roll), jnp.sin(half_roll)
    cp, sp = jnp.cos(half_pitch), jnp.sin(half_pitch)
    cy, sy = jnp.cos(half_yaw), jnp.sin(half_yaw)
    return jnp.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )


def quaternion_from_rotvec(rotation_vector: Array) -> Array:
    """Quaternion exponential map with a finite derivative at zero."""

    angle = safe_norm(rotation_vector, eps=1e-12)
    scale = 0.5 * jnp.sinc(angle / (2.0 * jnp.pi))
    xyz = rotation_vector * scale[..., None]
    return jnp.concatenate((xyz, jnp.cos(angle / 2.0)[..., None]), axis=-1)


def quaternion_to_rotvec(quaternion: Array) -> Array:
    """Shortest quaternion logarithm map."""

    quaternion = normalize(quaternion)
    quaternion = jnp.where(quaternion[..., 3:] < 0.0, -quaternion, quaternion)
    xyz, scalar = quaternion[..., :3], quaternion[..., 3]
    vector_norm = safe_norm(xyz, eps=1e-12)
    angle = 2.0 * jnp.arctan2(vector_norm, scalar)
    return xyz * (angle / vector_norm)[..., None]


def rotation_y(angle: Array) -> Array:
    """Right-handed rotation about the local positive y axis."""

    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    zero, one = jnp.zeros_like(angle), jnp.ones_like(angle)
    row_0 = jnp.stack((cosine, zero, sine), axis=-1)
    row_1 = jnp.stack((zero, one, zero), axis=-1)
    row_2 = jnp.stack((-sine, zero, cosine), axis=-1)
    return jnp.stack((row_0, row_1, row_2), axis=-2)


def matvec(matrix: Array, vector: Array) -> Array:
    return jnp.einsum("...ij,...j->...i", matrix, vector)
