"""Conversion between Cascade's native conventions and Glassbox's canonical rigid-body state.

Cascade works internally in NED world axes, FRD body axes, and scalar-last ``xyzw``
quaternions rotating body vectors into world. The sibling Glassbox project instead uses a
canonical 13-vector rigid-body state ``[position(3), velocity(3), quaternion wxyz(4), body
angular velocity(3)]`` in NWU world axes, FLU body axes, and a scalar-first quaternion. Its
schema string is :data:`CANONICAL_STATE_SCHEMA`.

Both frame changes are the same 180-degree rotation about x, ``S = diag(1, -1, -1)``: world
vectors and body vectors both convert with the sign flip ``[1, -1, -1]``, and quaternions
convert as ``(x, y, z, w) -> (w, x, -y, -z)``. Because ``S`` is self-inverse, every vector
conversion here is its own inverse, and :func:`attitude_from_canonical` is the exact inverse of
:func:`attitude_to_canonical`.

This module is the only place in Cascade where frame conversion between the two conventions
happens; nothing else in the package should hand-roll these sign flips.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cascade.math import normalize
from cascade.state import RigidBodyState

CANONICAL_STATE_SIZE = 13
CANONICAL_STATE_SCHEMA = "rigid_body_13_nwu_flu_wxyz_v1"

_WORLD_AND_BODY_SIGNS = jnp.array([1.0, -1.0, -1.0])


def ned_to_nwu(vector: Array) -> Array:
    """Convert a world vector (or wind) from NED to NWU. Self-inverse."""

    return vector * _WORLD_AND_BODY_SIGNS


def nwu_to_ned(vector: Array) -> Array:
    """Convert a world vector (or wind) from NWU to NED. Self-inverse."""

    return vector * _WORLD_AND_BODY_SIGNS


def frd_to_flu(vector: Array) -> Array:
    """Convert a body vector (e.g. angular velocity) from FRD to FLU. Self-inverse."""

    return vector * _WORLD_AND_BODY_SIGNS


def flu_to_frd(vector: Array) -> Array:
    """Convert a body vector (e.g. angular velocity) from FLU to FRD. Self-inverse."""

    return vector * _WORLD_AND_BODY_SIGNS


def attitude_to_canonical(quaternion_xyzw: Array) -> Array:
    """Convert a scalar-last NED/FRD attitude to a scalar-first NWU/FLU quaternion."""

    x, y, z, w = (quaternion_xyzw[..., index] for index in range(4))
    return jnp.stack((w, x, -y, -z), axis=-1)


def attitude_from_canonical(quaternion_wxyz: Array) -> Array:
    """Convert a scalar-first NWU/FLU attitude to a normalized scalar-last NED/FRD quaternion."""

    w, x, y, z = (quaternion_wxyz[..., index] for index in range(4))
    return normalize(jnp.stack((x, -y, -z, w), axis=-1))


def rigid_body_to_canonical(rigid_body: RigidBodyState) -> Array:
    """Pack a native :class:`RigidBodyState` into a batched canonical 13-vector."""

    return jnp.concatenate(
        (
            ned_to_nwu(rigid_body.position),
            ned_to_nwu(rigid_body.velocity),
            attitude_to_canonical(rigid_body.attitude),
            frd_to_flu(rigid_body.angular_velocity),
        ),
        axis=-1,
    )


def rigid_body_from_canonical(canonical: Array) -> RigidBodyState:
    """Unpack a batched canonical 13-vector into a native :class:`RigidBodyState`."""

    return RigidBodyState(
        position=nwu_to_ned(canonical[..., 0:3]),
        velocity=nwu_to_ned(canonical[..., 3:6]),
        attitude=attitude_from_canonical(canonical[..., 6:10]),
        angular_velocity=flu_to_frd(canonical[..., 10:13]),
    )
