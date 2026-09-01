import jax.numpy as jnp

from cascade.canonical import (
    CANONICAL_STATE_SCHEMA,
    attitude_from_canonical,
    attitude_to_canonical,
    flu_to_frd,
    frd_to_flu,
    ned_to_nwu,
    nwu_to_ned,
    rigid_body_from_canonical,
    rigid_body_to_canonical,
)
from cascade.math import quaternion_from_euler, quaternion_rotate
from cascade.state import RigidBodyState


def test_ned_to_nwu_flips_y_and_z_and_is_self_inverse():
    vector = jnp.array([10.0, 2.0, -1.0])
    converted = ned_to_nwu(vector)
    assert jnp.array_equal(converted, jnp.array([10.0, -2.0, 1.0]))
    assert jnp.array_equal(ned_to_nwu(converted), vector)
    assert jnp.array_equal(nwu_to_ned(converted), vector)


def test_frd_to_flu_flips_y_and_z_and_is_self_inverse():
    vector = jnp.array([0.1, 0.2, 0.3])
    converted = frd_to_flu(vector)
    assert jnp.allclose(converted, jnp.array([0.1, -0.2, -0.3]))
    assert jnp.allclose(frd_to_flu(converted), vector)
    assert jnp.allclose(flu_to_frd(converted), vector)


def test_identity_attitude_maps_to_identity_canonical_quaternion():
    identity_xyzw = jnp.array([0.0, 0.0, 0.0, 1.0])
    identity_wxyz = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.array_equal(attitude_to_canonical(identity_xyzw), identity_wxyz)
    assert jnp.allclose(attitude_from_canonical(identity_wxyz), identity_xyzw)


def _euler_zyx_quaternion_wxyz(
    roll: jnp.ndarray, pitch: jnp.ndarray, yaw: jnp.ndarray
) -> jnp.ndarray:
    """Scalar-first ZYX quaternion, built with the same half-angle formulas as
    :func:`cascade.math.quaternion_from_euler`, reordered to ``wxyz``."""

    half_roll, half_pitch, half_yaw = roll / 2.0, pitch / 2.0, yaw / 2.0
    cr, sr = jnp.cos(half_roll), jnp.sin(half_roll)
    cp, sp = jnp.cos(half_pitch), jnp.sin(half_pitch)
    cy, sy = jnp.cos(half_yaw), jnp.sin(half_yaw)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return jnp.stack((w, x, y, z), axis=-1)


def test_attitude_to_canonical_matches_sign_flipped_euler_angles():
    euler_triples = [
        (0.3, -0.2, 0.7),
        (1.1, 0.4, -0.9),
        (-0.5, 0.6, 1.3),
        (0.0, 0.0, 0.0),
    ]
    for roll, pitch, yaw in euler_triples:
        roll, pitch, yaw = jnp.array(roll), jnp.array(pitch), jnp.array(yaw)
        quaternion_xyzw = quaternion_from_euler(roll, pitch, yaw)
        canonical = attitude_to_canonical(quaternion_xyzw)
        expected = _euler_zyx_quaternion_wxyz(roll, -pitch, -yaw)
        assert jnp.allclose(canonical, expected, atol=1e-6) or jnp.allclose(
            canonical, -expected, atol=1e-6
        )


def test_rotation_is_consistent_across_frame_conversion():
    """Rotating a body vector and converting the world result matches converting the body
    vector and rotating it by the frame-converted attitude, showing the conversion is a
    proper change of frames (i.e. matches conjugation of the rotation by ``S``).

    Note: composing ``attitude_to_canonical`` with ``attitude_from_canonical`` is, by
    construction, an exact round trip back to the original ``xyzw`` quaternion (the frame
    change is self-inverse, so converting out and back necessarily recovers the input). The
    attitude that actually rotates FLU vectors into NWU results is therefore the canonical
    ``wxyz`` quaternion reinterpreted in ``xyzw`` order (no further sign flip) -- that is what
    is compared against below.
    """

    quaternions = [
        jnp.array([0.0, 0.0, jnp.sin(jnp.pi / 4.0), jnp.cos(jnp.pi / 4.0)]),
        jnp.array([0.18, -0.36, 0.54, 0.744]),
        jnp.array([-0.5, 0.5, -0.5, 0.5]),
    ]
    vectors = [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.4, -1.2, 3.1]),
        jnp.array([-2.0, 0.5, 0.5]),
    ]

    for quaternion, vector in zip(quaternions, vectors, strict=True):
        world_result = ned_to_nwu(quaternion_rotate(quaternion, vector))

        canonical_attitude_wxyz = attitude_to_canonical(quaternion)
        converted_attitude_xyzw = jnp.concatenate(
            (canonical_attitude_wxyz[..., 1:], canonical_attitude_wxyz[..., :1]), axis=-1
        )
        converted_result = quaternion_rotate(converted_attitude_xyzw, frd_to_flu(vector))

        assert jnp.allclose(world_result, converted_result, atol=1e-6)


def test_rigid_body_canonical_round_trip_is_batched():
    batch_shape = (5,)
    rigid_body = RigidBodyState(
        position=jnp.arange(15.0).reshape(*batch_shape, 3),
        attitude=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (*batch_shape, 1)),
        velocity=jnp.arange(15.0).reshape(*batch_shape, 3) * 0.1,
        angular_velocity=jnp.arange(15.0).reshape(*batch_shape, 3) * -0.1,
    )

    canonical = rigid_body_to_canonical(rigid_body)
    assert canonical.shape == (*batch_shape, 13)

    round_tripped = rigid_body_from_canonical(canonical)
    for original, restored in zip(rigid_body, round_tripped, strict=True):
        assert jnp.allclose(original, restored, atol=1e-6)


def test_canonical_state_schema_is_the_glassbox_literal():
    assert CANONICAL_STATE_SCHEMA == "rigid_body_13_nwu_flu_wxyz_v1"
