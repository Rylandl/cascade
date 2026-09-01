"""Hover guidance for thrust-borne flight: position and velocity to attitude and throttle.

A tailsitter hovers on its propellers with body ``x`` pointing up. This module turns a position
and velocity setpoint into the thrust direction the airframe must point at and the throttle that
produces the required thrust magnitude, leaving the wing plane free to face a chosen azimuth.
The output feeds the attitude and rate loops of :mod:`cascade.control`, so a transition is a
change of setpoint schedule, not of controller structure. Everything is pure and batched.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from cascade.math import normalize, quaternion_from_matrix, safe_norm
from cascade.model import AircraftModel
from cascade.state import AircraftState, Environment


class HoverGains(NamedTuple):
    """Position and velocity gains and the limits that keep hover commands sane."""

    position_kp: Array
    velocity_kp: Array
    acceleration_limit: Array
    tilt_limit: Array


class HoverSetpoint(NamedTuple):
    """Where to be, how fast to move, and which azimuth the wing's belly should face."""

    position_ned: Array
    velocity_ned: Array
    azimuth_rad: Array


def default_hover_gains() -> HoverGains:
    return HoverGains(
        position_kp=jnp.asarray(2.0),
        velocity_kp=jnp.asarray(2.5),
        acceleration_limit=jnp.asarray(5.0),
        tilt_limit=jnp.asarray(0.5),
    )


def thrust_direction_attitude(direction_world: Array, azimuth_rad: Array) -> Array:
    """Attitude whose body ``x`` points along ``direction_world`` with the belly toward an azimuth.

    Body ``z`` is the projection of the azimuth's horizontal unit vector orthogonal to the thrust
    axis, and body ``y`` completes the right-handed frame, so the wing plane is fixed by the
    azimuth rather than left to drift. The construction is regular whenever the thrust axis is
    not itself along the azimuth vector, which a hovering or climbing airframe never approaches.
    """

    x_body = normalize(direction_world)
    reference = jnp.stack(
        (jnp.cos(azimuth_rad), jnp.sin(azimuth_rad), jnp.zeros_like(azimuth_rad)), axis=-1
    )
    z_body = normalize(
        reference - jnp.sum(reference * x_body, axis=-1, keepdims=True) * x_body
    )
    y_body = jnp.cross(z_body, x_body, axis=-1)
    rotation = jnp.stack((x_body, y_body, z_body), axis=-1)
    return quaternion_from_matrix(rotation)


def hover_throttle(model: AircraftModel, thrust_total: Array, density: Array) -> Array:
    """Throttle for every motor so that static thrust sums to ``thrust_total``.

    Uses the static term of the thrust map, ``T = rho n^2 D^4 c_20``; inflow corrections are
    left to the feedback loops. Shape ``(..., P)``.
    """

    propellers = model.propellers
    static = propellers.thrust_map[..., 1, 0] * propellers.diameter**4
    per_motor = thrust_total[..., None] / model.n_propellers
    revolutions = jnp.sqrt(jnp.maximum(per_motor / (density[..., None] * static), 0.0))
    speed = 2.0 * jnp.pi * revolutions
    minimum, maximum = model.actuators.propeller_speed_min, model.actuators.propeller_speed_max
    return jnp.clip((speed - minimum) / jnp.maximum(maximum - minimum, 1e-6), 0.0, 1.0)


def hover_guidance(
    model: AircraftModel,
    gains: HoverGains,
    setpoint: HoverSetpoint,
    state: AircraftState,
    environment: Environment,
) -> tuple[Array, Array]:
    """Attitude setpoint (``xyzw``) and per-motor throttle for thrust-borne flight.

    The commanded acceleration is a limited proportional-derivative law on position and
    velocity; the required specific force is that acceleration minus gravity, its direction
    gives the thrust axis (tilt-limited about vertical) and its magnitude times mass gives the
    total thrust.
    """

    rigid_body = state.rigid_body
    acceleration = gains.position_kp * (
        setpoint.position_ned - rigid_body.position
    ) + gains.velocity_kp * (setpoint.velocity_ned - rigid_body.velocity)
    # safe_norm keeps the gradient finite at zero command, where the plain norm gives 0 / 0.
    magnitude = safe_norm(acceleration, keepdims=True)
    acceleration = acceleration * jnp.minimum(1.0, gains.acceleration_limit / magnitude)
    specific_force = acceleration - environment.gravity
    up = -normalize(environment.gravity)
    # Limit the tilt away from vertical by scaling the horizontal part of the specific force
    # so that its ratio to the vertical part never exceeds tan(tilt_limit). This is smooth at
    # zero tilt, unlike an angle computed through arccos.
    vertical = jnp.sum(specific_force * up, axis=-1, keepdims=True)
    horizontal = specific_force - vertical * up
    horizontal_norm = safe_norm(horizontal, keepdims=True)
    allowed = jnp.tan(gains.tilt_limit) * jnp.maximum(vertical, 1e-3)
    horizontal = horizontal * jnp.minimum(1.0, allowed / horizontal_norm)
    direction = normalize(vertical * up + horizontal)
    attitude = thrust_direction_attitude(direction, setpoint.azimuth_rad)
    thrust_total = model.mass * jnp.sum(specific_force * direction, axis=-1)
    throttle = hover_throttle(model, thrust_total, environment.density)
    return attitude, throttle


def velocity_ramp_schedule(
    steps: int,
    dt: float,
    *,
    start_position_ned: Array,
    heading_rad: Array,
    cruise_speed_m_s: Array,
    acceleration_m_s2: Array,
    hold_steps: int = 0,
) -> HoverSetpoint:
    """Time-major hover setpoints that hold, then accelerate along a heading to cruise speed.

    Velocity ramps linearly at ``acceleration_m_s2`` after ``hold_steps`` and saturates at
    ``cruise_speed_m_s``; position is the integral of that velocity from the start point, and
    the wing's belly faces the heading throughout. This is the setpoint side of a hover-to-cruise
    transition; the guidance law decides how far to tilt to follow it.
    """

    time = jnp.arange(steps) * dt
    ramp_time = jnp.maximum(time - hold_steps * dt, 0.0)
    speed = jnp.minimum(acceleration_m_s2 * ramp_time, cruise_speed_m_s)
    # Distance along the heading: integral of the saturating ramp.
    ramp_end = cruise_speed_m_s / acceleration_m_s2
    distance = jnp.where(
        ramp_time < ramp_end,
        0.5 * acceleration_m_s2 * ramp_time**2,
        0.5 * cruise_speed_m_s * ramp_end + cruise_speed_m_s * (ramp_time - ramp_end),
    )
    along = jnp.stack(
        (jnp.cos(heading_rad), jnp.sin(heading_rad), jnp.zeros_like(heading_rad)), axis=-1
    )
    position = start_position_ned + distance[:, None] * along
    velocity = speed[:, None] * along
    return HoverSetpoint(
        position_ned=position,
        velocity_ned=velocity,
        azimuth_rad=jnp.broadcast_to(heading_rad, (steps,)),
    )
