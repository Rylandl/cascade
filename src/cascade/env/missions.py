"""Validated, time-varying missions over the existing tracking task and dynamics.

Builders run on the host. Their NamedTuple results are PyTrees: evaluate one mission with
``jit`` or use ``vmap`` for a batch with the same number of schedule points.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.env.tasks import TrackingTask, tracking_task, trimmed_reference
from cascade.math import safe_norm


class MissionTarget(NamedTuple):
    """The instantaneous shared reward, observation, and guidance target.

    Position errors are actual minus desired in NED. Horizontal position cost is squared
    distance divided by ``position_scale_m`` squared; altitude retains TrackingTask's 10 m
    scale. Scheduled heading tasks disable horizontal position tracking.
    """

    tracking: TrackingTask
    position_ned: Array
    position_weight: Array
    position_scale_m: Array
    track_position: Array

    @property
    def airspeed_m_s(self):
        return self.tracking.airspeed_m_s

    @property
    def altitude_m(self):
        return self.tracking.altitude_m

    @property
    def heading_rad(self):
        return self.tracking.heading_rad

    def heading_error(self, rigid):
        return self.tracking.heading_error(rigid)

    def position_error(self, rigid):
        return jnp.where(
            self.track_position,
            rigid.position - self.position_ned,
            self.tracking.position_error(rigid),
        )

    def reference_speed(self):
        return self.airspeed_m_s

    def cost(self, rigid, environment, action):
        horizontal = (rigid.position[..., :2] - self.position_ned[..., :2]) / self.position_scale_m
        return self.tracking.cost(rigid, environment, action) + jnp.where(
            self.track_position,
            self.position_weight * jnp.sum(jnp.square(horizontal), axis=-1),
            0.0,
        )


def _tracking(template, speed, altitude, heading):
    return template._replace(airspeed_m_s=speed, altitude_m=altitude, heading_rad=heading)


def _reference(mission, model, environment):
    target = mission.at(jnp.asarray(0.0))
    reference = trimmed_reference(model, target.tracking, environment)
    rigid = reference.state.rigid_body
    position = jnp.where(target.track_position, target.position_ned, rigid.position)
    return reference._replace(
        state=reference.state._replace(rigid_body=rigid._replace(position=position))
    )


class ScheduledTrackingTask(NamedTuple):
    """Linearly interpolated speed, altitude, and unwrapped heading; endpoints are held."""

    times_s: Array
    airspeed_m_s: Array
    altitude_m: Array
    heading_rad: Array
    template: TrackingTask

    def at(self, time_s, rigid=None):
        speed = jnp.interp(time_s, self.times_s, self.airspeed_m_s)
        altitude = jnp.interp(time_s, self.times_s, self.altitude_m)
        heading = jnp.interp(time_s, self.times_s, self.heading_rad)
        return MissionTarget(
            _tracking(self.template, speed, altitude, heading),
            jnp.array([0.0, 0.0, -altitude]),
            jnp.asarray(0.0),
            jnp.asarray(10.0),
            jnp.asarray(False),
        )

    def reference(self, model, environment=None):
        return _reference(self, model, environment)

    def reference_speed(self):
        return self.airspeed_m_s[0]


class WaypointTask(NamedTuple):
    """A timed piecewise-linear NED path with lookahead point guidance.

    Position cost follows the point at the current time. Heading aims from the aircraft
    toward the point ``lookahead_s`` in the future. This is a tracking experiment, not an
    arrival-triggered route planner; the episode should end at the final waypoint time.
    """

    times_s: Array
    positions_ned: Array
    airspeed_m_s: Array
    lookahead_s: Array
    position_weight: Array
    position_scale_m: Array
    template: TrackingTask

    def _position(self, time_s):
        return jnp.stack(
            tuple(jnp.interp(time_s, self.times_s, self.positions_ned[:, i]) for i in range(3))
        )

    def at(self, time_s, rigid=None):
        desired = self._position(time_s)
        ahead = self._position(time_s + self.lookahead_s)
        current = desired if rigid is None else rigid.position
        direction = ahead[:2] - current[:2]
        segment = jnp.clip(
            jnp.searchsorted(self.times_s, time_s, side="right") - 1, 0, self.times_s.shape[0] - 2
        )
        positions = jnp.asarray(self.positions_ned)
        tangent = positions[segment + 1, :2] - positions[segment, :2]
        direction = jnp.where(jnp.sum(direction**2) > 1e-12, direction, tangent)
        heading = jnp.arctan2(direction[1], direction[0])
        speed = jnp.interp(time_s, self.times_s, self.airspeed_m_s)
        return MissionTarget(
            _tracking(self.template, speed, -desired[2], heading),
            desired,
            self.position_weight,
            self.position_scale_m,
            jnp.asarray(True),
        )

    def reference(self, model, environment=None):
        return _reference(self, model, environment)

    def reference_speed(self):
        return self.airspeed_m_s[0]


class OrbitTask(NamedTuple):
    """A horizontal ground-frame circle, with tangent heading plus radial capture.

    No phase/timing penalty is imposed. Radius, altitude, and airspeed are targets; wind
    compensation and bank feasibility are left to the controller/experimenter.
    """

    center_ned: Array
    radius_m: Array
    airspeed_m_s: Array
    direction: Array
    initial_phase_rad: Array
    capture_distance_m: Array
    position_weight: Array
    template: TrackingTask

    def at(self, time_s, rigid=None):
        if rigid is None:
            angle = self.initial_phase_rad
            radius = self.radius_m
        else:
            offset = rigid.position[:2] - self.center_ned[:2]
            radius = safe_norm(offset)
            nonzero = jnp.sum(offset**2) > 1e-12
            safe_offset = jnp.where(nonzero, offset, jnp.array([1.0, 0.0]))
            angle = jnp.where(
                nonzero, jnp.arctan2(safe_offset[1], safe_offset[0]), self.initial_phase_rad
            )
        radial = jnp.array([jnp.cos(angle), jnp.sin(angle)])
        position = jnp.asarray(self.center_ned).at[:2].add(self.radius_m * radial)
        heading = angle + self.direction * (
            jnp.pi / 2.0 + jnp.arctan2(radius - self.radius_m, self.capture_distance_m)
        )
        return MissionTarget(
            _tracking(self.template, self.airspeed_m_s, -self.center_ned[2], heading),
            position,
            self.position_weight,
            self.radius_m,
            jnp.asarray(True),
        )

    def reference(self, model, environment=None):
        # A straight tangent trim initializes the maneuver; this is not a steady-turn trim.
        return _reference(self, model, environment)

    def reference_speed(self):
        return self.airspeed_m_s


def _host_array(name, value):
    try:
        host = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must contain finite numbers")
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    with np.errstate(over="ignore", invalid="ignore"):
        native = host.astype(dtype)
    if not np.all(np.isfinite(native)):
        raise ValueError(f"{name} must be finite in the active JAX dtype")
    return native


def _scalar(name, value, *, positive=False, nonnegative=False):
    result = _host_array(name, value)
    if result.shape != () or (positive and result <= 0) or (nonnegative and result < 0):
        bound = "positive" if positive else "nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be a {bound} scalar")
    return jnp.asarray(result)


def _times(times_s):
    values = _host_array("times_s", times_s)
    if values.ndim != 1 or values.size < 2 or values[0] != 0 or np.any(np.diff(values) <= 0):
        raise ValueError("times_s must start at zero and contain at least two increasing times")
    return values


def _series(name, value, count, *, positive=False):
    result = _host_array(name, value)
    if result.shape == ():
        result = np.full(count, result, dtype=result.dtype)
    if result.shape != (count,) or (positive and np.any(result <= 0)):
        raise ValueError(
            f"{name} must be a {'positive ' if positive else ''}scalar or length {count}"
        )
    return result


def _template(speed, altitude, heading, weights):
    allowed = {
        "airspeed_weight",
        "altitude_weight",
        "heading_weight",
        "rate_weight",
        "effort_weight",
    }
    unknown = weights.keys() - allowed
    if unknown:
        raise TypeError(f"unknown tracking weights: {sorted(unknown)}")
    checked = {name: _scalar(name, value, nonnegative=True) for name, value in weights.items()}
    return tracking_task(speed, altitude, heading, **checked)


def scheduled_tracking_task(
    times_s, airspeed_m_s, altitude_m, heading_rad=0.0, **weights
) -> ScheduledTrackingTask:
    """Build a schedule with scalar or per-knot targets and nonnegative tracking weights.

    Adjacent headings follow the shortest angular arc; use intermediate knots to request
    turns of 180 degrees or more unambiguously. Values outside the time range hold endpoints.
    """

    times = _times(times_s)
    speed = _series("airspeed_m_s", airspeed_m_s, len(times), positive=True)
    altitude = _series("altitude_m", altitude_m, len(times))
    heading = _host_array("heading_rad", np.unwrap(_series("heading_rad", heading_rad, len(times))))
    return ScheduledTrackingTask(
        jnp.asarray(times),
        jnp.asarray(speed),
        jnp.asarray(altitude),
        jnp.asarray(heading),
        _template(speed[0], altitude[0], heading[0], weights),
    )


def waypoint_task(
    times_s,
    positions_ned,
    airspeed_m_s,
    *,
    lookahead_s=2.0,
    position_weight=1.0,
    position_scale_m=10.0,
    **weights,
) -> WaypointTask:
    """Build a timed NED path. Consecutive waypoints must differ horizontally.

    Airspeed is explicit (scalar or per knot); waypoint timings specify ground motion, so
    their feasibility in the selected wind must be checked by the caller.
    """

    times = _times(times_s)
    positions = _host_array("positions_ned", positions_ned)
    if positions.shape != (len(times), 3):
        raise ValueError(f"positions_ned must have shape ({len(times)}, 3)")
    if np.any(np.all(np.diff(positions[:, :2], axis=0) == 0, axis=1)):
        raise ValueError("consecutive waypoints must differ horizontally")
    speed = _series("airspeed_m_s", airspeed_m_s, len(times), positive=True)
    return WaypointTask(
        jnp.asarray(times),
        jnp.asarray(positions),
        jnp.asarray(speed),
        _scalar("lookahead_s", lookahead_s, positive=True),
        _scalar("position_weight", position_weight, nonnegative=True),
        _scalar("position_scale_m", position_scale_m, positive=True),
        _template(speed[0], -positions[0, 2], 0.0, weights),
    )


def orbit_task(
    center_ned,
    radius_m,
    airspeed_m_s,
    *,
    clockwise=True,
    initial_phase_rad=0.0,
    capture_distance_m=None,
    position_weight=1.0,
    **weights,
) -> OrbitTask:
    """Build a circle about a NED center; positive phase is clockwise from north.

    ``clockwise`` is a host boolean. Capture distance defaults to the radius and controls
    how strongly radial displacement turns the commanded heading toward the circle.
    """

    center = _host_array("center_ned", center_ned)
    if center.shape != (3,):
        raise ValueError("center_ned must have shape (3,)")
    if not isinstance(clockwise, (bool, np.bool_)):
        raise ValueError("clockwise must be a boolean")
    radius = _scalar("radius_m", radius_m, positive=True)
    speed = _scalar("airspeed_m_s", airspeed_m_s, positive=True)
    capture = radius_m if capture_distance_m is None else capture_distance_m
    return OrbitTask(
        jnp.asarray(center),
        radius,
        speed,
        jnp.asarray(1.0 if clockwise else -1.0),
        _scalar("initial_phase_rad", initial_phase_rad),
        _scalar("capture_distance_m", capture, positive=True),
        _scalar("position_weight", position_weight, nonnegative=True),
        _template(speed, -center[2], 0.0, weights),
    )


__all__ = [
    "MissionTarget",
    "OrbitTask",
    "ScheduledTrackingTask",
    "WaypointTask",
    "orbit_task",
    "scheduled_tracking_task",
    "waypoint_task",
]
