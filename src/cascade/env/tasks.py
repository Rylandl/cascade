"""Episode tasks and the reference flights they are drawn around.

Each task owns its cost, its heading and position errors, the speed that normalises its
observations, and how to build its reference flight (a trim for tracking, a static hover for
hover and transition), so the episode functions never branch on task type.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from cascade.analysis.trim import StraightFlightCondition, trim_straight_flight
from cascade.control.vtol import (
    hover_throttle,
    thrust_direction_attitude,
)
from cascade.initialization import (
    equilibrate_internal_state,
    standard_environment,
    zero_state,
)
from cascade.math import (
    normalize,
    quaternion_rotate,
    safe_norm,
)
from cascade.model import AircraftModel
from cascade.state import AircraftState, ControlInput, Environment


class TrackingTask(NamedTuple):
    """ReferenceFlight tracking: hold an airspeed, altitude, and heading.

    Weights multiply normalised squared errors (airspeed by the reference, altitude by 10 m,
    heading by ``1 - cos``), body rates, and the mean squared normalised action.
    """

    airspeed_m_s: Array
    altitude_m: Array
    heading_rad: Array
    airspeed_weight: Array
    altitude_weight: Array
    heading_weight: Array
    rate_weight: Array
    effort_weight: Array

    def heading_error(self, rigid) -> Array:
        return _nose_heading(rigid.attitude) - self.heading_rad

    def position_error(self, rigid) -> Array:
        """World-frame error; only altitude is referenced, so north and east are zero."""

        altitude_error = -rigid.position[..., 2] - self.altitude_m
        zeros = jnp.zeros_like(altitude_error)
        return jnp.stack((zeros, zeros, -altitude_error), axis=-1)

    def reference_speed(self) -> Array:
        return self.airspeed_m_s

    def reference(self, model: AircraftModel, environment: Environment | None = None):
        return trimmed_reference(model, self, environment)

    def cost(self, rigid, environment: Environment, action: Array) -> Array:
        airspeed = safe_norm(rigid.velocity - environment.wind)
        airspeed_error = (airspeed - self.airspeed_m_s) / jnp.maximum(self.airspeed_m_s, 1.0)
        altitude_error = (-rigid.position[..., 2] - self.altitude_m) / 10.0
        return (
            self.airspeed_weight * jnp.square(airspeed_error)
            + self.altitude_weight * jnp.square(altitude_error)
            + self.heading_weight * (1.0 - jnp.cos(self.heading_error(rigid)))
            + self.rate_weight * jnp.sum(jnp.square(rigid.angular_velocity), axis=-1)
            + self.effort_weight * jnp.mean(jnp.square(action), axis=-1)
        )


class HoverTask(NamedTuple):
    """Hold a position with the belly toward an azimuth: the tailsitter's hover.

    Position error is normalised by 1 m and velocity by 1 m/s (hover is a precision task);
    the azimuth is the belly direction (body z projected on the horizontal), which is what a
    tailsitter's hover attitude fixes, since its nose points up.
    """

    position_ned: Array
    azimuth_rad: Array
    position_weight: Array
    velocity_weight: Array
    azimuth_weight: Array
    rate_weight: Array
    effort_weight: Array

    def heading_error(self, rigid) -> Array:
        return _belly_azimuth(rigid.attitude) - self.azimuth_rad

    def position_error(self, rigid) -> Array:
        return self.position_ned - rigid.position

    def reference_speed(self) -> Array:
        return jnp.asarray(1.0)

    @property
    def hover_position_ned(self) -> Array:
        return self.position_ned

    @property
    def hover_azimuth_rad(self) -> Array:
        return self.azimuth_rad

    def reference(self, model: AircraftModel, environment: Environment | None = None):
        return hover_reference(model, self, environment)

    def cost(self, rigid, environment: Environment, action: Array) -> Array:
        return (
            self.position_weight * jnp.sum(jnp.square(self.position_error(rigid)), axis=-1)
            + self.velocity_weight * jnp.sum(jnp.square(rigid.velocity), axis=-1)
            + self.azimuth_weight * (1.0 - jnp.cos(self.heading_error(rigid)))
            + self.rate_weight * jnp.sum(jnp.square(rigid.angular_velocity), axis=-1)
            + self.effort_weight * jnp.mean(jnp.square(action), axis=-1)
        )


class TransitionTask(NamedTuple):
    """From hover, reach and hold a cruise airspeed, altitude, and heading: a tailsitter's
    forward transition. Heading is the belly azimuth, defined in hover and in cruise alike."""

    cruise_speed_m_s: Array
    altitude_m: Array
    heading_rad: Array
    airspeed_weight: Array
    altitude_weight: Array
    heading_weight: Array
    rate_weight: Array
    effort_weight: Array

    def heading_error(self, rigid) -> Array:
        return _belly_azimuth(rigid.attitude) - self.heading_rad

    def position_error(self, rigid) -> Array:
        altitude_error = -rigid.position[..., 2] - self.altitude_m
        zeros = jnp.zeros_like(altitude_error)
        return jnp.stack((zeros, zeros, -altitude_error), axis=-1)

    def reference_speed(self) -> Array:
        return self.cruise_speed_m_s

    @property
    def hover_position_ned(self) -> Array:
        return jnp.array([0.0, 0.0, 0.0]).at[2].set(-self.altitude_m)

    @property
    def hover_azimuth_rad(self) -> Array:
        return self.heading_rad

    def reference(self, model: AircraftModel, environment: Environment | None = None):
        return hover_reference(model, self, environment)

    def cost(self, rigid, environment: Environment, action: Array) -> Array:
        airspeed = safe_norm(rigid.velocity - environment.wind)
        airspeed_error = (airspeed - self.cruise_speed_m_s) / jnp.maximum(
            self.cruise_speed_m_s, 1.0
        )
        altitude_error = (-rigid.position[..., 2] - self.altitude_m) / 10.0
        return (
            self.airspeed_weight * jnp.square(airspeed_error)
            + self.altitude_weight * jnp.square(altitude_error)
            + self.heading_weight * (1.0 - jnp.cos(self.heading_error(rigid)))
            + self.rate_weight * jnp.sum(jnp.square(rigid.angular_velocity), axis=-1)
            + self.effort_weight * jnp.mean(jnp.square(action), axis=-1)
        )


Task = TrackingTask | HoverTask | TransitionTask


class ReferenceFlight(NamedTuple):
    """The flight an episode is drawn around: a state, the control that holds it, and the
    environment it was found in (a cruise trim, or a static hover)."""

    state: AircraftState
    control: ControlInput
    environment: Environment


# The name before 0.2; kept so existing imports keep working.
Reference = ReferenceFlight


def tracking_task(
    airspeed_m_s: float,
    altitude_m: float,
    heading_rad: float = 0.0,
    *,
    airspeed_weight: float = 1.0,
    altitude_weight: float = 1.0,
    heading_weight: float = 1.0,
    rate_weight: float = 0.05,
    effort_weight: float = 0.05,
) -> TrackingTask:
    return TrackingTask(
        airspeed_m_s=jnp.asarray(airspeed_m_s),
        altitude_m=jnp.asarray(altitude_m),
        heading_rad=jnp.asarray(heading_rad),
        airspeed_weight=jnp.asarray(airspeed_weight),
        altitude_weight=jnp.asarray(altitude_weight),
        heading_weight=jnp.asarray(heading_weight),
        rate_weight=jnp.asarray(rate_weight),
        effort_weight=jnp.asarray(effort_weight),
    )


def hover_task(
    position_ned,
    azimuth_rad: float = 0.0,
    *,
    position_weight: float = 1.0,
    velocity_weight: float = 0.1,
    azimuth_weight: float = 0.5,
    rate_weight: float = 0.02,
    effort_weight: float = 0.05,
) -> HoverTask:
    return HoverTask(
        position_ned=jnp.asarray(position_ned),
        azimuth_rad=jnp.asarray(azimuth_rad),
        position_weight=jnp.asarray(position_weight),
        velocity_weight=jnp.asarray(velocity_weight),
        azimuth_weight=jnp.asarray(azimuth_weight),
        rate_weight=jnp.asarray(rate_weight),
        effort_weight=jnp.asarray(effort_weight),
    )


def transition_task(
    cruise_speed_m_s: float,
    altitude_m: float,
    heading_rad: float = 0.0,
    *,
    airspeed_weight: float = 1.0,
    altitude_weight: float = 1.0,
    heading_weight: float = 0.5,
    rate_weight: float = 0.02,
    effort_weight: float = 0.05,
) -> TransitionTask:
    return TransitionTask(
        cruise_speed_m_s=jnp.asarray(cruise_speed_m_s),
        altitude_m=jnp.asarray(altitude_m),
        heading_rad=jnp.asarray(heading_rad),
        airspeed_weight=jnp.asarray(airspeed_weight),
        altitude_weight=jnp.asarray(altitude_weight),
        heading_weight=jnp.asarray(heading_weight),
        rate_weight=jnp.asarray(rate_weight),
        effort_weight=jnp.asarray(effort_weight),
    )


def trimmed_reference(
    model: AircraftModel,
    task: TrackingTask,
    environment: Environment | None = None,
    **trim_kwargs,
) -> ReferenceFlight:
    """Trim the model in the task's reference flight (a host-side solve, done once)."""

    environment = standard_environment() if environment is None else environment
    condition = StraightFlightCondition(
        airspeed_m_s=float(task.airspeed_m_s),
        heading_rad=float(task.heading_rad),
        altitude_m=float(task.altitude_m),
    )
    result = trim_straight_flight(model, condition, environment=environment, **trim_kwargs)
    if not result.success:
        raise ValueError(f"reference trim failed: {result.message}")
    return ReferenceFlight(state=result.state, control=result.control, environment=environment)


def hover_reference(
    model: AircraftModel, task: HoverTask | TransitionTask, environment: Environment | None = None
) -> ReferenceFlight:
    """The static hover of a tailsitter: nose up, belly toward the azimuth, throttle balancing
    weight from the static thrust map, elevons neutral. Not a trim (a wing in its own propwash
    carries a small camber force), so hold it with feedback. A transition task hovers at its
    altitude facing its heading."""

    environment = standard_environment() if environment is None else environment
    position, azimuth = task.hover_position_ned, task.hover_azimuth_rad
    up = -normalize(environment.gravity)
    attitude = thrust_direction_attitude(up, azimuth)
    weight = model.mass * safe_norm(environment.gravity)
    throttle = hover_throttle(model, weight, environment.density)
    state = zero_state(model)
    state = state._replace(
        rigid_body=state.rigid_body._replace(position=position, attitude=attitude)
    )
    control = ControlInput(propeller=throttle, channel=jnp.zeros(model.n_control_channels))
    state = equilibrate_internal_state(model, state, control, environment)
    return ReferenceFlight(state=state, control=control, environment=environment)


def reference_speed(task: Task) -> Array:
    """Speed that normalises velocity observations: the task's own reference speed."""

    return task.reference_speed()


def _nose_heading(attitude_xyzw: Array) -> Array:
    nose = quaternion_rotate(attitude_xyzw, jnp.array([1.0, 0.0, 0.0]))
    return jnp.arctan2(nose[..., 1], nose[..., 0])


def _belly_azimuth(attitude_xyzw: Array) -> Array:
    belly = quaternion_rotate(attitude_xyzw, jnp.array([0.0, 0.0, 1.0]))
    return jnp.arctan2(belly[..., 1], belly[..., 0])
