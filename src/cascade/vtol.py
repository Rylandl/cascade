"""Hover guidance for thrust-borne flight: position and velocity to attitude and throttle.

A tailsitter hovers on its propellers with body ``x`` pointing up. This module turns a position
and velocity setpoint into the thrust direction the airframe must point at and the throttle that
produces the required thrust magnitude, leaving the wing plane free to face a chosen azimuth.
The output feeds the attitude and rate loops of :mod:`cascade.control`, so a transition is a
change of setpoint schedule, not of controller structure. Everything is pure and batched.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.control import (
    AttitudeGains,
    ChannelMap,
    GuidanceGains,
    GuidanceSetpoint,
    GuidanceState,
    RateGains,
    RateState,
    attitude_controller,
    channel_map,
    guidance_controller,
    rate_controller,
)
from cascade.integration import StepFunction, rk4_step
from cascade.math import normalize, quaternion_from_matrix, safe_norm
from cascade.model import AircraftModel
from cascade.spec import AircraftSpec
from cascade.state import AircraftState, ControlInput, Environment


class HoverGains(NamedTuple):
    """Position, velocity, and integral gains and the limits that keep hover commands sane.

    The integral term removes the standing offset a constant force leaves behind, such as the
    camber lift of a wing sitting in its own propwash.
    """

    position_kp: Array
    position_ki: Array
    velocity_kp: Array
    integral_limit: Array
    acceleration_limit: Array
    tilt_limit: Array


class HoverState(NamedTuple):
    position_integral: Array


class HoverSetpoint(NamedTuple):
    """Where to be, how fast to move, and which azimuth the wing's belly should face."""

    position_ned: Array
    velocity_ned: Array
    azimuth_rad: Array


def default_hover_gains() -> HoverGains:
    return HoverGains(
        position_kp=jnp.asarray(2.0),
        position_ki=jnp.asarray(1.0),
        velocity_kp=jnp.asarray(2.5),
        integral_limit=jnp.asarray(2.0),
        acceleration_limit=jnp.asarray(5.0),
        tilt_limit=jnp.asarray(0.5),
    )


def initial_hover_state(batch_shape: tuple[int, ...] = ()) -> HoverState:
    return HoverState(position_integral=jnp.zeros((*batch_shape, 3)))


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
    hover_state: HoverState,
    setpoint: HoverSetpoint,
    state: AircraftState,
    environment: Environment,
    dt: float,
) -> tuple[Array, Array, HoverState]:
    """Attitude setpoint (``xyzw``), per-motor throttle, and next state for thrust-borne flight.

    The commanded acceleration is a limited proportional-integral-derivative law on position and
    velocity; the required specific force is that acceleration minus gravity, its direction
    gives the thrust axis (tilt-limited about vertical) and its magnitude times mass gives the
    total thrust. The integral is clipped for anti-windup.
    """

    rigid_body = state.rigid_body
    position_error = setpoint.position_ned - rigid_body.position
    position_integral = jnp.clip(
        hover_state.position_integral + position_error * dt,
        -gains.integral_limit,
        gains.integral_limit,
    )
    acceleration = (
        gains.position_kp * position_error
        + gains.position_ki * position_integral
        + gains.velocity_kp * (setpoint.velocity_ned - rigid_body.velocity)
    )
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
    return attitude, throttle, HoverState(position_integral=position_integral)


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


class TransitionController(NamedTuple):
    """Hover and forward-flight guidance sharing one attitude and rate stack.

    Both guidance laws run every step; their body-rate setpoints and throttles are blended by a
    smooth weight in airspeed centred on ``switch_airspeed_m_s``. Blending in the rate domain
    keeps the attitude handover free of quaternion interpolation and continuous in time.
    """

    channels: ChannelMap
    rate: RateGains
    attitude: AttitudeGains
    hover: HoverGains
    forward: GuidanceGains
    switch_airspeed_m_s: Array
    switch_width_m_s: Array


class TransitionState(NamedTuple):
    rate: RateState
    hover: HoverState
    forward: GuidanceState


def initial_transition_state(aircraft_state: AircraftState) -> TransitionState:
    batch_shape = aircraft_state.rigid_body.attitude.shape[:-1]
    axes = (*batch_shape, 3)
    return TransitionState(
        rate=RateState(integral=jnp.zeros(axes), previous_error=jnp.zeros(axes)),
        hover=initial_hover_state(batch_shape),
        forward=GuidanceState(airspeed_integral=jnp.zeros(batch_shape)),
    )


def forward_weight(controller: TransitionController, aircraft_state, environment) -> Array:
    """Blend weight of the forward-flight law, a sigmoid in airspeed."""

    airspeed = safe_norm(aircraft_state.rigid_body.velocity - environment.wind)
    return jax.nn.sigmoid(
        (airspeed - controller.switch_airspeed_m_s) / controller.switch_width_m_s
    )


def transition_step(
    model: AircraftModel,
    controller: TransitionController,
    state: TransitionState,
    hover_setpoint: HoverSetpoint,
    forward_setpoint: GuidanceSetpoint,
    aircraft_state: AircraftState,
    environment: Environment,
    dt: float,
) -> tuple[ControlInput, TransitionState, Array]:
    """One control step: blended guidance, attitude loop, rate loop, channel mapping."""

    weight = forward_weight(controller, aircraft_state, environment)
    # Each guidance law's integrator advances in proportion to its blend weight, so neither can
    # wind up while the other is flying the aircraft.
    hover_attitude, hover_throttle_command, hover_state = hover_guidance(
        model, controller.hover, state.hover, hover_setpoint, aircraft_state, environment,
        (1.0 - weight) * dt,
    )
    forward_attitude, forward_throttle, forward_state = guidance_controller(
        controller.forward, state.forward, forward_setpoint, aircraft_state, environment,
        weight * dt,
    )
    measured = aircraft_state.rigid_body.attitude
    hover_rates = attitude_controller(controller.attitude, hover_attitude, measured)
    forward_rates = attitude_controller(controller.attitude, forward_attitude, measured)
    rate_setpoint = (1.0 - weight[..., None]) * hover_rates + weight[..., None] * forward_rates
    throttle = (1.0 - weight[..., None]) * hover_throttle_command + weight[
        ..., None
    ] * forward_throttle
    command, rate_state = rate_controller(
        controller.rate, state.rate, rate_setpoint, aircraft_state.rigid_body.angular_velocity, dt
    )
    channels = jnp.clip(
        jnp.einsum("cr,...r->...c", controller.channels.matrix, command),
        -controller.channels.limit,
        controller.channels.limit,
    )
    control = ControlInput(propeller=throttle, channel=channels)
    next_state = TransitionState(rate=rate_state, hover=hover_state, forward=forward_state)
    return control, next_state, weight


def transition_rollout(
    model: AircraftModel,
    controller: TransitionController,
    aircraft_state: AircraftState,
    state: TransitionState,
    hover_setpoints: HoverSetpoint,
    forward_setpoints: GuidanceSetpoint,
    environment: Environment,
    dt: float,
    *,
    step: StepFunction = rk4_step,
) -> tuple[tuple[AircraftState, TransitionState], tuple[AircraftState, ControlInput, Array]]:
    """Scan the transition controller and the plant over time-major setpoint sequences.

    Returns the final aircraft and controller states and the time-major trajectory of aircraft
    states, control inputs, and forward-blend weights. Differentiable end to end, so a schedule
    or a gain can be tuned by gradient through the whole transition.
    """

    def scan_step(carry, setpoints):
        aircraft, controller_state = carry
        hover_setpoint, forward_setpoint = setpoints
        control, next_controller_state, weight = transition_step(
            model, controller, controller_state, hover_setpoint, forward_setpoint, aircraft,
            environment, dt,
        )
        next_aircraft = step(model, aircraft, control, environment, dt)
        return (next_aircraft, next_controller_state), (next_aircraft, control, weight)

    return jax.lax.scan(scan_step, (aircraft_state, state), (hover_setpoints, forward_setpoints))


def tailsitter_reference_controller(spec: AircraftSpec) -> TransitionController:
    """Transition controller for :func:`cascade.tailsitter_reference`.

    Gains were set by closed-loop step responses in hover and at 7 m/s cruise; see
    ``docs/tailsitter.md``. Elevator is trailing-edge-down positive and therefore nose-down, so
    its pitch role is negative, as on the other packaged aircraft.
    """

    return TransitionController(
        channels=channel_map(spec, roles={"aileron": "roll", "elevator": "-pitch"}, limits=1.0),
        rate=RateGains(
            kp=jnp.array([0.25, 0.25, 0.0]),
            ki=jnp.array([0.5, 0.5, 0.0]),
            kd=jnp.array([0.0, 0.0, 0.0]),
            integral_limit=jnp.array([0.4, 0.4, 0.0]),
            feedforward=jnp.array([0.0, 0.0, 0.0]),
        ),
        attitude=AttitudeGains(
            kp=jnp.array([4.0, 4.0, 2.0]), rate_limit=jnp.array([4.0, 4.0, 2.0])
        ),
        hover=default_hover_gains(),
        forward=GuidanceGains(
            airspeed_kp=jnp.asarray(0.1),
            airspeed_ki=jnp.asarray(0.05),
            throttle_trim=jnp.asarray(0.58),
            throttle_limits=jnp.array([0.2, 1.0]),
            altitude_kp=jnp.asarray(0.5),
            climb_rate_limit=jnp.asarray(1.5),
            pitch_trim=jnp.asarray(0.115),
            pitch_limits=jnp.array([-0.35, 0.6]),
            heading_kp=jnp.asarray(1.0),
            bank_limit=jnp.asarray(0.5),
            airspeed_pitch_kp=jnp.asarray(0.05),
        ),
        switch_airspeed_m_s=jnp.asarray(6.5),
        switch_width_m_s=jnp.asarray(0.5),
    )
