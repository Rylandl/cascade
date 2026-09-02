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
    coordinated_turn_rates,
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
    camber lift of a wing sitting in its own propwash. ``wing_speed_m_s`` is the airspeed at
    which the wing is credited with carrying the whole weight: the thrust law subtracts
    ``(airspeed / wing_speed)^2`` of gravity before balancing, so a tilted, fast aircraft is not
    driven forward by thrust the wing already supplies. Infinity disables the credit.
    ``position_error_limit_m`` clips the position error seen by the proportional and integral
    terms, so far from the setpoint the loop tracks the commanded velocity rather than lunging
    at a stale position (a schedule the aircraft fell behind during forward flight, say).
    """

    position_kp: Array
    position_ki: Array
    velocity_kp: Array
    integral_limit: Array
    acceleration_limit: Array
    tilt_limit: Array
    wing_speed_m_s: Array
    position_error_limit_m: Array


class HoverState(NamedTuple):
    position_integral: Array


class HoverSetpoint(NamedTuple):
    """Where to be, how fast to move, which azimuth the belly faces, and a scheduled tilt.

    ``tilt_forward_rad`` rotates the thrust axis toward the azimuth on top of the limited
    feedback tilt. It is how a transition is flown: a pitch ramp at high throttle carries the
    airframe along its thrust-borne trim branch until the forward-flight loops take over.
    """

    position_ned: Array
    velocity_ned: Array
    azimuth_rad: Array
    tilt_forward_rad: Array = jnp.asarray(0.0)


def default_hover_gains() -> HoverGains:
    return HoverGains(
        position_kp=jnp.asarray(2.0),
        position_ki=jnp.asarray(1.0),
        velocity_kp=jnp.asarray(2.5),
        integral_limit=jnp.asarray(2.0),
        acceleration_limit=jnp.asarray(5.0),
        tilt_limit=jnp.asarray(0.5),
        wing_speed_m_s=jnp.asarray(jnp.inf),
        position_error_limit_m=jnp.asarray(1.0),
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
    z_body = normalize(reference - jnp.sum(reference * x_body, axis=-1, keepdims=True) * x_body)
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
    position_error = jnp.clip(
        setpoint.position_ned - rigid_body.position,
        -gains.position_error_limit_m,
        gains.position_error_limit_m,
    )
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
    # Scheduled tilt toward the azimuth: Rodrigues rotation about the horizontal axis
    # perpendicular to travel, applied after the feedback so the feedback keeps its own limit.
    azimuth = setpoint.azimuth_rad
    along = jnp.stack((jnp.cos(azimuth), jnp.sin(azimuth), jnp.zeros_like(azimuth)), axis=-1)
    axis = normalize(jnp.cross(up, along, axis=-1))
    tilt = setpoint.tilt_forward_rad[..., None]
    direction = (
        direction * jnp.cos(tilt)
        + jnp.cross(axis, direction, axis=-1) * jnp.sin(tilt)
        + axis * jnp.sum(axis * direction, axis=-1, keepdims=True) * (1.0 - jnp.cos(tilt))
    )
    attitude = thrust_direction_attitude(direction, setpoint.azimuth_rad)
    # Thrust from vertical balance: the up component of the tilted thrust must supply the up
    # component of the required specific force. Capped where the axis nears horizontal.
    cosine = jnp.maximum(jnp.sum(direction * up, axis=-1), jnp.cos(1.4))
    airspeed = safe_norm(rigid_body.velocity - environment.wind)
    lift_fraction = jnp.minimum(jnp.square(airspeed / gains.wing_speed_m_s), 1.0)
    gravity = safe_norm(environment.gravity)
    vertical_demand = jnp.sum(specific_force * up, axis=-1) - gravity * lift_fraction
    thrust_total = model.mass * jnp.maximum(vertical_demand, 0.05 * gravity) / cosine
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
    tilt_at_cruise_rad: float = 1.0,
) -> HoverSetpoint:
    """Time-major hover setpoints that hold, then accelerate along a heading to cruise speed.

    Velocity ramps linearly at ``acceleration_m_s2`` after ``hold_steps`` and saturates at
    ``cruise_speed_m_s``; position is the integral of that velocity from the start point; the
    wing's belly faces the heading throughout; and the scheduled forward tilt rises in
    proportion to the commanded speed up to ``tilt_at_cruise_rad``. The feedback then only
    corrects about that ramp.
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
        tilt_forward_rad=tilt_at_cruise_rad * speed / cruise_speed_m_s,
    )


def hover_azimuth_across_wind(
    wind_ned: Array, preferred_azimuth_rad: Array, minimum_wind_m_s: float = 0.5
) -> Array:
    """Hover azimuth (belly direction) that puts the wing's span into the wind.

    A tailsitter hovering broadside to the wind presents its whole wing as a flat plate, and
    leaning into the wind exposes more of it; edge-on, the same wind is a small side load the
    differential-thrust yaw loop holds easily. Of the two azimuths perpendicular to the wind
    this returns the one nearer ``preferred_azimuth_rad`` (the heading the next transition will
    use, say). Below ``minimum_wind_m_s`` the wind direction is meaningless and the preferred
    azimuth is returned; the blend between the two is smooth in the wind speed.
    """

    north, east = wind_ned[..., 0], wind_ned[..., 1]
    wind_direction = jnp.arctan2(east, north)
    candidates = wind_direction[..., None] + jnp.array([0.5, -0.5]) * jnp.pi
    offsets = jnp.arctan2(
        jnp.sin(candidates - preferred_azimuth_rad[..., None]),
        jnp.cos(candidates - preferred_azimuth_rad[..., None]),
    )
    nearer = jnp.argmin(jnp.abs(offsets), axis=-1)
    across = (
        preferred_azimuth_rad + jnp.take_along_axis(offsets, nearer[..., None], axis=-1)[..., 0]
    )
    speed = jnp.hypot(north, east)
    weight = jax.nn.sigmoid((speed - minimum_wind_m_s) / (0.1 * minimum_wind_m_s))
    return preferred_azimuth_rad + weight * (across - preferred_azimuth_rad)


def trapezoid_speed_profile(
    steps: int,
    dt: float,
    *,
    hold_steps: int,
    cruise_speed_m_s: Array,
    acceleration_m_s2: Array,
    cruise_steps: int,
    deceleration_m_s2: Array,
) -> Array:
    """Commanded speed over time: hold, accelerate, cruise, decelerate, hold.

    The cruise phase lasts ``cruise_steps`` after the acceleration ramp reaches cruise speed;
    the deceleration ramp then runs back to zero and the schedule holds there.
    """

    time = jnp.arange(steps) * dt
    ramp_start = hold_steps * dt
    cruise_start = ramp_start + cruise_speed_m_s / acceleration_m_s2
    decel_start = cruise_start + cruise_steps * dt
    decel_end = decel_start + cruise_speed_m_s / deceleration_m_s2
    speed = jnp.where(
        time < cruise_start,
        acceleration_m_s2 * jnp.maximum(time - ramp_start, 0.0),
        jnp.where(
            time < decel_start,
            cruise_speed_m_s,
            cruise_speed_m_s
            - deceleration_m_s2 * jnp.minimum(time - decel_start, decel_end - decel_start),
        ),
    )
    return jnp.clip(speed, 0.0, cruise_speed_m_s)


def speed_profile_schedule(
    speed_m_s: Array,
    dt: float,
    *,
    start_position_ned: Array,
    heading_rad: Array,
    cruise_speed_m_s: Array,
    tilt_at_cruise_rad: float = 1.0,
) -> HoverSetpoint:
    """Time-major hover setpoints that follow a commanded speed profile along a heading.

    ``heading_rad`` is a scalar or a time-major profile. Position is the running integral of
    the commanded velocity from the start point, the belly faces the heading, and the scheduled
    forward tilt is proportional to the commanded speed up to ``tilt_at_cruise_rad``.
    Decelerating the profile through the transition controller's switch airspeed hands the
    aircraft back to hover guidance, so a trapezoid profile flies a full hover, cruise, hover
    round trip, and a heading profile turns it in cruise.
    """

    steps = speed_m_s.shape[0]
    heading = jnp.broadcast_to(jnp.asarray(heading_rad), (steps,))
    along = jnp.stack((jnp.cos(heading), jnp.sin(heading), jnp.zeros_like(heading)), axis=-1)
    velocity = speed_m_s[:, None] * along
    return HoverSetpoint(
        position_ned=start_position_ned + jnp.cumsum(velocity, axis=0) * dt,
        velocity_ned=velocity,
        azimuth_rad=heading,
        tilt_forward_rad=tilt_at_cruise_rad * speed_m_s / cruise_speed_m_s,
    )


class TransitionController(NamedTuple):
    """Hover and forward-flight guidance sharing one attitude and rate stack.

    Both guidance laws run every step; their body-rate setpoints and throttles are blended by a
    smooth weight in airspeed centred on ``switch_airspeed_m_s``. Blending in the rate domain
    keeps the attitude handover free of quaternion interpolation and continuous in time.

    ``differential_thrust`` (one entry per propeller) is the throttle increment per unit of the
    rate loop's body-z command: the yaw control of a twin-motor tailsitter. In hover body z is
    the belly normal, so this is what holds the wing's plane against a spanwise wind; in
    forward flight it is the rudder a flying wing does not have.
    """

    channels: ChannelMap
    rate: RateGains
    attitude: AttitudeGains
    hover: HoverGains
    forward: GuidanceGains
    switch_airspeed_m_s: Array
    switch_width_m_s: Array
    differential_thrust: Array


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


def forward_weight(
    controller: TransitionController, hover_setpoint: HoverSetpoint, aircraft_state, environment
) -> Array:
    """Blend weight of the forward-flight law.

    The product of two sigmoids about the switch airspeed: one in the measured airspeed, so the
    forward law never flies a wing that is not yet flying, and one in the commanded hover speed,
    so the schedule owns the mode. Decelerating the commanded speed through the switch hands the
    aircraft back to hover guidance while it is still fast, which is the back-transition: a
    pitch-up onto the thrust-borne branch, then a decelerating hover ramp.
    """

    airspeed = safe_norm(aircraft_state.rigid_body.velocity - environment.wind)
    commanded = safe_norm(hover_setpoint.velocity_ned)
    measured_gate = jax.nn.sigmoid(
        (airspeed - controller.switch_airspeed_m_s) / controller.switch_width_m_s
    )
    commanded_gate = jax.nn.sigmoid(
        (commanded - controller.switch_airspeed_m_s) / controller.switch_width_m_s
    )
    return measured_gate * commanded_gate


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

    weight = forward_weight(controller, hover_setpoint, aircraft_state, environment)
    # Each guidance law's integrator advances in proportion to its blend weight, so neither can
    # wind up while the other is flying the aircraft.
    hover_attitude, hover_throttle_command, hover_state = hover_guidance(
        model,
        controller.hover,
        state.hover,
        hover_setpoint,
        aircraft_state,
        environment,
        (1.0 - weight) * dt,
    )
    forward_attitude, forward_throttle, forward_state = guidance_controller(
        controller.forward,
        state.forward,
        forward_setpoint,
        aircraft_state,
        environment,
        weight * dt,
    )
    measured = aircraft_state.rigid_body.attitude
    hover_rates = attitude_controller(controller.attitude, hover_attitude, measured)
    forward_rates = attitude_controller(
        controller.attitude, forward_attitude, measured
    ) + coordinated_turn_rates(
        forward_attitude,
        safe_norm(aircraft_state.rigid_body.velocity - environment.wind),
        safe_norm(environment.gravity),
    )
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
    propeller = jnp.clip(throttle + command[..., 2:3] * controller.differential_thrust, 0.0, 1.0)
    control = ControlInput(propeller=propeller, channel=channels)
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
    environments: Environment | None = None,
) -> tuple[tuple[AircraftState, TransitionState], tuple[AircraftState, ControlInput, Array]]:
    """Scan the transition controller and the plant over time-major setpoint sequences.

    Returns the final aircraft and controller states and the time-major trajectory of aircraft
    states, control inputs, and forward-blend weights. ``environments`` is an optional
    time-major ``Environment`` (a gust sequence from ``cascade.gusts``, say) that replaces the
    constant ``environment`` step by step. Differentiable end to end, so a schedule or a gain
    can be tuned by gradient through the whole transition.
    """

    def scan_step(carry, inputs):
        aircraft, controller_state = carry
        hover_setpoint, forward_setpoint, step_environment = inputs
        current = environment if step_environment is None else step_environment
        control, next_controller_state, weight = transition_step(
            model,
            controller,
            controller_state,
            hover_setpoint,
            forward_setpoint,
            aircraft,
            current,
            dt,
        )
        next_aircraft = step(model, aircraft, control, current, dt)
        return (next_aircraft, next_controller_state), (next_aircraft, control, weight)

    return jax.lax.scan(
        scan_step, (aircraft_state, state), (hover_setpoints, forward_setpoints, environments)
    )


def tailsitter_reference_controller(spec: AircraftSpec) -> TransitionController:
    """Transition controller for :func:`cascade.tailsitter_reference`.

    Gains were set by closed-loop step responses in hover and at 7 m/s cruise; see
    ``docs/tailsitter.md``. Elevator is trailing-edge-down positive and therefore nose-down, so
    its pitch role is negative, as on the other packaged aircraft.
    """

    return TransitionController(
        channels=channel_map(spec, roles={"aileron": "roll", "elevator": "-pitch"}, limits=1.0),
        rate=RateGains(
            kp=jnp.array([0.25, 0.25, 0.15]),
            ki=jnp.array([0.5, 0.5, 0.15]),
            kd=jnp.array([0.0, 0.0, 0.0]),
            integral_limit=jnp.array([0.4, 0.4, 0.3]),
            feedforward=jnp.array([0.0, 0.0, 0.0]),
        ),
        attitude=AttitudeGains(
            kp=jnp.array([4.0, 4.0, 2.0]), rate_limit=jnp.array([4.0, 4.0, 2.0])
        ),
        hover=default_hover_gains()._replace(wing_speed_m_s=jnp.asarray(9.0)),
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
        differential_thrust=jnp.array([0.5, -0.5]),
    )
