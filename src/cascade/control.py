"""A PX4-style fixed-wing control cascade: rate, attitude, and guidance loops.

Every function here is a pure function of a PyTree of gains and a PyTree of controller state, so
the whole cascade is ``jit``/``vmap``/``grad`` compatible and batches over worlds with the same
leading dimensions as :class:`cascade.state.AircraftState`. Nothing branches in Python on an array
value; loop scheduling uses ``jnp.where`` so the traced program shape never depends on
``step_index``.

Loop structure, innermost first:

- **Rate loop** (:func:`rate_controller`): PID with feedforward on body roll/pitch/yaw rate,
  producing unit ``[roll, pitch, yaw]`` commands before channel mapping.
- **Attitude loop** (:func:`attitude_controller`): proportional-only, converting a quaternion
  attitude error into a body-rate setpoint for the rate loop. There is no attitude integrator; a
  steady attitude bias must be trimmed out or corrected by the rate loop's own integral term.
- **Guidance loop** (:func:`guidance_controller`): airspeed, altitude, and heading hold, each a
  single decoupled proportional (or PI) stage rather than energy-based (TECS) control. See its
  docstring for the exact simplifications.

:class:`CascadeController` composes the three loops with per-level update periods (in simulation
steps) and :func:`cascade_step` runs one scheduled update of all three, holding each level's
output between its own updates. :func:`closed_loop_rollout` scans that step together with plant
integration over a time-major sequence of guidance setpoints.

Conventions carried over from the rest of Cascade: body FRD, world NED, scalar-last ``xyzw``
quaternions rotating body into world (see :mod:`cascade.math`). Channels are the aircraft's
abstract control channels in the specification's own units (normalized ``[-1, 1]`` or radians);
this module takes per-channel limits from the caller rather than assuming units.

Positive-command conventions, enforced by :class:`ChannelMap` rather than assumed: positive roll
command demands positive body roll rate (right wing down), positive pitch command demands positive
body pitch rate (nose up), positive yaw command demands positive body yaw rate (nose right).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.integration import StepFunction, rk4_step
from cascade.math import (
    quaternion_conjugate,
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_to_rotvec,
    safe_norm,
)
from cascade.model import AircraftModel
from cascade.spec import AircraftSpec
from cascade.state import AircraftState, ControlInput, Environment

_AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}


class ChannelMap(NamedTuple):
    """Static mapping from roll/pitch/yaw unit commands to an aircraft's abstract channels."""

    matrix: Array
    limit: Array


def channel_map(
    spec: AircraftSpec, roles: Mapping[str, str], limits: Mapping[str, float] | float
) -> ChannelMap:
    """Build a :class:`ChannelMap` from named channel roles.

    ``roles`` assigns each named channel a role of ``"roll"``, ``"pitch"``, or ``"yaw"``; a
    channel absent from ``roles`` gets an all-zero row and never receives a command. Prefix a role
    with ``-`` (for example ``"-pitch"``) when a positive channel command drives that axis
    negative for this airframe: the sign is a property of the airframe's ``control_map_rad`` (or,
    for a coefficient-table aircraft, its ``body.deflection_map``), not of the channel's name, and
    must be checked per aircraft rather than assumed — for example by sweeping ``dCl/d(aileron)``
    and ``dCm/d(elevator)`` with :func:`cascade.aerodynamic_sweep` or by a short open-loop rollout
    from trim. ``limits`` is either one clip shared by every channel or a per-channel-name mapping,
    in the specification's own channel units.
    """

    channels = spec.control_channels
    matrix = [[0.0, 0.0, 0.0] for _ in channels]
    for name, role in roles.items():
        if name not in channels:
            raise ValueError(f"unknown control channel {name!r}; spec channels are {channels}")
        sign = -1.0 if role.startswith("-") else 1.0
        axis = role[1:] if role.startswith("-") else role
        if axis not in _AXIS_INDEX:
            raise ValueError(f"unknown control role {role!r}; expected roll, pitch, or yaw")
        matrix[channels.index(name)][_AXIS_INDEX[axis]] = sign
    if isinstance(limits, Mapping):
        limit = [float(limits[name]) for name in channels]
    else:
        limit = [float(limits)] * len(channels)
    return ChannelMap(matrix=jnp.asarray(matrix), limit=jnp.asarray(limit))


class RateGains(NamedTuple):
    """PID-plus-feedforward gains per body axis ``[roll, pitch, yaw]``."""

    kp: Array
    ki: Array
    kd: Array
    integral_limit: Array
    feedforward: Array


class RateState(NamedTuple):
    integral: Array
    previous_error: Array


def rate_controller(
    gains: RateGains,
    state: RateState,
    rate_setpoint: Array,
    rate_measured: Array,
    dt: float,
) -> tuple[Array, RateState]:
    """PID-plus-feedforward body-rate control, returning unit commands before channel mapping.

    The integral is clipped to ``gains.integral_limit`` every step (simple anti-windup); the
    derivative is a plain finite difference of the error, matching ``RateState`` carrying only
    ``integral`` and ``previous_error``. ``feedforward`` scales the setpoint itself directly into
    the output, standard practice for tracking a fast-moving rate command.
    """

    error = rate_setpoint - rate_measured
    integral = jnp.clip(
        state.integral + error * dt, -gains.integral_limit, gains.integral_limit
    )
    derivative = (error - state.previous_error) / dt
    output = (
        gains.kp * error
        + gains.ki * integral
        + gains.kd * derivative
        + gains.feedforward * rate_setpoint
    )
    return output, RateState(integral=integral, previous_error=error)


class AttitudeGains(NamedTuple):
    kp: Array
    rate_limit: Array


def attitude_controller(
    gains: AttitudeGains, attitude_setpoint_xyzw: Array, attitude_measured_xyzw: Array
) -> Array:
    """Proportional attitude control: shortest-rotation error to a saturated body-rate setpoint.

    There is no integral term, so a persistent attitude bias (trim drift, an un-modeled moment)
    is left for the rate loop's own integrator or for an updated trim, not corrected here.
    """

    error_quaternion = quaternion_multiply(
        quaternion_conjugate(attitude_measured_xyzw), attitude_setpoint_xyzw
    )
    rate_setpoint = gains.kp * quaternion_to_rotvec(error_quaternion)
    return jnp.clip(rate_setpoint, -gains.rate_limit, gains.rate_limit)


class GuidanceGains(NamedTuple):
    airspeed_kp: Array
    airspeed_ki: Array
    throttle_trim: Array
    throttle_limits: Array
    altitude_kp: Array
    climb_rate_limit: Array
    pitch_trim: Array
    pitch_limits: Array
    heading_kp: Array
    bank_limit: Array
    airspeed_pitch_kp: Array


class GuidanceState(NamedTuple):
    airspeed_integral: Array


class GuidanceSetpoint(NamedTuple):
    airspeed_m_s: Array
    altitude_m: Array
    heading_rad: Array


def guidance_controller(
    gains: GuidanceGains,
    state: GuidanceState,
    setpoint: GuidanceSetpoint,
    aircraft_state: AircraftState,
    environment: Environment,
    dt: float,
) -> tuple[Array, Array, GuidanceState]:
    """Decoupled airspeed/altitude/heading guidance, producing an attitude setpoint and throttle.

    This is three independent single-input single-output loops, not TECS: altitude and airspeed
    do not share energy, so a climb eats into airspeed (and vice versa) exactly as a real
    decoupled autopilot without a throttle/pitch energy mix would. Specifically:

    - Altitude error feeds a proportional climb-rate command, saturated to ``climb_rate_limit``.
      The climb-rate command is converted to a pitch offset by the small-angle relation
      ``climb_rate / airspeed`` (flight-path angle, not angle of attack) about ``pitch_trim``, and
      reduced by ``airspeed_pitch_kp`` times any airspeed deficit below the commanded airspeed as
      a stall-protection term, then saturated to ``pitch_limits``.
    - Airspeed error feeds a PI throttle about ``throttle_trim``, saturated to ``throttle_limits``;
      the integral is clamped to the value that alone would span the throttle range, a cheap
      anti-windup that needs no extra gain field.
    - Heading error (wrapped to ``[-pi, pi]``) feeds a proportional bank-angle command, saturated
      to ``bank_limit``. The attitude setpoint's yaw is set to the *current* measured yaw, not the
      commanded heading, so the attitude loop only closes roll and pitch and never fights the
      heading loop directly; the bank angle alone produces the coordinated turn.

    There is no wind feedforward: crosswind or a headwind/tailwind shift is corrected only through
    the airspeed and heading errors it eventually causes.
    """

    rigid_body = aircraft_state.rigid_body
    altitude = -rigid_body.position[..., 2]
    altitude_error = setpoint.altitude_m - altitude
    climb_rate_command = jnp.clip(
        gains.altitude_kp * altitude_error, -gains.climb_rate_limit, gains.climb_rate_limit
    )

    air_velocity_world = rigid_body.velocity - environment.wind
    airspeed = safe_norm(air_velocity_world)
    airspeed_error = setpoint.airspeed_m_s - airspeed

    throttle_span = gains.throttle_limits[..., 1] - gains.throttle_limits[..., 0]
    integral_limit = throttle_span / jnp.maximum(gains.airspeed_ki, 1e-6)
    airspeed_integral = jnp.clip(
        state.airspeed_integral + airspeed_error * dt, -integral_limit, integral_limit
    )
    throttle_command = (
        gains.throttle_trim
        + gains.airspeed_kp * airspeed_error
        + gains.airspeed_ki * airspeed_integral
    )
    throttle = jnp.clip(
        throttle_command, gains.throttle_limits[..., 0], gains.throttle_limits[..., 1]
    )[..., None]

    speed_floor = jnp.maximum(airspeed, 3.0)
    stall_margin = jnp.maximum(setpoint.airspeed_m_s - airspeed, 0.0)
    pitch_command = (
        gains.pitch_trim
        + climb_rate_command / speed_floor
        - gains.airspeed_pitch_kp * stall_margin
    )
    pitch_setpoint = jnp.clip(
        pitch_command, gains.pitch_limits[..., 0], gains.pitch_limits[..., 1]
    )

    heading_measured = _yaw_from_quaternion(rigid_body.attitude)
    heading_error = _wrap_to_pi(setpoint.heading_rad - heading_measured)
    bank_setpoint = jnp.clip(
        gains.heading_kp * heading_error, -gains.bank_limit, gains.bank_limit
    )

    attitude_setpoint = quaternion_from_euler(bank_setpoint, pitch_setpoint, heading_measured)
    return attitude_setpoint, throttle, GuidanceState(airspeed_integral=airspeed_integral)


def _wrap_to_pi(angle: Array) -> Array:
    return jnp.mod(angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def _yaw_from_quaternion(attitude_xyzw: Array) -> Array:
    """Body yaw of a scalar-last body-to-world quaternion, aerospace ZYX convention."""

    x, y, z, w = (attitude_xyzw[..., index] for index in range(4))
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class CascadeController(NamedTuple):
    """The full cascade: three loops' gains plus their update periods in simulation steps."""

    channels: ChannelMap
    rate: RateGains
    attitude: AttitudeGains
    guidance: GuidanceGains
    rate_period: int
    attitude_period: int
    guidance_period: int


class CascadeState(NamedTuple):
    rate: RateState
    guidance: GuidanceState
    held_attitude_setpoint: Array
    held_rate_setpoint: Array
    held_throttle: Array
    held_channels: Array
    step_index: Array


def initial_cascade_state(
    controller: CascadeController, aircraft_state: AircraftState, control: ControlInput
) -> CascadeState:
    """Zero every integrator and hold ``control`` (for example a trim) until the first update."""

    batch_shape = aircraft_state.rigid_body.attitude.shape[:-1]
    axis_shape = (*batch_shape, 3)
    propeller_shape = (*batch_shape, control.propeller.shape[-1])
    channel_shape = (*batch_shape, control.channel.shape[-1])
    return CascadeState(
        rate=RateState(integral=jnp.zeros(axis_shape), previous_error=jnp.zeros(axis_shape)),
        guidance=GuidanceState(airspeed_integral=jnp.zeros(batch_shape)),
        held_attitude_setpoint=jnp.broadcast_to(
            aircraft_state.rigid_body.attitude, (*batch_shape, 4)
        ),
        held_rate_setpoint=jnp.zeros(axis_shape),
        held_throttle=jnp.broadcast_to(control.propeller, propeller_shape),
        held_channels=jnp.broadcast_to(control.channel, channel_shape),
        step_index=jnp.zeros((), dtype=jnp.int32),
    )


def cascade_step(
    controller: CascadeController,
    cascade_state: CascadeState,
    setpoint: GuidanceSetpoint,
    aircraft_state: AircraftState,
    environment: Environment,
    dt: float,
) -> tuple[ControlInput, CascadeState]:
    """Advance the cascade by one simulation step, updating each loop on its own schedule.

    Every loop is evaluated every call (nothing here branches on an array value); a
    ``jnp.where(step_index % period == 0, fresh, held)`` selects whether a level's fresh output
    replaces its held one this step, matching a rate-scheduled flight-control stack without
    changing the traced program shape. A level's own state (its integrator) advances in step with
    its own schedule, using that level's *elapsed* time (``period * dt``) rather than the
    simulation ``dt``, so integral gains are tuned against the rate the loop actually runs at.
    """

    step_index = cascade_state.step_index
    guidance_due = (step_index % controller.guidance_period) == 0
    attitude_due = (step_index % controller.attitude_period) == 0
    rate_due = (step_index % controller.rate_period) == 0

    fresh_attitude_setpoint, fresh_throttle, fresh_guidance_state = guidance_controller(
        controller.guidance,
        cascade_state.guidance,
        setpoint,
        aircraft_state,
        environment,
        dt * controller.guidance_period,
    )
    attitude_setpoint = jnp.where(
        guidance_due, fresh_attitude_setpoint, cascade_state.held_attitude_setpoint
    )
    throttle = jnp.where(guidance_due, fresh_throttle, cascade_state.held_throttle)
    guidance_state = jax.tree.map(
        lambda fresh, held: jnp.where(guidance_due, fresh, held),
        fresh_guidance_state,
        cascade_state.guidance,
    )

    fresh_rate_setpoint = attitude_controller(
        controller.attitude, attitude_setpoint, aircraft_state.rigid_body.attitude
    )
    rate_setpoint = jnp.where(attitude_due, fresh_rate_setpoint, cascade_state.held_rate_setpoint)

    fresh_command, fresh_rate_state = rate_controller(
        controller.rate,
        cascade_state.rate,
        rate_setpoint,
        aircraft_state.rigid_body.angular_velocity,
        dt * controller.rate_period,
    )
    fresh_channels = jnp.clip(
        jnp.einsum("cr,...r->...c", controller.channels.matrix, fresh_command),
        -controller.channels.limit,
        controller.channels.limit,
    )
    channels = jnp.where(rate_due, fresh_channels, cascade_state.held_channels)
    rate_state = jax.tree.map(
        lambda fresh, held: jnp.where(rate_due, fresh, held), fresh_rate_state, cascade_state.rate
    )

    control = ControlInput(propeller=throttle, channel=channels)
    next_cascade_state = CascadeState(
        rate=rate_state,
        guidance=guidance_state,
        held_attitude_setpoint=attitude_setpoint,
        held_rate_setpoint=rate_setpoint,
        held_throttle=throttle,
        held_channels=channels,
        step_index=step_index + 1,
    )
    return control, next_cascade_state


def closed_loop_rollout(
    model: AircraftModel,
    controller: CascadeController,
    aircraft_state: AircraftState,
    cascade_state: CascadeState,
    setpoints: GuidanceSetpoint,
    environment: Environment,
    dt: float,
    *,
    step: StepFunction = rk4_step,
    environments: Environment | None = None,
) -> tuple[tuple[AircraftState, CascadeState], tuple[AircraftState, ControlInput, CascadeState]]:
    """Scan a time-major guidance-setpoint sequence, closing the loop between plant and cascade.

    Each iteration evaluates :func:`cascade_step` at the *current* state, then advances the plant
    one ``step`` with the resulting control. Returns the final ``(state, cascade_state)`` and the
    post-step, time-major ``(state, control, cascade_state)`` trajectory. ``environments`` follows
    :func:`cascade.integration.rollout`'s convention: an optional time-major :class:`Environment`
    applied per interval like ``setpoints``, or ``None`` to hold the per-world ``environment``.
    """

    def scan_step(carry, inputs):
        state, controller_state = carry
        setpoint, step_environment = inputs
        active_environment = environment if step_environment is None else step_environment
        control, next_controller_state = cascade_step(
            controller, controller_state, setpoint, state, active_environment, dt
        )
        next_state = step(model, state, control, active_environment, dt)
        return (next_state, next_controller_state), (next_state, control, next_controller_state)

    return jax.lax.scan(scan_step, (aircraft_state, cascade_state), (setpoints, environments))


def aerobatic_reference_controller() -> CascadeController:
    """Tuned cascade for :func:`cascade.aerobatic_reference`, gains found by closed-loop step
    response at 12 m/s / 20 m from :func:`cascade.trim_straight_flight`.

    Aileron and rudder command roll and yaw directly (positive channel drives positive rate);
    elevator commands pitch with a flipped sign, since positive elevator channel is trailing-edge
    down and drives a nose-down (negative) pitch rate here, confirmed with
    :func:`cascade.aerodynamic_sweep` and a short open-loop rollout from trim. Rate gains:
    ``kp=[0.12, 0.20, 0.35]``, ``ki=[1.0, 2.0, 3.0]``, ``kd=[0.001, 0.005, 0.005]``,
    ``integral_limit=0.6``, ``feedforward=[0.02, 0.02, 0.02]`` — pitch and yaw need higher gain
    than roll because their surfaces have less authority per unit channel at this trim. Attitude
    gains: ``kp=[4.0, 10.0, 3.0]``, ``rate_limit=[3.0, 4.0, 2.0]`` rad/s — pitch also needs a
    higher attitude ``kp`` than roll to settle in time, since the airframe's short-period mode is
    only lightly damped (``damping_ratio`` about 0.33 from :func:`cascade.linearize_step`).
    Guidance gains: ``airspeed_kp=0.12``, ``airspeed_ki=0.06``, ``throttle_trim=0.586``,
    ``throttle_limits=[0.0, 1.0]``, ``altitude_kp=1.0``, ``climb_rate_limit=4.0``,
    ``pitch_trim=0.086`` rad (the trimmed 4.9 deg angle of attack), ``pitch_limits=[-0.35, 0.35]``
    rad, ``heading_kp=1.0``, ``bank_limit=0.45`` rad, ``airspeed_pitch_kp=0.05``.
    """

    from cascade.reference import aerobatic_reference_spec

    channels = channel_map(
        aerobatic_reference_spec(),
        roles={"aileron": "roll", "elevator": "-pitch", "rudder": "yaw"},
        limits=1.0,
    )
    rate = RateGains(
        kp=jnp.array([0.12, 0.20, 0.35]),
        ki=jnp.array([1.0, 2.0, 3.0]),
        kd=jnp.array([0.001, 0.005, 0.005]),
        integral_limit=jnp.array([0.6, 0.6, 0.6]),
        feedforward=jnp.array([0.02, 0.02, 0.02]),
    )
    attitude = AttitudeGains(
        kp=jnp.array([4.0, 10.0, 3.0]), rate_limit=jnp.array([3.0, 4.0, 2.0])
    )
    guidance = GuidanceGains(
        airspeed_kp=jnp.asarray(0.12),
        airspeed_ki=jnp.asarray(0.06),
        throttle_trim=jnp.asarray(0.586),
        throttle_limits=jnp.array([0.0, 1.0]),
        altitude_kp=jnp.asarray(1.0),
        climb_rate_limit=jnp.asarray(4.0),
        pitch_trim=jnp.asarray(0.086),
        pitch_limits=jnp.array([-0.35, 0.35]),
        heading_kp=jnp.asarray(1.0),
        bank_limit=jnp.asarray(0.45),
        airspeed_pitch_kp=jnp.asarray(0.05),
    )
    return CascadeController(
        channels=channels,
        rate=rate,
        attitude=attitude,
        guidance=guidance,
        rate_period=1,
        attitude_period=2,
        guidance_period=10,
    )


def skywalker_x8_controller() -> CascadeController:
    """Tuned cascade for :func:`cascade.skywalker_x8`, gains found by closed-loop step response
    at 18 m/s / 100 m from :func:`cascade.trim_straight_flight`.

    The X8 has only aileron and elevator channels (no rudder); its yaw column is all zero and the
    rate loop's commanded yaw authority is unused. Elevator is trailing-edge-down positive, a
    nose-down pitching moment, so like the reference aircraft it maps with a flipped sign,
    confirmed with :func:`cascade.aerodynamic_sweep` and a short open-loop rollout from trim.
    Channel limits are narrower than the physical ``0.7`` rad elevon limit (``0.35`` rad each) so
    a combined aileron-plus-elevator command cannot drive one elevon past its own stall angle.
    Rate gains: ``kp=[0.5, 2.0, 0.0]``, ``ki=[2.0, 5.0, 0.0]``, ``kd=[0.0, 0.08, 0.0]``,
    ``integral_limit=0.3``, ``feedforward=[0.02, 0.02, 0.0]``. The pitch axis carries real
    derivative gain (roll needs none) because the X8's short-period mode is very lightly damped
    (``damping_ratio`` about 0.10, versus 0.33 for the reference aircraft, from
    :func:`cascade.linearize_step`); without it, pitch-rate tracking rings up the mode instead of
    damping it. Attitude gains: ``kp=[8.0, 3.0, 1.0]``, ``rate_limit=[4.0, 1.5, 1.0]`` rad/s.
    Guidance gains: ``airspeed_kp=0.08``, ``airspeed_ki=0.03``, ``throttle_trim=0.437``,
    ``throttle_limits=[0.0, 1.0]``, ``altitude_kp=0.35``, ``climb_rate_limit=3.0``,
    ``pitch_trim=0.024`` rad (the trimmed 1.4 deg angle of attack), ``pitch_limits=[-0.2, 0.2]``
    rad, ``heading_kp=0.8``, ``bank_limit=0.4`` rad, ``airspeed_pitch_kp=0.05``.
    """

    from cascade.reference import skywalker_x8_spec

    channels = channel_map(
        skywalker_x8_spec(),
        roles={"aileron": "roll", "elevator": "-pitch"},
        limits={"aileron": 0.35, "elevator": 0.35},
    )
    rate = RateGains(
        kp=jnp.array([0.5, 2.0, 0.0]),
        ki=jnp.array([2.0, 5.0, 0.0]),
        kd=jnp.array([0.0, 0.08, 0.0]),
        integral_limit=jnp.array([0.3, 0.3, 0.3]),
        feedforward=jnp.array([0.02, 0.02, 0.0]),
    )
    attitude = AttitudeGains(
        kp=jnp.array([8.0, 3.0, 1.0]), rate_limit=jnp.array([4.0, 1.5, 1.0])
    )
    guidance = GuidanceGains(
        airspeed_kp=jnp.asarray(0.08),
        airspeed_ki=jnp.asarray(0.03),
        throttle_trim=jnp.asarray(0.437),
        throttle_limits=jnp.array([0.0, 1.0]),
        altitude_kp=jnp.asarray(0.35),
        climb_rate_limit=jnp.asarray(3.0),
        pitch_trim=jnp.asarray(0.024),
        pitch_limits=jnp.array([-0.2, 0.2]),
        heading_kp=jnp.asarray(0.8),
        bank_limit=jnp.asarray(0.4),
        airspeed_pitch_kp=jnp.asarray(0.05),
    )
    return CascadeController(
        channels=channels,
        rate=rate,
        attitude=attitude,
        guidance=guidance,
        rate_period=1,
        attitude_period=2,
        guidance_period=10,
    )
