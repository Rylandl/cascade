"""Model-based baseline policies: the control cascade and the transition controller wrapped
as policies so a learner has a reference score on the same task and horizon."""

from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.control.loops import (
    CascadeController,
    CascadeState,
    GuidanceSetpoint,
    cascade_step,
    initial_cascade_state,
)
from cascade.control.scheduling import GainSchedule, controller_at_speed
from cascade.control.vtol import (
    HoverSetpoint,
    TransitionController,
    TransitionState,
    transition_step,
)
from cascade.env.episode import (
    EnvState,
    EpisodeConfig,
    control_to_action,
    current_environment,
    observation_layout,
    observation_size,
)
from cascade.env.tasks import ReferenceFlight, TrackingTask, TransitionTask, task_at
from cascade.math import quaternion_from_euler, safe_norm
from cascade.model import AircraftModel


class SensorObservation(NamedTuple):
    """Public measurement input with per-element acquisition ages and validity.

    All three arrays have the observation vector's shape; ``age_s`` is in seconds
    and ``valid`` is boolean. Invalid elements may have infinite ages. This is an
    optional input to :func:`observation_cascade_policy`; ordinary episode rollouts
    pass just the values and therefore cannot reject stale, finite measurements.
    """

    values: Array
    age_s: Array
    valid: Array


class ObservationCascadeState(NamedTuple):
    """Controller memory, independent clock, last action, and last update status.

    ``measurement_valid`` is false when the latest call held the previous action
    because required measurements or the resulting controller update were unusable.
    The clock advances on every call, including a held update.
    """

    cascade: CascadeState
    step: Array
    action: Array
    measurement_valid: Array


def observation_cascade_policy(
    controller: CascadeController | GainSchedule,
    model: AircraftModel,
    config: EpisodeConfig,
    task,
    reference: ReferenceFlight,
    *,
    max_sensor_age_s: float = 0.5,
):
    """Wrap the fixed-wing cascade using observations and policy memory only.

    Requires airspeed, rates, gravity, heading, and position-error blocks (both
    the default observation and ``onboard_observation()`` provide them). Gravity
    reconstructs roll/pitch; heading error supplies relative yaw guidance; the
    body position error projected onto gravity supplies altitude error in metres.
    The cascade runs in a synthetic frame with current yaw and altitude zero.
    It needs no true position, velocity, wind, actuator state, or episode clock.
    The third policy argument is accepted for rollout compatibility and ignored.

    The known mission supplies commanded speed at ``memory.step / frequency``;
    heading and altitude commands come entirely from observations, including for
    waypoint and orbit missions. Assume NED gravity, nonsingular pitch, one policy
    call per control period, and a fresh policy memory at each episode reset.
    All cascade loops run each call; actions are clipped to [-1, 1].

    A raw vector uses finite held/delayed readings as delivered. Without metadata,
    stale readings and missing zero rate/position readings cannot be distinguished
    from legitimate measurements. ``SensorObservation`` additionally rejects any
    required element marked invalid or older than ``max_sensor_age_s`` and uses
    airspeed acquisition age to undo scheduled-target speed normalization. Any
    unusable required block (including degenerate gravity/heading or nonpositive
    airspeed) freezes the cascade and holds the last action, initially the clipped
    reference action. This is a defined fallback, not a flight-safety controller.
    Asynchronous blocks are not extrapolated or fused into a state estimator.
    """

    if (
        isinstance(max_sensor_age_s, bool)
        or not isinstance(max_sensor_age_s, Real)
        or not math.isfinite(max_sensor_age_s)
        or max_sensor_age_s <= 0.0
    ):
        raise ValueError("max_sensor_age_s must be finite and positive")
    layout = observation_layout(model, config.observation)
    required = ("airspeed", "rates", "gravity", "heading", "position_error")
    missing = [name for name in required if getattr(layout, name) is None]
    if missing:
        raise ValueError(f"observation_cascade_policy requires observation blocks: {missing}")
    indices = jnp.asarray(
        [
            i
            for name in required
            for i in range(getattr(layout, name).start, getattr(layout, name).stop)
        ]
    )
    size = observation_size(model, config.observation)
    gravity = np.asarray(reference.environment.gravity)
    if (
        gravity.shape != (3,)
        or not np.all(np.isfinite(gravity))
        or not np.allclose(gravity[:2], 0.0, atol=1e-7)
        or gravity[2] <= 0.0
    ):
        raise ValueError("observation_cascade_policy requires positive world-NED gravity")
    target = task_at(task, jnp.asarray(0.0))
    if not isinstance(getattr(target, "tracking", target), TrackingTask):
        raise TypeError("observation_cascade_policy needs a TrackingTask or tracking mission")
    schedule = controller if isinstance(controller, GainSchedule) else None
    if schedule is not None:
        controller = controller_at_speed(schedule, target.airspeed_m_s)
    controller = controller._replace(rate_period=1, attitude_period=1, guidance_period=1)
    period = 1.0 / config.control_frequency_hz
    environment = reference.environment._replace(wind=jnp.zeros(3))

    def policy(memory: ObservationCascadeState, obs: Array | SensorObservation, env_state=None):
        del env_state
        values = jnp.asarray(obs.values if isinstance(obs, SensorObservation) else obs)
        if values.shape != (size,) or not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError(f"observation must be a floating vector of shape ({size},)")
        usable = jnp.all(jnp.isfinite(values[indices]))
        time_s = memory.step * period
        speed_sample_time = time_s
        if isinstance(obs, SensorObservation):
            ages, valid = jnp.asarray(obs.age_s), jnp.asarray(obs.valid)
            if ages.shape != values.shape or valid.shape != values.shape:
                raise ValueError("sensor ages and validity must match observation shape")
            if valid.dtype != jnp.bool_ or not jnp.issubdtype(ages.dtype, jnp.floating):
                raise ValueError("sensor validity must be boolean and ages floating point")
            usable &= jnp.all(
                valid[indices]
                & jnp.isfinite(ages[indices])
                & (ages[indices] >= 0.0)
                & (ages[indices] <= max_sensor_age_s)
            )
            speed_age = ages[layout.airspeed][0]
            speed_age = jnp.where(jnp.isfinite(speed_age), speed_age, 0.0)
            speed_sample_time = jnp.maximum(time_s - speed_age, 0.0)
        measured = jnp.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        gravity_body = measured[layout.gravity]
        # Scaling before normalizing avoids overflow for malformed large readings.
        gravity_scale = jnp.max(jnp.abs(gravity_body))
        gravity_body = gravity_body / jnp.maximum(gravity_scale, 1e-6)
        gravity_body = gravity_body / jnp.maximum(safe_norm(gravity_body), 1e-6)
        horizontal_gravity = jnp.hypot(gravity_body[1], gravity_body[2])
        heading = measured[layout.heading]
        normalized_speed = measured[layout.airspeed][0]
        usable &= (
            (gravity_scale > 1e-6)
            & (horizontal_gravity > 1e-4)
            & (jnp.max(jnp.abs(heading)) > 1e-6)
            & (normalized_speed > 0.0)
        )
        roll = jnp.arctan2(gravity_body[1], gravity_body[2])
        pitch = jnp.arctan2(-gravity_body[0], horizontal_gravity)
        target = task_at(task, time_s)
        sample_target = task_at(task, speed_sample_time)
        speed = normalized_speed * jnp.maximum(sample_target.airspeed_m_s, 1.0)
        altitude_error = 10.0 * jnp.dot(measured[layout.position_error], gravity_body)
        synthetic = reference.state._replace(
            rigid_body=reference.state.rigid_body._replace(
                position=jnp.zeros(3),
                attitude=quaternion_from_euler(roll, pitch, jnp.asarray(0.0)),
                velocity=jnp.stack((speed, jnp.zeros_like(speed), jnp.zeros_like(speed))),
                angular_velocity=measured[layout.rates],
            )
        )
        # Output clipping can conceal overflow in decoded measurements; reject
        # those inputs before considering the clipped controller result usable.
        usable &= (
            jnp.isfinite(speed)
            & jnp.isfinite(safe_norm(synthetic.rigid_body.velocity))
            & jnp.isfinite(altitude_error)
        )
        active = controller
        if schedule is not None:
            active = controller_at_speed(schedule, speed)._replace(
                rate_period=1, attitude_period=1, guidance_period=1
            )
        setpoint = GuidanceSetpoint(
            target.airspeed_m_s, altitude_error, -jnp.arctan2(heading[0], heading[1])
        )
        control, candidate = cascade_step(
            active, memory.cascade, setpoint, synthetic, environment, period
        )
        candidate_action = jnp.clip(control_to_action(config, control), -1.0, 1.0)
        usable &= jnp.all(jnp.isfinite(candidate_action))
        for leaf in jax.tree.leaves(candidate):
            usable &= jnp.all(jnp.isfinite(leaf))
        next_cascade = jax.tree.map(
            lambda new, old: jnp.where(usable, new, old), candidate, memory.cascade
        )
        action = jnp.where(usable, candidate_action, memory.action)
        return action, ObservationCascadeState(next_cascade, memory.step + 1, action, usable)

    return policy, ObservationCascadeState(
        initial_cascade_state(controller, reference.state, reference.control),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.clip(control_to_action(config, reference.control), -1.0, 1.0),
        jnp.asarray(False),
    )


def cascade_policy(
    controller: CascadeController | GainSchedule,
    model: AircraftModel,
    config: EpisodeConfig,
    task,
    reference: ReferenceFlight,
):
    """The control cascade as a policy for a tracking task or mission.

    Every loop runs at the environment's control rate (the controller's periods are replaced
    by one), matching the policy's action cadence and mission time. This is a privileged
    model-based baseline: it reads true EnvState and bypasses observation noise, delay, and
    dropout. Returns the policy function and its initial :class:`CascadeState`, holding the
    reference control until the first update.
    """

    target = task_at(task, jnp.asarray(0.0), reference.state.rigid_body)
    tracking = getattr(target, "tracking", target)
    if not isinstance(tracking, TrackingTask):
        raise TypeError("cascade_policy needs a TrackingTask; the cascade has no hover mode")
    schedule = controller if isinstance(controller, GainSchedule) else None
    if schedule is not None:
        speed = safe_norm(reference.state.rigid_body.velocity - reference.environment.wind)
        controller = controller_at_speed(schedule, speed)
    controller = controller._replace(rate_period=1, attitude_period=1, guidance_period=1)
    period = 1.0 / config.control_frequency_hz

    def policy(cascade_state: CascadeState, obs: Array, env_state: EnvState):
        target = task_at(task, env_state.time_s, env_state.aircraft.rigid_body)
        setpoint = GuidanceSetpoint(
            airspeed_m_s=target.airspeed_m_s,
            altitude_m=target.altitude_m,
            heading_rad=target.heading_rad,
        )
        environment = current_environment(reference, env_state)
        active_controller = controller
        if schedule is not None:
            speed = safe_norm(env_state.aircraft.rigid_body.velocity - environment.wind)
            active_controller = controller_at_speed(schedule, speed)._replace(
                rate_period=1, attitude_period=1, guidance_period=1
            )
        control, cascade_state = cascade_step(
            active_controller,
            cascade_state,
            setpoint,
            env_state.aircraft,
            environment,
            period,
        )
        return control_to_action(config, control), cascade_state

    return policy, initial_cascade_state(controller, reference.state, reference.control)


def transition_policy(
    controller: TransitionController,
    model: AircraftModel,
    config: EpisodeConfig,
    task: TransitionTask,
    reference: ReferenceFlight,
    hover_setpoints: HoverSetpoint,
    forward_setpoints,
):
    """The transition controller as a policy: the baseline for a transition task.

    ``hover_setpoints`` and ``forward_setpoints`` are time-major schedules of at least
    ``horizon_steps`` entries (a :func:`cascade.vtol.velocity_ramp_schedule`, say); the policy
    indexes them by the episode step and runs :func:`cascade.vtol.transition_step` at the
    control rate. Returns the policy and its initial :class:`TransitionState`.
    """

    from cascade.control.vtol import initial_transition_state

    period = 1.0 / config.control_frequency_hz
    last = config.horizon_steps - 1

    def policy(transition_state: TransitionState, obs: Array, env_state: EnvState):
        index = jnp.minimum(env_state.step, last)
        hover_setpoint = jax.tree.map(lambda leaf: leaf[index], hover_setpoints)
        forward_setpoint = jax.tree.map(lambda leaf: leaf[index], forward_setpoints)
        control, transition_state, _ = transition_step(
            model,
            controller,
            transition_state,
            hover_setpoint,
            forward_setpoint,
            env_state.aircraft,
            current_environment(reference, env_state),
            period,
        )
        return control_to_action(config, control), transition_state

    return policy, initial_transition_state(reference.state)
