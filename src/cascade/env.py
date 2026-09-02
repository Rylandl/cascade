"""Native-JAX episode environments over the functional core.

An environment here is a set of pure functions, not an object: :func:`reset` draws an initial
state around a trimmed reference flight, :func:`step` holds an action for one control period
of simulation and returns the next observation, reward, and termination flag, and
:func:`rollout_actions` scans a whole action sequence. Everything is jit-able, vmap-able over
keys, states, and models, and differentiable, so the same functions serve reinforcement
learning (vmap over thousands of episodes), trajectory optimisation (grad through the
episode), and system identification (vmap over model parameters).

The packaged tasks are reference tracking (hold an airspeed, altitude, and heading from a
trim), hover (hold a position and belly azimuth, for a tailsitter), and transition (from
hover, reach and hold cruise).
The reward is ``exp(-cost)`` in ``(0, 1]``, zero on the step an episode crashes, so an
undiscounted return counts "good steps" and a policy that only survives earns nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.analysis.trim import StraightFlightCondition, trim_straight_flight
from cascade.control import (
    CascadeController,
    CascadeState,
    GuidanceSetpoint,
    cascade_step,
    initial_cascade_state,
)
from cascade.initialization import (
    control_from_array,
    control_to_array,
    equilibrate_internal_state,
    standard_environment,
    zero_state,
)
from cascade.integration import StepFunction, repeat_control, rk4_step, rollout
from cascade.math import (
    normalize,
    quaternion_multiply,
    quaternion_rotate,
    quaternion_rotate_inverse,
    safe_norm,
)
from cascade.model import AircraftModel
from cascade.state import AircraftState, ControlInput, Environment
from cascade.vtol import (
    HoverSetpoint,
    TransitionController,
    TransitionState,
    hover_throttle,
    thrust_direction_attitude,
    transition_step,
)


@dataclass(frozen=True)
class EpisodeConfig:
    """Static episode settings (Python scalars, so they are compile-time constants).

    Actions are normalised to ``[-1, 1]``: throttles map to ``[0, 1]`` and channels are scaled
    by ``channel_scale`` into the aircraft's channel units (1.0 for a normalised spec, about
    0.5 rad for one commanding radians). The reset draws Gaussian perturbations of the trimmed
    reference with the listed standard deviations; the attitude perturbation is a body-frame
    rotation vector.
    """

    simulation_frequency_hz: float = 400.0
    control_frequency_hz: float = 40.0
    horizon_steps: int = 400
    channel_scale: float = 1.0
    reset_position_std_m: float = 2.0
    reset_velocity_std_m_s: float = 1.0
    reset_attitude_std_rad: float = 0.1
    reset_rate_std_rad_s: float = 0.2
    crash_altitude_m: float = 0.0
    upright_limit_rad: float = 1.4
    step: StepFunction = rk4_step

    def __post_init__(self) -> None:
        ratio = self.simulation_frequency_hz / self.control_frequency_hz
        if abs(ratio - round(ratio)) > 1e-9 or ratio < 1.0:
            raise ValueError(
                "simulation frequency must be an integer multiple of control frequency"
            )
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")

    @property
    def substeps(self) -> int:
        return int(round(self.simulation_frequency_hz / self.control_frequency_hz))

    @property
    def simulation_dt_s(self) -> float:
        return 1.0 / self.simulation_frequency_hz


class TrackingTask(NamedTuple):
    """Reference tracking: hold an airspeed, altitude, and heading.

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


class Reference(NamedTuple):
    """The trimmed flight an episode is drawn around: state, control, and environment."""

    state: AircraftState
    control: ControlInput
    environment: Environment


class EnvState(NamedTuple):
    aircraft: AircraftState
    step: Array
    key: Array


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


def trimmed_reference(
    model: AircraftModel,
    task: TrackingTask,
    environment: Environment | None = None,
    **trim_kwargs,
) -> Reference:
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
    return Reference(state=result.state, control=result.control, environment=environment)


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
        position_ned=jnp.asarray(position_ned, dtype=jnp.float32),
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


def hover_reference(
    model: AircraftModel, task: HoverTask | TransitionTask, environment: Environment | None = None
) -> Reference:
    """The static hover of a tailsitter: nose up, belly toward the azimuth, throttle balancing
    weight from the static thrust map, elevons neutral. Not a trim (a wing in its own propwash
    carries a small camber force), so hold it with feedback. A transition task hovers at its
    altitude facing its heading."""

    environment = standard_environment() if environment is None else environment
    if isinstance(task, TransitionTask):
        position = jnp.array([0.0, 0.0, 0.0]).at[2].set(-task.altitude_m)
        azimuth = task.heading_rad
    else:
        position, azimuth = task.position_ned, task.azimuth_rad
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
    return Reference(state=state, control=control, environment=environment)


def _quaternion_from_rotvec(rotvec: Array) -> Array:
    angle = safe_norm(rotvec, keepdims=True)
    vector = 0.5 * rotvec * jnp.sinc(angle / (2.0 * jnp.pi))
    return jnp.concatenate((vector, jnp.cos(0.5 * angle)), axis=-1)


def reset(
    model: AircraftModel,
    config: EpisodeConfig,
    task: TrackingTask,
    reference: Reference,
    key: Array,
) -> tuple[EnvState, Array]:
    """Draw an initial state around the reference; returns the state and first observation."""

    key, k_position, k_velocity, k_attitude, k_rate = jax.random.split(key, 5)
    rigid = reference.state.rigid_body
    position = rigid.position + config.reset_position_std_m * jax.random.normal(k_position, (3,))
    velocity = rigid.velocity + config.reset_velocity_std_m_s * jax.random.normal(k_velocity, (3,))
    rotation = config.reset_attitude_std_rad * jax.random.normal(k_attitude, (3,))
    attitude = normalize(quaternion_multiply(rigid.attitude, _quaternion_from_rotvec(rotation)))
    rate = rigid.angular_velocity + config.reset_rate_std_rad_s * jax.random.normal(k_rate, (3,))
    perturbed = reference.state._replace(
        rigid_body=rigid._replace(
            position=position, velocity=velocity, attitude=attitude, angular_velocity=rate
        )
    )
    aircraft = equilibrate_internal_state(
        model, perturbed, reference.control, reference.environment
    )
    state = EnvState(aircraft=aircraft, step=jnp.zeros((), jnp.int32), key=key)
    return state, observation(model, task, reference, state)


def reference_speed(task: Task) -> Array:
    """Speed that normalises velocity observations: the tracking airspeed, or 1 m/s in hover."""

    if isinstance(task, TrackingTask):
        return task.airspeed_m_s
    if isinstance(task, TransitionTask):
        return task.cruise_speed_m_s
    return jnp.asarray(1.0)


def _nose_heading(attitude_xyzw: Array) -> Array:
    nose = quaternion_rotate(attitude_xyzw, jnp.array([1.0, 0.0, 0.0]))
    return jnp.arctan2(nose[..., 1], nose[..., 0])


def _belly_azimuth(attitude_xyzw: Array) -> Array:
    belly = quaternion_rotate(attitude_xyzw, jnp.array([0.0, 0.0, 1.0]))
    return jnp.arctan2(belly[..., 1], belly[..., 0])


def observation(model: AircraftModel, task: Task, reference: Reference, state: EnvState) -> Array:
    """Body-frame observation vector, independent of world position except through the error.

    Layout: air velocity in body FRD over the reference speed (3), airspeed over the reference
    speed, alpha, beta (3), body rates (3), gravity direction in body axes (3), heading error as
    sin and cos (2), position error in body axes over 10 m (3), surface deflections (S),
    propeller speeds as a fraction of maximum (P). Tracking tasks reference only altitude, so
    their position error is vertical.
    """

    rigid = state.aircraft.rigid_body
    environment = reference.environment
    air_body = quaternion_rotate_inverse(rigid.attitude, rigid.velocity - environment.wind)
    airspeed = safe_norm(air_body)
    alpha = jnp.arctan2(air_body[..., 2], air_body[..., 0])
    beta = jnp.arcsin(jnp.clip(air_body[..., 1] / jnp.maximum(airspeed, 1e-3), -1.0, 1.0))
    gravity_body = quaternion_rotate_inverse(rigid.attitude, normalize(environment.gravity))
    heading_error = task.heading_error(rigid)
    position_error_body = quaternion_rotate_inverse(rigid.attitude, task.position_error(rigid))
    speed_scale = jnp.maximum(reference_speed(task), 1.0)
    return jnp.concatenate(
        (
            air_body / speed_scale,
            jnp.stack((airspeed / speed_scale, alpha, beta), axis=-1),
            rigid.angular_velocity,
            gravity_body,
            jnp.stack((jnp.sin(heading_error), jnp.cos(heading_error)), axis=-1),
            position_error_body / 10.0,
            state.aircraft.actuators.surface_deflection,
            state.aircraft.actuators.propeller_speed / model.actuators.propeller_speed_max,
        ),
        axis=-1,
    )


def action_size(model: AircraftModel) -> int:
    """Length of the normalised action: one throttle per propeller plus the control channels."""

    return model.n_propellers + model.n_control_channels


def action_to_control(model: AircraftModel, config: EpisodeConfig, action: Array) -> ControlInput:
    """Map a normalised ``[-1, 1]`` action to the aircraft's control input."""

    control = control_from_array(model, action)
    return ControlInput(
        propeller=jnp.clip(0.5 * (control.propeller + 1.0), 0.0, 1.0),
        channel=control.channel * config.channel_scale,
    )


def control_to_action(config: EpisodeConfig, control: ControlInput) -> Array:
    """Inverse of :func:`action_to_control`; the trim control becomes the reference action."""

    return control_to_array(
        ControlInput(
            propeller=2.0 * control.propeller - 1.0, channel=control.channel / config.channel_scale
        )
    )


def step(
    model: AircraftModel,
    config: EpisodeConfig,
    task: Task,
    reference: Reference,
    state: EnvState,
    action: Array,
) -> tuple[EnvState, Array, Array, Array, dict[str, Array]]:
    """Hold ``action`` for one control period; returns state, observation, reward, done, info.

    ``done`` is true on a crash (below ``crash_altitude_m``), on leaving the upright envelope
    (the body down axis more than ``upright_limit_rad`` from gravity), or at the horizon; the
    info dict separates ``crashed`` and ``truncated`` and reports the cost.
    """

    control = action_to_control(model, config, action)
    controls = repeat_control(control, config.substeps)
    aircraft, _ = rollout(
        model,
        state.aircraft,
        controls,
        reference.environment,
        config.simulation_dt_s,
        step=config.step,
    )
    rigid = aircraft.rigid_body
    cost = task.cost(rigid, reference.environment, action)
    down_body = quaternion_rotate_inverse(rigid.attitude, normalize(reference.environment.gravity))
    crashed = (-rigid.position[..., 2] < config.crash_altitude_m) | (
        down_body[..., 2] < jnp.cos(config.upright_limit_rad)
    )
    next_step = state.step + 1
    truncated = next_step >= config.horizon_steps
    reward = jnp.where(crashed, 0.0, jnp.exp(-cost))
    next_state = EnvState(aircraft=aircraft, step=next_step, key=state.key)
    info = {"cost": cost, "crashed": crashed, "truncated": truncated}
    return (
        next_state,
        observation(model, task, reference, next_state),
        reward,
        crashed | truncated,
        info,
    )


def rollout_actions(
    model: AircraftModel,
    config: EpisodeConfig,
    task: Task,
    reference: Reference,
    state: EnvState,
    actions: Array,
) -> tuple[EnvState, tuple[Array, Array, Array]]:
    """Scan a time-major action sequence; returns the final state and (observations, rewards,
    dones). Rewards after the first ``done`` are zeroed, so the sum is the episode return."""

    def scan_step(carry, action):
        state, finished = carry
        next_state, obs, reward, done, _ = step(model, config, task, reference, state, action)
        reward = jnp.where(finished, 0.0, reward)
        return (next_state, finished | done), (obs, reward, done)

    (final, _), outputs = jax.lax.scan(scan_step, (state, jnp.zeros((), bool)), actions)
    return final, outputs


def rollout_policy(
    model: AircraftModel,
    config: EpisodeConfig,
    task: Task,
    reference: Reference,
    state: EnvState,
    policy,
    policy_state,
) -> tuple[EnvState, tuple[Array, Array, Array, Array]]:
    """Scan a policy over the horizon; returns the final state and (observations, actions,
    rewards, dones), rewards zeroed after the first ``done``.

    ``policy(policy_state, observation, env_state) -> (action, policy_state)``: a learned policy
    reads the observation and ignores the environment state; a model-based baseline such as
    :func:`cascade_policy` may read the state directly.
    """

    first_observation = observation(model, task, reference, state)

    def scan_step(carry, _):
        state, obs, policy_state, finished = carry
        action, policy_state = policy(policy_state, obs, state)
        next_state, next_obs, reward, done, _ = step(model, config, task, reference, state, action)
        reward = jnp.where(finished, 0.0, reward)
        return (next_state, next_obs, policy_state, finished | done), (obs, action, reward, done)

    (final, _, _, _), outputs = jax.lax.scan(
        scan_step,
        (state, first_observation, policy_state, jnp.zeros((), bool)),
        None,
        length=config.horizon_steps,
    )
    return final, outputs


def cascade_policy(
    controller: CascadeController,
    model: AircraftModel,
    config: EpisodeConfig,
    task: TrackingTask,
    reference: Reference,
):
    """The control cascade as a policy for a tracking task: the reference score for a learner.

    Every loop runs at the environment's control rate (the controller's periods are replaced
    by one), so the baseline sees exactly what a learned policy sees in time. Returns the
    policy function and its initial :class:`CascadeState`, holding the reference control until
    the first update.
    """

    if not isinstance(task, TrackingTask):
        raise TypeError("cascade_policy needs a TrackingTask; the cascade has no hover mode")
    controller = controller._replace(rate_period=1, attitude_period=1, guidance_period=1)
    setpoint = GuidanceSetpoint(
        airspeed_m_s=task.airspeed_m_s, altitude_m=task.altitude_m, heading_rad=task.heading_rad
    )
    period = 1.0 / config.control_frequency_hz

    def policy(cascade_state: CascadeState, obs: Array, env_state: EnvState):
        control, cascade_state = cascade_step(
            controller, cascade_state, setpoint, env_state.aircraft, reference.environment, period
        )
        return control_to_action(config, control), cascade_state

    return policy, initial_cascade_state(controller, reference.state, reference.control)


def transition_policy(
    controller: TransitionController,
    model: AircraftModel,
    config: EpisodeConfig,
    task: TransitionTask,
    reference: Reference,
    hover_setpoints: HoverSetpoint,
    forward_setpoints,
):
    """The transition controller as a policy: the baseline for a transition task.

    ``hover_setpoints`` and ``forward_setpoints`` are time-major schedules of at least
    ``horizon_steps`` entries (a :func:`cascade.vtol.velocity_ramp_schedule`, say); the policy
    indexes them by the episode step and runs :func:`cascade.vtol.transition_step` at the
    control rate. Returns the policy and its initial :class:`TransitionState`.
    """

    from cascade.vtol import initial_transition_state

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
            reference.environment,
            period,
        )
        return control_to_action(config, control), transition_state

    return policy, initial_transition_state(reference.state)
