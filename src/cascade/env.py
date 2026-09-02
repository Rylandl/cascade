"""Native-JAX episode environments over the functional core.

An environment here is a set of pure functions, not an object: :func:`reset` draws an initial
state around a trimmed reference flight, :func:`step` holds an action for one control period
of simulation and returns the next observation, reward, and termination flag, and
:func:`rollout_actions` scans a whole action sequence. Everything is jit-able, vmap-able over
keys, states, and models, and differentiable, so the same functions serve reinforcement
learning (vmap over thousands of episodes), trajectory optimisation (grad through the
episode), and system identification (vmap over model parameters).

The packaged task is reference tracking: hold an airspeed, altitude, and heading from a trim.
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
from cascade.initialization import (
    control_from_array,
    control_to_array,
    equilibrate_internal_state,
    standard_environment,
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


def _heading(attitude_xyzw: Array) -> Array:
    nose = quaternion_rotate(attitude_xyzw, jnp.array([1.0, 0.0, 0.0]))
    return jnp.arctan2(nose[..., 1], nose[..., 0])


def observation(
    model: AircraftModel, task: TrackingTask, reference: Reference, state: EnvState
) -> Array:
    """Body-frame observation vector, independent of world position except through altitude.

    Layout: air velocity in body FRD (3), airspeed, alpha, beta, body rates (3), gravity
    direction in body axes (3), heading error as sin and cos (2), altitude error / 10 m (1),
    surface deflections (S), propeller speeds as a fraction of maximum (P).
    """

    rigid = state.aircraft.rigid_body
    environment = reference.environment
    air_body = quaternion_rotate_inverse(rigid.attitude, rigid.velocity - environment.wind)
    airspeed = safe_norm(air_body)
    alpha = jnp.arctan2(air_body[..., 2], air_body[..., 0])
    beta = jnp.arcsin(jnp.clip(air_body[..., 1] / jnp.maximum(airspeed, 1e-3), -1.0, 1.0))
    gravity_body = quaternion_rotate_inverse(rigid.attitude, normalize(environment.gravity))
    heading_error = _heading(rigid.attitude) - task.heading_rad
    altitude_error = (-rigid.position[..., 2] - task.altitude_m) / 10.0
    return jnp.concatenate(
        (
            air_body / jnp.maximum(task.airspeed_m_s, 1.0),
            jnp.stack((airspeed / jnp.maximum(task.airspeed_m_s, 1.0), alpha, beta), axis=-1),
            rigid.angular_velocity,
            gravity_body,
            jnp.stack((jnp.sin(heading_error), jnp.cos(heading_error)), axis=-1),
            altitude_error[..., None],
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


def _tracking_cost(task: TrackingTask, rigid, environment: Environment, action: Array) -> Array:
    airspeed = safe_norm(rigid.velocity - environment.wind)
    airspeed_error = (airspeed - task.airspeed_m_s) / jnp.maximum(task.airspeed_m_s, 1.0)
    altitude_error = (-rigid.position[..., 2] - task.altitude_m) / 10.0
    heading_error = 1.0 - jnp.cos(_heading(rigid.attitude) - task.heading_rad)
    return (
        task.airspeed_weight * jnp.square(airspeed_error)
        + task.altitude_weight * jnp.square(altitude_error)
        + task.heading_weight * heading_error
        + task.rate_weight * jnp.sum(jnp.square(rigid.angular_velocity), axis=-1)
        + task.effort_weight * jnp.mean(jnp.square(action), axis=-1)
    )


def step(
    model: AircraftModel,
    config: EpisodeConfig,
    task: TrackingTask,
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
    cost = _tracking_cost(task, rigid, reference.environment, action)
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
    task: TrackingTask,
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
