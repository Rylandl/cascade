"""Episode functions: reset, step, observation, action mapping, and rollouts.

Pure functions, jit-able, vmap-able over keys, states, models, and weather, and
differentiable through the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.env.sensors import SensorNoise, _noise_vectors, sensor_noise
from cascade.env.tasks import ReferenceFlight, Task, reference_speed
from cascade.env.weather import (
    WeatherCondition,
    initial_gust_state,
    mean_wind_ned,
    step_gust,
)
from cascade.initialization import (
    control_from_array,
    control_to_array,
    equilibrate_internal_state,
)
from cascade.integration import StepFunction, repeat_control, rk4_step, rollout
from cascade.math import (
    normalize,
    quaternion_multiply,
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
    rotation vector. ``observation_delay_steps`` returns the observation from that many control
    periods ago (the reset observation until the buffer fills), a latency the policy must live
    with. ``action_delay_steps`` applies the action commanded that many periods ago (the
    reference action until the buffer fills): sense-to-actuate latency. ``action_delay_range``
    draws the delay per episode, uniformly over the inclusive integer range, so latency is a
    randomisable leaf; it overrides the fixed value.
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
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    action_delay_range: tuple[int, int] | None = None
    step: StepFunction = rk4_step

    def __post_init__(self) -> None:
        ratio = self.simulation_frequency_hz / self.control_frequency_hz
        if abs(ratio - round(ratio)) > 1e-9 or ratio < 1.0:
            raise ValueError(
                "simulation frequency must be an integer multiple of control frequency"
            )
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.observation_delay_steps < 0:
            raise ValueError("observation_delay_steps must be non-negative")
        if self.action_delay_steps < 0:
            raise ValueError("action_delay_steps must be non-negative")
        if self.action_delay_range is not None:
            low, high = self.action_delay_range
            if low < 0 or high < low:
                raise ValueError("action_delay_range must be 0 <= low <= high")

    @property
    def max_action_delay(self) -> int:
        if self.action_delay_range is not None:
            return int(self.action_delay_range[1])
        return int(self.action_delay_steps)

    @property
    def substeps(self) -> int:
        return int(round(self.simulation_frequency_hz / self.control_frequency_hz))

    @property
    def simulation_dt_s(self) -> float:
        return 1.0 / self.simulation_frequency_hz


class EnvState(NamedTuple):
    aircraft: AircraftState
    step: Array
    key: Array
    sensor_bias: Array
    observation_buffer: Array
    gust: Array
    wind_ned: Array
    action_buffer: Array
    action_delay: Array


def current_environment(reference: ReferenceFlight, state: EnvState) -> Environment:
    """The reference environment with this step's wind (mean profile plus gust)."""

    return reference.environment._replace(wind=state.wind_ned)


def _quaternion_from_rotvec(rotvec: Array) -> Array:
    angle = safe_norm(rotvec, keepdims=True)
    vector = 0.5 * rotvec * jnp.sinc(angle / (2.0 * jnp.pi))
    return jnp.concatenate((vector, jnp.cos(0.5 * angle)), axis=-1)


def reset(
    model: AircraftModel,
    config: EpisodeConfig,
    task: Task,
    reference: ReferenceFlight,
    key: Array,
    noise: SensorNoise | None = None,
    weather: WeatherCondition | None = None,
) -> tuple[EnvState, Array]:
    """Draw an initial state around the reference; returns the state and first observation.

    ``noise`` (default none) adds white sensor noise to every observation and draws a
    per-episode bias here; the true observation is always available from :func:`observation`.
    ``weather`` (default the reference's own wind, usually none) sets the mean wind profile
    and turbulence for the episode; the initial ground velocity is shifted by the wind at the
    start altitude so the aircraft begins at its trimmed airspeed.
    """

    noise = sensor_noise() if noise is None else noise
    key, k_position, k_velocity, k_attitude, k_rate, k_bias, k_noise, k_delay = jax.random.split(
        key, 8
    )
    rigid = reference.state.rigid_body
    position = rigid.position + config.reset_position_std_m * jax.random.normal(k_position, (3,))
    if weather is None:
        wind = reference.environment.wind
    else:
        wind = mean_wind_ned(weather, -position[..., 2])
    velocity = (
        rigid.velocity
        + (wind - reference.environment.wind)
        + config.reset_velocity_std_m_s * jax.random.normal(k_velocity, (3,))
    )
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
    white, bias_std = _noise_vectors(model, noise)
    bias = bias_std * jax.random.normal(k_bias, bias_std.shape)
    partial = EnvState(
        aircraft=aircraft,
        step=jnp.zeros((), jnp.int32),
        key=key,
        sensor_bias=bias,
        observation_buffer=jnp.zeros((config.observation_delay_steps + 1, white.shape[0])),
        gust=initial_gust_state(),
        wind_ned=wind,
        action_buffer=jnp.broadcast_to(
            control_to_action(config, reference.control),
            (config.max_action_delay + 1, action_size(model)),
        ),
        action_delay=_draw_action_delay(config, k_delay),
    )
    sensed = _sense(model, task, reference, partial, white, k_noise)
    buffer = jnp.broadcast_to(sensed, partial.observation_buffer.shape)
    state = partial._replace(observation_buffer=buffer)
    return state, buffer[0]


def _draw_action_delay(config: EpisodeConfig, key: Array) -> Array:
    if config.action_delay_range is None:
        return jnp.asarray(config.action_delay_steps, jnp.int32)
    low, high = config.action_delay_range
    return jax.random.randint(key, (), low, high + 1).astype(jnp.int32)


def _sense(
    model: AircraftModel,
    task: Task,
    reference: ReferenceFlight,
    state: EnvState,
    white: Array,
    key: Array,
) -> Array:
    true = observation(model, task, reference, state)
    return true + state.sensor_bias + white * jax.random.normal(key, true.shape)


def observation(
    model: AircraftModel, task: Task, reference: ReferenceFlight, state: EnvState
) -> Array:
    """Body-frame observation vector, independent of world position except through the error.

    Layout: air velocity in body FRD over the reference speed (3), airspeed over the reference
    speed, alpha, beta (3), body rates (3), gravity direction in body axes (3), heading error as
    sin and cos (2), position error in body axes over 10 m (3), surface deflections (S),
    propeller speeds as a fraction of maximum (P). Tracking tasks reference only altitude, so
    their position error is vertical.
    """

    rigid = state.aircraft.rigid_body
    environment = current_environment(reference, state)
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


class ObservationLayout(NamedTuple):
    """Index slices of the observation vector from :func:`observation`."""

    air_velocity: slice
    air_data: slice
    rates: slice
    gravity: slice
    heading: slice
    position_error: slice
    surfaces: slice
    propellers: slice


OBSERVATION_FIXED_SIZE = 17


def observation_layout(model: AircraftModel) -> ObservationLayout:
    """Where each block sits in the observation of ``model``."""

    surfaces = model.n_surfaces
    return ObservationLayout(
        air_velocity=slice(0, 3),
        air_data=slice(3, 6),
        rates=slice(6, 9),
        gravity=slice(9, 12),
        heading=slice(12, 14),
        position_error=slice(14, 17),
        surfaces=slice(17, 17 + surfaces),
        propellers=slice(17 + surfaces, 17 + surfaces + model.n_propellers),
    )


def observation_size(model: AircraftModel) -> int:
    """Length of the observation vector for ``model``."""

    return OBSERVATION_FIXED_SIZE + model.n_surfaces + model.n_propellers


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
    reference: ReferenceFlight,
    state: EnvState,
    action: Array,
    noise: SensorNoise | None = None,
    weather: WeatherCondition | None = None,
) -> tuple[EnvState, Array, Array, Array, dict[str, Array]]:
    """Hold ``action`` for one control period; returns state, observation, reward, done, info.

    The returned observation carries the sensor ``noise`` (white noise drawn from the episode
    key, plus the bias drawn at reset) and the configured delay. With ``weather`` the wind
    for the period is the mean profile at the aircraft's altitude plus a Dryden gust advanced
    from the episode key; without it the reference wind holds. With an action delay the
    action applied is an earlier one (``info["applied_action"]``); the cost still charges the
    action commanded now.

    ``done`` is true on a crash (below ``crash_altitude_m``), on leaving the upright envelope
    (the body down axis more than ``upright_limit_rad`` from gravity), or at the horizon; the
    info dict separates ``crashed`` and ``truncated`` and reports the cost.
    """

    # The commanded action enters the buffer; the one applied is from ``action_delay``
    # periods ago (index max_delay - delay, the buffer's oldest entry being the most delayed).
    action_buffer = jnp.concatenate((state.action_buffer[1:], action[None]), axis=0)
    applied = jax.lax.dynamic_index_in_dim(
        action_buffer, config.max_action_delay - state.action_delay, axis=0, keepdims=False
    )
    control = action_to_control(model, config, applied)
    controls = repeat_control(control, config.substeps)
    environment = current_environment(reference, state)
    aircraft, _ = rollout(
        model,
        state.aircraft,
        controls,
        environment,
        config.simulation_dt_s,
        step=config.step,
    )
    rigid = aircraft.rigid_body
    cost = task.cost(rigid, environment, action)
    down_body = quaternion_rotate_inverse(rigid.attitude, normalize(reference.environment.gravity))
    crashed = (-rigid.position[..., 2] < config.crash_altitude_m) | (
        down_body[..., 2] < jnp.cos(config.upright_limit_rad)
    )
    next_step = state.step + 1
    truncated = next_step >= config.horizon_steps
    reward = jnp.where(crashed, 0.0, jnp.exp(-cost))
    noise = sensor_noise() if noise is None else noise
    key, k_noise, k_gust = jax.random.split(state.key, 3)
    white, _ = _noise_vectors(model, noise)
    if weather is None:
        gust, wind = state.gust, state.wind_ned
    else:
        air = rigid.velocity - environment.wind
        gust, gust_ned = step_gust(
            weather,
            state.gust,
            k_gust,
            1.0 / config.control_frequency_hz,
            airspeed_m_s=safe_norm(air),
            altitude_m=-rigid.position[..., 2],
            heading_rad=jnp.arctan2(air[..., 1], air[..., 0]),
        )
        wind = mean_wind_ned(weather, -rigid.position[..., 2]) + gust_ned
    advanced = state._replace(
        aircraft=aircraft,
        step=next_step,
        key=key,
        gust=gust,
        wind_ned=wind,
        action_buffer=action_buffer,
    )
    sensed = _sense(model, task, reference, advanced, white, k_noise)
    buffer = jnp.concatenate((state.observation_buffer[1:], sensed[None]), axis=0)
    next_state = advanced._replace(observation_buffer=buffer)
    info = {"cost": cost, "crashed": crashed, "truncated": truncated, "applied_action": applied}
    return next_state, buffer[0], reward, crashed | truncated, info


def rollout_actions(
    model: AircraftModel,
    config: EpisodeConfig,
    task: Task,
    reference: ReferenceFlight,
    state: EnvState,
    actions: Array,
    noise: SensorNoise | None = None,
    weather: WeatherCondition | None = None,
) -> tuple[EnvState, tuple[Array, Array, Array]]:
    """Scan a time-major action sequence; returns the final state and (observations, rewards,
    dones). Rewards after the first ``done`` are zeroed, so the sum is the episode return."""

    def scan_step(carry, action):
        state, finished = carry
        next_state, obs, reward, done, _ = step(
            model, config, task, reference, state, action, noise, weather
        )
        reward = jnp.where(finished, 0.0, reward)
        return (next_state, finished | done), (obs, reward, done)

    (final, _), outputs = jax.lax.scan(scan_step, (state, jnp.zeros((), bool)), actions)
    return final, outputs


def rollout_policy(
    model: AircraftModel,
    config: EpisodeConfig,
    task: Task,
    reference: ReferenceFlight,
    state: EnvState,
    policy,
    policy_state,
    noise: SensorNoise | None = None,
    weather: WeatherCondition | None = None,
) -> tuple[EnvState, tuple[Array, Array, Array, Array]]:
    """Scan a policy over the horizon; returns the final state and (observations, actions,
    rewards, dones), rewards zeroed after the first ``done``.

    ``policy(policy_state, observation, env_state) -> (action, policy_state)``: a learned policy
    reads the observation and ignores the environment state; a model-based baseline such as
    :func:`cascade_policy` may read the state directly.
    """

    first_observation = state.observation_buffer[0]

    def scan_step(carry, _):
        state, obs, policy_state, finished = carry
        action, policy_state = policy(policy_state, obs, state)
        next_state, next_obs, reward, done, _ = step(
            model, config, task, reference, state, action, noise, weather
        )
        reward = jnp.where(finished, 0.0, reward)
        return (next_state, next_obs, policy_state, finished | done), (obs, action, reward, done)

    (final, _, _, _), outputs = jax.lax.scan(
        scan_step,
        (state, first_observation, policy_state, jnp.zeros((), bool)),
        None,
        length=config.horizon_steps,
    )
    return final, outputs
