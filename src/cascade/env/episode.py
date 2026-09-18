"""Episode functions: reset, step, observation, action mapping, and rollouts.

Pure functions, jit-able, vmap-able over keys, states, models, and weather, and
differentiable through the episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.dynamics import evaluate_dynamics
from cascade.env.faults import FaultSchedule, apply_faults
from cascade.env.sensors import (
    GRAVITY_SCALE_M_S2,
    ObservationSpec,
    SensorNoise,
    SensorPipelineConfig,
    SensorState,
    _noise_vectors,
    block_sizes,
    initialize_sensor_pipeline,
    sensor_noise,
    step_sensor_pipeline,
)
from cascade.env.tasks import ReferenceFlight, Task, reference_speed, task_at
from cascade.env.weather import (
    WeatherCondition,
    discrete_gust_ned,
    initial_gust_state,
    isa_density,
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
    randomisable leaf; it overrides the fixed value. ``observation`` selects the blocks a
    policy sees (:class:`cascade.env.sensors.ObservationSpec`; everything by default).
    ``isa_density`` replaces the reference environment's density with the standard
    atmosphere at the aircraft's altitude every period. ``upright_limit_rad >= pi`` disables
    attitude-based termination for hover, inverted flight, and full-attitude tasks.
    ``sensors`` optionally adds block sample rates, random-walk bias, dropped packets,
    and acquisition/transport jitter before the whole-observation delay.
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
    observation: ObservationSpec = ObservationSpec()
    isa_density: bool = False
    step: StepFunction = rk4_step
    sensors: SensorPipelineConfig | None = None

    def __post_init__(self) -> None:
        for name in ("simulation_frequency_hz", "control_frequency_hz", "channel_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        ratio = self.simulation_frequency_hz / self.control_frequency_hz
        if not math.isfinite(ratio) or abs(ratio - round(ratio)) > 1e-9 or ratio < 1.0:
            raise ValueError(
                "simulation frequency must be an integer multiple of control frequency"
            )
        for name in ("horizon_steps", "observation_delay_steps", "action_delay_steps"):
            value = getattr(self, name)
            minimum = 1 if name == "horizon_steps" else 0
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.action_delay_range is not None:
            bounds = self.action_delay_range
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in bounds)
                or not 0 <= bounds[0] <= bounds[1]
            ):
                raise ValueError("action_delay_range must be integer bounds 0 <= low <= high")
        for name in (
            "reset_position_std_m",
            "reset_velocity_std_m_s",
            "reset_attitude_std_rad",
            "reset_rate_std_rad_s",
            "upright_limit_rad",
            "crash_altitude_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "crash_altitude_m" and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not callable(self.step):
            raise ValueError("step must be a callable integration function")
        if self.sensors is not None and not isinstance(self.sensors, SensorPipelineConfig):
            raise ValueError("sensors must be a SensorPipelineConfig or None")

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
    """Episode state, including delivered-measurement metadata.

    ``sensor_age_s`` and ``sensor_valid`` follow the observation layout and describe
    ``observation_buffer[0]`` after all delays. An unavailable measurement has age
    infinity, validity false, and value zero. Held measurements remain valid while
    their ages increase. ``sensor_state`` is present only with a sensor pipeline.
    """

    aircraft: AircraftState
    step: Array
    key: Array
    sensor_bias: Array
    observation_buffer: Array
    gust: Array
    wind_ned: Array
    action_buffer: Array
    action_delay: Array
    density: Array
    time_s: Array
    sensor_state: SensorState | None
    observation_sample_step_buffer: Array
    observation_valid_buffer: Array
    sensor_age_s: Array
    sensor_valid: Array


def current_environment(reference: ReferenceFlight, state: EnvState) -> Environment:
    """The reference environment with this step's wind (mean profile plus gusts) and density."""

    return reference.environment._replace(wind=state.wind_ned, density=state.density)


def _density(config: EpisodeConfig, reference: ReferenceFlight, altitude_m: Array) -> Array:
    if config.isa_density:
        return isa_density(altitude_m)
    return reference.environment.density


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
    A configured sensor pipeline acquires at time zero; blocks with transport delay
    or an initial dropout return zero until delivery, marked invalid in the state.
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
        wind = mean_wind_ned(weather, -position[..., 2]) + discrete_gust_ned(weather, 0.0)
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
    white, bias_std = _noise_vectors(model, noise, config.observation)
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
        density=_density(config, reference, -position[..., 2]),
        time_s=jnp.zeros(()),
        sensor_state=None,
        observation_sample_step_buffer=jnp.zeros(
            (config.observation_delay_steps + 1, white.shape[0]), jnp.int32
        ),
        observation_valid_buffer=jnp.ones(
            (config.observation_delay_steps + 1, white.shape[0]), bool
        ),
        sensor_age_s=jnp.zeros_like(white),
        sensor_valid=jnp.ones(white.shape, bool),
    )
    sensed = _sense(model, task, reference, partial, white, k_noise, config.observation)
    if config.sensors is not None:
        pipeline = initialize_sensor_pipeline(
            model, config.sensors, config.observation, sensed, jax.random.fold_in(k_noise, 1)
        )
        sensed = pipeline.reading
        partial = partial._replace(
            sensor_state=pipeline,
            observation_sample_step_buffer=jnp.broadcast_to(
                pipeline.sample_step,
                partial.observation_sample_step_buffer.shape,
            ),
            observation_valid_buffer=jnp.broadcast_to(
                pipeline.valid, partial.observation_valid_buffer.shape
            ),
            sensor_valid=pipeline.valid,
            sensor_age_s=jnp.where(pipeline.valid, 0.0, jnp.inf),
        )
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
    spec: ObservationSpec | None = None,
) -> Array:
    true = observation(model, task, reference, state, spec)
    return true + state.sensor_bias + white * jax.random.normal(key, true.shape)


def observation(
    model: AircraftModel,
    task: Task,
    reference: ReferenceFlight,
    state: EnvState,
    spec: ObservationSpec | None = None,
) -> Array:
    """Body-frame observation vector, independent of world position except through the error.

    Blocks, in order when selected by ``spec`` (all but specific force by default): air
    velocity in body FRD over the reference speed (3), airspeed over the reference speed (1),
    alpha and beta (2), body rates (3), gravity direction in body axes (3), heading error as
    sin and cos (2), position error in body axes over 10 m (3), surface deflections (S),
    propeller speeds as a fraction of maximum (P), specific force in body axes in g (3): what
    an accelerometer reads, the acceleration less gravity. Tracking tasks reference only
    altitude, so their position error is vertical.
    """

    spec = ObservationSpec() if spec is None else spec
    rigid = state.aircraft.rigid_body
    task = task_at(task, state.time_s, rigid)
    environment = current_environment(reference, state)
    air_body = quaternion_rotate_inverse(rigid.attitude, rigid.velocity - environment.wind)
    airspeed = safe_norm(air_body)
    alpha = jnp.arctan2(air_body[..., 2], air_body[..., 0])
    beta = jnp.arcsin(jnp.clip(air_body[..., 1] / jnp.maximum(airspeed, 1e-3), -1.0, 1.0))
    gravity_body = quaternion_rotate_inverse(rigid.attitude, normalize(environment.gravity))
    heading_error = task.heading_error(rigid)
    position_error_body = quaternion_rotate_inverse(rigid.attitude, task.position_error(rigid))
    speed_scale = jnp.maximum(reference_speed(task), 1.0)
    blocks = {
        "air_velocity": air_body / speed_scale,
        "airspeed": (airspeed / speed_scale)[..., None],
        "air_angles": jnp.stack((alpha, beta), axis=-1),
        "rates": rigid.angular_velocity,
        "gravity": gravity_body,
        "heading": jnp.stack((jnp.sin(heading_error), jnp.cos(heading_error)), axis=-1),
        "position_error": position_error_body / 10.0,
        "surfaces": state.aircraft.actuators.surface_deflection,
        "propellers": state.aircraft.actuators.propeller_speed
        / model.actuators.propeller_speed_max,
    }
    if spec.specific_force:
        # The velocity derivative depends on the actuator states, not on the command, so a zero
        # control gives the current acceleration exactly.
        zero = ControlInput(
            propeller=jnp.zeros(model.n_propellers), channel=jnp.zeros(model.n_control_channels)
        )
        acceleration = evaluate_dynamics(
            model, state.aircraft, zero, environment
        ).derivative.rigid_body.velocity
        specific = quaternion_rotate_inverse(rigid.attitude, acceleration - environment.gravity)
        blocks["specific_force"] = specific / GRAVITY_SCALE_M_S2
    selected = [blocks[name] for name in block_sizes(model) if getattr(spec, name)]
    return jnp.concatenate(selected, axis=-1)


class ObservationLayout(NamedTuple):
    """Index slices of the observation vector from :func:`observation` (``None`` for a block
    the spec leaves out)."""

    air_velocity: slice | None
    airspeed: slice | None
    air_angles: slice | None
    rates: slice | None
    gravity: slice | None
    heading: slice | None
    position_error: slice | None
    surfaces: slice | None
    propellers: slice | None
    specific_force: slice | None

    @property
    def air_data(self) -> slice | None:
        """Airspeed, alpha, beta together, when both blocks are present."""

        if self.airspeed is None or self.air_angles is None:
            return None
        return slice(self.airspeed.start, self.air_angles.stop)


OBSERVATION_FIXED_SIZE = 17  # the default spec's size without the surface and propeller blocks


def observation_layout(
    model: AircraftModel, spec: ObservationSpec | None = None
) -> ObservationLayout:
    """Where each block sits in the observation of ``model`` under ``spec``."""

    spec = ObservationSpec() if spec is None else spec
    slices = {}
    offset = 0
    for name, size in block_sizes(model).items():
        if getattr(spec, name):
            slices[name] = slice(offset, offset + size)
            offset += size
        else:
            slices[name] = None
    return ObservationLayout(**slices)


def observation_size(model: AircraftModel, spec: ObservationSpec | None = None) -> int:
    """Length of the observation vector for ``model`` under ``spec``."""

    spec = ObservationSpec() if spec is None else spec
    return sum(size for name, size in block_sizes(model).items() if getattr(spec, name))


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
    faults: FaultSchedule | None = None,
) -> tuple[EnvState, Array, Array, Array, dict[str, Array]]:
    """Hold ``action`` for one control period; returns state, observation, reward, done, info.

    The returned observation carries the sensor ``noise`` (white noise drawn from the episode
    key, plus the bias drawn at reset) and the configured delay. With ``weather`` the wind
    for the period is the mean profile at the aircraft's altitude plus a Dryden gust advanced
    from the episode key; without it the reference wind holds. With an action delay the
    action applied is an earlier one (``info["applied_action"]``); the cost still charges the
    action commanded now. ``faults`` (a :class:`cascade.env.faults.FaultSchedule`) applies
    whatever has failed by this period's time to the actuators; the policy is not told.

    ``done`` is true on a crash (below ``crash_altitude_m``), on leaving the upright envelope
    (the body down axis more than ``upright_limit_rad`` from gravity), or at the horizon; the
    info dict separates ``crashed`` and ``truncated`` and reports the cost.
    ``sensor_age_s``/``sensor_valid`` describe each returned observation element;
    ``sensor_sampled``/``sensor_dropped`` describe acquisition attempts on this tick
    before the whole-observation delay. Ages include all configured delays.
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
    physics = model
    if faults is not None:
        physics = apply_faults(model, faults, state.step / config.control_frequency_hz)
    aircraft, _ = rollout(
        physics,
        state.aircraft,
        controls,
        environment,
        config.simulation_dt_s,
        step=config.step,
    )
    rigid = aircraft.rigid_body
    next_step = state.step + 1
    next_time = next_step / config.control_frequency_hz
    down_body = quaternion_rotate_inverse(rigid.attitude, normalize(reference.environment.gravity))
    attitude_crash = (config.upright_limit_rad < math.pi) & (
        down_body[..., 2] < jnp.cos(config.upright_limit_rad)
    )
    crashed = (-rigid.position[..., 2] < config.crash_altitude_m) | attitude_crash
    truncated = next_step >= config.horizon_steps
    noise = sensor_noise() if noise is None else noise
    key, k_noise, k_gust = jax.random.split(state.key, 3)
    white, _ = _noise_vectors(model, noise, config.observation)
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
        wind = (
            mean_wind_ned(weather, -rigid.position[..., 2])
            + gust_ned
            + discrete_gust_ned(weather, next_step / config.control_frequency_hz)
        )
    advanced = state._replace(
        aircraft=aircraft,
        step=next_step,
        key=key,
        gust=gust,
        wind_ned=wind,
        action_buffer=action_buffer,
        density=_density(config, reference, -rigid.position[..., 2]),
        time_s=next_time,
    )
    active_task = task_at(task, next_time, rigid)
    cost = active_task.cost(rigid, current_environment(reference, advanced), action)
    reward = jnp.where(crashed, 0.0, jnp.exp(-cost))
    sensed = _sense(model, task, reference, advanced, white, k_noise, config.observation)
    pipeline = state.sensor_state
    if config.sensors is not None:
        if pipeline is None:
            raise ValueError("reset must use the same sensor pipeline configuration as step")
        pipeline = step_sensor_pipeline(
            model,
            config.sensors,
            config.observation,
            pipeline,
            sensed,
            jax.random.fold_in(k_noise, 1),
            1.0 / config.control_frequency_hz,
        )
        sensed = pipeline.reading
        sample_step = pipeline.sample_step
        valid = pipeline.valid
        sampled, dropped = pipeline.sampled, pipeline.dropped
    else:
        if pipeline is not None:
            raise ValueError("reset must use the same sensor pipeline configuration as step")
        sample_step = jnp.full(sensed.shape, next_step, jnp.int32)
        valid = jnp.ones(sensed.shape, bool)
        sampled, dropped = valid, jnp.zeros_like(valid)
    buffer = jnp.concatenate((state.observation_buffer[1:], sensed[None]), axis=0)
    sample_buffer = jnp.concatenate(
        (state.observation_sample_step_buffer[1:], sample_step[None]), axis=0
    )
    valid_buffer = jnp.concatenate((state.observation_valid_buffer[1:], valid[None]), axis=0)
    age = jnp.where(
        valid_buffer[0], (next_step - sample_buffer[0]) / config.control_frequency_hz, jnp.inf
    )
    fresh = valid_buffer[0] & (sample_buffer[0] > state.observation_sample_step_buffer[0])
    next_state = advanced._replace(
        observation_buffer=buffer,
        sensor_state=pipeline,
        observation_sample_step_buffer=sample_buffer,
        observation_valid_buffer=valid_buffer,
        sensor_age_s=age,
        sensor_valid=valid_buffer[0],
    )
    info = {
        "cost": cost,
        "crashed": crashed,
        "truncated": truncated,
        "applied_action": applied,
        "sensor_age_s": age,
        "sensor_valid": valid_buffer[0],
        "sensor_fresh": fresh,
        "sensor_sampled": sampled,
        "sensor_dropped": dropped,
    }
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
    faults: FaultSchedule | None = None,
) -> tuple[EnvState, tuple[Array, Array, Array]]:
    """Scan a time-major action sequence; returns the final state and (observations, rewards,
    dones). Rewards after the first ``done`` are zeroed, so the sum is the episode return."""

    def scan_step(carry, action):
        state, finished = carry
        next_state, obs, reward, done, _ = step(
            model, config, task, reference, state, action, noise, weather, faults
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
    faults: FaultSchedule | None = None,
) -> tuple[EnvState, tuple[Array, Array, Array, Array]]:
    """Scan a policy over the horizon; returns the final state and (observations, actions,
    rewards, dones), rewards zeroed after the first ``done``.

    ``policy(policy_state, observation, env_state) -> (action, policy_state)``: a learned policy
    reads the observation and ignores the environment state; a model-based baseline such as
    :func:`cascade_policy` may read the state directly. Wrap a two-argument
    ``policy(policy_state, SensorObservation)`` with :func:`cascade.env.sensor_policy`
    to receive the delivered values, acquisition ages and validity without hidden
    aircraft state. The rollout's returned observations remain ordinary value arrays.
    """

    first_observation = state.observation_buffer[0]

    def scan_step(carry, _):
        state, obs, policy_state, finished = carry
        action, policy_state = policy(policy_state, obs, state)
        next_state, next_obs, reward, done, _ = step(
            model, config, task, reference, state, action, noise, weather, faults
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
