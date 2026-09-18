"""What a policy observes, and how noisily.

:class:`ObservationSpec` selects the blocks of :func:`cascade.env.observation`; the default
is everything (a privileged observation useful for learning research), and
:func:`onboard_observation` is what a small autopilot actually measures: rates and specific
force from an IMU, an attitude estimate (gravity direction and heading), a pitot airspeed,
and a GNSS position error. :class:`SensorNoise` is white noise per block plus per-episode
biases on the gyros and accelerometers; :func:`sensor_noise_from_sensors` builds it from
datasheet units (m/s, rad, rad/s, m/s^2, m) for a task, so the conversion into observation
units is the library's, not the user's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.model import AircraftModel

GRAVITY_SCALE_M_S2 = 9.80665  # the specific-force block is in g units
POSITION_SCALE_M = 10.0


@dataclass(frozen=True)
class SensorBlockConfig:
    """Sampling/transport behavior for one complete observation block.

    Periods and jitter bounds are integer control steps. Acquisition intervals are
    uniform in ``sample_period_steps +/- sample_jitter_steps``; packet latency is
    uniform in ``delay_steps +/- delay_jitter_steps``. A dropout discards a whole
    acquired block. ``bias_walk_std`` is independent per-element Brownian drift in
    observation units per square-root second, in addition to :class:`SensorNoise`.
    """

    sample_period_steps: int = 1
    sample_jitter_steps: int = 0
    delay_steps: int = 0
    delay_jitter_steps: int = 0
    dropout_probability: float = 0.0
    bias_walk_std: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "sample_period_steps",
            "sample_jitter_steps",
            "delay_steps",
            "delay_jitter_steps",
        ):
            value = getattr(self, name)
            minimum = 1 if name == "sample_period_steps" else 0
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.sample_jitter_steps >= self.sample_period_steps:
            raise ValueError("sample_jitter_steps must be smaller than sample_period_steps")
        if self.delay_jitter_steps > self.delay_steps:
            raise ValueError("delay_jitter_steps must not exceed delay_steps")
        for name in ("dropout_probability", "bias_walk_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0 or (name == "dropout_probability" and value > 1.0):
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class SensorPipelineConfig:
    """Static block settings for ``EpisodeConfig(sensors=...)``.

    Blocks share acquisition/dropout/latency decisions across their elements. All
    defaults reproduce the existing sensor readings exactly. Unselected observation
    blocks are ignored. The usual whole-observation delay is applied afterwards.
    """

    air_velocity: SensorBlockConfig = SensorBlockConfig()
    airspeed: SensorBlockConfig = SensorBlockConfig()
    air_angles: SensorBlockConfig = SensorBlockConfig()
    rates: SensorBlockConfig = SensorBlockConfig()
    gravity: SensorBlockConfig = SensorBlockConfig()
    heading: SensorBlockConfig = SensorBlockConfig()
    position_error: SensorBlockConfig = SensorBlockConfig()
    surfaces: SensorBlockConfig = SensorBlockConfig()
    propellers: SensorBlockConfig = SensorBlockConfig()
    specific_force: SensorBlockConfig = SensorBlockConfig()

    def __post_init__(self) -> None:
        for field in fields(self):
            if not isinstance(getattr(self, field.name), SensorBlockConfig):
                raise ValueError(f"{field.name} must be a SensorBlockConfig")


class SensorState(NamedTuple):
    """JAX pipeline state; readings/flags/sample steps follow observation element order.

    ``valid`` means a sample has arrived at least once; held readings remain valid
    while their age grows. Missing readings are zero until the first delivery.
    ``sampled``/``dropped`` describe the current acquisition, and ``updated`` marks
    delivery of a newer sample. Older reordered packets never replace newer data.
    Queue rows are internal transport slots, not additional observation dimensions.
    """

    reading: Array
    bias_walk: Array
    step: Array
    next_sample_step: Array
    sample_step: Array
    valid: Array
    sampled: Array
    dropped: Array
    updated: Array
    pending_values: Array
    pending_sample_steps: Array
    pending_delivery_steps: Array
    pending_valid: Array


class ObservationSpec(NamedTuple):
    """Which observation blocks a policy sees, in this order when present: air velocity in
    body axes (3), airspeed (1), alpha and beta (2), body rates (3), gravity direction in body
    axes (3), heading error sin and cos (2), position error in body axes (3), surface
    deflections (S), propeller speed fractions (P), specific force in body axes in g (3)."""

    air_velocity: bool = True
    airspeed: bool = True
    air_angles: bool = True
    rates: bool = True
    gravity: bool = True
    heading: bool = True
    position_error: bool = True
    surfaces: bool = True
    propellers: bool = True
    specific_force: bool = False


def full_observation() -> ObservationSpec:
    return ObservationSpec()


def onboard_observation() -> ObservationSpec:
    """Rates and specific force (IMU), gravity direction and heading (attitude estimate), pitot
    airspeed, and position error (GNSS): no flow angles, no actuator states."""

    return ObservationSpec(
        air_velocity=False,
        airspeed=True,
        air_angles=False,
        rates=True,
        gravity=True,
        heading=True,
        position_error=True,
        surfaces=False,
        propellers=False,
        specific_force=True,
    )


def block_sizes(model: AircraftModel) -> dict[str, int]:
    return {
        "air_velocity": 3,
        "airspeed": 1,
        "air_angles": 2,
        "rates": 3,
        "gravity": 3,
        "heading": 2,
        "position_error": 3,
        "surfaces": model.n_surfaces,
        "propellers": model.n_propellers,
        "specific_force": 3,
    }


def _pipeline_blocks(model, config, spec):
    selected = [
        (name, size)
        for name, size in block_sizes(model).items()
        if getattr(spec, name) and size > 0
    ]
    if not selected:
        raise ValueError("sensor pipeline requires at least one observation element")
    blocks = tuple(getattr(config, name) for name, _ in selected)
    ids = jnp.asarray([i for i, (_, size) in enumerate(selected) for _ in range(size)], jnp.int32)
    return blocks, ids


def initialize_sensor_pipeline(
    model: AircraftModel,
    config: SensorPipelineConfig,
    spec: ObservationSpec,
    reading: Array,
    key: Array,
) -> SensorState:
    """Acquire at time zero, applying configured dropout and transport latency.

    ``reading`` includes the existing white noise and episode bias. Drift starts at
    zero. Until a first sample arrives the output is zero with ``valid=False``.
    This and :func:`step_sensor_pipeline` are pure JIT/vmap-compatible functions.
    """
    blocks, ids = _pipeline_blocks(model, config, spec)
    if reading.shape != ids.shape:
        raise ValueError("reading shape must match the selected observation layout")
    depth = 1 + max(block.delay_steps + block.delay_jitter_steps for block in blocks)
    state = SensorState(
        reading=jnp.zeros_like(reading),
        bias_walk=jnp.zeros_like(reading),
        step=jnp.asarray(0, jnp.int32),
        next_sample_step=jnp.zeros((len(blocks),), jnp.int32),
        sample_step=jnp.full(reading.shape, -1, jnp.int32),
        valid=jnp.zeros(reading.shape, bool),
        sampled=jnp.zeros(reading.shape, bool),
        dropped=jnp.zeros(reading.shape, bool),
        updated=jnp.zeros(reading.shape, bool),
        pending_values=jnp.zeros((depth, *reading.shape), reading.dtype),
        pending_sample_steps=jnp.full((depth, len(blocks)), -1, jnp.int32),
        pending_delivery_steps=jnp.full((depth, len(blocks)), -1, jnp.int32),
        pending_valid=jnp.zeros((depth, len(blocks)), bool),
    )
    return _update_pipeline(blocks, ids, state, reading, key, state.step, 0.0)


def step_sensor_pipeline(
    model: AircraftModel,
    config: SensorPipelineConfig,
    spec: ObservationSpec,
    state: SensorState,
    reading: Array,
    key: Array,
    dt_s: float,
) -> SensorState:
    """Advance one control period and return the latest delivered sensor values.

    Drift evolves every period, including unsampled/dropped intervals, by independent
    increments ``bias_walk_std * sqrt(dt_s) * N(0, 1)``. Acquisition, dropout, and
    delivery occur on control ticks; no sub-period interpolation is implied.
    """
    if not isinstance(dt_s, jax.core.Tracer) and (
        isinstance(dt_s, bool) or not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0
    ):
        raise ValueError("dt_s must be a finite positive scalar")
    blocks, ids = _pipeline_blocks(model, config, spec)
    if reading.shape != state.reading.shape or reading.shape != ids.shape:
        raise ValueError("reading shape must match the selected observation layout")
    depth = 1 + max(block.delay_steps + block.delay_jitter_steps for block in blocks)
    if state.pending_values.shape != (depth, reading.shape[0]) or state.next_sample_step.shape != (
        len(blocks),
    ):
        raise ValueError("sensor pipeline state must match its initialization configuration")
    return _update_pipeline(blocks, ids, state, reading, key, state.step + 1, dt_s)


def _update_pipeline(blocks, ids, state, reading, key, step, dt_s):
    k_walk, k_interval, k_delay, k_drop = jax.random.split(key, 4)

    def vector(name, dtype=jnp.int32):
        return jnp.asarray([getattr(block, name) for block in blocks], dtype)

    walk = vector("bias_walk_std", reading.dtype)[ids]
    drift = state.bias_walk + walk * jnp.sqrt(jnp.asarray(dt_s, reading.dtype)) * jax.random.normal(
        k_walk, reading.shape, dtype=reading.dtype
    )
    sampled = step >= state.next_sample_step
    jitter = vector("sample_jitter_steps")
    interval = vector("sample_period_steps") + jax.random.randint(
        k_interval, sampled.shape, -jitter, jitter + 1, dtype=jnp.int32
    )
    next_sample = jnp.where(sampled, step + interval, state.next_sample_step).astype(jnp.int32)
    delay_jitter = vector("delay_jitter_steps")
    delay = vector("delay_steps") + jax.random.randint(
        k_delay, sampled.shape, -delay_jitter, delay_jitter + 1, dtype=jnp.int32
    )
    dropped = sampled & (
        jax.random.uniform(k_drop, sampled.shape, dtype=reading.dtype)
        < vector("dropout_probability", reading.dtype)
    )
    acquired = sampled & ~dropped
    slot = jnp.mod(step, state.pending_values.shape[0])
    values = state.pending_values.at[slot].set(reading + drift)
    acquired_steps = state.pending_sample_steps.at[slot].set(step)
    delivery_steps = state.pending_delivery_steps.at[slot].set(step + delay)
    pending = state.pending_valid.at[slot].set(acquired)
    ready = pending & (delivery_steps <= step)
    available = jnp.where(ready, acquired_steps, -1)
    newest = jnp.max(available, axis=0)
    row = jnp.argmax(available, axis=0)
    updated = newest[ids] > state.sample_step
    selected = values[row[ids], jnp.arange(reading.shape[0])]
    return SensorState(
        reading=jnp.where(updated, selected, state.reading),
        bias_walk=drift,
        step=step,
        next_sample_step=next_sample,
        sample_step=jnp.where(updated, newest[ids], state.sample_step),
        valid=state.valid | updated,
        sampled=sampled[ids],
        dropped=dropped[ids],
        updated=updated,
        pending_values=values,
        pending_sample_steps=acquired_steps,
        pending_delivery_steps=delivery_steps,
        pending_valid=pending & ~ready,
    )


class SensorNoise(NamedTuple):
    """Per-block white-noise standard deviations in observation units, plus per-episode bias
    standard deviations for the gyros and accelerometers (drawn once in :func:`reset`).
    Zeros (the default) give the true state. See :func:`sensor_noise_from_sensors` for
    datasheet units."""

    air_std: Array
    angle_std: Array
    rate_std: Array
    rate_bias_std: Array
    gravity_std: Array
    heading_std: Array
    position_std: Array
    actuator_std: Array
    specific_force_std: Array
    specific_force_bias_std: Array


def sensor_noise(
    *,
    air_std: float = 0.0,
    angle_std: float = 0.0,
    rate_std: float = 0.0,
    rate_bias_std: float = 0.0,
    gravity_std: float = 0.0,
    heading_std: float = 0.0,
    position_std: float = 0.0,
    actuator_std: float = 0.0,
    specific_force_std: float = 0.0,
    specific_force_bias_std: float = 0.0,
) -> SensorNoise:
    values = (
        air_std,
        angle_std,
        rate_std,
        rate_bias_std,
        gravity_std,
        heading_std,
        position_std,
        actuator_std,
        specific_force_std,
        specific_force_bias_std,
    )
    for name, value in zip(SensorNoise._fields, values, strict=True):
        if isinstance(value, jax.core.Tracer):
            continue
        array = np.asarray(value)
        if (
            array.shape != ()
            or array.dtype.kind not in "iuf"
            or not math.isfinite(float(array))
            or float(array) < 0.0
        ):
            raise ValueError(f"{name} must be a finite nonnegative scalar")
    return SensorNoise(*(jnp.asarray(value) for value in values))


def sensor_noise_from_sensors(
    reference_speed_m_s: float,
    *,
    airspeed_std_m_s: float = 0.0,
    angle_std_rad: float = 0.0,
    gyro_std_rad_s: float = 0.0,
    gyro_bias_std_rad_s: float = 0.0,
    accelerometer_std_m_s2: float = 0.0,
    accelerometer_bias_std_m_s2: float = 0.0,
    attitude_std_rad: float = 0.0,
    heading_std_rad: float = 0.0,
    position_std_m: float = 0.0,
    actuator_std_rad: float = 0.0,
) -> SensorNoise:
    """Noise from datasheet units for a task whose observations are scaled by
    ``reference_speed_m_s`` (the task's own reference speed). Small angles: an attitude error
    of ``a`` rad moves a unit-vector component by about ``a``."""

    if (
        isinstance(reference_speed_m_s, bool)
        or not math.isfinite(float(reference_speed_m_s))
        or float(reference_speed_m_s) <= 0.0
    ):
        raise ValueError("reference_speed_m_s must be finite and positive")
    speed = max(float(reference_speed_m_s), 1.0)
    return sensor_noise(
        air_std=airspeed_std_m_s / speed,
        angle_std=angle_std_rad,
        rate_std=gyro_std_rad_s,
        rate_bias_std=gyro_bias_std_rad_s,
        gravity_std=attitude_std_rad,
        heading_std=heading_std_rad,
        position_std=position_std_m / POSITION_SCALE_M,
        actuator_std=actuator_std_rad,
        specific_force_std=accelerometer_std_m_s2 / GRAVITY_SCALE_M_S2,
        specific_force_bias_std=accelerometer_bias_std_m_s2 / GRAVITY_SCALE_M_S2,
    )


def _noise_vectors(
    model: AircraftModel, noise: SensorNoise, spec: ObservationSpec | None = None
) -> tuple[Array, Array]:
    """White-noise and bias standard deviations laid out like the observation of ``spec``."""

    spec = ObservationSpec() if spec is None else spec
    sizes = block_sizes(model)
    white_by_block = {
        "air_velocity": noise.air_std,
        "airspeed": noise.air_std,
        "air_angles": noise.angle_std,
        "rates": noise.rate_std,
        "gravity": noise.gravity_std,
        "heading": noise.heading_std,
        "position_error": noise.position_std,
        "surfaces": noise.actuator_std,
        "propellers": noise.actuator_std,
        "specific_force": noise.specific_force_std,
    }
    bias_by_block = {"rates": noise.rate_bias_std, "specific_force": noise.specific_force_bias_std}
    white, bias = [], []
    for name, size in sizes.items():
        if not getattr(spec, name):
            continue
        white.append(jnp.broadcast_to(white_by_block[name], (size,)))
        bias.append(jnp.broadcast_to(bias_by_block.get(name, jnp.zeros(())), (size,)))
    return jnp.concatenate(white), jnp.concatenate(bias)


__all__ = [
    "GRAVITY_SCALE_M_S2",
    "ObservationSpec",
    "SensorBlockConfig",
    "SensorNoise",
    "SensorPipelineConfig",
    "SensorState",
    "block_sizes",
    "full_observation",
    "initialize_sensor_pipeline",
    "onboard_observation",
    "sensor_noise",
    "sensor_noise_from_sensors",
    "step_sensor_pipeline",
]
