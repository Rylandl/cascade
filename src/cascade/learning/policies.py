"""Small sensor-only policies with explicit recurrent memory and saved normalization."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.env import SensorObservation


@dataclass(frozen=True)
class PolicyConfig:
    """Static architecture and feature normalization, serialized with a policy.

    Features concatenate scaled/clipped values, scaled/clipped acquisition ages, and
    validity indicators, in that order. The observation is the selected environment
    vector, not privileged state. Stale finite samples remain valid; their clipped age
    is supplied to the network. Missing/malformed samples get zero values and maximum age.
    """

    observation_size: int
    action_size: int
    hidden_size: int = 32
    architecture: str = "feedforward"
    observation_scale: tuple[float, ...] | None = None
    observation_clip: float = 10.0
    age_scale_s: float = 0.1
    age_clip_s: float = 1.0
    residual_scale: float = 0.5

    def __post_init__(self):
        for name in ("observation_size", "action_size", "hidden_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.architecture not in {"feedforward", "recurrent"}:
            raise ValueError("architecture must be feedforward or recurrent")
        for name in ("observation_clip", "age_scale_s", "age_clip_s", "residual_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not np.finfo(np.float32).tiny <= value <= np.finfo(np.float32).max
            ):
                raise ValueError(f"{name} must be finite, positive and representable in float32")
            object.__setattr__(self, name, float(value))
        if self.residual_scale > 2:
            raise ValueError("residual_scale must be at most two normalized action units")
        if self.observation_scale is not None:
            values = np.asarray(self.observation_scale)
            if (
                values.shape != (self.observation_size,)
                or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()
                or np.any(values < np.finfo(np.float32).tiny)
                or np.any(values > np.finfo(np.float32).max)
            ):
                raise ValueError(
                    "observation_scale must have one finite positive float32 scale per value"
                )
            object.__setattr__(self, "observation_scale", tuple(float(value) for value in values))


def parameter_shapes(config: PolicyConfig) -> dict[str, tuple[int, ...]]:
    """Named floating parameter leaves; no hidden configuration is stored in the PyTree."""
    result = {
        "input_weight": (3 * config.observation_size, config.hidden_size),
        "input_bias": (config.hidden_size,),
        "output_weight": (config.hidden_size, config.action_size),
        "output_bias": (config.action_size,),
    }
    if config.architecture == "recurrent":
        result["recurrent_weight"] = (config.hidden_size, config.hidden_size)
    return result


def initialize_policy(config: PolicyConfig, key: Array) -> dict[str, Array]:
    """Initialize exactly at the separately saved trim action, with small hidden weights."""
    input_key, recurrent_key = jax.random.split(key)
    shapes = parameter_shapes(config)
    parameters = {name: jnp.zeros(shape, jnp.float32) for name, shape in shapes.items()}
    parameters["input_weight"] = (
        0.1
        / jnp.sqrt(float(shapes["input_weight"][0]))
        * jax.random.normal(input_key, shapes["input_weight"], dtype=jnp.float32)
    )
    if config.architecture == "recurrent":
        parameters["recurrent_weight"] = (
            0.1
            / jnp.sqrt(float(config.hidden_size))
            * jax.random.normal(recurrent_key, shapes["recurrent_weight"], dtype=jnp.float32)
        )
    return parameters


def initial_memory(config: PolicyConfig, batch_shape: tuple[int, ...] = ()) -> Array:
    """Zero memory for an episode reset; batching adds leading dimensions."""
    if not isinstance(batch_shape, tuple) or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size < 0 for size in batch_shape
    ):
        raise ValueError("batch_shape must be a tuple of nonnegative integers")
    return jnp.zeros((*batch_shape, config.hidden_size), jnp.float32)


def sensor_features(observation: SensorObservation, config: PolicyConfig) -> Array:
    """Finite bounded features; malformed values cannot become giant neural inputs."""
    if not isinstance(observation, SensorObservation):
        raise TypeError("policy input must be SensorObservation(values, age_s, valid)")
    values, ages, valid = map(jnp.asarray, observation)
    if values.ndim < 1 or values.shape[-1] != config.observation_size:
        raise ValueError("observation values do not match PolicyConfig.observation_size")
    if ages.shape != values.shape or valid.shape != values.shape:
        raise ValueError("sensor ages and validity must match observation values")
    if not jnp.issubdtype(values.dtype, jnp.floating) or not jnp.issubdtype(
        ages.dtype, jnp.floating
    ):
        raise ValueError("sensor values and ages must be floating point")
    if valid.dtype != jnp.bool_:
        raise ValueError("sensor validity must be boolean")
    usable = valid & jnp.isfinite(values) & jnp.isfinite(ages) & (ages >= 0)
    safe_values = jnp.where(usable, values, 0.0)
    safe_ages = jnp.where(usable, ages, config.age_clip_s)
    scales = (
        jnp.ones(config.observation_size)
        if config.observation_scale is None
        else jnp.asarray(config.observation_scale)
    )
    # Clip before division too, so finite extremes and small scales do not overflow.
    scaled_values = (
        jnp.clip(safe_values, -config.observation_clip * scales, config.observation_clip * scales)
        / scales
    )
    scaled_values = jnp.clip(scaled_values, -config.observation_clip, config.observation_clip)
    normalized_ages = jnp.minimum(safe_ages, config.age_clip_s) / config.age_scale_s
    normalized_ages = jnp.clip(normalized_ages, 0.0, config.observation_clip)
    return jnp.concatenate((scaled_values, normalized_ages, usable.astype(values.dtype)), axis=-1)


def policy_step(
    parameters: dict[str, Array],
    memory: Array,
    observation: SensorObservation,
    config: PolicyConfig,
    trim_action: Array,
) -> tuple[Array, Array]:
    """Pure public-input policy update; close over config or mark it static under JIT.

    Output is ``clip(trim + residual_scale*tanh(network), -1, 1)``. Feedforward
    policies return zero memory; recurrent policies return their new tanh hidden state.
    Use ``vmap`` or matching leading observation/memory dimensions for independent worlds.
    """
    shapes = parameter_shapes(config)
    if set(parameters) != set(shapes) or any(
        parameters[name].shape != shape for name, shape in shapes.items()
    ):
        raise ValueError("policy parameter names/shapes do not match configuration")
    features = sensor_features(observation, config).astype(parameters["input_weight"].dtype)
    memory, trim_action = jnp.asarray(memory), jnp.asarray(trim_action)
    if memory.shape != (*features.shape[:-1], config.hidden_size):
        raise ValueError("policy memory shape does not match observation batch and hidden size")
    if trim_action.shape != (config.action_size,):
        raise ValueError("trim_action must have one normalized value per action")
    hidden = features @ parameters["input_weight"] + parameters["input_bias"]
    if config.architecture == "recurrent":
        hidden = hidden + memory @ parameters["recurrent_weight"]
    hidden = jnp.tanh(hidden)
    residual = jnp.tanh(hidden @ parameters["output_weight"] + parameters["output_bias"])
    action = jnp.clip(trim_action + config.residual_scale * residual, -1.0, 1.0)
    next_memory = hidden if config.architecture == "recurrent" else jnp.zeros_like(memory)
    return action, next_memory


def make_policy(parameters, config: PolicyConfig, trim_action):
    """Adapt the public sensor-only policy to existing episode rollouts without hidden state."""
    from cascade.env import sensor_policy

    trim_action = jnp.asarray(trim_action)
    if trim_action.shape != (config.action_size,):
        raise ValueError("trim_action must have one normalized value per action")
    if not isinstance(trim_action, jax.core.Tracer) and (
        not np.isfinite(np.asarray(trim_action)).all()
        or np.any(np.abs(np.asarray(trim_action)) > 1)
    ):
        raise ValueError("trim_action must be finite and in [-1, 1]")

    def public(memory, observation):
        return policy_step(parameters, memory, observation, config, trim_action)

    return sensor_policy(public), initial_memory(config)
