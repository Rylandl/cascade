"""Observation-only policy inputs and adapters for the episode callback interface."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, NamedTuple, TypeVar

from jax import Array

if TYPE_CHECKING:
    from cascade.env.episode import EnvState

Memory = TypeVar("Memory")


class SensorObservation(NamedTuple):
    """Delivered measurements with per-element acquisition ages and validity.

    All three arrays follow the selected observation layout. ``age_s`` is in
    seconds and ``valid`` is boolean. Unavailable elements have value zero,
    infinite age and false validity; held readings remain valid while aging.
    Both block transport and whole-observation delay are included in the ages.
    This NamedTuple is a JAX PyTree; it contains no hidden aircraft state.
    """

    values: Array
    age_s: Array
    valid: Array


def sensor_observation(state: EnvState) -> SensorObservation:
    """Return the current delivered observation and its metadata from episode state.

    This reads the oldest observation-buffer entry after all configured delays,
    without recomputing a measurement from the aircraft. It works with or without
    a sensor pipeline and with leading batch dimensions, including under ``vmap``.
    Use :func:`sensor_policy` inside a rollout to preserve its supplied observation.
    """

    return SensorObservation(
        state.observation_buffer[..., 0, :], state.sensor_age_s, state.sensor_valid
    )


def sensor_policy(
    policy: Callable[[Memory, SensorObservation], tuple[Array, Memory]],
) -> Callable[[Memory, Array, EnvState], tuple[Array, Memory]]:
    """Adapt ``policy(memory, SensorObservation)`` to the legacy episode callback.

    The returned function accepts ``(memory, observation, env_state)`` and passes
    only the supplied observation plus ``sensor_age_s``/``sensor_valid`` to the
    inner policy. It never reads the aircraft, recomputes true observations, or
    substitutes the state's observation buffer. The supplied values and state
    metadata must describe the same step and observation layout, as they do in
    ``rollout_policy`` and experiment ``Policy`` factories.

    The wrapper is compatible with ``jit``, ``vmap`` and differentiation whenever
    the inner policy is. It does not clip actions, impute missing readings, reject
    stale samples or sanitize infinite ages: those decisions belong to the policy.
    """

    if not callable(policy):
        raise TypeError("sensor_policy requires a callable policy")

    @wraps(policy)
    def adapted(memory: Memory, observation: Array, env_state: EnvState):
        reading = SensorObservation(observation, env_state.sensor_age_s, env_state.sensor_valid)
        return policy(memory, reading)

    return adapted
