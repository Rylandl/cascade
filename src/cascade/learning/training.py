"""Resumable finite-safe Adam ascent over caller-declared episode objectives."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.learning.policies import PolicyConfig, make_policy


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.003
    gradient_clip: float = 10.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, Integral)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        object.__setattr__(self, "batch_size", int(self.batch_size))
        for name in ("learning_rate", "gradient_clip", "epsilon", "beta1", "beta2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in {"beta1", "beta2"}:
                if not 0 <= value < 1 or np.float32(value) >= 1:
                    raise ValueError(f"{name} must stay in [0, 1) in float32")
            elif not np.finfo(np.float32).tiny <= value <= np.finfo(np.float32).max:
                raise ValueError(f"{name} must be positive and representable in float32")
            object.__setattr__(self, name, float(value))


class TrainingState(NamedTuple):
    parameters: object
    first_moment: object
    second_moment: object
    key: Array
    iteration: Array


class TrainingMetrics(NamedTuple):
    objective: Array
    gradient_norm: Array
    accepted: Array
    iteration: Array


def initialize_training(parameters, key) -> TrainingState:
    """Validate floating leaves and initialize moments, preserving the supplied PRNG key.

    A single typed or legacy JAX key is accepted. Typed keys retain their explicit
    implementation; legacy keys use the active JAX PRNG setting.
    """
    parameters = jax.tree.map(jnp.asarray, parameters)
    leaves = jax.tree.leaves(parameters)
    if not leaves or any(
        leaf.size == 0
        or leaf.dtype not in (jnp.dtype("float32"), jnp.dtype("float64"))
        or not np.isfinite(np.asarray(leaf)).all()
        for leaf in leaves
    ):
        raise ValueError("parameters must have nonempty finite float32/float64 leaves")
    try:
        key = jnp.asarray(key)
        implementation = jax.random.key_impl(key)
        raw_key = jax.random.key_data(key)
        expected = jax.random.key_data(jax.random.key(0, impl=implementation))
        if raw_key.shape != expected.shape or raw_key.dtype != jnp.uint32:
            raise ValueError("training requires a single PRNG key")
        jax.random.wrap_key_data(raw_key, impl=implementation)
    except (TypeError, ValueError) as exc:
        raise ValueError("training key must be a valid single JAX PRNG key") from exc
    zero = jax.tree.map(jnp.zeros_like, parameters)
    return TrainingState(parameters, zero, zero, key, jnp.asarray(0, jnp.int32))


def _all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)]))


def _clip_gradient(gradient, limit):
    """Scale before clipping to avoid squaring raw large gradients."""
    safe = jax.tree.map(lambda value: jnp.where(jnp.isfinite(value), value, 0.0), gradient)
    maximum = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in jax.tree.leaves(safe)]))
    denominator = jnp.where(maximum > 0, maximum, 1.0)
    normalized = jax.tree.map(lambda leaf: leaf / denominator, safe)
    norm = jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in jax.tree.leaves(normalized)))
    magnitude = jnp.minimum(maximum, limit / jnp.maximum(norm, 1.0))
    clipped = jax.tree.map(
        lambda leaf, original: (leaf * magnitude).astype(original.dtype), normalized, gradient
    )
    reported = jnp.minimum(maximum * norm, jnp.finfo(maximum.dtype).max)
    return clipped, reported


def make_train_step(objective, config: TrainingConfig):
    """Compile one Adam ascent update for scalar ``objective(parameters, batch_keys)``.

    Only finite objective/gradient/candidate updates are accepted. Rejection preserves
    parameters, moments and the accepted-update counter, advancing only the PRNG key.
    The host :func:`train` raises on rejection instead of reporting successful training.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")

    def advance(state: TrainingState):
        next_key, batch_key = jax.random.split(state.key)
        keys = jax.random.split(batch_key, config.batch_size)
        value, raw_gradient = jax.value_and_grad(objective)(state.parameters, keys)
        gradient, norm = _clip_gradient(raw_gradient, config.gradient_clip)

        def moment(old, g, beta, *, squared):
            beta = jnp.asarray(beta, old.dtype)
            return beta * old + (1 - beta) * (g * g if squared else g)

        first = jax.tree.map(
            lambda old, g: moment(old, g, config.beta1, squared=False), state.first_moment, gradient
        )
        second = jax.tree.map(
            lambda old, g: moment(old, g, config.beta2, squared=True), state.second_moment, gradient
        )
        iteration = state.iteration + 1

        def update_parameter(old, m, v):
            # Use the same rounded beta in moments and their correction. Python's
            # precomputed (1 - beta) can otherwise disagree badly near beta == 1.
            beta1, beta2 = (
                jnp.asarray(config.beta1, old.dtype),
                jnp.asarray(config.beta2, old.dtype),
            )
            first_scale, second_scale = 1 - beta1**iteration, 1 - beta2**iteration
            return old + config.learning_rate * (m / first_scale) / (
                jnp.sqrt(v / second_scale) + config.epsilon
            )

        parameters = jax.tree.map(update_parameter, state.parameters, first, second)
        accepted = (
            jnp.isfinite(value)
            & _all_finite(raw_gradient)
            & _all_finite((parameters, first, second))
            & (iteration > state.iteration)
        )

        def choose(new, old):
            return jnp.where(accepted, new, old)

        next_state = TrainingState(
            jax.tree.map(choose, parameters, state.parameters),
            jax.tree.map(choose, first, state.first_moment),
            jax.tree.map(choose, second, state.second_moment),
            next_key,
            jnp.where(accepted, iteration, state.iteration),
        )
        return next_state, TrainingMetrics(value, norm, accepted, next_state.iteration)

    return jax.jit(advance)


def train(state: TrainingState, train_step, iterations: int, callback=None):
    """Run further updates from this exact optimizer/key state, retaining per-step metrics.

    ``callback(state, metrics)`` runs after each accepted update. Failed updates raise
    ``FloatingPointError`` with ``state`` and ``metrics`` attributes for inspection/resume.
    Iterations count attempted updates in this call; zero is a no-op.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    history = []
    for _ in range(iterations):
        state, metrics = train_step(state)
        if not bool(metrics.accepted):
            error = FloatingPointError(
                "training rejected a nonfinite objective, gradient or optimizer update"
            )
            error.state, error.metrics = state, metrics
            raise error
        history.append(metrics)
        if callback is not None:
            callback(state, metrics)
    return state, tuple(history)


def make_episode_return_objective(
    model,
    episode_config,
    task,
    reference,
    policy_config: PolicyConfig,
    trim_action,
    *,
    noise=None,
    weather=None,
    faults=None,
):
    """Mean differentiable episode return over supplied keys, with public sensor inputs.

    Supports tracking and scheduled tracking; data-split/seed-pool selection belongs to
    the caller. Every episode starts with independent zero policy memory. Noise, weather
    and faults use the same declared episode path as evaluation.
    """
    from cascade.env import (
        ScheduledTrackingTask,
        TrackingTask,
        action_size,
        observation_size,
        reset,
        rollout_policy,
    )

    if not isinstance(task, (TrackingTask, ScheduledTrackingTask)):
        raise TypeError("learning objective requires tracking or scheduled tracking")
    if policy_config.observation_size != observation_size(model, episode_config.observation) or (
        policy_config.action_size != action_size(model)
    ):
        raise ValueError("policy dimensions do not match the episode observation/action contract")

    def objective(parameters, keys):
        policy, memory = make_policy(parameters, policy_config, trim_action)

        def episode(key):
            state, _ = reset(
                model, episode_config, task, reference, key, noise=noise, weather=weather
            )
            _, (_, _, rewards, _) = rollout_policy(
                model,
                episode_config,
                task,
                reference,
                state,
                policy,
                memory,
                noise=noise,
                weather=weather,
                faults=faults,
            )
            return jnp.sum(rewards)

        return jnp.mean(jax.vmap(episode)(keys))

    return objective
