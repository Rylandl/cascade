"""A single-episode Gymnasium-style wrapper over the pure episode functions.

Gymnasium is not a dependency: when it is installed this subclasses ``gymnasium.Env`` with
real spaces, otherwise it is a plain class with the same ``reset(seed=...)`` / ``step(action)``
contract. Batched training should stay in JAX and vmap the functions directly; this is for
tooling that expects the Gymnasium interface (evaluation scripts, wrappers, video loggers).
"""

from __future__ import annotations

import jax
import numpy as np

from cascade.env import (
    EpisodeConfig,
    action_size,
    observation_size,
    reset,
    step,
    tracking_task,
    trimmed_reference,
)
from cascade.reference import aerobatic_reference

try:  # pragma: no cover - depends on the optional package
    import gymnasium

    Base = gymnasium.Env
except ImportError:  # pragma: no cover
    gymnasium = None
    Base = object


class CascadeEnv(Base):
    """One aircraft, one task, one episode at a time, NumPy in and out."""

    metadata = {"render_modes": []}

    def __init__(self, model, task, config: EpisodeConfig | None = None, reference=None):
        self.model = model
        self.task = task
        self.config = EpisodeConfig() if config is None else config
        self.reference = trimmed_reference(model, task) if reference is None else reference
        self._reset = jax.jit(lambda key: reset(model, self.config, task, self.reference, key))
        self._step = jax.jit(lambda s, a: step(model, self.config, task, self.reference, s, a))
        self._state = None
        low, high = -np.inf, np.inf
        if gymnasium is not None:
            self.observation_space = gymnasium.spaces.Box(
                low, high, (observation_size(model, self.config.observation),), np.float32
            )
            self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (action_size(model),), np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        key = jax.random.PRNGKey(0 if seed is None else seed)
        self._state, observation = self._reset(key)
        return np.asarray(observation), {}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("call reset() first")
        self._state, observation, reward, done, info = self._step(
            self._state, np.asarray(action, dtype=np.float32)
        )
        terminated = bool(info["crashed"])
        truncated = bool(info["truncated"])
        return (
            np.asarray(observation),
            float(reward),
            terminated,
            truncated,
            {"cost": float(info["cost"])},
        )


def main() -> None:
    model = aerobatic_reference()
    env = CascadeEnv(model, tracking_task(12.0, 50.0, 0.0), EpisodeConfig(horizon_steps=80))
    observation, _ = env.reset(seed=3)
    total = 0.0
    for _ in range(80):
        action = np.zeros(action_size(model), dtype=np.float32)
        action[0] = 0.2  # a mild throttle above the neutral mapping
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    installed = gymnasium is not None
    print(f"gymnasium installed: {installed}; observation {observation.shape}")
    print(f"return {total:.1f} of 80")


if __name__ == "__main__":
    main()
