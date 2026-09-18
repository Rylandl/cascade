"""NumPy/Gymnasium boundary around Cascade's functional JAX environment.

Install ``cascade-flight[gymnasium]`` for real Gymnasium spaces and wrappers. Without the
extra, ``CascadeEnv`` retains its reset/step interface as a plain Python class. Vectorized
training should use the functional environment with JAX's ``vmap`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import jax
import numpy as np

from cascade.env.episode import EpisodeConfig, action_size, observation_size, reset, step

try:
    import gymnasium
except ImportError:  # The core distribution deliberately does not require Gymnasium.
    gymnasium = None

Base = object if gymnasium is None else gymnasium.Env


class CascadeEnv(Base):
    """One fixed aircraft/task/configuration with NumPy float32 observations and actions.

    ``task`` may be static or a mission. ``noise``/``weather`` apply at reset and step;
    ``faults`` applies at step. Sensor timing is set by ``config.sensors``. Reset ``options``
    may override noise, weather, and faults for that episode; omitting an option restores
    its constructor default on the next reset. Model/config/reference stay fixed so spaces
    cannot change during an episode. Rendering is not supplied by this adapter.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model,
        task,
        config: EpisodeConfig | None = None,
        reference=None,
        *,
        noise=None,
        weather=None,
        faults=None,
        render_mode=None,
    ):
        if render_mode is not None:
            raise ValueError("CascadeEnv has no render modes; use cascade.viz separately")
        self.model = model
        self.task = task
        self.config = EpisodeConfig() if config is None else config
        self.reference = task.reference(model) if reference is None else reference
        self.render_mode = None
        self._defaults = {"noise": noise, "weather": weather, "faults": faults}
        self._episode_options = self._defaults.copy()
        self._reset = jax.jit(
            lambda key, noise, weather: reset(
                model, self.config, task, self.reference, key, noise=noise, weather=weather
            )
        )
        self._step = jax.jit(
            lambda state, action, noise, weather, faults: step(
                model,
                self.config,
                task,
                self.reference,
                state,
                action,
                noise=noise,
                weather=weather,
                faults=faults,
            )
        )
        self._state = None
        self._needs_reset = True
        self._rng = np.random.default_rng()
        if gymnasium is not None:
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (observation_size(model, self.config.observation),), np.float32
            )
            self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (action_size(model),), np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("reset options must be a mapping")
        options = {} if options is None else dict(options)
        unknown = options.keys() - self._defaults.keys()
        if unknown:
            raise ValueError(f"unknown reset options: {sorted(unknown)}")
        if gymnasium is not None:
            super().reset(seed=None if seed is None else int(seed))
            rng = self.np_random
        else:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            rng = self._rng
        episode_options = self._defaults | options
        key = jax.random.PRNGKey(rng.integers(0, 2**32, dtype=np.uint32))
        state, observation = self._reset(key, episode_options["noise"], episode_options["weather"])
        self._state = state
        self._episode_options = episode_options
        self._needs_reset = False
        return np.asarray(observation, dtype=np.float32), {"time_s": 0.0}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("call reset() before stepping a new or completed episode")
        with np.errstate(over="ignore", invalid="ignore"):
            action = np.asarray(action, dtype=np.float32)
        if action.shape != (action_size(self.model),) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must contain {action_size(self.model)} finite values")
        if np.any(np.abs(action) > 1.0):
            raise ValueError("action values must lie in [-1, 1]")
        options = self._episode_options
        self._state, observation, reward, _, info = self._step(
            self._state, action, options["noise"], options["weather"], options["faults"]
        )
        terminated, truncated = bool(info["crashed"]), bool(info["truncated"])
        self._needs_reset = terminated or truncated
        result_info = {
            name: np.asarray(value).item() if np.asarray(value).shape == () else np.asarray(value)
            for name, value in info.items()
        }
        result_info["time_s"] = float(self._state.time_s)
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            result_info,
        )

    def close(self):
        """Release the stored episode; the adapter owns no rendering resources."""

        self._state = None
        self._needs_reset = True


__all__ = ["CascadeEnv"]
