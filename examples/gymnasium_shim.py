"""Use the packaged optional Gymnasium adapter; NumPy in/out, JAX physics underneath."""

import numpy as np

from cascade.env import EpisodeConfig, action_size, tracking_task
from cascade.integrations.gymnasium import CascadeEnv, gymnasium
from cascade.reference import aerobatic_reference


def main() -> None:
    model = aerobatic_reference()
    env = CascadeEnv(model, tracking_task(12.0, 50.0), EpisodeConfig(horizon_steps=80))
    observation, _ = env.reset(seed=3)
    total = 0.0
    for _ in range(80):
        action = np.zeros(action_size(model), dtype=np.float32)
        action[0] = 0.2
        observation, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    print(f"gymnasium installed: {gymnasium is not None}; observation {observation.shape}")
    print(f"return {total:.1f} of 80")
    env.close()


if __name__ == "__main__":
    main()
