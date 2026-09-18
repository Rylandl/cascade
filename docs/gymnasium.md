# Gymnasium integration

Install the optional integration with `pip install 'cascade-flight[gymnasium]'` and import
`CascadeEnv` from `cascade.integrations.gymnasium`. The numerical core does not import or
require Gymnasium. Without that extra the same class remains usable as a plain reset/step
adapter, but Gymnasium spaces and wrapper compatibility require the extra.

```python
import numpy as np
from cascade import aerobatic_reference
from cascade.env import EpisodeConfig, scheduled_tracking_task
from cascade.integrations.gymnasium import CascadeEnv

env = CascadeEnv(
    aerobatic_reference(),
    scheduled_tracking_task([0, 2, 4], [12, 12, 14], [50, 50, 55]),
    EpisodeConfig(horizon_steps=160),
)
observation, info = env.reset(seed=3)
for _ in range(160):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

The class subclasses `gymnasium.Env` when Gymnasium is installed. Observations and actions
at this boundary use NumPy float32, including when the underlying JAX simulation uses x64.
Observation dimensions follow `EpisodeConfig.observation`; actions are finite values in
[-1, 1]. Invalid action dimensions/ranges raise `ValueError`. A step before reset or after
termination/truncation raises `RuntimeError`; reset starts the next episode.

`reset(seed=n)` follows Gymnasium's RNG contract and repeats the same sequence. Subsequent
unseeded resets advance that sequence. Seeding does not promise bitwise equality across JAX
versions or devices. The adapter distinguishes crash/attitude termination from the horizon's
truncation and converts all functional `info` values to NumPy arrays or Python scalars. It
also returns `time_s`. Rendering is not implemented; use `cascade.viz` separately.

Static tasks and missions use `task.reference(model)` by default, or accept an explicit
`reference=`. Constructor keywords `noise=`, `weather=` and `faults=` pass through to the
functional episode. Sensor timing, dropout and drift are configured through
`EpisodeConfig.sensors`. For per-episode changes, use
`reset(options={"weather": condition, "faults": schedule})`. Only those three option names
are accepted; explicit `None` disables an option. Each reset starts from constructor
defaults before applying its overrides. Changing an option's structure may cause a JAX
recompilation. Model, task, configuration and reference stay fixed for the adapter's lifetime.

The adapter is for one unbatched aircraft. Use JAX `vmap` over the functional reset/step API
for accelerator-native batched learning. `examples/gymnasium_shim.py` is now a small client
of this packaged integration.
