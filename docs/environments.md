# Episode environments

`cascade.env` is a native-JAX episode environment over the functional core: pure functions
rather than an object, so one definition serves reinforcement learning (vmap over thousands
of episodes), trajectory optimisation (grad through an episode), and identification (vmap over
model parameters).

```python
import jax
import cascade
from cascade.env import EpisodeConfig, reset, rollout_actions, step, tracking_task, trimmed_reference

model = cascade.skywalker_x8()
task = tracking_task(airspeed_m_s=18.0, altitude_m=100.0, heading_rad=0.0)
reference = trimmed_reference(model, task)          # one host-side trim
config = EpisodeConfig(control_frequency_hz=40.0, horizon_steps=400, channel_scale=0.5)

keys = jax.random.split(jax.random.PRNGKey(0), 1024)
states, observations = jax.jit(jax.vmap(lambda k: reset(model, config, task, reference, k)))(keys)
env_step = jax.jit(jax.vmap(lambda s, a: step(model, config, task, reference, s, a)))
states, observations, rewards, dones, info = env_step(states, actions)   # actions: (1024, P + C)
```

## Pieces

| piece | role |
| --- | --- |
| `EpisodeConfig` | static settings: simulation and control rates, horizon, action scaling, reset spread, crash and upright limits, integrator |
| `TrackingTask` / `tracking_task` | hold an airspeed, altitude, and heading; weights on the normalised errors, body rates, and action effort |
| `HoverTask` / `hover_task` / `hover_reference` | hold a position with the belly toward an azimuth (a tailsitter's hover); the reference is the static hover from the thrust map, not a trim |
| `trimmed_reference` | trims the model in the task's flight once (host side); the episode is drawn around it |
| `reset` | Gaussian perturbation of the reference in position, velocity, body-frame attitude, and rates; actuators equilibrated to the trim control |
| `step` | holds a normalised `[-1, 1]` action for one control period of RK4 sub-steps; returns state, observation, reward, done, info |
| `rollout_actions` | scans a time-major action sequence; rewards after the first `done` are zeroed so the sum is the return |
| `rollout_policy` | scans `policy(policy_state, observation, env_state)` over the horizon; learned policies read the observation, model-based baselines may read the state |
| `cascade_policy` | the control cascade as a policy for a tracking task, every loop at the control rate: the reference score for a learner |
| `action_to_control` / `control_to_action` | throttles map `[-1, 1]` to `[0, 1]`; channels scale by `channel_scale` into spec units |

## Observation

Body-frame, so a policy never sees world position except through the altitude error:

| slice | content |
| --- | --- |
| 0:3 | air velocity in body FRD over the reference airspeed |
| 3:6 | airspeed over the reference, alpha, beta |
| 6:9 | body rates |
| 9:12 | gravity direction in body axes |
| 12:14 | heading error as sin and cos |
| 14:17 | position error in body axes over 10 m (vertical only for tracking tasks) |
| 17:17+S | surface deflections (rad) |
| 17+S: | propeller speeds as a fraction of maximum |

## Reward and termination

The reward is `exp(-cost)` in `(0, 1]`, with cost the task's weighted sum: for tracking, squared
normalised airspeed and altitude errors and `1 - cos` of the heading error; for hover, squared
position error in metres, squared velocity, and `1 - cos` of the belly-azimuth error; both add
squared body rates and mean squared action. Set `upright_limit_rad` above pi for a hover task,
whose nose-up attitude is 90° from the tracking task's upright. It is zero on the step an episode crashes (below `crash_altitude_m`, or the
body down axis more than `upright_limit_rad` from gravity), so an undiscounted return counts
good steps and survival alone earns nothing. `done` is crash or horizon; `info` separates
`crashed` and `truncated` and reports the cost.

## Gymnasium

There is no Gymnasium dependency. A single-episode `gymnasium.Env` is a few lines over these
functions: keep an `EnvState`, call `reset` with a fresh key in `reset(seed=...)`, call `step`
in `step(action)`, and convert with `np.asarray`. Batched training loops should stay in JAX and
vmap the functions directly; that is where the speed is.

## Baseline

`cascade_policy` wraps a tuned `CascadeController` as a policy, every loop at the environment's
control rate, so a learned policy has a reference score on the same task, resets, and horizon.
On the aerobatic reference's 12 m/s tracking task from perturbed starts (2 m, 1 m/s, 0.1 rad,
0.2 rad/s) at 40 Hz it crashes no episodes and earns a mean reward above 0.6 over 60 steps and
above 0.8 once settled.
