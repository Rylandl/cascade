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
| `TransitionTask` / `transition_task` | from hover, reach and hold a cruise airspeed, altitude, and heading (belly azimuth); starts from `hover_reference` |
| `transition_policy` | the transition controller with a setpoint schedule as a policy: the baseline for a transition task |
| `ReferenceFlight` / `trimmed_reference` / `hover_reference` | the flight an episode is drawn around (a cruise trim, or a static hover), built once host side; `task.reference(model)` picks the right one |
| `reset` | Gaussian perturbation of the reference in position, velocity, body-frame attitude, and rates; actuators equilibrated to the trim control |
| `step` | holds a normalised `[-1, 1]` action for one control period of RK4 sub-steps; returns state, observation, reward, done, info |
| `rollout_actions` | scans a time-major action sequence; rewards after the first `done` are zeroed so the sum is the return |
| `rollout_policy` | scans `policy(policy_state, observation, env_state)` over the horizon; learned policies read the observation, model-based baselines may read the state |
| `cascade_policy` | the control cascade as a policy for a tracking task, every loop at the control rate: the reference score for a learner |
| `action_to_control` / `control_to_action` | throttles map `[-1, 1]` to `[0, 1]`; channels scale by `channel_scale` into spec units |

## Observation

`observation_size(model)` gives the length and `observation_layout(model)` the slice of each
block, so a policy or an ablation never hardcodes offsets.

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

## Sensors

`reset` and `step` take an optional `SensorNoise` (`sensor_noise(...)`): white noise per
observation block (air data, angles, rates, gravity direction, heading, position, actuators)
plus a rate bias drawn once per episode. `EpisodeConfig.observation_delay_steps` returns the
reading from that many control periods ago. Both are pure functions of the episode key, so a
noisy episode is still reproducible and differentiable; the true observation is always
available from `observation`.

## Latency

`EpisodeConfig.action_delay_steps` applies the action commanded that many control periods
ago, the sense-to-actuate latency a real stack has (one to three periods at 40 Hz is common);
`action_delay_range` draws the delay per episode over an inclusive integer range, so latency
is a randomisable leaf like mass or a coefficient. The state carries an action buffer that
starts full of the reference action, `info["applied_action"]` reports what actually reached
the actuators, and the cost charges the commanded action. A policy trained at zero delay
oscillates on hardware; one trained across a range of delays does not, and this is where to
show it. The hand-tuned aerobatic cascade illustrates the sensitivity on the 12 m/s tracking
task from perturbed starts (8 episodes, 4 s): at 40 Hz it flies every episode with 25 ms of
latency, crashes 2 with 50 ms, 3 with 75 ms, and 7 with 100 ms; at 100 Hz it flies every
episode up to the 40 ms tested.

## Failures

`fault_schedule(model, jams={surface: t}, hardovers={surface: (t, sign)}, motor_out={propeller:
t}, partial_power={propeller: (t, fraction)})` builds a `FaultSchedule`; `step`, `rollout_actions`,
and `rollout_policy` take it as `faults` and apply whatever has failed by each period's time to
the actuators: a jam freezes a surface where it is, a hardover drives it to a limit at its own
rate and holds it, a motor-out spins a propeller down, partial power derates it. The policy is
not told. A batch of schedules is a batch of failure cases, and `apply_faults(model, schedule,
time)` is the pure function underneath for use outside the environment.

## Weather

`reset` and `step` take an optional `WeatherCondition` (`cascade.env.weather`): a mean wind with a
logarithmic profile over the site's roughness and Dryden turbulence advanced every period at
the aircraft's altitude, from a MIL-F-8785C class or a draw from station records. See
`docs/weather.md`.

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

## Domain randomisation

`randomisation(mass=(0.8, 1.2), inertia=(0.7, 1.4), lift_curve_slope=(0.9, 1.1),
surface_time_constant=(0.5, 2.0), thrust=(0.85, 1.15), center_of_mass_shift_m=(-0.02, 0.02))`
is a reviewable spec of multiplicative ranges over named model leaves plus a centre-of-mass
shift, and `sample_models(model, spec, key, n)` draws one factor per world per entry into a
batched model. Any other leaf can be named directly (`"surfaces.stall_angle"`). The mechanism
underneath is the one below, so hand-written updates still work:

`reset` and `step` take the model as an argument, so a batch of models is a batch of worlds:
`broadcast_model` the validated model to a leading batch shape, perturb leaves with indexed
updates (mass, inertia, a coefficient, an actuator lag), and vmap the episode functions over
models and keys together. The trimmed reference stays the nominal one, so each episode also
starts with the mismatch a real vehicle has from its nominal model.

```python
models = broadcast_model(model, (1024,))
models = models._replace(mass=models.mass * jax.random.uniform(key, (1024,), minval=0.8, maxval=1.2))
states, obs = jax.vmap(lambda m, k: reset(m, config, task, reference, k))(models, keys)
```

On the tailsitter's transition task (hover to 8 m/s at 1.5 m, 100 Hz control, 8 s horizon),
`transition_policy` with a 3.5 m/s² velocity ramp reaches cruise within 1.5 m/s, earns under 0.6
mean reward in the first second of hover and above 0.7 in the last second of cruise. A learned
policy that beats that curve has learned the transition.

## Learning by gradient through the dynamics

Because an episode is differentiable end to end, a policy can be trained by ascending the
return with the gradient taken straight through `rollout_policy`, no critic or replay buffer.
`examples/learn_tracking_policy.py` does that with a 32-unit tanh network initialised at the
trim action, Adam, and gradient clipping, on the aerobatic reference's 12 m/s tracking task
from perturbed starts (4 s horizon at 40 Hz, batches of 16 episodes):

| policy | mean return over 256 evaluation episodes (max 160) |
| --- | ---: |
| hold the trim action | 144.5 |
| control cascade baseline (`cascade_policy`) | 156.5 |
| learned, after 60 gradient steps (36 s after an 18 s compile) | 156.8 |

Sixty steps through the physics match a hand-tuned three-loop cascade. The same loop runs
unchanged on a batch of randomised models, which is how a robust policy is trained here.

## Throughput

`scripts/benchmark_env.py` emits this table for the current backend (run it on a GPU host for a
GPU row). One control step is ten RK4 sub-steps of the full model (six-surface panels, actuator lags,
stall dynamics, propeller inflow) at 400 Hz. Measured on an Apple M3 CPU while the machine was
also running other work, so these are conservative:

| aircraft | batch | ms per control step | env steps / s | RK4 steps / s |
| --- | ---: | ---: | ---: | ---: |
| aerobatic reference | 1 | 0.08 | 12 000 | 120 000 |
| aerobatic reference | 1024 | 13.7 | 75 000 | 750 000 |
| aerobatic reference | 4096 | 43.8 | 94 000 | 940 000 |
| Skywalker X8 (coefficient backend) | 1 | 0.07 | 15 000 | 150 000 |
| Skywalker X8 (coefficient backend) | 1024 | 7.9 | 130 000 | 1 300 000 |

A 4 s episode at 40 Hz is 160 control steps, so batch 1024 runs about 500 episodes per second
on this CPU; a GPU vmap is the same code.
