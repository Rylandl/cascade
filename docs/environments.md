# Episode environments

`cascade.env` is a native-JAX episode environment over the functional core: pure functions
rather than an object, so one definition serves reinforcement learning (vmap over thousands
of episodes), trajectory optimisation (grad through an episode), and identification (vmap over
model parameters).

```python
import jax
import jax.numpy as jnp
import cascade
from cascade.env import EpisodeConfig, action_size, reset, step, tracking_task, trimmed_reference

model = cascade.skywalker_x8()
task = tracking_task(airspeed_m_s=18.0, altitude_m=100.0, heading_rad=0.0)
reference = trimmed_reference(model, task)  # one host-side trim
config = EpisodeConfig(control_frequency_hz=40.0, horizon_steps=400, channel_scale=0.5)

keys = jax.random.split(jax.random.PRNGKey(0), 1024)
states, observations = jax.jit(jax.vmap(lambda k: reset(model, config, task, reference, k)))(keys)
env_step = jax.jit(jax.vmap(lambda s, a: step(model, config, task, reference, s, a)))
actions = jnp.zeros((1024, action_size(model)))  # throttle 0.5, neutral channels
states, observations, rewards, dones, info = env_step(states, actions)
```

## Pieces

| piece | role |
| --- | --- |
| `EpisodeConfig` | static settings: simulation and control rates, horizon, action scaling, reset spread, crash and upright limits, integrator |
| `TrackingTask` / `tracking_task` | hold an airspeed, altitude, and heading; weights on the normalised errors, body rates, and action effort |
| `HoverTask` / `hover_task` / `hover_reference` | hold a position with the belly toward an azimuth (a tailsitter's hover); the reference is the static hover from the thrust map, not a trim |
| `TransitionTask` / `transition_task` | from hover, reach and hold a cruise airspeed, altitude, and heading (belly azimuth); starts from `hover_reference` |
| Mission tasks | scheduled tracking, timed waypoints and orbit guidance; `task_at` resolves the shared target at each episode time; see [missions](missions.md) |
| `transition_policy` | the transition controller with a setpoint schedule as a policy: the baseline for a transition task |
| `ReferenceFlight` / `trimmed_reference` / `hover_reference` | the flight an episode is drawn around (a cruise trim, or a static hover), built once host side; `task.reference(model)` picks the right one |
| `reset` | Gaussian perturbation of the reference in position, velocity, body-frame attitude, and rates; actuators equilibrated to the trim control |
| `step` | holds a normalised `[-1, 1]` action for one control period of RK4 sub-steps; returns state, observation, reward, done, info |
| `rollout_actions` | scans a time-major action sequence; rewards after the first `done` are zeroed so the sum is the return |
| `rollout_policy` | scans `policy(policy_state, observation, env_state)` over the horizon; learned policies read the observation, model-based baselines may read the state |
| `cascade_policy` | the control cascade as a policy for a tracking task, every loop at the control rate: the reference score for a learner |
| `action_to_control` / `control_to_action` | throttles map `[-1, 1]` to `[0, 1]`; channels scale by `channel_scale` into spec units |

## Observation

`ObservationSpec` selects the blocks a policy sees (`EpisodeConfig.observation`); the default
is everything below except the specific-force block, a privileged view for learning research,
and `onboard_observation()` is what a small autopilot measures: rates and specific force from
an IMU, gravity direction and heading from an attitude estimate, pitot airspeed, and a GNSS
position error, with no flow angles and no actuator states. `observation_size(model, spec)`
gives the length and `observation_layout(model, spec)` the slice of each block (`None` when
absent), so a policy or an ablation never hardcodes offsets.

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
| (opt) | specific force in body axes in g: acceleration less gravity, what an accelerometer reads |

## Sensors

`reset` and `step` take an optional `SensorNoise`: white noise per observation block plus gyro
and accelerometer biases drawn once per episode. `sensor_noise_from_sensors(reference_speed,
airspeed_std_m_s=..., gyro_std_rad_s=..., accelerometer_std_m_s2=..., attitude_std_rad=...,
position_std_m=...)` builds it from datasheet units, so the conversion into observation units
is the library's; `sensor_noise(...)` takes observation units directly. `EpisodeConfig.observation_delay_steps` returns the
reading from that many control periods ago. Both are pure functions of the episode key, so a
noisy episode is still reproducible and differentiable; the true observation is always
available from `observation`.

`EpisodeConfig.sensors` optionally adds per-block sampling, acquisition and transport jitter,
delay, dropout and bias drift. Measurement age, validity and freshness are returned separately
in `info`; the observation vector size is preserved. See [sensor pipelines](sensors.md).

### Sensor-aware policies

`sensor_observation(state)` returns `SensorObservation(values, age_s, valid)` for
the current delivered observation. The fields share the selected observation
layout; ages include block and whole-observation delay. Missing readings have
zero values, infinite ages and false validity. Held readings remain valid while
their ages grow. Without a sensor pipeline, validity is true and ages still
include `observation_delay_steps`.

Write a policy as `policy(memory, reading) -> (action, memory)`, then wrap it with
`sensor_policy(policy)` for `rollout_policy` or an experiment `Policy` factory.
The adapter passes only the supplied observation, ages and validity to the inner
policy. It does not expose aircraft state, faults, wind, sensor internals, or the
episode clock, and it does not recompute truth-based measurements. Observation
values and metadata must retain the same layout and refer to the same step.

For example, enable the observed cascade's existing age and validity checks:

```python
from cascade.control import aerobatic_reference_controller
from cascade.env import observation_cascade_policy, rollout_policy, sensor_policy

observed, memory = observation_cascade_policy(
    aerobatic_reference_controller(),
    model,
    config,
    task,
    reference,
)
policy = sensor_policy(observed)
final_state, outputs = rollout_policy(
    model,
    config,
    task,
    reference,
    state,
    policy,
    memory,
)
```

An experiment factory returns `(sensor_policy(observed), memory)` in the same
way. Existing three-argument/raw-array policies and rollout outputs are unchanged.
The adapter works with `jit`, `vmap`, and differentiable policies; it leaves
action bounds and invalid/stale-reading behavior to the policy. In particular,
check validity before using infinite missing-reading ages in a learned policy.

## Latency

`EpisodeConfig.action_delay_steps` applies the action commanded that many control periods
ago, representing sense-to-actuate latency;
`action_delay_range` draws the delay per episode over an inclusive integer range, so latency
is a randomisable leaf like mass or a coefficient. The state carries an action buffer that
starts full of the reference action, `info["applied_action"]` reports what actually reached
the actuators, and the cost charges the commanded action. Evaluate the policy across the
expected delay range: randomizing delay enables an experiment, but does not itself guarantee
robustness on hardware. Historical crash-count sweeps are not release qualification results.

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

`cascade.integrations.gymnasium.CascadeEnv` is the optional public single-episode adapter.
The `gymnasium` extra supplies a real Gymnasium base class and spaces; core installation also
supports its plain reset/step fallback. `examples/gymnasium_shim.py` demonstrates the packaged
class. See [Gymnasium integration](gymnasium.md) for seeding, mission and sensor options.
Batched training loops can vmap the functional episode API directly.

## Deployment

A trained policy can be serialized through `jax.export`. `examples/export_policy.py` reads
a learning checkpoint, exports its learned parameters and sensor/memory contract, reloads
the inference bundle, and checks action and recurrent-memory agreement across seeded packets.
This is a JAX serialization round trip; it does not execute on onboard hardware or qualify
another runtime. Platform and version constraints apply as described in the
[JAX export documentation](https://docs.jax.dev/en/latest/export/export.html).
Install the `export` extra for serialization (`flatbuffers`, also in the dev group). See
[policy learning](learning.md) for the checkpoint and export commands.

## Baseline

`cascade_policy` wraps a tuned `CascadeController` as a policy, every loop at the environment's
control rate, so a learned policy has a reference score on the same task, resets, and horizon.
The sensor-aware learning workflow instead uses `observation_cascade_policy` through
`sensor_policy`, so the baseline and learned controllers use the same sensor-input contract,
including delivered values, ages and validity. It compares trim and independently trained controllers on frozen, disjoint
training/validation/evaluation episode seeds, including held-out sensor stress. Record
configuration and results rather than assuming one score applies across aircraft,
perturbations, and software versions.

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
models = models._replace(
    mass=models.mass * jax.random.uniform(key, (1024,), minval=0.8, maxval=1.2)
)
states, obs = jax.vmap(lambda m, k: reset(m, config, task, reference, k))(models, keys)
```

The tailsitter's transition task exercises hover-to-cruise behavior with the same episode
interface. Compare tracking errors, termination, and rewards on a declared evaluation set;
a better scalar reward alone does not establish flight transfer.

## Learning by gradient through the dynamics

Because an episode is differentiable end to end, a policy can be trained by ascending the
return with the gradient taken straight through `rollout_policy`, no critic or replay buffer.
`cascade.learning` provides feedforward and recurrent reference policies initialized at the
trim action, Adam ascent, clipping and finite-update checks. The default workflow uses the
aerobatic reference's 12 m/s tracking task from perturbed starts (4 s at 40 Hz, batch size 16),
three independent training seeds and onboard observation blocks. Run
`uv run --frozen python -m cascade.learning dist/learning --steps 60`
to retain actual trained checkpoints, learning curves, schemas, configuration hashes,
versions and matched validation/evaluation reports. Resume and export use the same weights;
see the [learning guide](learning.md) for commands, public APIs and interpretation limits.

## Throughput

`uv run --frozen python scripts/benchmark_env.py --output dist/throughput.json` records a table
for both backends and batches from 1 to 16384, with model hashes, seeds, versions, and
configuration. One control step is ten RK4 substeps at 400 Hz. Compilation is excluded by a
warm-up call; device work is synchronized before and after timing. The reported throughput
is execution throughput, not end-to-end training or real-time latency. Run on an otherwise
idle host and record load conditions alongside the JSON. The release record identifies the
actual measurement platform; GPU performance is not qualified by a CPU run.
