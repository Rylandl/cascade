# Observation-only fixed-wing baseline

`cascade.env.observation_cascade_policy` runs the existing fixed-wing control
cascade from the delivered observation and its own memory. Its optional third
policy argument is ignored: changing the aircraft state, wind, episode clock,
faults, or sensor internals cannot change its action when observation and memory
are unchanged. The existing `cascade_policy` remains a privileged baseline that
reads the true aircraft state.

```python
import jax
from cascade.control import aerobatic_reference_controller
from cascade.env import (
    EpisodeConfig,
    observation_cascade_policy,
    onboard_observation,
    reset,
    rollout_policy,
    tracking_task,
)
from cascade.reference import aerobatic_reference

model = aerobatic_reference()
task = tracking_task(12.0, 50.0)
reference = task.reference(model)
config = EpisodeConfig(observation=onboard_observation(), horizon_steps=400)
state, obs = reset(model, config, task, reference, jax.random.PRNGKey(0))
policy, memory = observation_cascade_policy(
    aerobatic_reference_controller(),
    model,
    config,
    task,
    reference,
)
final_state, (observations, actions, rewards, dones) = rollout_policy(
    model,
    config,
    task,
    reference,
    state,
    policy,
    memory,
)
```

The factory accepts a `CascadeController` or `GainSchedule` and returns the policy
plus `ObservationCascadeState(cascade, step, action, measurement_valid)`. Call the
policy once per control period and create fresh memory at every episode reset.
Both ordinary arrays and the optional metadata wrapper below work with `jit` and
`vmap`. Every cascade loop runs once per policy call. Gain schedules use measured
airspeed, and normalized actions are clipped to `[-1, 1]` after channel scaling.

## What the measurements mean

The required blocks are `airspeed`, `rates`, `gravity`, `heading`, and
`position_error`. Both the default observation and `onboard_observation()` contain
them. Unused blocks, including accelerometer readings, do not affect the policy.

| Observation | Use in the controller |
| --- | --- |
| Airspeed / commanded speed | Recover measured airspeed in m/s using the known mission schedule |
| Body rates in rad/s | Rate-loop feedback |
| Gravity direction in body FRD | Reconstruct roll and pitch |
| Sine/cosine of measured minus desired heading | Heading guidance with desired minus measured error |
| Body position error / 10 m | Project onto gravity and multiply by 10 to recover desired minus measured altitude |

The cascade receives a synthetic state with current yaw and altitude zero and
relative heading/altitude targets. It only uses velocity magnitude, so this
reconstruction needs no wind estimate or angle-of-attack estimate. With ideal,
simultaneous observations it reproduces the privileged cascade's bounded action.
It requires positive world-NED gravity and pitch away from the vertical
singularity; it is a fixed-wing tracking baseline, not a hover controller.

Tracking, scheduled tracking, waypoint, and orbit tasks are supported. Commanded
speed comes from the known mission at the policy's own clock; heading and altitude
guidance come from the delivered error measurements. The environment computes
these task-relative errors before sensor noise and transport, including the
lookahead/radial guidance of waypoint/orbit tasks. They are already processed
navigation inputs, not raw GNSS or magnetometer measurements. This baseline does
not implement a navigation estimator or redo mission guidance from raw sensors.

## Delayed, missing, and stale readings

Ordinary array input consumes finite held/delayed sensor readings as delivered.
Sampling, noise, bias, delay, and dropout therefore affect its actions. A raw
vector contains neither validity nor acquisition time: missing zero body-rate or
position readings are indistinguishable from legitimate zeros, and an unchanged
reading cannot establish staleness. For a time-varying speed target, raw delayed
airspeed is decoded using the current target, an explicit timing approximation.

To use delivery metadata, supply it explicitly as part of the observation:

```python
from cascade.env import SensorObservation, step

# At reset the delivered reading's metadata is on the returned state.
reading = SensorObservation(obs, state.sensor_age_s, state.sensor_valid)
action, memory = policy(memory, reading)  # no environment state is passed
state, obs, reward, done, info = step(model, config, task, reference, state, action)
reading = SensorObservation(obs, info["sensor_age_s"], info["sensor_valid"])
action, memory = policy(memory, reading)
```

`SensorObservation(values, age_s, valid)` has three arrays of the observation's
shape. Ages are floating-point seconds and validity is boolean. The wrapper
rejects required readings whose age is nonfinite, negative, or greater than
`max_sensor_age_s` (factory default: 0.5 seconds), or whose validity is false.
For a valid delayed pitot reading it also decodes normalization using the commanded
speed at acquisition time. To carry this metadata through ordinary `rollout_policy`
and experiment factories, wrap the controller with the public adapter:

```python
from cascade.env import sensor_policy

policy, memory = observation_cascade_policy(controller, model, config, task, reference)
policy = sensor_policy(policy)
# Pass policy and memory to rollout_policy, or return them from an experiment factory.
```

The adapter passes only the delivered values, acquisition ages and validity to the controller.
See [sensor-aware learning](learning.md) for trained policies using the same input contract.

With either input form, a nonfinite required reading, nonpositive airspeed,
degenerate gravity/heading vector, or near-vertical pitch causes the policy to
hold its previous bounded action and freeze the cascade integrators. The first
held action is the clipped reference control. `measurement_valid` reports whether
the latest update succeeded; the independent clock still advances. The same
fallback applies if reconstructed speed/altitude or controller outputs/state
become nonfinite, including overflow concealed by output clipping. Valid measurements
resume control on the next call. Asynchronous blocks are not extrapolated or
fused, and a prolonged held action is not a flight-safety guarantee.

Tests cover ideal-action agreement across supported missions and observation
layouts, hidden-state independence, JIT/vmap, sensor-pipeline influence, delayed
speed normalization, invalid/stale input handling, and action bounds. These are
software and controller-contract checks; performance under sustained sensor,
weather, and fault stress requires the separate benchmark campaign.
