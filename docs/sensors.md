# Sensor timing and availability

Cascade can sample observation blocks at different control ticks, hold the most recent
reading, evolve a bias random walk, drop acquisitions, and deliver packets with variable
latency. These are optional research models for the observation stream. Aircraft dynamics
continue to run at the configured simulation frequency.

The original `SensorNoise` API still provides white measurement noise and a bias drawn once
per episode. `EpisodeConfig(sensors=None)` preserves that behavior. Enabling a default
`SensorPipelineConfig()` produces the same values and uses the same existing noise,
weather, and reset random streams.

```python
from cascade.env import EpisodeConfig
from cascade.env.sensors import (
    SensorBlockConfig,
    SensorPipelineConfig,
    onboard_observation,
    sensor_noise_from_sensors,
)

config = EpisodeConfig(
    control_frequency_hz=40.0,
    observation=onboard_observation(),
    sensors=SensorPipelineConfig(
        rates=SensorBlockConfig(bias_walk_std=0.001),
        specific_force=SensorBlockConfig(bias_walk_std=0.002),
        airspeed=SensorBlockConfig(sample_period_steps=2),
        position_error=SensorBlockConfig(
            sample_period_steps=8,
            sample_jitter_steps=1,
            delay_steps=2,
            delay_jitter_steps=1,
            dropout_probability=0.1,
        ),
    ),
)
noise = sensor_noise_from_sensors(
    12.0,
    gyro_std_rad_s=0.01,
    gyro_bias_std_rad_s=0.02,
    accelerometer_std_m_s2=0.05,
    position_std_m=0.5,
)
# Pass noise to reset(...) and each step(...), as before.
```

In this example the IMU blocks sample at 40 Hz and airspeed at 20 Hz. Position-error samples
arrive from acquisitions every 7–9 control periods, with a transport delay of 1–3 periods.
The nominal position sampling frequency is 5 Hz. A dropout discards the whole position block
with probability 0.1 per acquisition attempt. Timing and dropout decisions are shared by all
elements of a block; independent blocks draw independent decisions.

`SensorPipelineConfig` has the same block names as `ObservationSpec`: `air_velocity`,
`airspeed`, `air_angles`, `rates`, `gravity`, `heading`, `position_error`, `surfaces`,
`propellers`, and `specific_force`. Unselected blocks consume no queue storage or random
decisions. Configuration objects are frozen dataclasses and support `dataclasses.asdict`.

| Block setting | Meaning |
| --- | --- |
| `sample_period_steps` | Nominal interval between acquisition attempts, at least 1 |
| `sample_jitter_steps` | Inclusive uniform variation around the interval; less than the nominal interval |
| `delay_steps` | Nominal transport delay, at least 0 |
| `delay_jitter_steps` | Inclusive uniform variation around the delay; no greater than the nominal delay |
| `dropout_probability` | Probability in `[0, 1]` of discarding an attempted sample |
| `bias_walk_std` | Independent per-element bias diffusion in observation units per square-root second |

For an interval of `dt` seconds, the drift increment has zero mean and standard deviation
`bias_walk_std * sqrt(dt)`. Drift evolves on every control tick, including unsampled or
dropped intervals. It adds to the fixed episode bias. For body rates these are rad/s per
square-root second; for specific force they are g per square-root second; for normalized
position error they are ten-metre units per square-root second. White noise is frozen into
a sample at acquisition, so held or delayed readings do not receive fresh noise each tick.

All sampling and delivery occurs on control ticks. Rates above the control frequency and
sub-tick latency are not represented. This model does not claim to reproduce an IMU filter,
GNSS estimator, or correlated hardware failure process.

## Reading ages and validity

Reset attempts a sample for every selected block at time zero. A zero-delay successful sample
is immediately available. Otherwise its returned value is zero until the first delivery.
Inspect `state.sensor_valid` instead of treating that placeholder as a real measurement.

`state.sensor_age_s` and `state.sensor_valid` are arrays with the same shape and ordering as
the returned observation. `step` also exposes them as `info["sensor_age_s"]` and
`info["sensor_valid"]`. Age is the current episode time minus the delivered sample's
acquisition time. Missing readings have `age=inf` and `valid=False`. Once a sample has arrived,
a held value remains valid and its age grows; applications can impose their own freshness
threshold. Slices from `observation_layout(model, config.observation)` give block metadata.

The existing `observation_delay_steps` delays the entire resulting observation after block
sampling and transport. Metadata passes through the same buffer, so reported ages include
both delays. Old packets arriving after a newer sample are discarded and cannot make the
measurement timestamp move backward.

`info["sensor_sampled"]` marks elements whose block attempted an acquisition on the current
tick. `info["sensor_dropped"]` marks attempts discarded by dropout. These acquisition events
are reported immediately, before the whole-observation delay; they do not imply a new value
has reached the policy. `state.sensor_state.updated` identifies deliveries accepted by the
pipeline on this tick, also before that final delay.
`info["sensor_fresh"]` identifies newly delivered values in the returned observation after
all delays, even when their age is nonzero. Rewards use the task and environment at the
returned episode time, independently of the age of the policy's observations.

## Functional use and reproducibility

`initialize_sensor_pipeline(model, config, observation_spec, reading, key)` and
`step_sensor_pipeline(model, config, observation_spec, state, reading, key, dt_s)` expose
the same pipeline independently of an episode. Inputs are one observation vector per
world; use `jax.vmap` across worlds and `jax.lax.scan` across time. Configuration and the
observation layout are static; the queue, clocks, drift, and readings are JAX arrays.

Given identical keys, configuration, and input readings, eager and compiled runs produce
the same sampling/dropout decisions. Separate keys produce independent streams. Do not
change configuration or observation layout between reset and step: queue shapes and stored
measurements belong to that configuration. The episode state now also carries `time_s` for
time-varying tasks; policies can choose whether to consume measurement-age metadata.

Run `uv run python examples/sensor_timing.py` to see a seeded episode's airspeed and position
measurement ages, including missing readings and held samples.
