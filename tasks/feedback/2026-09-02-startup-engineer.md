# Review: autonomy startup founding engineer (Opus role-play, 2026-09-02)

## 1. API and ergonomics

Good: the two-convention rule (traced NamedTuples, static frozen dataclasses) is applied consistently; `canonical.py` as the only frame-conversion site; tasks own cost/reference/heading so episode.py never branches; ChannelMap's "-pitch" role prefix with the verify-the-sign procedure. The learning example reproduces bit-for-bit (156.84 vs 156.8 documented).

Awkward: `step` takes eight parameters, four episode-invariant (a make_env closure factory would fix it); no `observation_size(model)` and the observation layout lives only in a docstring; `reset` returns buffer[-1] while `step` returns buffer[0]; `SensorNoise` is in observation units, not sensor units; `channel_scale` is one float for all channels and independent of ChannelMap.limit, so the baseline and a learned policy can have different authority; hardcoded float32 casts read as load-bearing but are not (x64 works end to end); `cascade.env.gusts` missing from the env init while weather imports its private filters.

## 2. Sim-to-real tooling

Present: observation white noise, per-episode rate bias, integer observation delay; Dryden at the aircraft's altitude with a log profile and station records; randomisation by batching model pytrees; first-order actuator lag with rate limiting and clip; Plant as a 40 Hz HIL-shaped boundary; MJCF/OBJ export.

Missing: actuation latency (only observations are delayed; the action lands at the top of the period), timing jitter, dropped frames; the observation is privileged (alpha, beta, true airspeed, surface deflections, prop RPM) with no mask; no IMU model (specific force, accel bias, scale/misalignment, quantisation, saturation, GPS dropout, baro drift); actuators lack deadband, backlash, PWM quantisation, battery sag; no named randomisation spec; gusts are CG-uniform with no rotary components; no trajectory logging/replay, no ULog/MAVLink/rosbag ingestion, no ONNX/StableHLO/jax.export or fixed-rate onboard inference.

## 3. Unserved needs, ranked

| # | Need | Why | Done looks like | Effort |
| --- | --- | --- | --- | --- |
| 1 | Actuation latency and jitter, randomisable | Largest single sim-to-real gap; zero-delay policies oscillate on hardware | action_delay_steps in EpisodeConfig, action buffer in EnvState, fractional hold, per-episode randomisation | S (~150 LOC) |
| 2 | Configurable observation spec | Default obs is unbuildable on real vehicles | ObservationSpec selecting blocks; noise in sensor units; IMU specific-force block | M |
| 3 | Policy export and onboard reference | JAX-only artefacts are not deployable to Jetson | jax.export/StableHLO example, fixed-rate C++/ONNX harness, sim-vs-onboard numeric check | M |
| 4 | Named domain-randomisation spec | Hand-written _replace is not reviewable or CI-able | Randomisation pytree over mass/inertia/coeffs/lags/latency; sample_models(key, n) | S to M |
| 5 | Trajectory logging and flight-log replay | Every sim-real investigation starts here | npz/HDF5 writer with a versioned schema; ULog to canonical loader driving Plant | M |

## 4. Red flags

Maturity: 48 commits, one author, 21 hours; alpha; vendor rather than depend. Licensing is the good news: MIT, deps JAX + SciPy + tomli-w, MuJoCo optional; but the X8 coefficients derive from a published NTNU model and provenance is TOML comments, not a data-licence statement. Performance claims are CPU-only; no GPU CI, no perf regression gate. Determinism is PRNG-reproducible but unstated (no dtype policy). `control_from_actuators` calls pinv on every telemetry read.

## 5. The first PR

Actuation latency mirroring the observation path: action_delay_steps in EpisodeConfig, an action_buffer in EnvState shaped like observation_buffer, applied in step, plumbed through rollouts and Plant.step. Zero new dependencies, no physics change, makes latency a randomisable leaf.
