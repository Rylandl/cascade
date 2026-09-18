# Supported interfaces and compatibility

Cascade 0.4.0rc1 adds research workflows to the 0.3.0rc1 foundation. The earlier candidate's
qualification record is in [release-readiness.md](release-readiness.md); it applies to those
archived artifacts. The development snapshot has its own
[feature record](feature-readiness.md); current candidate evidence and remaining gates are in
[rc-readiness.md](rc-readiness.md). A CI matrix is a plan
until it has run.

## Runtime support

Python 3.11–3.13 on macOS and Linux CPUs is the release target. The dependency range is JAX
0.7.x–0.11.x, NumPy 1.26–2.x, SciPy 1.12–1.x, and tomli-w 1.2–1.x. The resolver must also satisfy
JAX's own Python, jaxlib, NumPy, and SciPy requirements; these ranges are not a claim that every
Cartesian combination is valid. `uv.lock` specifies the reproducible development environment.
The minimum lane uses Python 3.11, JAX/jaxlib 0.7.0, NumPy 1.26.0, SciPy 1.12.0, tomli-w 1.2.0
and Gymnasium 1.0.0. Gymnasium is optional and supported from 1.0 through the 1.x series.

JAX GPU/TPU execution is experimental for this candidate: the functional interfaces compose
with JAX transformations, but this release does not supply GPU performance or hardware
qualification evidence. Windows and Intel macOS are not qualified by the local release run;
consult the executed platform table before adopting a platform. Optional rendering requires
MuJoCo and an OpenGL context; MP4 output also requires ffmpeg. Policy serialization requires
the `export` extra. Neither optional dependency is required to import or use the core.

## Public Python interface

The documented names in `cascade.__all__` and the `__all__` lists of `cascade.analysis`,
`cascade.control`, `cascade.env`, `cascade.design`, `cascade.viz`, `cascade.experiments`,
`cascade.learning`, and `cascade.calibration`
are public, as is `cascade.integrations.gymnasium.CascadeEnv`. Import optional integrations
from their explicit module; importing `cascade.integrations` does not import Gymnasium. Underscored
helpers and unexported implementation details are internal. Specifications and configuration
objects are host-side construction boundaries; validate them before compiling numerical code.
States, models, controls, and numerical results are JAX PyTrees. Read NamedTuple fields by
name: serialized positional tuples and raw PyTree layouts are not a stable interchange format.

The core dynamics, canonical conversion, specifications, trim/linearization, episode stepping,
and documented controllers are supported research interfaces. Archetype generation and
automatic tuning, learned-policy examples, and exported JAX
artifacts remain experimental. Generated airframes are plausible fixtures, not calibrated
models of a population of real aircraft. The new [learning interfaces](learning.md) are
experimental research APIs with versioned checkpoint/inference schemas. Their reference
networks use float32. Resume reproducibility requires the same software/backend/precision
configuration; JAX export compatibility also depends on platform and JAX version.

The [calibration interfaces](calibration.md) are experimental research APIs with a versioned
artifact schema. Bounded fitting uses the configured JAX precision and does not imply unique
physical parameters or measured-flight accuracy. Calibrated TOML files use the existing aircraft
specification schema and can be loaded independently of the fitting report.

The v0.2 compatibility modules (`cascade.vtol`, `autotune`, `weather`, `gusts`, `family`,
`archetypes`, `geometry`, `render`) and `cascade.env.Reference` remain available in this
candidate. New code should use the layered packages and `ReferenceFlight`. Breaking public
changes require a changelog entry and migration guidance; 0.x releases may still make such
changes. A future 1.0 release will explicitly commit to a longer compatibility policy.

## Units and frames

Physics uses SI units, right-handed NED world axes and FRD body axes. Position and velocity
in `RigidBodyState` are world-frame; angular velocity is body-frame rad/s. Attitude is the
scalar-last xyzw quaternion rotating body vectors into world coordinates. The canonical boundary
uses NWU world and FLU body axes and scalar-first wxyz quaternions, with layout `[position(3), velocity(3), quaternion(4),
angular_velocity(3)]`. Use the conversion functions instead of manual sign/slice operations.

Core `ControlInput.propeller` is normalized throttle in [0, 1]; channels use the units of the
aircraft specification's control map. Episode actions instead use [-1, 1], with throttle
remapped and channels multiplied by `EpisodeConfig.channel_scale`. Propeller speed bounds
are command-target bounds; actual nonnegative RPM approaches the target through the motor
lag/rate limit, including after a failure or derating. Integration timesteps must resolve the
fastest dynamics; clipping is not a substitute for a stable timestep.

## Shapes, observations, and precision

A single state's vector leaves end in the physical dimension: position/velocity/rates in 3,
attitude in 4, surface states in S, propeller states in P, channel commands in C. Leading
dimensions are batch dimensions. Core rollouts consume time-major controls
`(T, *batch, P/C)` and return post-step states `(T, *batch, ...)`; the initial state is not in
the returned trajectory. `vmap` is the supported way to batch episode reset/step and families.
Different aircraft topologies cannot share one dense compiled batch.

An `ObservationSpec` changes the vector length and selected blocks. Always obtain sizes and
named slices with `observation_size(model, spec)` and `observation_layout(model, spec)`;
persist the spec and action scaling with a learned policy. The full default observation
contains privileged simulator state. The onboard preset is an observation selection, not
an independently validated estimator or physical sensor simulation.

JAX's default is float32. Set `JAX_ENABLE_X64=1` before model/array creation for float64.
Constructing models once in float32 and later enabling x64 cannot recover lost precision.
Representative core checks run in both precisions. Numerical gradients are verified in smooth
regimes; clipping, hard termination, faults, and discrete choices are not globally smooth.
Seeds make stochastic draws repeatable within a fixed software/backend configuration;
bitwise equality across devices or JAX releases is not promised.

## Files and provenance

Aircraft TOML uses schema 2. `BodySpec.reference_position_m` is an optional zero-default
coefficient reference offset; see [aircraft-spec.md](aircraft-spec.md). Old schema-2 fixtures
continue to load. Export specifications or versioned trajectories rather than pickled models.

Trajectory schema `cascade_trajectory_v1` stores canonical states and native actuator and
separation arrays. `dt` is positive; `t0_s` defaults to zero for compatibility. For the
post-step states returned by `rollout`, pass `t0_s=dt` (or absolute initial time plus dt).
`controls[i]` is the command for the interval ending at state/time i. Loading validates the
state schema, uniform time axis, aligned array dimensions, and unit quaternions (absolute
norm tolerance 1e-4). Values that overflow the active JAX dtype are rejected. Older v1 files lacking `t0_s`
retain their zero-origin interpretation. Unknown schemas and malformed files are rejected.

Use `stamp(spec, model, seed=...)` with results. A source checkout's Git HEAD identifies the
base commit, not uncommitted changes; retain the source distribution or source diff and its
checksum for an uncommitted candidate. Installed wheels must not inherit another project's
Git identity. Provenance metadata is not a substitute for recording experiment configuration
and the data-selection protocol.

## Migration from 0.2

No compatibility shims are removed. Invalid configuration and malformed trajectory inputs
that previously slipped through now raise errors. Fault transients now follow the documented
motor dynamics. Coefficient-model CG randomization now moves the aerodynamic reference,
which intentionally changes affected randomized trajectories. `Plant.reset` normalizes finite
nonzero input quaternions robustly and rejects zero rotations or values that overflow the active dtype. Users of raw model tuples
should reconstruct from the specification because `BodyModel` gained a reference-position
field. Rebaseline results affected by these fixes rather than expecting old numerical hashes.

## Additions and migration in 0.4

The exported names in `cascade.experiments` and `cascade.integrations.gymnasium.CascadeEnv`
are new public research interfaces. Experiment and flight-pack JSON formats are versioned
separately from trajectory files. Their loaders validate recorded hashes and reject unsupported
schemas; hashes provide integrity, not proof of licensing or an untouched scientific holdout.

Mission objects resolve through `task_at(task, time_s, rigid)`; static tasks keep their existing
behavior. `Task` is now a runtime-checkable structural protocol for `reference(model,
environment)` and `reference_speed()`, covering static and mission tasks. It is no longer
a closed union of the three original task classes; use concrete classes for exhaustive case
analysis. Runtime protocol checks establish method presence, not configuration validity.
Existing observation dimensions remain unchanged. Sensor ages, validity and freshness
are separate info/state values so policies explicitly choose whether to consume them.
`EnvState` gains timing and sensor bookkeeping: initialize through `reset`, access named fields,
and do not restore old raw tuples or serialized PyTrees. Default sensor settings preserve legacy
noise/delay behavior. Reward under changing weather now uses the same post-step environment as
the returned state and observation; affected numerical reward baselines should be regenerated.

`cascade_policy` remains a privileged baseline: it reads true simulator state and does not
consume the noisy observation vector. The new `observation_cascade_policy` uses required
observation blocks and its own control-step clock; it ignores the simulator-state argument.
Passing `SensorObservation` also supplies measurement ages and validity so it can hold the
previous action when required data are unusable. Passing raw values alone cannot detect stale
finite measurements. Reset controller memory with the episode and call it once per control
step; see [observed control](observed-control.md) for its reconstruction assumptions.

Gain interpolation does not establish stability between knots; qualify the operating ranges
you use. Steady-turn trim solves the simulator's relative equilibrium, not a measured maneuver
envelope. The new synthetic flight pack verifies tooling and is not physical validation.
