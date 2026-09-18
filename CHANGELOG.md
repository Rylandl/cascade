# Changelog

## Unreleased: policy learning and aircraft calibration

- Add bounded aircraft calibration from frozen fitting recordings, using shared differentiable
  open-loop replay, explicit physical residual scales and JAX Jacobians.
- Add named parameter declarations, validated calibrated aircraft specifications, local
  sensitivity/bound diagnostics and versioned calibration artifacts bound to the exact pack.
- Add a fit/evaluate CLI and a synthetic mass/inertia recovery example. Held-out replay preserves
  optimizer outcomes and does not participate in parameter fitting or model selection.

- Add `cascade.learning` with sensor-aware feedforward and recurrent reference policies,
  configurable differentiable training, finite-update checks and explicit optimizer/RNG state.
- Add versioned, atomic, non-pickle checkpoints and inference bundles that preserve trained
  weights, normalization, sensor/action schemas, recurrent memory contracts and provenance.
- Add `sensor_observation` and `sensor_policy` adapters for delivered measurement values,
  ages and validity in ordinary rollouts and experiments. Existing array policies are unchanged.
- Add a frozen tracking/scheduled-tracking workflow with independent training seeds, disjoint
  episode splits, held-out sensor stress, observation-only baseline comparisons and saved curves.
- The learning example now takes a run directory and supports `--resume`; the export example
  takes a training checkpoint and output path. It no longer exports unrelated random weights.

These are feature-development changes; the 0.4.0rc1 artifacts and their evidence are preserved.

## 0.4.0rc1 (unreleased candidate)

- Add frozen experiment manifests with disjoint seed splits, matched policy comparisons,
  terminal-aware metrics, per-flight diagnostics and offline HTML inspection reports.
- Add scheduled tracking, timed waypoint and orbit tasks with shared targets for rewards,
  observations and baseline control; support gain schedules in the baseline policy.
- Add optional per-block sensor rates, drift, dropout, acquisition/transport jitter and delay,
  with measurement age, validity and freshness. Default sensing preserves previous behavior.
- Add steady-turn relative-equilibrium trim and accepted, interpolated airspeed gain schedules.
- Package the Gymnasium integration behind the `gymnasium` extra, retaining the plain adapter.
- Add versioned flight-data packs, explicit licensing and maneuver splits, CSV interchange,
  nominal/calibrated/persistence replay and a labeled synthetic evaluation example.
- Add an observation-only cascade baseline, with optional explicit measurement age/validity,
  and a frozen sustained benchmark campaign with declared acceptance thresholds.
- Describe static tasks and missions through the public structural `Task` protocol; use
  concrete task classes when an exhaustive closed list is required.
- Align post-step reward with the returned mission time, wind and density. This intentionally
  changes costs in episodes whose weather changes over a control period.

The verified 0.3.0rc1 and 0.4.0.dev0 distributions are preserved. This candidate has its own
verification record in `docs/rc-readiness.md`; earlier checks do not qualify later changes.

## 0.3.0rc1 (2026-09-18)

Release-candidate hardening of the research library. Existing v0.2 import shims remain.

- Correct motor-out and partial-power transients: command limits no longer instantly clip
  actual shaft speed. Limits of pi or more now disable episode attitude termination.
- Apply coefficient-aircraft CG shifts to the coefficient reference, local flow, and body
  moments. Optional zero-default `BodySpec.reference_position_m` preserves schema-2 inputs.
- Reject invalid episode/fault/randomization configuration and malformed trajectory data.
  Protect provenance and file metadata, prevent unrelated repository attribution, and add an
  explicit trajectory `t0_s` (zero by default; pass `dt` for a rollout starting at zero).
- Add numerical precision, batching, gradient, and timestep-convergence regression checks;
  qualify installation artifacts and document supported interfaces and migration.
- Withdraw the misleading unfitted-X8 headline; distinguish simulator checks, selected
  historical flight results, and experiments from independent flight validation.
- Clarify policy serialization as a JAX round-trip example and assert numerical agreement.
  Add the optional `export` dependency group.
- Include work since the v0.2 tag: downwash/reference fixes, separated surface centers of
  pressure, action latency, named randomization, observation selection and sensor-unit noise,
  fault schedules, versioned trajectories, provenance, discrete gusts and ISA density.

Runtime bounds now explicitly include NumPy and match JAX's minimum SciPy requirement.
Numerical model/specification hashes can change across versions when their representation
changes, even for a physically identical zero-default reference point. See
`docs/compatibility.md` for migration and `docs/release-readiness.md` for executed evidence.

## 0.2.0 (2026-09-02)

Package layout by layer. The top-level `cascade` namespace is the core only (states, models,
specifications, dynamics, integration, trim and analysis, the canonical boundary, the stepped
plant, the packaged aircraft). The layers above moved into packages:

- `cascade.control`: `loops` (the rate/attitude/guidance cascade), `vtol` (hover and
  transition), `autotune`, and `tuned` (the packaged aircraft's controllers).
- `cascade.env`: `episode` (reset, step, rollouts), `tasks`, `sensors`, `baselines`,
  `weather`, `gusts`, `family`.
- `cascade.design`: `archetypes`.
- `cascade.viz`: `geometry`, `render`.

Compatibility shims keep the old module paths importable (`cascade.vtol`, `cascade.autotune`,
`cascade.weather`, `cascade.gusts`, `cascade.family`, `cascade.archetypes`,
`cascade.geometry`, `cascade.render`), and `cascade.env` keeps its public names. Renamed:
`cascade.env.Reference` is now `ReferenceFlight` (the old name is an alias). Moved:
`control_authority` is in `cascade.analysis`. Tasks own their reference speed and reference
flight (`task.reference_speed()`, `task.reference(model)`), so nothing branches on task type.
Added `py.typed`, a `slow` pytest marker, and a MuJoCo `viz` extra.

## 0.1.0

First pass: panel and coefficient aerodynamics, actuators, RK4 rollouts, trim and
continuation, linearisation, the Skywalker X8 validated against real flight through
Glassbox, the tailsitter fixture, the control cascade and transition controller, the episode
environment with sensors, weather, and families, archetypes with automatic tuning, and
MuJoCo rendering.
