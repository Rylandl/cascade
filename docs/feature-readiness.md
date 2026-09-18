# Research workflow expansion: 0.4.0.dev0

This is the archived development-artifact record. Current release-candidate work is tracked
in [rc-readiness.md](rc-readiness.md); the evidence below does not qualify later changes.

Prepared locally on 2026-09-18. This development version implements the feature expansion
described below. It is separate from the preserved [0.3.0rc1 qualification](release-readiness.md).
The new distributions and execution evidence live in `dist/0.4.0.dev0/`, which is ignored by Git.

## Implemented scope

| Feature | Public entry point and guide |
| --- | --- |
| Frozen experiments, matched seeds, scoring and saved results | `cascade.experiments`; [experiments](experiments.md) |
| Offline trajectory inspection and two-flight comparison | `cascade.viz.trajectory_report`; [inspection](inspection.md) |
| Scheduled targets, timed waypoints and orbit following | `cascade.env`; [missions](missions.md) |
| Independent sensor rates, drift, dropout, jitter and delay | `SensorPipelineConfig`; [sensors](sensors.md) |
| Steady-turn trim and airspeed gain schedules | `cascade.analysis`, `cascade.control`; [envelope](envelope.md) |
| Optional NumPy/Gymnasium adapter | `cascade.integrations.gymnasium.CascadeEnv`; [Gymnasium](gymnasium.md) |
| Frozen recording packs and nominal/calibrated replay | `cascade.experiments`; [flight data](flight-data.md) |

Each feature includes examples, boundary validation and regression coverage. The experiment
example combines a scheduled mission, multirate sensors, wind, a motor derating and matched
trim/cascade policies. Its eight flights and comparison report are in `installed-research/`.
`installed-flight-pack/` demonstrates recording ingestion, declared fitting/evaluation splits,
content hashes and replay scores using explicitly synthetic data.

## Executed verification

| Platform / Python | Dependency profile and scope | Result |
| --- | --- | --- |
| Linux aarch64 container / 3.12.14 | Final installed wheel; JAX/jaxlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1 | 518 passed, 1 skipped; 867.89 s |
| macOS arm64 / 3.13.11 | Final installed wheel; locked JAX/jaxlib 0.11.1, NumPy 2.5.2, SciPy 1.18.1 | 518 passed, 1 skipped; 766.76 s |
| macOS arm64 / 3.11.14 | Feature source checks; exact core dependency minima | 146 passed, 14 skipped; 80.46 s |
| macOS arm64 / 3.11.14 | Final inspector boundaries; exact core dependency minima | 17 passed; 2.59 s |
| macOS arm64 / 3.11.14 | Installed wheel; Gymnasium 1.0.0 and exact core dependency minima | 25 passed, 1 skipped; 61.89 s |

Minimum core versions are JAX/jaxlib 0.7.0, NumPy 1.26.0, SciPy 1.12.0 and tomli-w 1.2.0.
The full-suite skip is intentional: the Gymnasium checker does not apply to the test's
simulated no-Gymnasium fallback. The minimum feature source run also skips thirteen tests
requiring the absent Gymnasium extra; the separate installed-wheel minimum-Gymnasium run
covers that adapter with the dependency present. Elapsed times include shared-machine load
and are not performance baselines.

Installed-package executions passed for the combined research workflow, flight-data replay,
flight envelope, sensor timing, Gymnasium adapter and saved-manifest CLI. The core-only wheel
smoke check passed with MuJoCo absent, including packaged aircraft resources, public exports,
both aerodynamic backends, episode stepping, trajectory round-trip, manifest round-trip and
HTML generation. A separate installed-wheel check rendered a real 160×120 RGB frame on macOS.
Its first attempt was blocked by the sandbox's CoreGraphics restriction; the rerun with
graphics access passed.

The offline inspector was exercised in a browser: comparison traces, shared time scrubbing,
sensor-signal selection, playback to the end and a jump to the 2.50 s motor-derating event all
worked without browser console errors. This is a browser smoke check, not cross-browser
qualification. Workflow YAML, ten shell blocks and three embedded Python programs were checked
locally. Ruff lint and formatting and the dependency lock check passed.

An earlier source full run started while inspector boundary checks were being added; its
already-imported implementation failed eight of those new checks. That log is retained as a
nonfinal run. Final regression results above use the frozen wheel containing those fixes.

## Artifacts and audit record

`dist/0.4.0.dev0/` contains the wheel, source distribution and `SHA256SUMS`.
`verification/manifest.json` records their hashes, full regression outcomes and hashes of
verification evidence and installed-example outputs. `verification/source-files.json` records
the project source snapshot. Clean-install, metadata and source-content checks have individual
logs in that directory; consult the manifest for the final artifact identities. The clean
source-distribution installation passed the core-only smoke check outside the checkout, and
Twine accepted both distributions. The rebuilt wheel is byte-identical to the wheel used by
the full regression runs.

The final wheel has SHA-256
`90d927658237a455df89081b4d148a6c01532cdb099b8dbdaa3167c9e21d01d0`.
All 68 packaged files match the source snapshot used for verification. The source archive
captures the project source, including uncommitted changes; the Git base commit alone does not
identify this work.

## Interpretation and remaining external work

This is a development release, with public research APIs rather than a promise of flight
qualification. The recording workflow is complete, but no redistributable measured-flight
dataset was supplied. The bundled synthetic example checks the software and does not establish
physical accuracy. Parameter fitting remains upstream; replay verifies declared fitting inputs
and evaluates a supplied specification rather than fitting it.

Timed waypoints follow a time-indexed path; they are not an arrival-triggered mission planner.
Gain interpolation does not establish stability between knots. Sensor effects operate on the
selected observation blocks and are not a calibrated physical sensor/estimator model.
The inspector is an offline plot viewer; reduced display samples are not numerical evidence.
See the individual guides and [compatibility](compatibility.md) for these contracts.

The CI workflow is configured and checked locally, but remote GitHub jobs have not run for
this uncommitted tree. Linux verification uses an aarch64 container, not a GitHub x86_64 runner.
Windows, Intel macOS, GPU/TPU execution and onboard deployment remain unqualified. Publication,
remote tagging and merging have not been performed. The original 0.3.0rc1 artifacts and their
checksums remain unchanged.
