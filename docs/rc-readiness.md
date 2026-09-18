# Release candidate: 0.4.0rc1

Preparation started on 2026-09-18. This is the current candidate record. The earlier
[0.3.0rc1](release-readiness.md) and [0.4.0.dev0](feature-readiness.md) records describe their
preserved distributions and do not qualify changed source or metadata.

## Scope and qualification protocol

The candidate adds reproducible experiments, offline trajectory inspection, missions,
multirate sensor effects, turn trim, gain scheduling, a public Gymnasium adapter and flight-data
evaluation packs. Release closeout adds a sensor-driven controller and a sustained benchmark
campaign, with explicit acceptance criteria and retained failure results.

The public API review preserves all previously exported names. `Task` is now a structural
protocol covering both static tasks and missions; see [compatibility](compatibility.md).
The observation controller has independent checks for units, signs, mission timing, hidden-state
independence, stale readings and numerical overflow. Its 21 focused tests passed on current
JAX 0.11.1 and minimum JAX 0.7.0. The task-interface checks passed on both stacks as well.

The [campaign protocol](benchmark-campaign.md) declares 40-second flights across four scenario
families, two policies and eight reserved evaluation seeds (201–208): 64 evaluation episodes,
preceded by 32 training/validation episodes. All nominal episodes must finish, remain finite
and avoid crashes, with altitude RMSE at most 3 m, airspeed RMSE at most 2 m/s, heading RMSE
at most 0.2 rad and route position RMSE at most 15 m. Stress results are characterized
separately. These are simulator regression limits, not an operational flight envelope.

Pilot development runs established feasibility before evaluation. They are retained under
`campaign-prefreeze*` and are not the final installed-candidate evidence. The final pilot and
evaluation must use the same frozen source, script and acceptance contract. Evaluation
rechecks all retained pilot results and artifact hashes, including each trajectory's seed.

Qualification is commit-bound. After committing the source, require a successful full
[GitHub CI run](https://github.com/Rylandl/cascade/actions/workflows/ci.yml) for that candidate,
passing installed-candidate campaign records, and clean wheel/sdist installation checks.
The immutable source archive is built before those runs; their outcomes and exact hashes
belong to the CI artifacts, the candidate pull request and the local verification manifest.
Configured checks and earlier-version results are not substitutes for executed evidence.

## Artifact and source identity

Local candidate artifacts and evidence use `dist/0.4.0rc1/`. The release source is prepared on
`codex/0.4-release-candidate`. The candidate pull request records its exact commit and CI run.
Build outputs remain ignored by Git, so retain checksums and verification records separately.
The GitHub workflow builds distributions once and tests those exact bytes across its matrix.
It retains source/runtime identity, dependency versions, JUnit reports, console logs, example
outputs and the smoke-campaign results, including failures. Artifacts expire after 30 days
unless retained separately. Download and preserve them with their checksums for release review.

The local `verification/manifest.json` and `verification/release-status.md` bind the executed
candidate checks to source/artifact hashes. `campaign/evaluation/acceptance.json` is the full
campaign decision record, and `campaign/evaluation/summary.txt` explains every threshold
failure and the sensor-ablation results. The short CI campaign uses separate seeds and does
not replace the 40-second evaluation. No failed seed is discarded.

## External requirements and limits

Code access to `Rylandl/cascade` is available. Protected GitHub environment settings and PyPI
Trusted Publishing require authenticated settings access and will not be inferred from the
workflow file. Package publication, tagging and merging have not been performed.

External tester recipients have been requested. The [tester guide](prerelease-feedback.md)
specifies installation, an end-to-end experiment and the environment/evidence to report. No external test result is
claimed before a participant actually reports one.

No redistributable measured-flight dataset was supplied. The flight-data workflow remains
qualified with explicitly synthetic data only; physical accuracy, real-flight robustness and
onboard deployment are outside the demonstrated evidence. Windows, Intel macOS and GPU/TPU
execution are not included in the local platform qualification.
