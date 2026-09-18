# 0.4 release-candidate closeout

The authorized scope is source review and commits, corrected release instructions, a sustained
benchmark campaign, an observation-only baseline, execution of the remote CI matrix and
preparation of an exact 0.4.0rc1 candidate. Preserve earlier hashed artifacts. External tester
feedback and measured-flight data require suitable participants/data; publication remains an
explicit final action after the candidate and publishing configuration are reviewable.

## Source preparation

- [x] Review public APIs, compatibility and the accumulated source changes.
- [x] Add an observation-only controller and verify that sensor degradation affects its input.
- [x] Freeze a sustained, seeded campaign and acceptance thresholds before evaluation.
- [x] Correct version-specific release commands and prepare tester instructions.

## Verification after the source commit

Execution outcomes are recorded in the candidate PR/CI artifacts and
`dist/0.4.0rc1/verification/release-status.md`, rather than changing the source archive after
its verification. Complete all of these gates for the exact candidate source and artifacts:

- Build 0.4.0rc1; verify distributions, clean installation and source/artifact identity.
- Commit the reviewed source on `codex/0.4-release-candidate` and open a draft pull request.
- Execute installed-candidate training/validation pilots and the frozen evaluation; retain
  successes and failures without changing thresholds or reusing evaluation data for tuning.
- Execute the full GitHub installed-artifact matrix; resolve actionable failures.
- Preserve commit-bound test/campaign evidence and exact distribution checksums.

## External follow-through

- Inspect protected GitHub/PyPI publishing settings when authenticated access is available.
- Obtain external tester feedback after recipients are identified.

See [current candidate record](../docs/rc-readiness.md) for executed results and outstanding
external requirements. Real-flight qualification requires licensed recordings and a frozen,
independent evaluation protocol; synthetic examples are not physical validation.
