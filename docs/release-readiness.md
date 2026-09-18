# Release candidate: 0.3.0rc1

This is the archived 0.3.0rc1 record. The working tree now develops 0.4.0.dev0; the checks below
apply to the preserved, hashed distributions, not subsequent feature changes.

Local candidate prepared on 2026-09-18. This record separates executed evidence from configured
CI. The release target is a dependable CPU research library; hardware deployment, new physics,
and the family-learning thesis benchmark are outside this candidate.

## Gates

- [x] Correctness: motor-out transient, coefficient-backend CG shift, host configuration checks.
- [x] Data contracts: validated trajectories, protected metadata, accurate provenance.
- [x] Numerical checks: float32/float64, both aerodynamic backends, JIT/batching, gradients,
      timestep convergence.
- [x] Complete final test suite passes against the installed candidate wheel; Ruff passes.
- [x] Supported Python/dependency matrix agrees with metadata and CI.
- [x] Wheel and source distribution build; clean installs outside the checkout pass.
- [x] Core works without visualization dependencies; a real frame renders separately.
- [x] Release-critical examples pass against the installed distribution.
- [x] Validation claims, public API, compatibility, limitations, and benchmark instructions agree
      with evidence.
- [x] Version/changelog/citation updated; verified-artifact publishing workflow prepared.

## Executed verification

The final installed-wheel suite includes all 363 collected tests, with 362 passing and one
intentional skip: the Gymnasium checker is inapplicable to the deliberately simulated
no-Gymnasium fallback. The checker also runs successfully with Gymnasium installed.

| Platform / Python | Dependency profile and scope | Result |
| --- | --- | --- |
| macOS arm64 / 3.13.11 | Final installed wheel; JAX/jaxlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1 | 362 passed, 1 skipped; 588.08 s |
| Linux aarch64 container / 3.12.14 | Final installed wheel; JAX/jaxlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1 | 362 passed, 1 skipped; 610.35 s |
| macOS arm64 / 3.13.11 | Earlier candidate source; locked JAX 0.11.1 | 312 passed; 584.60 s |
| macOS arm64 / 3.12.12 | Earlier candidate source; locked JAX 0.11.1 | 330 passed, 7 optional Gymnasium skips; 653.86 s |
| macOS arm64 / 3.11.14 | Earlier candidate source; exact direct dependency minima | 307 passed, 5 optional MuJoCo skips; 630.05 s |
| macOS arm64 / 3.11.14 | Final Plant/trajectory/provenance/Gymnasium/body regressions; same minima | 99 passed, 7 optional Gymnasium skips; 32.43 s |

Minimum versions were JAX/jaxlib 0.7.0, NumPy 1.26.0, SciPy 1.12.0, and tomli-w 1.2.0.
Earlier source runs preceded the last boundary regressions; they are not labeled final-wheel
full-suite runs. The final installed suites cover those additions. Both aerodynamic backends
pass float32/float64, scalar/native-batch/vmap/JIT, finite-difference gradient and RK4 convergence
checks. Ruff, `git diff --check`, `uv lock --check`, distribution metadata checks, workflow YAML
parsing, and workflow shell syntax checks passed.

Fresh wheel and sdist core installations passed outside the checkout with isolated Python and
MuJoCo absent. The checks cover packaged aircraft TOMLs and `py.typed`, public imports, both
backends, environment stepping, trajectories, and installed-package provenance. Separate
visualization environments rendered actual 160×120 RGB frames on macOS and Linux/OSMesa.
The minimum optional stack also passed: Python 3.11.14, JAX 0.7.0, flatbuffers 23.1.4, MuJoCo 3.1.0.
Policy serialization assertions passed on both platforms with current dependencies and on the
minimum stack (maximum absolute error 4.47e-8). This demonstrates JAX serialization only.

Installed-package executions passed for the README and environment quickstarts, Gymnasium
shim, high-alpha flight, X8 trim, trim envelope, tailsitter transition/tuning/corridor,
archetypes, family episodes, learning example, X8 panel comparison, and policy export.
The learning and coefficient comparisons retain seeds, configuration, and provenance.
`benchmark_env.py` also completed all ten cases and wrote `throughput.json`; concurrent test
work shared the machine, so those timings are an execution check, not an idle-machine baseline.
The historical selected X8 flight-score headline was withdrawn; see [validation](validation.md).

## Artifacts and audit record

Local artifacts are in `dist/`:

- `cascade_flight-0.3.0rc1-py3-none-any.whl`
- `cascade_flight-0.3.0rc1.tar.gz`
- `SHA256SUMS` and `verification/manifest.json`
- `verification/`: full-suite logs, dependency versions, example commands/script hashes,
  numerical outputs, rendering/export evidence, and final clean-install checks.

The final wheel's SHA-256 is
`55fdac8d909e9dfa210c025912917af167ce9a9e494bf240efc34d3b392294ce`.
The accompanying manifest records the final source archive hash and verification files.
The base commit is `d6613886f8bbdc514a9ae88ff344c23af994df6a`; this candidate contains
uncommitted changes. The source archive and hashes capture the candidate, not that commit
alone. `dist/` is ignored by Git; retain these local artifacts with the review record.

## Configured but not executed remotely

GitHub CI is configured for Linux/macOS Python 3.11–3.13 with locked dependencies, Linux
minimum/current lanes, fresh wheel/sdist installations, export checks, and actual rendering.
Its YAML and shell blocks were checked locally; GitHub runner jobs have not run for these
uncommitted changes. The Linux execution above used an aarch64 container, not a GitHub x86_64
runner. Windows, Intel macOS, GPU/TPU execution, and onboard deployment are not qualified by
this local run. No new held-out flight-accuracy claim is made.

## External actions

The candidate is ready for source and artifact review. Before publication, commit/review the
changes, run the complete configured GitHub workflow, and configure the protected PyPI
environment and Trusted Publisher described in [releasing](releasing.md). Publication uses
the exact artifacts from that successful workflow. Package publication, remote release tags,
and merging remain separate actions requiring authorization; none has been performed.
