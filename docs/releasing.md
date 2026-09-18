# Preparing a release candidate

The release target is a CPU research library on Python 3.11–3.13, Linux and macOS.
Follow [0.4 release readiness](rc-readiness.md) for the current evidence and remaining gates.
The [development feature record](feature-readiness.md) and
[0.3 qualification record](release-readiness.md) describe their archived artifacts.
CI configuration describes checks that will run; it is not evidence that a particular
commit, runner, or artifact has passed them.

## Local verification

From the repository, with `uv` installed:

```bash
uv sync --locked
uv run --locked ruff check .
CI=1 uv run --locked pytest
repo="$PWD"
version=$(.venv/bin/python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
candidate="$repo/dist/$version"
mkdir -p "$candidate"
uv build --no-sources --out-dir "$candidate"
uvx --from twine twine check "$candidate"/*.whl "$candidate"/*.tar.gz
(cd "$candidate" && shasum -a 256 ./*.whl ./*.tar.gz > SHA256SUMS)
```

Use the version-specific candidate directory throughout these Bash commands. Keep earlier
versions in place; do not select artifacts with root-level `dist/*` globs. Before rebuilding
the same version, retain its earlier artifacts and evidence in a separate archive. A version
directory should contain exactly one wheel and one source distribution for verification.
Keep the wheel and source distribution together with their SHA-256 hashes, test logs,
Python/dependency versions, and the source commit. A commit identifier alone does not
capture uncommitted changes. Do not call a modified worktree reproducible from its HEAD.

Test both built distributions using fresh environments outside the checkout. These Bash
commands resolve currently available dependencies within the package's declared ranges:

```bash
work=$(mktemp -d)
for artifact in "$candidate"/*.whl "$candidate"/*.tar.gz; do
  target="$work/$(basename "$artifact")"
  uv venv --python 3.13 "$target"
  uv pip install --python "$target/bin/python" "$artifact"
  (
    cd "$work"
    "$target/bin/python" -I "$repo/scripts/release_smoke.py" \
      --core-only --expected-version "$version"
  )
done
```

`release_smoke.py` rejects editable/source-tree imports and checks packaged aircraft TOMLs,
`py.typed`, public imports, JIT rollout with both aerodynamic backends, an environment step,
trajectory serialization, and installed-package provenance. `--core-only` also proves
MuJoCo is absent. The full test suite remains a separate gate.

Run `examples/high_alpha.py`, `examples/gymnasium_shim.py`, `examples/x8_trim.py`,
`examples/flight_envelope.py`, `examples/research_workflow.py`,
`examples/flight_data_evaluation.py`, and `examples/calibrate_aircraft.py`
with the installed wheel's Python and `-I`, from outside
the checkout. Retain generated reports and replay scores with the test evidence. Calibration
must retain optimizer status, synthetic recovery error and held-out replay results; it does not
qualify measured-flight accuracy. The Gymnasium
example should also be exercised with the `gymnasium` extra installed. For the policy export example,
create a fresh environment and install only the wheel's `export` extra:

```bash
uv venv --python 3.13 "$work/export"
wheels=("$candidate"/*.whl)
uv pip install --python "$work/export/bin/python" "${wheels[0]}[export]"
(
  cd "$work"
  "$work/export/bin/python" -I "$repo/scripts/release_smoke.py" --core-only --expected-version "$version"
  "$work/export/bin/python" -I -m cascade.learning "$work/learning" \
    --steps 2 --seeds 0 --horizon 8 --hidden-size 4 --batch-size 2 --evaluation-episodes 1
  "$work/export/bin/python" -I "$repo/examples/export_policy.py" \
    "$work/learning/seed-0/checkpoint.npz" "$work/policy.cascade-policy"
)
```

The example asserts action and memory agreement between the saved trained policy and its
deserialized JAX artifact across seeded sensor packets. The short training run above checks
the workflow, not learning performance. This qualifies the JAX serialization round trip only;
it does not qualify an onboard runtime or the performance of a trained policy.

For visualization, create another clean environment and install the wheel's `viz` extra.
Run the same script with `--viz`, from outside the checkout. It must render a nonempty
160×120 RGB frame, not merely import MuJoCo or construct an MJCF model. Headless Linux needs
an OpenGL backend; CI installs `libosmesa6` and uses `MUJOCO_GL=osmesa` with
`PYOPENGL_PLATFORM=osmesa`. On macOS, use a session with a working graphics context.
The smoke check does not encode video; video export additionally requires `ffmpeg`.

## CI gates and dependency profiles

[CI](../.github/workflows/ci.yml) builds the wheel and sdist once and records `SHA256SUMS`.
Every test job downloads those exact artifacts. The full suite runs against the installed
wheel with isolated Python, from outside the checkout, on Linux and macOS with Python
3.11, 3.12, and 3.13. Each lane also smoke-tests fresh wheel and sdist installations
without optional visualization dependencies, using constraints from that lane's environment.

The normal matrix uses `uv.lock`. A Python 3.13 Linux lane resolves current allowed
dependencies; a Python 3.11 Linux lane uses JAX/jaxlib 0.7.0, NumPy 1.26.0, SciPy 1.12.0,
tomli-w 1.2.0, and Gymnasium 1.0.0. The latter qualifies the direct dependency floors and
the optional Gymnasium adapter's lower bound, not every possible
combination of intermediate versions. A separate Linux lane renders through the installed
visualization extra. GPU execution is outside this release's qualified support matrix.
The current-dependency lane additionally executes the high-alpha, Gymnasium, X8 trim,
flight-envelope, experiment-runner, and synthetic flight-data examples against the installed
wheel and verifies policy serialization in a clean
`wheel[export]` environment. These examples are not duplicated across every matrix lane.
It also runs `scripts/benchmark_campaign.py --profile smoke --phase all` with the installed
wheel, retaining the frozen configuration, results and acceptance checks. This short software
check is separate from the sustained candidate campaign recorded in release readiness.

Every pull request runs this matrix; a workflow file alone does not qualify the source.
The uploaded artifact is named `cascade-distributions-COMMIT_SHA` and contains `dist/`
plus `SHA256SUMS`. Separate `cascade-evidence-*` artifacts retain dependency versions,
test logs and JUnit results, smoke logs, generated example outputs and campaign results,
including logs from failed jobs. Evidence names include the run attempt so reruns retain
earlier failure records. CI retains these artifacts for 30 days; download the
distributions and evidence for a permanent release archive.
Review the results of all matrix and visualization jobs before calling
the candidate verified. Any source or metadata change after verification requires a rebuild
and appropriate re-verification.

## Explicit publication

[Release candidate](../.github/workflows/release.yml) is manually dispatched. Its default
`publish=false` runs the entire reusable CI workflow and retains the artifacts for review.
It neither creates nor pushes a tag.

After publication has been authorized, the publication run must be dispatched from the
exact `vVERSION` tag with `publish=true`, where `VERSION` is the version recorded in the
candidate's wheel metadata. The workflow rejects a publication request from a branch.
All verification jobs run first. The publication job checks artifact hashes and the wheel's
version against the tag, then uploads those same wheel/sdist bytes without rebuilding them.

Before enabling publication, configure the repository's `pypi` environment with a required
reviewer and register a PyPI Trusted Publisher for this repository, workflow `release.yml`,
and environment `pypi`. These repository/account settings cannot be supplied by the workflow
file. The publishing job alone requests an OIDC token. See the
[PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
for the Trusted Publishing and protected-environment setup.

No tag, merge, or package publication is implied by preparing or testing a local candidate.
Use the [prerelease feedback checklist](prerelease-feedback.md) for external testing before
promoting a candidate to a final release.
