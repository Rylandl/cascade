# Testing a prerelease

Use the exact wheel and source distribution from a candidate's CI run. Record the candidate
version, source commit and artifact hashes with your feedback. The development-snapshot and
older release-candidate results do not qualify a new artifact. Current evidence is tracked in
[release readiness](rc-readiness.md).

## Install outside the source checkout

Download and extract the `cascade-distributions-COMMIT_SHA` artifact from the candidate's
GitHub Actions run. Its root contains `SHA256SUMS` and `dist/` with one wheel and one source
distribution. With Python 3.11, 3.12 or 3.13 available, run these Bash commands. Replace the
first path with the extracted artifact directory and use the Python version you want to test.

```bash
bundle=/absolute/path/to/extracted-artifact
(cd "$bundle" && shasum -a 256 -c SHA256SUMS)
work=$(mktemp -d)
python3.13 -m venv "$work/env"
py="$work/env/bin/python"
wheels=("$bundle"/dist/*.whl)
sdists=("$bundle"/dist/*.tar.gz)
"$py" -m pip install "${wheels[0]}"
"$py" -m pip check
"$py" -m pip freeze > "$work/dependencies.txt"
mkdir "$work/source"
tar -xzf "${sdists[0]}" -C "$work/source"
sources=("$work"/source/*)
source_dir="${sources[0]}"
version=$("$py" -I -c 'from importlib.metadata import version; print(version("cascade-flight"))')
cd "$work"
CI=1 JAX_PLATFORMS=cpu "$py" -I "$source_dir/scripts/release_smoke.py" \
  --core-only --expected-version "$version" > core-smoke.log 2>&1
```

The smoke check verifies that Cascade was imported from the environment's installed package,
not the extracted source. It checks both aerodynamic backends, packaged aircraft resources,
episodes, serialization and report generation without requiring MuJoCo. Review `core-smoke.log`
after the command completes; retain failures as well as successful output. Windows and hardware
accelerators are outside the candidate's qualified target matrix.

## Exercise the workflows you use

Continue in the same temporary directory. These commands create new output directories and
use the installed package with scripts from the matching source distribution:

```bash
CI=1 JAX_PLATFORMS=cpu "$py" -I "$source_dir/examples/research_workflow.py" \
  --output "$work/research" > research.log 2>&1
CI=1 JAX_PLATFORMS=cpu "$py" -I "$source_dir/examples/flight_data_evaluation.py" \
  --output "$work/flight-data" > flight-data.log 2>&1
"$py" -m pip install "${wheels[0]}[gymnasium]"
CI=1 JAX_PLATFORMS=cpu "$py" -I "$source_dir/examples/gymnasium_shim.py" > gymnasium.log 2>&1
"$py" -m pip freeze > "$work/dependencies-with-gymnasium.txt"
```

Open the generated research HTML reports locally and check playback, scrubbing, the signal
selector and comparison traces. The flight-data example is explicitly synthetic: its scores
verify the recording/replay software rather than the physical accuracy of an aircraft model.
For your own aircraft or controller, record the specification, observation layout, action
scaling, scenario, seed and relevant configuration. Avoid tuning against an evaluation split.

## Feedback checklist

- Candidate version, source commit and wheel/sdist SHA-256 hashes.
- OS, architecture, Python version and the saved dependency list.
- Command and input needed to reproduce the issue; expected and actual behavior.
- Full traceback or failing log, including the first error. For numerical failures, include
  the scenario, seed and first failing step rather than only a plot.
- For API compatibility, the previous working version and the public import, named field or
  file format that changed.
- For inspector issues, browser/version, the generated report and the failing interaction.
- For measured flight data, permission to share it and the distinction between fitting and
  held-out evaluation records. Do not include restricted data or credentials in a public issue.

Report issues at the [project issue tracker](https://github.com/Rylandl/cascade/issues), or
return the evidence to the person coordinating the candidate review. Successful checks are
also useful when they identify a tested platform and workflow. No feedback is sent automatically.
