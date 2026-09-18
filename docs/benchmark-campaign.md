# Sustained simulator benchmark campaign

`scripts/benchmark_campaign.py` freezes a matched-policy protocol, runs disjoint pilot splits,
and evaluates a held-out split once. It retains every result, including crashes, nonfinite
episodes and threshold failures. This campaign measures simulator regressions on the bundled
aerobatic reference aircraft. Its limits are chosen software-regression bounds, not flight
performance guarantees, an operational envelope, or measured-flight validation.

## Frozen full profile

All episodes last 40 seconds: 1,600 control steps at 40 Hz, with RK4 plant integration at
400 Hz. Each scenario uses training seeds `11, 12`, validation seeds `51, 52`, and evaluation
seeds `201` through `208`. Seeds are disjoint between splits and paired between policies and
between sensor-ablation scenarios. The full evaluation contains **64 episodes**:
four scenario families × two policies × eight seeds.

The aircraft is `aerobatic_reference_spec()`, the component-aerodynamics fixture. Reset
perturbations have standard deviations of 0.5 m position, 0.2 m/s velocity, 0.02 rad attitude
and 0.03 rad/s angular rate. Every scenario enables standard-atmosphere density by altitude.
The portable manifest contains all aircraft parameters and every scenario setting.

| Family | Mission and disturbances | Role |
| --- | --- | --- |
| `nominal-tracking` | Scheduled speed 12–13 m/s, altitude 49–52 m, heading −0.15–0.2 rad; mild wind and turbulence; onboard sensor noise and multirate position/pitot readings. | Supported regression regime. |
| `nominal-route` | Timed, gently turning 480 m NED path over 40 s, 12 m/s, altitude 50–52 m; calm air; the same nominal sensor settings. | Supported regression regime. |
| `stress-mixed` | Scheduled tracking with stronger wind/turbulence, vertical gust, motor derating, action latency, delayed/dropped sensor readings and gyro drift. | Intentional stress characterization. |
| `stress-clean-sensors` | Identical mission, weather, fault, action latency and seeds to `stress-mixed`, with ideal immediate sensor readings. | Paired sensor ablation; stress characterization. |

The scheduled mission interpolates targets at 0, 8, 16, 28 and 40 seconds. Its speed values
are `[12, 12, 13, 12, 12]`, altitude `[50, 50, 52, 49, 50]`, and heading
`[0, 0, 0.2, -0.15, 0]`. The route has waypoints at 0, 10, 20, 30 and 40 seconds:
`[[0,0,-50], [120,0,-50], [240,12,-52], [360,12,-52], [480,0,-50]]` in NED metres,
with a 3-second guidance lookahead. It is a timed path, not an arrival-triggered waypoint planner.

Nominal tracking uses a 0.5 m/s wind at 10 m, blowing from 1 rad clockwise from north,
with a 0.5 m/s turbulence-driving wind at 20 ft. Stress uses 2 m/s for both inputs,
a 1 m/s upward one-minus-cosine gust from 16 to 22 seconds, and motor power fraction 0.8
from 22 seconds. Mean wind varies with altitude according to the weather model.
Stress also adds one control period of action delay. These are synthetic disturbances.

Nominal sensors have pitot updates every two control steps and position updates every four,
with one-step position latency. White noise is 0.1 m/s airspeed, 0.003 rad/s gyro,
0.003 rad attitude, 0.005 rad heading, 0.1 m position and 0.02 m/s² specific force;
gyro bias standard deviation is 0.001 rad/s. Stress adds one-step gyro/pitot latency,
gyro dropout 3%, pitot dropout 5%, position dropout 10%, two-step position latency,
and gyro random-walk standard deviation 0.0002 in observation units per √s.
The clean-sensor ablation removes noise and the block pipeline while retaining plant,
weather, fault and action-delay settings.

## Two explicitly different information sets

`privileged-cascade` uses the existing cascade with true simulated state and wind. It
bypasses the observation pipeline and provides a model-based reference.

`observation-cascade` uses `observation_cascade_policy` with delivered
`onboard_observation()` values and its own clock. It ignores the episode-state argument.
Both policies use nominal autotuning at the initial 12 m/s reference and run all cascade
loops at the control cadence. Neither tunes on evaluation results.

The observation baseline reconstructs roll/pitch from the measured gravity direction,
uses the delivered heading error and body position-error projection for guidance, and uses
pitot airspeed and measured rates. It holds its last bounded action on unusable required
readings. The campaign supplies raw observation vectors; finite held readings can remain
stale, and there is no sensor-age rejection or asynchronous state estimator in this profile.
These observation blocks abstract onboard estimates; the campaign does not model a complete
navigation or attitude-estimation stack.

The full profile reports commanded-action RMS differences between each paired stress run
with and without sensor degradation. It compares their common prefix through termination
and records the number of paired steps. This shows whether delivered sensor quality changes
actions; a difference alone is not proof of better control. Ground-truth tracking scores
remain independent of sensor noise.

## Acceptance declared before evaluation

Both policies have the same per-episode nominal requirements:

| Check | Limit |
| --- | --- |
| Ground-truth state and diagnostics | Finite throughout the scored episode. |
| Crash | None. |
| Completion | All 1,600 expected steps. |
| Altitude RMSE | ≤ 3 m. |
| Airspeed RMSE | ≤ 2 m/s. |
| Heading-error RMSE | ≤ 0.2 rad. |
| Position RMSE, route only | ≤ 15 m. |

Every nominal episode must pass every check. There is no pass-fraction allowance and no
discarding failed seeds. Stress results are assessed against the same displayed tracking
bounds but reported separately; their failures do not constitute acceptance of a stress
capability and do not change the declared nominal regime.

Integrity checks apply to all scenarios. Missing, extra, duplicated or wrong-split episodes,
stale manifest/policy hashes, malformed or absent metrics, and inconsistent durations fail
acceptance. A short episode cannot pass by setting `completed=true`. Saved trajectories are
loaded through the package's finite-state/schema validator, and their length and experiment,
scenario, policy and seed provenance must match the score row. Artifact paths cannot be reused
across rows. Every saved trajectory and diagnostic
file receives an artifact hash. A simulation exception leaves an explicit failed acceptance
record and preserves files already emitted. Interrupted output directories cannot be reused.

## Run pilots, then evaluate once

From the repository:

```sh
python scripts/benchmark_campaign.py --profile full --phase pilot \
  --output dist/0.4.0rc1/campaign
python scripts/benchmark_campaign.py --profile full --phase evaluation \
  --output dist/0.4.0rc1/campaign
```

The pilot command writes the complete manifest and acceptance contract **before** running
training and validation. It never runs evaluation seeds. Inspect pilot failures and choose
a new protocol/version and fresh directory if development is needed. Do not change thresholds
or policies in response to held-out results and then describe the reused set as a blind holdout.
The evaluation command requires successful, hash-matched pilot acceptance, the same protocol,
policy source, package source and campaign script, and no existing evaluation directory.
It rereads and rescores pilot results and checks every retained artifact against its recorded
hash immediately before evaluation; a stale accepted summary is not sufficient.

An installed-wheel run can use an absolute script path from outside the checkout:

```sh
cd /tmp
/path/to/wheel-environment/bin/python -I /path/to/cascade/scripts/benchmark_campaign.py \
  --profile full --phase evaluation --output /path/to/frozen/campaign
```

Only the installed package is imported in that invocation; the campaign script must still
match the frozen script hash. Package-source hashing uses relative Python paths and contents,
so editable and installed copies with identical source agree. Dependencies, backend and
runtime provenance are also recorded by the experiment runner. A changed dependency stack
is a separate reproduction, not additional independent held-out data.

The lightweight CI command is:

```sh
python scripts/benchmark_campaign.py --profile smoke --phase all --output dist/campaign-smoke
```

Smoke runs shortened 4-second tracking and mixed-stress missions, using seeds 10011, 10051
and 10101 for training, validation and evaluation respectively, independent of the full profile,
and both policies. It includes training, validation and evaluation, for 12 short episodes.
It omits the timed route and paired clean-sensor ablation and reduces scheduled target changes.
It tests the end-to-end workflow and is explicitly **not** evidence that the 40-second release
profile passed. All commands return nonzero when nominal acceptance or integrity fails.

Before the final evaluation, the seed pools were separated from an exploratory smoke profile
that had reused seed 101. Full evaluation seeds 201–208 were reserved before execution;
the mission definitions and acceptance limits were unchanged. Exploratory pilot/smoke directories
are retained separately and are not the final release campaign.

Outputs are `manifest.json` and `campaign.json`, plus a directory for each executed split.
Each split contains the existing experiment `results.json`, CSV scores, trajectory and
diagnostic files, then `acceptance.json` and `summary.txt`. The acceptance JSON binds the
contract, result and artifact hashes and lists every per-seed decision. The human summary
reports all failures, completion/finiteness/crash counts, worst errors by scenario/policy,
and paired sensor-ablation action differences where available.

`tests/test_benchmark_campaign.py` uses fabricated metric rows only to test protocol handling:
exact seed coverage, malformed data rejection, failure retention and the distinction between
nominal gates and stress characterization. It does not substitute for executing the campaign.
