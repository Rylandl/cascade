# Reproducible experiments

`cascade.experiments` connects the native episode functions to frozen scenario definitions,
matched policy comparisons, trajectory files, and offline reports. It runs research experiments
on the host; the simulation inside each scenario is JIT compiled and vmapped over seeds.

```python
from cascade import aerobatic_reference_spec
from cascade.env import EpisodeConfig, tracking_task
from cascade.experiments import Experiment, Scenario, autotuned_policy, run_experiment, trim_policy

experiment = Experiment(
    "tracking",
    (
        Scenario(
            "training",
            aerobatic_reference_spec(),
            tracking_task(12.0, 50.0),
            EpisodeConfig(horizon_steps=160),
            seeds=(1, 2),
            split="training",
        ),
        Scenario(
            "evaluation",
            aerobatic_reference_spec(),
            tracking_task(12.0, 50.0),
            EpisodeConfig(horizon_steps=160),
            seeds=(101, 102),
        ),
    ),
)
experiment.save("dist/tracking-manifest.json")
restored = Experiment.load("dist/tracking-manifest.json")
results = run_experiment(restored, [trim_policy(), autotuned_policy()], "dist/tracking-results")
```

The output folder must be new or empty. A manifest contains the aircraft specification, task,
episode configuration, sensor noise/pipeline, weather, fault schedule, split and seeds. Its
content hash is checked on loading. Different splits must use disjoint seeds; policies within
one scenario receive identical seeds. This prevents accidental seed reuse across declared
splits, not all possible information leakage through research decisions.

Loading also validates the decoded task/schedule, aircraft, sensor settings, weather and
fault shapes/ranges; recomputing a hash does not make an invalid scenario valid. Array dtype
metadata and finite values are checked before execution. Manifests retain their recorded
array precision for content hashing; execution uses the active JAX precision. Inputs that
overflow that precision are rejected. Content hashes detect accidental changes, not the
authenticity or scientific quality of a manifest.

The selected split defaults to `evaluation`; request `training` or `validation` explicitly.
All scenarios retain their full definitions in the saved manifest. Topologies can differ between
scenarios; each scenario compiles independently. Models within a scenario share one airframe.

## Policies and outputs

`trim_policy()` holds the initial trim. `autotuned_policy()` builds a cruise-tuned cascade,
which supports the mission tasks. Custom policies use `Policy(name, factory, metadata=...)`:
the factory receives `(scenario, model, reference)` and returns the existing episode policy
function plus its initial state. The function accepts `(policy_state, observation, env_state)`.
Pure observation-based policies may ignore the simulator state; the built-in model-based
baseline deliberately uses it. Record checkpoint hashes and all policy settings in metadata.
Factories are Python code, not serialized executable content. Save that code or its package
version alongside results; a source hash alone does not reconstruct a checkpoint or closure.

Outputs include:

- `manifest.json`, `results.json`, and `scores.csv` with specification and environment provenance.
- Per-episode return, position/altitude/airspeed/heading RMSE, mean squared normalized action,
  saturation fraction, numerical-finiteness status and completion status.
- A versioned trajectory for every numerically finite flight; diagnostics for every flight,
  including failures. Result rows carry relative file paths; use them instead of inventing
  filenames. Paths are separated by scenario, policy and seed to avoid name collisions.
- Standalone HTML inspectors by default (`reports=False` disables them).

Scores stop at the first termination, inclusive. A flight that becomes nonfinite is retained
as a failed result rather than written to a valid-trajectory file. Completion means reaching
the episode horizon without a crash or numerical failure; it does not mean tracking errors
meet an application-specific success threshold. A hovered task's speed target is zero.
RMSEs use the actual task's reference and position-error definition.
Scoring reads true simulated state and wind, independently of the noisy/delayed observation
available to a policy. Sensor degradation changes the policy's information; it does not add
measurement noise to the ground-truth evaluation metrics.
Saved trajectory controls are the actions actually applied after action delay, converted
to native control units. Diagnostics retain both commanded and applied normalized actions.
Identical seeds match reset draws and random innovations across policies; state-dependent
wind/turbulence can still differ when the policies fly different trajectories.

## Command line

```bash
python -m cascade.experiments dist/tracking-manifest.json dist/tracking-results \
  --policies trim,cascade --split evaluation
```

`--no-reports` skips HTML generation. The CLI uses only the packaged `trim` and `cascade`
baselines; custom policy factories use the Python API. Run `--help` for arguments. The
cascade baseline supports tracking missions, not hover/transition control. Choose a new
output directory for each run; existing results are never overwritten.

Aggregation is per policy and split across the selected scenarios. It reports counts, means,
sample standard deviations and completion fractions. A standard deviation is not a population
confidence interval. Keep scenario-level rows when aircraft, task scales, or difficulties differ;
a pooled reward is not a universal comparison of controller quality.

`examples/research_workflow.py` builds scheduled missions with multirate sensors, wind and a
motor derating, runs matched trim/cascade policies, and writes a comparison report. The example
uses synthetic aircraft fixtures and makes no flight-transfer claim.
