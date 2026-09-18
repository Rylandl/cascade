# Sensor-aware policy learning

`cascade.learning` turns the differentiable tracking example into a reusable train/save/
resume/evaluate/export workflow. The reference policies consume delivered observations,
measurement ages and validity, with optional recurrent memory. They never receive the
aircraft's hidden state. Tracking and scheduled tracking are supported by the episode-return
objective; this is an experimental research layer over the unchanged dynamics core.

## Run a complete experiment

From a checkout with its dependencies installed:

```bash
uv run python -m cascade.learning dist/learning --steps 60
uv run python -m cascade.learning dist/learning --resume --steps 20
uv run python -m cascade.learning dist/learning --resume --steps 0 --export
```

The same commands work as `python -m cascade.learning` from an installed wheel. Export needs
`cascade-flight[export]` (flatbuffers); training and checkpoints need only core dependencies.
The first directory must be new or empty. `--steps` means **additional accepted updates per
training seed**; zero evaluates saved weights. `--resume` restores the frozen settings and
rejects configuration overrides, changed package source, model hashes and runtime settings.
The package source fingerprint works for both editable checkouts and installed wheels;
use the original code when resuming a run. Checkpoints are saved every ten updates and at completion;
after interruption, resume starts at the last complete checkpoint. Independent seeds that
had not started are initialized from their declared seed. Re-running a partially completed
multi-seed invocation adds the requested updates to each seed's saved position.

Defaults are three independent parameter/training seeds (0, 1, 2), a 32-unit feedforward
network, 60 updates, batch size 16 and 160 control steps (4 seconds). For a recurrent
scheduled-tracking run:

```bash
uv run python -m cascade.learning dist/recurrent-learning \
  --architecture recurrent --task scheduled --steps 60 --seeds 0 1 2
```

`--horizon`, `--hidden-size`, `--batch-size`, `--learning-rate`, and
`--evaluation-episodes` configure a new run. `--reports` also produces per-flight HTML
inspectors. Compilation and reverse-mode rollout memory grow with horizon and batch size;
start small when changing hardware. A two-update, eight-step run is a workflow smoke test,
not useful controller-performance evidence.

Outputs include:

- `run.json` and `experiment.json`: hashed settings, aircraft, mission, sensor conditions
  and fixed training/validation/evaluation episode seeds.
- `seed-N/checkpoint.npz`: learned parameters, optimizer moments, PRNG state, update count,
  normalization, trim action, observation/action schemas, runtime stamp and training history.
- `validation-HASH/` and `evaluation-HASH/`: immutable experiment results, scores, per-flight
  diagnostics and finite trajectories, tied to the checkpoint hashes.
- `summary.json` and `summary.md`: training-return/gradient curves, per-condition results and variation across
  independent training runs. Requested exports are `seed-N/checkpoint.cascade-policy`.

Validation results are reported without selecting checkpoints. Training samples only the
512 declared training episode seeds; validation and evaluation use disjoint ranges. Nominal
and sensor-stress evaluation share episode seeds across every policy. Stress includes a
whole-observation delay, delayed/dropped rate packets, slower airspeed samples and delayed
GNSS-position packets with 20% dropout; the manifest freezes the exact settings. This stress
distribution is held out from optimization.

The untrained reference network has zero output weights and produces the saved trim action,
so the `trim` policy is its exact untrained baseline. `observed-cascade` uses the packaged
reference controller through the same measurement-age/validity adapter. Results include
completion, return, tracking errors and saturation; failures remain in the reports. Episode
standard deviations and standard deviations of independent training-run mean returns are
separate, and neither is a population confidence interval. A learned policy need not beat
the cascade. The finite seeded simulator experiment does not establish flight transfer.

## Public training interfaces

`PolicyConfig` defines observation/action dimensions, hidden size, `feedforward` or
`recurrent` architecture, feature scales and the residual-action range. Policies concatenate
scaled/clipped measurements, bounded acquisition ages, and validity indicators. Missing or
malformed measurements are zeroed and assigned maximum age; finite held measurements retain
their validity while aging. Normalization lives in the checkpoint, not an inference caller.

`initialize_policy(config, key)` produces a parameter PyTree; `initial_memory(config)`
produces a zero vector of `hidden_size` for either architecture. Feedforward policies ignore
and zero this memory; recurrent policies update it. `policy_step` accepts
parameters, memory, a `SensorObservation`, configuration and trim action, returning action and
next memory. `make_policy` wraps that function for `rollout_policy` or experiment factories.
Each independently reset episode needs fresh memory. When batching, use independent memories
per episode; the built-in episode objective does this automatically with `vmap`.

```python
import jax
from cascade.learning import (
    PolicyConfig,
    TrainingConfig,
    initialize_policy,
    initialize_training,
    make_episode_return_objective,
    make_train_step,
    train,
)
from cascade.env import action_size, control_to_action, observation_size

# model, episode_config, task and reference are ordinary Cascade environment objects.
network = PolicyConfig(
    observation_size(model, episode_config.observation),
    action_size(model),
    architecture="recurrent",
)
parameter_key, training_key = jax.random.split(jax.random.PRNGKey(0))
state = initialize_training(initialize_policy(network, parameter_key), training_key)
objective = make_episode_return_objective(
    model,
    episode_config,
    task,
    reference,
    network,
    control_to_action(episode_config, reference.control),
)
update = make_train_step(objective, TrainingConfig(batch_size=16))
state, history = train(state, update, 60)
```

The generic optimizer accepts any differentiable scalar `objective(parameters, keys)` to
maximize. It splits the saved PRNG key, evaluates a batch, clips the global gradient, then
applies Adam ascent. A nonfinite objective, gradient, moment or candidate parameter rejects
the update; parameters/moments/update count remain unchanged. `train` raises a
`FloatingPointError` carrying the failed attempt's state and metrics. It never reports a
rejected update as successful. The rejected low-level step advances its random key so a
caller can explicitly decide how to recover. The workflow retains the last accepted saved
checkpoint. Resume equivalence is tested on a fixed stack, not promised across JAX releases,
backends or changed objective code.

## Checkpoints and inference

Use `save_checkpoint`/`load_checkpoint` with explicit observation/action schemas and provenance.
`observation_schema(model, episode_config)` and `action_schema(...)` build the standard schema
descriptions. Schemas fix the selected block layout and normalized action mapping; task
normalization, the aircraft and other environment settings belong in experiment provenance.
Callers creating their own objectives must save and check that context too. Expected-config,
schema and provenance arguments on load reject incompatible resumes.

Checkpoints use numeric NPZ arrays plus JSON metadata and content hashes, with pickle disabled.
Saving uses atomic file replacement. Loading validates versions, contents, dimensions, dtypes,
finite weights/optimizer state, PRNG state and schema/configuration consistency. Hashes detect
changed contents; they are not signatures establishing the author's identity.

```python
from cascade.learning import load_checkpoint

checkpoint = load_checkpoint("dist/learning/seed-0/checkpoint.npz")
policy, memory = checkpoint.make_policy()
# Pass policy and memory to rollout_policy, or return them from an experiment Policy factory.
```

The export example now uses those exact trained weights:

```bash
uv run python examples/export_policy.py dist/learning/seed-0/checkpoint.npz \
  dist/learning/controller.cascade-policy
```

An inference bundle contains a serialized JAX Exported program and hashed JSON metadata.
`load_exported_policy(path).call(values, age_s, valid, memory)` returns action and next memory.
The signature uses float32 values/ages/memory and boolean validity; `initial_memory()` creates
the correct reset state. `make_policy()` adapts a loaded export to the normal rollout callback.
Memory remains explicit even for recurrent policies. Serialization requires the optional export
dependency, and exported programs retain JAX's platform/version restrictions. The round trip
verifies JAX inference agreement, not standalone C, another runtime or onboard execution.

## Migration from the old examples

The old learning script's `--output results.json` only retained scores. It is replaced by a
positional **run directory**, with `summary.json` plus actual checkpoint files. The export
example now requires `CHECKPOINT OUTPUT`; it no longer creates random weights. The existing
three-argument policy callback and default array observations remain compatible. Opt into
sensor metadata using `sensor_policy`; `SensorObservation` remains importable from both
`cascade.env` and its previous `cascade.env.baselines` location.
