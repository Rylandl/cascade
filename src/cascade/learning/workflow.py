"""A reproducible tracking experiment built from the public learning interfaces.

This small reference workflow freezes its aircraft, tasks, sensor conditions and seed
splits before training. It is a simulator benchmark, not an aircraft qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cascade.control import aerobatic_reference_controller
from cascade.env import (
    EpisodeConfig,
    SensorBlockConfig,
    SensorPipelineConfig,
    action_size,
    control_to_action,
    observation_cascade_policy,
    observation_size,
    onboard_observation,
    scheduled_tracking_task,
    sensor_noise,
    sensor_policy,
    tracking_task,
)
from cascade.experiments import Experiment, Policy, Scenario, run_experiment, trim_policy
from cascade.experiments.manifest import content_hash
from cascade.provenance import stamp
from cascade.reference import aerobatic_reference_spec

from .checkpoint import action_schema, load_checkpoint, observation_schema, save_checkpoint
from .policies import PolicyConfig, initialize_policy
from .training import (
    TrainingConfig,
    initialize_training,
    make_episode_return_objective,
    make_train_step,
    train,
)


@dataclass(frozen=True)
class LearningRunConfig:
    """Frozen benchmark settings; ``seeds`` initialize independent training runs."""

    seeds: tuple[int, ...] = (0, 1, 2)
    architecture: str = "feedforward"
    task: str = "tracking"
    horizon_steps: int = 160
    hidden_size: int = 32
    batch_size: int = 16
    evaluation_episodes: int = 8
    learning_rate: float = 0.003

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be nonempty and unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32
            for seed in self.seeds
        ):
            raise ValueError("training seeds must be uint32 integers")
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if self.architecture not in {"feedforward", "recurrent"}:
            raise ValueError("architecture must be feedforward or recurrent")
        if self.task not in {"tracking", "scheduled"}:
            raise ValueError("task must be tracking or scheduled")
        for name in ("horizon_steps", "hidden_size", "batch_size", "evaluation_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.evaluation_episodes > 10000:
            raise ValueError("evaluation_episodes must be <= 10000 to preserve seed splits")
        TrainingConfig(batch_size=self.batch_size, learning_rate=self.learning_rate)
        object.__setattr__(self, "learning_rate", float(self.learning_rate))


def tracking_experiment(config: LearningRunConfig) -> Experiment:
    """Freeze training, validation and paired nominal/stress evaluation episodes."""
    aircraft = aerobatic_reference_spec()
    episode = EpisodeConfig(
        horizon_steps=config.horizon_steps,
        observation=onboard_observation(),
        sensors=SensorPipelineConfig(),
    )
    task = tracking_task(12.0, 50.0, 0.0)
    if config.task == "scheduled":
        duration = config.horizon_steps / episode.control_frequency_hz
        task = scheduled_tracking_task(
            [0.0, duration * 0.25, duration * 0.75, duration],
            [12.0, 12.0, 13.0, 13.0],
            [50.0, 50.0, 51.0, 51.0],
            [0.0, 0.0, 0.12, 0.12],
        )
    noise = sensor_noise(
        air_std=0.005,
        rate_std=0.002,
        gravity_std=0.002,
        heading_std=0.002,
        position_std=0.005,
        specific_force_std=0.002,
    )
    stress = replace(
        episode,
        observation_delay_steps=1,
        sensors=SensorPipelineConfig(
            rates=SensorBlockConfig(delay_steps=1, dropout_probability=0.05),
            airspeed=SensorBlockConfig(sample_period_steps=2, delay_steps=1),
            position_error=SensorBlockConfig(
                sample_period_steps=4, delay_steps=2, dropout_probability=0.2
            ),
        ),
    )
    evaluation = tuple(range(30000, 30000 + config.evaluation_episodes))
    return Experiment(
        "sensor-aware-learning",
        (
            Scenario(
                "training",
                aircraft,
                task,
                episode,
                tuple(range(10000, 10512)),
                "training",
                noise=noise,
            ),
            Scenario(
                "validation",
                aircraft,
                task,
                episode,
                tuple(range(20000, 20000 + config.evaluation_episodes)),
                "validation",
                noise=noise,
            ),
            Scenario("nominal", aircraft, task, episode, evaluation, noise=noise),
            Scenario("sensor-stress", aircraft, task, stress, evaluation, noise=noise),
        ),
        "Fixed-wing observation-only learning; stress is held out from optimization. "
        "Seeded simulator evidence only, with no physical-transfer claim.",
    )


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _implementation_hash():
    """Identify the package code, including uncommitted or installed source changes."""
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _write_json(path, payload):
    """Replace a report only after all its bytes have been written successfully."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _observed_factory(scenario, model, reference):
    policy, memory = observation_cascade_policy(
        aerobatic_reference_controller(), model, scenario.config, scenario.task, reference
    )
    return sensor_policy(lambda memory, reading: policy(memory, reading)), memory


def _checkpoint_policy(path, name):
    checkpoint = load_checkpoint(path)

    def factory(scenario, model, reference):
        if observation_schema(model, scenario.config) != checkpoint.observation_schema:
            raise ValueError("checkpoint observation schema does not match the scenario")
        if action_schema(model, scenario.config) != checkpoint.action_schema:
            raise ValueError("checkpoint action schema does not match the scenario")
        return checkpoint.make_policy()

    return Policy(
        name,
        factory,
        {
            "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "training_seed": checkpoint.provenance["training_seed"],
            "iteration": int(checkpoint.state.iteration),
            "observation_only": True,
        },
    )


def _cached_evaluation(directory, experiment, policies, split, *, reports):
    """Reuse a complete matching attempt, or choose a new output without overwriting one."""
    directory = Path(directory)
    expected_policies = [policy.provenance() for policy in policies]
    expected_episodes = sorted(
        (scenario.name, split, policy.name, seed)
        for scenario in experiment.scenarios
        if scenario.split == split
        for policy in policies
        for seed in scenario.seeds
    )
    prefix = f"{directory.name}-retry-"
    retries = [
        path
        for path in directory.parent.glob(f"{prefix}*")
        if path.name.removeprefix(prefix).isdigit()
    ]
    candidates = [directory, *sorted(retries, key=lambda path: int(path.name[len(prefix) :]))]
    for candidate in candidates:
        results_path = candidate / "results.json"
        if not results_path.is_file():
            continue
        try:
            result = json.loads(results_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # The experiment writer may have been interrupted during its final JSON write.
            continue
        if not isinstance(result, dict) or result.get("schema") != "cascade_results_v1":
            raise ValueError(f"saved evaluation schema mismatch: {results_path}")
        body = {name: value for name, value in result.items() if name != "results_sha256"}
        if result.get("results_sha256") != content_hash(body):
            raise ValueError(f"saved evaluation content hash mismatch: {results_path}")
        if result.get("experiment_sha256") != experiment.sha256:
            raise ValueError(f"saved evaluation manifest hash mismatch: {results_path}")
        if result.get("policies") != expected_policies:
            raise ValueError(f"saved evaluation policy provenance mismatch: {results_path}")
        try:
            actual_episodes = sorted(
                (row["scenario"], row["split"], row["policy"], row["seed"])
                for row in result["episodes"]
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"saved evaluation episode inventory is invalid: {results_path}"
            ) from error
        if actual_episodes != expected_episodes:
            raise ValueError(f"saved evaluation episode inventory mismatch: {results_path}")
        if reports:
            complete_reports = True
            for row in result["episodes"]:
                if row.get("trajectory") is None:
                    continue
                trajectory = Path(row["trajectory"])
                if trajectory.is_absolute() or ".." in trajectory.parts:
                    raise ValueError(f"saved evaluation trajectory path is invalid: {results_path}")
                complete_reports &= (candidate / trajectory.with_suffix(".html")).is_file()
            if not complete_reports:
                continue
        return candidate, result
    if not directory.exists() or not any(directory.iterdir()):
        return directory, None
    attempt = 1
    while directory.with_name(f"{directory.name}-retry-{attempt}").exists():
        attempt += 1
    return directory.with_name(f"{directory.name}-retry-{attempt}"), None


def _summary_rows(result, training_seeds):
    """Keep regimes separate and distinguish training-run variance from episode variance."""
    rows = []
    for scenario in sorted({row["scenario"] for row in result["episodes"]}):
        selected = [row for row in result["episodes"] if row["scenario"] == scenario]
        for policy in sorted({row["policy"] for row in selected}):
            group = [row for row in selected if row["policy"] == policy]
            record = {
                "scenario": scenario,
                "policy": policy,
                "episodes": len(group),
                "completion_rate": float(np.mean([r["completed"] for r in group])),
            }
            for metric in (
                "return",
                "position_rmse_m",
                "airspeed_rmse_m_s",
                "heading_rmse_rad",
                "saturation_fraction",
            ):
                values = [r[metric] for r in group if r[metric] is not None]
                record[metric] = {
                    "mean": float(np.mean(values)) if values else None,
                    "episode_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "count": len(values),
                }
            rows.append(record)
    variance = []
    for scenario in sorted({row["scenario"] for row in rows}):
        means = [
            row["return"]["mean"]
            for row in rows
            if row["scenario"] == scenario
            and row["policy"] in {f"learned-{seed}" for seed in training_seeds}
        ]
        valid = [value for value in means if value is not None]
        variance.append(
            {
                "scenario": scenario,
                "training_runs": len(means),
                "finite_runs": len(valid),
                "mean_return": float(np.mean(valid)) if valid else None,
                "training_run_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
            }
        )
    return rows, variance


def run_learning(
    output: str | Path,
    config: LearningRunConfig | None = None,
    *,
    steps: int = 60,
    resume: bool = False,
    export: bool = False,
    reports: bool = False,
    checkpoint_interval: int = 10,
) -> dict:
    """Train each seed for ``steps`` additional updates, then score held-out episodes.

    Resume restores the frozen configuration and optimizer/RNG state. The last complete
    checkpoint is authoritative after interruption. ``steps=0`` evaluates saved policies.
    Evaluation directories are immutable; repeated evaluation at the same state is reused.
    """
    for name, value, minimum in (
        ("steps", steps, 0),
        ("checkpoint_interval", checkpoint_interval, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    output = Path(output)
    configuration_path = output / "run.json"
    if resume:
        wrapper = json.loads(configuration_path.read_text())
        payload = wrapper["configuration"]
        if wrapper["sha256"] != content_hash(payload):
            raise ValueError("learning run configuration hash mismatch")
        restored = LearningRunConfig(**{**payload, "seeds": tuple(payload["seeds"])})
        if config is not None and config != restored:
            raise ValueError("resume configuration differs from the frozen run")
        config = restored
        experiment = Experiment.load(output / "experiment.json")
        if experiment.sha256 != tracking_experiment(config).sha256:
            raise ValueError(
                "frozen experiment differs from this workflow; use its original version"
            )
    else:
        config = LearningRunConfig() if config is None else config
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("learning output must be absent or empty; use resume=True")
        output.mkdir(parents=True, exist_ok=True)
        payload = _json_value(asdict(config))
        _write_json(configuration_path, {"configuration": payload, "sha256": content_hash(payload)})
        experiment = tracking_experiment(config)
        experiment.save(output / "experiment.json")

    scenario = next(s for s in experiment.scenarios if s.split == "training")
    model = scenario.aircraft.to_model()
    reference = scenario.task.reference(model)
    trim_action = control_to_action(scenario.config, reference.control).astype(jnp.float32)
    policy_config = PolicyConfig(
        observation_size(model, scenario.config.observation),
        action_size(model),
        hidden_size=config.hidden_size,
        architecture=config.architecture,
    )
    training_config = TrainingConfig(
        batch_size=config.batch_size, learning_rate=config.learning_rate
    )
    obs_schema = observation_schema(model, scenario.config)
    act_schema = action_schema(model, scenario.config)
    episode_objective = make_episode_return_objective(
        model,
        scenario.config,
        scenario.task,
        reference,
        policy_config,
        trim_action,
        noise=scenario.noise,
    )
    seed_pool = jnp.asarray(scenario.seeds, dtype=jnp.uint32)

    def objective(parameters, keys):
        indices = jax.vmap(lambda key: jax.random.randint(key, (), 0, len(scenario.seeds)))(keys)
        episode_keys = jax.vmap(jax.random.PRNGKey)(seed_pool[indices])
        return episode_objective(parameters, episode_keys)

    update = make_train_step(objective, training_config)
    histories, paths = {}, []
    runtime = stamp(scenario.aircraft, model)
    runtime_fields = (
        "cascade_version",
        "jax_version",
        "jaxlib_version",
        "backend",
        "x64_enabled",
        "spec_hash",
        "model_hash",
    )
    implementation = _implementation_hash()
    for seed in config.seeds:
        path = output / f"seed-{seed}" / "checkpoint.npz"
        if resume and path.exists():
            checkpoint = load_checkpoint(
                path,
                expected_policy_config=policy_config,
                expected_training_config=training_config,
                expected_observation_schema=obs_schema,
                expected_action_schema=act_schema,
            )
            provenance = checkpoint.provenance
            if (
                provenance["experiment_sha256"] != experiment.sha256
                or provenance["run_configuration_sha256"] != content_hash(payload)
                or provenance["training_seed"] != seed
            ):
                raise ValueError("checkpoint belongs to a different training run")
            if provenance.get("implementation_sha256") != implementation:
                raise ValueError("resume requires the original Cascade package source")
            if any(provenance["runtime"][field] != runtime[field] for field in runtime_fields):
                raise ValueError("resume requires the original JAX/backend/precision configuration")
            state, history = checkpoint.state, list(provenance["history"])
        else:
            parameter_key, training_key = jax.random.split(jax.random.PRNGKey(seed))
            state = initialize_training(
                initialize_policy(policy_config, parameter_key), training_key
            )
            history = []
            provenance = {
                "training_seed": seed,
                "experiment_sha256": experiment.sha256,
                "run_configuration_sha256": content_hash(payload),
                "runtime": runtime,
                "implementation_sha256": implementation,
            }

        def save(current, path=path, provenance=provenance, history=history):
            save_checkpoint(
                path,
                current,
                policy_config,
                training_config,
                trim_action,
                observation_schema=obs_schema,
                action_schema=act_schema,
                provenance={**provenance, "history": history},
            )

        if not path.exists():
            save(state)

        def progress(current, metrics, history=history, seed=seed, save=save):
            record = {
                "iteration": int(current.iteration),
                "objective": float(metrics.objective),
                "gradient_norm": float(metrics.gradient_norm),
            }
            history.append(record)
            if int(current.iteration) % checkpoint_interval == 0:
                save(current)
                print(
                    f"seed {seed}, step {record['iteration']}: "
                    f"training return {record['objective']:.3f}",
                    flush=True,
                )

        state, _ = train(state, update, steps, callback=progress)
        save(state)
        histories[str(seed)] = history
        paths.append(path)
        if export:
            from .export import export_policy

            export_policy(path, path.with_suffix(".cascade-policy"))

    policies = [
        trim_policy(),
        Policy(
            "observed-cascade",
            _observed_factory,
            {"observation_only": True, "sensor_metadata": True},
        ),
    ]
    policies.extend(
        _checkpoint_policy(path, f"learned-{seed}")
        for path, seed in zip(paths, config.seeds, strict=True)
    )
    checkpoint_hashes = {
        str(seed): hashlib.sha256(path.read_bytes()).hexdigest()
        for seed, path in zip(config.seeds, paths, strict=True)
    }
    evaluation_id = content_hash(checkpoint_hashes)[:16]
    evaluations = {}
    for split in ("validation", "evaluation"):
        directory, result = _cached_evaluation(
            output / f"{split}-{evaluation_id}", experiment, policies, split, reports=reports
        )
        if result is None:
            print(f"Scoring {split} episodes", flush=True)
            result = run_experiment(experiment, policies, directory, split=split, reports=reports)
        rows, variance = _summary_rows(result, config.seeds)
        evaluations[split] = {
            "results": str(directory.relative_to(output) / "results.json"),
            "by_scenario_policy": rows,
            "across_training_runs": variance,
        }
    summary = {
        "schema": "cascade_learning_run_v1",
        "configuration": payload,
        "experiment_sha256": experiment.sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "training_curves": histories,
        "evaluations": evaluations,
        "interpretation": "Trim is the untrained zero-residual policy. Validation is reported "
        "without checkpoint selection; evaluation is never used by the optimizer. Standard "
        "deviations describe this finite seed set, not confidence intervals. Sensor stress is "
        "held out from training. No superiority over the cascade or flight transfer is assumed.",
    }
    _write_json(output / "summary.json", summary)
    _write_summary(output / "summary.md", summary)
    print(f"Saved checkpoints, learning curves and scores to {output}", flush=True)
    return summary


def _write_summary(path, summary):
    lines = [
        "# Sensor-aware policy learning",
        "",
        f"Architecture: **{summary['configuration']['architecture']}**. "
        f"Task: **{summary['configuration']['task']}**.",
        "",
        "Frozen experiment: `" + summary["experiment_sha256"] + "`.",
        "",
        "The trim policy is the untrained zero-residual network. All policies use "
        "the same sensor settings and paired episode seeds. Figures describe this finite "
        "seeded simulator experiment.",
    ]
    for split, result in summary["evaluations"].items():
        lines.extend(
            [
                "",
                f"## {split.capitalize()}",
                "",
                f"[Detailed scores and trajectory references]({result['results']})",
                "",
                "| Condition | Policy | Completion | Mean return | "
                "Position RMSE (m) | Saturation |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["by_scenario_policy"]:

            def number(metric, row=row):
                value = row[metric]["mean"]
                return "undefined" if value is None else f"{value:.3f}"

            lines.append(
                f"| {row['scenario']} | {row['policy']} | {row['completion_rate']:.0%} | "
                f"{number('return')} | {number('position_rmse_m')} | "
                f"{number('saturation_fraction')} |"
            )
        lines.extend(["", "Variation across independent trained controllers:", ""])
        for row in result["across_training_runs"]:
            mean, std = row["mean_return"], row["training_run_std"]
            mean_text = "undefined" if mean is None else f"{mean:.3f}"
            std_text = "undefined" if std is None else f"{std:.3f}"
            lines.append(
                f"- {row['scenario']}: mean return {mean_text}, sample standard deviation "
                f"{std_text} across {row['training_runs']} training runs."
            )
    lines.extend(
        [
            "",
            "[Training curves and complete summary](summary.json)",
            "",
            summary["interpretation"],
            "",
        ]
    )
    Path(path).write_text("\n".join(lines))
