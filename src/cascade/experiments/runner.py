"""Run matched policy comparisons and retain the flights behind every score."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cascade.env import action_to_control, cascade_policy, control_to_action, reset, step
from cascade.env.tasks import HoverTask
from cascade.provenance import stamp
from cascade.trajectory import save_trajectory

from .manifest import Experiment, canonical_json, content_hash, valid_name


@dataclass(frozen=True)
class Policy:
    """Factory(scenario, model, reference) -> (policy function, initial policy state).

    Record configuration (including checkpoint hashes) in metadata for custom policies.
    A Python callable is not serialized; restoring it is the caller's responsibility.
    """

    name: str
    factory: Callable
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        valid_name(self.name)
        if not callable(self.factory):
            raise TypeError("policy factory must be callable")
        canonical_json(self.metadata)

    def provenance(self):
        try:
            source = inspect.getsource(self.factory).encode()
            digest = hashlib.sha256(source).hexdigest()
        except (OSError, TypeError):
            digest = None
        return {
            "name": self.name,
            "metadata": self.metadata,
            "factory_source_sha256": digest,
            "factory": (
                f"{getattr(self.factory, '__module__', type(self.factory).__module__)}."
                f"{getattr(self.factory, '__qualname__', type(self.factory).__qualname__)}"
            ),
        }


def _trim_factory(scenario, model, reference):
    action = control_to_action(scenario.config, reference.control)
    return lambda memory, observation, state: (action, memory), ()


def _cascade_factory(scenario, model, reference):
    from cascade.control import tune_cascade
    from cascade.env.tasks import task_at

    initial = task_at(scenario.task, 0.0, reference.state.rigid_body)
    if not hasattr(initial, "airspeed_m_s"):
        raise ValueError("the autotuned cascade policy requires a tracking task or mission")
    controller, _ = tune_cascade(
        scenario.aircraft,
        float(initial.airspeed_m_s),
        model=model,
        altitude_m=float(initial.altitude_m),
        environment=reference.environment,
        simulation_dt_s=scenario.config.simulation_dt_s,
    )
    return cascade_policy(controller, model, scenario.config, scenario.task, reference)


def trim_policy() -> Policy:
    return Policy("trim", _trim_factory, {"kind": "constant initial reference actuation"})


def autotuned_policy() -> Policy:
    return Policy("cascade", _cascade_factory, {"kind": "nominal cruise autotuning"})


def _simulate(scenario, model, reference, policy, memory):
    from cascade.env.tasks import task_at

    config, task = scenario.config, scenario.task
    dt = 1.0 / config.control_frequency_hz

    def run(seed):
        state, obs = reset(
            model,
            config,
            task,
            reference,
            jax.random.PRNGKey(seed),
            noise=scenario.noise,
            weather=scenario.weather,
        )

        def advance(carry, _):
            state, obs, memory = carry
            action, memory = policy(memory, obs, state)
            advanced, obs, reward, done, info = step(
                model,
                config,
                task,
                reference,
                state,
                action,
                noise=scenario.noise,
                weather=scenario.weather,
                faults=scenario.faults,
            )
            rigid = advanced.aircraft.rigid_body
            target = task_at(task, advanced.step * dt, rigid)
            airspeed = jnp.linalg.norm(rigid.velocity - advanced.wind_ned)
            desired_speed = (
                jnp.asarray(0.0)
                if isinstance(target, HoverTask)
                else getattr(target, "airspeed_m_s", getattr(target, "cruise_speed_m_s", 0.0))
            )
            diagnostics = {
                "commanded_action": action,
                "applied_action": info["applied_action"],
                "reward": reward,
                "done": done,
                "crashed": info["crashed"],
                "position_error_m": target.position_error(rigid),
                "airspeed_m_s": airspeed,
                "airspeed_error_m_s": airspeed - desired_speed,
                "heading_error_rad": jnp.arctan2(
                    jnp.sin(target.heading_error(rigid)), jnp.cos(target.heading_error(rigid))
                ),
                "wind_ned_m_s": advanced.wind_ned,
            }
            for name in ("sensor_age_s", "sensor_valid", "sensor_sampled", "sensor_dropped"):
                if name in info:
                    diagnostics[name] = info[name]
            return (advanced, obs, memory), (advanced.aircraft, diagnostics)

        return jax.lax.scan(advance, (state, obs, memory), None, length=config.horizon_steps)[1]

    return jax.jit(jax.vmap(run))


def episode_metrics(diagnostics: dict[str, np.ndarray], dt: float) -> dict[str, Any]:
    """Score through the first termination inclusive; later simulated samples never count."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("metric timestep must be finite and positive")
    done = np.asarray(diagnostics["done"], dtype=bool)
    if done.ndim != 1 or any(np.asarray(v).shape[:1] != done.shape for v in diagnostics.values()):
        raise ValueError("diagnostics must have one aligned time axis")
    length = int(np.flatnonzero(done)[0] + 1) if done.any() else len(done)
    if not length:
        raise ValueError("cannot score an empty episode")
    d = {name: np.asarray(value)[:length] for name, value in diagnostics.items()}
    finite = all(np.isfinite(v).all() for k, v in d.items() if k != "sensor_age_s")
    if "sensor_age_s" in d:
        age, valid = d["sensor_age_s"], d["sensor_valid"]
        finite = finite and bool(np.all(np.isfinite(age) | (~valid & np.isposinf(age))))
    crashed = bool(np.any(d["crashed"]))
    metrics = {
        "steps": length,
        "duration_s": length * dt,
        "finite": finite,
        "crashed": crashed,
        "completed": finite and not crashed,
    }
    for name, value in {
        "return": np.sum(d["reward"]),
        "position_rmse_m": np.sqrt(np.mean(np.sum(d["position_error_m"] ** 2, axis=-1))),
        "altitude_rmse_m": np.sqrt(np.mean(d["position_error_m"][..., 2] ** 2)),
        "airspeed_rmse_m_s": np.sqrt(np.mean(d["airspeed_error_m_s"] ** 2)),
        "heading_rmse_rad": np.sqrt(np.mean(d["heading_error_rad"] ** 2)),
        "mean_squared_action": np.mean(d["commanded_action"] ** 2),
        "saturation_fraction": np.mean(np.any(np.abs(d["commanded_action"]) >= 1 - 1e-6, axis=-1)),
    }.items():
        metrics[name] = float(value) if np.isfinite(value) else None
    return metrics


def _events(scenario):
    if scenario.faults is None:
        return []
    events = []
    for name, values in scenario.faults._asdict().items():
        if "time" in name:
            for i, time in enumerate(np.asarray(values).reshape(-1)):
                if np.isfinite(time):
                    events.append({"time_s": float(time), "label": f"{name}[{i}]"})
    return events


def _aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["policy"], row["split"]), []).append(row)
    result = []
    for (policy, split), group in sorted(groups.items()):
        means = {}
        for key in (
            "return",
            "position_rmse_m",
            "airspeed_rmse_m_s",
            "heading_rmse_rad",
            "saturation_fraction",
        ):
            values = [r[key] for r in group if r[key] is not None]
            means[key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "count": len(values),
            }
        result.append(
            {
                "policy": policy,
                "split": split,
                "episodes": len(group),
                "completion_rate": float(np.mean([r["completed"] for r in group])),
                "metrics": means,
            }
        )
    return result


def run_experiment(
    experiment: Experiment,
    policies: list[Policy],
    output: str | Path,
    *,
    split: str = "evaluation",
    reports: bool = True,
) -> dict[str, Any]:
    """Run the selected split with identical initial/weather seeds for every policy.

    Output must be absent or empty. Writes manifest, JSON/CSV scores, each finite flight,
    diagnostics and optional HTML inspectors. Compilation is per scenario/policy, seeds vmap.
    Aggregate standard deviations describe these episodes, not population confidence bounds.
    """
    if not policies or len({p.name for p in policies}) != len(policies):
        raise ValueError("policies must be nonempty with unique names")
    scenarios = [s for s in experiment.scenarios if s.split == split]
    if not scenarios:
        raise ValueError(f"no scenarios in split {split!r}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("experiment output must be empty; results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    experiment.save(output / "manifest.json")
    manifest_hash = experiment.sha256
    policy_records = [p.provenance() for p in policies]
    rows = []
    for scenario in scenarios:
        model = scenario.aircraft.to_model()
        reference = scenario.task.reference(model)
        for policy_spec in policies:
            policy, memory = policy_spec.factory(scenario, model, reference)
            trajectories, diagnostics = _simulate(scenario, model, reference, policy, memory)(
                jnp.asarray(scenario.seeds, dtype=jnp.uint32)
            )
            for i, seed in enumerate(scenario.seeds):
                d = jax.tree.map(lambda x, i=i: np.asarray(x[i]), diagnostics)
                metrics = episode_metrics(d, 1 / scenario.config.control_frequency_hz)
                trajectory = jax.tree.map(
                    lambda x, i=i, length=metrics["steps"]: x[i, :length], trajectories
                )
                finite_state = all(
                    np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(trajectory)
                )
                metrics["finite"] = metrics["finite"] and finite_state
                metrics["completed"] = metrics["completed"] and finite_state
                row = {
                    "scenario": scenario.name,
                    "policy": policy_spec.name,
                    "split": scenario.split,
                    "seed": seed,
                    **metrics,
                    "trajectory": None,
                }
                relative = Path("flights") / scenario.name / policy_spec.name / f"{seed}.npz"
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path = relative.with_suffix(".diagnostics.npz")
                np.savez_compressed(
                    output / diagnostics_path,
                    **{k: v[: metrics["steps"]] for k, v in d.items()},
                )
                row["diagnostics"] = diagnostics_path.as_posix()
                if metrics["finite"]:
                    applied = jnp.asarray(d["applied_action"][: metrics["steps"]])
                    controls = action_to_control(model, scenario.config, applied)
                    save_trajectory(
                        output / relative,
                        trajectory,
                        1 / scenario.config.control_frequency_hz,
                        t0_s=1 / scenario.config.control_frequency_hz,
                        controls=controls,
                        stamp=stamp(scenario.aircraft, model, seed=seed),
                        experiment_sha256=manifest_hash,
                        scenario=scenario.name,
                        policy=policy_spec.name,
                        events=_events(scenario),
                    )
                    row["trajectory"] = relative.as_posix()
                    if reports:
                        from cascade.viz.inspector import trajectory_report

                        trajectory_report(
                            output / relative, (output / relative).with_suffix(".html")
                        )
                rows.append(row)
    if experiment.sha256 != manifest_hash:
        raise RuntimeError("experiment inputs mutated during execution")
    result = {
        "schema": "cascade_results_v1",
        "experiment_sha256": manifest_hash,
        "provenance": stamp(),
        "policies": policy_records,
        "episodes": rows,
        "aggregate": _aggregate(rows),
    }
    result["results_sha256"] = content_hash(result)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result
