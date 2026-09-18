"""Freeze and score a synthetic simulator campaign; no measured-flight claim.

Full protocol: --phase pilot (training/validation), then --phase evaluation in the same
directory. Smoke CI: --profile smoke --phase all. Existing runs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

import cascade
from cascade.env import (
    EpisodeConfig,
    SensorBlockConfig,
    SensorPipelineConfig,
    fault_schedule,
    onboard_observation,
    scheduled_tracking_task,
    sensor_noise_from_sensors,
    waypoint_task,
    weather_condition,
)
from cascade.experiments import Experiment, Policy, Scenario, run_experiment
from cascade.experiments.manifest import content_hash

SCHEMA = "cascade_benchmark_campaign_v1"
POLICY_NAMES = ("privileged-cascade", "observation-cascade")
METRICS = (
    "position_rmse_m",
    "altitude_rmse_m",
    "airspeed_rmse_m_s",
    "heading_rmse_rad",
    "mean_squared_action",
    "saturation_fraction",
)


def build_experiment(profile="full"):
    """Declare all splits before any seeds run; the smoke profile is not release evidence."""
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be smoke or full")
    smoke = profile == "smoke"
    duration = 4.0 if smoke else 40.0
    seeds = {
        "training": (10011,) if smoke else (11, 12),
        "validation": (10051,) if smoke else (51, 52),
        "evaluation": (10101,) if smoke else tuple(range(201, 209)),
    }
    aircraft = cascade.aerobatic_reference_spec()
    schedule = scheduled_tracking_task(
        [0, 0.2 * duration, 0.4 * duration, 0.7 * duration, duration],
        [12, 12, 12.3 if smoke else 13, 12, 12],
        [50, 50, 50.3 if smoke else 52, 49.7 if smoke else 49, 50],
        [0, 0, 0.05 if smoke else 0.2, -0.03 if smoke else -0.15, 0],
    )
    nominal_sensors = SensorPipelineConfig(
        airspeed=SensorBlockConfig(sample_period_steps=2),
        position_error=SensorBlockConfig(sample_period_steps=4, delay_steps=1),
    )
    stress_sensors = SensorPipelineConfig(
        rates=SensorBlockConfig(delay_steps=1, dropout_probability=0.03, bias_walk_std=0.0002),
        airspeed=SensorBlockConfig(sample_period_steps=2, delay_steps=1, dropout_probability=0.05),
        position_error=SensorBlockConfig(
            sample_period_steps=4, delay_steps=2, dropout_probability=0.1
        ),
    )
    noise = sensor_noise_from_sensors(
        12.0,
        airspeed_std_m_s=0.1,
        gyro_std_rad_s=0.003,
        gyro_bias_std_rad_s=0.001,
        attitude_std_rad=0.003,
        heading_std_rad=0.005,
        position_std_m=0.1,
        accelerometer_std_m_s2=0.02,
    )

    def config(sensors, stress=False):
        return EpisodeConfig(
            horizon_steps=int(duration * 40),
            observation=onboard_observation(),
            sensors=sensors,
            reset_position_std_m=0.5,
            reset_velocity_std_m_s=0.2,
            reset_attitude_std_rad=0.02,
            reset_rate_std_rad_s=0.03,
            action_delay_steps=int(stress),
            isa_density=True,
        )

    families = [
        (
            "nominal-tracking",
            schedule,
            config(nominal_sensors),
            weather_condition(0.5, 1.0, turbulence_wind_20ft_m_s=0.5),
            None,
        ),
    ]
    if not smoke:
        route = waypoint_task(
            [0, 10, 20, 30, 40],
            [[0, 0, -50], [120, 0, -50], [240, 12, -52], [360, 12, -52], [480, 0, -50]],
            12.0,
            lookahead_s=3.0,
        )
        families.append(("nominal-route", route, config(nominal_sensors), None, None))
    families.append(
        (
            "stress-mixed",
            schedule,
            config(stress_sensors, True),
            weather_condition(
                2.0,
                1.0,
                turbulence_wind_20ft_m_s=2.0,
                gust_amplitude_m_s=1.0,
                gust_start_s=0.4 * duration,
                gust_duration_s=0.15 * duration,
            ),
            fault_schedule(aircraft.to_model(), partial_power={0: (0.55 * duration, 0.8)}),
        )
    )
    if not smoke:
        # Same state/weather/fault innovations, changing only delivered sensor quality.
        _, _, _, stress_weather, stress_faults = families[-1]
        families.append(
            ("stress-clean-sensors", schedule, config(None, True), stress_weather, stress_faults)
        )
    scenarios = tuple(
        Scenario(
            f"{family}-{split}",
            aircraft,
            task,
            settings,
            seeds[split],
            split,
            weather,
            None if family == "stress-clean-sensors" else noise,
            faults,
        )
        for split in seeds
        for family, task, settings, weather, faults in families
    )
    return Experiment(
        f"release-campaign-{profile}",
        scenarios,
        "Synthetic cruise missions: supported nominal tracking and separate mixed "
        "stress characterization. Fixed disjoint seeds; no flight-transfer claim.",
    )


def _tuned_controller(scenario, model, reference):
    from cascade.control import tune_cascade
    from cascade.env.tasks import task_at

    target = task_at(scenario.task, 0.0, reference.state.rigid_body)
    controller, _ = tune_cascade(
        scenario.aircraft,
        float(target.airspeed_m_s),
        model=model,
        altitude_m=float(target.altitude_m),
        environment=reference.environment,
        simulation_dt_s=scenario.config.simulation_dt_s,
    )
    return controller


def privileged_factory(scenario, model, reference):
    from cascade.env import cascade_policy

    return cascade_policy(
        _tuned_controller(scenario, model, reference),
        model,
        scenario.config,
        scenario.task,
        reference,
    )


def observation_factory(scenario, model, reference):
    from cascade.env import observation_cascade_policy

    return observation_cascade_policy(
        _tuned_controller(scenario, model, reference),
        model,
        scenario.config,
        scenario.task,
        reference,
    )


def campaign_policies():
    """Both use nominal autotuning; only the public baseline consumes delivered measurements."""
    return [
        Policy(
            POLICY_NAMES[0],
            privileged_factory,
            {
                "information": "true simulator state/wind, observation pipeline bypassed",
                "tuning": "initial nominal cruise; no evaluation tuning",
            },
        ),
        Policy(
            POLICY_NAMES[1],
            observation_factory,
            {
                "information": "delivered onboard observations and policy clock only",
                "tuning": "initial nominal cruise; no evaluation tuning",
            },
        ),
    ]


def package_source_hash():
    root = Path(inspect.getfile(cascade)).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_contract(experiment, policies, profile):
    """Predeclared per-episode bounds; stress outcomes are reported but not nominal gates."""
    criteria = {}
    for scenario in experiment.scenarios:
        role = "stress" if scenario.name.startswith("stress-") else "supported"
        bounds = {"altitude_rmse_m": 3.0, "airspeed_rmse_m_s": 2.0, "heading_rmse_rad": 0.2}
        if scenario.name.startswith("nominal-route"):
            bounds["position_rmse_m"] = 15.0
        criteria[scenario.name] = {
            "role": role,
            "policy_bounds": {policy.name: dict(bounds) for policy in policies},
            "require_finite": True,
            "require_no_crash": True,
            "require_full_horizon": True,
        }
    contract = {
        "schema": SCHEMA,
        "profile": profile,
        "experiment_sha256": experiment.sha256,
        "policies": [policy.provenance() for policy in policies],
        "package_source_sha256": package_source_hash(),
        "campaign_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "criteria": criteria,
        "acceptance": "all supported scenario-policy-seed episodes pass every bound; exact seed "
        "coverage and valid records are required for supported and stress cases",
        "stress_semantics": "same bounds shown for characterization; stress failures do not "
        "establish or revoke nominal regime acceptance",
        "evidence_kind": "synthetic simulator benchmark",
    }
    return {"sha256": content_hash(contract), "campaign": contract}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def score_acceptance(experiment, frozen, result, split):
    """Fail closed on malformed, stale, duplicated or incomplete experiment scores."""
    issues, rows = [], []
    contract = frozen.get("campaign", {}) if isinstance(frozen, dict) else {}
    try:
        if not isinstance(frozen, dict) or not isinstance(contract, dict):
            raise ValueError("campaign contract must be an object")
        if frozen.get("sha256") != content_hash(contract) or contract.get("schema") != SCHEMA:
            issues.append("campaign contract hash/schema mismatch")
        if contract.get("experiment_sha256") != experiment.sha256:
            issues.append("campaign experiment hash mismatch")
        if not isinstance(result, dict):
            raise ValueError("results must be an object")
        payload = {key: value for key, value in result.items() if key != "results_sha256"}
        if result.get("results_sha256") != content_hash(payload):
            issues.append("results content hash mismatch")
        if result.get("schema") != "cascade_results_v1":
            issues.append("unsupported results schema")
        if result.get("experiment_sha256") != experiment.sha256:
            issues.append("results experiment hash mismatch")
        if result.get("policies") != contract.get("policies"):
            issues.append("policy provenance differs from frozen contract")
        policies = [record["name"] for record in contract["policies"]]
        if policies != list(POLICY_NAMES):
            issues.append("campaign requires both declared policies in fixed order")
        if set(contract["criteria"]) != {s.name for s in experiment.scenarios}:
            issues.append("criteria must cover every frozen scenario exactly")
        for name, rule in contract["criteria"].items():
            if not isinstance(rule, dict):
                raise ValueError("scenario acceptance rule must be an object")
            if rule["role"] not in {"supported", "stress"} or any(
                rule.get(key) is not True
                for key in ("require_finite", "require_no_crash", "require_full_horizon")
            ):
                issues.append(f"invalid acceptance rule: {name}")
            if set(rule["policy_bounds"]) != set(policies):
                issues.append(f"policy bounds incomplete: {name}")
            for bounds in rule["policy_bounds"].values():
                if not isinstance(bounds, dict):
                    raise ValueError("metric bounds must be an object")
                if not {"altitude_rmse_m", "airspeed_rmse_m_s", "heading_rmse_rad"} <= set(bounds):
                    issues.append(f"tracking bounds incomplete: {name}")
                if any(
                    metric not in METRICS or not _number(limit) or limit <= 0
                    for metric, limit in bounds.items()
                ):
                    issues.append(f"invalid tracking bounds: {name}")
        scenarios = {s.name: s for s in experiment.scenarios if s.split == split}
        if not scenarios:
            issues.append("scoring split contains no scenarios")
        expected = {
            (s.name, policy, seed)
            for s in scenarios.values()
            for policy in policies
            for seed in s.seeds
        }
        seen = set()
        episodes = result.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("episodes must be a list")
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                issues.append(f"episode {index} is not an object")
                continue
            key = (episode.get("scenario"), episode.get("policy"), episode.get("seed"))
            if any(not isinstance(value, str) for value in key[:2]) or (
                not isinstance(key[2], int) or isinstance(key[2], bool)
            ):
                issues.append(f"episode {index} has malformed identity")
                continue
            if key not in expected or key in seen or episode.get("split") != split:
                issues.append(f"unexpected, duplicated or wrong-split episode {key}")
                continue
            seen.add(key)
            scenario = scenarios[key[0]]
            rule = contract["criteria"][scenario.name]
            reasons, malformed = [], []
            for field in ("finite", "crashed", "completed"):
                if type(episode.get(field)) is not bool:
                    malformed.append(f"{field} must be boolean")
            steps = episode.get("steps")
            if type(steps) is not int or not 1 <= steps <= scenario.config.horizon_steps:
                malformed.append("invalid step count")
            elif not _number(episode.get("duration_s")) or not math.isclose(
                episode["duration_s"],
                steps / scenario.config.control_frequency_hz,
                abs_tol=1e-7,
                rel_tol=1e-7,
            ):
                malformed.append("duration does not match step count")
            if not episode.get("finite"):
                reasons.append("nonfinite episode")
            if episode.get("crashed"):
                reasons.append("crash")
            if not episode.get("completed") or steps != scenario.config.horizon_steps:
                reasons.append("did not complete full horizon")
            if episode.get("completed") and (not episode.get("finite") or episode.get("crashed")):
                malformed.append("inconsistent completion flag")
            for metric in METRICS:
                value = episode.get(metric)
                if metric not in episode or (
                    value is not None and (not _number(value) or value < 0)
                ):
                    malformed.append(f"invalid or absent {metric}")
                elif value is None and episode.get("finite"):
                    malformed.append(f"finite episode lacks {metric}")
            if _number(episode.get("saturation_fraction")) and episode["saturation_fraction"] > 1:
                malformed.append("saturation fraction exceeds one")
            for metric, limit in rule["policy_bounds"][key[1]].items():
                value = episode.get(metric)
                if not _number(value) or value > limit:
                    reasons.append(f"{metric}={value!r} exceeds {limit}")
            if malformed:
                issues.append(f"{key}: " + "; ".join(malformed))
            rows.append(
                {
                    "scenario": key[0],
                    "policy": key[1],
                    "seed": key[2],
                    "role": rule["role"],
                    "passed": not reasons and not malformed,
                    "reasons": reasons + malformed,
                    "metrics": {metric: episode.get(metric) for metric in METRICS},
                    "finite": episode.get("finite"),
                    "crashed": episode.get("crashed"),
                    "completed": episode.get("completed"),
                    "steps": steps,
                }
            )
        missing = sorted(expected - seen)
        if missing:
            issues.append(f"missing {len(missing)} expected episode(s): {missing}")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        issues.append(f"malformed input: {error}")
    supported = [row for row in rows if row["role"] == "supported"]
    stress = [row for row in rows if row["role"] == "stress"]
    return {
        "schema": "cascade_campaign_acceptance_v1",
        "campaign_sha256": frozen.get("sha256") if isinstance(frozen, dict) else None,
        "split": split,
        "integrity_passed": not issues,
        "issues": issues,
        "accepted": not issues and bool(supported) and all(row["passed"] for row in supported),
        "supported_passed": sum(row["passed"] for row in supported),
        "supported_episodes": len(supported),
        "stress_passed": sum(row["passed"] for row in stress),
        "stress_episodes": len(stress),
        "episodes": rows,
    }


def summary_text(acceptance):
    lines = [
        f"{acceptance['split']}: nominal accepted={acceptance['accepted']}; "
        f"integrity={acceptance['integrity_passed']}",
        f"Supported episodes: {acceptance['supported_passed']}/{acceptance['supported_episodes']}",
        f"Stress characterization: {acceptance['stress_passed']}/{acceptance['stress_episodes']} "
        "meet the nominal tracking bounds (not an acceptance gate).",
    ]
    groups = {}
    for row in acceptance["episodes"]:
        groups.setdefault((row["scenario"], row["policy"]), []).append(row)
    for (scenario, policy), group in groups.items():
        maxima = []
        for metric in (
            "position_rmse_m",
            "altitude_rmse_m",
            "airspeed_rmse_m_s",
            "heading_rmse_rad",
        ):
            values = [row["metrics"][metric] for row in group if _number(row["metrics"][metric])]
            maxima.append(f"{metric}={max(values):.5g}" if values else f"{metric}=missing")
        lines.append(
            f"{scenario} {policy}: finite={sum(row['finite'] is True for row in group)}"
            f"/{len(group)}, crashed={sum(row['crashed'] is True for row in group)}, "
            f"completed={sum(row['completed'] is True for row in group)}/{len(group)}; "
            "worst " + ", ".join(maxima)
        )
    lines.extend(f"INTEGRITY: {issue}" for issue in acceptance["issues"])
    lines.extend(
        f"FAIL {row['scenario']} {row['policy']} seed={row['seed']}: " + "; ".join(row["reasons"])
        for row in acceptance["episodes"]
        if not row["passed"]
    )
    for policy in POLICY_NAMES:
        paired = [
            pair["commanded_action_rms_difference"]
            for pair in acceptance.get("sensor_ablation", [])
            if pair["policy"] == policy
        ]
        if paired:
            finite = [value for value in paired if _number(value)]
            lines.append(
                f"Sensor ablation {policy}: {len(finite)}/{len(paired)} finite pairs; "
                f"action RMS difference range={min(finite) if finite else None}"
                f"..{max(finite) if finite else None} (common prefix through termination)"
            )
    return "\n".join(lines) + "\n"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _artifact_hashes(output, result):
    hashes = {}
    resolved = set()
    for row in result["episodes"]:
        for field in ("diagnostics", "trajectory"):
            relative = row[field]
            if relative is None and field == "trajectory" and not row["finite"]:
                continue
            if not isinstance(relative, str):
                raise ValueError(f"episode lacks {field}")
            path = (output / relative).resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file():
                raise ValueError(f"missing or external artifact: {relative}")
            if path in resolved:
                raise ValueError(f"artifact reused across episode rows: {relative}")
            resolved.add(path)
            if field == "trajectory":
                trajectory, _, metadata = cascade.load_trajectory(path)
                if trajectory.rigid_body.position.shape[0] != row["steps"]:
                    raise ValueError("ground-truth trajectory length differs from scored horizon")
                if metadata.get("experiment_sha256") != result["experiment_sha256"]:
                    raise ValueError("ground-truth trajectory belongs to a different experiment")
                if (
                    metadata.get("scenario") != row["scenario"]
                    or metadata.get("policy") != row["policy"]
                ):
                    raise ValueError(
                        "ground-truth trajectory belongs to a different scenario/policy"
                    )
                if metadata.get("stamp", {}).get("seed") != row["seed"]:
                    raise ValueError("ground-truth trajectory belongs to a different seed")
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def sensor_ablation(output, result):
    """Retain paired action differences; no inference of improved control or statistical power."""
    episodes = {(row["scenario"], row["policy"], row["seed"]): row for row in result["episodes"]}
    pairs = []
    for (scenario, policy, seed), row in episodes.items():
        if not scenario.startswith("stress-mixed-"):
            continue
        other_name = scenario.replace("stress-mixed-", "stress-clean-sensors-", 1)
        other = episodes.get((other_name, policy, seed))
        if other is None:
            continue  # The bounded smoke profile intentionally omits this ablation.
        with (
            np.load(output / row["diagnostics"], allow_pickle=False) as a,
            np.load(output / other["diagnostics"], allow_pickle=False) as b,
        ):
            count = min(len(a["commanded_action"]), len(b["commanded_action"]))
            delta = a["commanded_action"][:count] - b["commanded_action"][:count]
        rms = float(np.sqrt(np.mean(delta**2)))
        pairs.append(
            {
                "policy": policy,
                "seed": seed,
                "paired_steps": count,
                "degraded_scenario": scenario,
                "clean_scenario": other_name,
                "commanded_action_rms_difference": rms if math.isfinite(rms) else None,
                "degraded_completed": row["completed"],
                "clean_completed": other["completed"],
            }
        )
    return pairs


def run_split(experiment, frozen, policies, output, split):
    destination = output / split
    if destination.exists():
        raise FileExistsError(f"refusing to rerun {split}: {destination}")
    try:
        result = run_experiment(experiment, policies, destination, split=split, reports=False)
        artifacts = _artifact_hashes(destination, result)
        acceptance = score_acceptance(experiment, frozen, result, split)
        acceptance["artifact_sha256"] = artifacts
        acceptance["results_sha256"] = result["results_sha256"]
        acceptance["sensor_ablation"] = sensor_ablation(destination, result)
    except Exception as error:
        # Preserve any emitted diagnostic files, and make interruption/failure unambiguously fail.
        destination.mkdir(parents=True, exist_ok=True)
        acceptance = score_acceptance(experiment, frozen, {}, split)
        acceptance["issues"].append(f"execution failed: {type(error).__name__}: {error}")
    acceptance["acceptance_sha256"] = content_hash(acceptance)
    _write_json(destination / "acceptance.json", acceptance)
    text = summary_text(acceptance)
    (destination / "summary.txt").write_text(text)
    print(text, flush=True)
    return acceptance


def verify_completed_split(experiment, frozen, output, split):
    """Recheck the complete pilot evidence chain immediately before consuming held-out seeds."""
    directory = output / split
    outcome = json.loads((directory / "acceptance.json").read_text())
    if not isinstance(outcome, dict) or outcome.get("acceptance_sha256") != content_hash(
        {key: value for key, value in outcome.items() if key != "acceptance_sha256"}
    ):
        raise ValueError(f"{split} acceptance hash mismatch")
    result = json.loads((directory / "results.json").read_text())
    rescored = score_acceptance(experiment, frozen, result, split)
    if (
        not rescored["accepted"]
        or any(outcome.get(key) != value for key, value in rescored.items())
        or outcome.get("results_sha256") != result.get("results_sha256")
    ):
        raise ValueError(f"{split} acceptance is stale, malformed or failed")
    if outcome.get("artifact_sha256") != _artifact_hashes(directory, result):
        raise ValueError(f"{split} artifact hash mismatch")
    if outcome.get("sensor_ablation") != sensor_ablation(directory, result):
        raise ValueError(f"{split} sensor ablation differs from retained diagnostics")
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--phase", choices=("pilot", "evaluation", "all"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "all" and args.profile != "smoke":
        parser.error("full campaigns require separate pilot and evaluation phases")
    output = args.output
    policies = campaign_policies()
    if args.phase in {"pilot", "all"}:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("campaign directory must be absent or empty")
        output.mkdir(parents=True, exist_ok=True)
        experiment = build_experiment(args.profile)
        experiment.save(output / "manifest.json")
        frozen = build_contract(experiment, policies, args.profile)
        _write_json(output / "campaign.json", frozen)
    else:
        experiment = Experiment.load(output / "manifest.json")
        frozen = json.loads((output / "campaign.json").read_text())
        current = build_contract(experiment, policies, args.profile)
        if frozen != current:
            raise ValueError(
                "frozen campaign, policy, package source or script differs; "
                "do not reuse held-out seeds after changing the protocol"
            )
        for pilot in ("training", "validation"):
            verify_completed_split(experiment, frozen, output, pilot)
    splits = (
        ("training", "validation")
        if args.phase == "pilot"
        else (("training", "validation", "evaluation") if args.phase == "all" else ("evaluation",))
    )
    outcomes = [run_split(experiment, frozen, policies, output, split) for split in splits]
    return 0 if all(outcome["accepted"] for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
