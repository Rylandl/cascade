"""Lightweight protocol tests; fabricated metric rows are not benchmark evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.experiments import Experiment
from cascade.experiments.manifest import content_hash

_SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_campaign.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_campaign", _SCRIPT)
campaign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campaign)


@pytest.fixture(scope="module")
def protocol():
    experiment = campaign.build_experiment("full")
    frozen = campaign.build_contract(experiment, campaign.campaign_policies(), "full")
    return experiment, frozen


def passing_result(experiment, frozen, split="evaluation"):
    rows = []
    for scenario in experiment.scenarios:
        if scenario.split != split:
            continue
        for policy in campaign.POLICY_NAMES:
            for seed in scenario.seeds:
                rows.append(
                    {
                        "scenario": scenario.name,
                        "policy": policy,
                        "seed": seed,
                        "split": split,
                        "steps": scenario.config.horizon_steps,
                        "duration_s": scenario.config.horizon_steps
                        / scenario.config.control_frequency_hz,
                        "finite": True,
                        "crashed": False,
                        "completed": True,
                        **dict.fromkeys(campaign.METRICS, 0.1),
                    }
                )
    return signed(
        {
            "schema": "cascade_results_v1",
            "experiment_sha256": experiment.sha256,
            "policies": copy.deepcopy(frozen["campaign"]["policies"]),
            "episodes": rows,
        }
    )


def signed(result):
    result["results_sha256"] = content_hash(
        {key: value for key, value in result.items() if key != "results_sha256"}
    )
    return result


def test_full_manifest_freezes_long_missions_disjoint_seeds_and_mixed_features(protocol, tmp_path):
    experiment, frozen = protocol
    path = experiment.save(tmp_path / "manifest.json")
    assert Experiment.load(path).sha256 == experiment.sha256
    assert len(experiment.scenarios) == 12
    by_split = {
        split: {seed for s in experiment.scenarios if s.split == split for seed in s.seeds}
        for split in ("training", "validation", "evaluation")
    }
    assert len(by_split["evaluation"]) == 8
    assert all(not by_split[a] & by_split[b] for a in by_split for b in by_split if a != b)
    for scenario in experiment.scenarios:
        assert scenario.config.horizon_steps / scenario.config.control_frequency_hz == 40
        if scenario.name.startswith("stress-clean"):
            assert scenario.noise is None and scenario.config.sensors is None
        else:
            assert scenario.noise is not None and scenario.config.sensors is not None
        if scenario.name.startswith("stress-mixed"):
            assert scenario.faults is not None and scenario.weather is not None
            assert scenario.config.sensors.rates.dropout_probability > 0
            assert scenario.config.sensors.position_error.delay_steps > 0
        assert set(frozen["campaign"]["criteria"][scenario.name]["policy_bounds"]) == set(
            campaign.POLICY_NAMES
        )


def test_exact_complete_results_pass(protocol):
    experiment, frozen = protocol
    acceptance = campaign.score_acceptance(
        experiment, frozen, passing_result(*protocol), "evaluation"
    )
    assert acceptance["accepted"] and acceptance["integrity_passed"]
    assert acceptance["supported_episodes"] == 32
    assert acceptance["stress_episodes"] == 32
    assert acceptance["supported_passed"] == 32


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "extra-seed", "wrong-split", "wrong-policy"]
)
def test_seed_and_policy_coverage_is_exact_not_a_pass_fraction(protocol, mutation):
    experiment, frozen = protocol
    result = passing_result(*protocol)
    if mutation == "missing":
        result["episodes"].pop()
    elif mutation == "duplicate":
        result["episodes"].append(copy.deepcopy(result["episodes"][0]))
    else:
        key, value = {
            "extra-seed": ("seed", 999),
            "wrong-split": ("split", "training"),
            "wrong-policy": ("policy", "trim"),
        }[mutation]
        result["episodes"][0][key] = value
    acceptance = campaign.score_acceptance(experiment, frozen, signed(result), "evaluation")
    assert not acceptance["accepted"] and not acceptance["integrity_passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("altitude_rmse_m", None),
        ("altitude_rmse_m", -1),
        ("heading_rmse_rad", "0.1"),
        ("airspeed_rmse_m_s", True),
        ("finite", "true"),
        ("completed", 1),
        ("crashed", None),
        ("steps", 1600.0),
        ("duration_s", 41.0),
        ("seed", True),
        ("saturation_fraction", 1.1),
    ],
)
def test_malformed_metrics_cannot_pass_after_rehashing(protocol, field, value):
    experiment, frozen = protocol
    result = passing_result(*protocol)
    result["episodes"][0][field] = value
    acceptance = campaign.score_acceptance(experiment, frozen, signed(result), "evaluation")
    assert not acceptance["accepted"] and not acceptance["integrity_passed"]


def test_missing_or_nan_metrics_and_bad_results_hash_fail_closed(protocol):
    experiment, frozen = protocol
    for mode in ("missing", "nan", "tampered"):
        result = passing_result(*protocol)
        if mode == "missing":
            del result["episodes"][0]["airspeed_rmse_m_s"]
            signed(result)
        else:
            result["episodes"][0]["airspeed_rmse_m_s"] = float("nan") if mode == "nan" else 0.2
        outcome = campaign.score_acceptance(experiment, frozen, result, "evaluation")
        assert not outcome["accepted"] and not outcome["integrity_passed"]


def test_one_nominal_failure_fails_without_dropping_it_from_summary(protocol):
    experiment, frozen = protocol
    result = passing_result(*protocol)
    result["episodes"][0]["heading_rmse_rad"] = 0.21
    acceptance = campaign.score_acceptance(experiment, frozen, signed(result), "evaluation")
    assert not acceptance["accepted"] and acceptance["integrity_passed"]
    assert acceptance["supported_passed"] == 31 and acceptance["supported_episodes"] == 32
    assert "heading_rmse_rad" in campaign.summary_text(acceptance)


def test_short_episode_and_crash_are_failures_even_with_small_tracking_error(protocol):
    experiment, frozen = protocol
    result = passing_result(*protocol)
    result["episodes"][0].update(steps=100, duration_s=2.5, completed=False, crashed=True)
    acceptance = campaign.score_acceptance(experiment, frozen, signed(result), "evaluation")
    assert not acceptance["accepted"]
    assert "crash" in acceptance["episodes"][0]["reasons"]


def test_stress_failure_is_retained_as_characterization_without_changing_nominal_gate(protocol):
    experiment, frozen = protocol
    result = passing_result(*protocol)
    stress = next(row for row in result["episodes"] if row["scenario"].startswith("stress"))
    stress.update(finite=False, completed=False, crashed=True, heading_rmse_rad=None)
    acceptance = campaign.score_acceptance(experiment, frozen, signed(result), "evaluation")
    assert acceptance["accepted"] and acceptance["integrity_passed"]
    assert acceptance["stress_passed"] == 31 and acceptance["stress_episodes"] == 32
    assert "stress-mixed" in campaign.summary_text(acceptance)


@pytest.mark.parametrize("mutation", ["policy", "contract", "role", "criteria", "bounds"])
def test_stale_provenance_or_incomplete_acceptance_contract_cannot_pass(protocol, mutation):
    experiment, frozen = protocol
    frozen = copy.deepcopy(frozen)
    result = passing_result(experiment, frozen)
    if mutation == "policy":
        result["policies"][0]["metadata"]["tuning"] = "changed"
        signed(result)
    elif mutation == "contract":
        frozen["campaign"]["profile"] = "changed without hash"
    else:
        rule = next(iter(frozen["campaign"]["criteria"].values()))
        if mutation == "role":
            rule["role"] = "ignore"
        elif mutation == "criteria":
            frozen["campaign"]["criteria"].pop(next(iter(frozen["campaign"]["criteria"])))
        else:
            rule["policy_bounds"][campaign.POLICY_NAMES[0]] = {}
        frozen["sha256"] = content_hash(frozen["campaign"])
    acceptance = campaign.score_acceptance(experiment, frozen, result, "evaluation")
    assert not acceptance["accepted"] and not acceptance["integrity_passed"]


def test_absent_or_empty_results_cannot_pass(protocol):
    experiment, frozen = protocol
    for result in (None, {}, [], {"episodes": []}):
        assert not campaign.score_acceptance(experiment, frozen, result, "evaluation")["accepted"]


def test_smoke_profile_is_explicitly_small_and_distinct_from_full(protocol):
    smoke = campaign.build_experiment("smoke")
    assert all(s.config.horizon_steps == 160 and len(s.seeds) == 1 for s in smoke.scenarios)
    assert smoke.sha256 != protocol[0].sha256
    smoke_seeds = {seed for scenario in smoke.scenarios for seed in scenario.seeds}
    full_seeds = {seed for scenario in protocol[0].scenarios for seed in scenario.seeds}
    assert not smoke_seeds & full_seeds
    with pytest.raises(ValueError, match="profile"):
        campaign.build_experiment("almost-full")


def test_artifact_index_rejects_missing_diagnostics_and_external_paths(tmp_path):
    result = {"episodes": [{"diagnostics": "missing.npz", "trajectory": None, "finite": False}]}
    with pytest.raises(ValueError, match="missing or external"):
        campaign._artifact_hashes(tmp_path, result)
    outside = tmp_path.parent / "outside.npz"
    outside.write_bytes(b"test")
    result["episodes"][0]["diagnostics"] = "../outside.npz"
    with pytest.raises(ValueError, match="missing or external"):
        campaign._artifact_hashes(tmp_path, result)


def make_artifacts(directory, result, *, wrong_seed=False):
    """Finite stand-in states test file integrity, not dynamics or metric accuracy."""
    directory.mkdir(parents=True, exist_ok=True)
    state = cascade.zero_state(cascade.aerobatic_reference())
    for i, row in enumerate(result["episodes"]):
        row["trajectory"] = f"flight-{i}.npz"
        row["diagnostics"] = f"diagnostics-{i}.npz"
        trajectory = jax.tree.map(
            lambda leaf, count=row["steps"]: jnp.broadcast_to(leaf, (count, *leaf.shape)), state
        )
        cascade.save_trajectory(
            directory / row["trajectory"],
            trajectory,
            0.025,
            stamp={"seed": row["seed"] + int(wrong_seed)},
            experiment_sha256=result["experiment_sha256"],
            scenario=row["scenario"],
            policy=row["policy"],
        )
        np.savez(directory / row["diagnostics"], commanded_action=np.zeros((row["steps"], 4)))
    return signed(result)


def test_trajectory_seed_identity_and_resolved_artifact_reuse_rejected(tmp_path):
    experiment = campaign.build_experiment("smoke")
    frozen = campaign.build_contract(experiment, campaign.campaign_policies(), "smoke")
    result = make_artifacts(tmp_path, passing_result(experiment, frozen), wrong_seed=True)
    with pytest.raises(ValueError, match="different seed"):
        campaign._artifact_hashes(tmp_path, result)
    result = make_artifacts(tmp_path, passing_result(experiment, frozen))
    campaign._artifact_hashes(tmp_path, result)
    alias = tmp_path / "alias.npz"
    alias.symlink_to(tmp_path / result["episodes"][0]["diagnostics"])
    result["episodes"][1]["diagnostics"] = alias.name
    with pytest.raises(ValueError, match="reused"):
        campaign._artifact_hashes(tmp_path, result)


@pytest.mark.parametrize("mutation", [None, "results", "diagnostics", "missing-diagnostics"])
def test_evaluation_preflight_reverifies_complete_pilot_evidence(tmp_path, mutation):
    experiment = campaign.build_experiment("smoke")
    frozen = campaign.build_contract(experiment, campaign.campaign_policies(), "smoke")
    directory = tmp_path / "training"
    result = make_artifacts(directory, passing_result(experiment, frozen, "training"))
    outcome = campaign.score_acceptance(experiment, frozen, result, "training")
    assert outcome["accepted"]
    outcome.update(
        artifact_sha256=campaign._artifact_hashes(directory, result),
        results_sha256=result["results_sha256"],
        sensor_ablation=[],
    )
    outcome["acceptance_sha256"] = content_hash(outcome)
    campaign._write_json(directory / "results.json", result)
    campaign._write_json(directory / "acceptance.json", outcome)
    if mutation == "results":
        result["episodes"][0]["altitude_rmse_m"] = 0.2
        campaign._write_json(directory / "results.json", result)
    elif mutation == "diagnostics":
        file = directory / result["episodes"][0]["diagnostics"]
        file.write_bytes(file.read_bytes() + b"altered bytes")
    elif mutation == "missing-diagnostics":
        (directory / result["episodes"][0]["diagnostics"]).unlink()
    if mutation:
        with pytest.raises(ValueError):
            campaign.verify_completed_split(experiment, frozen, tmp_path, "training")
    else:
        assert campaign.verify_completed_split(experiment, frozen, tmp_path, "training") == outcome
        assert json.loads((directory / "acceptance.json").read_text()) == outcome
