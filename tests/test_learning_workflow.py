import json

import numpy as np
import pytest

from cascade.learning.__main__ import main
from cascade.learning.checkpoint import load_checkpoint
from cascade.learning.workflow import (
    LearningRunConfig,
    _summary_rows,
    run_learning,
    tracking_experiment,
)


@pytest.mark.parametrize(
    "options",
    [
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"seeds": (True,)},
        {"seeds": (-1,)},
        {"architecture": "unknown"},
        {"task": "landing"},
        {"horizon_steps": 0},
        {"hidden_size": 2.5},
        {"evaluation_episodes": 10001},
        {"learning_rate": float("nan")},
    ],
)
def test_learning_run_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        LearningRunConfig(**options)


@pytest.mark.parametrize("task", ["tracking", "scheduled"])
def test_frozen_learning_splits_and_paired_stress(task, tmp_path):
    config = LearningRunConfig(task=task, evaluation_episodes=3)
    experiment = tracking_experiment(config)
    assert tracking_experiment(config).sha256 == experiment.sha256
    splits = {}
    for scenario in experiment.scenarios:
        splits.setdefault(scenario.split, set()).update(scenario.seeds)
        assert not scenario.config.observation.air_velocity
        assert not scenario.config.observation.surfaces
    assert not splits["training"] & splits["evaluation"]
    assert not splits["validation"] & (splits["training"] | splits["evaluation"])
    nominal, stress = experiment.scenarios[-2:]
    assert nominal.seeds == stress.seeds
    assert stress.config.sensors.position_error.dropout_probability > 0
    path = experiment.save(tmp_path / "manifest.json")
    assert type(experiment).load(path).sha256 == experiment.sha256


def test_cli_does_not_silently_override_resumed_training_configuration():
    with pytest.raises(SystemExit) as error:
        main(["absent", "--resume", "--batch-size", "2"])
    assert error.value.code == 2


def test_summary_separates_training_run_and_episode_variance():
    episodes = []
    for seed, returns in [(0, [1.0, 3.0]), (1, [3.0, 5.0])]:
        for value in returns:
            episodes.append(
                {
                    "scenario": "nominal",
                    "policy": f"learned-{seed}",
                    "completed": True,
                    "return": value,
                    "position_rmse_m": 1.0,
                    "airspeed_rmse_m_s": 1.0,
                    "heading_rmse_rad": 0.1,
                    "saturation_fraction": 0.0,
                }
            )
    rows, variance = _summary_rows({"episodes": episodes}, (0, 1))
    assert len(rows) == 2 and variance[0]["mean_return"] == 3.0
    np.testing.assert_allclose(variance[0]["training_run_std"], np.sqrt(2))
    assert all(row["return"]["count"] == 2 for row in rows)


@pytest.mark.slow
def test_train_resume_evaluate_workflow(tmp_path, monkeypatch):
    config = LearningRunConfig(
        seeds=(7,), horizon_steps=4, hidden_size=4, batch_size=1, evaluation_episodes=1
    )
    root = tmp_path / "learning"
    summary = run_learning(root, config, steps=1, checkpoint_interval=1)
    checkpoint = load_checkpoint(root / "seed-7/checkpoint.npz")
    assert int(checkpoint.state.iteration) == 1
    assert len(summary["training_curves"]["7"]) == 1
    assert set(summary["evaluations"]) == {"validation", "evaluation"}
    assert {
        row["scenario"] for row in summary["evaluations"]["evaluation"]["by_scenario_policy"]
    } == {"nominal", "sensor-stress"}
    for split in summary["evaluations"].values():
        results = json.loads((root / split["results"]).read_text())
        assert all(row["finite"] for row in results["episodes"])
        assert {policy["name"] for policy in results["policies"]} == {
            "trim",
            "observed-cascade",
            "learned-7",
        }
    resumed = run_learning(root, steps=0, resume=True)
    assert resumed["training_curves"] == summary["training_curves"]
    np.testing.assert_array_equal(
        load_checkpoint(root / "seed-7/checkpoint.npz").state.key, checkpoint.state.key
    )
    with pytest.raises(FileExistsError):
        run_learning(root, config, steps=0)
    with pytest.raises(ValueError, match="differs"):
        run_learning(root, LearningRunConfig(), steps=0, resume=True)
    assert (root / "summary.md").is_file()
    monkeypatch.setattr("cascade.learning.workflow._implementation_hash", lambda: "different")
    with pytest.raises(ValueError, match="package source"):
        run_learning(root, steps=0, resume=True)
