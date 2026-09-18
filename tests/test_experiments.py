import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade import aerobatic_reference_spec, load_trajectory
from cascade.env import (
    EpisodeConfig,
    SensorBlockConfig,
    SensorPipelineConfig,
    control_to_action,
    fault_schedule,
    observation_layout,
    orbit_task,
    scheduled_tracking_task,
    sensor_noise,
    tracking_task,
    waypoint_task,
    weather_condition,
)
from cascade.experiments import (
    Experiment,
    Policy,
    Scenario,
    episode_metrics,
    run_experiment,
    trim_policy,
)
from cascade.experiments.manifest import content_hash


def noisy_factory(scenario, model, reference):
    action = control_to_action(scenario.config, reference.control)
    rates = observation_layout(model, scenario.config.observation).rates

    def policy(memory, observation, state):
        # Echo sensed rate noise into a surface command: matched actions therefore test
        # matched sensor randomness, not just identical deterministic aircraft dynamics.
        return action.at[model.n_propellers].set(0.1 * jnp.tanh(observation[rates.start])), memory

    return policy, ()


def scenario(**kwargs):
    values = dict(
        name="calm",
        aircraft=aerobatic_reference_spec(),
        task=tracking_task(12.0, 50.0),
        config=EpisodeConfig(horizon_steps=6),
        seeds=(11, 12),
    )
    values.update(kwargs)
    return Scenario(**values)


def test_manifest_roundtrip_freezes_every_setting_and_rejects_tampering(tmp_path):
    spec = aerobatic_reference_spec()
    config = EpisodeConfig(
        horizon_steps=6,
        sensors=SensorPipelineConfig(
            rates=SensorBlockConfig(sample_period_steps=2, dropout_probability=0.2)
        ),
    )
    s = scenario(
        config=config,
        weather=weather_condition(2.0),
        noise=sensor_noise(rate_std=0.01),
        faults=fault_schedule(spec.to_model(), motor_out={0: 1.0}),
        task=scheduled_tracking_task([0, 1], [12, 14], [50, 55], [0, 0.2]),
    )
    experiment = Experiment("audit", (s,))
    path = experiment.save(tmp_path / "manifest.json")
    loaded = Experiment.load(path)
    assert loaded.sha256 == experiment.sha256
    assert loaded.scenarios[0].config.sensors.rates.sample_period_steps == 2
    np.testing.assert_array_equal(
        loaded.scenarios[0].faults.surface_jam_time_s, s.faults.surface_jam_time_s
    )
    data = json.loads(path.read_text())
    data["experiment"]["scenarios"][0]["name"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        Experiment.load(path)


def test_manifest_prevents_split_leakage_and_unsafe_output_names():
    with pytest.raises(ValueError, match="disjoint"):
        Experiment("leak", (scenario(name="train", split="training"), scenario(name="eval")))
    with pytest.raises(ValueError, match="safe"):
        scenario(name="../escape")
    with pytest.raises(ValueError, match="unique"):
        scenario(seeds=(1, 1))


def test_metrics_stop_at_first_done_and_allow_missing_sensor_age():
    d = dict(
        done=np.array([False, True, True]),
        crashed=np.array([False, True, False]),
        reward=np.array([0.5, 0, 999]),
        position_error_m=np.array([[3, 4, 0], [0, 0, 0], [9, 9, 9]]),
        heading_error_rad=np.zeros(3),
        airspeed_error_m_s=np.zeros(3),
        commanded_action=np.zeros((3, 4)),
        applied_action=np.zeros((3, 4)),
        sensor_age_s=np.full((3, 2), np.inf),
        sensor_valid=np.zeros((3, 2), bool),
    )
    m = episode_metrics(d, 0.1)
    assert m["steps"] == 2 and m["return"] == 0.5 and m["crashed"] and not m["completed"]
    assert m["finite"] and m["position_rmse_m"] == pytest.approx(np.sqrt(12.5))


def test_matched_policies_reproduce_flights_and_save_only_once(tmp_path):
    sensors = SensorPipelineConfig(rates=SensorBlockConfig(delay_steps=2, sample_period_steps=2))
    task = scheduled_tracking_task([0, 0.15], [12, 12.2], [50, 50.1], [0, 0.03])
    experiment = Experiment(
        "matched",
        (
            scenario(
                task=task,
                config=EpisodeConfig(horizon_steps=6, sensors=sensors, action_delay_steps=2),
                noise=sensor_noise(rate_std=0.05, rate_bias_std=0.1),
                weather=weather_condition(2.0, turbulence_wind_20ft_m_s=3.0),
            ),
        ),
    )
    first = Policy("noise-echo", noisy_factory)
    second = Policy("same-policy", first.factory, {"purpose": "paired reproducibility test"})
    result = run_experiment(experiment, [first, second], tmp_path / "results")
    rows = result["episodes"]
    assert len(rows) == 4 and all(r["finite"] for r in rows)
    for left, right in zip(rows[:2], rows[2:], strict=True):
        assert left["seed"] == right["seed"] and left["return"] == right["return"]
        a, ca, ma = load_trajectory(tmp_path / "results" / left["trajectory"])
        b, cb, mb = load_trajectory(tmp_path / "results" / right["trajectory"])
        for x, y in zip(jax.tree.leaves((a, ca)), jax.tree.leaves((b, cb)), strict=True):
            np.testing.assert_array_equal(x, y)
        assert ma["experiment_sha256"] == mb["experiment_sha256"] == experiment.sha256
        assert (tmp_path / "results" / left["trajectory"]).with_suffix(".html").is_file()
        with np.load(tmp_path / "results" / left["diagnostics"]) as diagnostic:
            assert not np.allclose(diagnostic["commanded_action"], diagnostic["applied_action"])
            np.testing.assert_allclose(
                control_to_action(experiment.scenarios[0].config, ca),
                diagnostic["applied_action"],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                diagnostic["applied_action"][2:],
                diagnostic["commanded_action"][:-2],
                atol=1e-6,
            )
    with pytest.raises(FileExistsError):
        run_experiment(experiment, [first], tmp_path / "results")


@pytest.mark.parametrize("kind", ["schedule", "noise", "fault", "weather", "dtype", "config"])
def test_resigned_manifest_still_rejects_invalid_inputs(tmp_path, kind):
    spec = aerobatic_reference_spec()
    s = scenario(
        task=scheduled_tracking_task([0, 1], [12, 14], 50),
        noise=sensor_noise(rate_std=0.1),
        faults=fault_schedule(spec.to_model()),
        weather=weather_condition(),
    )
    path = Experiment("invalid", (s,)).save(tmp_path / "invalid.json")
    wrapper = json.loads(path.read_text())
    data = wrapper["experiment"]["scenarios"][0]
    if kind == "schedule":
        data["task"]["fields"]["times_s"]["array"] = [0.0, 0.0]
    elif kind == "noise":
        data["noise"]["fields"]["rate_std"]["array"] = -1.0
    elif kind == "fault":
        data["faults"]["fields"]["propeller_power_fraction"]["array"] = [2.0]
    elif kind == "weather":
        data["weather"]["fields"]["roughness_length_m"]["array"] = 10.0
    elif kind == "dtype":
        data["task"]["fields"]["times_s"]["array"] = [0.0, 1e100]
    else:
        data["config"]["fields"]["horizon_steps"] = 0
    wrapper["sha256"] = content_hash(wrapper["experiment"])
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError):
        Experiment.load(path)


@pytest.mark.parametrize(
    "mission",
    [
        waypoint_task([0, 1], [[0, 0, -50], [12, 0, -50]], 12),
        orbit_task([0, 0, -50], 100, 12),
    ],
)
def test_restored_waypoint_and_orbit_run_under_jit_vmap(tmp_path, mission):
    experiment = Experiment(
        "restored", (scenario(task=mission, config=EpisodeConfig(horizon_steps=2)),)
    )
    path = experiment.save(tmp_path / "manifest.json")
    restored = Experiment.load(path)
    assert restored.sha256 == experiment.sha256
    result = run_experiment(restored, [trim_policy()], tmp_path / "results", reports=False)
    assert len(result["episodes"]) == 2
    assert all(row["finite"] and row["steps"] == 2 for row in result["episodes"])


def test_callable_factory_and_names_with_double_hyphens_do_not_collide(tmp_path):
    class Factory:
        def __call__(self, scenario, model, reference):
            return trim_policy().factory(scenario, model, reference)

    policies = [Policy("c", Factory()), Policy("b--c", Factory())]
    experiment = Experiment(
        "names",
        tuple(
            scenario(name=name, seeds=(1,), config=EpisodeConfig(horizon_steps=1))
            for name in ("a--b", "a")
        ),
    )
    result = run_experiment(experiment, policies, tmp_path / "results", reports=False)
    paths = [row["trajectory"] for row in result["episodes"]]
    assert len(set(paths)) == len(paths) == 4
    assert all((tmp_path / "results" / path).exists() for path in paths)
    assert all("Factory" in record["factory"] for record in result["policies"])


def test_nonfinite_policy_retains_failure_diagnostics(tmp_path):
    def factory(scenario, model, reference):
        return lambda memory, obs, state: (jnp.full(4, jnp.nan), memory), ()

    experiment = Experiment(
        "failed", (scenario(seeds=(1,), config=EpisodeConfig(horizon_steps=2)),)
    )
    result = run_experiment(
        experiment, [Policy("nonfinite", factory)], tmp_path / "results", reports=False
    )
    row = result["episodes"][0]
    assert not row["finite"] and not row["completed"] and row["trajectory"] is None
    # Crash reward can legitimately be zero even when the command is nonfinite;
    # the undefined action metric must remain missing rather than serialized as NaN.
    assert row["mean_squared_action"] is None
    with np.load(tmp_path / "results" / row["diagnostics"]) as diagnostic:
        assert np.isnan(diagnostic["commanded_action"]).all()


def test_cli_help_and_unknown_policy_errors():
    result = subprocess.run(
        [sys.executable, "-m", "cascade.experiments", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "--policies" in result.stdout and "--split" in result.stdout
    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "cascade.experiments",
            "absent.json",
            "out",
            "--policies",
            "unknown",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2 and "unique names from trim,cascade" in invalid.stderr


def test_cli_runs_selected_split_and_policy(tmp_path, capsys):
    from cascade.experiments.__main__ import main

    experiment = Experiment(
        "cli",
        (
            scenario(
                name="train", split="training", seeds=(1,), config=EpisodeConfig(horizon_steps=1)
            ),
            scenario(name="eval", seeds=(2,), config=EpisodeConfig(horizon_steps=1)),
        ),
    )
    manifest = experiment.save(tmp_path / "manifest.json")
    result = main(
        [
            str(manifest),
            str(tmp_path / "results"),
            "--policies",
            "trim",
            "--split",
            "training",
            "--no-reports",
        ]
    )
    assert len(result["episodes"]) == 1
    assert result["episodes"][0]["split"] == "training"
    assert result["episodes"][0]["policy"] == "trim"
    assert "Scored 1 episodes" in capsys.readouterr().out


def test_metrics_reject_invalid_time_and_misaligned_arrays():
    with pytest.raises(ValueError, match="timestep"):
        episode_metrics({"done": np.array([False])}, 0.0)
    with pytest.raises(ValueError, match="aligned"):
        episode_metrics({"done": np.array([False]), "reward": np.zeros(2)}, 0.1)
