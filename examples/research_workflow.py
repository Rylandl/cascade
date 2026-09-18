"""A frozen matched-controller experiment, with offline flight inspection.

Run: python examples/research_workflow.py --output dist/research-workflow
Each output folder is immutable: choose a new folder to repeat a run.
"""

import argparse

from cascade import aerobatic_reference_spec
from cascade.env import (
    EpisodeConfig,
    SensorBlockConfig,
    SensorPipelineConfig,
    fault_schedule,
    scheduled_tracking_task,
    weather_condition,
)
from cascade.experiments import Experiment, Scenario, autotuned_policy, run_experiment, trim_policy
from cascade.viz import trajectory_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/research-workflow")
    args = parser.parse_args()
    aircraft = aerobatic_reference_spec()
    config = EpisodeConfig(
        horizon_steps=160,
        sensors=SensorPipelineConfig(
            rates=SensorBlockConfig(sample_period_steps=2, bias_walk_std=0.002),
            position_error=SensorBlockConfig(
                sample_period_steps=4, delay_steps=1, dropout_probability=0.05
            ),
        ),
    )
    task = scheduled_tracking_task(
        [0, 1, 3, 4], [12, 12, 13, 13], [50, 50, 52, 52], [0, 0, 0.15, 0.15]
    )
    experiment = Experiment(
        "maneuver-comparison",
        (
            Scenario("training", aircraft, task, config, seeds=(1, 2), split="training"),
            Scenario("calm", aircraft, task, config, seeds=(101, 102)),
            Scenario(
                "wind-and-derating",
                aircraft,
                task,
                config,
                seeds=(101, 102),
                weather=weather_condition(1.5, 1.0, turbulence_wind_20ft_m_s=2.0),
                faults=fault_schedule(aircraft.to_model(), partial_power={0: (2.5, 0.9)}),
            ),
        ),
        "Illustrative simulator comparison; no flight-transfer claim.",
    )
    result = run_experiment(experiment, [trim_policy(), autotuned_policy()], args.output)
    from pathlib import Path

    root = Path(args.output)
    comparison = {
        row["policy"]: row["trajectory"]
        for row in result["episodes"]
        if row["scenario"] == "calm" and row["seed"] == 101
    }
    if any(comparison.get(name) is None for name in ("trim", "cascade")):
        raise RuntimeError("comparison flight failed; inspect results.json and diagnostics")
    trajectory_report(
        root / comparison["trim"],
        root / "comparison.html",
        compare=root / comparison["cascade"],
        labels=("Trim hold", "Autotuned cascade"),
        title="Tracking a changing mission",
    )
    for summary in result["aggregate"]:
        print(
            summary["policy"],
            "completion",
            summary["completion_rate"],
            "mean return",
            summary["metrics"]["return"]["mean"],
        )
    print(f"Experiment {result['experiment_sha256']}\nReport: {root / 'comparison.html'}")


if __name__ == "__main__":
    main()
