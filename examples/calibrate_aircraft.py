"""Estimate mass and inertia from a synthetic flight pack, then score held-out maneuvers.

This demonstrates parameter recovery in the simulator. It contains no measured flights
and makes no claim about real-aircraft identification or predictive accuracy.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.calibration import (
    CalibrationConfig,
    FitParameter,
    evaluate_calibration,
    fit_flight_pack,
)
from cascade.canonical import rigid_body_to_canonical
from cascade.experiments import FlightRecord, create_flight_pack


def synthetic_pack(output: Path, *, steps: int = 80, substeps: int = 4):
    """Freeze distinct maneuvers; return the manifest and known generating parameters."""
    nominal = cascade.aerobatic_reference_spec()
    truth = {"mass_kg": nominal.mass_kg * 1.08, "inertia_scale": 1.12}
    generator = replace(
        nominal,
        mass_kg=truth["mass_kg"],
        inertia_kg_m2=tuple(
            tuple(value * truth["inertia_scale"] for value in row) for row in nominal.inertia_kg_m2
        ),
    )
    model = generator.to_model()
    environment = cascade.standard_environment()
    trim = cascade.trim_straight_flight(
        model, cascade.StraightFlightCondition(12.0, altitude_m=50.0)
    )
    if not trim.success:
        raise RuntimeError("synthetic fixture trim did not converge")
    dt = 0.025

    def record(name, elevator, aileron, throttle):
        controls = cascade.repeat_control(trim.control, steps)
        first, second, third = steps // 5, 2 * steps // 5, 3 * steps // 5
        channel = controls.channel.at[first:second, 1].add(elevator)
        channel = channel.at[second:third, 0].add(aileron)
        propeller = controls.propeller.at[third:, 0].add(throttle)
        controls = controls._replace(channel=channel, propeller=propeller)
        held = jax.tree.map(lambda value: jnp.repeat(value, substeps, axis=0), controls)
        _, states = jax.jit(cascade.rollout)(model, trim.state, held, environment, dt / substeps)
        canonical = rigid_body_to_canonical(states.rigid_body)[substeps - 1 :: substeps]
        initial = rigid_body_to_canonical(trim.state.rigid_body)
        observed = jnp.concatenate([initial[None], canonical], axis=0)
        commands = jnp.concatenate(
            [cascade.control_to_array(trim.control)[None], cascade.control_to_array(controls)],
            axis=0,
        )
        return FlightRecord(
            name,
            name,
            np.arange(steps + 1) * dt,
            np.asarray(observed),
            np.asarray(commands),
            kind="synthetic",
        )

    manifest = create_flight_pack(
        output,
        nominal,
        {
            "fitting": [
                record("fit-pitch-roll", 0.035, 0.035, 0.035),
                record("fit-opposite", -0.025, -0.025, -0.025),
            ],
            "validation": [record("validation-mixed", 0.020, -0.030, 0.020)],
            "evaluation": [
                record("evaluation-recovery", -0.030, 0.025, 0.025),
                record("evaluation-cross-axis", 0.030, -0.020, -0.020),
            ],
        },
        license="MIT",
        source="examples/calibrate_aircraft.py: synthetic simulator recordings",
        description=(
            "Known mass/inertia perturbation. Two fitting, one validation, and two evaluation "
            "maneuvers; no measured-flight evidence."
        ),
    )
    return manifest, nominal, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/synthetic-calibration"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max-nfev", type=int, default=40)
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("--steps must be at least 10 to preserve the excitation sequence")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("example output directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    pack, nominal, truth = synthetic_pack(args.output / "pack", steps=args.steps)
    config = CalibrationConfig(
        parameters=(
            FitParameter("mass_kg", nominal.mass_kg * 0.8, nominal.mass_kg * 1.25),
            FitParameter("inertia_scale", 0.75, 1.3),
        ),
        substeps=4,
        max_nfev=args.max_nfev,
    )
    (args.output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    artifact = fit_flight_pack(pack, config, output=args.output / "calibration")
    if not artifact.report["optimizer"]["success"]:
        raise RuntimeError("synthetic fit did not converge; inspect the saved calibration report")
    evaluate_calibration(artifact, pack, split="validation", output=args.output / "validation.json")
    result = evaluate_calibration(
        artifact, pack, split="evaluation", output=args.output / "evaluation.json"
    )
    estimates = {
        "mass_kg": artifact.spec.mass_kg,
        "inertia_scale": artifact.spec.inertia_kg_m2[0][0] / nominal.inertia_kg_m2[0][0],
    }
    recovery = {
        "evidence_kind": "synthetic",
        "truth": truth,
        "estimated": estimates,
        "relative_error": {name: abs(estimates[name] / truth[name] - 1) for name in truth},
    }
    (args.output / "recovery.json").write_text(
        json.dumps(recovery, indent=2, allow_nan=False) + "\n"
    )
    if any(error > 0.02 for error in recovery["relative_error"].values()):
        raise RuntimeError("synthetic parameters were not recovered within two percent")
    nominal_scores = {r["record"]: r for r in result["scores"] if r["model"] == "nominal"}
    if any(
        r["velocity_rmse_m_s"] >= 0.1 * nominal_scores[r["record"]]["velocity_rmse_m_s"]
        for r in result["scores"]
        if r["model"] == "calibrated"
    ):
        raise RuntimeError("synthetic held-out velocity error did not improve tenfold")
    print("SYNTHETIC ONLY: estimated parameters from fitting maneuvers")
    for name in truth:
        print(f"  {name}: fitted {estimates[name]:.6g}; generator {truth[name]:.6g}")
    print("Held-out velocity RMSE (m/s):")
    for row in result["scores"]:
        print(f"  {row['record']} / {row['model']}: {row['velocity_rmse_m_s']:.6g}")
    print("Calibration:", artifact.path)


if __name__ == "__main__":
    main()
