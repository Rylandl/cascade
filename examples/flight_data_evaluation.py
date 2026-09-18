"""Build and replay a labeled synthetic evaluation pack; this is not flight validation.

The optional comparison model is the known simulation generator, not a model fitted to the
held-out recording. A real calibration pipeline should supply its independently fitted spec
and record the fitting IDs and parameter hash through the same interface.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cascade import (
    StraightFlightCondition,
    aerobatic_reference_spec,
    control_to_array,
    repeat_control,
    rollout,
    spec_hash,
    standard_environment,
    trim_straight_flight,
)
from cascade.canonical import rigid_body_to_canonical
from cascade.experiments import FlightRecord, create_flight_pack, evaluate_flight_pack
from cascade.experiments.manifest import content_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/synthetic-flight-pack")
    args = parser.parse_args()
    nominal = aerobatic_reference_spec()
    generator = replace(nominal, mass_kg=nominal.mass_kg * 1.08)
    model = generator.to_model()
    environment = standard_environment()
    trim = trim_straight_flight(model, StraightFlightCondition(12.0, altitude_m=50.0))
    assert trim.success
    dt, count = 0.025, 80

    def record(name, elevator):
        controls = repeat_control(trim.control, count)
        controls = controls._replace(channel=controls.channel.at[15:35, 1].add(elevator))
        subcontrols = jax.tree.map(lambda x: jnp.repeat(x, 10, axis=0), controls)
        _, states = jax.jit(rollout)(model, trim.state, subcontrols, environment, dt / 10)
        trajectory = jax.tree.map(
            lambda initial, x: jnp.concatenate([initial[None], x[9::10]], axis=0),
            trim.state,
            states,
        )
        commands = jnp.concatenate(
            [control_to_array(trim.control)[None], control_to_array(controls)], axis=0
        )
        return FlightRecord(
            name,
            name,
            np.arange(count + 1) * dt,
            np.asarray(rigid_body_to_canonical(trajectory.rigid_body)),
            np.asarray(commands),
            "synthetic",
        )

    fitting, heldout = record("fit-pitch-up", 0.04), record("eval-pitch-down", -0.03)
    root = Path(args.output)
    path = create_flight_pack(
        root,
        nominal,
        {"fitting": [fitting], "evaluation": [heldout]},
        license="MIT",
        source="examples/flight_data_evaluation.py",
        description="Generated simulator fixture; contains no measured flights.",
    )
    import json

    pack_hash = content_hash(json.loads(path.read_text())["pack"])
    result = evaluate_flight_pack(
        path,
        calibrated=generator,
        substeps=10,
        calibration={
            "pack_sha256": pack_hash,
            "calibrated_spec_sha256": spec_hash(generator),
            "fitting_records": [fitting.name],
            "method": "known synthetic generator, not fitted",
        },
        output=root / "replay-results.json",
    )
    rows = {r["model"]: r for r in result["scores"]}
    assert rows["calibrated"]["velocity_rmse_m_s"] < rows["nominal"]["velocity_rmse_m_s"]
    print("SYNTHETIC ONLY: velocity RMSE m/s")
    for name, row in rows.items():
        print(name, row["velocity_rmse_m_s"])
    print("Pack:", path)


if __name__ == "__main__":
    main()
