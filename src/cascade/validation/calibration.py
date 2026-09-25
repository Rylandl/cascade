"""Fit Cascade to independently propagated, explicitly synthetic JSBSim recordings."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from cascade.analysis.trim import StraightFlightCondition, trim_straight_flight
from cascade.calibration import (
    CalibrationConfig,
    FitParameter,
    evaluate_calibration,
    fit_flight_pack,
)
from cascade.experiments import FlightRecord, create_flight_pack
from cascade.spec import save_aircraft_spec
from cascade.state import ControlInput

from .campaign import PROTOCOL, jsbsim_trajectory, json_write

CALIBRATION_PROTOCOL = {
    "generator_mass_scale": 1.08,
    "generator_inertia_scale": 1.12,
    "parameter_relative_error_limit": 0.02,
    "heldout_velocity_error_ratio_limit": 0.1,
    "recordings": {
        "fitting": [[0.035, 0.03, 0.04], [-0.025, -0.035, -0.03]],
        "validation": [[0.02, -0.02, 0.025]],
        "evaluation": [[-0.03, 0.025, 0.03], [0.025, -0.03, -0.025]],
    },
    "duration_s": 2.0,
    "sample_period_s": 0.02,
    "reference_dt_s": 0.000625,
    "integration_substeps": 8,
    "max_nfev": 40,
}


def canonical(states):
    """Independent NED/FRD -> NWU/FLU adapter for the documented flight-pack layout."""
    flip = np.diag([1.0, -1.0, -1.0])
    result = states.copy()
    result[:, :3] = states[:, :3] @ flip
    result[:, 3:6] = states[:, 3:6] @ flip
    rotation = Rotation.from_matrix(flip @ Rotation.from_quat(states[:, 6:10]).as_matrix() @ flip)
    xyzw = rotation.as_quat()
    result[:, 6:10] = xyzw[:, [3, 0, 1, 2]]
    result[:, 10:] = states[:, 10:] @ flip
    return result


def run_calibration(spec, speed, directory):
    """Freeze generator/splits before fitting; only fit records enter the optimizer."""
    directory.mkdir()
    json_write(directory / "protocol.json", CALIBRATION_PROTOCOL)
    truth = {
        "mass_kg": spec.mass_kg * CALIBRATION_PROTOCOL["generator_mass_scale"],
        "inertia_scale": CALIBRATION_PROTOCOL["generator_inertia_scale"],
    }
    generator = replace(
        spec,
        mass_kg=truth["mass_kg"],
        inertia_kg_m2=tuple(
            tuple(x * truth["inertia_scale"] for x in row) for row in spec.inertia_kg_m2
        ),
    )
    save_aircraft_spec(generator, directory / "generator.toml")
    nominal = spec.to_model()
    trim = trim_straight_flight(nominal, StraightFlightCondition(speed, altitude_m=1000))
    if not trim.success:
        raise RuntimeError("calibration recording initial condition did not trim")
    dt = CALIBRATION_PROTOCOL["reference_dt_s"]
    sample_period = CALIBRATION_PROTOCOL["sample_period_s"]
    if sample_period != PROTOCOL["sample_period_s"]:
        raise ValueError("recording sampling must match the trajectory sampler")
    samples = round(CALIBRATION_PROTOCOL["duration_s"] / sample_period)
    stride = round(sample_period / dt)
    groups = {}
    for split, excitations in CALIBRATION_PROTOCOL["recordings"].items():
        groups[split] = []
        for index, (elevator, aileron, throttle) in enumerate(excitations):
            name = f"{split}-{index}"
            propeller = np.broadcast_to(
                trim.control.propeller, (samples, nominal.n_propellers)
            ).copy()
            channel = np.broadcast_to(
                trim.control.channel, (samples, nominal.n_control_channels)
            ).copy()
            channel[samples // 5 : 2 * samples // 5, 1] += elevator
            channel[2 * samples // 5 : 3 * samples // 5, 0] += aileron
            propeller[3 * samples // 5 :] += throttle
            controls = ControlInput(
                jnp.asarray(np.repeat(propeller, stride, axis=0)),
                jnp.asarray(np.repeat(channel, stride, axis=0)),
            )
            states = jsbsim_trajectory(generator, trim.state, controls, dt, directory / name)
            command = np.concatenate((propeller, channel), axis=1)
            command = np.concatenate((command[:1], command), axis=0)
            record = FlightRecord(
                name,
                name,
                np.arange(samples + 1) * sample_period,
                canonical(states),
                command,
                kind="synthetic",
            )
            groups[split].append(record)
    manifest = create_flight_pack(
        directory / "pack",
        spec,
        groups,
        license="MIT",
        source="Native JSBSim 1.3.1 XML equations and propagation; cascade.validation.calibration",
        description=(
            "Two fitting, one validation, two evaluation maneuvers from a declared mass/inertia "
            "perturbation. Synthetic shared-model verification only."
        ),
    )
    config = CalibrationConfig(
        parameters=(
            FitParameter("mass_kg", spec.mass_kg * 0.8, spec.mass_kg * 1.25),
            FitParameter("inertia_scale", 0.75, 1.3),
        ),
        substeps=CALIBRATION_PROTOCOL["integration_substeps"],
        max_nfev=CALIBRATION_PROTOCOL["max_nfev"],
    )
    json_write(directory / "config.json", config.to_dict())
    artifact = fit_flight_pack(manifest, config, output=directory / "fit")
    evaluate_calibration(
        artifact, manifest, split="validation", output=directory / "validation.json"
    )
    evaluation = evaluate_calibration(
        artifact, manifest, split="evaluation", output=directory / "evaluation.json"
    )
    estimated = {
        "mass_kg": artifact.spec.mass_kg,
        "inertia_scale": artifact.spec.inertia_kg_m2[0][0] / spec.inertia_kg_m2[0][0],
    }
    errors = {k: abs(estimated[k] / truth[k] - 1) for k in truth}
    nominal_scores = {
        x["record"]: x["velocity_rmse_m_s"] for x in evaluation["scores"] if x["model"] == "nominal"
    }
    ratios = {
        x["record"]: x["velocity_rmse_m_s"] / nominal_scores[x["record"]]
        for x in evaluation["scores"]
        if x["model"] == "calibrated"
    }
    result = {
        "truth": truth,
        "estimated": estimated,
        "relative_errors": errors,
        "heldout_velocity_error_ratios": ratios,
        "optimizer": artifact.report["optimizer"],
        "passed": bool(
            artifact.report["optimizer"]["success"]
            and max(errors.values()) <= CALIBRATION_PROTOCOL["parameter_relative_error_limit"]
            and max(ratios.values()) <= CALIBRATION_PROTOCOL["heldout_velocity_error_ratio_limit"]
        ),
    }
    json_write(directory / "recovery.json", result)
    return result
