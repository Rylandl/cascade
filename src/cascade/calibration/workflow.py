"""Fit frozen flight-pack fitting records and score explicitly held-out splits."""

from __future__ import annotations

from cascade.experiments.flight_data import evaluate_flight_pack, load_flight_pack
from cascade.experiments.manifest import content_hash

from .artifacts import (
    CalibrationArtifact,
    _empty_output,
    _validate_artifact,
    _write_json,
    load_calibration,
    save_calibration,
)
from .fitting import CalibrationConfig, fit_records


def fit_flight_pack(path, config: CalibrationConfig, *, output) -> CalibrationArtifact:
    """Fit only the pack's fitting arrays and atomically save a labelled result.

    The pack loader verifies every record's integrity, but validation/evaluation
    arrays are discarded before optimization. No model selection uses either
    held-out split. Budget exhaustion can produce a valid artifact with explicit
    ``optimizer.success=False``; numerical/invalid-candidate failures still raise.
    """
    _empty_output(output)
    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be a CalibrationConfig")
    nominal, records, metadata = load_flight_pack(path)
    fitting = tuple(record for split, record in records if split == "fitting")
    del records
    pack_hash = content_hash(metadata)
    result = fit_records(nominal, fitting, config)
    return save_calibration(output, result, pack=path, expected_pack_sha256=pack_hash)


def evaluate_calibration(artifact, pack, *, split="evaluation", output=None, substeps=None):
    """Compare the bound fit, nominal model and persistence without model selection.

    Scoring uses full-record replay, including any prefix excluded from fitting
    by ``warmup_steps``. Optimizer success/status remain explicit in the result;
    evaluating an unconverged fit does not declare it accepted or identified.
    """
    if not isinstance(artifact, CalibrationArtifact):
        artifact = load_calibration(artifact)
    _validate_artifact(artifact)
    nominal, records, metadata = load_flight_pack(pack)
    del nominal, records
    if content_hash(metadata) != artifact.calibration["pack_sha256"]:
        raise ValueError("calibration artifact belongs to a different flight pack")
    result = evaluate_flight_pack(
        pack,
        calibrated=artifact.spec,
        calibration=artifact.calibration,
        substeps=artifact.config.substeps if substeps is None else substeps,
        split=split,
    )
    result["calibration_artifact_sha256"] = artifact.sha256
    result["calibration_status"] = {
        "optimizer": artifact.report["optimizer"],
        "fitting_warmup_steps": artifact.config.warmup_steps,
        "evaluation_warmup_steps": 0,
        "scoring": "full-record replay, including any prefix excluded from fitting",
        "model_selection": "none; optimizer status is reported without acceptance promotion",
    }
    if output is not None:
        _write_json(output, result)
    return result


__all__ = ["evaluate_calibration", "fit_flight_pack"]
