"""Versioned, auditable calibration artifacts with atomic directory publication.

The artifact contains an ordinary aircraft TOML file and finite JSON metadata. It
does not contain pickle or recording arrays. Hashes detect accidental changes;
they are not signatures or evidence that a model is physically identified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cascade.experiments.flight_data import PACK_SCHEMA, load_flight_pack
from cascade.experiments.manifest import content_hash
from cascade.provenance import spec_hash
from cascade.spec import AircraftSpec, load_aircraft_spec, save_aircraft_spec

from .fitting import CalibrationConfig, FitResult
from .parameters import Parameterization

CALIBRATION_SCHEMA = "cascade_calibration_artifact_v1"


def _json(value):
    """Validate finiteness and take an independent, canonical JSON snapshot."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError("calibration metadata must contain finite JSON-compatible values") from exc


def _empty_output(output):
    output = Path(output)
    if output.is_symlink():
        raise ValueError("calibration output must not be a symbolic link")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("calibration output must be an absent or empty directory")
    return output


def _write_json(path, value):
    """Atomically write an evaluation report; artifacts publish their whole directory."""
    path = Path(path)
    value = _json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CalibrationArtifact:
    """A validated fitted specification, replay contract, and solver diagnostics.

    ``path`` is the artifact directory. ``calibration`` is directly accepted by
    :func:`cascade.experiments.evaluate_flight_pack`. A valid artifact may contain
    an unconverged solver result; inspect ``report['optimizer']['success']``.
    """

    path: Path
    spec: AircraftSpec
    config: CalibrationConfig
    calibration: dict[str, Any]
    report: dict[str, Any]
    metadata: dict[str, Any]
    sha256: str

    @property
    def manifest_path(self):
        return self.path / "metadata.json"


def _check_digest(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 hex digest")


def _selected_records(records, pack):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("calibration requires nonempty selected fitting records")
    fitting = {entry["name"]: entry for entry in pack["records"] if entry["split"] == "fitting"}
    names = []
    for record in records:
        if not isinstance(record, dict) or record.get("name") not in fitting:
            raise ValueError("calibration record must belong to the pack's fitting split")
        name = record["name"]
        if name in names:
            raise ValueError("selected fitting records must be unique")
        for field in ("maneuver_id", "kind", "content_sha256"):
            if record.get(field) != fitting[name][field]:
                raise ValueError(f"selected fitting record {name} has inconsistent {field}")
        names.append(name)
    return names


def _validate_report(report, config, nominal, fitted):
    if not isinstance(report, dict) or report.get("schema") != "cascade_calibration_fit_v1":
        raise ValueError("unsupported calibration fit report schema")
    if (
        report["nominal_spec_sha256"] != spec_hash(nominal)
        or report["calibrated_spec_sha256"] != spec_hash(fitted)
        or report["configuration"] != config.to_dict()
    ):
        raise ValueError("fit report specification or configuration binding is inconsistent")
    optimizer = report["optimizer"]
    if not isinstance(optimizer, dict) or type(optimizer.get("success")) is not bool:
        raise ValueError("optimizer success must be explicitly boolean")
    status = optimizer.get("status")
    if (
        type(status) is not int
        or status not in range(-1, 5)
        or optimizer["success"] != (status > 0)
    ):
        raise ValueError("optimizer success and status are inconsistent")
    if optimizer.get("method") != "scipy.optimize.least_squares":
        raise ValueError("unsupported optimizer method")
    if not isinstance(optimizer.get("message"), str) or not optimizer["message"].strip():
        raise ValueError("optimizer message must describe the solver outcome")
    for name in ("nfev", "njev"):
        value = optimizer.get(name)
        if name == "njev" and value is None:
            continue
        if type(value) is not int or not 1 <= value <= config.max_nfev:
            raise ValueError(f"optimizer {name} must be within the configured evaluation budget")
    tolerances = optimizer.get("tolerances")
    if not isinstance(tolerances, dict) or set(tolerances) != {"ftol", "xtol", "gtol"}:
        raise ValueError("optimizer tolerances must declare ftol, xtol and gtol")
    for name, value in [
        *((name, report[name]) for name in ("initial_data_loss", "final_data_loss")),
        ("optimality", optimizer.get("optimality")),
        *tolerances.items(),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")
        if name in tolerances and value <= 0:
            raise ValueError("optimizer tolerances must be positive")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("fit report requires selected fitting record metadata")
    for record in records:
        if (
            not isinstance(record, dict)
            or type(record.get("samples")) is not int
            or type(record.get("scored_samples")) is not int
            or record["samples"] < 2
            or record["scored_samples"] != record["samples"] - 1 - config.warmup_steps
            or record["scored_samples"] < 1
        ):
            raise ValueError("fit report record sample counts are inconsistent with warmup")
    parameters = Parameterization(nominal, config.parameters)
    rows = report["parameters"]
    if not isinstance(rows, list) or len(rows) != parameters.size:
        raise ValueError("fit report parameter inventory does not match configuration")
    values = []
    for index, (parameter, row) in enumerate(zip(parameters.parameters, rows, strict=True)):
        if (
            not isinstance(row, dict)
            or row.get("path") != parameter.path
            or row.get("lower") != parameter.lower
            or row.get("upper") != parameter.upper
        ):
            raise ValueError("fit report parameter paths or bounds differ from configuration")
        initial = float(np.asarray(parameters.initial_values)[index])
        for name in ("initial", "value", "normalized_value"):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
            ):
                raise ValueError(f"reported parameter {name} must be a finite number")
        if not np.isclose(row["initial"], initial, rtol=1e-6, atol=0.0):
            raise ValueError("fit report initial parameter differs from nominal specification")
        normalized = row["normalized_value"]
        if not 0 <= normalized <= 1 or not parameter.lower <= row["value"] <= parameter.upper:
            raise ValueError("reported fitted parameter is outside its bounds")
        expected = (row["value"] - parameter.lower) / (parameter.upper - parameter.lower)
        if not np.isclose(normalized, expected, rtol=1e-6, atol=1e-7):
            raise ValueError("reported normalized parameter coordinate is inconsistent")
        bound = "lower" if normalized <= 1e-6 else "upper" if normalized >= 1 - 1e-6 else None
        if row.get("bound_hit") != bound:
            raise ValueError("reported parameter bound-hit flag is inconsistent")
        values.append(row["value"])
    reconstructed = parameters.apply_spec(np.asarray(values, dtype=np.float64))
    if spec_hash(reconstructed) != spec_hash(fitted):
        raise ValueError("calibrated specification differs from the declared fitted parameters")
    diagnostics = report["identifiability"]
    if not isinstance(diagnostics, dict):
        raise ValueError("identifiability diagnostics must be an object")
    residual_count = 12 * sum(record["scored_samples"] for record in records)
    singular = np.asarray(diagnostics["singular_values"])
    if (
        singular.shape != (min(residual_count, parameters.size),)
        or singular.dtype.kind not in "iuf"
        or not np.isfinite(singular).all()
        or np.any(singular < 0)
        or np.any(singular[:-1] < singular[1:])
    ):
        raise ValueError(
            "diagnostic singular values must have the correct size and descending order"
        )
    rank, tolerance = diagnostics["rank"], diagnostics["rank_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (float, int))
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("diagnostic rank tolerance must be finite and nonnegative")
    if (
        type(rank) is not int
        or rank != int(np.count_nonzero(singular > tolerance))
        or type(diagnostics.get("parameter_count")) is not int
        or diagnostics["parameter_count"] != parameters.size
    ):
        raise ValueError("diagnostic rank or parameter count is inconsistent")
    condition = diagnostics.get("condition")
    if rank < parameters.size:
        if condition is not None:
            raise ValueError("rank-deficient diagnostic condition must be null")
    else:
        if (
            isinstance(condition, bool)
            or not isinstance(condition, (float, int))
            or not np.isfinite(condition)
            or condition < 1
            or not np.isclose(condition, singular[0] / singular[-1], rtol=1e-6)
        ):
            raise ValueError("diagnostic condition is inconsistent with singular values")
    if diagnostics.get("coordinates") != "unit-interval parameters; scaled data residuals only":
        raise ValueError("unsupported diagnostic parameter coordinates")


def _validate_metadata(metadata, fitted):
    """Check semantic bindings in addition to the enclosing metadata checksum."""
    _json(metadata)
    expected = {
        "schema",
        "pack",
        "pack_sha256",
        "nominal_spec",
        "nominal_spec_sha256",
        "calibrated_spec",
        "configuration",
        "configuration_sha256",
        "fitting_records",
        "calibration",
        "report",
        "provenance",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != expected
        or metadata.get("schema") != CALIBRATION_SCHEMA
    ):
        raise ValueError("unsupported or malformed calibration artifact schema")
    pack = metadata["pack"]
    if not isinstance(pack, dict) or pack.get("schema") != PACK_SCHEMA:
        raise ValueError("unsupported bound flight pack schema")
    if metadata["pack_sha256"] != content_hash(pack):
        raise ValueError("bound flight pack hash mismatch")
    nominal = AircraftSpec.from_dict(metadata["nominal_spec"])
    nominal_hash = spec_hash(nominal)
    if nominal_hash != metadata["nominal_spec_sha256"] or nominal_hash != pack["aircraft_sha256"]:
        raise ValueError("nominal specification does not match the bound flight pack")
    config = CalibrationConfig.from_dict(metadata["configuration"])
    if metadata["configuration_sha256"] != content_hash(config.to_dict()):
        raise ValueError("calibration configuration hash mismatch")
    file = metadata["calibrated_spec"]
    if (
        not isinstance(file, dict)
        or set(file) != {"path", "file_sha256", "spec_sha256"}
        or file["path"] != "calibrated.toml"
    ):
        raise ValueError("calibrated specification must be the local calibrated.toml file")
    _check_digest(file["file_sha256"], "calibrated file hash")
    if file["spec_sha256"] != spec_hash(fitted):
        raise ValueError("calibrated specification hash mismatch")
    names = _selected_records(metadata["fitting_records"], pack)
    if metadata["report"].get("records") != metadata["fitting_records"]:
        raise ValueError("fit report records differ from selected fitting records")
    _validate_report(metadata["report"], config, nominal, fitted)
    optimizer = metadata["report"]["optimizer"]
    calibration = {
        "pack_sha256": metadata["pack_sha256"],
        "nominal_spec_sha256": nominal_hash,
        "calibrated_spec_sha256": spec_hash(fitted),
        "configuration_sha256": metadata["configuration_sha256"],
        "report_sha256": content_hash(metadata["report"]),
        "fitting_records": names,
        "fitting_record_sha256": {
            record["name"]: record["content_sha256"] for record in metadata["fitting_records"]
        },
        "optimizer_success": optimizer["success"],
        "optimizer_status": optimizer["status"],
        "optimizer_message": optimizer["message"],
    }
    if metadata["calibration"] != calibration:
        raise ValueError("calibration replay binding differs from artifact contents")
    provenance = metadata["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"runtime", "implementation_sha256"}:
        raise ValueError("calibration provenance must identify runtime and implementation")
    _check_digest(provenance["implementation_sha256"], "implementation hash")
    if (
        not isinstance(provenance["runtime"], dict)
        or provenance["runtime"].get("spec_hash") != nominal_hash
    ):
        raise ValueError("calibration runtime stamp must identify the nominal specification")
    return config


def save_calibration(output, result: FitResult, *, pack, expected_pack_sha256=None):
    """Atomically save a finite fit result bound to this exact flight pack.

    A valid result is saved even when the optimizer exhausted its budget. The
    recorded success/status are never rewritten. Output must be absent or empty.
    """
    output = _empty_output(output)
    if not isinstance(result, FitResult):
        raise TypeError("result must be a FitResult")
    nominal, source_records, pack_metadata = load_flight_pack(pack)
    pack_hash = content_hash(pack_metadata)
    if expected_pack_sha256 is not None and expected_pack_sha256 != pack_hash:
        raise ValueError("flight pack changed while calibration was running")
    if result.nominal_spec_sha256 != spec_hash(nominal):
        raise ValueError("fit result belongs to a different nominal specification")
    records = _json(result.records)
    names = _selected_records(records, pack_metadata)
    source_counts = {record.name: len(record.time_s) for _, record in source_records}
    if any(record.get("samples") != source_counts[record["name"]] for record in records):
        raise ValueError("fit result sample counts differ from source recordings")
    report = _json(result.report)
    if not isinstance(report, dict) or report.get("records") != records:
        raise ValueError("fit result records differ from fit report")
    _validate_report(report, result.config, nominal, result.spec)
    config = _json(result.config.to_dict())
    optimizer = report["optimizer"]
    calibration = {
        "pack_sha256": pack_hash,
        "nominal_spec_sha256": spec_hash(nominal),
        "calibrated_spec_sha256": spec_hash(result.spec),
        "configuration_sha256": content_hash(config),
        "report_sha256": content_hash(report),
        "fitting_records": names,
        "fitting_record_sha256": {record["name"]: record["content_sha256"] for record in records},
        "optimizer_success": optimizer["success"],
        "optimizer_status": optimizer["status"],
        "optimizer_message": optimizer["message"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        spec_path = temporary / "calibrated.toml"
        save_aircraft_spec(result.spec, spec_path)
        metadata = _json(
            {
                "schema": CALIBRATION_SCHEMA,
                "pack": pack_metadata,
                "pack_sha256": pack_hash,
                "nominal_spec": nominal.to_dict(),
                "nominal_spec_sha256": spec_hash(nominal),
                "calibrated_spec": {
                    "path": "calibrated.toml",
                    "file_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                    "spec_sha256": spec_hash(result.spec),
                },
                "configuration": config,
                "configuration_sha256": content_hash(config),
                "fitting_records": records,
                "calibration": calibration,
                "report": report,
                "provenance": result.provenance,
            }
        )
        _validate_metadata(metadata, result.spec)
        _write_json(
            temporary / "metadata.json", {"sha256": content_hash(metadata), "artifact": metadata}
        )
        with spec_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _empty_output(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_calibration(output)


def load_calibration(path) -> CalibrationArtifact:
    """Read an artifact directory or metadata.json and verify all saved bindings."""
    path = Path(path)
    manifest = path / "metadata.json" if path.is_dir() else path
    if manifest.name != "metadata.json" or manifest.is_symlink():
        raise ValueError("calibration manifest must be the local metadata.json file")
    try:
        wrapper = json.loads(manifest.read_text())
        if not isinstance(wrapper, dict) or set(wrapper) != {"sha256", "artifact"}:
            raise ValueError("malformed calibration metadata wrapper")
        metadata = _json(wrapper["artifact"])
        if wrapper["sha256"] != content_hash(metadata):
            raise ValueError("calibration metadata checksum mismatch")
        file = metadata["calibrated_spec"]
        if not isinstance(file, dict) or file.get("path") != "calibrated.toml":
            raise ValueError("calibrated specification must be the local calibrated.toml file")
        spec_path = manifest.parent / file["path"]
        if spec_path.is_symlink() or spec_path.resolve().parent != manifest.parent.resolve():
            raise ValueError("calibrated specification must be local to the artifact")
        if hashlib.sha256(spec_path.read_bytes()).hexdigest() != file["file_sha256"]:
            raise ValueError("calibrated TOML file checksum mismatch")
        spec = load_aircraft_spec(spec_path)
        config = _validate_metadata(metadata, spec)
        return CalibrationArtifact(
            manifest.parent,
            spec,
            config,
            metadata["calibration"],
            metadata["report"],
            metadata,
            wrapper["sha256"],
        )
    except (KeyError, TypeError, UnicodeError, AttributeError) as exc:
        raise ValueError(f"malformed calibration artifact: {exc}") from exc


def _validate_artifact(artifact):
    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("expected a CalibrationArtifact")
    if content_hash(_json(artifact.metadata)) != artifact.sha256:
        raise ValueError("calibration artifact metadata was changed after loading")
    config = _validate_metadata(artifact.metadata, artifact.spec)
    if (
        artifact.calibration != artifact.metadata["calibration"]
        or artifact.report != artifact.metadata["report"]
        or artifact.config != config
    ):
        raise ValueError("calibration artifact bindings were changed after loading")


__all__ = ["CALIBRATION_SCHEMA", "CalibrationArtifact", "load_calibration", "save_calibration"]
