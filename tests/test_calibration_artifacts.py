"""Calibration persistence, fitting-only data flow, and explicitly labelled replay."""

import copy
import json
from dataclasses import replace

import jax
import numpy as np
import pytest

from cascade.calibration.artifacts import load_calibration, save_calibration
from cascade.calibration.fitting import CalibrationConfig, FitResult
from cascade.calibration.parameters import FitParameter, Parameterization
from cascade.calibration.workflow import evaluate_calibration, fit_flight_pack
from cascade.experiments import FlightRecord, create_flight_pack, load_flight_pack
from cascade.experiments.manifest import content_hash
from cascade.experiments.replay import prepare_record, replay_record
from cascade.provenance import spec_hash, stamp
from cascade.reference import aerobatic_reference_spec


def recording(name, *, offset=0.0):
    states = np.zeros((5, 13))
    states[:, 0] = np.arange(5) * 0.2 + offset
    states[:, 2] = 50
    states[:, 3] = 8
    states[:, 6] = 1
    return FlightRecord(name, name, np.arange(5) * 0.025, states, np.zeros((5, 4)))


def make_pack(path, *, fitting=None, heldout_offset=1.0):
    return create_flight_pack(
        path,
        aerobatic_reference_spec(),
        {
            "fitting": [recording("fit") if fitting is None else fitting],
            "validation": [recording("validation", offset=heldout_offset)],
            "evaluation": [recording("evaluation", offset=heldout_offset + 1)],
        },
        license="MIT",
        source="synthetic software regression fixture, not flight validation",
    )


def declared_result(nominal, fitting, config=None, *, converged=True):
    """Construct a solver-result fixture; separate tests exercise real numerical fitting."""
    config = config or CalibrationConfig(
        (FitParameter("mass_kg", 0.8, 1.6), FitParameter("inertia_scale", 0.8, 1.4)),
        substeps=1,
        warmup_steps=1,
        max_nfev=5,
    )
    parameterization = Parameterization(nominal, config.parameters)
    values = [1.3 if p.path == "mass_kg" else 1.05 for p in config.parameters]
    fitted = parameterization.apply_spec(np.asarray(values))
    records = tuple(
        {
            "name": record.name,
            "maneuver_id": record.maneuver_id,
            "kind": record.kind,
            "content_sha256": record.sha256,
            "samples": len(record.time_s),
            "scored_samples": len(record.time_s) - 1 - config.warmup_steps,
        }
        for record in fitting
    )
    singular = list(reversed(range(1, len(config.parameters) + 1)))
    report = {
        "schema": "cascade_calibration_fit_v1",
        "nominal_spec_sha256": spec_hash(nominal),
        "calibrated_spec_sha256": spec_hash(fitted),
        "configuration": config.to_dict(),
        "records": list(records),
        "optimizer": {
            "method": "scipy.optimize.least_squares",
            "success": converged,
            "status": 1 if converged else 0,
            "message": "fixture convergence" if converged else "maximum evaluations exceeded",
            "nfev": 2 if converged else config.max_nfev,
            "njev": 2,
            "optimality": 0.01,
            "tolerances": {"ftol": 1e-6, "xtol": 1e-6, "gtol": 1e-6},
        },
        "initial_data_loss": 2.0,
        "final_data_loss": 0.1,
        "parameters": [
            {
                "path": p.path,
                "initial": float(parameterization.initial_values[i]),
                "value": value,
                "lower": p.lower,
                "upper": p.upper,
                "normalized_value": (value - p.lower) / (p.upper - p.lower),
                "bound_hit": None,
            }
            for i, (p, value) in enumerate(zip(config.parameters, values, strict=True))
        ],
        "identifiability": {
            "coordinates": "unit-interval parameters; scaled data residuals only",
            "singular_values": singular,
            "rank": len(singular),
            "parameter_count": len(singular),
            "rank_tolerance": 1e-5,
            "condition": singular[0] / singular[-1],
        },
    }
    provenance = {
        "runtime": stamp(nominal, nominal.to_model()),
        "implementation_sha256": "1" * 64,
    }
    return FitResult(fitted, config, records, spec_hash(nominal), report, provenance)


@pytest.fixture
def case(tmp_path):
    pack = make_pack(tmp_path / "pack")
    nominal, records, _ = load_flight_pack(pack)
    result = declared_result(nominal, tuple(r for split, r in records if split == "fitting"))
    return pack, result


def test_artifact_roundtrip_binds_pack_specs_records_configuration_and_solver(tmp_path, case):
    pack, result = case
    output = tmp_path / "artifact"
    output.mkdir()  # Atomic publication also supports a preexisting empty directory.
    saved = save_calibration(output, result, pack=pack)
    loaded = load_calibration(saved.manifest_path)
    assert loaded.path == output
    assert set(p.name for p in output.iterdir()) == {"calibrated.toml", "metadata.json"}
    assert spec_hash(loaded.spec) == spec_hash(result.spec)
    assert loaded.report == result.report
    assert loaded.config == result.config
    assert loaded.calibration["fitting_records"] == ["fit"]
    assert loaded.calibration["fitting_record_sha256"] == {
        "fit": result.records[0]["content_sha256"]
    }
    assert loaded.calibration["optimizer_success"] is True
    assert loaded.sha256 == content_hash(loaded.metadata)
    assert loaded.metadata["provenance"]["runtime"]["spec_hash"] == result.nominal_spec_sha256
    assert len(loaded.metadata["provenance"]["implementation_sha256"]) == 64


def test_delayed_save_preserves_fitting_runtime_and_source_snapshot(tmp_path, case, monkeypatch):
    import cascade.calibration.fitting as fitting

    pack, result = case
    snapshot = copy.deepcopy(result.provenance)
    later_stamp = {
        **snapshot["runtime"],
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "x64_enabled": not snapshot["runtime"]["x64_enabled"],
    }
    monkeypatch.setattr(fitting, "stamp", lambda *args, **kwargs: later_stamp)
    monkeypatch.setattr(fitting, "_implementation_hash", lambda: "2" * 64)
    original_precision = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", later_stamp["x64_enabled"])
    try:
        artifact = save_calibration(tmp_path / "artifact", result, pack=pack)
    finally:
        jax.config.update("jax_enable_x64", original_precision)
    assert artifact.metadata["provenance"] == snapshot
    assert load_calibration(artifact.path).metadata["provenance"] == snapshot
    result.provenance["runtime"]["timestamp_utc"] = "changed after publication"
    assert artifact.metadata["provenance"] == snapshot


def test_output_refuses_existing_data_and_symlinks(tmp_path, case):
    pack, result = case
    output = tmp_path / "artifact"
    save_calibration(output, result, pack=pack)
    original = (output / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError, match="absent or empty"):
        save_calibration(output, result, pack=pack)
    assert (output / "metadata.json").read_bytes() == original
    link = tmp_path / "link"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        save_calibration(link, result, pack=pack)


def test_interrupted_publication_preserves_empty_output(tmp_path, case, monkeypatch):
    import cascade.calibration.artifacts as module

    pack, result = case
    output = tmp_path / "artifact"
    output.mkdir()
    replace_file = module.os.replace

    def interrupted(source, destination):
        if destination == output:
            raise OSError("simulated interrupted publication")
        return replace_file(source, destination)

    monkeypatch.setattr(module.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        save_calibration(output, result, pack=pack)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".artifact.*"))


def rewrite_metadata(path, change, *, resign=True):
    wrapper = json.loads(path.read_text())
    change(wrapper["artifact"])
    if resign:
        wrapper["sha256"] = content_hash(wrapper["artifact"])
    path.write_text(json.dumps(wrapper))


@pytest.mark.parametrize(
    "damage",
    [
        "metadata",
        "toml",
        "schema",
        "path",
        "symlink",
        "nominal",
        "configuration",
        "heldout_record",
        "record_hash",
        "report_spec",
        "report_config",
        "report_records",
        "sample_count",
    ],
)
def test_artifact_tamper_and_binding_failures_are_rejected(tmp_path, case, damage):
    pack, result = case
    artifact = save_calibration(tmp_path / "artifact", result, pack=pack)
    if damage == "metadata":
        rewrite_metadata(
            artifact.manifest_path, lambda d: d.update(pack_sha256="0" * 64), resign=False
        )
    elif damage == "toml":
        with (artifact.path / "calibrated.toml").open("a") as stream:
            stream.write("\n# changed bytes\n")
    elif damage == "symlink":
        outside = tmp_path / "outside.toml"
        (artifact.path / "calibrated.toml").rename(outside)
        (artifact.path / "calibrated.toml").symlink_to(outside)
    else:

        def change(data):
            if damage == "schema":
                data["schema"] = "unsupported"
            elif damage == "path":
                data["calibrated_spec"]["path"] = "../outside.toml"
            elif damage == "nominal":
                data["nominal_spec_sha256"] = "0" * 64
            elif damage == "configuration":
                data["configuration"]["max_nfev"] = 10
            elif damage == "heldout_record":
                data["fitting_records"][0]["name"] = "evaluation"
            elif damage == "record_hash":
                data["fitting_records"][0]["content_sha256"] = "0" * 64
            elif damage == "report_spec":
                data["report"]["calibrated_spec_sha256"] = "0" * 64
            elif damage == "report_config":
                data["report"]["configuration"]["max_nfev"] = 10
            elif damage == "report_records":
                data["report"]["records"][0]["name"] = "evaluation"
            else:
                data["report"]["records"][0]["scored_samples"] += 1
                data["fitting_records"] = copy.deepcopy(data["report"]["records"])

        rewrite_metadata(artifact.manifest_path, change)
    with pytest.raises(ValueError):
        load_calibration(artifact.path)


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "success",
        "nfev_zero",
        "nfev_budget",
        "optimality",
        "tolerances",
        "rank",
        "singular_shape",
        "singular_order",
        "condition",
        "normalized",
        "bound_hit",
        "initial",
        "value",
    ],
)
def test_rehashed_malformed_solver_diagnostics_are_rejected(tmp_path, case, damage):
    pack, result = case
    artifact = save_calibration(tmp_path / "artifact", result, pack=pack)

    def change(data):
        report = data["report"]
        if damage == "status":
            report["optimizer"]["status"] = 7
        elif damage == "success":
            report["optimizer"]["success"] = False
        elif damage == "nfev_zero":
            report["optimizer"]["nfev"] = 0
        elif damage == "nfev_budget":
            report["optimizer"]["nfev"] = 100
        elif damage == "optimality":
            report["optimizer"]["optimality"] = -1
        elif damage == "tolerances":
            report["optimizer"]["tolerances"]["ftol"] = 0
        elif damage == "rank":
            report["identifiability"]["rank"] = 0
        elif damage == "singular_shape":
            report["identifiability"]["singular_values"] = [2]
        elif damage == "singular_order":
            report["identifiability"]["singular_values"] = [1, 2]
        elif damage == "condition":
            report["identifiability"]["condition"] = None
        elif damage == "normalized":
            report["parameters"][0]["normalized_value"] = 0.1
        elif damage == "bound_hit":
            report["parameters"][0]["bound_hit"] = "upper"
        elif damage == "initial":
            report["parameters"][0]["initial"] = 1.5
        else:
            report["parameters"][0]["value"] = 2.0
        data["calibration"]["report_sha256"] = content_hash(report)
        for field in ("success", "status", "message"):
            data["calibration"][f"optimizer_{field}"] = report["optimizer"][field]

    rewrite_metadata(artifact.manifest_path, change)
    with pytest.raises(ValueError):
        load_calibration(artifact.path)


def test_nonfinite_json_is_rejected_even_with_resigned_wrapper(tmp_path, case):
    pack, result = case
    artifact = save_calibration(tmp_path / "artifact", result, pack=pack)
    wrapper = json.loads(artifact.manifest_path.read_text())
    wrapper["artifact"]["report"]["final_data_loss"] = float("nan")
    artifact.manifest_path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="finite JSON"):
        load_calibration(artifact.path)


@pytest.mark.parametrize("damage", ["nominal", "sample_count", "unselected_parameter"])
def test_save_rejects_misbound_fit_result(tmp_path, case, damage):
    pack, result = case
    if damage == "nominal":
        result = replace(result, nominal_spec_sha256="0" * 64)
    elif damage == "sample_count":
        records = copy.deepcopy(result.records)
        records[0]["samples"] += 1
        records[0]["scored_samples"] += 1
        report = copy.deepcopy(result.report)
        report["records"] = list(records)
        result = replace(result, records=records, report=report)
    else:
        fitted = replace(result.spec, reference_area_m2=result.spec.reference_area_m2 * 1.1)
        report = copy.deepcopy(result.report)
        report["calibrated_spec_sha256"] = spec_hash(fitted)
        result = replace(result, spec=fitted, report=report)
    with pytest.raises(ValueError):
        save_calibration(tmp_path / "artifact", result, pack=pack)
    assert not (tmp_path / "artifact").exists()


def test_only_fitting_arrays_reach_optimizer_and_pack_changes_are_bound(tmp_path, monkeypatch):
    import cascade.calibration.workflow as workflow

    first = make_pack(tmp_path / "first")
    second = make_pack(tmp_path / "second", heldout_offset=100)
    config = CalibrationConfig((FitParameter("mass_kg", 0.8, 1.6),), substeps=1, max_nfev=5)
    observed = []

    def fitting_only(nominal, records, settings):
        assert tuple(r.name for r in records) == ("fit",)
        observed.append(tuple(r.sha256 for r in records))
        return declared_result(nominal, records, settings, converged=False)

    monkeypatch.setattr(workflow, "fit_records", fitting_only)
    left = fit_flight_pack(first, config, output=tmp_path / "left")
    right = fit_flight_pack(second, config, output=tmp_path / "right")
    assert observed[0] == observed[1]
    assert spec_hash(left.spec) == spec_hash(right.spec)
    assert left.report == right.report
    assert left.calibration["pack_sha256"] != right.calibration["pack_sha256"]
    assert left.calibration["optimizer_success"] is False
    with pytest.raises(ValueError, match="different flight pack"):
        evaluate_calibration(left, second)


def test_pack_change_during_fit_prevents_publication(tmp_path, case, monkeypatch):
    import cascade.calibration.workflow as workflow

    pack, result = case

    def changed_pack(*args):
        wrapper = json.loads(pack.read_text())
        wrapper["pack"]["description"] = "changed during optimization"
        wrapper["sha256"] = content_hash(wrapper["pack"])
        pack.write_text(json.dumps(wrapper))
        return result

    monkeypatch.setattr(workflow, "fit_records", changed_pack)
    with pytest.raises(ValueError, match="changed while calibration"):
        fit_flight_pack(pack, result.config, output=tmp_path / "artifact")


def test_nonconverged_artifact_scores_validation_and_evaluation_honestly(tmp_path, case):
    pack, result = case
    nominal, records, _ = load_flight_pack(pack)
    result = declared_result(
        nominal, tuple(r for s, r in records if s == "fitting"), converged=False
    )
    artifact = save_calibration(tmp_path / "artifact", result, pack=pack)
    destination = tmp_path / "evaluation.json"
    evaluated = evaluate_calibration(artifact.path, pack, output=destination)
    assert evaluated == json.loads(destination.read_text())
    assert evaluated["split"] == "evaluation"
    assert {row["record"] for row in evaluated["scores"]} == {"evaluation"}
    assert {row["model"] for row in evaluated["scores"]} == {"nominal", "calibrated", "persistence"}
    assert evaluated["calibration_status"]["optimizer"]["success"] is False
    assert evaluated["calibration_status"]["optimizer"]["status"] == 0
    assert evaluated["calibration_status"]["fitting_warmup_steps"] == 1
    assert evaluated["calibration_status"]["evaluation_warmup_steps"] == 0
    validated = evaluate_calibration(artifact, pack, split="validation")
    assert {row["record"] for row in validated["scores"]} == {"validation"}
    assert artifact.report == result.report
    artifact.calibration["optimizer_success"] = True
    with pytest.raises(ValueError, match="changed after loading"):
        evaluate_calibration(artifact, pack)


@pytest.mark.slow
def test_real_fitting_is_independent_of_heldout_arrays(tmp_path):
    nominal = aerobatic_reference_spec()
    truth = replace(nominal, mass_kg=1.35)
    original = recording("fit")
    commands = original.command.copy()
    commands[:, 0] = 0.6
    commands[1:, 2] = [0.02, -0.01, 0.03, 0.0]
    original = replace(original, command=commands)
    inputs = prepare_record(original, truth.to_model(), substeps=1)
    predicted = np.asarray(jax.jit(replay_record)(truth.to_model(), inputs))
    measured = replace(
        original, canonical_state=np.vstack([original.canonical_state[0], predicted])
    )
    left_pack = make_pack(tmp_path / "left-pack", fitting=measured)
    right_pack = make_pack(tmp_path / "right-pack", fitting=measured, heldout_offset=200)
    config = CalibrationConfig((FitParameter("mass_kg", 0.8, 1.6),), substeps=1, max_nfev=5)
    left = fit_flight_pack(left_pack, config, output=tmp_path / "left")
    right = fit_flight_pack(right_pack, config, output=tmp_path / "right")
    assert spec_hash(left.spec) == spec_hash(right.spec)
    assert left.report == right.report
    assert left.report["final_data_loss"] < left.report["initial_data_loss"]
    assert left.calibration["pack_sha256"] != right.calibration["pack_sha256"]
    assert left.spec.mass_kg == pytest.approx(1.35, abs=0.03)
