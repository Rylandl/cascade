"""Synthetic fixtures verify replay plumbing; these tests are not flight validation."""

import hashlib
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.canonical import rigid_body_to_canonical
from cascade.experiments import (
    FlightRecord,
    create_flight_pack,
    evaluate_flight_pack,
    load_flight_pack,
)
from cascade.experiments.manifest import content_hash
from cascade.state import Environment


def record(name="fit", *, offset=0.0):
    states = np.zeros((5, 13))
    states[:, 0] = np.arange(5) + offset
    states[:, 3] = 8
    states[:, 6] = 1
    return FlightRecord(name, name, np.arange(5) * 0.125, states, np.zeros((5, 4)))


def pack(tmp_path, fitting=None, evaluation=None):
    return create_flight_pack(
        tmp_path / "pack",
        cascade.aerobatic_reference_spec(),
        {
            "fitting": [record() if fitting is None else fitting],
            "evaluation": [record("eval", offset=2) if evaluation is None else evaluation],
        },
        license="MIT",
        source="generated synthetic test fixture",
    )


def edit_manifest(path, edit):
    data = json.loads(path.read_text())["pack"]
    edit(data)
    path.write_text(json.dumps({"sha256": content_hash(data), "pack": data}))


def rewrite_record(path, index, change, *, rehash_content=True):
    data = json.loads(path.read_text())["pack"]
    entry = data["records"][index]
    file = path.parent / entry["path"]
    with np.load(file) as stored:
        arrays = dict(stored)
    change(arrays)
    np.savez_compressed(file, **arrays)
    entry["file_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    if rehash_content:
        entry["content_sha256"] = FlightRecord(
            entry["name"],
            entry["maneuver_id"],
            kind=entry["kind"],
            **arrays,
        ).sha256
    path.write_text(json.dumps({"sha256": content_hash(data), "pack": data}))


def test_pack_roundtrip_preserves_license_hashes_and_environment(tmp_path):
    original = replace(record(), wind_ned_m_s=[2, -1, 0.2], density_kg_m3=1.1)
    path = pack(tmp_path, fitting=original)
    aircraft, records, metadata = load_flight_pack(path)
    restored = records[0][1]
    assert metadata["license"] == "MIT"
    assert metadata["source"] == "generated synthetic test fixture"
    assert metadata["aircraft_sha256"] == cascade.spec_hash(aircraft)
    assert restored.sha256 == original.sha256
    assert np.array_equal(restored.wind_ned_m_s, original.wind_ned_m_s)
    assert not restored.canonical_state.flags.writeable
    assert restored.kind == "synthetic"
    with pytest.raises(FileExistsError):
        pack(tmp_path)


@pytest.mark.parametrize("key,value", [("license", ""), ("license", None), ("source", " ")])
def test_license_and_source_validated_on_creation_and_load(tmp_path, key, value):
    kwargs = {"license": "MIT", "source": "synthetic"} | {key: value}
    with pytest.raises(ValueError, match="license and source"):
        create_flight_pack(
            tmp_path / "bad",
            cascade.aerobatic_reference_spec(),
            {"fitting": [record()], "evaluation": [record("eval", offset=2)]},
            **kwargs,
        )
    path = pack(tmp_path)
    edit_manifest(path, lambda data: data.update({key: value}))
    with pytest.raises(ValueError, match="license and source"):
        load_flight_pack(path)


def test_manifest_aircraft_file_and_content_hashes_are_verified(tmp_path):
    path = pack(tmp_path)
    wrapper = json.loads(path.read_text())
    wrapper["pack"]["source"] = "changed without hash"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="schema or hash"):
        load_flight_pack(path)
    edit_manifest(path, lambda data: data.update(source="rehash"))
    file = path.parent / "fit.npz"
    original_bytes = file.read_bytes()
    file.write_bytes(original_bytes + b"tamper")
    with pytest.raises(ValueError, match="file hash"):
        load_flight_pack(path)
    file.write_bytes(original_bytes)
    rewrite_record(
        path,
        0,
        lambda arrays: arrays["canonical_state"].__setitem__((0, 0), 99),
        rehash_content=False,
    )
    with pytest.raises(ValueError, match="content hash"):
        load_flight_pack(path)
    cascade.save_aircraft_spec(
        replace(cascade.aerobatic_reference_spec(), mass_kg=9), path.parent / "aircraft.toml"
    )
    with pytest.raises(ValueError, match="aircraft hash"):
        load_flight_pack(path)


@pytest.mark.parametrize("same_maneuver", [False, True])
def test_split_leakage_rejected_on_create_and_load(tmp_path, same_maneuver):
    fitting = record()
    evaluation = replace(fitting, name="eval", maneuver_id="fit" if same_maneuver else "eval")
    if same_maneuver:
        evaluation = replace(evaluation, canonical_state=record(offset=2).canonical_state)
    with pytest.raises(ValueError, match="crosses data splits"):
        pack(tmp_path, fitting, evaluation)
    path = pack(tmp_path)
    edit_manifest(path, lambda data: data["records"][1].update(maneuver_id="fit"))
    with pytest.raises(ValueError, match="leakage"):
        load_flight_pack(path)


def test_unused_row_zero_and_antipodal_quaternions_do_not_bypass_duplicate_guard(tmp_path):
    fitting = record()
    states = fitting.canonical_state.copy()
    states[::2, 6:10] *= -1
    commands = fitting.command.copy()
    commands[0] = [0.5, 1, 2, 3]
    wind = fitting.wind_ned_m_s.copy()
    wind[0] = [50, 50, 50]
    duplicate = replace(
        fitting,
        name="eval",
        maneuver_id="eval",
        time_s=fitting.time_s + 40,
        canonical_state=states,
        command=commands,
        wind_ned_m_s=wind,
    )
    assert fitting.sha256 == duplicate.sha256
    with pytest.raises(ValueError, match="duplicate"):
        pack(tmp_path, fitting, duplicate)


def test_required_splits_and_command_contract_enforced_on_load(tmp_path):
    path = pack(tmp_path)
    edit_manifest(path, lambda data: data["records"][1].update(split="validation"))
    with pytest.raises(ValueError, match="nonempty fitting and evaluation"):
        load_flight_pack(path)
    edit_manifest(path, lambda data: data["records"][1].update(split="evaluation"))
    rewrite_record(path, 0, lambda arrays: arrays["command"].__setitem__((1, 0), 1.2))
    with pytest.raises(ValueError, match="throttle"):
        load_flight_pack(path)
    rewrite_record(path, 0, lambda arrays: arrays.update(command=np.zeros((5, 5))))
    with pytest.raises(ValueError, match="width"):
        load_flight_pack(path)


@pytest.mark.parametrize("kind", ["record", "aircraft"])
def test_pack_paths_cannot_follow_symlinks_outside_pack(tmp_path, kind):
    path = pack(tmp_path)
    name = "fit.npz" if kind == "record" else "aircraft.toml"
    file = path.parent / name
    outside = tmp_path / name
    file.rename(outside)
    file.symlink_to(outside)
    with pytest.raises(ValueError, match="within the pack"):
        load_flight_pack(path)


def test_csv_roundtrip_and_environment_defaults(tmp_path):
    original = replace(
        record(),
        wind_ned_m_s=np.arange(15).reshape(5, 3) * 0.1,
        density_kg_m3=np.linspace(1.1, 1.2, 5),
    )
    csv = tmp_path / "record.csv"
    original.to_csv(csv)
    restored = FlightRecord.from_csv(csv, name="fit", maneuver_id="fit", kind="synthetic")
    assert restored.sha256 == original.sha256
    assert np.array_equal(restored.canonical_state, original.canonical_state)
    lines = [",".join(line.split(",")[:-4]) for line in csv.read_text().splitlines()]
    csv.write_text("\n".join(lines) + "\n")
    restored = FlightRecord.from_csv(csv, name="fit", maneuver_id="fit")
    assert restored.kind == "measured"
    assert np.all(restored.wind_ned_m_s == 0)
    assert np.all(restored.density_kg_m3 == 1.225)


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("state_0", "state_1", "unique"),
        ("state_0", " state_0", "unique"),
        ("state_0", "other", "canonical state"),
        ("command_1", "command_9", "consecutively"),
        ("wind_ned_z_m_s", "other", "all three"),
    ],
)
def test_csv_malformed_headers_rejected(tmp_path, old, new, match):
    csv = tmp_path / "record.csv"
    record().to_csv(csv)
    csv.write_text(csv.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=match):
        FlightRecord.from_csv(csv, name="fit", maneuver_id="fit")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"time_s": [0, 0.1, 0.2, 0.4, 0.5]}, "uniformly"),
        ({"time_s": [0, 0, 1, 2, 3]}, "increasing"),
        ({"time_s": [0, 1, 2, 3, np.nan]}, "finite"),
        ({"time_s": np.arange(5).astype(complex)}, "real numeric"),
        ({"canonical_state": np.zeros((5, 13))}, "unit norm"),
        ({"command": np.zeros((4, 4))}, "states must"),
        ({"wind_ned_m_s": [1, 2]}, "shape"),
        ({"density_kg_m3": 0}, "positive"),
        ({"density_kg_m3": np.nan}, "finite"),
    ],
)
def test_record_numeric_contract(changes, match):
    with pytest.raises(ValueError, match=match):
        replace(record(), **changes)


@pytest.fixture(scope="module")
def synthetic_replay(tmp_path_factory):
    directory = tmp_path_factory.mktemp("known-synthetic-generator")
    nominal = cascade.aerobatic_reference_spec()
    truth = replace(nominal, mass_kg=nominal.mass_kg * 1.15)
    model = truth.to_model()
    environment = cascade.standard_environment()._replace(
        wind=jnp.array([2.0, -1.0, 0.1]), density=jnp.array(1.1)
    )
    trimmed = cascade.trim_straight_flight(
        model, cascade.StraightFlightCondition(12, altitude_m=40), environment
    )
    assert trimmed.success
    count, substeps, dt = 30, 4, 0.02
    wind = np.tile(np.asarray(environment.wind), (count + 1, 1))
    wind[11:, 1] -= 0.2
    density = np.full(count + 1, 1.1)
    density[16:] = 1.15
    environments = Environment(
        jnp.repeat(jnp.asarray(density[1:]), substeps),
        jnp.repeat(jnp.asarray(wind[1:]), substeps, axis=0),
        jnp.tile(environment.gravity, (count * substeps, 1)),
    )
    controls = cascade.repeat_control(trimmed.control, count * substeps)
    _, states = jax.jit(cascade.rollout)(
        model, trimmed.state, controls, environment, dt / substeps, environments=environments
    )
    states = jax.tree.map(
        lambda initial, sequence: jnp.concatenate(
            [initial[None], sequence[substeps - 1 :: substeps]], axis=0
        ),
        trimmed.state,
        states,
    )
    evaluation = FlightRecord(
        "eval",
        "eval",
        np.arange(count + 1) * dt,
        np.asarray(rigid_body_to_canonical(states.rigid_body)),
        np.tile(cascade.control_to_array(trimmed.control), (count + 1, 1)),
        wind_ned_m_s=wind,
        density_kg_m3=density,
    )
    path = pack(directory, evaluation=evaluation)
    metadata = {
        "pack_sha256": content_hash(json.loads(path.read_text())["pack"]),
        "calibrated_spec_sha256": cascade.spec_hash(truth),
        "fitting_records": ["fit"],
        "method": "known synthetic generator, not fitted",
    }
    return path, truth, metadata, substeps


def test_known_generator_replay_matches_with_recorded_variable_environment(
    synthetic_replay, tmp_path
):
    path, truth, metadata, substeps = synthetic_replay
    output = tmp_path / "scores.json"
    result = evaluate_flight_pack(
        path, calibrated=truth, calibration=metadata, substeps=substeps, output=output
    )
    scores = {row["model"]: row for row in result["scores"]}
    assert result["evidence_kind"] == ["synthetic"]
    assert scores["calibrated"]["position_rmse_m"] < 1e-4
    assert scores["calibrated"]["velocity_rmse_m_s"] < 1e-4
    assert scores["nominal"]["velocity_rmse_m_s"] > 0.1
    assert scores["persistence"]["position_rmse_m"] > 2
    assert json.loads(output.read_text())["calibration"] == metadata
    assert "recorded NED" in result["assumptions"]["wind"]


@pytest.mark.parametrize(
    "change",
    [
        {"pack_sha256": "wrong"},
        {"calibrated_spec_sha256": "wrong"},
        {"fitting_records": ["eval"]},
        {"fitting_records": []},
        {"fitting_records": ["fit", "fit"]},
        {"fitting_records": "fit"},
    ],
)
def test_calibration_bound_to_parameter_hash_and_fitting_only_ids(synthetic_replay, change):
    path, truth, metadata, _ = synthetic_replay
    with pytest.raises(ValueError, match="fitting-only"):
        evaluate_flight_pack(path, calibrated=truth, calibration=metadata | change)


def test_calibration_requires_model_and_unchanged_input_semantics(synthetic_replay):
    path, truth, metadata, _ = synthetic_replay
    with pytest.raises(ValueError, match="requires a calibrated model"):
        evaluate_flight_pack(path, calibration=metadata)
    reordered = replace(truth, control_channels=truth.control_channels[::-1])
    with pytest.raises(ValueError, match="ordered channels"):
        evaluate_flight_pack(
            path,
            calibrated=reordered,
            calibration=metadata | {"calibrated_spec_sha256": cascade.spec_hash(reordered)},
        )


@pytest.mark.parametrize("substeps", [0, -1, True, 1.5])
def test_invalid_substeps_rejected_before_loading(substeps):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_flight_pack("does-not-exist.json", substeps=substeps)


def test_replay_rejects_underflowing_timestep_before_integration(tmp_path):
    tiny_dt = float(np.finfo(np.asarray(jnp.asarray(1.0)).dtype).smallest_subnormal)
    path = pack(
        tmp_path, evaluation=replace(record("eval", offset=2), time_s=np.arange(5) * tiny_dt)
    )
    with pytest.raises(ValueError, match="timestep.*JAX precision"):
        evaluate_flight_pack(path)
