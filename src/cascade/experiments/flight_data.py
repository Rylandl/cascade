"""Auditable maneuver packs and nominal/calibrated replay evaluation.

This module consumes canonical recordings prepared by an upstream log adapter. It does not
infer sensors, fit parameters, or claim that synthetic trajectories are flight measurements.
"""

from __future__ import annotations

import hashlib
import json
from csv import reader
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cascade.canonical import (
    CANONICAL_STATE_SCHEMA,
    rigid_body_from_canonical,
    rigid_body_to_canonical,
)
from cascade.initialization import (
    control_from_array,
    equilibrate_internal_state,
    standard_environment,
    zero_state,
)
from cascade.integration import rollout
from cascade.provenance import model_hash, spec_hash, stamp
from cascade.spec import AircraftSpec, load_aircraft_spec, save_aircraft_spec
from cascade.state import Environment

from .manifest import content_hash, valid_name

PACK_SCHEMA = "cascade_flight_pack_v1"


@dataclass(frozen=True)
class FlightRecord:
    """Uniformly sampled canonical states plus native throttle/channel commands.

    State[0] is the initial observation. command[i] acts on (time[i-1], time[i]];
    command[0] is unused in replay. Unknown internal actuator/separation states are
    initialized at equilibrium under command[1], explicitly reported by the evaluator.
    ``wind_ned_m_s`` is a constant NED vector or (T,3) interval-end sequence; density is a
    scalar or (T,) sequence. Their first rows, like command[0], are unused in replay.
    """

    name: str
    maneuver_id: str
    time_s: np.ndarray
    canonical_state: np.ndarray
    command: np.ndarray
    kind: str = "synthetic"
    wind_ned_m_s: Any = (0.0, 0.0, 0.0)
    density_kg_m3: Any = 1.225

    def __post_init__(self):
        valid_name(self.name)
        valid_name(self.maneuver_id)
        if self.kind not in {"synthetic", "measured"}:
            raise ValueError("record kind must be synthetic or measured")
        for name in ("time_s", "canonical_state", "command"):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind not in "iuf":
                raise ValueError(f"{name} must contain real numeric values")
            value = value.astype(np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite numeric values")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        t, state, command = self.time_s, self.canonical_state, self.command
        with np.errstate(over="ignore"):
            intervals = np.diff(t) if t.ndim == 1 else np.array([])
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(intervals).all() or np.any(intervals <= 0):
            raise ValueError("time_s must contain at least two increasing samples")
        if not np.allclose(intervals, intervals[0], rtol=1e-7, atol=1e-9):
            raise ValueError("recordings must be uniformly sampled; resample in the log adapter")
        if state.shape != (len(t), 13) or command.ndim != 2 or len(command) != len(t):
            raise ValueError("states must be (T, 13) and command must be (T, inputs)")
        if not np.allclose(np.linalg.norm(state[:, 6:10], axis=1), 1, rtol=0, atol=1e-4):
            raise ValueError("canonical wxyz quaternion must have unit norm")
        for name, shape, constant_shape in (
            ("wind_ned_m_s", (len(t), 3), (3,)),
            ("density_kg_m3", (len(t),), ()),
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite real values")
            if value.shape == constant_shape:
                value = np.broadcast_to(value, shape)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {constant_shape} or {shape}")
            value = np.array(value, dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} overflows float64 storage")
            if name == "density_kg_m3" and np.any(value <= 0):
                raise ValueError("density must be positive")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def sha256(self):
        """Replay-content hash: relative timestamps, antipodal-normalized attitudes, active rows.

        File hashes separately protect every stored byte. Ignored row-zero inputs cannot make
        a duplicate recording appear distinct across data splits.
        """
        digest = hashlib.sha256()
        state = self.canonical_state.copy()
        quaternion = state[:, 6:10]
        pivot = np.argmax(np.abs(quaternion), axis=1)
        quaternion *= np.sign(quaternion[np.arange(len(state)), pivot])[:, None]
        for array in (
            self.time_s - self.time_s[0],
            state,
            self.command[1:],
            self.wind_ned_m_s[1:],
            self.density_kg_m3[1:],
        ):
            array = np.array(array, dtype="<f8", copy=True)
            array[array == 0] = 0.0  # normalize signed zero
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def to_csv(self, path: str | Path) -> None:
        """Write lossless decimal columns accepted by :meth:`from_csv`, including environment."""
        names = ["time_s", *[f"state_{i}" for i in range(13)]]
        names += [f"command_{i}" for i in range(self.command.shape[1])]
        names += ["wind_ned_x_m_s", "wind_ned_y_m_s", "wind_ned_z_m_s", "density_kg_m3"]
        values = np.column_stack(
            (
                self.time_s,
                self.canonical_state,
                self.command,
                self.wind_ned_m_s,
                self.density_kg_m3,
            )
        )
        np.savetxt(path, values, delimiter=",", header=",".join(names), comments="", fmt="%.17g")

    @classmethod
    def from_csv(cls, path, *, name, maneuver_id, kind="measured"):
        """Read explicit columns: time_s, state_0..state_12, command_0..command_N.

        States are canonical NWU/FLU, scalar-first quaternion, SI. Column names deliberately
        carry no inferred mapping to vendor-specific logs; log adapters own that conversion.
        """
        with Path(path).open(newline="") as stream:
            header = next(reader(stream), [])
        if len(set(header)) != len(header) or any(name != name.strip() for name in header):
            raise ValueError("CSV column names must be explicit and unique")
        values = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
        names = values.dtype.names or ()
        if tuple(header) != names:
            raise ValueError("CSV column names must use the exact documented spelling")
        expected = ["time_s", *[f"state_{i}" for i in range(13)]]
        commands = [n for n in names if n.startswith("command_")]
        try:
            commands.sort(key=lambda n: int(n.split("_")[-1]))
        except ValueError as e:
            raise ValueError("command columns must be consecutively numbered") from e
        if commands != [f"command_{i}" for i in range(len(commands))] or not commands:
            raise ValueError("command columns must be consecutively numbered from zero")
        wind_names = ["wind_ned_x_m_s", "wind_ned_y_m_s", "wind_ned_z_m_s"]
        environment_names = set(names) & {*wind_names, "density_kg_m3"}
        if environment_names & set(wind_names) and not set(wind_names) <= environment_names:
            raise ValueError("CSV wind requires all three NED components")
        if set(names) != set(expected + commands) | environment_names:
            raise ValueError("CSV must use explicit canonical state and command columns")
        return cls(
            name,
            maneuver_id,
            values["time_s"],
            np.stack([values[n] for n in expected[1:]], axis=-1),
            np.stack([values[n] for n in commands], axis=-1),
            kind,
            wind_ned_m_s=(
                np.stack([values[n] for n in wind_names], axis=-1)
                if set(wind_names) <= environment_names
                else (0.0, 0.0, 0.0)
            ),
            density_kg_m3=values["density_kg_m3"] if "density_kg_m3" in names else 1.225,
        )


def _validate_metadata(license, source):
    if any(not isinstance(value, str) or not value.strip() for value in (license, source)):
        raise ValueError("license and source must be nonempty strings")


def _validate_commands(record, model):
    if record.command.shape[1] != model.n_propellers + model.n_control_channels:
        raise ValueError("record command width differs from aircraft")
    if np.any(record.command[:, : model.n_propellers] < 0) or np.any(
        record.command[:, : model.n_propellers] > 1
    ):
        raise ValueError("native throttle commands must be in [0, 1]")


def create_flight_pack(
    output: str | Path,
    aircraft: AircraftSpec,
    splits: dict[str, list[FlightRecord]],
    *,
    license: str,
    source: str,
    description: str = "",
) -> Path:
    """Freeze recordings and maneuver-level splits with declared licensing and content hashes.

    The caller must have redistribution rights. Recording a license is not license validation.
    Maneuvers and exact duplicate recordings cannot cross fitting/validation/evaluation splits.
    """
    _validate_metadata(license, source)
    if not splits.get("fitting") or not splits.get("evaluation"):
        raise ValueError("provide nonempty fitting and evaluation splits")
    if set(splits) - {"fitting", "validation", "evaluation"}:
        raise ValueError("unsupported split")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("flight pack directory must be empty")
    names, maneuvers, hashes = set(), {}, {}
    model = aircraft.to_model()
    records = []
    for split, flights in splits.items():
        for record in flights:
            if not isinstance(record, FlightRecord):
                raise ValueError("flight pack entries must be FlightRecord values")
            if record.name in names:
                raise ValueError("record names must be unique")
            _validate_commands(record, model)
            if (
                maneuvers.get(record.maneuver_id, split) != split
                or hashes.get(record.sha256, split) != split
            ):
                raise ValueError("a maneuver or duplicate recording crosses data splits")
            names.add(record.name)
            maneuvers[record.maneuver_id] = hashes[record.sha256] = split
            records.append((split, record))
    output.mkdir(parents=True, exist_ok=True)
    save_aircraft_spec(aircraft, output / "aircraft.toml")
    entries = []
    for split, record in records:
        path = output / f"{record.name}.npz"
        np.savez_compressed(
            path,
            time_s=record.time_s,
            canonical_state=record.canonical_state,
            command=record.command,
            wind_ned_m_s=record.wind_ned_m_s,
            density_kg_m3=record.density_kg_m3,
        )
        entries.append(
            {
                "name": record.name,
                "maneuver_id": record.maneuver_id,
                "split": split,
                "kind": record.kind,
                "path": path.name,
                "content_sha256": record.sha256,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    data = {
        "schema": PACK_SCHEMA,
        "state_schema": CANONICAL_STATE_SCHEMA,
        "license": license,
        "source": source,
        "description": description,
        "aircraft_sha256": spec_hash(aircraft),
        "records": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps({"sha256": content_hash(data), "pack": data}, indent=2, allow_nan=False) + "\n"
    )
    return output / "manifest.json"


def load_flight_pack(path: str | Path):
    """Verify hashes and path confinement before loading any recording."""
    path = Path(path)
    wrapper = json.loads(path.read_text())
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("pack"), dict):
        raise ValueError("flight pack must contain a manifest object")
    data = wrapper["pack"]
    if data.get("schema") != PACK_SCHEMA or wrapper.get("sha256") != content_hash(data):
        raise ValueError("flight pack schema or hash mismatch")
    if data.get("state_schema") != CANONICAL_STATE_SCHEMA:
        raise ValueError("unsupported state schema")
    _validate_metadata(data.get("license"), data.get("source"))
    aircraft_path = (path.parent / "aircraft.toml").resolve()
    if aircraft_path.parent != path.parent.resolve():
        raise ValueError("aircraft must be local within the pack")
    aircraft = load_aircraft_spec(aircraft_path)
    if spec_hash(aircraft) != data["aircraft_sha256"]:
        raise ValueError("aircraft hash mismatch")
    records = []
    if not isinstance(data.get("records"), list):
        raise ValueError("records must be a list")
    model = aircraft.to_model()
    names, groups, fingerprints = set(), {}, {}
    for entry in data["records"]:
        file = (path.parent / entry["path"]).resolve()
        if file.parent != path.parent.resolve() or file.suffix != ".npz":
            raise ValueError("record must be a local NPZ within the pack")
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry["file_sha256"]:
            raise ValueError("record file hash mismatch")
        with np.load(file, allow_pickle=False) as a:
            record = FlightRecord(
                entry["name"],
                entry["maneuver_id"],
                a["time_s"],
                a["canonical_state"],
                a["command"],
                entry["kind"],
                wind_ned_m_s=a["wind_ned_m_s"],
                density_kg_m3=a["density_kg_m3"],
            )
        _validate_commands(record, model)
        if record.sha256 != entry["content_sha256"]:
            raise ValueError("record content hash mismatch")
        split = entry["split"]
        if split not in {"fitting", "validation", "evaluation"} or record.name in names:
            raise ValueError("invalid split or duplicate record name")
        if (
            groups.get(record.maneuver_id, split) != split
            or fingerprints.get(record.sha256, split) != split
        ):
            raise ValueError("data leakage across splits")
        names.add(record.name)
        groups[record.maneuver_id] = fingerprints[record.sha256] = split
        records.append((split, record))
    if not {"fitting", "evaluation"} <= {split for split, _ in records}:
        raise ValueError("provide nonempty fitting and evaluation splits")
    return aircraft, records, data


def _errors(predicted, observed):
    quaternion_dot = np.sum(predicted[:, 6:10] * observed[:, 6:10], axis=-1)
    norm = np.linalg.norm(predicted[:, 6:10], axis=-1) * np.linalg.norm(observed[:, 6:10], axis=-1)
    angular = 2 * np.arccos(np.clip(np.abs(quaternion_dot) / norm, 0, 1))
    return {
        "position_rmse_m": float(
            np.sqrt(np.mean(np.sum((predicted[:, :3] - observed[:, :3]) ** 2, axis=-1)))
        ),
        "velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.sum((predicted[:, 3:6] - observed[:, 3:6]) ** 2, axis=-1)))
        ),
        "attitude_rmse_rad": float(np.sqrt(np.mean(angular**2))),
        "rate_rmse_rad_s": float(
            np.sqrt(np.mean(np.sum((predicted[:, 10:13] - observed[:, 10:13]) ** 2, axis=-1)))
        ),
    }


def evaluate_flight_pack(
    path: str | Path,
    *,
    calibrated: AircraftSpec | None = None,
    calibration: dict[str, Any] | None = None,
    substeps: int = 4,
    split: str = "evaluation",
    output: str | Path | None = None,
):
    """Open-loop replay from each maneuver's initial state, plus constant-state persistence.

    A calibrated model requires fitting record IDs, pack hash and calibrated specification hash.
    Parameter selection stays upstream (e.g. Glassbox); this API checks declared split use
    but cannot establish that a human never inspected held-out data. Recorded NED wind and
    density are applied per interval; defaults are calm/standard if no environment was supplied.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    nominal, records, data = load_flight_pack(path)
    pack_hash = content_hash(data)
    if split not in {"fitting", "validation", "evaluation"}:
        raise ValueError("unsupported scoring split")
    if calibrated is None and calibration is not None:
        raise ValueError("calibration metadata requires a calibrated model")
    if calibrated is not None:
        fitting = {r.name for s, r in records if s == "fitting"}
        if (
            not isinstance(calibration, dict)
            or calibration.get("pack_sha256") != pack_hash
            or calibration.get("calibrated_spec_sha256") != spec_hash(calibrated)
            or not isinstance(calibration.get("fitting_records"), list)
            or not calibration["fitting_records"]
            or not all(isinstance(name, str) for name in calibration["fitting_records"])
            or len(set(calibration["fitting_records"])) != len(calibration["fitting_records"])
            or not set(calibration["fitting_records"]) <= fitting
        ):
            raise ValueError(
                "calibration must identify this pack, calibrated spec and fitting-only record IDs"
            )
        calibration = json.loads(json.dumps(calibration, allow_nan=False))
        if calibrated.control_channels != nominal.control_channels or any(
            tuple(item.name for item in getattr(calibrated, group))
            != tuple(item.name for item in getattr(nominal, group))
            for group in ("surfaces", "propellers")
        ):
            raise ValueError(
                "calibrated model must preserve ordered channels, surfaces and propellers"
            )
    models = {"nominal": nominal.to_model()}
    if calibrated is not None:
        models["calibrated"] = calibrated.to_model()
    nominal_model = models["nominal"]
    for model in models.values():
        if (model.n_propellers, model.n_control_channels, model.n_surfaces) != (
            nominal_model.n_propellers,
            nominal_model.n_control_channels,
            nominal_model.n_surfaces,
        ):
            raise ValueError("calibrated model must preserve actuator and surface topology")
    standard = standard_environment()
    rows = []
    for record_split, record in records:
        if record_split != split:
            continue
        if record.command.shape[1] != nominal_model.n_propellers + nominal_model.n_control_channels:
            raise ValueError("record command width differs from aircraft")
        dt = float(record.time_s[1] - record.time_s[0])
        step_dt = jnp.asarray(dt / substeps)
        initial_observation = jnp.asarray(record.canonical_state[0])
        interval_commands = jnp.asarray(record.command[1:])
        if not np.isfinite(float(step_dt)) or float(step_dt) <= 0:
            raise ValueError("record timestep is outside the configured JAX precision")
        if (
            not np.isfinite(np.asarray(initial_observation)).all()
            or not np.isfinite(np.asarray(interval_commands)).all()
        ):
            raise ValueError("record state or commands overflow the configured JAX precision")
        environment = Environment(
            density=jnp.asarray(record.density_kg_m3[1]),
            wind=jnp.asarray(record.wind_ned_m_s[1]),
            gravity=standard.gravity,
        )
        environments = Environment(
            density=jnp.repeat(jnp.asarray(record.density_kg_m3[1:]), substeps, axis=0),
            wind=jnp.repeat(jnp.asarray(record.wind_ned_m_s[1:]), substeps, axis=0),
            gravity=jnp.broadcast_to(standard.gravity, ((len(record.time_s) - 1) * substeps, 3)),
        )
        if not all(
            np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(environments)
        ) or np.any(np.asarray(environments.density) <= 0):
            raise ValueError("record environment overflows the configured JAX precision")
        observed = record.canonical_state[1:]
        rows.append(
            {
                "record": record.name,
                "model": "persistence",
                "kind": record.kind,
                **_errors(np.broadcast_to(record.canonical_state[0], observed.shape), observed),
            }
        )
        for label, model in models.items():
            first_control = control_from_array(model, interval_commands[0])
            state = zero_state(model)._replace(
                rigid_body=rigid_body_from_canonical(initial_observation)
            )
            state = equilibrate_internal_state(model, state, first_control, environment)
            commands = jnp.repeat(interval_commands, substeps, axis=0)
            controls = control_from_array(model, commands)
            _, states = jax.jit(rollout)(
                model, state, controls, environment, step_dt, environments=environments
            )
            states = jax.tree.map(lambda x: x[substeps - 1 :: substeps], states)
            predicted = np.asarray(rigid_body_to_canonical(states.rigid_body))
            if not np.isfinite(predicted).all():
                raise ValueError(f"{label} replay of {record.name} became nonfinite")
            rows.append(
                {
                    "record": record.name,
                    "model": label,
                    "kind": record.kind,
                    **_errors(predicted, observed),
                }
            )
    if not rows:
        raise ValueError("no recordings in requested scoring split")
    result = {
        "schema": "cascade_replay_results_v1",
        "pack_sha256": pack_hash,
        "split": split,
        "provenance": stamp(),
        "model_sha256": {n: model_hash(m) for n, m in models.items()},
        "calibration": calibration,
        "substeps": substeps,
        "scores": rows,
        "assumptions": {
            "wind": "recorded NED vectors, interval-end rows; default zero",
            "density_kg_m3": "recorded interval-end values; default 1.225",
            "gravity_ned_m_s2": np.asarray(standard.gravity).tolist(),
            "internal_initialization": "equilibrium at first interval command",
            "persistence": "all 13 initial state components held constant",
            "timestamps": "commands[i] end at state/time[i]; row zero unused",
        },
        "evidence_kind": sorted({r["kind"] for r in rows}),
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
