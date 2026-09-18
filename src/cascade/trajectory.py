"""Trajectory files: a versioned, self-describing record of a flight for replay and comparison.

:func:`save_trajectory` writes a time-major :class:`cascade.state.AircraftState` (and the
controls that produced it, when given) to a compressed ``.npz`` with the rigid body in the
canonical NWU/FLU 13-vector, the actuator and separation states, the time axis, and a JSON
metadata block carrying the schema version, the timestep, and a provenance stamp.
:func:`load_trajectory` reads it back into native states. The same file format is what a
flight log looks like once a loader has put it into canonical state, so simulated and flown
trajectories compare on equal terms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from cascade.canonical import (
    CANONICAL_STATE_SCHEMA,
    rigid_body_from_canonical,
    rigid_body_to_canonical,
)
from cascade.state import ActuatorState, AeroState, AircraftState, ControlInput

TRAJECTORY_SCHEMA = "cascade_trajectory_v1"
_RESERVED_METADATA = {"schema", "state_schema", "dt_s", "t0_s", "steps", "stamp"}
_STATE_ARRAYS = (
    "canonical_state",
    "surface_deflection_rad",
    "propeller_speed_rad_s",
    "separation",
)


def _finite_scalar(name: str, value: Any, *, positive: bool = False) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(array)
    if positive and scalar <= 0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _finite_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real numbers")
    return array


def _validate_arrays(arrays: dict[str, np.ndarray], header: dict[str, Any]) -> None:
    """Validate the on-disk contract before creating JAX arrays or writing a file."""
    if header.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError(f"unsupported trajectory schema {header.get('schema')!r}")
    if header.get("state_schema") != CANONICAL_STATE_SCHEMA:
        raise ValueError(f"unsupported state schema {header.get('state_schema')!r}")
    if header.get("stamp") is not None and not isinstance(header["stamp"], dict):
        raise ValueError("stamp must be a JSON object or null")
    json.dumps(header, allow_nan=False)
    dt = _finite_scalar("dt_s", header.get("dt_s"), positive=True)
    # Original v1 files always started at zero and did not carry t0_s.
    t0 = _finite_scalar("t0_s", header.get("t0_s", 0.0))
    steps = header.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    for name in ("time_s", *_STATE_ARRAYS):
        if name not in arrays:
            raise ValueError(f"trajectory is missing {name}")
    for name, array in arrays.items():
        _finite_array(name, array)
    canonical = arrays["canonical_state"]
    if canonical.ndim < 2 or canonical.shape[0] != steps or canonical.shape[-1] != 13:
        raise ValueError("canonical_state must have shape (steps, *batch, 13)")
    quaternion = canonical[..., 6:10]
    # Bound components before squaring so even malformed huge finite values are safe.
    if np.any(np.abs(quaternion) > 1.0001) or not np.allclose(
        np.linalg.norm(quaternion.astype(np.float64), axis=-1), 1.0, rtol=0.0, atol=1e-4
    ):
        raise ValueError("canonical_state quaternion norms must be within 1e-4 of one")
    sample_shape = canonical.shape[:-1]
    for name in _STATE_ARRAYS[1:]:
        if arrays[name].ndim != canonical.ndim or arrays[name].shape[:-1] != sample_shape:
            raise ValueError(f"{name} must have shape (steps, *batch, components)")
    if arrays["separation"].shape != arrays["surface_deflection_rad"].shape:
        raise ValueError("separation and surface_deflection_rad must have the same shape")
    has_propeller = "control_propeller" in arrays
    has_channel = "control_channel" in arrays
    if has_propeller != has_channel:
        raise ValueError("control_propeller and control_channel must both be present or absent")
    if has_propeller:
        if arrays["control_propeller"].shape != arrays["propeller_speed_rad_s"].shape:
            raise ValueError("control_propeller must match the propeller_speed_rad_s shape")
        if (
            arrays["control_channel"].ndim != canonical.ndim
            or arrays["control_channel"].shape[:-1] != sample_shape
        ):
            raise ValueError("control_channel must have shape (steps, *batch, channels)")
    times = arrays["time_s"]
    if times.shape != (steps,):
        raise ValueError("time_s must have shape (steps,)")
    expected = t0 + np.arange(steps, dtype=np.float64) * dt
    # Permit float32 log timestamps without accepting a different timestep or time origin.
    tolerance = 8 * np.finfo(times.dtype if times.dtype.kind == "f" else np.float64).eps
    if not np.all(np.isfinite(expected)) or not np.allclose(
        times, expected, rtol=tolerance, atol=tolerance * dt
    ):
        raise ValueError("time_s must equal t0_s + arange(steps) * dt_s")
    if steps > 1 and np.any(times[1:] <= times[:-1]):
        raise ValueError("time_s must be strictly increasing")


def save_trajectory(
    path: str | Path,
    trajectory: AircraftState,
    dt: float,
    *,
    controls: ControlInput | None = None,
    stamp: dict[str, Any] | None = None,
    t0_s: float = 0.0,
    **metadata: Any,
) -> Path:
    """Write a finite trajectory with shape ``(steps, *batch, components)``.

    ``t0_s`` is the timestamp of the *first stored state*, not the simulation start.
    For :func:`cascade.integration.rollout` begun at time zero, pass ``t0_s=dt``:
    that function stores post-step states. The compatibility default is zero.
    When supplied, ``controls[i]`` is the control applied during the interval ending
    at ``t0_s + i * dt``. Include no initial state when pairing rollout controls.

    Schema, timing, step count, and provenance fields cannot be replaced by custom
    metadata. Metadata must be JSON serializable with finite numeric values. Empty
    trajectories are supported. Validation is host-side and cannot be JIT compiled.
    Quaternion norms must be within ``1e-4`` of one; scaled or zero quaternions are
    not valid trajectory orientations. Loading normalizes accepted roundoff error.
    Returns the actual filename, including NumPy's appended ``.npz`` when needed.
    """
    reserved = _RESERVED_METADATA.intersection(metadata)
    if reserved:
        raise ValueError(f"reserved trajectory metadata: {', '.join(sorted(reserved))}")
    dt = _finite_scalar("dt", dt, positive=True)
    t0_s = _finite_scalar("t0_s", t0_s)
    position = _finite_array("position", trajectory.rigid_body.position)
    if position.ndim < 2 or position.shape[-1] != 3:
        raise ValueError("position must have shape (steps, *batch, 3)")
    for name, width in (("velocity", 3), ("attitude", 4), ("angular_velocity", 3)):
        array = _finite_array(name, getattr(trajectory.rigid_body, name))
        if array.shape != (*position.shape[:-1], width):
            raise ValueError(f"{name} must have shape {(*position.shape[:-1], width)}")
    steps = int(position.shape[0])
    canonical = np.asarray(rigid_body_to_canonical(trajectory.rigid_body))
    arrays = {
        "time_s": t0_s + np.arange(steps, dtype=np.float64) * dt,
        "canonical_state": canonical,
        "surface_deflection_rad": np.asarray(trajectory.actuators.surface_deflection),
        "propeller_speed_rad_s": np.asarray(trajectory.actuators.propeller_speed),
        "separation": np.asarray(trajectory.aero.separation),
    }
    if controls is not None:
        arrays["control_propeller"] = np.asarray(controls.propeller)
        arrays["control_channel"] = np.asarray(controls.channel)
    header = {
        "schema": TRAJECTORY_SCHEMA,
        "state_schema": CANONICAL_STATE_SCHEMA,
        "dt_s": dt,
        "t0_s": t0_s,
        "steps": steps,
        "stamp": stamp,
        **metadata,
    }
    _validate_arrays(arrays, header)
    arrays["metadata_json"] = np.array(json.dumps(header, sort_keys=True, allow_nan=False))
    path = Path(path)
    if not str(path).endswith(".npz"):
        path = Path(str(path) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_trajectory(path: str | Path) -> tuple[AircraftState, ControlInput | None, dict[str, Any]]:
    """Read and validate native time-major states, controls, and metadata.

    Requires the supported trajectory and canonical coordinate schemas and consistent,
    finite arrays. Original v1 files without ``t0_s`` retain a zero time origin. The
    returned metadata always includes ``t0_s`` so timestamps can be reconstructed.
    JAX's active x64 setting determines whether float64 state arrays retain precision.
    Rejects values that become nonfinite in the active JAX dtype. Quaternion norms
    must be within ``1e-4`` of one and accepted attitudes are normalized on loading.
    """

    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data or data["metadata_json"].shape != ():
            raise ValueError("metadata_json must be a scalar JSON object")
        metadata = json.loads(str(data["metadata_json"]))
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must contain a JSON object")
        metadata.setdefault("t0_s", 0.0)
        arrays = {name: data[name] for name in data.files if name != "metadata_json"}
        _validate_arrays(arrays, metadata)
        names = list(_STATE_ARRAYS)
        if "control_propeller" in arrays:
            names.extend(("control_propeller", "control_channel"))
        converted = {name: jnp.asarray(arrays[name]) for name in names}
        for name, array in converted.items():
            _finite_array(f"{name} after JAX dtype conversion", array)
        trajectory = AircraftState(
            rigid_body=rigid_body_from_canonical(converted["canonical_state"]),
            actuators=ActuatorState(
                surface_deflection=converted["surface_deflection_rad"],
                propeller_speed=converted["propeller_speed_rad_s"],
            ),
            aero=AeroState(separation=converted["separation"]),
        )
        controls = None
        if "control_propeller" in arrays:
            controls = ControlInput(
                propeller=converted["control_propeller"],
                channel=converted["control_channel"],
            )
    return trajectory, controls, metadata


__all__ = ["TRAJECTORY_SCHEMA", "load_trajectory", "save_trajectory"]
