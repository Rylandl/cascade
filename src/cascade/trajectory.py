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

import jax
import jax.numpy as jnp
import numpy as np

from cascade.canonical import (
    CANONICAL_STATE_SCHEMA,
    rigid_body_from_canonical,
    rigid_body_to_canonical,
)
from cascade.state import ActuatorState, AeroState, AircraftState, ControlInput

TRAJECTORY_SCHEMA = "cascade_trajectory_v1"


def save_trajectory(
    path: str | Path,
    trajectory: AircraftState,
    dt: float,
    *,
    controls: ControlInput | None = None,
    stamp: dict[str, Any] | None = None,
    **metadata: Any,
) -> Path:
    """Write a time-major trajectory sampled every ``dt`` seconds; returns the path."""

    steps = int(trajectory.rigid_body.position.shape[0])
    canonical = np.asarray(jax.vmap(rigid_body_to_canonical)(trajectory.rigid_body))
    arrays = {
        "time_s": np.arange(steps) * float(dt),
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
        "dt_s": float(dt),
        "steps": steps,
        "stamp": stamp,
        **metadata,
    }
    arrays["metadata_json"] = np.array(json.dumps(header, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_trajectory(path: str | Path) -> tuple[AircraftState, ControlInput | None, dict[str, Any]]:
    """Read a trajectory file back into native time-major states, controls, and metadata."""

    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("schema") != TRAJECTORY_SCHEMA:
            raise ValueError(f"unsupported trajectory schema {metadata.get('schema')!r}")
        trajectory = AircraftState(
            rigid_body=rigid_body_from_canonical(jnp.asarray(data["canonical_state"])),
            actuators=ActuatorState(
                surface_deflection=jnp.asarray(data["surface_deflection_rad"]),
                propeller_speed=jnp.asarray(data["propeller_speed_rad_s"]),
            ),
            aero=AeroState(separation=jnp.asarray(data["separation"])),
        )
        controls = None
        if "control_propeller" in data:
            controls = ControlInput(
                propeller=jnp.asarray(data["control_propeller"]),
                channel=jnp.asarray(data["control_channel"]),
            )
    return trajectory, controls, metadata


__all__ = ["TRAJECTORY_SCHEMA", "load_trajectory", "save_trajectory"]
