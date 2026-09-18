"""Dependency-free, offline HTML inspection of versioned trajectory files."""

from __future__ import annotations

import html
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from cascade.trajectory import load_trajectory


def _diagnostics(path, count, controls):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    matrices = {"commanded_action", "applied_action", "sensor_age_s", "sensor_valid"}
    for name, array in arrays.items():
        if array.ndim < 1 or array.shape[0] != count or array.dtype.kind not in "biuf":
            raise ValueError(f"diagnostic {name} has invalid shape or type")
        if name in matrices and array.ndim != 2:
            raise ValueError(f"diagnostic {name} must have shape (time, channels)")
        if name in ("reward", "airspeed_m_s", "heading_error_rad") and array.ndim != 1:
            raise ValueError(f"diagnostic {name} must have shape (time,)")
    valid = arrays.get("sensor_valid")
    if valid is not None and valid.dtype.kind != "b":
        raise ValueError("diagnostic sensor_valid must contain boolean values")
    ages = arrays.get("sensor_age_s")
    if ages is not None:
        if ages.dtype.kind not in "iuf" or np.any(ages < 0):
            raise ValueError("diagnostic sensor_age_s must contain nonnegative ages")
        if valid is not None and valid.shape != ages.shape:
            raise ValueError("diagnostic sensor_age_s must match the sensor_valid shape")
    commanded, applied = arrays.get("commanded_action"), arrays.get("applied_action")
    if (commanded is None) != (applied is None):
        raise ValueError("commanded_action and applied_action must both be present or absent")
    if commanded is not None:
        if commanded.shape != applied.shape:
            raise ValueError("commanded_action and applied_action must have matching shapes")
        if controls is not None:
            channels = controls.propeller.shape[-1] + controls.channel.shape[-1]
            if commanded.shape[1] != channels:
                raise ValueError("action diagnostic width must match the stored control channels")
    for name, array in arrays.items():
        if np.isfinite(array).all():
            continue
        if (
            name != "sensor_age_s"
            or valid is None
            or not np.all(np.isfinite(array) | (~valid & np.isposinf(array)))
        ):
            raise ValueError(f"diagnostic {name} contains nonfinite values")
    return arrays


def _flight(path, label, max_points):
    path = Path(path)
    states, controls, metadata = load_trajectory(path)
    position = np.asarray(states.rigid_body.position)
    if position.ndim != 2 or not len(position):
        raise ValueError("inspector requires one nonempty, unbatched trajectory")
    count = len(position)
    indices = np.unique(np.linspace(0, count - 1, min(count, max_points), dtype=int))
    time = metadata.get("t0_s", 0.0) + np.arange(count) * metadata["dt_s"]
    data = {
        "Altitude · m": -position[:, 2],
        "Ground speed · m/s": np.hypot.reduce(
            np.asarray(states.rigid_body.velocity, dtype=np.float64), axis=-1
        ),
    }
    for group, array, labels in (
        ("Body rate · rad/s", states.rigid_body.angular_velocity, ("roll", "pitch", "yaw")),
        ("Surface · rad", states.actuators.surface_deflection, None),
        ("Propeller · rad/s", states.actuators.propeller_speed, None),
        ("Separation", states.aero.separation, None),
    ):
        for i in range(array.shape[-1]):
            data[f"{group} {labels[i] if labels else i}"] = np.asarray(array[:, i])
    if controls is not None:
        for group, array in (
            ("Applied throttle", controls.propeller),
            ("Applied channel", controls.channel),
        ):
            for i in range(array.shape[-1]):
                data[f"{group} {i}"] = np.asarray(array[:, i])
    diagnostics_path = path.with_suffix(".diagnostics.npz")
    if diagnostics_path.exists():
        for name, array in _diagnostics(diagnostics_path, count, controls).items():
            if name in ("commanded_action", "applied_action", "sensor_age_s", "sensor_valid"):
                for i in range(array.shape[1]):
                    data[f"{name} {i}"] = array[:, i]
            elif name in ("reward", "airspeed_m_s", "heading_error_rad"):
                data[name] = array
    events = metadata.get("events", [])
    if not isinstance(events, list) or any(
        not isinstance(e, dict)
        or not isinstance(e.get("label"), str)
        or not isinstance(e.get("time_s"), (int, float))
        or isinstance(e.get("time_s"), bool)
        or not np.isfinite(e["time_s"])
        for e in events
    ):
        raise ValueError("events must have finite time_s and a string label")
    return {
        "label": label or path.stem,
        "time": time[indices].tolist(),
        "north": position[indices, 0].tolist(),
        "east": position[indices, 1].tolist(),
        "series": {
            k: [float(x) if np.isfinite(x) else None for x in np.asarray(v)[indices]]
            for k, v in data.items()
        },
        "events": events,
        "metadata": metadata,
        "original_samples": count,
    }


def trajectory_report(
    trajectory: str | Path,
    output: str | Path,
    *,
    compare: str | Path | None = None,
    title: str = "Flight inspector",
    labels: tuple[str, ...] | None = None,
    max_points: int = 2500,
) -> Path:
    """Write a standalone report with shared time cursor and optional second-flight overlay.

    No network requests, plotting dependencies, GL context or web server are required.
    Sidecar *.diagnostics.npz files from the experiment runner add command/sensor plots.
    Display downsampling preserves endpoints; original files remain the analysis source.
    Comparison uses stored timestamps without silently aligning or resampling flights.
    Sidecar action arrays must be paired with matching channel order/width. Sensor
    validity is boolean; ages are nonnegative, allowing positive infinity only for
    invalid measurements. The output cannot replace either trajectory or sidecar.
    """
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 2:
        raise ValueError("max_points must be an integer >= 2")
    paths = [trajectory] if compare is None else [trajectory, compare]
    if labels is not None and (
        isinstance(labels, (str, bytes))
        or len(labels) != len(paths)
        or any(not isinstance(label, str) for label in labels)
    ):
        raise ValueError("provide one string label for each trajectory")
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    output = Path(output)
    protected = [
        p for path in paths for p in (Path(path), Path(path).with_suffix(".diagnostics.npz"))
    ]
    if output.resolve() in {path.resolve() for path in protected} or (
        output.exists() and any(path.exists() and output.samefile(path) for path in protected)
    ):
        raise ValueError("report must not overwrite an input trajectory or diagnostics file")
    flights = [_flight(p, labels[i] if labels else None, max_points) for i, p in enumerate(paths)]
    payload = (
        json.dumps(flights, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    template = files("cascade.viz").joinpath("_inspector.html").read_text()
    output.write_text(
        template.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)
    )
    return output
