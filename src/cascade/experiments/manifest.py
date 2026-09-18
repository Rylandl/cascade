"""Portable, content-addressed scenario definitions; no executable objects in JSON."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import jax
import numpy as np

from cascade.env.episode import EpisodeConfig
from cascade.spec import AircraftSpec

SCHEMA = "cascade_experiment_v1"


def _registry():
    from cascade.env import faults, missions, sensors, tasks, weather
    from cascade.integration import euler_step, rk4_step

    types = {"EpisodeConfig": EpisodeConfig}
    for module in (faults, missions, sensors, tasks, weather):
        for name, value in vars(module).items():
            if (
                isinstance(value, type)
                and value.__module__ == module.__name__
                and (hasattr(value, "_fields") or dataclasses.is_dataclass(value))
            ):
                types[name] = value
    return types, {"rk4_step": rk4_step, "euler_step": euler_step}


def encode(value: Any) -> Any:
    """Encode only explicit data and known pure integrators, never pickle or import paths."""
    if isinstance(value, AircraftSpec):
        return {"type": "AircraftSpec", "fields": value.to_dict()}
    if dataclasses.is_dataclass(value) or (isinstance(value, tuple) and hasattr(value, "_fields")):
        types, _ = _registry()
        if types.get(type(value).__name__) is not type(value):
            raise TypeError(f"unsupported manifest type {type(value).__name__!r}")
    if dataclasses.is_dataclass(value):
        return {
            "type": type(value).__name__,
            "fields": {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            "type": type(value).__name__,
            "fields": {k: encode(v) for k, v in value._asdict().items()},
        }
    if callable(value):
        _, integrators = _registry()
        for name, fn in integrators.items():
            if value is fn:
                return {"integrator": name}
        raise TypeError("portable manifests support only rk4_step and euler_step")
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {"tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "shape"):
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError("manifest arrays must be numeric")
        # Fault schedules use +inf for never; encode it explicitly, not nonstandard JSON.
        return {
            "array": np.where(np.isposinf(array), 0, array).tolist(),
            "positive_infinity": np.isposinf(array).tolist(),
            "dtype": str(array.dtype),
        }
    raise TypeError(f"unsupported manifest value {type(value).__name__}")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    types, integrators = _registry()
    if set(value) == {"integrator"}:
        if value["integrator"] not in integrators:
            raise ValueError("unsupported integrator")
        return integrators[value["integrator"]]
    if set(value) == {"array", "positive_infinity", "dtype"}:
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("unsupported array dtype")
        raw = np.asarray(value["array"])
        mask = np.asarray(value["positive_infinity"])
        if raw.dtype.kind not in "biuf" or not np.isfinite(raw).all():
            raise ValueError("array payload must contain finite numbers")
        if mask.dtype.kind != "b" and mask.size:
            raise ValueError("infinity mask must contain booleans")
        mask = mask.astype(bool)
        if dtype.kind in "iub":
            if mask.any():
                raise ValueError("integer arrays cannot contain infinity")
            bounds = (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
            if np.any(raw < bounds[0]) or np.any(raw > bounds[1]) or np.any(raw != np.floor(raw)):
                raise ValueError("array values are not representable by their integer dtype")
        with np.errstate(over="ignore", invalid="ignore"):
            array = raw.astype(dtype)
        if not np.isfinite(array).all():
            raise ValueError("array values overflow their declared dtype")
        if array.shape != mask.shape:
            raise ValueError("infinity mask shape mismatch")
        return np.where(mask, np.inf, array) if mask.any() else array
    if set(value) == {"tuple"}:
        return tuple(decode(v) for v in value["tuple"])
    if set(value) == {"type", "fields"}:
        if value["type"] == "AircraftSpec":
            return AircraftSpec.from_dict(value["fields"])
        if value["type"] not in types:
            raise ValueError(f"unsupported manifest type {value['type']!r}")
        return types[value["type"]](**{k: decode(v) for k, v in value["fields"].items()})
    return {k: decode(v) for k, v in value.items()}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def valid_name(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("name must be 1–80 safe letters, digits, dots, underscores or hyphens")


def _numeric(value, name, shape, *, positive=False, nonnegative=False, allow_infinity=False):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have numeric shape {shape}")
    finite = np.isfinite(array) | (np.isposinf(array) if allow_infinity else False)
    if not finite.all() or (positive and np.any(array <= 0)) or (nonnegative and np.any(array < 0)):
        raise ValueError(f"invalid values for {name}")
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    with np.errstate(over="ignore", invalid="ignore"):
        native = array.astype(dtype)
    if np.any(np.isfinite(array) & ~np.isfinite(native)):
        raise ValueError(f"{name} overflows the active JAX dtype")
    if positive and np.any(native <= 0):
        raise ValueError(f"{name} must stay positive in the active JAX dtype")
    return array


def _validate_task(task):
    from cascade.env import missions, tasks

    static = (tasks.TrackingTask, tasks.HoverTask, tasks.TransitionTask)
    if isinstance(task, static):
        for name, value in task._asdict().items():
            _numeric(
                value,
                name,
                (3,) if name == "position_ned" else (),
                positive=name in {"airspeed_m_s", "cruise_speed_m_s"},
                nonnegative=name.endswith("_weight"),
            )
        return
    if not isinstance(
        task, (missions.ScheduledTrackingTask, missions.WaypointTask, missions.OrbitTask)
    ):
        raise ValueError("scenario task must be a supported static task or mission")
    if not isinstance(task.template, tasks.TrackingTask):
        raise ValueError("mission template must be a TrackingTask")
    _validate_task(task.template)
    weights = {
        name: value for name, value in task.template._asdict().items() if name.endswith("_weight")
    }
    # Re-run the validated host builder without replacing the stored arrays/dtypes: hashes
    # describe portable input data, even when the executing JAX precision differs.
    if isinstance(task, missions.ScheduledTrackingTask):
        missions.scheduled_tracking_task(
            task.times_s, task.airspeed_m_s, task.altitude_m, task.heading_rad, **weights
        )
    elif isinstance(task, missions.WaypointTask):
        missions.waypoint_task(
            task.times_s,
            task.positions_ned,
            task.airspeed_m_s,
            lookahead_s=task.lookahead_s,
            position_weight=task.position_weight,
            position_scale_m=task.position_scale_m,
            **weights,
        )
    else:
        direction = _numeric(task.direction, "orbit direction", ())
        if direction not in (-1, 1):
            raise ValueError("orbit direction must be -1 or +1")
        missions.orbit_task(
            task.center_ned,
            task.radius_m,
            task.airspeed_m_s,
            clockwise=bool(direction > 0),
            initial_phase_rad=task.initial_phase_rad,
            capture_distance_m=task.capture_distance_m,
            position_weight=task.position_weight,
            **weights,
        )


def _validate_inputs(scenario):
    from cascade.env.faults import FaultSchedule
    from cascade.env.sensors import SensorNoise
    from cascade.env.weather import WeatherCondition

    scenario.aircraft.validate()
    _validate_task(scenario.task)
    if scenario.noise is not None:
        if not isinstance(scenario.noise, SensorNoise):
            raise ValueError("noise must be SensorNoise or None")
        for name, value in scenario.noise._asdict().items():
            _numeric(value, name, (), nonnegative=True)
    if scenario.weather is not None:
        if not isinstance(scenario.weather, WeatherCondition):
            raise ValueError("weather must be WeatherCondition or None")
        for name, value in scenario.weather._asdict().items():
            _numeric(
                value,
                name,
                (3,) if name == "gust_direction_ned" else (),
                positive=name in {"roughness_length_m", "gust_duration_s"},
                nonnegative=name in {"wind_speed_m_s", "turbulence_wind_20ft_m_s", "gust_start_s"},
            )
        if float(scenario.weather.roughness_length_m) >= 10:
            raise ValueError("weather roughness length must be below the 10 m reference height")
        if not np.isclose(np.linalg.norm(scenario.weather.gust_direction_ned), 1, atol=1e-4):
            raise ValueError("weather gust direction must be a unit vector")
    if scenario.faults is not None:
        if not isinstance(scenario.faults, FaultSchedule):
            raise ValueError("faults must be FaultSchedule or None")
        for name, value in scenario.faults._asdict().items():
            count = (
                len(scenario.aircraft.surfaces)
                if name.startswith("surface")
                else len(scenario.aircraft.propellers)
            )
            array = _numeric(
                value, name, (count,), nonnegative="time" in name, allow_infinity="time" in name
            )
            if name.endswith("sign") and not np.isin(array, [-1, 1]).all():
                raise ValueError("hardover signs must be -1 or +1")
            if name.endswith("fraction") and np.any((array < 0) | (array > 1)):
                raise ValueError("partial-power fractions must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    aircraft: AircraftSpec
    task: Any
    config: EpisodeConfig = EpisodeConfig()
    seeds: tuple[int, ...] = (0,)
    split: str = "evaluation"
    weather: Any = None
    noise: Any = None
    faults: Any = None

    def __post_init__(self):
        valid_name(self.name)
        if self.split not in {"training", "validation", "evaluation"}:
            raise ValueError("split must be training, validation or evaluation")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("scenario seeds must be nonempty and unique")
        if any(
            isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in self.seeds
        ):
            raise ValueError("seeds must be integers in [0, 2**32)")
        if not isinstance(self.aircraft, AircraftSpec) or not isinstance(
            self.config, EpisodeConfig
        ):
            raise TypeError("scenario requires AircraftSpec and EpisodeConfig")
        _validate_inputs(self)
        canonical_json(self.to_dict())

    def to_dict(self):
        return {f.name: encode(getattr(self, f.name)) for f in dataclasses.fields(self)}


@dataclasses.dataclass(frozen=True)
class Experiment:
    name: str
    scenarios: tuple[Scenario, ...]
    description: str = ""

    def __post_init__(self):
        valid_name(self.name)
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        if any(not isinstance(s, Scenario) for s in self.scenarios):
            raise ValueError("experiment scenarios must be Scenario instances")
        if not self.scenarios or len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("experiment must have scenarios with unique names")
        splits: dict[str, set[int]] = {}
        for scenario in self.scenarios:
            splits.setdefault(scenario.split, set()).update(scenario.seeds)
        for a, seeds in splits.items():
            if any(seeds & other for b, other in splits.items() if a != b):
                raise ValueError("seeds must be disjoint across training/validation/evaluation")

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "name": self.name,
            "description": self.description,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    @property
    def sha256(self):
        return content_hash(self.to_dict())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        path.write_text(
            json.dumps(
                {"sha256": content_hash(payload), "experiment": payload}, indent=2, allow_nan=False
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Experiment:
        wrapper = json.loads(Path(path).read_text())
        if not isinstance(wrapper, dict) or set(wrapper) != {"sha256", "experiment"}:
            raise ValueError("malformed experiment wrapper")
        data = wrapper["experiment"]
        if not isinstance(data, dict) or set(data) != {
            "schema",
            "name",
            "description",
            "scenarios",
        }:
            raise ValueError("malformed experiment fields")
        if data.get("schema") != SCHEMA or wrapper.get("sha256") != content_hash(data):
            raise ValueError("experiment schema or content hash mismatch")
        try:
            return cls(
                data["name"],
                tuple(Scenario(**{k: decode(v) for k, v in s.items()}) for s in data["scenarios"]),
                data["description"],
            )
        except (TypeError, KeyError, OverflowError) as error:
            raise ValueError(f"malformed experiment: {error}") from error
