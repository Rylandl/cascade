from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import jax.numpy as jnp
import tomli_w

from cascade.model import (
    ActuatorModel,
    AircraftModel,
    BodyModel,
    DragCoefficients,
    LateralCoefficients,
    LongitudinalCoefficients,
    PropellerModel,
    SurfaceModel,
    validate_model,
    zero_body,
)

SCHEMA_VERSION = 2


class SpecError(ValueError):
    """Raised when an aircraft specification is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    name: str
    position_m: tuple[float, float, float]
    body_from_surface: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    area_m2: float
    chord_m: float
    lift_coefficient_zero: float
    lift_curve_slope_rad: float
    drag_coefficient_zero: float
    induced_drag_factor: float
    moment_coefficient_zero: float
    moment_coefficient_alpha_rad: float
    stall_angle_rad: float
    stall_width_rad: float
    normal_force_coefficient: float
    edge_drag_coefficient: float
    span_drag_coefficient: float
    separation_time_constant_s: float
    reattachment_time_constant_s: float
    all_moving_fraction: float
    flap_effectiveness: float
    moment_coefficient_flap_rad: float
    drag_coefficient_flap_rad2: float
    control_map_rad: tuple[float, ...]
    actuator_bias_rad: float
    actuator_limit_rad: float
    actuator_time_constant_s: float
    actuator_rate_limit_rad_s: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        values = dict(data)
        values["position_m"] = _vector(values["position_m"], 3, "surface.position_m")
        values["body_from_surface"] = _matrix3(values["body_from_surface"])
        values["control_map_rad"] = tuple(float(value) for value in values["control_map_rad"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            field: _lists(value)
            for field, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class PropellerSpec:
    name: str
    position_m: tuple[float, float, float]
    direction_body: tuple[float, float, float]
    diameter_m: float
    thrust_map: tuple[tuple[float, float, float], tuple[float, float, float]]
    torque_coefficient_static: float
    spin_direction: float
    slipstream_weights: tuple[float, ...]
    speed_min_rad_s: float
    speed_max_rad_s: float
    time_constant_s: float
    acceleration_limit_rad_s2: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        values = dict(data)
        values["position_m"] = _vector(values["position_m"], 3, "propeller.position_m")
        values["direction_body"] = _vector(values["direction_body"], 3, "propeller.direction_body")
        rows = tuple(_vector(row, 3, "propeller.thrust_map row") for row in values["thrust_map"])
        if len(rows) != 2:
            raise SpecError("propeller.thrust_map must contain two rows (n and n squared)")
        values["thrust_map"] = rows
        values["slipstream_weights"] = tuple(float(value) for value in values["slipstream_weights"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            field: _lists(value)
            for field, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class LongitudinalCoefficientSpec:
    """Lift or pitching moment: ``zero + alpha_rad * a + q * (c q / 2 V_a) + elevator_rad * de``."""

    zero: float
    alpha_rad: float
    q: float
    elevator_rad: float


@dataclass(frozen=True, slots=True)
class DragCoefficientSpec:
    zero: float
    alpha_rad: float
    alpha_sq_rad2: float
    beta_rad: float
    beta_sq_rad2: float
    q: float
    elevator_sq_rad2: float


@dataclass(frozen=True, slots=True)
class LateralCoefficientSpec:
    """Side force, rolling moment, or yawing moment in sideslip, rates, and controls."""

    zero: float
    beta_rad: float
    p: float
    r: float
    aileron_rad: float
    rudder_rad: float


@dataclass(frozen=True, slots=True)
class BodySpec:
    """Whole-aircraft coefficient table about the center of mass, in the published convention.

    ``deflection_map`` has one row per generalized control (aileron, elevator, rudder) and one
    column per surface, forming those angles from the physical surface deflections.
    """

    lift: LongitudinalCoefficientSpec
    drag: DragCoefficientSpec
    side: LateralCoefficientSpec
    roll: LateralCoefficientSpec
    pitch: LongitudinalCoefficientSpec
    yaw: LateralCoefficientSpec
    stall_angle_rad: float
    stall_width_rad: float
    normal_force_coefficient: float
    pitch_flat_plate: float
    deflection_map: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        rows = tuple(tuple(float(value) for value in row) for row in data["deflection_map"])
        if len(rows) != 3:
            raise SpecError("body.deflection_map must contain three rows")
        return cls(
            lift=LongitudinalCoefficientSpec(**_floats(data["lift"])),
            drag=DragCoefficientSpec(**_floats(data["drag"])),
            side=LateralCoefficientSpec(**_floats(data["side"])),
            roll=LateralCoefficientSpec(**_floats(data["roll"])),
            pitch=LongitudinalCoefficientSpec(**_floats(data["pitch"])),
            yaw=LateralCoefficientSpec(**_floats(data["yaw"])),
            stall_angle_rad=float(data["stall_angle_rad"]),
            stall_width_rad=float(data["stall_width_rad"]),
            normal_force_coefficient=float(data["normal_force_coefficient"]),
            pitch_flat_plate=float(data["pitch_flat_plate"]),
            deflection_map=rows,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lift": asdict(self.lift),
            "drag": asdict(self.drag),
            "side": asdict(self.side),
            "roll": asdict(self.roll),
            "pitch": asdict(self.pitch),
            "yaw": asdict(self.yaw),
            "stall_angle_rad": self.stall_angle_rad,
            "stall_width_rad": self.stall_width_rad,
            "normal_force_coefficient": self.normal_force_coefficient,
            "pitch_flat_plate": self.pitch_flat_plate,
            "deflection_map": _lists(self.deflection_map),
        }

    def to_model(self) -> BodyModel:
        return BodyModel(
            lift=_longitudinal(self.lift),
            drag=DragCoefficients(
                zero=jnp.asarray(self.drag.zero),
                alpha=jnp.asarray(self.drag.alpha_rad),
                alpha_sq=jnp.asarray(self.drag.alpha_sq_rad2),
                beta=jnp.asarray(self.drag.beta_rad),
                beta_sq=jnp.asarray(self.drag.beta_sq_rad2),
                q=jnp.asarray(self.drag.q),
                elevator_sq=jnp.asarray(self.drag.elevator_sq_rad2),
            ),
            side=_lateral(self.side),
            roll=_lateral(self.roll),
            pitch=_longitudinal(self.pitch),
            yaw=_lateral(self.yaw),
            stall_angle=jnp.asarray(self.stall_angle_rad),
            stall_width=jnp.asarray(self.stall_width_rad),
            normal_force_coefficient=jnp.asarray(self.normal_force_coefficient),
            pitch_flat_plate=jnp.asarray(self.pitch_flat_plate),
            deflection_map=jnp.asarray(self.deflection_map),
        )


@dataclass(frozen=True, slots=True)
class AircraftSpec:
    name: str
    description: str
    mass_kg: float
    inertia_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    reference_area_m2: float
    reference_chord_m: float
    reference_span_m: float
    control_channels: tuple[str, ...]
    surfaces: tuple[SurfaceSpec, ...]
    propellers: tuple[PropellerSpec, ...]
    body: BodySpec | None = None
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> Self:
        if self.schema_version != SCHEMA_VERSION:
            raise SpecError(
                f"unsupported schema version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.name.strip():
            raise SpecError("aircraft name must not be empty")
        if not self.surfaces:
            raise SpecError("at least one aerodynamic surface is required")
        _require_unique(self.control_channels, "control channel")
        _require_unique((surface.name for surface in self.surfaces), "surface")
        _require_unique((propeller.name for propeller in self.propellers), "propeller")
        for surface in self.surfaces:
            if len(surface.control_map_rad) != len(self.control_channels):
                raise SpecError(
                    f"surface {surface.name!r} has {len(surface.control_map_rad)} control-map "
                    f"entries; expected {len(self.control_channels)}"
                )
        for propeller in self.propellers:
            if len(propeller.slipstream_weights) != len(self.surfaces):
                raise SpecError(
                    f"propeller {propeller.name!r} has {len(propeller.slipstream_weights)} "
                    f"slipstream weights; expected {len(self.surfaces)}"
                )
        if self.body is not None:
            for row in self.body.deflection_map:
                if len(row) != len(self.surfaces):
                    raise SpecError(
                        f"body.deflection_map rows must have {len(self.surfaces)} entries"
                    )
        return self

    def to_model(self) -> AircraftModel:
        """Compile the named host-side specification into a numerical JAX PyTree."""

        self.validate()
        surfaces = self.surfaces
        propellers = self.propellers
        n_propeller, n_surface = len(propellers), len(surfaces)
        surface_model = SurfaceModel(
            position=jnp.asarray([surface.position_m for surface in surfaces]),
            body_from_surface=jnp.asarray([surface.body_from_surface for surface in surfaces]),
            area=jnp.asarray([surface.area_m2 for surface in surfaces]),
            chord=jnp.asarray([surface.chord_m for surface in surfaces]),
            lift_coefficient_zero=jnp.asarray(
                [surface.lift_coefficient_zero for surface in surfaces]
            ),
            lift_curve_slope=jnp.asarray([surface.lift_curve_slope_rad for surface in surfaces]),
            drag_coefficient_zero=jnp.asarray(
                [surface.drag_coefficient_zero for surface in surfaces]
            ),
            induced_drag_factor=jnp.asarray([surface.induced_drag_factor for surface in surfaces]),
            moment_coefficient_zero=jnp.asarray(
                [surface.moment_coefficient_zero for surface in surfaces]
            ),
            moment_coefficient_alpha=jnp.asarray(
                [surface.moment_coefficient_alpha_rad for surface in surfaces]
            ),
            stall_angle=jnp.asarray([surface.stall_angle_rad for surface in surfaces]),
            stall_width=jnp.asarray([surface.stall_width_rad for surface in surfaces]),
            normal_force_coefficient=jnp.asarray(
                [surface.normal_force_coefficient for surface in surfaces]
            ),
            edge_drag_coefficient=jnp.asarray(
                [surface.edge_drag_coefficient for surface in surfaces]
            ),
            span_drag_coefficient=jnp.asarray(
                [surface.span_drag_coefficient for surface in surfaces]
            ),
            separation_time_constant=jnp.asarray(
                [surface.separation_time_constant_s for surface in surfaces]
            ),
            reattachment_time_constant=jnp.asarray(
                [surface.reattachment_time_constant_s for surface in surfaces]
            ),
            all_moving_fraction=jnp.asarray([surface.all_moving_fraction for surface in surfaces]),
            flap_effectiveness=jnp.asarray([surface.flap_effectiveness for surface in surfaces]),
            moment_coefficient_flap=jnp.asarray(
                [surface.moment_coefficient_flap_rad for surface in surfaces]
            ),
            drag_coefficient_flap=jnp.asarray(
                [surface.drag_coefficient_flap_rad2 for surface in surfaces]
            ),
        )
        propeller_model = PropellerModel(
            position=jnp.asarray([propeller.position_m for propeller in propellers]).reshape(
                n_propeller, 3
            ),
            direction=jnp.asarray([propeller.direction_body for propeller in propellers]).reshape(
                n_propeller, 3
            ),
            diameter=jnp.asarray([propeller.diameter_m for propeller in propellers]),
            torque_coefficient=jnp.asarray(
                [propeller.torque_coefficient_static for propeller in propellers]
            ),
            spin_direction=jnp.asarray([propeller.spin_direction for propeller in propellers]),
            thrust_map=jnp.asarray([propeller.thrust_map for propeller in propellers]).reshape(
                n_propeller, 2, 3
            ),
            slipstream_map=jnp.asarray(
                [propeller.slipstream_weights for propeller in propellers]
            ).reshape(n_propeller, n_surface),
        )
        actuator_model = ActuatorModel(
            surface_map=jnp.asarray([surface.control_map_rad for surface in surfaces]).reshape(
                n_surface, len(self.control_channels)
            ),
            surface_bias=jnp.asarray([surface.actuator_bias_rad for surface in surfaces]),
            surface_limit=jnp.asarray([surface.actuator_limit_rad for surface in surfaces]),
            surface_time_constant=jnp.asarray(
                [surface.actuator_time_constant_s for surface in surfaces]
            ),
            surface_rate_limit=jnp.asarray(
                [surface.actuator_rate_limit_rad_s for surface in surfaces]
            ),
            propeller_speed_min=jnp.asarray(
                [propeller.speed_min_rad_s for propeller in propellers]
            ),
            propeller_speed_max=jnp.asarray(
                [propeller.speed_max_rad_s for propeller in propellers]
            ),
            propeller_time_constant=jnp.asarray(
                [propeller.time_constant_s for propeller in propellers]
            ),
            propeller_acceleration_limit=jnp.asarray(
                [propeller.acceleration_limit_rad_s2 for propeller in propellers]
            ),
        )
        inertia = jnp.asarray(self.inertia_kg_m2)
        return validate_model(
            AircraftModel(
                mass=jnp.asarray(self.mass_kg),
                inertia=inertia,
                inertia_inverse=jnp.linalg.inv(inertia),
                reference_area=jnp.asarray(self.reference_area_m2),
                reference_chord=jnp.asarray(self.reference_chord_m),
                reference_span=jnp.asarray(self.reference_span_m),
                surfaces=surface_model,
                propellers=propeller_model,
                actuators=actuator_model,
                body=zero_body(n_surface) if self.body is None else self.body.to_model(),
            )
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        try:
            rigid_body = data["rigid_body"]
            reference = data["reference"]
            controls = data["controls"]
            spec = cls(
                schema_version=int(data["schema_version"]),
                name=str(data["name"]),
                description=str(data.get("description", "")),
                mass_kg=float(rigid_body["mass_kg"]),
                inertia_kg_m2=_matrix3(rigid_body["inertia_kg_m2"]),
                reference_area_m2=float(reference["area_m2"]),
                reference_chord_m=float(reference["chord_m"]),
                reference_span_m=float(reference["span_m"]),
                control_channels=tuple(str(value) for value in controls["channels"]),
                surfaces=tuple(SurfaceSpec.from_dict(value) for value in data["surfaces"]),
                propellers=tuple(
                    PropellerSpec.from_dict(value) for value in data.get("propellers", [])
                ),
                body=BodySpec.from_dict(data["body"]) if "body" in data else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SpecError(f"invalid aircraft specification: {error}") from error
        return spec.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "rigid_body": {
                "mass_kg": self.mass_kg,
                "inertia_kg_m2": _lists(self.inertia_kg_m2),
            },
            "reference": {
                "area_m2": self.reference_area_m2,
                "chord_m": self.reference_chord_m,
                "span_m": self.reference_span_m,
            },
            "controls": {"channels": list(self.control_channels)},
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "propellers": [propeller.to_dict() for propeller in self.propellers],
            **({} if self.body is None else {"body": self.body.to_dict()}),
        }


def load_aircraft_spec(path: str | Path) -> AircraftSpec:
    with Path(path).open("rb") as source:
        return AircraftSpec.from_dict(tomllib.load(source))


def save_aircraft_spec(spec: AircraftSpec, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        tomli_w.dump(spec.to_dict(), output)


def _floats(values: Any) -> dict[str, float]:
    return {str(key): float(value) for key, value in dict(values).items()}


def _longitudinal(spec: LongitudinalCoefficientSpec) -> LongitudinalCoefficients:
    return LongitudinalCoefficients(
        zero=jnp.asarray(spec.zero),
        alpha=jnp.asarray(spec.alpha_rad),
        q=jnp.asarray(spec.q),
        elevator=jnp.asarray(spec.elevator_rad),
    )


def _lateral(spec: LateralCoefficientSpec) -> LateralCoefficients:
    return LateralCoefficients(
        zero=jnp.asarray(spec.zero),
        beta=jnp.asarray(spec.beta_rad),
        p=jnp.asarray(spec.p),
        r=jnp.asarray(spec.r),
        aileron=jnp.asarray(spec.aileron_rad),
        rudder=jnp.asarray(spec.rudder_rad),
    )


def _vector(values: Any, length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise SpecError(f"{name} must contain {length} values")
    return result


def _matrix3(values: Any):
    rows = tuple(_vector(row, 3, "3x3 matrix row") for row in values)
    if len(rows) != 3:
        raise SpecError("matrix must contain three rows")
    return rows


def _require_unique(values, label: str) -> None:
    values = tuple(values)
    if len(values) != len(set(values)):
        raise SpecError(f"{label} names must be unique")


def _lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    return value
