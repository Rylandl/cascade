"""Bounded host declarations for a small, differentiable aircraft parameter vector.

Paths use specification names and units, never model-array indices. Geometry, topology,
actuator limits/maps and propeller maps are deliberately excluded. The nominal reference
specification is immutable; every application starts from that same baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Real

import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.model import AircraftModel, validate_model
from cascade.spec import AircraftSpec

# specification field -> (model field, physical domain)
_SURFACE_FIELDS = {
    "lift_coefficient_zero": ("lift_coefficient_zero", "signed"),
    "lift_curve_slope_rad": ("lift_curve_slope", "nonnegative"),
    "drag_coefficient_zero": ("drag_coefficient_zero", "nonnegative"),
    "induced_drag_factor": ("induced_drag_factor", "nonnegative"),
    "moment_coefficient_zero": ("moment_coefficient_zero", "signed"),
    "moment_coefficient_alpha_rad": ("moment_coefficient_alpha", "signed"),
    "stall_angle_rad": ("stall_angle", "positive"),
    "stall_width_rad": ("stall_width", "positive"),
    "normal_force_coefficient": ("normal_force_coefficient", "nonnegative"),
    "edge_drag_coefficient": ("edge_drag_coefficient", "nonnegative"),
    "span_drag_coefficient": ("span_drag_coefficient", "nonnegative"),
    "separation_time_constant_s": ("separation_time_constant", "positive"),
    "reattachment_time_constant_s": ("reattachment_time_constant", "positive"),
    "flap_effectiveness": ("flap_effectiveness", "nonnegative"),
    "moment_coefficient_flap_rad": ("moment_coefficient_flap", "signed"),
    "drag_coefficient_flap_rad2": ("drag_coefficient_flap", "nonnegative"),
    "actuator_time_constant_s": ("surface_time_constant", "positive"),
}
_LONGITUDINAL = {"zero": "zero", "alpha_rad": "alpha", "q": "q", "elevator_rad": "elevator"}
_LATERAL = {
    "zero": "zero",
    "beta_rad": "beta",
    "p": "p",
    "r": "r",
    "aileron_rad": "aileron",
    "rudder_rad": "rudder",
}
_DRAG = {
    "zero": "zero",
    "alpha_rad": "alpha",
    "alpha_sq_rad2": "alpha_sq",
    "beta_rad": "beta",
    "beta_sq_rad2": "beta_sq",
    "q": "q",
    "elevator_sq_rad2": "elevator_sq",
}
_BODY_GROUPS = {
    "lift": _LONGITUDINAL,
    "drag": _DRAG,
    "side": _LATERAL,
    "roll": _LATERAL,
    "pitch": _LONGITUDINAL,
    "yaw": _LATERAL,
}
_BODY_SCALARS = {
    "stall_angle_rad": ("stall_angle", "positive"),
    "stall_width_rad": ("stall_width", "positive"),
    "normal_force_coefficient": ("normal_force_coefficient", "nonnegative"),
    "pitch_flat_plate": ("pitch_flat_plate", "signed"),
}


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not boolean")
    try:
        converted = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a finite host float") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be representable as a finite host float")
    return converted


@dataclass(frozen=True, slots=True)
class FitParameter:
    """One absolute scalar parameter and inclusive bounds in specification units.

    ``inertia_scale`` is the sole relative parameter: one means nominal inertia.
    Bounds must have positive width; fixed quantities should simply be omitted.
    """

    path: str
    lower: float
    upper: float

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path or self.path.strip() != self.path:
            raise ValueError("parameter path must be a nonempty exact specification path")
        lower, upper = _real(self.lower, "lower bound"), _real(self.upper, "upper bound")
        if lower >= upper:
            raise ValueError("parameter bounds must satisfy lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class _Target:
    kind: str
    spec_field: str
    model_path: tuple[str, ...]
    initial: float
    domain: str
    index: int | None = None
    group: str | None = None


def _resolve(spec, path):
    if path == "mass_kg":
        return _Target("mass", "mass_kg", ("mass",), spec.mass_kg, "positive")
    if path == "inertia_scale":
        return _Target("inertia", "inertia_kg_m2", ("inertia",), 1.0, "positive")
    if path.startswith("surfaces.") and "." in path.removeprefix("surfaces."):
        name, spec_field = path.removeprefix("surfaces.").rsplit(".", 1)
        names = [surface.name for surface in spec.surfaces]
        if name not in names:
            raise ValueError(f"unknown surface name in parameter path {path!r}")
        if spec_field in _SURFACE_FIELDS:
            index = names.index(name)
            model_field, domain = _SURFACE_FIELDS[spec_field]
            group = "actuators" if spec_field == "actuator_time_constant_s" else "surfaces"
            return _Target(
                "surface",
                spec_field,
                (group, model_field),
                getattr(spec.surfaces[index], spec_field),
                domain,
                index=index,
            )
    if path.startswith("body."):
        if spec.body is None:
            raise ValueError("body parameters require an existing whole-aircraft coefficient block")
        parts = path.split(".")
        if len(parts) == 2 and parts[1] in _BODY_SCALARS:
            model_field, domain = _BODY_SCALARS[parts[1]]
            return _Target(
                "body", parts[1], ("body", model_field), getattr(spec.body, parts[1]), domain
            )
        if len(parts) == 3 and parts[1] in _BODY_GROUPS:
            group, spec_field = parts[1:]
            if spec_field in _BODY_GROUPS[group]:
                domain = "signed"
                if (
                    group == "drag"
                    and spec_field in {"zero", "alpha_sq_rad2", "beta_sq_rad2", "elevator_sq_rad2"}
                ) or (group == "lift" and spec_field == "alpha_rad"):
                    domain = "nonnegative"
                return _Target(
                    "body",
                    spec_field,
                    ("body", group, _BODY_GROUPS[group][spec_field]),
                    getattr(getattr(spec.body, group), spec_field),
                    domain,
                    group=group,
                )
    raise ValueError(f"unsupported calibration parameter path {path!r}")


def _set_model(model, path, value, index):
    name, *remaining = path
    if remaining:
        return model._replace(**{name: _set_model(getattr(model, name), remaining, value, index)})
    if index is not None:
        value = getattr(model, name).astype(value.dtype).at[index].set(value)
    return model._replace(**{name: value})


@dataclass(frozen=True, slots=True, eq=False)
class Parameterization:
    """Resolve named scalar parameters once, then apply a vector without host tracing.

    ``initial_values``, ``lower_bounds`` and ``upper_bounds`` are JAX vectors in declaration
    order and the JAX floating precision active at construction. Nonzero subnormal values,
    collapsed bounds and overflowing spans are rejected. Bounds are inclusive in this
    captured working precision, including when :meth:`apply_spec` checks fitted values.

    ``apply_model`` is a pure, unclipped transform; the optimizer must enforce its bounds.
    ``apply_spec`` is the host boundary that rejects invalid values and validates the saved
    specification/model. Allowed coefficient ranges do not guarantee stable flight or
    identifiability from the supplied data.
    """

    spec: AircraftSpec
    parameters: tuple[FitParameter, ...]
    initial_values: Array = field(init=False, repr=False, compare=False)
    lower_bounds: Array = field(init=False, repr=False, compare=False)
    upper_bounds: Array = field(init=False, repr=False, compare=False)
    _model: AircraftModel = field(init=False, repr=False, compare=False)
    _targets: tuple[_Target, ...] = field(init=False, repr=False, compare=False)
    _dtype: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.spec, AircraftSpec):
            raise ValueError("Parameterization requires an AircraftSpec")
        try:
            parameters = tuple(self.parameters)
        except TypeError as error:
            raise ValueError("parameters must be a nonempty sequence of FitParameter") from error
        if not parameters or any(not isinstance(p, FitParameter) for p in parameters):
            raise ValueError("parameters must be a nonempty sequence of FitParameter")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "_model", self.spec.to_model())
        object.__setattr__(self, "_dtype", jnp.asarray(0.0).dtype)
        targets = tuple(_resolve(self.spec, p.path) for p in parameters)
        resolved = [(t.model_path, t.index) for t in targets]
        if len(resolved) != len(set(resolved)):
            raise ValueError("parameter paths must resolve to unique model targets")
        initial = []
        for parameter, target in zip(parameters, targets, strict=True):
            value = _real(target.initial, f"nominal {parameter.path}")
            if not parameter.lower <= value <= parameter.upper:
                raise ValueError(f"nominal {parameter.path} must lie inside its bounds")
            if target.domain == "positive" and parameter.lower <= 0:
                raise ValueError(f"{parameter.path} bounds must be strictly positive")
            if target.domain == "nonnegative" and parameter.lower < 0:
                raise ValueError(f"{parameter.path} bounds must be nonnegative")
            initial.append(value)
        object.__setattr__(self, "_targets", targets)
        for name, values in (
            ("initial_values", initial),
            ("lower_bounds", [p.lower for p in parameters]),
            ("upper_bounds", [p.upper for p in parameters]),
        ):
            object.__setattr__(self, name, jnp.asarray(self._host_values(values), self._dtype))
        with np.errstate(over="ignore", invalid="ignore"):
            spans = np.asarray(self.upper_bounds) - np.asarray(self.lower_bounds)
        if not np.isfinite(spans).all() or np.any(spans <= 0):
            raise ValueError(
                "parameter bounds must have finite positive width in working precision"
            )
        # All fitted coordinates have independent scalar domains. Uniform inertia scaling
        # preserves the inertia shape/physical inequalities; endpoint checks catch overflow
        # or an unrepresentable cached inverse in either compiled or saved-spec arithmetic.
        for bounds in (self.lower_bounds, self.upper_bounds):
            validate_model(self.apply_model(bounds))
            self.apply_spec(bounds)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(parameter.path for parameter in self.parameters)

    @property
    def size(self) -> int:
        return len(self.parameters)

    def _host_values(self, values):
        try:
            values = np.asarray(values)
        except (TypeError, ValueError) as error:
            raise ValueError("parameter values must be a real numeric vector") from error
        if values.shape != (self.size,) or values.dtype.kind not in "iuf":
            raise ValueError(f"parameter values must be a real vector with shape ({self.size},)")
        limits = np.finfo(self._dtype)
        # Promote integer checks so abs(min_int64) cannot wrap around its dtype.
        checked = values.astype(np.float64) if values.dtype.kind in "iu" else values
        if (
            not np.isfinite(checked).all()
            or np.any(np.abs(checked) > limits.max)
            or np.any((checked != 0) & (np.abs(checked) < limits.tiny))
        ):
            raise ValueError(
                "parameter values must be finite and representable in working precision"
            )
        return values.astype(self._dtype)

    def apply_model(self, values) -> AircraftModel:
        """Apply absolute values to the captured nominal model; supports JIT/grad/vmap.

        Shape and real dtype are checked statically. Values are intentionally neither
        clipped nor converted to host arrays, so an optimizer retains true derivatives.
        """
        values = jnp.asarray(values)
        if values.shape != (self.size,) or not (
            jnp.issubdtype(values.dtype, jnp.floating) or jnp.issubdtype(values.dtype, jnp.integer)
        ):
            raise ValueError(f"parameter values must be a real vector with shape ({self.size},)")
        values = values.astype(self._dtype)
        model = self._model
        for target, value in zip(self._targets, values, strict=True):
            if target.kind == "inertia":
                model = model._replace(
                    inertia=self._model.inertia * value,
                    inertia_inverse=self._model.inertia_inverse / value,
                )
            else:
                model = _set_model(model, target.model_path, value, target.index)
        return model

    def apply_spec(self, values) -> AircraftSpec:
        """Validate values and return a new specification, retaining supplied host precision.

        Rounded JAX endpoints are accepted; their saved scalar is clamped back to the
        original host bound. This only removes rounding accepted by the working-precision
        check, never clips an optimizer excursion to make an out-of-bounds fit pass.
        """
        working = self._host_values(values)
        if np.any(working < np.asarray(self.lower_bounds)) or np.any(
            working > np.asarray(self.upper_bounds)
        ):
            raise ValueError("parameter values lie outside their inclusive bounds")
        values = np.asarray(values)
        spec = self.spec
        for parameter, target, raw_value in zip(
            self.parameters, self._targets, values, strict=True
        ):
            # Convert the NumPy scalar first: NumPy's weak scalar promotion would
            # otherwise round the Python bounds back to float32 before clipping.
            value = min(parameter.upper, max(parameter.lower, float(raw_value)))
            if target.kind == "mass":
                spec = replace(spec, mass_kg=value)
            elif target.kind == "inertia":
                inertia = tuple(
                    tuple(float(entry) * value for entry in row) for row in self.spec.inertia_kg_m2
                )
                spec = replace(spec, inertia_kg_m2=inertia)
            elif target.kind == "surface":
                surfaces = list(spec.surfaces)
                surfaces[target.index] = replace(
                    surfaces[target.index], **{target.spec_field: value}
                )
                spec = replace(spec, surfaces=tuple(surfaces))
            else:
                body = spec.body
                if target.group is None:
                    body = replace(body, **{target.spec_field: value})
                else:
                    group = replace(getattr(body, target.group), **{target.spec_field: value})
                    body = replace(body, **{target.group: group})
                spec = replace(spec, body=body)
        spec.to_model()  # Validate physical invariants as well as host schema/topology.
        return spec


__all__ = ["FitParameter", "Parameterization"]
