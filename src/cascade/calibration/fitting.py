"""Bounded offline parameter fitting against explicitly supplied flight recordings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import least_squares

from cascade.experiments.flight_data import FlightRecord
from cascade.experiments.replay import prepare_record, replay_record
from cascade.math import quaternion_conjugate, quaternion_multiply, quaternion_to_rotvec
from cascade.provenance import spec_hash, stamp
from cascade.spec import AircraftSpec

from .parameters import FitParameter, Parameterization


def _implementation_hash():
    """Fingerprint the installed package's Python sources with stable relative paths."""
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidualScales:
    """Positive physical scales for dimensionless position, velocity, attitude and rate errors."""

    position_m: float = 1.0
    velocity_m_s: float = 1.0
    attitude_rad: float = 0.1
    rate_rad_s: float = 0.1

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{item.name} must be finite and positive")
            if value <= 0:
                raise ValueError(f"{item.name} must be finite and positive")
            object.__setattr__(self, item.name, float(value))


@dataclass(frozen=True)
class CalibrationConfig:
    """Frozen fitting choices; only supplied records enter the objective.

    Each record contributes its mean squared scaled state-error norm, with equal weight
    across records. ``warmup_steps`` excludes that many initial interval endpoints from
    scoring, while replay still integrates the complete recording from its initial state.
    Positive residual scales have physical units; they are not estimated noise variances.
    There are no implicit priors or parameter selection against held-out data.
    """

    parameters: tuple[FitParameter, ...]
    substeps: int = 4
    warmup_steps: int = 0
    max_nfev: int = 100
    scales: ResidualScales = field(default_factory=ResidualScales)

    def __post_init__(self):
        parameters = tuple(self.parameters)
        if not parameters or any(not isinstance(value, FitParameter) for value in parameters):
            raise ValueError("parameters must contain at least one FitParameter")
        if len({parameter.path for parameter in parameters}) != len(parameters):
            raise ValueError("parameter paths must be unique")
        object.__setattr__(self, "parameters", parameters)
        if not isinstance(self.scales, ResidualScales):
            raise TypeError("scales must be ResidualScales")
        for name, minimum in (("substeps", 1), ("warmup_steps", 0), ("max_nfev", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, Any]:
        """Return finite JSON data, with ordered parameter declarations."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationConfig:
        """Restore explicit data; unknown fields and malformed parameter/scales objects fail."""
        if not isinstance(data, dict) or set(data) - {item.name for item in fields(cls)}:
            raise ValueError("unsupported calibration configuration fields")
        if not isinstance(data.get("parameters"), list):
            raise ValueError("configuration parameters must be a list")
        if any(
            not isinstance(item, dict) or set(item) != {"path", "lower", "upper"}
            for item in data["parameters"]
        ):
            raise ValueError("each parameter requires path, lower and upper")
        scales = data.get("scales", {})
        if not isinstance(scales, dict) or set(scales) - {
            item.name for item in fields(ResidualScales)
        }:
            raise ValueError("unsupported residual scales fields")
        return cls(
            **{
                **data,
                "parameters": tuple(FitParameter(**item) for item in data["parameters"]),
                "scales": ResidualScales(**scales),
            }
        )


@dataclass(frozen=True)
class FitResult:
    """A finite validated candidate, optimizer outcome, and data-sensitivity diagnostics.

    ``report['optimizer']['success']`` means the numerical solver met a stopping criterion;
    it does not establish identifiable parameters or physical model accuracy. Budget-limited
    finite candidates are returned with success false. ``records`` contains provenance
    metadata only; persistence and held-out evaluation belong to the pack workflow.
    ``provenance`` snapshots the fitting runtime and package sources before optimization,
    so delayed artifact publication cannot substitute its own runtime for the fitting one.
    """

    spec: AircraftSpec
    config: CalibrationConfig
    records: tuple[dict[str, Any], ...]
    nominal_spec_sha256: str
    report: dict[str, Any]
    provenance: dict[str, Any]


def _state_residual(predicted, observed, scales: ResidualScales):
    """Scaled canonical state errors with an antipodal-invariant local rotation residual."""
    # Canonical wxyz -> the math module's xyzw convention; both remain in NWU/FLU frames.
    predicted_q = jnp.concatenate((predicted[..., 7:10], predicted[..., 6:7]), axis=-1)
    observed_q = jnp.concatenate((observed[..., 7:10], observed[..., 6:7]), axis=-1)
    relative_q = quaternion_multiply(quaternion_conjugate(observed_q), predicted_q)
    pivot = jnp.take_along_axis(
        relative_q, jnp.argmax(jnp.abs(relative_q), axis=-1)[..., None], axis=-1
    )
    # Choose the same representative even at a half-turn, where w and -w are both zero.
    relative_q = jnp.where(pivot < 0, -relative_q, relative_q)
    attitude = quaternion_to_rotvec(relative_q)
    return jnp.concatenate(
        (
            (predicted[..., :3] - observed[..., :3]) / scales.position_m,
            (predicted[..., 3:6] - observed[..., 3:6]) / scales.velocity_m_s,
            attitude / scales.attitude_rad,
            (predicted[..., 10:13] - observed[..., 10:13]) / scales.rate_rad_s,
        ),
        axis=-1,
    )


def _finite_array(value, context):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"calibration {context} became nonfinite")
    return array


def fit_records(
    spec: AircraftSpec, fitting_records: tuple[FlightRecord, ...], config: CalibrationConfig
) -> FitResult:
    """Fit a bounded parameter subset using only explicitly supplied recordings.

    All records are replayed open-loop with the same timing and initialization as the
    evaluator. The solver uses unit-interval parameter coordinates and forward-mode JAX
    Jacobians. Every residual/Jacobian evaluation is checked for finite values; failed
    arithmetic raises ``FloatingPointError`` rather than returning a claimed calibration.
    Data-loss and Jacobian diagnostics omit priors (none are added) and never use evaluation
    records. Record selection and provenance binding are handled by the pack-level API.
    """
    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be CalibrationConfig")
    records = tuple(fitting_records)
    if not records or any(not isinstance(record, FlightRecord) for record in records):
        raise ValueError("fitting_records must contain at least one FlightRecord")
    if len({record.name for record in records}) != len(records):
        raise ValueError("fitting record names must be unique")
    parameterization = Parameterization(spec, config.parameters)
    model = spec.to_model()
    provenance = {
        "runtime": stamp(
            spec, model, numpy_version=np.__version__, scipy_version=scipy.__version__
        ),
        "implementation_sha256": _implementation_hash(),
    }
    inputs = tuple(prepare_record(record, model, substeps=config.substeps) for record in records)
    if any(record.observed.shape[0] <= config.warmup_steps for record in inputs):
        raise ValueError("warmup_steps must leave at least one scored interval in every record")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scales = np.asarray(
            tuple(asdict(config.scales).values()), dtype=np.dtype(jnp.asarray(0.0).dtype)
        )
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("residual scales must be positive in the configured JAX precision")
    lower, upper = parameterization.lower_bounds, parameterization.upper_bounds
    span = upper - lower
    if not np.isfinite(np.asarray(span)).all() or np.any(np.asarray(span) <= 0):
        raise ValueError("parameter bound spans must be positive in the configured JAX precision")

    def residual_model(candidate):
        residuals = []
        for record in inputs:
            predicted = replay_record(candidate, record)[config.warmup_steps :]
            observed = record.observed[config.warmup_steps :]
            residual = _state_residual(predicted, observed, config.scales)
            residuals.append(residual.reshape(-1) / np.sqrt(len(inputs) * len(observed)))
        return jnp.concatenate(residuals)

    def residual_values(values):
        return residual_model(parameterization.apply_model(values))

    compiled_model_residual = jax.jit(residual_model)
    compiled_residual = jax.jit(residual_values)
    compiled_jacobian = jax.jit(jax.jacfwd(residual_values))

    def objective(normalized):
        return _finite_array(
            compiled_residual(lower + span * jnp.asarray(normalized, dtype=lower.dtype)), "residual"
        )

    def jacobian(normalized):
        return _finite_array(
            compiled_jacobian(lower + span * jnp.asarray(normalized, dtype=lower.dtype)) * span,
            "Jacobian",
        )

    initial = np.asarray(parameterization.initial_values, dtype=np.float64)
    initial_normalized = np.clip(
        (initial - np.asarray(lower, dtype=np.float64)) / np.asarray(span, dtype=np.float64),
        0.0,
        1.0,
    )
    initial_residual = _finite_array(compiled_model_residual(model), "initial residual")
    tolerances = {"ftol": 1e-6, "xtol": 1e-6, "gtol": 1e-6}
    optimized = least_squares(
        objective,
        initial_normalized,
        jac=jacobian,
        bounds=(np.zeros(len(initial)), np.ones(len(initial))),
        max_nfev=config.max_nfev,
        **tolerances,
    )
    # Rebuild the serializable candidate from exact host bounds, not rounded JAX endpoints.
    exact_lower = np.asarray([parameter.lower for parameter in config.parameters])
    exact_upper = np.asarray([parameter.upper for parameter in config.parameters])
    values = np.clip(
        exact_lower + (exact_upper - exact_lower) * optimized.x, exact_lower, exact_upper
    )
    fitted_spec = parameterization.apply_spec(values)
    fitted_model = fitted_spec.to_model()
    final_residual = _finite_array(compiled_model_residual(fitted_model), "final residual")
    final_jacobian = _finite_array(
        compiled_jacobian(jnp.asarray(values, dtype=lower.dtype)) * span, "final Jacobian"
    )
    singular_values = _finite_array(
        np.linalg.svd(final_jacobian, compute_uv=False), "Jacobian singular values"
    )
    tolerance = float(
        max(final_jacobian.shape) * np.finfo(np.dtype(lower.dtype)).eps * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1]) if rank == len(config.parameters) else None
    )
    record_metadata = tuple(
        {
            "name": record.name,
            "maneuver_id": record.maneuver_id,
            "kind": record.kind,
            "content_sha256": record.sha256,
            "samples": len(record.time_s),
            "scored_samples": len(record.time_s) - 1 - config.warmup_steps,
        }
        for record in records
    )
    report = {
        "schema": "cascade_calibration_fit_v1",
        "nominal_spec_sha256": spec_hash(spec),
        "calibrated_spec_sha256": spec_hash(fitted_spec),
        "configuration": config.to_dict(),
        "records": list(record_metadata),
        "optimizer": {
            "method": "scipy.optimize.least_squares",
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "message": str(optimized.message),
            "nfev": int(optimized.nfev),
            "njev": None if optimized.njev is None else int(optimized.njev),
            "optimality": float(optimized.optimality),
            "tolerances": tolerances,
        },
        "initial_data_loss": float(initial_residual @ initial_residual),
        "final_data_loss": float(final_residual @ final_residual),
        "loss_definition": "Mean across records of the mean squared scaled state-error norm; "
        "12 residual components per sample, with 3 attitude rotation-vector components. "
        "Warm-up endpoints excluded; no prior residuals. Initial/final losses replay the "
        "nominal/fitted serialized specifications, including recompilation of cached inertia.",
        "parameters": [
            {
                "path": parameter.path,
                "initial": float(start),
                "value": float(value),
                "lower": parameter.lower,
                "upper": parameter.upper,
                "normalized_value": float(normalized),
                "bound_hit": "lower"
                if normalized <= 1e-6
                else "upper"
                if normalized >= 1 - 1e-6
                else None,
            }
            for parameter, start, value, normalized in zip(
                config.parameters, initial, values, optimized.x, strict=True
            )
        ],
        "identifiability": {
            "coordinates": "unit-interval parameters; scaled data residuals only",
            "singular_values": singular_values.tolist(),
            "rank": rank,
            "parameter_count": len(config.parameters),
            "rank_tolerance": tolerance,
            "condition": condition,
            "interpretation": "Local numerical sensitivity, not covariance, unique physical "
            "identification, or a physical accuracy claim. Solver success is separate from rank. "
            "Jacobian uses the differentiable parameter transform at the fitted host values; "
            "the serialized-spec replay may differ at floating-point roundoff.",
        },
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    return FitResult(fitted_spec, config, record_metadata, spec_hash(spec), report, provenance)
