"""Steady turning relative equilibria of the full aircraft dynamics."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import least_squares

from cascade.analysis.trim import (
    StraightFlightCondition,
    TrimResult,
    _decision_bounds,
    _initial_decision,
    _trim_candidate,
    _validate_unbatched_environment,
    trim_straight_flight,
)
from cascade.dynamics import evaluate_dynamics
from cascade.initialization import equilibrate_internal_state, standard_environment
from cascade.math import quaternion_rotate_inverse
from cascade.model import AircraftModel
from cascade.state import Environment


@dataclass(frozen=True, slots=True)
class SteadyTurnCondition:
    """Air-relative helix: positive turn rate rotates course clockwise in world NED.

    Roll, pitch, airspeed and climb angle remain constant. A constant wind translates the
    air-relative circle/helix, so the ground track need not be circular. Level flight is the
    default. This is an unconstrained-slip trim; a small regularizer prefers coordinated flight.
    """

    airspeed_m_s: float
    turn_rate_rad_s: float
    flight_path_angle_rad: float = 0.0
    heading_rad: float = 0.0
    altitude_m: float = 0.0

    def straight_condition(self) -> StraightFlightCondition:
        return StraightFlightCondition(
            self.airspeed_m_s, self.flight_path_angle_rad, self.heading_rad, self.altitude_m
        )

    def validate(self) -> None:
        self.straight_condition().validate()
        if not np.isfinite(self.turn_rate_rad_s):
            raise ValueError("turn rate must be finite")


@dataclass(frozen=True, slots=True)
class TurnTrimResult(TrimResult):
    """Relative equilibrium with residual ``[a_world - a_centripetal, omega_dot_body]``.

    Residual units are m/s² and rad/s². A successful turn has nonzero translational
    acceleration; the inherited ``acceleration_norm`` measures its error, not its magnitude.
    """

    condition: SteadyTurnCondition


def _candidate(model, condition, environment, decision):
    state, control = _trim_candidate(model, condition[:4], environment, decision)
    rates = quaternion_rotate_inverse(
        state.rigid_body.attitude, jnp.array([0.0, 0.0, condition[4]])
    )
    state = state._replace(rigid_body=state.rigid_body._replace(angular_velocity=rates))
    return equilibrate_internal_state(model, state, control, environment), control


def _balance(model, condition, environment, decision):
    state, control = _candidate(model, condition, environment, decision)
    dynamics = evaluate_dynamics(model, state, control, environment)
    air_velocity = state.rigid_body.velocity - environment.wind
    target = jnp.cross(jnp.array([0.0, 0.0, condition[4]]), air_velocity)
    return jnp.concatenate(
        (
            dynamics.derivative.rigid_body.velocity - target,
            dynamics.derivative.rigid_body.angular_velocity,
        )
    )


def _scaled_balance(model, condition, environment, decision):
    gravity = jnp.maximum(jnp.linalg.norm(environment.gravity), 1.0)
    angular_scale = gravity / jnp.array(
        [model.reference_span, model.reference_chord, model.reference_span]
    )
    scales = jnp.concatenate((jnp.full(3, gravity), angular_scale))
    return _balance(model, condition, environment, decision) / scales


_compiled_balance = jax.jit(_scaled_balance)
_compiled_jacobian = jax.jit(jax.jacfwd(_scaled_balance, argnums=3))


def trim_steady_turn(
    model: AircraftModel,
    condition: SteadyTurnCondition,
    environment: Environment | None = None,
    *,
    initial_decision: Array | TrimResult | None = None,
    residual_tolerance: float = 1e-4,
    max_evaluations: int = 300,
) -> TurnTrimResult:
    """Solve full force/gyroscopic moment balance for a constant-course-rate turn.

    The decision layout and actuator bounds match :func:`trim_straight_flight`. Body rates
    are the world vertical turn-rate vector rotated into body axes, not ``[0, 0, turn_rate]``.
    Constant vertical gravity and spatially uniform wind/density are required for a relative
    equilibrium. Failed/infeasible candidates remain inspectable with ``success=False``.
    Zero turn rate delegates to the straight-flight solver exactly.
    """

    condition.validate()
    if model.mass.ndim != 0:
        raise ValueError("turn trim requires an unbatched model")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, Integral)
        or max_evaluations <= 0
    ):
        raise ValueError("max_evaluations must be a positive integer")
    environment = standard_environment() if environment is None else environment
    _validate_unbatched_environment(environment)
    if condition.turn_rate_rad_s == 0:
        straight = trim_straight_flight(
            model,
            condition.straight_condition(),
            environment,
            initial_decision=initial_decision,
            residual_tolerance=residual_tolerance,
            max_evaluations=max_evaluations,
        )
        values = {field.name: getattr(straight, field.name) for field in fields(TrimResult)}
        return TurnTrimResult(**(values | {"condition": condition}))
    gravity = np.asarray(environment.gravity)
    if not np.allclose(gravity[:2], 0.0, atol=1e-8) or gravity[2] <= 0:
        raise ValueError("steady turns require positive vertical NED gravity")
    vector = jnp.array(
        [
            condition.airspeed_m_s,
            condition.flight_path_angle_rad,
            condition.heading_rad,
            condition.altitude_m,
            condition.turn_rate_rad_s,
        ]
    )
    lower, upper = _decision_bounds(model)
    initial = np.asarray(
        _initial_decision(model, condition.straight_condition(), initial_decision), dtype=float
    ).copy()
    bank_seed = np.arctan2(
        condition.airspeed_m_s
        * np.cos(condition.flight_path_angle_rad)
        * condition.turn_rate_rad_s,
        gravity[2],
    )
    if initial_decision is None:
        initial[0] = np.clip(bank_seed, lower[0] + 1e-5, upper[0] - 1e-5)
    regularizer = np.zeros((2, initial.size))
    regularizer[0, 0] = regularizer[1, 2] = 1e-3
    target = np.array([bank_seed * 1e-3, 0.0])

    def residual(decision):
        physical = np.asarray(_compiled_balance(model, vector, environment, jnp.asarray(decision)))
        return np.concatenate((physical, regularizer @ decision - target)).astype(float)

    def jacobian(decision):
        physical = _compiled_jacobian(model, vector, environment, jnp.asarray(decision))
        return np.concatenate((np.asarray(physical), regularizer)).astype(float)

    optimized = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=max_evaluations,
    )
    decision = jnp.asarray(optimized.x)
    state, control = _candidate(model, vector, environment, decision)
    balance = _balance(model, vector, environment, decision)
    scaled = _scaled_balance(model, vector, environment, decision)
    norm = float(jnp.linalg.norm(scaled))
    success = bool(optimized.success and np.isfinite(norm) and norm <= residual_tolerance)
    air = quaternion_rotate_inverse(
        state.rigid_body.attitude, state.rigid_body.velocity - environment.wind
    )
    message = str(optimized.message)
    if not success:
        message += f" Scaled turn balance norm {norm:.3e}; tolerance {residual_tolerance:.3e}."
    return TurnTrimResult(
        condition=condition,
        state=state,
        control=control,
        decision=decision,
        residual=balance,
        scaled_residual=scaled,
        angle_of_attack_rad=float(jnp.arctan2(air[2], air[0])),
        sideslip_rad=float(jnp.arctan2(air[1], jnp.hypot(air[0], air[2]))),
        success=success,
        optimizer_success=bool(optimized.success),
        cost=float(optimized.cost),
        optimality=float(optimized.optimality),
        evaluations=int(optimized.nfev),
        message=message,
    )


__all__ = ["SteadyTurnCondition", "TurnTrimResult", "trim_steady_turn"]
