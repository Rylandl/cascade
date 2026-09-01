from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import least_squares

from cascade.dynamics import evaluate_dynamics
from cascade.initialization import equilibrate_internal_state, standard_environment, zero_state
from cascade.math import quaternion_from_euler
from cascade.model import AircraftModel
from cascade.state import AircraftState, ControlInput, Environment


@dataclass(frozen=True, slots=True)
class StraightFlightCondition:
    """Requested air-relative path for a constant-velocity, zero-rate trim.

    Flight-path angle is positive in a climb. Heading is the air-relative course clockwise from
    north in NED; body yaw differs from it by the trim's yaw offset, which is the sideslip a
    rudderless or torque-loaded aircraft needs to balance yaw. Wind is added to this
    air-relative velocity, so the resulting ground track can differ from heading.
    """

    airspeed_m_s: float
    flight_path_angle_rad: float = 0.0
    heading_rad: float = 0.0
    altitude_m: float = 0.0

    def validate(self) -> None:
        values = (
            self.airspeed_m_s,
            self.flight_path_angle_rad,
            self.heading_rad,
            self.altitude_m,
        )
        if not all(np.isfinite(values)):
            raise ValueError("trim condition must contain only finite values")
        if self.airspeed_m_s <= 0.0:
            raise ValueError("trim airspeed must be positive")
        if abs(self.flight_path_angle_rad) >= np.pi / 2.0:
            raise ValueError("flight-path angle must lie strictly between -pi/2 and pi/2")


@dataclass(frozen=True, slots=True)
class TrimResult:
    """A steady-flight candidate and its physical balance error."""

    condition: StraightFlightCondition
    state: AircraftState
    control: ControlInput
    decision: Array
    residual: Array
    scaled_residual: Array
    angle_of_attack_rad: float
    sideslip_rad: float
    success: bool
    optimizer_success: bool
    cost: float
    optimality: float
    evaluations: int
    message: str

    @property
    def acceleration_norm(self) -> float:
        return float(jnp.linalg.norm(self.residual[:3]))

    @property
    def angular_acceleration_norm(self) -> float:
        return float(jnp.linalg.norm(self.residual[3:]))


def trim_straight_flight(
    model: AircraftModel,
    condition: StraightFlightCondition,
    environment: Environment | None = None,
    *,
    initial_decision: Array | TrimResult | None = None,
    residual_tolerance: float = 1e-4,
    max_evaluations: int = 300,
) -> TrimResult:
    """Solve force and moment balance for straight, constant-velocity flight.

    The decision vector is ``[roll, pitch, yaw offset, propeller commands..., control
    channels...]``, where the yaw offset is body yaw minus air-relative heading.
    The returned state has zero body rates and internally equilibrated actuators and aerodynamic
    separation, making it suitable as a rollout or linearization initial state.
    """

    condition.validate()
    environment = standard_environment() if environment is None else environment
    _validate_unbatched_environment(environment)
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")

    initial = _initial_decision(model, condition, initial_decision)
    lower, upper = _decision_bounds(model)
    condition_vector = jnp.asarray(
        [
            condition.airspeed_m_s,
            condition.flight_path_angle_rad,
            condition.heading_rad,
            condition.altitude_m,
        ]
    )

    def scipy_residual(decision: np.ndarray) -> np.ndarray:
        value = _compiled_scaled_balance(
            model, condition_vector, environment, jnp.asarray(decision)
        )
        return np.asarray(jax.device_get(value), dtype=float)

    def scipy_jacobian(decision: np.ndarray) -> np.ndarray:
        value = _compiled_balance_jacobian(
            model, condition_vector, environment, jnp.asarray(decision)
        )
        return np.asarray(jax.device_get(value), dtype=float)

    optimizer = least_squares(
        scipy_residual,
        np.asarray(initial, dtype=float),
        jac=scipy_jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=max_evaluations,
    )
    decision = jnp.asarray(optimizer.x)
    state, control = _trim_candidate(model, condition_vector, environment, decision)
    dynamics = evaluate_dynamics(model, state, control, environment)
    residual = jnp.concatenate(
        (
            dynamics.derivative.rigid_body.velocity,
            dynamics.derivative.rigid_body.angular_velocity,
        )
    )
    scaled_residual = _scaled_balance(model, condition_vector, environment, decision)
    air_velocity = dynamics.air_velocity_body
    planar_speed = jnp.hypot(air_velocity[0], air_velocity[2])
    angle_of_attack = jnp.arctan2(air_velocity[2], air_velocity[0])
    sideslip = jnp.arctan2(air_velocity[1], planar_speed)
    balance_norm = float(jnp.linalg.norm(scaled_residual))
    success = bool(optimizer.success and balance_norm <= residual_tolerance)
    message = str(optimizer.message)
    if optimizer.success and not success:
        message = (
            f"{message} Optimizer stopped with scaled balance norm {balance_norm:.3e}, "
            f"above tolerance {residual_tolerance:.3e}."
        )
    return TrimResult(
        condition=condition,
        state=state,
        control=control,
        decision=decision,
        residual=residual,
        scaled_residual=scaled_residual,
        angle_of_attack_rad=float(angle_of_attack),
        sideslip_rad=float(sideslip),
        success=success,
        optimizer_success=bool(optimizer.success),
        cost=float(optimizer.cost),
        optimality=float(optimizer.optimality),
        evaluations=int(optimizer.nfev),
        message=message,
    )


def continue_trims(
    model: AircraftModel,
    conditions: Iterable[StraightFlightCondition],
    environment: Environment | None = None,
    *,
    initial_decision: Array | TrimResult | None = None,
    residual_tolerance: float = 1e-4,
    max_evaluations: int = 300,
) -> tuple[TrimResult, ...]:
    """Solve an ordered envelope, warm-starting each point from the previous solution."""

    results = []
    seed = initial_decision
    for condition in conditions:
        result = trim_straight_flight(
            model,
            condition,
            environment,
            initial_decision=seed,
            residual_tolerance=residual_tolerance,
            max_evaluations=max_evaluations,
        )
        results.append(result)
        seed = result
    return tuple(results)


def _initial_decision(
    model: AircraftModel,
    condition: StraightFlightCondition,
    initial: Array | TrimResult | None,
) -> Array:
    size = 3 + model.n_propellers + model.n_control_channels
    if isinstance(initial, TrimResult):
        initial = initial.decision
    if initial is not None:
        value = np.asarray(initial, dtype=float)
        if value.shape != (size,):
            raise ValueError(f"initial trim decision must have shape {(size,)}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("initial trim decision must be finite")
        return jnp.asarray(value)

    # A modest positive pitch and half throttle is a useful generic seed for small conventional
    # aircraft; low throttle can start inside a propeller's windmilling valley, where thrust
    # first falls with throttle before rising, and trap a local solver at zero throttle.
    # Continuation or an explicit decision should seed alternate/high-alpha branches.
    return jnp.concatenate(
        (
            jnp.array([0.0, condition.flight_path_angle_rad + np.deg2rad(5.0), 0.0]),
            jnp.full((model.n_propellers,), 0.5),
            jnp.zeros((model.n_control_channels,)),
        )
    )


def _trim_candidate(
    model: AircraftModel,
    condition: Array,
    environment: Environment,
    decision: Array,
) -> tuple[AircraftState, ControlInput]:
    speed, flight_path_angle, heading, altitude = condition
    roll, pitch, yaw_offset = decision[0], decision[1], decision[2]
    propeller_end = 3 + model.n_propellers
    control = ControlInput(
        propeller=decision[3:propeller_end],
        channel=decision[propeller_end:],
    )
    cosine_gamma = jnp.cos(flight_path_angle)
    air_velocity_world = speed * jnp.array(
        [
            cosine_gamma * jnp.cos(heading),
            cosine_gamma * jnp.sin(heading),
            -jnp.sin(flight_path_angle),
        ]
    )
    state = zero_state(model, altitude=altitude)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            attitude=quaternion_from_euler(roll, pitch, heading + yaw_offset),
            velocity=air_velocity_world + environment.wind,
        )
    )
    return equilibrate_internal_state(model, state, control, environment), control


def _scaled_balance(
    model: AircraftModel,
    condition: Array,
    environment: Environment,
    decision: Array,
) -> Array:
    state, control = _trim_candidate(model, condition, environment, decision)
    result = evaluate_dynamics(model, state, control, environment)
    acceleration = result.derivative.rigid_body.velocity
    angular_acceleration = result.derivative.rigid_body.angular_velocity
    gravity_scale = jnp.maximum(jnp.linalg.norm(environment.gravity), 1.0)
    angular_scale = gravity_scale / jnp.array(
        [model.reference_span, model.reference_chord, model.reference_span]
    )
    return jnp.concatenate((acceleration / gravity_scale, angular_acceleration / angular_scale))


_compiled_scaled_balance = jax.jit(_scaled_balance)
_compiled_balance_jacobian = jax.jit(jax.jacfwd(_scaled_balance, argnums=3))


def _decision_bounds(model: AircraftModel) -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate(
        (
            np.array([-np.deg2rad(80.0), -np.deg2rad(89.0), -np.deg2rad(30.0)]),
            np.zeros(model.n_propellers),
            -np.ones(model.n_control_channels),
        )
    )
    upper = np.concatenate(
        (
            np.array([np.deg2rad(80.0), np.deg2rad(89.0), np.deg2rad(30.0)]),
            np.ones(model.n_propellers),
            np.ones(model.n_control_channels),
        )
    )
    return lower, upper


def _validate_unbatched_environment(environment: Environment) -> None:
    if np.shape(environment.density) != ():
        raise ValueError("trim requires a scalar environment density")
    if np.shape(environment.wind) != (3,) or np.shape(environment.gravity) != (3,):
        raise ValueError("trim requires unbatched wind and gravity vectors with shape (3,)")
    if not all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (environment.density, environment.wind, environment.gravity)
    ):
        raise ValueError("trim environment must contain only finite values")
    if float(np.asarray(environment.density)) <= 0.0:
        raise ValueError("trim environment density must be positive")
