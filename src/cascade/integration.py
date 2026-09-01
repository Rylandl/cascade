from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from cascade.dynamics import derivative
from cascade.math import normalize
from cascade.model import AircraftModel
from cascade.state import (
    ActuatorState,
    AeroState,
    AircraftDerivative,
    AircraftState,
    ControlInput,
    Environment,
    RigidBodyState,
)

StepFunction = Callable[
    [AircraftModel, AircraftState, ControlInput, Environment, float], AircraftState
]


def add_derivative(
    state: AircraftState, state_derivative: AircraftDerivative, scale: float
) -> AircraftState:
    """Advance every state leaf linearly without projecting intermediate RK stages."""

    rigid_body = RigidBodyState(
        *(
            value + scale * rate
            for value, rate in zip(state.rigid_body, state_derivative.rigid_body, strict=True)
        )
    )
    actuators = ActuatorState(
        *(
            value + scale * rate
            for value, rate in zip(state.actuators, state_derivative.actuators, strict=True)
        )
    )
    aero = AeroState(
        *(
            value + scale * rate
            for value, rate in zip(state.aero, state_derivative.aero, strict=True)
        )
    )
    return AircraftState(rigid_body=rigid_body, actuators=actuators, aero=aero)


def weighted_derivative(
    first: AircraftDerivative,
    second: AircraftDerivative,
    third: AircraftDerivative,
    fourth: AircraftDerivative,
) -> AircraftDerivative:
    return jax.tree.map(
        lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0,
        first,
        second,
        third,
        fourth,
    )


def project_state(model: AircraftModel, state: AircraftState) -> AircraftState:
    """Restore quaternion and bounded physical-state invariants after an integration step."""

    rigid_body = state.rigid_body._replace(attitude=normalize(state.rigid_body.attitude))
    actuators = ActuatorState(
        surface_deflection=jnp.clip(
            state.actuators.surface_deflection,
            -model.actuators.surface_limit,
            model.actuators.surface_limit,
        ),
        propeller_speed=jnp.clip(
            state.actuators.propeller_speed,
            model.actuators.propeller_speed_min,
            model.actuators.propeller_speed_max,
        ),
    )
    aero = AeroState(separation=jnp.clip(state.aero.separation, 0.0, 1.0))
    return AircraftState(rigid_body=rigid_body, actuators=actuators, aero=aero)


def euler_step(
    model: AircraftModel,
    state: AircraftState,
    control: ControlInput,
    environment: Environment,
    dt: float,
) -> AircraftState:
    next_state = add_derivative(state, derivative(model, state, control, environment), dt)
    return project_state(model, next_state)


def rk4_step(
    model: AircraftModel,
    state: AircraftState,
    control: ControlInput,
    environment: Environment,
    dt: float,
) -> AircraftState:
    """Fourth-order Runge-Kutta step with one final physical-state projection."""

    k1 = derivative(model, state, control, environment)
    k2 = derivative(model, add_derivative(state, k1, dt / 2.0), control, environment)
    k3 = derivative(model, add_derivative(state, k2, dt / 2.0), control, environment)
    k4 = derivative(model, add_derivative(state, k3, dt), control, environment)
    next_state = add_derivative(state, weighted_derivative(k1, k2, k3, k4), dt)
    return project_state(model, next_state)


def rollout(
    model: AircraftModel,
    initial_state: AircraftState,
    controls: ControlInput,
    environment: Environment,
    dt: float,
    *,
    step: StepFunction = rk4_step,
) -> tuple[AircraftState, AircraftState]:
    """Scan a time-major control sequence, returning final and post-step trajectory states."""

    def scan_step(state: AircraftState, control: ControlInput):
        next_state = step(model, state, control, environment, dt)
        return next_state, next_state

    return jax.lax.scan(scan_step, initial_state, controls)


def repeat_control(control: ControlInput, steps: int) -> ControlInput:
    """Make a time-major constant control sequence for :func:`rollout`."""

    return jax.tree.map(lambda value: jnp.broadcast_to(value, (steps, *value.shape)), control)
