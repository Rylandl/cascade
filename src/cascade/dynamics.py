from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from cascade.actuators import actuator_dynamics
from cascade.aerodynamics import (
    AerodynamicResult,
    PropulsionResult,
    aerodynamics,
    propulsion,
    separation_derivative,
)
from cascade.math import matvec, quaternion_derivative, quaternion_rotate, quaternion_rotate_inverse
from cascade.model import AircraftModel
from cascade.state import (
    ActuatorDerivative,
    AeroDerivative,
    AircraftDerivative,
    AircraftState,
    ControlInput,
    Environment,
    RigidBodyDerivative,
)


class DynamicsResult(NamedTuple):
    derivative: AircraftDerivative
    aerodynamics: AerodynamicResult
    propulsion: PropulsionResult
    air_velocity_body: Array
    force_body: Array
    moment_body: Array


def evaluate_dynamics(
    model: AircraftModel,
    state: AircraftState,
    control: ControlInput,
    environment: Environment,
) -> DynamicsResult:
    """Evaluate state derivatives and diagnostic wrenches without mutating state."""

    rigid_body = state.rigid_body
    air_velocity_world = rigid_body.velocity - environment.wind
    air_velocity_body = quaternion_rotate_inverse(rigid_body.attitude, air_velocity_world)

    propeller_result = propulsion(model, state, environment, air_velocity_body)
    aerodynamic_result = aerodynamics(
        model,
        state,
        environment,
        air_velocity_body,
        propeller_result.induced_velocity,
    )
    force_body = aerodynamic_result.force_body + propeller_result.force_body
    moment_body = aerodynamic_result.moment_body + propeller_result.moment_body

    force_world = quaternion_rotate(rigid_body.attitude, force_body)
    acceleration_world = force_world / model.mass[..., None] + environment.gravity
    angular_momentum = matvec(model.inertia, rigid_body.angular_velocity)
    gyroscopic_moment = jnp.cross(rigid_body.angular_velocity, angular_momentum, axis=-1)
    angular_acceleration = matvec(model.inertia_inverse, moment_body - gyroscopic_moment)

    actuator_rate: ActuatorDerivative = actuator_dynamics(model, state.actuators, control)
    derivative = AircraftDerivative(
        rigid_body=RigidBodyDerivative(
            position=rigid_body.velocity,
            attitude=quaternion_derivative(rigid_body.attitude, rigid_body.angular_velocity),
            velocity=acceleration_world,
            angular_velocity=angular_acceleration,
        ),
        actuators=actuator_rate,
        aero=AeroDerivative(
            separation=separation_derivative(
                model, state.aero, aerodynamic_result.air.separation_equilibrium
            )
        ),
    )
    return DynamicsResult(
        derivative=derivative,
        aerodynamics=aerodynamic_result,
        propulsion=propeller_result,
        air_velocity_body=air_velocity_body,
        force_body=force_body,
        moment_body=moment_body,
    )


def derivative(
    model: AircraftModel,
    state: AircraftState,
    control: ControlInput,
    environment: Environment,
) -> AircraftDerivative:
    """Return only the derivative, allowing unused diagnostics to be eliminated by XLA."""

    return evaluate_dynamics(model, state, control, environment).derivative
