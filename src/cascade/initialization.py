from __future__ import annotations

import jax.numpy as jnp

from cascade._typing import BatchShape
from cascade.actuators import actuator_targets
from cascade.aerodynamics import propulsion, surface_air_data
from cascade.math import quaternion_rotate_inverse
from cascade.model import AircraftModel
from cascade.state import (
    ActuatorState,
    AeroState,
    AircraftState,
    ControlInput,
    Environment,
    RigidBodyState,
)


def zero_state(
    model: AircraftModel,
    batch_shape: BatchShape = (),
    *,
    altitude: float = 0.0,
    forward_speed: float = 0.0,
) -> AircraftState:
    """Construct a level state at rest or moving north in still air."""

    vector_shape = (*batch_shape, 3)
    position = jnp.zeros(vector_shape).at[..., 2].set(-altitude)
    attitude = jnp.zeros((*batch_shape, 4)).at[..., 3].set(1.0)
    velocity = jnp.zeros(vector_shape).at[..., 0].set(forward_speed)
    angular_velocity = jnp.zeros(vector_shape)
    return AircraftState(
        rigid_body=RigidBodyState(
            position=position,
            attitude=attitude,
            velocity=velocity,
            angular_velocity=angular_velocity,
        ),
        actuators=ActuatorState(
            surface_deflection=jnp.broadcast_to(
                model.actuators.surface_bias, (*batch_shape, model.n_surfaces)
            ),
            propeller_speed=jnp.broadcast_to(
                model.actuators.propeller_speed_min,
                (*batch_shape, model.n_propellers),
            ),
        ),
        aero=AeroState(separation=jnp.zeros((*batch_shape, model.n_surfaces))),
    )


def zero_control(model: AircraftModel, batch_shape: BatchShape = ()) -> ControlInput:
    return ControlInput(
        propeller=jnp.zeros((*batch_shape, model.n_propellers)),
        channel=jnp.zeros((*batch_shape, model.n_control_channels)),
    )


def control_from_array(model: AircraftModel, action: jnp.ndarray) -> ControlInput:
    """Split a dense action ``[..., P + C]`` into its typed control representation."""

    split = model.n_propellers
    return ControlInput(propeller=action[..., :split], channel=action[..., split:])


def control_to_array(control: ControlInput) -> jnp.ndarray:
    return jnp.concatenate((control.propeller, control.channel), axis=-1)


def standard_environment(
    batch_shape: BatchShape = (),
    *,
    density: float = 1.225,
    gravity: float = 9.80665,
) -> Environment:
    """Still-air sea-level environment in world-NED coordinates."""

    return Environment(
        density=jnp.full(batch_shape, density),
        wind=jnp.zeros((*batch_shape, 3)),
        gravity=jnp.broadcast_to(jnp.array([0.0, 0.0, gravity]), (*batch_shape, 3)),
    )


def equilibrate_internal_state(
    model: AircraftModel,
    state: AircraftState,
    control: ControlInput,
    environment: Environment,
) -> AircraftState:
    """Set actuator and separation states to their instantaneous command/flow equilibria.

    This is appropriate for reset states representing an aircraft already in steady conditions.
    Leave the internal state untouched when initializing a rapid maneuver whose actuator or stall
    history is intentionally part of the initial condition.
    """

    actuators = actuator_targets(model, control)
    state = state._replace(actuators=actuators)
    air_velocity_world = state.rigid_body.velocity - environment.wind
    air_velocity_body = quaternion_rotate_inverse(state.rigid_body.attitude, air_velocity_world)
    propeller_result = propulsion(model, state, environment, air_velocity_body)
    air, _ = surface_air_data(
        model,
        state,
        environment,
        air_velocity_body,
        propeller_result.induced_velocity,
    )
    return state._replace(aero=AeroState(separation=air.separation_equilibrium))
