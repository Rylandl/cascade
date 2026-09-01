from __future__ import annotations

import jax.numpy as jnp

from cascade.model import AircraftModel
from cascade.state import ActuatorDerivative, ActuatorState, ControlInput


def actuator_targets(model: AircraftModel, control: ControlInput) -> ActuatorState:
    """Map normalized control inputs to physical surface angles and propeller speeds."""

    actuators = model.actuators
    channel = jnp.clip(control.channel, -1.0, 1.0)
    surface = actuators.surface_bias + jnp.einsum(
        "...sc,...c->...s", actuators.surface_map, channel
    )
    surface = jnp.clip(surface, -actuators.surface_limit, actuators.surface_limit)

    throttle = jnp.clip(control.propeller, 0.0, 1.0)
    propeller = actuators.propeller_speed_min + throttle * (
        actuators.propeller_speed_max - actuators.propeller_speed_min
    )
    return ActuatorState(surface_deflection=surface, propeller_speed=propeller)


def actuator_dynamics(
    model: AircraftModel, state: ActuatorState, control: ControlInput
) -> ActuatorDerivative:
    """Smooth first-order actuator dynamics with asymptotic rate limiting."""

    target = actuator_targets(model, control)
    actuators = model.actuators

    desired_surface_rate = (
        target.surface_deflection - state.surface_deflection
    ) / actuators.surface_time_constant
    surface_rate = actuators.surface_rate_limit * jnp.tanh(
        desired_surface_rate / actuators.surface_rate_limit
    )

    desired_propeller_acceleration = (
        target.propeller_speed - state.propeller_speed
    ) / actuators.propeller_time_constant
    propeller_acceleration = actuators.propeller_acceleration_limit * jnp.tanh(
        desired_propeller_acceleration / actuators.propeller_acceleration_limit
    )
    return ActuatorDerivative(
        surface_deflection=surface_rate, propeller_speed=propeller_acceleration
    )
