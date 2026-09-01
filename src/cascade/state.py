from __future__ import annotations

from typing import NamedTuple

from jax import Array


class RigidBodyState(NamedTuple):
    """Common six-degree-of-freedom state.

    Attitude is an ``xyzw`` quaternion rotating body-FRD vectors into world-NED. Linear velocity
    is expressed in world coordinates and angular velocity in body coordinates.
    """

    position: Array
    attitude: Array
    velocity: Array
    angular_velocity: Array


class ActuatorState(NamedTuple):
    """Physical actuator state, not merely the latest command."""

    surface_deflection: Array
    propeller_speed: Array


class AeroState(NamedTuple):
    """Unsteady aerodynamic state for every surface.

    ``separation`` is zero for fully attached flow and one for fully separated flow.
    """

    separation: Array


class AircraftState(NamedTuple):
    rigid_body: RigidBodyState
    actuators: ActuatorState
    aero: AeroState


class ControlInput(NamedTuple):
    """Normalized direct-actuator input.

    Propeller commands use ``[0, 1]``. Abstract surface channels use ``[-1, 1]`` and are mapped to
    physical surfaces by :class:`cascade.model.ActuatorModel`.
    """

    propeller: Array
    channel: Array


class Environment(NamedTuple):
    """Per-world atmospheric state in world-NED coordinates."""

    density: Array
    wind: Array
    gravity: Array


class RigidBodyDerivative(NamedTuple):
    position: Array
    attitude: Array
    velocity: Array
    angular_velocity: Array


class ActuatorDerivative(NamedTuple):
    surface_deflection: Array
    propeller_speed: Array


class AeroDerivative(NamedTuple):
    separation: Array


class AircraftDerivative(NamedTuple):
    rigid_body: RigidBodyDerivative
    actuators: ActuatorDerivative
    aero: AeroDerivative
