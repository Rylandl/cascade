from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cascade.math import (
    normalize,
    quaternion_conjugate,
    quaternion_from_rotvec,
    quaternion_multiply,
    quaternion_to_rotvec,
)
from cascade.model import AircraftModel
from cascade.state import ActuatorState, AeroState, AircraftState, ControlInput, RigidBodyState


def tangent_state_size(model: AircraftModel) -> int:
    """Dimension of the minimal local state: 12 rigid-body plus actuator/aero states."""

    return 12 + 2 * model.n_surfaces + model.n_propellers


def state_retract(model: AircraftModel, state: AircraftState, delta: Array) -> AircraftState:
    """Apply a minimal local perturbation to an aircraft state.

    Attitude perturbations are body-local rotation vectors composed on the quaternion's right.
    """

    n_surface, n_propeller = model.n_surfaces, model.n_propellers
    surface_start = 12
    propeller_start = surface_start + n_surface
    separation_start = propeller_start + n_propeller
    attitude_delta = quaternion_from_rotvec(delta[..., 3:6])
    attitude = normalize(quaternion_multiply(state.rigid_body.attitude, attitude_delta))
    return AircraftState(
        rigid_body=RigidBodyState(
            position=state.rigid_body.position + delta[..., 0:3],
            attitude=attitude,
            velocity=state.rigid_body.velocity + delta[..., 6:9],
            angular_velocity=state.rigid_body.angular_velocity + delta[..., 9:12],
        ),
        actuators=ActuatorState(
            surface_deflection=(
                state.actuators.surface_deflection + delta[..., surface_start:propeller_start]
            ),
            propeller_speed=(
                state.actuators.propeller_speed + delta[..., propeller_start:separation_start]
            ),
        ),
        aero=AeroState(separation=state.aero.separation + delta[..., separation_start:]),
    )


def state_difference(reference: AircraftState, value: AircraftState) -> Array:
    """Return the minimal local coordinates taking ``reference`` to ``value``."""

    attitude_delta = quaternion_multiply(
        quaternion_conjugate(reference.rigid_body.attitude), value.rigid_body.attitude
    )
    return jnp.concatenate(
        (
            value.rigid_body.position - reference.rigid_body.position,
            quaternion_to_rotvec(attitude_delta),
            value.rigid_body.velocity - reference.rigid_body.velocity,
            value.rigid_body.angular_velocity - reference.rigid_body.angular_velocity,
            value.actuators.surface_deflection - reference.actuators.surface_deflection,
            value.actuators.propeller_speed - reference.actuators.propeller_speed,
            value.aero.separation - reference.aero.separation,
        ),
        axis=-1,
    )


def control_retract(model: AircraftModel, control: ControlInput, delta: Array) -> ControlInput:
    n_propeller = model.n_propellers
    return ControlInput(
        propeller=control.propeller + delta[..., :n_propeller],
        channel=control.channel + delta[..., n_propeller:],
    )


def tangent_state_labels(model: AircraftModel) -> tuple[str, ...]:
    return (
        "north_m",
        "east_m",
        "down_m",
        "attitude_x_rad",
        "attitude_y_rad",
        "attitude_z_rad",
        "velocity_north_m_s",
        "velocity_east_m_s",
        "velocity_down_m_s",
        "roll_rate_rad_s",
        "pitch_rate_rad_s",
        "yaw_rate_rad_s",
        *(f"surface_{index}_rad" for index in range(model.n_surfaces)),
        *(f"propeller_{index}_rad_s" for index in range(model.n_propellers)),
        *(f"separation_{index}" for index in range(model.n_surfaces)),
    )
