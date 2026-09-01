from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid

from cascade.math import rotation_y, smooth_abs
from cascade.model import AircraftModel
from cascade.state import AeroState, AircraftState, Environment


class PropulsionResult(NamedTuple):
    force_body: Array
    moment_body: Array
    force_per_propeller: Array
    induced_velocity: Array


class SurfaceAirData(NamedTuple):
    velocity: Array
    airspeed: Array
    angle_of_attack: Array
    sideslip: Array
    dynamic_pressure: Array
    separation_equilibrium: Array


class AerodynamicResult(NamedTuple):
    force_body: Array
    moment_body: Array
    force_per_surface: Array
    moment_per_surface: Array
    air: SurfaceAirData


def deflected_surface_frames(model: AircraftModel, deflection: Array) -> Array:
    """Return body-from-surface rotations after physical surface deflection."""

    deflection_rotation = rotation_y(deflection)
    return jnp.einsum(
        "...sij,...sjk->...sik", model.surfaces.body_from_surface, deflection_rotation
    )


def propulsion(model: AircraftModel, propeller_speed: Array, density: Array) -> PropulsionResult:
    """Calculate propeller wrench and static induced slipstream velocity."""

    propellers = model.propellers
    speed_squared = jnp.square(jnp.maximum(propeller_speed, 0.0))
    thrust = propellers.thrust_coefficient * speed_squared
    force_per_propeller = thrust[..., None] * propellers.direction

    arm_moment = jnp.cross(propellers.position, force_per_propeller, axis=-1)
    reaction_moment = (-propellers.spin_direction * propellers.torque_coefficient * speed_squared)[
        ..., None
    ] * propellers.direction

    force_body = jnp.sum(force_per_propeller, axis=-2)
    moment_body = jnp.sum(arm_moment + reaction_moment, axis=-2)
    induced_speed_squared = jnp.maximum(thrust, 0.0) / (
        2.0 * density[..., None] * propellers.disk_area + 1e-8
    )
    induced_velocity = jnp.sqrt(induced_speed_squared + 1e-8) - 1e-4
    return PropulsionResult(
        force_body=force_body,
        moment_body=moment_body,
        force_per_propeller=force_per_propeller,
        induced_velocity=induced_velocity,
    )


def slipstream_velocity(model: AircraftModel, induced_velocity: Array) -> Array:
    """Distribute propeller induced velocity to every aerodynamic surface."""

    return jnp.einsum(
        "...p,...ps,...pj->...sj",
        induced_velocity,
        model.propellers.slipstream_map,
        model.propellers.direction,
    )


def surface_air_data(
    model: AircraftModel,
    state: AircraftState,
    environment: Environment,
    air_velocity_body: Array,
    induced_velocity: Array,
) -> tuple[SurfaceAirData, Array]:
    """Calculate local flow and deflected coordinate frame for every surface."""

    surfaces = model.surfaces
    frames = deflected_surface_frames(model, state.actuators.surface_deflection)
    rotational_velocity = jnp.cross(
        state.rigid_body.angular_velocity[..., None, :], surfaces.position, axis=-1
    )
    local_velocity_body = (
        air_velocity_body[..., None, :]
        + rotational_velocity
        + slipstream_velocity(model, induced_velocity)
    )
    local_velocity = jnp.einsum("...sij,...si->...sj", frames, local_velocity_body)

    axial, spanwise, normal = (
        local_velocity[..., 0],
        local_velocity[..., 1],
        local_velocity[..., 2],
    )
    planar_speed_squared = jnp.square(axial) + jnp.square(normal)
    planar_speed = jnp.sqrt(planar_speed_squared + 1e-8)
    airspeed = jnp.sqrt(planar_speed_squared + jnp.square(spanwise) + 1e-8)
    # Angle is physically undefined at zero airspeed. Regularizing only within roughly 1 mm/s
    # chooses a benign zero-angle limit and, unlike atan2(0, 0), keeps autodiff finite.
    angle_regularizer = 1e-3 * jnp.exp(-planar_speed_squared / 1e-6)
    angle_of_attack = jnp.arctan2(normal, axial + angle_regularizer)
    sideslip = jnp.arctan2(spanwise, planar_speed)
    dynamic_pressure = 0.5 * environment.density[..., None] * planar_speed_squared
    separation_equilibrium = sigmoid(
        (smooth_abs(angle_of_attack) - surfaces.stall_angle) / surfaces.stall_width
    )
    return (
        SurfaceAirData(
            velocity=local_velocity,
            airspeed=airspeed,
            angle_of_attack=angle_of_attack,
            sideslip=sideslip,
            dynamic_pressure=dynamic_pressure,
            separation_equilibrium=separation_equilibrium,
        ),
        frames,
    )


def aerodynamic_coefficients(
    model: AircraftModel, aero_state: AeroState, angle_of_attack: Array
) -> tuple[Array, Array, Array]:
    """Blend attached and flat-plate coefficients over the full angular envelope."""

    surfaces = model.surfaces
    # The attached approximation cannot diverge when its lag state is stale during a violent
    # maneuver. It is softly bounded and forcibly faded beyond twice the static stall angle.
    attached_angle_limit = 3.0 * surfaces.stall_angle
    attached_angle = attached_angle_limit * jnp.tanh(angle_of_attack / attached_angle_limit)
    lift_attached = surfaces.lift_coefficient_zero + surfaces.lift_curve_slope * attached_angle
    drag_attached = surfaces.drag_coefficient_zero + surfaces.induced_drag_factor * jnp.square(
        lift_attached
    )
    moment_attached = (
        surfaces.moment_coefficient_zero + surfaces.moment_coefficient_alpha * attached_angle
    )

    sine, cosine = jnp.sin(angle_of_attack), jnp.cos(angle_of_attack)
    absolute_sine = smooth_abs(sine)
    normal = surfaces.normal_force_coefficient * sine * absolute_sine
    lift_separated = normal * cosine
    drag_separated = (
        surfaces.normal_force_coefficient * absolute_sine** 3
        + surfaces.edge_drag_coefficient * jnp.square(cosine)
    )
    moment_separated = jnp.zeros_like(moment_attached)

    separation_state = jnp.clip(aero_state.separation, 0.0, 1.0)
    forced_separation = sigmoid(
        (smooth_abs(angle_of_attack) - 2.0 * surfaces.stall_angle) / surfaces.stall_width
    )
    separation = 1.0 - (1.0 - separation_state) * (1.0 - forced_separation)
    attached = 1.0 - separation
    lift = attached * lift_attached + separation * lift_separated
    drag = attached * drag_attached + separation * drag_separated
    moment = attached * moment_attached + separation * moment_separated
    return lift, drag, moment


def aerodynamics(
    model: AircraftModel,
    state: AircraftState,
    environment: Environment,
    air_velocity_body: Array,
    induced_velocity: Array,
) -> AerodynamicResult:
    """Evaluate component aerodynamic forces and moments about the center of mass."""

    surfaces = model.surfaces
    air, frames = surface_air_data(model, state, environment, air_velocity_body, induced_velocity)
    lift_coefficient, drag_coefficient, moment_coefficient = aerodynamic_coefficients(
        model, state.aero, air.angle_of_attack
    )

    lift = air.dynamic_pressure * surfaces.area * lift_coefficient
    drag = air.dynamic_pressure * surfaces.area * drag_coefficient
    sine, cosine = jnp.sin(air.angle_of_attack), jnp.cos(air.angle_of_attack)

    force_axial = -drag * cosine + lift * sine
    force_normal = -drag * sine - lift * cosine
    spanwise_velocity = air.velocity[..., 1]
    force_spanwise = (
        -0.5
        * environment.density[..., None]
        * surfaces.area
        * surfaces.span_drag_coefficient
        * spanwise_velocity
        * smooth_abs(spanwise_velocity)
    )
    force_surface = jnp.stack((force_axial, force_spanwise, force_normal), axis=-1)
    force_body = jnp.einsum("...sij,...sj->...si", frames, force_surface)

    pitching_moment = air.dynamic_pressure * surfaces.area * surfaces.chord * moment_coefficient
    intrinsic_moment_surface = jnp.stack(
        (jnp.zeros_like(pitching_moment), pitching_moment, jnp.zeros_like(pitching_moment)),
        axis=-1,
    )
    intrinsic_moment_body = jnp.einsum("...sij,...sj->...si", frames, intrinsic_moment_surface)
    moment_body = jnp.cross(surfaces.position, force_body, axis=-1) + intrinsic_moment_body

    return AerodynamicResult(
        force_body=jnp.sum(force_body, axis=-2),
        moment_body=jnp.sum(moment_body, axis=-2),
        force_per_surface=force_body,
        moment_per_surface=moment_body,
        air=air,
    )


def separation_derivative(model: AircraftModel, state: AeroState, equilibrium: Array) -> Array:
    """Continuous stall hysteresis with different separation and reattachment lags."""

    delta = equilibrium - state.separation
    separating = 0.5 * (1.0 + jnp.tanh(delta / 0.05))
    time_constant = (
        separating * model.surfaces.separation_time_constant
        + (1.0 - separating) * model.surfaces.reattachment_time_constant
    )
    return delta / time_constant
