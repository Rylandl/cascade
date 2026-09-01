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
    thrust: Array
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
    """Return body-from-surface rotations after physical surface deflection.

    Only the all-moving share of a deflection rotates the surface frame. The flap share changes
    the attached-flow coefficients instead, see :func:`aerodynamic_coefficients`.
    """

    deflection_rotation = rotation_y(model.surfaces.all_moving_fraction * deflection)
    return jnp.einsum(
        "...sij,...sjk->...sik", model.surfaces.body_from_surface, deflection_rotation
    )


def propulsion(
    model: AircraftModel,
    state: AircraftState,
    environment: Environment,
    air_velocity_body: Array,
) -> PropulsionResult:
    """Calculate propeller wrench and momentum-theory induced velocity with axial inflow.

    Thrust follows ``T = rho n^2 D^4 C_T0 (1 - J / J_0)`` with advance ratio ``J = V_a / (n D)``
    written without dividing by ``n``, so it is exactly zero for a stopped propeller and turns
    into windmilling drag beyond the zero-thrust advance ratio. The induced velocity is the
    momentum-theory wake increment with axial inflow; ``validate_model`` bounds ``C_T0`` so its
    discriminant is a sum of squares and the result stays finite and differentiable everywhere.
    """

    propellers = model.propellers
    revolutions = jnp.maximum(state.actuators.propeller_speed, 0.0) / (2.0 * jnp.pi)
    diameter = propellers.diameter
    rotational_velocity = jnp.cross(
        state.rigid_body.angular_velocity[..., None, :], propellers.position, axis=-1
    )
    local_velocity = air_velocity_body[..., None, :] + rotational_velocity
    axial_speed = jnp.sum(local_velocity * propellers.direction, axis=-1)

    tip_advance = revolutions * diameter
    thrust_per_density = (
        propellers.thrust_coefficient
        * jnp.square(diameter)
        * tip_advance
        * (tip_advance - axial_speed / propellers.zero_thrust_advance_ratio)
    )
    density = environment.density[..., None]
    thrust = density * thrust_per_density
    torque = density * jnp.square(revolutions) * diameter**5 * propellers.torque_coefficient
    force_per_propeller = thrust[..., None] * propellers.direction

    arm_moment = jnp.cross(propellers.position, force_per_propeller, axis=-1)
    reaction_moment = (-propellers.spin_direction * torque)[..., None] * propellers.direction
    force_body = jnp.sum(force_per_propeller, axis=-2)
    moment_body = jnp.sum(arm_moment + reaction_moment, axis=-2)

    # Momentum theory ``v_i (|V_a| + v_i) = T / (2 rho A)`` solved by the cancellation-free root
    # ``2k / (sqrt(V_a^2 + 4k) + |V_a|)``: exactly zero at zero thrust, ``sqrt(k)`` in hover,
    # negative when windmilling, and treated symmetrically in reverse flow where momentum
    # theory has no valid branch anyway.
    disk_area = 0.25 * jnp.pi * jnp.square(diameter)
    momentum = thrust_per_density / (2.0 * disk_area)
    root = jnp.sqrt(jnp.maximum(jnp.square(axial_speed) + 4.0 * momentum, 0.0) + 1e-12)
    induced_velocity = 2.0 * momentum / (root + smooth_abs(axial_speed) + 1e-6)
    return PropulsionResult(
        force_body=force_body,
        moment_body=moment_body,
        thrust=thrust,
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
    model: AircraftModel, aero_state: AeroState, angle_of_attack: Array, deflection: Array
) -> tuple[Array, Array, Array]:
    """Blend attached and flat-plate coefficients over the full angular envelope.

    ``deflection`` is the physical surface angle. Its flap share shifts both the attached lift
    curve and the separated flat-plate incidence by ``flap_effectiveness`` times the angle, and
    adds intrinsic pitching-moment and profile-drag increments to attached flow only. The
    all-moving share has already rotated the frame and therefore ``angle_of_attack``.
    """

    surfaces = model.surfaces
    flap_deflection = (1.0 - surfaces.all_moving_fraction) * deflection
    # The attached approximation cannot diverge when its lag state is stale during a violent
    # maneuver. It is softly bounded and forcibly faded beyond twice the static stall angle.
    attached_angle_limit = 3.0 * surfaces.stall_angle
    attached_angle = attached_angle_limit * jnp.tanh(angle_of_attack / attached_angle_limit)
    effective_angle = attached_angle + surfaces.flap_effectiveness * flap_deflection
    lift_attached = surfaces.lift_coefficient_zero + surfaces.lift_curve_slope * effective_angle
    drag_attached = (
        surfaces.drag_coefficient_zero
        + surfaces.induced_drag_factor * jnp.square(lift_attached)
        + surfaces.drag_coefficient_flap * jnp.square(flap_deflection)
    )
    moment_attached = (
        surfaces.moment_coefficient_zero
        + surfaces.moment_coefficient_alpha * attached_angle
        + surfaces.moment_coefficient_flap * flap_deflection
    )

    # A deflected flap on a separated surface still rotates the plate's mean line, so the same
    # effectiveness shifts the flat-plate incidence. This is what gives stalled ailerons and
    # elevators their remaining authority in post-stall and prop-hanging flight.
    separated_angle = angle_of_attack + surfaces.flap_effectiveness * flap_deflection
    sine, cosine = jnp.sin(separated_angle), jnp.cos(separated_angle)
    absolute_sine = smooth_abs(sine)
    normal = surfaces.normal_force_coefficient * sine * absolute_sine
    lift_separated = normal * cosine
    drag_separated = (
        surfaces.normal_force_coefficient * absolute_sine**3
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
        model, state.aero, air.angle_of_attack, state.actuators.surface_deflection
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
