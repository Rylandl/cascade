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


class BodyResult(NamedTuple):
    force_body: Array
    moment_body: Array
    coefficients: Array
    angle_of_attack: Array
    sideslip: Array
    airspeed: Array


class AerodynamicResult(NamedTuple):
    force_body: Array
    moment_body: Array
    force_per_surface: Array
    moment_per_surface: Array
    air: SurfaceAirData
    body: BodyResult


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

    Thrust follows the polynomial map ``T / rho = D^4 sum_ij c_ij n^(i+1) (V_a / D)^j`` in shaft
    speed and axial inflow, which is exactly zero for a stopped propeller and turns into
    windmilling drag where the map goes negative. The induced velocity is the momentum-theory
    wake increment with axial inflow; ``validate_model`` checks the map over the operating range
    so the root's discriminant stays non-negative and the result finite and differentiable.
    """

    propellers = model.propellers
    revolutions = jnp.maximum(state.actuators.propeller_speed, 0.0) / (2.0 * jnp.pi)
    diameter = propellers.diameter
    rotational_velocity = jnp.cross(
        state.rigid_body.angular_velocity[..., None, :], propellers.position, axis=-1
    )
    local_velocity = air_velocity_body[..., None, :] + rotational_velocity
    axial_speed = jnp.sum(local_velocity * propellers.direction, axis=-1)

    speed_powers = jnp.stack((revolutions, jnp.square(revolutions)), axis=-1)
    inflow = axial_speed / diameter
    inflow_powers = jnp.stack((jnp.ones_like(inflow), inflow, jnp.square(inflow)), axis=-1)
    thrust_per_density = diameter**4 * jnp.einsum(
        "...pij,...pi,...pj->...p", propellers.thrust_map, speed_powers, inflow_powers
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
    adds an intrinsic pitching-moment increment in both regimes and a profile-drag increment
    in attached flow. The all-moving share has already rotated the frame and therefore
    ``angle_of_attack``.
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
        surfaces.normal_force_coefficient * absolute_sine** 3
        + surfaces.edge_drag_coefficient * jnp.square(cosine)
    )
    # The flap's extra normal force acts on the flap, aft of the quarter chord. The attached
    # flap moment and lift increment fix that arm, -Cm_flap / (CL_alpha tau) chords, and the
    # separated load keeps it, so a stalled elevon still pitches the surface. Without this the
    # post-stall pitch authority of a flying wing is only the panel's own lever arm about the
    # centre of mass, and a tailsitter cannot pitch up out of forward flight.
    flap_lift_slope = surfaces.lift_curve_slope * surfaces.flap_effectiveness
    flap_arm = jnp.where(
        flap_lift_slope > 0.0,
        -surfaces.moment_coefficient_flap / jnp.where(flap_lift_slope > 0.0, flap_lift_slope, 1.0),
        0.0,
    )
    clean_sine = jnp.sin(angle_of_attack)
    normal_clean = surfaces.normal_force_coefficient * clean_sine * smooth_abs(clean_sine)
    moment_separated = -flap_arm * (normal - normal_clean)

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


def body_aerodynamics(
    model: AircraftModel,
    state: AircraftState,
    environment: Environment,
    air_velocity_body: Array,
) -> BodyResult:
    """Evaluate the whole-aircraft coefficient block about the center of mass.

    Forces are formed in the wind frame from ``C_L``, ``C_D``, ``C_Y`` and rotated to the body
    with the standard wind-to-body rotation; moments are ``q S b C_l``, ``q S c C_m``,
    ``q S b C_n`` directly in body axes. Rate terms are written with one power of airspeed
    fewer so that the ``1 / V_a`` of the non-dimensional rate never appears and the block is
    finite at rest. Slipstream does not reach the body block; it belongs to component surfaces.
    """

    body = model.body
    axial, spanwise, normal = (
        air_velocity_body[..., 0],
        air_velocity_body[..., 1],
        air_velocity_body[..., 2],
    )
    planar_speed_squared = jnp.square(axial) + jnp.square(normal)
    planar_speed = jnp.sqrt(planar_speed_squared + 1e-8)
    airspeed = jnp.sqrt(planar_speed_squared + jnp.square(spanwise) + 1e-8)
    angle_regularizer = 1e-3 * jnp.exp(-planar_speed_squared / 1e-6)
    alpha = jnp.arctan2(normal, axial + angle_regularizer)
    beta = jnp.arctan2(spanwise, planar_speed)
    rate_p, rate_q, rate_r = (
        state.rigid_body.angular_velocity[..., 0],
        state.rigid_body.angular_velocity[..., 1],
        state.rigid_body.angular_velocity[..., 2],
    )
    deflection = jnp.einsum(
        "...ds,...s->...d", body.deflection_map, state.actuators.surface_deflection
    )
    aileron, elevator, rudder = deflection[..., 0], deflection[..., 1], deflection[..., 2]

    separation = sigmoid((smooth_abs(alpha) - body.stall_angle) / body.stall_width)
    attached = 1.0 - separation
    sine, cosine = jnp.sin(alpha), jnp.cos(alpha)
    absolute_sine = smooth_abs(sine)
    lift, drag, side = body.lift, body.drag, body.side
    roll, pitch, yaw = body.roll, body.pitch, body.yaw

    plate_normal = body.normal_force_coefficient * sine * absolute_sine
    lift_static = (
        attached * (lift.zero + lift.alpha * alpha)
        + separation * plate_normal * cosine
        + lift.elevator * elevator
    )
    drag_static = (
        attached * (drag.zero + drag.alpha * alpha + drag.alpha_sq * jnp.square(alpha))
        + separation * body.normal_force_coefficient * absolute_sine**3
        + drag.beta * beta
        + drag.beta_sq * jnp.square(beta)
        + drag.elevator_sq * jnp.square(elevator)
    )
    side_static = side.zero + side.beta * beta + side.aileron * aileron + side.rudder * rudder
    roll_static = roll.zero + roll.beta * beta + roll.aileron * aileron + roll.rudder * rudder
    pitch_static = (
        attached * (pitch.zero + pitch.alpha * alpha)
        + separation * body.pitch_flat_plate * sine * absolute_sine
        + pitch.elevator * elevator
    )
    yaw_static = yaw.zero + yaw.beta * beta + yaw.aileron * aileron + yaw.rudder * rudder

    half_chord = 0.5 * model.reference_chord
    half_span = 0.5 * model.reference_span
    lift_rate = lift.q * half_chord * rate_q
    drag_rate = drag.q * half_chord * rate_q
    side_rate = half_span * (side.p * rate_p + side.r * rate_r)
    roll_rate = half_span * (roll.p * rate_p + roll.r * rate_r)
    pitch_rate = pitch.q * half_chord * rate_q
    yaw_rate = half_span * (yaw.p * rate_p + yaw.r * rate_r)

    dynamic_pressure = 0.5 * environment.density * jnp.square(airspeed)
    rate_pressure = 0.5 * environment.density * airspeed
    area = model.reference_area
    lift_force = area * (dynamic_pressure * lift_static + rate_pressure * lift_rate)
    drag_force = area * (dynamic_pressure * drag_static + rate_pressure * drag_rate)
    side_force = area * (dynamic_pressure * side_static + rate_pressure * side_rate)
    span_area = area * model.reference_span
    chord_area = area * model.reference_chord
    roll_moment = span_area * (dynamic_pressure * roll_static + rate_pressure * roll_rate)
    pitch_moment = chord_area * (dynamic_pressure * pitch_static + rate_pressure * pitch_rate)
    yaw_moment = span_area * (dynamic_pressure * yaw_static + rate_pressure * yaw_rate)

    sine_beta, cosine_beta = jnp.sin(beta), jnp.cos(beta)
    # Standard wind-to-body rotation applied to (-D, Y, -L).
    axial_force = -drag_force * cosine_beta - side_force * sine_beta
    force_body = jnp.stack(
        (
            axial_force * cosine + lift_force * sine,
            -drag_force * sine_beta + side_force * cosine_beta,
            axial_force * sine - lift_force * cosine,
        ),
        axis=-1,
    )
    moment_body = jnp.stack((roll_moment, pitch_moment, yaw_moment), axis=-1)
    coefficients = jnp.stack(
        (
            lift_static + lift_rate / airspeed,
            drag_static + drag_rate / airspeed,
            side_static + side_rate / airspeed,
            roll_static + roll_rate / airspeed,
            pitch_static + pitch_rate / airspeed,
            yaw_static + yaw_rate / airspeed,
        ),
        axis=-1,
    )
    return BodyResult(
        force_body=force_body,
        moment_body=moment_body,
        coefficients=coefficients,
        angle_of_attack=alpha,
        sideslip=beta,
        airspeed=airspeed,
    )


def aerodynamics(
    model: AircraftModel,
    state: AircraftState,
    environment: Environment,
    air_velocity_body: Array,
    induced_velocity: Array,
) -> AerodynamicResult:
    """Evaluate component and whole-aircraft aerodynamics about the center of mass."""

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

    body = body_aerodynamics(model, state, environment, air_velocity_body)
    return AerodynamicResult(
        force_body=jnp.sum(force_body, axis=-2) + body.force_body,
        moment_body=jnp.sum(moment_body, axis=-2) + body.moment_body,
        force_per_surface=force_body,
        moment_per_surface=moment_body,
        air=air,
        body=body,
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
