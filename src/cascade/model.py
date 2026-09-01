from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class SurfaceModel(NamedTuple):
    """Geometry and full-envelope coefficients for ``S`` aerodynamic elements.

    A surface with zero area contributes no force and exists only to carry a physical, lagged,
    limited actuator, for example the elevons of an aircraft modelled by a whole-aircraft
    coefficient block.
    """

    position: Array
    body_from_surface: Array
    area: Array
    chord: Array
    lift_coefficient_zero: Array
    lift_curve_slope: Array
    drag_coefficient_zero: Array
    induced_drag_factor: Array
    moment_coefficient_zero: Array
    moment_coefficient_alpha: Array
    stall_angle: Array
    stall_width: Array
    normal_force_coefficient: Array
    edge_drag_coefficient: Array
    span_drag_coefficient: Array
    separation_time_constant: Array
    reattachment_time_constant: Array
    all_moving_fraction: Array
    flap_effectiveness: Array
    moment_coefficient_flap: Array
    drag_coefficient_flap: Array


class PropellerModel(NamedTuple):
    """Geometry and a polynomial thrust map for ``P`` propellers.

    ``thrust_map`` has shape ``[P, 2, 3]`` and defines
    ``T / rho = D^4 sum_ij thrust_map[i, j] n^(i + 1) (V_a / D)^j`` with ``n`` in revolutions
    per second and ``V_a`` the axial inflow, so a stopped propeller produces exactly no force.
    The classical linear ``C_T(J) = C_T0 (1 - J / J_0)`` is the entry ``[[0, -C_T0 / J_0, 0],
    [C_T0, 0, 0]]``; measured maps and published exit-velocity laws fit the same form.
    ``torque_coefficient`` is the static ``C_Q = Q / (rho n^2 D^5)``.
    """

    position: Array
    direction: Array
    diameter: Array
    torque_coefficient: Array
    spin_direction: Array
    thrust_map: Array
    slipstream_map: Array


class ActuatorModel(NamedTuple):
    """Mapping and dynamics for ``S`` surfaces, ``P`` motors, and ``C`` control channels."""

    surface_map: Array
    surface_bias: Array
    surface_limit: Array
    surface_time_constant: Array
    surface_rate_limit: Array
    propeller_speed_min: Array
    propeller_speed_max: Array
    propeller_time_constant: Array
    propeller_acceleration_limit: Array


class LongitudinalCoefficients(NamedTuple):
    """Lift or pitching-moment polynomial in angle of attack, pitch rate, and elevator."""

    zero: Array
    alpha: Array
    q: Array
    elevator: Array


class DragCoefficients(NamedTuple):
    """Drag polynomial in angle of attack, sideslip, pitch rate, and elevator."""

    zero: Array
    alpha: Array
    alpha_sq: Array
    beta: Array
    beta_sq: Array
    q: Array
    elevator_sq: Array


class LateralCoefficients(NamedTuple):
    """Side-force, rolling-moment, or yawing-moment polynomial in sideslip, rates, and controls."""

    zero: Array
    beta: Array
    p: Array
    r: Array
    aileron: Array
    rudder: Array


class BodyModel(NamedTuple):
    """Whole-aircraft polynomial aerodynamics about the center of mass.

    This is the classical Beard and McLain coefficient form used by published small-UAV models.
    It is evaluated from the air velocity at the center of mass and added to the component
    surfaces, so an aircraft can be described by a coefficient table alone (with zero-area
    surfaces carrying its physical actuators), by components alone, or by a mix. All entries
    are per-world scalars except ``deflection_map`` with shape ``[3, S]``, which forms the
    generalized aileron, elevator, and rudder angles from the physical surface deflections.
    Static angle-of-attack polynomials blend beyond ``stall_angle`` to a flat plate with
    ``normal_force_coefficient`` (about 2 for a thin plate) and ``pitch_flat_plate``.
    """

    lift: LongitudinalCoefficients
    drag: DragCoefficients
    side: LateralCoefficients
    roll: LateralCoefficients
    pitch: LongitudinalCoefficients
    yaw: LateralCoefficients
    stall_angle: Array
    stall_width: Array
    normal_force_coefficient: Array
    pitch_flat_plate: Array
    deflection_map: Array


def zero_body(n_surfaces: int) -> BodyModel:
    """A body block that contributes nothing, for component-only aircraft."""

    zero = jnp.asarray(0.0)
    longitudinal = LongitudinalCoefficients(zero, zero, zero, zero)
    lateral = LateralCoefficients(zero, zero, zero, zero, zero, zero)
    return BodyModel(
        lift=longitudinal,
        drag=DragCoefficients(zero, zero, zero, zero, zero, zero, zero),
        side=lateral,
        roll=lateral,
        pitch=longitudinal,
        yaw=lateral,
        stall_angle=jnp.asarray(0.3),
        stall_width=jnp.asarray(0.05),
        normal_force_coefficient=zero,
        pitch_flat_plate=zero,
        deflection_map=jnp.zeros((3, n_surfaces)),
    )


class AircraftModel(NamedTuple):
    mass: Array
    inertia: Array
    inertia_inverse: Array
    reference_area: Array
    reference_chord: Array
    reference_span: Array
    surfaces: SurfaceModel
    propellers: PropellerModel
    actuators: ActuatorModel
    body: BodyModel

    @property
    def n_surfaces(self) -> int:
        return self.surfaces.area.shape[-1]

    @property
    def n_propellers(self) -> int:
        return self.propellers.diameter.shape[-1]

    @property
    def n_control_channels(self) -> int:
        return self.actuators.surface_map.shape[-1]


def with_computed_inverse(model: AircraftModel) -> AircraftModel:
    """Return a model whose cached inverse agrees with its inertia tensor."""

    return model._replace(inertia_inverse=jnp.linalg.inv(model.inertia))


def broadcast_model(model: AircraftModel, batch_shape: tuple[int, ...]) -> AircraftModel:
    """Broadcast an unbatched validated model for per-world parameter variation.

    The returned arrays are views where possible. Callers can use immutable JAX indexed updates to
    randomize individual parameters while preserving the static surface/propeller topology.
    """

    return jax.tree.map(lambda value: jnp.broadcast_to(value, (*batch_shape, *value.shape)), model)


def _validate_momentum_discriminant(propellers: PropellerModel, actuators: ActuatorModel) -> None:
    """Require ``V_a^2 / 4 + T / (2 rho A) >= 0`` over the whole operating range.

    That is the discriminant of the momentum-theory induced-velocity root. For a fixed shaft
    speed the thrust map makes it a quadratic in axial airspeed, so its minimum is checked
    exactly at every speed of a fine shaft-speed grid up to the propeller's maximum.
    """

    thrust_map = np.asarray(propellers.thrust_map)
    diameters = np.asarray(propellers.diameter)
    maxima = np.asarray(actuators.propeller_speed_max) / (2.0 * np.pi)
    for index in range(thrust_map.shape[0]):
        diameter = diameters[index]
        disk_area = 0.25 * np.pi * diameter**2
        coefficients = thrust_map[index]
        for revolutions in np.linspace(0.0, maxima[index], 65):
            powers = np.array([revolutions, revolutions**2])
            constant = diameter**4 * (coefficients[:, 0] @ powers) / (2.0 * disk_area)
            linear = diameter**3 * (coefficients[:, 1] @ powers) / (2.0 * disk_area)
            quadratic = 0.25 + diameter**2 * (coefficients[:, 2] @ powers) / (2.0 * disk_area)
            if quadratic < -1e-12:
                minimum = -np.inf
            elif quadratic <= 1e-12:
                minimum = constant if abs(linear) <= 1e-9 else -np.inf
            else:
                minimum = constant - linear**2 / (4.0 * quadratic)
            if minimum < -1e-9:
                raise ValueError(
                    f"propeller {index} thrust map violates the momentum-theory bound at "
                    f"{revolutions:.1f} rev/s; induced velocity would become complex"
                )


def validate_model(model: AircraftModel) -> AircraftModel:
    """Validate static shapes and physical invariants at model-construction time."""

    surfaces, propellers, actuators = model.surfaces, model.propellers, model.actuators
    n_surface = surfaces.area.shape[-1]
    n_propeller = propellers.diameter.shape[-1]

    surface_vectors = {
        "position": (n_surface, 3),
        "body_from_surface": (n_surface, 3, 3),
    }
    for name, expected in surface_vectors.items():
        actual = getattr(surfaces, name).shape
        if actual != expected:
            raise ValueError(f"surfaces.{name} must have shape {expected}, got {actual}")
    for name in SurfaceModel._fields[2:]:
        actual = getattr(surfaces, name).shape
        if actual != (n_surface,):
            raise ValueError(f"surfaces.{name} must have shape {(n_surface,)}, got {actual}")

    if propellers.position.shape != (n_propeller, 3):
        raise ValueError("propellers.position must have shape (P, 3)")
    if propellers.direction.shape != (n_propeller, 3):
        raise ValueError("propellers.direction must have shape (P, 3)")
    for name in ("diameter", "torque_coefficient", "spin_direction"):
        if getattr(propellers, name).shape != (n_propeller,):
            raise ValueError(f"propellers.{name} must have shape {(n_propeller,)}")
    if propellers.thrust_map.shape != (n_propeller, 2, 3):
        raise ValueError("propellers.thrust_map must have shape (P, 2, 3)")
    if propellers.slipstream_map.shape != (n_propeller, n_surface):
        raise ValueError("propellers.slipstream_map must have shape (P, S)")

    if actuators.surface_map.shape[0] != n_surface:
        raise ValueError("actuators.surface_map must have shape (S, C)")
    for name in ActuatorModel._fields[1:5]:
        if getattr(actuators, name).shape != (n_surface,):
            raise ValueError(f"actuators.{name} must have shape {(n_surface,)}")
    for name in ActuatorModel._fields[5:]:
        if getattr(actuators, name).shape != (n_propeller,):
            raise ValueError(f"actuators.{name} must have shape {(n_propeller,)}")

    body = model.body
    for name, group in zip(BodyModel._fields, body, strict=True):
        if name == "deflection_map":
            continue
        for leaf in jax.tree.leaves(group):
            if leaf.shape != ():
                raise ValueError(f"body.{name} coefficients must be scalars, got {leaf.shape}")
    if body.deflection_map.shape != (3, n_surface):
        raise ValueError("body.deflection_map must have shape (3, S)")

    arrays = jax.tree.leaves(model)
    if not all(np.all(np.isfinite(np.asarray(jax.device_get(value)))) for value in arrays):
        raise ValueError("model contains non-finite values")
    if float(np.asarray(model.mass)) <= 0:
        raise ValueError("mass must be positive")
    if any(
        float(np.asarray(value)) <= 0
        for value in (model.reference_area, model.reference_chord, model.reference_span)
    ):
        raise ValueError("reference area, chord, and span must be positive")
    if np.any(np.asarray(surfaces.area) < 0):
        raise ValueError("surface areas must be non-negative")
    if np.any(np.asarray(surfaces.chord) <= 0):
        raise ValueError("surface chords must be positive")
    if np.any(np.asarray(surfaces.stall_width) <= 0):
        raise ValueError("stall widths must be positive")
    if np.any(np.asarray(surfaces.separation_time_constant) <= 0):
        raise ValueError("separation time constants must be positive")
    if np.any(np.asarray(surfaces.reattachment_time_constant) <= 0):
        raise ValueError("reattachment time constants must be positive")
    all_moving = np.asarray(surfaces.all_moving_fraction)
    if np.any(all_moving < 0) or np.any(all_moving > 1):
        raise ValueError("all-moving fractions must lie in [0, 1]")
    if np.any(np.asarray(surfaces.flap_effectiveness) < 0):
        raise ValueError("flap effectiveness must be non-negative")
    if np.any(np.asarray(surfaces.drag_coefficient_flap) < 0):
        raise ValueError("flap drag coefficients must be non-negative")
    if float(np.asarray(body.stall_angle)) <= 0 or float(np.asarray(body.stall_width)) <= 0:
        raise ValueError("body stall angle and width must be positive")
    if float(np.asarray(body.normal_force_coefficient)) < 0:
        raise ValueError("body normal-force coefficient must be non-negative")
    if np.any(np.asarray(propellers.diameter) <= 0):
        raise ValueError("propeller diameters must be positive")
    if np.any(np.asarray(propellers.torque_coefficient) < 0):
        raise ValueError("propeller torque coefficients must be non-negative")
    _validate_momentum_discriminant(propellers, actuators)
    if np.any(np.asarray(propellers.slipstream_map) < 0):
        raise ValueError("slipstream weights must be non-negative")
    if np.any(np.asarray(actuators.surface_limit) < 0):
        raise ValueError("surface limits must be non-negative")
    if np.any(np.asarray(actuators.surface_time_constant) <= 0):
        raise ValueError("surface actuator time constants must be positive")
    if np.any(np.asarray(actuators.surface_rate_limit) <= 0):
        raise ValueError("surface rate limits must be positive")
    if np.any(np.asarray(actuators.propeller_time_constant) <= 0):
        raise ValueError("propeller time constants must be positive")
    if np.any(np.asarray(actuators.propeller_acceleration_limit) <= 0):
        raise ValueError("propeller acceleration limits must be positive")
    if np.any(
        np.asarray(actuators.propeller_speed_max) < np.asarray(actuators.propeller_speed_min)
    ):
        raise ValueError("propeller maximum speed must not be below minimum speed")

    orientation = np.asarray(surfaces.body_from_surface)
    identity = np.eye(3)
    if not np.allclose(np.swapaxes(orientation, -1, -2) @ orientation, identity, atol=1e-5):
        raise ValueError("surface frames must be orthonormal")
    if not np.allclose(np.linalg.det(orientation), 1.0, atol=1e-5):
        raise ValueError("surface frames must be proper right-handed rotations")
    directions = np.asarray(propellers.direction)
    if not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-5):
        raise ValueError("propeller directions must be unit vectors")
    if not np.allclose(np.asarray(model.inertia) @ np.asarray(model.inertia_inverse), identity):
        raise ValueError("inertia_inverse does not invert inertia")
    inertia = np.asarray(model.inertia)
    if not np.allclose(inertia, inertia.T, atol=1e-7):
        raise ValueError("inertia must be symmetric")
    if np.any(np.linalg.eigvalsh(inertia) <= 0):
        raise ValueError("inertia must be positive definite")
    return model
