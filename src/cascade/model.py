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
    """Geometry and advance-ratio thrust parameters for ``P`` propellers.

    ``thrust_coefficient`` is the static ``C_T = T / (rho n^2 D^4)`` and ``torque_coefficient``
    the static ``C_Q = Q / (rho n^2 D^5)`` with ``n`` in revolutions per second. Thrust falls
    linearly with advance ratio and reaches zero at ``zero_thrust_advance_ratio``.
    """

    position: Array
    direction: Array
    diameter: Array
    thrust_coefficient: Array
    zero_thrust_advance_ratio: Array
    torque_coefficient: Array
    spin_direction: Array
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
    for name in PropellerModel._fields[2:-1]:
        if getattr(propellers, name).shape != (n_propeller,):
            raise ValueError(f"propellers.{name} must have shape {(n_propeller,)}")
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
    if np.any(np.asarray(propellers.diameter) <= 0):
        raise ValueError("propeller diameters must be positive")
    if np.any(np.asarray(propellers.thrust_coefficient) < 0):
        raise ValueError("propeller thrust coefficients must be non-negative")
    if np.any(np.asarray(propellers.zero_thrust_advance_ratio) <= 0):
        raise ValueError("zero-thrust advance ratios must be positive")
    if np.any(np.asarray(propellers.torque_coefficient) < 0):
        raise ValueError("propeller torque coefficients must be non-negative")
    momentum_bound = 0.5 * np.pi * np.square(np.asarray(propellers.zero_thrust_advance_ratio))
    if np.any(np.asarray(propellers.thrust_coefficient) > momentum_bound):
        raise ValueError(
            "propeller thrust coefficient exceeds the momentum-theory bound "
            "(pi / 2) * zero_thrust_advance_ratio**2"
        )
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
