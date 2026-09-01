from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.dynamics import evaluate_dynamics
from cascade.initialization import (
    equilibrate_internal_state,
    standard_environment,
    zero_control,
    zero_state,
)
from cascade.model import AircraftModel
from cascade.state import AircraftState, ControlInput, Environment


@dataclass(frozen=True, slots=True)
class AerodynamicSweep:
    """Vectorized equilibrium aerodynamic data in body-axis coefficient convention."""

    angle_of_attack_rad: Array
    sideslip_rad: Array
    airspeed_m_s: Array
    state: AircraftState
    control: ControlInput
    force_body_n: Array
    moment_body_nm: Array
    force_coefficient_body: Array
    moment_coefficient_body: Array
    surface_separation: Array


def velocity_from_air_angles(airspeed_m_s: Array, alpha_rad: Array, beta_rad: Array) -> Array:
    """Convert airspeed, angle of attack, and sideslip to an FRD air-velocity vector."""

    cosine_beta = jnp.cos(beta_rad)
    return jnp.stack(
        (
            airspeed_m_s * jnp.cos(alpha_rad) * cosine_beta,
            airspeed_m_s * jnp.sin(beta_rad),
            airspeed_m_s * jnp.sin(alpha_rad) * cosine_beta,
        ),
        axis=-1,
    )


def aerodynamic_sweep(
    model: AircraftModel,
    angle_of_attack_rad: Array,
    *,
    airspeed_m_s: Array | float = 12.0,
    sideslip_rad: Array | float = 0.0,
    control: ControlInput | None = None,
    environment: Environment | None = None,
) -> AerodynamicSweep:
    """Evaluate arbitrary broadcastable air-angle grids in one batched dynamics call.

    Actuator and separation states are placed at their local equilibria. Force coefficients are
    ``[CX, CY, CZ]`` in body FRD; moment coefficients are ``[Cl, Cm, Cn]`` using reference span,
    chord, and span respectively. Propeller thrust is excluded, while its slipstream effect on the
    aerodynamic surfaces is retained.
    """

    environment = standard_environment() if environment is None else environment
    alpha, beta, speed = jnp.broadcast_arrays(
        jnp.asarray(angle_of_attack_rad),
        jnp.asarray(sideslip_rad),
        jnp.asarray(airspeed_m_s),
    )
    # Retain useful host-side errors without breaking jit/grad when these values are tracers.
    if not any(isinstance(value, jax.core.Tracer) for value in (alpha, beta, speed)):
        if np.any(np.asarray(speed) <= 0.0) or not np.all(np.isfinite(np.asarray(speed))):
            raise ValueError("sweep airspeed must be finite and positive")
        if not np.all(np.isfinite(np.asarray(alpha))) or not np.all(np.isfinite(np.asarray(beta))):
            raise ValueError("sweep angles must be finite")

    batch_shape = alpha.shape
    environment = _broadcast_environment(environment, batch_shape)
    control = (
        zero_control(model, batch_shape)
        if control is None
        else _broadcast_control(model, control, batch_shape)
    )
    air_velocity_body = velocity_from_air_angles(speed, alpha, beta)
    state = zero_state(model, batch_shape)
    state = state._replace(
        rigid_body=state.rigid_body._replace(velocity=air_velocity_body + environment.wind)
    )
    state = equilibrate_internal_state(model, state, control, environment)
    result = evaluate_dynamics(model, state, control, environment)

    dynamic_pressure = 0.5 * environment.density * jnp.square(speed)
    force_scale = dynamic_pressure * model.reference_area
    moment_scale = force_scale[..., None] * jnp.array(
        [model.reference_span, model.reference_chord, model.reference_span]
    )
    return AerodynamicSweep(
        angle_of_attack_rad=alpha,
        sideslip_rad=beta,
        airspeed_m_s=speed,
        state=state,
        control=control,
        force_body_n=result.aerodynamics.force_body,
        moment_body_nm=result.aerodynamics.moment_body,
        force_coefficient_body=result.aerodynamics.force_body / force_scale[..., None],
        moment_coefficient_body=result.aerodynamics.moment_body / moment_scale,
        surface_separation=state.aero.separation,
    )


def _broadcast_environment(environment: Environment, batch_shape: tuple[int, ...]) -> Environment:
    return Environment(
        density=jnp.broadcast_to(environment.density, batch_shape),
        wind=jnp.broadcast_to(environment.wind, (*batch_shape, 3)),
        gravity=jnp.broadcast_to(environment.gravity, (*batch_shape, 3)),
    )


def _broadcast_control(
    model: AircraftModel, control: ControlInput, batch_shape: tuple[int, ...]
) -> ControlInput:
    return ControlInput(
        propeller=jnp.broadcast_to(control.propeller, (*batch_shape, model.n_propellers)),
        channel=jnp.broadcast_to(control.channel, (*batch_shape, model.n_control_channels)),
    )
