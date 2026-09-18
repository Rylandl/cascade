"""Host-validated recording inputs and a shared differentiable open-loop replay."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.canonical import rigid_body_from_canonical, rigid_body_to_canonical
from cascade.initialization import (
    control_from_array,
    equilibrate_internal_state,
    standard_environment,
    zero_state,
)
from cascade.integration import rollout
from cascade.model import AircraftModel
from cascade.state import Environment

if TYPE_CHECKING:
    from cascade.experiments.flight_data import FlightRecord


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["initial", "observed", "commands", "environments", "step_dt"],
    meta_fields=["substeps"],
)
@dataclass(frozen=True)
class ReplayInputs:
    """Prepared JAX arrays; observed/predicted rows exclude the initial sample.

    Commands and environments have already been repeated for each integration substep.
    ``substeps`` is static PyTree metadata so ``replay_record`` works directly under JIT.
    Construct through :func:`prepare_record` at the host boundary.
    """

    initial: Array
    observed: Array
    commands: Array
    environments: Environment
    step_dt: Array
    substeps: int


def prepare_record(
    record: FlightRecord, model: AircraftModel, *, substeps: int = 4
) -> ReplayInputs:
    """Validate a recording against a model and the active JAX precision before replay.

    Row zero supplies only the initial rigid-body state. Command, NED wind and density at
    row ``i`` are held throughout interval ``(time[i-1], time[i]]``. All observed states
    must be representable in the active precision, including later fitting targets.
    """
    from cascade.experiments.flight_data import FlightRecord

    if not isinstance(record, FlightRecord):
        raise TypeError("record must be a FlightRecord")
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    substeps = int(substeps)
    if record.command.shape[1] != model.n_propellers + model.n_control_channels:
        raise ValueError("record command width differs from aircraft")
    if np.any(record.command[:, : model.n_propellers] < 0) or np.any(
        record.command[:, : model.n_propellers] > 1
    ):
        raise ValueError("native throttle commands must be in [0, 1]")
    dtype = np.dtype(jnp.asarray(0.0).dtype)
    # Check the precision conversion on the host before JAX allocates or traces inputs.
    # Overflow is a boundary error, rather than a RuntimeWarning followed by bad dynamics.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        step_dt = np.asarray(float(record.time_s[1] - record.time_s[0]) / substeps, dtype=dtype)
        states = np.asarray(record.canonical_state, dtype=dtype)
        commands = np.asarray(record.command[1:], dtype=dtype)
        density = np.asarray(record.density_kg_m3[1:], dtype=dtype)
        wind = np.asarray(record.wind_ned_m_s[1:], dtype=dtype)
    if not np.isfinite(float(step_dt)) or float(step_dt) <= 0:
        raise ValueError("record timestep is outside the configured JAX precision")
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError("record state or commands overflow the configured JAX precision")
    if not np.isfinite(density).all() or not np.isfinite(wind).all() or np.any(density <= 0):
        raise ValueError("record environment overflows the configured JAX precision")
    environments = Environment(
        density=jnp.repeat(jnp.asarray(density), substeps, axis=0),
        wind=jnp.repeat(jnp.asarray(wind), substeps, axis=0),
        gravity=jnp.broadcast_to(
            standard_environment().gravity, ((len(record.time_s) - 1) * substeps, 3)
        ),
    )
    return ReplayInputs(
        jnp.asarray(states[0]),
        jnp.asarray(states[1:]),
        jnp.repeat(jnp.asarray(commands), substeps, axis=0),
        environments,
        jnp.asarray(step_dt),
        substeps,
    )


def replay_record(model: AircraftModel, inputs: ReplayInputs) -> Array:
    """Pure-JAX canonical predictions at every observed interval endpoint.

    The initial rigid-body state comes from the recording. Unknown actuator and separation
    states are initialized at equilibrium under the first active command and environment.
    Integration then runs open-loop without resetting to subsequent observations. Gradients
    include this model-dependent equilibrium initialization. Output has shape ``(T-1, 13)``.
    No host conversions, clipping of model parameters or nonfinite repair are performed here.
    """
    first_environment = jax.tree.map(lambda value: value[0], inputs.environments)
    first_control = control_from_array(model, inputs.commands[0])
    state = zero_state(model)._replace(rigid_body=rigid_body_from_canonical(inputs.initial))
    state = equilibrate_internal_state(model, state, first_control, first_environment)
    _, states = rollout(
        model,
        state,
        control_from_array(model, inputs.commands),
        first_environment,
        inputs.step_dt,
        environments=inputs.environments,
    )
    rigid_body = jax.tree.map(
        lambda value: value[inputs.substeps - 1 :: inputs.substeps], states.rigid_body
    )
    return rigid_body_to_canonical(rigid_body)
