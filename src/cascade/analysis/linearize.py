from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.analysis.coordinates import (
    control_retract,
    state_difference,
    state_retract,
    tangent_state_labels,
    tangent_state_size,
)
from cascade.integration import StepFunction, rk4_step
from cascade.model import AircraftModel
from cascade.state import AircraftState, ControlInput, Environment


@dataclass(frozen=True, slots=True)
class StepLinearization:
    """Minimal-coordinate discrete-time linearization about one state and input."""

    state_matrix: Array
    input_matrix: Array
    nominal_next_state: AircraftState
    timestep: float
    state_labels: tuple[str, ...]

    @property
    def discrete_eigenvalues(self) -> Array:
        return jnp.linalg.eigvals(self.state_matrix)

    def predict(self, state_delta: Array, input_delta: Array) -> Array:
        return self.state_matrix @ state_delta + self.input_matrix @ input_delta


@dataclass(frozen=True, slots=True)
class StabilityMode:
    discrete_eigenvalue: complex
    continuous_eigenvalue: complex
    stable: bool
    frequency_hz: float
    damping_ratio: float | None
    time_constant_s: float | None


def linearize_step(
    model: AircraftModel,
    state: AircraftState,
    control: ControlInput,
    environment: Environment,
    timestep: float,
    *,
    step: StepFunction = rk4_step,
) -> StepLinearization:
    """Differentiate one simulator step in quaternion-safe local coordinates."""

    nominal_next = step(model, state, control, environment, timestep)
    state_size = tangent_state_size(model)
    input_size = model.n_propellers + model.n_control_channels
    zero_state_delta = jnp.zeros(state_size)
    zero_input_delta = jnp.zeros(input_size)

    def transition(state_delta: Array, input_delta: Array) -> Array:
        perturbed_state = state_retract(model, state, state_delta)
        perturbed_control = control_retract(model, control, input_delta)
        next_state = step(model, perturbed_state, perturbed_control, environment, timestep)
        return state_difference(nominal_next, next_state)

    state_matrix, input_matrix = jax.jacfwd(transition, argnums=(0, 1))(
        zero_state_delta, zero_input_delta
    )
    return StepLinearization(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        nominal_next_state=nominal_next,
        timestep=timestep,
        state_labels=tangent_state_labels(model),
    )


def stability_modes(linearization: StepLinearization) -> tuple[StabilityMode, ...]:
    """Convert discrete eigenvalues into continuous rates and readable mode metrics."""

    discrete = np.asarray(jax.device_get(linearization.discrete_eigenvalues), dtype=complex)
    continuous = np.log(discrete.astype(complex)) / linearization.timestep
    modes = []
    for discrete_value, continuous_value in zip(discrete, continuous, strict=True):
        magnitude = abs(continuous_value)
        real = continuous_value.real
        modes.append(
            StabilityMode(
                discrete_eigenvalue=complex(discrete_value),
                continuous_eigenvalue=complex(continuous_value),
                stable=bool(abs(discrete_value) < 1.0 + 1e-7),
                frequency_hz=float(abs(continuous_value.imag) / (2.0 * np.pi)),
                damping_ratio=None if magnitude < 1e-12 else float(-real / magnitude),
                time_constant_s=None if abs(real) < 1e-12 else float(1.0 / abs(real)),
            )
        )
    return tuple(sorted(modes, key=lambda mode: mode.continuous_eigenvalue.real, reverse=True))
