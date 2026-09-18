"""A stepped, stateful plant over the functional core for identification and control tooling.

The plant mirrors the shape of a hardware-in-the-loop interface: reset to a state, hold one
command for one sample interval, read back telemetry. State crosses the boundary as the
canonical NWU/FLU scalar-first 13-vector from :mod:`cascade.canonical`; commands and reported
controls use the aircraft specification's channel units. Only this wrapper holds state; every
step is one jitted :func:`cascade.integration.rollout` of the pure dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cascade.actuators import control_from_actuators
from cascade.canonical import (
    CANONICAL_STATE_SIZE,
    ned_to_nwu,
    nwu_to_ned,
    rigid_body_from_canonical,
    rigid_body_to_canonical,
)
from cascade.initialization import (
    control_from_array,
    control_to_array,
    equilibrate_internal_state,
    standard_environment,
    zero_state,
)
from cascade.integration import StepFunction, repeat_control, rk4_step, rollout
from cascade.model import AircraftModel
from cascade.spec import AircraftSpec
from cascade.state import AircraftState, ControlInput, Environment


@dataclass(frozen=True)
class PlantConfig:
    """Fixed execution contract for one plant, with positive integer frequencies in Hz."""

    simulation_frequency_hz: int = 400
    control_frequency_hz: int = 40
    density_kg_m3: float = 1.225
    gravity_m_s2: float = 9.80665
    step: StepFunction = rk4_step

    def __post_init__(self) -> None:
        for name in ("simulation_frequency_hz", "control_frequency_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.simulation_frequency_hz % self.control_frequency_hz:
            raise ValueError("simulation frequency must be a multiple of the control frequency")
        if (
            isinstance(self.density_kg_m3, bool)
            or not isinstance(self.density_kg_m3, Real)
            or not math.isfinite(self.density_kg_m3)
            or self.density_kg_m3 <= 0.0
        ):
            raise ValueError("density must be finite and positive")
        if (
            isinstance(self.gravity_m_s2, bool)
            or not isinstance(self.gravity_m_s2, Real)
            or not math.isfinite(self.gravity_m_s2)
        ):
            raise ValueError("gravity must be finite")
        if not callable(self.step):
            raise ValueError("step must be a callable integration function")

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.control_frequency_hz

    @property
    def simulation_steps_per_control(self) -> int:
        return int(self.simulation_frequency_hz // self.control_frequency_hz)


@dataclass(frozen=True)
class PlantSample:
    """Canonical telemetry with both requested and applied actuation."""

    time_s: float
    state: np.ndarray
    commanded_control: np.ndarray
    applied_control: np.ndarray
    surface_deflection_rad: np.ndarray
    propeller_speed_rad_s: np.ndarray
    wind_nwu_m_s: np.ndarray


class Plant:
    """Single-world plant that holds each command for one control interval."""

    def __init__(
        self,
        spec: AircraftSpec,
        config: PlantConfig | None = None,
        *,
        model: AircraftModel | None = None,
    ) -> None:
        self.spec = spec
        self.config = PlantConfig() if config is None else config
        self.model = spec.to_model() if model is None else model
        self.control_names: tuple[str, ...] = (
            *(propeller.name for propeller in spec.propellers),
            *spec.control_channels,
        )
        self._environment: Environment = standard_environment(
            density=_native_array("density", self.config.density_kg_m3),
            gravity=_native_array("gravity", self.config.gravity_m_s2),
        )
        self._state: AircraftState | None = None
        self._control = control_from_array(self.model, jnp.zeros(self.control_size))
        self._steps = 0

        steps = self.config.simulation_steps_per_control
        dt = 1.0 / self.config.simulation_frequency_hz
        model = self.model
        integrator = self.config.step

        def advance(state: AircraftState, control: ControlInput, environment: Environment):
            final, _ = rollout(
                model, state, repeat_control(control, steps), environment, dt, step=integrator
            )
            return final

        self._advance = jax.jit(advance)

    @property
    def control_size(self) -> int:
        return self.model.n_propellers + self.model.n_control_channels

    @property
    def sample_period_s(self) -> float:
        return self.config.sample_period_s

    @property
    def time_s(self) -> float:
        return self._steps / self.config.simulation_frequency_hz

    def reset(
        self,
        state: Any,
        *,
        applied_control: Any | None = None,
        wind_nwu: Any | None = None,
    ) -> PlantSample:
        """Reset to a canonical state with actuators and separation at their equilibria.

        A finite, nonzero canonical quaternion is scale-normalized before conversion to the
        working precision. Its magnitude is immaterial; an all-zero quaternion is invalid.
        Inputs must remain finite in that precision. Rejected resets leave the plant unchanged.
        """

        canonical = _finite_vector("canonical state", state, CANONICAL_STATE_SIZE).copy()
        quaternion = canonical[6:10]
        magnitude = np.max(np.abs(quaternion))
        if magnitude == 0.0:
            raise ValueError("canonical state quaternion must be nonzero")
        # Rescaling first avoids overflow/underflow when squaring very large/small finite
        # input quaternions; the resulting norm lies in [1, 2]. Do this before a float32 cast.
        quaternion /= magnitude
        quaternion /= np.linalg.norm(quaternion)
        applied = (
            np.zeros(self.control_size)
            if applied_control is None
            else _finite_vector("applied control", applied_control, self.control_size)
        )
        control = control_from_array(self.model, _native_array("applied control", applied))
        environment = self._environment_with_wind(wind_nwu)
        base = zero_state(self.model)
        rigid_body = rigid_body_from_canonical(_native_array("canonical state", canonical))
        next_state = equilibrate_internal_state(
            self.model, base._replace(rigid_body=rigid_body), control, environment
        )
        _require_finite_state("reset", next_state)
        self._state = next_state
        self._environment = environment
        self._control = control
        self._steps = 0
        return self.snapshot()

    def step(self, command: Any, *, wind_nwu: Any | None = None) -> PlantSample:
        """Hold one command for one control interval and return the new telemetry.

        Inputs must remain finite in the working precision. Nonfinite integration results
        raise an error without changing the plant, so an unstable step is never committed.
        """

        if self._state is None:
            raise RuntimeError("reset the plant before stepping it")
        control = control_from_array(
            self.model,
            _native_array("command", _finite_vector("command", command, self.control_size)),
        )
        environment = self._environment_with_wind(wind_nwu)
        next_state = self._advance(self._state, control, environment)
        _require_finite_state("step", next_state)
        self._state = next_state
        self._environment = environment
        self._control = control
        self._steps += self.config.simulation_steps_per_control
        return self.snapshot()

    def snapshot(self) -> PlantSample:
        """Return current canonical telemetry without advancing the plant."""

        if self._state is None:
            raise RuntimeError("reset the plant before reading it")
        state = self._state
        applied = control_from_actuators(self.model, state.actuators)
        return PlantSample(
            time_s=self.time_s,
            state=np.asarray(rigid_body_to_canonical(state.rigid_body), dtype=np.float64),
            commanded_control=np.asarray(control_to_array(self._control), dtype=np.float64),
            applied_control=np.asarray(control_to_array(applied), dtype=np.float64),
            surface_deflection_rad=np.asarray(state.actuators.surface_deflection, np.float64),
            propeller_speed_rad_s=np.asarray(state.actuators.propeller_speed, np.float64),
            wind_nwu_m_s=np.asarray(ned_to_nwu(self._environment.wind), dtype=np.float64),
        )

    def _environment_with_wind(self, wind_nwu: Any | None) -> Environment:
        if wind_nwu is None:
            return self._environment
        wind = _finite_vector("wind", wind_nwu, 3)
        return self._environment._replace(wind=nwu_to_ned(_native_array("wind", wind)))


def _finite_vector(name: str, value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _native_array(name: str, value: Any) -> jax.Array:
    """Check representability before crossing the host-to-JAX precision boundary."""
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must remain finite in the working precision ({np.dtype(dtype)})")
    return jnp.asarray(array)


def _require_finite_state(operation: str, state: AircraftState) -> None:
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(state)):
        raise ValueError(f"plant {operation} produced nonfinite state; check inputs and timestep")
