"""A stepped, stateful plant over the functional core for identification and control tooling.

The plant mirrors the shape of a hardware-in-the-loop interface: reset to a state, hold one
command for one sample interval, read back telemetry. State crosses the boundary as the
canonical NWU/FLU scalar-first 13-vector from :mod:`cascade.canonical`; commands and reported
controls use the aircraft specification's channel units. Only this wrapper holds state; every
step is one jitted :func:`cascade.integration.rollout` of the pure dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Fixed execution contract for one plant."""

    simulation_frequency_hz: int = 400
    control_frequency_hz: int = 40
    density_kg_m3: float = 1.225
    gravity_m_s2: float = 9.80665
    step: StepFunction = rk4_step

    def __post_init__(self) -> None:
        if self.simulation_frequency_hz < 1 or self.control_frequency_hz < 1:
            raise ValueError("plant frequencies must be positive")
        if self.simulation_frequency_hz % self.control_frequency_hz:
            raise ValueError("simulation frequency must be a multiple of the control frequency")
        if not np.isfinite(self.density_kg_m3) or self.density_kg_m3 <= 0.0:
            raise ValueError("density must be finite and positive")
        if not np.isfinite(self.gravity_m_s2):
            raise ValueError("gravity must be finite")

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.control_frequency_hz

    @property
    def simulation_steps_per_control(self) -> int:
        return self.simulation_frequency_hz // self.control_frequency_hz


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
            density=self.config.density_kg_m3, gravity=self.config.gravity_m_s2
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
        """Reset to a canonical state with actuators and separation at their equilibria."""

        canonical = _finite_vector("canonical state", state, CANONICAL_STATE_SIZE)
        applied = (
            np.zeros(self.control_size)
            if applied_control is None
            else _finite_vector("applied control", applied_control, self.control_size)
        )
        control = control_from_array(self.model, jnp.asarray(applied))
        self._set_wind(wind_nwu)
        base = zero_state(self.model)
        rigid_body = rigid_body_from_canonical(jnp.asarray(canonical))
        self._state = equilibrate_internal_state(
            self.model, base._replace(rigid_body=rigid_body), control, self._environment
        )
        self._control = control
        self._steps = 0
        return self.snapshot()

    def step(self, command: Any, *, wind_nwu: Any | None = None) -> PlantSample:
        """Hold one command for one control interval and return the new telemetry."""

        if self._state is None:
            raise RuntimeError("reset the plant before stepping it")
        control = control_from_array(
            self.model, jnp.asarray(_finite_vector("command", command, self.control_size))
        )
        self._set_wind(wind_nwu)
        self._state = self._advance(self._state, control, self._environment)
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

    def _set_wind(self, wind_nwu: Any | None) -> None:
        if wind_nwu is None:
            return
        wind = _finite_vector("wind", wind_nwu, 3)
        self._environment = self._environment._replace(wind=nwu_to_ned(jnp.asarray(wind)))


def _finite_vector(name: str, value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result
