"""Failure injection: time-indexed faults on control surfaces and propellers.

Faults are edits of the compiled model's actuator leaves selected by time, so the core
dynamics stay untouched and everything remains jit-able, batchable, and differentiable:

- a **jam** freezes a surface where it is (its time constant becomes effectively infinite);
- a **hardover** drives a surface to a physical limit and holds it there (its command row is
  zeroed and its bias set to the limit, so the actuator itself moves it at its own rate);
- a **motor-out** takes a propeller's target speed range to zero (it spins down through its
  time constant and acceleration limit);
- **partial power** scales a propeller's maximum target speed, with the same motor dynamics.

:func:`apply_faults` returns the faulted model for a given time; :func:`cascade.env.step`
applies it every control period when a schedule is given, and a batch of schedules is a batch
of failure cases.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from cascade.model import AircraftModel

NEVER = float("inf")
JAMMED_TIME_CONSTANT_S = 1e9


class FaultSchedule(NamedTuple):
    """Fault times per surface and propeller (seconds; infinity means never)."""

    surface_jam_time_s: Array
    surface_hardover_time_s: Array
    surface_hardover_sign: Array
    propeller_failure_time_s: Array
    propeller_power_time_s: Array
    propeller_power_fraction: Array


def no_faults(model: AircraftModel) -> FaultSchedule:
    surfaces, propellers = model.n_surfaces, model.n_propellers
    return FaultSchedule(
        surface_jam_time_s=jnp.full(surfaces, NEVER),
        surface_hardover_time_s=jnp.full(surfaces, NEVER),
        surface_hardover_sign=jnp.ones(surfaces),
        propeller_failure_time_s=jnp.full(propellers, NEVER),
        propeller_power_time_s=jnp.full(propellers, NEVER),
        propeller_power_fraction=jnp.ones(propellers),
    )


def fault_schedule(
    model: AircraftModel,
    *,
    jams: dict[int, float] | None = None,
    hardovers: dict[int, tuple[float, float]] | None = None,
    motor_out: dict[int, float] | None = None,
    partial_power: dict[int, tuple[float, float]] | None = None,
) -> FaultSchedule:
    """Build a schedule from indices: ``jams={surface: time}``, ``hardovers={surface: (time,
    sign)}``, ``motor_out={propeller: time}``, ``partial_power={propeller: (time, fraction)}``.

    Indices must address an existing actuator, times must be nonnegative (infinity means
    never), hardover signs must be -1 or +1, and target-speed fractions must lie in [0, 1].
    This host-side constructor validates inputs before building the traceable schedule.
    """

    schedule = no_faults(model)
    for surface, time in (jams or {}).items():
        _validate_fault(surface, time, model.n_surfaces, "surface")
        schedule = schedule._replace(
            surface_jam_time_s=schedule.surface_jam_time_s.at[surface].set(time)
        )
    for surface, (time, sign) in (hardovers or {}).items():
        _validate_fault(surface, time, model.n_surfaces, "surface")
        if isinstance(sign, bool) or not isinstance(sign, Real) or sign not in (-1, 1):
            raise ValueError("hardover sign must be -1 or +1")
        schedule = schedule._replace(
            surface_hardover_time_s=schedule.surface_hardover_time_s.at[surface].set(time),
            surface_hardover_sign=schedule.surface_hardover_sign.at[surface].set(sign),
        )
    for propeller, time in (motor_out or {}).items():
        _validate_fault(propeller, time, model.n_propellers, "propeller")
        schedule = schedule._replace(
            propeller_failure_time_s=schedule.propeller_failure_time_s.at[propeller].set(time)
        )
    for propeller, (time, fraction) in (partial_power or {}).items():
        _validate_fault(propeller, time, model.n_propellers, "propeller")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, Real)
            or not math.isfinite(fraction)
            or not 0.0 <= fraction <= 1.0
        ):
            raise ValueError("partial-power target-speed fraction must be finite and in [0, 1]")
        schedule = schedule._replace(
            propeller_power_time_s=schedule.propeller_power_time_s.at[propeller].set(time),
            propeller_power_fraction=schedule.propeller_power_fraction.at[propeller].set(fraction),
        )
    return schedule


def _validate_fault(index: int, time: float, count: int, kind: str) -> None:
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < count:
        raise ValueError(f"{kind} index must be an integer in [0, {count})")
    if isinstance(time, bool) or not isinstance(time, Real) or math.isnan(time) or time < 0.0:
        raise ValueError("fault time must be nonnegative; use infinity for never")


def apply_faults(model: AircraftModel, schedule: FaultSchedule, time_s: Array) -> AircraftModel:
    """The model with every fault whose time has come applied to its actuator leaves."""

    actuators = model.actuators
    time = jnp.asarray(time_s)
    jammed = time >= schedule.surface_jam_time_s
    hardover = time >= schedule.surface_hardover_time_s
    failed = time >= schedule.propeller_failure_time_s
    derated = time >= schedule.propeller_power_time_s
    surface_time_constant = jnp.where(
        jammed, JAMMED_TIME_CONSTANT_S, actuators.surface_time_constant
    )
    surface_map = jnp.where(hardover[..., None], 0.0, actuators.surface_map)
    surface_bias = jnp.where(
        hardover, schedule.surface_hardover_sign * actuators.surface_limit, actuators.surface_bias
    )
    power = jnp.where(derated, schedule.propeller_power_fraction, 1.0)
    speed_max = jnp.where(failed, 0.0, actuators.propeller_speed_max * power)
    speed_min = jnp.where(failed, 0.0, jnp.minimum(actuators.propeller_speed_min, speed_max))
    return model._replace(
        actuators=actuators._replace(
            surface_time_constant=surface_time_constant,
            surface_map=surface_map,
            surface_bias=surface_bias,
            propeller_speed_max=speed_max,
            propeller_speed_min=speed_min,
        )
    )


__all__ = ["NEVER", "FaultSchedule", "apply_faults", "fault_schedule", "no_faults"]
