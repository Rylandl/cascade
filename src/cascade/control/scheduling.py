"""Host-built airspeed schedules with pure JAX gain interpolation and stateful control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.control.autotune import StepReport, TuningReport, step_response, tune_cascade
from cascade.control.loops import (
    CascadeController,
    CascadeState,
    GuidanceSetpoint,
    cascade_step,
)
from cascade.integration import StepFunction, rk4_step
from cascade.math import safe_norm
from cascade.model import AircraftModel
from cascade.spec import AircraftSpec
from cascade.state import AircraftState, Environment


class GainSchedule(NamedTuple):
    """Increasing airspeeds ``[K]`` and controllers with a leading ``K`` axis.

    Construct on the host with :func:`build_gain_schedule`. Channel maps and loop periods
    are identical at every knot; gains, limits, rate feedforward, trim throttle and trim pitch
    interpolate linearly. Interpolation alone does not establish closed-loop stability.
    """

    airspeeds_m_s: Array
    controllers: CascadeController


@dataclass(frozen=True)
class ScheduleReport:
    tuning: tuple[TuningReport, ...]
    responses: tuple[StepReport, ...]
    simulation_dt_s: float
    response_duration_s: float


def _grid(airspeeds_m_s):
    values = np.asarray(airspeeds_m_s, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(
            "airspeeds must be at least two finite, positive, strictly increasing knots"
        )
    return values


def build_gain_schedule(
    airspeeds_m_s: Sequence[float], controllers: Sequence[CascadeController]
) -> GainSchedule:
    """Validate and stack already accepted unbatched controllers; no tuning is performed.

    The caller is responsible for accepting custom points. :func:`tune_gain_schedule` also
    checks each tuned point against a closed-loop step response before constructing the table.
    """

    speeds = _grid(airspeeds_m_s)
    points = tuple(controllers)
    if len(points) != len(speeds):
        raise ValueError("one controller is required per airspeed knot")
    if any(not isinstance(point, CascadeController) for point in points):
        raise ValueError("schedule points must be CascadeController values")
    shapes = jax.tree.map(np.shape, points[0])
    for index, point in enumerate(points):
        if jax.tree.map(np.shape, point) != shapes:
            raise ValueError("all schedule controllers must have the same topology and leaf shapes")
        if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(point)):
            raise ValueError(f"controller {index} contains non-finite parameters")
        if point.channels.matrix.ndim != 2 or point.channels.matrix.shape[-1] != 3:
            raise ValueError("channel matrix must have unbatched shape (C, 3)")
        if point.channels.limit.shape != (point.channels.matrix.shape[0],):
            raise ValueError("channel limits must have shape (C,)")
        if np.any(np.asarray(point.channels.limit) < 0):
            raise ValueError("channel limits must be non-negative")
        for group in (point.rate, point.attitude):
            if any(np.shape(value) != (3,) for value in group):
                raise ValueError("rate and attitude gains must be unbatched three-axis vectors")
        for name, value in zip(point.guidance._fields, point.guidance, strict=True):
            expected = (2,) if name in ("throttle_limits", "pitch_limits") else ()
            if np.shape(value) != expected:
                raise ValueError(f"guidance.{name} must have shape {expected}")
        for name in ("rate_period", "attitude_period", "guidance_period"):
            period = np.asarray(getattr(point, name))
            if period.shape != () or period.dtype.kind not in "iu" or period <= 0:
                raise ValueError("loop periods must be positive integers")
            if int(period) != int(getattr(points[0], name)):
                raise ValueError("loop scheduling periods must be identical at every knot")
        for name in ("matrix", "limit"):
            if not np.array_equal(getattr(point.channels, name), getattr(points[0].channels, name)):
                raise ValueError("channel mapping and limits must be identical at every knot")
        for limit in (point.guidance.throttle_limits, point.guidance.pitch_limits):
            if float(limit[0]) > float(limit[1]):
                raise ValueError("guidance limit lower bound must not exceed upper bound")
        if any(
            np.any(np.asarray(value) < 0)
            for value in (
                point.rate.integral_limit,
                point.attitude.rate_limit,
                point.guidance.climb_rate_limit,
                point.guidance.bank_limit,
            )
        ):
            raise ValueError("controller magnitude limits must be non-negative")
    speed_array = jnp.asarray(speeds)
    if not np.all(np.isfinite(np.asarray(speed_array))) or not np.all(
        np.diff(np.asarray(speed_array)) > 0
    ):
        raise ValueError("airspeed knots collapse in the configured JAX precision")
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *points)
    if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(stacked)):
        raise ValueError("controller parameters overflow in the configured JAX precision")
    return GainSchedule(speed_array, stacked)


def tune_gain_schedule(
    spec: AircraftSpec,
    airspeeds_m_s: Sequence[float],
    *,
    environment: Environment | None = None,
    altitude_m: float = 50.0,
    simulation_dt_s: float = 0.0025,
    maximum_rate_bandwidth_rad_s: float = 12.0,
    response_duration_s: float = 12.0,
) -> tuple[GainSchedule, ScheduleReport]:
    """Tune a single aircraft across speeds and accept only settled step-response points.

    Each point must trim and pass :func:`step_response` at the requested integration step
    and duration. Failed knots raise with their speed; they are never silently dropped.
    Acceptance at knots does not guarantee behavior between them or during fast scheduling.
    """

    speeds = _grid(airspeeds_m_s)
    for name, value in (
        ("simulation_dt_s", simulation_dt_s),
        ("response_duration_s", response_duration_s),
        ("maximum_rate_bandwidth_rad_s", maximum_rate_bandwidth_rad_s),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if response_duration_s <= 2.0 or response_duration_s / simulation_dt_s < 2:
        raise ValueError("response duration must extend beyond the 2 s command step")
    model = spec.to_model()
    controllers, reports, responses = [], [], []
    for speed in speeds:
        try:
            controller, report = tune_cascade(
                spec,
                float(speed),
                model=model,
                environment=environment,
                altitude_m=altitude_m,
                simulation_dt_s=simulation_dt_s,
                maximum_rate_bandwidth_rad_s=maximum_rate_bandwidth_rad_s,
            )
        except ValueError as error:
            raise ValueError(f"schedule knot {speed:g} m/s could not be tuned: {error}") from error
        response = step_response(
            model,
            controller,
            report.trim,
            environment=environment,
            simulation_dt_s=simulation_dt_s,
            duration_s=response_duration_s,
        )
        if not response.finite or not response.settled:
            raise ValueError(f"schedule knot {speed:g} m/s failed its step response: {response}")
        controllers.append(controller)
        reports.append(report)
        responses.append(response)
    return build_gain_schedule(speeds, controllers), ScheduleReport(
        tuple(reports), tuple(responses), simulation_dt_s, response_duration_s
    )


def controller_at_speed(
    schedule: GainSchedule, airspeed_m_s: Array | float, *, bounds: str = "clamp"
) -> CascadeController:
    """Interpolate one controller at a scalar airspeed (use ``vmap`` for a batch).

    ``clamp`` holds the nearest endpoint outside the knot interval and works under JIT.
    ``raise`` rejects out-of-range values on the host; it rejects traced calls rather than
    silently clamping them. Non-finite host queries always raise. Traced non-finite inputs
    propagate non-finite gains, so callers should validate their measurement boundary.
    """

    if bounds not in ("clamp", "raise"):
        raise ValueError("bounds must be 'clamp' or 'raise'")
    speed = jnp.asarray(airspeed_m_s)
    if speed.ndim != 0:
        raise ValueError("airspeed query must be scalar; use vmap for batches")
    if isinstance(speed, jax.core.Tracer):
        if bounds == "raise":
            raise ValueError("bounds='raise' is host-only; validate before tracing or use clamp")
    else:
        if not np.isfinite(float(speed)):
            raise ValueError("airspeed query must be finite")
        if bounds == "raise" and not (
            float(schedule.airspeeds_m_s[0]) <= float(speed) <= float(schedule.airspeeds_m_s[-1])
        ):
            raise ValueError("airspeed is outside the gain schedule")
    knots = schedule.airspeeds_m_s
    speed = jnp.where(jnp.isfinite(speed), jnp.clip(speed, knots[0], knots[-1]), jnp.nan)
    lower = jnp.clip(jnp.searchsorted(knots, speed, side="right") - 1, 0, knots.shape[0] - 2)
    weight = (speed - knots[lower]) / (knots[lower + 1] - knots[lower])

    def interpolate(value):
        interpolated = value[lower] + weight * (value[lower + 1] - value[lower])
        # Preserve stored endpoints exactly, including where subtraction/addition would round.
        return jnp.where(
            speed <= knots[0],
            value[0],
            jnp.where(speed >= knots[-1], value[-1], interpolated),
        )

    points = schedule.controllers
    return CascadeController(
        channels=jax.tree.map(lambda x: x[0], points.channels),
        rate=jax.tree.map(interpolate, points.rate),
        attitude=jax.tree.map(interpolate, points.attitude),
        guidance=jax.tree.map(interpolate, points.guidance),
        rate_period=points.rate_period[0],
        attitude_period=points.attitude_period[0],
        guidance_period=points.guidance_period[0],
    )


def scheduled_cascade_step(
    schedule: GainSchedule,
    cascade_state: CascadeState,
    setpoint: GuidanceSetpoint,
    aircraft_state: AircraftState,
    environment: Environment,
    dt: float,
):
    """Schedule on measured airspeed and perform the usual cascade update without resets.

    Integrals, derivative history, held outputs and the scheduling counter pass directly
    through :func:`cascade_step`; crossing a knot does not reinitialize controller state.
    Endpoint gains are clamped. Use ``vmap`` to schedule different worlds independently.
    """

    speed = safe_norm(aircraft_state.rigid_body.velocity - environment.wind)
    controller = controller_at_speed(schedule, speed)
    return cascade_step(controller, cascade_state, setpoint, aircraft_state, environment, dt)


def scheduled_closed_loop_rollout(
    model: AircraftModel,
    schedule: GainSchedule,
    aircraft_state: AircraftState,
    cascade_state: CascadeState,
    setpoints: GuidanceSetpoint,
    environment: Environment,
    dt: float,
    *,
    step: StepFunction = rk4_step,
    environments: Environment | None = None,
):
    """Scan scheduled control plus integration; trajectory entries are post-step states."""

    def advance(carry, inputs):
        state, controller_state = carry
        setpoint, varying_environment = inputs
        active = environment if varying_environment is None else varying_environment
        control, next_controller = scheduled_cascade_step(
            schedule, controller_state, setpoint, state, active, dt
        )
        next_state = step(model, state, control, active, dt)
        return (next_state, next_controller), (next_state, control, next_controller)

    return jax.lax.scan(advance, (aircraft_state, cascade_state), (setpoints, environments))


__all__ = [
    "GainSchedule",
    "ScheduleReport",
    "build_gain_schedule",
    "tune_gain_schedule",
    "controller_at_speed",
    "scheduled_cascade_step",
    "scheduled_closed_loop_rollout",
]
