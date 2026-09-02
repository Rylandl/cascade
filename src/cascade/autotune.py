"""Tune the control cascade for any aircraft from its own trim and linearisation.

No hand tuning: trim at the design cruise, measure each axis's control authority (angular
acceleration per unit channel, through the actuator map) and rate damping (from the
linearised step), and place the rate loops at a bandwidth the actuators can support. The
attitude loops sit below the rate loops, and the airspeed loop's gain comes from the measured
acceleration per unit throttle. The result is the reference controller every sampled
airframe gets without a human, and a yardstick for learners that see none of this.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from cascade.analysis.linearize import linearize_step
from cascade.analysis.trim import StraightFlightCondition, TrimResult, trim_straight_flight
from cascade.archetypes import control_authority
from cascade.control import (
    AttitudeGains,
    CascadeController,
    GuidanceGains,
    GuidanceSetpoint,
    RateGains,
    channel_map,
    closed_loop_rollout,
    initial_cascade_state,
)
from cascade.dynamics import evaluate_dynamics
from cascade.initialization import standard_environment
from cascade.model import AircraftModel
from cascade.spec import AircraftSpec
from cascade.state import Environment

ROLES = {"aileron": (0, "roll"), "elevator": (1, "pitch"), "rudder": (2, "yaw")}


@dataclass(frozen=True)
class TuningReport:
    """What the tuner measured and chose."""

    trim: TrimResult
    authority_rad_s2: np.ndarray
    damping_1_s: np.ndarray
    rate_bandwidth_rad_s: np.ndarray
    throttle_acceleration_m_s2: float


def throttle_acceleration(model: AircraftModel, state, control, environment: Environment) -> float:
    """Forward acceleration per unit throttle at the trim, propellers at their steady speed."""

    actuators = model.actuators

    def acceleration(throttle):
        speed = actuators.propeller_speed_min + throttle * (
            actuators.propeller_speed_max - actuators.propeller_speed_min
        )
        placed = state._replace(actuators=state.actuators._replace(propeller_speed=speed))
        result = evaluate_dynamics(model, placed, control._replace(propeller=throttle), environment)
        forward = result.derivative.rigid_body.velocity
        # Along the flight direction.
        direction = state.rigid_body.velocity / jnp.maximum(
            jnp.linalg.norm(state.rigid_body.velocity), 1e-3
        )
        return jnp.sum(forward * direction)

    jacobian = jax.jacfwd(acceleration)(control.propeller)
    return float(jnp.sum(jacobian))


def tune_cascade(
    spec: AircraftSpec,
    cruise_speed_m_s: float,
    *,
    model: AircraftModel | None = None,
    altitude_m: float = 50.0,
    environment: Environment | None = None,
    simulation_dt_s: float = 0.0025,
    maximum_rate_bandwidth_rad_s: float = 12.0,
) -> tuple[CascadeController, TuningReport]:
    """Trim, measure, and place the loops. Returns the controller and what was measured."""

    model = spec.to_model() if model is None else model
    environment = standard_environment() if environment is None else environment
    trim = trim_straight_flight(
        model,
        StraightFlightCondition(cruise_speed_m_s, altitude_m=altitude_m),
        environment=environment,
    )
    if not trim.success:
        raise ValueError(f"cannot tune without a cruise trim: {trim.message}")
    authority = control_authority(model, trim.state, trim.control, environment)
    linearization = linearize_step(model, trim.state, trim.control, environment, simulation_dt_s)
    labels = linearization.state_labels
    continuous = (np.asarray(linearization.state_matrix) - np.eye(len(labels))) / simulation_dt_s
    rate_index = [
        labels.index(n) for n in ("roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s")
    ]
    damping = np.array([continuous[i, i] for i in rate_index])
    lag = np.asarray(model.actuators.surface_time_constant)
    surface_map = np.asarray(model.actuators.surface_map)

    kp = np.zeros(3)
    ki = np.zeros(3)
    feedforward = np.zeros(3)
    bandwidth = np.zeros(3)
    roles = {}
    for name, (axis, role) in ROLES.items():
        if name not in spec.control_channels:
            continue
        column = spec.control_channels.index(name)
        effectiveness = float(authority[axis, column])
        if abs(effectiveness) < 1e-6:
            continue
        driven = np.abs(surface_map[:, column]) > 1e-9
        slowest = float(np.max(lag[driven])) if np.any(driven) else 0.05
        target = min(maximum_rate_bandwidth_rad_s, 0.35 / max(slowest, 1e-3))
        if axis == 2:
            target *= 0.5
        bandwidth[axis] = target
        kp[axis] = (target - min(damping[axis], 0.0)) / abs(effectiveness)
        ki[axis] = kp[axis] * target
        feedforward[axis] = 0.5 * abs(min(damping[axis], 0.0)) / abs(effectiveness)
        roles[name] = role if effectiveness > 0.0 else f"-{role}"

    attitude_kp = np.where(bandwidth > 0.0, bandwidth / 2.5, 0.0)
    thrust_gain = throttle_acceleration(model, trim.state, trim.control, environment)
    airspeed_kp = 1.0 / (2.5 * max(thrust_gain, 0.5))
    pitch_trim = float(trim.decision[1])
    # The nose may not be raised into the stall: a climb at the rate limit adds its flight-path
    # angle to the trim incidence, and the pitch ceiling keeps the sum below stall by 2 degrees.
    stall = float(np.min(np.asarray(model.surfaces.stall_angle)))
    climb_rate = 0.12 * cruise_speed_m_s
    climb_angle = float(np.arcsin(climb_rate / cruise_speed_m_s))
    pitch_ceiling = min(pitch_trim + 0.35, stall - 0.035 + climb_angle)
    rudderless = "rudder" not in roles
    guidance = GuidanceGains(
        airspeed_kp=jnp.asarray(airspeed_kp),
        airspeed_ki=jnp.asarray(airspeed_kp / 5.0),
        throttle_trim=jnp.asarray(float(jnp.mean(trim.control.propeller))),
        throttle_limits=jnp.array([0.0, 1.0]),
        altitude_kp=jnp.asarray(0.8),
        climb_rate_limit=jnp.asarray(climb_rate),
        pitch_trim=jnp.asarray(pitch_trim),
        pitch_limits=jnp.array([pitch_trim - 0.35, pitch_ceiling]),
        heading_kp=jnp.asarray(0.8 if rudderless else 1.0),
        bank_limit=jnp.asarray(0.4 if rudderless else 0.45),
        airspeed_pitch_kp=jnp.asarray(0.05),
    )
    controller = CascadeController(
        channels=channel_map(spec, roles, 1.0),
        rate=RateGains(
            kp=jnp.asarray(kp),
            ki=jnp.asarray(ki),
            kd=jnp.zeros(3),
            integral_limit=jnp.full(3, 0.6),
            feedforward=jnp.asarray(feedforward),
        ),
        attitude=AttitudeGains(kp=jnp.asarray(attitude_kp), rate_limit=jnp.array([3.0, 3.0, 2.0])),
        guidance=guidance,
        rate_period=1,
        attitude_period=2,
        guidance_period=10,
    )
    report = TuningReport(
        trim=trim,
        authority_rad_s2=authority,
        damping_1_s=damping,
        rate_bandwidth_rad_s=bandwidth,
        throttle_acceleration_m_s2=thrust_gain,
    )
    return controller, report


@dataclass(frozen=True)
class StepReport:
    """Closed-loop response to a heading and an altitude step from the trim."""

    finite: bool
    heading_error_deg: float
    altitude_error_m: float
    airspeed_error_m_s: float
    max_bank_deg: float
    settled: bool


def step_response(
    model: AircraftModel,
    controller: CascadeController,
    trim: TrimResult,
    *,
    environment: Environment | None = None,
    duration_s: float = 12.0,
    simulation_dt_s: float = 0.0025,
    heading_step_rad: float = 0.5,
    altitude_step_m: float = 5.0,
) -> StepReport:
    """Fly a heading and an altitude step under the cascade and report where it ends up."""

    environment = standard_environment() if environment is None else environment
    steps = int(duration_s / simulation_dt_s)
    time = jnp.arange(steps) * simulation_dt_s
    condition = trim.condition
    after = time >= 2.0
    setpoints = GuidanceSetpoint(
        airspeed_m_s=jnp.full(steps, condition.airspeed_m_s),
        altitude_m=jnp.where(after, condition.altitude_m + altitude_step_m, condition.altitude_m),
        heading_rad=jnp.where(
            after, condition.heading_rad + heading_step_rad, condition.heading_rad
        ),
    )
    cascade_state = initial_cascade_state(controller, trim.state, trim.control)
    (final, _), (trajectory, _, _) = jax.jit(
        lambda: closed_loop_rollout(
            model, controller, trim.state, cascade_state, setpoints, environment, simulation_dt_s
        )
    )()
    position = np.asarray(trajectory.rigid_body.position)
    velocity = np.asarray(final.rigid_body.velocity)
    attitude = np.asarray(trajectory.rigid_body.attitude)
    x, y, z, w = attitude.T
    bank = np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    heading = np.degrees(np.arctan2(velocity[1], velocity[0]))
    heading_error = float(
        abs(
            (heading - np.degrees(condition.heading_rad + heading_step_rad) + 180.0) % 360.0 - 180.0
        )
    )
    altitude_error = float(abs(-position[-1, 2] - (condition.altitude_m + altitude_step_m)))
    airspeed_error = float(
        abs(np.linalg.norm(velocity - np.asarray(environment.wind)) - condition.airspeed_m_s)
    )
    finite = bool(np.all(np.isfinite(position)))
    settled = finite and heading_error < 5.0 and altitude_error < 1.5 and airspeed_error < 2.0
    return StepReport(
        finite=finite,
        heading_error_deg=heading_error,
        altitude_error_m=altitude_error,
        airspeed_error_m_s=airspeed_error,
        max_bank_deg=float(np.max(np.abs(bank))) if finite else float("nan"),
        settled=settled,
    )


__all__ = ["StepReport", "TuningReport", "step_response", "throttle_acceleration", "tune_cascade"]
