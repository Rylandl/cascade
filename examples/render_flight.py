"""Render flights to MP4 with MuJoCo: a tailsitter round trip and a conventional archetype
flying heading and altitude steps under its auto-tuned cascade.

Usage: ``uv run python examples/render_flight.py [output_dir [width height]]``. Needs the
``viz`` extra (MuJoCo) and ffmpeg on the path. Panels are coloured by their separation state, so the
tailsitter's wings turn red through the stalled transition and grey again in cruise.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

import cascade
from cascade.archetypes import ConventionalDesign, cruise_speed, design_spec
from cascade.autotune import tune_cascade
from cascade.control import GuidanceSetpoint, closed_loop_rollout, initial_cascade_state
from cascade.math import quaternion_from_euler
from cascade.render import render_trajectory
from cascade.vtol import (
    initial_transition_state,
    speed_profile_schedule,
    tailsitter_reference_controller,
    transition_rollout,
    trapezoid_speed_profile,
)


def tailsitter_round_trip(dt: float = 0.005, steps: int = 3200):
    spec = cascade.tailsitter_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    controller = tailsitter_reference_controller(spec)
    state = cascade.zero_state(model, altitude=1.5)
    state = state._replace(
        rigid_body=state.rigid_body._replace(attitude=quaternion_from_euler(0.0, jnp.pi / 2.0, 0.0))
    )
    control = cascade.ControlInput(propeller=jnp.array([0.78, 0.78]), channel=jnp.zeros(2))
    state = cascade.equilibrate_internal_state(model, state, control, environment)
    speed = trapezoid_speed_profile(
        steps,
        dt,
        hold_steps=400,
        cruise_speed_m_s=jnp.asarray(8.0),
        acceleration_m_s2=jnp.asarray(3.5),
        cruise_steps=600,
        deceleration_m_s2=jnp.asarray(2.0),
    )
    hover = speed_profile_schedule(
        speed,
        dt,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.asarray(8.0),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.maximum(speed, 6.5),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=jnp.zeros(steps),
    )
    (_, _), (trajectory, _, _) = jax.jit(
        lambda: transition_rollout(
            model,
            controller,
            state,
            initial_transition_state(state),
            hover,
            forward,
            environment,
            dt,
        )
    )()
    return spec, trajectory, dt


def conventional_steps(dt: float = 0.0025, duration_s: float = 12.0):
    design = ConventionalDesign()
    spec = design_spec(design)
    model = spec.to_model()
    environment = cascade.standard_environment()
    controller, report = tune_cascade(spec, cruise_speed(design), model=model)
    steps = int(duration_s / dt)
    time = jnp.arange(steps) * dt
    condition = report.trim.condition
    setpoints = GuidanceSetpoint(
        airspeed_m_s=jnp.full(steps, condition.airspeed_m_s),
        altitude_m=jnp.where(time >= 2.0, condition.altitude_m + 5.0, condition.altitude_m),
        heading_rad=jnp.where(time >= 2.0, 0.5, 0.0),
    )
    cascade_state = initial_cascade_state(controller, report.trim.state, report.trim.control)
    (_, _), (trajectory, _, _) = jax.jit(
        lambda: closed_loop_rollout(
            model, controller, report.trim.state, cascade_state, setpoints, environment, dt
        )
    )()
    return spec, trajectory, dt


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("renders")
    size = {"width": int(sys.argv[2]), "height": int(sys.argv[3])} if len(sys.argv) > 3 else {}
    spec, trajectory, dt = tailsitter_round_trip()
    for camera in ("follow", "ground"):
        path = render_trajectory(
            spec,
            trajectory,
            dt,
            output / f"tailsitter_round_trip_{camera}.mp4",
            camera=camera,
            **size,
        )
        print("wrote", path)
    spec, trajectory, dt = conventional_steps()
    path = render_trajectory(
        spec, trajectory, dt, output / "conventional_steps_chase.mp4", camera="chase", **size
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
