"""Check steady-turn relative equilibria and track a speed change with scheduled gains."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.analysis import SteadyTurnCondition, trim_steady_turn
from cascade.control import (
    GuidanceSetpoint,
    controller_at_speed,
    initial_cascade_state,
    scheduled_closed_loop_rollout,
    tune_gain_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    spec = cascade.aerobatic_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    turns = []
    for rate in (-0.2, 0.2):
        trim = trim_steady_turn(model, SteadyTurnCondition(12.0, rate, altitude_m=50.0))
        assert trim.success, trim.message
        final, _ = jax.jit(cascade.rollout)(
            model, trim.state, cascade.repeat_control(trim.control, 400), environment, 0.005
        )
        expected = np.array(
            [
                12 / rate * np.sin(rate * 2),
                12 / rate * (1 - np.cos(rate * 2)),
                -50,
            ]
        )
        error = float(np.linalg.norm(np.asarray(final.rigid_body.position) - expected))
        assert error < 0.01, error
        turns.append(
            {
                "turn_rate_rad_s": rate,
                "bank_rad": float(trim.decision[0]),
                "scaled_balance_norm": float(jnp.linalg.norm(trim.scaled_residual)),
                "two_second_position_error_m": error,
            }
        )
        print(f"turn {rate:+.2f} rad/s: circle position error {error:.6f} m")

    schedule, report = tune_gain_schedule(spec, [12.0, 16.0])
    trim = report.tuning[0].trim
    memory = initial_cascade_state(controller_at_speed(schedule, 12.0), trim.state, trim.control)
    dt, count = 0.0025, 4800
    setpoints = GuidanceSetpoint(
        jnp.where(jnp.arange(count) * dt < 2, 12.0, 15.0),
        jnp.full(count, 50.0),
        jnp.zeros(count),
    )
    (final, controller_state), (trajectory, _, _) = jax.jit(scheduled_closed_loop_rollout)(
        model, schedule, trim.state, memory, setpoints, environment, dt
    )
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(trajectory))
    speed_error = abs(float(jnp.linalg.norm(final.rigid_body.velocity)) - 15.0)
    altitude_error = abs(float(final.rigid_body.position[2]) + 50.0)
    assert speed_error < 1.0 and altitude_error < 1.5
    assert int(controller_state.step_index) == count
    print(
        f"scheduled tracking: speed error {speed_error:.4f} m/s, "
        f"altitude error {altitude_error:.4f} m"
    )
    if arguments.output:
        record = {
            "provenance": cascade.stamp(spec, model),
            "turn_configuration": {"speed_m_s": 12.0, "dt_s": 0.005, "steps": 400},
            "turns": turns,
            "schedule_configuration": {
                "knots_m_s": [12.0, 16.0],
                "target_m_s": 15.0,
                "step_at_s": 2.0,
                "dt_s": dt,
                "steps": count,
                "all_knots_settled": True,
            },
            "scheduled_tracking": {
                "speed_error_m_s": speed_error,
                "altitude_error_m": altitude_error,
            },
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
