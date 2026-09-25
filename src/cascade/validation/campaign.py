"""Frozen, reproducible comparison campaign with retained per-case discrepancies."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from cascade.analysis.sweep import velocity_from_air_angles
from cascade.analysis.trim import StraightFlightCondition, trim_straight_flight
from cascade.dynamics import evaluate_dynamics
from cascade.initialization import equilibrate_internal_state, standard_environment, zero_state
from cascade.integration import rollout
from cascade.math import quaternion_from_euler, quaternion_rotate
from cascade.provenance import spec_hash, stamp
from cascade.reference import (
    aerobatic_reference_spec,
    skywalker_x8_panels_spec,
    skywalker_x8_spec,
    tailsitter_reference_spec,
)
from cascade.spec import save_aircraft_spec
from cascade.state import AeroState, ControlInput

from .jsbsim import JSBSIM_VERSION, JSBSimAircraft

# Frozen before the campaign. Euler-vs-RK4 trajectories have first-order reference
# error; static loads/accelerations and trim use much tighter independent limits.
PROTOCOL = {
    "schema": "cascade_jsbsim_comparison_v1",
    "jsbsim_version": JSBSIM_VERSION,
    "density_kg_m3": 1.225,
    "altitude_m": 1000.0,
    "duration_s": 2.0,
    "sample_period_s": 0.02,
    "timesteps_s": [0.0025, 0.00125, 0.000625],
    "calibration_models": ["x8", "aerobatic"],
    "limits": {
        "scaled_load_error": 5e-6,
        "acceleration_error_m_s2": 2e-5,
        "angular_acceleration_error_rad_s2": 2e-5,
        "equilibrium_error": 2e-6,
        "trim_acceleration_m_s2": 0.002,
        "trim_angular_acceleration_rad_s2": 0.002,
        "position_rmse_m": 0.03,
        "velocity_rmse_m_s": 0.03,
        "attitude_rmse_rad": 0.003,
        "rate_rmse_rad_s": 0.02,
        "convergence_ratio": 0.8,
    },
    "claims": (
        "Implementation agreement for shared model assumptions; "
        "synthetic, not measured-flight validation."
    ),
}


def fixtures():
    base = aerobatic_reference_spec()
    downwash = np.zeros((len(base.surfaces), len(base.surfaces)))
    downwash[2, :2] = 0.12
    return {
        "x8": (skywalker_x8_spec(), 18.0),
        "x8-panels": (skywalker_x8_panels_spec(), 18.0),
        "aerobatic": (base, 12.0),
        "tailsitter": (tailsitter_reference_spec(), 8.0),
        "aerobatic-downwash": (
            replace(
                base, name="aerobatic-downwash-check", downwash_map=tuple(map(tuple, downwash))
            ),
            12.0,
        ),
    }


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def state_array(state):
    rb = state.rigid_body
    return np.concatenate(
        [np.asarray(x) for x in (rb.position, rb.velocity, rb.attitude, rb.angular_velocity)],
        axis=-1,
    )


def trajectory_errors(actual, expected):
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 13:
        raise ValueError("trajectories must have matching (T,13) shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("nonfinite comparison trajectory")
    attitude = (
        Rotation.from_quat(expected[:, 6:10]).inv() * Rotation.from_quat(actual[:, 6:10])
    ).magnitude()
    return {
        "position_rmse_m": float(
            np.sqrt(np.mean(np.sum((actual[:, :3] - expected[:, :3]) ** 2, axis=1)))
        ),
        "velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.sum((actual[:, 3:6] - expected[:, 3:6]) ** 2, axis=1)))
        ),
        "attitude_rmse_rad": float(np.sqrt(np.mean(attitude**2))),
        "rate_rmse_rad_s": float(
            np.sqrt(np.mean(np.sum((actual[:, 10:] - expected[:, 10:]) ** 2, axis=1)))
        ),
    }


def point_cases(model, speed, *, smoke):
    """Deterministic grid; no fitted parameters or randomized case selection."""
    cases = []
    angles = [-8, 0, 6, 12] if not smoke else [0, 12]
    for speed_scale in [0.7, 1.0, 1.3] if not smoke else [1.0]:
        for alpha in angles:
            for beta in [-8, 0, 8]:
                cases.append((speed * speed_scale, alpha, beta, 0.35, False))
    # Large angles, still air and stale separation verify implementation only.
    cases += [(speed, a, 5, 0.5, True) for a in [-160, -45, 45, 160]]
    cases += [(0, 0, 0, 0.55, False)]
    for n, (v, a, b, t, stale) in enumerate(cases):
        sign = -1 if n % 2 else 1
        control = ControlInput(
            jnp.full((model.n_propellers,), t),
            jnp.linspace(-0.12, 0.1, model.n_control_channels) * sign,
        )
        rotation = quaternion_from_euler(0.17 * sign, 0.11, -0.31 * sign)
        state = zero_state(model, altitude=1000)
        state = state._replace(
            rigid_body=state.rigid_body._replace(
                attitude=rotation,
                velocity=quaternion_rotate(
                    rotation, velocity_from_air_angles(v, np.deg2rad(a), np.deg2rad(b))
                ),
                angular_velocity=jnp.array([0.25, -0.18, 0.3]) * sign,
            )
        )
        state = equilibrate_internal_state(model, state, control, standard_environment())
        if stale:
            state = state._replace(aero=AeroState(jnp.full((model.n_surfaces,), 0.2)))
        yield (
            {
                "speed_m_s": v,
                "alpha_deg": a,
                "beta_deg": b,
                "throttle": t,
                "stale_separation": stale,
            },
            state,
            control,
        )


def compare_points(spec, speed, directory, *, smoke):
    model, env = spec.to_model(), standard_environment()
    reference = JSBSimAircraft(spec, directory / "model")
    evaluate = jax.jit(evaluate_dynamics)
    rows = []
    limits = PROTOCOL["limits"]
    for index, (case, state, control) in enumerate(point_cases(model, speed, smoke=smoke)):
        reference.initialize(state)
        expected = reference.loads()
        result = evaluate(model, state, control, env)
        actual = {
            "force": result.force_body,
            "moment": result.moment_body,
            "aero-force": result.aerodynamics.force_body,
            "aero-moment": result.aerodynamics.moment_body,
            "propulsion-force": result.propulsion.force_body,
            "propulsion-moment": result.propulsion.moment_body,
        }
        scaled = max(
            float(np.max(abs(np.asarray(value) - expected[key]) / (1 + abs(expected[key]))))
            for key, value in actual.items()
        )
        acceleration, angular = reference.accelerations()
        errors = {
            "scaled_load_error": scaled,
            "acceleration_error_m_s2": float(
                np.max(abs(acceleration - result.derivative.rigid_body.velocity))
            ),
            "angular_acceleration_error_rad_s2": float(
                np.max(abs(angular - result.derivative.rigid_body.angular_velocity))
            ),
        }
        # Independent equilibrium initialization tests the promised reset contract.
        if not case["stale_separation"]:
            reference.initialize(state, equilibrate=True, control=control)
            errors["equilibrium_error"] = float(
                np.max(abs(reference.separation - np.asarray(state.aero.separation)))
            )
        rows.append(
            {
                "index": index,
                "case": case,
                "input_state_ned_frd_xyzw": state_array(state).tolist(),
                "input_surface_rad": np.asarray(state.actuators.surface_deflection).tolist(),
                "input_motor_rad_s": np.asarray(state.actuators.propeller_speed).tolist(),
                "input_separation": np.asarray(state.aero.separation).tolist(),
                "errors": errors,
                "passed": all(v <= limits[k] for k, v in errors.items()),
                "cascade": {k: np.asarray(v).tolist() for k, v in actual.items()},
                "jsbsim": {k: v.tolist() for k, v in expected.items()},
            }
        )
    json_write(directory / "points.json", rows)
    return {
        "cases": len(rows),
        "passed": all(x["passed"] for x in rows),
        "maximum_errors": {k: max(x["errors"].get(k, 0) for x in rows) for k in rows[0]["errors"]},
    }


def maneuver_commands(model, trim_control, maneuver, *, dt, duration):
    time = np.arange(round(duration / dt)) * dt
    propeller = np.broadcast_to(trim_control.propeller, (len(time), model.n_propellers)).copy()
    channel = np.broadcast_to(trim_control.channel, (len(time), model.n_control_channels)).copy()
    doublet = ((time >= 0.4) & (time < 0.8)).astype(float) - ((time >= 0.8) & (time < 1.2)).astype(
        float
    )
    if maneuver == "throttle-step":
        propeller += 0.04 * (time >= 0.4)[:, None]
    elif maneuver != "trim":
        channel[:, 0 if maneuver == "aileron-doublet" else 1] += 0.025 * doublet
    return ControlInput(jnp.asarray(propeller), jnp.asarray(channel))


def jsbsim_trajectory(spec, state, commands, dt, directory):
    reference = JSBSimAircraft(spec, directory, dt=dt)
    reference.initialize(state)
    propeller, channel = np.asarray(commands.propeller), np.asarray(commands.channel)
    samples = [reference.snapshot()]
    stride = round(PROTOCOL["sample_period_s"] / dt)
    for i in range(len(propeller)):
        value = reference.step(ControlInput(propeller[i], channel[i]))
        if (i + 1) % stride == 0:
            samples.append(value)
    return np.array(samples)


def compare_flights(spec, speed, directory, *, smoke):
    model = spec.to_model()
    trim = trim_straight_flight(model, StraightFlightCondition(speed, altitude_m=1000))
    if not trim.success:
        return {
            "passed": False,
            "trim": {"cascade_success": False, "message": trim.message},
            "flights": [],
        }
    reference = JSBSimAircraft(spec, directory / "trim-model")
    reference.initialize(trim.state, equilibrate=True, control=trim.control)
    acceleration, angular = reference.accelerations()
    trim_report = {
        "cascade_success": True,
        "jsbsim_acceleration_m_s2": float(np.linalg.norm(acceleration)),
        "jsbsim_angular_acceleration_rad_s2": float(np.linalg.norm(angular)),
        "decision": np.asarray(trim.decision).tolist(),
    }
    limits = PROTOCOL["limits"]
    trim_report["passed"] = bool(
        trim_report["jsbsim_acceleration_m_s2"] <= limits["trim_acceleration_m_s2"]
        and trim_report["jsbsim_angular_acceleration_rad_s2"]
        <= limits["trim_angular_acceleration_rad_s2"]
    )
    rows = []
    duration = 0.4 if smoke else PROTOCOL["duration_s"]
    maneuvers = (
        ["trim"] if smoke else ["trim", "elevator-doublet", "aileron-doublet", "throttle-step"]
    )
    compiled = jax.jit(rollout)
    for maneuver in maneuvers:
        folder = directory / maneuver
        folder.mkdir()
        dt = PROTOCOL["timesteps_s"][-1]
        commands = maneuver_commands(model, trim.control, maneuver, dt=dt, duration=duration)
        _, history = compiled(model, trim.state, commands, standard_environment(), dt)
        stride = round(PROTOCOL["sample_period_s"] / dt)
        baseline = np.concatenate(
            (state_array(trim.state)[None], state_array(history)[stride - 1 :: stride])
        )
        trajectories = []
        for step in PROTOCOL["timesteps_s"]:
            jsb = jsbsim_trajectory(
                spec,
                trim.state,
                maneuver_commands(model, trim.control, maneuver, dt=step, duration=duration),
                step,
                folder / f"dt-{step}",
            )
            errors = trajectory_errors(jsb[1:], baseline[1:])
            trajectories.append((step, jsb, errors))
        # Require error to decrease at the expected first-order rate, unless already
        # below 1% of the absolute acceptance limits (roundoff/flat-Earth floor).
        convergence = {
            k: bool(
                trajectories[-1][2][k]
                <= max(0.01 * limits[k], limits["convergence_ratio"] * trajectories[-2][2][k])
            )
            for k in trajectories[-1][2]
        }
        errors = trajectories[-1][2]
        passed = all(errors[k] <= limits[k] for k in errors) and all(convergence.values())
        np.savez_compressed(
            folder / "trajectories.npz",
            time_s=np.arange(len(baseline)) * PROTOCOL["sample_period_s"],
            cascade=baseline,
            **{f"jsbsim_{i}": x[1] for i, x in enumerate(trajectories)},
        )
        row = {
            "maneuver": maneuver,
            "passed": bool(passed),
            "convergence": convergence,
            "runs": [{"dt_s": step, **err} for step, _, err in trajectories],
        }
        json_write(folder / "errors.json", row)
        rows.append(row)
    return {
        "passed": trim_report["passed"] and all(x["passed"] for x in rows),
        "trim": trim_report,
        "flights": rows,
    }


def run_campaign(output, *, models=None, smoke=False):
    if not jax.config.x64_enabled:
        raise ValueError("run the comparison with JAX_ENABLE_X64=1")
    selected = fixtures()
    models = list(selected) if models is None else list(models)
    if not models or any(x not in selected for x in models) or len(set(models)) != len(models):
        raise ValueError("models must be unique known fixture names")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("comparison output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    json_write(
        output / "protocol.json",
        {**PROTOCOL, "profile": "smoke" if smoke else "full", "models": models},
    )
    result = {
        "schema": PROTOCOL["schema"],
        "profile": "smoke" if smoke else "full",
        "models": {},
        "runtime": stamp(),
        "python_platform": platform.platform(),
    }
    for name in models:
        print(f"Comparing {name}: forces, equilibrium, trim and trajectories", flush=True)
        spec, speed = selected[name]
        directory = output / name
        directory.mkdir()
        save_aircraft_spec(spec, directory / "aircraft.toml")
        points = compare_points(spec, speed, directory, smoke=smoke)
        flights = compare_flights(spec, speed, directory, smoke=smoke)
        result["models"][name] = {
            "spec_sha256": spec_hash(spec),
            "points": points,
            **flights,
            "passed": points["passed"] and flights["passed"],
        }
        if not smoke and name in PROTOCOL["calibration_models"]:
            from .calibration import run_calibration

            print(f"Calibrating {name} against JSBSim-generated recordings", flush=True)
            calibration = run_calibration(spec, speed, directory / "calibration")
            result["models"][name]["calibration"] = calibration
            result["models"][name]["passed"] &= calibration["passed"]
        json_write(output / "results.json", result)
    if "x8" in models and "x8-panels" in models:
        from .approximation import compare_x8_rates

        directory = output / "x8-approximation"
        directory.mkdir()
        result["x8_approximation"] = compare_x8_rates(directory)
    result["passed"] = all(x["passed"] for x in result["models"].values())
    result["file_sha256"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "results.json"
    }
    result["implementation_sha256"] = {
        str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.parent.rglob("*.py"))
    }
    json_write(output / "results.json", result)
    lines = [
        "# JSBSim comparison",
        "",
        PROTOCOL["claims"],
        "",
        f"Profile: {result['profile']}. Overall pass: {result['passed']}.",
        "",
        "| Model | Load/equilibrium checks | Trim and flights | Calibration | Overall |",
        "|---|---|---|---|---|",
    ]
    for name, item in result["models"].items():
        lines.append(
            f"| {name} | {item['points']['cases']} cases; {item['points']['passed']} "
            f"| trim {item['trim'].get('passed', False)}; {len(item['flights'])} flights "
            f"| {item.get('calibration', {}).get('passed', 'not run')} | {item['passed']} |"
        )
    lines.extend(
        [
            "",
            "See `results.json` for all errors and hashes, each model's `points.json` for loads, "
            "and maneuver `trajectories.npz` files for all three JSBSim timesteps. "
            "No failed cases are discarded.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines))
    return result
