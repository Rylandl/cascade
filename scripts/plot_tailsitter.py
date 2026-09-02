"""Figures for docs/tailsitter.md: the trim corridor, the round trip, and a gusty round trip.

Run with ``uv run --with matplotlib python scripts/plot_tailsitter.py``; matplotlib is not a
project dependency. Writes SVGs into docs/figures/.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import cascade  # noqa: E402
from cascade.analysis import StraightFlightCondition, continue_trims  # noqa: E402
from cascade.control import GuidanceSetpoint  # noqa: E402
from cascade.gusts import dryden_environment_sequence, dryden_low_altitude  # noqa: E402
from cascade.math import quaternion_from_euler, quaternion_rotate  # noqa: E402
from cascade.vtol import (  # noqa: E402
    initial_transition_state,
    speed_profile_schedule,
    tailsitter_reference_controller,
    transition_rollout,
    trapezoid_speed_profile,
)

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
DT = 0.005
STEPS = 3200


def corridor_figure(model):
    thrust_borne_speeds = (0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0)
    conventional_speeds = (9.0, 8.0, 7.0)
    seed = jnp.array([0.0, np.deg2rad(70.0), 0.0, 0.77, 0.77, 0.0, 0.0])
    thrust_borne = continue_trims(
        model,
        (StraightFlightCondition(v, altitude_m=1.5) for v in thrust_borne_speeds),
        initial_decision=seed,
        residual_tolerance=1e-3,
    )
    conventional = continue_trims(
        model,
        (StraightFlightCondition(v, altitude_m=1.5) for v in conventional_speeds),
        residual_tolerance=1e-3,
    )

    def unpack(results, speeds):
        rows = [(v, r) for v, r in zip(speeds, results, strict=False) if r.success]
        return (
            np.array([v for v, _ in rows]),
            np.array([np.degrees(float(r.decision[1])) for _, r in rows]),
            np.array([float(r.control.propeller[0]) for _, r in rows]),
            np.array([float(r.control.channel[1]) for _, r in rows]),
        )

    tb = unpack(thrust_borne, thrust_borne_speeds)
    cv = unpack(conventional, conventional_speeds)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    for ax, index, label in zip(
        axes, (1, 2, 3), ("pitch (deg)", "throttle", "elevator"), strict=False
    ):
        ax.plot(tb[0], tb[index], "o-", label="thrust-borne branch")
        ax.plot(cv[0], cv[index], "s-", label="conventional branch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("airspeed (m/s)")
    fig.suptitle("Tailsitter trim corridor: two straight-flight branches")
    fig.tight_layout()
    fig.savefig(FIGURES / "tailsitter_corridor.svg")
    plt.close(fig)


def hover_start(model, environment):
    state = cascade.zero_state(model, altitude=1.5)
    state = state._replace(
        rigid_body=state.rigid_body._replace(attitude=quaternion_from_euler(0.0, jnp.pi / 2.0, 0.0))
    )
    control = cascade.ControlInput(propeller=jnp.array([0.78, 0.78]), channel=jnp.zeros(2))
    return cascade.equilibrate_internal_state(model, state, control, environment)


def round_trip(model, controller, state, environment, environments=None):
    speed = trapezoid_speed_profile(
        STEPS,
        DT,
        hold_steps=400,
        cruise_speed_m_s=jnp.asarray(8.0),
        acceleration_m_s2=jnp.asarray(3.5),
        cruise_steps=600,
        deceleration_m_s2=jnp.asarray(2.0),
    )
    hover = speed_profile_schedule(
        speed,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.asarray(8.0),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.maximum(speed, 6.5),
        altitude_m=jnp.full(STEPS, 1.5),
        heading_rad=jnp.zeros(STEPS),
    )
    (_, _), (trajectory, controls, weight) = jax.jit(
        lambda envs: transition_rollout(
            model,
            controller,
            state,
            initial_transition_state(state),
            hover,
            forward,
            environment,
            DT,
            environments=envs,
        )
    )(environments)
    position = np.asarray(trajectory.rigid_body.position)
    airspeed = np.linalg.norm(np.asarray(trajectory.rigid_body.velocity), axis=1)
    x_body = np.asarray(
        jax.vmap(lambda q: quaternion_rotate(q, jnp.array([1.0, 0.0, 0.0])))(
            trajectory.rigid_body.attitude
        )
    )
    tilt = np.degrees(np.arccos(np.clip(-x_body[:, 2], -1.0, 1.0)))
    return {
        "time": np.arange(STEPS) * DT,
        "command": np.asarray(speed),
        "speed": airspeed,
        "tilt": tilt,
        "altitude": -position[:, 2],
        "weight": np.asarray(weight),
        "throttle": np.asarray(controls.propeller[:, 0]),
        "elevator": np.asarray(controls.channel[:, 1]),
    }


def round_trip_figure(calm, gusty):
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.4), sharex=True)
    axes[0].plot(calm["time"], calm["command"], "k--", label="commanded")
    axes[0].plot(calm["time"], calm["speed"], label="calm")
    axes[0].plot(gusty["time"], gusty["speed"], alpha=0.7, label="gusts, 4 m/s reference wind")
    axes[0].set_ylabel("airspeed (m/s)")
    axes[0].legend(loc="upper right")
    axes[1].plot(calm["time"], calm["tilt"])
    axes[1].plot(gusty["time"], gusty["tilt"], alpha=0.7)
    axes[1].set_ylabel("tilt from vertical (deg)")
    axes[2].plot(calm["time"], calm["altitude"])
    axes[2].plot(gusty["time"], gusty["altitude"], alpha=0.7)
    axes[2].axhline(1.5, color="k", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("altitude (m)")
    axes[3].plot(calm["time"], calm["throttle"], label="throttle")
    axes[3].plot(calm["time"], calm["elevator"], label="elevator")
    axes[3].plot(calm["time"], calm["weight"], label="forward blend weight")
    axes[3].set_ylabel("controls")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Hover, transition, cruise at 8 m/s, back-transition, hover")
    fig.tight_layout()
    fig.savefig(FIGURES / "tailsitter_round_trip.svg")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    spec = cascade.tailsitter_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    controller = tailsitter_reference_controller(spec)
    state = hover_start(model, environment)
    corridor_figure(model)
    calm = round_trip(model, controller, state, environment)
    gusts = dryden_environment_sequence(
        jax.random.PRNGKey(1),
        environment,
        STEPS,
        DT,
        airspeed_m_s=jnp.asarray(4.0),
        parameters=dryden_low_altitude(jnp.asarray(1.5), jnp.asarray(4.0)),
    )
    gusty = round_trip(model, controller, state, environment, gusts)
    round_trip_figure(calm, gusty)
    print("wrote", sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
