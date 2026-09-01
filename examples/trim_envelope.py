"""Trace conventional and post-stall trim branches, then linearize one point."""

import jax.numpy as jnp
import numpy as np

import cascade


def print_branch(name: str, results: tuple[cascade.TrimResult, ...]) -> None:
    print(f"\n{name}")
    print("speed  alpha  throttle  elevator  balance")
    for result in results:
        print(
            f"{result.condition.airspeed_m_s:5.1f} "
            f"{np.rad2deg(result.angle_of_attack_rad):6.1f} "
            f"{float(result.control.propeller[0]):9.3f} "
            f"{float(result.control.channel[1]):9.3f} "
            f"{float(jnp.linalg.norm(result.scaled_residual)):8.1e}"
        )


def main() -> None:
    model = cascade.aerobatic_reference()
    environment = cascade.standard_environment()

    conventional = cascade.continue_trims(
        model,
        (cascade.StraightFlightCondition(speed) for speed in (16.0, 14.0, 12.0, 10.0)),
        environment,
    )

    # Decision order is roll, pitch, yaw offset, propeller commands, then aileron/elevator/rudder.
    # A high-alpha seed intentionally selects the post-stall branch of this non-convex problem.
    post_stall_seed = jnp.array([0.0, np.deg2rad(75.0), 0.0, 0.7, 0.0, -0.85, 0.0])
    post_stall = cascade.continue_trims(
        model,
        (cascade.StraightFlightCondition(speed) for speed in (4.0, 5.0, 6.0, 7.0, 8.0)),
        environment,
        initial_decision=post_stall_seed,
        residual_tolerance=1e-3,
    )

    print_branch("Conventional branch", conventional)
    print_branch("Post-stall branch", post_stall)

    linearization = cascade.linearize_step(
        model, conventional[2].state, conventional[2].control, environment, 0.01
    )
    dominant = cascade.stability_modes(linearization)[:5]
    print("\nDominant local modes at 12 m/s (continuous eigenvalues)")
    for mode in dominant:
        print(f"{mode.continuous_eigenvalue.real:+8.3f} {mode.continuous_eigenvalue.imag:+8.3f}j")

    sweep = cascade.aerodynamic_sweep(
        model, jnp.deg2rad(jnp.linspace(-180.0, 180.0, 145)), airspeed_m_s=12.0
    )
    assert jnp.all(jnp.isfinite(sweep.force_coefficient_body))
    print("\nFull-envelope coefficient sweep: 145/145 finite points")


if __name__ == "__main__":
    main()
