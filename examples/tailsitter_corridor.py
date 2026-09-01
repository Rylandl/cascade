"""Trace the tailsitter's two steady-flight branches: conventional and thrust-borne."""

import jax.numpy as jnp
import numpy as np

import cascade
from cascade.reference import tailsitter_reference


def print_branch(name: str, results: tuple[cascade.TrimResult, ...]) -> None:
    print(f"\n{name}")
    print("speed  alpha  pitch  throttle  elevator  balance")
    for result in results:
        pitch = np.rad2deg(float(result.decision[1]))
        print(
            f"{result.condition.airspeed_m_s:5.1f} "
            f"{np.rad2deg(result.angle_of_attack_rad):6.1f} {pitch:6.1f} "
            f"{float(result.control.propeller[0]):9.3f} "
            f"{float(result.control.channel[1]):9.3f} "
            f"{float(jnp.linalg.norm(result.scaled_residual)):8.1e}"
            f"{'' if result.success else '  (no trim)'}"
        )


def main() -> None:
    model = tailsitter_reference()
    conventional = cascade.continue_trims(
        model,
        (cascade.StraightFlightCondition(v, altitude_m=1.5) for v in (9.0, 8.0, 7.0, 6.5, 6.0)),
        residual_tolerance=1e-3,
    )
    # Decision order: roll, pitch, yaw offset, two motors, aileron, elevator. A steep, high-throttle
    # seed selects the thrust-borne branch, which continues from near hover up through cruise.
    seed = jnp.array([0.0, np.deg2rad(70.0), 0.0, 0.77, 0.77, 0.0, 0.0])
    thrust_borne = cascade.continue_trims(
        model,
        (
            cascade.StraightFlightCondition(v, altitude_m=1.5)
            for v in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        ),
        initial_decision=seed,
        residual_tolerance=1e-3,
    )
    print_branch("Conventional branch (exists above the stall speed)", conventional)
    print_branch("Thrust-borne branch (near hover to cruise)", thrust_borne)
    print(
        "\nBoth branches coexist above about 6.5 m/s; below it only the thrust-borne branch "
        "remains, and around 3-4.5 m/s it needs nearly full nose-up elevon."
    )


if __name__ == "__main__":
    main()
