"""Express the NTNU Skywalker X8 propulsion law as a Cascade thrust map, exactly.

The published X8 propulsion (pyfly ``x8_param.mat``; Reinhardt et al. 2022) is
``T = 1/2 rho S_prop C_prop V_d (V_d - V_a)`` with ``V_d = V_a + dt (k_motor - V_a)``, where
``dt`` is normalized throttle. Cascade's propeller is the polynomial map
``T / rho = D^4 sum_ij c_ij n^(i+1) (V_a / D)^j`` with ``n = dt n_max`` in rev/s. Expanding the
published law in ``dt`` and ``V_a`` gives every ``c_ij`` in closed form, so the map reproduces
it to rounding. ``n_max`` is a modelling choice (the published law has no shaft speed); it only
scales the propeller-speed state. Run ``uv run python scripts/fit_x8_propeller.py``; the values
are recorded in ``src/cascade/aircraft/skywalker_x8.toml``.
"""

from __future__ import annotations

import numpy as np

RHO = 1.225
S_PROP = 0.10179  # m^2
C_PROP = 0.248
K_MOTOR = 37.42  # m/s
N_MAX = 150.0  # rev/s at full throttle
DIAMETER = float(np.sqrt(4.0 * S_PROP / np.pi))


def published_thrust(throttle: np.ndarray, airspeed: np.ndarray) -> np.ndarray:
    exit_speed = airspeed + throttle * (K_MOTOR - airspeed)
    return 0.5 * RHO * S_PROP * C_PROP * exit_speed * (exit_speed - airspeed)


def thrust_map() -> np.ndarray:
    """Rows multiply n, n^2; columns multiply (V_a / D)^0, ^1, ^2."""

    half = 0.5 * S_PROP * C_PROP
    d = DIAMETER
    return np.array(
        [
            [0.0, half * K_MOTOR / (N_MAX * d**3), -half / (N_MAX * d**2)],
            [
                half * K_MOTOR**2 / (N_MAX**2 * d**4),
                -2.0 * half * K_MOTOR / (N_MAX**2 * d**3),
                half / (N_MAX**2 * d**2),
            ],
        ]
    )


def map_thrust(
    coefficients: np.ndarray, throttle: np.ndarray, airspeed: np.ndarray
) -> np.ndarray:
    revolutions = throttle * N_MAX
    inflow = airspeed / DIAMETER
    speed_powers = np.stack((revolutions, revolutions**2), axis=-1)
    inflow_powers = np.stack((np.ones_like(inflow), inflow, inflow**2), axis=-1)
    thrust_per_density = np.einsum("ij,...i,...j->...", coefficients, speed_powers, inflow_powers)
    return RHO * DIAMETER**4 * thrust_per_density


if __name__ == "__main__":
    coefficients = thrust_map()
    throttle, airspeed = np.meshgrid(
        np.linspace(0.0, 1.0, 41), np.linspace(-5.0, 30.0, 71), indexing="ij"
    )
    mismatch = map_thrust(coefficients, throttle, airspeed) - published_thrust(throttle, airspeed)
    error = np.max(np.abs(mismatch))
    print(f"diameter_m = {DIAMETER:.5f}")
    print(f"speed_max_rad_s = {2.0 * np.pi * N_MAX:.3f}")
    print("thrust_map = [")
    for row in coefficients:
        print("    [" + ", ".join(f"{value:.6e}" for value in row) + "],")
    print("]")
    print(f"max |map - published| over throttle 0..1, V_a -5..30 m/s: {error:.2e} N")
    for dt, va in ((0.45, 18.0), (0.5, 18.0), (1.0, 0.0), (1.0, 18.0)):
        published = published_thrust(np.asarray(dt), np.asarray(va))
        mapped = map_thrust(coefficients, np.asarray(dt), np.asarray(va))
        print(
            f"  throttle {dt:.2f}, V_a {va:4.1f} m/s: "
            f"published {published:6.2f} N, map {mapped:6.2f} N"
        )
