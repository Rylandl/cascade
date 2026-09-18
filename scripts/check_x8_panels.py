"""Read-only comparison of the packaged X8 panel and coefficient models.

Run ``python scripts/check_x8_panels.py --output dist/verification/x8-panels.json``.
Unlike ``fit_x8_panels.py``, this checks the stored specifications without fitting or rewriting
them. It compares models, not flight data, and records hashes and the numerical configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from fit_x8_panels import COEFFICIENT_NAMES, PUBLISHED_RATES, fit_grid, rate_derivatives

import cascade


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    table_spec, panel_spec = cascade.skywalker_x8_spec(), cascade.skywalker_x8_panels_spec()
    table, panels = table_spec.to_model(), panel_spec.to_model()
    alpha, beta, control = fit_grid()

    def coefficients(model):
        sweep = cascade.aerodynamic_sweep(
            model, alpha, sideslip_rad=beta, airspeed_m_s=18.0, control=control
        )
        return jnp.concatenate(
            (sweep.force_coefficient_body, sweep.moment_coefficient_body), axis=-1
        )

    residual = np.asarray(coefficients(panels) - coefficients(table))
    if not np.isfinite(residual).all():
        raise ValueError("X8 comparison produced non-finite coefficients")
    rms = np.sqrt(np.mean(residual**2, axis=0))
    rates = {
        "coefficient_model": rate_derivatives(table, math.radians(3.0), 18.0, 0.2),
        "panel_model": rate_derivatives(panels, math.radians(3.0), 18.0, 0.2),
        "published_xflr5": PUBLISHED_RATES,
    }
    record = {
        "comparison": "packaged models; no fitting and no flight observations",
        "grid": {
            "alpha_deg": np.linspace(-8.0, 10.0, 7).tolist(),
            "beta_deg": np.linspace(-8.0, 8.0, 5).tolist(),
            "aileron_rad": [-0.2, 0.0, 0.2],
            "elevator_rad": [-0.2, 0.0, 0.2],
            "airspeed_m_s": 18.0,
            "points": int(residual.shape[0]),
        },
        "residual_rms": dict(zip(COEFFICIENT_NAMES, rms.tolist(), strict=True)),
        "rate_configuration": {
            "alpha_deg": 3.0,
            "airspeed_m_s": 18.0,
            "one_axis_at_a_time_rad_s": 0.2,
        },
        "rate_derivatives": rates,
        "coefficient_provenance": cascade.stamp(table_spec, table),
        "panel_provenance": cascade.stamp(panel_spec, panels),
    }
    output = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
