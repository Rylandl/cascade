"""Sample airframes from the archetypes, check they fly, and tune a cascade for each.

Every row is a different aircraft a learner could be handed with no other information: the
design parameters are drawn at random, the spec is built from textbook relations, the design
is trimmed and checked, and the control cascade is tuned from that trim and linearisation
alone. The spread of cruise speed, short-period frequency, and tuned gains is the point.
"""

import jax
import numpy as np

from cascade.archetypes import (
    ConventionalDesign,
    FlyingWingDesign,
    design_spec,
    sample_designs,
    validate_design,
)
from cascade.autotune import step_response, tune_cascade


def main() -> None:
    key = jax.random.PRNGKey(0)
    print(
        "archetype     layout        span   mass  cruise  alpha  thr | SP Hz zeta | "
        "rate kp roll/pitch/yaw   | step: hdg  alt  settled"
    )
    for archetype, count in ((FlyingWingDesign, 6), (ConventionalDesign, 6)):
        for design in sample_designs(archetype, key, count):
            report = validate_design(design)
            layout = getattr(design, "motors", None) or getattr(design, "tail", "")
            spec = design_spec(design)
            head = (
                f"{archetype.__name__[:-6]:13s} {layout:12s} {design.span_m:5.2f} "
                f"{spec.mass_kg:6.2f} {report.cruise_speed_m_s:6.1f} "
                f"{np.degrees(report.angle_of_attack_rad):6.1f} {report.throttle:4.2f} | "
                f"{report.short_period_hz:5.2f} {report.short_period_damping:4.2f} | "
            )
            if not report.valid:
                print(head + f"rejected: {'; '.join(report.reasons)}")
                continue
            controller, tuning = tune_cascade(spec, report.cruise_speed_m_s)
            response = step_response(spec.to_model(), controller, tuning.trim)
            kp = np.asarray(controller.rate.kp)
            print(
                head + f"{kp[0]:5.3f} {kp[1]:5.3f} {kp[2]:5.3f}          | "
                f"{response.heading_error_deg:5.1f} {response.altitude_error_m:5.2f} "
                f"{response.settled}"
            )
        key, _ = jax.random.split(key)


if __name__ == "__main__":
    main()
