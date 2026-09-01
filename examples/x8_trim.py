"""Trim the published Skywalker X8 model at cruise for the documented parameter variants."""

from dataclasses import replace

import numpy as np

import cascade
from cascade.spec import LongitudinalCoefficientSpec

FLIGHT_TUNED_PITCH = LongitudinalCoefficientSpec(0.02275, -0.4629, -1.3, -0.2292)


def main() -> None:
    base = cascade.skywalker_x8_spec()
    variants = {
        "wind-tunnel pitch, 3.364 kg": base,
        "flight-tuned pitch, 3.364 kg": replace(
            base, body=replace(base.body, pitch=FLIGHT_TUNED_PITCH)
        ),
        "wind-tunnel pitch, 4.0 kg": replace(base, mass_kg=4.0),
        "flight-tuned pitch, 4.0 kg": replace(
            base, mass_kg=4.0, body=replace(base.body, pitch=FLIGHT_TUNED_PITCH)
        ),
    }
    print("variant                          alpha  elevator  throttle  balance")
    for name, spec in variants.items():
        result = cascade.trim_straight_flight(
            spec.to_model(), cascade.StraightFlightCondition(airspeed_m_s=18.0, altitude_m=100.0)
        )
        status = "ok " if result.success else "FAIL"
        print(
            f"{name:32s} {np.rad2deg(result.angle_of_attack_rad):5.1f}  "
            f"{np.rad2deg(float(result.control.channel[1])):+7.2f}  "
            f"{float(result.control.propeller[0]):8.3f}  "
            f"{float(np.linalg.norm(result.scaled_residual)):8.1e} {status}"
        )


if __name__ == "__main__":
    main()
