"""Characterize the X8 panel approximation separately from implementation agreement."""

import jax.numpy as jnp
import numpy as np

from cascade.initialization import zero_control, zero_state
from cascade.reference import skywalker_x8_panels_spec, skywalker_x8_spec

from .campaign import json_write
from .jsbsim import JSBSimAircraft


def compare_x8_rates(directory):
    """Independent JSBSim central differences; descriptive, never an acceptance gate."""
    speed, alpha, perturbation, rho = 18.0, np.deg2rad(3), 0.02, 1.225
    table = {}
    for name, spec in [
        ("coefficient", skywalker_x8_spec()),
        ("panels", skywalker_x8_panels_spec()),
    ]:
        model = spec.to_model()
        state = zero_state(model, altitude=1000)
        state = state._replace(
            rigid_body=state.rigid_body._replace(
                velocity=jnp.array([speed * np.cos(alpha), 0, speed * np.sin(alpha)])
            )
        )
        reference = JSBSimAircraft(spec, directory / name)
        derivatives = {}
        qS = 0.5 * rho * speed**2 * spec.reference_area_m2
        for axis, letter in enumerate("pqr"):
            values = []
            for sign in [-1, 1]:
                changed = state._replace(
                    rigid_body=state.rigid_body._replace(
                        angular_velocity=jnp.eye(3)[axis] * perturbation * sign
                    )
                )
                reference.initialize(changed, equilibrate=True, control=zero_control(model))
                loads = reference.loads()
                force, moment = loads["aero-force"] / qS, loads["aero-moment"] / qS
                values.append(
                    np.array(
                        [
                            np.sin(alpha) * force[0] - np.cos(alpha) * force[2],
                            force[1],
                            moment[0] / spec.reference_span_m,
                            moment[1] / spec.reference_chord_m,
                            moment[2] / spec.reference_span_m,
                        ]
                    )
                )
            length = spec.reference_chord_m if axis == 1 else spec.reference_span_m
            derivative = (values[1] - values[0]) / (perturbation * length / speed)
            for index in [0, 3] if axis == 1 else [1, 2, 4]:
                derivatives[["CL", "CY", "Cl", "Cm", "Cn"][index] + "_" + letter] = float(
                    derivative[index]
                )
        table[name] = derivatives
    result = {
        "description": (
            "Model approximation differences, not implementation failures or flight accuracy"
        ),
        "configuration": {
            "alpha_deg": 3,
            "airspeed_m_s": speed,
            "central_rate_step_rad_s": perturbation,
            "propellers": "stopped",
            "separation": "equilibrium",
        },
        "derivatives": table,
        "opposite_signs": [
            key
            for key in table["coefficient"]
            if table["coefficient"][key] * table["panels"][key] < 0
        ],
    }
    json_write(directory / "rate-comparison.json", result)
    return result
