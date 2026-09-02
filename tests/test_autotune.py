import jax
import pytest

import cascade
from cascade.archetypes import (
    ConventionalDesign,
    FlyingWingDesign,
    cruise_speed,
    design_spec,
    sample_designs,
    validate_design,
)
from cascade.autotune import step_response, tune_cascade

CASES = (
    ("aerobatic", cascade.aerobatic_reference_spec(), 12.0),
    ("x8", cascade.skywalker_x8_spec(), 18.0),
    ("flying-wing", design_spec(FlyingWingDesign()), cruise_speed(FlyingWingDesign())),
    (
        "tailsitter-wing",
        design_spec(FlyingWingDesign(motors="twin_tractor")),
        cruise_speed(FlyingWingDesign(motors="twin_tractor")),
    ),
    ("conventional", design_spec(ConventionalDesign()), cruise_speed(ConventionalDesign())),
    (
        "v-tail",
        design_spec(ConventionalDesign(tail="v_tail")),
        cruise_speed(ConventionalDesign(tail="v_tail")),
    ),
)


@pytest.mark.parametrize("name,spec,speed", CASES, ids=[c[0] for c in CASES])
def test_tuned_cascade_settles_heading_and_altitude_steps(name, spec, speed):
    controller, report = tune_cascade(spec, speed)
    # Signs came from the measured authority, gains from bandwidth over authority.
    assert float(report.throttle_acceleration_m_s2) > 0.0
    assert bool(jax.numpy.all(controller.rate.kp >= 0.0))
    response = step_response(spec.to_model(), controller, report.trim)
    assert response.finite
    assert response.settled, response


def test_tuning_generalises_over_sampled_designs():
    settled = 0
    tried = 0
    for archetype in (FlyingWingDesign, ConventionalDesign):
        for design in sample_designs(archetype, jax.random.PRNGKey(3), 5):
            report = validate_design(design)
            if not report.valid:
                continue
            spec = design_spec(design)
            controller, tuning = tune_cascade(spec, report.cruise_speed_m_s)
            response = step_response(spec.to_model(), controller, tuning.trim)
            tried += 1
            settled += int(response.settled)
    assert tried >= 5
    assert settled >= 0.8 * tried, (settled, tried)
