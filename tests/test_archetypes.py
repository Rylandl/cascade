import jax
import numpy as np
import pytest

from cascade.archetypes import (
    ConventionalDesign,
    FlyingWingDesign,
    design_spec,
    flap_effectiveness,
    flap_moment_coefficient,
    sample_designs,
    validate_design,
    wing_lift_slope,
)
from cascade.spec import load_aircraft_spec, save_aircraft_spec

NOMINALS = (
    FlyingWingDesign(),
    FlyingWingDesign(motors="twin_tractor"),
    ConventionalDesign(),
    ConventionalDesign(tail="v_tail"),
)


def test_relations_match_the_textbook():
    # Helmbold tends to the section slope at infinite aspect ratio and to pi AR / 2 at low.
    assert abs(wing_lift_slope(1e6) - 5.7) < 0.01
    assert wing_lift_slope(4.0) < wing_lift_slope(8.0) < 5.7
    # A quarter-chord plain flap is about 0.6 effective and pitches nose-down.
    assert 0.5 < flap_effectiveness(0.25) < 0.65
    assert flap_moment_coefficient(0.25) < -0.3
    assert flap_effectiveness(0.4) > flap_effectiveness(0.2)


@pytest.mark.parametrize("design", NOMINALS, ids=lambda d: f"{type(d).__name__}")
def test_nominal_designs_trim_and_are_flyable(design):
    report = validate_design(design)
    assert report.valid, report.reasons
    assert 0.3 < report.throttle < 0.9
    assert report.pitch_authority_rad_s2 > 20.0 and report.roll_authority_rad_s2 > 20.0
    assert 1.0 < report.short_period_hz < 5.0


def test_specs_round_trip_through_toml(tmp_path):
    for index, design in enumerate(NOMINALS):
        spec = design_spec(design)
        path = tmp_path / f"design_{index}.toml"
        save_aircraft_spec(spec, path)
        loaded = load_aircraft_spec(path)
        assert loaded.mass_kg == pytest.approx(spec.mass_kg)
        assert len(loaded.surfaces) == len(spec.surfaces)
        loaded.to_model()


def test_static_margin_moves_the_centre_of_mass_aft_of_the_panels():
    tight = design_spec(FlyingWingDesign(static_margin=0.02))
    loose = design_spec(FlyingWingDesign(static_margin=0.15))
    # More static margin puts the centre of mass further forward, so panels sit further aft.
    assert np.mean([s.position_m[0] for s in loose.surfaces]) < np.mean(
        [s.position_m[0] for s in tight.surfaces]
    )


def test_sampled_designs_are_mostly_valid_and_visibly_diverse():
    for archetype, minimum_rate in ((FlyingWingDesign, 0.4), (ConventionalDesign, 0.6)):
        reports = [validate_design(d) for d in sample_designs(archetype, jax.random.PRNGKey(0), 12)]
        valid = [r for r in reports if r.valid]
        assert len(valid) >= minimum_rate * len(reports), [r.reasons for r in reports]
        speeds = np.array([r.cruise_speed_m_s for r in valid])
        periods = np.array([r.short_period_hz for r in valid])
        authority = np.array([r.pitch_authority_rad_s2 for r in valid])
        assert speeds.max() / speeds.min() > 1.3
        assert periods.max() / periods.min() > 1.3
        assert authority.max() / authority.min() > 2.0
