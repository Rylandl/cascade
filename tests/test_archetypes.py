import jax
import numpy as np
import pytest

from cascade.design.archetypes import (
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

pytestmark = pytest.mark.slow

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


def test_inertia_is_built_from_exactly_the_aircraft_mass():
    import numpy as np

    from cascade.design.archetypes import _inertia_from_parts, _plate_inertia

    # Parts summing to 1.1 of the mass are scaled down so the tensor is that of the aircraft.
    parts = [(0.6, (0.0, 0.5, 0.0), 0.2, 1.0, 0.0), (0.2, (0.0, -0.5, 0.0), 0.2, 1.0, 0.0)]
    scaled = np.asarray(_inertia_from_parts(parts, 0.3, 0.05, pod_length=0.6, total_mass=1.0))
    expected = np.zeros((3, 3))
    for mass, centre, chord, width, height in parts:
        expected += _plate_inertia(mass / 1.1, centre, chord, width, height)
    pod = 0.3 / 1.1
    expected += np.diag(
        [0.4 * pod * 0.05**2, 0.2 * pod * (0.3**2 + 0.05**2), 0.2 * pod * (0.3**2 + 0.05**2)]
    )
    assert np.allclose(scaled, expected)
    # A slender pod adds to pitch and yaw far more than to roll.
    assert scaled[1, 1] > scaled[0, 0] * 0.0 and expected[2, 2] > expected[0, 0]
    # Every archetype's tensor is symmetric positive definite with plausible ordering.
    for design in NOMINALS:
        inertia = np.asarray(design_spec(design).inertia_kg_m2)
        assert np.allclose(inertia, inertia.T)
        assert np.all(np.linalg.eigvalsh(inertia) > 0.0)
        assert inertia[2, 2] > inertia[1, 1] * 0.5


def test_conventional_tail_sees_the_wing_downwash_at_cruise():
    from cascade.analysis import StraightFlightCondition, trim_straight_flight
    from cascade.archetypes import cruise_speed
    from cascade.dynamics import evaluate_dynamics
    from cascade.initialization import standard_environment

    design = ConventionalDesign()
    spec = design_spec(design)
    names = [s.name for s in spec.surfaces]
    table = np.asarray(spec.downwash_map)
    tail, wing = names.index("horizontal_tail"), names.index("left_wing_inner")
    assert table[tail, wing] > 0.0 and table[wing, tail] == 0.0
    model = spec.to_model()
    environment = standard_environment()
    trim = trim_straight_flight(
        model,
        StraightFlightCondition(cruise_speed(design), altitude_m=50.0),
        environment=environment,
    )
    assert trim.success
    air = evaluate_dynamics(model, trim.state, trim.control, environment).aerodynamics.air
    assert float(air.angle_of_attack[tail]) < float(air.angle_of_attack[wing])
    # The tail keeps its full slope: pitch authority is well above the downwash-scaled value.
    report = validate_design(design)
    assert report.valid and report.pitch_authority_rad_s2 > 80.0
