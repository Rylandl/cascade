"""Synthetic recovery and numerical contracts; these are not measured-flight evidence."""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy

import cascade
from cascade.calibration.fitting import (
    CalibrationConfig,
    ResidualScales,
    _state_residual,
    fit_records,
)
from cascade.calibration.parameters import FitParameter
from cascade.canonical import rigid_body_to_canonical
from cascade.experiments import FlightRecord
from cascade.experiments.replay import prepare_record, replay_record
from cascade.state import Environment


@pytest.fixture(scope="module")
def recordings():
    nominal = cascade.aerobatic_reference_spec()
    truth = replace(nominal, mass_kg=nominal.mass_kg * 1.08)
    model = truth.to_model()
    environment = cascade.standard_environment(density=1.18)._replace(
        wind=jnp.array([0.7, -0.2, 0.0])
    )
    trimmed = cascade.trim_straight_flight(
        model, cascade.StraightFlightCondition(12.0, altitude_m=50.0), environment=environment
    )
    assert trimmed.success
    count, substeps, dt = 32, 4, 0.025

    def generate(name, pulse):
        controls = cascade.repeat_control(trimmed.control, count)
        controls = controls._replace(
            channel=controls.channel.at[6:16, 1].add(pulse).at[18:26, 0].add(pulse * 0.6),
            propeller=controls.propeller.at[22:].add(0.025),
        )
        wind = np.tile(np.asarray(environment.wind), (count + 1, 1))
        wind[18:, 0] += 0.08
        density = np.full(count + 1, 1.18)
        density[20:] -= 0.02
        environments = Environment(
            jnp.repeat(jnp.asarray(density[1:]), substeps),
            jnp.repeat(jnp.asarray(wind[1:]), substeps, axis=0),
            jnp.tile(environment.gravity, (count * substeps, 1)),
        )
        _, states = jax.jit(cascade.rollout)(
            model,
            trimmed.state,
            jax.tree.map(lambda value: jnp.repeat(value, substeps, axis=0), controls),
            environment,
            dt / substeps,
            environments=environments,
        )
        observed = jnp.concatenate(
            (
                rigid_body_to_canonical(trimmed.state.rigid_body)[None],
                rigid_body_to_canonical(states.rigid_body)[substeps - 1 :: substeps],
            )
        )
        commands = jnp.concatenate(
            (cascade.control_to_array(trimmed.control)[None], cascade.control_to_array(controls))
        )
        return FlightRecord(
            name,
            name,
            np.arange(count + 1) * dt,
            np.asarray(observed),
            np.asarray(commands),
            wind_ned_m_s=wind,
            density_kg_m3=density,
        )

    return nominal, truth, generate("fit-pitch", 0.04), generate("heldout-pitch", -0.03)


def mass_config(spec, **options):
    return CalibrationConfig(
        (FitParameter("mass_kg", spec.mass_kg * 0.8, spec.mass_kg * 1.2),), **options
    )


@pytest.mark.parametrize(
    "options",
    [
        {"substeps": True},
        {"substeps": 0},
        {"warmup_steps": -1},
        {"max_nfev": 0},
        {"max_nfev": 1.5},
        {"scales": {}},
    ],
)
def test_config_validates_static_choices(options):
    with pytest.raises((TypeError, ValueError)):
        CalibrationConfig((FitParameter("mass_kg", 1.0, 2.0),), **options)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_residual_scales_require_finite_positive_values(value):
    with pytest.raises(ValueError, match="finite and positive"):
        ResidualScales(attitude_rad=value)


def test_configuration_json_roundtrip_and_unknown_fields():
    config = CalibrationConfig(
        (FitParameter("mass_kg", 1.0, 2.0),),
        warmup_steps=3,
        scales=ResidualScales(position_m=2.0),
    )
    payload = json.loads(json.dumps(config.to_dict(), allow_nan=False))
    assert CalibrationConfig.from_dict(payload) == config
    for malformed in (
        {**payload, "unknown": 1},
        {**payload, "parameters": [{"path": "mass_kg", "lower": 1, "upper": 2, "extra": 0}]},
        {**payload, "scales": {"unknown": 1}},
    ):
        with pytest.raises(ValueError):
            CalibrationConfig.from_dict(malformed)


def test_shared_replay_matches_independent_rollout_and_ignores_unused_rows(recordings):
    _, truth, record, _ = recordings
    model = truth.to_model()
    inputs = prepare_record(record, model, substeps=4)
    predicted = jax.jit(replay_record)(model, inputs)
    np.testing.assert_allclose(predicted, record.canonical_state[1:], rtol=2e-5, atol=3e-5)
    command, wind, density = (
        record.command.copy(),
        record.wind_ned_m_s.copy(),
        record.density_kg_m3.copy(),
    )
    command[0] = 0
    wind[0] = [5, 2, 1]
    density[0] = 2
    changed = replace(record, command=command, wind_ned_m_s=wind, density_kg_m3=density)
    np.testing.assert_array_equal(
        predicted, jax.jit(replay_record)(model, prepare_record(changed, model, substeps=4))
    )


def test_replay_mass_jacobian_matches_finite_difference(recordings):
    nominal, _, record, _ = recordings
    model = nominal.to_model()
    inputs = prepare_record(record, model, substeps=4)

    def final_velocity(mass):
        return replay_record(model._replace(mass=mass), inputs)[-1, 3:6]

    mass = jnp.asarray(nominal.mass_kg)
    derivative = jax.jit(jax.jacfwd(final_velocity))(mass)
    evaluate = jax.jit(final_velocity)
    delta = 0.002 * nominal.mass_kg
    finite_difference = (evaluate(mass + delta) - evaluate(mass - delta)) / (2 * delta)
    assert np.linalg.norm(derivative) > 0.01
    np.testing.assert_allclose(derivative, finite_difference, rtol=3e-3, atol=2e-3)


def test_attitude_residual_is_antipodal_invariant_with_finite_exact_fit_derivatives():
    state = jnp.zeros((2, 13)).at[:, 6].set(1)
    opposite = state.at[:, 6:10].multiply(-1)
    residual = jax.jit(
        lambda predicted, observed: _state_residual(predicted, observed, ResidualScales())
    )
    np.testing.assert_array_equal(residual(state, state), residual(opposite, state))
    half_turn = state.at[:, 6].set(0).at[:, 7].set(1)
    np.testing.assert_array_equal(
        residual(half_turn, state), residual(half_turn.at[:, 6:10].multiply(-1), state)
    )
    assert np.isfinite(jax.jacfwd(lambda value: residual(value, state))(state)).all()
    assert np.isfinite(jax.jacfwd(lambda value: residual(value, opposite))(state)).all()


def test_mass_recovery_improves_separate_maneuver(recordings):
    nominal, truth, fitting, heldout = recordings
    fitted = fit_records(nominal, (fitting,), mass_config(nominal, max_nfev=20))
    assert fitted.report["optimizer"]["success"]
    np.testing.assert_allclose(fitted.spec.mass_kg, truth.mass_kg, rtol=2e-3)
    assert fitted.report["final_data_loss"] < fitted.report["initial_data_loss"] * 0.01
    assert fitted.report["identifiability"]["rank"] == 1
    assert fitted.records[0]["name"] == fitting.name
    assert heldout.name not in json.dumps(fitted.report)
    nominal_model, calibrated_model = nominal.to_model(), fitted.spec.to_model()
    inputs = prepare_record(heldout, nominal_model, substeps=4)
    replay = jax.jit(replay_record)
    nominal_error = np.linalg.norm(
        np.asarray(replay(nominal_model, inputs))[:, 3:6] - heldout.canonical_state[1:, 3:6]
    )
    calibrated_error = np.linalg.norm(
        np.asarray(replay(calibrated_model, inputs))[:, 3:6] - heldout.canonical_state[1:, 3:6]
    )
    assert calibrated_error < 0.05 * nominal_error
    json.dumps(fitted.report, allow_nan=False)


def test_budget_limit_returns_truthful_finite_candidate(recordings):
    nominal, _, fitting, _ = recordings
    fitted = fit_records(nominal, (fitting,), mass_config(nominal, max_nfev=1))
    assert not fitted.report["optimizer"]["success"]
    assert fitted.report["optimizer"]["status"] == 0
    assert fitted.report["optimizer"]["nfev"] == 1
    fitted.spec.to_model()
    np.testing.assert_allclose(fitted.spec.mass_kg, nominal.mass_kg, rtol=2e-6)
    runtime = fitted.provenance["runtime"]
    fitting_precision = bool(jax.config.x64_enabled)
    assert runtime["x64_enabled"] is fitting_precision
    assert runtime["numpy_version"] == np.__version__
    assert runtime["scipy_version"] == scipy.__version__
    assert runtime["spec_hash"] == fitted.nominal_spec_sha256
    assert len(fitted.provenance["implementation_sha256"]) == 64
    # Delayed publication may happen under a different runtime; the fit snapshot stays fixed.
    try:
        jax.config.update("jax_enable_x64", not fitting_precision)
        assert cascade.stamp()["x64_enabled"] is not fitting_precision
        assert fitted.provenance["runtime"]["x64_enabled"] is fitting_precision
    finally:
        jax.config.update("jax_enable_x64", fitting_precision)


def test_warmup_excludes_targets_without_resetting_replay(recordings):
    nominal, _, record, _ = recordings
    altered = record.canonical_state.copy()
    altered[1:4, :6] += 50
    changed = replace(record, canonical_state=altered)
    config = mass_config(nominal, warmup_steps=3, max_nfev=1)
    original = fit_records(nominal, (record,), config)
    excluded = fit_records(nominal, (changed,), config)
    assert original.report["initial_data_loss"] == excluded.report["initial_data_loss"]
    assert excluded.records[0]["scored_samples"] == len(record.time_s) - 4
    with pytest.raises(ValueError, match="leave at least one"):
        fit_records(nominal, (record,), replace(config, warmup_steps=len(record.time_s) - 1))


def test_equal_record_weighting_uses_each_records_mean(recordings):
    nominal, _, long, _ = recordings
    short = replace(
        long,
        name="short",
        maneuver_id="short",
        time_s=long.time_s[:7],
        canonical_state=long.canonical_state[:7],
        command=long.command[:7],
        wind_ned_m_s=long.wind_ned_m_s[:7],
        density_kg_m3=long.density_kg_m3[:7],
    )
    config = mass_config(nominal, max_nfev=1)
    fitted = fit_records(nominal, (short, long), config)
    model = nominal.to_model()
    losses = []
    for record in (short, long):
        inputs = prepare_record(record, model, substeps=4)
        residual = _state_residual(
            jax.jit(replay_record)(model, inputs), inputs.observed, config.scales
        )
        losses.append(float(jnp.mean(jnp.sum(residual**2, axis=1))))
    np.testing.assert_allclose(fitted.report["initial_data_loss"], np.mean(losses), rtol=2e-5)


def test_unexcited_parameter_reports_rank_zero(recordings):
    nominal, _, original, _ = recordings
    # A zero-area surface produces no force for any value of its lift-zero coefficient.
    surfaces = tuple(replace(surface, area_m2=0.0) for surface in nominal.surfaces)
    spec = replace(nominal, surfaces=surfaces)
    path = f"surfaces.{surfaces[0].name}.lift_coefficient_zero"
    center = surfaces[0].lift_coefficient_zero
    config = CalibrationConfig((FitParameter(path, center - 0.1, center + 0.1),), max_nfev=2)
    result = fit_records(spec, (original,), config)
    assert result.report["optimizer"]["success"]
    assert result.report["identifiability"]["rank"] == 0
    assert result.report["identifiability"]["condition"] is None
    assert result.report["identifiability"]["singular_values"] == [0.0]


def test_nonfinite_replay_is_not_returned_as_a_fit(recordings):
    nominal, _, original, _ = recordings
    state = original.canonical_state.copy()
    state[0, 3:6] = 1e20  # representable input, overflowing aerodynamic arithmetic
    with pytest.raises(FloatingPointError, match="nonfinite"):
        fit_records(nominal, (replace(original, canonical_state=state),), mass_config(nominal))


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_nominal_at_decimal_bound_is_a_valid_solver_start(recordings, side):
    nominal, _, original, _ = recordings
    spec = replace(nominal, mass_kg=1.1)
    bounds = (1.1, 1.9) if side == "lower" else (0.3, 1.1)
    config = CalibrationConfig((FitParameter("mass_kg", *bounds),), max_nfev=1)
    fitted = fit_records(spec, (original,), config)
    assert np.isfinite(fitted.report["final_data_loss"])
    assert fitted.report["parameters"][0]["bound_hit"] == side
    np.testing.assert_allclose(fitted.spec.mass_kg, spec.mass_kg, rtol=2e-6)


def test_prepare_rejects_unrepresentable_later_targets(recordings):
    nominal, _, original, _ = recordings
    if jax.config.x64_enabled:
        pytest.skip("float32 overflow boundary")
    state = original.canonical_state.copy()
    state[-1, 0] = 1e100
    with pytest.raises(ValueError, match="state or commands overflow"):
        prepare_record(replace(original, canonical_state=state), nominal.to_model())
