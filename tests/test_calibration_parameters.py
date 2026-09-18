"""Calibration parameter bounds and spec/model numerical agreement."""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade.calibration.parameters import FitParameter, Parameterization
from cascade.dynamics import evaluate_dynamics
from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.reference import aerobatic_reference_spec, skywalker_x8_spec
from cascade.spec import load_aircraft_spec, save_aircraft_spec


def assert_models_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=1e-7)


def bounds_around(value):
    return (min(value * 0.5, value * 1.5), max(value * 0.5, value * 1.5)) if value else (0, 0.1)


def test_mass_inertia_roundtrip_preserves_tensor_shape_and_cached_inverse(tmp_path):
    original = aerobatic_reference_spec()
    parameterization = Parameterization(
        original, [FitParameter("mass_kg", 0.8, 1.8), FitParameter("inertia_scale", 0.5, 2)]
    )
    assert parameterization.paths == ("mass_kg", "inertia_scale")
    assert parameterization.size == 2 and isinstance(parameterization.parameters, tuple)
    np.testing.assert_allclose(parameterization.initial_values, [original.mass_kg, 1])
    values = jnp.array([1.5, 1.25])
    fitted = jax.jit(parameterization.apply_model)(values)
    nominal = original.to_model()
    np.testing.assert_allclose(fitted.inertia, nominal.inertia * 1.25)
    np.testing.assert_allclose(fitted.inertia_inverse, nominal.inertia_inverse / 1.25)
    np.testing.assert_allclose(fitted.inertia @ fitted.inertia_inverse, np.eye(3), atol=1e-6)
    spec = parameterization.apply_spec(values)
    assert spec.mass_kg == 1.5 and spec.surfaces == original.surfaces
    assert spec.propellers == original.propellers and spec.body == original.body
    path = tmp_path / "fitted.toml"
    save_aircraft_spec(spec, path)
    loaded = load_aircraft_spec(path)
    assert loaded == spec
    assert_models_close(fitted, loaded.to_model())
    assert_models_close(parameterization.apply_model(parameterization.initial_values), nominal)
    # Applications start from the nominal tensor, never compound a previous scale.
    assert_models_close(parameterization.apply_model(values), fitted)
    derivative = jax.jacfwd(lambda v: parameterization.apply_model(v).inertia_inverse)(values)
    np.testing.assert_allclose(derivative[..., 1], -nominal.inertia_inverse / 1.25**2, rtol=1e-6)


def test_all_supported_surface_scalars_agree_with_spec_compilation():
    spec = aerobatic_reference_spec()
    selected = (
        "lift_coefficient_zero",
        "lift_curve_slope_rad",
        "drag_coefficient_zero",
        "induced_drag_factor",
        "moment_coefficient_zero",
        "moment_coefficient_alpha_rad",
        "stall_angle_rad",
        "stall_width_rad",
        "normal_force_coefficient",
        "edge_drag_coefficient",
        "span_drag_coefficient",
        "separation_time_constant_s",
        "reattachment_time_constant_s",
        "flap_effectiveness",
        "moment_coefficient_flap_rad",
        "drag_coefficient_flap_rad2",
        "actuator_time_constant_s",
    )
    parameters = tuple(
        FitParameter(f"surfaces.left_wing.{name}", *bounds_around(getattr(spec.surfaces[0], name)))
        for name in selected
    )
    fit = Parameterization(spec, parameters)
    values = fit.initial_values + 0.25 * (fit.upper_bounds - fit.initial_values)
    compiled = jax.jit(fit.apply_model)(values)
    saved = fit.apply_spec(values)
    assert_models_close(compiled, saved.to_model())
    for name, value in zip(selected, values, strict=True):
        assert getattr(saved.surfaces[0], name) == float(value)
    assert saved.surfaces[1:] == spec.surfaces[1:]
    for name in (
        "position_m",
        "body_from_surface",
        "area_m2",
        "chord_m",
        "control_map_rad",
        "actuator_limit_rad",
        "actuator_rate_limit_rad_s",
        "separated_center_of_pressure",
    ):
        assert getattr(saved.surfaces[0], name) == getattr(spec.surfaces[0], name)


def test_body_polynomials_and_blend_scalars_agree_with_spec_compilation():
    spec = skywalker_x8_spec()
    parameters = []
    for group in ("lift", "drag", "side", "roll", "pitch", "yaw"):
        coefficient = getattr(spec.body, group)
        for field in fields(coefficient):
            parameters.append(
                FitParameter(
                    f"body.{group}.{field.name}", *bounds_around(getattr(coefficient, field.name))
                )
            )
    for field in (
        "stall_angle_rad",
        "stall_width_rad",
        "normal_force_coefficient",
        "pitch_flat_plate",
    ):
        parameters.append(FitParameter(f"body.{field}", *bounds_around(getattr(spec.body, field))))
    fit = Parameterization(spec, parameters)
    values = fit.initial_values + 0.2 * (fit.upper_bounds - fit.initial_values)
    changed = fit.apply_spec(values)
    assert_models_close(jax.jit(fit.apply_model)(values), changed.to_model())
    assert changed.surfaces == spec.surfaces and changed.propellers == spec.propellers
    assert changed.body.deflection_map == spec.body.deflection_map
    assert changed.body.reference_position_m == spec.body.reference_position_m


def test_named_surface_target_follows_names_after_spec_reordering():
    original = aerobatic_reference_spec()
    reordered = replace(
        original, surfaces=(original.surfaces[1], original.surfaces[0], *original.surfaces[2:])
    )
    fit = Parameterization(
        reordered, [FitParameter("surfaces.left_wing.lift_curve_slope_rad", 3, 7)]
    )
    model = fit.apply_model(jnp.array([6.0]))
    assert float(model.surfaces.lift_curve_slope[1]) == 6
    assert float(model.surfaces.lift_curve_slope[0]) == pytest.approx(4.8)
    assert fit.apply_spec([6.0]).surfaces[1].lift_curve_slope_rad == 6


def test_parameterized_dynamics_gradients_match_finite_difference_and_batching():
    fit = Parameterization(
        aerobatic_reference_spec(),
        (
            FitParameter("mass_kg", 0.9, 1.8),
            FitParameter("surfaces.left_wing.lift_coefficient_zero", -0.2, 0.6),
        ),
    )
    model = fit.apply_model(fit.initial_values)
    state = zero_state(model, forward_speed=12, altitude=50)
    control, environment = zero_control(model), standard_environment()

    def acceleration(values):
        result = evaluate_dynamics(fit.apply_model(values), state, control, environment)
        return result.derivative.rigid_body.velocity

    jacobian = jax.jit(jax.jacfwd(acceleration))(fit.initial_values)
    assert np.isfinite(jacobian).all() and np.all(np.linalg.norm(jacobian, axis=0) > 0)
    finite_difference = np.column_stack(
        [
            (
                np.asarray(acceleration(fit.initial_values.at[index].add(1e-3)))
                - np.asarray(acceleration(fit.initial_values.at[index].add(-1e-3)))
            )
            / 2e-3
            for index in range(fit.size)
        ]
    )
    np.testing.assert_allclose(jacobian, finite_difference, atol=2e-3, rtol=2e-3)
    values = jnp.stack([fit.initial_values, fit.initial_values + jnp.array([0.1, 0.02])])
    batched = jax.jit(jax.vmap(fit.apply_model))(values)
    np.testing.assert_allclose(batched.mass, values[:, 0])
    np.testing.assert_allclose(batched.surfaces.lift_coefficient_zero[:, 0], values[:, 1])


def test_model_transform_is_unclipped_but_host_spec_enforces_bounds_and_precision():
    spec = replace(aerobatic_reference_spec(), mass_kg=1.0)
    fit = Parameterization(spec, [FitParameter("mass_kg", 0.9, 1.1)])
    assert float(jax.jit(fit.apply_model)(jnp.array([1.2])).mass) > 1.1
    with pytest.raises(ValueError, match="outside"):
        fit.apply_spec([1.2])
    assert fit.apply_spec([1.1]).mass_kg == 1.1
    assert fit.apply_spec(fit.upper_bounds).mass_kg == 1.1
    assert fit.apply_spec(fit.lower_bounds).mass_kg == 0.9
    # Supplied double-precision host values remain reproducible in the saved spec.
    value = 1.0123456789
    assert fit.apply_spec([value]).mass_kg == value


@pytest.mark.parametrize(
    "lower,upper",
    [
        (True, 2),
        (0, False),
        (1, 1),
        (2, 1),
        (float("nan"), 2),
        (0, float("inf")),
        ("0", 2),
        (0, 10**1000),
    ],
)
def test_parameter_declaration_rejects_invalid_bounds(lower, upper):
    with pytest.raises(ValueError):
        FitParameter("mass_kg", lower, upper)


@pytest.mark.parametrize(
    "path",
    [
        "mass",
        "inertia_inverse",
        "inertia_kg_m2",
        "reference_area_m2",
        "surfaces.left_wing.area_m2",
        "surfaces.left_wing.position_m",
        "surfaces.left_wing.control_map_rad",
        "surfaces.left_wing.actuator_limit_rad",
        "surfaces.left_wing.actuator_rate_limit_rad_s",
        "surfaces.left_wing.actuator_bias_rad",
        "surfaces.left_wing.all_moving_fraction",
        "surfaces.left_wing.separated_center_of_pressure",
        "propellers.nose_propeller.thrust_map",
        "surfaces.0.lift_coefficient_zero",
    ],
)
def test_geometry_topology_arrays_maps_and_limits_are_not_parameters(path):
    with pytest.raises(ValueError, match="unsupported|unknown surface"):
        Parameterization(aerobatic_reference_spec(), [FitParameter(path, -10, 10)])


def test_invalid_targets_nominal_bounds_and_physical_domains_are_rejected():
    spec = aerobatic_reference_spec()
    with pytest.raises(ValueError, match="unique"):
        Parameterization(spec, [FitParameter("mass_kg", 1, 2)] * 2)
    with pytest.raises(ValueError, match="inside"):
        Parameterization(spec, [FitParameter("mass_kg", 2, 3)])
    with pytest.raises(ValueError, match="positive"):
        Parameterization(spec, [FitParameter("inertia_scale", 0, 2)])
    with pytest.raises(ValueError, match="nonnegative"):
        Parameterization(spec, [FitParameter("surfaces.left_wing.drag_coefficient_zero", -1, 1)])
    with pytest.raises(ValueError, match="existing"):
        Parameterization(spec, [FitParameter("body.pitch.zero", -1, 1)])
    with pytest.raises(ValueError, match="nonempty"):
        Parameterization(spec, [])
    with pytest.raises(ValueError, match="unique"):
        Parameterization(
            replace(spec, surfaces=(spec.surfaces[0], spec.surfaces[0])),
            [FitParameter("mass_kg", 1, 2)],
        )


@pytest.mark.parametrize("lower,upper", [(1.2, 1.2 + 1e-10), (1e-45, 2), (1, 1e40)])
def test_bounds_must_be_representable_in_working_precision(lower, upper):
    with pytest.raises(ValueError, match="working precision"):
        Parameterization(aerobatic_reference_spec(), [FitParameter("mass_kg", lower, upper)])


def test_working_precision_is_captured_when_parameters_are_resolved():
    prior = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        fit = Parameterization(
            aerobatic_reference_spec(), [FitParameter("mass_kg", 1.2, 1.2 + 1e-10)]
        )
        assert fit.initial_values.dtype == jnp.float64
        assert float(fit.upper_bounds[0] - fit.lower_bounds[0]) > 0
        assert_models_close(
            fit.apply_model(fit.upper_bounds), fit.apply_spec(fit.upper_bounds).to_model()
        )
    finally:
        jax.config.update("jax_enable_x64", prior)


@pytest.mark.parametrize(
    "values", [[float("nan")], [float("inf")], [True], [[1]], [1, 2], [1 + 1j]]
)
def test_saved_spec_rejects_malformed_values(values):
    fit = Parameterization(aerobatic_reference_spec(), [FitParameter("mass_kg", 0.8, 1.8)])
    with pytest.raises(ValueError):
        fit.apply_spec(values)


def test_overflowing_inertia_update_is_rejected_at_construction():
    with pytest.raises(ValueError, match="finite|inertia"):
        Parameterization(aerobatic_reference_spec(), [FitParameter("inertia_scale", 1e-38, 2)])
