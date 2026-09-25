"""Independent JSBSim checks, optional in core installs and required by validation CI."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import cascade
from cascade.validation.campaign import (
    compare_flights,
    fixtures,
    point_cases,
    trajectory_errors,
)
from cascade.validation.jsbsim import JSBSimAircraft

pytest.importorskip("jsbsim")


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("name", list(fixtures()))
def test_native_jsbsim_loads_accelerations_and_equilibrium(name, tmp_path):
    spec, speed = fixtures()[name]
    model = spec.to_model()
    reference = JSBSimAircraft(spec, tmp_path)
    evaluate = jax.jit(cascade.evaluate_dynamics)
    for case, state, control in point_cases(model, speed, smoke=True):
        reference.initialize(state)
        # JSBSim silently creates unknown property names. These boundary checks catch
        # mistyped IC setters, including the non-obvious 'ic/psi-true-rad'.
        snapshot = reference.snapshot()
        np.testing.assert_allclose(snapshot[3:6], state.rigid_body.velocity, atol=1e-10)
        delta = Rotation.from_quat(snapshot[6:10]).inv() * Rotation.from_quat(
            state.rigid_body.attitude
        )
        assert delta.magnitude() < 1e-10
        result = evaluate(model, state, control, cascade.standard_environment())
        np.testing.assert_allclose(reference.loads()["force"], result.force_body, atol=1e-8)
        np.testing.assert_allclose(reference.loads()["moment"], result.moment_body, atol=1e-8)
        accel, angular = reference.accelerations()
        np.testing.assert_allclose(accel, result.derivative.rigid_body.velocity, atol=1e-6)
        np.testing.assert_allclose(
            angular, result.derivative.rigid_body.angular_velocity, atol=1e-7
        )
        if not case["stale_separation"]:
            reference.initialize(state, equilibrate=True, control=control)
            np.testing.assert_allclose(reference.separation, state.aero.separation, atol=2e-7)


def test_cross_inertia_and_coefficient_reference_translation(tmp_path):
    spec = cascade.skywalker_x8_spec()
    spec = replace(
        spec,
        inertia_kg_m2=((0.335, 0.014, -0.029), (0.014, 0.14, -0.018), (-0.029, -0.018, 0.40)),
        body=replace(spec.body, reference_position_m=(0.11, -0.07, 0.04)),
    )
    model = spec.to_model()
    _, state, control = next(point_cases(model, 18.0, smoke=True))
    reference = JSBSimAircraft(spec, tmp_path)
    reference.initialize(state)
    result = cascade.evaluate_dynamics(model, state, control, cascade.standard_environment())
    np.testing.assert_allclose(reference.loads()["moment"], result.moment_body, atol=1e-9)
    np.testing.assert_allclose(
        reference.accelerations()[1], result.derivative.rigid_body.angular_velocity, atol=1e-8
    )


def test_reference_is_sensitive_to_its_own_model_not_cascade_outputs(tmp_path):
    spec = cascade.skywalker_x8_spec()
    model = spec.to_model()
    _, state, control = next(point_cases(model, 18.0, smoke=True))
    perturbed = replace(
        spec, body=replace(spec.body, lift=replace(spec.body.lift, zero=spec.body.lift.zero + 0.1))
    )
    reference = JSBSimAircraft(perturbed, tmp_path)
    reference.initialize(state)
    result = cascade.evaluate_dynamics(model, state, control, cascade.standard_environment())
    assert np.linalg.norm(reference.loads()["force"] - result.force_body) > 5


def test_load_derivative_matches_jsbsim_central_difference(tmp_path):
    spec = cascade.skywalker_x8_spec()
    model, env = spec.to_model(), cascade.standard_environment()
    _, state, control = next(point_cases(model, 18.0, smoke=True))
    reference = JSBSimAircraft(spec, tmp_path)

    def force(delta):
        changed = state._replace(
            actuators=state.actuators._replace(
                surface_deflection=state.actuators.surface_deflection.at[0].add(delta)
            )
        )
        return cascade.evaluate_dynamics(model, changed, control, env).force_body

    values = []
    h = 1e-5
    for delta in [-h, h]:
        changed = state._replace(
            actuators=state.actuators._replace(
                surface_deflection=state.actuators.surface_deflection.at[0].add(delta)
            )
        )
        reference.initialize(changed)
        values.append(reference.loads()["force"])
    np.testing.assert_allclose(
        jax.jacfwd(force)(0.0), (values[1] - values[0]) / (2 * h), rtol=2e-7, atol=1e-7
    )


@pytest.mark.slow
def test_x8_native_propagation_smoke(tmp_path):
    spec, speed = fixtures()["x8"]
    result = compare_flights(spec, speed, tmp_path, smoke=True)
    assert result["passed"]


def test_error_metric_uses_quaternion_distance_and_rejects_nonfinite():
    values = np.zeros((3, 13))
    values[:, 9] = 1
    antipodal = values.copy()
    antipodal[:, 6:10] *= -1
    assert all(x == 0 for x in trajectory_errors(antipodal, values).values())
    values[1, 0] = np.nan
    with pytest.raises(FloatingPointError):
        trajectory_errors(antipodal, values)
