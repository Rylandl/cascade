from typing import Any

import jax.numpy as jnp
import pytest

from cascade.dynamics import evaluate_dynamics
from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.model import validate_model
from cascade.reference import aerobatic_reference


def nested_replace(value: Any, field: str, replacement: Any):
    return value._replace(**{field: replacement})


def test_reference_model_is_valid():
    model = aerobatic_reference()
    assert model.n_surfaces == 4
    assert model.n_propellers == 1
    assert model.n_control_channels == 3


def test_invalid_surface_frame_is_rejected():
    model = aerobatic_reference()
    frames = model.surfaces.body_from_surface.at[0, 0, 0].set(2.0)
    surfaces = nested_replace(model.surfaces, "body_from_surface", frames)
    with pytest.raises(ValueError, match="orthonormal"):
        validate_model(nested_replace(model, "surfaces", surfaces))


def test_invalid_slipstream_shape_is_rejected():
    model = aerobatic_reference()
    propellers = nested_replace(model.propellers, "slipstream_map", jnp.zeros((1, 3)))
    with pytest.raises(ValueError, match="slipstream_map"):
        validate_model(nested_replace(model, "propellers", propellers))


def test_momentum_bound_rejects_a_thrust_map_that_windmills_too_hard():
    model = aerobatic_reference()
    # A steep negative airspeed slope makes the momentum-theory discriminant negative at
    # moderate airspeed, which would give a complex induced velocity.
    steep = model.propellers.thrust_map.at[0, 0, 1].set(-5.0)
    propellers = nested_replace(model.propellers, "thrust_map", steep)
    with pytest.raises(ValueError, match="momentum"):
        validate_model(nested_replace(model, "propellers", propellers))


def test_linear_thrust_law_satisfies_the_momentum_bound_up_to_its_limit():
    model = aerobatic_reference()
    thrust_map = model.propellers.thrust_map
    static, slope = float(thrust_map[0, 1, 0]), float(thrust_map[0, 0, 1])
    zero_thrust_advance_ratio = -static / slope
    bound = 0.5 * jnp.pi * zero_thrust_advance_ratio**2
    # C_T0 <= (pi / 2) J_0^2 is the closed-form condition for this special case.
    def scaled_model(scale):
        thrust = thrust_map.at[0, 1, 0].set(scale * bound)
        thrust = thrust.at[0, 0, 1].set(-scale * bound / zero_thrust_advance_ratio)
        propellers = nested_replace(model.propellers, "thrust_map", thrust)
        return nested_replace(model, "propellers", propellers)

    validate_model(scaled_model(0.99))
    with pytest.raises(ValueError, match="momentum"):
        validate_model(scaled_model(1.1))


def test_zero_area_surface_is_valid_and_produces_no_force():
    model = aerobatic_reference()
    surfaces = nested_replace(model.surfaces, "area", model.surfaces.area.at[0].set(0.0))
    model = validate_model(nested_replace(model, "surfaces", surfaces))
    state = zero_state(model, forward_speed=12.0)

    result = evaluate_dynamics(model, state, zero_control(model), standard_environment())

    assert jnp.allclose(result.aerodynamics.force_per_surface[0], 0.0)
    assert jnp.linalg.norm(result.aerodynamics.force_per_surface[1]) > 0.0
