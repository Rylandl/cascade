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


def test_momentum_bound_rejects_excessive_static_thrust():
    model = aerobatic_reference()
    bound = 0.5 * jnp.pi * jnp.square(model.propellers.zero_thrust_advance_ratio)
    propellers = nested_replace(model.propellers, "thrust_coefficient", 1.1 * bound)
    with pytest.raises(ValueError, match="momentum"):
        validate_model(nested_replace(model, "propellers", propellers))


def test_zero_area_surface_is_valid_and_produces_no_force():
    model = aerobatic_reference()
    surfaces = nested_replace(model.surfaces, "area", model.surfaces.area.at[0].set(0.0))
    model = validate_model(nested_replace(model, "surfaces", surfaces))
    state = zero_state(model, forward_speed=12.0)

    result = evaluate_dynamics(model, state, zero_control(model), standard_environment())

    assert jnp.allclose(result.aerodynamics.force_per_surface[0], 0.0)
    assert jnp.linalg.norm(result.aerodynamics.force_per_surface[1]) > 0.0
