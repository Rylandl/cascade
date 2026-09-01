from typing import Any

import jax.numpy as jnp
import pytest

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
