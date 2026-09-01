from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from cascade.reference import aerobatic_reference_spec
from cascade.spec import SpecError, load_aircraft_spec, save_aircraft_spec


def test_packaged_reference_spec_preserves_names_and_compiles():
    spec = aerobatic_reference_spec()
    model = spec.to_model()

    assert spec.name == "cascade-aerobatic-reference"
    assert spec.control_channels == ("aileron", "elevator", "rudder")
    assert tuple(surface.name for surface in spec.surfaces) == (
        "left_wing",
        "right_wing",
        "horizontal_tail",
        "vertical_tail",
    )
    assert tuple(propeller.name for propeller in spec.propellers) == ("nose_propeller",)
    assert model.n_surfaces == 4
    assert model.n_propellers == 1


def test_aircraft_spec_toml_round_trip(tmp_path):
    original = aerobatic_reference_spec()
    path = tmp_path / "aircraft.toml"

    save_aircraft_spec(original, path)
    loaded = load_aircraft_spec(path)

    assert loaded == original
    assert jax.tree.all(jax.tree.map(jnp.array_equal, loaded.to_model(), original.to_model()))


def test_aircraft_spec_rejects_duplicate_names():
    spec = aerobatic_reference_spec()
    duplicate_channels = replace(spec, control_channels=("aileron", "aileron", "rudder"))

    with pytest.raises(SpecError, match="control channel names must be unique"):
        duplicate_channels.validate()
