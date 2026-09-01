from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files

from cascade.model import AircraftModel
from cascade.spec import AircraftSpec, load_aircraft_spec


@lru_cache(maxsize=1)
def aerobatic_reference_spec() -> AircraftSpec:
    """Load the packaged illustrative aircraft specification."""

    resource = files("cascade.aircraft").joinpath("aerobatic_reference.toml")
    with as_file(resource) as path:
        return load_aircraft_spec(path)


def aerobatic_reference() -> AircraftModel:
    """Compile the illustrative reference aircraft into a numerical model.

    Its parameters are plausible software fixtures, not measurements from a real aircraft.
    """

    return aerobatic_reference_spec().to_model()
