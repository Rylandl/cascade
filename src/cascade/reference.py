from __future__ import annotations

from functools import cache
from importlib.resources import as_file, files

from cascade.model import AircraftModel
from cascade.spec import AircraftSpec, load_aircraft_spec


@cache
def _packaged_spec(filename: str) -> AircraftSpec:
    resource = files("cascade.aircraft").joinpath(filename)
    with as_file(resource) as path:
        return load_aircraft_spec(path)


def aerobatic_reference_spec() -> AircraftSpec:
    """Load the packaged illustrative aircraft specification."""

    return _packaged_spec("aerobatic_reference.toml")


def aerobatic_reference() -> AircraftModel:
    """Compile the illustrative reference aircraft into a numerical model.

    Its parameters are plausible software fixtures, not measurements from a real aircraft.
    """

    return aerobatic_reference_spec().to_model()


def skywalker_x8_spec() -> AircraftSpec:
    """Load the packaged Skywalker X8 flying wing built from the published NTNU model.

    Static aerodynamics are the wind-tunnel column of Gryte et al. (ICUAS 2018), rate
    derivatives their XFLR5 column, inertia the bifilar-pendulum tensor of Reinhardt et al.
    (ICUAS 2022), and propulsion a fit to the NTNU exit-velocity thrust law. The specification
    records every choice between conflicting published values in its comments.
    """

    return _packaged_spec("skywalker_x8.toml")


def skywalker_x8() -> AircraftModel:
    """Compile the packaged Skywalker X8 specification into a numerical model."""

    return skywalker_x8_spec().to_model()
