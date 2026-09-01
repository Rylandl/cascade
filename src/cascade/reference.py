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


def skywalker_x8_panels_spec() -> AircraftSpec:
    """Load the packaged component-panel Skywalker X8 specification.

    This is a from-geometry component model (no ``[body]`` table): a center body panel, swept
    inboard and outboard panels per side with elevons on the outer panels, and tip winglets acting
    as vertical stabilizers. Its static coefficients are fitted by ``scripts/fit_x8_panels.py`` to
    reproduce ``skywalker_x8_spec()``'s published static polynomial; its rate derivatives are not
    fitted and are geometry predictions. See ``docs/skywalker-x8.md``.
    """

    return _packaged_spec("skywalker_x8_panels.toml")


def skywalker_x8_panels() -> AircraftModel:
    """Compile the packaged component-panel Skywalker X8 specification into a numerical model."""

    return skywalker_x8_panels_spec().to_model()


def tailsitter_reference_spec() -> AircraftSpec:
    """Load the packaged illustrative twin-motor flying-wing tailsitter.

    An indoor-class fixture whose propwash covers its elevons, so it has pitch and roll authority
    at zero airspeed and can hover, transition, and cruise. Plausible numbers, not an identified
    vehicle.
    """

    return _packaged_spec("tailsitter_reference.toml")


def tailsitter_reference() -> AircraftModel:
    """Compile the illustrative tailsitter into a numerical model."""

    return tailsitter_reference_spec().to_model()
