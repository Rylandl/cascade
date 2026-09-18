"""Named domain randomisation over a model: a reviewable spec instead of hand-written updates.

A :class:`Randomisation` lists multiplicative ranges for named leaves of the compiled model
(``"mass"``, ``"inertia"``, ``"surfaces.lift_curve_slope"``, ``"actuators.surface_time_constant"``,
...) and a centre-of-mass shift; :func:`sample_models` draws one factor per world per entry
and returns a batched model ready for ``jax.vmap`` over :func:`cascade.env.reset` and
:func:`cascade.env.step`. Composes with :mod:`cascade.env.family` (randomise a family's
models the same way) and with the environment's latency and sensor settings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.model import AircraftModel, broadcast_model

Range = tuple[float, float]

# Zero scale factors are valid for a coefficient or an area, but not for quantities that
# appear in denominators or must remain positive definite.
_POSITIVE_SCALES = frozenset(
    {
        "mass",
        "inertia",
        "reference_area",
        "reference_chord",
        "reference_span",
        "surfaces.chord",
        "surfaces.stall_angle",
        "surfaces.stall_width",
        "surfaces.separation_time_constant",
        "surfaces.reattachment_time_constant",
        "propellers.diameter",
        "actuators.surface_time_constant",
        "actuators.surface_rate_limit",
        "actuators.propeller_time_constant",
        "actuators.propeller_acceleration_limit",
        "body.stall_angle",
        "body.stall_width",
    }
)
_UNIT_LEAVES = frozenset(
    {"surfaces.body_from_surface", "propellers.direction", "propellers.spin_direction"}
)


class Randomisation(NamedTuple):
    """Multiplicative ranges ``(low, high)`` per named model leaf, a centre-of-mass shift
    range in metres along body x (positive forward). Scale factors must be non-negative
    (strictly positive for mass, inertia, lengths and time constants). ``thrust`` in the
    convenience constructor scales the entire thrust map. Absent entries are left at nominal.
    Inertia's inverse is derived automatically; unit directions/frames cannot be scaled.
    Bounds and names are host-side configuration, not traced JAX values."""

    scales: dict[str, Range]
    center_of_mass_shift_m: Range | None = None


def randomisation(
    *,
    mass: Range | None = None,
    inertia: Range | None = None,
    lift_curve_slope: Range | None = None,
    drag_coefficient_zero: Range | None = None,
    flap_effectiveness: Range | None = None,
    surface_time_constant: Range | None = None,
    propeller_time_constant: Range | None = None,
    thrust: Range | None = None,
    center_of_mass_shift_m: Range | None = None,
) -> Randomisation:
    """The common knobs by name; any other leaf can be given directly in ``scales``."""

    named = {
        "mass": mass,
        "inertia": inertia,
        "surfaces.lift_curve_slope": lift_curve_slope,
        "surfaces.drag_coefficient_zero": drag_coefficient_zero,
        "surfaces.flap_effectiveness": flap_effectiveness,
        "actuators.surface_time_constant": surface_time_constant,
        "actuators.propeller_time_constant": propeller_time_constant,
        "propellers.thrust_map": thrust,
    }
    spec = Randomisation(
        scales={name: value for name, value in named.items() if value is not None},
        center_of_mass_shift_m=center_of_mass_shift_m,
    )
    _validate_spec(spec)
    return spec


def _validate_range(bounds: Range, name: str, *, scale: bool) -> None:
    try:
        values = tuple(bounds)
    except TypeError as error:
        raise ValueError(f"{name} must be a (low, high) pair of finite real numbers") from error
    if len(values) != 2 or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{name} must be a (low, high) pair of finite real numbers")
    low, high = values
    if low > high:
        raise ValueError(f"{name} lower bound must not exceed its upper bound")
    if scale and low < 0:
        raise ValueError(f"{name} scale factors must be non-negative")
    if scale and name in _POSITIVE_SCALES and low <= 0:
        raise ValueError(f"{name} scale factors must be strictly positive")


def _validate_spec(spec: Randomisation) -> None:
    if not isinstance(spec, Randomisation) or not isinstance(spec.scales, Mapping):
        raise ValueError("spec must be a Randomisation with a mapping of model leaves to ranges")
    for name, bounds in spec.scales.items():
        if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
            raise ValueError("randomisation scale names must be non-empty model leaf paths")
        _validate_range(bounds, name, scale=True)
        if name == "inertia_inverse":
            raise ValueError("scale inertia instead of the derived inertia_inverse")
        if name in _UNIT_LEAVES and tuple(bounds) != (1.0, 1.0):
            raise ValueError(f"{name} must remain unit length and cannot be scaled")
    if spec.center_of_mass_shift_m is not None:
        _validate_range(spec.center_of_mass_shift_m, "center_of_mass_shift_m", scale=False)


def _get(tree, path: str):
    node = tree
    for part in path.split("."):
        if part not in getattr(node, "_fields", ()):
            raise ValueError(f"unknown randomisation model leaf {path!r}")
        node = getattr(node, part)
    if not hasattr(node, "shape"):
        raise ValueError(f"randomisation path {path!r} must name an array leaf")
    return node


def _set(tree, path: str, value):
    parts = path.split(".")
    if len(parts) == 1:
        return tree._replace(**{parts[0]: value})
    child = getattr(tree, parts[0])
    return tree._replace(**{parts[0]: _set(child, ".".join(parts[1:]), value)})


def sample_models(
    model: AircraftModel, spec: Randomisation, key: Array, count: int
) -> AircraftModel:
    """``count`` worlds of ``model`` with each named leaf scaled by a uniform draw from its
    range (one factor per world, shared over the leaf's elements) and the centre of mass
    shifted by moving every surface, propeller and body coefficient reference the other way.
    The inertia tensor is still about the new center of mass: this perturbation does not
    infer a mass redistribution or apply a parallel-axis correction. Supply an inertia scale
    separately if required. Input bounds and paths are checked before drawing samples.
    """

    _validate_spec(spec)
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("count must be a positive integer")
    if model.mass.ndim != 0:
        raise ValueError("sample_models requires an unbatched model; vmap over a model family")
    names = sorted(spec.scales)
    for name in names:
        _get(model, name)
    keys = jax.random.split(key, len(names) + 1)
    batched = broadcast_model(model, (count,))
    for name, draw_key in zip(names, keys[:-1], strict=True):
        low, high = spec.scales[name]
        factor = jax.random.uniform(draw_key, (count,), minval=low, maxval=high)
        leaf = _get(batched, name)
        shape = (count,) + (1,) * (leaf.ndim - 1)
        scaled = leaf * factor.reshape(shape)
        batched = _set(batched, name, scaled)
        if name == "inertia":
            batched = batched._replace(inertia_inverse=jnp.linalg.inv(scaled))
    if spec.center_of_mass_shift_m is not None:
        low, high = spec.center_of_mass_shift_m
        shift = jax.random.uniform(keys[-1], (count,), minval=low, maxval=high)
        offset = jnp.stack((-shift, jnp.zeros(count), jnp.zeros(count)), axis=-1)
        batched = batched._replace(
            surfaces=batched.surfaces._replace(
                position=batched.surfaces.position + offset[:, None, :]
            ),
            propellers=batched.propellers._replace(
                position=batched.propellers.position + offset[:, None, :]
            ),
            body=batched.body._replace(reference_position=batched.body.reference_position + offset),
        )
    return batched


__all__ = ["Randomisation", "randomisation", "sample_models"]
