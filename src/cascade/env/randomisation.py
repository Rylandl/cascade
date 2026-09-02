"""Named domain randomisation over a model: a reviewable spec instead of hand-written updates.

A :class:`Randomisation` lists multiplicative ranges for named leaves of the compiled model
(``"mass"``, ``"inertia"``, ``"surfaces.lift_curve_slope"``, ``"actuators.surface_time_constant"``,
...) and a centre-of-mass shift; :func:`sample_models` draws one factor per world per entry
and returns a batched model ready for ``jax.vmap`` over :func:`cascade.env.reset` and
:func:`cascade.env.step`. Composes with :mod:`cascade.env.family` (randomise a family's
models the same way) and with the environment's latency and sensor settings.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.model import AircraftModel, broadcast_model

Range = tuple[float, float]


class Randomisation(NamedTuple):
    """Multiplicative ranges ``(low, high)`` per named model leaf, a centre-of-mass shift
    range in metres along body x (positive forward), and a thrust range applied to the
    propellers' static thrust coefficient. Absent entries are left at nominal."""

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
    return Randomisation(
        scales={name: value for name, value in named.items() if value is not None},
        center_of_mass_shift_m=center_of_mass_shift_m,
    )


def _get(tree, path: str):
    node = tree
    for part in path.split("."):
        node = getattr(node, part)
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
    shifted by moving every surface and propeller the other way."""

    names = sorted(spec.scales)
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
        )
    return batched


__all__ = ["Randomisation", "randomisation", "sample_models"]
