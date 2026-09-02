"""Families of airframes: sampled designs, trimmed, tuned, and stacked so that one vmap flies
them all, with the design parameters kept apart from what an episode exposes.

A :class:`Family` is the Glassbox-facing object for "control diverse airframes with no a
priori information": ``models``, ``tasks``, ``references``, and ``controllers`` are stacked
pytrees with a leading family axis, ready for ``jax.vmap`` over :func:`cascade.env.reset`,
:func:`cascade.env.step`, and :func:`cascade.env.rollout_policy`; ``designs`` and ``reports``
are the hidden truth, kept for analysis and never fed to a policy.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.control.autotune import tune_cascade
from cascade.control.loops import CascadeController
from cascade.design.archetypes import (
    Design,
    DesignReport,
    design_spec,
    sample_design,
    validate_design,
)
from cascade.env.tasks import ReferenceFlight, TrackingTask, tracking_task, trimmed_reference
from cascade.model import AircraftModel


class Family(NamedTuple):
    models: AircraftModel
    tasks: TrackingTask
    references: ReferenceFlight
    controllers: CascadeController
    cruise_speeds_m_s: Array
    designs: tuple[Design, ...]
    reports: tuple[DesignReport, ...]

    def __len__(self) -> int:
        return len(self.designs)


def stack_pytrees(items):
    """Stack a list of pytrees along a new leading axis. Python scalars (a controller's update
    periods, say) become arrays too, so every leaf carries the family axis and ``jax.vmap``
    maps the whole tree; the consumers all accept 0-d arrays where they took ints."""

    return jax.tree.map(lambda *leaves: jnp.stack([jnp.asarray(leaf) for leaf in leaves]), *items)


def sample_family(
    archetype: type[Design],
    key: Array,
    count: int,
    *,
    altitude_m: float = 50.0,
    ranges=None,
    max_attempts: int | None = None,
) -> Family:
    """Draw ``count`` valid designs, trim each at its cruise, tune a cascade for each, and
    stack everything. Invalid draws are skipped (about one in ten); ``max_attempts`` bounds
    the search (default four draws per requested design)."""

    max_attempts = 4 * count if max_attempts is None else max_attempts
    designs, reports, models, tasks, references, controllers, speeds = [], [], [], [], [], [], []
    for attempt_key in jax.random.split(key, max_attempts):
        if len(designs) >= count:
            break
        design = sample_design(archetype, attempt_key, ranges)
        report = validate_design(design, altitude_m=altitude_m)
        if not report.valid:
            continue
        spec = design_spec(design)
        model = spec.to_model()
        task = tracking_task(report.cruise_speed_m_s, altitude_m, 0.0)
        reference = trimmed_reference(model, task)
        controller, _ = tune_cascade(
            spec, report.cruise_speed_m_s, model=model, altitude_m=altitude_m
        )
        designs.append(design)
        reports.append(report)
        models.append(model)
        tasks.append(task)
        references.append(reference)
        controllers.append(controller)
        speeds.append(report.cruise_speed_m_s)
    if len(designs) < count:
        raise ValueError(
            f"only {len(designs)} of {count} designs were valid in {max_attempts} draws"
        )
    return Family(
        models=stack_pytrees(models),
        tasks=stack_pytrees(tasks),
        references=stack_pytrees(references),
        controllers=stack_pytrees(controllers),
        cruise_speeds_m_s=jnp.asarray(np.asarray(speeds, dtype=float)),
        designs=tuple(designs),
        reports=tuple(reports),
    )


def family_member(family: Family, index: int):
    """One member's (model, task, reference, controller) as unbatched pytrees."""

    def take(tree):
        return jax.tree.map(lambda leaf: leaf[index], tree)

    return (
        take(family.models),
        take(family.tasks),
        take(family.references),
        take(family.controllers),
    )


__all__ = ["Family", "family_member", "sample_family", "stack_pytrees"]
