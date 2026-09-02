"""Visualisation: geometry from the spec, OBJ and MJCF export, and MuJoCo video.
See ``docs/rendering.md``."""

from cascade.viz.geometry import *  # noqa: F403
from cascade.viz.geometry import __all__ as _geometry_all

__all__ = [*_geometry_all, "Scene", "render_trajectory"]  # noqa: F405


def __getattr__(name):
    # MuJoCo is optional: the renderer is imported on first use.
    if name in ("Scene", "render_trajectory"):
        from cascade.viz import render

        return getattr(render, name)
    raise AttributeError(name)
