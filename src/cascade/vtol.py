"""Compatibility shim: :mod:`cascade.control.vtol` moved into the control package."""

from cascade.control.tuned import tailsitter_reference_controller  # noqa: F401
from cascade.control.vtol import *  # noqa: F403
