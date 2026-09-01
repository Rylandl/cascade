"""Trim, envelope, sweep, and local-linear analysis tools."""

from cascade.analysis.coordinates import (
    control_retract,
    state_difference,
    state_retract,
    tangent_state_size,
)
from cascade.analysis.linearize import (
    StabilityMode,
    StepLinearization,
    linearize_step,
    stability_modes,
)
from cascade.analysis.sweep import AerodynamicSweep, aerodynamic_sweep, velocity_from_air_angles
from cascade.analysis.trim import (
    StraightFlightCondition,
    TrimResult,
    continue_trims,
    trim_straight_flight,
)

__all__ = [
    "AerodynamicSweep",
    "StabilityMode",
    "StraightFlightCondition",
    "StepLinearization",
    "TrimResult",
    "aerodynamic_sweep",
    "continue_trims",
    "control_retract",
    "linearize_step",
    "stability_modes",
    "state_difference",
    "state_retract",
    "tangent_state_size",
    "trim_straight_flight",
    "velocity_from_air_angles",
]
