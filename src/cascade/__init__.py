"""Differentiable full-envelope fixed-wing dynamics."""

from cascade.analysis import (
    AerodynamicSweep,
    StabilityMode,
    StepLinearization,
    StraightFlightCondition,
    TrimResult,
    aerodynamic_sweep,
    continue_trims,
    linearize_step,
    stability_modes,
    trim_straight_flight,
)
from cascade.dynamics import DynamicsResult, derivative, evaluate_dynamics
from cascade.initialization import (
    control_from_array,
    control_to_array,
    equilibrate_internal_state,
    standard_environment,
    zero_control,
    zero_state,
)
from cascade.integration import euler_step, repeat_control, rk4_step, rollout
from cascade.model import (
    ActuatorModel,
    AircraftModel,
    PropellerModel,
    SurfaceModel,
    broadcast_model,
    validate_model,
)
from cascade.reference import aerobatic_reference, aerobatic_reference_spec
from cascade.spec import (
    AircraftSpec,
    PropellerSpec,
    SpecError,
    SurfaceSpec,
    load_aircraft_spec,
    save_aircraft_spec,
)
from cascade.state import AircraftState, ControlInput, Environment

__all__ = [
    "ActuatorModel",
    "AerodynamicSweep",
    "AircraftModel",
    "AircraftSpec",
    "AircraftState",
    "ControlInput",
    "DynamicsResult",
    "Environment",
    "PropellerModel",
    "PropellerSpec",
    "SpecError",
    "StabilityMode",
    "StepLinearization",
    "StraightFlightCondition",
    "SurfaceModel",
    "SurfaceSpec",
    "TrimResult",
    "aerodynamic_sweep",
    "aerobatic_reference",
    "aerobatic_reference_spec",
    "broadcast_model",
    "control_from_array",
    "control_to_array",
    "continue_trims",
    "derivative",
    "equilibrate_internal_state",
    "euler_step",
    "evaluate_dynamics",
    "load_aircraft_spec",
    "linearize_step",
    "repeat_control",
    "rk4_step",
    "rollout",
    "save_aircraft_spec",
    "standard_environment",
    "stability_modes",
    "trim_straight_flight",
    "validate_model",
    "zero_control",
    "zero_state",
]
