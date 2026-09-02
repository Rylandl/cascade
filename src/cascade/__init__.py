"""Differentiable full-envelope fixed-wing dynamics.

The top level is the core: states, models, specifications, dynamics, integration, trim and
analysis, the canonical state boundary, the stepped plant, and the packaged aircraft. The
layers above live in their own packages: ``cascade.control`` (the cascade, VTOL transition,
automatic tuning), ``cascade.env`` (episodes, tasks, sensors, weather, families),
``cascade.design`` (archetypes), and ``cascade.viz`` (geometry, MJCF, MuJoCo video).
"""

from cascade.actuators import control_from_actuators
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
    BodyModel,
    PropellerModel,
    SurfaceModel,
    broadcast_model,
    validate_model,
)
from cascade.plant import Plant, PlantConfig, PlantSample
from cascade.provenance import model_hash, spec_hash, stamp, write_stamp
from cascade.reference import (
    aerobatic_reference,
    aerobatic_reference_spec,
    skywalker_x8,
    skywalker_x8_panels,
    skywalker_x8_panels_spec,
    skywalker_x8_spec,
    tailsitter_reference,
    tailsitter_reference_spec,
)
from cascade.spec import (
    AircraftSpec,
    BodySpec,
    DragCoefficientSpec,
    LateralCoefficientSpec,
    LongitudinalCoefficientSpec,
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
    "BodyModel",
    "BodySpec",
    "ControlInput",
    "DragCoefficientSpec",
    "DynamicsResult",
    "Environment",
    "LateralCoefficientSpec",
    "LongitudinalCoefficientSpec",
    "Plant",
    "PlantConfig",
    "PlantSample",
    "PropellerModel",
    "PropellerSpec",
    "SpecError",
    "StabilityMode",
    "StepLinearization",
    "StraightFlightCondition",
    "SurfaceModel",
    "SurfaceSpec",
    "TrimResult",
    "aerobatic_reference",
    "aerobatic_reference_spec",
    "aerodynamic_sweep",
    "broadcast_model",
    "continue_trims",
    "control_from_actuators",
    "control_from_array",
    "control_to_array",
    "derivative",
    "equilibrate_internal_state",
    "euler_step",
    "evaluate_dynamics",
    "linearize_step",
    "load_aircraft_spec",
    "model_hash",
    "repeat_control",
    "rk4_step",
    "rollout",
    "save_aircraft_spec",
    "skywalker_x8",
    "skywalker_x8_panels",
    "skywalker_x8_panels_spec",
    "skywalker_x8_spec",
    "spec_hash",
    "stamp",
    "stability_modes",
    "standard_environment",
    "tailsitter_reference",
    "tailsitter_reference_spec",
    "trim_straight_flight",
    "validate_model",
    "write_stamp",
    "zero_control",
    "zero_state",
]
