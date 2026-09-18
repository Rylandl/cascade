"""Bounded aircraft calibration from frozen flight-recording packs.

Fitting is an experimental research workflow. Numerical convergence and local
sensitivity diagnostics do not establish unique physical parameters or flight accuracy.
"""

from .artifacts import (
    CALIBRATION_SCHEMA,
    CalibrationArtifact,
    load_calibration,
    save_calibration,
)
from .fitting import CalibrationConfig, FitResult, ResidualScales, fit_records
from .parameters import FitParameter, Parameterization
from .workflow import evaluate_calibration, fit_flight_pack

__all__ = [
    "CALIBRATION_SCHEMA",
    "CalibrationArtifact",
    "CalibrationConfig",
    "FitParameter",
    "FitResult",
    "Parameterization",
    "ResidualScales",
    "evaluate_calibration",
    "fit_flight_pack",
    "fit_records",
    "load_calibration",
    "save_calibration",
]
