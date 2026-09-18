# Aircraft calibration from flight recordings

`cascade.calibration` fits a small, explicitly chosen set of aircraft parameters to a
[frozen flight-recording pack](flight-data.md). It uses differentiable open-loop replay,
bounded least squares, and the same timing and environment conventions as the replay evaluator.
The result is a normal aircraft TOML specification with a separate calibration record.

The included demonstration uses synthetic recordings with known parameter changes. Recovering
those changes tests the fitting software; it does not establish measured-flight accuracy.

## Run a complete example

```bash
uv run python examples/calibrate_aircraft.py --output dist/synthetic-calibration
```

The directory must be new or empty. The example generates two fitting maneuvers, one validation
maneuver and two evaluation maneuvers. It estimates mass and a uniform inertia multiplier from
the fitting recordings, saves the resulting aircraft, then reports validation and evaluation
errors. The known generator is used only to generate recordings and report recovery error;
it is not supplied to the optimizer.

Outputs include the frozen `pack/`, `config.json`, the saved `calibration/`, full replay scores
in `validation.json` and `evaluation.json`, and the synthetic parameter comparison in
`recovery.json`. The original nominal specification remains in the pack.

## Fit an existing pack

Declare the parameters, bounds, residual scales and numerical settings before fitting:

```python
from cascade.calibration import (
    CalibrationConfig,
    FitParameter,
    ResidualScales,
    evaluate_calibration,
    fit_flight_pack,
)

config = CalibrationConfig(
    parameters=(
        FitParameter("mass_kg", 0.9, 1.5),
        FitParameter("inertia_scale", 0.75, 1.3),
    ),
    scales=ResidualScales(
        position_m=1.0,
        velocity_m_s=1.0,
        attitude_rad=0.1,
        rate_rad_s=0.1,
    ),
    substeps=4,
    warmup_steps=0,
    max_nfev=100,
)
artifact = fit_flight_pack("flight-pack/manifest.json", config, output="calibration")
print(artifact.report["optimizer"])
result = evaluate_calibration(artifact, "flight-pack/manifest.json", split="evaluation")
```

Choose bounds for the nominal aircraft in your pack; the nominal values must lie inside them.
The configuration is serializable with `config.to_dict()` and
`CalibrationConfig.from_dict(...)`. The CLI accepts that same JSON representation:

```bash
python -m cascade.calibration fit flight-pack/manifest.json calibration --config config.json
python -m cascade.calibration evaluate flight-pack/manifest.json calibration \
  --split evaluation --output evaluation.json
```

The fitting command exits with status zero when the optimizer converges. A finite result that
exhausts its budget is saved with `success: false`, and the fitting command exits with status
three. Input errors or nonfinite optimization fail with status two. Evaluation preserves the
saved optimizer outcome; it never relabels a failed fit as converged.

## Parameters and fitting objective

`FitParameter(path, lower, upper)` uses absolute values in the aircraft specification's units.
The exception is `inertia_scale`, a positive multiplier of the complete nominal inertia tensor.
The cached inverse is updated consistently. No individual tensor entries are fitted.

Supported paths include:

- `mass_kg` and `inertia_scale`.
- Named surface scalars such as `surfaces.left_wing.lift_curve_slope_rad`,
  `surfaces.left_wing.drag_coefficient_zero`, and
  `surfaces.left_wing.actuator_time_constant_s`.
- Whole-aircraft polynomial coefficients such as `body.pitch.alpha_rad` and `body.drag.zero`,
  when the nominal specification has a body coefficient model.

The parameter layer allows selected aerodynamic coefficients, stall/separation settings and
time constants. It rejects unsupported fields, duplicate parameters, invalid physical bounds,
and bounds that collapse in the configured JAX precision. Geometry, topology, channel maps,
actuator limits and propulsion maps remain fixed. Surface names refer to the specification's
named components, not array indices.

`Parameterization(spec, parameters)` exposes the nominal values and bounds. Its `apply_model`
method composes with JAX differentiation and compilation; it assumes the caller supplies valid
in-bounds values. Its host-side `apply_spec` method checks bounds and returns a validated
specification. `fit_records(spec, records, config)` is the lower-level fitter for explicitly
provided fitting records; the pack workflow additionally enforces split membership.

The optimizer uses normalized parameter coordinates and JAX automatic differentiation. Residuals include
position, velocity, attitude and body angular rate, divided by the declared physical scales.
Attitude residuals treat antipodal quaternions as the same orientation. Each recording has
equal weight, and each is normalized by its retained sample count. Scales set the relative
importance of different physical quantities; they are not automatically estimated noise levels.

Replay starts at the first recorded rigid-body state, equilibrates unknown internal states
under the first active command/environment, then integrates the whole maneuver without resets
to later observations. `warmup_steps` excludes that many initial replay intervals from the
fitting residual only. The prefix is still simulated. Validation and evaluation report the
complete replay after the initial sample, including any fitting warm-up intervals.

The pack loader verifies the entire pack's integrity, but only fitting recordings reach the
optimizer. Validation and evaluation do not select parameters, bounds, stopping criteria or a
best model automatically. Repeated human inspection of evaluation scores changes their status
as a blind holdout; declare any such selection in your experiment protocol.

## Inspect and reuse a calibration

```python
from cascade.calibration import load_calibration

artifact = load_calibration("calibration")
model = artifact.spec.to_model()
print(artifact.report["initial_data_loss"], artifact.report["final_data_loss"])
print(artifact.report["optimizer"])
```

The report retains the solver status, parameter values, bound hits and singular values/rank
of the scaled data-residual Jacobian. Rank deficiency indicates locally indistinguishable
parameter directions under the chosen recordings, bounds and residual scales. Full local rank
does not establish a unique physical explanation, confidence intervals, or predictive accuracy
outside those maneuvers. Wind, density, timing, actuation and force coefficients can compensate
for one another; begin with a small parameter set and recordings that excite it. Initial/final
reported losses replay the nominal/saved specifications; the sensitivity calculation uses the
differentiable parameter transform, whose inertia arithmetic can differ at floating-point roundoff.

The artifact directory contains `calibrated.toml` and versioned `metadata.json`. Its hashes bind
the exact pack, nominal and fitted specifications, configuration, fitting-record identities,
report and fitting-time runtime/source provenance, including JAX, NumPy and SciPy versions.
Saving a previously computed fit retains its original provenance. Loading validates those contracts.
`evaluate_calibration`
also checks the supplied pack and defaults to the integration substeps used during fitting.
Artifacts retain optimizer failures honestly and do not provide statistical uncertainty estimates.

Use correctly synchronized and framed recordings, document wind/density assumptions and unknown
initial actuator states, and check timestep convergence for the chosen sampling rate. Float64
can help with sensitive fits (`JAX_ENABLE_X64=1` before starting Python), but it does not fix an
unidentifiable parameter set or incorrect observations.
