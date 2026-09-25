# Cross-simulator verification with JSBSim

Cascade has an optional, headless comparison harness for **JSBSim 1.3.1**. It starts
with the coefficient Skywalker X8 and also covers the X8 panel model, aerobatic
reference, tailsitter in cruise, and an aerobatic fixture with wing-to-tail downwash.

This checks two implementations of shared aircraft assumptions. It does not make
the aircraft parameters independently measured or establish flight accuracy. The
JSBSim aircraft XML is generated from Cascade's specifications; it is **not** a
comparison against a separately identified community aircraft model.

## Run it

```bash
uv sync --locked --extra validation
JAX_ENABLE_X64=1 uv run --locked --extra validation python -m cascade.validation dist/jsbsim-check
```

The output directory must be new or empty. Exit status is zero only if every selected
acceptance check passes. The default full campaign includes:

- 205 fixed-state cases: five models, 41 cases each. Ordinary cases span three speeds,
  four incidences and three sideslip angles, with nonzero rates, controls and attitude.
  Additional reversed-flow, stale-separation and still-air cases check implementation
  behavior beyond ordinary flight; they supply no physical post-stall qualification.
- Twenty two-second flights: a trimmed hold, elevator doublet, aileron doublet and
  throttle step for each model. Every reference flight runs at 2.5, 1.25 and 0.625 ms.
- Ten JSBSim-generated recordings for two calibration experiments, X8 and aerobatic.
  Each uses two fitting, one validation and two evaluation maneuvers. Mass and uniform
  inertia scale are fitted only on fitting maneuvers.
- A descriptive X8 coefficient-versus-panel rate-derivative comparison using JSBSim
  central differences. Approximation differences are reported separately from the
  implementation acceptance checks.

To run only the X8, add `--models x8`. Available names are `x8`, `x8-panels`,
`aerobatic`, `tailsitter`, and `aerobatic-downwash`. `--smoke` uses 11 point cases per
model and a 0.4-second trim hold; it skips calibration and is only a workflow check.

JSBSim remains an optional dependency. Core simulation, learning and calibration do
not import it. The dedicated CI job installs the built wheel with the validation
extra, runs adapter/regression tests and executes the full campaign outside the checkout.

## What is independent

`cascade.validation._xml` expresses the body coefficients, panel local flow and force
resolution, propeller thrust/torque, slipstream, downwash and separation equations as
native JSBSim XML functions. JSBSim evaluates those functions and propagates its own
rigid-body state. The reference never calls Cascade's force, derivative or integrator
functions. Specifications, commanded inputs and initial conditions are intentionally
shared. Regression tests perturb only the reference model and verify that the
comparison detects the discrepancy.

The reference adapter independently advances actuator and separation states with
forward Euler. JSBSim uses its Euler rigid-body integrators. Cascade uses RK4, so
trajectory error is expected to decrease with the JSBSim timestep. The campaign
checks convergence as well as absolute errors, retaining all three runs. Cascade trim
candidates are checked for balance after an independent JSBSim internal-equilibrium
initialization; this is a trim residual cross-check, not a separate JSBSim trim search.

Frame and environment choices are explicit:

- SI units inside the generated equations; conversions to JSBSim's feet, pounds-force
  and slugs occur at the integration boundary. Products of inertia include JSBSim's
  structural-to-body sign convention. Tests exercise all off-diagonal inertia entries
  and a displaced coefficient reference point.
- Flight comparison arrays use position/velocity NED, body-to-world **xyzw** quaternion,
  and body FRD rates. The calibration-pack adapter converts these to the existing
  **NWU/FLU, wxyz** canonical format. Quaternion errors are sign-invariant angles.
- Constant density 1.225 kg/m³, no wind, no ground contacts, and no planet rotation.
  A spherical planet of radius 10⁹ m approximates Cascade's flat world. Its gravity is
  normalized to 9.80665 m/s² at the initial altitude of 1000 m. States are transformed
  back to the initial tangent frame; the tiny remaining curvature/gravity differences
  are part of the declared comparison error, not adjusted away using results.
- Commands are piecewise constant over declared intervals. Recordings use the flight
  pack's interval-end convention. Actuator/separation histories remain active during
  flights; they are not reset at each observation.

## Acceptance and retained evidence

`protocol.json` is written before execution. Limits are defined in the source protocol,
not inferred from observed errors. Fixed-state tests compare separate aerodynamic and
propulsion loads, total loads, linear/angular acceleration, and separation equilibrium.
For loads, the error is `abs(Cascade - JSBSim) / (1 + abs(JSBSim))` in SI units.

At the finest timestep, each flight must have position RMSE at most 0.03 m, velocity
RMSE at most 0.03 m/s, attitude RMSE at most 0.003 rad and rate RMSE at most 0.02 rad/s.
Halving the final timestep must reduce each error by at least 20%, unless that error
is already below 1% of its absolute limit. Calibration requires solver convergence,
parameter recovery within 2%, and at least tenfold held-out velocity-error improvement.
All these thresholds test software behavior under the frozen synthetic experiment.

The output contains:

- `results.json` and `summary.md`: every model's decision, errors, runtime information,
  specification hashes, generated-artifact hashes and package-source hashes.
- Per-model `aircraft.toml`, generated JSBSim XML and planet definitions, `points.json`
  with actual loads/input states, and per-maneuver `trajectories.npz` plus `errors.json`.
- X8/aerobatic calibration protocols, generator specifications, frozen recording packs,
  fitted specifications, solver status and all held-out scores.
- `x8-approximation/rate-comparison.json` with coefficient and panel rate derivatives.

Failures remain in the output. Existing runs are never overwritten. Source fingerprints
identify the exact implementation even when running an uncommitted checkout; a Git HEAD
stamp alone does not identify uncommitted changes. These checksums detect changes, not
authenticate the author. CI retains full output for 30 days; archive it for longer use.

## Executed local campaign, 2026-09-25

The [source-fingerprinted result record](verification/jsbsim-2026-09-25.json) preserves
the full campaign's summary on macOS arm64, Python 3.13, JAX 0.11.1 and JSBSim 1.3.1.
All 205 point cases, 20 flights, convergence checks and both calibration experiments
passed. The complete local artifact tree is `dist/jsbsim-validation/final/`; the JSON
record binds its result file and exact Python sources by SHA-256.

Worst RMSE across each model's four flights at the finest reference timestep:

| Model | Position (m) | Velocity (m/s) | Attitude (rad) | Rate (rad/s) |
|---|---:|---:|---:|---:|
| X8 coefficients | 0.000248 | 0.000737 | 0.0000345 | 0.000278 |
| X8 panels | 0.000146 | 0.000425 | 0.0000439 | 0.000296 |
| Aerobatic | 0.000170 | 0.000157 | 0.0000427 | 0.000413 |
| Tailsitter, cruise | 0.000775 | 0.001421 | 0.000207 | 0.002730 |
| Aerobatic, downwash | 0.000602 | 0.000751 | 0.0000517 | 0.000413 |

The maximum scaled load discrepancy was below 7×10⁻¹³. X8 mass/inertia recovery
errors were 0.0020%/0.1923%; aerobatic errors were 0.0011%/0.2349%. Every held-out
velocity error improved by more than 500 times over the nominal model. Those small
errors reflect a controlled, noiseless, shared-model experiment with known environment,
actuation and initial state, not corresponding real-aircraft accuracy.

## Refinement found by this harness

The first run found that `equilibrate_internal_state` initialized separation using
geometric incidence, while `aerodynamics` subsequently used downwash-adjusted incidence.
For the downwash fixture, the maximum initial separation discrepancy was about 0.196.
A reported 12 m/s trim had an independent pitch/angular acceleration norm of about
0.0768 rad/s² when initialized at the actual separation equilibrium.

Initialization now solves the coupled upstream-lift/separation fixed point using
48 damped iterations. A zero downwash map preserves the previous geometric result.
Regression tests cover positive/negative incidence, a sustained trimmed rollout,
batching and differentiation. Very strongly coupled cyclic downwash maps may still
fail to converge; check the resulting separation derivative for such custom models.

The packaged X8 coefficients were not retuned to reduce a simulator comparison error.
Its panel approximation's existing rate-derivative discrepancies remain model limits.
Matching the same equations in JSBSim cannot establish which approximation best
predicts a real X8. That next step needs the frozen, measured-flight protocol described
in [validation](validation.md).

## References and scope

- [JSBSim aerodynamic functions and force axes](https://jsbsim-team.github.io/jsbsim-reference-manual/user/concepts/forces-and-moments/)
- [JSBSim Python execution interface](https://jsbsim-team.github.io/jsbsim/python/FGFDMExec.html)
- [Pinned inertia conversion implementation](https://github.com/JSBSim-Team/jsbsim/blob/v1.3.1/src/models/FGMassBalance.cpp)
- [Pinned planet/gravity implementation](https://github.com/JSBSim-Team/jsbsim/blob/v1.3.1/src/models/FGInertial.cpp)

JSBSim is an external LGPL dependency; no bundled JSBSim aircraft assets are copied.
The generated XML and synthetic recordings use this repository's specifications.
Wind, ground handling, long-duration flight, real sensor errors, tailsitter transitions,
and physical stall/spin accuracy are outside this campaign's evidence.
