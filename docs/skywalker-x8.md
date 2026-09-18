# The Skywalker X8: two models of one airframe

Cascade ships two specifications of the Skywalker X8 flying wing: a coefficient model assembled
from published sources and a component model fitted to its static aerodynamic response. Comparing
them checks model implementation and approximation; it does not establish flight accuracy. See
[validation status](validation.md) for the limits of the available flight evidence.

## `skywalker_x8` — the whole-aircraft coefficient table

`src/cascade/aircraft/skywalker_x8.toml` (`cascade.skywalker_x8_spec()` /
`cascade.skywalker_x8()`) carries the classical `[body]` polynomial fitted directly to wind-tunnel
and XFLR5 data by Gryte, Hann, Alexandersen, and Johansen ("Aerodynamic modeling of the Skywalker
X8 Fixed-Wing Unmanned Aerial Vehicle", ICUAS 2018), with inertia from the bifilar-pendulum
measurement of Reinhardt, Gryte, and Johansen ("Modeling of the Skywalker X8 Fixed-Wing UAV:
Flight Tests and System Identification", ICUAS 2022) and propulsion and mass constants from the
pyfly project (Bohn et al., ICUAS 2019). The specification records the source and modeling
choices for each group: wind-tunnel static coefficients, XFLR5 rate derivatives, symmetry
assumptions, unmeasured terms set to zero, and a
post-stall extension. Those terms do not all have the same experimental support. Two zero-area
elevon surfaces carry actuator dynamics; the `[body]` block supplies the aerodynamic loads.

## `skywalker_x8_panels` — the component-panel reconstruction

`src/cascade/aircraft/skywalker_x8_panels.toml` (`cascade.skywalker_x8_panels_spec()` /
`cascade.skywalker_x8_panels()`) has no `[body]` table. Its geometry is reconstructed as a
center body panel, two swept panels per side, and two tip winglets, each an ordinary component
surface with its own local flow, body-rate contribution, and attached/separated aerodynamics
(`docs/architecture.md`, "Component aerodynamic model"). Geometry facts are taken from the same
papers plus Winter et al. ("Improved Wind Estimation for a Flying Wing", EUCASS 2019, X8
dimensions): span 2.1 m, reference area 0.75 m² (the exposed wing is roughly 0.4 m², the rest is
the blended center body), mean aerodynamic chord 0.357 m, about 30° of leading-edge/quarter-chord
sweep, and a CG a little under half a meter aft of the nose. Mass, inertia, the `[reference]`
block, and the pusher propeller are copied unchanged from `skywalker_x8.toml`; the propeller's
slipstream weights are zero on every panel (wake interaction is out of scope for this experiment).

| panel (× left/right except center) | area (m²) | chord (m) | span station \|y\| (m) | sweep |
|---|---|---|---|---|
| `center_body` | 0.30 | 0.55 | 0 | none |
| `left_inner` / `right_inner` | 0.14 each | 0.40 | 0.47 | 30° |
| `left_outer` / `right_outer` (elevon) | 0.075 each | 0.22 | 0.85 | 30° |
| `left_winglet` / `right_winglet` | 0.01 each | 0.12 | 1.05 (tip) | vertical fin |

Total area is 0.75 m², matching the reference exactly. The outer panels carry the elevons as pure
flaps (`all_moving_fraction = 0`) with the same mixing as `skywalker_x8.toml`: left = aileron +
elevator, right = elevator − aileron. A swept panel's frame is the unswept frame rotated about
body `z` by the sweep angle Λ — `R_z(+Λ)` on the right wing, `R_z(−Λ)` on the left — which is the
rotation that keeps both wings' local chord axis tilting toward its own tip while leaving `y`
pointing generally to the right for both sides (consistent with how the unswept aerobatic-reference
wings share one frame convention for left and right). The winglets reuse the aerobatic reference's
vertical-tail frame (chord along body `x`, span along body `-z`, "lift" axis along body `y`), so a
winglet's own aerodynamic response is a side force, exactly like a small vertical stabilizer.

Panel locations, orientations, chords, and areas stay fixed during the static fit. The fitted
lift and moment coefficients also influence the rate response when local flow changes, so the
resulting rate derivatives depend on both that geometry and the fitted static aerodynamics.

## The fit

`scripts/fit_x8_panels.py` fits only the panels' *static* coefficients — lift, drag, and pitching
moment about each panel's own aerodynamic center, plus the outer panels' flap terms and the
winglets' lift slope and zero drag — so that `cascade.aerodynamic_sweep` of the panel model matches
the body-block sweep of `skywalker_x8_spec()` over alpha ∈ [−8°, 10°] (7 points), beta ∈ [−8°, 8°]
(5 points), aileron and elevator ∈ {−0.2, 0, 0.2} rad (315 grid points), at 18 m/s and zero rates.
Center, inner, and outer panels each get one independent set of coefficients shared between their
left and right instances (23 free parameters total); rate derivatives are never part of the
objective. It uses `scipy.optimize.least_squares` with a JAX `jacfwd` Jacobian, exactly like
`cascade/analysis/trim.py`, built once from the spec and re-evaluated by `_replace`-ing coefficient
leaves inside a jitted residual — the surface arrays' static topology never changes.

The fit uses a quadratic prior with weight 0.35 in each parameter's own (possibly log) units,
plus bounds: lift-curve slope 1.5 to 2π/rad, flap effectiveness 0.2–1, positive bounded drag
terms, zero-lift and zero-moment terms within ±0.3, and moment-curve slope within ±1.5.
These are modeling constraints, not independently identified section properties. In the stored
fit, the center-body lift slope reaches 2π/rad and its moment-curve slope is about 1.48/rad.
The decomposition can trade lift forces against intrinsic moments, so fitting the whole-aircraft
static table does not uniquely identify the physical properties of each panel.

To reproduce the tables below without changing any aircraft file, run:

```sh
python scripts/check_x8_panels.py --output dist/verification/x8-panels.json
```

The JSON records the grid, model and specification hashes, JAX/runtime versions, dtype setting,
and computed values. The following rounded values were reproduced with Python 3.13.11,
JAX 0.11.1, float32, CPU on macOS arm64. This check evaluates the stored fit; it does not refit
or use flight observations. `scripts/fit_x8_panels.py` performs a new fit and rewrites the panel
TOML, so it is a separate calibration operation.

**Fit residual RMS, candidate − target, over the 315-point grid:**

| C_X | C_Y | C_Z | C_l | C_m | C_n |
|---|---|---|---|---|---|
| 0.0075 | 0.0111 | 0.0293 | 0.0103 | 0.0149 | 0.0017 |

These residuals quantify agreement with the target polynomial on the fitting grid.
`tests/test_x8_panels.py` evaluates a coarser grid with different angle points and checks
wind-axis `C_L` within 0.05 and `C_m` within 0.02
of the coefficient model. Both grids evaluate the same fitted polynomial target; neither is
an independent flight dataset.

## Rate response with fixed geometry and fitted static coefficients

Both models are evaluated with `aerodynamic_sweep(..., angular_velocity_rad_s=...)` at alpha 3°,
18 m/s, with p, q, r each set to 0.2 rad/s one at a time (all controls neutral), and
non-dimensionalized with `p̂ = b p / 2V`, `q̂ = c q / 2V`, `r̂ = b r / 2V` (b = 2.1 m, c = 0.357 m).
"target model" is `skywalker_x8` evaluated the same way, confirming the harness reproduces the
stored XFLR5 derivatives to floating-point precision; "panel model" is the fitted
`skywalker_x8_panels`. Rate derivatives were excluded from the static fitting objective.

| derivative | published (XFLR5) | target model | panel model |
|---|---|---|---|
| C_lp | −0.404 | −0.404 | −0.287 |
| C_lr | 0.0555 | 0.0555 | 0.0303 |
| C_np | 0.00437 | 0.00437 | −0.0319 |
| C_nr | −0.012 | −0.012 | −0.0242 |
| C_Yp | −0.137 | −0.137 | 0.0432 |
| C_Yr | 0.0839 | 0.0839 | 0.0740 |
| C_Lq | 3.87 | 3.87 | 2.784 |
| C_mq | −1.3 | −1.3 | −2.584 |

### Interpreting the comparison

The panel model matches the sign of six of these eight rate derivatives. `C_Yr` is close in
magnitude, while `C_nr` and `C_mq` are about twice the target magnitude. `C_np` and `C_Yp` have
the opposite sign. These discrepancies remain limitations of this panel approximation.

Local flow contains `omega × r`, so panel location, orientation, and the fitted static response
all matter. For example, yaw rate changes both axial flow across the span and side flow at a
winglet's fore/aft station; roll rate introduces side flow through its height relative to the
CG. The comparison alone cannot attribute a discrepancy to winglet height, sweep, lift slope,
or missing unsteady effects. Such explanations would require controlled geometry/parameter
perturbations or additional measurements.

## Against flight

A historical Glassbox replay selected mass, inertia, CG, and inferred-wind adjustments using
the same campaign maneuvers on which its best score was reported. That is a tuned comparison,
not held-out validation. Local artifacts include a result summary and processed X8 windows,
but not the original canonical recording bundle expected by the replay harness or a frozen
held-out protocol. Those artifacts do not reproduce the original replay end to end or establish
that the panel and coefficient models predict flight equally well.

The release therefore makes no quantitative flight-accuracy claim from that experiment. A new
flight validation needs accessible inputs and their hashes, the exact harness and configuration,
a frozen calibration split, and evaluation on separate maneuvers. See
[validation status](validation.md) for the evidence inventory and required follow-up.
