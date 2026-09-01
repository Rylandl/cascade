# The Skywalker X8: two models of one airframe

Cascade ships two specifications of the Skywalker X8 flying wing. They are not alternatives for
the same purpose: one is a published reference, the other is an experiment in whether geometry
alone reproduces what that reference measured.

## `skywalker_x8` — the whole-aircraft coefficient table

`src/cascade/aircraft/skywalker_x8.toml` (`cascade.skywalker_x8_spec()` /
`cascade.skywalker_x8()`) carries the classical `[body]` polynomial fitted directly to wind-tunnel
and XFLR5 data by Gryte, Hann, Alexandersen, and Johansen ("Aerodynamic modeling of the Skywalker
X8 Fixed-Wing Unmanned Aerial Vehicle", ICUAS 2018), with inertia from the bifilar-pendulum
measurement of Reinhardt, Gryte, and Johansen ("Modeling of the Skywalker X8 Fixed-Wing UAV:
Flight Tests and System Identification", ICUAS 2022) and propulsion and mass constants from the
pyfly project (Bohn et al., ICUAS 2019). Every static and rate coefficient — `C_L0`, `C_Lalpha`,
`C_mq`, `C_nr`, and the rest — is a directly identified number from the published papers. It has
two zero-area elevon actuator surfaces and no other aerodynamic geometry; the `[body]` block does
all the work. This is Cascade's physically identified reference.

## `skywalker_x8_panels` — the component-panel reconstruction

`src/cascade/aircraft/skywalker_x8_panels.toml` (`cascade.skywalker_x8_panels_spec()` /
`cascade.skywalker_x8_panels()`) has no `[body]` table. It is a from-geometry reconstruction: a
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

Positions and every quantity that could affect a rate derivative — panel location, orientation,
chord, area — are geometry, chosen once from the facts above and never touched by the fit.

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

An unregularized fit runs away: with 23 parameters and only 6 output directions per grid point, a
panel's own lift0/lift-slope feed `C_m` both directly (its own moment coefficients) and indirectly
(moment-arm times force at its position), so several panels trade off against each other at
near-equal residual. A fully unbounded solve found a solution with a 10.4/rad lift-curve slope, a
lift-coefficient-zero of 6.8, and a flap effectiveness of 2.16 — all outside any physical airfoil's
range — while fitting the data about as well as a sane solution. Bounding every parameter to a
conventional range did not fix this: at three different box widths, tried in turn, most parameters
sat exactly on an edge rather than settling inside. The fit therefore adds a small quadratic prior
pulling every parameter toward a sensible initial guess (weight 0.35, in each parameter's own —
possibly log — units) on top of a generous safety-net box (lift-curve slope 1.5 to 2π/rad, thin-airfoil
theory's own ceiling; flap effectiveness 0.2–1; drag and induced-drag terms positive and modest;
zero-lift and zero-moment terms within ±0.3; moment-curve slope within ±1.5). This picks the
minimum-norm point among the near-equal-residual solutions instead of an arbitrary corner. Every
panel's fitted coefficients now land inside an ordinary small-UAV wing-section range *except* the
center-body panel, whose lift-curve slope still saturates at 2π/rad and whose intrinsic
moment-curve slope reaches 1.48/rad — both reported honestly below rather than hidden, because a
single "center body" panel is being asked to carry more of the whole-aircraft `C_m(alpha)` curve
than one clean airfoil section normally would with only three independent panel groups to work
with.

**Fit residual RMS, candidate − target, over the 315-point grid:**

| C_X | C_Y | C_Z | C_l | C_m | C_n |
|---|---|---|---|---|---|
| 0.0075 | 0.0111 | 0.0293 | 0.0103 | 0.0149 | 0.0017 |

These are small next to the coefficients' own scale (`C_Z` and `C_l` range over roughly ±1 and
±0.15 across the grid), so the panel geometry, with plausible per-panel airfoil coefficients, can
reproduce the published static polynomial closely. `tests/test_x8_panels.py` checks a coarser,
independent grid and finds wind-axis `C_L` within 0.05 and `C_m` within 0.02 of the published
values (actual max error on that grid: `C_L` 0.028, `C_m` 0.015).

## The test: rate derivatives from geometry alone

Both models are evaluated with `aerodynamic_sweep(..., angular_velocity_rad_s=...)` at alpha 3°,
18 m/s, with p, q, r each set to 0.2 rad/s one at a time (all controls neutral), and
non-dimensionalized with `p̂ = b p / 2V`, `q̂ = c q / 2V`, `r̂ = b r / 2V` (b = 2.1 m, c = 0.357 m).
"target model" is `skywalker_x8` evaluated the same way, confirming the harness reproduces the
published numbers exactly; "panel model" is the fitted `skywalker_x8_panels`, a prediction from
geometry that was never fitted to any of these numbers.

| derivative | published (XFLR5) | target model | panel model |
|---|---|---|---|
| C_lp | −0.404 | −0.404 | −0.287 |
| C_lr | 0.0555 | 0.0555 | 0.0303 |
| C_np | 0.00437 | 0.00437 | −0.0319 |
| C_nr | −0.012 | −0.012 | −0.0242 |
| C_Yp | −0.137 | −0.137 | 0.0432 |
| C_Yr | 0.0839 | 0.0839 | 0.0740 |
| C_Lq | 3.87 | 3.87 | 2.785 |
| C_mq | −1.3 | −1.3 | −2.584 |

### Reading the agreements

**C_Yr (side force due to yaw rate)** is the closest match (0.074 vs 0.084, 88%). Yaw rate gives
the winglets, sitting at `|y| = 1.05 m`, a differential local sideslip through `omega × r`; a
vertical fin turns local sideslip directly into side force along its own lift axis, which is
exactly the mechanism this geometry captures well: two winglets, correctly positioned far out on
the span, doing what a vertical stabilizer does.

**C_lp, C_Lq, and C_lr** get the right sign and a sizeable fraction of the published magnitude (71%,
72%, and 55% respectively) with no fitting of any rate-affecting quantity. C_lr is classically a
sweep effect — yaw rate gives the advancing, swept-aft wing more forward velocity than the
retreating wing, and a swept panel converts that into a lift asymmetry — so recovering its sign and
half its magnitude from a 30° sweep angle and nothing else is a real, if partial, confirmation that
the swept-panel frame construction captures simple sweep theory. C_Lq did not need a tail to
appear either: with no discrete horizontal stabilizer, it comes entirely from the wing panels' own
aft-of-CG moment arms seeing a pitch-rate-induced local angle-of-attack change through `omega × r`
— the same mechanism a tailed aircraft's tail uses, just with a shorter arm and more area. Reaching
72% of the published value from geometry alone, with no tail, is the paper's headline positive
result.

**C_nr is the right sign but about 2× too large** (−0.024 vs −0.012). Yaw rate gives the winglets
differential sideslip, producing a differential side force and hence a restoring yaw moment —
qualitatively the winglet does exactly what it is supposed to do, and doing so at roughly double
strength is a much smaller miss than getting the sign wrong. It suggests either the winglet area
(0.01 m² each, a deliberately modest "large winglet" relative to the 0.75 m² reference) or its
fitted lift-curve slope (4.62/rad, comfortably inside the fit's safety-net box) gives more
side-force authority per radian of local sideslip than the real, three-dimensional, small-aspect-
ratio winglet achieves — plausible, since a real small fin's effective lift slope typically sits
well below what a strip-theory panel is allowed to reach.

### Reading the disagreements

**C_np and C_Yp have the wrong sign.** Both are driven, physically, by how far the winglets sit
above the roll axis: a real vertical fin mounted with real height above the CG sweeps sideways as
the aircraft rolls, producing a restoring side force and a coupled yaw moment. In this geometry the
winglets are positioned mostly *outboard* (`|y| = 1.05 m`) with only a small height offset
(`z = −0.05 m`, roughly a fin's half-height) from the wing plane. `omega × r` at that station is
dominated by the outboard lever arm, which for a vertical-span fin projects mostly into *spanwise*
flow along the fin (crossflow drag, not lift), not into the fin's own lift-generating axis. The
mechanism that gives a real vertical tail its C_Yp — fin height above the roll axis — is present in
this geometry only weakly, so the panel model's C_Yp and C_np come instead from a smaller,
different effect (most likely the swept wing panels' own asymmetric response) and land on the wrong
side of zero. Both published values are themselves small (|C_np| < 0.005, |C_Yp| < 0.14), so this
is a case where a modest geometric simplification — a winglet modelled with too little height
relative to its span — flips the sign of a small, second-order coupling term rather than
contradicting a first-order effect.

**C_mq overshoots by about 2×** (−2.58 vs −1.3). The X8 has no horizontal tail, so all of its
published pitch damping has to come from the wing's own moment arm and each panel's own intrinsic
pitching-moment response. The outer and inner panels sit far enough aft of the CG (0.25–0.47 m)
that their `omega × r` response is already substantial, and the fitted per-panel
`moment_coefficient_alpha` terms (allowed to be any sign, representing non-ideal-AC camber/twist
effects rather than pure geometry, and pushed as large as 0.7–1.5/rad by the static fit — see "The
fit" above) add to that rather than opposing it. This is a case where a tailless flying wing's
damping is genuinely a whole-airframe effect, and this simple panel decomposition, without a
downwash or unsteady-wake model between panels, has no mechanism to reduce it back toward the
measured value once the static fit has pushed those intrinsic moment terms up to explain `C_m0`
and `C_malpha` with only three independent panel groups.

## Summary

Geometry alone — reasonable panel positions and shared, regularized-but-still-fitted per-panel
airfoil coefficients, fitted only against the published *static* polynomial — reproduces the sign
of six of the eight published rate derivatives (C_lp, C_lr, C_nr, C_Yr, C_Lq, and C_mq) and lands
within roughly a factor of two of magnitude for five of those six. It gets the two smallest,
second-order cross-coupling terms, C_np and C_Yp, backwards. The pattern is consistent with what
this geometry can and cannot see: derivatives driven by in-plane lever arms and sweep (roll and yaw
damping, lift-due-to-pitch-rate, side-force-due-to-yaw-rate) come through with the right sign and
the right order of magnitude; the two derivatives that depend on a fin's real height above the roll
axis and its precise three-dimensional lift slope — both of which this single-strip-per-panel model
approximates crudely — do not.


## Against flight

Both models were replayed against the NTNU X8 flight campaign through Glassbox's rolling-window
protocol (Glassbox `docs/cascade-x8-validation.md`, 2026-09-01). At the same documented-uncertainty
variant (CG 50 mm forward, 4 kg, twice the pendulum inertia, half the campaign's inferred vertical
wind) the coefficient table scores 0.677 of kinematic persistence and the panel model 0.675, so
statics fitted to the table plus rate derivatives from geometry predict flight as well as the
published derivatives do. One-step residual regressions split that: the panel model halves the
roll-moment residual (2.0 against 3.4 N m rms) with its aileron and dihedral terms matching flight,
while its geometric pitch damping of −2.6 overstates the flying wing's weak pitch damping more than
XFLR5's −1.3 does. The next question for this model is what layout or unsteady term gives a
tailless wing its measured pitch damping.
