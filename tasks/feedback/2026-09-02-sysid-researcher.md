# Review: flight-dynamics and system-identification researcher (Opus role-play, 2026-09-02)

## 1. Physics I trust and physics I don't

Trust: the rigid-body core and wind-to-body rotation; the momentum-theory inflow through the cancellation-free root with the offline discriminant check (right limits at hover, zero thrust, windmilling; differentiable throughout); the Dryden filters (the sqrt(3) factorisation reproduces the numerator exactly, the 1/sqrt(2) normalisation is right, the low-altitude scales are 8785C); the separated-flap arm -Cm_flap/(CLa tau) as a defensible closure that docs/tailsitter.md shows is load-bearing.

Don't trust:
- No separated centre-of-pressure shift on component surfaces: moment_separated is identically zero at zero flap, so a stalled panel's normal force stays at the quarter chord. The flat-plate CP march to mid-chord (the stall pitch break) exists only in the body block. Archetypes and the X8 panels have no intrinsic stall break.
- Downwash: no downwash map. The archetype folds it in statically and double-counts: scaling the tail slope by (1 - de/da) is right for the neutral point but also scales C_mq and elevator power by about 0.6 at AR 7, so archetypes understate pitch damping and elevator authority by about 40%. No downwash lag.
- Stall hysteresis: separation_derivative is a first-order lag with constants in seconds, not c/2V, and no alpha-dot term; reduced frequency never appears. The whole-aircraft backend ignores AeroState entirely, so the validated X8 has zero stall dynamics.
- Propeller inflow is axial only: no oblique inflow or prop normal force (a first-order pitch term at the tailsitter's 50 deg trim alphas), no swirl, wake lag, or contraction; slipstream lands instantly and uniformly; no gyroscopic moment; shaft speed is a pure function of throttle with no aerodynamic unloading.
- Archetype inertia: thin plates plus a sphere for the pod overstates Ixx and understates Iyy/Izz; the parts list sums to 1.06x (flying wing) and 1.08x (conventional) of the mass; validate_model checks SPD but not the triangle inequality.
- The body block's lateral and control terms are never blended past stall: linear values at alpha 90, beta 180.

## 2. Validation evidence

The real evidence lives in Glassbox's cascade-x8-validation doc and is honest: published-as-is scores 2.76, three times worse than persistence; the headline 0.679 selects CG +50 mm, 4 kg, inertia x2, half vertical wind from a 54-cell grid scored on the same four maneuvers. That is model selection on the test set with four effective free parameters, and the README does not say so. Missing: a holdout split for the grid; independent airspeed truth (a 1.2% pitot bias generates the whole vertical-wind estimate); a weighed and swung airframe; any measurement above 12 deg alpha.

Next experiment: refit the variant grid on maneuvers 1 to 8, score on 9 to 17, report both. Then a purpose-built identification flight: elevator 3-2-1-1 and frequency sweeps at two alphas, roll doublets; fit C_mq, C_lp, C_lda by output error.

## 3. Unserved needs, ranked

1. Output-error / filter-error identification over Plant: identify(spec, dataset, free_params) -> spec + covariance. 2 to 4 weeks.
2. Identifiability diagnostics: per-maneuver Cramer-Rao bound and parameter correlation from the existing Jacobians. Days.
3. Trim beyond straight flight: steady turn, pull-up, steady sideslip. About a week.
4. Convective-time stall (Goman-Khrabrov with tau c/2V and an alpha-dot term) shared by both backends. 1 to 2 weeks plus data.
5. Spatially sampled wind so rotational gusts exist. About a week.

## 4. Red flags

- trim.py hard-codes control-channel bounds to +-1 while channels may be radians or normalised; a degree-unit spec would silently trim within +-1 deg. Bound by the mapped physical limit instead.
- float32 hard-coded in env/gusts.py and env/weather.py: silent downcast under x64.
- The diagnostic coefficient array divides by an airspeed floored at 1e-4: spikes in near-zero-speed sweeps.
- Dryden with V clamped to 1 m/s gives about 15 s time constants in hover; frozen-field turbulence is not valid at hover, yet the tailsitter gust results rest on it.
- mean_wind_ned has no vertical component while the X8 result hinges on vertical wind.
- Three X8 numbers in circulation.
- The attached-angle tanh guard bends the lift curve about 1.6% low at 10 deg for a 15 deg stall; a fitted CLa will absorb a numerical guard.

## 5. The one thing I'd ask for first

Rerun the X8 variant grid with selection on a disjoint set of maneuvers, and change the README headline to the as-published number with the tuned number labelled as tuned.
