# Cascade milestone: X8 plant validated against real flight through Glassbox

Plan of record: `~/.claude/plans/refactored-wishing-neumann.md` (approved 2026-09-01).

## Phase 0: repository foundations
- [x] git init, baseline commit, branch `main`
- [x] LICENSE (MIT), CI workflow, pyproject metadata, `dist/` removed
- [x] Regression test: torque-free tumbling conserves angular momentum and energy
- [x] Regression test: roll moment monotonic in aileron (xfail strict until Phase 1)
- [x] `uv run pytest` and `uv run ruff check .` green; commit

## Phase 1: schema v2 physics
- [x] 1a flapped and all-moving surfaces (`all_moving_fraction`, `flap_effectiveness`, flap moment and drag)
- [x] 1b propeller: polynomial thrust map (linear CT(J) and the NTNU X8 law as exact cases), momentum-theory induced velocity, discriminant check
- [x] 1c whole-aircraft coefficient backend (`BodyModel`, `[body]` table, `deflection_map`, zero-area surfaces)
- [x] 1d channels as linear coordinates; `control_from_actuators`
- [x] Schema v2 in `spec.py`, fixture TOML converted, docs updated
- [x] Tests listed in the plan; un-xfail roll monotonicity; examples still run; commit

## Phase 2: canonical conversion, plant, environment sequences
- [x] `cascade/canonical.py` with tests (no Glassbox import)
- [x] `rollout(environments=...)` time-major path with tests
- [x] `cascade/plant.py` with tests; exports; README and architecture doc; commit

## Phase 3: Skywalker X8 spec from the published model
- [x] `aircraft/skywalker_x8.toml` with provenance; loader; propeller fit script and residual
- [x] Tests: coefficient reproduction, control derivative signs, trim at 18 m/s (both pitch variants)
- [x] `examples/x8_trim.py`; commit
- [x] Trim gains a yaw-offset (sideslip) decision variable so rudderless aircraft balance yaw

## Phase 4: Glassbox integration and X8 replay validation
- [x] Glassbox `cascade` extra via uv path source; pytest marker
- [x] `glassbox/integrations/cascade.py` (plant adapter, `predict_windows`)
- [x] `glassbox-x8 evaluate-cascade` with a (CG shift x mass x yaw damping x vertical wind) variant grid, reusing x8_evaluation's protocol helpers
- [x] Opt-in tests; run the evaluation; record the result and the vertical-wind finding; commit

## Phase 4 follow-ups (2026-09-01, Glassbox commit 6efdc5b)
- [x] `diagnose-cascade` residual regressions; lag-aware actuator initialization; inertia-scale axis
- [x] Validation rerun: best 0.677 vs persistence (CG +50 mm, 4 kg, inertia x2, half vertical wind)

## Phase 5 (follow-on)
- [x] Component-panel X8 fitted to the coefficient backend; rate derivatives predicted from geometry (docs/skywalker-x8.md)
- [x] Dryden gust sequences (`cascade.gusts`) with tests and docs (export added with the panel integration)

## Review (2026-09-01)

- Cascade: 71 tests, ruff clean, five commits on `main` (baseline, Phase 0, Phase 1 x2, Phase 2 x2, Phase 3).
- Physics changes beyond the plan, each forced by evidence: the propeller became a polynomial
  thrust map because a linear C_T(J) could not represent the NTNU law (0.6 N RMS, negative J0);
  flap effectiveness also shifts the separated flat-plate incidence because otherwise stalled
  ailerons had no authority and the post-stall trim branch could not balance propeller torque;
  the body block needed its own normal-force coefficient so an empty body is silent post-stall;
  trim gained a yaw-offset variable because a rudderless wing balances aileron-induced yaw with
  sideslip; the default trim seed moved to half throttle to avoid the windmilling valley.
- Validation (Glassbox `docs/cascade-x8-validation.md`): published model as-is scores 2.77 vs
  persistence (untrimmed, pitches away); with a 50 mm forward CG and a quarter of the campaign's
  vertical wind it scores 0.735, 1.44x the fitted structured model. Findings: vertical wind about
  0.4 of the estimate by lift balance; CG/wind trade; 2.7 N forward-force excess; rate residuals
  implicate the EKF rate signals rather than any single coefficient.
- Glassbox working tree had unrelated uncommitted edits (Crazyflow throw campaign) from another
  session; only the Cascade integration files were committed.

## Glassbox readiness (branch `cascade-integration` in `~/projects/glassbox-cascade`)
- [x] Isolated worktree with its own venv; `cascade-flight` from GitHub instead of `../cascade`
- [x] CI workflow (`-m "not px4_sitl and not crazyflow"`), README extras and Python range, repository URL
- [ ] Full suite green in the fresh copy (running)
- [ ] LICENSE (Ryland to choose); push branch when Ryland says so

## Phase 6: control cascade and tailsitter flagship (started 2026-09-01)
Direction: docs/architecture.md Milestone 3 and the thesis (fielded fixed-wing autonomy is VTOL
fixed-wing; hover-to-cruise transition is post-stall flight). Assumptions stated in the recap.
- [x] 6a `cascade/control.py`: rate PID, quaternion attitude P, airspeed/altitude/heading guidance,
      rate-scheduled composition, `closed_loop_rollout`, gains tuned by step-response tests on the
      aerobatic reference and the X8, gradient through gains (docs/control.md)
- [x] 6b `aircraft/tailsitter_reference.toml`: washed/clean panels, hover balance, zero-airspeed
      authority, transition corridor with both trim branches (docs/tailsitter.md)
- [x] 6c transition example: hover -> cruise under the cascade, differentiated end to end
      (`examples/tailsitter_transition.py`, `tests/test_transition.py`: 2 s hover hold, 3.5 m/s^2
      tilt ramp reaches 7 m/s at t = 4 s, cruise at 7.1 m/s with pitch ~9 deg, altitude excursion
      < 1.6 m, finite gradient of final speed in the ramp acceleration)
- [x] 6d cruise -> hover back-transition: `trapezoid_speed_profile` + `speed_profile_schedule`
      fly hover -> 8 m/s -> hover (examples/tailsitter_transition.py, tests/test_transition.py).
      Needed: (1) blend weight gated on commanded as well as measured speed so the schedule owns
      the mode; (2) hover thrust law credits wing lift (`wing_speed_m_s`) and clips the position
      error (`position_error_limit_m`); (3) a model fix: the separated-flow flap load now keeps
      the attached flap moment arm, without which full elevon could not hold alpha > 30 deg at
      7 m/s and the corridor above 3 m/s was a rolled/sideslipping trim family; (4) trim now pulls
      roll and yaw offset toward zero (1e-3) so the symmetric member of the slip family is chosen
      on every platform (this was the ubuntu-only CI failure)
- [ ] 6e hover-hold integrator gain pass (the hold still drifts 0.15 m against the propwash camber
      lift before the integral catches it), then a heading change during cruise and a gusty
      round trip with `cascade.gusts`
