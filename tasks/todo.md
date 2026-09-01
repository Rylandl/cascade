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

## Phase 5 (follow-on)
- [ ] Component-panel X8 fitted to the coefficient backend; `downwash_map` placeholder in docs

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
