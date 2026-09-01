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
- [x] 1b propeller `CT(J)` with momentum-theory induced velocity and `CT0 <= (pi/2) J0^2` bound
- [x] 1c whole-aircraft coefficient backend (`BodyModel`, `[body]` table, `deflection_map`, zero-area surfaces)
- [x] 1d channels as linear coordinates; `control_from_actuators`
- [x] Schema v2 in `spec.py`, fixture TOML converted, docs updated
- [x] Tests listed in the plan; un-xfail roll monotonicity; examples still run; commit

## Phase 2: canonical conversion, plant, environment sequences
- [x] `cascade/canonical.py` with tests (no Glassbox import)
- [x] `rollout(environments=...)` time-major path with tests
- [ ] `cascade/plant.py` with tests; exports; README and architecture doc; commit

## Phase 3: Skywalker X8 spec from the published model
- [ ] `aircraft/skywalker_x8.toml` with provenance; loader; propeller fit script and residual
- [ ] Tests: coefficient reproduction, control derivative signs, trim at 18 m/s (both pitch variants)
- [ ] `examples/x8_trim.py`; commit

## Phase 4: Glassbox integration and X8 replay validation
- [ ] Glassbox `cascade` extra via uv path source; pytest marker
- [ ] `glassbox/integrations/cascade.py` (plant adapter, `predict_windows`)
- [ ] `x8_evaluation.score_predictor`; `glassbox-x8 evaluate-cascade` with variant matrix
- [ ] Opt-in tests; run the evaluation; record the result and the vertical-wind finding; commit

## Phase 5 (follow-on)
- [ ] Component-panel X8 fitted to the coefficient backend; `downwash_map` placeholder in docs

## Review
_(filled in at the end of each phase)_
