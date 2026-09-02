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
- [x] 6e hover-hold gain pass: kp/ki/kv 4/4/4 halves the standing drift (0.10 m) but costs the
      round trip (altitude dip 0.68 m, final position error 0.59 m from integrator wind-up through
      the transition); 2/1/2.5 stays. The drift is the initial transient against the propwash
      camber lift, not a steady offset.
- [x] 6f gusty round trip: `transition_rollout(..., environments=)`; found that hover yaw (body z,
      the belly normal) had no control at all: a 1 m/s spanwise wind tipped the wing over. Added
      `TransitionController.differential_thrust` (rate-loop z command onto the motors). Gusty
      round trips (W20 2 and 4 m/s) survive; broadside hover in >= 2 m/s wind drifts because the
      plate drag is 40% of weight (hover edge-on instead; documented)
- [x] 6g heading change in cruise: `speed_profile_schedule` takes a heading profile; a 90 deg ramp
      over 3 s is tracked with ~1 s lag at 20 deg bank and the back-transition lands facing the new
      heading (0.15 m). Added `coordinated_turn_rates` feedforward to the rate setpoint (cascade
      and transition): without it the differential-thrust yaw loop fought the turn (0.08 split)
      and the nose dropped 0.9 m. `hover_azimuth_across_wind` gives the edge-on hover azimuth.
- [x] 6h gradient demo: `examples/tailsitter_tuning.py` tunes acceleration, deceleration, and
      cruise tilt through the 16 s round trip (0.2 s per value-and-gradient after a 7 s compile;
      cost 0.40 -> 0.28 in 12 steps); test asserts one descent step improves the cost
- [x] 6i `cascade.env`: native-JAX episode functions (reset/step/rollout_actions) over a trimmed
      tracking task, vmap-batched, differentiable through the episode (docs/environments.md)
- [x] 6j `HoverTask` + `hover_reference` for `cascade.env` (tailsitter hover: position, velocity,
      belly azimuth); body-frame position error joins the observation
- [x] 6k `rollout_policy` + `cascade_policy` baseline (the cascade at the control rate; on the
      aerobatic tracking task from perturbed starts: no crashes, mean reward > 0.6, > 0.8 settled)
- [x] 6l `TransitionTask` + `transition_policy` baseline (tailsitter hover -> 8 m/s at 100 Hz:
      reaches cruise within 1.5 m/s, reward < 0.6 in the first second, > 0.7 in the last);
      domain-randomisation recipe (vmap over `broadcast_model` batches) with a test
- [x] 6m `examples/learn_tracking_policy.py`: a 32-unit policy trained by gradient through the
      dynamics (Adam, clipping) on the 4 s tracking task: trim-hold 144.5, cascade 156.5, learned
      156.8 of 160 after 60 steps (36 s of training); test asserts one step improves the return
- [x] 6n `SensorNoise` (white noise per observation block + per-episode rate bias) and
      `observation_delay_steps` in the env, pure in the episode key
- [x] 6o learning the transition by gradient through the dynamics from a hover-initialised
      policy does not work as is: return 17 -> 28 of 200 in 120 steps against the transition
      controller's 142; the first gradient norm is 26k (chaotic sensitivity through the stall)
      and the policy settles in a hover-ish local optimum. Recipe for later: behaviour-clone the
      transition policy, then fine-tune by gradient; or curriculum on the ramp acceleration
- [x] 6p figures: `scripts/plot_tailsitter.py` -> docs/figures/{tailsitter_corridor,
      tailsitter_round_trip}.svg, referenced from docs/tailsitter.md
- [ ] 6q candidates: Glassbox re-evaluation of the X8 panels model after the post-stall
      flap-moment change (the campaign reaches alpha 20 deg); imitation + fine-tune transition
      learning; README figure
- [x] env throughput in docs/environments.md (M3 CPU: 75k aerobatic / 130k X8 env steps/s at
      batch 1024, ten RK4 sub-steps each)

## Phase 7: archetypes, diverse airframes, and weather (proposed 2026-09-02)
Direction: the Glassbox demonstration that a learner with no a priori information can control
diverse fixed-wing airframes quickly in real weather. Cascade supplies the family and the
weather; Glassbox the adaptation and the metric.
- [x] 7a `cascade.archetypes` (2026-09-02): parametric designs -> `AircraftSpec` on the panel backend.
      Flying wing (span, aspect ratio, wing loading, sweep, taper, washout, reflex, static margin,
      elevon span/chord fractions, winglet size, thrust-to-weight, prop diameter, motor layout
      incl. the tailsitter twin) and conventional (span, aspect ratio, wing loading, camber,
      dihedral, tail arm, horizontal/vertical tail volume coefficients, static margin, control
      chord fractions, tail arrangement: conventional / V-tail / cruciform, thrust-to-weight,
      prop diameter). Derived physics: panel discretisation by sweep/taper, lift slope from
      aspect ratio (Helmbold), induced-drag factor from span efficiency, flap effectiveness and
      flap moment from chord fraction (thin-airfoil), static wing downwash folded into the tail's
      effective slope and incidence, propwash weights from disk coverage, inertia from geometry
      and a mass split (wing, pod/fuselage, tail, battery). `sample_designs(archetype, key, n,
      ranges)` and `validate_design` (trim at design speed within limits, positive static
      margin, control authority floor, damped short period) with resampling. Tests: nominal
      designs trim and are controllable; sampled batches are mostly valid and visibly diverse
      (spread of trim alpha, throttle, short-period and roll modes).
- [x] 7b auto-tuned baseline (`cascade.autotune`, 2026-09-02): trim -> linearise -> rate/attitude gains from the linear model
      -> guidance gains from timescales -> step-response check, for any spec. The reference
      controller every sampled airframe gets without a human, and the yardstick for "learned
      quickly with no a priori information".
- [x] 7c weather (2026-09-02): `cascade.weather` (log profile, classes, per-step Dryden gusts at
      the aircraft's altitude, station records CSV + sampling) wired into `cascade.env`
      reset/step/rollouts. Not done: discrete 1-cos gusts, ISA density with altitude.
- [x] 7d `cascade.family` (2026-09-02): `sample_family` -> stacked models/tasks/references/
      auto-tuned controllers (fixed topology per archetype), one vmap per family with per-episode
      weather; designs and reports kept as hidden truth. `examples/family_episode.py`.
- [ ] 7e next: the Glassbox side (adaptation protocol across a family, the flight-minutes-to-
      baseline metric); discrete 1-cos gusts and ISA density; a learned policy across a family
      (imitation of the per-design baselines, then fine-tuning)

## Phase 8: visualisation (2026-09-02)
- [x] 8a `cascade.geometry`: parts from the spec (boxes, hinged flaps, spinning props, pod, fallback
      wing for coefficient aircraft), OBJ and MJCF writers
- [x] 8b `cascade.render`: MuJoCo kinematic playback to MP4 via ffmpeg, chase/side/ground cameras,
      stall colouring; `viz` extra; `examples/render_flight.py`
- [ ] 8c browser viewer (three.js artifact) and a matplotlib fallback for machines without GL;
      wind/gust arrows and propwash in the scene; a README video

## Phase 9: packaging (2026-09-02)
- [x] 9a layout by layer: `cascade.control` (loops, vtol, autotune, tuned), `cascade.env` (episode,
      tasks, sensors, baselines, weather, gusts, family), `cascade.design`, `cascade.viz`; top-level
      namespace trimmed to the core (55 names); compatibility shims for the old module paths
- [x] 9b `env.Reference` -> `ReferenceFlight` (alias kept); tasks own `reference_speed()` and
      `reference(model)`; `control_authority` -> `cascade.analysis`; tuned controllers out of the loops
- [x] 9c `py.typed`, 0.2.0, CHANGELOG, project URLs, `slow` marker (`pytest -m "not slow"` ~109 tests)
- [ ] 9d after Glassbox re-pins: drop the shims in 0.3

## Phase 10: evidence and hygiene (proposed from the 2026-09-02 reviews; see tasks/feedback/)
- [ ] 10a X8 numbers: re-score the variant grid with selection on disjoint maneuvers (fit 1-8,
      score 9-17), relabel the README headline (as-published 2.76; 0.68 is tuned), one number
      everywhere (README, docs/skywalker-x8.md, todo). Needs Glassbox's harness (user's agent).
- [x] 10b archetype defects: parts mass scaled to the aircraft mass; slender-body pod inertia; the
      static downwash fold replaced by a real `downwash_map` in the spec/model/aerodynamics (two-pass
      surface evaluation; zero map is bitwise the old single pass); conventional tail keeps its full
      slope (pitch authority +15% on the nominal design)
- [x] 10c trim channel bounds from the mapped physical limit (`channel_bounds`); diagnostic
      coefficients report against a 1 m/s floor; `vertical_wind_m_s` in WeatherCondition; dtype
      policy stated, float32 casts removed; Dryden-at-hover validity note
- [x] 10d doc drift fixed; gusts exported from cascade.env with public filter names;
      `observation_size` / `observation_layout`; reset and step return the same buffer end.
      Deferred: precomputing the actuator pseudo-inverse (a 6x3 pinv per telemetry read; cheap)
- [x] 10e reproducibility: `cascade.provenance` stamp (spec/model hashes, versions, backend, x64,
      seed, git commit); `scripts/benchmark_env.py` (backend-aware, GPU-ready) and
      `scripts/archetype_statistics.py` emit the documented tables; CITATION.cff; `uv build`
      produces 0.2.0 wheels. Not done: publishing to PyPI (user's call), DOI (Zenodo on a tag)
- [ ] 10f identifiability diagnostics from the existing Jacobians (Cramer-Rao, correlations):
      Glassbox's (identification); Cascade already exposes the Jacobians
- [x] 10g separated centre of pressure on component surfaces (`separated_center_of_pressure`,
      optional per surface, 0.25 flat plate in the archetypes, 0 in the packaged fixtures): the
      stall pitch break the review said panels lacked; validity stats unchanged (36/40, 38/40)
## Phase 11: sim-to-real tooling
- [x] 11a action latency: `action_delay_steps` / per-episode `action_delay_range`, action buffer in
      the state, `info["applied_action"]`. Finding: the hand-tuned aerobatic cascade at 40 Hz flies
      with 25 ms, crashes 2/8 at 50 ms, 3/8 at 75 ms, 7/8 at 100 ms; at 100 Hz fine to 40 ms
- [x] 11b `cascade.env.randomisation`: `randomisation(...)` ranges over named leaves plus a
      centre-of-mass shift; `sample_models(model, spec, key, n)`; latency randomises through
      `action_delay_range`
- [ ] 11c observation spec (select blocks; IMU specific-force block; noise in sensor units)
- [ ] 11d failure-injection API (time-indexed faults on surfaces and propellers: jam, hardover,
      motor-out, partial power)
- [ ] 11e trajectory logging with a versioned schema (npz) and a canonical-state replay over Plant
- [ ] 11f weather: 1-cos discrete gust, shear, density altitude, rotary gust components; action
      jitter and dropped frames
## Phase 12: the thesis harness (proposed, with Glassbox)
- [ ] frozen two-archetype manifest with hashes and splits; shipped station-record weather set;
      flight-minutes-to-baseline metric; three baselines incl. a non-adaptive family-trained
      policy; five-seed variance; reproduction container
## Phase 13: flight-stack bridge (proposed)
- [ ] PX4/ArduPlane parameter export with a documented gain mapping; log-fit loop; lockstep HIL
      over Plant; policy export (jax.export/StableHLO) with an onboard numeric check
