# Feedback synthesis (2026-09-02)

Six Opus reviewers in role read Cascade v0.2.0 read-only: an academic RL/control professor, a
flight-dynamics and system-identification researcher, a small-UAS manufacturer's VP of
engineering, a government autonomy program manager, an autonomy startup's founding engineer,
and a flight-test and airworthiness lead. Full reviews sit beside this file.

## Where they agree the fit is real

- The layered package, the two configuration conventions, and `canonical.py` as the single
  frame-conversion site were called out as discipline that is rare in research sims. The
  learning example reproduced bit for bit for the startup engineer.
- Families of auto-tuned archetypes, batched for one vmap with design parameters hidden, is
  the differentiator versus Crazyflow and the quad ecosystem: a morphology-generalisation
  suite with a non-strawman baseline built in (academic, program manager).
- The transition corridor analysis is already a test-planning artefact: it identifies the
  authority pinch and the branch coexistence that should set build-up order and abort
  triggers (safety lead).
- The verification tests are strong for an open-source sim: conservation to 1e-5, finiteness
  and gradients over the full angle range, coefficient tables to 2e-5, Dryden variances
  against the standard (safety lead).
- MIT licensing with light runtime dependencies blocks nothing (everyone).

## Where they agree it falls short

Ranked by how many roles raised it and how hard they pressed.

1. **The validation story is thin, off-repo, and inconsistent.** One airframe, one campaign, a
   rolling-window score below persistence, chosen from a variant grid (2x inertia, half the
   inferred vertical wind), and quoted three ways: 0.68/1.33x in the README, 0.677 in the X8
   doc, 0.735/1.44x in the todo. The real-flight chain runs through Glassbox whose licence is
   undecided. Every role except the startup engineer led with this; the program manager and
   safety lead would not put the number on a slide. Asked for: a pre-registered variant, the
   data and a replay script in this repo, tolerance-banded overlays per regime instead of a
   scalar, and a configuration-stamped run record (model hash, code version, seed, dtype).
2. **No path to a flight stack.** No PX4/ArduPilot parameter export, no ULog/DataFlash
   ingestion, no MAVLink/HIL bridge, no policy export; `Plant` is HIL-shaped with nothing on
   the other end. The manufacturer's single adoption feature is a closed loop from their flight
   log to a gain set that flies on the first sortie; the startup wants `jax.export` and a
   fixed-rate onboard harness; the program manager wants sim-tuned gains flown on two
   airframes.
3. **Sim-to-real realism stops at the observation.** Actuation latency and jitter are absent
   (the startup's first PR), the observation is privileged (alpha, beta, true airspeed,
   deflections, RPM) with no mask or spec, there is no IMU model, sensor noise is in
   observation units rather than sensor units, and gusts have no rotary components. A named
   randomisation spec is wanted instead of hand-written `_replace`.
4. **Ground contact, takeoff, and landing are absent by design**, which is where most mishaps
   live and where field use is dominated (four of five roles).
5. **No failure injection**: motor-out, jam, hardover, partial power, GNSS or link loss. Half
   the qualification hazard list is unreachable by API (safety lead, manufacturer).
6. **Weather is a point Dryden with a log profile.** Missing the certification-standard
   one-minus-cosine gust, shear and thermals, vertical mean wind, density altitude, and any
   shipped station data.
7. **Known physics gaps that touch control design**: no inter-surface downwash, static
   slipstream, quasi-steady stall with one lag state, the panel X8's pitch damping at twice
   the measured value and its C_np and C_Yp signs wrong. The program manager calls the damping
   error a bandwidth-setting error; the safety lead notes those terms govern departure. The
   system-identification reviewer adds four specifics: component surfaces have no separated
   centre-of-pressure shift (the stall pitch break exists only in the body block); the
   archetypes' static downwash fold scales pitch damping and elevator power by the same factor
   as the neutral point, understating both by about 40% at aspect ratio 7; the stall lag is in
   seconds rather than convective time and the coefficient backend has no stall dynamics at
   all; propeller inflow is axial only, with no propeller normal force, swirl, wake lag, or
   gyroscopic moment, and shaft speed ignores aerodynamic load.
8. **Benchmarks and reproducibility.** All throughput numbers are one laptop CPU; no GPU table
   or GPU CI; no seeded PPO-versus-gradient learning curve; the doc tables are prose rather than
   scripts; no PyPI release, DOI, or CITATION.cff. The academic's first ask is one script
   emitting the GPU table and the learning curve.
9. **The programme thesis metric does not exist.** Time-to-competence on a held-out airframe
   is future work; there is no frozen design manifest, no train/test split, no seed
   discipline, no non-adaptive family-trained baseline, no variance reported anywhere.
10. **Project risk**: one author, two days of history, alpha, a module reshuffle already, doc
    drift after the reshuffle. Vendor and pin, say two of them.

## Where they disagree

- **What "integration" means.** The manufacturer wants PX4/ArduPlane parameters and a
  quadplane mixer (their product line is 4+1, not tailsitters); the startup wants ONNX or
  StableHLO export and an onboard harness; the academic wants Gymnasium and PPO glue. Three
  different bridges from the same core.
- **Fidelity versus tooling.** The safety lead and program manager want months of physics
  (downwash, unsteady stall, spin) before trusting the post-stall claim; the manufacturer and
  startup would take today's physics with better tooling and honest labels.
- **What counts as evidence.** The academic accepts replay scores with a seeded benchmark;
  the program manager and safety lead want closed-loop flight and pre-registration.

## Cheap fixes to do now (days)

Re-score the X8 variant grid with selection on a disjoint set of maneuvers and relabel the
README headline (as-published 2.76 of persistence; the 0.68 is tuned), since the current
number is selection on the evaluation set with four effective free parameters; fix the
archetype mass split, which sums to 1.06 to 1.08 of the mass, and replace the sphere pod
inertia with a slender body; bound trim channels by the mapped physical limit instead of a
hard-coded +-1; floor the diagnostic coefficients' airspeed sensibly; give the mean wind a
vertical component; note that frozen-field Dryden is not valid at hover; fix the doc drift the
restructure left (`docs/control.md` module path, `env/baselines.py` importing the `vtol`
shim, `cascade.env.gusts` missing from the env init); add `observation_size(model)` and
observation index constants; make `reset` and `step` return the same end of the buffer;
remove load-bearing-looking float32 casts and state a dtype policy; precompute the actuator
pseudo-inverse on the model; add `CITATION.cff`; a configuration stamp on results.

## Recommended sequence

- **Phase 10, evidence and hygiene (about 2 weeks).** The cheap fixes above; a
  configuration-stamped run record; the X8 replay harness and its data in this repo with a
  held-out split and tolerance-band overlays; identifiability diagnostics (Cramer-Rao bound and
  parameter correlations from the existing Jacobians, which would have caught the CG, inertia,
  and wind degeneracy); per-table scripts for every documented number; a GPU benchmark script
  (numbers need a GPU); PyPI release and DOI.
- **Phase 11, sim-to-real tooling (about 3 weeks).** Actuation latency and jitter; an
  observation spec with an IMU block and sensor-unit noise; a named randomisation spec;
  a time-indexed failure-injection API on surfaces and propellers; trajectory logging with a
  versioned schema and a ULog-to-canonical loader that drives `Plant`; the one-minus-cosine
  gust, shear, and density altitude.
- **Phase 12, the thesis harness (2 to 3 months, with Glassbox).** A frozen two-archetype
  manifest with hashes and splits, a station-record weather set that ships, the
  flight-minutes-to-baseline metric, three baselines including a non-adaptive family-trained
  policy, five-seed variance, one reproduction container.
- **Phase 13, the flight-stack bridge.** PX4/ArduPlane parameter export with a documented
  gain mapping, then the manufacturer's log-fit loop, then a lockstep HIL bridge over `Plant`.
- **Later, in this order:** ground contact and a landing task; quadplane allocation and
  transition; a per-surface downwash map, a separated centre-of-pressure shift on panels, and
  a convective-time stall term shared by both backends, each re-scored against flight; trim
  beyond straight flight (steady turn, pull-up, sideslip) and output-error identification
  over `Plant` for the identification workflow.

## One-line asks, by role

- Academic: one script for the GPU table and a seeded PPO-versus-gradient curve.
- System identification: rerun the X8 variant grid with selection on disjoint maneuvers and relabel the README headline as tuned.
- Manufacturer: flight log in, PX4/ArduPlane gains out, with the validation score in between.
- Program manager: a six-month time-to-competence harness with a pre-registered X8 anchor.
- Startup: actuation latency as a randomisable leaf, mirroring the observation delay.
- Safety lead: an envelope clearance card with a validation-pedigree block.
