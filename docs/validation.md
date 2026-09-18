# Evidence and model limitations

Numerical verification establishes properties of Cascade's equations and implementation.
It does not establish the accuracy of those equations for an aircraft or flight regime.
The release record links executed checks, configurations, and artifacts in
[rc-readiness.md](rc-readiness.md). Earlier [0.3.0rc1](release-readiness.md) and
[0.4.0.dev0](feature-readiness.md) records apply to their archived distributions.

## Packaged airframes

- `aerobatic_reference` and `tailsitter_reference` are illustrative software fixtures.
- `skywalker_x8` uses the published NTNU coefficient model and records parameter provenance
  in its TOML. Agreement with those coefficients is not independent flight validation.
- `skywalker_x8_panels` fits static behavior to that coefficient model. Its rate derivatives
  are geometric predictions with known errors; see [skywalker-x8.md](skywalker-x8.md).
- Generated archetypes provide diversity for simulation experiments. They have not been
  validated as a statistical model of real airframe families.

## Correction to the historical X8 headline

The previous README described a roughly 0.68-to-persistence replay score as "unfitted".
That description was misleading: the best variant was selected across CG, mass, inertia,
and inferred-wind choices using the scored maneuvers. It is a tuned/selected result, not an
independent held-out estimate for the packaged nominal model. The differing numbers in old
task notes also refer to different configurations and must not be combined.

The local Glassbox checkout contains the historical result JSON and the variant-grid code
(`docs/results/cascade-x8-validation-results.json` and
`src/glassbox/workflows/benchmarks/cascade_x8.py`). Its report identifies a best model and
marks acceptance as `not_scored`. This candidate does not ship the original flight-data
bundle or a frozen held-out replay protocol, and the historical grid is not reproduced as
release evidence. The old quantitative headline has therefore been withdrawn.

To restore a quantitative flight-accuracy claim: make the data/licensing and adapter version
reproducible, select parameters only on fitting maneuvers, freeze the variant and scoring
protocol, evaluate untouched maneuvers, and publish per-regime errors alongside the nominal
model and persistence baseline with full provenance. The earlier maneuvers already examined
during selection must not be relabeled as a new blind holdout.

## Scope of current evidence

The test suite covers conservation, coordinate transforms, published coefficient recovery,
actuator behavior, trim, controllers, episode behavior, stochastic models, JIT/batching and
autodiff. Release numerical checks add finite-difference gradient agreement and RK4 timestep
convergence for both aerodynamic backends, in addition to representative dtype checks.
Pass/fail results are recorded only after executing those checks.

Controller, learning, and throughput examples are simulator experiments with declared seeds
and configurations. Their results are not flight performance promises, population statistics,
or GPU measurements. Performance depends on device, compiler, precision, batch size and
system load. Keep warm-up/compilation separate from execution timing and save the emitted
provenance with a result.

Full-envelope finiteness means the functions remain numerically defined through unusual
attitudes/flows. High-alpha accuracy still requires airframe-specific measurements. Ground
contact, landing, wake dynamics, flight-stack integration, and onboard policy execution are
outside this candidate's demonstrated capabilities.
