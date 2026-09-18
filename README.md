# Cascade

Cascade is a differentiable, massively batchable fixed-wing flight-dynamics core built with JAX.
It is aimed at learning, planning, control, and system identification across the full flight
envelope, including high-angle-of-attack and post-stall flight.

The first milestone deliberately focuses on one thing: high-quality airborne dynamics. Rendering,
autopilots, ground contact, hardware interfaces, and identification tooling are separate layers over
the same pure functional core.

## Design principles

- **Full-envelope by construction.** Aerodynamic functions remain finite through stall, inverted
  flight, sideways flow, and near-zero airspeed.
- **Component aerodynamics.** Wings, tails, and control surfaces see their own local flow, including
  body rotation and propeller slipstream. Flapped and all-moving controls are distinct.
- **Published models drop in.** A whole-aircraft coefficient table in the classical polynomial
  form can describe an airframe alone or alongside components, blended to a flat plate past stall.
- **Dynamics, not animation.** Actuator lag, rate limits, propeller dynamics, and continuous flow-
  separation states are part of the simulated state.
- **JAX-native.** State and model objects are PyTrees; stepping, batching, differentiation, and
  rollouts compose with `jax.jit`, `jax.vmap`, `jax.grad`, and `jax.lax.scan`.
- **Explicit conventions.** Physics uses right-handed NED world axes and FRD body axes. Rendering
  and identification adapters convert at the boundary; `cascade.canonical` provides the NWU/FLU
  scalar-first 13-vector state used by Glassbox-style tooling.
- **Airframe-specific truth.** The engine is reusable, but high-alpha parameters and residuals are
  expected to be identified for each airframe.

Current candidate: **0.4.0rc1**. See the [current candidate checks](docs/rc-readiness.md).
Archived [0.4.0.dev0 checks](docs/feature-readiness.md) and
[0.3.0rc1 checks](docs/release-readiness.md) apply to their preserved artifacts. See
[supported interfaces and migration notes](docs/compatibility.md) and [validation limits](docs/validation.md).

New research workflows connect [frozen experiments and controller comparisons](docs/experiments.md)
to an [offline HTML flight inspector](docs/inspection.md). Episodes now support
[scheduled, waypoint and orbit missions](docs/missions.md), [multirate sensors with dropout,
drift and timing jitter](docs/sensors.md), and a packaged [Gymnasium integration](docs/gymnasium.md).
[Turn trim and airspeed-dependent gains](docs/envelope.md) expand the analysis/control layer;
[flight-data packs](docs/flight-data.md) provide explicit splits and replay scoring.

Run `uv run python examples/research_workflow.py --output dist/my-experiment` to generate
a matched controller comparison and standalone HTML reports. The folder must be new or empty.
The flight-data example uses explicitly synthetic recordings; it supplies no new flight-accuracy claim.

The [observation-only controller](docs/observed-control.md) consumes the same sensor vector as
a learned policy. The [sustained benchmark campaign](docs/benchmark-campaign.md) freezes longer
missions, seed splits and acceptance limits before evaluation, including paired sensor stress.
For candidate installation and feedback, use the [prerelease tester guide](docs/prerelease-feedback.md).

Development after that candidate adds [sensor-aware policy learning](docs/learning.md):
feedforward/recurrent controllers, resumable training checkpoints, and evaluation/export of
the same learned weights. Run `uv run python -m cascade.learning dist/learning --steps 60`
for a frozen, three-seed tracking benchmark. Its verification is recorded separately from
the preserved candidate artifacts.

Quick test run: `uv run --frozen pytest -m "not slow"` (the full suite takes several minutes).

See [the architecture document](docs/architecture.md) for scope, equations, extension points,
the package layout, and the roadmap. The [analysis guide](docs/analysis.md) covers trim, post-stall branch continuation,
coefficient sweeps, and local linearization; [aircraft specifications](docs/aircraft-spec.md)
documents the versioned TOML format; [the control guide](docs/control.md) covers the rate/attitude/
guidance cascade, channel-map sign conventions, tuning, and differentiable-tuning examples;
[environments](docs/environments.md) covers the native-JAX episode functions for learning and
trajectory optimisation; [archetypes](docs/archetypes.md) covers parametric airframe families
and automatic controller tuning; [weather](docs/weather.md) covers wind profiles, turbulence
classes, and station records; [rendering](docs/rendering.md) covers geometry from the spec,
MJCF export, and MuJoCo video ([tailsitter round trip](docs/media/tailsitter_round_trip_follow.mp4)).

## Install

```bash
git clone https://github.com/Rylandl/cascade
cd cascade
uv sync --frozen --python 3.13
```

Alternatively, `pip install .` from the checkout, or install a verified wheel using the
[release instructions](docs/releasing.md). Candidate publication is a separate release step.
Python 3.11 to 3.13; runtime dependencies are JAX, NumPy, SciPy, and tomli-w. MuJoCo rendering
is optional (`viz` extra); JAX policy serialization needs the `export` extra.

## Development

```bash
uv sync --frozen --python 3.13
uv run --frozen pytest
uv run --frozen ruff check .
```

The bundled aerobatic reference aircraft is intentionally an illustrative dynamics fixture.
`cascade.skywalker_x8()` is assembled from the published NTNU Skywalker X8 model with full
provenance. The former flight-replay headline selected a parameter variant using the scored
maneuvers and should not have been described as unfitted. It is withdrawn pending a
reproducible held-out evaluation; see [the evidence statement](docs/validation.md). Full-envelope
numerical behavior is tested, while physical accuracy requires airframe-specific calibration.

## Minimal rollout

```python
import jax
import jax.numpy as jnp
import cascade

model = cascade.aerobatic_reference()
state = cascade.zero_state(model, batch_shape=(4096,), altitude=20.0, forward_speed=12.0)
control = cascade.ControlInput(
    propeller=jnp.full((4096, 1), 0.6),
    channel=jnp.zeros((4096, 3)),
)
environment = cascade.standard_environment(batch_shape=(4096,))
state = cascade.equilibrate_internal_state(model, state, control, environment)
controls = cascade.repeat_control(control, steps=100)

final_state, trajectory = jax.jit(cascade.rollout)(model, state, controls, environment, 0.01)
```

`cascade.tailsitter_reference()` is an indoor-class flying-wing tailsitter fixture whose propwash
gives its elevons authority at zero airspeed; `cascade.control.vtol` adds hover guidance and a
hover-to-cruise transition controller over the same loops (`docs/tailsitter.md`).

`cascade.env.gusts` generates Dryden turbulence as a time-major environment sequence for
`rollout`, with per-world realizations from a PRNG key.

`cascade.Plant` wraps the same core as a stepped hidden plant for identification tooling: reset to
a canonical state, hold one command per control interval, read back commanded and applied
actuation.

Run `uv run python examples/high_alpha.py` for a post-stall rollout differentiated with respect to
the elevator command. Run `uv run python examples/trim_envelope.py` to trace conventional and
high-alpha equilibrium branches and linearize a trim point.
