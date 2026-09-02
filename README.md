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

See [the architecture document](docs/architecture.md) for scope, equations, extension points, and
the roadmap. The [analysis guide](docs/analysis.md) covers trim, post-stall branch continuation,
coefficient sweeps, and local linearization; [aircraft specifications](docs/aircraft-spec.md)
documents the versioned TOML format; [the control guide](docs/control.md) covers the rate/attitude/
guidance cascade, channel-map sign conventions, tuning, and differentiable-tuning examples;
[environments](docs/environments.md) covers the native-JAX episode functions for learning and
trajectory optimisation; [archetypes](docs/archetypes.md) covers parametric airframe families
and automatic controller tuning.

## Install

```bash
uv add "cascade-flight @ git+https://github.com/Rylandl/cascade"
```

or with pip, `pip install "cascade-flight @ git+https://github.com/Rylandl/cascade"`. Python 3.11
to 3.13; the only runtime dependencies are JAX, SciPy, and tomli-w.

## Development

```bash
uv sync --python 3.13
uv run pytest
uv run ruff check .
```

The bundled aerobatic reference aircraft is intentionally an illustrative dynamics fixture.
`cascade.skywalker_x8()` is assembled from the published NTNU Skywalker X8 model with full
provenance; its first validation against real flight, through Glassbox's X8 campaign adapter,
is recorded in Glassbox's `docs/cascade-x8-validation.md` (unfitted, within the paper's stated
CG and inertia uncertainty: 0.68 of kinematic persistence, 1.33x the fitted effective model).

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
gives its elevons authority at zero airspeed; `cascade.vtol` adds hover guidance and a
hover-to-cruise transition controller over the same loops (`docs/tailsitter.md`).

`cascade.gusts` generates Dryden turbulence as a time-major environment sequence for
`rollout`, with per-world realizations from a PRNG key.

`cascade.Plant` wraps the same core as a stepped hidden plant for identification tooling: reset to
a canonical state, hold one command per control interval, read back commanded and applied
actuation.

Run `uv run python examples/high_alpha.py` for a post-stall rollout differentiated with respect to
the elevator command. Run `uv run python examples/trim_envelope.py` to trace conventional and
high-alpha equilibrium branches and linearize a trim point.
