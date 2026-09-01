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
  body rotation and propeller slipstream.
- **Dynamics, not animation.** Actuator lag, rate limits, propeller dynamics, and continuous flow-
  separation states are part of the simulated state.
- **JAX-native.** State and model objects are PyTrees; stepping, batching, differentiation, and
  rollouts compose with `jax.jit`, `jax.vmap`, `jax.grad`, and `jax.lax.scan`.
- **Explicit conventions.** Physics uses right-handed NED world axes and FRD body axes. Rendering
  adapters must perform their coordinate conversion at the boundary.
- **Airframe-specific truth.** The engine is reusable, but high-alpha parameters and residuals are
  expected to be identified for each airframe.

See [the architecture document](docs/architecture.md) for scope, equations, extension points, and
the roadmap. The [analysis guide](docs/analysis.md) covers trim, post-stall branch continuation,
coefficient sweeps, and local linearization; [aircraft specifications](docs/aircraft-spec.md)
documents the versioned TOML format.

## Development

```bash
uv sync --python 3.13
uv run pytest
uv run ruff check .
```

The bundled reference aircraft is intentionally an illustrative dynamics fixture, not a validated
real vehicle.

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

Run `uv run python examples/high_alpha.py` for a post-stall rollout differentiated with respect to
the elevator command. Run `uv run python examples/trim_envelope.py` to trace conventional and
high-alpha equilibrium branches and linearize a trim point.
