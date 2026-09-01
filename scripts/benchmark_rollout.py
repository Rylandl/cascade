"""Measure batched RK4 rollout throughput for the packaged aircraft on the current device.

Run with ``uv run python scripts/benchmark_rollout.py``. Each row reports the steady-state
world-steps per second for a jitted 100-step rollout after one warm-up call, so compile time is
excluded. Numbers depend on the machine and on scheduling: run on an idle machine at normal
priority (background QoS on Apple silicon pins the job to efficiency cores and cuts throughput
several-fold), and record results with the device they came from.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import cascade

STEPS = 100
WORLDS = (1, 256, 4096, 16384)


def benchmark(name: str, spec: cascade.AircraftSpec, worlds: int, throttle: float) -> float:
    model = spec.to_model()
    state = cascade.zero_state(model, batch_shape=(worlds,), altitude=50.0, forward_speed=15.0)
    control = cascade.ControlInput(
        propeller=jnp.full((worlds, model.n_propellers), throttle),
        channel=jnp.zeros((worlds, model.n_control_channels)),
    )
    environment = cascade.standard_environment(batch_shape=(worlds,))
    state = cascade.equilibrate_internal_state(model, state, control, environment)
    controls = cascade.repeat_control(control, steps=STEPS)
    step = jax.jit(cascade.rollout)
    jax.block_until_ready(step(model, state, controls, environment, 0.01))
    start = time.perf_counter()
    repeats = 3
    for _ in range(repeats):
        jax.block_until_ready(step(model, state, controls, environment, 0.01))
    elapsed = (time.perf_counter() - start) / repeats
    return worlds * STEPS / elapsed


def main() -> None:
    print(f"device: {jax.default_backend()}; steps per rollout: {STEPS}; RK4")
    print(f"{'aircraft':22s} {'worlds':>7s} {'world-steps/s':>15s}")
    for name, loader, throttle in (
        ("aerobatic_reference", cascade.aerobatic_reference_spec, 0.6),
        ("skywalker_x8", cascade.skywalker_x8_spec, 0.45),
    ):
        spec = loader()
        for worlds in WORLDS:
            rate = benchmark(name, spec, worlds, throttle)
            print(f"{name:22s} {worlds:7d} {rate / 1e6:13.2f} M")


if __name__ == "__main__":
    main()
