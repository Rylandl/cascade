"""Environment throughput versus batch size on the current JAX backend.

Emits the table in docs/environments.md ("Throughput"). Run on a GPU host to add a GPU row:
``uv run python scripts/benchmark_env.py``. One control step is ten RK4 sub-steps.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from cascade.env import EpisodeConfig, action_size, control_to_action, reset, step, tracking_task
from cascade.env.tasks import trimmed_reference
from cascade.provenance import stamp
from cascade.reference import aerobatic_reference_spec, skywalker_x8_spec


def benchmark(name, model, config, task, reference, batch):
    action = control_to_action(config, reference.control)
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    states, _ = jax.jit(jax.vmap(lambda k: reset(model, config, task, reference, k)))(keys)
    actions = jnp.broadcast_to(action, (batch, action_size(model)))
    env_step = jax.jit(jax.vmap(lambda s, a: step(model, config, task, reference, s, a)))
    states, *_ = env_step(states, actions)
    jax.block_until_ready(states)
    repeats = 100 if batch <= 1024 else 20
    start = time.perf_counter()
    for _ in range(repeats):
        states, *_ = env_step(states, actions)
    jax.block_until_ready(states)
    elapsed = (time.perf_counter() - start) / repeats
    print(
        f"{name:14s} {batch:6d} {elapsed * 1e3:9.2f} {batch / elapsed:13.0f} "
        f"{batch * config.substeps / elapsed:13.0f}"
    )
    return {
        "aircraft": name,
        "batch": batch,
        "milliseconds_per_step": elapsed * 1e3,
        "environment_steps_per_second": batch / elapsed,
        "rk4_steps_per_second": batch * config.substeps / elapsed,
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write results and provenance as JSON")
    args = parser.parse_args()
    record = stamp()
    print(f"backend {record['backend']}  jax {record['jax_version']}  {record['platform']}")
    print("aircraft        batch   ms/step   env steps/s   RK4 steps/s")
    cases = (
        ("aerobatic", aerobatic_reference_spec(), 12.0, 50.0, 1.0),
        ("skywalker_x8", skywalker_x8_spec(), 18.0, 100.0, 0.5),
    )
    results = {"provenance": record, "warmup_calls": 1, "cases": []}
    for name, spec, speed, altitude, scale in cases:
        model = spec.to_model()
        task = tracking_task(speed, altitude, 0.0)
        reference = trimmed_reference(model, task)
        config = EpisodeConfig(channel_scale=scale)
        case = {
            "provenance": stamp(spec, model, seed=0),
            "config": {
                "speed_m_s": speed,
                "altitude_m": altitude,
                "channel_scale": scale,
                "control_frequency_hz": config.control_frequency_hz,
                "simulation_frequency_hz": config.simulation_frequency_hz,
            },
            "rows": [],
        }
        for batch in (1, 64, 1024, 4096, 16384):
            case["rows"].append(benchmark(name, model, config, task, reference, batch))
        results["cases"].append(case)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
