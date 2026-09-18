"""Export a saved trained controller and verify its explicit sensor/memory signature.

First train with ``python -m cascade.learning dist/learning --seeds 0 --steps 60``.
Then run ``python examples/export_policy.py dist/learning/seed-0/checkpoint.npz
dist/learning/policy.cascade-policy``. Requires the ``export`` extra. This verifies JAX
serialization, not another runtime or onboard hardware.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from cascade.env import SensorObservation
from cascade.learning.checkpoint import load_checkpoint
from cascade.learning.export import export_policy, load_exported_policy
from cascade.learning.policies import initial_memory, policy_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    export_policy(checkpoint, args.output)
    exported = load_exported_policy(args.output)
    config = checkpoint.policy_config
    memory = initial_memory(config)
    exported_memory = exported.initial_memory()
    largest_error = 0.0
    for index in range(8):
        values = jax.random.normal(jax.random.PRNGKey(index), (config.observation_size,))
        valid = jnp.arange(config.observation_size) % 3 != index % 3
        age = jnp.where(valid, index * 0.025, jnp.inf)
        reading = SensorObservation(values, age, valid)
        expected, memory = policy_step(
            checkpoint.state.parameters, memory, reading, config, checkpoint.trim_action
        )
        actual, exported_memory = exported.call(values, age, valid, exported_memory)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(exported_memory, memory, rtol=2e-5, atol=2e-6)
        largest_error = max(largest_error, float(jnp.max(jnp.abs(actual - expected))))
    print(
        f"Exported trained {config.architecture} policy at update {int(checkpoint.state.iteration)}"
    )
    print(f"Saved {args.output}; maximum action error {largest_error:.2e}")


if __name__ == "__main__":
    main()
