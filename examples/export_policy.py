"""Export a policy for deployment with ``jax.export`` and check it against the JAX call.

The learned tracking policy of ``learn_tracking_policy.py`` is a pure function of its
parameters and the observation; ``jax.export`` lowers it to StableHLO, a serialised artifact
that runs without Python or JAX at inference (through the XLA or IREE runtimes, or by
conversion to ONNX). This example exports a policy with a fixed observation size, reloads the
artifact, and checks the outputs agree bit for bit, which is the sim-versus-onboard check a
deployment needs before a first flight. Serialisation needs the ``flatbuffers`` package
(``uv run --with flatbuffers python examples/export_policy.py``; it is in the dev group).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from cascade.env import (
    EpisodeConfig,
    action_size,
    control_to_action,
    observation_size,
    tracking_task,
    trimmed_reference,
)
from cascade.reference import aerobatic_reference

HIDDEN = 32


def policy_network(params, observation, trim_action):
    hidden = jnp.tanh(observation @ params["w1"] + params["b1"])
    return jnp.clip(trim_action + 0.5 * jnp.tanh(hidden @ params["w2"] + params["b2"]), -1.0, 1.0)


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("renders") / "policy.stablehlo"
    model = aerobatic_reference()
    task = tracking_task(12.0, 50.0, 0.0)
    reference = trimmed_reference(model, task)
    config = EpisodeConfig()
    trim_action = control_to_action(config, reference.control)
    key = jax.random.PRNGKey(0)
    params = {
        "w1": 0.1 * jax.random.normal(key, (observation_size(model), HIDDEN)),
        "b1": jnp.zeros(HIDDEN),
        "w2": 0.05 * jax.random.normal(jax.random.fold_in(key, 1), (HIDDEN, action_size(model))),
        "b2": jnp.zeros(action_size(model)),
    }

    def act(observation):
        return policy_network(params, observation, trim_action)

    exported = export.export(jax.jit(act))(
        jax.ShapeDtypeStruct((observation_size(model),), jnp.float32)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(exported.serialize())
    reloaded = export.deserialize(output.read_bytes())
    observation = jax.random.normal(jax.random.fold_in(key, 2), (observation_size(model),))
    expected = np.asarray(act(observation))
    actual = np.asarray(reloaded.call(observation))
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    print(f"observation size {observation_size(model)}, action size {action_size(model)}")
    print(f"max |exported - jax| = {np.max(np.abs(actual - expected)):.2e}")
    print("platforms:", exported.platforms)


if __name__ == "__main__":
    main()
