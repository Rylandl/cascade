"""Learn a tracking policy by gradient through the dynamics.

A small policy network is initialised at the trim action and trained by ascending the episode
return with the gradient taken straight through :func:`cascade.env.rollout_policy`: the plant,
the actuators, the stall dynamics, and the policy are one differentiable program. On the
aerobatic reference's 12 m/s tracking task from perturbed starts, the example compares sixty
training steps with the hand-tuned control cascade on separate evaluation seeds. This is one
seeded simulator experiment, not a general sample-efficiency claim.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from cascade.control import aerobatic_reference_controller
from cascade.env import (
    EpisodeConfig,
    action_size,
    cascade_policy,
    control_to_action,
    observation_size,
    reset,
    rollout_policy,
    tracking_task,
    trimmed_reference,
)
from cascade.provenance import stamp
from cascade.reference import aerobatic_reference_spec

HORIZON = 160
BATCH = 16
ITERATIONS = 60
HIDDEN = 32
LEARNING_RATE = 3e-3
GRADIENT_CLIP = 10.0


def initial_parameters(key, observation_size, action_size_):
    scale_key, _ = jax.random.split(key)
    return {
        "w1": 0.1 * jax.random.normal(scale_key, (observation_size, HIDDEN)),
        "b1": jnp.zeros(HIDDEN),
        "w2": jnp.zeros((HIDDEN, action_size_)),
        "b2": jnp.zeros(action_size_),
    }


def policy_network(params, observation, trim_action):
    hidden = jnp.tanh(observation @ params["w1"] + params["b1"])
    return jnp.clip(trim_action + 0.5 * jnp.tanh(hidden @ params["w2"] + params["b2"]), -1.0, 1.0)


def clip_gradient(gradient):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(gradient)))
    scale = jnp.minimum(1.0, GRADIENT_CLIP / (norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, gradient), norm


def adam_ascent(params, first, second, gradient, iteration):
    first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, gradient)
    second = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, second, gradient)
    first_hat = jax.tree.map(lambda m: m / (1.0 - 0.9**iteration), first)
    second_hat = jax.tree.map(lambda v: v / (1.0 - 0.999**iteration), second)
    params = jax.tree.map(
        lambda p, m, v: p + LEARNING_RATE * m / (jnp.sqrt(v) + 1e-8), params, first_hat, second_hat
    )
    return params, first, second


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write results and provenance as JSON")
    args = parser.parse_args()
    spec = aerobatic_reference_spec()
    model = spec.to_model()
    task = tracking_task(12.0, 50.0, 0.0)
    reference = trimmed_reference(model, task)
    config = EpisodeConfig(horizon_steps=HORIZON)
    trim_action = control_to_action(config, reference.control)
    observations = observation_size(model, config.observation)

    def episode_return(policy, policy_state, key):
        state, _ = reset(model, config, task, reference, key)
        _, (_, _, rewards, _) = rollout_policy(
            model, config, task, reference, state, policy, policy_state
        )
        return jnp.sum(rewards)

    def mean_return(params, keys):
        def policy(policy_state, observation, env_state):
            return policy_network(params, observation, trim_action), policy_state

        return jnp.mean(jax.vmap(lambda key: episode_return(policy, None, key))(keys))

    def baseline_return(keys):
        policy, policy_state = cascade_policy(
            aerobatic_reference_controller(), model, config, task, reference
        )
        return jnp.mean(jax.vmap(lambda key: episode_return(policy, policy_state, key))(keys))

    evaluation_keys = jax.random.split(jax.random.PRNGKey(999), 256)
    params = initial_parameters(jax.random.PRNGKey(0), observations, action_size(model))
    evaluate = jax.jit(mean_return)
    print(f"horizon {HORIZON} steps at {config.control_frequency_hz:.0f} Hz; max return {HORIZON}")
    baseline = float(jax.jit(baseline_return)(evaluation_keys))
    untrained = float(evaluate(params, evaluation_keys))
    print(f"control cascade baseline: {baseline:7.2f}")
    print(f"untrained policy (trim):  {untrained:7.2f}")

    value_and_grad = jax.jit(jax.value_and_grad(mean_return))
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    start = time.time()
    for iteration in range(1, ITERATIONS + 1):
        keys = jax.random.split(jax.random.PRNGKey(iteration), BATCH)
        value, gradient = value_and_grad(params, keys)
        gradient, norm = clip_gradient(gradient)
        params, first, second = adam_ascent(params, first, second, gradient, iteration)
        if iteration % 10 == 0 or iteration == 1:
            print(
                f"iteration {iteration:3d}  training return {float(value):7.2f}  "
                f"gradient norm {float(norm):7.2f}  [{time.time() - start:3.0f} s]"
            )
    trained = float(evaluate(params, evaluation_keys))
    print(f"trained policy:           {trained:7.2f}")
    if args.output:
        record = {
            "provenance": stamp(spec, model, seed=0),
            "config": {
                "horizon": HORIZON,
                "training_batch": BATCH,
                "iterations": ITERATIONS,
                "hidden": HIDDEN,
                "learning_rate": LEARNING_RATE,
                "gradient_clip": GRADIENT_CLIP,
                "training_keys": "PRNGKey(iteration), iterations 1..60",
                "evaluation_seed": 999,
                "evaluation_episodes": 256,
                "control_frequency_hz": config.control_frequency_hz,
                "simulation_frequency_hz": config.simulation_frequency_hz,
            },
            "returns": {"trim": untrained, "cascade": baseline, "learned": trained},
            "training_and_final_evaluation_seconds": time.time() - start,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
