"""Public sensor policies and finite, reproducible optimization contracts."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade.env import SensorObservation
from cascade.learning import (
    PolicyConfig,
    TrainingConfig,
    initial_memory,
    initialize_policy,
    initialize_training,
    make_episode_return_objective,
    make_policy,
    make_train_step,
    policy_step,
    sensor_features,
    train,
)


def reading(values):
    values = jnp.asarray(values, jnp.float32)
    return SensorObservation(values, jnp.zeros_like(values), jnp.ones_like(values, dtype=bool))


def assert_tree_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for lhs, rhs in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(lhs, rhs)


@pytest.mark.parametrize("architecture", ["feedforward", "recurrent"])
def test_policy_starts_exactly_at_trim_and_is_batched_jittable(architecture):
    config = PolicyConfig(3, 2, hidden_size=5, architecture=architecture)
    params = initialize_policy(config, jax.random.PRNGKey(7))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    observations = reading([[1, 0, -3], [2, 1, 4]])
    memory = initial_memory(config, (2,))
    trim = jnp.array([0.1, -0.9], jnp.float32)
    apply = jax.jit(lambda m, o: policy_step(params, m, o, config, trim))
    actions, next_memory = apply(memory, observations)
    np.testing.assert_array_equal(actions, jnp.broadcast_to(trim, (2, 2)))
    vmapped = jax.jit(jax.vmap(apply))(memory, observations)
    assert_tree_equal((actions, next_memory), vmapped)
    assert bool(jnp.isfinite(next_memory).all())


def test_normalization_retains_age_and_validity_and_sanitizes_missing_values():
    config = PolicyConfig(5, 1, observation_scale=(2, 4, 1, 1, 1))
    observation = SensorObservation(
        jnp.array([4.0, -8.0, jnp.nan, 1e30, 5.0]),
        jnp.array([0.2, 0.4, 0.0, jnp.inf, -0.1]),
        jnp.array([True, True, True, False, True]),
    )
    features = jax.jit(lambda obs: sensor_features(obs, config))(observation)
    np.testing.assert_allclose(features[:5], [2, -2, 0, 0, 0])
    np.testing.assert_allclose(features[5:10], [2, 4, 10, 10, 10])
    np.testing.assert_array_equal(features[10:], [1, 1, 0, 0, 0])
    assert bool(jnp.isfinite(features).all())
    # Large finite measurements saturate, while held samples carry their actual age.
    changed = observation._replace(values=observation.values.at[0].set(1e30))
    assert float(sensor_features(changed, config)[0]) == config.observation_clip


def test_finite_extreme_normalization_stays_bounded():
    config = PolicyConfig(2, 1, observation_scale=(1e-30, 1e30), age_scale_s=1e-30)
    obs = SensorObservation(
        jnp.array([1e30, -1e30]), jnp.array([1e30, 0.0]), jnp.array([True, True])
    )
    features = jax.jit(lambda o: sensor_features(o, config))(obs)
    assert bool(jnp.isfinite(features).all())
    assert float(jnp.max(jnp.abs(features))) <= config.observation_clip


@pytest.mark.parametrize("architecture", ["feedforward", "recurrent"])
def test_policy_gradients_actions_and_memory_reset(architecture):
    config = PolicyConfig(3, 2, hidden_size=5, architecture=architecture)
    params = initialize_policy(config, jax.random.PRNGKey(0))
    params["output_weight"] = 0.2 * jnp.ones_like(params["output_weight"])
    obs, memory, trim = reading([0.2, 0.3, -0.4]), initial_memory(config), jnp.zeros(2)
    apply = jax.jit(lambda p, m: policy_step(p, m, obs, config, trim))
    first, next_memory = apply(params, memory)
    repeated, repeated_memory = apply(params, initial_memory(config))
    assert_tree_equal((first, next_memory), (repeated, repeated_memory))
    second, _ = apply(params, next_memory)
    if architecture == "recurrent":
        assert not np.allclose(first, second)
        assert bool(jnp.any(next_memory != 0))
    else:
        np.testing.assert_array_equal(first, second)
        assert bool(jnp.all(next_memory == 0))
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(apply(p, memory)[0])))(params)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(gradient))
    assert float(jnp.linalg.norm(gradient["input_weight"])) > 0
    assert float(jnp.max(jnp.abs(first))) <= 1


def test_sensor_adapter_uses_delivered_metadata_and_never_reads_aircraft():
    config = PolicyConfig(2, 1, hidden_size=2)
    params = initialize_policy(config, jax.random.PRNGKey(1))
    params["input_weight"] = jnp.ones_like(params["input_weight"]) * 0.2
    params["output_weight"] = jnp.ones_like(params["output_weight"])
    policy, memory = make_policy(params, config, jnp.zeros(1))
    state = SimpleNamespace(sensor_age_s=jnp.zeros(2), sensor_valid=jnp.ones(2, dtype=bool))
    action, _ = policy(memory, jnp.ones(2), state)
    stale = SimpleNamespace(sensor_age_s=jnp.full(2, 0.5), sensor_valid=state.sensor_valid)
    stale_action, _ = policy(memory, jnp.ones(2), stale)
    assert not np.allclose(action, stale_action)
    # These states have no aircraft or true position attributes at all.
    expected, _ = policy_step(params, memory, reading([1, 1]), config, jnp.zeros(1))
    np.testing.assert_array_equal(action, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_size": True},
        {"action_size": 0},
        {"hidden_size": -1},
        {"architecture": "unknown"},
        {"observation_scale": (1,)},
        {"observation_scale": (1, 0)},
        {"age_scale_s": 0},
        {"age_clip_s": float("nan")},
        {"observation_clip": float("inf")},
        {"residual_scale": 3},
    ],
)
def test_invalid_policy_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PolicyConfig(**({"observation_size": 2, "action_size": 1} | kwargs))


def test_policy_shape_contract_and_trim_validation():
    config = PolicyConfig(2, 1)
    params = initialize_policy(config, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="trim_action"):
        make_policy(params, config, jnp.array([jnp.nan]))
    with pytest.raises(ValueError, match="memory shape"):
        policy_step(params, jnp.zeros(1), reading([1, 2]), config, jnp.zeros(1))
    with pytest.raises(TypeError, match="SensorObservation"):
        sensor_features(jnp.zeros(2), config)
    with pytest.raises(ValueError, match="validity"):
        sensor_features(reading([1, 2])._replace(valid=jnp.ones(1, dtype=bool)), config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": False},
        {"learning_rate": 0},
        {"gradient_clip": -1},
        {"beta1": 1},
        {"beta2": 1 - 1e-12},
        {"epsilon": float("nan")},
    ],
)
def test_invalid_training_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_adam_ascent_improves_objective_and_resume_is_exact():
    config = TrainingConfig(batch_size=4, learning_rate=0.05, gradient_clip=1)
    initial = initialize_training({"x": jnp.zeros(3)}, jax.random.PRNGKey(8))

    def objective(params, keys):
        targets = 1 + 0.01 * jax.vmap(lambda key: jax.random.normal(key, (3,)))(keys)
        return -jnp.mean((params["x"] - targets) ** 2)

    update = make_train_step(objective, config)
    callback_iterations = []
    uninterrupted, history = train(
        initial,
        update,
        12,
        callback=lambda state, metrics: callback_iterations.append(int(state.iteration)),
    )
    partial, _ = train(initial, update, 5)
    resumed, remainder = train(partial, update, 7)
    assert_tree_equal(uninterrupted, resumed)
    assert callback_iterations == list(range(1, 13)) and len(remainder) == 7
    assert all(bool(metrics.accepted) for metrics in history)
    keys = jax.random.split(jax.random.PRNGKey(900), 8)
    assert (
        float(objective(uninterrupted.parameters, keys))
        > float(objective(initial.parameters, keys)) + 0.5
    )
    assert_tree_equal(train(resumed, update, 0)[0], resumed)


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg"])
def test_training_preserves_typed_random_key_implementation(implementation):
    key = jax.random.key(17, impl=implementation)
    initial = initialize_training({"x": jnp.ones(1)}, key)
    assert initial.key.dtype == key.dtype
    np.testing.assert_array_equal(jax.random.key_data(initial.key), jax.random.key_data(key))
    update = make_train_step(
        lambda params, keys: -jnp.sum(params["x"] ** 2), TrainingConfig(batch_size=2)
    )
    final, metrics = update(initial)
    assert bool(metrics.accepted) and final.key.dtype == key.dtype
    np.testing.assert_array_equal(
        jax.random.key_data(final.key), jax.random.key_data(jax.random.split(key)[0])
    )


def test_mixed_precision_parameter_leaves_keep_their_dtypes():
    prior = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        config = PolicyConfig(2, 1, architecture="recurrent")
        params = initialize_policy(config, jax.random.PRNGKey(1))
        assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(params))
        state = initialize_training(
            {"single": jnp.ones(1, jnp.float32), "double": jnp.ones(1, jnp.float64)},
            jax.random.PRNGKey(1),
        )
        update = make_train_step(
            lambda p, keys: -sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(p)),
            TrainingConfig(),
        )
        final, metrics = update(state)
        assert bool(metrics.accepted)
        for tree in final[:3]:
            assert tree["single"].dtype == jnp.float32
            assert tree["double"].dtype == jnp.float64
    finally:
        jax.config.update("jax_enable_x64", prior)


@pytest.mark.parametrize("key", [jnp.zeros(2), jnp.zeros((2, 2), dtype=jnp.uint32)])
def test_invalid_or_batched_training_key_is_rejected(key):
    with pytest.raises(ValueError, match="single JAX PRNG key"):
        initialize_training({"x": jnp.ones(1)}, key)


@pytest.mark.parametrize("bad_gradient_only", [False, True])
def test_invalid_objective_or_gradient_rejects_without_poisoning_state(bad_gradient_only):
    state = initialize_training({"x": jnp.zeros(1)}, jax.random.PRNGKey(4))

    def objective(params, keys):
        del keys
        if bad_gradient_only:
            return jnp.sqrt(params["x"])[0]  # Finite value with an infinite derivative.
        return params["x"][0] * jnp.nan

    update = make_train_step(objective, TrainingConfig())
    rejected, metrics = update(state)
    assert not bool(metrics.accepted)
    if bad_gradient_only:
        assert bool(jnp.isfinite(metrics.objective))
    assert_tree_equal(rejected[:3], state[:3])
    assert int(rejected.iteration) == 0
    assert not np.array_equal(rejected.key, state.key)
    calls = []
    with pytest.raises(FloatingPointError, match="rejected") as caught:
        train(state, update, 2, callback=lambda *args: calls.append(args))
    assert not calls
    assert_tree_equal(caught.value.state, rejected)


def test_large_finite_gradient_is_clipped_before_adam_squaring():
    state = initialize_training({"x": jnp.zeros(3)}, jax.random.PRNGKey(3))
    update = make_train_step(
        lambda params, keys: jnp.sum(params["x"] * 1e30),
        TrainingConfig(gradient_clip=1, learning_rate=0.01),
    )
    next_state, metrics = update(state)
    assert bool(metrics.accepted) and bool(jnp.isfinite(metrics.gradient_norm))
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(next_state))
    np.testing.assert_allclose(next_state.parameters["x"], 0.01, atol=1e-6)
    # The normalized gradient's squared norm is one before the beta2 weight.
    assert float(jnp.sum(next_state.second_moment["x"])) == pytest.approx(0.001, rel=2e-5)


@pytest.mark.parametrize("beta1,beta2", [(0.9, 0.999), (0.99999997, 0), (0, 0.99999995)])
def test_adam_first_step_bias_correction_matches_moment_rounding(beta1, beta2):
    state = initialize_training({"x": jnp.zeros(1)}, jax.random.PRNGKey(0))
    config = TrainingConfig(learning_rate=0.1, beta1=beta1, beta2=beta2)
    update = make_train_step(lambda p, keys: p["x"][0], config)
    final, metrics = update(state)
    assert bool(metrics.accepted)
    np.testing.assert_allclose(final.parameters["x"], 0.1, atol=1e-7)


def test_nonfinite_optimizer_candidate_is_rejected():
    state = initialize_training({"x": jnp.array([3e38])}, jax.random.PRNGKey(3))
    update = make_train_step(
        lambda params, keys: params["x"][0], TrainingConfig(learning_rate=3e38)
    )
    candidate, metrics = update(state)
    assert not bool(metrics.accepted)
    assert_tree_equal(candidate.parameters, state.parameters)
    assert_tree_equal(candidate.first_moment, state.first_moment)


@pytest.fixture(scope="module")
def aircraft_setup():
    from cascade.env import EpisodeConfig, onboard_observation, tracking_task, trimmed_reference
    from cascade.reference import aerobatic_reference

    model = aerobatic_reference()
    task = tracking_task(12, 50, 0)
    reference = trimmed_reference(model, task)
    config = EpisodeConfig(horizon_steps=40, observation=onboard_observation())
    return model, config, task, reference


@pytest.mark.slow
def test_learning_improves_a_short_aircraft_return(aircraft_setup):
    from cascade.env import action_size, control_to_action, observation_size

    model, episode, task, reference = aircraft_setup
    policy_config = PolicyConfig(
        observation_size(model, episode.observation), action_size(model), hidden_size=8
    )
    params = initialize_policy(policy_config, jax.random.PRNGKey(0))
    objective = make_episode_return_objective(
        model,
        episode,
        task,
        reference,
        policy_config,
        control_to_action(episode, reference.control),
    )
    # Fixed declared training realizations isolate optimization from Monte Carlo variation.
    # Generalization is assessed by the separate workflow's disjoint evaluation split.
    training_keys = jax.random.split(jax.random.PRNGKey(10), 4)
    fixed_objective = lambda p, keys: objective(p, training_keys)  # noqa: E731
    update = make_train_step(fixed_objective, TrainingConfig(batch_size=4, learning_rate=0.003))
    initial = initialize_training(params, jax.random.PRNGKey(1))
    final, history = train(initial, update, 8)
    after = float(jax.jit(objective)(final.parameters, training_keys))
    before = float(history[0].objective)
    assert all(bool(jnp.isfinite(m.gradient_norm)) and bool(m.accepted) for m in history)
    assert float(history[0].gradient_norm) > 0
    assert after > before + 0.01


@pytest.mark.slow
def test_recurrent_scheduled_objective_accepts_sensor_weather_and_faults(aircraft_setup):
    from cascade.env import (
        SensorBlockConfig,
        SensorPipelineConfig,
        action_size,
        control_to_action,
        fault_schedule,
        observation_size,
        scheduled_tracking_task,
        sensor_noise,
    )
    from cascade.env.weather import weather_condition

    model, config, _, reference = aircraft_setup
    config = replace(
        config,
        horizon_steps=12,
        sensors=SensorPipelineConfig(
            rates=SensorBlockConfig(delay_steps=1, dropout_probability=0.2),
            airspeed=SensorBlockConfig(sample_period_steps=2),
        ),
    )
    task = scheduled_tracking_task([0, 0.3], [12, 12.2], [50, 50.1], [0, 0.03])
    policy_config = PolicyConfig(
        observation_size(model, config.observation),
        action_size(model),
        hidden_size=4,
        architecture="recurrent",
    )
    params = initialize_policy(policy_config, jax.random.PRNGKey(2))
    params["output_weight"] = 0.01 * jnp.ones_like(params["output_weight"])
    objective = make_episode_return_objective(
        model,
        config,
        task,
        reference,
        policy_config,
        control_to_action(config, reference.control),
        noise=sensor_noise(rate_std=0.01),
        weather=weather_condition(0.5, 0.2),
        faults=fault_schedule(model, partial_power={0: (0.1, 0.9)}),
    )
    keys = jax.random.split(jax.random.PRNGKey(20), 2)
    value, gradient = jax.jit(jax.value_and_grad(objective))(params, keys)
    assert 0 < float(value) <= config.horizon_steps
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(gradient))
    assert float(jnp.linalg.norm(gradient["recurrent_weight"])) > 0
