"""Checkpoint integrity, exact optimizer continuation, and trained export agreement."""

import json
import zipfile
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade.env import EpisodeConfig, SensorObservation, onboard_observation
from cascade.learning.checkpoint import (
    action_schema,
    load_checkpoint,
    observation_schema,
    save_checkpoint,
)
from cascade.learning.export import export_policy, load_exported_policy
from cascade.learning.policies import PolicyConfig, initial_memory, initialize_policy, policy_step
from cascade.learning.training import TrainingConfig, initialize_training, make_train_step, train
from cascade.reference import aerobatic_reference


@pytest.fixture(params=["feedforward", "recurrent"])
def learning_case(request):
    config = PolicyConfig(
        3, 2, hidden_size=5, architecture=request.param, observation_scale=(1.0, 2.0, 3.0)
    )
    training = TrainingConfig(batch_size=4, learning_rate=0.01)
    trim = jnp.array([0.1, -0.2], dtype=jnp.float32)
    params = initialize_policy(config, jax.random.PRNGKey(41))
    state = initialize_training(params, jax.random.PRNGKey(42))

    def objective(parameters, keys):
        def episode(key):
            values = jax.random.normal(key, (3,), dtype=jnp.float32)
            packet = SensorObservation(values, jnp.zeros(3), jnp.ones(3, dtype=bool))
            action, memory = policy_step(parameters, initial_memory(config), packet, config, trim)
            next_action, _ = policy_step(parameters, memory, packet, config, trim)
            target = 0.3 * jnp.tanh(values[:2])
            return -jnp.sum((action - target) ** 2 + (next_action - target) ** 2)

        return jnp.mean(jax.vmap(episode)(keys))

    update = make_train_step(objective, training)
    metadata = {
        "observation_schema": {"schema": "test_packet_v1", "size": 3, "blocks": ["a", "b", "c"]},
        "action_schema": {"schema": "test_action_v1", "size": 2, "channels": ["left", "right"]},
        "provenance": {"experiment_sha256": "f" * 64, "seed": 42, "objective": "test-quadratic-v1"},
    }
    return config, training, trim, state, update, metadata


def _write_case(path, case, state=None):
    config, training, trim, initial, _, metadata = case
    return save_checkpoint(
        path, initial if state is None else state, config, training, trim, **metadata
    )


def _assert_state_equal(actual, expected):
    for name in ("parameters", "first_moment", "second_moment"):
        for left, right in zip(
            jax.tree.leaves(getattr(actual, name)),
            jax.tree.leaves(getattr(expected, name)),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(
        jax.random.key_data(actual.key), jax.random.key_data(expected.key)
    )
    np.testing.assert_array_equal(actual.iteration, expected.iteration)


def test_checkpoint_resumes_real_optimizer_exactly(tmp_path, learning_case):
    config, training, trim, state, update, metadata = learning_case
    uninterrupted, _ = train(state, update, 7)
    partial, _ = train(state, update, 3)
    path = tmp_path / "policy.npz"
    saved = _write_case(path, learning_case, partial)
    checkpoint = load_checkpoint(
        path,
        expected_policy_config=config,
        expected_training_config=training,
        expected_observation_schema=metadata["observation_schema"],
        expected_action_schema=metadata["action_schema"],
        expected_provenance=metadata["provenance"],
    )
    assert checkpoint.metadata == saved
    assert checkpoint.policy_config == config
    assert checkpoint.training_config == training
    np.testing.assert_array_equal(checkpoint.trim_action, trim)
    resumed, _ = train(checkpoint.state, update, 4)
    _assert_state_equal(resumed, uninterrupted)
    assert int(resumed.iteration) == 7
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(resumed.first_moment))
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind in "biuf" for name in archive.files)


def test_typed_key_preserves_implementation(tmp_path, learning_case):
    state = learning_case[3]._replace(key=jax.random.key(42, impl="threefry2x32"))
    path = tmp_path / "typed-key.npz"
    _write_case(path, learning_case, state)
    restored = load_checkpoint(path).state
    assert jax.dtypes.issubdtype(restored.key.dtype, jax.dtypes.prng_key)
    assert str(jax.random.key_impl(restored.key)) == str(jax.random.key_impl(state.key))
    _assert_state_equal(restored, state)


@pytest.mark.parametrize(
    "contract",
    ["policy_config", "training_config", "observation_schema", "action_schema", "provenance"],
)
def test_incompatible_checkpoint_contracts_rejected(tmp_path, learning_case, contract):
    path = tmp_path / "policy.npz"
    _write_case(path, learning_case)
    config, training, _, _, _, metadata = learning_case
    expected = {
        "policy_config": replace(config, hidden_size=7),
        "training_config": replace(training, learning_rate=0.03),
        "observation_schema": dict(metadata["observation_schema"], blocks=["b", "a", "c"]),
        "action_schema": dict(metadata["action_schema"], channels=["right", "left"]),
        "provenance": dict(metadata["provenance"], seed=43),
    }[contract]
    with pytest.raises(ValueError, match=f"{contract} is incompatible"):
        load_checkpoint(path, **{f"expected_{contract}": expected})


def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mutate(arrays)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


@pytest.mark.parametrize("damage", ["checksum", "shape", "missing", "schema", "metadata", "object"])
def test_corrupt_checkpoints_rejected(tmp_path, learning_case, damage):
    path = tmp_path / "policy.npz"
    _write_case(path, learning_case)

    def mutate(arrays):
        if damage == "checksum":
            arrays["parameters/output_bias"] = arrays["parameters/output_bias"] + 0.1
        elif damage == "shape":
            arrays["parameters/output_bias"] = np.zeros(3, dtype=np.float32)
        elif damage == "missing":
            del arrays["key"]
        elif damage == "object":
            arrays["key"] = np.array([{"payload": "never unpickle"}], dtype=object)
        else:
            record = json.loads(arrays["metadata"].tobytes())
            if damage == "schema":
                record["schema"] = "future-schema"
            else:
                record["provenance"]["seed"] = 100
            arrays["metadata"] = np.frombuffer(json.dumps(record).encode(), dtype=np.uint8)

    _rewrite_npz(path, mutate)
    with pytest.raises(ValueError):
        load_checkpoint(path)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "second_moment", "trim", "iteration"])
def test_invalid_save_preserves_existing_checkpoint(tmp_path, learning_case, invalid):
    path = tmp_path / "policy.npz"
    _write_case(path, learning_case)
    before = path.read_bytes()
    config, training, trim, state, _, metadata = learning_case
    if invalid in {"shape", "nonfinite"}:
        parameters = dict(state.parameters)
        parameters["output_bias"] = (
            jnp.zeros(3) if invalid == "shape" else jnp.array([jnp.nan, 0.0])
        )
        state = state._replace(parameters=parameters)
    elif invalid == "second_moment":
        state = state._replace(second_moment=jax.tree.map(lambda x: x - 1, state.second_moment))
    elif invalid == "trim":
        trim = trim + 3
    else:
        state = state._replace(iteration=jnp.array(-1, dtype=jnp.int32))
    with pytest.raises(ValueError):
        save_checkpoint(path, state, config, training, trim, **metadata)
    assert path.read_bytes() == before


def test_interrupted_save_preserves_existing_checkpoint(tmp_path, learning_case, monkeypatch):
    import cascade.learning.checkpoint as checkpoint_module

    path = tmp_path / "policy.npz"
    _write_case(path, learning_case)
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        _write_case(path, learning_case)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_trained_sensor_policy_export_roundtrip(tmp_path, learning_case):
    pytest.importorskip("flatbuffers")
    config, _, trim, state, update, _ = learning_case
    trained, _ = train(state, update, 5)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(trained.parameters), jax.tree.leaves(state.parameters), strict=True
        )
    )
    checkpoint_path = tmp_path / "trained.npz"
    _write_case(checkpoint_path, learning_case, trained)
    output = tmp_path / "trained.cascade-policy"
    metadata = export_policy(checkpoint_path, output)
    loaded = load_exported_policy(output)
    checkpoint = load_checkpoint(checkpoint_path)
    rollout_policy, rollout_memory = checkpoint.make_policy()
    export_rollout, export_memory = loaded.make_policy()
    assert loaded.metadata == metadata
    assert metadata["training_iteration"] == 5
    expected_memory = initial_memory(config)
    actual_memory = loaded.initial_memory()
    for index in range(5):
        values = jax.random.normal(jax.random.PRNGKey(index), (3,), dtype=jnp.float32)
        valid = jnp.array([True, index % 2 == 0, False])
        ages = jnp.array([index * 0.02, 0.1, jnp.inf], dtype=jnp.float32)
        packet = SensorObservation(values, ages, valid)
        action, expected_memory = policy_step(
            trained.parameters, expected_memory, packet, config, trim
        )
        actual, actual_memory = loaded.call(values, ages, valid, actual_memory)
        sensor_state = SimpleNamespace(sensor_age_s=ages, sensor_valid=valid)
        rollout_action, rollout_memory = rollout_policy(rollout_memory, values, sensor_state)
        export_action, export_memory = export_rollout(export_memory, values, sensor_state)
        np.testing.assert_allclose(actual, action, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(rollout_action, action, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(export_action, action, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual_memory, expected_memory, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(loaded.initial_memory(), jnp.zeros(config.hidden_size))


def test_export_rejects_mutated_checkpoint(tmp_path, learning_case):
    path = tmp_path / "policy.npz"
    _write_case(path, learning_case)
    checkpoint = load_checkpoint(path)
    checkpoint.state.parameters["output_bias"] = jnp.ones(2, dtype=jnp.float32)
    with pytest.raises(ValueError, match="arrays were changed"):
        export_policy(checkpoint, tmp_path / "policy.cascade-policy")


def test_wrong_dtype_is_not_silently_converted_on_load(tmp_path, learning_case):
    import hashlib

    path = tmp_path / "policy.npz"
    _write_case(path, learning_case)

    def mutate(arrays):
        key = "parameters/output_bias"
        arrays[key] = arrays[key].astype(np.float64)
        record = json.loads(arrays["metadata"].tobytes())
        record["arrays"][key]["dtype"] = "float64"
        record["arrays"][key]["sha256"] = hashlib.sha256(arrays[key].tobytes()).hexdigest()
        del record["metadata_sha256"]
        body = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        record["metadata_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        arrays["metadata"] = np.frombuffer(json.dumps(record).encode(), dtype=np.uint8)

    _rewrite_npz(path, mutate)
    with pytest.raises(ValueError, match="dtype float32"):
        load_checkpoint(path)


def test_export_corruption_is_detected(tmp_path, learning_case):
    pytest.importorskip("flatbuffers")
    checkpoint_path = tmp_path / "trained.npz"
    _write_case(checkpoint_path, learning_case)
    output = tmp_path / "trained.cascade-policy"
    export_policy(checkpoint_path, output)
    with zipfile.ZipFile(output) as archive:
        metadata = archive.read("metadata.json")
        artifact = archive.read("policy.jax")
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("metadata.json", metadata)
        archive.writestr("policy.jax", artifact + b"damaged")
    with pytest.raises(ValueError, match="artifact checksum"):
        load_exported_policy(output)


@pytest.mark.parametrize(
    "damage, message",
    [
        ("nested_hash", "provenance checksum"),
        ("policy_config", "hidden_size"),
        ("memory_size", "contracts do not match"),
        ("observation_schema", "observation_schema"),
        ("field_names", "contracts do not match"),
        ("training_config", "beta1"),
        ("platforms", "platforms differ"),
        ("avals", "argument contract differs"),
    ],
)
def test_export_contracts_checked_beyond_checksums(tmp_path, learning_case, damage, message):
    import hashlib

    pytest.importorskip("flatbuffers")
    path = tmp_path / "policy.npz"
    _write_case(path, learning_case)
    output = tmp_path / "policy.cascade-policy"
    export_policy(path, output)
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        artifact = archive.read("policy.jax")
    if damage == "nested_hash":
        metadata["provenance"]["seed"] = 100
    elif damage == "policy_config":
        metadata["policy_config"]["hidden_size"] = 0
    elif damage in {"memory_size", "avals"}:
        metadata["policy_config"]["hidden_size"] = 6
        if damage == "avals":
            metadata["inputs"][3]["shape"] = [6]
            metadata["outputs"][1]["shape"] = [6]
    elif damage == "observation_schema":
        metadata["observation_schema"]["size"] = 99
    elif damage == "field_names":
        metadata["inputs"][0]["name"] = "age_s"
    elif damage == "training_config":
        metadata["training_config"]["beta1"] = 1.0
    else:
        metadata["platforms"] = ["fake-platform"]

    def digest(value):
        body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(body.encode()).hexdigest()

    if damage != "nested_hash":
        metadata["hashes"] = {name: digest(metadata[name]) for name in metadata["hashes"]}
    del metadata["metadata_sha256"]
    metadata["metadata_sha256"] = digest(metadata)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("policy.jax", artifact)
    with pytest.raises(ValueError, match=message):
        load_exported_policy(output)


def test_environment_contract_schemas():
    model = aerobatic_reference()
    config = EpisodeConfig(observation=onboard_observation(), channel_scale=0.5)
    observations = observation_schema(model, config)
    actions = action_schema(model, config)
    assert observations["size"] == 15
    assert observations["blocks"]["air_angles"] is None
    assert observations["blocks"]["airspeed"] == [0, 1]
    assert observations["missing"]["age"] == "positive_infinity"
    assert actions["size"] == model.n_propellers + model.n_control_channels
    assert actions["channel_scale"] == 0.5
    assert actions["propellers"] == [0, model.n_propellers]
