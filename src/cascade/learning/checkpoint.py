"""Atomic, versioned learning checkpoints with JSON metadata and numeric arrays.

Checkpoints contain no pickle or executable Python. Checksums detect accidental
corruption, not malicious replacement. Resume reproducibility is scoped to the
same software, hardware, configuration, and objective; a checkpoint cannot
serialize a Python objective or establish equivalence across JAX versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cascade.env import action_size, observation_layout, observation_size
from cascade.learning.policies import PolicyConfig, make_policy, parameter_shapes
from cascade.learning.training import TrainingConfig, TrainingState

CHECKPOINT_SCHEMA = "cascade_learning_checkpoint_v1"


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must contain finite JSON-compatible values") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _atomic_write(path: str | Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def observation_schema(model, episode_config) -> dict[str, Any]:
    """Describe selected observation blocks and the sensor-aware input contract.

    Target-dependent normalization remains part of the frozen task/experiment
    provenance. The schema fixes block order and dimensions, not a whole task.
    """
    layout = observation_layout(model, episode_config.observation)
    return {
        "schema": "cascade_sensor_observation_v1",
        "size": observation_size(model, episode_config.observation),
        "blocks": {
            name: None if block is None else [block.start, block.stop]
            for name, block in layout._asdict().items()
        },
        "values_dtype": "float32",
        "age_dtype": "float32",
        "age_units": "seconds",
        "valid_dtype": "bool",
        "missing": {"value": 0.0, "age": "positive_infinity", "valid": False},
    }


def action_schema(model, episode_config) -> dict[str, Any]:
    """Describe propeller-first normalized actions and channel scaling."""
    return {
        "schema": "cascade_normalized_action_v1",
        "size": action_size(model),
        "dtype": "float32",
        "bounds": [-1.0, 1.0],
        "propellers": [0, model.n_propellers],
        "channels": [model.n_propellers, action_size(model)],
        "propeller_mapping": "(action + 1) / 2",
        "channel_scale": float(episode_config.channel_scale),
    }


@dataclass(frozen=True)
class Checkpoint:
    """Restored training state and the contracts needed to deploy its policy."""

    state: TrainingState
    policy_config: PolicyConfig
    training_config: TrainingConfig
    trim_action: jax.Array
    observation_schema: dict[str, Any]
    action_schema: dict[str, Any]
    provenance: dict[str, Any]
    metadata: dict[str, Any]

    def make_policy(self):
        """Return the same weights as an observation-only rollout policy and memory."""
        return make_policy(self.state.parameters, self.policy_config, self.trim_action)


def _validate_schema(schema, size: int, name: str) -> None:
    if (
        not isinstance(schema, dict)
        or not schema
        or isinstance(schema.get("size"), bool)
        or schema.get("size") != size
    ):
        raise ValueError(f"{name} must be a JSON object with size={size}")
    _json_bytes(schema)


def _validate_arrays(state, policy_config, trim_action) -> dict[str, np.ndarray]:
    expected = parameter_shapes(policy_config)
    arrays = {}
    for group in ("parameters", "first_moment", "second_moment"):
        tree = getattr(state, group)
        if not isinstance(tree, dict) or set(tree) != set(expected):
            raise ValueError(f"{group} keys do not match policy architecture")
        for name, template in expected.items():
            array = np.asarray(tree[name])
            if array.shape != template or array.dtype != np.dtype("float32"):
                raise ValueError(f"{group}.{name} must have shape {template} and dtype float32")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{group}.{name} contains nonfinite values")
            if group == "second_moment" and np.any(array < 0):
                raise ValueError(f"{group}.{name} must be nonnegative")
            arrays[f"{group}/{name}"] = array
    trim = np.asarray(trim_action)
    if trim.shape != (policy_config.action_size,) or trim.dtype != np.dtype("float32"):
        raise ValueError("trim_action must match action_size and have dtype float32")
    if not np.all(np.isfinite(trim)) or np.any(np.abs(trim) > 1):
        raise ValueError("trim_action must be finite and within [-1, 1]")
    iteration = np.asarray(state.iteration)
    if iteration.shape != () or iteration.dtype != np.dtype("int32") or iteration < 0:
        raise ValueError("iteration must be a nonnegative int32 scalar")
    try:
        key = np.asarray(jax.random.key_data(state.key))
        implementation = jax.random.key_impl(state.key)
        expected_key = jax.random.key_data(jax.random.key(0, impl=implementation))
        if key.shape != expected_key.shape or key.dtype != np.uint32:
            raise ValueError("training requires a single PRNG key")
        # Validate shape against the key implementation, including legacy keys.
        jax.random.wrap_key_data(jnp.asarray(key), impl=implementation)
    except (TypeError, ValueError) as exc:
        raise ValueError("training key must be a valid JAX PRNG key") from exc
    arrays.update(trim_action=trim, iteration=iteration, key=key)
    return arrays


def _validate_checkpoint(checkpoint: Checkpoint) -> None:
    """Reject in-memory mutations that would disconnect exported data from its stamp."""
    metadata = checkpoint.metadata
    body = {k: v for k, v in metadata.items() if k != "metadata_sha256"}
    if metadata.get("metadata_sha256") != _hash(body):
        raise ValueError("checkpoint metadata was changed; save a new checkpoint before export")
    for name in (
        "policy_config",
        "training_config",
        "observation_schema",
        "action_schema",
        "provenance",
    ):
        value = getattr(checkpoint, name)
        if name.endswith("config"):
            value = asdict(value)
        if (
            _hash(value) != metadata["hashes"][name]
            or _hash(metadata[name]) != metadata["hashes"][name]
        ):
            raise ValueError(f"checkpoint {name} was changed; save a new checkpoint before export")
    arrays = _validate_arrays(checkpoint.state, checkpoint.policy_config, checkpoint.trim_action)
    descriptions = {
        name: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": _array_hash(a)}
        for name, a in arrays.items()
    }
    if descriptions != metadata["arrays"]:
        raise ValueError("checkpoint arrays were changed; save a new checkpoint before export")


def save_checkpoint(
    path: str | Path,
    state: TrainingState,
    policy_config: PolicyConfig,
    training_config: TrainingConfig,
    trim_action,
    *,
    observation_schema: dict[str, Any],
    action_schema: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Validate then atomically replace a checkpoint; return its JSON metadata.

    Schemas must include their vector ``size``. Provenance should identify the
    frozen experiment, objective, model, software, and seed. Passing the same
    provenance as ``expected_provenance`` on load prevents accidental resumption
    against a different experiment.
    """
    if not isinstance(state, TrainingState):
        raise ValueError("state must be a TrainingState")
    if not isinstance(policy_config, PolicyConfig) or not isinstance(
        training_config, TrainingConfig
    ):
        raise ValueError("policy_config and training_config must be learning configuration objects")
    _validate_schema(observation_schema, policy_config.observation_size, "observation_schema")
    _validate_schema(action_schema, policy_config.action_size, "action_schema")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")
    arrays = _validate_arrays(state, policy_config, trim_action)
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "policy_config": asdict(policy_config),
        "training_config": asdict(training_config),
        "observation_schema": observation_schema,
        "action_schema": action_schema,
        "provenance": provenance,
        "key": {
            "implementation": str(jax.random.key_impl(state.key)),
            "typed": bool(jax.dtypes.issubdtype(state.key.dtype, jax.dtypes.prng_key)),
        },
        "arrays": {
            name: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": _array_hash(a)}
            for name, a in arrays.items()
        },
    }
    metadata["hashes"] = {
        name: _hash(metadata[name])
        for name in (
            "policy_config",
            "training_config",
            "observation_schema",
            "action_schema",
            "provenance",
        )
    }
    metadata["metadata_sha256"] = _hash(metadata)
    metadata = json.loads(_json_bytes(metadata))
    payload = dict(arrays, metadata=np.frombuffer(_json_bytes(metadata), dtype=np.uint8))
    _atomic_write(path, lambda f: np.savez_compressed(f, **payload))
    return metadata


def load_checkpoint(
    path: str | Path,
    *,
    expected_policy_config: PolicyConfig | None = None,
    expected_training_config: TrainingConfig | None = None,
    expected_observation_schema: dict[str, Any] | None = None,
    expected_action_schema: dict[str, Any] | None = None,
    expected_provenance: dict[str, Any] | None = None,
) -> Checkpoint:
    """Read a checkpoint without pickle and reject corrupt or incompatible data.

    Optional expected contracts are checked before returning any training state.
    File-not-found and filesystem errors retain their usual Python exceptions.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            raw = archive["metadata"]
            if raw.dtype != np.uint8 or raw.ndim != 1:
                raise ValueError("metadata must be a UTF-8 JSON byte vector")
            metadata = json.loads(raw.tobytes())
            if not isinstance(metadata, dict) or metadata.get("schema") != CHECKPOINT_SCHEMA:
                raise ValueError("unsupported checkpoint schema")
            saved_hash = metadata.get("metadata_sha256")
            body = {k: v for k, v in metadata.items() if k != "metadata_sha256"}
            if saved_hash != _hash(body):
                raise ValueError("checkpoint metadata checksum mismatch")
            descriptions = metadata["arrays"]
            if (
                set(archive.files) != {"metadata", *descriptions}
                or len(archive.files) != len(descriptions) + 1
            ):
                raise ValueError("checkpoint array inventory mismatch")
            arrays = {}
            for name, description in descriptions.items():
                array = archive[name]
                if (
                    list(array.shape) != description["shape"]
                    or str(array.dtype) != description["dtype"]
                    or _array_hash(array) != description["sha256"]
                ):
                    raise ValueError(f"checkpoint array checksum or shape mismatch: {name}")
                arrays[name] = array
        for name, expected in (
            (
                "policy_config",
                None if expected_policy_config is None else asdict(expected_policy_config),
            ),
            (
                "training_config",
                None if expected_training_config is None else asdict(expected_training_config),
            ),
            ("observation_schema", expected_observation_schema),
            ("action_schema", expected_action_schema),
            ("provenance", expected_provenance),
        ):
            if metadata["hashes"][name] != _hash(metadata[name]):
                raise ValueError(f"checkpoint {name} checksum mismatch")
            if expected is not None and _hash(expected) != _hash(metadata[name]):
                raise ValueError(f"checkpoint {name} is incompatible with expected {name}")
        config = dict(metadata["policy_config"])
        if config.get("observation_scale") is not None:
            config["observation_scale"] = tuple(config["observation_scale"])
        policy_config = PolicyConfig(**config)
        training_config = TrainingConfig(**metadata["training_config"])
        _validate_schema(
            metadata["observation_schema"], policy_config.observation_size, "observation_schema"
        )
        _validate_schema(metadata["action_schema"], policy_config.action_size, "action_schema")
        if arrays["key"].dtype != np.uint32:
            raise ValueError("checkpoint PRNG key data must have dtype uint32")
        key = jnp.asarray(arrays["key"])
        if metadata["key"]["typed"]:
            key = jax.random.wrap_key_data(key, impl=metadata["key"]["implementation"])
        elif str(jax.random.key_impl(key)) != metadata["key"]["implementation"]:
            raise ValueError(
                "legacy PRNG implementation differs; use the checkpoint's JAX PRNG setting"
            )
        trees = [
            {
                name.split("/", 1)[1]: array
                for name, array in arrays.items()
                if name.startswith(group + "/")
            }
            for group in ("parameters", "first_moment", "second_moment")
        ]
        state = TrainingState(*trees, key, arrays["iteration"])
        expected_arrays = _validate_arrays(state, policy_config, arrays["trim_action"])
        if set(arrays) != set(expected_arrays):
            raise ValueError("checkpoint contains unexpected arrays")
        if not isinstance(metadata["provenance"], dict):
            raise ValueError("provenance must be a JSON object")
        state = jax.tree.map(jnp.asarray, state)
        return Checkpoint(
            state,
            policy_config,
            training_config,
            jnp.asarray(arrays["trim_action"]),
            metadata["observation_schema"],
            metadata["action_schema"],
            metadata["provenance"],
            metadata,
        )
    except (zipfile.BadZipFile, KeyError, TypeError, UnicodeError, EOFError) as exc:
        raise ValueError(f"malformed learning checkpoint: {exc}") from exc


__all__ = [
    "CHECKPOINT_SCHEMA",
    "Checkpoint",
    "action_schema",
    "load_checkpoint",
    "observation_schema",
    "save_checkpoint",
]
