"""Portable JAX export bundles for trained, sensor-aware learning policies.

An export bundles a serialized ``jax.export.Exported`` and its input/output
contract in one atomic ZIP file. This validates the JAX round trip on compatible
platforms; it does not qualify an onboard runtime or promise cross-JAX support.
The optional ``cascade-flight[export]`` extra is needed only when exporting or
loading these bundles.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from cascade.env import SensorObservation
from cascade.learning.checkpoint import (
    Checkpoint,
    _atomic_write,
    _hash,
    _json_bytes,
    _validate_checkpoint,
    _validate_schema,
    load_checkpoint,
)
from cascade.learning.policies import PolicyConfig, initial_memory, policy_step
from cascade.learning.training import TrainingConfig

EXPORT_SCHEMA = "cascade_learning_export_v1"


def _jax_export():
    try:
        from jax import export
    except ImportError as exc:
        raise ImportError("policy export requires the 'cascade-flight[export]' extra") from exc
    return export


def _input_contract(config):
    return [
        {"name": "values", "shape": [config.observation_size], "dtype": "float32"},
        {"name": "age_s", "shape": [config.observation_size], "dtype": "float32"},
        {"name": "valid", "shape": [config.observation_size], "dtype": "bool"},
        {"name": "memory", "shape": [config.hidden_size], "dtype": "float32"},
    ]


def _output_contract(config):
    return [
        {"name": "action", "shape": [config.action_size], "dtype": "float32"},
        {"name": "memory", "shape": [config.hidden_size], "dtype": "float32"},
    ]


@dataclass(frozen=True)
class ExportedPolicy:
    """A loaded JAX export with explicit measurement and recurrent-state inputs."""

    exported: Any
    metadata: dict[str, Any]

    def call(self, values, age_s, valid, memory):
        """Return ``(action, next_memory)`` for one sensor packet.

        Values, ages, and memory use float32; validity uses bool. Reset memory at
        every episode boundary, including truncation. Missing readings may use
        infinite age with false validity, matching the environment contract.
        """
        return self.exported.call(values, age_s, valid, memory)

    def initial_memory(self):
        """Allocate zero policy memory for a new episode."""
        return jnp.zeros((self.metadata["policy_config"]["hidden_size"],), dtype=jnp.float32)

    def make_policy(self):
        """Adapt the export to ordinary sensor-aware environment rollouts."""
        from cascade.env import sensor_policy

        def policy(memory, observation):
            return self.call(observation.values, observation.age_s, observation.valid, memory)

        return sensor_policy(policy), self.initial_memory()


def export_policy(checkpoint: Checkpoint | str | Path, path: str | Path) -> dict[str, Any]:
    """Export saved learned weights, metadata, and explicit recurrent memory.

    ``path`` is a single ZIP bundle, conventionally ending in ``.cascade-policy``.
    Policy parameters and the trim action are constants inside the artifact.
    """
    if not isinstance(checkpoint, Checkpoint):
        checkpoint = load_checkpoint(checkpoint)
    _validate_checkpoint(checkpoint)
    config = checkpoint.policy_config

    def act(values, age_s, valid, memory):
        return policy_step(
            checkpoint.state.parameters,
            memory,
            SensorObservation(values, age_s, valid),
            config,
            checkpoint.trim_action,
        )

    exported = _jax_export().export(jax.jit(act))(
        jax.ShapeDtypeStruct((config.observation_size,), jnp.float32),
        jax.ShapeDtypeStruct((config.observation_size,), jnp.float32),
        jax.ShapeDtypeStruct((config.observation_size,), jnp.bool_),
        jax.ShapeDtypeStruct(initial_memory(config).shape, jnp.float32),
    )
    artifact = bytes(exported.serialize())
    metadata = {
        "schema": EXPORT_SCHEMA,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "checkpoint_metadata_sha256": checkpoint.metadata["metadata_sha256"],
        "policy_config": checkpoint.metadata["policy_config"],
        "training_config": checkpoint.metadata["training_config"],
        "observation_schema": checkpoint.observation_schema,
        "action_schema": checkpoint.action_schema,
        "provenance": checkpoint.provenance,
        "hashes": checkpoint.metadata["hashes"],
        "training_iteration": int(checkpoint.state.iteration),
        "jax_version": jax.__version__,
        "platforms": list(exported.platforms),
        "inputs": _input_contract(config),
        "outputs": _output_contract(config),
        "memory_reset": "zero at every episode boundary (terminated or truncated)",
    }
    metadata["metadata_sha256"] = _hash(metadata)

    def write(f):
        with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", _json_bytes(metadata))
            archive.writestr("policy.jax", artifact)

    _atomic_write(path, write)
    return metadata


def load_exported_policy(path: str | Path) -> ExportedPolicy:
    """Load a bundle after checking its schema, checksums, and argument contracts."""
    try:
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ["metadata.json", "policy.jax"]:
                raise ValueError(
                    "export bundle must contain metadata.json and policy.jax exactly once"
                )
            metadata = json.loads(archive.read("metadata.json"))
            artifact = archive.read("policy.jax")
        if not isinstance(metadata, dict) or metadata.get("schema") != EXPORT_SCHEMA:
            raise ValueError("unsupported policy export schema")
        body = {k: v for k, v in metadata.items() if k != "metadata_sha256"}
        if metadata.get("metadata_sha256") != _hash(body):
            raise ValueError("policy export metadata checksum mismatch")
        if metadata.get("artifact_sha256") != hashlib.sha256(artifact).hexdigest():
            raise ValueError("policy export artifact checksum mismatch")
        policy_config = PolicyConfig(**metadata["policy_config"])
        TrainingConfig(**metadata["training_config"])
        for name in (
            "policy_config",
            "training_config",
            "observation_schema",
            "action_schema",
            "provenance",
        ):
            if metadata["hashes"][name] != _hash(metadata[name]):
                raise ValueError(f"policy export {name} checksum mismatch")
        _validate_schema(
            metadata["observation_schema"], policy_config.observation_size, "observation_schema"
        )
        _validate_schema(metadata["action_schema"], policy_config.action_size, "action_schema")
        if not isinstance(metadata["provenance"], dict):
            raise ValueError("policy export provenance must be a JSON object")
        iteration = metadata["training_iteration"]
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("policy export training_iteration must be a nonnegative integer")
        if metadata["inputs"] != _input_contract(policy_config) or metadata[
            "outputs"
        ] != _output_contract(policy_config):
            raise ValueError(
                "policy export input/output contracts do not match policy configuration"
            )
        exported = _jax_export().deserialize(artifact)
        if metadata["platforms"] != list(exported.platforms):
            raise ValueError("policy export platforms differ from metadata")
        for avals, records in (
            (exported.in_avals, metadata["inputs"]),
            (exported.out_avals, metadata["outputs"]),
        ):
            if len(avals) != len(records):
                raise ValueError("policy export argument count differs from metadata")
            for aval, record in zip(avals, records, strict=True):
                if list(aval.shape) != record["shape"] or str(aval.dtype) != record["dtype"]:
                    raise ValueError("policy export argument contract differs from metadata")
        return ExportedPolicy(exported, metadata)
    except (zipfile.BadZipFile, KeyError, TypeError, UnicodeError) as exc:
        raise ValueError(f"malformed policy export: {exc}") from exc


__all__ = ["EXPORT_SCHEMA", "ExportedPolicy", "export_policy", "load_exported_policy"]
