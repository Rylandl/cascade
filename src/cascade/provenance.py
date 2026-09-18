"""Configuration stamps for results: what code, what model, what numerics, what seed.

A result without its stamp cannot be reproduced or traced to a build. :func:`stamp` collects
the package version, git commit (when the source tree is at hand), JAX and jaxlib versions,
the default backend and platform, the x64 setting, hashes of the specification and of the
compiled model's leaves, the seed, and a timestamp, as a plain dict ready for JSON.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import jax
import numpy as np

from cascade.model import AircraftModel
from cascade.spec import AircraftSpec

STAMP_SCHEMA = "cascade_stamp_v1"
_RESERVED_FIELDS = {
    "schema",
    "timestamp_utc",
    "cascade_version",
    "git_commit",
    "jax_version",
    "jaxlib_version",
    "backend",
    "platform",
    "python",
    "x64_enabled",
    "seed",
    "spec_name",
    "spec_hash",
    "model_hash",
}


def spec_hash(spec: AircraftSpec) -> str:
    """SHA-256 of the specification's canonical JSON (sorted keys, no whitespace)."""

    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def model_hash(model: AircraftModel) -> str:
    """SHA-256 over the compiled model's leaves (shape, dtype, and bytes), in tree order."""

    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(model):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    """The source checkout's HEAD, or ``None`` for an installed distribution.

    Only a checkout containing this module under ``src/cascade`` is eligible. In
    particular, a virtual environment inside another repository must not inherit
    that repository's commit. HEAD identifies the base commit, not uncommitted edits.
    """

    try:
        module_path = Path(__file__).resolve()
        root = module_path.parents[2]
        if module_path != root / "src" / "cascade" / "provenance.py":
            return None
        if not (root / ".git").exists():
            return None
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(lines) != 2 or Path(lines[0]).resolve() != root:
        return None
    return lines[1]


def stamp(
    spec: AircraftSpec | None = None,
    model: AircraftModel | None = None,
    *,
    seed: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Record environment and model identifiers as a JSON-ready dict.

    Custom fields may add experiment context but cannot overwrite reserved fields,
    including optional model/specification hashes even when those inputs are absent.
    Values must be JSON serializable and numeric values must be finite. A stamp does
    not bundle the model, inputs, external datasets, or uncommitted source changes;
    those must be preserved separately to reproduce an experiment.
    """

    reserved = _RESERVED_FIELDS.intersection(extra)
    if reserved:
        raise ValueError(f"reserved provenance fields: {', '.join(sorted(reserved))}")
    try:
        version = metadata.version("cascade-flight")
    except metadata.PackageNotFoundError:
        version = "unknown"
    record: dict[str, Any] = {
        "schema": STAMP_SCHEMA,
        "timestamp_utc": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
        "cascade_version": version,
        "git_commit": git_commit(),
        "jax_version": jax.__version__,
        "jaxlib_version": metadata.version("jaxlib") if _installed("jaxlib") else None,
        "backend": jax.default_backend(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "seed": seed,
    }
    if spec is not None:
        record["spec_name"] = spec.name
        record["spec_hash"] = spec_hash(spec)
    if model is not None:
        record["model_hash"] = model_hash(model)
    record.update(extra)
    json.dumps(record, allow_nan=False)
    return record


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def write_stamp(path: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Write :func:`stamp` as JSON next to a result and return it."""

    record = stamp(*args, **kwargs)
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


__all__ = ["STAMP_SCHEMA", "git_commit", "model_hash", "spec_hash", "stamp", "write_stamp"]
