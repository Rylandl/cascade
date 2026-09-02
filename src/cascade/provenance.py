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


def spec_hash(spec: AircraftSpec) -> str:
    """SHA-256 of the specification's canonical JSON (sorted keys, no whitespace)."""

    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
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
    """The source tree's commit, when running from a checkout; ``None`` otherwise."""

    try:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def stamp(
    spec: AircraftSpec | None = None,
    model: AircraftModel | None = None,
    *,
    seed: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Everything needed to reproduce a result, as a JSON-ready dict."""

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
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


__all__ = ["STAMP_SCHEMA", "git_commit", "model_hash", "spec_hash", "stamp", "write_stamp"]
