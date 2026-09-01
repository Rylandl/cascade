from __future__ import annotations

from typing import TypeAlias

from jax import Array

ArrayLike: TypeAlias = Array | float | int
BatchShape: TypeAlias = tuple[int, ...]
