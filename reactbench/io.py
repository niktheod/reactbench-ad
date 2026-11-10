"""JSON and hashing helpers shared by all stages."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from pathlib import Path


def digest(value) -> str:
    """SHA-256 of a JSON-serializable value (keys sorted)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def jsonable(value):
    """Plain JSON values from numpy arrays and scalars, tensors, paths and tuples, recursively."""
    import numpy as np
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def finite_json(value):
    """Replace non-finite floats by None, recursively."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, value):
    """Atomic replacement through a sibling staging file on the destination filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.publishing")
    try:
        staging.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def relative(value):
    """Absolute paths (values and keys) reduced to their last two components, recursively (for released files)."""
    if isinstance(value, dict):
        return {relative(k): relative(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relative(v) for v in value]
    if isinstance(value, str) and value.startswith("/") and "\n" not in value:
        return "/".join(Path(value).parts[-2:])
    return value
