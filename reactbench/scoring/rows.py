"""Read a scored run's per-sample rows after checking them against their manifest and inputs."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..io import file_digest, read_json


@lru_cache(maxsize=65536)
def _prediction_digest(path, size, modified_ns, changed_ns):
    """Reuse hashes of unchanged archives shared by conditions and placements."""
    return file_digest(path)


def read_rows(root, prefix):
    root = Path(root)
    path = root / "metrics" / f"{prefix}_per_sample.json"
    manifest = read_json(path.with_name(f"{prefix}_manifest.json"), {})
    if not path.is_file():
        return None, manifest
    if manifest.get("per_sample_sha256") != file_digest(path):
        raise ValueError(f"Metric checksum mismatch: {path}")
    if manifest.get("exported"):
        # Released scores (reactbench.export): the predictions they were computed from are not part of them.
        return read_json(path), manifest
    for source, sha in manifest.get("source_documents", {}).items():
        if not Path(source).is_file() or file_digest(source) != sha:
            raise ValueError(f"Metrics are stale relative to inference document: {source}")
    rows = read_json(path)
    seen = set()
    for row in rows:
        for source in row.get("prediction_sources", {}).values():
            filename, expected = source.get("path"), source.get("sha256")
            if not filename or not expected or (filename, expected) in seen:
                continue
            prediction = Path(filename)
            if not prediction.is_file():
                raise ValueError(f"Scored trajectory is missing: {prediction}")
            stat = prediction.stat()
            actual = _prediction_digest(str(prediction), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if actual != expected:
                raise ValueError(f"Metrics are stale relative to saved trajectory: {prediction}")
            seen.add((filename, expected))
    return rows, manifest
