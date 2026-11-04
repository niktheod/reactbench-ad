"""Original-scene predictions are computed once per model and reused by every edited run (with a provenance check)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CACHE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "evaluation_manifest.json"


@dataclass(frozen=True)
class OriginalCacheCohort:
    """Validated trajectories and exclusions for one edited evaluation."""

    trajectories: dict[str, np.ndarray]
    eligible_tokens: list[str]
    excluded_failed_tokens: list[str]
    manifest_path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_signature(
    *,
    evaluator: str,
    config_path: Path,
    interval_length_s: float,
    inference_options: Mapping[str, Any],
    seed_name: str,
    seed_base: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact inference/source signature required by an edited run."""

    return {
        "evaluator": evaluator,
        "evaluation_config_sha256": file_sha256(config_path),
        "interval_length_s": float(interval_length_s),
        "inference_options": dict(inference_options),
        "seed_name": seed_name,
        "seed_base": int(seed_base),
        "source": dict(source),
    }


def save_original_manifest(
    original_result_dir: Path,
    *,
    signature: Mapping[str, Any],
    source_run_dir: Path | None,
    selected_tokens: Sequence[str],
    successful_tokens: Sequence[str],
    source_contracts: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write the provenance contract for an original-only run."""

    original_result_dir.mkdir(parents=True, exist_ok=True)
    path = original_result_dir / MANIFEST_FILENAME
    temporary = path.with_name(f".{path.name}.tmp")
    document = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "mode": "original",
        "signature": dict(signature),
        "source_run_dir": str(source_run_dir) if source_run_dir is not None else None,
        "selected_tokens": list(selected_tokens),
        "successful_tokens": list(successful_tokens),
        "source_contracts": dict(source_contracts or {}),
    }
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_cached_trajectory(
    path: Path, *, expected_interval_length_s: float
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached original trajectory: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "trajectory" not in archive:
            raise KeyError(f"Cached trajectory has no 'trajectory' array: {path}")
        poses = np.asarray(archive["trajectory"], dtype=np.float32)
        if "interval_length_s" not in archive:
            raise KeyError(
                f"Cached trajectory has no 'interval_length_s' value: {path}"
            )
        stored_interval = float(np.asarray(archive["interval_length_s"]).item())
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 3:
        raise ValueError(f"Invalid cached trajectory shape {poses.shape}: {path}")
    if not np.isfinite(poses).all():
        raise ValueError(f"Cached trajectory contains non-finite values: {path}")
    if not np.isclose(
        stored_interval, float(expected_interval_length_s), atol=1e-6
    ):
        raise ValueError(
            f"Cached trajectory interval {stored_interval} does not match "
            f"{expected_interval_length_s}: {path}"
        )
    return poses


def _load_validated_manifest(
    original_result_dir: Path,
    *,
    expected_signature: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = original_result_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Original cache manifest is missing: {manifest_path}. "
            "Run this evaluator once with --original first."
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {manifest_path}")
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported original cache schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("signature") != dict(expected_signature):
        raise ValueError(
            f"Original cache settings do not match this edited run: "
            f"{manifest_path}. Rerun --original with the same configuration."
        )
    return manifest_path, manifest


def _manifest_token_list(
    manifest: Mapping[str, Any], field: str, manifest_path: Path
) -> list[str]:
    value = manifest.get(field)
    if (
        not isinstance(value, list)
        or not all(isinstance(token, str) and token for token in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"Invalid {field} in {manifest_path}")
    return value


def _source_contract_identity(contract: Any) -> Any:
    """Return the source fields that determine original PAI-AV inputs."""

    if not isinstance(contract, Mapping):
        return contract
    identity = dict(contract)
    if identity.get("version") == "paiav_archive_camera_frames_v1":
        identity.pop("source_reconstruction_file", None)
    return identity


def validate_original_cache(
    original_result_dir: Path,
    *,
    tokens: Sequence[str],
    expected_signature: Mapping[str, Any],
    expected_source_contracts: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Validate an original cache and load all trajectories needed by a run."""

    manifest_path, manifest = _load_validated_manifest(
        original_result_dir,
        expected_signature=expected_signature,
    )
    successful_tokens = _manifest_token_list(
        manifest, "successful_tokens", manifest_path
    )
    missing_tokens = sorted(set(tokens).difference(successful_tokens))
    if missing_tokens:
        raise ValueError(
            f"Original cache is missing {len(missing_tokens)} selected token(s): "
            f"{missing_tokens[:10]}. Rerun --original for the required selection."
        )

    if expected_source_contracts is not None:
        cached_contracts = manifest.get("source_contracts")
        if not isinstance(cached_contracts, dict):
            raise ValueError(f"Invalid source_contracts in {manifest_path}")
        mismatched = [
            token
            for token in tokens
            if _source_contract_identity(cached_contracts.get(token))
            != _source_contract_identity(expected_source_contracts.get(token))
        ]
        if mismatched:
            raise ValueError(
                "Original image provenance differs from the cached baseline for "
                f"{len(mismatched)} token(s): {mismatched[:10]}"
            )

    interval_length_s = float(expected_signature["interval_length_s"])
    trajectory_dir = original_result_dir / "trajectories"
    return {
        token: load_cached_trajectory(
            trajectory_dir / f"{token}.npz",
            expected_interval_length_s=interval_length_s,
        )
        for token in tokens
    }


def validate_original_cache_cohort(
    original_result_dir: Path,
    *,
    tokens: Sequence[str],
    expected_signature: Mapping[str, Any],
    expected_source_contracts: Mapping[str, Any] | None = None,
) -> OriginalCacheCohort:
    """Load the successful original subset while rejecting unattempted tokens.

    Tokens explicitly selected by the completed original run but absent from
    its ``successful_tokens`` list are returned as exclusions. A requested
    token absent from ``selected_tokens`` is an incompatible/incomplete cache
    and remains a hard error.
    """

    manifest_path, manifest = _load_validated_manifest(
        original_result_dir,
        expected_signature=expected_signature,
    )
    selected_tokens = _manifest_token_list(
        manifest, "selected_tokens", manifest_path
    )
    successful_tokens = _manifest_token_list(
        manifest, "successful_tokens", manifest_path
    )
    selected_set = set(selected_tokens)
    successful_set = set(successful_tokens)
    invalid_successes = sorted(successful_set.difference(selected_set))
    if invalid_successes:
        raise ValueError(
            f"Original cache marks {len(invalid_successes)} unselected token(s) "
            f"as successful in {manifest_path}: {invalid_successes[:10]}"
        )

    unattempted_tokens = [token for token in tokens if token not in selected_set]
    if unattempted_tokens:
        raise ValueError(
            f"Original cache never attempted {len(unattempted_tokens)} requested "
            f"token(s): {unattempted_tokens[:10]}. Rerun --original for the "
            "required selection."
        )

    eligible_tokens = [token for token in tokens if token in successful_set]
    excluded_failed_tokens = [
        token for token in tokens if token not in successful_set
    ]
    if not eligible_tokens:
        raise ValueError(
            "None of the requested tokens succeeded in the original cache; "
            "there is no paired cohort to evaluate."
        )

    if expected_source_contracts is not None:
        cached_contracts = manifest.get("source_contracts")
        if not isinstance(cached_contracts, dict):
            raise ValueError(f"Invalid source_contracts in {manifest_path}")
        mismatched = [
            token
            for token in eligible_tokens
            if _source_contract_identity(cached_contracts.get(token))
            != _source_contract_identity(expected_source_contracts.get(token))
        ]
        if mismatched:
            raise ValueError(
                "Original image provenance differs from the cached baseline for "
                f"{len(mismatched)} token(s): {mismatched[:10]}"
            )

    interval_length_s = float(expected_signature["interval_length_s"])
    trajectory_dir = original_result_dir / "trajectories"
    trajectories = {
        token: load_cached_trajectory(
            trajectory_dir / f"{token}.npz",
            expected_interval_length_s=interval_length_s,
        )
        for token in eligible_tokens
    }
    return OriginalCacheCohort(
        trajectories=trajectories,
        eligible_tokens=eligible_tokens,
        excluded_failed_tokens=excluded_failed_tokens,
        manifest_path=manifest_path,
    )
