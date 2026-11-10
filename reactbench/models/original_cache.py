"""Original-scene predictions are computed once per model and reused by every edited run (with a provenance check).

``evaluation_manifest.json`` of a run records its cache signature (model, configuration, seeds, inputs), the
selected and successful tokens and, where a loader provides them, per-token source contracts of the original
inputs. An edited run accepts an original run only with the same signature and source contracts; tokens whose
original prediction failed are left out of its pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import file_digest, jsonable, read_json, write_json

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "evaluation_manifest.json"


@dataclass(frozen=True)
class OriginalCacheCohort:
    """The cached original trajectories of the pairs an edited run can form."""

    trajectories: dict[str, np.ndarray]
    eligible_tokens: list[str]
    excluded_failed_tokens: list[str]
    manifest_path: Path


def cache_signature(*, evaluator: str, config_path: Path, interval_length_s: float,
                    inference_options: Mapping[str, Any], seed_name: str, seed_base: int,
                    source: Mapping[str, Any]) -> dict[str, Any]:
    """The inference and input settings an original cache must share with an edited run."""
    return {"evaluator": evaluator, "evaluation_config_sha256": file_digest(config_path),
            "interval_length_s": float(interval_length_s), "inference_options": jsonable(dict(inference_options)),
            "seed_name": seed_name, "seed_base": int(seed_base), "source": jsonable(dict(source))}


def save_manifest(result_dir: Path, *, mode: str, signature: Mapping[str, Any], source_run_dir: Path | None,
                  selected_tokens: Sequence[str], successful_tokens: Sequence[str],
                  source_contracts: Mapping[str, Any] | None = None, **extra: Any) -> Path:
    """Write ``evaluation_manifest.json`` of a finished run (``mode``: original, paired, cot_swap, fixed_cots)."""
    path = Path(result_dir) / MANIFEST_FILENAME
    write_json(path, jsonable({
        "schema_version": SCHEMA_VERSION, "mode": mode, "signature": dict(signature),
        "source_run_dir": None if source_run_dir is None else str(source_run_dir),
        "selected_tokens": list(selected_tokens), "successful_tokens": list(successful_tokens),
        "source_contracts": dict(source_contracts or {}), **extra}))
    return path


def load_cached_trajectory(path: Path, *, expected_interval_length_s: float) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached trajectory: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "trajectory" not in archive or "interval_length_s" not in archive:
            raise KeyError(f"Cached trajectory needs 'trajectory' and 'interval_length_s': {path}")
        poses = np.asarray(archive["trajectory"], dtype=np.float32)
        stored_interval = float(np.asarray(archive["interval_length_s"]).item())
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 3:
        raise ValueError(f"Invalid cached trajectory shape {poses.shape}: {path}")
    if not np.isfinite(poses).all():
        raise ValueError(f"Cached trajectory contains non-finite values: {path}")
    if not np.isclose(stored_interval, float(expected_interval_length_s), atol=1e-6):
        raise ValueError(f"Cached trajectory interval {stored_interval} does not match "
                         f"{expected_interval_length_s}: {path}")
    return poses


def source_contract_identity(contract: Any) -> Any:
    """The fields of a source contract that determine the original inputs (not where the index file lies)."""
    if not isinstance(contract, Mapping):
        return contract
    identity = jsonable(dict(contract))
    if identity.get("version") == "paiav_archive_camera_frames_v1":
        identity.pop("source_reconstruction_file", None)
    return identity


def token_list(manifest: Mapping[str, Any], field: str, path: Path) -> list[str]:
    value = manifest.get(field)
    if (not isinstance(value, list) or not all(isinstance(token, str) and token for token in value)
            or len(value) != len(set(value))):
        raise ValueError(f"Invalid {field} in {path}")
    return value


def read_manifest(result_dir: Path, *, mode: str, signature: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """A finished run's manifest; its mode and signature must match."""
    path = Path(result_dir) / MANIFEST_FILENAME
    manifest = read_json(path)
    if manifest is None:
        raise FileNotFoundError(f"No finished {mode} run: {path} is missing" +
                                (" (run the evaluator with --original first)" if mode == "original" else ""))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("mode") != mode:
        raise ValueError(f"{path} is not the manifest of a finished {mode} run")
    expected = jsonable(dict(signature))
    found = manifest.get("signature") or {}
    cached = (found.get("inference_options") or {}).get("hardware_class")
    wanted = (expected.get("inference_options") or {}).get("hardware_class")
    if cached != wanted:
        raise ValueError(f"The {mode} run in {result_dir} was computed on {cached}, this job runs on {wanted}; "
                         "seeds reproduce a rollout only on the same GPU architecture")
    if found != expected:
        raise ValueError(f"The settings of the {mode} run in {result_dir} differ from this run's "
                         "(model, configuration, seeds or inputs)")
    if not set(token_list(manifest, "successful_tokens", path)) <= set(token_list(manifest, "selected_tokens", path)):
        raise ValueError(f"{path} marks unselected tokens as successful")
    return path, manifest


def load_original_cache(original_result_dir: Path, *, tokens: Sequence[str], signature: Mapping[str, Any],
                        source_contracts: Mapping[str, Any] | None = None) -> OriginalCacheCohort:
    """The cached original trajectories of ``tokens``.

    A token the original run never attempted is an error (rerun ``--original`` on that selection); a token whose
    original prediction failed is excluded from the pairs.
    """
    path, manifest = read_manifest(original_result_dir, mode="original", signature=signature)
    selected = set(manifest["selected_tokens"])
    successful = set(manifest["successful_tokens"])
    unattempted = [token for token in tokens if token not in selected]
    if unattempted:
        raise ValueError(f"The original run never attempted {len(unattempted)} requested token(s): "
                         f"{unattempted[:10]}. Rerun --original for the required selection.")
    eligible = [token for token in tokens if token in successful]
    if not eligible:
        raise ValueError("None of the requested tokens succeeded in the original run; there are no pairs")
    if source_contracts is not None:
        cached = manifest.get("source_contracts")
        if not isinstance(cached, dict):
            raise ValueError(f"Invalid source_contracts in {path}")
        mismatched = [token for token in eligible if source_contract_identity(cached.get(token))
                      != source_contract_identity(source_contracts.get(token))]
        if mismatched:
            raise ValueError(f"The original inputs of {len(mismatched)} token(s) differ from the cached run's: "
                             f"{mismatched[:10]}")
    interval = float(signature["interval_length_s"])
    trajectories = {token: load_cached_trajectory(Path(original_result_dir) / "trajectories" / f"{token}.npz",
                                                  expected_interval_length_s=interval) for token in eligible}
    return OriginalCacheCohort(trajectories, eligible, [token for token in tokens if token not in successful], path)
