"""Helpers shared by the model runners: compositor runs, NAVSIM annotations and saved predictions.

A compositor run directory holds ``final_images/<token>/<camera>/<frame>.png``,
``metadata/<token>.json`` (placement of the inserted object) and ``run_config.json``. Predictions
are saved as ``trajectories/<token>.npz`` with rear-axle x/y/heading poses and their interval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import paths

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
MAX_TORCH_SEED = (1 << 63) - 1


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def resolve_directory(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_dir():
        raise NotADirectoryError(
            f"{label} does not exist or is not a directory: {result}"
        )
    return result


def resolve_file(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file: {result}"
        )
    return result


def load_run_config(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    config_candidates = tuple(
        (run_dir / filename).expanduser().resolve()
        for filename in ("receiver_preview_config.json", "run_config.json")
    )
    config_path = next(
        (path for path in config_candidates if path.is_file()),
        None,
    )
    if config_path is None:
        expected = ", ".join(str(path) for path in config_candidates)
        raise FileNotFoundError(
            f"Run configuration does not exist; expected one of: {expected}"
        )
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    for required_key in ("scene_filter", "data_root"):
        if not config.get(required_key):
            raise KeyError(
                f"Run configuration is missing {required_key!r}: {config_path}"
            )
    return config_path, config


def discover_tokens(
    final_images_dir: Path, run_config: dict[str, Any]
) -> list[str]:
    final_tokens = {
        path.name for path in final_images_dir.iterdir() if path.is_dir()
    }
    if not final_tokens:
        raise ValueError(f"No token directories found under {final_images_dir}")

    configured_tokens = run_config.get("selected_tokens")
    if isinstance(configured_tokens, list):
        ordered = [
            token
            for token in configured_tokens
            if isinstance(token, str) and token in final_tokens
        ]
        ordered.extend(sorted(final_tokens.difference(ordered)))
        return ordered
    return sorted(final_tokens)


def load_annotations_index(path: Path) -> dict[str, dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    by_token: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Expected annotation objects in {path}")
        token = record.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(f"Annotation without a valid token in {path}")
        if token in by_token:
            raise ValueError(f"Duplicate annotation token {token!r} in {path}")
        by_token[token] = record
    return by_token


def resolve_edited_front_image(final_images_dir: Path, token: str) -> Path:
    frame_stem = final_images_dir / token / "CAM_F0" / "3"
    matches = [
        frame_stem.with_suffix(suffix)
        for suffix in SUPPORTED_IMAGE_SUFFIXES
        if frame_stem.with_suffix(suffix).is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"Missing edited current CAM_F0 image for token={token} under "
            f"{final_images_dir}"
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple edited current CAM_F0 images for token={token}: {matches}"
        )
    return matches[0]


def current_speed_mps(annotation: dict[str, Any]) -> float:
    velocities = annotation.get("velocity")
    if not isinstance(velocities, list) or not velocities:
        raise ValueError("Annotation is missing velocity history")
    velocity = np.asarray(velocities[-1], dtype=np.float64)
    if velocity.ndim != 1 or velocity.size < 2:
        raise ValueError(
            "Expected the current velocity to contain at least two values, "
            f"got {velocity.shape}"
        )
    speed = float(np.linalg.norm(velocity[:2]))
    if not np.isfinite(speed):
        raise ValueError(f"Current ego speed is not finite: {velocity}")
    return speed


def trajectory_poses(trajectory: Any) -> np.ndarray:
    poses = np.asarray(
        getattr(trajectory, "poses", trajectory), dtype=np.float32
    )
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
        raise ValueError(
            f"Expected predicted trajectory shape (N, 3), got {poses.shape}"
        )
    if not np.all(np.isfinite(poses)):
        raise ValueError("Predicted trajectory contains non-finite values")
    return poses


def save_prediction(
    trajectory_dir: Path,
    output_stem: str,
    trajectory: Any,
    *,
    interval_length_s: float,
) -> np.ndarray:
    poses = trajectory_poses(trajectory)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trajectory_dir / f"{output_stem}.npz",
        trajectory=poses,
        interval_length_s=np.float32(interval_length_s),
    )
    return poses


def load_requested_tokens(tokens_file: Path, available_tokens: list[str]) -> list[str]:
    """Load an exact, ordered token selection (``{"tokens": [...]}``) and validate its availability."""

    manifest_path = resolve_file(tokens_file, "token manifest")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {manifest_path}")
    tokens = manifest.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"Expected a non-empty 'tokens' list in {manifest_path}")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError(f"Every token in {manifest_path} must be a non-empty string")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate tokens found in {manifest_path}")
    missing_tokens = sorted(set(tokens).difference(available_tokens))
    if missing_tokens:
        raise ValueError(f"Requested tokens are missing from the input source: {missing_tokens}")
    return tokens


def annotations_path(run_config: dict[str, Any]) -> Path:
    """NAVSIM annotations of the run's scene filter, made by ``reactbench.datasets.navsim_annotations``."""
    name = Path(run_config["scene_filter"]).stem
    return resolve_file(paths.get("annotations", paths.outputs("annotations")) / f"{name}.json", "NAVSIM annotations")


def predictions_dir(model: str, dataset: str, run_name: str) -> Path:
    """Where a model's predictions for one run are saved: <outputs>/predictions/<model>/<dataset>/<run>."""
    return paths.outputs("predictions", model, dataset, run_name)


def select_tokens(available: list[str], *, tokens_file: Optional[Path] = None,
                  max_samples: Optional[int] = None) -> list[str]:
    if tokens_file is not None:
        return load_requested_tokens(tokens_file, available)
    if max_samples is not None:
        return available[:max_samples]
    return available


def expand_config(text: str) -> str:
    """Expand the location placeholders used in the model configurations."""
    def location(key, default=None, function=None):
        try:
            return str(function() if function else paths.get(key, default))
        except KeyError:
            return "{" + key + "}"
    replacements = {"{checkpoints}": location("checkpoints", "checkpoints"),
                    "{third_party}": str(paths.REPO_ROOT / "third_party"),
                    "{sensor_blobs_test}": location("sensor_blobs_test", function=lambda: paths.navsim_sensor_blobs("test"))}
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text
