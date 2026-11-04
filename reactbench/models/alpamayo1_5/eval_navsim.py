"""Run Alpamayo 1.5 on a ReactNAV compositor run (three front cameras, four history frames).

    python -m reactbench.models.alpamayo1_5.eval_navsim RUN_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo1_5.eval_navsim RUN_DIR               # edited predictions + scoring

NAVSIM's front-left, front and front-right cameras are mapped to Alpamayo's cross-left, front-wide and
cross-right inputs (frames 0.5 s apart instead of 0.1 s); the four NAVSIM ego poses are linearly
interpolated to the 16 poses at 10 Hz the model expects, and NAVSIM's driving command is passed as a
navigation instruction. Predictions go to ``<outputs>/predictions/alpamayo1_5/<dataset>/<run>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml

from ...scoring.actor_motion import inserted_actor_motion
from ...scoring.geometry import project_acceleration_onto_velocity
from ...scoring.motion_bins import NAVSIM_BIN_SPEC_ID
from ...scoring.records import run_summary, sample_record
from ..common import (MAX_TORCH_SEED, SUPPORTED_IMAGE_SUFFIXES, annotations_path, current_speed_mps,
                      discover_tokens, expand_config, load_annotations_index, load_json, load_run_config,
                      predictions_dir, resolve_directory, resolve_file, select_tokens, trajectory_poses)
from ..original_cache import cache_signature, save_original_manifest, validate_original_cache

MODEL_NAME = "alpamayo1_5"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_navsim.yaml")
CAMERA_SPECS = (
    ("front_left_camera", "CAM_L0"),
    ("front_camera", "CAM_F0"),
    ("front_right_camera", "CAM_R0"),
)

FRAMES_PER_CAMERA = 4

CURRENT_FRAME_INDEX = 3

def diffusion_seed_for_token(token: str, base_seed: int) -> int:
    """Derive a stable per-token seed shared by its original/edited pair."""

    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    base_seed = int(base_seed)
    if not 0 <= base_seed <= MAX_TORCH_SEED:
        raise ValueError(
            "base_seed must be between 0 and 2**63 - 1, "
            f"got {base_seed}"
        )

    material = f"alpamayo-diffusion:{base_seed}:{token}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big") >> 1


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, indent=2)
        handle.write("\n")


def original_camera_paths(
    annotation: dict[str, Any],
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for annotation_key, _ in CAMERA_SPECS:
        values = annotation.get(annotation_key)
        if not isinstance(values, list) or len(values) != FRAMES_PER_CAMERA:
            raise ValueError(
                f"Expected {FRAMES_PER_CAMERA} paths in {annotation_key!r}"
            )
        paths = [Path(value).expanduser() for value in values]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing original images in {annotation_key!r}: {missing}"
            )
        result[annotation_key] = paths
    return result


def resolve_edited_frame(
    final_images_dir: Path,
    token: str,
    camera: str,
    frame_index: int,
) -> Path:
    frame_stem = final_images_dir / token / camera / str(frame_index)
    matches = [
        frame_stem.with_suffix(suffix)
        for suffix in SUPPORTED_IMAGE_SUFFIXES
        if frame_stem.with_suffix(suffix).is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"Missing edited image for token={token}, camera={camera}, "
            f"frame={frame_index} under {final_images_dir}"
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple edited images for token={token}, camera={camera}, "
            f"frame={frame_index}: {matches}"
        )
    return matches[0]


def edited_camera_paths(
    final_images_dir: Path, token: str
) -> dict[str, list[Path]]:
    return {
        annotation_key: [
            resolve_edited_frame(final_images_dir, token, camera, frame_index)
            for frame_index in range(FRAMES_PER_CAMERA)
        ]
        for annotation_key, camera in CAMERA_SPECS
    }


def metric_trajectory(
    poses: np.ndarray,
    *,
    interval_length_s: float,
    metric_horizon_s: float | None,
) -> np.ndarray:
    """Return the common-horizon prefix used by robustness metrics."""

    poses = trajectory_poses(poses)
    if metric_horizon_s is None:
        return poses
    horizon = float(metric_horizon_s)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError(
            f"metric_horizon_s must be finite and positive, got {horizon}"
        )
    count = int(round(horizon / interval_length_s))
    if count <= 0 or not np.isclose(
        count * interval_length_s, horizon, atol=1e-6
    ):
        raise ValueError(
            "metric_horizon_s must be an integer multiple of "
            f"interval_length_s ({interval_length_s})"
        )
    if count > poses.shape[0]:
        raise ValueError(
            f"Metric horizon requires {count} poses, but only "
            f"{poses.shape[0]} were predicted"
        )
    return poses[:count]


def make_input(annotation: dict[str, Any], camera_paths: dict[str, list[Path]]) -> dict[str, Any]:
    return {
        "camera_paths": camera_paths,
        "history_poses": annotation["past_trajectory"],
        "navigation_command": annotation["navigation_command"],
    }


def save_prediction(trajectory_dir: Path, token: str, prediction: Any, *, interval_length_s: float) -> np.ndarray:
    poses = trajectory_poses(prediction)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_dir / f"{token}.npz", trajectory=poses,
                        interval_length_s=np.float32(interval_length_s))
    return poses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Compositor run (final_images/, metadata/, run_config.json)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default="reactnav")
    parser.add_argument("--original", action="store_true", help="Run and cache only original-image predictions")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-samples", type=int, help="Evaluate only the first N tokens (smoke test)")
    selection.add_argument("--tokens-file", type=Path, help="JSON with the exact ordered 'tokens' to evaluate")
    parser.add_argument("--scoring", choices=("score", "off"), default="score")
    parser.add_argument("--scoring-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    from hydra.utils import instantiate
    from ...scoring.navsim import postprocess_evaluation
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args(argv)
    run_dir = resolve_directory(args.run_dir, "run directory")
    config_path = resolve_file(args.config, "Alpamayo configuration")
    cfg = yaml.safe_load(expand_config(config_path.read_text()))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    final_images_dir = resolve_directory(run_dir / "final_images", "final edited images")
    run_config_path, run_config = load_run_config(run_dir)
    annotations_file = annotations_path(run_config)
    annotations_by_token = load_annotations_index(annotations_file)

    result_dir = predictions_dir(MODEL_NAME, args.dataset, "original" if args.original else run_dir.name)
    original_result_dir = predictions_dir(MODEL_NAME, args.dataset, "original")
    trajectory_dir, reasoning_dir = result_dir / "trajectories", result_dir / "reasoning"
    failures_path = result_dir / "failed_samples.json"
    result_dir.mkdir(parents=True, exist_ok=True)

    tokens = select_tokens(discover_tokens(final_images_dir, run_config),
                           tokens_file=args.tokens_file, max_samples=args.max_samples)
    interval_length_s = float(cfg.get("trajectory_sampling", {}).get("interval_length", 0.1))
    metric_horizon_s = float(cfg.get("metric_horizon_s", 4.0))
    diffusion_seed_base = int(cfg.get("diffusion_seed_base", 42))
    signature = cache_signature(
        evaluator="alpamayo-navsim", config_path=config_path, interval_length_s=interval_length_s,
        inference_options={}, seed_name="diffusion_seed_base", seed_base=diffusion_seed_base,
        source={"annotations_path": str(annotations_file)},
    )
    cached_originals = ({} if args.original else
                        validate_original_cache(original_result_dir, tokens=tokens, expected_signature=signature))

    print(f"Evaluating {len(tokens)} {'original' if args.original else 'edited'} samples from {run_dir}", flush=True)
    agent = instantiate(cfg)
    agent.initialize()
    failed_samples: list[dict[str, Any]] = []
    successful_tokens: list[str] = []
    metric_records: list[dict[str, Any]] = []
    for sample_index, token in enumerate(tokens, start=1):
        stage = "load_input"
        seed: Optional[int] = None
        try:
            seed = diffusion_seed_for_token(token, agent.diffusion_seed_base)
            if token not in annotations_by_token:
                raise KeyError(f"Token {token!r} is absent from {annotations_file}")
            annotation = annotations_by_token[token]
            if args.original:
                stage = "original_inference"
                prediction = agent.compute_trajectory(make_input(annotation, original_camera_paths(annotation)),
                                                      diffusion_seed=seed)
                save_prediction(trajectory_dir, token, prediction, interval_length_s=interval_length_s)
                save_json(reasoning_dir / f"{token}.json", prediction.extra)
                successful_tokens.append(token)
            else:
                edited_paths = edited_camera_paths(final_images_dir, token)
                speed_mps = current_speed_mps(annotation)
                acceleration = project_acceleration_onto_velocity(annotation["velocity"][-1], annotation["acceleration"][-1])
                metadata_path = run_dir / "metadata" / f"{token}.json"
                actor_motion = inserted_actor_motion(load_json(metadata_path))
                original_poses = cached_originals[token]
                stage = "edited_inference"
                prediction = agent.compute_trajectory(make_input(annotation, edited_paths), diffusion_seed=seed)
                edited_poses = save_prediction(trajectory_dir, token, prediction, interval_length_s=interval_length_s)
                save_json(reasoning_dir / f"{token}.json", prediction.extra)
                stage = "metrics"
                original_metric = metric_trajectory(original_poses, interval_length_s=interval_length_s,
                                                    metric_horizon_s=metric_horizon_s)
                edited_metric = metric_trajectory(edited_poses, interval_length_s=interval_length_s,
                                                  metric_horizon_s=metric_horizon_s)
                record = sample_record(bin_spec=NAVSIM_BIN_SPEC_ID, token=token, output_stem=token,
                                       original_trajectory=original_metric, edited_trajectory=edited_metric,
                                       actor_motion=actor_motion, interval_length_s=interval_length_s,
                                       speed_mps=speed_mps, longitudinal_acceleration_mps2=acceleration)
                record.update(placement_metadata_path=str(metadata_path), diffusion_seed=seed,
                              native_pose_count=int(original_poses.shape[0]),
                              metric_pose_count=int(original_metric.shape[0]),
                              metric_horizon_s=float(original_metric.shape[0]) * interval_length_s)
                metric_records.append(record)
            print(f"Completed {sample_index}/{len(tokens)} token={token}", flush=True)
        except Exception as exc:
            failed_samples.append({"index": sample_index, "token": token, "stage": stage, "diffusion_seed": seed,
                                   "error_type": type(exc).__name__, "error": str(exc),
                                   "traceback": traceback.format_exc()})
            save_json(failures_path, failed_samples)
            print(f"Failed {sample_index}/{len(tokens)} token={token} stage={stage}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)

    save_json(failures_path, failed_samples)
    if args.original:
        manifest_path = save_original_manifest(result_dir, signature=signature, source_run_dir=run_dir,
                                               selected_tokens=tokens, successful_tokens=successful_tokens)
        print(f"Saved {len(successful_tokens)} original predictions to {result_dir}; "
              f"{len(failed_samples)} samples failed. Manifest: {manifest_path}", flush=True)
        postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)
        return
    metrics_dir = result_dir / "metrics"
    save_json(metrics_dir / "per_sample.json", metric_records)
    summary = run_summary(metric_records)
    summary.update({
        "model": MODEL_NAME, "dataset": args.dataset, "run_dir": str(run_dir),
        "run_config_path": str(run_config_path), "annotations_path": str(annotations_file),
        "final_images_dir": str(final_images_dir), "result_dir": str(result_dir),
        "original_result_dir": str(original_result_dir), "evaluation_config_path": str(config_path),
        "model_name_or_path": cfg.get("model_name_or_path"), "attn_implementation": cfg.get("attn_implementation"),
        "vlm_seed": agent.seed, "diffusion_seed_base": agent.diffusion_seed_base,
        "diffusion_seed_strategy": "SHA-256 of alpamayo-diffusion:<base>:<token>; first 63 bits shared by each pair",
        "interval_length_s": interval_length_s, "metric_horizon_s": metric_horizon_s,
        "selected_token_count": len(tokens), "successful_pair_count": len(metric_records),
        "failed_sample_count": len(failed_samples),
        "input_contract": {"cameras": ["CAM_L0", "CAM_F0", "CAM_R0"], "frames_per_camera": FRAMES_PER_CAMERA,
                           "navsim_image_interval_s": 0.5, "alpamayo_nominal_image_interval_s": 0.1,
                           "history_resampling": "4x0.5s NAVSIM poses to 16x0.1s"},
    })
    save_json(metrics_dir / "summary.json", summary)
    print(f"Saved {len(metric_records)} paired results to {result_dir}; {len(failed_samples)} samples failed.", flush=True)
    postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)


if __name__ == "__main__":
    main()
