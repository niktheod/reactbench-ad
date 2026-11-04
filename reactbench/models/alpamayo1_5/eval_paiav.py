"""Run Alpamayo 1.5 on a compact ReactPAI run: original and edited scenes.

    python -m reactbench.models.alpamayo1_5.eval_paiav RUN_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo1_5.eval_paiav RUN_DIR               # edited predictions + scoring

The CoT interventions (CoT swap, fixed CoTs) are ``python -m reactbench.analysis.cot.run_alpamayo1_5``.

Inputs: four 10 Hz frames of the cross-left, front-wide, cross-right and front-tele cameras and the
native 16-step ego history; no navigation command. Original images are decoded from the PAI-AV
archives; edited ones come from the run's final_images/. Predictions go to
``<outputs>/predictions/alpamayo1_5/<dataset>/<run>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

from ...datasets.paiav import CAMERA_NAMES, PAIAVCompactRun
from ...scoring.actor_motion import actor_motion_from_metadata
from ...scoring.paiav import postprocess_evaluation as postprocess_collisions
from ...scoring.records import run_summary, sample_record
from ..common import expand_config, predictions_dir
from ..original_cache import cache_signature, save_original_manifest, validate_original_cache
from .inference import AlpamayoInference

MODEL_NAME = "alpamayo1_5"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_paiav.yaml")


def parse_args(
    argv: Sequence[str] | None = None, *, default_config: Path = DEFAULT_CONFIG_PATH
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Alpamayo on archive-reconstructed originals (--original) or, "
            "by default, edited images using a cached original baseline."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Compact PAI-AV run directory")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--original",
        action="store_true",
        help="Run and cache only original-image predictions.",
    )
    sample_selection = parser.add_mutually_exclusive_group()
    sample_selection.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N selected clips for a smoke test.",
    )
    sample_selection.add_argument(
        "--tokens-file",
        type=Path,
        help=(
            "JSON manifest containing the exact ordered token subset under "
            "a 'tokens' key."
        ),
    )
    parser.add_argument(
        "--original-result-dir",
        type=Path,
        help=(
            "Read the compatible original cache from this directory instead "
            "of the selected output root. Intended for isolated recoveries."
        ),
    )
    parser.add_argument("--dataset", default="reactpai")
    parser.add_argument("--scoring", choices=("score", "off"), default="score")
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.original and args.original_result_dir is not None:
        parser.error("--original-result-dir cannot be used with --original")
    return args


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(jsonable(value), handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def load_requested_tokens(
    tokens_file: Path, available_tokens: Sequence[str]
) -> list[str]:
    """Load an exact, ordered recovery cohort and validate its availability."""

    manifest_path = resolve_file(tokens_file, "token manifest")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {manifest_path}")
    tokens = manifest.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(
            f"Expected a non-empty 'tokens' list in {manifest_path}"
        )
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError(
            f"Every token in {manifest_path} must be a non-empty string"
        )
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate tokens found in {manifest_path}")
    missing = sorted(set(tokens).difference(available_tokens))
    if missing:
        raise ValueError(
            f"Requested tokens are missing from the compact run: {missing}"
        )
    return tokens


def make_input(ego_history: Mapping[str, Any], camera_paths: dict[str, list[Path]]) -> dict[str, Any]:
    return {"camera_paths": camera_paths, "navigation_command": None, "alpamayo_ego_history": dict(ego_history)}


def trajectory_poses(trajectory: Any) -> np.ndarray:
    poses = np.asarray(getattr(trajectory, "poses", trajectory), dtype=np.float32)
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
        raise ValueError("metric_horizon_s must be finite and positive")
    count = int(round(horizon / interval_length_s))
    if count <= 0 or not np.isclose(
        count * interval_length_s, horizon, atol=1e-6
    ):
        raise ValueError(
            "metric_horizon_s must be an integer multiple of interval_length_s"
        )
    if count > poses.shape[0]:
        raise ValueError(
            f"Metric horizon requires {count} poses, but only {poses.shape[0]} "
            "were predicted"
        )
    return poses[:count]


def result_name_for_run(run_dir: Path, *, original: bool = False) -> str:
    return "original" if original else run_dir.name


def main(argv: Sequence[str] | None = None) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    config_path = resolve_file(args.config, "Alpamayo evaluation configuration")
    tokens_file = (
        None
        if args.tokens_file is None
        else resolve_file(args.tokens_file, "token manifest")
    )
    cfg = yaml.safe_load(expand_config(config_path.read_text()))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    inference = AlpamayoInference.from_config(cfg)
    trajectory_config = cfg.get("trajectory_sampling", {})
    interval_length_s = float(trajectory_config.get("interval_length", 0.1))
    metric_horizon_value = cfg.get("metric_horizon_s")
    metric_horizon_s = (
        None if metric_horizon_value is None else float(metric_horizon_value)
    )
    result_dir = predictions_dir(MODEL_NAME, args.dataset, result_name_for_run(run_dir, original=args.original))
    default_original_result_dir = predictions_dir(MODEL_NAME, args.dataset, result_name_for_run(run_dir, original=True))
    original_result_dir = (
        default_original_result_dir
        if args.original_result_dir is None
        else args.original_result_dir.expanduser().resolve()
    )
    trajectory_dir = result_dir / "trajectories"
    reasoning_dir = result_dir / "reasoning"
    metrics_dir = result_dir / "metrics"
    failures_path = result_dir / "failed_samples.json"
    result_dir.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(
        evaluator=inference.evaluator,
        config_path=config_path,
        interval_length_s=interval_length_s,
        inference_options=inference.options({}),
        seed_name=f"{inference.seed_key}_base",
        seed_base=inference.seed_base,
        source={"dataset": "pai-av", "dataset_group": run_dir.parent.name},
    )

    from hydra.utils import instantiate

    failed_samples: list[dict[str, Any]] = []
    successful_tokens: list[str] = []
    metric_records: list[dict[str, Any]] = []
    with PAIAVCompactRun(run_dir) as loader:
        available_tokens = list(loader.tokens)
        if tokens_file is not None:
            tokens = load_requested_tokens(tokens_file, available_tokens)
        elif args.max_samples is not None:
            tokens = available_tokens[: args.max_samples]
        else:
            tokens = available_tokens
        source_contracts = {
            token: loader.original_source_contract(token) for token in tokens
        }
        cached_originals = (
            {}
            if args.original
            else validate_original_cache(
                original_result_dir,
                tokens=tokens,
                expected_signature=signature,
                expected_source_contracts=source_contracts,
            )
        )
        mode_name = "original" if args.original else "edited"
        print(f"Evaluating {len(tokens)} PAI-AV {mode_name} samples from {run_dir}", flush=True)
        agent = instantiate(cfg)
        agent.initialize()

        for sample_index, token in enumerate(tokens, start=1):
            output_stem: Optional[str] = None
            pair_seed: Optional[int] = None
            source_contract: Optional[dict[str, Any]] = None
            stage = "load_input"
            try:
                output_stem = token
                pair_seed = inference.seed_for_token(token)
                ego_history, history_source = loader.ego_history(token)
                source_contract = source_contracts[token]
                if args.original:
                    with loader.materialize_original_images(
                        token
                    ) as original_paths:
                        stage = "original_inference"
                        original_prediction = agent.compute_trajectory(
                            make_input(
                                ego_history,
                                original_paths,
                            ),
                            **inference.seed_kwargs(pair_seed),
                        )
                        save_prediction(
                            trajectory_dir,
                            output_stem,
                            original_prediction,
                            interval_length_s=interval_length_s,
                        )
                        save_json(
                            reasoning_dir / f"{output_stem}.json",
                            original_prediction.extra,
                        )
                    successful_tokens.append(token)
                else:
                    edited_paths = loader.edited_camera_paths(token)
                    ego_geometry, geometry_source = (
                        loader.ego_vehicle_geometry(token)
                    )
                    speed_mps = loader.current_speed_mps(token)
                    longitudinal_acceleration_mps2 = (
                        loader.current_longitudinal_acceleration_mps2(token)
                    )
                    metadata, metadata_path = loader.metadata(token)
                    actor_motion = actor_motion_from_metadata(
                        metadata, camera_name=None
                    )
                    original_poses = cached_originals[token]

                    stage = "edited_inference"
                    edited_prediction = agent.compute_trajectory(
                        make_input(
                            ego_history,
                            edited_paths,
                        ),
                        **inference.seed_kwargs(pair_seed),
                    )
                    edited_poses = save_prediction(
                        trajectory_dir,
                        output_stem,
                        edited_prediction,
                        interval_length_s=interval_length_s,
                    )
                    save_json(
                        reasoning_dir / f"{output_stem}.json",
                        edited_prediction.extra,
                    )

                    stage = "metrics"
                    original_metric_poses = metric_trajectory(
                        original_poses,
                        interval_length_s=interval_length_s,
                        metric_horizon_s=metric_horizon_s,
                    )
                    # The edited plan must also cover the metric horizon.
                    metric_trajectory(
                        edited_poses,
                        interval_length_s=interval_length_s,
                        metric_horizon_s=metric_horizon_s,
                    )
                    metrics = sample_record(
                        token=token,
                        output_stem=output_stem,
                        actor_motion=actor_motion,
                        vehicle_geometry=ego_geometry,
                        speed_mps=speed_mps,
                        longitudinal_acceleration_mps2=(
                            longitudinal_acceleration_mps2
                        ),
                    )
                    metrics.update(
                        {
                            "placement_metadata_path": str(metadata_path),
                            inference.seed_key: pair_seed,
                            "native_pose_count": int(original_poses.shape[0]),
                            "metric_pose_count": int(
                                original_metric_poses.shape[0]
                            ),
                            "metric_horizon_s": (
                                float(original_metric_poses.shape[0])
                                * interval_length_s
                            ),
                            "ego_vehicle_geometry": ego_geometry,
                            "ego_vehicle_geometry_source": geometry_source,
                            "alpamayo_ego_history_source": history_source,
                            "original_image_sources": source_contract,
                        }
                    )
                    metric_records.append(metrics)
                print(
                    f"Completed {sample_index}/{len(tokens)} token={token}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "index": sample_index,
                    "token": token,
                    "output_stem": output_stem,
                    inference.seed_key: pair_seed,
                    "original_image_sources": source_contract,
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed_samples.append(failure)
                save_json(failures_path, failed_samples)
                print(
                    f"Failed {sample_index}/{len(tokens)} token={token} "
                    f"stage={stage}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        preflight = loader.preflight_report()

    save_json(failures_path, failed_samples)
    if args.original:
        manifest_path = save_original_manifest(
            result_dir,
            signature=signature,
            source_run_dir=run_dir,
            selected_tokens=tokens,
            successful_tokens=successful_tokens,
            source_contracts=source_contracts,
        )
        print(
            f"Saved {len(successful_tokens)} original predictions to "
            f"{result_dir}; {len(failed_samples)} samples failed. "
            f"Manifest: {manifest_path}",
            flush=True,
        )
        return

    # Bookkeeping only; every metric comes from protocol V2 scoring (postprocess_collisions).
    save_json(metrics_dir / "per_sample.json", metric_records)
    summary = run_summary(metric_records)
    summary.update(
        {
            "dataset": "pai-av",
            "model": MODEL_NAME,
            "run_dir": str(run_dir),
            "run_config_path": str(run_dir / "run_config.json"),
            "annotations_path": str(run_dir / "annotations.json"),
            "source_reconstruction_path": str(
                run_dir / "source_reconstruction.json"
            ),
            "compaction_report_path": str(run_dir / "compaction_report.json"),
            "final_images_dir": str(run_dir / "final_images"),
            "result_dir": str(result_dir),
            "original_result_dir": str(original_result_dir),
            "evaluation_config_path": str(config_path),
            "model_name_or_path": cfg.get("model_name_or_path"),
            "attn_implementation": cfg.get("attn_implementation"),
            **inference.metadata(config_path),
            "interval_length_s": interval_length_s,
            "metric_horizon_s": metric_horizon_s,
            "selected_token_count": len(tokens),
            "successful_pair_count": len(metric_records),
            "failed_sample_count": len(failed_samples),
            "tokens_file": None if tokens_file is None else str(tokens_file),
            "tokens_file_sha256": (
                None
                if tokens_file is None
                else hashlib.sha256(tokens_file.read_bytes()).hexdigest()
            ),
            "compact_run_preflight": preflight,
            "input_contract": {
                "camera_fields": list(CAMERA_NAMES),
                "camera_ids": [0, 1, 2, 6],
                "frames_per_camera": 4,
                "image_interval_s": 0.1,
                "ego_history": "16 native 3D current-ego poses at 10 Hz",
                "navigation_text": None,
                "original_images": "decoded temporarily from PAI-AV archives",
                "edited_images": "compact run final_images only",
            },
        }
    )
    save_json(metrics_dir / "summary.json", summary)

    print(
        f"Saved {len(metric_records)} paired results to {result_dir}; "
        f"{len(failed_samples)} samples failed.",
        flush=True,
    )
    postprocess_collisions(result_dir, args.scoring)


if __name__ == "__main__":
    main()
