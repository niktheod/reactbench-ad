"""Run DriveWAM on a ReactNAV compositor run (original or edited front-camera images).

    python -m reactbench.models.drivewam.eval RUN_DIR --original     # cache original predictions
    python -m reactbench.models.drivewam.eval RUN_DIR                # edited predictions + scoring

Predictions go to ``<outputs>/predictions/drivewam/<dataset>/<run>``. The original and edited
rollouts of a scene share all video and action noise.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import yaml

from ...scoring.actor_motion import inserted_actor_motion
from ...scoring.geometry import project_acceleration_onto_velocity
from ...scoring.records import run_summary, sample_record
from ..seeds import paired_seed
from ..common import (annotations_path, current_speed_mps, discover_tokens, expand_config,
                      load_annotations_index,
                      load_json, load_run_config, predictions_dir, resolve_directory, resolve_edited_front_image,
                      resolve_file, save_json, save_prediction, select_tokens)
from ..original_cache import cache_signature, save_original_manifest, validate_original_cache

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


SEED_NAMESPACE = "drivewam-rollout"  # per-token seeds: models/seeds.py


def make_input(annotation: dict[str, Any], image_path: Path) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "velocity": annotation["velocity"][-1],
        "acceleration": annotation["acceleration"][-1],
        "history_poses": annotation["past_trajectory"],
        "navigation_command": annotation["navigation_command"],
    }


def load_actor_motion(run_dir: Path, token: str) -> tuple[dict[str, Any], Path]:
    metadata_path = run_dir / "metadata" / f"{token}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No compositor metadata for token {token}: {metadata_path}")
    return inserted_actor_motion(load_json(metadata_path)), metadata_path


def load_config(path: Path) -> dict[str, Any]:
    path = resolve_file(path, "DriveWAM configuration")
    config = yaml.safe_load(expand_config(path.read_text()))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Compositor run (final_images/, metadata/, run_config.json)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def main(argv=None) -> None:
    from hydra.utils import instantiate
    from ...scoring.navsim import postprocess_evaluation
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args(argv)
    run_dir = resolve_directory(args.run_dir, "run directory")
    config_path = resolve_file(args.config, "DriveWAM configuration")
    cfg = load_config(config_path)
    final_images_dir = resolve_directory(run_dir / "final_images", "final edited images")
    run_config_path, run_config = load_run_config(run_dir)
    annotations_file = annotations_path(run_config)
    annotations_by_token = load_annotations_index(annotations_file)

    name = cfg.pop("name", "drivewam")
    result_dir = predictions_dir(name, args.dataset, "original" if args.original else run_dir.name)
    original_result_dir = predictions_dir(name, args.dataset, "original")
    trajectory_dir = result_dir / "trajectories"
    failures_path = result_dir / "failed_samples.json"
    result_dir.mkdir(parents=True, exist_ok=True)

    tokens = select_tokens(discover_tokens(final_images_dir, run_config),
                           tokens_file=args.tokens_file, max_samples=args.max_samples)
    interval_length_s = float(cfg.get("trajectory_sampling", {}).get("interval_length", 0.5))
    signature = cache_signature(
        evaluator="drivewam-navsim", config_path=config_path, interval_length_s=interval_length_s,
        inference_options={}, seed_name="rollout_seed_base", seed_base=int(cfg.get("seed", 0)),
        source={"annotations_path": str(annotations_file)},
    )
    if not args.original:
        # The original-scene predictions must exist for every token and match this configuration.
        validate_original_cache(original_result_dir, tokens=tokens, expected_signature=signature)

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
            seed = paired_seed(SEED_NAMESPACE, agent.seed, token)
            if token not in annotations_by_token:
                raise KeyError(f"Token {token!r} is absent from {annotations_file}")
            annotation = annotations_by_token[token]
            if args.original:
                stage = "original_inference"
                trajectory = agent.compute_trajectory(make_input(annotation, Path(annotation["front_camera"][-1])),
                                                      rollout_seed=seed)
                save_prediction(trajectory_dir, token, trajectory, interval_length_s=interval_length_s)
                successful_tokens.append(token)
            else:
                edited_image = resolve_edited_front_image(final_images_dir, token)
                speed_mps = current_speed_mps(annotation)
                acceleration = project_acceleration_onto_velocity(annotation["velocity"][-1], annotation["acceleration"][-1])
                actor_motion, metadata_path = load_actor_motion(run_dir, token)
                stage = "edited_inference"
                trajectory = agent.compute_trajectory(make_input(annotation, edited_image), rollout_seed=seed)
                save_prediction(trajectory_dir, token, trajectory, interval_length_s=interval_length_s)
                stage = "metrics"
                record = sample_record(token=token, output_stem=token,
                                       actor_motion=actor_motion, speed_mps=speed_mps, longitudinal_acceleration_mps2=acceleration)
                record.update(placement_metadata_path=str(metadata_path), rollout_seed=seed)
                metric_records.append(record)
            print(f"Completed {sample_index}/{len(tokens)} token={token}", flush=True)
        except Exception as exc:
            failed_samples.append({"index": sample_index, "token": token, "stage": stage, "rollout_seed": seed,
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
        "model": name, "dataset": args.dataset, "run_dir": str(run_dir), "run_config_path": str(run_config_path),
        "annotations_path": str(annotations_file), "final_images_dir": str(final_images_dir),
        "result_dir": str(result_dir), "original_result_dir": str(original_result_dir),
        "rollout_seed_base": agent.seed,
        "rollout_seed_strategy": "SHA-256 of drivewam-rollout:<base>:<token>; first 63 bits shared by each pair",
        "interval_length_s": interval_length_s, "selected_token_count": len(tokens),
        "successful_pair_count": len(metric_records), "failed_sample_count": len(failed_samples),
    })
    save_json(metrics_dir / "summary.json", summary)
    print(f"Saved {len(metric_records)} paired results to {result_dir}; {len(failed_samples)} samples failed.", flush=True)
    postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)


if __name__ == "__main__":
    main()
