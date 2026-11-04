"""Run AutoVLA on a ReactNAV compositor run (three front cameras, four history frames).

    python -m reactbench.models.autovla.eval RUN_DIR --original     # cache original predictions
    python -m reactbench.models.autovla.eval RUN_DIR                # edited predictions + scoring

Inputs are AutoVLA's own per-token no-CoT samples (``autovla_inputs`` in configs/paths.yaml, made
with AutoVLA's preprocessing); for edited runs the front, front-left and front-right images of all
four history frames are replaced by the edited ones. Predictions go to
``<outputs>/predictions/autovla/<dataset>/<run>``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from ... import paths
from ...scoring.actor_motion import inserted_actor_motion
from ...scoring.geometry import project_acceleration_onto_velocity
from ...scoring.motion_bins import NAVSIM_BIN_SPEC_ID
from ...scoring.records import run_summary, sample_record
from ..common import (MAX_TORCH_SEED, SUPPORTED_IMAGE_SUFFIXES, discover_tokens, expand_config, load_json,
                      load_run_config, predictions_dir, resolve_directory, resolve_file, save_json, select_tokens,
                      trajectory_poses)
from ..original_cache import cache_signature, save_original_manifest, validate_original_cache_cohort

AUTOVLA_REPO_ROOT = Path(os.environ.get("AUTOVLA_REPO_ROOT", paths.REPO_ROOT / "third_party" / "AutoVLA"))
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
CAMERA_INPUT_KEYS = {
    "CAM_F0": "front_camera_paths",
    "CAM_L0": "front_left_camera_paths",
    "CAM_R0": "front_right_camera_paths",
}
CURRENT_FRAME_INDEX = 3


def generation_seed_for_token(token: str, base_seed: int) -> int:
    """Derive a stable generation seed shared by an original/edited pair."""

    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    base_seed = int(base_seed)
    if not 0 <= base_seed <= MAX_TORCH_SEED:
        raise ValueError(
            "base_seed must be between 0 and 2**63 - 1, "
            f"got {base_seed}"
        )

    material = f"autovla-generation:{base_seed}:{token}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big") >> 1


def load_token_agent_input(json_data_path: Path, token: str) -> dict[str, Any]:
    json_path = json_data_path / f"{token}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Token input JSON does not exist: {json_path}")
    agent_input = load_json(json_path)
    if not isinstance(agent_input, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")
    if agent_input.get("token") != token:
        raise ValueError(f"Token mismatch in {json_path}: expected {token}, got {agent_input.get('token')}")
    return agent_input


def resolve_edited_frame(final_images_dir: Path, token: str, camera: str, frame_index: int) -> Path:
    frame_stem = final_images_dir / token / camera / str(frame_index)
    matches = [frame_stem.with_suffix(s) for s in SUPPORTED_IMAGE_SUFFIXES if frame_stem.with_suffix(s).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one edited image for token={token}, camera={camera}, "
                                f"frame={frame_index} under {final_images_dir}, found {len(matches)}")
    return matches[0]


def replace_front_camera_history(agent_input: dict[str, Any], final_images_dir: Path, token: str) -> Path:
    edited_current_front: Optional[Path] = None
    for camera, input_key in CAMERA_INPUT_KEYS.items():
        original_paths = agent_input.get(input_key)
        if not isinstance(original_paths, list) or len(original_paths) != 4:
            raise ValueError(f"Expected four paths in {input_key!r} for token {token}")
        edited_paths = [str(resolve_edited_frame(final_images_dir, token, camera, frame_index))
                        for frame_index in range(len(original_paths))]
        agent_input[input_key] = edited_paths
        if camera == "CAM_L0" and "left_camera_paths" in agent_input:
            agent_input["left_camera_paths"] = list(edited_paths)
        if camera == "CAM_R0" and "right_camera_paths" in agent_input:
            agent_input["right_camera_paths"] = list(edited_paths)
        if camera == "CAM_F0":
            edited_current_front = Path(edited_paths[CURRENT_FRAME_INDEX])
    if edited_current_front is None:
        raise RuntimeError(f"Failed to resolve edited CAM_F0 frame for {token}")
    return edited_current_front


def current_speed_mps(agent_input: dict[str, Any]) -> float:
    velocity = np.asarray(agent_input["velocity"], dtype=np.float64)
    if velocity.ndim != 1 or velocity.size < 2:
        raise ValueError(f"Expected a current velocity vector with at least two values, got {velocity.shape}")
    speed = float(np.linalg.norm(velocity[:2]))
    if not np.isfinite(speed):
        raise ValueError(f"Current ego speed is not finite: {velocity}")
    return speed


def format_cot_output(cot_results: Any) -> str:
    if cot_results is None:
        return ""
    if isinstance(cot_results, str):
        return cot_results.strip()
    if isinstance(cot_results, np.ndarray):
        cot_results = cot_results.tolist()
    if isinstance(cot_results, (list, tuple)):
        return "\n".join(str(item).strip() for item in cot_results)
    return str(cot_results).strip()


def save_prediction(trajectory_dir: Path, cot_dir: Path, token: str, prediction: Any, *,
                    interval_length_s: float) -> np.ndarray:
    trajectory, cot = prediction if isinstance(prediction, tuple) and len(prediction) == 2 else (prediction, None)
    poses = trajectory_poses(trajectory)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_dir / f"{token}.npz", trajectory=poses,
                        interval_length_s=np.float32(interval_length_s))
    cot_dir.mkdir(parents=True, exist_ok=True)
    content = format_cot_output(cot)
    (cot_dir / f"{token}.txt").write_text((content + "\n") if content else "\n", encoding="utf-8")
    return poses


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
    sys.path.insert(0, str(AUTOVLA_REPO_ROOT))
    sys.path.insert(0, str(AUTOVLA_REPO_ROOT / "navsim"))
    from hydra.utils import instantiate
    from ...scoring.navsim import postprocess_evaluation
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args(argv)
    run_dir = resolve_directory(args.run_dir, "run directory")
    config_path = resolve_file(args.config, "AutoVLA configuration")
    cfg = yaml.safe_load(expand_config(config_path.read_text()))
    final_images_dir = resolve_directory(run_dir / "final_images", "final edited images")
    run_config_path, run_config = load_run_config(run_dir)
    json_data_path = resolve_directory(paths.get("autovla_inputs"), "AutoVLA token JSON directory")

    name = cfg.pop("name", "autovla")
    result_dir = predictions_dir(name, args.dataset, "original" if args.original else run_dir.name)
    original_result_dir = predictions_dir(name, args.dataset, "original")
    trajectory_dir, cot_dir = result_dir / "trajectories", result_dir / "cot_outputs"
    failures_path = result_dir / "failed_samples.json"
    result_dir.mkdir(parents=True, exist_ok=True)

    tokens = select_tokens(discover_tokens(final_images_dir, run_config),
                           tokens_file=args.tokens_file, max_samples=args.max_samples)
    interval_length_s = float(cfg.get("trajectory_sampling", {}).get("interval_length", 0.5))
    metric_horizon_s = float(cfg.pop("metric_horizon_s", 4.0))
    generation_seed_base = int(cfg.pop("generation_seed_base", 42))
    signature = cache_signature(
        evaluator="autovla-navsim", config_path=config_path, interval_length_s=interval_length_s,
        inference_options={"thinking": False}, seed_name="generation_seed_base", seed_base=generation_seed_base,
        source={"json_data_path": str(json_data_path)},
    )
    requested_tokens = list(tokens)
    excluded_original_tokens: list[str] = []
    cached_originals: dict[str, np.ndarray] = {}
    if not args.original:
        # Originals whose generation failed are excluded from the pairs.
        cohort = validate_original_cache_cohort(original_result_dir, tokens=tokens, expected_signature=signature)
        cached_originals, tokens = cohort.trajectories, cohort.eligible_tokens
        excluded_original_tokens = cohort.excluded_failed_tokens

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
            seed = generation_seed_for_token(token, generation_seed_base)
            original_input = load_token_agent_input(json_data_path, token)
            if args.original:
                stage = "original_inference"
                torch.manual_seed(seed)
                save_prediction(trajectory_dir, cot_dir, token, agent.compute_trajectory(original_input),
                                interval_length_s=interval_length_s)
                successful_tokens.append(token)
            else:
                speed_mps = current_speed_mps(original_input)
                acceleration = project_acceleration_onto_velocity(original_input["velocity"], original_input["acceleration"])
                metadata_path = run_dir / "metadata" / f"{token}.json"
                actor_motion = inserted_actor_motion(load_json(metadata_path))
                stage = "prepare_edited_input"
                edited_input = copy.deepcopy(original_input)
                replace_front_camera_history(edited_input, final_images_dir, token)
                original_poses = cached_originals[token]
                stage = "edited_inference"
                torch.manual_seed(seed)
                edited_poses = save_prediction(trajectory_dir, cot_dir, token, agent.compute_trajectory(edited_input),
                                               interval_length_s=interval_length_s)
                stage = "metrics"
                count = int(round(metric_horizon_s / interval_length_s))
                record = sample_record(bin_spec=NAVSIM_BIN_SPEC_ID, token=token, output_stem=token,
                                       original_trajectory=original_poses[:count], edited_trajectory=edited_poses[:count],
                                       actor_motion=actor_motion, interval_length_s=interval_length_s,
                                       speed_mps=speed_mps, longitudinal_acceleration_mps2=acceleration)
                record.update(placement_metadata_path=str(metadata_path), generation_seed=seed,
                              native_pose_count=int(original_poses.shape[0]), metric_pose_count=count,
                              metric_horizon_s=count * interval_length_s)
                metric_records.append(record)
            print(f"Completed {sample_index}/{len(tokens)} token={token}", flush=True)
        except Exception as exc:
            failed_samples.append({"index": sample_index, "token": token, "stage": stage, "generation_seed": seed,
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
        "final_images_dir": str(final_images_dir), "result_dir": str(result_dir),
        "original_result_dir": str(original_result_dir), "inference_options": signature["inference_options"],
        "generation_seed_base": generation_seed_base,
        "generation_seed_strategy": "SHA-256 of autovla-generation:<base>:<token>; first 63 bits shared by each pair",
        "interval_length_s": interval_length_s, "metric_horizon_s": metric_horizon_s,
        "selected_token_count": len(requested_tokens), "original_success_cohort_count": len(tokens),
        "excluded_original_tokens": excluded_original_tokens,
        "successful_pair_count": len(metric_records), "failed_sample_count": len(failed_samples),
    })
    save_json(metrics_dir / "summary.json", summary)
    print(json.dumps({"saved_pairs": len(metric_records), "failed": len(failed_samples), "result_dir": str(result_dir)}),
          flush=True)
    postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)


if __name__ == "__main__":
    main()
