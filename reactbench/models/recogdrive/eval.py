"""Run ReCogDrive on a ReactNAV compositor run (original or edited front-camera images).

    python -m reactbench.models.recogdrive.eval RUN_DIR --original --config rl   # cache original predictions
    python -m reactbench.models.recogdrive.eval RUN_DIR --config rl              # edited predictions + scoring

``--config`` is one of the files in ``configs/`` (``vlm_sft``, ``il``, ``rl``, ``il_rt``) or a path.
Predictions go to ``<outputs>/predictions/<name>/<dataset>/<run>``, where ``name`` comes from the
configuration. The original and edited rollouts of a scene share all diffusion noise.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from ...scoring.actor_motion import inserted_actor_motion
from ...scoring.geometry import project_acceleration_onto_velocity
from ...scoring.motion_bins import NAVSIM_BIN_SPEC_ID
from ...scoring.records import run_summary, sample_record
from ..common import (MAX_TORCH_SEED, annotations_path, current_speed_mps, discover_tokens, expand_config,
                      load_annotations_index,
                      load_json, load_run_config, predictions_dir, resolve_directory, resolve_edited_front_image,
                      resolve_file, save_json, save_prediction, select_tokens)
from ..original_cache import (cache_signature, save_original_manifest, validate_original_cache,
                              validate_original_cache_cohort)

CONFIG_DIR = Path(__file__).with_name("configs")
VLM_TARGET = "reactbench.models.recogdrive.vlm_agent.ReCogDriveVLMAgent"


def diffusion_seed_for_token(token: str, base_seed: int) -> int:
    """Derive a stable diffusion seed shared by an original/edited pair."""

    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    base_seed = int(base_seed)
    if not 0 <= base_seed <= MAX_TORCH_SEED:
        raise ValueError(
            "base_seed must be between 0 and 2**63 - 1, "
            f"got {base_seed}"
        )

    material = f"recogdrive-diffusion:{base_seed}:{token}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big") >> 1


def compute_trajectory_with_seed(agent: Any, agent_input: Any, seed: int) -> Any:
    """Run one stochastic rollout without advancing surrounding Torch RNGs."""

    seed = int(seed)
    if not 0 <= seed <= MAX_TORCH_SEED:
        raise ValueError(f"seed must be between 0 and 2**63 - 1, got {seed}")
    device = torch.device(agent.device)
    rng_devices: list[int] = []
    if device.type == "cuda":
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        return agent.compute_trajectory(agent_input)


def load_config(value: str) -> tuple[Path, dict[str, Any]]:
    """Resolve a configuration name or path; ``{checkpoints}`` expands to the configured checkpoint root."""
    path = CONFIG_DIR / f"{value}.yaml" if (CONFIG_DIR / f"{value}.yaml").is_file() else Path(value)
    path = resolve_file(path, "ReCogDrive configuration")
    text = expand_config(path.read_text())
    config = yaml.safe_load(text)
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError(f"Expected a YAML mapping with a 'name' in {path}")
    backend = config.get("backend", "diffusion")
    if backend not in ("diffusion", "vlm") or (backend == "vlm") != (config.get("_target_") == VLM_TARGET):
        raise ValueError(f"backend={backend} does not match _target_={config.get('_target_')} in {path}")
    return path, config


def agent_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only the configuration accepted by the agent constructor."""
    return {k: v for k, v in config.items() if k not in ("name", "backend", "diffusion_seed_base")}


def make_input(annotation: dict[str, Any], image_path: Path) -> Any:
    from .inputs import make_recogdrive_agent_input
    return make_recogdrive_agent_input(
        image_path=image_path,
        velocities=annotation["velocity"],
        accelerations=annotation["acceleration"],
        history_trajectories=annotation["past_trajectory"],
        navigation_command=annotation["navigation_command"],
    )


def compute_prediction(agent: Any, agent_input: Any, seed: Optional[int], *, response_path: Optional[Path] = None) -> Any:
    if response_path is None:
        return compute_trajectory_with_seed(agent, agent_input, seed)
    agent.last_generation = None
    try:
        return agent.compute_trajectory(agent_input)
    finally:
        # Persist the generated text even if trajectory parsing raises.
        if agent.last_generation is not None:
            save_json(response_path, agent.last_generation)


def load_actor_motion(run_dir: Path, token: str) -> tuple[dict[str, Any], Path]:
    metadata_path = run_dir / "metadata" / f"{token}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No compositor metadata for token {token}: {metadata_path}")
    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}")
    return inserted_actor_motion(metadata), metadata_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Compositor run (final_images/, metadata/, run_config.json)")
    parser.add_argument("--config", default="rl", help="vlm_sft, il, rl, il_rt, or a path")
    parser.add_argument("--dataset", default="reactnav", help="Dataset folder of the predictions (default reactnav)")
    parser.add_argument("--original", action="store_true", help="Run and cache only original-image predictions")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-samples", type=int, help="Evaluate only the first N tokens (smoke test)")
    selection.add_argument("--tokens-file", type=Path, help="JSON with the exact ordered 'tokens' to evaluate")
    parser.add_argument("--scoring", choices=("score", "off"), default="score",
                        help="Score the edited run on CPU after inference (default) or not")
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
    config_path, cfg = load_config(args.config)
    is_vlm = cfg.get("backend", "diffusion") == "vlm"
    final_images_dir = resolve_directory(run_dir / "final_images", "final edited images")
    run_config_path, run_config = load_run_config(run_dir)
    annotations_file = annotations_path(run_config)
    annotations_by_token = load_annotations_index(annotations_file)

    result_dir = predictions_dir(cfg["name"], args.dataset, "original" if args.original else run_dir.name)
    original_result_dir = predictions_dir(cfg["name"], args.dataset, "original")
    trajectory_dir = result_dir / "trajectories"
    metrics_dir = result_dir / "metrics"
    failures_path = result_dir / "failed_samples.json"
    result_dir.mkdir(parents=True, exist_ok=True)

    tokens = select_tokens(discover_tokens(final_images_dir, run_config),
                           tokens_file=args.tokens_file, max_samples=args.max_samples)
    interval_length_s = float(cfg.get("trajectory_sampling", {}).get("interval_length", 0.5))
    diffusion_seed_base = None if is_vlm else int(cfg.get("diffusion_seed_base", 42))
    agent = instantiate(agent_config(cfg))
    provenance = agent.provenance() if is_vlm else None
    signature = cache_signature(
        evaluator="recogdrive-vlm-navsim" if is_vlm else "recogdrive-navsim",
        config_path=config_path,
        interval_length_s=interval_length_s,
        inference_options=provenance if is_vlm else {},
        seed_name="unused_greedy" if is_vlm else "diffusion_seed_base",
        seed_base=0 if is_vlm else diffusion_seed_base,
        source={"annotations_path": str(annotations_file)},
    )
    selected_tokens = list(tokens)
    excluded_original_tokens: list[str] = []
    cached_originals = {}
    if not args.original and is_vlm:
        # Unparseable VLM answers leave original gaps; evaluate the successful subset.
        cohort = validate_original_cache_cohort(original_result_dir, tokens=tokens, expected_signature=signature)
        cached_originals, tokens = cohort.trajectories, cohort.eligible_tokens
        excluded_original_tokens = cohort.excluded_failed_tokens
    elif not args.original:
        cached_originals = validate_original_cache(original_result_dir, tokens=tokens, expected_signature=signature)

    print(f"Evaluating {len(tokens)} {'original' if args.original else 'edited'} samples from {run_dir}", flush=True)
    agent.initialize()
    failed_samples: list[dict[str, Any]] = []
    successful_tokens: list[str] = []
    metric_records: list[dict[str, Any]] = []
    for sample_index, token in enumerate(tokens, start=1):
        stage = "load_input"
        pair_seed: Optional[int] = None
        response_path = result_dir / "responses" / f"{token}.json" if is_vlm else None
        try:
            if not is_vlm:
                pair_seed = diffusion_seed_for_token(token, diffusion_seed_base)
            if token not in annotations_by_token:
                raise KeyError(f"Token {token!r} is absent from {annotations_file}")
            annotation = annotations_by_token[token]
            if args.original:
                stage = "original_inference"
                image = Path(annotation["front_camera"][-1])
                trajectory = compute_prediction(agent, make_input(annotation, image), pair_seed, response_path=response_path)
                save_prediction(trajectory_dir, token, trajectory, interval_length_s=interval_length_s)
                successful_tokens.append(token)
            else:
                edited_image = resolve_edited_front_image(final_images_dir, token)
                speed_mps = current_speed_mps(annotation)
                acceleration = project_acceleration_onto_velocity(annotation["velocity"][-1], annotation["acceleration"][-1])
                actor_motion, metadata_path = load_actor_motion(run_dir, token)
                original_poses = cached_originals[token]
                stage = "edited_inference"
                trajectory = compute_prediction(agent, make_input(annotation, edited_image), pair_seed,
                                                response_path=response_path)
                edited_poses = save_prediction(trajectory_dir, token, trajectory, interval_length_s=interval_length_s)
                stage = "metrics"
                record = sample_record(bin_spec=NAVSIM_BIN_SPEC_ID, token=token, output_stem=token,
                                       original_trajectory=original_poses, edited_trajectory=edited_poses,
                                       actor_motion=actor_motion, interval_length_s=interval_length_s,
                                       speed_mps=speed_mps, longitudinal_acceleration_mps2=acceleration)
                record["placement_metadata_path"] = str(metadata_path)
                if is_vlm:
                    record["response_path"] = str(response_path)
                else:
                    record["diffusion_seed"] = pair_seed
                metric_records.append(record)
            print(f"Completed {sample_index}/{len(tokens)} token={token}", flush=True)
        except Exception as exc:
            failed_samples.append({"index": sample_index, "token": token, "stage": stage, "diffusion_seed": pair_seed,
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

    # Bookkeeping only; every metric comes from scoring.
    save_json(metrics_dir / "per_sample.json", metric_records)
    summary = run_summary(metric_records)
    summary.update({
        "model": cfg["name"], "dataset": args.dataset,
        "run_dir": str(run_dir), "run_config_path": str(run_config_path), "annotations_path": str(annotations_file),
        "final_images_dir": str(final_images_dir), "result_dir": str(result_dir),
        "original_result_dir": str(original_result_dir), "interval_length_s": interval_length_s,
        "selected_token_count": len(selected_tokens), "successful_pair_count": len(metric_records),
        "failed_sample_count": len(failed_samples),
        "excluded_original_tokens": excluded_original_tokens,
    })
    if not is_vlm:
        summary.update(diffusion_seed_base=diffusion_seed_base,
                       diffusion_seed_strategy="SHA-256 of recogdrive-diffusion:<base>:<token>; "
                                               "first 63 bits shared by each original/edited pair")
    else:
        summary["inference_provenance"] = provenance
    save_json(metrics_dir / "summary.json", summary)
    print(f"Saved {len(metric_records)} paired results to {result_dir}; {len(failed_samples)} samples failed.", flush=True)
    postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)


if __name__ == "__main__":
    main()
