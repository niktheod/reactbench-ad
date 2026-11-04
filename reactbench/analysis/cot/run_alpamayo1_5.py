"""Alpamayo 1.5 CoT interventions on ReactPAI (paper section 4.2 and appendix): CoT swap and fixed CoTs.

    python -m reactbench.analysis.cot.run_alpamayo1_5 RUN_DIR --cot-swap     # O/E images x O/E CoTs
    python -m reactbench.analysis.cot.run_alpamayo1_5 RUN_DIR --fixed-cots   # the five CoTs of fixed_cots.yaml

Both reuse the run's natural original and edited predictions from
``python -m reactbench.models.alpamayo1_5.eval_paiav`` (same configuration and per-token seeds) and force
the CoT through Alpamayo 1.5's agent (``models/alpamayo1_5/forced_cot.py``). Results go to
``<outputs>/predictions/alpamayo1_5/reactpai/<run>_cot_swap`` and ``<run>_fixed_cots``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Optional

from contextlib import ExitStack

import numpy as np
import yaml

from ...models.common import expand_config
from ...models.original_cache import cache_signature, file_sha256, load_cached_trajectory, validate_original_cache
from .fixed_cot import (
    BASELINE, CONDITIONS, ORIGINAL, SCHEMA_VERSION, ResumeConflict, aggregate, condition_metrics,
    contrast, digest, input_provenance, jsonable, load_completed, load_config,
    object_type, prepare_manifest, read_json, resolve_conditions, save_completed,
    save_json,
)


from ...models.alpamayo1_5 import eval_paiav as evaluation
from ...models.alpamayo1_5.eval_paiav import (
    MODEL_NAME, DEFAULT_CONFIG_PATH, load_requested_tokens, make_input, metric_trajectory, resolve_file,
    result_name_for_run, save_json as save_result_json, save_prediction,
)
from ...models.alpamayo1_5.inference import AlpamayoInference
from ...models.common import predictions_dir
from ...datasets.paiav import PAIAVCompactRun
from ...scoring.actor_motion import actor_motion_from_metadata
from ...scoring.paiav import postprocess_evaluation as postprocess_collisions
from .cot_swap import (
    CROSSED_EDITED_IMAGE, CROSSED_ORIGINAL_IMAGE, FACTORIAL_CONDITIONS, NATURAL_EDITED, NATURAL_ORIGINAL,
    aggregate_cot_swap_records, build_cot_swap_record, intervention_reasoning, load_cot_swap_sources,
)

COT_SWAP_RESULT_SUFFIX = "_cot_swap"
FIXED_COTS = Path(__file__).resolve().with_name("fixed_cots.yaml")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Compact ReactPAI run directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cot-swap", action="store_true",
                      help="Run original-image/edited-CoT and edited-image/original-CoT rollouts")
    mode.add_argument("--fixed-cots", type=Path, metavar="PATH", nargs="?", const=FIXED_COTS,
                      help="Evaluate fixed CoTs (default: the bundled fixed_cots.yaml)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-samples", type=int, help="Only the first N clips (a smoke test)")
    selection.add_argument("--tokens-file", type=Path, help="JSON with the ordered clip subset under 'tokens'")
    parser.add_argument("--original-result-dir", type=Path,
                        help="Read the natural original predictions from this directory")
    parser.add_argument("--dataset", default="reactpai")
    parser.add_argument("--scoring", choices=("score", "off"), default="score")
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def cot_swap(args: argparse.Namespace) -> None:
    """Original/edited images crossed with original/edited CoTs (2 x 2), with the natural rollouts' seeds."""
    from hydra.utils import instantiate

    os.environ.setdefault("MPLBACKEND", "Agg")
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
    natural_edited_result_dir = predictions_dir(MODEL_NAME, args.dataset, result_name_for_run(run_dir))
    result_dir = predictions_dir(MODEL_NAME, args.dataset, result_name_for_run(run_dir) + COT_SWAP_RESULT_SUFFIX)
    default_original_result_dir = predictions_dir(MODEL_NAME, args.dataset, result_name_for_run(run_dir, original=True))
    original_result_dir = (
        default_original_result_dir
        if args.original_result_dir is None
        else args.original_result_dir.expanduser().resolve()
    )
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

    failed_samples: list[dict[str, Any]] = []
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
        cached_originals = validate_original_cache(
            original_result_dir,
            tokens=tokens,
            expected_signature=signature,
            expected_source_contracts=source_contracts,
        )
        cot_swap_sources = load_cot_swap_sources(
            original_result_dir,
            natural_edited_result_dir,
            tokens=tokens,
            interval_length_s=interval_length_s,
            expected_edited_summary={
                "model_name_or_path": cfg.get("model_name_or_path"),
                "attn_implementation": cfg.get("attn_implementation"),
                "vlm_seed": int(cfg.get("seed", 42)),
                f"{inference.seed_key}_base": inference.seed_base,
                "interval_length_s": interval_length_s,
                "metric_horizon_s": metric_horizon_s,
            },
        )
        print(f"Evaluating {len(tokens)} PAI-AV CoT-swap samples from {run_dir}", flush=True)
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
                original_cot = cot_swap_sources.original_cots[token]
                edited_cot = cot_swap_sources.edited_cots[token]
                original_poses = cached_originals[token]
                natural_edited_poses = (
                    cot_swap_sources.edited_trajectories[token]
                )
                condition_provenance: dict[str, dict[str, Any]] = {
                    NATURAL_ORIGINAL: {
                        "inference": "natural",
                        "trajectory_path": str(
                            original_result_dir
                            / "trajectories"
                            / f"{token}.npz"
                        ),
                        "reasoning_path": str(original_cot.path),
                        "reasoning_sha256": original_cot.sha256,
                    },
                    NATURAL_EDITED: {
                        "inference": "natural",
                        "trajectory_path": str(
                            natural_edited_result_dir
                            / "trajectories"
                            / f"{token}.npz"
                        ),
                        "reasoning_path": str(edited_cot.path),
                        "reasoning_sha256": edited_cot.sha256,
                    },
                }

                with loader.materialize_original_images(
                    token
                ) as original_paths:
                    stage = "original_image_edited_cot_inference"
                    crossed_original_prediction = agent.compute_trajectory(
                        make_input(
                            ego_history,
                            original_paths,
                        ),
                        forced_cot=edited_cot.text,
                        **inference.seed_kwargs(pair_seed),
                    )
                    crossed_original_root = (
                        result_dir / CROSSED_ORIGINAL_IMAGE
                    )
                    crossed_original_poses = save_prediction(
                        crossed_original_root / "trajectories",
                        token,
                        crossed_original_prediction,
                        interval_length_s=interval_length_s,
                    )
                    crossed_original_reasoning_path = (
                        crossed_original_root / "reasoning" / f"{token}.json"
                    )
                    save_result_json(
                        crossed_original_reasoning_path,
                        intervention_reasoning(
                            crossed_original_prediction.extra,
                            condition=CROSSED_ORIGINAL_IMAGE,
                            donor_cot=edited_cot,
                            **inference.seed_kwargs(pair_seed),
                        ),
                    )
                    condition_provenance[CROSSED_ORIGINAL_IMAGE] = {
                        "inference": "forced_cot_token_replay",
                        "trajectory_path": str(
                            crossed_original_root
                            / "trajectories"
                            / f"{token}.npz"
                        ),
                        "reasoning_path": str(
                            crossed_original_reasoning_path
                        ),
                        "donor_reasoning_path": str(edited_cot.path),
                        "donor_reasoning_sha256": edited_cot.sha256,
                    }

                stage = "edited_image_original_cot_inference"
                crossed_edited_prediction = agent.compute_trajectory(
                    make_input(
                        ego_history,
                        edited_paths,
                    ),
                    forced_cot=original_cot.text,
                    **inference.seed_kwargs(pair_seed),
                )
                crossed_edited_root = result_dir / CROSSED_EDITED_IMAGE
                crossed_edited_poses = save_prediction(
                    crossed_edited_root / "trajectories",
                    token,
                    crossed_edited_prediction,
                    interval_length_s=interval_length_s,
                )
                crossed_edited_reasoning_path = (
                    crossed_edited_root / "reasoning" / f"{token}.json"
                )
                save_result_json(
                    crossed_edited_reasoning_path,
                    intervention_reasoning(
                        crossed_edited_prediction.extra,
                        condition=CROSSED_EDITED_IMAGE,
                        donor_cot=original_cot,
                        **inference.seed_kwargs(pair_seed),
                    ),
                )
                condition_provenance[CROSSED_EDITED_IMAGE] = {
                    "inference": "forced_cot_token_replay",
                    "trajectory_path": str(
                        crossed_edited_root
                        / "trajectories"
                        / f"{token}.npz"
                    ),
                    "reasoning_path": str(crossed_edited_reasoning_path),
                    "donor_reasoning_path": str(original_cot.path),
                    "donor_reasoning_sha256": original_cot.sha256,
                }

                stage = "cot_swap_metrics"
                factorial_full = {
                    NATURAL_ORIGINAL: original_poses,
                    NATURAL_EDITED: natural_edited_poses,
                    CROSSED_ORIGINAL_IMAGE: crossed_original_poses,
                    CROSSED_EDITED_IMAGE: crossed_edited_poses,
                }
                factorial_metric = {
                    condition: metric_trajectory(
                        poses,
                        interval_length_s=interval_length_s,
                        metric_horizon_s=metric_horizon_s,
                    )
                    for condition, poses in factorial_full.items()
                }
                metrics = build_cot_swap_record(
                    token=token,
                    trajectories=factorial_metric,
                    actor_motion=actor_motion,
                    interval_length_s=interval_length_s,
                    speed_mps=speed_mps,
                    longitudinal_acceleration_mps2=(
                        longitudinal_acceleration_mps2
                    ),
                    **inference.seed_kwargs(pair_seed),
                    condition_provenance=condition_provenance,
                )
                metrics.update(
                    {
                        "placement_metadata_path": str(metadata_path),
                        "native_pose_counts": {
                            condition: int(poses.shape[0])
                            for condition, poses in factorial_full.items()
                        },
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
                save_result_json(failures_path, failed_samples)
                print(
                    f"Failed {sample_index}/{len(tokens)} token={token} "
                    f"stage={stage}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        preflight = loader.preflight_report()

    save_result_json(failures_path, failed_samples)
    save_result_json(metrics_dir / "per_sample.json", metric_records)
    summary = aggregate_cot_swap_records(metric_records)
    summary.update(
        {
            "mode": "cot_swap",
            "dataset": "pai-av",
            "run_dir": str(run_dir),
            "result_dir": str(result_dir),
            "natural_original_result_dir": str(original_result_dir),
            "natural_edited_result_dir": str(
                natural_edited_result_dir
            ),
            "natural_edited_summary_path": str(
                cot_swap_sources.edited_summary_path
            ),
            "natural_edited_summary_sha256": (
                cot_swap_sources.edited_summary_sha256
            ),
            "evaluation_config_path": str(config_path),
            "model_name_or_path": cfg.get("model_name_or_path"),
            "attn_implementation": cfg.get("attn_implementation"),
            **inference.metadata(config_path),
            "intervention": (
                "decoded donor CoT is forced as assistant tokens; hidden "
                "states are recomputed with the selected image"
            ),
            "conditions": summary["conditions"],
            "condition_order": list(FACTORIAL_CONDITIONS),
            "interval_length_s": interval_length_s,
            "metric_horizon_s": metric_horizon_s,
            "selected_token_count": len(tokens),
            "successful_token_count": len(metric_records),
            "failed_sample_count": len(failed_samples),
            "compact_run_preflight": preflight,
            "metric_definitions": {
                "condition_metrics": (
                    "distance driven; collision metrics come from "
                    "protocol V2 scoring (metrics/paiav_at_fault_*)"
                ),
                "contrast_ade_m": (
                    "mean corresponding-pose x/y displacement from the "
                    "named from_condition to to_condition"
                ),
                "contrast_delta": (
                    "to_condition value minus from_condition value"
                ),
                "interaction_difference_in_differences": (
                    "(edited-image effect with edited CoT) minus "
                    "(edited-image effect with original CoT)"
                ),
            },
        }
    )
    save_result_json(metrics_dir / "summary.json", summary)
    print(
        f"Saved {len(metric_records)} PAI-AV CoT-swap results to "
        f"{result_dir}; {len(failed_samples)} samples failed.",
        flush=True,
    )
    postprocess_collisions(result_dir, args.scoring)


class BaselineMismatch(ValueError):
    """A present baseline belongs to different inference inputs."""


class ModelInitializationError(RuntimeError):
    """Stop the run if its model cannot be initialized."""


class Inputs:
    """Edited inputs of a compact ReactPAI run, as the natural evaluation builds them."""

    def __init__(self, evaluator, args, stack):
        self.e, self.args = evaluator, args
        self.run = args.run_dir.expanduser().resolve()
        self.options = {}
        self.loader = stack.enter_context(evaluator.PAIAVCompactRun(self.run))
        self.config_path = self.loader.run_config_path
        self.config = self.loader.run_config
        self.tokens = list(self.loader.tokens)
        self.source = {"dataset": "pai-av", "dataset_group": self.run.parent.name}
        self.source_files = {str(p): file_sha256(p) for p in (
            self.loader.annotations_path, self.loader.source_index_path)}
        self.kind = object_type(self.config)
        if args.tokens_file is not None:
            self.tokens = evaluator.load_requested_tokens(args.tokens_file, self.tokens)
        elif args.max_samples is not None:
            self.tokens = self.tokens[:args.max_samples]
        if not self.tokens:
            raise ValueError("No samples selected")

    def name(self, original=False):
        return self.e.result_name_for_run(self.run, original=original)

    def sample(self, token):
        e = self.e
        history, _ = self.loader.ego_history(token)
        agent_input = e.make_input(history, self.loader.edited_camera_paths(token))
        metadata, path = self.loader.metadata(token)
        actor = e.actor_motion_from_metadata(metadata, camera_name=None)
        geometry, _ = self.loader.ego_vehicle_geometry(token)
        speed = self.loader.current_speed_mps(token)
        source_contract = self.loader.original_source_contract(token)
        if object_type(actor) != self.kind:
            raise ValueError(f"Actor metadata differs from run object type for {token}")
        return agent_input, actor, path, geometry, speed, source_contract


class Baselines:
    """Missing caches are reported; existing incompatible caches fail preflight."""

    def __init__(self, original, edited, signature, expected_summary):
        self.original, self.edited, self.signature = original, edited, signature
        self.manifest = self.summary = None
        self.edited_tokens = set()
        self.status = {}
        manifest_path = original / "evaluation_manifest.json"
        if manifest_path.exists():
            validate_original_cache(original, tokens=[], expected_signature=signature)
            self.manifest = read_json(manifest_path)
            self.status[ORIGINAL] = {"path": str(manifest_path),
                                     "sha256": file_sha256(manifest_path)}
        else:
            self.status[ORIGINAL] = {"missing": str(manifest_path)}
        summary_path = edited / "metrics" / "summary.json"
        records_path = edited / "metrics" / "per_sample.json"
        if summary_path.exists() and records_path.exists() and self.manifest is not None:
            self.summary = read_json(summary_path)
            for key, expected in expected_summary.items():
                actual = self.summary.get(key, False if isinstance(expected, bool) else None)
                equal = (np.isclose(actual, expected) if isinstance(expected, float)
                         and isinstance(actual, (int, float)) else actual == expected)
                if not equal:
                    raise ValueError(f"Incompatible natural edited cache {summary_path}: "
                                     f"{key}={actual!r}, expected {expected!r}")
            self.edited_records = {r["token"]: r for r in read_json(records_path)}
            self.edited_tokens = set(self.edited_records)
            self.status[BASELINE] = {"path": str(summary_path),
                                     "sha256": file_sha256(summary_path)}
        else:
            self.status[BASELINE] = {"missing": "Compatible original manifest and natural "
                                    f"edited summary/per_sample required under {edited}"}

    def load(self, token, source_contract, seed):
        results, issues, provenance = {}, {}, {}
        if self.manifest is not None and token in self.manifest["successful_tokens"]:
            if source_contract is not None:
                before = dict(self.manifest.get("source_contracts", {}).get(token, {}))
                after = dict(source_contract)
                for value in (before, after):
                    if value.get("version") == "paiav_archive_camera_frames_v1":
                        value.pop("source_reconstruction_file", None)
                if before != after:
                    raise BaselineMismatch(f"Original source contract differs for {token}")
            names = [ORIGINAL]
        else:
            names = []
            issues[ORIGINAL] = "Token absent from compatible original successes"
        if token in self.edited_tokens:
            seed_key = self.signature["seed_name"].removesuffix("_base")
            if self.edited_records[token].get(seed_key) != seed:
                raise BaselineMismatch(f"Natural edited seed mismatch for {token}")
            names.append(BASELINE)
        else:
            issues[BASELINE] = "Token absent from compatible natural edited metrics"
        for name in names:
            path = (self.original if name == ORIGINAL else self.edited) / "trajectories" / f"{token}.npz"
            try:
                results[name] = load_cached_trajectory(path, expected_interval_length_s=
                                                      self.signature["interval_length_s"])
                provenance[name] = {"path": str(path), "sha256": file_sha256(path)}
            except (FileNotFoundError, ValueError, KeyError) as exc:
                issues[name] = str(exc)
        return results, issues, provenance


def run_fixed_cots(evaluator, args, *, inference):
    """Entry point called only after the evaluator's normal CLI validation."""
    from hydra.utils import instantiate

    e = evaluator
    backend = "alpamayo"
    fixed_path = args.fixed_cots.expanduser().resolve()
    fixed = load_config(fixed_path)
    config_path = e.resolve_file(args.config, "evaluation configuration")
    cfg = yaml.safe_load(expand_config(config_path.read_text()))
    dt = float(cfg.get("trajectory_sampling", {}).get("interval_length", 0.1))
    horizon = float(cfg.get("metric_horizon_s") or 4.0)
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(horizon, 4.0):
        raise ValueError("Fixed-CoT evaluation requires positive sampling interval and four-second horizon")
    if fixed["stop_duration_s"] > horizon:
        raise ValueError("stop_duration_s exceeds the metric horizon")
    e.metric_trajectory(np.zeros((int(round(horizon / dt)), 3)),
                        interval_length_s=dt, metric_horizon_s=horizon)
    seed_key = f"{inference.seed_key}_base"
    seed_base = inference.seed_base
    seed_for = lambda token, base: inference.seed_for_token(token)  # noqa: E731
    with ExitStack() as stack:
        inputs = Inputs(e, args, stack)
        texts = resolve_conditions(fixed, inputs.kind)
        edited = e.predictions_dir(e.MODEL_NAME, args.dataset, inputs.name())
        original = e.predictions_dir(e.MODEL_NAME, args.dataset, inputs.name(original=True))
        if getattr(args, "original_result_dir", None):
            original = args.original_result_dir.expanduser().resolve()
        root = edited.with_name(edited.name + "_fixed_cots")
        signature = cache_signature(
            evaluator=inference.evaluator, config_path=config_path, interval_length_s=dt,
            inference_options=inference.options(inputs.options), seed_name=seed_key, seed_base=seed_base,
            source=inputs.source)
        expected = {"run_dir": str(inputs.run), "interval_length_s": dt,
                    "metric_horizon_s": horizon, seed_key: seed_base,
                    "model_name_or_path": cfg.get("model_name_or_path"),
                    "attn_implementation": cfg.get("attn_implementation"),
                    "vlm_seed": int(cfg.get("seed", 42))}
        baselines = Baselines(original, edited, signature, expected)
        identity = {
            "schema_version": SCHEMA_VERSION, "backend": backend,
            "signature": signature, "run_dir": str(inputs.run),
            "run_config_sha256": file_sha256(inputs.config_path),
            "source_files": inputs.source_files, "fixed_config": fixed,
            "fixed_config_sha256": file_sha256(fixed_path), "conditions": texts,
            "object_type": inputs.kind, "insertion_timestep": inputs.config.get("future_timestep"),
            "metric_horizon_s": horizon, "baseline_dirs": {
                BASELINE: str(edited), ORIGINAL: str(original)},
        }
        prepare_manifest(root, identity, inputs.tokens)
        print(f"Fixed-CoT {backend}: {len(inputs.tokens)} samples × {len(texts)} conditions; {root}", flush=True)
        print(f"Baseline availability: {baselines.status}", flush=True)
        agent = None  # A completely resumed run needs no model/GPU initialization.
        records, failures = [], []
        for index, token in enumerate(inputs.tokens, 1):
            record = {"token": token, "conditions": {}, "contrasts": {}}
            try:
                agent_input, actor, metadata_path, geometry, speed, source_contract = inputs.sample(token)
                provenance = input_provenance(agent_input, metadata_path)
                seed = seed_for(token, seed_base)
                record.update(actor_motion=actor, ego_geometry=geometry,
                              provenance=provenance, seed=seed)
                trajectories, issues, references = baselines.load(token, source_contract, seed)
                record.update(baseline_issues=issues, baseline_provenance=references)
                for name, text in texts.items():
                    intervention = {"condition": name, "forced_cot": text,
                                    "text_sha256": digest(text), "seed": seed,
                                    "input_provenance": provenance,
                                    "identity_sha256": digest(identity)}
                    condition_root = root / name
                    # A conflicting committed result is a hard error, not a failed sample.
                    poses = load_completed(condition_root, token, intervention, dt)
                    try:
                        if poses is None:
                            if agent is None:
                                try:
                                    agent = instantiate(cfg)
                                    agent.initialize()
                                except Exception as exc:
                                    raise ModelInitializationError(str(exc)) from exc
                            prediction = agent.compute_trajectory(agent_input, forced_cot=text,
                                                                  **inference.seed_kwargs(seed))
                            trajectory, output = prediction, prediction.extra
                            actual = output.get("cot")
                            while isinstance(actual, list) and len(actual) == 1:
                                actual = actual[0]
                            if actual != text:
                                raise ValueError("Alpamayo output did not preserve fixed reasoning")
                            poses = e.trajectory_poses(trajectory)
                            e.metric_trajectory(poses, interval_length_s=dt, metric_horizon_s=horizon)
                            save_completed(condition_root, token, poses, output, intervention, dt)
                        trajectories[name] = poses
                    except ModelInitializationError:
                        raise
                    except Exception as exc:
                        failures.append({"token": token, "condition": name,
                                         "stage": "inference", "error": str(exc)})
                        print(f"{token} {name}: {exc}", flush=True)
                metric_poses = {}
                for name, poses in trajectories.items():
                    try:
                        poses = e.metric_trajectory(poses, interval_length_s=dt, metric_horizon_s=horizon)
                        metrics = condition_metrics(poses, dt=dt, initial_speed=speed, config=fixed)
                        record["conditions"][name] = metrics
                        metric_poses[name] = poses
                    except Exception as exc:
                        failures.append({"token": token, "condition": name,
                                         "stage": "metrics", "error": str(exc)})
                for before in metric_poses:
                    for after in CONDITIONS:
                        if after in metric_poses and before != after:
                            record["contrasts"][f"{before}->{after}"] = contrast(
                                metric_poses[before], metric_poses[after],
                                record["conditions"][before], record["conditions"][after])
                # Small arrays permit standalone CPU sample plots with no source caches.
                record["trajectories"] = jsonable(metric_poses)
            except (ResumeConflict, BaselineMismatch, ModelInitializationError):
                raise
            except Exception as exc:
                failures.append({"token": token, "condition": None,
                                 "stage": "sample", "error": str(exc)})
            records.append(record)
            save_json(root / "metrics" / "samples" / f"{token}.json", record)
            save_json(root / "failed_samples.json", failures)
            print(f"[{index}/{len(inputs.tokens)}] {token}: "
                  f"{sum(c in record['conditions'] for c in CONDITIONS)}/5 conditions", flush=True)
        save_json(root / "metrics" / "per_sample.json", records)
        summary = {**aggregate(records, inputs.tokens), "identity": identity,
                   "baselines": baselines.status,
                   "failure_count": len(failures), "result_dir": str(root),
                   "metric_definitions": {
                       "speed": "Planar segment distance / dt, origin included; times are segment midpoints",
                       "stop": "First low-speed segment start with at least stop_duration_s consecutive low-speed segments",
                       "stop_time_null": "No sustained stop within the four-second horizon",
                       "initially_stationary": "Recorded initial speed <= stop_speed_mps; moving_stopping_rate excludes these samples",
                   }}
        save_json(root / "metrics" / "summary.json", summary)
        print(f"Saved fixed-CoT results: {root}; paired comparison count={summary['comparison_count']}", flush=True)
        from ...scoring.paiav import postprocess_evaluation
        postprocess_evaluation(root, args.scoring)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.cot_swap:
        return cot_swap(args)
    config_path = resolve_file(args.config, "Alpamayo evaluation configuration")
    cfg = yaml.safe_load(expand_config(config_path.read_text()))
    return run_fixed_cots(evaluation, args, inference=AlpamayoInference.from_config(cfg))


if __name__ == "__main__":
    main()
