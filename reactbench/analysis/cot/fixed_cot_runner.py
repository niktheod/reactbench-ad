"""Fixed-CoT interventions of Alpamayo 1.5 on ReactPAI (paper appendix): replace the generated CoT.

Called by ``python -m reactbench.models.alpamayo1_5.eval_paiav RUN_DIR --fixed-cots
reactbench/analysis/cot/fixed_cots.yaml``; reuses the run's natural original and edited predictions.
"""

from __future__ import annotations

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
        expected.update(inference.identity_metadata(config_path))
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
