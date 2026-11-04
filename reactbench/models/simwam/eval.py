"""Basic SimWAM original-cache and paired edited evaluation on NAVSIM compositor runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import yaml

from ...io import digest, read_json, write_json
from ..common import expand_config, predictions_dir
from ..original_cache import (
    cache_signature, file_sha256, save_original_manifest, validate_original_cache_cohort,
)
from .agent import SimWAMAgent
from .navsim_loader import NAVSIMRun


CONFIG_DIR = Path(__file__).with_name("configs")
AGENT_TARGET = "reactbench.models.simwam.agent.SimWAMAgent"
RUNNER_KEYS = ("_target_", "name", "result_suffix")


def parse_args(argv=None, *, preflight=False):
    parser = argparse.ArgumentParser(
        description=f"SimWAM NAVSIM {'CPU preflight' if preflight else 'trajectory evaluation'}",
        allow_abbrev=False)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", default="rl", help="rl, il, or a path")
    parser.add_argument("--dataset", default="reactnav")
    parser.add_argument("--original", action="store_true", help="Use pristine images; cache originals in evaluation mode")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-samples", type=int)
    selection.add_argument("--tokens-file", type=Path, help="JSON object with an exact ordered 'tokens' list")
    if preflight:
        parser.add_argument("--report", type=Path, help="Optionally save the CPU preflight report")
    else:
        parser.add_argument("--original-result-dir", type=Path)
        parser.add_argument("--scoring", choices=("score", "off"), default="score")
        parser.add_argument("--scoring-workers", type=int, default=1)
    args = parser.parse_args(argv)
    path = CONFIG_DIR / f"{args.config}.yaml"
    args.config = path if path.is_file() else Path(args.config)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if not preflight and args.original and args.original_result_dir is not None:
        parser.error("--original-result-dir is only for edited evaluation")
    return args


def selected_tokens(args, available):
    tokens = list(available)
    if args.tokens_file is not None:
        document = read_json(args.tokens_file)
        tokens = document.get("tokens") if isinstance(document, dict) else None
        if (not isinstance(tokens, list) or not tokens
                or any(not isinstance(token, str) or not token for token in tokens)
                or len(tokens) != len(set(tokens))):
            raise ValueError("Token manifest needs a nonempty list of unique strings")
        if set(tokens) - set(available):
            raise ValueError("Token manifest requests samples missing from the edited run")
    elif args.max_samples is not None:
        tokens = tokens[:args.max_samples]
    return tokens


def load_agent(path):
    cfg = yaml.safe_load(expand_config(Path(path).read_text()))
    if not isinstance(cfg, dict) or cfg.get("_target_") != AGENT_TARGET:
        raise ValueError(f"SimWAM configuration must be a mapping with _target_: {AGENT_TARGET}")
    suffix = str(cfg.get("result_suffix") or "")
    if Path(suffix).name != suffix or "/" in suffix:
        raise ValueError("result_suffix must be a plain name suffix")
    return cfg, SimWAMAgent(**{key: value for key, value in cfg.items() if key not in RUNNER_KEYS}), suffix


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def failure(token, stage, exc):
    return {"token": token, "stage": stage, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc()}


def save_prediction(result_dir, token, poses, *, interval_length_s):
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise ValueError(f"Prediction must be finite (N, 3) poses, got {poses.shape}")
    directory = result_dir / "trajectories"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{token}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, trajectory=poses, interval_length_s=np.float32(interval_length_s),
                                ego_origin="rear_axle")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return poses


def hardware_class(device):
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.device(device))


def signature_for(args, agent, loader, prepared, hardware):
    identity = agent.identity()
    files = [Path(__file__), Path(__file__).with_name("agent.py"), Path(__file__).with_name("navsim_loader.py")]
    identity["adapter_source_sha256"] = digest({p.name: file_sha256(p) for p in files})
    source = {"dataset": "navsim", "dataset_group": loader.run_dir.parent.name,
              "annotations_path": str(loader.annotations_path),
              "annotations_sha256": file_sha256(loader.annotations_path)}
    signature = cache_signature(
        evaluator="simwam-navsim", config_path=args.config, interval_length_s=agent.interval_length_s,
        # Pairs must share one GPU architecture so only the image differs between members.
        inference_options={"identity": identity, "input_profile": loader.input_profile,
                           "prompt_embedding_sha256": prepared["prompt_embedding_sha256"],
                           "data_settings": prepared["data_settings"], "hardware_class": hardware},
        seed_name="action_noise_seed_base", seed_base=agent.seed, source=source)
    return identity, signature


def preflight_main(argv=None):
    args = parse_args(argv, preflight=True)
    _, agent, _ = load_agent(args.config)
    failures, successes, checks = [], [], {}
    with NAVSIMRun(args.run_dir) as loader:
        tokens = selected_tokens(args, loader.tokens)
        for token in tokens:
            try:
                loader.original_source_contract(token)
                loader.metric_context(token)
                if not args.original:
                    checks[token] = loader.check_edited_inputs(token)
                loader.model_input(token, loader.image_path(token, original=args.original))
                successes.append(token)
            except Exception as exc:
                failures.append(failure(token, "preflight", exc))
        report = {"dataset": "navsim", "identity": agent.identity(), "input_contract": loader.input_profile,
                  "run": loader.preflight_report(), "selected_tokens": tokens,
                  "successful_tokens": successes, "edited_input_checks": checks, "failures": failures}
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return int(bool(failures))


def main(argv=None):
    args = parse_args(argv)
    args.config = args.config.expanduser().resolve()
    cfg, agent, suffix = load_agent(args.config)
    run_dir = args.run_dir.expanduser().resolve()
    name = cfg.get("name", "simwam")
    result_dir = predictions_dir(name, args.dataset, ("original" if args.original else run_dir.name) + suffix)
    original_dir = (args.original_result_dir or predictions_dir(name, args.dataset, f"original{suffix}")).expanduser().resolve()
    if result_dir.exists():
        raise FileExistsError(f"Result directory already exists: {result_dir}. Use a fresh results root.")
    failures, records, successful, contracts = [], [], [], {}
    interval = agent.interval_length_s
    with NAVSIMRun(run_dir) as loader:
        tokens = selected_tokens(args, loader.tokens)
        prepared = agent.prepare()
        hardware = hardware_class(agent.device)
        identity, signature = signature_for(args, agent, loader, prepared, hardware)
        for token in tokens:
            try:
                contracts[token] = loader.original_source_contract(token)
            except Exception as exc:
                failures.append(failure(token, "source_contract", exc))
        eligible = [token for token in tokens if token in contracts]
        cached, excluded_original = {}, []
        if not args.original:
            baseline_manifest = read_json(original_dir / "evaluation_manifest.json", {})
            if baseline_manifest.get("mode") != "original":
                raise ValueError(f"A completed --original cache is required at {original_dir}")
            cached_class = (baseline_manifest.get("signature") or {}).get(
                "inference_options", {}).get("hardware_class")
            if cached_class != hardware:
                raise ValueError(f"The original cache in {original_dir} was computed on {cached_class}, but this "
                                 f"job runs on {hardware}; pairs must share a GPU architecture.")
            cohort = validate_original_cache_cohort(
                original_dir, tokens=eligible, expected_signature=signature, expected_source_contracts=contracts)
            cached, excluded_original = cohort.trajectories, cohort.excluded_failed_tokens
            eligible = [token for token in eligible if token in cached]
            failures.extend({"token": token, "stage": "original_cache",
                             "error": "Original inference failed; excluded from paired evaluation"}
                            for token in excluded_original)
        # Claim the destination atomically; simultaneous jobs must never merge outputs.
        result_dir.parent.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(exist_ok=False)
        if eligible:
            try:
                agent.load()
            except BaseException:
                try:
                    result_dir.rmdir()  # Still empty: release the claim for a rerun.
                except OSError:
                    pass
                raise
        for index, token in enumerate(eligible, 1):
            stage = "input"
            seed = agent.rollout_seed(token)
            basic = {"token": token, "output_stem": token, "dataset": "navsim", "ego_origin": "rear_axle",
                     "checkpoint_file": identity["checkpoint_file"], "action_noise_seed": seed,
                     "native_pose_count": 8, "metric_horizon_s": 4.0}
            try:
                if not args.original:
                    stage = "edited_input_check"
                    basic.update(loader.check_edited_inputs(token))
                stage = "original_inference" if args.original else "edited_inference"
                image_path = loader.image_path(token, original=args.original)
                model_input = loader.model_input(token, image_path)
                poses = agent.compute_trajectory(model_input, rollout_seed=seed)
                stage = "save_prediction"
                poses = save_prediction(result_dir, token, poses, interval_length_s=interval)
                successful.append(token)
                basic.update(image_path=str(image_path), current_timestamp_us=model_input["current_timestamp_us"],
                             ego_state=model_input["state"].tolist())
                if args.original:
                    records.append(basic)
                else:
                    stage = "metrics"
                    # Bookkeeping only; every metric comes from protocol V2 scoring (postprocess_evaluation).
                    from ...scoring.records import sample_record
                    context = loader.metric_context(token)
                    metrics = sample_record(
                        token=token, output_stem=token,
                        actor_motion=context["actor_motion"], speed_mps=context["speed_mps"],
                        longitudinal_acceleration_mps2=context["longitudinal_acceleration_mps2"])
                    records.append({**metrics, **basic, **context})
            except Exception as exc:
                failures.append(failure(token, stage, exc))
                if stage == "metrics":
                    records.append({**basic, "metrics_status": "failed"})
            print(f"[{index}/{len(eligible)}] {token}: {stage}", flush=True)

        # Commit inference artifacts before independent CPU scene scoring.
        metrics_dir = result_dir / "metrics"
        write_json(result_dir / "failed_samples.json", failures)
        write_json(metrics_dir / "per_sample.json", jsonable(records))
        valid_records = [row for row in records if "metrics_status" not in row]
        from ...scoring.records import run_summary
        summary = run_summary(valid_records)
        summary.update({"dataset": args.dataset, "model": name, "mode": "original" if args.original else "paired",
                        "run_dir": str(run_dir), "run_config_path": str(loader.run_config_path),
                        "annotations_path": str(loader.annotations_path), "result_dir": str(result_dir),
                        "original_result_dir": str(original_dir), "evaluation_config_path": str(args.config),
                        "checkpoint_path": cfg["checkpoint_path"], "identity": {"signature": signature, **identity},
                        "model_load": {key: agent.load_report.get(key) for key in
                                       ("checkpoint_step", "missing_keys", "model_paths", "prompt_embedding")},
                        "interval_length_s": interval, "metric_horizon_s": 4.0,
                        "trajectory_convention": "8 x/y/heading poses at 0.5 s, current rear-axle frame, "
                                                 "current pose excluded",
                        "input_contract": loader.input_profile, "selected_token_count": len(tokens),
                        "successful_inference_count": len(successful),
                        "successful_pair_count": 0 if args.original else len(valid_records),
                        "failed_sample_count": len({row["token"] for row in failures}),
                        "inference_failure_count": sum(row["stage"] not in ("metrics", "edited_input_check")
                                                       for row in failures),
                        "metric_failure_count": sum(row["stage"] == "metrics" for row in failures),
                        "edited_input_exclusion_count": sum(row["stage"] == "edited_input_check" for row in failures),
                        "excluded_failed_original_tokens": excluded_original,
                        "hardware_class": hardware})
        write_json(metrics_dir / "summary.json", jsonable(summary))
        if args.original:
            save_original_manifest(result_dir, signature=signature, source_run_dir=run_dir,
                                   selected_tokens=tokens, successful_tokens=successful,
                                   source_contracts=jsonable(contracts))
        else:
            write_json(result_dir / "evaluation_manifest.json", {
                "schema_version": 1, "mode": "paired", "signature": signature,
                "source_run_dir": str(run_dir), "selected_tokens": tokens,
                "successful_tokens": successful, "source_contracts": jsonable(contracts)})

    from ...scoring.navsim import postprocess_evaluation
    postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)
    print(f"Saved {len(successful)}/{len(tokens)} predictions to {result_dir}; {len(failures)} failures.", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
