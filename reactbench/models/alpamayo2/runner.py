"""Basic original-cache and paired edited evaluation for both datasets."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import traceback

import numpy as np
import yaml

from ...io import digest, read_json, write_json
from ..common import expand_config, predictions_dir
from ..original_cache import cache_signature, file_sha256, save_original_manifest, validate_original_cache_cohort
from .inference import input_tensors, paired_seed

MODEL_NAME = "alpamayo2"
HERE = Path(__file__).resolve().parent
AGENTS = {"pai-av": "reactbench.models.alpamayo2.agent_paiav.Alpamayo2Agent",
          "navsim": "reactbench.models.alpamayo2.agent_navsim.Alpamayo2Agent"}
CONFIGS = {"pai-av": HERE / "config_paiav.yaml", "navsim": HERE / "config_navsim.yaml"}
DATASET_FOLDERS = {"pai-av": "reactpai", "navsim": "reactnav"}


def parse_args(dataset, argv=None, *, preflight=False):
    parser = argparse.ArgumentParser(
        description=f"Alpamayo2 basic {dataset} {'CPU preflight' if preflight else 'trajectory evaluation'}",
        allow_abbrev=False)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIGS[dataset])
    parser.add_argument("--dataset", default=DATASET_FOLDERS[dataset], help="Dataset folder of the predictions")
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


def load_agent(dataset, path):
    cfg = yaml.safe_load(expand_config(Path(path).read_text()))
    if not isinstance(cfg, dict):
        raise ValueError("Model configuration must be a YAML mapping")
    target = AGENTS[dataset]
    if cfg.get("_target_") != target:
        raise ValueError(f"Basic {dataset} evaluation requires _target_: {target}")
    module, name = target.rsplit(".", 1)
    agent = getattr(importlib.import_module(module), name)(
        **{key: value for key, value in cfg.items() if not key.startswith("_")})
    return cfg, agent


def load_run(dataset, path):
    if dataset == "pai-av":
        from .paiav_loader import PAIAVRun
        return PAIAVRun(path)
    from .navsim_loader import NAVSIMRun
    return NAVSIMRun(path)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def failure(token, stage, exc):
    return {"token": token, "stage": stage, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc()}


def save_prediction(result_dir, token, prediction):
    poses = np.asarray(prediction.poses, dtype=np.float32)
    xyz, rotations = np.asarray(prediction.xyz), np.asarray(prediction.rotations)
    if poses.shape != (64, 3) or xyz.shape != (64, 3) or rotations.shape != (64, 3, 3):
        raise ValueError("Prediction must contain all 64 poses, XYZ positions, and rotation matrices")
    if not all(np.isfinite(value).all() for value in (poses, xyz, rotations)):
        raise ValueError("Prediction contains non-finite values")
    directory = result_dir / "trajectories"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{token}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, trajectory=poses, pred_xyz=xyz, pred_rot=rotations,
                                interval_length_s=np.float32(0.1), ego_origin="rear_axle")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(result_dir / "reasoning" / f"{token}.json", jsonable(prediction.extra))
    return poses


def recorded_future_errors(poses, recorded):
    """Original prediction versus the recorded future; very small errors hint at memorised clips."""
    distances = np.linalg.norm(np.asarray(poses)[:, :2] - np.asarray(recorded)[:, :2], axis=1)
    return {"recorded_future_ade_4s_m": float(distances[:40].mean()),
            "recorded_future_fde_4s_m": float(distances[39]),
            "recorded_future_ade_6p4s_m": float(distances.mean()),
            "recorded_future_fde_6p4s_m": float(distances[-1])}


def summarize_recorded_future(records):
    rows = [row for row in records if "recorded_future_ade_4s_m" in row]
    if not rows and not any("recorded_future_error" in row for row in records):
        return None
    summary = {"purpose": "memorisation diagnostic: original predictions versus the recorded future",
               "count": len(rows),
               "unavailable_count": sum("recorded_future_error" in row for row in records)}
    for field in ("ade_4s", "fde_4s", "ade_6p4s", "fde_6p4s"):
        values = [row[f"recorded_future_{field}_m"] for row in rows]
        summary[f"mean_{field}_m"] = float(np.mean(values)) if values else None
        summary[f"median_{field}_m"] = float(np.median(values)) if values else None
    return summary


def signature_for(dataset, args, agent, loader, hardware):
    identity = agent.identity()
    # Include workspace input/inference code as well as upstream/model commits.
    suffix = "paiav" if dataset == "pai-av" else "navsim"
    files = [HERE / "runner.py", HERE / "inference.py", HERE / f"agent_{suffix}.py",
             HERE / ("paiav_loader.py" if dataset == "pai-av" else "navsim_loader.py"),
             HERE.parent / "alpamayo1_5" / "agent_paiav.py",
             HERE.parent.parent / "datasets" / "paiav.py" if dataset == "pai-av" else HERE.parent / "alpamayo1_5" / "agent_navsim.py"]
    package_root = HERE.parent.parent
    identity["adapter_source_sha256"] = digest({str(p.relative_to(package_root)): file_sha256(p) for p in files})
    source = {"dataset": dataset, "dataset_group": loader.run_dir.parent.name}
    if dataset == "navsim":
        source.update(annotations_path=str(loader.annotations_path),
                      annotations_sha256=file_sha256(loader.annotations_path))
    signature = cache_signature(
        evaluator=f"alpamayo2-{dataset}", config_path=args.config,
        interval_length_s=0.1,
        # Seeds reproduce rollouts only on one GPU architecture; pairs must share it.
        inference_options={"identity": identity, "camera_profile": loader.camera_profile,
                           "text_seed_base": agent.seed, "hardware_class": hardware["class"]},
        seed_name="diffusion_seed_base", seed_base=agent.diffusion_seed_base, source=source)
    return identity, signature


def preflight_main(dataset, argv=None):
    args = parse_args(dataset, argv, preflight=True)
    _, agent = load_agent(dataset, args.config)
    identity = agent.identity()
    failures, successes, checks = [], [], {}
    with load_run(dataset, args.run_dir) as loader:
        tokens = selected_tokens(args, loader.tokens)
        for token in tokens:
            try:
                loader.original_source_contract(token)
                loader.metric_context(token)
                if not args.original:
                    checks[token] = loader.check_edited_inputs(token)
                with loader.materialize_inputs(token, original=args.original) as paths:
                    input_tensors(loader.model_input(token, paths), agent.camera_loader)
                successes.append(token)
            except Exception as exc:
                failures.append(failure(token, "preflight", exc))
        report = {"dataset": dataset, "identity": identity, "input_contract": loader.camera_profile,
                  "run": loader.preflight_report(), "selected_tokens": tokens,
                  "successful_tokens": successes, "edited_input_checks": checks, "failures": failures}
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return int(bool(failures))


def main(dataset, argv=None):
    args = parse_args(dataset, argv)
    cfg, agent = load_agent(dataset, args.config)
    run_dir = args.run_dir.expanduser().resolve()
    result_dir = predictions_dir(MODEL_NAME, args.dataset, "original" if args.original else run_dir.name)
    original_dir = (args.original_result_dir or predictions_dir(MODEL_NAME, args.dataset, "original")).expanduser().resolve()
    if result_dir.exists():
        raise FileExistsError(f"Result directory already exists: {result_dir}. Use a fresh results root.")
    failures, records, successful, contracts = [], [], [], {}
    with load_run(dataset, run_dir) as loader:
        tokens = selected_tokens(args, loader.tokens)
        hardware = agent.hardware()
        identity, signature = signature_for(dataset, args, agent, loader, hardware)
        for token in tokens:
            try:
                contracts[token] = loader.original_source_contract(token)
            except Exception as exc:
                failures.append(failure(token, "source_contract", exc))
        eligible = [token for token in tokens if token in contracts]
        cached = {}
        excluded_original = []
        if not args.original:
            baseline_manifest = read_json(original_dir / "evaluation_manifest.json", {})
            if baseline_manifest.get("mode") != "original":
                raise ValueError("A completed --original cache is required; paired results are not baselines")
            cached_class = (baseline_manifest.get("signature") or {}).get(
                "inference_options", {}).get("hardware_class")
            if cached_class != hardware["class"]:
                raise ValueError(
                    f"The original cache in {original_dir} was computed on {cached_class} GPUs, but this job runs "
                    f"on {hardware['class']}; pairs must share a GPU architecture.")
            cohort = validate_original_cache_cohort(
                original_dir, tokens=tokens, expected_signature=signature,
                expected_source_contracts=contracts)
            cached = cohort.trajectories
            excluded_original = cohort.excluded_failed_tokens
            eligible = [token for token in eligible if token in cached]
            failures.extend({"token": token, "stage": "original_cache",
                             "error": "Original inference failed; excluded from paired evaluation"}
                            for token in excluded_original)
            if any(poses.shape != (64, 3) for poses in cached.values()):
                raise ValueError("Original cache must contain all 64 native poses")
        # Validate the environment, weights and GPU placement before claiming the
        # destination, so an unusable allocation leaves nothing behind.
        if eligible:
            agent.prepare()
        # Claim the destination atomically before expensive model loading. Two
        # simultaneous jobs must never merge predictions or baseline manifests.
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
            text_seed = paired_seed(token, agent.seed, "text")
            diffusion_seed = paired_seed(token, agent.diffusion_seed_base, "diffusion")
            basic = {"token": token, "output_stem": token, "dataset": dataset,
                     "model_name_or_path": cfg["model_name_or_path"], "ego_origin": "rear_axle",
                     "text_seed": text_seed, "diffusion_seed": diffusion_seed,
                     "native_pose_count": 64, "metric_pose_count": 40, "metric_horizon_s": 4.0}
            try:
                if not args.original:
                    stage = "edited_input_check"
                    basic.update(loader.check_edited_inputs(token))
                stage = "original_inference" if args.original else "edited_inference"
                with loader.materialize_inputs(token, original=args.original) as paths:
                    model_input = loader.model_input(token, paths)
                    prediction = agent.compute_trajectory(
                        model_input, text_seed=text_seed, diffusion_seed=diffusion_seed)
                stage = "save_prediction"
                poses = save_prediction(result_dir, token, prediction)
                successful.append(token)
                basic.update(image_timestamps_us=model_input["image_timestamps_us"],
                             navigation_command=model_input.get("navigation_command"))
                if args.original:
                    try:
                        recorded = loader.recorded_future(token)
                        if recorded is not None:
                            basic.update(recorded_future_errors(poses, recorded))
                    except Exception as exc:
                        basic["recorded_future_error"] = f"{type(exc).__name__}: {exc}"
                    records.append(basic)
                else:
                    stage = "metrics"
                    context = loader.metric_context(token)
                    metrics_kwargs = {key: context[key] for key in
                        ("actor_motion", "speed_mps", "longitudinal_acceleration_mps2", "vehicle_geometry")
                        if key in context}
                    if dataset == "navsim":
                        from ...scoring.motion_bins import NAVSIM_BIN_SPEC_ID
                        metrics_kwargs["bin_spec"] = NAVSIM_BIN_SPEC_ID
                    # Bookkeeping only; every metric comes from protocol V2 scoring (postprocess_evaluation).
                    from ...scoring.records import sample_record
                    metrics = sample_record(
                        token=token, output_stem=token, original_trajectory=cached[token][:40],
                        edited_trajectory=poses[:40], interval_length_s=0.1, **metrics_kwargs)
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
        summary.update({"dataset": dataset, "mode": "original" if args.original else "paired",
                        "run_dir": str(run_dir), "run_config_path": str(loader.run_config_path),
                        "annotations_path": str(loader.annotations_path), "result_dir": str(result_dir),
                        "original_result_dir": str(original_dir), "evaluation_config_path": str(args.config.resolve()),
                        "model_name_or_path": cfg["model_name_or_path"], "identity": {"signature": signature, **identity},
                        "interval_length_s": 0.1, "metric_horizon_s": 4.0,
                        "input_contract": loader.camera_profile, "selected_token_count": len(tokens),
                        "successful_inference_count": len(successful),
                        "successful_pair_count": 0 if args.original else len(valid_records),
                        "failed_sample_count": len({row["token"] for row in failures}),
                        "inference_failure_count": sum(row["stage"] not in ("metrics", "edited_input_check")
                                                       for row in failures),
                        "metric_failure_count": sum(row["stage"] == "metrics" for row in failures),
                        "excluded_failed_original_tokens": excluded_original,
                        "edited_input_exclusion_count": sum(row["stage"] == "edited_input_check" for row in failures),
                        "compact_run_preflight": loader.preflight_report(),
                        "hardware": hardware, "gpu_placement": agent.placement_report(),
                        "training_clip_exclusion": (
                            "PhysicalAI-AV is in the model's training data; clip-level split unverified"
                            if dataset == "pai-av" else None)})
        if args.original:
            summary["recorded_future_agreement"] = summarize_recorded_future(records)
        write_json(metrics_dir / "summary.json", jsonable(summary))
        if args.original:
            save_original_manifest(result_dir, signature=signature, source_run_dir=run_dir,
                selected_tokens=tokens, successful_tokens=successful, source_contracts=jsonable(contracts))
        else:
            write_json(result_dir / "evaluation_manifest.json", {
                "schema_version": 1, "mode": "paired", "signature": signature,
                "source_run_dir": str(run_dir), "selected_tokens": tokens,
                "successful_tokens": successful, "source_contracts": jsonable(contracts)})

    if dataset == "navsim":
        from ...scoring.navsim import postprocess_evaluation
        postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)
    else:
        from ...scoring.paiav import postprocess_evaluation
        postprocess_evaluation(result_dir, args.scoring)
    print(f"Saved {len(successful)}/{len(tokens)} predictions to {result_dir}; {len(failures)} failures.", flush=True)
    return 0 if successful else 1
