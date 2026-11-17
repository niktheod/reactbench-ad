"""Fixed CoTs (paper appendix): five hand-written CoTs forced on the edited images, for both Alpamayo models.

Each condition of ``fixed_cots.yaml`` (accelerate, resume speed, slow down, yield, immediate stop; worded per
object) is forced on every edited sample with the token's natural seeds, one directory per condition in
``<variant>_fixed_cots/``; the natural original and edited runs (``natural.py``) are the references. Speeds and stops
are computed here; collision metrics come from protocol V2 scoring (``metrics/paiav_at_fault_*``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from ...io import digest, file_digest, jsonable
from ...models import runner
from ...models.original_cache import save_manifest
from ...scoring.geometry import average_displacement_error, trajectory_distance_driven
from .natural import NaturalMismatch, natural_runs

CONDITIONS = ("accelerate", "resume_speed", "slow_down", "yield", "immediate_stop")
REASONS = {
    "car": "vehicle blocking the lane ahead",
    "standing_pedestrian": "pedestrian blocking the lane ahead",
    "crossing_pedestrian": "pedestrian crossing ahead",
}
BASELINE = "natural_edited"
ORIGINAL = "natural_original"
SCHEMA_VERSION = 1
METRIC_DEFINITIONS = {
    "speed": "Planar segment distance / dt, origin included; times are segment midpoints",
    "stop": "First low-speed segment start with at least stop_duration_s consecutive low-speed segments",
    "stop_time_null": "No sustained stop within the four-second horizon",
    "initially_stationary": "Recorded initial speed <= stop_speed_mps; moving_stopping_rate excludes these samples",
}


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Fixed-CoT config requires version: 1")
    if set(value.get("conditions", {})) != set(CONDITIONS):
        raise ValueError(f"Fixed-CoT config requires exactly {CONDITIONS}")
    for text in value["conditions"].values():
        if not isinstance(text, str) or not text.strip() or "<" in text or ">" in text:
            raise ValueError("CoTs must be nonempty plain text without control tags")
        for reason in REASONS.values():
            try:
                text.format(reason=reason)
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError("Only the {reason} template field is supported") from exc
    for key, default in (("stop_speed_mps", 0.1), ("stop_duration_s", 0.5)):
        number = float(value.get(key, default))
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{key} must be finite and positive")
        value[key] = number
    for key, default in (("sample_plot_count", 20), ("sample_plot_seed", 0)):
        number = value.get(key, default)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        value[key] = number
    return value


def object_type(metadata):
    """The inserted object of a variant's configuration or actor motion (never a directory name)."""
    actor = metadata.get("actor_type", "pedestrian")
    if actor == "car":
        return "car"
    if actor != "pedestrian":
        raise ValueError(f"Unsupported inserted actor: {actor!r}")
    behavior = metadata.get("behavior", metadata.get("actor_behavior", metadata.get("pedestrian_behavior")))
    velocity = metadata.get("velocity_xy_mps")
    speed = np.linalg.norm(velocity) if velocity is not None else metadata.get("speed_mps")
    if speed is None or not np.isfinite(float(speed)) or float(speed) < 0:
        raise ValueError("Pedestrian metadata requires a finite nonnegative speed")
    stationary = float(speed) <= 1e-6
    if behavior not in (None, "standing", "stationary", "crossing", "moving"):
        raise ValueError(f"Unsupported pedestrian behavior: {behavior!r}")
    if behavior in ("standing", "stationary") and not stationary:
        raise ValueError("Standing pedestrian has nonzero speed")
    if behavior in ("crossing", "moving") and stationary:
        raise ValueError("Crossing pedestrian has zero speed")
    return "standing_pedestrian" if stationary else "crossing_pedestrian"


def resolve_conditions(config, kind):
    return {name: config["conditions"][name].format(reason=REASONS[kind]) for name in CONDITIONS}


def input_provenance(loader, token, model_input, contract, metadata_path):
    """Fingerprint of the edited inputs: the model input, the edited image files and the archive sources."""
    from ...models.original_cache import source_contract_identity
    images = {str(camera): [file_digest(Path(path)) for path in paths]
              for camera, paths in loader.edited_camera_paths(token).items()}
    inputs = {key: value for key, value in model_input.items() if key != "camera_paths"}
    return {"model_input_sha256": digest(jsonable(inputs)), "edited_image_sha256": images,
            "source_contract_sha256": digest(source_contract_identity(contract)),
            "metadata_path": str(metadata_path), "metadata_sha256": file_digest(Path(metadata_path))}


def speed_metrics(poses, dt, initial_speed, config):
    """Segment-average planar speeds, including origin to the first pose."""
    xy = np.vstack(([0.0, 0.0], np.asarray(poses)[:, :2]))
    speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    threshold = config["stop_speed_mps"]
    window = max(1, int(math.ceil(config["stop_duration_s"] / dt - 1e-9)))
    low = speeds <= threshold
    start = next((i for i in range(len(low) - window + 1) if low[i:i + window].all()), None)
    return {
        "speed_mps": speeds.tolist(),
        "speed_time_s": ((np.arange(len(speeds)) + 0.5) * dt).tolist(),
        "final_speed_mps": float(speeds[-1]),
        "initial_speed_mps": float(initial_speed),
        "initially_stationary": bool(initial_speed <= threshold),
        "sustained_stop": start is not None,
        "stop_time_s": None if start is None else float(start * dt),
    }


def condition_metrics(poses, *, dt, initial_speed, config):
    return {"distance_driven_m": trajectory_distance_driven(poses), **speed_metrics(poses, dt, initial_speed, config)}


def contrast(before_poses, after_poses, before, after):
    return {"ade_m": average_displacement_error(before_poses, after_poses),
            **{f"{key}_delta": after[key] - before[key] for key in ("distance_driven_m", "final_speed_mps")}}


def aggregate(records, tokens):
    complete = [r for r in records if all(c in r["conditions"] for c in CONDITIONS)]
    paired = [r for r in complete if BASELINE in r["conditions"]]

    # All main bars/curves use precisely this cohort; fixed-only stays available when natural runs are missing.
    def summarize(cohort, names):
        result = {}
        for name in names:
            values = [r["conditions"][name] for r in cohort]
            moving = [v for v in values if not v["initially_stationary"]]
            mean = lambda key, rows=values: (float(np.mean([v[key] for v in rows])) if rows else None)  # noqa: E731
            result[name] = {
                "count": len(values), "stopping_rate": mean("sustained_stop"),
                "moving_stopping_rate": mean("sustained_stop", moving), "moving_count": len(moving),
                "distance_driven_m": mean("distance_driven_m"), "final_speed_mps": mean("final_speed_mps"),
            }
        return result
    contrasts = {}
    for before in (*CONDITIONS, BASELINE, ORIGINAL):
        cohort = [r for r in complete if before in r["conditions"]]
        for after in CONDITIONS:
            if before == after:
                continue
            key = f"{before}->{after}"
            values = [r["contrasts"][key] for r in cohort if key in r["contrasts"]]
            contrasts[key] = {"count": len(values), **{
                field: float(np.mean([v[field] for v in values])) if values else None
                for field in ("ade_m", "distance_driven_m_delta", "final_speed_mps_delta")}}
    return {
        "requested_count": len(tokens), "fixed_complete_count": len(complete), "comparison_count": len(paired),
        "comparison_tokens": [r["token"] for r in paired],
        "condition_counts": {name: {
            "attempted": len(tokens), "successful": sum(name in r["conditions"] for r in records),
            "failed": len(tokens) - sum(name in r["conditions"] for r in records)} for name in CONDITIONS},
        "conditions": summarize(paired, (*CONDITIONS, BASELINE)),
        "fixed_only_conditions": summarize(complete, CONDITIONS),
        "contrasts": contrasts,
    }


def run(s) -> int:
    """The fixed CoTs of ``s.tokens`` (``natural.main`` prepares ``s``); 0 if any condition succeeded."""
    model, loader, root, dt = s.model, s.model.loader, s.result_dir, s.model.interval_length_s
    fixed_path = s.args.fixed_cots.expanduser().resolve()
    fixed = load_config(fixed_path)
    if fixed["stop_duration_s"] > runner.METRIC_HORIZON_S:
        raise ValueError("stop_duration_s exceeds the metric horizon")
    kind = object_type(loader.run_config)
    texts = resolve_conditions(fixed, kind)
    for text in texts.values():
        model.forced_prefix(text)
    # A missing natural run is reported (the fixed-only cohort remains); an incompatible one is fatal.
    natural, baselines = natural_runs(s, required=False)
    natural = {ORIGINAL: natural["original"], BASELINE: natural["edited"]}
    baselines = {ORIGINAL: baselines["original"], BASELINE: baselines["edited"]}
    # Hashed into every intervention: the model's settings, the fixed CoTs and the inputs they apply to.
    identity = {
        "schema_version": SCHEMA_VERSION, "signature": s.signature, "run_dir": str(model.variant_dir),
        "run_config_sha256": file_digest(loader.run_config_path), "fixed_config": fixed,
        "fixed_config_sha256": file_digest(fixed_path), "conditions": texts, "object_type": kind,
        "insertion_timestep": loader.run_config.get("future_timestep"),
        "metric_horizon_s": runner.METRIC_HORIZON_S,
        "baseline_dirs": {BASELINE: str(s.edited_dir), ORIGINAL: str(s.original_dir)}, **model.cot_identity()}
    runner.prepare_output(root, s.args.overwrite, remove=True)
    print(f"Fixed CoTs: {len(s.tokens)} samples x {len(texts)} conditions; {root}", flush=True)
    print(f"Baseline availability: {baselines}", flush=True)
    model.load()
    records, failures, successful = [], [], []
    for index, token in enumerate(s.tokens, 1):
        seeds = model.seeds(token)
        record = {"token": token, "output_stem": token, "conditions": {}, "contrasts": {}, **seeds}
        stage = "source_contract"
        try:
            contract = loader.original_source_contract(token)
            stage = "edited_input_check"
            record.update(model.check_edited_inputs(token))
            stage = "metric_context"
            context = model.metric_context(token)
            actor, speed = context["actor_motion"], context["speed_mps"]
            if object_type(actor) != kind:
                raise ValueError(f"Actor metadata differs from the variant's object type for {token}")
            record.update(actor_motion=actor, ego_geometry=context.get("vehicle_geometry"), speed_mps=speed,
                          longitudinal_acceleration_mps2=context.get("longitudinal_acceleration_mps2"),
                          placement_metadata_path=context["placement_metadata_path"], ego_origin="rear_axle")
            stage = "natural_baselines"
            trajectories, issues, references = {}, {}, {}
            for name, run_ in natural.items():
                reason = "No compatible natural run" if run_ is None else run_.unavailable(token)
                if reason:
                    issues[name] = reason
                    continue
                try:
                    output = run_.load(token, contract, seeds)
                except NaturalMismatch:
                    raise
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    issues[name] = f"{type(exc).__name__}: {exc}"
                    continue
                trajectories[name] = output.poses
                references[name] = {"path": str(output.trajectory_path), "sha256": output.trajectory_sha256}
            record.update(baseline_issues=issues, baseline_provenance=references)
            stage = "edited_inputs"
            with model.inputs(token, False) as model_input:
                provenance = input_provenance(loader, token, model_input, contract, context["placement_metadata_path"])
                record["provenance"] = provenance
                for name, text in texts.items():
                    intervention = {"condition": name, "forced_cot": text, "assistant_prefix": model.forced_prefix(text),
                                    "text_sha256": digest(text), **seeds, "input_provenance": provenance,
                                    "identity_sha256": digest(jsonable(identity))}
                    try:
                        prediction = model.force(model_input, text, seeds)
                        poses = runner.trajectory_poses(prediction)
                        path = root / name / "trajectories" / f"{token}.npz"
                        runner.save_trajectory(path, poses, dt, **model.arrays(prediction))
                        runner.save_json(root / name / "reasoning" / f"{token}.json", {
                            "intervention": intervention, "model_output": prediction.extra,
                            "trajectory_sha256": file_digest(path)})
                        trajectories[name] = poses
                    except Exception as exc:
                        failures.append(runner.failure(index, token, f"{name}_inference", exc, **seeds))
                        print(f"{token} {name}: {exc}", flush=True)
            stage = "metrics"
            metrics = {}
            for name, poses in trajectories.items():
                try:
                    poses = runner.metric_poses(poses, dt)
                    record["conditions"][name] = condition_metrics(poses, dt=dt, initial_speed=speed, config=fixed)
                    metrics[name] = poses
                except Exception as exc:
                    failures.append(runner.failure(index, token, f"{name}_metrics", exc, **seeds))
            for before in metrics:
                for after in CONDITIONS:
                    if after in metrics and before != after:
                        record["contrasts"][f"{before}->{after}"] = contrast(
                            metrics[before], metrics[after], record["conditions"][before], record["conditions"][after])
            # Small arrays permit standalone CPU sample plots with no source caches.
            record["trajectories"] = jsonable(metrics)
        except NaturalMismatch:
            raise
        except Exception as exc:
            failures.append(runner.failure(index, token, stage, exc, **seeds))
        if record["conditions"].keys() & set(CONDITIONS):
            successful.append(token)
        records.append(record)
        runner.save_json(root / "failed_samples.json", failures)
        print(f"[{index}/{len(s.tokens)}] {token}: "
              f"{sum(name in record['conditions'] for name in CONDITIONS)}/{len(CONDITIONS)} conditions", flush=True)
    runner.save_json(root / "metrics" / "per_sample.json", records)
    summary = {**aggregate(records, s.tokens), "mode": "fixed_cots", "model": model.model, "dataset": model.dataset,
               "run_dir": str(model.variant_dir), "identity": identity, "baselines": baselines,
               "failure_count": len(failures), "result_dir": str(root), "interval_length_s": dt,
               "metric_horizon_s": runner.METRIC_HORIZON_S, "gpus": runner.gpu_names(),
               "metric_definitions": METRIC_DEFINITIONS, **model.summary_extras()}
    runner.save_json(root / "metrics" / "summary.json", summary)
    save_manifest(root, mode="fixed_cots", signature=s.signature, source_run_dir=model.variant_dir,
                  selected_tokens=s.tokens, successful_tokens=successful, identity=identity)
    print(f"Saved fixed-CoT results: {root}; paired comparison count={summary['comparison_count']}", flush=True)
    return 0 if successful else 1
