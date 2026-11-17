"""Score saved ReactPAI predictions (protocol V2: at-fault NC, NCins, contact speed, ADE).

Usage::

    python -m reactbench.scoring.paiav RESULT_DIR [RESULT_DIR ...] [--output-root DIR]

Each run gets ``metrics/paiav_at_fault_{per_sample,summary,manifest}.json``. The recorded PAI-AV
obstacle labels come with the ReactPAI download (``python -m reactbench.datasets.download_reactpai``).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

from .. import paths
from ..io import digest, file_digest, read_json, write_json
from . import results as saved
from .motion_bins import motion_binning_policy, speed_bin_label, speed_bin_sort_key
from .collisions import paired_outcome, score_trajectory, validate_trajectory
from .paiav_world import BackgroundWorlds
from .at_fault import CONTACT_SPEED_CONTRACT as AT_FAULT_CONTACT_SPEED

PREFIX = "paiav_at_fault"
SCOPES = ("inserted", "background", "all")
DEFAULTS = {"version": 1, "workers": 4, "horizon_s": 4.0, "interval_length_s": 0.1, "geometry_interval_s": 0.01,
            "max_track_gap_s": 0.25,
            # Evaluate only within each annotated track's lifetime; never extrapolate.
            "track_lifetime_policy": "annotated_lifetime"}


def load_config(overrides=None):
    """Data locations come from configs/paths.yaml; ``overrides`` replaces individual keys."""
    config = {**DEFAULTS,
              "data_root": str(paths.get("paiav_root", required=False) or ""),
              "selection_manifest": str(paths.get("reactpai_manifest", "data/reactpai/selection_manifest.json")),
              # The obstacle labels are part of the PAI-AV download (download_reactpai).
              "obstacle_root": str(paths.get("paiav_root", required=False) or ""),
              "work_root": str(paths.get("scoring_work_paiav", paths.outputs("work", "paiav")))}
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ("data_root", "selection_manifest", "obstacle_root", "work_root"):
        config[key] = str(paths.resolve(config[key]))
    if config["version"] != 1 or config["track_lifetime_policy"] != "annotated_lifetime":
        raise ValueError("Unsupported collision schema or track-lifetime policy")
    if not isinstance(config["workers"], int) or isinstance(config["workers"], bool) or config["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    for key in ("geometry_interval_s", "max_track_gap_s"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["geometry_interval_s"] != 0.01:
        raise ValueError("Version 1 preserves the existing 0.01 s geometry grid")
    validate_trajectory(np.zeros((40, 3)), config["interval_length_s"], config["horizon_s"])
    return config


def implementation_fingerprint(config):
    """Identity of the physical-contact code and settings (keys the contact cache)."""
    here = Path(__file__).parent
    settings = {k: config[k] for k in ("version", "horizon_s", "interval_length_s", "geometry_interval_s",
                                        "max_track_gap_s", "track_lifetime_policy")}
    return digest({"code": {name: file_digest(here / name) for name in (
        "collisions.py", "paiav_world.py", "geometry.py", "actor_motion.py", "pedestrian_geometry.py")},
        "loader": file_digest(here.parent / "datasets" / "paiav.py"), "settings": settings})


def at_fault_fingerprint(config):
    """Artifact identity: physical contacts plus the at-fault rules and ADE code."""
    here = Path(__file__).parent
    return digest({"physics": implementation_fingerprint(config),
                   "code": {name: file_digest(here / name) for name in ("at_fault.py", "ade.py", "paiav.py")}})


def inventory_result(root, *, variant_dir=None):
    inventory = saved.inventory_result(root, variant_dir=variant_dir)
    root = Path(root)
    summary = read_json(root / "metrics/summary.json", {})
    manifest = read_json(root / "evaluation_manifest.json", {})
    identity = summary.get("identity", manifest.get("identity", {}))
    recorded = summary.get("run_dir", identity.get("run_dir", manifest.get("source_run_dir")))
    inventory["run_dir"] = (str(Path(variant_dir).resolve()) if variant_dir else
                            str((root / recorded).resolve()) if recorded else None)
    records = {str(r["token"]): r for r in read_json(root / "metrics/per_sample.json", [])}
    for task in inventory["tasks"]:
        if inventory["dataset"] != "pai-av":
            task.update(status="inapplicable", reason="This scorer supports only PAI-AV")
        elif task["status"] == "unsupported":
            task["status"] = "ready"
            task.pop("reason", None)
        row = records.get(task["token"], {})
        for name, contrast in row.get("contrasts", {}).items():
            if "from_condition" in contrast and "to_condition" in contrast:
                task["pairs"][f"contrast/{name}"] = [contrast["from_condition"], contrast["to_condition"]]
        if inventory["mode"] == "fixed_cots":
            for name in identity.get("conditions", {}):
                task["pairs"][f"intervention/{name}"] = ["natural_edited", name]
    return inventory


def _poses(source):
    if source.get("error"):
        raise ValueError(source["error"])
    if "poses" in source:
        poses = np.asarray(source["poses"])
    else:
        path = Path(source["path"])
        if file_digest(path) != source.get("sha256"):
            raise ValueError("Saved trajectory changed after inventory")
        with np.load(path, allow_pickle=False) as archive:
            poses = archive["trajectories"][source["branch_index"]] if "branch_index" in source else archive["trajectory"]
    return validate_trajectory(poses, source["interval_length_s"])


def _score_cached(source, actor, geometry, world, world_fingerprint, config, contract):
    poses = _poses(source)
    fingerprint = digest({"poses": poses.tolist(), "actor": actor, "geometry": geometry,
                          "world": world_fingerprint, "contract": contract})
    path = Path(config["work_root"]) / "scores" / f"{fingerprint}.json"
    cached = read_json(path)
    if cached and cached.get("fingerprint") == fingerprint and cached.get("payload_sha256") == digest(cached["score"]):
        return {**cached["score"], "fingerprint": fingerprint}
    score = score_trajectory(poses, actor, geometry, world, interval_length_s=source["interval_length_s"],
                             geometry_interval_s=config["geometry_interval_s"])
    write_json(path, {"fingerprint": fingerprint, "score": score, "payload_sha256": digest(score)})
    return {**score, "fingerprint": fingerprint}


def _row(task, loader, worlds, config, contract):
    from .actor_motion import actor_motion_from_metadata, actor_footprint_corners
    row = {"token": task["token"], "world": task["world"], "status": task["status"], "reason": task.get("reason"),
           "groups": dict(task["groups"]), "scores": {}, "pairs": {}, "primary_pair": task["primary_condition"],
           "original_collision_all": None, "edited_collision_all": None, "collision_benefit": None,
           "contact_speed_reduction_mps": None,
           "speed_kmh": None, "longitudinal_acceleration_mps2": None,
           "interaction_difference_in_differences": None,
           "prediction_sources": task["predictions"], "errors": {}}
    if task["status"] != "ready":
        return row
    try:
        token = task["token"]
        context = loader.collision_context(token)
        world, fingerprint = worlds.load(context)
        actor = None
        if task["world"] == "augmented":
            metadata, path = loader.metadata(token)
            actor = actor_motion_from_metadata(metadata, camera_name=None)
            if task.get("actor_motion"):
                for t in (0.0, config["horizon_s"]):
                    if not np.allclose(actor_footprint_corners(actor, t), actor_footprint_corners(task["actor_motion"], t), atol=1e-6):
                        raise ValueError("Saved insertion motion differs from the rendering metadata")
            row["metadata_sha256"] = file_digest(path)
            row["groups"].update(actor_type=actor.get("actor_type"), placement_timestep=actor.get("future_timestep"))
        row.update(world_fingerprint=digest({"background": fingerprint, "actor": actor}),
                   background_fingerprint=fingerprint, world_coverage=world["coverage"],
                   actor_motion=actor,
                   geometry=context["ego_geometry"], current_timestamp_us=context["current_timestamp_us"])
        # Retain numeric measurements for rebinning; reporting must not reject valid scores.
        for value_field, field, compute, label_for in (
            ("speed_kmh", "speed_bin_kmh", lambda: loader.current_speed_mps(token) * 3.6, speed_bin_label),
            ("longitudinal_acceleration_mps2", None, lambda: loader.current_longitudinal_acceleration_mps2(token), None),
        ):
            if loader is None:
                continue
            try:
                value = compute()
                if field is not None:
                    row["groups"][field] = label_for(value)
                row[value_field] = float(value) if value is not None else None
            except (ValueError, TypeError) as exc:
                if field is not None:
                    row["groups"][field] = None
                row.setdefault("bin_warnings", {})[field or value_field] = str(exc)
        from .at_fault import classify_score
        for name, source in task["predictions"].items():
            try:
                physical = _score_cached(source, actor, context["ego_geometry"], world, fingerprint, config, contract)
                score = classify_score(physical, _poses(source), source["interval_length_s"], actor_motion=actor,
                                       tracks=world["tracks"], ego_geometry=context["ego_geometry"], dataset="pai-av")
                row["scores"][name] = {**score, "nc": int(not score["all"]), "nc_inserted": int(not score["inserted"]),
                                       "physical_fingerprint": physical["fingerprint"]}
            except Exception as exc:
                row["scores"][name] = None
                row["errors"][name] = f"{type(exc).__name__}: {exc}"
        if task["world"] == "augmented" and {"original", "edited"} <= set(task["predictions"]):
            from .ade import paired_ade
            try:
                row["ade_m"] = paired_ade({k: task["predictions"][k] for k in ("original", "edited")})
            except Exception as exc:
                row["ade_error"] = f"{type(exc).__name__}: {exc}"
        for name, (before, after) in task["pairs"].items():
            a, b = row["scores"].get(before), row["scores"].get(after)
            row["pairs"][name] = ({"from_condition": before, "to_condition": after,
                                    **paired_outcome(a, b)} if a is not None and b is not None else None)
        factorial = [row["scores"].get(name) for name in (
            "original_image_original_cot", "edited_image_original_cot",
            "original_image_edited_cot", "edited_image_edited_cot")]
        if all(value is not None for value in factorial):
            oo, eo, oe, ee = factorial
            row["interaction_difference_in_differences"] = {
                scope: (int(ee[scope]) - int(oe[scope])) - (int(eo[scope]) - int(oo[scope])) for scope in SCOPES}
        primary = row["pairs"].get(row["primary_pair"])
        if primary:
            row.update(original_collision_all=primary["all"]["original_collision"],
                       edited_collision_all=primary["all"]["edited_collision"], collision_benefit=primary["all"]["benefit"],
                       contact_speed_reduction_mps=primary["contact_speed"]["reduction_mps"])
        row["status"] = "partial" if row["errors"] else "scored"
    except Exception as exc:
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return row


def contact_speed_summary(pairs):
    """Describe paired contact speeds without treating avoidance/missing data as zero."""
    values = [p["contact_speed"] for p in pairs if p.get("contact_speed", {}).get("status") == "scored"]
    deltas = np.array([v["reduction_mps"] for v in values])
    count = len(values)
    return {"count": count, "valid_pair_count": len(pairs),
            "both_collide_count": sum(p["all"]["original_collision"] and p["all"]["edited_collision"] for p in pairs),
            "status_counts": dict(Counter(p.get("contact_speed", {}).get("status", "missing_contact_speed") for p in pairs)),
            "mean_from_speed_mps": float(np.mean([v["from_speed_mps"] for v in values])) if count else None,
            "mean_to_speed_mps": float(np.mean([v["to_speed_mps"] for v in values])) if count else None,
            "mean_reduction_mps": float(np.mean(deltas)) if count else None,
            "median_reduction_mps": float(np.median(deltas)) if count else None,
            "lower_speed_count": int(np.sum(deltas > 1e-6)),
            "higher_speed_count": int(np.sum(deltas < -1e-6)),
            "unchanged_speed_count": int(np.sum(np.abs(deltas) <= 1e-6)),
            "unchanged_tolerance_mps": 1e-6,
            "shared_first_actor_count": sum(v["shared_first_contact_actor"] for v in values),
            "different_first_actor_count": sum(not v["shared_first_contact_actor"] for v in values)}


def aggregate(rows):
    coverage = dict(Counter(r["status"] for r in rows))
    pair_names = sorted({name for r in rows for name in r["pairs"]})
    pairs = {}
    for name in pair_names:
        values = [r["pairs"][name] for r in rows if r["pairs"].get(name) is not None]
        pairs[name] = {"count": len(values), "scopes": {}, "contact_speed": contact_speed_summary(values)}
        for scope in SCOPES:
            before = sum(v[scope]["original_collision"] for v in values)
            after = sum(v[scope]["edited_collision"] for v in values)
            count = len(values)
            pairs[name]["scopes"][scope] = {"original_collision_count": before, "edited_collision_count": after,
                "original_collision_rate": before / count if count else None,
                "edited_collision_rate": after / count if count else None,
                "collision_rate_reduction_pp": (before - after) / count * 100 if count else None,
                "transitions": dict(Counter(v[scope]["transition"] for v in values))}
    conditions = {}
    for name in sorted({name for r in rows for name in r["scores"]}):
        values = [r["scores"][name] for r in rows if r["scores"].get(name) is not None]
        conditions[name] = {"count": len(values), "cohort": "available predictions; may differ between conditions",
                            **{f"{scope}_collision_rate": sum(v[scope] for v in values) / len(values) if values else None for scope in SCOPES}}
    primary = [r for r in rows if r["collision_benefit"] is not None]
    n = len(primary)
    interactions = [r["interaction_difference_in_differences"] for r in rows
                    if r.get("interaction_difference_in_differences") is not None]
    return {"sample_count": len(rows), "coverage": coverage, "by_pair": pairs, "by_condition": conditions,
            "interaction_difference_in_differences": {
                "count": len(interactions),
                "definition": "(C_edited_image_edited_cot - C_original_image_edited_cot) - (C_edited_image_original_cot - C_original_image_original_cot)",
                "mean_pp": {s: 100 * sum(v[s] for v in interactions) / len(interactions) if interactions else None for s in SCOPES}},
            "contact_speed": contact_speed_summary([p for r in primary
                if (p := r["pairs"].get(r.get("primary_pair"))) is not None]),
            "paired_count": n, "original_collision_rate": sum(r["original_collision_all"] for r in primary) / n if n else None,
            "edited_collision_rate": sum(r["edited_collision_all"] for r in primary) / n if n else None,
            "collision_rate_reduction_pp": 100 * sum(r["collision_benefit"] for r in primary) / n if n else None}


def summarize(rows):
    groups = {}
    for field in ("actor_type", "placement_timestep", "speed_bin_kmh"):
        bins = defaultdict(list)
        for row in rows:
            if row["groups"].get(field) is not None:
                bins[str(row["groups"][field])].append(row)
        labels = sorted(bins, key=speed_bin_sort_key if field == "speed_bin_kmh" else None)
        groups[field] = {label: aggregate(bins[label]) for label in labels}
    from .reporting import summarize as raw_summary
    return {"raw_metrics": raw_summary(rows, "nc"), "version": 2,
            "metric": "PAI-AV annotated-scene at-fault collision (reactbench/scoring/at_fault.py)",
            "units": "rates [0,1]; collision-rate reductions in percentage points; contact speeds/reductions in m/s",
            "contact_speed_metric": AT_FAULT_CONTACT_SPEED,
            "overall": aggregate(rows), "groups": groups}


def process_result(root, config, *, tokens=None, output_root=None, variant_dir=None):
    """Score one run; write metrics under ``output_root`` (default: the run itself).

    ``variant_dir``: the benchmark variant, for predictions saved without the runners' bookkeeping.
    """
    from ..datasets.paiav import PAIAVCompactVariant
    root = Path(root).resolve()
    inventory = inventory_result(root, variant_dir=variant_dir)
    if inventory["mode"] == "original":
        return {"status": "inapplicable", "result_dir": str(root), "mode": "original"}
    tasks = inventory["tasks"]
    if tokens is not None:
        missing = set(tokens) - {t["token"] for t in tasks}
        if missing:
            raise ValueError(f"Requested tokens absent from result: {sorted(missing)[:5]}")
        tasks = [t for t in tasks if t["token"] in set(tokens)]
    contract = implementation_fingerprint(config)
    worlds = BackgroundWorlds(config)
    rows, loader = [], None
    try:
        if any(t["status"] == "ready" for t in tasks):
            try:
                loader = PAIAVCompactVariant(inventory["run_dir"], validate_archive_members=False)
            except Exception as exc:
                for task in tasks:
                    if task["status"] == "ready":
                        task.update(status="failed", reason=f"Source variant unavailable: {exc}")
        for i, task in enumerate(tasks, 1):
            rows.append(_row(task, loader, worlds, config, contract))
            if i % 50 == 0:
                print(f"{root.name}: {i}/{len(tasks)} samples", flush=True)
    finally:
        worlds.close()
        if loader is not None:
            loader.close()
    summary = summarize(rows)
    summary["motion_binning"] = motion_binning_policy()
    output = (Path(output_root).resolve() if output_root else root) / "metrics"
    write_json(output / f"{PREFIX}_per_sample.json", rows)
    write_json(output / f"{PREFIX}_summary.json", summary)
    from .at_fault import AT_FAULT_CONTRACT
    write_json(output / f"{PREFIX}_manifest.json", {"version": 2, "result_dir": str(root),
        "run_dir": inventory["run_dir"], "mode": inventory["mode"], "config": config,
        "contract_fingerprint": at_fault_fingerprint(config), "physical_contract_fingerprint": contract,
        "collision_contract": AT_FAULT_CONTRACT,
        "contact_speed_metric": AT_FAULT_CONTACT_SPEED,
        "motion_binning": motion_binning_policy(),
        "source_documents": inventory["sources"], "selection": [t["token"] for t in tasks],
        "partial_selection": tokens is not None, "per_sample_sha256": file_digest(output / f"{PREFIX}_per_sample.json"),
        "status": "complete" if all(r["status"] in ("scored", "inapplicable") for r in rows) else "incomplete"})
    (output / f"{PREFIX}_error.json").unlink(missing_ok=True)
    return {"result_dir": str(root), **summary["overall"]}


def postprocess_evaluation(root, mode="score"):
    """Called by the runners after inference; a scoring failure never hides saved predictions."""
    if mode == "off":
        return
    try:
        summary = process_result(root, load_config())
        print(f"ReactPAI scoring: {root}; coverage={summary.get('coverage')}", flush=True)
    except Exception as exc:
        # Earlier valid scores stay readable; the error goes next to them.
        write_json(Path(root) / "metrics" / f"{PREFIX}_error.json", {"status": "failed", "reason": str(exc)})
        print(f"ReactPAI scoring failed; saved predictions remain available: {exc}", flush=True)


def _score_one(job):
    run, config, output = job
    try:
        result = process_result(run, config, output_root=output)
        return str(run), result.get("coverage"), None
    except Exception as exc:
        return str(run), None, f"{type(exc).__name__}: {exc}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path, help="Run directories, or directories containing runs")
    parser.add_argument("--workers", type=int, help="Runs scored in parallel (default 4)")
    parser.add_argument("--output-root", type=Path,
                        help="Write metrics/ for each run under OUTPUT_ROOT/<run path relative to its root>")
    args = parser.parse_args(argv)
    config = load_config({"workers": args.workers})
    jobs = []
    for base in args.roots:
        base = base.resolve()
        for run in saved.discover_result_dirs([base]):
            output = args.output_root / run.relative_to(base) if args.output_root else None
            jobs.append((run, config, output))
    if config["workers"] == 1:
        results = [_score_one(job) for job in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            results = list(pool.map(_score_one, jobs))
    unresolved = 0
    for run, coverage, error in results:
        print(run, coverage if error is None else f"FAILED {error}", flush=True)
        unresolved += bool(error or any((coverage or {}).get(k) for k in ("failed", "partial", "invalid")))
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
