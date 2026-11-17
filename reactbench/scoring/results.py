"""Inventory of a saved run: which predictions, insertions and pairs to score (read only).

A predictions run is ``<model>/<reactnav|reactpai>/<run>/trajectories/<token>.npz`` next to
``<model>/<dataset>/original/trajectories/<token>.npz`` (see README "Evaluating your model"). The runners also write
``metrics/summary.json`` and ``metrics/per_sample.json``, which name the benchmark run and the original
predictions; without them, the benchmark run is passed in (``run_dir``) and the originals are the sibling
``original/``. CoT runs keep one directory per condition.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from ..io import digest, file_digest, finite_json, read_json


def _path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def discover_result_dirs(roots):
    """Find run roots (directories with an evaluation manifest or metrics), without double counting."""
    found = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("plots", "videos", "reasoning", "cots", "__pycache__", "pdms")
                       and not d.startswith(".")]
            here = Path(directory)
            if "evaluation_manifest.json" in files or ("metrics" in dirs and any(
                    (here / "metrics" / name).is_file() for name in ("summary.json", "per_sample.json", "per_input.json"))):
                found.add(here)
                dirs[:] = []
                continue
            if here.name == "metrics" and ("per_sample.json" in files or "per_input.json" in files):
                found.add(here.parent)
    # A condition member belongs to its enclosing run. Distinct sibling runs survive.
    return sorted(p for p in found if not any(parent in found for parent in p.parents))


def _source_path(path, **extra):
    path = Path(path)
    value = {"path": str(path), **extra}
    if path.is_file():
        value["sha256"] = file_digest(path)
    return value


def _archive_paths(root):
    return sorted(p for name in ("trajectories", "rollouts")
                  for p in (root / name).glob("*.npz"))


def _archive_for(root, stem):
    existing = [root / name / f"{stem}.npz" for name in ("trajectories", "rollouts")
                if (root / name / f"{stem}.npz").is_file()]
    if len(existing) > 1:
        raise ValueError(f"Ambiguous saved trajectory for {stem} under {root}")
    return existing[0] if existing else root / "trajectories" / f"{stem}.npz"


def _prediction_sources(path, dt):
    """Expand a rollout archive into named branches, retaining action identity."""
    source = _source_path(path, interval_length_s=dt)
    if not path.is_file():
        return {"": source}
    with np.load(path, allow_pickle=False) as archive:
        if "interval_length_s" in archive:
            stored = float(np.asarray(archive["interval_length_s"]).item())
            if dt is not None and not np.isclose(float(dt), stored, atol=1e-6):
                raise ValueError("Archive interval disagrees with result summary")
            dt = stored
        if dt is None or not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("Missing or invalid trajectory interval")
        source["interval_length_s"] = float(dt)
        if "trajectory" in archive:
            poses = archive["trajectory"]
            if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
                raise ValueError("Invalid x/y/heading trajectory array")
            if len(poses) * dt < 4.0 - 1e-6:
                raise ValueError("Trajectory is shorter than four seconds")
            return {"": source}
        if "trajectories" not in archive:
            return {"": {**source, "error": "Archive has no full trajectory"}}
        trajectories = archive["trajectories"]
        if trajectories.ndim != 3:
            raise ValueError(f"Invalid branch array in {path}")
        if trajectories.shape[2] != 3 or not np.isfinite(trajectories).all():
            raise ValueError("Invalid rollout x/y/heading array")
        if trajectories.shape[1] * dt < 4.0 - 1e-6:
            raise ValueError("Rollout trajectories are shorter than four seconds")
        count = len(trajectories)
        ranks = archive["candidate_ranks"] if "candidate_ranks" in archive else np.arange(1, count + 1)
        ids = archive["candidate_token_ids"] if "candidate_token_ids" in archive else [None] * count
        if len(ranks) != count or len(ids) != count or len(set(ranks.tolist())) != count:
            raise ValueError(f"Invalid branch identities in {path}")
        return {f"rank_{int(ranks[i])}": {**source, "branch_index": i,
                 "candidate_rank": int(ranks[i]),
                 "candidate_token_id": None if ids[i] is None else int(ids[i])}
                for i in range(count)}


def _put_predictions(target, condition, path, dt):
    try:
        sources = _prediction_sources(path, dt)
    except Exception as exc:
        sources = {"": {"path": str(path), "interval_length_s": dt,
                        "error": f"{type(exc).__name__}: {exc}"}}
    for branch, source in sources.items():
        target[f"{condition}/{branch}" if branch else condition] = source


DATASET_FOLDERS = {"reactnav": "navsim", "reactpai": "pai-av"}


def inventory_result(result_dir, *, run_dir=None, dataset=None):
    """Read only: resolve every saved/expected sample into a scoring request.

    ``run_dir``: the benchmark run (default: the one the run's bookkeeping names). ``dataset``: "navsim" or
    "pai-av" (default: from the parent folder, ``reactnav`` or ``reactpai``).
    """
    root = Path(result_dir).resolve()
    summary = read_json(root / "metrics/summary.json", {})
    manifest = read_json(root / "evaluation_manifest.json", {})
    records = read_json(root / "metrics/per_sample.json", [])
    if not isinstance(records, list) or not isinstance(summary, dict) or not isinstance(manifest, dict):
        raise ValueError(f"Invalid result documents in {root}")
    identity = summary.get("identity", manifest.get("identity", {}))
    signature = identity.get("signature", manifest.get("signature", {}))
    mode = summary.get("mode", manifest.get("mode", "paired"))
    if mode != "fixed_cots" and identity.get("fixed_config"):
        mode = "fixed_cots"
    dataset = dataset or DATASET_FOLDERS.get(root.parent.name)
    if dataset is None:
        # Runs outside the outputs layout (the research repo's): guess from the bookkeeping.
        dataset_text = " ".join(str(v) for v in (summary.get("dataset", ""), signature.get("evaluator", ""),
                       signature.get("source", {}).get("dataset", ""), identity.get("backend", "")))
        dataset = "pai-av" if ("pai" in dataset_text.lower() or "alpamayo" in root.parts) else "navsim"
    run_value = summary.get("run_dir", identity.get("run_dir", manifest.get("source_run_dir")))
    run_dir = Path(run_dir).resolve() if run_dir else (_path(run_value, root) if run_value else None)
    dt = summary.get("interval_length_s", signature.get("interval_length_s"))
    sources = {str(p): file_digest(p) for p in (root / "metrics/summary.json",
               root / "metrics/per_sample.json", root / "evaluation_manifest.json") if p.is_file()}
    by_token = {}
    for row in records:
        token = str(row.get("token", row.get("output_stem", "")))
        if not token or token in by_token:
            raise ValueError(f"Empty or duplicate sample token {token!r} in {root}")
        by_token[token] = row
    expected = set(by_token)
    expected.update(str(t) for t in manifest.get("selected_tokens", []))
    for path in _archive_paths(root):
        expected.add(path.stem)
    for row in read_json(root / "failed_samples.json", []):
        if row.get("token"):
            expected.add(str(row["token"]))
    for member in ("original", "edited"):
        for path in _archive_paths(root / member):
            expected.add(path.stem)
    if mode == "fixed_cots":
        for condition in identity.get("conditions", {}):
            for path in _archive_paths(root / condition):
                expected.add(path.stem)
    if mode == "cot_swap":
        for name in ("original_image_edited_cot", "edited_image_original_cot"):
            for path in _archive_paths(root / name):
                expected.add(path.stem)
    if not expected:
        return {"result_dir": str(root), "dataset": dataset, "mode": mode,
                "sources": sources, "tasks": [], "status": "inapplicable",
                "reason": "No saved or expected trajectory samples found"}
    if mode == "original":
        # Original-scene predictions are the inputs of the paired runs; no table scores them alone.
        return {"result_dir": str(root), "dataset": dataset, "mode": mode, "sources": sources, "tasks": [],
                "status": "inapplicable", "reason": "Original-scene predictions are scored in their pairs"}
    original_value = summary.get("original_result_dir", summary.get("natural_original_result_dir",
                                    identity.get("baseline_dirs", {}).get("natural_original")))
    original_root = (_path(original_value, root) if original_value else
                     root / "original" if (root / "original").is_dir() else root.parent / "original")
    tasks = []
    for token in sorted(expected):
        row = by_token.get(token, {})
        task = {"token": token, "dataset": dataset, "world": "augmented",
                "predictions": {}, "pairs": {}, "primary_condition": None,
                "groups": {k: row[k] for k in ("speed_bin_kmh", "actor_type", "speed_mps",
                                               "longitudinal_acceleration_mps2") if k in row}}
        try:
            stem = str(row.get("output_stem", token))
            if Path(stem).name != stem or Path(token).name != token:
                raise ValueError("Sample identifier is not a filename")
            if row.get("ego_origin", "rear_axle") != "rear_axle":
                raise ValueError("Raw scoring requires trajectories expressed at the ego rear axle")
            if dataset != "navsim":
                # Completed by the PAI-AV scorer, which loads its own background world.
                task.update(status="unsupported", reason="PAI-AV needs compatible maps/routes and background tracks")
            metadata_value = row.get("placement_metadata_path", row.get("metadata_path",
                                 row.get("provenance", {}).get("metadata_path")))
            metadata_path = (_path(metadata_value, root) if metadata_value else
                             run_dir / "metadata" / f"{token}.json" if run_dir else None)
            motion = row.get("actor_motion")
            if metadata_path and metadata_path.is_file() and dataset == "navsim":
                from .actor_motion import inserted_actor_motion
                metadata = read_json(metadata_path)
                parsed = inserted_actor_motion(metadata)
                if motion is not None:
                    # Verify physically relevant geometry/motion, ignoring extra bookkeeping keys.
                    for time_s in (0.0, 5.0):
                        a, b = (actor_box_state(m, [0, 0, 0], time_s) for m in (motion, parsed))
                        for key in ("pose", "length", "width", "velocity"):
                            if not np.allclose(a[key], b[key], atol=1e-6):
                                raise ValueError("Saved actor motion disagrees with compositor metadata")
                motion = parsed
                task["metadata_source"] = _source_path(metadata_path)
            if motion is None and dataset == "navsim":
                raise ValueError("Missing insertion metadata/actor motion")
            task["actor_motion"] = motion
            if motion:
                if dataset == "navsim":
                    actor_box_state(motion, [0, 0, 0], 0)
                    if motion.get("actor_type", "pedestrian") == "pedestrian" and not motion.get("geometry"):
                        raise ValueError("Missing measured pedestrian geometry; legacy fixed footprints cannot be scored")
                task["groups"]["actor_type"] = motion.get("actor_type", "pedestrian")
                task["groups"]["placement_timestep"] = motion.get("future_timestep")
            predictions = task["predictions"]
            if mode == "fixed_cots":
                names = set(identity.get("conditions", {})) | set(row.get("conditions", {}))
                names.update(("natural_original", "natural_edited"))
                for name in sorted(names):
                    base = identity.get("baseline_dirs", {}).get(name)
                    path = _archive_for(_path(base, root) if base else root / name, stem)
                    if path.is_file() or name not in row.get("trajectories", {}):
                        _put_predictions(predictions, name, path, dt)
                    else:
                        predictions[name] = {"poses": row["trajectories"][name], "interval_length_s": dt,
                                             "source_document": sources.get(str(root / "metrics/per_sample.json"))}
                task["primary_condition"] = "natural_edited"
                task["pairs"] = {name: ["natural_original", name] for name in names if name != "natural_original"}
            elif mode == "cot_swap" or "original_image_original_cot" in row.get("conditions", {}):
                conditions = row.get("conditions", {})
                names = set(conditions) | {"original_image_original_cot", "edited_image_edited_cot",
                                          "original_image_edited_cot", "edited_image_original_cot"}
                for name in sorted(names):
                    value = conditions.get(name, {}).get("trajectory_path")
                    base = original_root if name == "original_image_original_cot" else root / name
                    if name == "edited_image_edited_cot" and summary.get("natural_edited_result_dir"):
                        base = _path(summary["natural_edited_result_dir"], root)
                    _put_predictions(predictions, name, _path(value, root) if value else _archive_for(base, stem), dt)
                task["primary_condition"] = "edited_image_edited_cot"
                task["pairs"] = {name: ["original_image_original_cot", name] for name in names
                                 if name != "original_image_original_cot"}
            else:
                _put_predictions(predictions, "original", _archive_for(original_root, stem), dt)
                edited_root = root / "edited" if (root / "edited").is_dir() else root
                _put_predictions(predictions, "edited", _archive_for(edited_root, stem), dt)
                if "edited" in predictions:
                    task["primary_condition"] = "edited"
                    task["pairs"] = {"edited": ["original", "edited"]}
                else:
                    # Branch ranks may correspond to different first actions. Pair only matching tokens.
                    before = {s.get("candidate_token_id"): n for n, s in predictions.items()
                              if n.startswith("original/") and s.get("candidate_token_id") is not None}
                    task["pairs"] = {name: [before[source["candidate_token_id"]], name]
                                     for name, source in predictions.items() if name.startswith("edited/")
                                     and source.get("candidate_token_id") in before}
            for name, source in predictions.items():
                if "path" in source and not Path(source["path"]).is_file():
                    source["error"] = "Missing saved trajectory"
            if dataset == "navsim":
                from .motion_bins import speed_bin_label
                groups = task["groups"]
                speed = groups.get("speed_mps")
                if speed is None:
                    speed = next((v.get("initial_speed_mps") for v in row.get("conditions", {}).values()
                                  if v.get("initial_speed_mps") is not None), None)
                # Binning must not invalidate an otherwise scoreable trajectory.
                try:
                    if speed is not None and "speed_bin_kmh" not in groups:
                        groups["speed_bin_kmh"] = speed_bin_label(float(speed)*3.6)
                except ValueError as exc:
                    task["grouping_warning"] = str(exc)
            task.setdefault("status", "ready")
        except Exception as exc:
            task.update(status="invalid", reason=f"{type(exc).__name__}: {exc}")
        task = finite_json(task)
        task["input_fingerprint"] = digest(task)
        tasks.append(task)
    return {"result_dir": str(root), "dataset": dataset, "mode": mode,
            "sources": sources, "tasks": tasks, "status": "inventoried"}


def load_prediction(source):
    """Validate source identity and require a full four seconds; no endpoint padding."""
    if source.get("error"):
        raise ValueError(source["error"])
    dt = source.get("interval_length_s")
    if "path" in source:
        path = Path(source["path"])
        if source.get("sha256") != file_digest(path):
            raise ValueError(f"Trajectory changed since inventory: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if "interval_length_s" in archive:
                stored = float(np.asarray(archive["interval_length_s"]).item())
                if dt is not None and not np.isclose(float(dt), stored, atol=1e-6):
                    raise ValueError(f"Trajectory interval disagrees with result provenance: {path}")
                dt = stored
            if "branch_index" in source:
                poses = np.asarray(archive["trajectories"][source["branch_index"]], dtype=float)
            else:
                poses = np.asarray(archive["trajectory"], dtype=float)
    else:
        poses = np.asarray(source["poses"], dtype=float)
    dt = float(dt) if dt is not None else math.nan
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Missing or invalid trajectory sampling interval")
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise ValueError("Trajectory must contain finite rear-axle x/y/heading poses")
    if len(poses) * dt < 4.0 - 1e-6:
        raise ValueError("Trajectory is shorter than the four-second evaluation horizon")
    return poses, dt


def actor_box_state(motion, ego_pose, time_s):
    """Return world box pose/dimensions/velocity, correcting asymmetric asset roots."""
    from .actor_motion import validated_motion
    motion = validated_motion(motion)
    if motion.get("actor_type") not in ("pedestrian", "car", "animal"):
        raise ValueError("Insertion scoring supports inserted pedestrians, cars and animals")
    pose = np.asarray(ego_pose, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(time_s):
        raise ValueError("Non-finite ego pose or actor time")
    heading = motion["heading_rad"]
    rotation = lambda a: np.array([[math.cos(a), -math.sin(a)],
                                   [math.sin(a), math.cos(a)]])
    offset = np.array([(motion["front_length_m"] - motion["rear_length_m"]) / 2,
                       (motion["left_width_m"] - motion["right_width_m"]) / 2])
    local = (np.asarray(motion["initial_center_xy_m"])
             + time_s * np.asarray(motion["velocity_xy_mps"]) + rotation(heading) @ offset)
    center = pose[:2] + rotation(pose[2]) @ local
    yaw = math.atan2(math.sin(pose[2] + heading), math.cos(pose[2] + heading))
    velocity = rotation(pose[2]) @ np.asarray(motion["velocity_xy_mps"])
    # Height has no effect on NAVSIM's BEV metrics. Use measured height if available.
    bounds = motion.get("geometry", {}).get("local_bounds_m", {})
    z = bounds.get("z")
    height = float(z[1] - z[0]) if z is not None else float(motion.get("height_m", 1.0))
    if not math.isfinite(height) or height <= 0:
        raise ValueError("Actor height must be finite and positive")
    return {"pose": [float(center[0]), float(center[1]), yaw],
            "length": motion["length_m"], "width": motion["width_m"], "height": height,
            "velocity": velocity.tolist(), "actor_type": motion["actor_type"]}
