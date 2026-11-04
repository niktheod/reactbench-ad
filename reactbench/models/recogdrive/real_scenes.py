"""Real-scenes test set (paper section 5): recorded stopped vehicles ahead and an injected ego speed.

The cohort (``python -m reactbench.selection.real_scenes``) keeps slow-ego navtest scenes with a recorded
stopped vehicle wholly in the ego lane 5-15 m ahead. Per headway h the ego status is replaced by a constant
speed v = gap / h along the route centerline (zero acceleration, past poses redrawn), so a non-reacting plan
reaches the vehicle after h. Predictions are scored in the recorded world with the target vehicle frozen at its
t = 0 pose (some "stopped" vehicles start moving within 4 s): P = NC x DAC, NC_ins (no at-fault contact with the
target, which takes the role of the inserted object) and CS (ego speed at the first at-fault contact with the
target). These scenes have no unedited counterpart, so the paper reports levels, averaged over the headways.

    python -m reactbench.models.recogdrive.real_scenes build --cohort outputs/real_scenes/cohort.json
    python -m reactbench.models.recogdrive.real_scenes infer --name il --config il      # GPU; then scores
    python -m reactbench.models.recogdrive.real_scenes infer --name il_rt --config il_rt
    python -m reactbench.models.recogdrive.real_scenes summarize

``build`` reads the navtest metric cache (``navsim_metric_cache``, NAVSIM v2) and writes the scene file.
Outputs live under ``<outputs>/real_scenes/``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

from ... import paths
from ..seeds import paired_seed

ROOT = paths.outputs("real_scenes")
PROTOCOL = "recogdrive-stopped-vehicle-probe-v2"
INFERENCE_PROTOCOL = "recogdrive-speed-injection-probe-v1"


RELEASED_SCENES = paths.REPO_ROOT / "data" / "real_scenes" / "navtest_v2.json"


def default_scenes():
    """The released scene file (308 scenes), unless one was built with ``build``."""
    built = ROOT / "scenes" / "navtest_v2.json"
    return built if built.is_file() else RELEASED_SCENES


def run_dir(scenes, name):
    return ROOT / "runs" / Path(scenes).stem / name


# Placement checks of the built scene file (``build``); ``hit`` and ``robust`` are recorded in it.
CRITERIA = {
    "max_ego_offset_m": 2., "max_ego_alignment_deg": 15.,
    "hit": "non-reacting plan: constant v along the route centerline, ego footprint vs the recorded target "
           "box at 0.1 s steps over 4 s",
    "robust": {"speed_factors": [.8, 1., 1.2], "lateral_offsets_m": [-.5, 0., .5]},
}

HISTORY_TIMES_S = (-1.5, -1.0, -.5, 0.)

STEPS = 40  # future boxes at t = 0.1 .. 4.0 s

def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def pose_on(line, s, step=.5):
    """(x, y, heading) at arc length ``s``; beyond either end the line continues straight."""
    length = line.length
    inside = min(max(s, 0.), length)
    a, b = line.interpolate(max(inside - step, 0.)), line.interpolate(min(inside + step, length))
    heading = math.atan2(b.y - a.y, b.x - a.x)
    point = line.interpolate(inside)
    extra = s - inside
    return point.x + extra * math.cos(heading), point.y + extra * math.sin(heading), heading

def injected_history(line, s0, speed_mps):
    """Four past poses (t-1.5 .. t0) along ``line`` ending at arc length ``s0``, in the current pose's frame."""
    if not math.isfinite(speed_mps) or speed_mps <= 0:
        raise ValueError("The injected speed must be positive")
    x0, y0, h0 = pose_on(line, s0)
    cos, sin = math.cos(h0), math.sin(h0)
    poses = []
    for t in HISTORY_TIMES_S:
        x, y, h = pose_on(line, s0 + speed_mps * t)
        dx, dy = x - x0, y - y0
        poses.append([dx * cos + dy * sin, -dx * sin + dy * cos, wrap(h - h0)])
    poses[-1] = [0., 0., 0.]
    return poses

def headway_record(line, s0, gap_m, headway_s, hit_s, robust=None):
    speed = gap_m / headway_s
    return {"speed_mps": speed, "velocity": [[speed, 0.]] * 4, "acceleration": [[0., 0.]] * 4,
            "past_trajectory": injected_history(line, s0, speed), "non_reacting_hit_s": hit_s,
            "robust_hit": robust}

class SceneView:
    """Ego, route centerline and the objects now and at t = 0.1 .. 4.0 s (token -> footprint)."""

    def __init__(self, ego_xyh, ego_speed, front, rear, half, line, now, futures):
        self.x0, self.y0, self.h0 = ego_xyh
        self.speed, self.front, self.rear, self.half, self.line = ego_speed, front, rear, half, line
        self.now = now          # [(token, kind, footprint, speed_mps, (x, y) centre)]
        self.futures = futures  # [{token: footprint}] for t = 0.1 .. 4.0 s

def _speed(obj):
    velocity = getattr(obj, "velocity", None)
    return 0. if velocity is None else math.hypot(velocity.x, velocity.y)

def _ego(cache):
    ego = cache.ego_state
    params = ego.car_footprint.vehicle_parameters
    return ((ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading), float(ego.dynamic_car_state.rear_axle_velocity_2d.x),
            params.front_length, params.rear_length, params.half_width)

def view_v2(cache):
    """NAVSIM v2 caches: tracked objects now and in 40 future frames."""
    def objects(detections):
        tracks = getattr(detections, "tracked_objects", detections)
        return list(getattr(tracks, "tracked_objects", tracks))
    now = [(str(o.track_token), o.tracked_object_type.name, o.box.geometry, _speed(o), (o.center.x, o.center.y))
           for o in objects(cache.current_tracked_objects[0])]
    futures = [{str(o.track_token): o.box.geometry for o in objects(frame)}
               for frame in cache.future_tracked_objects[:STEPS]]
    return SceneView(*_ego(cache), cache.centerline.linestring, now, futures)

def _box(x, y, heading, front, rear, half):
    from shapely.geometry import Polygon
    c, s = math.cos(heading), math.sin(heading)
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in
                    ((front, half), (front, -half), (-rear, -half), (-rear, half))])

def hit_time(view, s0, token, speed_mps, lateral_m=0.):
    """First time a constant-speed plan along the centerline (shifted laterally) overlaps the token's box."""
    for index, frame in enumerate(view.futures):
        footprint = frame.get(token)
        if footprint is None:
            continue
        t = (index + 1) * .1
        x, y, h = pose_on(view.line, s0 + speed_mps * t)
        x, y = x - lateral_m * math.sin(h), y + lateral_m * math.cos(h)
        if _box(x, y, h, view.front, view.rear, view.half).intersects(footprint):
            return round(t, 1)
    return None

def robust_hit(view, s0, token, speed_mps):
    robust = CRITERIA["robust"]
    return all(hit_time(view, s0, token, f * speed_mps, d) is not None
               for f in robust["speed_factors"] for d in robust["lateral_offsets_m"])


INTERVAL_S = .5

EMERGENCY_DECEL_MPS2 = 3.

def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()

def load_scenes(paths):
    """Scenes of every file (unique tokens) and the shared input conditions: real, then each headway."""
    scenes, sources, headways = [], [], None
    for path in paths:
        value = json.loads(Path(path).read_text())
        sources.append({"path": str(Path(path).resolve()), "sha256": file_sha256(path)})
        for scene in value["scenes"]:
            keys = sorted(scene["headways"], key=float)
            if headways is None:
                headways = keys
            elif keys != headways:
                raise ValueError("Every scene needs the same headways")
            scenes.append({"split": "test", **scene})
    tokens = [scene["token"] for scene in scenes]
    if not scenes or len(tokens) != len(set(tokens)):
        raise ValueError("Scene files must hold unique tokens")
    return scenes, sources, ["real", *headways]

def scenes_by_split(scenes):
    split = {}
    for scene in scenes:
        split.setdefault(scene["split"], []).append(scene)
    return split

def injected_annotation(annotation, headway):
    """The real annotation record with the injected velocity, acceleration and past trajectory."""
    record = dict(annotation)
    for key, width in (("velocity", 2), ("acceleration", 2), ("past_trajectory", 3)):
        value = np.asarray(headway[key], dtype=float)
        if value.shape != (4, width) or not np.isfinite(value).all():
            raise ValueError(f"Injected {key} must be four finite {width}-vectors")
        record[key] = value.tolist()
    return record

def condition_annotation(annotation, scene, condition):
    if condition == "real":
        return annotation
    return injected_annotation(annotation, scene["headways"][condition])

def initial_speed(annotation):
    return float(np.linalg.norm(np.asarray(annotation["velocity"][-1], dtype=float)[:2]))

def planner_state(model_shapes, checkpoint_state, prefix="action_head."):
    """Checkpoint tensors for every planner weight; a missing or reshaped weight is an error.

    The upstream agent's initialize() silently skips mismatched names/shapes, so an incompatible
    checkpoint would otherwise run with part of the IL planner.
    """
    stripped = {(k[len("agent."):] if k.startswith("agent.") else k): v for k, v in checkpoint_state.items()}
    expected = {k: tuple(shape) for k, shape in model_shapes.items() if k.startswith(prefix)}
    missing = sorted(k for k in expected if k not in stripped)
    reshaped = sorted(k for k in expected if k in stripped and tuple(stripped[k].shape) != expected[k])
    if not expected or missing or reshaped:
        raise ValueError(f"Checkpoint lacks planner weights: {len(missing)} missing (e.g. {missing[:3]}), "
                         f"{len(reshaped)} reshaped (e.g. {reshaped[:3]})")
    return {k: stripped[k] for k in expected}

def motion(poses, start_speed_mps):
    """Mean speed per 0.5 s interval and the decelerations between them (from the given start speed)."""
    poses = np.asarray(poses, dtype=float)
    if poses.shape != (8, 3) or not np.isfinite(poses).all():
        raise ValueError("Expected eight finite native poses")
    points = np.vstack(([0., 0.], poses[:, :2]))
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / INTERVAL_S
    steps = np.concatenate(([INTERVAL_S / 2], np.full(len(speeds) - 1, INTERVAL_S)))  # interval centres
    decel = -np.diff(np.concatenate(([start_speed_mps], speeds))) / steps
    peak = float(max(decel.max(), 0.))
    return {"start_speed_mps": float(start_speed_mps), "speed_1s_mps": float(speeds[1]),
            "speed_2s_mps": float(speeds[3]), "speed_4s_mps": float(speeds[-1]),
            "peak_deceleration_mps2": peak, "emergency_braking": peak >= EMERGENCY_DECEL_MPS2,
            "displacement_4s_m": float(np.linalg.norm(points[-1]))}

def target_contacts(score, track_token):
    at_fault = [c for c in score.get("contacts", []) if c.get("actor_id") == track_token]
    exempt = [c for c in score.get("exempt_contacts", []) if c.get("actor_id") == track_token]
    first = min(at_fault + exempt, key=lambda c: c["time_s"], default=None)
    return {"target_contact": bool(at_fault or exempt), "target_contact_at_fault": bool(at_fault),
            "target_contact_type": None if first is None else first.get("type"),
            "target_contact_time_s": None if first is None else first["time_s"]}

def parse_checkpoints(values):
    checkpoints = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path or name in checkpoints:
            raise ValueError(f"Checkpoints are NAME=PATH with unique names: {value}")
        checkpoints[name] = Path(path).expanduser().resolve()
    return checkpoints

def split_config(config, split):
    """The raw-scoring configuration with the scene's log split (navsim_logs/<split>)."""
    return {**config, "log_path": str(Path(config["log_path"]).parent / split)}


HEADWAYS = ("0.5", "1.0", "1.5", "2.0")

SPEED_BINS_KMH = ((5, 10), (10, 15), (15, 20), (20, 25), (25, 35), (35, 50), (50, 100))

def headway_key(value):
    """Canonical headway key ("1" -> "1.0")."""
    return str(float(value))

def conditions_of(rows):
    """"real" then the headways, in increasing order."""
    headways = sorted({r["condition"] for r in rows} - {"real"}, key=float)
    return (["real"] if any(r["condition"] == "real" for r in rows) else []) + headways

def front_gap(line, footprint, s0, front):
    """Route-centerline arc length from the front bumper (at s0 + front) to the footprint's nearest point."""
    from shapely.geometry import Point
    coords = [xy for part in getattr(footprint, "geoms", [footprint]) for xy in part.exterior.coords]
    return min(line.project(Point(xy)) for xy in coords) - (s0 + front)

def scene_record(view, track, headways=HEADWAYS):
    """Injected inputs for one scene view: route frame, target gap and per-headway records (or an error string)."""
    from shapely.geometry import LineString, Point
    line = view.line
    s0 = line.project(Point(view.x0, view.y0))
    offset = line.distance(Point(view.x0, view.y0))
    alignment = math.degrees(abs(wrap(view.h0 - pose_on(line, s0)[2])))
    history = "route_centerline"
    if offset > CRITERIA["max_ego_offset_m"] or alignment > CRITERIA["max_ego_alignment_deg"]:
        c, s = math.cos(view.h0), math.sin(view.h0)
        line = LineString([(view.x0 - 100 * c, view.y0 - 100 * s), (view.x0 + 300 * c, view.y0 + 300 * s)])
        s0, history = 100., "ego_heading"
        view = SceneView((view.x0, view.y0, view.h0), view.speed, view.front, view.rear, view.half,
                              line, view.now, view.futures)
    now = next((o for o in view.now if o[0] == track), None)
    if now is None:
        return f"target {track} not in the metric cache"
    _, kind, footprint, speed, centre = now
    gap = front_gap(line, footprint, s0, view.front)
    if not gap > 0:
        return f"non-positive gap {gap:.2f} m"
    records = {}
    for h in map(headway_key, headways):
        v = gap / float(h)
        hit = hit_time(view, s0, track, v)
        robust = hit is not None and robust_hit(view, s0, track, v)
        records[h] = {**headway_record(line, s0, gap, float(h), hit, robust), "gap_m": gap}
    return {"history_frame": history, "ego_offset_m": offset, "ego_alignment_deg": alignment,
            "target": {"kind": kind, "track_token": track, "gap_m": gap, "lateral_m": line.distance(Point(*centre)),
                       "speed_mps": speed}, "headways": records}

def freeze_target(world, track):
    """A copy of the world whose target track holds its t = 0 pose and size over the whole horizon."""
    frozen = copy.deepcopy(world)
    segments = [t for t in frozen["tracks"] if t["actor_id"] == track]
    if not segments:
        raise ValueError(f"Track {track} is not in the world")
    first = min(segments, key=lambda t: t["times_s"][0])
    if first["times_s"][0] != 0:
        raise ValueError(f"Track {track} is not observed at t = 0")
    steps = int(round(world["horizon_s"] / .5)) + 1
    held = {**first, "times_s": [i * .5 for i in range(steps)], "poses": [first["poses"][0]] * steps,
            "sizes": [first["sizes"][0]] * steps}
    frozen["tracks"] = [t for t in frozen["tracks"] if t["actor_id"] != track] + [held]
    return frozen

def track_motion(world, track):
    """Largest displacement (m) of the track's centre from its first pose within the world horizon."""
    poses = [np.asarray(p[:2], dtype=float) for t in world["tracks"] if t["actor_id"] == track for p in t["poses"]]
    return max((float(np.linalg.norm(p - poses[0])) for p in poses), default=None)

def contact_fields(result, track, poses, prefix, interval_length_s=INTERVAL_S):
    """P, NC_ins and CS of one V2 result, with the target as the inserted object."""
    from ...scoring.collisions import first_contact_speed
    on_target = [c for c in result.get("contacts", []) if c.get("actor_id") == track]  # at-fault contacts only
    contacts = target_contacts(result, track)
    first = first_contact_speed(poses, on_target, interval_length_s)
    speed = None if first is None or first["ego_speed_mps"] is None else 3.6 * first["ego_speed_mps"]
    return {f"{prefix}nc": result["nc"], f"{prefix}dac": result["dac"], f"{prefix}p": result["nc_dac"],
            f"{prefix}nc_ins": int(not on_target), f"{prefix}cs_kmh": speed,
            f"{prefix}cs_time_s": None if first is None else first["time_s"],
            f"{prefix}target_contact": contacts["target_contact"],
            f"{prefix}other_contact": any(c.get("actor_id") != track for c in result.get("contacts", []))}

def rate(rows, key, value=1):
    return 100. * sum(r[key] == value for r in rows) / len(rows) if rows else None

def mean_speed(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.mean(values)) if values else None), len(values)

def summarize_records(records):
    """Per model and condition: P, NC_ins and CS, recorded and frozen, plus braking."""
    out = {}
    for model in sorted({r["model"] for r in records}):
        rows_m = [r for r in records if r["model"] == model]
        for condition in conditions_of(rows_m):
            rows = [r for r in rows_m if r["condition"] == condition]
            ratios = [r["speed_2s_mps"] / r["start_speed_mps"] for r in rows if r["start_speed_mps"] > .5]
            entry = {"n": len(rows), "emergency_braking": rate(rows, "emergency_braking", True),
                     "speed_2s_ratio_median": float(np.median(ratios)) if ratios else None}
            for prefix, tag in (("", ""), ("frozen_", "_frozen")):
                cs, n_cs = mean_speed(rows, f"{prefix}cs_kmh")
                entry.update({f"p{tag}": rate(rows, f"{prefix}p"), f"nc_ins{tag}": rate(rows, f"{prefix}nc_ins"),
                              f"cs_kmh{tag}": cs, f"cs_n{tag}": n_cs})
            out.setdefault(model, {})[condition] = entry
    return out

def speed_bins(records, model, condition):
    rows = [r for r in records if r["model"] == model and r["condition"] == condition]
    out = {}
    for lo, hi in SPEED_BINS_KMH:
        sel = [r for r in rows if lo <= 3.6 * r["injected_speed_mps"] < hi]
        out[f"{lo}-{hi}"] = {"n": len(sel), "p_frozen": rate(sel, "frozen_p"), "nc_ins_frozen": rate(sel, "frozen_nc_ins")}
    return out


# --- build ------------------------------------------------------------------------------------------

def extract_lanes(scenes, work_root):
    """Lane sidecars for the at-fault lateral rule, per scene."""
    from ...scoring.navsim_lanes import extract_one
    return {scene["token"]: extract_one((scene["token"], Path(scene["metric_cache"]), str(work_root)))[1]
            for scene in scenes}


def build(args):
    import lzma
    import pickle
    cohort = json.loads(Path(args.cohort).read_text())
    headways = [headway_key(h) for h in args.headways]
    output = Path(args.output or ROOT / "scenes" / "navtest_v2.json")
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output")
    metadata = Path(args.metric_cache or paths.get("navsim_metric_cache")) / "metadata" / "metric_cache_metadata_node_0.csv"
    with metadata.open(newline="") as handle:
        caches = {Path(row["file_name"]).parent.name: row["file_name"] for row in csv.DictReader(handle)}
    scenes, problems = [], []
    for sample in cohort["samples"]:
        token, target = sample["token"], sample["target"]
        with lzma.open(caches[token], "rb") as handle:
            cache = pickle.load(handle)
        view = view_v2(cache)
        record = scene_record(view, target["track_token"], headways)
        if isinstance(record, str):
            problems.append(f"{token}: {record}")
            continue
        record["target"].update(gap_lanepath_m=target["gap_lanepath_m"], strip_fraction=target["strip_fraction"],
                                motion_4s_m=target["motion_4s_m"])
        scenes.append({"token": token, "split": "test", "source": "navtest", "log": sample["log"],
                       "metric_cache": str(caches[token]), "group": "stopped_vehicle",
                       "ego_speed_mps": view.speed,
                       "ego_acceleration_mps2": float(cache.ego_state.dynamic_car_state.rear_axle_acceleration_2d.x),
                       **record})
    if problems:
        raise RuntimeError(f"{len(problems)} scenes failed: {problems[:5]}")
    from ...scoring.navsim import load_config
    lanes = extract_lanes(scenes, Path(args.work_root or load_config()["work_root"]))
    value = {"version": 2, "probe": PROTOCOL, "split": "navtest", "groups": ["stopped_vehicle"],
             "cohort": {"path": str(Path(args.cohort).resolve()), "sha256": file_sha256(args.cohort),
                        "criteria": cohort["criteria"], "funnel": cohort["funnel"]},
             "criteria": {"headways_s": [float(h) for h in headways],
                          "injection": "v = gap / h, gap = route-centerline arc length from the front bumper to the "
                                       "target's current footprint; velocity [v, 0] x 4, acceleration 0, four past "
                                       "poses v x 0.5 s apart along the route centerline, or straight back along the "
                                       "ego heading when the ego is > 2 m or > 15 deg off it (history_frame)",
                          "hit": CRITERIA["hit"], "robust": CRITERIA["robust"]},
             "scenes_checked": len(cohort["samples"]), "lanes": lanes, "scenes": scenes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=1) + "\n")
    print(json.dumps({"output": str(output), "scenes": len(scenes), "headways": headways}, indent=1))


# --- infer ------------------------------------------------------------------------------------------

def build_annotations(scenes):
    """Real annotation records exactly as reactbench.datasets.navsim_annotations writes them."""
    from navsim.common.dataclasses import SceneFilter
    from ...datasets import navsim_annotations
    navsim_annotations.configure_navsim_environment()
    annotations = {}
    for split, members in scenes_by_split(scenes).items():
        tokens = [scene["token"] for scene in members]
        scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                   max_scenes=None, log_names=sorted({scene["log"] for scene in members}), tokens=tokens)
        loader = navsim_annotations.build_loader(split, scene_filter)
        missing = sorted(set(tokens) - set(loader.tokens))
        if missing:
            raise ValueError(f"The {split} scene loader lacks {len(missing)} tokens, e.g. {missing[:3]}")
        annotations.update({token: navsim_annotations.build_sample(loader, token) for token in tokens})
    return annotations


def infer(args):
    """Run every checkpoint on the real and injected inputs of every scene; then score the run."""
    import torch
    from hydra.utils import instantiate
    from . import eval as evaluation
    from ..common import trajectory_poses
    output = run_dir(args.scenes, args.name)
    if (output / "predictions.json").exists():
        raise FileExistsError(f"{output} already holds predictions; choose a new --name")
    config_path, cfg = evaluation.load_config(args.config)
    checkpoints = parse_checkpoints(args.checkpoint or [f"{args.name}={cfg['checkpoint_path']}"])
    scenes, sources, conditions = load_scenes([args.scenes])
    seed_base = int(cfg.get("diffusion_seed_base", 42))
    annotations = build_annotations(scenes)
    agent = instantiate(evaluation.agent_config(cfg))
    agent.initialize()
    shapes = {f"action_head.{k}": tuple(v.shape) for k, v in agent.action_head.state_dict().items()}
    predictions, provenance = {}, {}
    for name, path in checkpoints.items():
        state = planner_state(shapes, torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        agent.action_head.load_state_dict({k[len("action_head."):]: v for k, v in state.items()}, strict=True)
        provenance[name] = {"path": str(path), "sha256": file_sha256(path), "planner_weights": len(state)}
        print(f"{name}: loaded all {len(state)} planner weights from {path}", flush=True)
        for scene in scenes:
            token = scene["token"]
            seed = paired_seed(evaluation.SEED_NAMESPACE, seed_base, token)
            for condition in conditions:
                record = condition_annotation(annotations[token], scene, condition)
                trajectory = evaluation.compute_trajectory_with_seed(
                    agent, evaluation.make_input(record, Path(record["front_camera"][-1])), seed)
                poses = np.asarray(trajectory_poses(trajectory), dtype=float)
                predictions.setdefault(name, {}).setdefault(token, {})[condition] = poses.tolist()
    value = {"protocol": INFERENCE_PROTOCOL, "scenes": sources,
             "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
             "diffusion_seed_base": seed_base, "conditions": conditions, "checkpoints": provenance,
             "reference": next(iter(checkpoints)),
             "real_start_speed_mps": {t: initial_speed(a) for t, a in annotations.items()},
             "predictions": predictions}
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(value) + "\n")
    # Predictions differ slightly across GPU models (e.g. A100 vs H100), so record the device.
    (output / "run_info.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=1) + "\n")
    print(f"Wrote {output / 'predictions.json'}", flush=True)
    if not args.no_score:
        run_scoring_backend(output)


# --- score ------------------------------------------------------------------------------------------

def run_scoring_backend(out):
    """Score in the configured scoring Python (NAVSIM devkit + nuplan-devkit)."""
    from ...scoring.navsim import load_config
    config = load_config()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((config["navsim_root"], str(paths.REPO_ROOT)))
    environment["NUPLAN_MAPS_ROOT"] = config["maps_root"]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    subprocess.run([config["backend_python"], "-m", "reactbench.models.recogdrive.real_scenes", "score",
                    "--run-dir", str(out)], env=environment, cwd=paths.REPO_ROOT, check=True)


def score(args):
    from ...scoring.navsim import load_config
    config = load_config()
    # navsim.common.dataclasses reads the maps root when it is first imported.
    os.environ.setdefault("NUPLAN_MAPS_ROOT", config["maps_root"])
    from ...scoring.navsim_metrics import MapContext, score_prediction_at_fault
    from ...scoring.navsim_world import RawWorlds
    out = Path(args.run_dir).expanduser().resolve()
    value = json.loads((out / "predictions.json").read_text())
    for item in value["scenes"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"A scenes file changed after inference: {item['path']}")
    scenes, _, _ = load_scenes([item["path"] for item in value["scenes"]])
    worlds = {split: RawWorlds(split_config(config, split)) for split in scenes_by_split(scenes)}
    records = []
    for scene in scenes:
        token, track = scene["token"], scene["target"]["track_token"]
        poses_by = {(model, condition): np.asarray(poses, dtype=float)
                    for model, tokens in value["predictions"].items() for condition, poses in tokens[token].items()}
        reach = max(float(np.linalg.norm(p[:, :2], axis=1).max()) for p in poses_by.values())
        world = worlds[scene["split"]].load(token, max(100., 100. * math.ceil((reach + 10.) / 100.)))
        frozen = freeze_target(world, track)
        contexts = (MapContext(token, world, config["work_root"]), MapContext(token, frozen, config["work_root"]))
        moved = track_motion(world, track)
        for (model, condition), poses in poses_by.items():
            headway = None if condition == "real" else scene["headways"][condition]
            start = value["real_start_speed_mps"][token] if headway is None else headway["speed_mps"]
            record = {"model": model, "token": token, "split": scene["split"], "condition": condition,
                      "injected_speed_mps": None if headway is None else headway["speed_mps"],
                      "gap_m": scene["target"]["gap_m"], "target_motion_4s_m": moved,
                      "non_reacting_hit_s": None if headway is None else headway["non_reacting_hit_s"],
                      **motion(poses, start)}
            for prefix, w, context in (("", world, contexts[0]), ("frozen_", frozen, contexts[1])):
                record.update(contact_fields(score_prediction_at_fault(poses, INTERVAL_S, None, w, context),
                                             track, poses, prefix))
            records.append(record)
    (out / "scores.json").write_text(json.dumps({"protocol": PROTOCOL, "records": records}) + "\n")
    summary = summarize_records(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"run": str(out), "records": len(records), "summary": summary}, indent=1))


# --- summarize --------------------------------------------------------------------------------------

def summarize(args):
    """Per run: P, NC_ins and CS with the target frozen, per headway and their mean (paper table)."""
    stem = Path(args.scenes).stem
    root = ROOT / "runs" / stem
    names = args.models or sorted(p.parent.name for p in root.glob("*/scores.json"))
    records = {}
    for name in names:
        rows = json.loads((root / name / "scores.json").read_text())["records"]
        records[name] = [r for r in rows if r["model"] == name]
    out_dir = ROOT / "summary" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: summarize_records(records[name])[name] for name in names}
    conditions = conditions_of(records[names[0]])
    headways = [c for c in conditions if c != "real"]
    bins = {name: {h: speed_bins(records[name], name, h) for h in headways} for name in names}
    mean_over_headways = {name: {key: float(np.mean([summaries[name][h][key] for h in headways]))
                                 for key in ("p_frozen", "nc_ins_frozen", "cs_kmh_frozen")}
                          for name in names}
    (out_dir / "summary.json").write_text(json.dumps({"scenes": stem, "runs": names, "summary": summaries,
                                                      "mean_over_headways": mean_over_headways,
                                                      "speed_bins": bins}, indent=1) + "\n")
    fmt = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    n = summaries[names[0]][headways[0]]["n"]
    lines = [f"Real scenes, {stem} (n = {n}). Target frozen at t = 0. P and NC_ins in %, CS = mean ego speed at "
             "the first at-fault contact with the target (km/h; n plans with such a contact).", "",
             "| Model | " + " | ".join(f"h = {h} s: P / NC_ins / CS (n)" for h in headways) + " | Mean: P / CS |",
             "|---" * (len(headways) + 2) + "|"]
    for name in names:
        s, m = summaries[name], mean_over_headways[name]
        cells = [f"{fmt(s[h]['p_frozen'])} / {fmt(s[h]['nc_ins_frozen'])} / {fmt(s[h]['cs_kmh_frozen'])} "
                 f"({s[h]['cs_n_frozen']})" for h in headways]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fmt(m['p_frozen'])} / {fmt(m['cs_kmh_frozen'])} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "table.md").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="Cohort -> scene file with the injected ego statuses")
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--headways", nargs="+", default=list(HEADWAYS))
    p.add_argument("--output", type=Path)
    p.add_argument("--metric-cache", type=Path, help="Default: navsim_metric_cache in configs/paths.yaml")
    p.add_argument("--work-root", type=Path, help="Scoring work root that receives the lane sidecars")
    p.set_defaults(function=build)
    p = sub.add_parser("infer", help="Run a ReCogDrive checkpoint on a scene file (GPU), then score")
    p.add_argument("--scenes", type=Path, default=default_scenes())
    p.add_argument("--name", required=True, help="Run name, e.g. il or il_rt")
    p.add_argument("--config", default="il", help="ReCogDrive configuration (its planner is replaced by --checkpoint)")
    p.add_argument("--checkpoint", nargs="*", help="NAME=PATH planner checkpoints (default: the configuration's)")
    p.add_argument("--no-score", action="store_true")
    p.set_defaults(function=infer)
    p = sub.add_parser("score", help="Score one run (CPU; started by infer)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.set_defaults(function=score)
    p = sub.add_parser("summarize", help="Table over the scored runs of a scene file")
    p.add_argument("--scenes", type=Path, default=default_scenes())
    p.add_argument("--models", nargs="*")
    p.set_defaults(function=summarize)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
