#!/usr/bin/env python
"""Real-scenes test set (paper appendix C.4): real stopped vehicles ahead of a slow ego vehicle.

Model-agnostic: reads only the OpenScene logs and the nuPlan map. Per scene (its current frame):

- ego lane chain: the nuPlan lane or lane connector under the rear axle (on-route first, then the lowest
  heading error, below 90 deg), followed by successors until >= 60 m of lane lie ahead of the ego (on-route
  first, then the branch holding most of the logged ego future, then the one closest to it);
- target: the nearest vehicle at < 0.5 m/s whose whole box lies inside that chain, 5-15 m ahead (arc length
  along the chain centreline from the ego front bumper to the box's nearest point), with no vehicle, bicycle or
  pedestrian (any speed) nearer in the ego-width strip along the centreline;
- kept when the real ego speed is < 3 m/s and at least half of the target's box area lies in that strip.

The output lists every kept scene with its target, plus the selection funnel (12,146 navtest scenes -> 308).
Needs the nuPlan map API (nuplan-devkit):

    python -m reactbench.selection.real_scenes --output outputs/real_scenes/cohort.json --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import pickle
from pathlib import Path

import numpy as np

from .. import paths

NAVTEST_FILTER = (paths.REPO_ROOT / "third_party/recogdrive/navsim/planning/script/config/common/"
                  "train_test_split/scene_filter/navtest.yaml")
EGO = {"front_m": 4.049, "rear_m": 1.127, "half_width_m": 1.1485}  # nuPlan Pacifica, origin = rear axle
CRITERIA = {
    "version": 1,
    "max_ego_speed_mps": 3.0,
    "max_vehicle_speed_mps": 0.5,
    "gap_m": [5.0, 15.0],
    "min_strip_fraction": 0.5,
    "path_blockers": ["vehicle", "bicycle", "pedestrian"],
    "lookahead_m": 60.0,
    "max_lanes": 30,
    "lane_seam_m": 0.01,
    "future_frames": 10,
    "ego_lane": "lane or lane connector under the rear axle (on-route first, then lowest heading error < 90 deg); "
                "successors until >= lookahead_m lie ahead of the ego (on-route first, then most logged future ego "
                "positions inside, then smallest mean distance to them)",
    "gap": "arc length along the lane-chain centreline from the ego front bumper to the nearest box corner",
    "strip": "lane-chain centreline buffered by the ego half width (flat caps)",
    "first_in_path": "no other path_blockers box (any speed) intersects the strip nearer than the target, now",
    "target": "nearest qualifying vehicle wholly inside the lane chain, then strip fraction >= min_strip_fraction",
}


# --- Geometry (pure; tests pass synthetic lanes) ---------------------------------------------------

def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def box_corners(box):
    """Four (x, y) corners of a gt_boxes row (x, y, z, length, width, height, yaw) in its frame."""
    x, y, _, length, width, _, yaw = box[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([length, length, -length, -length]) / 2
    dy = np.array([width, -width, -width, width]) / 2
    return np.stack([x + c * dx - s * dy, y + s * dx + c * dy], 1)


def _line(lane):
    from shapely.geometry import LineString
    return LineString([(q.x, q.y) for q in lane.baseline_path.discrete_path])


def lane_heading_at(lane, xy):
    path = np.array([[q.x, q.y, q.heading] for q in lane.baseline_path.discrete_path])
    return path[np.argmin(np.hypot(path[:, 0] - xy[0], path[:, 1] - xy[1])), 2]


def ego_lane_chain(map_api, xy, heading, route, future_xy, criteria=CRITERIA, layers=None, point=None):
    """(lanes, corridor) of the ego lane chain, or (None, None) when no lane fits the ego."""
    from shapely.geometry import Point
    from shapely.ops import unary_union
    if layers is None:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        layers = (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR)
    if point is None:
        from nuplan.common.actor_state.state_representation import Point2D as point
    candidates = []
    for layer in layers:
        for lane in map_api.get_all_map_objects(point(*xy), layer):
            error = abs(wrap(lane_heading_at(lane, xy) - heading))
            if error < np.pi / 2:
                candidates.append((lane.get_roadblock_id() not in route, error, lane))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: (c[0], c[1]))
    chain = [candidates[0][2]]
    base = _line(chain[0])
    ahead = base.length - base.project(Point(*xy))
    future = [Point(*p) for p in future_xy]

    def key(lane):
        inside = sum(lane.polygon.contains(p) for p in future)
        line = _line(lane)
        return (lane.get_roadblock_id() not in route, -inside,
                float(np.mean([line.distance(p) for p in future])) if future else 0.)

    while ahead < criteria["lookahead_m"] and len(chain) < criteria["max_lanes"]:
        successors = chain[-1].outgoing_edges
        if not successors:
            break
        chain.append(min(successors, key=key))
        ahead += chain[-1].baseline_path.length
    corridor = unary_union([lane.polygon for lane in chain]).buffer(criteria["lane_seam_m"])
    return chain, corridor


def chain_centreline(chain):
    from shapely.geometry import LineString
    points = []
    for lane in chain:
        for q in lane.baseline_path.discrete_path:
            if not points or (q.x - points[-1][0]) ** 2 + (q.y - points[-1][1]) ** 2 > 1e-6:
                points.append((q.x, q.y))
    return LineString(points)


def span(line, polygon, s_bumper):
    """Arc-length extent of a polygon along ``line``, relative to the front bumper."""
    from shapely.geometry import Point
    proj = [line.project(Point(xy)) for xy in polygon.exterior.coords]
    return min(proj) - s_bumper, max(proj) - s_bumper


def select_target(objects, corridor, line, s_bumper, half_width, criteria=CRITERIA):
    """The scene's target and funnel flags.

    ``objects``: [(track, class name, global footprint polygon, speed m/s)]. Returns (target or None, flags).
    """
    strip = line.buffer(half_width, cap_style=2)
    lo, hi = criteria["gap_m"]
    candidates = []
    for track, name, polygon, speed in objects:
        if name != "vehicle" or speed >= criteria["max_vehicle_speed_mps"] or not polygon.within(corridor):
            continue
        gap = span(line, polygon, s_bumper)[0]
        if lo <= gap < hi:
            candidates.append((gap, track, polygon, speed))
    unblocked = []
    for gap, track, polygon, speed in candidates:
        blocked = False
        for other, name, poly, _ in objects:
            if other != track and name in criteria["path_blockers"] and poly.intersects(strip):
                near, far = span(line, poly, s_bumper)
                if far > 0 and near < gap:
                    blocked = True
                    break
        if not blocked:
            unblocked.append((gap, track, polygon, speed))
    flags = {"in_lane_candidate": bool(candidates), "first_in_path": bool(unblocked), "kept": False}
    if not unblocked:
        return None, flags
    gap, track, polygon, speed = min(unblocked, key=lambda c: c[0])
    fraction = float(polygon.intersection(strip).area / polygon.area)
    target = {"track_token": track, "gap_lanepath_m": float(gap), "strip_fraction": fraction, "speed_mps": float(speed)}
    flags["kept"] = fraction >= criteria["min_strip_fraction"]
    return (target if flags["kept"] else None), flags


# --- Logs ------------------------------------------------------------------------------------------

def frame_objects(frame, classes):
    """[(track, class, global footprint, speed)] of the frame's annotations of the given classes."""
    from shapely.geometry import Polygon
    anns = frame["anns"]
    names = np.asarray(anns["gt_names"], dtype=object)
    if len(names) == 0:
        return []
    T = np.asarray(frame["ego2global"])
    boxes, velocity = np.asarray(anns["gt_boxes"]), np.asarray(anns["gt_velocity_3d"])
    return [(anns["track_tokens"][j], str(names[j]), Polygon(box_corners(boxes[j]) @ T[:2, :2].T + T[:2, 3]),
             float(np.linalg.norm(velocity[j][:2]))) for j in range(len(names)) if names[j] in classes]


def target_motion(frames, index, track, seconds=4.):
    """Largest displacement (m) of the track's box centre from now over the next ``seconds`` (2 Hz frames)."""
    start = None
    largest = 0.
    for frame in frames[index:index + int(round(seconds / .5)) + 1]:
        anns = frame["anns"]
        tokens = list(anns["track_tokens"])
        if track not in tokens:
            continue
        T = np.asarray(frame["ego2global"])
        centre = T[:2, :2] @ np.asarray(anns["gt_boxes"])[tokens.index(track)][:2] + T[:2, 3]
        start = centre if start is None else start
        largest = max(largest, float(np.linalg.norm(centre - start)))
    return largest


_MAPS = {}


def _map(maps_root, name):
    if name not in _MAPS:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        _MAPS[name] = get_maps_api(str(maps_root), "nuplan-maps-v1.0", name)
    return _MAPS[name]


def analyse_log(job):
    """Kept samples and funnel counts of one log."""
    path, tokens, maps_root, criteria = job
    frames = pickle.load(open(path, "rb"))
    index = {frame["token"]: i for i, frame in enumerate(frames)}
    counts = {"scenes": 0, "slow_ego": 0, "no_ego_lane": 0, "in_lane_candidate": 0, "first_in_path": 0, "kept": 0}
    samples = []
    for token in sorted(set(tokens) & set(index)):
        counts["scenes"] += 1
        i = index[token]
        frame = frames[i]
        ego_speed = float(np.hypot(*frame["ego_dynamic_state"][:2]))
        if ego_speed >= criteria["max_ego_speed_mps"]:
            continue
        counts["slow_ego"] += 1
        objects = frame_objects(frame, set(criteria["path_blockers"]))
        T = np.asarray(frame["ego2global"])
        xy, heading = T[:2, 3], float(np.arctan2(T[1, 0], T[0, 0]))
        future = [np.asarray(f["ego2global"])[:2, 3] for f in frames[i + 1:i + 1 + criteria["future_frames"]]]
        chain, corridor = ego_lane_chain(_map(maps_root, frame["map_location"]), xy, heading,
                                         set(frame["roadblock_ids"]), future, criteria)
        if chain is None:
            counts["no_ego_lane"] += 1
            continue
        from shapely.geometry import Point
        line = chain_centreline(chain)
        s_bumper = line.project(Point(*xy)) + EGO["front_m"]
        target, flags = select_target(objects, corridor, line, s_bumper, EGO["half_width_m"], criteria)
        for name in ("in_lane_candidate", "first_in_path", "kept"):
            counts[name] += flags[name]
        if target is None:
            continue
        target["motion_4s_m"] = target_motion(frames, i, target["track_token"])
        samples.append({"token": token, "log": frame["log_name"], "map_location": frame["map_location"],
                        "ego_speed_mps": ego_speed, "lane_ids": [str(lane.id) for lane in chain], "target": target})
    return samples, counts


def scene_tokens(source=NAVTEST_FILTER):
    """{log name: tokens} of the navtest scenes (tokens are matched per log file)."""
    import yaml
    config = yaml.safe_load(Path(source).read_text())
    tokens = set(config["tokens"])
    return {log: tokens for log in config["log_names"]}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=None, help="Default: navsim_logs_test in configs/paths.yaml")
    parser.add_argument("--maps", type=Path, default=None, help="Default: nuplan_maps in configs/paths.yaml")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new --output")
    log_dir = args.logs or paths.navsim_logs("test")
    maps_root = args.maps or paths.nuplan_maps()
    jobs = [(log_dir / f"{log}.pkl", tokens, maps_root, CRITERIA) for log, tokens in sorted(scene_tokens().items())]
    missing = [str(job[0]) for job in jobs if not job[0].is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} logs missing, e.g. {missing[0]}")
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(analyse_log, jobs, chunksize=1)
    samples = sorted((s for found, _ in results for s in found), key=lambda s: s["token"])
    funnel = {key: sum(counts[key] for _, counts in results) for key in results[0][1]}
    value = {"version": 1, "criteria": CRITERIA, "ego": EGO, "split": "navtest",
             "provenance": {"tokens": {"path": str(NAVTEST_FILTER), "sha256": file_sha256(NAVTEST_FILTER)},
                            "logs": str(log_dir), "maps": str(maps_root), "code": "reactbench/selection/real_scenes.py"},
             "funnel": funnel, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "funnel": funnel}, indent=1))


if __name__ == "__main__":
    main()
