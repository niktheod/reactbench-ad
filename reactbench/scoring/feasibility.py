"""Braking feasibility of ReactBench samples (unavoidable collisions, paper appendix).

A sample (dataset, inserted object, look-ahead time tau, scene token) is *feasible* when braking along the
logged ground-truth (GT) path from the measured initial speed v0, at some constant deceleration
a <= A_MAX_MPS2 (8 m/s^2, as the ReactTrain targets) and then staying stopped, has no at-fault contact with
the inserted object within the 4 s horizon. The verdict is model-agnostic; other road users and the drivable
area are ignored.

- GT path: the ReactTrain target construction (``gt_path``/``path_poses`` in
  third_party/recogdrive/navsim/planning/training/recogdrive_kinematic_sft.py), mirrored here: a cubic spline through
  the origin and the GT rear-axle waypoints after merging backward and sub-5 cm steps, with chord-length knots,
  the initial tangent along the ego heading and a natural end; heading is the path tangent and the ego
  continues straight beyond the path end. ReactNAV: the eight logged 2 Hz poses; ReactPAI: the forty 10 Hz
  offline-egomotion poses that the scorer uses to register its world.
- Footprints as in the scorers: the record's rear-axle ego footprint and the inserted actor's oriented box
  moving at its constant velocity (reactbench.scoring.actor_motion). A contact is a SAT gap of at most
  COLLISION_EPSILON_M, checked at 100 Hz over [0, 4] s.
- Only the first contact counts, classified with the inserted-object branch of reactbench/scoring/at_fault.py.
  It is exempt if the ego is stopped (<= 0.05 m/s, using the speed at the preceding 10 ms sample, an upper
  bound) or if the actor's centre lies more than 150 deg from the ego heading (rear contact). Otherwise it is at
  fault: a stationary actor, a frontal contact, or a lateral contact with a pedestrian.
- a_min_avoid: smallest avoiding deceleration on the grid 0, 0.25, ..., 12 m/s^2. When the avoiding grid
  values form an upper set (monotone), it is refined by bisection to 1e-3 m/s^2. It is None if no grid value
  avoids the contact. ``feasible`` uses only the grid values up to A_MAX_MPS2.

Inputs are the scored runs of any one model (the verdict does not depend on the model), e.g.

    python -m reactbench.scoring.feasibility --reactnav outputs/predictions/recogdrive/reactnav \
        --reactpai outputs/predictions/alpamayo1_5/reactpai --workers 8

Writes {reactnav,reactpai}_per_sample.json to outputs/summaries/feasibility/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import math
import os
from pathlib import Path
import pickle
import zipfile

import numpy as np

from .. import paths
from ..io import digest, file_digest, read_json, write_json
from ..runs import parse_run_name
from .at_fault import BEHIND_ANGLE_DEG, STOPPED_SPEED_MPS

A_MAX_MPS2 = 8.0  # As the ReactTrain targets (paper §5): about the dry-road braking limit of a passenger car.
DECELERATIONS_MPS2 = np.round(np.arange(0., 12. + 1e-9, 0.25), 10)
HORIZON_S = 4.0
TIMES_S = np.linspace(0., HORIZON_S, 401)  # 100 Hz, as the scorers' collision checks
MERGE_DISTANCE_M = 0.05  # ReactTrain path: forward steps below 5 cm are merged
PATH_STEP_M = 0.01  # arc-length lookup resolution of the spline
REFINE_TOLERANCE_MPS2 = 1e-3
DATASETS = {
    "reactnav": {"dataset": "navsim", "prefix": "navsim_at_fault", "waypoint_interval_s": 0.5},
    "reactpai": {"dataset": "pai-av", "prefix": "paiav_at_fault", "waypoint_interval_s": 0.1},
}


def paiav_worlds():
    from .paiav import load_config
    return Path(load_config()["work_root"]) / "worlds"
CONTRACT = {
    "version": 1,
    "feasible": "some constant deceleration a <= a_max along the GT path from the measured v0, then held at rest, "
                "has no at-fault contact with the inserted object within 4 s; other road users and DAC ignored",
    "a_max_mps2": A_MAX_MPS2,
    "decelerations_mps2": "grid 0:0.25:12; a_min_avoid refined by bisection to 1e-3 when the avoiding set is "
                          "an upper set (monotone); feasible uses the grid values <= a_max only",
    "v0": "per-sample record groups.speed_mps (measured speed at the current frame)",
    "path": "ReactTrain construction (recogdrive_kinematic_sft.gt_path/path_poses): cubic spline through the origin "
            "and the merged GT rear-axle waypoints (backward or < 5 cm steps merged), chord-length knots, initial "
            "tangent along the ego heading, natural end; tangent heading; straight beyond the path end",
    "gt": {"reactnav": "8 logged 2 Hz poses (log ego2global matrices) in the current rear-axle frame",
           "reactpai": "40 offline-egomotion 10 Hz rig poses (paiav_world.pose_matrices) in the current rig"},
    "geometry": "record ego footprint (vehicle_footprint_parameters) and inserted-actor box at constant velocity "
                "(actor_footprint_corners); contact = SAT gap <= COLLISION_EPSILON_M at 100 Hz over [0, 4] s",
    "classification": "first contact only; exempt if the ego is stopped (speed at the preceding 10 ms sample <= "
                      "0.05 m/s) or the actor centroid is > 150 deg from the ego heading (rear); else at fault",
    "anchor_check": "|GT front bumper at tau - placement anchor|: pedestrians interception_center_xy_m, car rear "
                    "bumper (initial_center_xy_m - rear_length_m along heading_rad)",
}


# --- GT path ----------------------------------------------------------------------------------

def merge_waypoints(points, merge_distance=MERGE_DISTANCE_M):
    """Kept path points as in ReactTrain's ``_merge_waypoints``.

    A step from the last kept point that turns more than 90 deg from the direction of travel (backward), or
    one shorter than ``merge_distance``, is merged into that point.
    """
    points = np.asarray(points, dtype=np.float64)
    kept, direction = [points[0]], np.array([1., 0.])
    for point in points[1:]:
        step = point - kept[-1]
        if step @ direction < 0 or math.hypot(*step) < merge_distance:
            continue
        kept.append(point)
        direction = step
    return np.array(kept)


class GTPath:
    """Cubic path through the origin and the merged GT waypoints, evaluated by arc length."""

    def __init__(self, waypoints_xy, merge_distance=MERGE_DISTANCE_M, step_m=PATH_STEP_M):
        from scipy.interpolate import CubicSpline
        waypoints = np.asarray(waypoints_xy, dtype=np.float64)[:, :2]
        if not np.isfinite(waypoints).all():
            raise ValueError("GT waypoints must be finite")
        kept = merge_waypoints(np.vstack(([0., 0.], waypoints)), merge_distance)
        if len(kept) < 2:
            raise ValueError("GT path has no length")
        knots = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(kept, axis=0), axis=1))))
        spline = CubicSpline(knots, kept, bc_type=((1, np.array([1., 0.])), (2, np.zeros(2))))
        u = np.linspace(0., knots[-1], max(2, int(math.ceil(knots[-1] / step_m)) + 1))
        self.xy, tangent = spline(u), spline(u, 1)
        self.degenerate = bool(np.linalg.norm(tangent, axis=1).min() < 1e-3)
        self.arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))))
        self.heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        self.length = float(self.arc[-1])

    def poses(self, arc):
        """(x, y, heading) at arc lengths of any shape; the ego continues straight beyond the path end."""
        arc = np.asarray(arc, dtype=np.float64)
        flat = arc.ravel()
        inside = np.clip(flat, 0., self.length)
        x = np.interp(inside, self.arc, self.xy[:, 0])
        y = np.interp(inside, self.arc, self.xy[:, 1])
        heading = np.interp(inside, self.arc, self.heading)
        extra = np.maximum(flat - self.length, 0.)
        poses = np.column_stack((x + extra * np.cos(heading), y + extra * np.sin(heading), heading))
        return poses.reshape(arc.shape + (3,))


# --- Braking oracle ---------------------------------------------------------------------------

def braking_motion(v0, decelerations, times=TIMES_S):
    """Arc length and speed, shape (len(decelerations), len(times)), of braking at constant a from v0 at
    t = 0 and holding at rest once stopped; a = 0 keeps v0."""
    a = np.asarray(decelerations, dtype=np.float64)[:, None]
    t = np.asarray(times, dtype=np.float64)[None, :]
    stop = np.divide(v0, a, out=np.full(a.shape, np.inf), where=a > 0)
    held = np.minimum(t, stop)
    return v0 * held - 0.5 * a * held ** 2, np.maximum(v0 - a * t, 0.)


def footprints_overlap(first, second):
    """Rectangles (..., 4, 2) in consecutive corner order: True where the SAT gap is <= COLLISION_EPSILON_M,
    the contact criterion of the scorers (``signed_clearances``)."""
    from .geometry import COLLISION_EPSILON_M
    edges = np.concatenate((first[..., 1:3, :] - first[..., :2, :], second[..., 1:3, :] - second[..., :2, :]),
                           axis=-2)
    axes = np.stack((-edges[..., 1], edges[..., 0]), axis=-1)
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
    a = np.einsum("...vi,...ki->...kv", first, axes)
    b = np.einsum("...vi,...ki->...kv", second, axes)
    gaps = np.maximum(b.min(axis=-1) - a.max(axis=-1), a.min(axis=-1) - b.max(axis=-1))
    return gaps.max(axis=-1) <= COLLISION_EPSILON_M


class BrakingOracle:
    """Constant-deceleration braking along one GT path towards one inserted actor."""

    def __init__(self, path, v0, actor_motion, ego_geometry, times=TIMES_S):
        from .actor_motion import actor_footprint_corners
        from .geometry import vehicle_footprint_parameters
        if not math.isfinite(v0) or v0 < 0:
            raise ValueError("The initial speed must be finite and non-negative")
        self.path, self.v0, self.times = path, float(v0), np.asarray(times, dtype=np.float64)
        velocity = np.asarray(actor_motion["velocity_xy_mps"], dtype=np.float64)
        self.actor = actor_footprint_corners(actor_motion, 0.)[None] + self.times[:, None, None] * velocity
        self.actor_centre = self.actor.mean(axis=1)
        self.actor_radius = float(np.linalg.norm(self.actor[0] - self.actor_centre[0], axis=1).max())
        self.actor_stopped = float(np.linalg.norm(velocity)) <= STOPPED_SPEED_MPS
        footprint = vehicle_footprint_parameters(ego_geometry)
        front, rear = footprint["front_length_m"], footprint["rear_length_m"]
        left = footprint.get("left_width_m", footprint["width_m"] / 2)
        right = footprint.get("right_width_m", footprint["width_m"] / 2)
        self.local = np.array([[front, left], [front, -right], [-rear, -right], [-rear, left]])  # FL, FR, RR, RL
        self.centre_offset = np.array([(front - rear) / 2, (left - right) / 2])
        self.ego_radius = math.hypot((front + rear) / 2, (left + right) / 2)

    def _ego_corners(self, poses):
        cos, sin = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
        x = poses[:, :1] + cos * self.local[:, 0] - sin * self.local[:, 1]
        y = poses[:, 1:2] + sin * self.local[:, 0] + cos * self.local[:, 1]
        return np.stack((x, y), axis=-1)

    def contacts(self, decelerations):
        """(at_fault, kind, first-contact time or NaN) of the first contact per deceleration."""
        a = np.atleast_1d(np.asarray(decelerations, dtype=np.float64))
        arcs, speeds = braking_motion(self.v0, a, self.times)
        poses = self.path.poses(arcs)  # (A, T, 3)
        cos, sin = np.cos(poses[..., 2]), np.sin(poses[..., 2])
        centre = poses[..., :2] + np.stack((cos * self.centre_offset[0] - sin * self.centre_offset[1],
                                            sin * self.centre_offset[0] + cos * self.centre_offset[1]), axis=-1)
        near = (np.linalg.norm(centre - self.actor_centre[None], axis=-1)
                <= self.ego_radius + self.actor_radius + 1e-6)
        hit = np.zeros(near.shape, dtype=bool)
        rows, columns = np.nonzero(near)
        if len(rows):
            hit[rows, columns] = footprints_overlap(self._ego_corners(poses[rows, columns]), self.actor[columns])
        has, first = hit.any(axis=1), hit.argmax(axis=1)
        index = np.arange(len(a))
        incoming = np.where(first > 0, speeds[index, np.maximum(first - 1, 0)], self.v0)
        pose = poses[index, first]
        relative = self.actor_centre[first] - pose[:, :2]
        distance = np.linalg.norm(relative, axis=1)
        cosine = (relative[:, 0] * np.cos(pose[:, 2]) + relative[:, 1] * np.sin(pose[:, 2])) / np.where(
            distance > 0, distance, 1.)
        behind = (distance > 0) & (np.degrees(np.arccos(np.clip(cosine, -1., 1.))) > BEHIND_ANGLE_DEG)
        stopped_ego = incoming <= STOPPED_SPEED_MPS
        kind = np.where(~has, "none", np.where(stopped_ego, "stopped_ego", np.where(
            self.actor_stopped, "stopped_actor", np.where(behind, "rear", "moving_actor"))))
        at_fault = has & ~stopped_ego & (self.actor_stopped | ~behind)
        return at_fault, kind, np.where(has, self.times[first], np.nan)

    def evaluate(self, a_max=A_MAX_MPS2, grid=DECELERATIONS_MPS2, tolerance=REFINE_TOLERANCE_MPS2):
        grid = np.asarray(grid, dtype=np.float64)
        if not np.isclose(grid, a_max).any():
            raise ValueError("a_max must lie on the deceleration grid")
        at_fault, kind, time = self.contacts(grid)
        avoid = ~at_fault
        a_min, monotone = None, True
        if avoid.any():
            k = int(np.argmax(avoid))
            monotone = bool(avoid[k:].all())
            a_min = float(grid[k])
            if monotone and k > 0:
                low, high = float(grid[k - 1]), float(grid[k])
                while high - low > tolerance:
                    middle = (low + high) / 2
                    if self.contacts([middle])[0][0]:
                        low = middle
                    else:
                        high = middle
                a_min = high
        at_max = int(np.argmin(np.abs(grid - a_max)))
        return {"feasible": bool(avoid[grid <= a_max + 1e-9].any()),
                "a_min_avoid_mps2": None if a_min is None else round(a_min, 4),
                "monotone": monotone, "contact_at_zero": bool(at_fault[0]) if grid[0] == 0 else None,
                "kind_at_zero": str(kind[0]), "kind_at_amax": str(kind[at_max]),
                "contact_time_at_amax_s": None if math.isnan(time[at_max]) else round(float(time[at_max]), 3)}


# --- Placement check --------------------------------------------------------------------------

def anchor_xy(motion):
    """Placement point on the GT path: the GT front bumper at tau, in the current ego frame."""
    if motion.get("interception_center_xy_m") is not None:
        return np.asarray(motion["interception_center_xy_m"], dtype=np.float64)
    centre = np.asarray(motion["initial_center_xy_m"], dtype=np.float64)
    if motion.get("actor_type") == "car":  # rear bumper at the GT front bumper
        heading = float(motion["heading_rad"])
        return centre - float(motion["rear_length_m"]) * np.array([math.cos(heading), math.sin(heading)])
    return centre


def front_bumper(pose, ego_geometry):
    from .geometry import vehicle_footprint_parameters
    front = vehicle_footprint_parameters(ego_geometry)["front_length_m"]
    return np.asarray(pose[:2], dtype=np.float64) + front * np.array([math.cos(pose[2]), math.sin(pose[2])])


# --- GT sources -------------------------------------------------------------------------------

def to_local(poses, origin):
    """Global (x, y, heading) -> the frame of ``origin``."""
    poses = np.asarray(poses, dtype=np.float64)
    cos, sin = math.cos(origin[2]), math.sin(origin[2])
    xy = (poses[:, :2] - origin[:2]) @ np.array([[cos, -sin], [sin, cos]])
    heading = np.arctan2(np.sin(poses[:, 2] - origin[2]), np.cos(poses[:, 2] - origin[2]))
    return np.column_stack((xy, heading))


def navsim_gt(task):
    """{token: [8, 3] GT rear-axle poses at 0.5 s steps in the current frame} for tokens of one NAVSIM log."""
    log_path, tokens = task
    with open(log_path, "rb") as handle:
        frames = pickle.load(handle)
    index = {frame["token"]: i for i, frame in enumerate(frames)}
    result = {}
    for token in tokens:
        i = index[token]
        if i < 3 or i + 8 >= len(frames):  # as navsim_raw_world.SceneRepository.scene
            raise ValueError(f"Scene {token} lacks four history frames or four seconds of future")
        matrices = np.array([frames[j]["ego2global"] for j in range(i, i + 9)], dtype=np.float64)
        poses = np.column_stack((matrices[:, 0, 3], matrices[:, 1, 3], np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])))
        result[token] = to_local(poses[1:], poses[0])
    return result


def paiav_gt(task):
    """(token, [40, 3] GT rig poses at 0.1 s steps in the current rig), from the offline egomotion that the
    scorer's cached world for this clip was built from."""
    import pandas as pd
    from .paiav_world import pose_matrices
    token, background_fingerprint, current_timestamp_us, worlds = task
    stored = read_json(Path(worlds) / f"{background_fingerprint}.json")
    if not stored:
        raise FileNotFoundError(f"Missing PAI-AV world cache for {token}")
    provenance = stored["world"]["provenance"]
    if provenance["token"] != token or int(provenance["current_timestamp_us"]) != int(current_timestamp_us):
        raise ValueError(f"PAI-AV world provenance does not match {token}")
    with zipfile.ZipFile(provenance["offline_egomotion_path"]) as archive:
        data = archive.read(provenance["offline_egomotion_member"])
    if hashlib.sha256(data).hexdigest() != provenance["offline_egomotion_member_sha256"]:
        raise ValueError(f"Offline egomotion of {token} differs from the scored world")
    t0 = int(current_timestamp_us)
    matrices = pose_matrices(pd.read_parquet(io.BytesIO(data)), t0 + np.arange(41) * 100_000)
    relative = np.linalg.inv(matrices[0])[None] @ matrices
    poses = np.column_stack((relative[:, 0, 3], relative[:, 1, 3], np.arctan2(relative[:, 1, 0], relative[:, 0, 0])))
    return token, poses[1:]


# --- Driver -----------------------------------------------------------------------------------

def collect(dataset, roots):
    """Per (object, timestep): {token: sample inputs}, checked identical across every scored record."""
    from .results import discover_result_dirs
    spec = DATASETS[dataset]
    cells, sources, skipped = defaultdict(dict), {}, Counter()
    for run in discover_result_dirs(roots):
        parsed = parse_run_name(run.name, spec["dataset"])
        path = run / "metrics" / f"{spec['prefix']}_per_sample.json"
        if parsed is None or parsed[0] == "standing_animal" or not path.is_file():
            continue
        sources[str(path)] = file_digest(path)
        obj, timestep, _ = parsed
        for row in read_json(path):
            if not row.get("actor_motion"):
                skipped[f"{obj}/{timestep}/no_actor_motion"] += 1
                continue
            gt = (row["world_source"]["log_path"] if dataset == "reactnav"
                  else [row["background_fingerprint"], int(row["current_timestamp_us"])])
            entry = {"motion": row["actor_motion"], "geometry": row["geometry"],
                     "v0": float(row["groups"]["speed_mps"]), "gt": gt}
            previous = cells[obj, timestep].setdefault(row["token"], entry)
            # Speeds may differ in the last ulp between models' records (22 ReactPAI clips); the rest is exact.
            if previous is not entry and (abs(previous["v0"] - entry["v0"]) > 1e-9 or digest(
                    {k: v for k, v in previous.items() if k != "v0"}) != digest(
                    {k: v for k, v in entry.items() if k != "v0"})):
                raise ValueError(f"Models disagree on the inputs of {dataset} {obj} {timestep} {row['token']}")
    return dict(cells), sources, dict(skipped)


def token_verdicts(task):
    dataset, token, waypoints, items = task
    interval = DATASETS[dataset]["waypoint_interval_s"]
    waypoints = np.asarray(waypoints, dtype=np.float64)
    path = GTPath(waypoints[:, :2])
    rows = []
    for obj, timestep, motion, geometry, v0 in items:
        verdict = BrakingOracle(path, v0, motion, geometry).evaluate()
        error = np.linalg.norm(front_bumper(waypoints[timestep - 1], geometry) - anchor_xy(motion))
        rows.append({"token": token, "object": obj, "timestep": timestep, "tau_s": round(timestep * interval, 3),
                     "v0_mps": v0, **verdict, "anchor_error_m": round(float(error), 5),
                     "path_length_m": round(path.length, 3), "path_degenerate": path.degenerate})
    return rows


def evaluate_dataset(dataset, roots, workers):
    cells, sources, skipped = collect(dataset, roots)
    if not cells:
        raise ValueError(f"No scored {dataset} insertion runs under {roots}")
    inputs = defaultdict(list)
    for (obj, timestep), samples in sorted(cells.items()):
        for token, entry in samples.items():
            inputs[token].append((obj, timestep, entry["motion"], entry["geometry"], entry["v0"], entry["gt"]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if dataset == "reactnav":
            logs = defaultdict(set)
            for token, items in inputs.items():
                if len({item[5] for item in items}) != 1:
                    raise ValueError(f"Several logs for {token}")
                logs[items[0][5]].add(token)
            gt = {}
            for part in pool.map(navsim_gt, sorted((log, sorted(tokens)) for log, tokens in logs.items())):
                gt.update(part)
        else:
            keys = {}
            for token, items in inputs.items():
                if len({tuple(item[5]) for item in items}) != 1:
                    raise ValueError(f"Several PAI-AV worlds for {token}")
                keys[token] = items[0][5]
            worlds = str(paiav_worlds())
            gt = dict(pool.map(paiav_gt, [(t, *keys[t], worlds) for t in sorted(keys)], chunksize=8))
        tasks = [(dataset, token, gt[token], [item[:5] for item in items]) for token, items in sorted(inputs.items())]
        rows = [row for part in pool.map(token_verdicts, tasks, chunksize=16) for row in part]
    rows.sort(key=lambda r: (r["object"], r["timestep"], r["token"]))
    return rows, sources, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reactnav", nargs="*", type=Path, default=[],
                        help="Scored ReactNAV runs of one model (or directories containing them)")
    parser.add_argument("--reactpai", nargs="*", type=Path, default=[],
                        help="Scored ReactPAI runs of one model (or directories containing them)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--output-dir", type=Path, default=paths.outputs("summaries", "feasibility"))
    args = parser.parse_args()
    for dataset, roots in (("reactnav", args.reactnav), ("reactpai", args.reactpai)):
        if not roots:
            continue
        rows, sources, skipped = evaluate_dataset(dataset, roots, args.workers)
        write_json(args.output_dir / f"{dataset}_per_sample.json",
                   {"version": 1, "contract": CONTRACT, "dataset": dataset, "sources_sha256": sources,
                    "skipped_rows": skipped, "samples": rows})
        unavoidable = sum(not r["feasible"] for r in rows)
        print(f"{dataset}: {len(rows)} samples, {unavoidable} unavoidable at {A_MAX_MPS2:g} m/s^2", flush=True)


if __name__ == "__main__":
    main()
