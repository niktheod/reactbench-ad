"""Physical contacts between a predicted trajectory and the scene (inserted object + recorded road users).

Used for both ReactNAV and ReactPAI: footprints are sampled every 0.01 s along the piecewise-linear
trajectory, and each contact is refined to the instant of first touch. at_fault then classifies
the contacts.
"""
from __future__ import annotations

import math
import numpy as np

from .actor_motion import actor_footprint_corners
from .geometry import (
    COLLISION_EPSILON_M, bounded_minimum, first_contact_time, footprint_signed_clearance,
    vehicle_footprint_corners, vehicle_footprint_parameters,
)

CONTACT_SPEED_CONTRACT = {
    "version": 1,
    "scope": "first contact with any annotated background object or inserted actor",
    "definition": "from ego contact speed minus to ego contact speed; positive means lower to-condition speed",
    "units": "m/s",
    "eligibility": "both predictions collide and both first-contact speeds are available; contacted actors may differ",
    "estimator": "left derivative of piecewise-linear rear-axle x/y at each prediction's own first contact; native 0.1 s segments including origin",
    "initial_contact": "retain collision outcome; contact speed undefined at t=0 because no preceding predicted segment",
    "interpretation": "ego contact-speed reduction, not relative impact speed or a collision-severity estimate",
}


def first_contact_speed(poses, contacts, interval_length_s=0.1):
    """Describe first scene contact; never use the outgoing segment at a node."""
    if not contacts:
        return None
    time = min(float(c["time_s"]) for c in contacts)
    actors = [{k: c[k] for k in ("scope", "actor_id", "classes") if k in c}
              for c in contacts if math.isclose(c["time_s"], time, rel_tol=0, abs_tol=1e-6)]
    value = {"time_s": time, "actors": actors, "ego_speed_mps": None,
             "speed_interval_start_s": None, "speed_interval_end_s": None,
             "speed_status": "initial_contact" if time == 0 else "available"}
    if time == 0:
        return value
    nodes = np.vstack((np.zeros((1, 2)), np.asarray(poses, dtype=float)[:, :2]))
    ends = np.arange(1, len(nodes)) * interval_length_s
    if not np.isfinite(time) or time < 0 or time > ends[-1] + 1e-9:
        raise ValueError("Contact time must lie within the predicted horizon")
    index = min(int(np.searchsorted(ends, time, side="left")), len(ends) - 1)
    value.update(ego_speed_mps=float(np.linalg.norm(nodes[index + 1] - nodes[index]) / interval_length_s),
                 speed_interval_start_s=float(index * interval_length_s),
                 speed_interval_end_s=float(ends[index]))
    return value


def paired_contact_speed(before, after):
    a, b = before.get("first_contact"), after.get("first_contact")
    value = {"status": "not_both_collide", "from_speed_mps": None, "to_speed_mps": None,
             "reduction_mps": None, "shared_first_contact_actor": None}
    if not (before["all"] and after["all"]):
        return value
    if not a or not b:
        return {**value, "status": "missing_contact_speed"}
    value.update(from_speed_mps=a["ego_speed_mps"], to_speed_mps=b["ego_speed_mps"],
                 shared_first_contact_actor=bool(
                     {(c["scope"], c["actor_id"]) for c in a["actors"]} &
                     {(c["scope"], c["actor_id"]) for c in b["actors"]}))
    if a["time_s"] == 0 or b["time_s"] == 0:
        return {**value, "status": "initial_contact"}
    speeds = (a["ego_speed_mps"], b["ego_speed_mps"])
    if any(s is None or not np.isfinite(s) or s < 0 for s in speeds):
        return {**value, "status": "missing_contact_speed"}
    return {**value, "status": "scored", "reduction_mps": float(speeds[0] - speeds[1])}


def validate_trajectory(poses, interval_length_s=0.1, horizon_s=4.0):
    poses = np.asarray(poses, dtype=float)
    if not np.isclose(interval_length_s, 0.1, rtol=0, atol=1e-8):
        raise ValueError("PAI-AV collision scoring requires native 0.1 s poses")
    if not np.isclose(horizon_s, 4.0, rtol=0, atol=1e-8):
        raise ValueError("PAI-AV collision horizon must be four seconds")
    count = round(horizon_s / interval_length_s)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < count or not np.isfinite(poses).all():
        raise ValueError("Missing, nonfinite, malformed, or short x/y/heading trajectory")
    return poses[:count]


def rectangle_corners(poses, sizes):
    poses, sizes = np.asarray(poses), np.asarray(sizes)
    forward = np.stack((np.cos(poses[:, 2]), np.sin(poses[:, 2])), axis=-1)
    left = np.stack((-forward[:, 1], forward[:, 0]), axis=-1)
    return (poses[:, None, :2] +
            forward[:, None, :] * sizes[:, None, :1] * np.array([.5, .5, -.5, -.5])[None, :, None] +
            left[:, None, :] * sizes[:, None, 1:] * np.array([.5, -.5, -.5, .5])[None, :, None])


def signed_clearances(a, b):
    """Vectorized equivalent of the existing rectangular signed-clearance helper."""
    edges = np.concatenate((a[:, 1:3] - a[:, :2], b[:, 1:3] - b[:, :2]), axis=1)
    axes = np.stack((-edges[:, :, 1], edges[:, :, 0]), axis=-1)
    axes /= np.linalg.norm(axes, axis=-1)[:, :, None]
    pa, pb = np.einsum("nvi,nai->nav", a, axes), np.einsum("nvi,nai->nav", b, axes)
    gaps = np.maximum(pb.min(axis=-1) - pa.max(axis=-1), pa.min(axis=-1) - pb.max(axis=-1))
    result = gaps.max(axis=-1)
    separated = result > COLLISION_EPSILON_M
    if separated.any():
        distances = []
        for points, polygon in ((a[separated], b[separated]), (b[separated], a[separated])):
            start = polygon
            segment = np.roll(polygon, -1, axis=1) - start
            offset = points[:, :, None, :] - start[:, None, :, :]
            fraction = np.clip(np.sum(offset * segment[:, None], axis=-1) /
                               np.sum(segment * segment, axis=-1)[:, None], 0, 1)
            distance = np.linalg.norm(offset - fraction[:, :, :, None] * segment[:, None], axis=-1)
            distances.append(distance.min(axis=(1, 2)))
        result[separated] = np.minimum(*distances)
    return result


def _sample_ego(poses, times, geometry, interval_length_s=0.1):
    nodes = np.vstack((np.zeros((1, 3)), poses))
    times_nodes = np.arange(len(nodes)) * interval_length_s
    sampled = np.column_stack([np.interp(times, times_nodes, nodes[:, k] if k < 2 else np.unwrap(nodes[:, k]))
                               for k in range(3)])
    footprint = vehicle_footprint_parameters(geometry)
    front, rear = footprint["front_length_m"], footprint["rear_length_m"]
    left = footprint.get("left_width_m", footprint["width_m"] / 2)
    right = footprint.get("right_width_m", footprint["width_m"] / 2)
    forward = np.column_stack((np.cos(sampled[:, 2]), np.sin(sampled[:, 2])))
    lateral = np.column_stack((-forward[:, 1], forward[:, 0]))
    sampled[:, :2] += forward * ((front - rear) / 2) + lateral * ((left - right) / 2)
    return rectangle_corners(sampled, np.tile([front + rear, left + right], (len(times), 1)))


def track_corners(track, times):
    nodes, sizes = np.asarray(track["poses"]), np.asarray(track["sizes"])
    ts = np.asarray(track["times_s"])
    if np.min(times) < ts[0] - 1e-9 or np.max(times) > ts[-1] + 1e-9:
        raise ValueError("Cannot extrapolate an annotated actor track")
    poses = np.column_stack([np.interp(times, ts, nodes[:, k] if k < 2 else np.unwrap(nodes[:, k])) for k in range(3)])
    sizes = np.column_stack([np.interp(times, ts, sizes[:, k]) for k in range(2)])
    return rectangle_corners(poses, sizes)


def _contact(times, distances, clearance, *, refine_all=False):
    """Reuse legacy refinement for the insertion; refine each background local minimum."""
    coarse = int(np.argmin(distances))
    minima = [coarse]
    if refine_all and len(times) > 2:
        minima = sorted(set(minima) | set((np.flatnonzero((distances[1:-1] <= distances[:-2]) &
                                                         (distances[1:-1] <= distances[2:]) &
                            ((distances[1:-1] < distances[:-2]) | (distances[1:-1] < distances[2:]))) + 1).tolist()))
    candidates = [(float(times[i]), float(distances[i])) for i in np.flatnonzero(distances <= COLLISION_EPSILON_M)]
    for i in minima:
        lo, hi = max(0, i - 1), min(len(times) - 1, i + 1)
        t, distance = bounded_minimum(clearance, float(times[lo]), float(times[hi]))
        if distance <= COLLISION_EPSILON_M:
            candidates.append((t, distance))
    if not candidates:
        return None
    first = min(t for t, _ in candidates)
    index = max(0, int(np.searchsorted(times, first, side="left")) - 1)
    return float(first_contact_time(clearance, float(times[index]), first))


def score_trajectory(poses, actor_motion, ego_geometry, world, *, interval_length_s=0.1,
                     geometry_interval_s=0.01):
    poses = validate_trajectory(poses, interval_length_s, world["horizon_s"])
    return score_raw_trajectory(poses, actor_motion, ego_geometry, world, interval_length_s=interval_length_s,
                                geometry_interval_s=geometry_interval_s)


def score_raw_trajectory(poses, actor_motion, ego_geometry, world, *, interval_length_s, geometry_interval_s=0.01):
    """Same collision geometry on arbitrary native intervals; no controller or resampling of speed."""
    poses = np.asarray(poses, dtype=float)
    if (not np.isfinite(interval_length_s) or interval_length_s <= 0 or poses.ndim != 2
            or poses.shape[1] != 3 or not np.isfinite(poses).all()
            or len(poses) * interval_length_s < world["horizon_s"] - 1e-8):
        raise ValueError("Missing, malformed, or short native trajectory")
    horizon = world["horizon_s"]
    times = np.linspace(0, horizon, max(2, math.ceil(horizon / geometry_interval_s) + 1))
    ego = _sample_ego(poses, times, ego_geometry, interval_length_s)
    inserted_time = None
    if actor_motion is not None:
        initial_actor = actor_footprint_corners(actor_motion, 0.0)
        velocity = np.asarray(actor_motion["velocity_xy_mps"], dtype=float)
        actor = initial_actor[None] + times[:, None, None] * velocity[None, None]

        def insertion_clearance(t):
            return footprint_signed_clearance(vehicle_footprint_corners(
                sample_pose(poses, t, interval_length_s), ego_geometry), actor_footprint_corners(actor_motion, t))

        inserted_time = _contact(times, signed_clearances(ego, actor), insertion_clearance)
    contacts = []
    if inserted_time is not None:
        contacts.append({"scope": "inserted", "actor_id": "inserted", "time_s": inserted_time})
    for track in world["tracks"]:
        start, end = max(0, track["times_s"][0]), min(horizon, track["times_s"][-1])
        ts = np.unique(np.concatenate(([start, end], times[(times >= start) & (times <= end)],
                           np.asarray(track["times_s"])[(np.asarray(track["times_s"]) >= start) &
                                                       (np.asarray(track["times_s"]) <= end)])))
        boxes = track_corners(track, ts)
        ego_boxes = _sample_ego(poses, ts, ego_geometry, interval_length_s)
        # A swept axis-aligned bound cheaply eliminates spatially disjoint tracks.
        if np.any(boxes.min(axis=(0, 1)) > ego_boxes.max(axis=(0, 1))) or np.any(ego_boxes.min(axis=(0, 1)) > boxes.max(axis=(0, 1))):
            continue

        def clearance(t):
            return footprint_signed_clearance(vehicle_footprint_corners(
                sample_pose(poses, t, interval_length_s), ego_geometry), track_corners(track, [t])[0])

        contact_time = _contact(ts, signed_clearances(ego_boxes, boxes), clearance, refine_all=True)
        if contact_time is not None:
            contacts.append({"scope": "background", "actor_id": track["actor_id"],
                             "classes": track["classes"], "time_s": contact_time})
    contacts.sort(key=lambda c: (c["time_s"], c["actor_id"]))
    return {"inserted": inserted_time is not None, "background": any(c["scope"] == "background" for c in contacts),
            "all": bool(contacts), "contacts": contacts,
            "first_contact": first_contact_speed(poses, contacts, interval_length_s)}


def paired_outcome(before, after):
    result = {"contact_speed": paired_contact_speed(before, after)}
    for scope in ("inserted", "background", "all"):
        a, b = bool(before[scope]), bool(after[scope])
        result[scope] = {"original_collision": a, "edited_collision": b, "benefit": int(a) - int(b),
                         "transition": "avoided" if a and not b else "introduced" if b and not a else
                                       "persistent_collision" if a else "persistent_clear"}
    return result


def sample_pose(poses, time_s, interval_length_s):
    nodes = np.vstack((np.zeros((1, 3)), poses))
    times = np.arange(len(nodes)) * interval_length_s
    return np.array([np.interp(time_s, times, nodes[:, k] if k < 2 else np.unwrap(nodes[:, k])) for k in range(3)])
