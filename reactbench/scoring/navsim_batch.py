"""Batched ReactNAV scoring of several predictions for one scene.

Gives exactly the contacts of the scalar reference (``collisions.score_raw_trajectory``) while
reusing ego samples and rejecting distant road users early.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .collisions import (
    COLLISION_EPSILON_M, _contact, _sample_ego, actor_footprint_corners, first_contact_speed,
    footprint_signed_clearance, sample_pose, signed_clearances, track_corners,
    vehicle_footprint_corners, vehicle_footprint_parameters,
)


def implementation_identity():
    """Record this execution backend separately from the frozen input artifact."""
    return {"name": "navsim-raw-batch-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _track_times(track, times, horizon):
    start, end = max(0, track["times_s"][0]), min(horizon, track["times_s"][-1])
    nodes = np.asarray(track["times_s"])
    return np.unique(np.concatenate(([start, end], times[(times >= start) & (times <= end)],
                                     nodes[(nodes >= start) & (nodes <= end)])))


def _possible_candidates(track, ego_min, ego_max):
    """Conservative continuous bounds, before interpolating an actor's boxes.

Linear centers stay inside their node bounds. The maximum corner radius also
bounds interpolated sizes at every heading, including rotation between nodes.
Roundoff padding makes this only a broad-phase rejection; the evaluator's
original sampled-box rejection and contact tolerances still decide the result.
"""
    centers = np.asarray(track["poses"], dtype=float)[:, :2]
    sizes = np.asarray(track["sizes"], dtype=float)
    radius = np.hypot(sizes[:, 0], sizes[:, 1]).max() / 2
    lower, upper = centers.min(axis=0) - radius, centers.max(axis=0) + radius
    padding = 1e-9 + 16 * np.finfo(float).eps * np.maximum(np.abs(lower), np.abs(upper))
    return ~((lower - padding > ego_max).any(axis=1) | (ego_min > upper + padding).any(axis=1))


def _corner_speed_bound(poses, times, radius, sizes=None):
    """Upper bound on every footprint point's speed, across all native knots."""
    if len(times) < 2:
        return 0.
    dt = np.diff(times)
    rate = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) / dt
    rate += np.abs(np.diff(np.unwrap(poses[:, 2]))) / dt * radius
    if sizes is not None:
        rate += .5 * np.linalg.norm(np.diff(sizes, axis=0), axis=1) / dt
    return float(rate.max())


def _separated_throughout(distances, times, speed_bound, coordinate_scale):
    """Prove that refinement cannot find a contact between these samples.

    Polygon distance can shrink at most by the sum of their point-speed bounds
    times elapsed time. Every time lies within half the largest grid gap of a
    sample. Only skip when even that conservative movement cannot close the
    smallest sampled gap, including contact tolerance and roundoff padding.
    Otherwise use the *unchanged* scalar refinement and first-contact search.
    """
    half_gap = np.diff(times).max(initial=0.) / 2
    padding = 1e-9 + 128 * np.finfo(float).eps * max(1., coordinate_scale, speed_bound)
    return distances.min() > speed_bound * half_gap + COLLISION_EPSILON_M + padding


def _batch_dac(poses, interval, world):
    if world.get("map_status") != "available":
        return [{"dac": None, "dac_status": "unavailable",
                 "dac_reason": world.get("map_error", "Missing map geometry")} for _ in poses]
    from shapely import wkb
    from shapely.vectorized import contains

    times = np.arange(41) * .1
    corners = np.stack([_sample_ego(p, times, world["geometry"], interval) for p in poses])
    theta = world["origin"][2]
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    points = corners @ rotation.T + np.asarray(world["origin"][:2])
    inside = np.zeros(points.shape[:-1], dtype=bool)
    for value in world["map_polygons_wkb"]:
        polygon = wkb.loads(value, hex=True)
        inside |= contains(polygon, points[..., 0], points[..., 1])
    results = []
    for mask in inside:
        failures = np.flatnonzero(~mask.all(axis=1))
        results.append({"dac": int(not len(failures)), "dac_status": "available",
                        "first_non_drivable_time_s": float(times[failures[0]]) if len(failures) else None})
    return results


def score_predictions(poses, interval_length_s, actor_motion, world):
    """Return scalar-evaluator dictionaries for a scene's (G, T, 3) candidates.

    Keep the exact 100 Hz grid plus each actor's annotation knots/lifetime,
    signed-clearance kernel, local-minimum refinement and incoming-segment
    contact-speed estimator. Batching changes reuse, not collision resolution.
    """
    poses = np.asarray(poses, dtype=float)
    horizon = world["horizon_s"]
    if (not np.isfinite(interval_length_s) or interval_length_s <= 0 or poses.ndim != 3
            or poses.shape[2] != 3 or not np.isfinite(poses).all()
            or poses.shape[1] * interval_length_s < horizon - 1e-8):
        raise ValueError("Missing, malformed, or short native trajectory batch")
    if not len(poses):
        return []

    times = np.linspace(0, horizon, max(2, math.ceil(horizon / .01) + 1))
    tracks = world["tracks"]
    track_times = [_track_times(track, times, horizon) for track in tracks]
    # Taking a subset of this union gives precisely the scalar caller's time
    # grid, even for one-frame tracks, irregular knots and missing annotations.
    union = np.unique(np.concatenate([times, *track_times]))
    ego = np.stack([_sample_ego(p, union, world["geometry"], interval_length_s) for p in poses])
    ego_min, ego_max = ego.min(axis=(1, 2)), ego.max(axis=(1, 2))
    footprint = vehicle_footprint_parameters(world["geometry"])
    radius = math.hypot(max(footprint["front_length_m"], footprint["rear_length_m"]),
        max(footprint.get("left_width_m", footprint["width_m"]/2),
            footprint.get("right_width_m", footprint["width_m"]/2)))
    node_times = np.arange(poses.shape[1] + 1) * interval_length_s
    ego_rates = [_corner_speed_bound(np.vstack((np.zeros((1, 3)), p)), node_times, radius) for p in poses]
    contacts = [[] for _ in poses]

    if actor_motion is not None:
        initial = actor_footprint_corners(actor_motion, 0.)
        velocity = np.asarray(actor_motion["velocity_xy_mps"], dtype=float)
        actor = initial[None] + times[:, None, None] * velocity[None, None]
        for i, p in enumerate(poses):
            def clearance(t):
                return footprint_signed_clearance(vehicle_footprint_corners(
                    sample_pose(p, t, interval_length_s), world["geometry"]),
                    actor_footprint_corners(actor_motion, t))

            contact = _contact(times, signed_clearances(ego[i, np.searchsorted(union, times)], actor), clearance)
            if contact is not None:
                contacts[i].append({"scope": "inserted", "actor_id": "inserted", "time_s": contact})

    # Reuse ego samples and their bounds for actors with the same time grid.
    grids = {}
    for track, ts in zip(tracks, track_times):
        possible = _possible_candidates(track, ego_min, ego_max)
        if not possible.any():
            continue
        key = ts.tobytes()
        if key not in grids:
            boxes = ego[:, np.searchsorted(union, ts)]
            grids[key] = (boxes, boxes.min(axis=(1, 2)), boxes.max(axis=(1, 2)))
        ego_boxes, lower, upper = grids[key]
        boxes = track_corners(track, ts)
        # Exactly the scalar evaluator's sampled swept-box rejection.
        possible &= ~((boxes.min(axis=(0, 1)) > upper).any(axis=1) |
                      (lower > boxes.max(axis=(0, 1))).any(axis=1))
        indices = np.flatnonzero(possible)
        if not len(indices):
            continue
        selected = ego_boxes[indices]
        distances = signed_clearances(selected.reshape(-1, 4, 2),
            np.broadcast_to(boxes, selected.shape).reshape(-1, 4, 2)).reshape(len(indices), len(ts))
        sizes = np.asarray(track["sizes"], dtype=float)
        actor_radius = np.hypot(sizes[:, 0], sizes[:, 1]).max() / 2
        actor_rate = _corner_speed_bound(np.asarray(track["poses"], dtype=float),
                                        np.asarray(track["times_s"]), actor_radius, sizes)
        coordinate_scale = max(float(np.abs(boxes).max()), float(np.abs(selected).max()))
        for i, ds in zip(indices, distances):
            if _separated_throughout(ds, ts, ego_rates[i] + actor_rate, coordinate_scale):
                continue

            def clearance(t):
                return footprint_signed_clearance(vehicle_footprint_corners(
                    sample_pose(poses[i], t, interval_length_s), world["geometry"]), track_corners(track, [t])[0])

            contact = _contact(ts, ds, clearance, refine_all=True)
            if contact is not None:
                contacts[i].append({"scope": "background", "actor_id": track["actor_id"],
                                    "classes": track["classes"], "time_s": contact})

    results = []
    for p, cs, dac in zip(poses, contacts, _batch_dac(poses, interval_length_s, world)):
        cs.sort(key=lambda c: (c["time_s"], c["actor_id"]))
        inserted = any(c["scope"] == "inserted" for c in cs)
        results.append({"inserted": inserted, "background": any(c["scope"] == "background" for c in cs),
            "all": bool(cs), "contacts": cs, "first_contact": first_contact_speed(p, cs, interval_length_s),
            **dac, "nc_all": int(not cs), "nc_inserted": int(not inserted),
            "nc_dac": int(not cs) * dac["dac"] if dac["dac"] is not None else None,
            "interval_length_s": interval_length_s})
    return results
