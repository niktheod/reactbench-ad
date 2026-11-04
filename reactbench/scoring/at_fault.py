"""At-fault classification of raw-trajectory contacts (evaluation protocol V2).

Stock PDMS collision classes (nuPlan ``get_collision_type`` as NAVSIM scores them),
applied to the raw predicted trajectory at each actor's refined first-contact time,
with two additions. A physical score from ``score_raw_trajectory`` already holds
one contact per actor segment; this module only classifies, so the physical
contacts are identical to the V1 (NC-all) protocol by construction.

Order (first match wins), ego speed = incoming native-segment speed:
1. ego stopped (<= 0.05 m/s): exempt;
2. other actor stopped (<= 0.05 m/s; static objects always): at fault;
3. actor centre within 30 deg of straight behind the rear axle: exempt (rear),
   unless the ego moves backwards (displacement against its predicted heading);
4. ego front-bumper edge touches the actor: at fault (front);
5. otherwise lateral: at fault for pedestrians/cyclists, or when the ego spans
   several lanes (stock lane test) or is off-road; PAI-AV has no map, so lateral
   contacts with vehicles are never at fault there.
"""
from __future__ import annotations

import math

import numpy as np

from .actor_motion import actor_footprint_corners
from .geometry import point_segment_distance, vehicle_footprint_corners

STOPPED_SPEED_MPS = 0.05  # stock nuPlan/NAVSIM threshold for ego and tracks
BEHIND_ANGLE_DEG = 150.0  # stock nuPlan is_agent_behind
FRONT_TOUCH_M = 1e-6  # numerical tolerance at the exact first-touch time
SPEED_WINDOW_S = 1.0  # recorded-track speed: displacement over this window around the contact

# Pedestrian/cyclist classes for the added lateral rule.
VULNERABLE = {"navsim": frozenset({"PEDESTRIAN", "BICYCLE"}),
              "pai-av": frozenset({"person", "rider", "stroller"})}
# Static objects never move (stock: non-Agent tracks are always stopped).
STATIC = {"navsim": frozenset({"CZONE_SIGN", "BARRIER", "TRAFFIC_CONE", "GENERIC_OBJECT"}),
          "pai-av": frozenset({"protruding_object"})}
INSERTED_VULNERABLE = frozenset({"pedestrian"})

AT_FAULT_CONTRACT = {
    "version": 2,
    "rules": "stock PDMS collision classes on the raw trajectory at each actor's first-contact time "
             "(first contact per actor id, as PDMS ignores repeat contacts); additions: lateral contact with a "
             "pedestrian/cyclist while the ego moves is at fault; a contact while the ego moves backwards is never "
             "exempt as rear",
    "ego_speed": "incoming native-segment speed (first segment at t=0); stopped <= 0.05 m/s",
    "actor_speed": "inserted: known velocity; recorded agents: displacement over 1 s around the contact; "
                   "static objects always stopped; stopped <= 0.05 m/s",
    "behind": "actor footprint centroid more than 150 deg from the ego heading, seen from the rear axle",
    "front": "ego front-bumper edge touches the actor footprint at the first-contact time (1e-6 m tolerance)",
    "lateral": "at fault for pedestrians/cyclists, or ego spanning several lanes (stock PDMS lane test, NAVSIM "
               "metric-cache lanes) or off-road (any ego corner outside the drivable polygons); PAI-AV: vehicles "
               "never at fault (no map)",
    "vulnerable_classes": {k: sorted(v) for k, v in VULNERABLE.items()},
    "static_classes": {k: sorted(v) for k, v in STATIC.items()},
}

CONTACT_SPEED_CONTRACT = {
    "version": 2,
    "scope": "first at-fault contact with any annotated object or the inserted actor",
    "definition": "from ego contact speed minus to ego contact speed; positive means lower to-condition speed",
    "units": "m/s",
    "eligibility": "both predictions have an at-fault contact and both first at-fault-contact speeds are available; "
                   "contacted actors may differ",
    "estimator": "left derivative of piecewise-linear rear-axle x/y at each prediction's own first at-fault contact; "
                 "native segments including origin",
    "initial_contact": "retain collision outcome; contact speed undefined at t=0 because no preceding predicted segment",
    "interpretation": "ego contact-speed reduction, not relative impact speed or a collision-severity estimate",
}


def ego_motion(poses, time_s, interval_length_s):
    """Incoming native-segment displacement and the interpolated predicted pose at ``time_s``."""
    from .collisions import sample_pose
    nodes = np.vstack((np.zeros((1, 3)), np.asarray(poses, dtype=float)))
    # Same segment choice as the contact-speed estimator (first_contact_speed).
    ends = np.arange(1, len(nodes)) * interval_length_s
    index = min(int(np.searchsorted(ends, time_s, side="left")), len(ends) - 1)
    return nodes[index + 1, :2] - nodes[index, :2], sample_pose(poses, time_s, interval_length_s)


def track_speed(track, time_s, window_s=SPEED_WINDOW_S):
    """Mean speed over ``window_s`` around ``time_s``, clipped to the track's annotated lifetime."""
    times = np.asarray(track["times_s"], dtype=float)
    xy = np.asarray(track["poses"], dtype=float)[:, :2]
    lower, upper = max(times[0], time_s - window_s / 2), min(times[-1], time_s + window_s / 2)
    if upper - lower < 1e-6:
        return 0.
    a = np.array([np.interp(lower, times, xy[:, k]) for k in range(2)])
    b = np.array([np.interp(upper, times, xy[:, k]) for k in range(2)])
    return float(np.linalg.norm(b - a) / (upper - lower))


def _segment_distance(a, b, c, d):
    """Distance between segments ab and cd (0 when they intersect)."""

    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    s1, s2, s3, s4 = side(a, b, c), side(a, b, d), side(c, d, a), side(c, d, b)
    if s1 * s2 < 0 and s3 * s4 < 0:
        return 0.
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def segment_box_distance(a, b, corners):
    """Distance from segment ab to a convex 4-corner footprint (0 when touching or inside); numpy only."""
    corners = np.asarray(corners, dtype=float)
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    crosses = [((q[0] - p[0]) * (a[1] - p[1]) - (q[1] - p[1]) * (a[0] - p[0])) for p, q in edges]
    if all(c >= 0 for c in crosses) or all(c <= 0 for c in crosses):
        return 0.  # an endpoint lies inside (or on) the footprint
    return min(_segment_distance(np.asarray(a), np.asarray(b), p, q) for p, q in edges)


def _to_global(points, origin):
    theta = origin[2]
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return np.asarray(points) @ rotation.T + np.asarray(origin[:2])


def spans_several_lanes(corners_global, lanes):
    """Stock PDMS: >1 lane polygon holds an ego corner and none holds all four (boundary excluded)."""
    from shapely import contains_xy
    hits = np.array([contains_xy(p, corners_global[:, 0], corners_global[:, 1]) for p in lanes])
    if not len(hits):
        return False
    counts = hits.sum(axis=1)
    return bool((counts > 0).sum() > 1 and not (counts == 4).any())


def off_road(corners_global, drivable):
    """Stock PDMS: at least one ego corner lies in no drivable polygon (boundary excluded)."""
    from shapely import contains_xy
    inside = np.zeros(len(corners_global), dtype=bool)
    for polygon in drivable:
        inside |= contains_xy(polygon, corners_global[:, 0], corners_global[:, 1])
    return bool(not inside.all())


def classify_contact(poses, interval_length_s, time_s, ego_geometry, other_corners, *,
                     other_stopped, vulnerable, map_context=None):
    """Return (stock-style type, at_fault) for one contact.

    ``map_context`` (NAVSIM) is ``{"origin", "drivable", "lanes"}`` with shapely polygons
    in global coordinates; ``None`` (PAI-AV) means lateral contacts with vehicles are exempt.
    """
    displacement, pose = ego_motion(poses, time_s, interval_length_s)
    if np.linalg.norm(displacement) / interval_length_s <= STOPPED_SPEED_MPS:
        return "stopped_ego", False
    if other_stopped:
        return "stopped_actor", True
    heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
    other = np.asarray(other_corners, dtype=float)
    centre = other.mean(axis=0) - pose[:2]  # footprints are rectangles: centroid = corner mean
    norm = np.linalg.norm(centre)
    if norm > 0 and math.degrees(math.acos(np.clip(centre @ heading / norm, -1., 1.))) > BEHIND_ANGLE_DEG:
        backwards = bool(displacement @ heading < 0)
        return ("rear_reversing", True) if backwards else ("rear", False)
    ego = np.asarray(vehicle_footprint_corners(pose, ego_geometry))
    if segment_box_distance(ego[0], ego[1], other) <= FRONT_TOUCH_M:  # corners 0, 1: front-left, front-right
        return "front", True
    if vulnerable:
        return "lateral_vulnerable", True
    if map_context is None:
        return "lateral", False
    corners = _to_global(ego, map_context["origin"])
    if off_road(corners, map_context["drivable"]):
        return "lateral_off_road", True
    if spans_several_lanes(corners, map_context["lanes"]):
        return "lateral_several_lanes", True
    return "lateral", False


def classify_score(score, poses, interval_length_s, *, actor_motion, tracks, ego_geometry, dataset,
                   map_context=None):
    """Turn a physical ``score_raw_trajectory`` result into a V2 (at-fault) collision result.

    Every actor's first contact is classified. ``contacts`` keeps the at-fault ones and
    ``exempt_contacts`` the rest; ``all``/``inserted``/``background`` and ``first_contact``
    refer to at-fault contacts only.
    """
    from .collisions import first_contact_speed, track_corners
    if dataset not in VULNERABLE:
        raise ValueError(f"Unknown dataset for at-fault rules: {dataset}")
    first = {}
    for contact in score["contacts"]:
        key = (contact["scope"], contact["actor_id"])
        if key not in first or contact["time_s"] < first[key]["time_s"]:
            first[key] = contact
    segments = {}
    for track in tracks:
        segments.setdefault(track["actor_id"], []).append(track)
    at_fault, exempt = [], []
    for contact in sorted(first.values(), key=lambda c: (c["time_s"], c["actor_id"])):
        time_s = float(contact["time_s"])
        if contact["scope"] == "inserted":
            corners = actor_footprint_corners(actor_motion, time_s)
            stopped = float(np.linalg.norm(actor_motion["velocity_xy_mps"])) <= STOPPED_SPEED_MPS
            vulnerable = str(actor_motion.get("actor_type", "pedestrian")) in INSERTED_VULNERABLE
        else:
            candidates = [t for t in segments.get(contact["actor_id"], [])
                          if t["times_s"][0] - 1e-6 <= time_s <= t["times_s"][-1] + 1e-6]
            if not candidates:
                raise ValueError(f"No track segment covers contact {contact['actor_id']} at {time_s}")
            track = candidates[0]
            classes = set(track.get("classes", []))
            inside = min(max(time_s, track["times_s"][0]), track["times_s"][-1])
            corners = track_corners(track, [inside])[0]
            stopped = bool(classes & STATIC[dataset]) or track_speed(track, inside) <= STOPPED_SPEED_MPS
            vulnerable = bool(classes & VULNERABLE[dataset])
        kind, fault = classify_contact(poses, interval_length_s, time_s, ego_geometry, corners,
                                       other_stopped=stopped, vulnerable=vulnerable, map_context=map_context)
        (at_fault if fault else exempt).append({**contact, "type": kind, "at_fault": fault})
    inserted = any(c["scope"] == "inserted" for c in at_fault)
    background = any(c["scope"] == "background" for c in at_fault)
    return {"inserted": inserted, "background": background, "all": bool(at_fault),
            "contacts": at_fault, "exempt_contacts": exempt,
            "first_contact": first_contact_speed(poses, at_fault, interval_length_s)}
