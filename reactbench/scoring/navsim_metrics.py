"""ReactNAV scoring of one native prediction: at-fault NC, NCins and DAC (protocol V2)."""
from __future__ import annotations
import math
import numpy as np
from .collisions import _sample_ego, score_raw_trajectory
from .at_fault import AT_FAULT_CONTRACT
from .navsim_lanes import LANES_CONTRACT

SCORING_CONTRACT = {"version": 1, "horizon_s": 4., "geometry_interval_s": .01, "dac_interval_s": .1,
                    "interpolation": "origin + native xy and unwrapped heading, piecewise linear",
                    "contact_speed": "incoming native xy segment derivative at own first contact; t=0 unavailable",
                    "primary": "binary nc_all * dac per prediction", "controller": None}


def raw_dac(poses, interval_length_s, world):
    if world.get("map_status") != "available":
        return {"dac": None, "dac_status": "unavailable", "dac_reason": world.get("map_error", "Missing map geometry")}
    from shapely import wkb
    from shapely.vectorized import contains
    times = np.arange(41) * .1
    corners = _sample_ego(poses, times, world["geometry"], interval_length_s)
    theta = world["origin"][2]
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    points = corners @ rotation.T + np.asarray(world["origin"][:2])
    inside = np.zeros(points.shape[:-1], dtype=bool)
    for value in world["map_polygons_wkb"]:
        polygon = wkb.loads(value, hex=True)
        inside |= contains(polygon, points[..., 0], points[..., 1])
    failures = np.flatnonzero(~inside.all(axis=1))
    return {"dac": int(not len(failures)), "dac_status": "available",
            "first_non_drivable_time_s": float(times[failures[0]]) if len(failures) else None}


AT_FAULT_SCORING_CONTRACT = {
    **{k: v for k, v in SCORING_CONTRACT.items() if k not in ("version", "contact_speed", "primary")},
    "version": 2, "collision": AT_FAULT_CONTRACT, "lanes": LANES_CONTRACT,
    "contact_speed": "incoming native xy segment derivative at the first at-fault contact; t=0 unavailable",
    "primary": "binary nc * dac per prediction (nc: no at-fault collision)"}


class MapContext:
    """NAVSIM map geometry for the lateral rule, parsed only when a lateral contact needs it."""

    def __init__(self, token, world, work_root):
        self.token, self.world, self.work_root = token, world, work_root
        self._drivable = self._lanes = None

    def _require_map(self):
        # Only predictions with a lateral contact need the map; they fail rather than silently pass.
        if self.world.get("map_status") != "available":
            raise ValueError(f"Map unavailable for the at-fault lateral rule: {self.world.get('map_error')}")

    @property
    def origin(self):
        self._require_map()
        return self.world["origin"]

    @property
    def drivable(self):
        self._require_map()
        if self._drivable is None:
            from shapely import wkb
            self._drivable = [wkb.loads(v, hex=True) for v in self.world["map_polygons_wkb"]]
        return self._drivable

    @property
    def lanes(self):
        self._require_map()
        if self._lanes is None:
            from .navsim_lanes import load_lanes
            self._lanes = load_lanes(self.token, self.work_root, origin=self.origin)["polygons"]
        return self._lanes

    def __getitem__(self, key):  # classify_contact reads origin/drivable/lanes by key
        return getattr(self, key)


def at_fault_from_physical(physical, poses, interval_length_s, actor_motion, world, map_context):
    """V2 score from a physical (V1-geometry) collision result plus its DAC."""
    from .at_fault import classify_score
    collision = classify_score(physical, poses, interval_length_s, actor_motion=actor_motion,
                               tracks=world["tracks"], ego_geometry=world["geometry"], dataset="navsim",
                               map_context=map_context)
    dac = {k: physical[k] for k in ("dac", "dac_status", "dac_reason", "first_non_drivable_time_s") if k in physical}
    if "dac" not in dac:
        dac = raw_dac(poses, interval_length_s, world)
    return {**collision, **dac, "nc": int(not collision["all"]), "nc_inserted": int(not collision["inserted"]),
            "nc_dac": int(not collision["all"]) * dac["dac"] if dac["dac"] is not None else None,
            "interval_length_s": interval_length_s}


def score_prediction_at_fault(poses, interval_length_s, actor_motion, world, map_context):
    """V2 scalar reference: physical contacts as V1, then at-fault classification."""
    physical = score_raw_trajectory(poses, actor_motion, world["geometry"], world,
                                    interval_length_s=interval_length_s)
    return at_fault_from_physical(physical, poses, interval_length_s, actor_motion, world, map_context)
