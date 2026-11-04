"""ReactTrain fine-tuning: kinematic stop targets, their artifact and SFT inputs.

Each hazardous edit's target keeps the logged GT path (the cubic path of the minimum-jerk
reconstruction, its speed profile discarded). From the measured initial speed v0 the ego brakes
at the constant deceleration a = min(v0^2 / (2 ds), a_max) from t = 0 and then holds, stopping
ds = d - m along the path. d is the arc length at which the ego footprint first meets the
inserted object's footprint at any time of the metric cache's horizon (for standing objects,
its footprint; for walkers, the area they cover), so a stopped ego is never reached by the
object. When a_max caps the deceleration the stop falls inside the margin, or the contact
happens at the lowest reachable speed. Nothing ranks or selects targets: contacts and stock
PDMS are recorded for the audit only. Original and benign samples keep their GT. Each training
epoch is every edit plus a disjoint chunk of the originals (``EpochOriginalSampler``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from .recogdrive_mixed_data import KINEMATIC_SFT_PROTOCOL, KINEMATIC_SFT_TIMESTEPS, SFT_ACTORS, check_file, digest

TARGETS_PROTOCOL = "recogdrive-kinematic-sft-targets-v1"
SAMPLER = "all-edits-plus-disjoint-original-chunks-of-seeded-pool-passes-v2"
POOL_ORDER_KEY = 1 << 32  # Seeds the passes' pool permutations apart from the per-epoch shuffles.
# (stored name, PDMResults field) of the stock PDMS subscores.
PDMS_FIELDS = (("nc", "no_at_fault_collisions"), ("dac", "drivable_area_compliance"), ("ep", "ego_progress"),
               ("ttc", "time_to_collision_within_bound"), ("comfort", "comfort"),
               ("ddc", "driving_direction_compliance"))
CONTACT_FIELDS = ("inserted", "logged_behind", "logged_other")
WAYPOINT_TIMES_S = np.arange(1, 9) * 0.5  # the eight 2 Hz poses
STATE_TIMES_S = np.arange(41) * 0.1  # 10 Hz motion over the 4 s horizon (contacts)
SPEED_BIN_EDGES_KMH = (0., 5., 10., 15., 20., 25., 30., 35., 40.)  # the last bin is open
GT_KEPT_REASONS = ("no_path", "degenerate_path", "no_gt_contact")


@dataclass(frozen=True)
class KinematicSettings:
    margin_m: float = 1.0  # The minimum gap of NAVSIM's PDM-Closed IDM (metric caching).
    max_deceleration_mps2: float = 8.0  # About 0.8 g, the dry-road friction limit.
    contact_step_m: float = 0.05  # First-contact grid along the path, then bisection.
    contact_tolerance_m: float = 1e-3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.contact_tolerance_m > self.contact_step_m:
            raise ValueError("contact_tolerance_m must not exceed contact_step_m")

    def as_dict(self):
        return {name: float(value) for name, value in asdict(self).items()}


def reconstruction():
    """The minimum-jerk reconstruction settings (waypoint merging, at-rest speed) of the GT path."""
    from navsim.evaluate.minimum_jerk_pdm_score import MinimumJerkRewardConfig
    return MinimumJerkRewardConfig().reconstruction()


def target_conventions(settings):
    """Conventions and code identity recorded in the artifact (audit only; never compared)."""
    from navsim.evaluate import minimum_jerk_trajectory, pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_scorer
    code = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (minimum_jerk_trajectory, pdm_score, pdm_scorer)}
    code[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = reconstruction()
    return {"version": 1, "path": "minimum-jerk-reconstruction-cubic-path-of-merged-gt-waypoints-v1",
            "merge_distance_m": float(config.merge_distance_m),
            "stopped_initial_speed_mps": float(config.stopped_initial_speed_mps),
            "initial_speed": "measured-longitudinal-rear-axle-v1",
            "object": "union-of-inserted-footprints-over-the-metric-cache-observation-v1",
            "first_contact": "pacifica-footprint-along-path-grid-then-bisection-v1",
            "profile": "constant-deceleration-from-t0-then-hold-v1", "heading": "path-tangent",
            "gt_kept": "gt-with-logged-heading-when-no-path-or-no-contact-v1",
            "contacts": {"target": "exact-target-motion-10hz-v1",
                         "executed": "pdm-simulator-10hz-v1",
                         "rule": "any-footprint-overlap-excluding-red-lights-and-initial-overlaps-v1"},
            "pdms": {"sampling": [40, 0.1], "execution": "pdm-closed-reference-lqr-bicycle-v1"},
            "implementation": code}


# --- Geometry ---------------------------------------------------------------------------------

def gt_path(gt_xy):
    """(path, None), or (None, reason): the cubic path the minimum-jerk reconstruction fits through
    the origin and the merged GT waypoints (backward or sub-centimetre steps merged)."""
    from navsim.evaluate.minimum_jerk_trajectory import _Path, _merge_waypoints
    points = np.vstack(([0., 0.], np.asarray(gt_xy, dtype=np.float64)))
    kept, _, _ = _merge_waypoints(points, reconstruction().merge_distance_m)
    path = _Path(kept)
    if path.spline is None:
        return None, "no_path"
    if path.degenerate:
        return None, "degenerate_path"
    return path, None


def path_poses(path, arc):
    """Local-frame (x, y, heading) at arc lengths; heading is the path tangent, and the ego continues
    straight beyond the path end."""
    arc = np.asarray(arc, dtype=np.float64)
    position, heading, _, _ = path.evaluate(np.minimum(arc, path.length))
    extra = np.maximum(arc - path.length, 0.)
    position = position + extra[:, None] * np.column_stack((np.cos(heading), np.sin(heading)))
    return np.column_stack((position, heading))


def initial_pose(metric_cache):
    rear_axle = metric_cache.ego_state.rear_axle
    return np.array([rear_axle.x, rear_axle.y, rear_axle.heading], dtype=np.float64)


def to_global(local, pose):
    """Local rear-axle poses in the initial ego frame -> global poses (as the reconstruction transforms)."""
    local = np.asarray(local, dtype=np.float64)
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    position = local[:, :2] @ np.array([[cos, -sin], [sin, cos]]).T + pose[:2]
    return np.column_stack((position, local[:, 2] + pose[2]))


def to_local(points, pose):
    """Global XY -> the initial ego frame (inverse of ``to_global`` for positions)."""
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    return (np.asarray(points, dtype=np.float64) - pose[:2]) @ np.array([[cos, -sin], [sin, cos]])


def state_array(global_poses):
    """PDM state arrays (only the SE2 columns are used) for ego footprints and contact checks."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    poses = np.asarray(global_poses, dtype=np.float64)
    states = np.zeros((len(poses), StateIndex.size()), dtype=np.float64)
    states[:, StateIndex.STATE_SE2] = poses
    return states


def ego_footprints(global_poses):
    """Pacifica footprints (shapely polygons) of rear-axle poses, as the PDM scorer builds them."""
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        coords_array_to_polygon_array, state_array_to_coords_array)
    states = state_array(global_poses)[None]
    return coords_array_to_polygon_array(state_array_to_coords_array(states, get_pacifica_parameters()))[0]


def observation_frames(observation):
    """Every occupancy map of a metric cache's observation (10 Hz, from 0 s to its horizon)."""
    return [observation[index] for index in range(len(observation._global_to_local_idcs))]


def object_region(observation, token):
    """Union of the object's footprints over the observation, or None if it is never present."""
    from shapely.ops import unary_union
    footprints = [frame[token] for frame in observation_frames(observation) if token in frame.tokens]
    return unary_union(footprints) if footprints else None


def first_contact_arc(path, pose, region, settings):
    """Smallest arc length (to ``contact_tolerance_m``) at which the ego footprint on the path meets
    ``region``; 0 if it already does at the start, None if it never does along the path."""
    import shapely
    if region is None or region.is_empty:
        return None
    count = int(math.ceil(path.length / settings.contact_step_m)) + 1
    grid = np.linspace(0., path.length, count)
    hits = shapely.intersects(ego_footprints(to_global(path_poses(path, grid), pose)), region)
    if not hits.any():
        return None
    first = int(np.argmax(hits))
    if first == 0:
        return 0.
    low, high = float(grid[first - 1]), float(grid[first])
    while high - low > settings.contact_tolerance_m:
        middle = (low + high) / 2
        footprint = ego_footprints(to_global(path_poses(path, np.array([middle])), pose))[0]
        if shapely.intersects(footprint, region):
            high = middle
        else:
            low = middle
    return high


# --- Speed profile ------------------------------------------------------------------------------

def stop_profile(v0, delta_s, settings):
    """Constant deceleration from t = 0 that stops after ``delta_s`` (capped at a_max), then hold.

    An ego at rest (measured speed at or below the reconstruction's threshold) stays put.
    ``a_required`` is None when the stop point is at or behind the start (no deceleration suffices).
    """
    if not math.isfinite(v0) or not math.isfinite(delta_s):
        raise ValueError("The stop profile needs a finite initial speed and distance")
    if v0 <= reconstruction().stopped_initial_speed_mps:
        return {"a": 0., "a_required": 0., "capped": False, "t_stop": 0., "stop_arc": 0.}
    required = v0 * v0 / (2 * delta_s) if delta_s > 0 else math.inf
    a = min(required, settings.max_deceleration_mps2)
    return {"a": float(a), "a_required": None if math.isinf(required) else float(required),
            "capped": bool(required > settings.max_deceleration_mps2), "t_stop": float(v0 / a),
            "stop_arc": float(v0 * v0 / (2 * a))}


def profile_arc(v0, profile, times):
    """Distance travelled along the path at ``times``: v0 t - a t^2 / 2 until the stop, then held."""
    t = np.minimum(np.asarray(times, dtype=np.float64), profile["t_stop"])
    return v0 * t - 0.5 * profile["a"] * t * t if profile["a"] > 0 else np.zeros_like(t)


# --- Inputs and stock PDMS ------------------------------------------------------------------------

def sample_file(sample, name):
    """Frozen stamp of one prepared file: ``internvl_feature.gz``, ``trajectory_target.gz`` or ``metric``."""
    matches = [r for r in sample["files"]
               if (r["path"] == sample["metric_path"] if name == "metric" else Path(r["path"]).name == name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one prepared {name} file for {sample['feature_path']}")
    return matches[0]


def load_gt(reference):
    """Logged GT poses (x, y, heading) at 0.5-4.0 s from a stamped ``trajectory_target.gz``."""
    check_file(reference)
    with gzip.open(reference["path"], "rb") as handle:
        gt = np.asarray(pickle.load(handle)["trajectory"], dtype=np.float64)
    if gt.shape != (8, 3) or not np.isfinite(gt).all():
        raise ValueError(f"Invalid GT trajectory: {reference['path']}")
    return gt


def load_metric_cache(reference):
    check_file(reference)
    with lzma.open(reference["path"], "rb") as handle:
        return pickle.load(handle)


def stock_pdms():
    """Unmodified NAVSIM PDMS: 4 s at 10 Hz, PDM-Closed reference, LQR + bicycle execution, weights 5/5/2."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    return sampling, PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())


def pdms_result(metric_cache, poses, stock=None):
    """Stock executed PDMS of one trajectory scored on its own (``pdm_score``, PDM-Closed reference)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    sampling, simulator, scorer = stock_pdms() if stock is None else stock
    result = pdm_score(metric_cache, Trajectory(np.asarray(poses, dtype=np.float64)), sampling, simulator, scorer)
    values = {"score": float(result.score), **{name: float(getattr(result, field)) for name, field in PDMS_FIELDS}}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite stock PDMS or subscore")
    return values


def executed_states(metric_cache, poses, stock=None):
    """The trajectory's LQR + bicycle states exactly as ``pdm_score`` simulates them (beside the reference)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    sampling, simulator, _ = stock_pdms() if stock is None else stock
    initial = metric_cache.ego_state
    proposals = np.stack([
        get_trajectory_as_array(metric_cache.trajectory, sampling, initial.time_point),
        get_trajectory_as_array(transform_trajectory(Trajectory(np.asarray(poses, dtype=np.float64)), initial),
                                sampling, initial.time_point)])
    return simulator.simulate_proposals(proposals, initial)[1]


def track_contacts(states, observation, inserted_token):
    """Which tracks the ego footprint touches at 0-4 s, any fault: the inserted actor, and logged actors
    behind the ego (centroid within 30 deg of straight behind the rear axle) or elsewhere. Red lights and
    tracks already overlapping the ego at 0 s (stock initial collisions) are skipped. None without states."""
    if states is None:
        return None
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.observation.idm.utils import is_agent_behind
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        coords_array_to_polygon_array, state_array_to_coords_array)
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    polygons = coords_array_to_polygon_array(state_array_to_coords_array(states[None], get_pacifica_parameters()))[0]
    result = dict.fromkeys(CONTACT_FIELDS, False)
    seen = set(observation.collided_track_ids)
    for time_idx, polygon in enumerate(polygons):
        occupancy = observation[time_idx]
        for token in occupancy.intersects(polygon):
            if observation.red_light_token in token or token in seen:
                continue
            seen.add(token)
            if token == inserted_token:
                result["inserted"] = True
                continue
            centroid = occupancy[token].centroid
            behind = is_agent_behind(StateSE2(*states[time_idx, StateIndex.STATE_SE2]),
                                     StateSE2(centroid.x, centroid.y, 0.))
            result["logged_behind" if behind else "logged_other"] = True
    return result


# --- Targets -----------------------------------------------------------------------------------

def kinematic_target(metric_cache, gt, inserted_token, settings=None):
    """(target poses [8, 3], record) for one hazardous edit; never mutates the metric cache.

    The GT is kept (logged heading) when it has no usable path or never touches the object.
    """
    settings = KinematicSettings() if settings is None else settings
    gt = np.asarray(gt, dtype=np.float64)
    if gt.shape != (8, 3) or not np.isfinite(gt).all():
        raise ValueError("Kinematic targets require a finite [8, 3] GT")
    observation = metric_cache.observation
    if inserted_token not in observation.unique_objects:
        raise ValueError(f"The metric cache does not track the inserted object {inserted_token}")
    pose = initial_pose(metric_cache)
    v0 = float(metric_cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    record = {"v0": v0, "d": None, "delta_s": None, "a": None, "a_required": None, "capped": None,
              "t_stop": None, "stop_arc": None, "path_length": None, "travel_m": None, "executed_travel_m": None}
    contacts = {"target": None, "target_executed": None}
    path, reason = gt_path(gt[:, :2])
    if path is not None:
        record["path_length"] = float(path.length)
        d = first_contact_arc(path, pose, object_region(observation, inserted_token), settings)
        if d is None:
            reason = "no_gt_contact"
    if reason is not None:
        return gt, {"origin": "gt", "gt_kept_reason": reason, **record, "contacts": contacts, "pdms": None}
    profile = stop_profile(v0, d - settings.margin_m, settings)
    record.update(d=float(d), delta_s=float(d - settings.margin_m), **profile)
    target = path_poses(path, profile_arc(v0, profile, WAYPOINT_TIMES_S))
    states = state_array(to_global(path_poses(path, profile_arc(v0, profile, STATE_TIMES_S)), pose))
    stock = stock_pdms()
    contacts["target"] = track_contacts(states, observation, inserted_token)
    executed = executed_states(metric_cache, target, stock)
    contacts["target_executed"] = track_contacts(executed, observation, inserted_token)
    # Distance travelled in 4 s by the target and by its LQR + bicycle execution (audit of the tracking lag).
    record.update(travel_m=float(profile_arc(v0, profile, [STATE_TIMES_S[-1]])[0]),
                  executed_travel_m=travel_distance(executed))
    return target, {"origin": "kinematic", "gt_kept_reason": None, **record, "contacts": contacts,
                    "pdms": pdms_result(metric_cache, target, stock)}


def travel_distance(states):
    """Path length of a PDM state array's rear-axle positions."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    points = np.asarray(states, dtype=np.float64)[:, StateIndex.POINT]
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


# --- Audit ------------------------------------------------------------------------------------

def _quantiles(values):
    return [float(q) for q in np.percentile(values, (10, 50, 90))] if values else None


def _mean(values):
    return float(np.mean(values)) if values else None


def speed_bin(v0):
    kmh = 3.6 * v0
    for low, high in zip(SPEED_BIN_EDGES_KMH, SPEED_BIN_EDGES_KMH[1:]):
        if low <= kmh < high:
            return f"[{low:g},{high:g})"
    return f"[{SPEED_BIN_EDGES_KMH[-1]:g},inf)" if kmh >= SPEED_BIN_EDGES_KMH[-1] else "negative"


def _group_summary(rows):
    targets = [r for r in rows if r["origin"] == "kinematic"]
    count = {f"{who}_{field}": sum(bool((r["contacts"][who] or {}).get(field)) for r in rows)
             for who in ("target", "target_executed") for field in CONTACT_FIELDS}
    pdms = [r["pdms"] for r in targets if r["pdms"] is not None]
    return {"samples": len(rows), "kinematic_targets": len(targets),
            "gt_kept": {reason: sum(r["gt_kept_reason"] == reason for r in rows) for reason in GT_KEPT_REASONS},
            "capped": sum(bool(r["capped"]) for r in targets),
            "stops_within_horizon": sum(r["t_stop"] <= 4. for r in targets),
            "deceleration_mps2_p10_p50_p90": _quantiles([r["a"] for r in targets]),
            "stop_time_s_p10_p50_p90": _quantiles([r["t_stop"] for r in targets]),
            "margin_m_p10_p50_p90": _quantiles([r["d"] - r["stop_arc"] for r in targets]),
            "gap_d_m_p10_p50_p90": _quantiles([r["d"] for r in targets]),
            "execution_overshoot_m_p10_p50_p90": _quantiles([r["executed_travel_m"] - r["travel_m"] for r in targets]),
            "contacts": count,
            "pdms": {name: _mean([p[name] for p in pdms]) for name in ("score", "nc", "dac", "ep", "ttc")}}


def target_summary(samples):
    """Audit overall, per actor/timestep variant and per initial-speed bin."""
    groups = {}
    for record in samples.values():
        for key in ("all", f"variant/{record['variant']}", f"speed_kmh/{speed_bin(record['v0'])}"):
            groups.setdefault(key, []).append(record)
    return {key: _group_summary(rows) for key, rows in sorted(groups.items())}


# --- Training inputs --------------------------------------------------------------------------

def hazardous_edits(manifest):
    """Every hazardous edit ID (t1-t4) of a kinematic SFT view's sources, selected for training or not."""
    return {f"{source}__hazardous_{actor}_t{step}" for source in manifest["original_sources"]
            for actor in SFT_ACTORS for step in KINEMATIC_SFT_TIMESTEPS}


def load_targets(manifest):
    """Verified kinematic targets (hazardous edit ID -> float64 [8, 3]) of a kinematic SFT view."""
    if manifest.get("protocol") != KINEMATIC_SFT_PROTOCOL:
        raise ValueError("Kinematic targets require a kinematic SFT manifest")
    reference = manifest["kinematic_targets"]
    stamps = [r for r in manifest["sources"] if r.get("path") == reference["path"] and "sha256" in r]
    if len(stamps) != 1:
        raise ValueError("Kinematic SFT manifest must sign its target artifact exactly once")
    data = Path(reference["path"]).read_bytes()
    if len(data) != stamps[0]["size"] or hashlib.sha256(data).hexdigest() != stamps[0]["sha256"]:
        raise ValueError(f"Kinematic target artifact changed: {reference['path']}")
    value = json.loads(data)
    identity = value.pop("artifact_sha256", None)
    if (identity != digest(value) or identity != reference["artifact_sha256"]
            or value.get("protocol") != TARGETS_PROTOCOL or value.get("settings") != reference["settings"]
            or value.get("complete") is not True):
        raise ValueError("Kinematic target artifact checksum/protocol/settings/completeness mismatch")
    if (value["mixed_manifest"] != manifest["parent_identity"]
            or value["extra_edited_manifest"] != manifest["extra_edited_manifest_identity"]):
        raise ValueError("Kinematic targets were built for different edited caches")
    if set(value["samples"]) != hazardous_edits(manifest):
        raise ValueError("Kinematic targets must cover exactly every hazardous edit of the view's sources")
    targets = {}
    for token, record in value["samples"].items():
        target = np.asarray(record["target"], dtype=np.float64)
        if (target.shape != (8, 3) or not np.isfinite(target).all()
                or record["source_token"] != token.split("__", 1)[0]):
            raise ValueError(f"Invalid kinematic target record: {token}")
        targets[token] = target
    return targets


class HazardTargetDataset(Dataset):
    """Training samples in sorted token order; hazardous targets replaced by the frozen SFT targets.

    The sorted order makes sampler indices independent of directory listing order on each rank.
    Original and benign samples keep their cached GT; replacements keep the GT tensor dtype.
    """

    def __init__(self, base, kinds, targets):
        self.base = base
        self.tokens = sorted(base.tokens)
        self._valid_cache_paths = base._valid_cache_paths
        self._base_index = {token: index for index, token in enumerate(base.tokens)}
        if len(self._base_index) != len(self.tokens):
            raise ValueError("Duplicate training tokens")
        self.kinds = {token: kinds[token] for token in self.tokens}
        hazardous = [token for token in self.tokens if self.kinds[token] == "hazardous"]
        missing = [token for token in hazardous if token not in targets]
        if missing:
            raise ValueError(f"Hazardous training samples without SFT targets: {missing[:5]}")
        self.targets = {token: np.array(targets[token], dtype=np.float64) for token in hazardous}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        features, targets, loaded = self.base[self._base_index[token]]
        if loaded != token:
            raise RuntimeError(f"Cached sample identity mismatch: {loaded} != {token}")
        if token in self.targets:
            gt = targets["trajectory"]
            if tuple(gt.shape) != (8, 3):
                raise ValueError(f"Unexpected cached GT shape for {token}: {tuple(gt.shape)}")
            targets = {**targets, "trajectory": torch.tensor(self.targets[token], dtype=gt.dtype)}
        return features, targets, token


class EpochOriginalSampler(DistributedSampler):
    """Every edited sample plus ``originals_per_epoch`` originals, disjoint across epochs.

    The view's originals (the pool) are permuted once per pass, seeded by (seed, pass), and each
    pass is cut into ``len(pool) // originals_per_epoch`` disjoint chunks, one per epoch; later
    epochs start a new pass. A pool of exactly k x originals_per_epoch therefore uses every
    original once over k epochs. The epoch's index list depends only on (seed, epoch), is
    identical on every rank, and is shuffled and then sharded like ``DistributedSampler`` (tail
    padded by repetition). Being a DistributedSampler, Lightning keeps it and calls
    ``set_epoch`` before each epoch.
    """

    def __init__(self, dataset, originals_per_epoch, *, seed, num_replicas, rank):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed, drop_last=False)
        kinds = [dataset.kinds[token] for token in dataset.tokens]
        self.edited = np.flatnonzero([kind != "original" for kind in kinds])
        self.originals = np.flatnonzero([kind == "original" for kind in kinds])
        if (type(originals_per_epoch) is not int or not 0 < originals_per_epoch <= len(self.originals)
                or not len(self.edited)):
            raise ValueError("originals_per_epoch must be a positive integer up to the number of originals, "
                             "with edited samples present")
        self.originals_per_epoch = originals_per_epoch
        self.disjoint_epochs = len(self.originals) // originals_per_epoch  # Epochs per pass over the pool.
        self.epoch_size = len(self.edited) + originals_per_epoch
        self.num_samples = math.ceil(self.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def epoch_originals(self, epoch):
        """The dataset indices of one epoch's originals: one chunk of its pass's pool permutation."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        epoch_pass, chunk = divmod(epoch, self.disjoint_epochs)
        order = np.random.default_rng([int(self.seed), POOL_ORDER_KEY, epoch_pass]).permutation(self.originals)
        return order[chunk * self.originals_per_epoch:(chunk + 1) * self.originals_per_epoch]

    def epoch_indices(self, epoch=None):
        """The full (unsharded, padded) index list of one epoch."""
        epoch = self.epoch if epoch is None else epoch
        chosen = self.epoch_originals(int(epoch))
        rng = np.random.default_rng([int(self.seed), int(epoch)])
        indices = np.concatenate((self.edited, chosen))
        rng.shuffle(indices)
        indices = indices.tolist()
        padding = self.total_size - len(indices)
        while padding > 0:
            indices += indices[:padding]
            padding = self.total_size - len(indices)
        return indices

    def __iter__(self):
        return iter(self.epoch_indices()[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples




def sampler_protocol(manifest, originals_per_epoch, seed):
    """Recorded in the mixed-training protocol, so a resume cannot change the population or draw."""
    edits = manifest["counts"]["hazardous"] + manifest["counts"]["benign"]
    originals = manifest["counts"]["original"]
    return {"version": 2, "sampler": SAMPLER, "seed": seed,
            "originals_per_epoch": originals_per_epoch, "epoch_size": edits + originals_per_epoch,
            "originals": originals, "disjoint_epochs": originals // originals_per_epoch,
            "original_pool": manifest.get("original_pool"), "variant_selection": manifest["variant_selection"],
            "targets": manifest["kinematic_targets"]["artifact_sha256"], "target_rule": TARGETS_PROTOCOL,
            "target_settings": manifest["kinematic_targets"]["settings"],
            "loss": "stage2-epsilon-mse-per-sample-mean-v1"}


def training_inputs(manifest, base, originals_per_epoch, *, seed, num_replicas, rank):
    """(dataset, sampler, sampler record) of a kinematic SFT view's training split ``base``.

    The dataset supplies the frozen hazardous targets. ``originals_per_epoch`` None selects the
    number of hazardous edits.
    """
    if originals_per_epoch is None:
        originals_per_epoch = manifest["counts"]["hazardous"]
    data = HazardTargetDataset(
        base, {t: s["kind"] for t, s in manifest["samples"].items() if s["split"] == "train"}, load_targets(manifest))
    sampler = EpochOriginalSampler(data, originals_per_epoch, seed=seed, num_replicas=num_replicas, rank=rank)
    return data, sampler, sampler_protocol(manifest, originals_per_epoch, seed)
