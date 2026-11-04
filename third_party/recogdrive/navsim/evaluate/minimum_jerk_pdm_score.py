"""Training-only minimum-jerk reward on reconstructed candidate motion.

Stock PDM NC/DAC geometry is evaluated on the path + progress reconstruction
(see minimum_jerk_trajectory). NC adds the pedestrian/cyclist rule; TTC is a
graded stock-style check on logged actor positions; EP is smooth route progress
relative to the clean GT; comfort uses exact derivatives. Contact speed, a
universal speed limit, physical feasibility and backward waypoints (merged by the
reconstruction, charged by their largest distance) then shape the reward:

    R0 = NC * DAC * (w_ttc TTC + w_ep EP + w_c Comfort) / W - w_contact P_contact
    R  = (1 - gamma) R0 - gamma w_contact,  gamma = max(alpha_speed, beta_feasibility, beta_backward)
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import math
from typing import List, Optional, Sequence

import numpy as np
from shapely import Point, creation

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.evaluate.minimum_jerk_trajectory import (
    NUM_KNOTS, NUM_WAYPOINTS, WAYPOINT_INTERVAL_S, ReconstructedMotion, ReconstructionConfig, reconstruct_motion,
)
from navsim.evaluate.raw_pdm_score import validate_raw_observation
from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as comfort_bounds
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex, EgoAreaIndex, MultiMetricIndex, StateIndex, WeightedMetricIndex,
)

SAMPLING = TrajectorySampling(num_poses=NUM_KNOTS - 1, interval_length=0.1)
VULNERABLE_ROAD_USERS = frozenset({TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE})
_ZERO_ACCELERATION = 1e-9


@dataclass(frozen=True)
class MinimumJerkRewardConfig:
    contact_weight: float = 1.0
    contact_speed_cap_mps: float = 50 / 3.6
    ttc_horizon_s: float = 0.9
    ep_epsilon_m: float = 1.0
    merge_distance_m: float = 0.05
    stopped_initial_speed_mps: float = 0.05
    low_speed_threshold_mps: float = 0.5  # yaw comfort checks and curvature feasibility apply above it
    speed_limit_mps: float = 80 / 3.6  # universal; map limits were implausibly low for many scenes
    speed_limit_ramp: float = 0.5  # alpha reaches 1 at (1 + ramp) times the limit
    total_acceleration_limit_mps2: float = 9.0
    forward_acceleration_limit_mps2: float = 4.0
    curvature_limit_per_m: float = 0.2
    feasibility_ramp: float = 0.2  # beta reaches 1 at (1 + ramp) times a limit
    backward_saturation_m: float = 0.5  # beta_backward rises linearly from 0 and reaches 1 at this distance

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"training_minimum_jerk.{item.name} must be a finite number")
            if value < 0 or (value == 0 and item.name != "contact_weight"):
                raise ValueError(f"training_minimum_jerk.{item.name} must be positive")
        steps = self.ttc_horizon_s / SAMPLING.interval_length
        # Cached observations end at 5.0 s: t <= 4.0 s plus the horizon must stay within 4.9 s.
        if abs(steps - round(steps)) > 1e-9 or not 1 <= round(steps) <= 9:
            raise ValueError("training_minimum_jerk.ttc_horizon_s must be a multiple of 0.1 s in [0.1, 0.9]")

    @classmethod
    def from_mapping(cls, values) -> "MinimumJerkRewardConfig":
        values = dict(values or {})
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training_minimum_jerk settings: {sorted(unknown)}")
        return cls(**values)

    def reconstruction(self) -> ReconstructionConfig:
        return ReconstructionConfig(self.merge_distance_m, self.stopped_initial_speed_mps)

    def as_dict(self) -> dict:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class MinimumJerkResult:
    score: float
    # PDM-named components; zero (fail-all) for an infeasible fit.
    no_at_fault_collisions: float
    drivable_area_compliance: float
    time_to_collision_within_bound: float
    ego_progress: float
    comfort: float
    base_reward: float
    contact_penalty: float
    contact_speed: Optional[float]  # first future at-fault contact only
    initial_contact: bool
    vru_rule_contact: bool
    max_speed_ratio: float
    alpha: float
    feasibility_ratio: float
    beta: float
    total_acceleration_violation: bool
    forward_acceleration_violation: bool
    curvature_violation: bool
    infeasible_fit: bool
    infeasible_reason: Optional[str]
    qp_fallback: bool
    backward_distance_m: float = 0.  # reported even when not charged
    beta_backward: float = 0.


class MinimumJerkScorer(PDMScorer):
    """Stock ego areas/DAC/progress; NC with the pedestrian/cyclist rule; graded TTC."""

    def __init__(self, config: PDMScorerConfig, ttc_horizon_s: float):
        super().__init__(SAMPLING, config)
        self._ttc_steps = int(round(ttc_horizon_s / SAMPLING.interval_length))

    def score_minimum_jerk(self, states, observation, centerline, route_lane_ids, drivable_area_map):
        self._reset(states, observation, centerline, route_lane_ids, drivable_area_map)
        self._calculate_ego_area()
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()
        self._calculate_progress()
        self._calculate_ttc()
        return {"nc": self._multi_metrics[MultiMetricIndex.NO_COLLISION].copy(),
                "dac": self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA].copy(),
                "ttc": self._weighted_metrics[WeightedMetricIndex.TTC].copy(),
                "progress": self._progress_raw.copy(),
                "first_future_contact": self._first_future_contact.copy(),
                "initial_contact": self._initial_contact.copy(),
                "vru_rule_contact": self._vru_rule_contact.copy()}

    def _calculate_no_at_fault_collision(self) -> None:
        """Stock at-fault rule; any contact with a pedestrian/cyclist while moving is at fault.

        Stopped-ego contacts stay not at fault, and the per-track ignore list is
        never extended with pedestrians/cyclists. Tracks first contacted at t=0
        are initial contacts and never trigger the contact-speed penalty.
        """
        n = self._num_proposals
        scores = np.ones(n, dtype=np.float64)
        ignored = {p: copy.deepcopy(self._observation.collided_track_ids) for p in range(n)}
        initial_tokens = [set() for _ in range(n)]
        self._first_future_contact = np.full(n, -1, dtype=np.int64)
        self._initial_contact = np.zeros(n, dtype=bool)
        self._vru_rule_contact = np.zeros(n, dtype=bool)
        for time_idx in range(self.proposal_sampling.num_poses + 1):
            intersecting = self._observation[time_idx].query(self._ego_polygons[:, time_idx], predicate="intersects")
            if len(intersecting) == 0:
                continue
            for p, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx]
                if self._observation.red_light_token in token or token in ignored[p]:
                    continue
                multiple_or_nondrivable = (self._ego_areas[p, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                                           or self._ego_areas[p, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA])
                tracked_object = self._observation.unique_objects[token]
                collision_type = get_collision_type(self._states[p, time_idx], self._ego_polygons[p, time_idx],
                                                    tracked_object, self._observation[time_idx][token])
                stock_at_fault = (collision_type in (CollisionType.ACTIVE_FRONT_COLLISION,
                                                     CollisionType.STOPPED_TRACK_COLLISION)
                                  or (multiple_or_nondrivable and collision_type == CollisionType.ACTIVE_LATERAL_COLLISION))
                vulnerable = tracked_object.tracked_object_type in VULNERABLE_ROAD_USERS
                if stock_at_fault or (vulnerable and collision_type != CollisionType.STOPPED_EGO_COLLISION):
                    value = 0.0 if tracked_object.tracked_object_type in AGENT_TYPES else 0.5
                    scores[p] = min(scores[p], value)
                    self._collision_time_idcs[p] = min(time_idx, self._collision_time_idcs[p])
                    self._vru_rule_contact[p] |= not stock_at_fault
                    if time_idx == 0:
                        initial_tokens[p].add(token)
                        self._initial_contact[p] = True
                    elif token not in initial_tokens[p] and self._first_future_contact[p] < 0:
                        self._first_future_contact[p] = time_idx
                elif not vulnerable:
                    ignored[p].append(token)
        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = scores

    def _calculate_ttc(self) -> None:
        """Smallest stock-style projection time to a counted hit, over future samples, as a fraction of the horizon."""
        n, interval, steps = self._num_proposals, self.proposal_sampling.interval_length, self._ttc_steps
        horizon = steps * interval
        tau = np.full(n, horizon, dtype=np.float64)
        ignored = {p: copy.deepcopy(self._observation.collided_track_ids) for p in range(n)}
        offsets = np.arange(steps + 1)
        coords = self._ego_coords.copy()
        coords[:, :, BBCoordsIndex.CENTER, :] = coords[:, :, BBCoordsIndex.FRONT_LEFT, :]
        speeds = np.hypot(self._states[..., StateIndex.VELOCITY_X], self._states[..., StateIndex.VELOCITY_Y])
        velocity = np.stack([np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                             np.sin(self._states[..., StateIndex.HEADING]) * speeds], axis=-1)
        projected = (coords[:, :, None] + velocity[:, :, None, None, :]
                     * (offsets * interval)[None, None, :, None, None])
        polygons = creation.polygons(projected)
        for time_idx in range(1, self.proposal_sampling.num_poses + 1):
            for step in offsets:
                current = time_idx + int(step)
                intersecting = self._observation[current].query(polygons[:, time_idx, step], predicate="intersects")
                if len(intersecting) == 0:
                    continue
                for p, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current].tokens[geometry_idx]
                    if (self._observation.red_light_token in token or token in ignored[p]
                            or speeds[p, time_idx] < self._config.stopped_speed_threshold):
                        continue
                    multiple_or_nondrivable = (self._ego_areas[p, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                                               or self._ego_areas[p, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA])
                    ego_rear_axle = StateSE2(*self._states[p, time_idx, StateIndex.STATE_SE2])
                    centroid = self._observation[current][token].centroid
                    track_state = StateSE2(centroid.x, centroid.y,
                                           self._observation.unique_objects[token].box.center.heading)
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (multiple_or_nondrivable or self._drivable_area_map.is_in_layer(
                            ego_rear_axle.point, layer=SemanticMapLayer.INTERSECTION))
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        tau[p] = min(tau[p], step * interval)
                        self._ttc_time_idcs[p] = min(time_idx, self._ttc_time_idcs[p])
                    else:
                        ignored[p].append(token)
        self._weighted_metrics[WeightedMetricIndex.TTC] = tau / horizon


def motion_states(motion: ReconstructedMotion) -> np.ndarray:
    """PDM state array; velocity/acceleration are body-frame, heading wrapped."""
    states = np.zeros((NUM_KNOTS, StateIndex.size()), dtype=np.float64)
    states[:, StateIndex.X] = motion.rear_axle_xy[:, 0]
    states[:, StateIndex.Y] = motion.rear_axle_xy[:, 1]
    states[:, StateIndex.HEADING] = (motion.heading + np.pi) % (2 * np.pi) - np.pi
    states[:, StateIndex.VELOCITY_X] = motion.speed
    states[:, StateIndex.ACCELERATION_X] = motion.accel_lon
    states[:, StateIndex.ACCELERATION_Y] = motion.accel_lat
    states[:, StateIndex.ANGULAR_VELOCITY] = motion.yaw_rate
    states[:, StateIndex.ANGULAR_ACCELERATION] = motion.yaw_accel
    return states


def is_comfortable(motion: ReconstructedMotion, low_speed_threshold: float) -> bool:
    """PDMS quantities/strict bounds from exact derivatives; yaw checks skipped at low speed."""
    accel = np.hypot(motion.accel_lon, motion.accel_lat)
    accel_rate = np.hypot(motion.jerk_lon, motion.jerk_lat)  # one-sided limit at zero acceleration
    magnitude_jerk = np.divide(motion.accel_lon * motion.jerk_lon + motion.accel_lat * motion.jerk_lat, accel,
                               out=accel_rate.copy(), where=accel > _ZERO_ACCELERATION)
    moving = motion.speed >= low_speed_threshold
    return bool(np.all((motion.accel_lon > comfort_bounds.min_lon_accel) & (motion.accel_lon < comfort_bounds.max_lon_accel))
                and np.all(np.abs(motion.accel_lat) < comfort_bounds.max_abs_lat_accel)
                and np.all(np.abs(magnitude_jerk) < comfort_bounds.max_abs_mag_jerk)
                and np.all(np.abs(motion.jerk_lon) < comfort_bounds.max_abs_lon_jerk)
                and np.all(np.abs(motion.yaw_rate[moving]) < comfort_bounds.max_abs_yaw_rate)
                and np.all(np.abs(motion.yaw_accel[moving]) < comfort_bounds.max_abs_yaw_accel))


def feasibility(motion: ReconstructedMotion, config: MinimumJerkRewardConfig):
    """Worst value/limit ratio over future samples, and which limits are exceeded."""
    future = slice(1, None)
    total = np.hypot(motion.accel_lon[future], motion.accel_lat[future]) / config.total_acceleration_limit_mps2
    forward = np.maximum(motion.accel_lon[future], 0.) / config.forward_acceleration_limit_mps2
    moving = motion.speed[future] >= config.low_speed_threshold_mps
    curvature = np.where(moving, np.abs(motion.curvature[future]), 0.) / config.curvature_limit_per_m
    ratios = [float(total.max()), float(forward.max()), float(curvature.max())]
    return max(ratios), [ratio > 1 for ratio in ratios]


def ramp(ratio: float, width: float) -> float:
    return float(np.clip((ratio - 1) / width, 0., 1.))


def gt_route_progress(metric_cache, gt_poses: np.ndarray, rear_axle_to_center: float) -> float:
    """Route progress of the recorded clean GT between t=0 and t=4 s (vehicle centers)."""
    gt = np.asarray(gt_poses, dtype=np.float64)
    if gt.shape != (NUM_WAYPOINTS, 3) or not np.isfinite(gt).all():
        raise ValueError("Minimum-jerk EP requires the clean GT as eight finite (x, y, heading) poses")
    rear = metric_cache.ego_state.rear_axle
    cos, sin = math.cos(rear.heading), math.sin(rear.heading)
    end_heading = rear.heading + gt[-1, 2]
    end = (rear.x + cos * gt[-1, 0] - sin * gt[-1, 1] + rear_axle_to_center * math.cos(end_heading),
           rear.y + sin * gt[-1, 0] + cos * gt[-1, 1] + rear_axle_to_center * math.sin(end_heading))
    start = (rear.x + rear_axle_to_center * cos, rear.y + rear_axle_to_center * sin)
    progress = metric_cache.centerline.project([Point(*start), Point(*end)])
    value = max(float(progress[1] - progress[0]), 0.)
    if not math.isfinite(value):
        raise ValueError("Nonfinite clean-GT route progress")
    return value


def _waypoints(trajectory) -> np.ndarray:
    poses = np.asarray(trajectory.poses, dtype=np.float64)
    if (poses.ndim != 2 or poses.shape[0] != NUM_WAYPOINTS or poses.shape[1] < 2
            or not np.isclose(trajectory.trajectory_sampling.interval_length, WAYPOINT_INTERVAL_S, rtol=0, atol=1e-12)):
        raise ValueError("Minimum-jerk reward requires eight waypoints at 0.5-4.0 s")
    return poses[:, :2]


def minimum_jerk_scores(metric_cache, trajectories: Sequence, gt_poses, scorer_config: PDMScorerConfig,
                        config: MinimumJerkRewardConfig, charge_backward: bool = True) -> List[MinimumJerkResult]:
    """Score one scene's candidates independently of each other.

    ``charge_backward=False`` scores each candidate as its merged waypoints (no
    backward penalty), as sibling families do: their members share one base path.
    """
    if not trajectories:
        return []
    weights = np.array([scorer_config.ttc_weight, scorer_config.progress_weight, scorer_config.comfortable_weight],
                       dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Minimum-jerk TTC/EP/comfort weights must be finite, nonnegative, with a positive sum")
    validate_raw_observation(metric_cache, SAMPLING)
    initial = metric_cache.ego_state
    pose = (initial.rear_axle.x, initial.rear_axle.y, initial.rear_axle.heading)
    speed = initial.dynamic_car_state.rear_axle_velocity_2d.x
    reconstruction = config.reconstruction()
    motions = [reconstruct_motion(_waypoints(trajectory), pose, speed, reconstruction) for trajectory in trajectories]
    fitted = [index for index, motion in enumerate(motions) if motion.feasible]
    scorer = MinimumJerkScorer(scorer_config, config.ttc_horizon_s)
    gt_progress = gt_route_progress(metric_cache, gt_poses, scorer._vehicle_parameters.rear_axle_to_center)
    metrics = None
    if fitted:
        metrics = scorer.score_minimum_jerk(np.stack([motion_states(motions[i]) for i in fitted]),
                                            metric_cache.observation, metric_cache.centerline,
                                            metric_cache.route_lane_ids, metric_cache.drivable_area_map)
    floor = -config.contact_weight
    results = []
    for index, motion in enumerate(motions):
        if not motion.feasible:
            results.append(MinimumJerkResult(
                floor, 0., 0., 0., 0., 0., 0., 0., None, False, False, 0., 0., 0., 1., False, False, False,
                True, motion.reason, False))
            continue
        k = fitted.index(index)
        nc, dac, ttc = float(metrics["nc"][k]), float(metrics["dac"][k]), float(metrics["ttc"][k])
        ep = float(np.clip((metrics["progress"][k] + config.ep_epsilon_m) / (gt_progress + config.ep_epsilon_m), 0., 1.))
        comfort = float(is_comfortable(motion, config.low_speed_threshold_mps))
        contact_speed, penalty = None, 0.
        if metrics["first_future_contact"][k] >= 0:
            contact_speed = float(motion.speed[metrics["first_future_contact"][k]])
            penalty = min(contact_speed, config.contact_speed_cap_mps) / config.contact_speed_cap_mps
        base = float(nc * dac * (weights @ [ttc, ep, comfort]) / weights.sum() - config.contact_weight * penalty)
        speed_ratio = float(motion.speed[1:].max() / config.speed_limit_mps)
        alpha = ramp(speed_ratio, config.speed_limit_ramp)
        ratio, violations = feasibility(motion, config)
        beta = ramp(ratio, config.feasibility_ramp)
        backward = motion.backward_distance_m
        beta_backward = min(backward / config.backward_saturation_m, 1.) if charge_backward else 0.
        gamma = max(alpha, beta, beta_backward)
        score = (1 - gamma) * base + gamma * floor
        values = (score, nc, dac, ttc, ep, comfort, base, penalty, speed_ratio, alpha, ratio, beta, backward,
                  beta_backward)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite minimum-jerk reward or component")
        results.append(MinimumJerkResult(
            float(score), nc, dac, ttc, ep, comfort, base, float(penalty), contact_speed,
            bool(metrics["initial_contact"][k]), bool(metrics["vru_rule_contact"][k]), speed_ratio, alpha, ratio,
            beta, *violations, False, None, motion.qp_used, backward, beta_backward))
    return results


def protocol_metadata(settings=None) -> dict:
    """Resolved settings, reconstruction conventions and code identity for resume checks."""
    import hashlib
    from pathlib import Path
    from navsim.evaluate import minimum_jerk_trajectory

    config = settings if isinstance(settings, MinimumJerkRewardConfig) else MinimumJerkRewardConfig.from_mapping(settings)
    code = {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (minimum_jerk_trajectory.__file__, __file__)}
    return {
        "version": 2, "settings": config.as_dict(), "speed_limit": "universal-v1",
        "reconstruction": {
            "path": "cubic-spline-chord-length-start-heading-natural-end-v1",
            "progress": "c2-quintic-0.1s-knots-min-jerk-measured-speed-free-acceleration-nonnegative-speed-v1",
            "heading": "path-tangent-v1", "speed_check_interval_s": 0.01,
            "start_cluster": "forward-displacement-unless-at-rest-v1",
            "backward_waypoints": "chord-turn-over-90deg-merged-into-last-kept-point-v1",
        },
        "backward_penalty": "max-distance-linear-to-saturation-in-gamma-v1",
        "contact": "first-future-at-fault-ego-speed-initial-tracks-excluded-v1",
        "ttc": "stock-exemptions-logged-positions-graded-v1", "ep": "smooth-clean-gt-route-progress-v1",
        "comfort": "pdms-bounds-exact-derivatives-low-speed-yaw-skip-v1",
        "implementation": code,
    }


# Rank-local sums logged by the Lightning module (reduced across ranks before division).
DIAGNOSTIC_FIELDS = (
    "candidates", "fitted", "infeasible_fit", "qp_fallback", "beta_sum",
    "nc_sum", "dac_sum", "ttc_sum", "ep_sum", "comfort_sum", "base_reward_sum", "contact_penalty_sum",
    "max_speed_ratio_sum", "alpha_sum", "feasibility_ratio_sum", "total_acceleration_violation",
    "forward_acceleration_violation", "curvature_violation", "initial_contact", "vru_rule_contact",
    "contact_count", "contact_speed_sum", "beta_backward_sum", "backward_count", "backward_distance_sum",
)


def diagnostic_sums(results: Sequence[MinimumJerkResult]) -> np.ndarray:
    """Counts over all candidates; component/ratio sums over fitted candidates only."""
    totals = dict.fromkeys(DIAGNOSTIC_FIELDS, 0.)
    for result in results:
        totals["candidates"] += 1
        totals["infeasible_fit"] += result.infeasible_fit
        totals["beta_sum"] += result.beta
        if result.infeasible_fit:
            continue
        totals["fitted"] += 1
        totals["qp_fallback"] += result.qp_fallback
        for name, value in (("nc_sum", result.no_at_fault_collisions), ("dac_sum", result.drivable_area_compliance),
                            ("ttc_sum", result.time_to_collision_within_bound), ("ep_sum", result.ego_progress),
                            ("comfort_sum", result.comfort), ("base_reward_sum", result.base_reward),
                            ("contact_penalty_sum", result.contact_penalty),
                            ("max_speed_ratio_sum", result.max_speed_ratio), ("alpha_sum", result.alpha),
                            ("feasibility_ratio_sum", result.feasibility_ratio),
                            ("beta_backward_sum", result.beta_backward)):
            totals[name] += value
        for name in ("total_acceleration_violation", "forward_acceleration_violation", "curvature_violation",
                     "initial_contact", "vru_rule_contact"):
            totals[name] += getattr(result, name)
        if result.contact_speed is not None:
            totals["contact_count"] += 1
            totals["contact_speed_sum"] += result.contact_speed
        if result.backward_distance_m > 0:
            totals["backward_count"] += 1
            totals["backward_distance_sum"] += result.backward_distance_m
    return np.array([totals[name] for name in DIAGNOSTIC_FIELDS], dtype=np.float64)
