"""Reward-ranked speed siblings of the student's deterministic output (stage-3 L_SIB targets).

Training only. The base's path and minimum-jerk
progress are reconstructed exactly as by the minimum-jerk reward. Each sibling
keeps the path and offsets the base's longitudinal acceleration by -da, ramped
in with the minimum-jerk smoothstep over ``ramp_s`` from its onset:

    a_j(t) = a_base(t) - da S((t - t0) / ramp),   v_j = max(v_base + int(a_j - a_base), 0)

latched at zero once braking stops the car, s_j = int v_j. Waypoints are
re-sampled at 0.5-4.0 s with the path tangent as heading; faster siblings
continue straight along the final heading. The family (the base first) is
scored with the training reward without its backward-waypoint penalty (siblings
never step back; the base counts as its merged waypoints), and the best member
becomes the target if it beats the base by more than ``margin``; ties prefer
the smallest |da|, then the earlier onset.

The reconstruction reuses the reward module's private helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.minimum_jerk_pdm_score import MinimumJerkRewardConfig, minimum_jerk_scores
from navsim.evaluate.minimum_jerk_trajectory import (
    _HERMITE, _SCALE, KNOT_INTERVAL_S, KNOTS_PER_WAYPOINT, NUM_KNOTS, NUM_WAYPOINTS, WAYPOINT_INTERVAL_S,
    ReconstructionConfig, _merge_waypoints, _Path, fit_progress,
)

HORIZON_S = NUM_WAYPOINTS * WAYPOINT_INTERVAL_S
WAYPOINT_TIMES_S = np.arange(1, NUM_WAYPOINTS + 1) * WAYPOINT_INTERVAL_S
_STOP_GRID_S = 1e-3  # latching search grid; the stop time is then refined to ~1e-12 s


@dataclass(frozen=True)
class SiblingSettings:
    decelerations_mps2: Tuple[float, ...] = (-1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.)  # onset 0 s; 0 = base
    late_decelerations_mps2: Tuple[float, ...] = (2., 3., 4., 6.)
    late_onset_s: float = .5
    ramp_s: float = .5
    extension_m: float = 40.
    margin: float = .1

    def __post_init__(self):
        for name in ("decelerations_mps2", "late_decelerations_mps2"):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values or len(set(values)) != len(values)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                           for v in values)):
                raise ValueError(f"training_siblings.{name} must be distinct finite numbers")
        if 0 not in self.decelerations_mps2:
            raise ValueError("training_siblings.decelerations_mps2 must include 0 (the base)")
        if 0 in self.late_decelerations_mps2:
            raise ValueError("training_siblings.late_decelerations_mps2 must not repeat the base (0)")
        for name in ("late_onset_s", "ramp_s", "extension_m", "margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"training_siblings.{name} must be a finite number")
        if not 0 < self.late_onset_s < HORIZON_S or not 0 < self.ramp_s <= HORIZON_S:
            raise ValueError("training_siblings onset and ramp must lie within the 4 s horizon")
        if self.extension_m <= 0 or self.margin < 0:
            raise ValueError("training_siblings.extension_m must be positive and margin nonnegative")

    @classmethod
    def from_mapping(cls, values) -> "SiblingSettings":
        values = dict(values or {})
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training_siblings settings: {sorted(unknown)}")
        for name in ("decelerations_mps2", "late_decelerations_mps2"):
            if name in values:
                if isinstance(values[name], (str, bytes)) or not hasattr(values[name], "__iter__"):
                    raise ValueError(f"training_siblings.{name} must be a list of numbers")
                values[name] = tuple(values[name])
        return cls(**values)

    def as_dict(self) -> dict:
        return {"decelerations_mps2": [float(v) for v in self.decelerations_mps2],
                "late_decelerations_mps2": [float(v) for v in self.late_decelerations_mps2],
                **{name: float(getattr(self, name)) for name in ("late_onset_s", "ramp_s", "extension_m", "margin")}}

    def family(self) -> List[Tuple[float, float]]:
        """(deceleration, onset) per member in grid order; the first member is the base."""
        return ([(0., 0.)] + [(float(d), 0.) for d in self.decelerations_mps2 if d != 0]
                + [(float(d), float(self.late_onset_s)) for d in self.late_decelerations_mps2])


@dataclass(frozen=True)
class BaseMotion:
    """The reward's reconstruction of the base in its local frame: path and progress knots (0-4 s, 0.1 s)."""
    path: _Path
    arc: np.ndarray
    speed: np.ndarray
    accel: np.ndarray


def reconstruct_base(waypoints_xy, initial_speed: float, config: ReconstructionConfig):
    """(BaseMotion, None), or (None, reason) when the reward's reconstruction is infeasible.

    Mirrors ``minimum_jerk_trajectory.reconstruct_motion`` up to the progress
    fit, without the global transform.
    """
    waypoints = np.asarray(waypoints_xy, dtype=np.float64)
    if waypoints.shape != (NUM_WAYPOINTS, 2) or not np.isfinite(waypoints).all() or not math.isfinite(initial_speed):
        raise ValueError("Sibling bases require eight finite XY waypoints and a finite initial speed")
    at_rest = initial_speed <= config.stopped_initial_speed_mps
    v0 = 0. if at_rest else float(initial_speed)
    points = np.vstack(([0., 0.], waypoints))
    kept, owner, _ = _merge_waypoints(points, config.merge_distance_m)
    path = _Path(kept)
    if path.spline is not None and path.degenerate:
        return None, "degenerate_path"
    knot_arc = path.knot_arc if path.spline is not None else np.zeros(1)
    targets = knot_arc[owner].astype(np.float64)
    start = owner == 0
    if not at_rest:
        targets[start] = np.maximum.accumulate(np.maximum(points[start, 0], 0.))
    targets[0] = 0.
    if targets[1] == 0 and not at_rest:
        return None, "stop_without_distance"
    progress = fit_progress(targets, v0)
    if progress is None:
        return None, "reversal_required"
    arc, speed, accel, _, _ = progress
    return BaseMotion(path, arc, speed, accel), None


def _progress(base: BaseMotion, times, order: int) -> np.ndarray:
    """Exact quintic progress (order 0), speed (1) or acceleration (2) at arbitrary times in [0, 4] s."""
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    segment = np.clip(np.floor(times / KNOT_INTERVAL_S).astype(int), 0, NUM_KNOTS - 2)
    tau = times / KNOT_INTERVAL_S - segment
    monomials = np.zeros((len(times), 6))
    for j in range(order, 6):
        monomials[:, j] = math.factorial(j) / math.factorial(j - order) * tau ** (j - order)
    rows = (monomials @ _HERMITE) * _SCALE / KNOT_INTERVAL_S ** order
    knots = np.column_stack((base.arc[segment], base.speed[segment], base.accel[segment],
                             base.arc[segment + 1], base.speed[segment + 1], base.accel[segment + 1]))
    return np.einsum("ij,ij->i", rows, knots)


def _smoothstep_integrals(tau):
    """Integrals of the minimum-jerk smoothstep S: I = int_0^tau S and J = int_0^tau I."""
    tau = np.asarray(tau, dtype=np.float64)
    inside = np.clip(tau, 0., 1.)
    first = np.where(tau >= 1, tau - .5, 2.5 * inside ** 4 - 3 * inside ** 5 + inside ** 6)
    second = np.where(tau >= 1, 1 / 7 + tau ** 2 / 2 - tau / 2, .5 * inside ** 5 - .5 * inside ** 6 + inside ** 7 / 7)
    return np.where(tau > 0, first, 0.), np.where(tau > 0, second, 0.)


def _offsets(times, deceleration: float, onset: float, ramp: float):
    """Speed and progress offsets of the ramped acceleration offset -deceleration * S."""
    first, second = _smoothstep_integrals((np.asarray(times, dtype=np.float64) - onset) / ramp)
    return -deceleration * ramp * first, -deceleration * ramp ** 2 * second


def _stop_time(base: BaseMotion, deceleration: float, onset: float, ramp: float) -> Optional[float]:
    """First time after the onset at which braking brings the sibling's speed to zero, if any."""
    if deceleration <= 0:
        return None
    grid = np.arange(int(round(HORIZON_S / _STOP_GRID_S)) + 1) * _STOP_GRID_S
    speed = lambda t: _progress(base, t, 1) + _offsets(t, deceleration, onset, ramp)[0]
    stopped = np.flatnonzero((grid > onset) & (speed(grid) <= 0))
    if not stopped.size:
        return None
    high = float(grid[stopped[0]])
    low = max(float(grid[stopped[0] - 1]), onset)
    if speed(low)[0] <= 0:
        return low
    return float(brentq(lambda t: float(speed(t)[0]), low, high, xtol=1e-12, rtol=4 * np.finfo(float).eps))


def sibling_arc(base: BaseMotion, deceleration: float, onset: float, ramp: float) -> np.ndarray:
    """Sibling progress at the eight waypoint times."""
    stop = _stop_time(base, deceleration, onset, ramp)
    arc = base.arc[KNOTS_PER_WAYPOINT * np.arange(1, NUM_WAYPOINTS + 1)] + _offsets(
        WAYPOINT_TIMES_S, deceleration, onset, ramp)[1]
    if stop is not None:
        arc = np.where(WAYPOINT_TIMES_S > stop, _progress(base, stop, 0)[0] + _offsets(stop, deceleration, onset, ramp)[1], arc)
    return arc


def sibling_waypoints(base: BaseMotion, deceleration: float, onset: float, settings: SiblingSettings) -> np.ndarray:
    """Local-frame (x, y, heading) at 0.5-4.0 s; heading is the path tangent."""
    arc = sibling_arc(base, deceleration, onset, settings.ramp_s)
    path = base.path
    if path.spline is None:  # the base stays at the start: its path is the initial-heading ray
        if arc.max() > settings.extension_m:
            raise ValueError("Sibling travels beyond the straight path extension")
        return np.column_stack((arc, np.zeros_like(arc), np.zeros_like(arc)))
    beyond = arc - path.length
    if beyond.max() > settings.extension_m:
        raise ValueError("Sibling travels beyond the straight path extension")
    position, heading, _, _ = path.evaluate(np.minimum(arc, path.length))
    extra = np.maximum(beyond, 0.)
    position = position + extra[:, None] * np.column_stack((np.cos(heading), np.sin(heading)))
    return np.column_stack((position, heading))


@dataclass(frozen=True)
class SiblingOutcome:
    feasible: bool  # the base's reconstruction (no siblings otherwise)
    reason: Optional[str] = None
    base_score: float = math.nan
    best_score: float = math.nan  # over the family, including the base
    base_contact: bool = False
    target: Optional[np.ndarray] = None  # (8, 3) local waypoints
    deceleration: Optional[float] = None
    onset: Optional[float] = None
    target_contact: bool = False
    base_backward: bool = False  # backward waypoints merged (not charged within the family)


def select_target(family: Sequence[Tuple[float, float]], scores: Sequence[float], margin: float) -> Optional[int]:
    """Index of the best member if it beats the base (index 0) by more than ``margin``."""
    order = sorted(range(len(family)), key=lambda i: (abs(family[i][0]), family[i][1]))
    best = max(order, key=lambda i: scores[i])  # first maximum in tie order
    return best if scores[best] > scores[0] + margin else None


def sibling_outcome(metric_cache, base_poses, gt_poses, scorer_config, reward_config: MinimumJerkRewardConfig,
                    settings: SiblingSettings) -> SiblingOutcome:
    """Build and score one base's family; never mutates the metric cache."""
    base_poses = np.asarray(base_poses, dtype=np.float64)
    if base_poses.shape != (NUM_WAYPOINTS, 3):
        raise ValueError("Sibling bases must be eight (x, y, heading) poses")
    speed = metric_cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.x
    base, reason = reconstruct_base(base_poses[:, :2], speed, reward_config.reconstruction())
    if base is None:
        return SiblingOutcome(False, reason)
    family = settings.family()
    poses = [base_poses] + [sibling_waypoints(base, d, onset, settings) for d, onset in family[1:]]
    results = minimum_jerk_scores(metric_cache, [Trajectory(p) for p in poses], gt_poses, scorer_config, reward_config,
                                  charge_backward=False)
    if results[0].infeasible_fit:
        raise RuntimeError("Sibling and reward reconstructions disagree on the base")
    scores = [result.score for result in results]
    best = select_target(family, scores, settings.margin)
    outcome = dict(feasible=True, base_score=scores[0], best_score=max(scores),
                   base_contact=results[0].contact_speed is not None,
                   base_backward=results[0].backward_distance_m > 0)
    if best is not None:
        outcome.update(target=poses[best], deceleration=family[best][0], onset=family[best][1],
                       target_contact=results[best].contact_speed is not None)
    return SiblingOutcome(**outcome)


def protocol_metadata(settings=None) -> dict:
    """Resolved settings, conventions and code identity for stage-3 resume checks."""
    import hashlib
    from pathlib import Path

    config = settings if isinstance(settings, SiblingSettings) else SiblingSettings.from_mapping(settings)
    return {
        "version": 2, "settings": config.as_dict(),
        "family": "acceleration-offset-min-jerk-smoothstep-latched-stop-v1",
        "path": "base-path-straight-extension-tangent-heading-v1",
        "selection": "best-of-family-over-margin-ties-smallest-offset-then-onset-v1",
        "backward": "family-scored-as-merged-waypoints-without-backward-penalty-v1",
        "loss": "stage2-epsilon-mse-student-conditioning-sum-over-micro-batch-size-v1",
        "implementation": {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }


# Per-scene rows summed over ranks by the Lightning module before division.
SIBLING_FIELDS = ("scenes", "feasible_base", "targets", "margin_sum", "deceleration_sum", "late_onset",
                  "base_contact", "target_contact", "base_infeasible", "base_backward", "loss_sum")


def diagnostic_row(outcome: SiblingOutcome, loss: float = 0.) -> np.ndarray:
    row = dict.fromkeys(SIBLING_FIELDS, 0.)
    row.update(scenes=1., loss_sum=float(loss))
    if not outcome.feasible:
        row.update(base_infeasible=1.)
    else:
        row.update(feasible_base=1., margin_sum=outcome.best_score - outcome.base_score,
                   base_contact=float(outcome.base_contact), base_backward=float(outcome.base_backward))
        if outcome.target is not None:
            row.update(targets=1., deceleration_sum=outcome.deceleration, late_onset=float(outcome.onset > 0),
                       target_contact=float(outcome.target_contact))
    return np.array([row[name] for name in SIBLING_FIELDS], dtype=np.float64)
