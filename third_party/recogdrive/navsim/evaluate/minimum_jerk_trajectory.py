"""Path + progress reconstruction of predicted waypoints for the minimum-jerk reward.

Training only. The eight predicted rear-axle XY waypoints (0.5-4.0 s, initial
ego frame) define a cubic path; progress s(t) along it is the minimum-jerk
quintic spline through the waypoint arc lengths, matching the measured initial
speed, with s'(t) >= 0. The initial acceleration is left to the fit: the noisy
measured value otherwise forces jerk spikes and inflates braking. Heading is the
path tangent, so it never flips and stays defined when stopped. The predicted
heading channel is unused.

Progress uses C2 quintic segments with knots every 0.1 s. Without the speed
constraint the optimum is the natural quintic spline with knots at the
waypoints (it lies in this class). Only when that spline would reverse is the
convex QP solved; the finer knots let a stop complete within a waypoint
interval. A waypoint behind the last kept path point (its step turns more than
90 degrees from the direction of travel) is merged into that point, a stop, and
its distance is reported for the reward's backward penalty. Progress targets
that still cannot be met without reversing are an infeasible fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve, inv, lstsq, solve_triangular

HORIZON_S = 4.0
WAYPOINT_INTERVAL_S = 0.5
NUM_WAYPOINTS = 8
KNOT_INTERVAL_S = 0.1
NUM_KNOTS = 41  # reward samples coincide with the progress knots
KNOTS_PER_WAYPOINT = 5
CHECKS_PER_KNOT = 10  # speed constraint grid: 0.01 s
NUM_CHECKS = (NUM_KNOTS - 1) * CHECKS_PER_KNOT + 1
SAMPLE_TIMES_S = np.arange(NUM_KNOTS) * KNOT_INTERVAL_S

# Arc length: 5-point Gauss-Legendre on 16 pieces per path interval.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(5)
_ARC_PIECES = 16
_NEWTON_ITERATIONS = 8
_MIN_PATH_PARAMETER_SPEED = 1e-3  # |dP/du| for chord-length parameters is ~1
_SPEED_TOLERANCE_MPS = 1e-9  # closed-form/QP feasibility on the check grid
_FINAL_SPEED_TOLERANCE_MPS = 1e-6  # larger violations are numerical failures
_QP_MAX_ITERATIONS = 2000

INFEASIBLE_REASONS = ("degenerate_path", "stop_without_distance", "reversal_required")


@dataclass(frozen=True)
class ReconstructionConfig:
    merge_distance_m: float = 0.05
    stopped_initial_speed_mps: float = 0.05

    def __post_init__(self):
        for name in ("merge_distance_m", "stopped_initial_speed_mps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ReconstructedMotion:
    """Global rear-axle motion at t = 0, 0.1, ..., 4.0 s; arrays are None if infeasible."""
    feasible: bool
    reason: Optional[str] = None
    qp_used: bool = False
    rear_axle_xy: Optional[np.ndarray] = None
    heading: Optional[np.ndarray] = None  # unwrapped
    speed: Optional[np.ndarray] = None
    accel_lon: Optional[np.ndarray] = None
    accel_lat: Optional[np.ndarray] = None
    jerk_lon: Optional[np.ndarray] = None
    jerk_lat: Optional[np.ndarray] = None
    yaw_rate: Optional[np.ndarray] = None
    yaw_accel: Optional[np.ndarray] = None
    curvature: Optional[np.ndarray] = None
    backward_distance_m: float = 0.  # largest distance of a backward waypoint from its merged position


# ---------------------------------------------------------------------------
# Progress: C2 quintic Hermite segments, variables (s, v, a) at each knot.

def _monomial_row(tau: float, order: int) -> np.ndarray:
    row = np.zeros(6)
    for j in range(order, 6):
        row[j] = math.factorial(j) / math.factorial(j - order) * tau ** (j - order)
    return row


# Monomial coefficients in tau = (t - t_i) / h from scaled endpoint values
# (s0, h v0, h^2 a0, s1, h v1, h^2 a1).
_HERMITE = inv(np.array([_monomial_row(t, k) for t in (0., 1.) for k in range(3)]))
_SCALE = np.array([1., KNOT_INTERVAL_S, KNOT_INTERVAL_S ** 2] * 2)
_JERK_GRAM = np.zeros((6, 6))
for _i in range(3, 6):
    for _j in range(3, 6):
        _JERK_GRAM[_i, _j] = (math.factorial(_i) / math.factorial(_i - 3) * math.factorial(_j) / math.factorial(_j - 3)
                              / (_i + _j - 5))


def _segment_indices(segment: int) -> np.ndarray:
    n = NUM_KNOTS
    return np.array([segment, n + segment, 2 * n + segment, segment + 1, n + segment + 1, 2 * n + segment + 1])


def _local_row(tau: float, order: int) -> np.ndarray:
    return _monomial_row(tau, order) @ _HERMITE * _SCALE / KNOT_INTERVAL_S ** order


def _global_row(segment: int, tau: float, order: int) -> np.ndarray:
    row = np.zeros(3 * NUM_KNOTS)
    row[_segment_indices(segment)] = _local_row(tau, order)
    return row


@lru_cache(maxsize=1)
def _progress_basis():
    h = KNOT_INTERVAL_S
    local = (_HERMITE * _SCALE).T @ _JERK_GRAM @ (_HERMITE * _SCALE) / h ** 5
    quadratic = np.zeros((3 * NUM_KNOTS, 3 * NUM_KNOTS))
    for segment in range(NUM_KNOTS - 1):
        index = _segment_indices(segment)
        quadratic[np.ix_(index, index)] += local
    quadratic = (quadratic + quadratic.T) / 2
    # Jerk is one-sided at knots: right limit, except the final sample.
    jerk = np.array([_global_row(i, 0., 3) if i < NUM_KNOTS - 1 else _global_row(NUM_KNOTS - 2, 1., 3)
                     for i in range(NUM_KNOTS)])
    checks = np.array([_global_row(min(j // CHECKS_PER_KNOT, NUM_KNOTS - 2),
                                   (j - min(j // CHECKS_PER_KNOT, NUM_KNOTS - 2) * CHECKS_PER_KNOT) / CHECKS_PER_KNOT, 1)
                       for j in range(NUM_CHECKS)])
    for array in (quadratic, jerk, checks):
        array.setflags(write=False)
    return quadratic, jerk, checks


@lru_cache(maxsize=256)
def _progress_system(stopped: Tuple[bool, ...]):
    """Free/fixed partition and factorizations for one stopped-interval pattern."""
    quadratic, _, checks = _progress_basis()
    n = NUM_KNOTS
    fixed = {KNOTS_PER_WAYPOINT * k for k in range(NUM_WAYPOINTS + 1)} | {n}
    for interval, is_stopped in enumerate(stopped):
        if is_stopped:
            for knot in range(KNOTS_PER_WAYPOINT * interval, KNOTS_PER_WAYPOINT * (interval + 1) + 1):
                fixed |= {knot, n + knot, 2 * n + knot}
    fixed = np.array(sorted(fixed))
    free = np.setdiff1d(np.arange(3 * n), fixed)
    hessian = quadratic[np.ix_(free, free)]
    # Remaining at rest throughout fixes every variable.
    factor = cho_factor(hessian, lower=True) if free.size else None
    lower = np.tril(factor[0]) if free.size else np.zeros((0, 0))
    cross = quadratic[np.ix_(free, fixed)]
    rows = checks[:, free]
    constrained = np.abs(rows).max(axis=1, initial=0.) > 0
    system = {"fixed": fixed, "free": free, "hessian": hessian, "factor": factor, "lower": lower,
              "cross": cross, "check_free": rows[constrained], "check_fixed": checks[np.ix_(constrained, fixed)]}
    for value in system.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return system


def _fixed_values(system, targets: np.ndarray, stopped, v0: float) -> np.ndarray:
    n = NUM_KNOTS
    values = {}
    for k, target in enumerate(targets):
        values[KNOTS_PER_WAYPOINT * k] = target
    for interval, is_stopped in enumerate(stopped):
        if is_stopped:
            for knot in range(KNOTS_PER_WAYPOINT * interval, KNOTS_PER_WAYPOINT * (interval + 1) + 1):
                values[knot] = targets[interval]
                values[n + knot] = 0.
                values[2 * n + knot] = 0.
    values[n] = v0
    return np.array([values[index] for index in system["fixed"]], dtype=np.float64)


def dual_active_set_qp(lower, z0, constraints, bounds, tolerance, max_iterations=_QP_MAX_ITERATIONS):
    """Goldfarb-Idnani dual active set for min 1/2 z'Hz + q'z s.t. Cz >= b.

    ``z0`` is the unconstrained minimizer and ``lower`` the Cholesky factor of H.
    Returns (z, active, multipliers) or None if the constraints are infeasible.
    """
    z = np.array(z0, dtype=np.float64)
    active, multipliers = [], np.zeros(0)
    for _ in range(max_iterations):
        slack = constraints @ z - bounds
        p = int(np.argmin(slack))
        if slack[p] >= -tolerance:
            return z, active, multipliers
        normal, added = constraints[p], 0.
        for _ in range(max_iterations):
            projected = solve_triangular(lower, normal, lower=True)
            if active:
                columns = solve_triangular(lower, constraints[active].T, lower=True)
                dual_step = lstsq(columns, projected)[0]
                residual = projected - columns @ dual_step
            else:
                dual_step, residual = np.zeros(0), projected
            primal_step = solve_triangular(lower.T, residual, lower=False)
            curvature = float(normal @ primal_step)
            full = (math.inf if curvature <= 1e-14 * float(projected @ projected)
                    else -(float(normal @ z) - bounds[p]) / curvature)
            positive = np.flatnonzero(dual_step > 1e-14 * max(1., np.abs(dual_step).max(initial=0.)))
            partial, drop = math.inf, None
            if positive.size:
                ratios = multipliers[positive] / dual_step[positive]
                drop = int(positive[np.argmin(ratios)])
                partial = float(ratios.min())
            if math.isinf(full) and math.isinf(partial):
                return None
            step = min(full, partial)
            if not math.isinf(full):
                z = z + step * primal_step
            multipliers = multipliers - step * dual_step
            added += step
            if full <= partial:
                active.append(p)
                multipliers = np.append(multipliers, added)
                break
            active.pop(drop)
            multipliers = np.delete(multipliers, drop)
        else:
            raise RuntimeError("Minimum-jerk QP did not converge (inner loop)")
    raise RuntimeError("Minimum-jerk QP did not converge")


def _check_kkt(system, z, gradient_constant, active, multipliers, bounds):
    constraints = system["check_free"]
    stationarity = system["hessian"] @ z + gradient_constant
    if active:
        stationarity = stationarity - constraints[active].T @ multipliers
    scale = max(1., float(np.abs(gradient_constant).max()), float(np.abs(system["hessian"] @ z).max()))
    if (np.abs(stationarity).max() > 1e-6 * scale or (multipliers.size and multipliers.min() < -1e-8 * scale)
            or (constraints @ z - bounds).min() < -_FINAL_SPEED_TOLERANCE_MPS):
        raise RuntimeError("Minimum-jerk QP solution failed its optimality check")


def fit_progress(targets: np.ndarray, v0: float):
    """Fit s(t) through ``targets`` (s at 0, 0.5, ..., 4 s) with s' >= 0.

    Returns (s, v, a, jerk at the 41 samples, qp_used) or None if infeasible.
    Consecutive equal targets are an exact stop over that interval; the initial
    acceleration is free unless the first interval is such a stop.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (NUM_WAYPOINTS + 1,) or not np.isfinite(targets).all() or targets[0] != 0:
        raise ValueError("Progress targets must be nine finite values starting at zero")
    if np.any(np.diff(targets) < 0):
        raise ValueError("Progress targets must be nondecreasing")
    stopped = tuple(bool(targets[k + 1] == targets[k]) for k in range(NUM_WAYPOINTS))
    if stopped[0] and v0 != 0:
        return None
    system = _progress_system(stopped)
    fixed = _fixed_values(system, targets, stopped, v0)
    gradient_constant = system["cross"] @ fixed
    z = -cho_solve(system["factor"], gradient_constant) if system["free"].size else np.zeros(0)
    bounds = -(system["check_fixed"] @ fixed)
    qp_used = bool(system["check_free"].shape[0] and (system["check_free"] @ z - bounds).min() < -_SPEED_TOLERANCE_MPS)
    if qp_used:
        solution = dual_active_set_qp(system["lower"], z, system["check_free"], bounds, _SPEED_TOLERANCE_MPS)
        if solution is None:
            return None
        z, active, multipliers = solution
        _check_kkt(system, z, gradient_constant, active, multipliers, bounds)
    x = np.empty(3 * NUM_KNOTS)
    x[system["fixed"]] = fixed
    x[system["free"]] = z
    _, jerk_rows, checks = _progress_basis()
    if (checks @ x).min() < -_FINAL_SPEED_TOLERANCE_MPS:
        raise RuntimeError("Minimum-jerk progress reverses beyond numerical tolerance")
    n = NUM_KNOTS
    return x[:n].copy(), x[n:2 * n].copy(), x[2 * n:].copy(), jerk_rows @ x, qp_used


# ---------------------------------------------------------------------------
# Path: cubic spline through merged waypoints, chord-length parametrized.

class _Path:
    """Arc-length evaluation of the cubic path; a straight ray if it has no length."""

    def __init__(self, points: np.ndarray):
        self.points = points
        if len(points) < 2:
            self.spline, self.length = None, 0.
            return
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.knots = np.concatenate(([0.], np.cumsum(chords)))
        self.spline = CubicSpline(self.knots, points, bc_type=((1, np.array([1., 0.])), (2, np.zeros(2))))
        pieces = np.concatenate([np.linspace(a, b, _ARC_PIECES + 1)[:-1] for a, b in zip(self.knots[:-1], self.knots[1:])]
                                + [self.knots[-1:]])
        self.piece_bounds = pieces
        lengths = self._integral(pieces[:-1], pieces[1:])
        self.piece_arc = np.concatenate(([0.], np.cumsum(lengths)))
        self.knot_arc = self.piece_arc[::_ARC_PIECES]
        self.length = float(self.piece_arc[-1])
        nodes = (pieces[:-1, None] + pieces[1:, None]) / 2 + (pieces[1:, None] - pieces[:-1, None]) / 2 * _GL_NODES
        speeds = np.linalg.norm(self.spline(np.concatenate((nodes.ravel(), pieces)), 1), axis=1)
        self.degenerate = bool(speeds.min() < _MIN_PATH_PARAMETER_SPEED)

    def _integral(self, start, end):
        mid, half = (start + end) / 2, (end - start) / 2
        nodes = mid[..., None] + half[..., None] * _GL_NODES
        speeds = np.linalg.norm(self.spline(nodes.ravel(), 1), axis=1).reshape(nodes.shape)
        return half * (speeds @ _GL_WEIGHTS)

    def _parameters(self, arc: np.ndarray) -> np.ndarray:
        bounds, cumulative = self.piece_bounds, self.piece_arc
        index = np.clip(np.searchsorted(cumulative, arc, side="right") - 1, 0, len(bounds) - 2)
        low, high = bounds[index], bounds[index + 1]
        span = cumulative[index + 1] - cumulative[index]
        u = low + np.divide(arc - cumulative[index], span, out=np.zeros_like(arc), where=span > 0) * (high - low)
        for _ in range(_NEWTON_ITERATIONS):
            error = cumulative[index] + self._integral(low, u) - arc
            u = np.clip(u - error / np.linalg.norm(self.spline(u, 1), axis=1), low, high)
        error = cumulative[index] + self._integral(low, u) - arc
        if np.abs(error).max(initial=0.) > 1e-7:
            raise RuntimeError("Path arc-length inversion did not converge")
        return u

    def evaluate(self, arc: np.ndarray):
        """Position, tangent angle, curvature and d(curvature)/ds at arc lengths."""
        arc = np.asarray(arc, dtype=np.float64)
        if self.spline is None:
            zeros = np.zeros_like(arc)
            return np.column_stack((arc, zeros)), zeros, zeros, zeros
        u = self._parameters(np.clip(arc, 0., self.length))
        position = self.spline(u)
        d1, d2, d3 = (self.spline(u, k) for k in (1, 2, 3))
        speed = np.linalg.norm(d1, axis=1)
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        dot = d1[:, 0] * d2[:, 0] + d1[:, 1] * d2[:, 1]
        cross3 = d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]
        curvature = cross / speed ** 3
        curvature_s = (cross3 * speed ** 2 - 3 * cross * dot) / speed ** 6
        return position, np.arctan2(d1[:, 1], d1[:, 0]), curvature, curvature_s


def _merge_waypoints(points: np.ndarray, merge_distance: float):
    """Kept path points, each waypoint's kept point and the largest backward distance.

    A step from the last kept point that turns more than 90 degrees from the
    direction of travel (the last kept chord, initially the heading) is backward:
    the waypoint is merged into that point and its distance counts. A forward
    step shorter than ``merge_distance`` is merged at no cost (path geometry only).
    """
    kept, owner, backward = [points[0]], [0], 0.
    direction = np.array([1., 0.])
    for point in points[1:]:
        step = point - kept[-1]
        distance = float(np.hypot(*step))
        if step @ direction < 0:
            backward = max(backward, distance)
        elif distance >= merge_distance:
            kept.append(point)
            direction = step
        owner.append(len(kept) - 1)
    return np.array(kept), np.array(owner), backward


def reconstruct_motion(waypoints_xy, initial_pose, initial_speed: float,
                       config: ReconstructionConfig = ReconstructionConfig()) -> ReconstructedMotion:
    """Reconstruct the rear-axle motion of eight local-frame XY waypoints.

    ``initial_pose`` is the measured global rear-axle (x, y, heading) and
    ``initial_speed`` its measured longitudinal speed. At or below
    ``stopped_initial_speed_mps`` the ego is at rest (speed 0). Backward
    waypoints are merged; ``backward_distance_m`` reports the largest distance.
    """
    waypoints = np.asarray(waypoints_xy, dtype=np.float64)
    pose = np.asarray(initial_pose, dtype=np.float64)
    if waypoints.shape != (NUM_WAYPOINTS, 2) or not np.isfinite(waypoints).all():
        raise ValueError("Minimum-jerk reward requires eight finite XY waypoints")
    if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(initial_speed):
        raise ValueError("Minimum-jerk reward requires a finite measured initial state")
    at_rest = initial_speed <= config.stopped_initial_speed_mps
    v0 = 0. if at_rest else float(initial_speed)

    points = np.vstack(([0., 0.], waypoints))
    kept, owner, backward = _merge_waypoints(points, config.merge_distance_m)
    path = _Path(kept)
    if path.spline is not None and path.degenerate:
        return ReconstructedMotion(False, "degenerate_path")
    knot_arc = path.knot_arc if path.spline is not None else np.zeros(1)
    targets = knot_arc[owner].astype(np.float64)
    start = owner == 0
    if not at_rest:
        # Waypoints merged into the start keep their forward displacement (< merge distance; none if backward).
        targets[start] = np.maximum.accumulate(np.maximum(points[start, 0], 0.))
    targets[0] = 0.
    if targets[1] == 0 and not at_rest:
        return ReconstructedMotion(False, "stop_without_distance")
    progress = fit_progress(targets, v0)
    if progress is None:
        return ReconstructedMotion(False, "reversal_required")
    arc, speed, accel_lon, jerk_lon, qp_used = progress
    speed = np.maximum(speed, 0.)
    position, heading, curvature, curvature_s = path.evaluate(np.clip(arc, 0., targets[-1]))
    heading = np.unwrap(heading)
    accel_lat = curvature * speed ** 2
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[cos, -sin], [sin, cos]])
    motion = ReconstructedMotion(
        True, None, qp_used, position @ rotation.T + pose[:2], heading + pose[2], speed, accel_lon, accel_lat,
        jerk_lon, curvature_s * speed ** 3 + 2 * curvature * speed * accel_lon, curvature * speed,
        curvature_s * speed ** 2 + curvature * accel_lon, curvature, backward)
    for name in ("rear_axle_xy", "heading", "speed", "accel_lon", "accel_lat", "jerk_lon", "jerk_lat",
                 "yaw_rate", "yaw_accel", "curvature"):
        if not np.isfinite(getattr(motion, name)).all():
            raise RuntimeError(f"Nonfinite reconstructed {name}")
    return motion
