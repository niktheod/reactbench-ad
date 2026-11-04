"""Ego-vehicle geometry: footprints, trajectory interpolation and signed clearances.

Predicted trajectories are rear-axle x/y/heading poses at a native interval. Between poses they
are piecewise linear in x, y and unwrapped heading, starting from the current pose at t = 0.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .motion_bins import MINIMUM_PROJECTION_SPEED_MPS, project_longitudinal_acceleration

# NAVSIM predictions are rear-axle poses for the standard Pacifica.
VEHICLE_FRONT_LENGTH_M = 4.049
VEHICLE_REAR_LENGTH_M = 1.127
VEHICLE_WIDTH_M = 2.297

# Collision checks sample the trajectories every 0.01 s, then refine each contact.
GEOMETRY_SAMPLE_INTERVAL_S = 0.01
COLLISION_EPSILON_M = 1e-9


def trajectory_array(trajectory: Any, name: str) -> np.ndarray:
    poses = getattr(trajectory, "poses", trajectory)
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            f"{name} trajectory must have shape (N, >=2), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} trajectory must contain at least one pose")
    if not np.all(np.isfinite(array[:, :2])):
        raise ValueError(f"{name} trajectory contains non-finite positions")
    return array


def average_displacement_error(original: Any, edited: Any) -> float:
    """Return mean 2D displacement between corresponding predicted poses."""

    original_poses = trajectory_array(original, "original")
    edited_poses = trajectory_array(edited, "edited")
    if original_poses.shape[0] != edited_poses.shape[0]:
        raise ValueError(
            "Original and edited trajectories have different pose counts: "
            f"{original_poses.shape[0]} != {edited_poses.shape[0]}"
        )
    distances = np.linalg.norm(
        original_poses[:, :2] - edited_poses[:, :2], axis=1
    )
    return float(np.mean(distances))


def trajectory_distance_driven(trajectory: Any) -> float:
    """Return cumulative 2D path length from the current rear-axle origin."""

    poses = trajectory_array(trajectory, "predicted")
    positions = np.vstack(
        (np.zeros((1, 2), dtype=np.float64), poses[:, :2])
    )
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def finite_xy(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values, got {value!r}")
    return array


def interpolate_trajectory_pose(
    trajectory: Any, time_s: float, interval_length_s: float
) -> np.ndarray:
    """Interpolate a rear-axle pose, including the current pose at t=0."""

    poses = trajectory_array(trajectory, "predicted")
    if poses.shape[1] < 3 or not np.all(np.isfinite(poses[:, 2])):
        raise ValueError("Geometry evaluation requires finite x/y/heading poses")
    interval = float(interval_length_s)
    time_s = float(time_s)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(
            f"interval_length_s must be finite and positive, got {interval}"
        )
    horizon = poses.shape[0] * interval
    if not math.isfinite(time_s) or time_s < -1e-9 or time_s > horizon + 1e-9:
        raise ValueError(
            f"time_s {time_s} is outside trajectory horizon [0, {horizon}]"
        )
    time_s = min(max(time_s, 0.0), horizon)
    node_times = np.arange(poses.shape[0] + 1, dtype=np.float64) * interval
    nodes = np.vstack((np.zeros((1, poses.shape[1]), dtype=np.float64), poses))
    headings = np.unwrap(nodes[:, 2])
    return np.array(
        [
            np.interp(time_s, node_times, nodes[:, 0]),
            np.interp(time_s, node_times, nodes[:, 1]),
            np.interp(time_s, node_times, headings),
        ],
        dtype=np.float64,
    )


def oriented_box_corners(
    center: np.ndarray,
    heading: float,
    front_length: float,
    rear_length: float,
    left_width: float,
    right_width: float,
) -> np.ndarray:
    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    left = np.array([-forward[1], forward[0]], dtype=np.float64)
    return np.array(
        [
            center + front_length * forward + left_width * left,
            center + front_length * forward - right_width * left,
            center - rear_length * forward - right_width * left,
            center - rear_length * forward + left_width * left,
        ],
        dtype=np.float64,
    )


def vehicle_footprint_parameters(
    vehicle_geometry: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Normalize a rear-axle vehicle footprint.

    ``None`` preserves the NAVSIM Pacifica dimensions used by all historical
    callers.  PAI-AV adapters pass their recorded
    ``ego_vehicle_geometry.footprint_rear_axle_m`` mapping.
    """

    if vehicle_geometry is None:
        return {
            "front_length_m": VEHICLE_FRONT_LENGTH_M,
            "rear_length_m": VEHICLE_REAR_LENGTH_M,
            "width_m": VEHICLE_WIDTH_M,
        }
    if not isinstance(vehicle_geometry, Mapping):
        raise ValueError("vehicle_geometry must be a mapping")
    footprint_value = vehicle_geometry.get("footprint_rear_axle_m")
    footprint = (
        footprint_value
        if isinstance(footprint_value, Mapping)
        else vehicle_geometry
    )
    try:
        x_min = float(footprint["x_min"])
        x_max = float(footprint["x_max"])
        y_min = float(footprint["y_min"])
        y_max = float(footprint["y_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "vehicle_geometry must contain finite x_min/x_max/y_min/y_max "
            "rear-axle footprint bounds"
        ) from exc
    bounds = np.asarray([x_min, x_max, y_min, y_max], dtype=np.float64)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("vehicle_geometry footprint contains non-finite bounds")
    if not x_min < 0.0 < x_max or not y_min < 0.0 < y_max:
        raise ValueError(
            "vehicle_geometry rear axle and centerline must lie inside the "
            f"footprint, got {footprint!r}"
        )
    result = {
        "front_length_m": x_max,
        "rear_length_m": -x_min,
        "width_m": y_max - y_min,
        "left_width_m": y_max,
        "right_width_m": -y_min,
    }
    version = vehicle_geometry.get("version")
    if version is not None:
        result["source_geometry_version"] = str(version)
    return result


def vehicle_footprint_corners(
    pose: Any,
    vehicle_geometry: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape[0] < 3 or not np.all(np.isfinite(pose_array[:3])):
        raise ValueError(f"Vehicle pose must contain finite x/y/heading, got {pose!r}")
    footprint = vehicle_footprint_parameters(vehicle_geometry)
    return oriented_box_corners(
        pose_array[:2],
        float(pose_array[2]),
        float(footprint["front_length_m"]),
        float(footprint["rear_length_m"]),
        float(footprint.get("left_width_m", footprint["width_m"] / 2.0)),
        float(footprint.get("right_width_m", footprint["width_m"] / 2.0)),
    )


def _polygon_axes(corners: np.ndarray) -> list[np.ndarray]:
    axes = []
    for start, end in ((corners[0], corners[1]), (corners[1], corners[2])):
        edge = end - start
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            raise ValueError("Degenerate collision footprint")
        axes.append(normal / norm)
    return axes


def point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / squared_length)
    projection = start + min(max(fraction, 0.0), 1.0) * segment
    return float(np.linalg.norm(point - projection))


def footprint_signed_clearance(first: np.ndarray, second: np.ndarray) -> float:
    """Return polygon clearance; negative values are SAT penetration depth."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4, 2) or second.shape != (4, 2):
        raise ValueError("Collision footprints must both have shape (4, 2)")

    gaps = []
    for axis in _polygon_axes(first) + _polygon_axes(second):
        first_projection = first @ axis
        second_projection = second @ axis
        gap = max(
            float(second_projection.min() - first_projection.max()),
            float(first_projection.min() - second_projection.max()),
        )
        gaps.append(gap)
    if max(gaps) <= COLLISION_EPSILON_M:
        return -float(min(-gap for gap in gaps))

    distances = []
    for source, target in ((first, second), (second, first)):
        for point in source:
            for edge_index in range(4):
                distances.append(
                    point_segment_distance(
                        point,
                        target[edge_index],
                        target[(edge_index + 1) % 4],
                    )
                )
    return float(min(distances))


def bounded_minimum(
    function: Callable[[float], float], lower: float, upper: float
) -> tuple[float, float]:
    if upper <= lower:
        value = float(function(lower))
        return lower, value
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = float(function(left))
    right_value = float(function(right))
    for _ in range(48):
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = float(function(left))
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = float(function(right))
    time_s = (lower + upper) / 2.0
    return time_s, float(function(time_s))


def first_contact_time(
    clearance: Callable[[float], float], lower: float, upper: float
) -> float:
    if clearance(lower) <= COLLISION_EPSILON_M:
        return lower
    for _ in range(48):
        midpoint = (lower + upper) / 2.0
        if clearance(midpoint) <= COLLISION_EPSILON_M:
            upper = midpoint
        else:
            lower = midpoint
    return upper


STATIONARY_SPEED_EPSILON_MPS = MINIMUM_PROJECTION_SPEED_MPS


def project_acceleration_onto_velocity(
    velocity: Any,
    acceleration: Any,
    *,
    stationary_speed_epsilon_mps: float = STATIONARY_SPEED_EPSILON_MPS,
) -> Optional[float]:
    """Return signed longitudinal acceleration ``dot(a, v) / ||v||``.

    The direction of travel is undefined for a stationary ego vehicle, so
    samples at or below ``stationary_speed_epsilon_mps`` return ``None``.
    """

    velocity_array = np.asarray(velocity, dtype=np.float64)
    acceleration_array = np.asarray(acceleration, dtype=np.float64)
    if (
        velocity_array.ndim != 1
        or velocity_array.size < 2
        or not np.isfinite(velocity_array[:2]).all()
    ):
        raise ValueError(
            "velocity must contain at least two finite values, "
            f"got {velocity!r}"
        )
    if (
        acceleration_array.ndim != 1
        or acceleration_array.size < 2
        or not np.isfinite(acceleration_array[:2]).all()
    ):
        raise ValueError(
            "acceleration must contain at least two finite values, "
            f"got {acceleration!r}"
        )
    velocity_xy = velocity_array[:2]
    acceleration_xy = acceleration_array[:2]
    epsilon = float(stationary_speed_epsilon_mps)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(
            "stationary_speed_epsilon_mps must be finite and non-negative"
        )
    return project_longitudinal_acceleration(
        velocity_xy,
        acceleration_xy,
        minimum_speed_mps=epsilon,
    )
