"""Scalar references for tests: physical contacts plus DAC (any contact), and the per-sample insertion geometry."""
import math
from typing import Any, Mapping, Optional

import numpy as np

from reactbench.scoring.actor_motion import actor_footprint_corners, validated_motion
from reactbench.scoring.geometry import (
    COLLISION_EPSILON_M, GEOMETRY_SAMPLE_INTERVAL_S, bounded_minimum, first_contact_time, footprint_signed_clearance,
    interpolate_trajectory_pose, trajectory_array, vehicle_footprint_corners,
)
from reactbench.scoring.collisions import score_raw_trajectory
from reactbench.scoring.navsim_metrics import raw_dac


def score_prediction(poses, interval_length_s, actor_motion, world):
    """Physical contacts with any road user, not only at-fault ones (protocol V1), plus DAC; actor_motion=None
    scores an unedited scene."""
    collision = score_raw_trajectory(poses, actor_motion, world["geometry"], world,
                                     interval_length_s=interval_length_s)
    dac = raw_dac(poses, interval_length_s, world)
    return {**collision, **dac, "nc_all": int(not collision["all"]), "nc_inserted": int(not collision["inserted"]),
            "nc_dac": int(not collision["all"]) * dac["dac"] if dac["dac"] is not None else None,
            "interval_length_s": interval_length_s}


def evaluate_trajectory_geometry(
    trajectory: Any,
    actor_motion: Mapping[str, Any],
    interval_length_s: float,
    vehicle_geometry: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate closest approach and physical footprint contact over the horizon."""

    poses = trajectory_array(trajectory, "predicted")
    if poses.shape[1] < 3 or not np.all(np.isfinite(poses[:, 2])):
        raise ValueError("Geometry evaluation requires finite x/y/heading poses")
    interval = float(interval_length_s)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(
            f"interval_length_s must be finite and positive, got {interval}"
        )
    motion = validated_motion(actor_motion)
    horizon = poses.shape[0] * interval

    def clearance(time_s: float) -> float:
        vehicle = vehicle_footprint_corners(
            interpolate_trajectory_pose(poses, time_s, interval),
            vehicle_geometry,
        )
        actor = actor_footprint_corners(motion, time_s)
        return footprint_signed_clearance(vehicle, actor)

    sample_count = max(2, int(math.ceil(horizon / GEOMETRY_SAMPLE_INTERVAL_S)) + 1)
    times = np.linspace(0.0, horizon, sample_count, dtype=np.float64)
    clearances = np.asarray([clearance(float(time_s)) for time_s in times])
    coarse_index = int(np.argmin(clearances))
    lower_index = max(0, coarse_index - 1)
    upper_index = min(sample_count - 1, coarse_index + 1)
    refined_time_s, refined_clearance_m = bounded_minimum(
        clearance,
        float(times[lower_index]),
        float(times[upper_index]),
    )
    critical_time_s, minimum_clearance_m = min(
        (
            (float(times[lower_index]), float(clearances[lower_index])),
            (float(times[coarse_index]), float(clearances[coarse_index])),
            (float(times[upper_index]), float(clearances[upper_index])),
            (refined_time_s, refined_clearance_m),
        ),
        key=lambda candidate: (candidate[1], candidate[0]),
    )

    collision = minimum_clearance_m <= COLLISION_EPSILON_M
    collision_time_s: Optional[float] = None
    if collision:
        colliding_indices = np.flatnonzero(clearances <= COLLISION_EPSILON_M)
        if colliding_indices.size:
            collision_index = int(colliding_indices[0])
            if collision_index == 0:
                collision_time_s = 0.0
            else:
                collision_time_s = first_contact_time(
                    clearance,
                    float(times[collision_index - 1]),
                    float(times[collision_index]),
                )
        else:
            # A narrow overlap can be exposed only by the refined minimum.
            left_index = max(0, coarse_index - 1)
            while left_index > 0 and clearances[left_index] <= COLLISION_EPSILON_M:
                left_index -= 1
            collision_time_s = first_contact_time(
                clearance, float(times[left_index]), critical_time_s
            )

    return {
        "time_to_critical_s": float(critical_time_s),
        "minimum_clearance_m": float(minimum_clearance_m),
        "collision": bool(collision),
        "time_to_collision_s": collision_time_s,
        "horizon_s": float(horizon),
    }
