"""Deterministic lateral insertions shared by rendering and candidate checks."""

from __future__ import annotations

import math
import numpy as np
from shapely.geometry import MultiPoint

VERSION = "benign_lateral_opposite_car_v1"
SIDE_SELECTION_POLICY = "car_left_first_pedestrians_right_first_all_timesteps_v1"


def preferred_sides(actor_mode):
    if actor_mode not in ("car", "standing", "crossing"):
        raise ValueError(f"Unknown actor mode: {actor_mode}")
    return ("left", "right") if actor_mode == "car" else ("right", "left")


def add_arguments(parser):
    parser.add_argument("--benign", action="store_true", help="Insert beside the future ego path with full-path clearance checks.")
    parser.add_argument("--benign-side", choices=("auto", "left", "right"), default="auto",
                        help="Use the classified side by default; left/right force a side.")
    parser.add_argument("--benign-offset-m", type=float, default=4.0)
    parser.add_argument("--benign-walking-direction", choices=("forward", "backward"), default="forward")
    parser.add_argument("--benign-clearance-m", type=float, default=1.0)


def configuration(args):
    offset = float(getattr(args, "benign_offset_m", 4.0))
    clearance = float(getattr(args, "benign_clearance_m", 1.0))
    side = getattr(args, "benign_side", "auto")
    walking = getattr(args, "benign_walking_direction", "forward")
    if not math.isfinite(offset) or offset <= 0:
        raise ValueError("--benign-offset-m must be finite and positive")
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("--benign-clearance-m must be finite and positive")
    if side not in ("auto", "left", "right") or walking not in ("forward", "backward"):
        raise ValueError("Invalid benign side or walking direction")
    return dict(placement_version=VERSION, benign=True, benign_side=side,
                benign_offset_m=offset, benign_walking_direction=walking,
                benign_clearance_m=clearance, clearance_step_s=0.05,
                clearance_policy="synchronized_swept_footprints_with_rotation_bound_v1",
                vehicle_heading_policy="opposite_future_ego_forward")


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def corners(bounds):
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    result = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    if not np.isfinite(result).all() or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid actor footprint")
    return result


def clearance_diagnostics(scene_frames, current_index, origin, velocity, actor_yaw,
                          actor_bounds, ego_geometry, margin, step_s=0.05):
    """Conservative sweeps for linear XY / shortest-yaw interpolation of ego poses.

    Each interval uses convex hulls of endpoint footprints. The ego hull's
    distance is reduced by a bound on the arc between its rotated corners,
    avoiding misses between sampled poses. Both actors use the same intervals.
    """
    times = np.array([f["timestamp"] for f in scene_frames], dtype=float) / 1e6
    poses = np.array([f["ego2global"] for f in scene_frames], dtype=float)
    if (len(times) <= current_index + 1 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or poses.shape != (len(times), 4, 4)
            or not np.isfinite(poses).all()):
        raise ValueError("Clearance requires valid history and future timestamps/poses")
    ego = corners({"x": [-ego_geometry["rear_length_m"], ego_geometry["front_length_m"]],
                   "y": [-ego_geometry["width_m"] / 2, ego_geometry["width_m"] / 2]})
    actor = corners(actor_bounds) @ rotation(actor_yaw).T
    radius = float(np.linalg.norm(ego, axis=1).max())
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    t0 = times[current_index]
    minimum, minimum_time, count = math.inf, 0.0, 0
    for i in range(len(times) - 1):
        subdivisions = max(1, int(math.ceil((times[i + 1] - times[i]) / step_s)))
        for j in range(subdivisions):
            polygons, actor_polygons, angles = [], [], []
            for fraction in (j / subdivisions, (j + 1) / subdivisions):
                t = times[i] + fraction * (times[i + 1] - times[i])
                xy = poses[i, :2, 3] * (1 - fraction) + poses[i + 1, :2, 3] * fraction
                angle = yaw[i] * (1 - fraction) + yaw[i + 1] * fraction
                polygons.extend(ego @ rotation(angle).T + xy)
                actor_polygons.extend(actor + origin + velocity * (t - t0))
                angles.append(angle)
            swept_ego = MultiPoint(polygons).convex_hull
            swept_actor = MultiPoint(actor_polygons).convex_hull
            arc_bound = radius * (1 - math.cos(abs(angles[1] - angles[0]) / 2))
            distance = max(0.0, float(swept_ego.distance(swept_actor)) - arc_bound)
            count += 1
            if distance < minimum:
                minimum = distance
                minimum_time = times[i] + j / subdivisions * (times[i + 1] - times[i]) - t0
    return dict(passed=minimum >= margin, minimum_clearance_lower_bound_m=minimum,
                required_clearance_m=margin, minimum_interval_start_s=minimum_time,
                horizon_start_s=float(times[0] - t0), horizon_end_s=float(times[-1] - t0),
                interval_count=count, max_interval_s=step_s)


def resolve(args, fast, scene_frames, timestamps_s, front_bumper_offset_m,
            local_bounds, *, timestep=None, actor_mode=None):
    cfg = configuration(args)
    if cfg["benign_side"] == "auto":
        raise ValueError("Resolve the token's automatic benign side before constructing its trajectory")
    timestep = args.timestep if timestep is None else timestep
    mode = actor_mode or ("car" if getattr(args, "car", False) else
                          "standing" if getattr(args, "standing", False) else "crossing")
    if mode not in ("car", "standing", "crossing"):
        raise ValueError(f"Unsupported benign actor mode: {mode}")
    current_index = fast.EXPECTED_HISTORY_FRAMES - 1
    reference = fast.resolve_placement(scene_frames, timestep, front_bumper_offset_m,
                                       current_frame_index=current_index)
    pose = np.asarray(scene_frames[current_index + timestep]["ego2global"], dtype=float)
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    f = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-f[1], f[0]])
    sign = 1 if cfg["benign_side"] == "left" else -1
    bumper = np.asarray(reference["future_front_bumper_global_xy_m"])
    origin = bumper + sign * cfg["benign_offset_m"] * n
    basis_forward, actor_heading = f, -math.pi / 2
    velocity = np.zeros(2)
    if mode == "car":
        # Shift the hazardous footprint, then rotate it about its own centre.
        local_center = corners(local_bounds).mean(axis=0)
        hazardous_origin = bumper - float(local_bounds["x"][0]) * f
        center = hazardous_origin + rotation(yaw) @ local_center + sign * cfg["benign_offset_m"] * n
        origin = center - rotation(yaw + math.pi) @ local_center
        basis_forward, actor_heading = -f, 0.0
        actor_yaw = yaw + math.pi
    elif mode == "crossing":
        walking_sign = 1 if cfg["benign_walking_direction"] == "forward" else -1
        velocity = float(args.speed_mps) * walking_sign * f
        actor_heading = walking_sign * math.pi / 2
        actor_yaw = yaw + actor_heading
    else:
        actor_yaw = yaw + actor_heading
    timestamps_s = np.asarray(timestamps_s, dtype=float)
    expected = np.array([frame["timestamp"] for frame in scene_frames[:current_index + 1]]) / 1e6
    if timestamps_s.shape != expected.shape or not np.allclose(timestamps_s, expected, rtol=0, atol=1e-6):
        raise ValueError("Benign history timestamps must match source frames")
    global_xys = origin + (timestamps_s - timestamps_s[-1])[:, None] * velocity
    current_pose = np.asarray(scene_frames[current_index]["ego2global"], dtype=float)
    current_xys = np.array([fast.inverse_transform_xy_planar(current_pose, xy) for xy in global_xys])
    ego = getattr(args, "ego_vehicle_geometry", None)
    if ego is None:
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        vehicle = get_pacifica_parameters()
        ego = dict(front_length_m=vehicle.front_length, rear_length_m=vehicle.rear_length,
                   width_m=vehicle.width)
    diagnostics = clearance_diagnostics(scene_frames, current_index, origin, velocity,
                                        actor_yaw, local_bounds, ego, cfg["benign_clearance_m"])
    if not diagnostics["passed"]:
        raise fast.TokenFailure("benign_ego_clearance", f"Benign clearance {diagnostics['minimum_clearance_lower_bound_m']:.3f} m < required {cfg['benign_clearance_m']:.3f} m at interval starting {diagnostics['minimum_interval_start_s']:.3f} s")
    future_end_s = float(scene_frames[-1]["timestamp"]) / 1e6
    placement = {**reference, **cfg, "actor_mode": mode,
                 "side_selection_source": getattr(args, "benign_selection_source", "explicit_side"),
                 "actor_behavior": "parallel_walking" if mode == "crossing" else "stationary",
                 "anchor_global_xy_m": origin.tolist(),
                 "anchor_current_ego_xy_m": fast.inverse_transform_xy_planar(current_pose, origin).tolist(),
                 "actor_root_yaw_global_rad": actor_yaw,
                 "actor_heading_rad": actor_heading,
                 "walking_velocity_global_xy_mps": velocity.tolist(),
                 "clearance": diagnostics,
                 "ego_vehicle_geometry": dict(ego),
                 "footprint_local_bounds_m": local_bounds,
                 "future_actor_endpoint_global_xy_m": (origin + velocity * (future_end_s - timestamps_s[-1])).tolist()}
    return dict(placement=placement, reference_frame_index=current_index + timestep,
                reference_vehicle_forward_global_xy=f, reference_vehicle_left_global_xy=n,
                actor_basis_forward_global_xy=basis_forward,
                actor_heading_rad=actor_heading, actor_root_yaw_global_rad=actor_yaw,
                anchor_global_xy=origin, anchor_current_ego_xy=placement["anchor_current_ego_xy_m"],
                walking_velocity_global_xy_mps=velocity, actor_global_xys=global_xys,
                actor_current_xys=current_xys, crossing_offsets_m=np.full(len(timestamps_s), sign * cfg["benign_offset_m"]),
                scene_collision_path_global_xys=np.vstack([global_xys, placement["future_actor_endpoint_global_xy_m"]]))


def direction_lidar(basis, actor_heading):
    """Rocketbox's local forward is -Y before its renderer-root rotation."""
    return basis @ np.array([math.sin(actor_heading), -math.cos(actor_heading), 0.0])
