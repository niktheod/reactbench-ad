#!/usr/bin/env python3
"""Create cheap, calibrated NAVSIM receiver-marker previews.

This module deliberately does not import or invoke Blender. It reproduces the
compositor's token selection, asset selection, crossing trajectory, and
per-frame LiDAR grounding, then projects a small stance region into one camera
per timestep. The original sensor image remains untouched except for a thin
ground ellipse and center cross.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import yaml


SCHEMA_VERSION = 1
PIPELINE_VERSION = (
    "paiav_projected_receiver_preview_v9_measured_pedestrian_geometry"
)
SCENE_INPUT_VERSION = "original_navsim_final_timestep_triplet_v1"
RECEIVER_INPUT_VERSION = (
    "projected_receiver_ellipse_cross_or_contact_only_fallback_v3"
)
SELECTION_VERSION = (
    "largest_visible_body_receiver_or_contact_only_fallback_v2"
)
CONTACT_ONLY_FALLBACK_POLICY_VERSION = (
    "receiver_outside_image_contact_only_v1"
)
CONTACT_ONLY_FALLBACK_STATUS = "contact_only_fallback"
CONTACT_ONLY_FALLBACK_REASON = (
    "receiver_outside_image_while_body_proxy_visible"
)
MARKER_LENGTH_M = 0.62
MARKER_WIDTH_M = 0.42
MARKER_LIFT_M = 0.012
MARKER_SAMPLE_COUNT = 48
CAMERA_TIE_ORDER = ("CAM_F0", "CAM_L0", "CAM_R0")
DIRECT_SHADOW_PRESETS = {
    "contact_only_shadow": {
        "strength": "contact_only",
        "softness": "soft",
    },
    "moderate_directional_shadow": {
        "strength": "medium",
        "softness": "medium",
    },
    "clear_directional_shadow": {
        "strength": "strong",
        "softness": "hard",
    },
}

_FINAL_ARGS = None
_FAST = None
_PIPELINE_DIR = None


def set_final_context(args, fast, pipeline_dir: Path) -> None:
    """Set compatibility context for the temporary legacy wrapper."""
    global _FINAL_ARGS, _FAST, _PIPELINE_DIR
    _FINAL_ARGS = args
    _FAST = fast
    _PIPELINE_DIR = Path(pipeline_dir).resolve()


def _instructions_path(pipeline_dir: Path) -> Path:
    return (
        Path(pipeline_dir).resolve().parent
        / "agent_inference_alt"
        / "instructions.md"
    )


def _write_json(path: Path, value: Any, fast) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=fast.jsonable) + "\n"
    )
    os.replace(temporary, path)


def _validate_numeric_args(args, fast) -> None:
    fast.placement_request(args)
    if args.speed_mps <= 0:
        raise ValueError("fixed pedestrian speed must be positive")
    if args.gait_frame_advance <= 0:
        raise ValueError("--gait-frame-advance must be positive")
    if args.gait_timestep_s <= 0:
        raise ValueError("--gait-timestep-s must be positive")
    if args.stride_length_m is not None and args.stride_length_m <= 0:
        raise ValueError("--stride-length-m must be positive")
    if not 0.0 <= args.max_ground_tilt_deg <= 15.0:
        raise ValueError("--max-ground-tilt-deg must be in [0, 15]")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")


def _light_asset(path: Path, args, fast) -> dict:
    blend_path = path.resolve()
    sidecar_path = blend_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Generated pedestrian metadata sidecar not found: {sidecar_path}"
        )
    metadata = json.loads(sidecar_path.read_text())
    required = {
        "id",
        "character_height_m",
        "animation_version",
        "animation_frame_start",
        "walk_cycle_frames",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Asset metadata {sidecar_path} is missing {missing}")
    height = float(metadata["character_height_m"])
    if not 0.5 <= height <= 2.3:
        raise ValueError(
            f"Unreasonable asset height {height:.3f} m in {sidecar_path}"
        )
    cycle = int(metadata["walk_cycle_frames"])
    start = int(metadata["animation_frame_start"])
    if cycle <= 0:
        raise ValueError(f"Invalid walk-cycle length {cycle} in {sidecar_path}")
    action_range = (start, start + cycle)
    unavailable = [
        frame
        for frame in fast.NAVSIM_POSE_SEQUENCE
        if not action_range[0] <= frame < action_range[1]
    ]
    if unavailable:
        raise ValueError(
            f"Requested pose frames {unavailable} are outside asset action "
            f"range {action_range[0]}..{action_range[1] - 1} in {blend_path}"
        )
    if args.stride_length_m is None:
        stride = (
            cycle
            * args.speed_mps
            * args.gait_timestep_s
            / args.gait_frame_advance
        )
        stride_source = "derived_from_frame_advance"
    else:
        stride = float(args.stride_length_m)
        stride_source = "explicit"
    declared_actor_type = str(metadata.get("actor_type", "pedestrian")).lower()
    is_pedestrian = declared_actor_type in {"", "human", "pedestrian"}
    actor_geometry = None
    if is_pedestrian:
        actor_geometry = fast.normalize_pedestrian_geometry(
            metadata.get("actor_geometry"),
            expected_asset_id=str(metadata["id"]),
        )
    stat = blend_path.stat()
    result = {
        "path": blend_path,
        # Preview startup intentionally avoids hashing multi-GB .blend pools.
        # Size/mtime plus the hashed semantic sidecar are checked again before
        # final rendering; the fast renderer then computes its normal SHA-256.
        "preview_asset_identity": {
            "path": str(blend_path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "metadata_path": str(sidecar_path.resolve()),
            "metadata_sha256": fast.file_sha256(sidecar_path),
        },
        "metadata_path": sidecar_path.resolve(),
        "metadata": metadata,
        "id": metadata["id"],
        "actor_height_m": height,
        "animation_version": metadata["animation_version"],
        "action_range": action_range,
        "walk_cycle_frames": cycle,
        "stride_length_m": float(stride),
        "stride_length_source": stride_source,
        "actor_type": "pedestrian" if is_pedestrian else declared_actor_type,
    }
    if actor_geometry is not None:
        result["actor_geometry"] = actor_geometry
    return result


def _load_light_assets(args, fast) -> List[dict]:
    asset_dir = args.asset_dir.expanduser().resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Pedestrian asset directory not found: {asset_dir}")
    paths = sorted(path for path in asset_dir.glob("*.blend") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .blend assets found directly under: {asset_dir}")
    return [_light_asset(path, args, fast) for path in paths]


def _asset_record(asset: dict) -> dict:
    record = {
        **asset["preview_asset_identity"],
        "id": asset["id"],
        "actor_height_m": asset["actor_height_m"],
        "animation_version": asset["animation_version"],
        "action_range": list(asset["action_range"]),
        "walk_cycle_frames": asset["walk_cycle_frames"],
        "stride_length_m": asset["stride_length_m"],
        "stride_length_source": asset["stride_length_source"],
        "actor_type": asset.get("actor_type", "pedestrian"),
    }
    if "actor_geometry" in asset:
        record["actor_geometry"] = asset["actor_geometry"]
    return record


def _placement_config(
    args,
    fast,
    tokens: Sequence[str],
    assets: Sequence[dict],
    front_bumper_offset_m: float,
) -> dict:
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "fast_pipeline_source_sha256": fast.file_sha256(Path(fast.__file__)),
        "scene_filter": str(args.scene_filter.expanduser().resolve()),
        "scene_filter_sha256": fast.file_sha256(args.scene_filter),
        "data_root": str(args.data_root.expanduser().resolve()),
        "split": args.split,
        "output_root": str(args.output_root.expanduser().resolve()),
        "run_name": args.run_name,
        "asset_directory": str(args.asset_dir.expanduser().resolve()),
        "asset_selection": "blake2b_token_seed_v1",
        "asset_pool_preview_identity": [_asset_record(asset) for asset in assets],
        **fast.placement_request(args),
        **fast.direction_classification_provenance(args),
        "front_bumper_offset_m": float(front_bumper_offset_m),
        "speed_mps": float(args.speed_mps),
        "seed": int(args.seed),
        "stride_length_m": (
            None if args.stride_length_m is None else float(args.stride_length_m)
        ),
        "gait_frame_advance": int(args.gait_frame_advance),
        "gait_timestep_s": float(args.gait_timestep_s),
        "max_ground_tilt_degrees": float(args.max_ground_tilt_deg),
        "grounding_method": fast.GROUNDING_METHOD,
        "max_samples": args.max_samples,
        "selected_tokens": list(tokens),
        "selected_tokens_sha256": fast.canonical_hash(list(tokens)),
        "cameras": list(fast.CAMERA_ORDER),
        "history_frames": list(fast.HISTORY_FRAME_INDICES),
        "marker": {
            "shape": "road_aligned_ellipse_outline",
            "length_m": MARKER_LENGTH_M,
            "width_m": MARKER_WIDTH_M,
            "lift_above_fitted_ground_m": MARKER_LIFT_M,
            "center_indicator": "cross",
        },
        "deterministic_contact_only_fallback": {
            "policy_version": CONTACT_ONLY_FALLBACK_POLICY_VERSION,
            "trigger": CONTACT_ONLY_FALLBACK_REASON,
            "scope": "per_token_history_frame",
        },
        "scene_input_version": SCENE_INPUT_VERSION,
        "receiver_input_version": RECEIVER_INPUT_VERSION,
        "preview_selection_version": SELECTION_VERSION,
    }
    config["placement_config_sha256"] = fast.canonical_hash(config)
    return config


def _placement_core(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key != "placement_config_sha256"
    }


def _validate_existing_config(path: Path, expected: dict, fast) -> None:
    if not path.is_file():
        return
    existing = json.loads(path.read_text())
    if existing.get("placement_config_sha256") != expected[
        "placement_config_sha256"
    ] or fast.canonical_hash(_placement_core(existing)) != expected[
        "placement_config_sha256"
    ]:
        raise RuntimeError(
            f"Receiver-preview configuration conflicts with {path}; choose a "
            "different --run-name"
        )


def _polygon_mask(
    points_lidar: np.ndarray,
    calibration: dict,
    width: int,
    height: int,
    fast,
) -> Tuple[np.ndarray, int, Optional[np.ndarray], np.ndarray, bool]:
    uv, depth = fast.geometry.project_distorted(points_lidar, calibration)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0.2)
    mask = np.zeros((height, width), dtype=np.uint8)
    if not np.all(valid):
        return mask, 0, None, uv, False
    # OpenCV clips polygons at the image boundary.  A generous clamp protects
    # its integer conversion from pathological off-camera distortion values.
    clipped = np.column_stack(
        [
            np.clip(uv[:, 0], -4 * width, 5 * width),
            np.clip(uv[:, 1], -4 * height, 5 * height),
        ]
    )
    polygon = np.rint(clipped).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask, int(np.count_nonzero(mask)), polygon, uv, True


def _body_proxy_mask(
    bbox_lidar: np.ndarray,
    calibration: dict,
    width: int,
    height: int,
    fast,
) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not np.any(fast.geometry.camera_frustum_mask(bbox_lidar, calibration)):
        return mask, 0, None
    uv, depth = fast.geometry.project_distorted(bbox_lidar, calibration)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0.2)
    if np.count_nonzero(valid) < 3:
        return mask, 0, None
    clipped = np.column_stack(
        [
            np.clip(uv[valid, 0], -4 * width, 5 * width),
            np.clip(uv[valid, 1], -4 * height, 5 * height),
        ]
    )
    hull = cv2.convexHull(np.rint(clipped).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask, int(np.count_nonzero(mask)), hull.reshape(-1, 2)


def _project_point(point: np.ndarray, calibration: dict, fast) -> Optional[List[float]]:
    uv, depth = fast.geometry.project_distorted(point[None], calibration)
    if (
        not np.isfinite(uv[0]).all()
        or not np.isfinite(depth[0])
        or depth[0] <= 0.2
    ):
        return None
    return [float(uv[0, 0]), float(uv[0, 1])]


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    selected = mask > 0
    if not np.any(selected):
        return
    original = image[selected].astype(np.float32)
    image[selected] = np.clip(
        (1.0 - alpha) * original + alpha * np.asarray(color, dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)


def _draw_marker(
    source: np.ndarray,
    polygon: np.ndarray,
    center: Optional[List[float]],
) -> np.ndarray:
    image = source.copy()
    height, width = image.shape[:2]
    halo = np.zeros((height, width), dtype=np.uint8)
    accent = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(halo, [polygon], True, 255, thickness=7, lineType=cv2.LINE_AA)
    cv2.polylines(accent, [polygon], True, 255, thickness=3, lineType=cv2.LINE_AA)
    _blend_mask(image, halo, (12, 20, 24), 0.72)
    _blend_mask(image, accent, (0, 238, 255), 0.92)

    if center is not None:
        centre_px = tuple(
            np.rint(
                np.clip(center, [-4 * width, -4 * height], [5 * width, 5 * height])
            ).astype(int)
        )
        cross_halo = np.zeros((height, width), dtype=np.uint8)
        cross = np.zeros((height, width), dtype=np.uint8)
        cv2.drawMarker(
            cross_halo,
            centre_px,
            255,
            markerType=cv2.MARKER_CROSS,
            markerSize=15,
            thickness=5,
            line_type=cv2.LINE_AA,
        )
        cv2.drawMarker(
            cross,
            centre_px,
            255,
            markerType=cv2.MARKER_CROSS,
            markerSize=13,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        _blend_mask(image, cross_halo, (12, 20, 24), 0.65)
        _blend_mask(image, cross, (0, 238, 255), 0.90)

    return image


def _marker_points(actor_lidar: np.ndarray, basis: np.ndarray, lateral_sign: float):
    up = np.asarray(basis[:, 2], dtype=np.float64)
    walking = np.asarray(basis[:, 1], dtype=np.float64) * float(lateral_sign)
    walking /= np.linalg.norm(walking)
    across = np.cross(up, walking)
    across /= np.linalg.norm(across)
    centre = actor_lidar + MARKER_LIFT_M * up
    angles = np.linspace(0.0, 2.0 * math.pi, MARKER_SAMPLE_COUNT, endpoint=False)
    points = (
        centre[None]
        + np.cos(angles)[:, None] * (MARKER_LENGTH_M / 2.0) * walking[None]
        + np.sin(angles)[:, None] * (MARKER_WIDTH_M / 2.0) * across[None]
    )
    return points, centre


def _prepare_geometry(args, fast, loader, token: str, front_bumper: float, asset: dict):
    scene_frames = loader.scene_frames_dicts[token]
    frames = scene_frames[: fast.EXPECTED_HISTORY_FRAMES]
    if len(frames) != fast.EXPECTED_HISTORY_FRAMES:
        raise fast.TokenFailure(
            "missing_sensor",
            "expected "
            f"{fast.EXPECTED_HISTORY_FRAMES} history frames, got {len(frames)}",
        )
    for frame in frames:
        for camera in fast.CAMERA_ORDER:
            source = fast.frame_source_path(args.data_root, args.split, frame, camera)
            if not source.is_file():
                raise fast.TokenFailure(
                    "missing_sensor", f"missing {camera} image: {source}"
                )
    timestamps = np.asarray(
        [frame["timestamp"] for frame in frames], dtype=np.float64
    ) / 1e6
    if np.any(np.diff(timestamps) <= 0):
        raise fast.TokenFailure(
            "invalid_timestamps", f"non-increasing timestamps: {timestamps.tolist()}"
        )
    direction, lateral_sign, actor_heading = fast.crossing_policy(
        token,
        args.direction_classification_labels,
    )
    trajectory = fast.resolve_crossing_trajectory(
        scene_frames,
        timestamps,
        timestep=args.timestep,
        front_bumper_offset_m=front_bumper,
        lateral_sign=lateral_sign,
        speed_mps=args.speed_mps,
    )
    placement = trajectory["placement"]
    animation = fast.animation_frames(
        timestamps,
        args.speed_mps,
        asset["stride_length_m"],
        asset["action_range"][0],
        asset["action_range"][1],
        fast.NAVSIM_POSE_SEQUENCE[0],
    )
    actor_current_xys = trajectory["actor_current_xys"]
    actor_global_xys = trajectory["actor_global_xys"]
    actor_lidar_xys = [
        fast.inverse_transform_xy_planar(
            np.asarray(frame["lidar2global"], dtype=np.float64),
            actor_global_xy,
        )
        for frame, actor_global_xy in zip(frames, actor_global_xys)
    ]
    frame_points = [
        fast.geometry.load_lidar_xyz(args.data_root, args.split, frame)
        for frame in frames
    ]
    (
        ground_planes,
        ground_diagnostics,
        strict_ground_failures,
    ) = fast.fit_ground_sequence_best_effort(
        frames,
        frame_points,
        actor_lidar_xys,
        actor_current_xys,
    )

    states = []
    for index, (
        frame,
        animation_frame,
        actor_xy,
        plane,
        diagnostics,
    ) in enumerate(
        zip(
            frames,
            animation,
            actor_lidar_xys,
            ground_planes,
            ground_diagnostics,
        )
    ):
        lidar_to_global = np.asarray(frame["lidar2global"], dtype=np.float64)
        actor = np.array(
            [actor_xy[0], actor_xy[1], fast.geometry.ground_z(plane, *actor_xy)],
            dtype=np.float64,
        )
        gravity_up = lidar_to_global[:3, :3].T @ np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        basis = fast.road_basis_aligned_to_global_forward(
            plane,
            lidar_to_global,
            trajectory["reference_vehicle_forward_global_xy"],
            args.max_ground_tilt_deg,
        )
        direction_lidar = basis[:, 1] * lateral_sign
        yaw = math.atan2(direction_lidar[1], direction_lidar[0]) + math.pi / 2.0
        diagnostics.update(
            {
                "frame_index": index,
                "plane_z_ax_by_c_lidar": plane.tolist(),
                "gravity_up_lidar": gravity_up.tolist(),
                "actor_ground_contact_residual_m": float(
                    actor[2]
                    - fast.geometry.ground_z(plane, actor[0], actor[1])
                ),
            }
        )
        states.append(
            {
                "frame_index": index,
                "actor_lidar": actor,
                "road_plane_lidar": plane,
                "road_basis_lidar": basis,
                "ground_diagnostics": diagnostics,
                "yaw_lidar": yaw,
                "animation_frame": int(animation_frame),
            }
        )
    ground_failure = fast.ground_fit_potential_failure(
        token,
        strict_ground_failures,
        ground_diagnostics,
    )
    return {
        "frames": frames,
        "timestamps_s": timestamps,
        "direction_classification": args.direction_classification_labels[token],
        "direction": direction,
        "lateral_sign": lateral_sign,
        "actor_heading": float(actor_heading),
        "lateral_positions_m": trajectory["crossing_offsets_m"],
        "crossing_offsets_m": trajectory["crossing_offsets_m"],
        "walking_velocity_global_xy_mps": trajectory[
            "walking_velocity_global_xy_mps"
        ],
        "anchor_global_xy": trajectory["anchor_global_xy"],
        "anchor_current_ego_xy": trajectory["anchor_current_ego_xy"],
        "placement": placement,
        **(
            {
                "longitudinal_x_from_rear_axle_m": placement[
                    "longitudinal_x_from_rear_axle_m"
                ]
            }
            if "longitudinal_x_from_rear_axle_m" in placement
            else {}
        ),
        "states": states,
        "potential_failures": (
            [ground_failure] if ground_failure is not None else []
        ),
    }


def _source_descriptor(path: Path, camera: str, frame_index: int, fast) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "camera": camera,
        "frame_index": int(frame_index),
        "source_path": str(path.resolve()),
        "image_sha256": fast.file_sha256(path),
        "width": int(width),
        "height": int(height),
    }


def _select_camera(records: dict) -> Optional[str]:
    eligible = [
        camera
        for camera in CAMERA_TIE_ORDER
        if records[camera]["projected_body_pixel_count"] > 0
        and records[camera]["receiver_marker_pixel_count"] > 0
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda camera: (
            records[camera]["projected_body_pixel_count"],
            1 if camera == "CAM_F0" else 0,
            -CAMERA_TIE_ORDER.index(camera),
        ),
    )


def _select_contact_only_fallback_camera(records: dict) -> Optional[str]:
    """Select a useful body view whose receiver projects fully off-image."""
    eligible = [
        camera
        for camera in CAMERA_TIE_ORDER
        if records[camera]["projected_body_pixel_count"] > 0
        and records[camera]["receiver_projection_valid"]
        and records[camera]["receiver_marker_pixel_count"] == 0
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda camera: (
            records[camera]["projected_body_pixel_count"],
            1 if camera == "CAM_F0" else 0,
            -CAMERA_TIE_ORDER.index(camera),
        ),
    )


def _token_metadata(
    args,
    fast,
    token: str,
    geometry: dict,
    asset: dict,
    run_dir: Path,
    config: dict,
) -> dict:
    preview_selections = {}
    scene_inputs = []
    frames = geometry["frames"]
    states = geometry["states"]
    actor_height = float(asset["actor_height_m"])
    actor_heading = float(geometry["actor_heading"])

    for camera in fast.CAMERA_ORDER:
        source = fast.frame_source_path(args.data_root, args.split, frames[-1], camera)
        scene_inputs.append(_source_descriptor(source, camera, 3, fast))

    for state, frame in zip(states, frames):
        marker_points, marker_center = _marker_points(
            state["actor_lidar"], state["road_basis_lidar"], geometry["lateral_sign"]
        )
        bbox = fast.render_utils.actor_bbox_lidar(
            state["actor_lidar"],
            state["road_basis_lidar"],
            actor_heading,
            asset["actor_geometry"]["local_bounds_m"],
        )
        candidates = {}
        render_payload = {}
        for camera in fast.CAMERA_ORDER:
            source = fast.frame_source_path(
                args.data_root, args.split, frame, camera
            )
            with Image.open(source) as image:
                width, height = image.size
            calibration = frame["cams"][camera]
            (
                receiver_mask,
                receiver_count,
                receiver_polygon,
                receiver_uv,
                receiver_projection_valid,
            ) = (
                _polygon_mask(
                    marker_points, calibration, width, height, fast
                )
            )
            _body_mask, body_count, body_hull = _body_proxy_mask(
                bbox, calibration, width, height, fast
            )
            centre_uv = _project_point(marker_center, calibration, fast)
            candidates[camera] = {
                "projected_body_pixel_count": int(body_count),
                "receiver_marker_pixel_count": int(receiver_count),
                "receiver_projection_valid": bool(receiver_projection_valid),
                "receiver_projected_bounds": (
                    [
                        float(np.min(receiver_uv[:, 0])),
                        float(np.min(receiver_uv[:, 1])),
                        float(np.max(receiver_uv[:, 0])),
                        float(np.max(receiver_uv[:, 1])),
                    ]
                    if receiver_projection_valid
                    else None
                ),
                "projected_body_hull": (
                    None if body_hull is None else body_hull.tolist()
                ),
            }
            render_payload[camera] = {
                "source": source,
                "receiver_mask": receiver_mask,
                "receiver_polygon": receiver_polygon,
                "receiver_uv": receiver_uv,
                "center_uv": centre_uv,
            }
        selected = _select_camera(candidates)
        frame_index = int(state["frame_index"])
        if selected is None:
            body_visible = any(
                record["projected_body_pixel_count"] > 0
                for record in candidates.values()
            )
            fallback_camera = _select_contact_only_fallback_camera(candidates)
            if fallback_camera is not None:
                payload = render_payload[fallback_camera]
                source_descriptor = _source_descriptor(
                    payload["source"], fallback_camera, frame_index, fast
                )
                mechanical = {
                    "status": CONTACT_ONLY_FALLBACK_STATUS,
                    "reason": CONTACT_ONLY_FALLBACK_REASON,
                    "frame_index": frame_index,
                    "reference_camera": fallback_camera,
                    "source_image": source_descriptor,
                    "camera_projection_counts": candidates,
                    "selected_projected_body_pixel_count": candidates[
                        fallback_camera
                    ]["projected_body_pixel_count"],
                    "selected_receiver_marker_pixel_count": 0,
                    "receiver_center_pixel": payload["center_uv"],
                    "receiver_projected_bounds": candidates[fallback_camera][
                        "receiver_projected_bounds"
                    ],
                    "deterministic_shadow_override": "contact_only_shadow",
                    "decision_source": CONTACT_ONLY_FALLBACK_POLICY_VERSION,
                    "selection_version": SELECTION_VERSION,
                }
                mechanical["preview_selection_sha256"] = fast.canonical_hash(
                    mechanical
                )
                preview_selections[str(frame_index)] = mechanical
                continue
            if body_visible:
                raise fast.TokenFailure(
                    "receiver_projection_failed",
                    f"{token}/{frame_index}: a body proxy is visible but the "
                    "stance receiver has no valid off-image projection in any "
                    "supported camera",
                )
            preview_selections[str(frame_index)] = {
                "status": "not_published",
                "reason": "no_camera_with_visible_body_proxy_and_receiver_marker",
                "camera_projection_counts": candidates,
                "selection_version": SELECTION_VERSION,
            }
            continue
        payload = render_payload[selected]
        polygon = payload["receiver_polygon"]
        if polygon is None:
            raise fast.TokenFailure(
                "receiver_projection_failed",
                f"{token}/{frame_index}/{selected}: selected marker has no polygon",
            )
        source_array = np.asarray(Image.open(payload["source"]).convert("RGB"))
        preview = _draw_marker(
            source_array,
            polygon,
            payload["center_uv"],
        )
        relative = Path(token) / selected / f"{frame_index}.png"
        preview_path = run_dir / "preview_images" / relative
        mask_path = run_dir / "preview_masks" / relative
        fast.save_png(preview_path, preview)
        fast.save_png(mask_path, payload["receiver_mask"])
        source_descriptor = _source_descriptor(
            payload["source"], selected, frame_index, fast
        )
        mechanical = {
            "status": "published",
            "frame_index": frame_index,
            "reference_camera": selected,
            "relative_path": str(relative),
            "preview_path": str(preview_path.resolve()),
            "preview_sha256": fast.file_sha256(preview_path),
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": fast.file_sha256(mask_path),
            "source_image": source_descriptor,
            "camera_projection_counts": candidates,
            "selected_projected_body_pixel_count": candidates[selected][
                "projected_body_pixel_count"
            ],
            "selected_receiver_marker_pixel_count": candidates[selected][
                "receiver_marker_pixel_count"
            ],
            "receiver_outline_pixels": polygon.tolist(),
            "receiver_center_pixel": payload["center_uv"],
            "selection_version": SELECTION_VERSION,
        }
        mechanical["preview_selection_sha256"] = fast.canonical_hash(mechanical)
        preview_selections[str(frame_index)] = mechanical

    if not any(
        record.get("status") in {"published", CONTACT_ONLY_FALLBACK_STATUS}
        for record in preview_selections.values()
    ):
        raise fast.TokenFailure(
            "no_camera_visibility",
            "body proxy and receiver marker are outside all supported views",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "document_role": "receiver_preview_metadata",
        "pipeline_version": PIPELINE_VERSION,
        "token": token,
        "placement_config_sha256": config["placement_config_sha256"],
        "asset": _asset_record(asset),
        "actor_type": asset.get("actor_type", "pedestrian"),
        **(
            {"pedestrian_geometry": asset["actor_geometry"]}
            if asset.get("actor_type", "pedestrian") == "pedestrian"
            else {}
        ),
        "crossing": {
            "classification": geometry["direction_classification"],
            "direction": geometry["direction"],
            "lateral_sign": float(geometry["lateral_sign"]),
            "lateral_positions_m": geometry["lateral_positions_m"].tolist(),
            "crossing_offsets_m": geometry["crossing_offsets_m"].tolist(),
            "final_crossing_offset_m": float(
                geometry["crossing_offsets_m"][-1]
            ),
            "walking_velocity_global_xy_mps": geometry[
                "walking_velocity_global_xy_mps"
            ].tolist(),
            "speed_mps": float(args.speed_mps),
            "placement": geometry["placement"],
            **fast.placement_metadata_fields(geometry["placement"]),
        },
        "timestamps_s": geometry["timestamps_s"].tolist(),
        "actor_height_m": actor_height,
        "actor_states": [
            {
                "frame_index": int(state["frame_index"]),
                "actor_lidar": state["actor_lidar"].tolist(),
                "road_plane_lidar": state["road_plane_lidar"].tolist(),
                "road_basis_lidar": state["road_basis_lidar"].tolist(),
                "yaw_lidar": float(state["yaw_lidar"]),
                "animation_frame": int(state["animation_frame"]),
                "ground_diagnostics": state["ground_diagnostics"],
            }
            for state in states
        ],
        "scene_input_images": scene_inputs,
        "preview_selections": preview_selections,
        "grounding_method": fast.GROUNDING_METHOD,
        "potential_failures": geometry["potential_failures"],
        "scene_input_version": SCENE_INPUT_VERSION,
        "receiver_input_version": RECEIVER_INPUT_VERSION,
        "preview_selection_version": SELECTION_VERSION,
    }


def _template_record(metadata: dict, metadata_path: Path, fast) -> dict:
    timesteps = {}
    for frame_key, selection in metadata["preview_selections"].items():
        status = selection.get("status")
        if status == CONTACT_ONLY_FALLBACK_STATUS:
            timesteps[frame_key] = {
                "reference_camera": selection["reference_camera"],
                "source_image_sha256": selection["source_image"][
                    "image_sha256"
                ],
                "preview_selection_sha256": selection[
                    "preview_selection_sha256"
                ],
                "evidence": [],
                "recommended_shadow": "contact_only_shadow",
                "deterministic_shadow_override": "contact_only_shadow",
                "decision_source": CONTACT_ONLY_FALLBACK_POLICY_VERSION,
                "fallback_reason": CONTACT_ONLY_FALLBACK_REASON,
            }
            continue
        if status != "published":
            continue
        timesteps[frame_key] = {
            "reference_camera": selection["reference_camera"],
            "relative_path": selection["relative_path"],
            "image_sha256": selection["preview_sha256"],
            "preview_selection_sha256": selection[
                "preview_selection_sha256"
            ],
            "evidence": [],
            "receiver_visibility": None,
            "confidence": None,
        }
    return {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": fast.file_sha256(metadata_path),
        "scene": {
            "reference_frame": fast.EXPECTED_HISTORY_FRAMES - 1,
            "input_images": metadata["scene_input_images"],
            "evidence": [],
            "direct_sunlight_condition": None,
            "confidence": None,
        },
        "timesteps": timesteps,
    }


def run_receiver_preview(args, fast, pipeline_dir: Path) -> None:
    _validate_numeric_args(args, fast)
    args.scene_filter = args.scene_filter.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.asset_dir = args.asset_dir.expanduser().resolve()
    if not args.scene_filter.is_file():
        raise FileNotFoundError(f"Scene filter does not exist: {args.scene_filter}")

    fast.configure_navsim(args.data_root)
    filter_config = yaml.safe_load(args.scene_filter.read_text())
    loader, tokens = fast.build_loader(args, filter_config)
    fast.validate_direction_classifications(
        tokens,
        args.direction_classification_labels,
    )
    assets = _load_light_assets(args, fast)
    run_dir = args.output_root / args.scene_filter.stem / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_config.json").exists():
        raise RuntimeError(
            f"{run_dir} already contains a final compositor run_config.json; "
            "choose a different --run-name for receiver previews"
        )
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    front_bumper = float(get_pacifica_parameters().front_length)
    config = _placement_config(args, fast, tokens, assets, front_bumper)
    config_path = run_dir / "receiver_preview_config.json"
    _validate_existing_config(config_path, config, fast)
    _write_json(config_path, config, fast)
    labels = {}
    failures = {}
    potential_failures = []
    contact_only_fallback_tokens = set()
    contact_only_fallback_timesteps = 0
    successes = 0
    for index, token in enumerate(tokens):
        # Keep marker provenance separate from the final compositor's metadata
        # success records.  A failed final render must never inherit a preview
        # JSON that aggregate_results() could mistake for a completed token.
        metadata_path = run_dir / "receiver_preview_metadata" / f"{token}.json"
        failure_path = run_dir / "receiver_preview_failures" / f"{token}.json"
        if (
            failure_path.is_file()
            and not (args.retry_failures or args.overwrite or args.refinalize)
        ):
            failure = json.loads(failure_path.read_text())
            failures[token] = failure
            print(f"[{index + 1}/{len(tokens)}] cached failure {token}")
            continue
        try:
            cached = False
            if metadata_path.is_file() and not (args.overwrite or args.refinalize):
                metadata = json.loads(metadata_path.read_text())
                cached = (
                    metadata.get("placement_config_sha256")
                    == config["placement_config_sha256"]
                    and all(
                        selection.get("status") != "published"
                        or (
                            (run_dir / "preview_images" / selection["relative_path"]).is_file()
                            and fast.file_sha256(
                                run_dir
                                / "preview_images"
                                / selection["relative_path"]
                            )
                            == selection.get("preview_sha256")
                            and (run_dir / "preview_masks" / selection["relative_path"]).is_file()
                            and fast.file_sha256(
                                run_dir
                                / "preview_masks"
                                / selection["relative_path"]
                            )
                            == selection.get("mask_sha256")
                        )
                        for selection in metadata.get("preview_selections", {}).values()
                    )
                )
            if not cached:
                asset = fast.select_asset(token, args.seed, assets)
                geometry = _prepare_geometry(
                    args, fast, loader, token, front_bumper, asset
                )
                metadata = _token_metadata(
                    args, fast, token, geometry, asset, run_dir, config
                )
                _write_json(metadata_path, metadata, fast)
            labels[token] = _template_record(metadata, metadata_path, fast)
            fallback_count = sum(
                selection.get("status") == CONTACT_ONLY_FALLBACK_STATUS
                for selection in metadata.get("preview_selections", {}).values()
                if isinstance(selection, dict)
            )
            if fallback_count:
                contact_only_fallback_tokens.add(token)
                contact_only_fallback_timesteps += fallback_count
            potential_failures.extend(
                fast.normalize_potential_failures(
                    metadata.get("potential_failures", [])
                )
            )
            successes += 1
            if failure_path.exists():
                failure_path.unlink()
            print(
                f"[{index + 1}/{len(tokens)}] "
                f"{'cached' if cached else 'created'} receiver previews {token}"
            )
        except Exception as exc:
            code = exc.code if isinstance(exc, fast.TokenFailure) else "preview_failed"
            failure = {
                "token": token,
                "code": code,
                "message": str(exc),
                "placement_config_sha256": config["placement_config_sha256"],
            }
            failures[token] = failure
            _write_json(failure_path, failure, fast)
            print(f"[{index + 1}/{len(tokens)}] failed {token}: {code}: {exc}")

    prompt_path = _instructions_path(pipeline_dir)
    template = {
        "schema_version": SCHEMA_VERSION,
        "document_role": "shadow_labels",
        "scene_input_version": SCENE_INPUT_VERSION,
        "receiver_input_version": RECEIVER_INPUT_VERSION,
        "preview_selection_version": SELECTION_VERSION,
        "deterministic_contact_only_fallback_policy": (
            CONTACT_ONLY_FALLBACK_POLICY_VERSION
        ),
        "input_root": str((run_dir / "preview_images").resolve()),
        "metadata_root": str((run_dir / "receiver_preview_metadata").resolve()),
        "placement_config_path": str(config_path.resolve()),
        "placement_config_sha256": config["placement_config_sha256"],
        "prompt_path": str(prompt_path),
        "prompt_sha256": (
            fast.file_sha256(prompt_path) if prompt_path.is_file() else None
        ),
        "model": None,
        "model_revision": None,
        "labels": labels,
        "failures": failures,
    }
    template_path = run_dir / "receiver_label_template.json"
    _write_json(template_path, template, fast)
    potential_failures = fast.deduplicate_potential_failures(
        potential_failures
    )
    _write_json(
        run_dir / "receiver_preview_potential_failures.json",
        potential_failures,
        fast,
    )
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "selected_tokens": len(tokens),
        "successful_tokens": successes,
        "failed_tokens": len(failures),
        "contact_only_fallback_tokens": len(contact_only_fallback_tokens),
        "contact_only_fallback_timesteps": contact_only_fallback_timesteps,
        "potential_failure_tokens": len(
            {item["token"] for item in potential_failures}
        ),
        "run_dir": str(run_dir),
        "label_template": str(template_path),
        "instructions": str(prompt_path),
    }
    _write_json(run_dir / "receiver_preview_summary.json", summary, fast)
    print(
        f"Finished: {successes} successful, {len(failures)} failed\n"
        f"Selective contact-only fallbacks: "
        f"{contact_only_fallback_timesteps} timestep(s) across "
        f"{len(contact_only_fallback_tokens)} token(s)\n"
        f"Output: {run_dir}\n"
        f"Label template: {template_path}"
    )


def _valid_evidence(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _same_mechanical(actual: dict, expected: dict, keys: Sequence[str]) -> List[str]:
    return [key for key in keys if actual.get(key) != expected.get(key)]


def load_receiver_shadow_labels(
    path: Path,
    args=None,
    fast=None,
    pipeline_dir: Optional[Path] = None,
):
    """Validate direct labels and adapt them to the fast renderer's API."""
    if args is None:
        args = _FINAL_ARGS
    if fast is None:
        fast = _FAST
    if pipeline_dir is None:
        pipeline_dir = _PIPELINE_DIR
    if args is None or fast is None or pipeline_dir is None:
        raise RuntimeError("Receiver shadow-label loader was not initialized")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Shadow-label file does not exist: {path}")
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read shadow-label JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Shadow-label root must be an object")
    expected_contract = {
        "schema_version": SCHEMA_VERSION,
        "document_role": "direct_shadow_labels",
    }
    errors = [
        f"{key}={document.get(key)!r}, expected {value!r}"
        for key, value in expected_contract.items()
        if document.get(key) != value
    ]

    run_dir = (
        args.output_root.expanduser().resolve()
        / args.scene_filter.stem
        / args.run_name
    )
    expected_input_root = (run_dir / "preview_images").resolve()
    expected_metadata_root = (run_dir / "receiver_preview_metadata").resolve()
    expected_final_metadata_root = (run_dir / "metadata").resolve()
    expected_config_path = (run_dir / "receiver_preview_config.json").resolve()
    expected_template_path = (run_dir / "receiver_label_template.json").resolve()

    source_path_value = document.get("source_template_path")
    source_path = (
        Path(source_path_value).expanduser().resolve()
        if isinstance(source_path_value, str)
        else None
    )
    source_document: dict[str, Any] = {}
    if source_path != expected_template_path:
        errors.append(
            f"source_template_path must resolve to {expected_template_path}"
        )
    elif not source_path.is_file():
        errors.append(f"source receiver template does not exist: {source_path}")
    else:
        try:
            source_value = json.loads(source_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read source receiver template: {exc}")
        else:
            if isinstance(source_value, dict):
                source_document = source_value
            else:
                errors.append("source receiver template root must be an object")
        if document.get("source_template_sha256") != fast.file_sha256(
            source_path
        ):
            errors.append("source receiver template SHA-256 mismatch")

    source_contract = {
        "schema_version": SCHEMA_VERSION,
        "document_role": "shadow_labels",
        "scene_input_version": SCENE_INPUT_VERSION,
        "receiver_input_version": RECEIVER_INPUT_VERSION,
        "preview_selection_version": SELECTION_VERSION,
        "deterministic_contact_only_fallback_policy": (
            CONTACT_ONLY_FALLBACK_POLICY_VERSION
        ),
    }
    if source_document:
        for key, expected in source_contract.items():
            if source_document.get(key) != expected:
                errors.append(
                    f"source template {key}={source_document.get(key)!r}, "
                    f"expected {expected!r}"
                )

    path_fields = {
        "input_root": expected_input_root,
        "metadata_root": expected_metadata_root,
        "placement_config_path": expected_config_path,
    }
    for key, expected in path_fields.items():
        value = source_document.get(key)
        if not isinstance(value, str) or Path(value).expanduser().resolve() != expected:
            errors.append(f"source template {key} must resolve to {expected}")

    if not expected_config_path.is_file():
        errors.append(f"missing receiver preview config: {expected_config_path}")
        config = None
    else:
        try:
            config_value = json.loads(expected_config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read receiver preview config: {exc}")
            config = None
        else:
            config = config_value if isinstance(config_value, dict) else None
            if config is None:
                errors.append("receiver preview config root must be an object")
    if config is not None:
        config_hash = config.get("placement_config_sha256")
        if fast.canonical_hash(_placement_core(config)) != config_hash:
            errors.append("receiver preview config has an invalid self-hash")
        if source_document.get("placement_config_sha256") != config_hash:
            errors.append(
                "source template placement_config_sha256 does not match "
                "preview config"
            )
        current_paths = {
            "scene_filter": str(args.scene_filter.expanduser().resolve()),
            "direction_classifications_path": str(
                args.direction_classifications.expanduser().resolve()
            ),
            "data_root": str(args.data_root.expanduser().resolve()),
            "split": args.split,
            "output_root": str(args.output_root.expanduser().resolve()),
            "run_name": args.run_name,
            "asset_directory": str(args.asset_dir.expanduser().resolve()),
        }
        for key, expected in current_paths.items():
            if config.get(key) != expected:
                errors.append(f"preview config {key}={config.get(key)!r}, expected {expected!r}")
        scalar_fields = {
            **fast.placement_request(args),
            "speed_mps": float(args.speed_mps),
            "seed": int(args.seed),
            "stride_length_m": (
                None if args.stride_length_m is None else float(args.stride_length_m)
            ),
            "gait_frame_advance": int(args.gait_frame_advance),
            "gait_timestep_s": float(args.gait_timestep_s),
            "max_ground_tilt_degrees": float(args.max_ground_tilt_deg),
            "grounding_method": fast.GROUNDING_METHOD,
            "max_samples": args.max_samples,
            "direction_policy": fast.DIRECTION_CLASSIFICATION_POLICY_VERSION,
            "direction_classification_mapping": (
                fast.direction_classification_mapping()
            ),
            "deterministic_contact_only_fallback": {
                "policy_version": CONTACT_ONLY_FALLBACK_POLICY_VERSION,
                "trigger": CONTACT_ONLY_FALLBACK_REASON,
                "scope": "per_token_history_frame",
            },
        }
        for key, expected in scalar_fields.items():
            if config.get(key) != expected:
                errors.append(f"preview config {key}={config.get(key)!r}, expected {expected!r}")
        if args.scene_filter.is_file() and config.get("scene_filter_sha256") != fast.file_sha256(args.scene_filter):
            errors.append("scene-filter contents changed after preview generation")
        if (
            config.get("direction_classifications_sha256")
            != args.direction_classifications_sha256
        ):
            errors.append(
                "direction classifications changed after preview generation"
            )
        if config.get("fast_pipeline_source_sha256") != fast.file_sha256(
            Path(fast.__file__)
        ):
            # The compositor also contains rendering, cache, scheduling, and
            # diagnostics code. Requiring a byte-identical whole file makes
            # safe fixes in those unrelated areas invalidate every expensive
            # receiver-label run. Final preparation recomputes each token's
            # placement and validate_preview_placement() rejects any material
            # difference before a render can be accepted.
            print(
                "WARNING: 3d_compositing_fast_v2 source changed after preview "
                "generation; final placements will be revalidated against "
                "receiver-preview provenance for every token",
                file=sys.stderr,
                flush=True,
            )
        try:
            from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

            current_front_bumper = float(get_pacifica_parameters().front_length)
            if config.get("front_bumper_offset_m") != current_front_bumper:
                errors.append("vehicle front-bumper geometry changed after preview generation")
        except Exception as exc:
            errors.append(f"cannot validate vehicle front-bumper geometry: {exc}")
        try:
            current_assets = [_asset_record(asset) for asset in _load_light_assets(args, fast)]
            if config.get("asset_pool_preview_identity") != current_assets:
                errors.append("asset pool changed after preview generation")
        except Exception as exc:
            errors.append(f"cannot validate preview asset pool: {exc}")

    source_labels = source_document.get("labels")
    if not isinstance(source_labels, dict):
        errors.append("source template labels must be an object")
        source_labels = {}
    source_failures = source_document.get("failures")
    if not isinstance(source_failures, dict):
        errors.append("source template failures must be an object")
        source_failures = {}

    labels = document.get("labels")
    if not isinstance(labels, dict):
        errors.append("labels must be an object")
        labels = {}
    failures = document.get("failures")
    if not isinstance(failures, dict):
        errors.append("failures must be an object")
        failures = {}

    expected_tokens = set(config.get("selected_tokens", [])) if config else set()
    source_label_tokens = set(source_labels)
    source_failure_tokens = set(source_failures)
    if source_label_tokens & source_failure_tokens:
        errors.append("source template labels and failures overlap")
    if source_label_tokens | source_failure_tokens != expected_tokens:
        missing = sorted(
            expected_tokens - source_label_tokens - source_failure_tokens
        )
        extra = sorted(
            (source_label_tokens | source_failure_tokens) - expected_tokens
        )
        errors.append(
            "source template token set mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    label_tokens = set(labels)
    failure_tokens = set(failures)
    if label_tokens & failure_tokens:
        errors.append("direct labels and failures overlap")
    if label_tokens != source_label_tokens:
        missing = sorted(source_label_tokens - label_tokens)
        extra = sorted(label_tokens - source_label_tokens)
        errors.append(
            f"direct label token set mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if failures != source_failures:
        errors.append(
            "direct-label failures differ from source preview failures"
        )

    lookup: Dict[Tuple[str, int], dict] = {}
    for token, token_record in labels.items():
        location = token
        if not isinstance(token_record, dict):
            errors.append(f"{location}: token record must be an object")
            continue
        source_record = source_labels.get(token)
        if not isinstance(source_record, dict):
            errors.append(f"{location}: source token record must be an object")
            continue
        metadata_path_value = source_record.get("metadata_path")
        expected_metadata_path = expected_metadata_root / f"{token}.json"
        if not isinstance(metadata_path_value, str) or Path(metadata_path_value).expanduser().resolve() != expected_metadata_path:
            errors.append(f"{location}: metadata_path must be {expected_metadata_path}")
            continue
        if not expected_metadata_path.is_file():
            errors.append(f"{location}: missing metadata file")
            continue
        if fast.file_sha256(expected_metadata_path) != source_record.get(
            "metadata_sha256"
        ):
            errors.append(f"{location}: metadata SHA-256 mismatch")
            continue
        try:
            metadata_value = json.loads(expected_metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{location}: cannot read metadata: {exc}")
            continue
        if not isinstance(metadata_value, dict):
            errors.append(f"{location}: metadata root must be an object")
            continue
        metadata = metadata_value
        if metadata.get("token") != token:
            errors.append(f"{location}: metadata token mismatch")
        metadata_contract = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "scene_input_version": SCENE_INPUT_VERSION,
            "receiver_input_version": RECEIVER_INPUT_VERSION,
            "preview_selection_version": SELECTION_VERSION,
        }
        for key, expected in metadata_contract.items():
            if metadata.get(key) != expected:
                errors.append(
                    f"{location}: metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if config and metadata.get("placement_config_sha256") != config.get("placement_config_sha256"):
            errors.append(f"{location}: metadata belongs to a different placement config")
        crossing = metadata.get("crossing")
        preview_placement = (
            crossing.get("placement") if isinstance(crossing, dict) else None
        )
        if not isinstance(preview_placement, dict):
            errors.append(f"{location}: metadata is missing placement provenance")
        for scene_input in metadata.get("scene_input_images", []):
            if not isinstance(scene_input, dict):
                errors.append(f"{location}: invalid scene input descriptor")
                continue
            source_value = scene_input.get("source_path")
            source = Path(source_value).expanduser().resolve() if isinstance(source_value, str) else None
            if (
                source is None
                or not source.is_file()
                or fast.file_sha256(source) != scene_input.get("image_sha256")
            ):
                errors.append(
                    f"{location}: original scene input is missing or changed: "
                    f"{source_value!r}"
                )

        timesteps = token_record.get("timesteps")
        source_timesteps = source_record.get("timesteps")
        expected_selections = {
            key: value
            for key, value in metadata.get("preview_selections", {}).items()
            if isinstance(value, dict)
            and value.get("status")
            in {"published", CONTACT_ONLY_FALLBACK_STATUS}
        }
        if not isinstance(timesteps, dict):
            errors.append(f"{location}/timesteps: must be an object")
            continue
        if not isinstance(source_timesteps, dict):
            errors.append(f"{location}: source timesteps must be an object")
            continue
        if set(source_timesteps) != set(expected_selections):
            errors.append(
                f"{location}: source timestep keys differ from published "
                "previews and deterministic fallbacks"
            )
        if set(timesteps) != set(expected_selections):
            errors.append(
                f"{location}/timesteps: keys differ from published previews "
                "and deterministic fallbacks"
            )
        for frame_key, expected in expected_selections.items():
            record = timesteps.get(frame_key)
            source_timestep = source_timesteps.get(frame_key)
            item_location = f"{location}/{frame_key}"
            if not isinstance(record, dict):
                errors.append(f"{item_location}: label must be an object")
                continue
            if not isinstance(source_timestep, dict):
                errors.append(
                    f"{item_location}: source timestep must be an object"
                )
                continue
            is_contact_only_fallback = (
                expected.get("status") == CONTACT_ONLY_FALLBACK_STATUS
            )
            if is_contact_only_fallback:
                fallback_metadata_mismatches = [
                    key
                    for key, expected_value in {
                        "reason": CONTACT_ONLY_FALLBACK_REASON,
                        "deterministic_shadow_override": (
                            "contact_only_shadow"
                        ),
                        "decision_source": (
                            CONTACT_ONLY_FALLBACK_POLICY_VERSION
                        ),
                        "selected_receiver_marker_pixel_count": 0,
                    }.items()
                    if expected.get(key) != expected_value
                ]
                if (
                    not isinstance(
                        expected.get("selected_projected_body_pixel_count"),
                        int,
                    )
                    or expected.get("selected_projected_body_pixel_count", 0)
                    <= 0
                ):
                    fallback_metadata_mismatches.append(
                        "selected_projected_body_pixel_count"
                    )
                if fallback_metadata_mismatches:
                    errors.append(
                        f"{item_location}: invalid fallback metadata fields "
                        f"{fallback_metadata_mismatches}"
                    )
                source_image = expected.get("source_image")
                source_image_sha256 = (
                    source_image.get("image_sha256")
                    if isinstance(source_image, dict)
                    else None
                )
                expected_mechanical = {
                    "reference_camera": expected.get("reference_camera"),
                    "source_image_sha256": source_image_sha256,
                    "preview_selection_sha256": expected.get(
                        "preview_selection_sha256"
                    ),
                    "recommended_shadow": "contact_only_shadow",
                    "deterministic_shadow_override": "contact_only_shadow",
                    "decision_source": CONTACT_ONLY_FALLBACK_POLICY_VERSION,
                    "fallback_reason": CONTACT_ONLY_FALLBACK_REASON,
                }
            else:
                expected_mechanical = {
                    "reference_camera": expected["reference_camera"],
                    "relative_path": expected["relative_path"],
                    "image_sha256": expected["preview_sha256"],
                    "preview_selection_sha256": expected[
                        "preview_selection_sha256"
                    ],
                }
            source_changed = _same_mechanical(
                source_timestep,
                expected_mechanical,
                tuple(expected_mechanical),
            )
            if source_changed:
                errors.append(
                    f"{item_location}: source mechanical fields changed "
                    f"{source_changed}"
                )
            changed = _same_mechanical(
                record,
                expected_mechanical,
                tuple(expected_mechanical),
            )
            if changed:
                errors.append(f"{item_location}: changed mechanical fields {changed}")
            if is_contact_only_fallback:
                source_image = expected.get("source_image")
                source_value = (
                    source_image.get("source_path")
                    if isinstance(source_image, dict)
                    else None
                )
                source_path_for_frame = (
                    Path(source_value).expanduser().resolve()
                    if isinstance(source_value, str)
                    else None
                )
                if (
                    source_path_for_frame is None
                    or not source_path_for_frame.is_file()
                    or fast.file_sha256(source_path_for_frame)
                    != expected_mechanical["source_image_sha256"]
                ):
                    errors.append(
                        f"{item_location}: fallback source image missing or "
                        "SHA-256 changed"
                    )
            else:
                preview_path = expected_input_root / expected["relative_path"]
                if not preview_path.is_file() or fast.file_sha256(preview_path) != expected["preview_sha256"]:
                    errors.append(f"{item_location}: preview missing or SHA-256 changed")
            if not _valid_evidence(record.get("evidence")):
                errors.append(
                    f"{item_location}: evidence must be a list of non-empty strings"
                )
            shadow_class = record.get("recommended_shadow")
            preset = DIRECT_SHADOW_PRESETS.get(shadow_class)
            if preset is None:
                errors.append(
                    f"{item_location}: invalid recommended_shadow "
                    f"{shadow_class!r}"
                )
            else:
                experimental_scene = token_record.get("experimental_scene")
                scene_label = (
                    experimental_scene.get("direct_sunlight_condition")
                    if isinstance(experimental_scene, dict)
                    else None
                )
                receiver = record.get("experimental_receiver_visibility")
                decision_source = (
                    CONTACT_ONLY_FALLBACK_POLICY_VERSION
                    if is_contact_only_fallback
                    else "direct_shadow_labels_v1"
                )
                lookup[(token, int(frame_key))] = {
                    "scene_decision": {
                        "direct_sunlight_condition": scene_label,
                    },
                    "observations": {"receiver_visibility": receiver},
                    "direct_shadow_class": shadow_class,
                    "recommended_shadow": dict(preset),
                    "confidence": None,
                    "evidence": list(record.get("evidence", [])),
                    "reference_camera": record.get("reference_camera"),
                    "scene_confidence": None,
                    "receiver_confidence": None,
                    "scene_evidence": (
                        list(experimental_scene.get("evidence", []))
                        if isinstance(experimental_scene, dict)
                        and isinstance(experimental_scene.get("evidence"), list)
                        else []
                    ),
                    "receiver_evidence": record.get("evidence"),
                    "decision_source": decision_source,
                    "placement": preview_placement,
                }

    if errors:
        sample = "; ".join(errors[:12])
        suffix = f"; ... and {len(errors) - 12} more" if len(errors) > 12 else ""
        raise ValueError(
            f"Invalid direct shadow-label file ({len(errors)} error(s)): "
            f"{sample}{suffix}"
        )
    if not lookup:
        raise ValueError(f"Shadow-label file contains no successful labels: {path}")
    metadata = {
        "path": str(path),
        "sha256": fast.file_sha256(path),
        "schema_version": SCHEMA_VERSION,
        "document_role": "direct_shadow_labels",
        "model": document.get("components"),
        "model_revision": None,
        "prompt_sha256": None,
        "scene_prompt_sha256": None,
        "receiver_prompt_sha256": None,
        # The fast compositor reserves run_dir/metadata for final success
        # records.  Preserve the actual labeled marker provenance separately.
        "metadata_root": str(expected_final_metadata_root),
        "receiver_preview_metadata_root": source_document.get("metadata_root"),
        "scene_input_version": SCENE_INPUT_VERSION,
        "receiver_input_version": RECEIVER_INPUT_VERSION,
        "preview_selection_version": SELECTION_VERSION,
        "input_root": source_document.get("input_root"),
        "label_count": len(lookup),
        "deterministic_contact_only_fallback_count": sum(
            label.get("decision_source")
            == CONTACT_ONLY_FALLBACK_POLICY_VERSION
            for label in lookup.values()
        ),
        "token_count": len(labels),
        "source_failure_count": len(failures),
        "source_template_path": str(source_path) if source_path else None,
        "source_template_sha256": document.get("source_template_sha256"),
        "decision_policy": (
            "authoritative_direct_shadow_class_with_geometry_fallback_v2"
        ),
        "placement_config_sha256": source_document.get(
            "placement_config_sha256"
        ),
    }
    return lookup, metadata
