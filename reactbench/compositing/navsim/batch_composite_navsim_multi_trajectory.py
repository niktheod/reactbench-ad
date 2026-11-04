#!/usr/bin/env python3
"""Render several future-bumper actor placements into the same image.

For every requested future timestep, this coordinator places one actor at the
corresponding interception/contact anchor across the four NAVSIM history
frames.  A run contains exactly one actor mode: an externally label-directed
crossing pedestrian, a standing pedestrian, or a stationary car.  All
renderable timestep placements are rendered together with one deterministically
selected asset in one complete camera view per history frame when possible,
or up to two nonempty camera views when no single frame fully contains every
actor bounding box.

Strict ground fitting and trajectory collision diagnostics do not suppress
rendering when a usable fallback transform exists.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import json
import math
import os
from pathlib import Path
from queue import Queue
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import yaml

from .compositor import engine as fast


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = fast.DEFAULT_OUTPUT_ROOT
PIPELINE_VERSION = "multi_timestep_actor_only_grouped_modes_v7_measured_geometry"
CLASSIFICATION_SCHEMA_VERSION = 6
TIE_BREAK_NAMESPACE = "multi_trajectory_both_directions_pass"
TIE_BREAK_METHOD = "token_seeded_uniform_hash_v1"
CAMERA_SELECTION_POLICY_VERSION = "minimum_views_full_actor_containment_v2"
FRONT_CAMERA = "CAM_F0"
CAMERA_SELECTION_PRIORITY = (
    FRONT_CAMERA,
    "CAM_L0",
    "CAM_R0",
)
DEFAULT_BLENDER = fast.DEFAULT_BLENDER
ACTOR_MODES = ("crossing", "standing", "car")

# ``left`` and ``right`` describe where the pedestrian is walking relative to
# the selected future vehicle heading.  The longer crossing names are retained
# in metadata to make the sign convention unambiguous.
DIRECTION_SPECS = (
    {
        "classification": "left",
        "crossing_name": "right_to_left",
        "lateral_sign": 1.0,
        "actor_heading_rad": math.pi,
    },
    {
        "classification": "right",
        "crossing_name": "left_to_right",
        "lateral_sign": -1.0,
        "actor_heading_rad": 0.0,
    },
)
DIRECTION_BY_CLASSIFICATION = {
    item["classification"]: item for item in DIRECTION_SPECS
}


def normalize_timesteps(values: Sequence[int]) -> Tuple[int, ...]:
    """Validate requested future timesteps without silently changing order."""
    if not values:
        raise ValueError("--timesteps requires at least one positive integer")
    normalized = []
    seen = set()
    for value in values:
        if isinstance(value, bool) or int(value) < 1:
            raise ValueError(
                f"future timesteps must be positive integers, got {value!r}"
            )
        timestep = int(value)
        if timestep in seen:
            raise ValueError(
                f"future timestep {timestep} was supplied more than once"
            )
        seen.add(timestep)
        normalized.append(timestep)
    return tuple(normalized)


def classify_direction_results(
    direction_results: Dict[str, dict],
    *,
    token: str,
    seed: int,
    sample_failed: bool = False,
    failure_reason: Optional[str] = None,
) -> dict:
    """Classify the all-timestep results, randomly breaking a passing tie."""
    if sample_failed:
        return {
            "classification": "failure",
            "classification_reason": failure_reason or "sample_failed",
        }
    left_passes = bool(
        direction_results.get("left", {}).get(
            "passes_all_requested_timesteps", False
        )
    )
    right_passes = bool(
        direction_results.get("right", {}).get(
            "passes_all_requested_timesteps", False
        )
    )
    if left_passes != right_passes:
        winner = "left" if left_passes else "right"
        return {
            "classification": winner,
            "classification_reason": (
                "unique_direction_passing_all_requested_timesteps"
            ),
        }
    if left_passes:
        random_value = fast.token_uniform(
            str(token),
            int(seed),
            TIE_BREAK_NAMESPACE,
        )
        winner = "left" if random_value < 0.5 else "right"
        return {
            "classification": winner,
            "classification_reason": (
                "both_directions_pass_seeded_random_choice"
            ),
            "classification_tie_break": {
                "applied": True,
                "eligible_directions": ["left", "right"],
                "method": TIE_BREAK_METHOD,
                "namespace": TIE_BREAK_NAMESPACE,
                "seed": int(seed),
                "uniform_value": float(random_value),
            },
        }
    return {
        "classification": "TBD",
        "classification_reason": (
            "no_direction_passes_all_requested_timesteps"
        ),
    }


def build_direction_results(
    candidates: Sequence[dict],
    timesteps: Sequence[int],
) -> Dict[str, dict]:
    """Aggregate per-anchor candidate checks into one result per direction."""
    by_key = {
        (int(candidate["future_timestep"]), str(candidate["direction"])): candidate
        for candidate in candidates
    }
    results = {}
    for direction in ("left", "right"):
        by_timestep = {}
        selected = []
        for timestep in timesteps:
            candidate = by_key.get((int(timestep), direction))
            if candidate is None:
                candidate = {
                    "future_timestep": int(timestep),
                    "direction": direction,
                    "evaluation_complete": False,
                    "renderable": False,
                    "passed": False,
                    "failures": [
                        {
                            "code": "candidate_missing",
                            "message": "candidate evaluation was not produced",
                        }
                    ],
                }
            selected.append(candidate)
            by_timestep[str(int(timestep))] = public_candidate(candidate)
        results[direction] = {
            "crossing_name": DIRECTION_BY_CLASSIFICATION[direction][
                "crossing_name"
            ],
            "evaluation_complete": all(
                bool(item.get("evaluation_complete")) for item in selected
            ),
            "passes_all_requested_timesteps": all(
                bool(item.get("passed")) for item in selected
            ),
            "renderable_at_any_requested_timestep": any(
                bool(item.get("renderable")) for item in selected
            ),
            "by_timestep": by_timestep,
        }
    return results


def select_cameras_for_actor_visibility(
    actor_ids: Sequence[str],
    fully_visible_actor_ids_by_camera: Dict[str, Sequence[str]],
    visible_actor_ids_by_camera: Dict[str, Sequence[str]],
) -> dict:
    """Choose one fully containing camera, otherwise useful best views."""
    ordered_actor_ids = list(dict.fromkeys(str(value) for value in actor_ids))
    if not ordered_actor_ids:
        raise ValueError("camera selection requires at least one actor")
    actor_id_set = set(ordered_actor_ids)
    cameras = [
        camera
        for camera in CAMERA_SELECTION_PRIORITY
        if camera in fast.CAMERA_ORDER
    ]
    cameras.extend(
        camera
        for camera in fast.CAMERA_ORDER
        if camera not in cameras
    )
    if len(cameras) < 2:
        raise ValueError("camera selection requires at least two cameras")

    fully_visible_sets = {}
    fully_visible_lists = {}
    visible_sets = {}
    visible_lists = {}
    for camera in cameras:
        reported_fully_visible = {
            str(value)
            for value in fully_visible_actor_ids_by_camera.get(camera, [])
        }
        reported_visible = {
            str(value)
            for value in visible_actor_ids_by_camera.get(camera, [])
        }
        fully_visible = actor_id_set & reported_fully_visible
        visible = actor_id_set & (
            reported_visible | reported_fully_visible
        )
        fully_visible_sets[camera] = fully_visible
        visible_sets[camera] = visible
        fully_visible_lists[camera] = [
            actor_id
            for actor_id in ordered_actor_ids
            if actor_id in fully_visible
        ]
        visible_lists[camera] = [
            actor_id
            for actor_id in ordered_actor_ids
            if actor_id in visible
        ]

    complete_cameras = [
        camera
        for camera in cameras
        if fully_visible_sets[camera] == actor_id_set
    ]
    if complete_cameras:
        selected_cameras = [complete_cameras[0]]
        selection_mode = "single_camera_all_actors"
        if (
            len(complete_cameras) > 1
            and FRONT_CAMERA in complete_cameras
        ):
            selection_reason = (
                "front_camera_preferred_among_complete_views"
            )
        elif len(complete_cameras) == 1:
            selection_reason = "only_complete_camera"
        else:
            selection_reason = "preferred_complete_camera"
    else:
        useful_cameras = [
            camera for camera in cameras if visible_sets[camera]
        ]
        pair_candidates = list(combinations(useful_cameras, 2))
        ranked_pairs = []
        for priority_index, pair in enumerate(pair_candidates):
            combined_fully_visible = (
                fully_visible_sets[pair[0]]
                | fully_visible_sets[pair[1]]
            )
            combined_visible = (
                visible_sets[pair[0]] | visible_sets[pair[1]]
            )
            ranked_pairs.append(
                (
                    (
                        len(combined_fully_visible),
                        len(combined_visible),
                        int(FRONT_CAMERA in pair),
                        len(fully_visible_sets[pair[0]])
                        + len(fully_visible_sets[pair[1]]),
                        len(visible_sets[pair[0]])
                        + len(visible_sets[pair[1]]),
                        -priority_index,
                    ),
                    pair,
                )
            )
        if ranked_pairs:
            _score, selected_pair = max(
                ranked_pairs, key=lambda item: item[0]
            )
            selected_cameras = list(selected_pair)
            selection_mode = "two_camera_maximum_full_containment_union"
            selection_reason = (
                "no_single_camera_fully_contains_all_actors"
            )
        elif useful_cameras:
            selected_cameras = [useful_cameras[0]]
            selection_mode = "single_useful_camera_no_second_view"
            selection_reason = (
                "no_other_camera_contains_any_actor"
            )
        else:
            selected_cameras = []
            selection_mode = "no_visible_camera"
            selection_reason = "no_camera_contains_any_actor"

    selected_fully_visible_union = set().union(
        *(fully_visible_sets[camera] for camera in selected_cameras)
    )
    selected_visible_union = set().union(
        *(visible_sets[camera] for camera in selected_cameras)
    )
    return {
        "policy_version": CAMERA_SELECTION_POLICY_VERSION,
        "mode": selection_mode,
        "reason": selection_reason,
        "selected_cameras": selected_cameras,
        "complete_cameras": complete_cameras,
        "actor_ids": ordered_actor_ids,
        "fully_visible_actor_ids_by_camera": fully_visible_lists,
        "visible_actor_ids_by_camera": visible_lists,
        "selected_fully_visible_union_actor_ids": [
            actor_id
            for actor_id in ordered_actor_ids
            if actor_id in selected_fully_visible_union
        ],
        "selected_visible_union_actor_ids": [
            actor_id
            for actor_id in ordered_actor_ids
            if actor_id in selected_visible_union
        ],
        "actors_not_fully_visible_in_selection": [
            actor_id
            for actor_id in ordered_actor_ids
            if actor_id not in selected_fully_visible_union
        ],
        "actors_not_visible_in_selection": [
            actor_id
            for actor_id in ordered_actor_ids
            if actor_id not in selected_visible_union
        ],
        "all_actors_fully_visible_by_selection": (
            selected_fully_visible_union == actor_id_set
        ),
        "all_actors_visible_by_selection": (
            selected_visible_union == actor_id_set
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-filter", type=Path, required=True)
    parser.add_argument(
        "--actor-mode",
        choices=ACTOR_MODES,
        default="crossing",
        help=(
            "Render one combined image set containing all requested "
            "timestep placements for this actor mode."
        ),
    )
    parser.add_argument(
        "--direction-classifications",
        type=Path,
        help=(
            "JSON object mapping NAVSIM tokens to left/right. Required only "
            "for crossing mode."
        ),
    )
    parser.add_argument(
        "--asset-source",
        type=Path,
        required=True,
        help=(
            "One packed actor .blend file, or a directory of packed "
            "assets. A directory retains deterministic per-token selection; "
            "each token still uses exactly one asset for every copy."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        required=True,
        help="Future 0.5-second NAVSIM timestep anchors, for example 1 2 4.",
    )
    parser.add_argument(
        "--actor-scale",
        type=float,
        default=1.0,
        help="Uniform scale shared by every actor instance (cars require 1).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=fast.locations.NAVSIM_DATA,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=fast.PEDESTRIAN_SPEED_MPS,
    )
    parser.add_argument("--stride-length-m", type=float)
    parser.add_argument("--gait-frame-advance", type=int, default=23)
    parser.add_argument("--gait-timestep-s", type=float, default=0.5)
    parser.add_argument(
        "--max-ground-tilt-deg",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Cycles samples for the fixed neutral actor render.",
    )
    parser.add_argument(
        "--world-strength",
        type=float,
        default=0.8,
        help=(
            "Fixed neutral environment strength. It is never estimated from "
            "the source image."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--blender",
        type=Path,
        default=DEFAULT_BLENDER,
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=HERE / "compositor" / "blender_asset",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--render-batch-size", type=int, default=96)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument(
        "--finalize-workers",
        type=int,
        default=4,
        help=(
            "Bounded CPU workers that finalize ready tokens while Blender "
            "continues rendering."
        ),
    )
    parser.add_argument(
        "--render-region-padding-px",
        type=int,
        default=fast.RENDER_REGION_PADDING_PX,
        help=(
            "Conservative ideal-image padding around the union of all actors."
        ),
    )
    parser.add_argument(
        "--experimental-render-region",
        action="store_true",
        help=(
            "Opt in to Cycles border rendering and cropped raw storage. "
            "This is faster, but validation found small RGB differences from "
            "full-frame rendering, so it is disabled by default."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.standing = args.actor_mode == "standing"
    args.car = args.actor_mode == "car"
    args.timesteps = normalize_timesteps(args.timesteps)
    if not args.scene_filter.is_file():
        raise FileNotFoundError(
            f"Scene filter does not exist: {args.scene_filter}"
        )
    if not args.asset_source.exists():
        raise FileNotFoundError(
            f"Asset source does not exist: {args.asset_source}"
        )
    if args.asset_source.is_file() and args.asset_source.suffix != ".blend":
        raise ValueError(
            "--asset-source file must have a .blend extension"
        )
    if not args.asset_source.is_file() and not args.asset_source.is_dir():
        raise ValueError(
            "--asset-source must be a .blend file or asset directory"
        )
    if not args.run_name.strip():
        raise ValueError("--run-name must not be empty")
    if (
        not math.isfinite(args.actor_scale)
        or args.actor_scale <= 0.0
    ):
        raise ValueError("--actor-scale must be finite and positive")
    if args.car and not math.isclose(args.actor_scale, 1.0):
        raise ValueError("car mode requires --actor-scale 1.0")
    if not math.isfinite(args.speed_mps) or args.speed_mps <= 0.0:
        raise ValueError("--speed-mps must be finite and positive")
    if (
        args.stride_length_m is not None
        and (
            not math.isfinite(args.stride_length_m)
            or args.stride_length_m <= 0.0
        )
    ):
        raise ValueError("--stride-length-m must be finite and positive")
    if args.gait_frame_advance <= 0:
        raise ValueError("--gait-frame-advance must be positive")
    if (
        not math.isfinite(args.gait_timestep_s)
        or args.gait_timestep_s <= 0.0
    ):
        raise ValueError("--gait-timestep-s must be finite and positive")
    if not 0.0 <= args.max_ground_tilt_deg <= 15.0:
        raise ValueError("--max-ground-tilt-deg must be in [0, 15]")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if (
        not math.isfinite(args.world_strength)
        or args.world_strength <= 0.0
    ):
        raise ValueError("--world-strength must be finite and positive")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.render_batch_size < 1:
        raise ValueError("--render-batch-size must be positive")
    if args.render_workers < 1:
        raise ValueError("--render-workers must be positive")
    if args.finalize_workers < 1:
        raise ValueError("--finalize-workers must be positive")
    if args.render_region_padding_px < 0:
        raise ValueError("--render-region-padding-px must be non-negative")
    if not args.blender.is_file():
        raise FileNotFoundError(f"Blender runtime is missing: {args.blender}")
    fast.configure_direction_classifications(args)


def load_assets(args: argparse.Namespace) -> List[dict]:
    source = args.asset_source.resolve()
    if source.is_file():
        assets = [fast.load_asset(source, args)]
    else:
        args.asset_dir = source
        assets = fast.load_asset_pool(args)
    if args.car:
        invalid = [
            asset["id"]
            for asset in assets
            if asset.get("actor_type") != "car"
        ]
        if invalid:
            raise ValueError(f"car mode received non-car assets: {invalid[:5]}")
    else:
        invalid = [
            asset["id"]
            for asset in assets
            if asset.get("actor_type") != "pedestrian"
        ]
        if invalid:
            raise ValueError(
                "pedestrian modes require human pedestrian assets; "
                f"invalid={invalid[:5]}"
            )
    missing_geometry = [
        asset["id"] for asset in assets if "actor_geometry" not in asset
    ]
    if missing_geometry:
        raise ValueError(
            "multi-actor rendering requires measured asset geometry; "
            f"missing={missing_geometry[:5]}"
        )
    return assets


def select_token_asset(
    token: str,
    args: argparse.Namespace,
    assets: Sequence[dict],
) -> dict:
    if len(assets) == 1:
        return assets[0]
    return fast.select_asset(token, args.seed, assets)


def diagnostic(
    code: str,
    message: str,
    **details: Any,
) -> dict:
    return {
        "code": str(code),
        "message": str(message),
        **details,
    }


def collision_path_with_interception(
    actor_global_xys: Sequence[Sequence[float]],
    interception_global_xy: Sequence[float],
) -> np.ndarray:
    """Append the unrendered future interception to the history sweep."""
    history = np.asarray(actor_global_xys, dtype=np.float64)
    interception = np.asarray(
        interception_global_xy, dtype=np.float64
    ).reshape(-1)
    if (
        history.ndim != 2
        or history.shape[0] == 0
        or history.shape[1] != 2
        or interception.shape != (2,)
        or not np.isfinite(history).all()
        or not np.isfinite(interception).all()
    ):
        raise ValueError(
            "collision path requires finite history XY points and one "
            "finite interception XY point"
        )
    return np.concatenate([history, interception[None]], axis=0)


def exception_diagnostic(exc: Exception, **details: Any) -> dict:
    code = exc.code if isinstance(exc, fast.TokenFailure) else "evaluation_failed"
    return diagnostic(
        code,
        str(exc),
        exception_type=type(exc).__name__,
        **details,
    )


def rotation_z_matrix(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    return result


def scaled_actor_bbox_lidar(
    position: np.ndarray,
    actor_basis_lidar: np.ndarray,
    actor_heading: float,
    local_bounds_m: dict,
    scale: float,
) -> np.ndarray:
    """Transform and scale measured geometry around its grounded root."""
    return fast.render_utils.actor_bbox_lidar(
        position,
        actor_basis_lidar,
        actor_heading,
        local_bounds_m,
        scale=float(scale),
    )


def scaled_local_bounds(local_bounds_m: dict, scale: float) -> dict:
    """Scale measured local bounds around the asset root."""
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"actor scale must be finite and positive, got {scale}")
    try:
        return {
            axis: [
                scale * float(local_bounds_m[axis][0]),
                scale * float(local_bounds_m[axis][1]),
            ]
            for axis in ("x", "y", "z")
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "local_bounds_m must contain numeric x/y/z intervals"
        ) from exc


def actor_camera_visibility(
    bbox_lidar: np.ndarray,
    calibration: dict,
) -> dict:
    """Report frame intersection and strict full-box containment separately."""
    corner_mask = np.asarray(
        fast.geometry.camera_frustum_mask(
            bbox_lidar,
            calibration,
            margin=0.0,
        ),
        dtype=bool,
    ).reshape(-1)
    if corner_mask.size != len(bbox_lidar):
        raise ValueError(
            "camera frustum mask does not match actor bounding-box corners"
        )
    return {
        "visible": bool(np.any(corner_mask)),
        "fully_visible": bool(
            corner_mask.size > 0 and np.all(corner_mask)
        ),
        "inside_corner_count": int(np.count_nonzero(corner_mask)),
        "corner_count": int(corner_mask.size),
    }


def validate_token_inputs(
    args: argparse.Namespace,
    scene_frames: Sequence[dict],
    *,
    lidar_loader: Optional[Callable[[dict], np.ndarray]] = None,
) -> Tuple[List[dict], np.ndarray, List[np.ndarray]]:
    frames = list(scene_frames[: fast.EXPECTED_HISTORY_FRAMES])
    if len(frames) != fast.EXPECTED_HISTORY_FRAMES:
        raise fast.TokenFailure(
            "missing_sensor",
            f"expected {fast.EXPECTED_HISTORY_FRAMES} history frames, "
            f"got {len(frames)}",
        )
    for frame in frames:
        for camera in fast.CAMERA_ORDER:
            source_path = fast.frame_source_path(
                args.data_root, args.split, frame, camera
            )
            if not source_path.is_file():
                raise fast.TokenFailure(
                    "missing_sensor",
                    f"missing {camera} image: {source_path}",
                )
        lidar_path_text = frame.get("lidar_path")
        if not isinstance(lidar_path_text, str) or not lidar_path_text:
            raise fast.TokenFailure(
                "missing_sensor",
                "history frame does not contain a LiDAR path",
            )
        lidar_path = (
            args.data_root
            / "sensor_blobs"
            / args.split
            / lidar_path_text
        )
        if not lidar_path.is_file():
            raise fast.TokenFailure(
                "missing_sensor",
                f"missing LiDAR point cloud: {lidar_path}",
            )
    map_locations = {
        str(frame.get("map_location", "")) for frame in frames
    }
    if len(map_locations) != 1:
        raise fast.TokenFailure(
            "inconsistent_map_location",
            f"history frames span multiple maps: {sorted(map_locations)}",
        )
    timestamps_s = (
        np.asarray(
            [frame.get("timestamp", math.nan) for frame in frames],
            dtype=np.float64,
        )
        / 1e6
    )
    if (
        not np.isfinite(timestamps_s).all()
        or np.any(np.diff(timestamps_s) <= 0.0)
    ):
        raise fast.TokenFailure(
            "invalid_timestamps",
            f"history timestamps are invalid: {timestamps_s.tolist()}",
        )
    if lidar_loader is None:
        lidar_loader = lambda frame: fast.geometry.load_lidar_xyz(
            args.data_root,
            args.split,
            frame,
        )
    frame_lidar_points = [lidar_loader(frame) for frame in frames]
    return frames, timestamps_s, frame_lidar_points


def evaluate_candidate(
    *,
    args: argparse.Namespace,
    token: str,
    scene_frames: Sequence[dict],
    frames: Sequence[dict],
    timestamps_s: np.ndarray,
    frame_lidar_points: Sequence[np.ndarray],
    global_to_current: np.ndarray,
    front_bumper_offset_m: float,
    asset: dict,
    animation: Sequence[int],
    future_timestep: int,
    direction_spec: Optional[dict] = None,
) -> dict:
    actor_mode = str(args.actor_mode)
    benign = bool(getattr(args, "benign", False))
    if benign:
        direction = str(args.benign_side)
        crossing_name = None
        lateral_sign = 0.0
        actor_heading_rad = 0.0
    elif actor_mode == "crossing":
        if direction_spec is None:
            raise ValueError("crossing evaluation requires a direction spec")
        direction = str(direction_spec["classification"])
        crossing_name = str(direction_spec["crossing_name"])
        lateral_sign = float(direction_spec["lateral_sign"])
        actor_heading_rad = float(direction_spec["actor_heading_rad"])
    elif actor_mode == "standing":
        direction = "standing"
        crossing_name = None
        lateral_sign = 0.0
        actor_heading_rad = float(fast.STANDING_ACTOR_HEADING_RAD)
    elif actor_mode == "car":
        direction = "stationary"
        crossing_name = None
        lateral_sign = 0.0
        actor_heading_rad = 0.0
    else:
        raise ValueError(f"unsupported actor mode: {actor_mode}")
    candidate = {
        "candidate_id": f"t{int(future_timestep)}_{actor_mode}_{direction}",
        "future_timestep": int(future_timestep),
        "actor_mode": actor_mode,
        "direction": direction,
        "crossing_name": crossing_name,
        "lateral_sign": lateral_sign,
        "actor_heading_rad": actor_heading_rad,
        "evaluation_complete": False,
        "renderable": False,
        "passed": False,
        "failures": [],
        "_states": [],
    }
    try:
        local_bounds_m = scaled_local_bounds(
            asset["actor_geometry"]["local_bounds_m"],
            float(args.actor_scale),
        )
        if benign:
            trajectory = fast.benign_placements.resolve(
                args, fast, scene_frames, timestamps_s, front_bumper_offset_m,
                local_bounds_m, timestep=int(future_timestep), actor_mode=actor_mode,
            )
            actor_heading_rad = trajectory["actor_heading_rad"]
            candidate["actor_heading_rad"] = actor_heading_rad
            candidate["candidate_id"] = f"t{future_timestep}_benign_{actor_mode}_{direction}"
        elif actor_mode == "crossing":
            trajectory = fast.resolve_crossing_trajectory(
                scene_frames,
                timestamps_s,
                timestep=int(future_timestep),
                front_bumper_offset_m=front_bumper_offset_m,
                lateral_sign=lateral_sign,
                speed_mps=args.speed_mps,
            )
        elif actor_mode == "standing":
            trajectory = fast.resolve_standing_trajectory(
                scene_frames,
                timestamps_s,
                timestep=int(future_timestep),
                front_bumper_offset_m=front_bumper_offset_m,
            )
        else:
            ego_geometry = getattr(args, "ego_vehicle_geometry", None)
            if not isinstance(ego_geometry, dict):
                raise RuntimeError(
                    "car evaluation requires ego vehicle geometry"
                )
            trajectory = fast.resolve_stationary_vehicle_trajectory(
                scene_frames,
                timestamps_s,
                timestep=int(future_timestep),
                front_bumper_offset_m=front_bumper_offset_m,
                car_local_bounds=asset["actor_geometry"]["local_bounds_m"],
                ego_rear_length_m=float(ego_geometry["rear_length_m"]),
                ego_width_m=float(ego_geometry["width_m"]),
            )
        basis_forward = trajectory.get("actor_basis_forward_global_xy", trajectory["reference_vehicle_forward_global_xy"])
        actor_reference_current_xys = trajectory["actor_current_xys"]
        actor_global_xys = trajectory["actor_global_xys"]
        actor_lidar_xys = [
            fast.inverse_transform_xy_planar(
                np.asarray(frame["lidar2global"], dtype=np.float64),
                actor_global_xy,
            )
            for frame, actor_global_xy in zip(frames, actor_global_xys)
        ]
        (
            ground_planes,
            ground_diagnostics,
            strict_ground_failures,
        ) = fast.fit_ground_sequence_best_effort(
            frames,
            frame_lidar_points,
            actor_lidar_xys,
            actor_reference_current_xys,
        )

        if strict_ground_failures:
            candidate["failures"].append(
                diagnostic(
                    "ground_fit_failed",
                    (
                        "the strict same-frame local ground fit failed at "
                        "one or more history frames"
                    ),
                    failed_frame_indices=[
                        int(item["frame_index"])
                        for item in strict_ground_failures
                    ],
                    strict_failures=strict_ground_failures,
                )
            )

        canonical_actor_to_global = None
        if actor_mode == "car":
            final_lidar_to_global = np.asarray(
                frames[-1]["lidar2global"], dtype=np.float64
            )
            final_actor_lidar = fast.ground_point_lidar_for_global_xy(
                final_lidar_to_global,
                ground_planes[-1],
                trajectory["anchor_global_xy"],
            )
            final_basis = fast.road_basis_aligned_to_global_forward(
                ground_planes[-1],
                final_lidar_to_global,
                basis_forward,
                args.max_ground_tilt_deg,
            )
            canonical_actor_to_global = (
                final_lidar_to_global
                @ fast.render_utils.matrix4(final_basis, final_actor_lidar)
            )

        states = []
        per_frame_collisions = []
        for frame_index, (
            frame,
            lidar_xyz,
            actor_lidar_xy,
            actor_global_xy,
            plane,
            ground_details,
            animation_frame,
        ) in enumerate(
            zip(
                frames,
                frame_lidar_points,
                actor_lidar_xys,
                actor_global_xys,
                ground_planes,
                ground_diagnostics,
                animation,
            )
        ):
            lidar_to_global = np.asarray(
                frame["lidar2global"], dtype=np.float64
            )
            if (
                lidar_to_global.shape != (4, 4)
                or not np.isfinite(lidar_to_global).all()
            ):
                raise fast.TokenFailure(
                    "invalid_lidar_pose",
                    f"history frame {frame_index} has an invalid lidar2global",
                )
            if actor_mode == "car":
                if canonical_actor_to_global is None:
                    raise RuntimeError("car pose was not initialized")
                actor_to_lidar = (
                    np.linalg.inv(lidar_to_global)
                    @ canonical_actor_to_global
                )
                actor_lidar = actor_to_lidar[:3, 3].copy()
                basis = actor_to_lidar[:3, :3].copy()
                actor_global = canonical_actor_to_global[:3, 3].copy()
            else:
                actor_lidar = np.array(
                    [
                        float(actor_lidar_xy[0]),
                        float(actor_lidar_xy[1]),
                        fast.geometry.ground_z(
                            plane,
                            float(actor_lidar_xy[0]),
                            float(actor_lidar_xy[1]),
                        ),
                    ],
                    dtype=np.float64,
                )
                basis = fast.road_basis_aligned_to_global_forward(
                    plane,
                    lidar_to_global,
                    basis_forward,
                    args.max_ground_tilt_deg,
                )
                actor_to_lidar = fast.render_utils.matrix4(
                    basis, actor_lidar
                )
                actor_global = fast.geometry.transform_points(
                    lidar_to_global, actor_lidar[None]
                )[0]
            lidar_to_current = global_to_current @ lidar_to_global
            actor_to_current = lidar_to_current @ actor_to_lidar
            root_to_current = (
                actor_to_current
                @ rotation_z_matrix(actor_heading_rad)
            )
            if actor_mode == "car":
                direction_lidar = basis[:, 0]
                yaw_lidar = math.atan2(
                    float(direction_lidar[1]), float(direction_lidar[0])
                )
                collision = fast.stationary_vehicle_collision_diagnostics(
                    lidar_xyz,
                    np.linalg.inv(actor_to_lidar),
                    asset["actor_geometry"]["local_bounds_m"],
                )
            else:
                direction_lidar = (fast.benign_placements.direction_lidar(basis, actor_heading_rad) if benign else
                                   -basis[:, 0] if actor_mode == "standing" else basis[:, 1] * lateral_sign)
                yaw_lidar = (
                    math.atan2(
                        float(direction_lidar[1]),
                        float(direction_lidar[0]),
                    )
                    + math.pi / 2.0
                )
                collision = fast.frame_collision_diagnostics(
                    lidar_xyz,
                    np.linalg.inv(actor_to_lidar),
                    actor_heading_rad,
                    local_bounds_m,
                )
            collision["frame_index"] = int(frame_index)
            per_frame_collisions.append(collision)
            bbox_lidar = scaled_actor_bbox_lidar(
                actor_lidar,
                basis,
                actor_heading_rad,
                asset["actor_geometry"]["local_bounds_m"],
                float(args.actor_scale),
            )
            camera_visibility = {
                camera: actor_camera_visibility(
                    bbox_lidar,
                    frame["cams"][camera],
                )
                for camera in fast.CAMERA_ORDER
            }
            visible_cameras = [
                camera
                for camera in fast.CAMERA_ORDER
                if camera_visibility[camera]["visible"]
            ]
            fully_visible_cameras = [
                camera
                for camera in fast.CAMERA_ORDER
                if camera_visibility[camera]["fully_visible"]
            ]
            actor_global = fast.geometry.transform_points(
                lidar_to_global, actor_lidar[None]
            )[0]
            actor_current = fast.geometry.transform_points(
                global_to_current, actor_global[None]
            )[0]
            states.append(
                {
                    "frame_index": int(frame_index),
                    "animation_frame": int(animation_frame),
                    "actor_lidar": actor_lidar,
                    "actor_global": actor_global,
                    "actor_current": actor_current,
                    "actor_global_xy": np.asarray(
                        actor_global_xy, dtype=np.float64
                    ),
                    "road_plane_lidar": np.asarray(
                        plane, dtype=np.float64
                    ),
                    "road_basis_lidar": basis,
                    "root_to_current": root_to_current,
                    "yaw_lidar_rad": float(yaw_lidar),
                    "bbox_lidar": bbox_lidar,
                    "visible_cameras": visible_cameras,
                    "fully_visible_cameras": fully_visible_cameras,
                    "camera_visibility": camera_visibility,
                    "ground_diagnostics": ground_details,
                    "collision_diagnostics": collision,
                }
            )

        if actor_mode == "car":
            swept_collision = dict(per_frame_collisions[-1])
            swept_collision.update(
                {
                    "path_start_xy_m": states[-1]["actor_lidar"][
                        :2
                    ].tolist(),
                    "path_end_xy_m": states[-1]["actor_lidar"][
                        :2
                    ].tolist(),
                    "sample_count": 1,
                }
            )
            swept_path_point_count = 1
            swept_includes_interception = False
        else:
            # The rendered pedestrian history ends before interception. Keep
            # diagnostics conservative by checking that final segment too.
            collision_path_global_xys = collision_path_with_interception(
                actor_global_xys,
                trajectory["anchor_global_xy"],
            )
            if benign:
                collision_path_global_xys = trajectory["scene_collision_path_global_xys"]
            final_positions_lidar = [
                fast.inverse_transform_xy_planar(
                    np.asarray(frames[-1]["lidar2global"], dtype=np.float64),
                    actor_global_xy,
                )
                for actor_global_xy in collision_path_global_xys
            ]
            swept_collision = fast.trajectory_collision_diagnostics(
                frame_lidar_points[-1],
                states[-1]["road_plane_lidar"],
                final_positions_lidar,
                actor_basis_lidar=states[-1]["road_basis_lidar"],
                actor_heading=actor_heading_rad,
                local_bounds_m=local_bounds_m,
            )
            swept_path_point_count = len(final_positions_lidar)
            swept_includes_interception = True
        occupied_frames = [
            int(item["frame_index"])
            for item in per_frame_collisions
            if item["occupied"]
        ]
        collision_occupied = bool(
            swept_collision["occupied"]
            or any(item["occupied"] for item in per_frame_collisions)
        )
        if collision_occupied:
            candidate["failures"].append(
                diagnostic(
                    "trajectory_collision",
                    (
                        "LiDAR-supported scene geometry intersects the "
                        f"{actor_mode} footprint or swept path"
                    ),
                    occupied_frame_indices=occupied_frames,
                    swept_collision=swept_collision,
                )
            )

        fast.front_visibility.check_candidate(args, fast, frames, states, candidate)
        visible_view_count = sum(
            len(state["visible_cameras"]) for state in states
        )
        fully_visible_view_count = sum(
            len(state["fully_visible_cameras"]) for state in states
        )
        if visible_view_count == 0:
            candidate["failures"].append(
                diagnostic(
                    "no_camera_visibility",
                    (
                        "the candidate bounding volume is outside all "
                        "four-history-frame front-camera views"
                    ),
                )
            )

        candidate.update(
            {
                "evaluation_complete": True,
                "renderable": True,
                "passed": not candidate["failures"],
                "placement": trajectory["placement"],
                "anchor_global_xy_m": trajectory[
                    "anchor_global_xy"
                ].tolist(),
                "walking_velocity_global_xy_mps": trajectory[
                    "walking_velocity_global_xy_mps"
                ].tolist(),
                "crossing_offsets_m": trajectory[
                    "crossing_offsets_m"
                ].tolist(),
                "strict_ground_failures": strict_ground_failures,
                "collision_diagnostics": {
                    "per_frame": per_frame_collisions,
                    "swept_final_frame": swept_collision,
                    "swept_path_includes_future_interception": (
                        swept_includes_interception and not benign
                    ),
                    "swept_path_point_count": swept_path_point_count,
                    "occupied": collision_occupied,
                },
                "visible_view_count": int(visible_view_count),
                "fully_visible_view_count": int(
                    fully_visible_view_count
                ),
                "_states": states,
            }
        )
    except Exception as exc:
        candidate["failures"].append(
            exception_diagnostic(
                exc,
                future_timestep=int(future_timestep),
                direction=direction,
            )
        )
        candidate.update(
            {
                "evaluation_complete": False,
                "renderable": False,
                "passed": False,
                "_states": [],
            }
        )
    return candidate


def public_state(state: dict) -> dict:
    return {
        "frame_index": int(state["frame_index"]),
        "animation_frame": int(state["animation_frame"]),
        "actor_lidar_m": state["actor_lidar"].tolist(),
        "actor_global_m": state["actor_global"].tolist(),
        "actor_current_ego_m": state["actor_current"].tolist(),
        "road_plane_lidar": state["road_plane_lidar"].tolist(),
        "road_basis_lidar": state["road_basis_lidar"].tolist(),
        "yaw_lidar_rad": float(state["yaw_lidar_rad"]),
        "visible_cameras": list(state["visible_cameras"]),
        "fully_visible_cameras": list(state["fully_visible_cameras"]),
        "camera_visibility": state["camera_visibility"],
        "ground_diagnostics": state["ground_diagnostics"],
        "collision_diagnostics": state["collision_diagnostics"],
    }


def public_candidate(candidate: dict) -> dict:
    return {
        key: value
        for key, value in {
            "candidate_id": candidate.get("candidate_id"),
            "future_timestep": int(candidate["future_timestep"]),
            "actor_mode": candidate.get("actor_mode"),
            "direction": str(candidate["direction"]),
            "crossing_name": candidate.get("crossing_name"),
            "lateral_sign": candidate.get("lateral_sign"),
            "actor_heading_rad": candidate.get("actor_heading_rad"),
            "evaluation_complete": bool(
                candidate.get("evaluation_complete")
            ),
            "renderable": bool(candidate.get("renderable")),
            "passed": bool(candidate.get("passed")),
            "failures": candidate.get("failures", []),
            "placement": candidate.get("placement"),
            "front_visibility": candidate.get("front_visibility"),
            "anchor_global_xy_m": candidate.get("anchor_global_xy_m"),
            "walking_velocity_global_xy_mps": candidate.get(
                "walking_velocity_global_xy_mps"
            ),
            "crossing_offsets_m": candidate.get("crossing_offsets_m"),
            "strict_ground_failures": candidate.get(
                "strict_ground_failures", []
            ),
            "collision_diagnostics": candidate.get(
                "collision_diagnostics"
            ),
            "visible_view_count": candidate.get("visible_view_count", 0),
            "fully_visible_view_count": candidate.get(
                "fully_visible_view_count", 0
            ),
            "historical_states": [
                public_state(state)
                for state in candidate.get("_states", [])
            ],
        }.items()
        if value is not None
    }


def requested_mode_classification(
    args: argparse.Namespace,
    token: str,
) -> dict:
    if args.actor_mode == "crossing":
        label = str(args.direction_classification_labels[token])
        return {
            "classification": label,
            "classification_reason": "external_direction_label",
            "direction_label": label,
        }
    return {
        "classification": str(args.actor_mode),
        "classification_reason": "requested_actor_mode",
    }


def build_timestep_results(
    candidates: Sequence[dict],
    timesteps: Sequence[int],
) -> dict:
    by_timestep = {
        str(int(candidate["future_timestep"])): public_candidate(candidate)
        for candidate in candidates
    }
    missing = [
        int(timestep)
        for timestep in timesteps
        if str(int(timestep)) not in by_timestep
    ]
    return {
        "all_requested_timesteps_present": not missing,
        "all_requested_timesteps_renderable": (
            not missing
            and all(
                bool(by_timestep[str(int(timestep))].get("renderable"))
                for timestep in timesteps
            )
        ),
        "missing_timesteps": missing,
        "by_timestep": by_timestep,
    }


def multi_actor_render_region(
    states: Sequence[dict],
    calibration: dict,
    intrinsic: np.ndarray,
    resolution: Sequence[int],
    padding_px: int,
) -> List[int]:
    """Return a conservative crop containing every rendered actor."""
    width, height = [int(value) for value in resolution]
    full_region = [0, 0, width, height]
    if not states:
        return full_region
    camera_to_lidar = fast.render_utils.camera_to_lidar(calibration)
    regions = [
        fast.actor_shadow_render_region(
            state["bbox_lidar"],
            camera_to_lidar,
            intrinsic,
            resolution,
            (0.0, 0.0, -1.0),
            0.0,
            0.0,
            padding_px=padding_px,
        )
        for state in states
    ]
    region = [
        min(item[0] for item in regions),
        min(item[1] for item in regions),
        max(item[2] for item in regions),
        max(item[3] for item in regions),
    ]
    if not (
        0 <= region[0] < region[2] <= width
        and 0 <= region[1] < region[3] <= height
    ):
        return full_region
    region_area = (region[2] - region[0]) * (region[3] - region[1])
    if region_area >= 0.95 * width * height:
        return full_region
    return region


def manifest_cache_key(manifest: dict) -> str:
    return fast.canonical_hash(
        {
            key: value
            for key, value in manifest.items()
            if key not in ("output_path", "render_info_path")
        }
    )


def prepare_token(
    *,
    args: argparse.Namespace,
    loader: Any,
    token: str,
    run_dir: Path,
    front_bumper_offset_m: float,
    asset: dict,
) -> dict:
    scene_frames = loader.scene_frames_dicts[token]
    (
        frames,
        timestamps_s,
        frame_lidar_points,
    ) = validate_token_inputs(args, scene_frames)
    current_ego_to_global = np.asarray(
        frames[-1].get("ego2global", []), dtype=np.float64
    )
    if (
        current_ego_to_global.shape != (4, 4)
        or not np.isfinite(current_ego_to_global).all()
    ):
        raise fast.TokenFailure(
            "invalid_ego_pose",
            "final history frame has an invalid ego2global transform",
        )
    global_to_current = np.linalg.inv(current_ego_to_global)
    if args.actor_mode == "crossing":
        animation = fast.animation_frames(
            timestamps_s,
            args.speed_mps,
            asset["stride_length_m"],
            asset["action_range"][0],
            asset["action_range"][1],
            fast.NAVSIM_POSE_SEQUENCE[0],
        )
        direction_label = str(args.direction_classification_labels[token])
        direction_spec = DIRECTION_BY_CLASSIFICATION[direction_label]
    else:
        animation = [1] * fast.EXPECTED_HISTORY_FRAMES
        direction_spec = None

    candidates = [
        evaluate_candidate(
            args=args,
            token=token,
            scene_frames=scene_frames,
            frames=frames,
            timestamps_s=timestamps_s,
            frame_lidar_points=frame_lidar_points,
            global_to_current=global_to_current,
            front_bumper_offset_m=front_bumper_offset_m,
            asset=asset,
            animation=animation,
            future_timestep=int(future_timestep),
            direction_spec=direction_spec,
        )
        for future_timestep in args.timesteps
    ]
    timestep_results = build_timestep_results(candidates, args.timesteps)
    locatable_candidates = [
        candidate for candidate in candidates if candidate["renderable"]
    ]
    if not timestep_results["all_requested_timesteps_renderable"]:
        missing = [
            int(candidate["future_timestep"])
            for candidate in candidates
            if not candidate["renderable"]
        ]
        raise fast.TokenFailure(
            "incomplete_timestep_group",
            (
                "cannot render all requested timesteps in one image; "
                f"unrenderable_timesteps={missing}"
            )
        )
    classification = requested_mode_classification(args, token)

    view_records = []
    frame_camera_selections = []
    jobs = []
    for frame_index, frame in enumerate(frames):
        frame_candidates = [
            (candidate, candidate["_states"][frame_index])
            for candidate in locatable_candidates
            if len(candidate["_states"]) == fast.EXPECTED_HISTORY_FRAMES
        ]
        if not frame_candidates:
            continue
        lidar_to_global = np.asarray(
            frame["lidar2global"], dtype=np.float64
        )
        lidar_to_current = global_to_current @ lidar_to_global
        animation_frame = int(animation[frame_index])
        actor_ids = [
            candidate["candidate_id"]
            for candidate, _state in frame_candidates
        ]
        visible_actor_ids_by_camera = {
            camera: [
                candidate["candidate_id"]
                for candidate, state in frame_candidates
                if camera in state["visible_cameras"]
            ]
            for camera in fast.CAMERA_ORDER
        }
        fully_visible_actor_ids_by_camera = {
            camera: [
                candidate["candidate_id"]
                for candidate, state in frame_candidates
                if camera in state["fully_visible_cameras"]
            ]
            for camera in fast.CAMERA_ORDER
        }
        camera_selection = select_cameras_for_actor_visibility(
            actor_ids,
            fully_visible_actor_ids_by_camera,
            visible_actor_ids_by_camera,
        )
        frame_camera_selections.append(
            {
                "frame_index": int(frame_index),
                **camera_selection,
            }
        )
        for camera in camera_selection["selected_cameras"]:
            visible_actor_ids = visible_actor_ids_by_camera[camera]
            fully_visible_actor_ids = (
                fully_visible_actor_ids_by_camera[camera]
            )
            source_path = fast.frame_source_path(
                args.data_root, args.split, frame, camera
            )
            with Image.open(source_path) as source_image:
                width, height = source_image.size
            calibration = frame["cams"][camera]
            overscan = fast.render_utils.distortion_overscan(
                calibration, width, height
            )
            camera_to_current = (
                lidar_to_current
                @ fast.render_utils.camera_to_lidar(calibration)
            )
            if args.experimental_render_region:
                render_region = multi_actor_render_region(
                    [state for _candidate, state in frame_candidates],
                    calibration,
                    np.asarray(overscan["intrinsic"], dtype=np.float64),
                    overscan["resolution"],
                    args.render_region_padding_px,
                )
            else:
                render_region = [
                    0,
                    0,
                    int(overscan["resolution"][0]),
                    int(overscan["resolution"][1]),
                ]
            raw_path = (
                run_dir
                / "raw_actor_only"
                / token
                / camera
                / f"{frame_index}.png"
            )
            info_path = raw_path.with_suffix(".render.json")
            output_path = (
                run_dir
                / "images"
                / token
                / camera
                / f"{frame_index}.png"
            )
            mask_path = (
                run_dir
                / "masks"
                / token
                / camera
                / f"{frame_index}.png"
            )
            manifest_path = (
                run_dir
                / "manifests"
                / token
                / camera
                / f"{frame_index}.json"
            )
            manifest = {
                "job_id": f"{token}:{camera}:{frame_index}",
                "pipeline_version": PIPELINE_VERSION,
                "asset_path": str(asset["path"]),
                "asset_sha256": asset["sha256"],
                "asset_id": asset["id"],
                "actor_mode": args.actor_mode,
                "actor_type": asset["actor_type"],
                "actor_root_name": asset["actor_root_name"],
                "standing_pose": args.actor_mode == "standing",
                "standing_pose_version": (
                    fast.STANDING_POSE_VERSION
                    if args.actor_mode == "standing"
                    else None
                ),
                "resolution": list(overscan["resolution"]),
                "sensor_resolution": [int(width), int(height)],
                "ideal_origin": list(overscan["origin"]),
                "render_region": render_region,
                "raw_render_storage": fast.RAW_RENDER_STORAGE_VERSION,
                "render_region_padding_px": int(
                    args.render_region_padding_px
                ),
                "camera_intrinsic_matrix": np.asarray(
                    overscan["intrinsic"], dtype=np.float64
                ).tolist(),
                "camera_to_world": camera_to_current.tolist(),
                "animation_frame": animation_frame,
                "actor_scale": float(args.actor_scale),
                "actors": [
                    {
                        "actor_id": candidate["candidate_id"],
                        "future_timestep": int(
                            candidate["future_timestep"]
                        ),
                        "actor_mode": candidate["actor_mode"],
                        "direction": candidate["direction"],
                        "crossing_name": candidate["crossing_name"],
                        "root_to_world": state[
                            "root_to_current"
                        ].tolist(),
                    }
                    for candidate, state in frame_candidates
                ],
                "visible_actor_ids": visible_actor_ids,
                "fully_visible_actor_ids": fully_visible_actor_ids,
                "camera_selection": camera_selection,
                "samples": int(args.samples),
                "world_strength": float(args.world_strength),
                "output_path": str(raw_path.resolve()),
                "render_info_path": str(info_path.resolve()),
            }
            manifest["cache_key"] = manifest_cache_key(manifest)
            fast.write_json(manifest_path, manifest)
            if args.overwrite or not fast.cache_matches(
                raw_path,
                info_path,
                manifest["cache_key"],
            ):
                jobs.append(manifest)
            view_records.append(
                {
                    "frame_index": int(frame_index),
                    "camera": camera,
                    "source_path": source_path,
                    "output_path": output_path,
                    "mask_path": mask_path,
                    "raw_path": raw_path,
                    "render_info_path": info_path,
                    "manifest_path": manifest_path,
                    "manifest": manifest,
                    "calibration": calibration,
                    "sensor_resolution": [int(width), int(height)],
                    "visible_actor_ids": visible_actor_ids,
                    "fully_visible_actor_ids": fully_visible_actor_ids,
                    "camera_selection": camera_selection,
                }
            )

    if not view_records:
        classification = {
            "classification": "failure",
            "classification_reason": "no_visible_renderable_views",
        }
    return {
        "token": token,
        "asset": asset,
        "timestamps_s": timestamps_s,
        "candidates": candidates,
        "timestep_results": timestep_results,
        **classification,
        "frame_camera_selections": frame_camera_selections,
        "view_records": view_records,
        "jobs": jobs,
    }


def crop_render_storage_after_blender(job: dict) -> dict:
    """Losslessly crop an experimental border-render PNG after Blender exits.

    This pixel crop itself is exact. Cycles border rendering is not guaranteed
    to match full-frame sampling, which is why callers only reach this path
    after an explicit ``--experimental-render-region`` opt-in.
    """
    output_path = Path(job["output_path"])
    info_path = Path(job["render_info_path"])
    if not output_path.is_file() or not info_path.is_file():
        return {"status": "missing"}
    render_info = json.loads(info_path.read_text())
    pending = render_info.get("pending_storage_crop_region")
    if pending is None:
        return {
            "status": "full",
            "reason": render_info.get("crop_fallback_reason"),
        }
    region = [int(value) for value in pending]
    if region != [int(value) for value in job["render_region"]]:
        raise ValueError(
            f"renderer pending crop {region} does not match manifest "
            f"region {job['render_region']}"
        )
    width, height = [int(value) for value in job["resolution"]]
    x_min, y_min, x_max, y_max = region
    if not (
        0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    ):
        raise ValueError(
            f"pending render crop {region} is outside {width}x{height}"
        )

    with Image.open(output_path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    if rgba.shape != (height, width, 4):
        raise ValueError(
            f"border-render image has shape {rgba.shape}, expected "
            f"{(height, width, 4)}"
        )
    cropped = rgba[y_min:y_max, x_min:x_max].copy()
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".crop.tmp"
    )
    try:
        Image.fromarray(cropped).save(
            temporary_path,
            format="PNG",
            compress_level=3,
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    render_info.update(
        {
            "stored_render_region": region,
            "stored_resolution": [x_max - x_min, y_max - y_min],
            "pending_storage_crop_region": None,
            "storage_crop_completed": True,
            "storage_crop_method": "coordinator_lossless_png_pixel_crop_v1",
        }
    )
    fast.write_json(info_path, render_info)
    return {
        "status": "cropped",
        "region": region,
        "stored_resolution": render_info["stored_resolution"],
    }


def run_blender_batches(
    args: argparse.Namespace,
    run_dir: Path,
    jobs: Sequence[dict],
    on_batches_submitted: Optional[Callable[[], None]] = None,
    on_batch_complete: Optional[Callable[[dict], None]] = None,
) -> List[dict]:
    """Run the dedicated renderer while preserving existing GPU isolation."""
    if not jobs:
        if on_batches_submitted is not None:
            on_batches_submitted()
        return []
    renderer = (
        args.asset_root / "scripts/render_multi_trajectory_batch.py"
    )
    if not renderer.is_file():
        raise FileNotFoundError(
            f"Multi-trajectory Blender renderer is missing: {renderer}"
        )
    log_dir = run_dir / "render_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs_by_asset: Dict[str, List[dict]] = {}
    for job in jobs:
        jobs_by_asset.setdefault(str(job["asset_path"]), []).append(job)

    with tempfile.TemporaryDirectory(
        prefix="navsim-multi-trajectory-jobs-"
    ) as temporary_directory:
        job_dir = Path(temporary_directory)
        batches = []
        for asset_path, asset_jobs in jobs_by_asset.items():
            for start in range(0, len(asset_jobs), args.render_batch_size):
                batch_jobs = asset_jobs[
                    start : start + args.render_batch_size
                ]
                batch_index = len(batches)
                job_path = job_dir / f"batch_{batch_index:05d}.jsonl"
                job_path.write_text(
                    "".join(
                        json.dumps(job, default=fast.jsonable) + "\n"
                        for job in batch_jobs
                    )
                )
                batches.append(
                    {
                        "batch_index": batch_index,
                        "asset_path": asset_path,
                        "job_path": job_path,
                        "log_path": (
                            log_dir / f"batch_{batch_index:05d}.log"
                        ),
                        "job_ids": [
                            str(job["job_id"]) for job in batch_jobs
                        ],
                        "jobs": list(batch_jobs),
                    }
                )

        worker_count = min(args.render_workers, len(batches))
        gpu_devices = fast.render_worker_devices(
            args.device, worker_count
        )
        try:
            allocated_cpus = int(
                os.environ.get(
                    "SLURM_CPUS_PER_TASK",
                    os.cpu_count() or worker_count,
                )
            )
        except ValueError:
            allocated_cpus = os.cpu_count() or worker_count
        native_threads = max(1, allocated_cpus // worker_count)
        print(
            f"Rendering {len(batches)} batch(es) with "
            f"{worker_count} Blender worker(s)"
        )

        available_devices: Queue[Optional[str]] = Queue()
        for gpu_device in gpu_devices:
            available_devices.put(gpu_device)

        def run_one(batch: dict) -> dict:
            gpu_device = available_devices.get()
            try:
                result = fast.run_blender_batch(
                    args,
                    renderer,
                    batch,
                    gpu_device,
                    native_threads,
                )
                result["job_ids"] = list(batch["job_ids"])
                storage_results = []
                for job in batch["jobs"]:
                    try:
                        storage_results.append(
                            crop_render_storage_after_blender(job)
                        )
                    except Exception as exc:
                        storage_results.append(
                            {
                                "status": "crop_failed_kept_full",
                                "job_id": str(job["job_id"]),
                                "error": repr(exc),
                            }
                        )
                        print(
                            "WARNING: raw render storage crop failed for "
                            f"{job['job_id']}; keeping the full PNG: {exc}",
                            file=sys.stderr,
                        )
                result["storage_results"] = storage_results
                return result
            finally:
                available_devices.put(gpu_device)

        results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_one, batch): batch
                for batch in batches
            }
            if on_batches_submitted is not None:
                on_batches_submitted()
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result["returncode"] != 0:
                    print(
                        "WARNING: Blender batch "
                        f"{result['batch_index']} failed; see "
                        f"{result['log_path']}",
                        file=sys.stderr,
                    )
                if on_batch_complete is not None:
                    on_batch_complete(result)
        return results


def finalize_token(
    args: argparse.Namespace,
    run_dir: Path,
    prepared: dict,
) -> dict:
    remap_cache = {}
    rendered_views = []
    render_failures = []
    for record in prepared["view_records"]:
        raw_path = Path(record["raw_path"])
        info_path = Path(record["render_info_path"])
        manifest = record["manifest"]
        if not fast.cache_matches(
            raw_path, info_path, manifest["cache_key"]
        ):
            render_failures.append(
                diagnostic(
                    "render_failed",
                    "actor-only render is missing or stale",
                    frame_index=record["frame_index"],
                    camera=record["camera"],
                    raw_path=str(raw_path),
                )
            )
            continue
        try:
            source = np.asarray(
                Image.open(record["source_path"]).convert("RGB")
            )
            ideal = np.asarray(Image.open(raw_path).convert("RGBA"))
            render_info = json.loads(info_path.read_text())
            effective_manifest = dict(manifest)
            effective_manifest["render_region"] = render_info.get(
                "stored_render_region", manifest["render_region"]
            )
            expected_width, expected_height = (
                fast.render_storage_resolution(effective_manifest)
            )
            if ideal.shape != (
                expected_height,
                expected_width,
                4,
            ):
                raise ValueError(
                    f"raw actor image has shape {ideal.shape}, expected "
                    f"{(expected_height, expected_width, 4)}"
                )
            sensor_width, sensor_height = record["sensor_resolution"]
            distorted = fast.render_utils.distort_rgba(
                ideal,
                record["calibration"],
                remap_cache,
                output_size=(sensor_width, sensor_height),
                ideal_origin=fast.render_storage_ideal_origin(
                    effective_manifest
                ),
            )
            actor_pixels = int(np.count_nonzero(distorted[..., 3]))
            if actor_pixels == 0:
                raise fast.TokenFailure(
                    "render_visibility_mismatch",
                    "projected actors produced no visible alpha pixels",
                )
            edited, actor_mask = fast.render_utils.alpha_composite(
                source, distorted
            )
            fast.save_png(Path(record["output_path"]), edited)
            fast.save_png(Path(record["mask_path"]), actor_mask)
            rendered_views.append(
                {
                    "frame_index": int(record["frame_index"]),
                    "camera": str(record["camera"]),
                    "source_path": str(record["source_path"]),
                    "output_path": str(record["output_path"]),
                    "mask_path": str(record["mask_path"]),
                    "actor_pixel_count": actor_pixels,
                    "requested_render_region": manifest["render_region"],
                    "stored_render_region": effective_manifest[
                        "render_region"
                    ],
                    "crop_fallback_reason": render_info.get(
                        "crop_fallback_reason"
                    ),
                    "visible_actor_ids": list(
                        record["visible_actor_ids"]
                    ),
                    "fully_visible_actor_ids": list(
                        record["fully_visible_actor_ids"]
                    ),
                    "camera_selection": record["camera_selection"],
                }
            )
        except Exception as exc:
            render_failures.append(
                exception_diagnostic(
                    exc,
                    frame_index=int(record["frame_index"]),
                    camera=str(record["camera"]),
                )
            )

    if not rendered_views:
        classification = {
            "classification": "failure",
            "classification_reason": "no_edited_images_produced",
        }
        render_status = "failure"
    elif render_failures:
        classification = {
            key: prepared[key]
            for key in (
                "classification",
                "classification_reason",
                "direction_label",
            )
            if key in prepared
        }
        render_status = "partial"
    else:
        classification = {
            key: prepared[key]
            for key in (
                "classification",
                "classification_reason",
                "direction_label",
            )
            if key in prepared
        }
        render_status = "success"

    metadata = {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "actor_mode": args.actor_mode,
        "token": prepared["token"],
        **classification,
        "classification_semantics": {
            "left": "label-directed crossing toward vehicle-left",
            "right": "label-directed crossing toward vehicle-right",
            "standing": "four stationary pedestrian placements",
            "car": "four stationary vehicle placements",
            "failure": "the sample could not produce an edited image",
        },
        "requested_future_timesteps": [
            int(value) for value in args.timesteps
        ],
        "asset": fast.asset_config(prepared["asset"]),
        "actor_type": (
            "car" if args.actor_mode == "car" else "pedestrian"
        ),
        "actor_geometry": prepared["asset"]["actor_geometry"],
        **(
            {
                "pedestrian_geometry": prepared["asset"][
                    "actor_geometry"
                ]
            }
            if args.actor_mode != "car"
            else {}
        ),
        "actor_scale": float(args.actor_scale),
        "speed_mps": (
            float(args.speed_mps) if args.actor_mode == "crossing" else 0.0
        ),
        "timestep_results": prepared["timestep_results"],
        "frame_camera_selections": prepared[
            "frame_camera_selections"
        ],
        "render_status": render_status,
        "rendered_views": rendered_views,
        "render_failures": render_failures,
    }
    fast.write_json(
        run_dir / "metadata" / f"{prepared['token']}.json",
        metadata,
    )
    return metadata


def finalize_prepared_token(
    args: argparse.Namespace,
    run_dir: Path,
    prepared: dict,
) -> dict:
    """Finalize one token and publish a stable failure record if needed."""
    token = prepared["token"]
    failure_path = run_dir / "failures" / f"{token}.json"
    try:
        metadata = finalize_token(args, run_dir, prepared)
    except Exception as exc:
        failure = exception_diagnostic(exc, stage="finalization")
        metadata = failed_token_metadata(
            args=args,
            token=token,
            failure=failure,
        )
        fast.write_json(
            run_dir / "metadata" / f"{token}.json", metadata
        )
    if metadata["classification"] == "failure":
        fast.write_json(
            failure_path,
            {
                "status": "failed",
                "token": token,
                "code": metadata["classification_reason"],
                "message": "sample did not produce an edited image",
                "render_failures": metadata.get(
                    "render_failures", []
                ),
            },
        )
    elif failure_path.exists():
        failure_path.unlink()
    return metadata


def run_blender_with_streaming_finalization(
    args: argparse.Namespace,
    run_dir: Path,
    prepared_tokens: Sequence[dict],
    jobs: Sequence[dict],
) -> Dict[str, dict]:
    """Finalize tokens as soon as all of their render batches complete."""
    prepared_by_token = {
        str(prepared["token"]): prepared for prepared in prepared_tokens
    }
    pending_job_ids = {
        token: {
            str(job["job_id"]) for job in prepared["jobs"]
        }
        for token, prepared in prepared_by_token.items()
    }
    job_owner = {
        str(job["job_id"]): token
        for token, prepared in prepared_by_token.items()
        for job in prepared["jobs"]
    }
    submitted_tokens = set()
    finalization_futures = {}
    worker_count = min(
        len(prepared_tokens),
        max(1, int(args.finalize_workers)),
    )
    if prepared_tokens:
        print(
            f"Finalizing {len(prepared_tokens)} token(s) with "
            f"{worker_count} bounded worker(s) as renders complete"
        )

    def submit_ready(executor: ThreadPoolExecutor) -> None:
        for prepared in prepared_tokens:
            token = str(prepared["token"])
            if token in submitted_tokens or pending_job_ids[token]:
                continue
            submitted_tokens.add(token)
            future = executor.submit(
                finalize_prepared_token,
                args,
                run_dir,
                prepared,
            )
            finalization_futures[future] = token

    metadata_by_token = {}
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:

        def batch_complete(result: dict) -> None:
            for job_id in result["job_ids"]:
                token = job_owner.get(str(job_id))
                if token is not None:
                    pending_job_ids[token].discard(str(job_id))
            submit_ready(executor)

        run_blender_batches(
            args,
            run_dir,
            jobs,
            on_batches_submitted=lambda: submit_ready(executor),
            on_batch_complete=batch_complete,
        )
        submit_ready(executor)
        for future in as_completed(finalization_futures):
            token = finalization_futures[future]
            metadata_by_token[token] = future.result()

    incomplete = {
        token: sorted(job_ids)
        for token, job_ids in pending_job_ids.items()
        if token not in submitted_tokens and job_ids
    }
    if incomplete:
        raise RuntimeError(
            "Blender scheduler returned without completing token jobs: "
            f"{incomplete}"
        )
    return metadata_by_token


def failed_token_metadata(
    *,
    args: argparse.Namespace,
    token: str,
    failure: dict,
) -> dict:
    return {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "token": token,
        "actor_mode": args.actor_mode,
        "classification": "failure",
        "classification_reason": "token_preparation_failed",
        "requested_future_timesteps": [
            int(value) for value in args.timesteps
        ],
        "timestep_results": {},
        "frame_camera_selections": [],
        "render_status": "failure",
        "rendered_views": [],
        "render_failures": [failure],
    }


def write_run_outputs(
    run_dir: Path,
    metadata_records: Sequence[dict],
) -> None:
    actor_mode = (
        str(metadata_records[0].get("actor_mode", "unknown"))
        if metadata_records
        else "unknown"
    )
    classifications = {
        str(record["token"]): str(record["classification"])
        for record in metadata_records
    }
    details = {
        str(record["token"]): {
            "classification": str(record["classification"]),
            "classification_reason": str(
                record["classification_reason"]
            ),
            "render_status": str(record.get("render_status", "unknown")),
            **(
                {"direction_label": record["direction_label"]}
                if "direction_label" in record
                else {}
            ),
        }
        for record in metadata_records
    }
    result_document = {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "actor_mode": actor_mode,
        "classification_semantics": {
            "left": "label-directed crossing toward vehicle-left",
            "right": "label-directed crossing toward vehicle-right",
            "standing": "combined standing-pedestrian timestep group",
            "car": "combined stationary-car timestep group",
            "failure": "sample could not be edited",
        },
        "tokens": classifications,
        "details": details,
    }
    fast.write_json(run_dir / "actor_render_results.json", result_document)
    if actor_mode == "crossing":
        fast.write_json(
            run_dir / "direction_classifications.json",
            result_document,
        )
    counts = Counter(classifications.values())
    failures = [
        {
            "token": record["token"],
            "classification": "failure",
            "reason": record["classification_reason"],
            "render_failures": record.get("render_failures", []),
        }
        for record in metadata_records
        if record["classification"] == "failure"
    ]
    (run_dir / "failures.jsonl").write_text(
        "".join(json.dumps(item, default=fast.jsonable) + "\n" for item in failures)
    )
    fast.write_json(
        run_dir / "summary.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "actor_mode": actor_mode,
            "sample_count": len(metadata_records),
            "classification_counts": dict(sorted(counts.items())),
            "edited_sample_count": sum(
                bool(record.get("rendered_views"))
                for record in metadata_records
            ),
            "failure_count": len(failures),
        },
    )


def run_config(
    args: argparse.Namespace,
    assets: Sequence[dict],
) -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "actor_mode": args.actor_mode,
        "scene_filter": str(args.scene_filter.resolve()),
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "asset_source": str(args.asset_source.resolve()),
        "asset_selection": (
            "single_global_asset"
            if len(assets) == 1
            else "one_deterministic_asset_per_token"
        ),
        "asset_pool": [fast.asset_config(asset) for asset in assets],
        "requested_future_timesteps": [
            int(value) for value in args.timesteps
        ],
        **(
            fast.direction_classification_provenance(args)
            if args.actor_mode == "crossing"
            else {}
        ),
        "placement_grouping": (
            "one_actor_per_requested_timestep_composited_in_one_image"
        ),
        "classification_policy": (
            "external_direction_label"
            if args.actor_mode == "crossing"
            else "requested_actor_mode"
        ),
        "strict_ground_fit_recorded_as_diagnostic": True,
        "ground_fallback_used_for_rendering": True,
        "trajectory_collision_recorded_as_diagnostic": True,
        "trajectory_collision_path": (
            "stationary_vehicle_pose"
            if args.actor_mode == "car"
            else "rendered_history_plus_future_interception_segment"
        ),
        "synthetic_actor_collisions_ignored": True,
        "camera_selection": {
            "policy_version": CAMERA_SELECTION_POLICY_VERSION,
            "single_camera_rule": (
                "use one camera whose frame fully contains every "
                "renderable actor bounding box"
            ),
            "complete_view_tie_break": (
                "prefer CAM_F0, then CAM_L0, then CAM_R0"
            ),
            "no_complete_view_rule": (
                "use up to two nonempty cameras maximizing the union of "
                "fully contained actors, then partial visibility; prefer "
                "a CAM_F0 pair on equal coverage"
            ),
            "maximum_views_per_history_frame": 2,
            "full_containment_test": (
                "all eight scaled actor bounding-box corners inside "
                "the zero-margin camera frustum"
            ),
        },
        "actor_scale": float(args.actor_scale),
        "speed_mps": (
            float(args.speed_mps) if args.actor_mode == "crossing" else 0.0
        ),
        "seed": int(args.seed),
        "samples": int(args.samples),
        "world_strength": float(args.world_strength),
        "render_batch_size": int(args.render_batch_size),
        "render_workers": int(args.render_workers),
        "finalize_workers": int(args.finalize_workers),
        "raw_render_storage": fast.RAW_RENDER_STORAGE_VERSION,
        "render_region": {
            "enabled": bool(args.experimental_render_region),
            "status": (
                "experimental_opt_in"
                if args.experimental_render_region
                else "disabled_for_full_frame_pixel_equivalence"
            ),
            "padding_px": int(args.render_region_padding_px),
            "alignment_px": int(fast.RENDER_REGION_ALIGNMENT_PX),
            "fallback": "full_frame_when_boundary_alpha_is_detected",
            "equivalence_note": (
                "Cycles border rendering produced small RGB differences in "
                "full-versus-border validation even when alpha was identical"
            ),
        },
        "compositing": "nonzero_alpha_bounding_box_pixel_exact",
        "lighting": "fixed_neutral_environment_no_scene_estimation",
        "shadows": "disabled_no_receiver_geometry",
        "occlusion": "none",
        "harmonization": "none",
        "cameras": list(fast.CAMERA_ORDER),
        "history_frames": list(fast.HISTORY_FRAME_INDICES),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.scene_filter = args.scene_filter.resolve()
    args.asset_source = args.asset_source.resolve()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.asset_root = args.asset_root.resolve()
    args.blender = args.blender.resolve()
    finalization_native_threads = (
        fast.configure_finalization_native_threads(args.finalize_workers)
    )
    print(
        f"Finalization CPU budget: {args.finalize_workers} token worker(s), "
        f"up to {finalization_native_threads} OpenCV thread(s) per operation"
    )

    fast.configure_navsim(args.data_root)
    filter_config = yaml.safe_load(args.scene_filter.read_text())
    available_future_frames = int(filter_config["num_future_frames"])
    unavailable = [
        timestep
        for timestep in args.timesteps
        if timestep > available_future_frames
    ]
    if unavailable:
        raise ValueError(
            f"requested future timestep(s) {unavailable} exceed the scene "
            f"filter's {available_future_frames} future frames"
        )

    loader, tokens = fast.build_loader(args, filter_config)
    if args.actor_mode == "crossing":
        fast.validate_direction_classifications(
            tokens,
            args.direction_classification_labels,
        )
    assets = load_assets(args)
    from nuplan.common.actor_state.vehicle_parameters import (
        get_pacifica_parameters,
    )

    ego_vehicle = get_pacifica_parameters()
    front_bumper_offset_m = float(ego_vehicle.front_length)
    args.ego_vehicle_geometry = {
        "front_length_m": front_bumper_offset_m,
        "rear_length_m": float(ego_vehicle.rear_length),
        "width_m": float(ego_vehicle.width),
    }
    run_dir = (
        args.output_root / args.scene_filter.stem / args.run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration = run_config(args, assets)
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text())
        if existing != configuration:
            raise RuntimeError(
                f"Run configuration conflicts with {config_path}; "
                "choose a different --run-name"
            )
    else:
        fast.write_json(config_path, configuration)

    prepared_tokens = []
    all_jobs = []
    metadata_by_token: Dict[str, dict] = {}
    for index, token in enumerate(tokens):
        metadata_path = run_dir / "metadata" / f"{token}.json"
        failure_path = run_dir / "failures" / f"{token}.json"
        if metadata_path.is_file() and not args.overwrite:
            cached = json.loads(metadata_path.read_text())
            if (
                cached.get("classification") != "failure"
                or not args.retry_failures
            ):
                metadata_by_token[token] = cached
                print(
                    f"[{index + 1}/{len(tokens)}] cached "
                    f"{cached.get('classification')} {token}"
                )
                continue
        try:
            asset = select_token_asset(token, args, assets)
            prepared = prepare_token(
                args=args,
                loader=loader,
                token=token,
                run_dir=run_dir,
                front_bumper_offset_m=front_bumper_offset_m,
                asset=asset,
            )
            prepared_tokens.append(prepared)
            all_jobs.extend(prepared["jobs"])
            print(
                f"[{index + 1}/{len(tokens)}] prepared {token}: "
                f"{prepared['classification']}, "
                f"{len(prepared['view_records'])} visible view(s)"
            )
        except Exception as exc:
            failure = exception_diagnostic(exc, stage="preparation")
            metadata = failed_token_metadata(
                args=args,
                token=token,
                failure=failure,
            )
            fast.write_json(metadata_path, metadata)
            fast.write_json(
                failure_path,
                {
                    "status": "failed",
                    "token": token,
                    **failure,
                },
            )
            metadata_by_token[token] = metadata
            print(
                f"[{index + 1}/{len(tokens)}] failure {token}: "
                f"{failure['code']}: {failure['message']}",
                file=sys.stderr,
            )

    print(
        f"Rendering {len(all_jobs)} uncached combined camera views "
        f"for {len(prepared_tokens)} token(s)"
    )
    metadata_by_token.update(
        run_blender_with_streaming_finalization(
            args,
            run_dir,
            prepared_tokens,
            all_jobs,
        )
    )

    ordered_metadata = [
        metadata_by_token[token]
        for token in tokens
        if token in metadata_by_token
    ]
    write_run_outputs(run_dir, ordered_metadata)
    print(
        f"Finished {len(ordered_metadata)} sample(s); outputs: {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
