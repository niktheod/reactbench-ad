#!/usr/bin/env python3
"""Insert crossing or stationary actors into four PAI-AV camera histories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
COMPOSITOR_ROOT = HERE / "compositor"

from .compositor import engine as compositor
from . import ego_masks
from .. import edit_preparation, locations
from .paiav_io import (
    ALPAMAYO_HISTORY_FRAME_COUNT,
    CAMERA_NAMES,
    HISTORY_FRAME_COUNT,
    HISTORY_INTERVAL_US,
    POSE_SEQUENCE,
    PAIAVSceneLoader,
    frame_source_path,
    load_lidar_xyz,
    load_selection_manifest,
)


DEFAULT_DATA_ROOT = locations.PAIAV_ROOT
DEFAULT_MANIFEST = locations.REACTPAI_MANIFEST
DEFAULT_OUTPUT_ROOT = locations.REACTPAI
DEFAULT_ASSET_DIR = locations.ASSETS / "rocketbox" / "assets"
DEFAULT_CAR_ASSET_DIR = locations.ASSETS / "openx" / "assets"
DEFAULT_BLENDER = locations.BLENDER
DEFAULT_DEPTH_PYTHON = locations.DEPTH_PYTHON
DEFAULT_DEPTH_CACHE = locations.MODEL_CACHE
PAIAV_RENDER_PIPELINE_VERSION = (
    "paiav_photoreal_counterfactual_v3_measured_geometry_contact_only"
)
PAIAV_STANDING_RENDER_PIPELINE_VERSION = (
    "paiav_photoreal_counterfactual_standing_v1_contact_only"
)
PAIAV_CAR_RENDER_PIPELINE_VERSION = (
    "paiav_photoreal_counterfactual_car_v1_contact_only"
)
PAIAV_PLACEMENT_VERSION = "paiav_future_ego_front_bumper_fixed_10hz_v1"
PAIAV_STANDING_PLACEMENT_VERSION = (
    "paiav_future_ego_front_bumper_stationary_obstacle_10hz_v1"
)
PAIAV_CAR_PLACEMENT_VERSION = (
    "paiav_future_ego_front_to_stationary_car_rear_contact_10hz_v1"
)
PAIAV_DIRECTION_POLICY_VERSION = "paiav_balanced_deterministic_direction_v1"
PAIAV_GROUNDING_VERSION = (
    "paiav_per_point_lidar_sequence_consistent_grounding_v3_nearby_support_required"
)
PAIAV_CAR_GROUNDING_VERSION = (
    "paiav_single_rigid_world_pose_grounding_v2_nearby_support_required"
)
_BASE_FIT_GROUND_SEQUENCE = compositor.fit_ground_sequence_best_effort
_BASE_ANNOTATION_CONTEXT = compositor.annotation_context


def load_navigation_commands(tokens: Sequence[str]) -> dict[str, str | None]:
    """Navigation is disabled: navigation_command is null in every output annotation."""
    return {token: None for token in tokens}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    ego_masks.add_arguments(parser)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--timestep",
        type=int,
        required=True,
        help=(
            "Future 0.1-second ego step used for pedestrian interception or "
            "stationary-car contact."
        ),
    )
    actor_mode = parser.add_mutually_exclusive_group()
    actor_mode.add_argument(
        "--standing",
        action="store_true",
        help=(
            "Place a naturally posed stationary human at the future ego "
            "front-bumper point, facing the approaching ego."
        ),
    )
    actor_mode.add_argument(
        "--car",
        action="store_true",
        help=(
            "Insert a stationary OpenX/XOMA car whose rear bumper first "
            "contacts the future ego front bumper."
        ),
    )
    parser.add_argument(
        "--anchor-time-s",
        type=float,
        default=5.1,
        help="Nominal time of the fourth edited image (default: 5.1 seconds).",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        help=(
            "Homogeneous actor directory. Defaults to the local Rocketbox "
            "pool, or the local prepared OpenX pool with --car."
        ),
    )
    parser.add_argument(
        "--direction-classifications",
        type=Path,
        help=(
            "Optional JSON mapping clip IDs to left/right. By default, the "
            "selected clips are divided deterministically between both directions."
        ),
    )
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed-mps", type=float, default=1.4)
    parser.add_argument("--stride-length-m", type=float)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--render-scale", type=int, choices=(1, 2), default=1)
    parser.add_argument("--actor-blur-sigma", type=float, default=0.55)
    parser.add_argument("--max-actor-noise-std", type=float, default=1.5)
    parser.add_argument("--disable-motion-blur", action="store_true")
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="gpu")
    parser.add_argument("--harmonization-strength", type=float, default=0.65)
    parser.add_argument(
        "--occlusion-mode",
        choices=("none", "lidar", "dense"),
        default="dense",
        help=(
            "Actor/scene occlusion policy. The default keeps only dense-depth "
            "components seeded by per-pixel-depth-gated LiDAR evidence and "
            "preserves the actor when dense alignment is unavailable."
        ),
    )
    parser.add_argument(
        "--strict-depth-alignment",
        action="store_true",
        help="Make dense-depth alignment fatal instead of falling back to LiDAR.",
    )
    parser.add_argument(
        "--max-unconfirmed-occlusion-fraction",
        type=float,
        default=0.20,
        help=(
            "Maximum dense-depth actor fraction removable without closer "
            "LiDAR support before falling back to supported occlusion."
        ),
    )
    parser.add_argument(
        "--min-projected-bbox-area",
        type=float,
        default=64.0,
        help="Skip Blender for camera-edge projections smaller than this many pixels.",
    )
    parser.add_argument("--max-ground-tilt-deg", type=float, default=2.0)
    parser.add_argument("--disable-collision-check", action="store_true")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--depth-python", type=Path, default=DEFAULT_DEPTH_PYTHON)
    parser.add_argument("--depth-model-cache", type=Path, default=DEFAULT_DEPTH_CACHE)
    parser.add_argument("--depth-prediction-cache", type=Path)
    parser.add_argument("--render-batch-size", type=int, default=96)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--finalize-workers", type=int, default=4)
    parser.add_argument("--stage", choices=("all", "prepare", "render"), default="all",
                        help="CPU preparation, render saved preparation, or the combined pipeline.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Materialize inputs and render manifests, but do not run depth or Blender.",
    )
    args = parser.parse_args()
    if args.prepare_only:
        if args.stage == "render":
            parser.error("--prepare-only cannot be combined with --stage render")
        args.stage = "prepare"
    args.prepare_only = args.stage == "prepare"
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.asset_dir is None:
        args.asset_dir = (
            DEFAULT_CAR_ASSET_DIR
            if getattr(args, "car", False)
            else DEFAULT_ASSET_DIR
        )
    if not args.run_name or "/" in args.run_name:
        raise ValueError("--run-name must be one nonempty directory name")
    if args.timestep < 1:
        raise ValueError("--timestep must be positive")
    minimum_anchor_time_s = (
        ALPAMAYO_HISTORY_FRAME_COUNT - 1
    ) * HISTORY_INTERVAL_US / 1_000_000
    if (
        not math.isfinite(args.anchor_time_s)
        or args.anchor_time_s < minimum_anchor_time_s
    ):
        raise ValueError(
            "--anchor-time-s must be finite and at least "
            f"{minimum_anchor_time_s:g} to record Alpamayo ego history"
        )
    if args.max_clips is not None and args.max_clips < 0:
        raise ValueError("--max-clips must be non-negative")
    if not math.isfinite(args.speed_mps) or args.speed_mps <= 0:
        raise ValueError("--speed-mps must be finite and positive")
    if args.stride_length_m is not None and args.stride_length_m <= 0:
        raise ValueError("--stride-length-m must be positive")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.actor_blur_sigma < 0 or args.max_actor_noise_std < 0:
        raise ValueError("Camera degradation parameters must be non-negative")
    if not 0 <= args.harmonization_strength <= 1:
        raise ValueError("--harmonization-strength must be in [0, 1]")
    if not 0 <= args.max_unconfirmed_occlusion_fraction <= 1:
        raise ValueError(
            "--max-unconfirmed-occlusion-fraction must be in [0, 1]"
        )
    if args.min_projected_bbox_area < 0:
        raise ValueError("--min-projected-bbox-area must be non-negative")
    if not 0 <= args.max_ground_tilt_deg <= 15:
        raise ValueError("--max-ground-tilt-deg must be in [0, 15]")
    for name in ("render_batch_size", "render_workers", "finalize_workers"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def detect_asset_profile(asset_dir: Path, *, car: bool = False) -> str:
    asset_dir = asset_dir.expanduser().resolve()
    blend_paths = sorted(
        asset_dir.rglob("*.blend") if car else asset_dir.glob("*.blend")
    )
    if not blend_paths:
        scope = "recursively under" if car else "directly under"
        raise FileNotFoundError(f"No .blend assets found {scope}: {asset_dir}")
    if car:
        missing_xoma = [
            path.with_suffix(".xoma")
            for path in blend_paths
            if not path.with_suffix(".xoma").is_file()
        ]
        if missing_xoma:
            raise FileNotFoundError(
                f"OpenX XOMA sidecar not found: {missing_xoma[0]}"
            )
        missing_preparation = [
            path.with_suffix(".preparation.json")
            for path in blend_paths
            if not path.with_suffix(".preparation.json").is_file()
        ]
        if missing_preparation:
            raise FileNotFoundError(
                "OpenX preparation sidecar not found: "
                f"{missing_preparation[0]}"
            )
        return "car"
    for blend_path in blend_paths:
        sidecar = blend_path.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Asset sidecar not found: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("species"):  # human assets carry no species; animal assets are not used on PAI-AV
            raise ValueError(f"ReactPAI inserts humans or cars, not {metadata['species']!r}: {sidecar}")
    return "human"


def validate_actor_asset_profile(args: argparse.Namespace, asset_profile: str) -> None:
    if getattr(args, "car", False):
        if asset_profile != "car":
            raise ValueError("--car requires prepared OpenX vehicle assets")
        return
    if asset_profile == "car":
        raise ValueError("OpenX vehicle assets require --car")
    if getattr(args, "standing", False) and asset_profile != "human":
        raise ValueError("--standing requires Rocketbox human assets")


def fixed_animation_frames(
    timestamps_s: np.ndarray,
    _speed: float,
    _stride_length: float,
    start: int,
    end: int,
    _initial_frame: int,
) -> list[int]:
    if len(timestamps_s) != HISTORY_FRAME_COUNT:
        raise ValueError(
            f"PAI-AV editing requires {HISTORY_FRAME_COUNT} history timestamps"
        )
    unavailable = [frame for frame in POSE_SEQUENCE if not start <= frame < end]
    if unavailable:
        raise ValueError(
            f"Fixed PAI-AV poses {unavailable} are outside action range {start}..{end - 1}"
        )
    return list(POSE_SEQUENCE)


def contact_only_sun(frame: dict) -> dict:
    """Supply a disabled direct-light record; PAI-AV has no solar coordinates."""
    return {
        "method": "unavailable_paiav_contact_only_v1",
        "timestamp_us": int(frame["timestamp"]),
        "map_location": str(frame.get("map_location", "pai-av-local")),
        "direction_to_sun_global_utm": [0.0, 0.0, 1.0],
        "sun_ray_direction_global_utm": [0.0, 0.0, -1.0],
        "direct_sun_enabled": False,
        "reason": "PAI-AV selection lacks precise solar inputs; contact-only policy",
    }


def _ground_diagnostic_has_local_support(diagnostic: dict) -> bool:
    method = str(diagnostic.get("method", ""))
    if method == "strict_same_timestamp_local_lidar":
        return int(diagnostic.get("local_support_count", 0)) >= 12
    local_support = int(diagnostic.get("local_support_count_2p5m", 0))
    nearest = diagnostic.get("nearest_candidate_distance_m")
    if nearest is None:
        nearest = diagnostic.get("nearest_distance_m")
    if local_support > 0 and nearest is not None and float(nearest) <= 2.5:
        return True
    return (
        method == "nearest_accumulated_low_surface_horizontal_emergency"
        and nearest is not None
        and float(nearest) <= 2.5
        and int(diagnostic.get("low_surface_count", 0)) >= 6
    )


def paiav_sequence_consistent_ground(
    frames: Sequence[dict],
    frame_points: Sequence[np.ndarray],
    actor_lidar_xy: Sequence[Sequence[float]],
    actor_reference_current_xy: Sequence[Sequence[float]],
):
    """Use one road plane with accepted nearby support in at least one frame."""
    planes, diagnostics, strict_failures = _BASE_FIT_GROUND_SEQUENCE(
        frames,
        frame_points,
        actor_lidar_xy,
        actor_reference_current_xy,
    )
    reliable_indices = [
        index
        for index, diagnostic in enumerate(diagnostics)
        if _ground_diagnostic_has_local_support(diagnostic)
    ]
    if not reliable_indices:
        evidence = [
            {
                "frame_index": index,
                "method": diagnostic.get("method"),
                "local_support_count": diagnostic.get(
                    "local_support_count",
                    diagnostic.get("local_support_count_2p5m", 0),
                ),
                "nearest_candidate_distance_m": diagnostic.get(
                    "nearest_candidate_distance_m",
                    diagnostic.get("nearest_distance_m"),
                ),
            }
            for index, diagnostic in enumerate(diagnostics)
        ]
        raise compositor.TokenFailure(
            "ground_fit_failed",
            "No history frame has accepted LiDAR ground support within 2.5 m "
            "of the actor; refusing low-confidence sequence fallback. "
            f"evidence={json.dumps(evidence, separators=(',', ':'))}",
        )

    global_to_common = compositor._common_frame_from_final_ego(
        np.asarray(frames[-1]["ego2global"], dtype=np.float64)
    )
    lidar_to_common = [
        global_to_common @ np.asarray(frame["lidar2global"], dtype=np.float64)
        for frame in frames
    ]
    common_planes = np.stack(
        [
            compositor._plane_lidar_to_common(
                planes[index], lidar_to_common[index]
            )
            for index in reliable_indices
        ]
    )
    sequence_plane_common = np.median(common_planes, axis=0)
    support_policy = "nearby_lidar_supported_sequence_median"
    sequence_slope = float(np.linalg.norm(sequence_plane_common[:2]))
    if not np.isfinite(sequence_plane_common).all() or sequence_slope > 0.22:
        raise compositor.TokenFailure(
            "ground_fit_failed",
            "PAI-AV sequence ground plane is implausible: "
            f"slope={sequence_slope:.3f}, sources={reliable_indices}",
        )

    consistent_planes = [
        compositor._plane_common_to_lidar(
            sequence_plane_common, lidar_to_common[index]
        )
        for index in range(len(frames))
    ]
    consistent_diagnostics = [
        {
            **diagnostic,
            "original_method": diagnostic.get("method"),
            "method": "paiav_sequence_consistent_gravity_aligned_plane_v2",
            "sequence_plane_common": sequence_plane_common.tolist(),
            "sequence_plane_slope": sequence_slope,
            "sequence_support_frame_indices": reliable_indices,
            "local_support_accepted": index in reliable_indices,
            "sequence_support_policy": support_policy,
        }
        for index, diagnostic in enumerate(diagnostics)
    ]
    return consistent_planes, consistent_diagnostics, strict_failures


def placement_request(args: argparse.Namespace) -> dict:
    if args.timestep < 1:
        raise ValueError("--timestep must be positive")
    request = {
        "placement_mode": "future_timestep",
        "placement_version": PAIAV_PLACEMENT_VERSION,
        "future_timestep": int(args.timestep),
        "expected_future_offset_s": float(args.timestep * 0.1),
        "history_interval_s": 0.1,
    }
    if getattr(args, "car", False):
        request.update(
            {
                "placement_version": PAIAV_CAR_PLACEMENT_VERSION,
                "actor_type": "car",
                "vehicle_behavior": "stationary",
                "vehicle_position_policy": (
                    "rear_bumper_at_future_ego_front_bumper"
                ),
                "vehicle_heading_policy": "future_ego_forward",
            }
        )
    elif getattr(args, "standing", False):
        request.update(
            {
                "placement_version": PAIAV_STANDING_PLACEMENT_VERSION,
                "actor_type": "pedestrian",
                "pedestrian_behavior": "standing",
                "standing_pose_version": compositor.STANDING_POSE_VERSION,
                "standing_position_policy": (
                    "stationary_at_future_ego_front_bumper"
                ),
                "standing_facing_policy": "opposite_future_ego_forward",
            }
        )
    return request


def paiav_annotation_sample(token: str, context: dict, output_paths: dict) -> dict:
    sample = {
        "token": token,
        "clip_id": token,
        **{
            camera: [
                str(output_paths[(index, camera)])
                for index in range(HISTORY_FRAME_COUNT)
            ]
            for camera in CAMERA_NAMES
        },
        "history_interval_s": 0.1,
        **context,
    }
    if context.get("actor_mode", "crossing") == "crossing":
        sample["animation_pose_sequence"] = list(POSE_SEQUENCE)
    return sample


def paiav_annotation_context(loader: PAIAVSceneLoader, token: str) -> dict:
    """Add PAI-AV evaluation fields without changing shared four-step fields."""
    context = {
        **_BASE_ANNOTATION_CONTEXT(loader, token),
        "alpamayo_ego_history": loader.alpamayo_ego_history(token),
        "ego_vehicle_geometry": loader.ego_vehicle_geometry(token),
    }
    context["navigation_command"] = loader.navigation_commands[token]
    return context


def install_paiav_adapter(asset_profile: str, actor_mode: str = "crossing") -> Path:
    asset_root = COMPOSITOR_ROOT / "blender_asset"
    if asset_profile == "car":
        compositor.RENDER_PIPELINE_VERSION = PAIAV_CAR_RENDER_PIPELINE_VERSION
    elif actor_mode == "standing":
        compositor.RENDER_PIPELINE_VERSION = (
            PAIAV_STANDING_RENDER_PIPELINE_VERSION
        )
    else:
        compositor.RENDER_PIPELINE_VERSION = PAIAV_RENDER_PIPELINE_VERSION

    compositor.CAMERA_ORDER = CAMERA_NAMES
    compositor.NAVSIM_POSE_SEQUENCE = POSE_SEQUENCE
    compositor.EXPECTED_HISTORY_FRAMES = HISTORY_FRAME_COUNT
    compositor.HISTORY_FRAME_INDICES = tuple(range(HISTORY_FRAME_COUNT))
    compositor.NAVSIM_TIMESTEP_S = 0.1
    compositor.FUTURE_TIMESTEP_PLACEMENT_VERSION = PAIAV_PLACEMENT_VERSION
    compositor.STANDING_PLACEMENT_VERSION = PAIAV_STANDING_PLACEMENT_VERSION
    compositor.CAR_PLACEMENT_VERSION = PAIAV_CAR_PLACEMENT_VERSION
    compositor.DIRECTION_CLASSIFICATION_POLICY_VERSION = (
        PAIAV_DIRECTION_POLICY_VERSION
    )
    compositor.GROUNDING_METHOD = PAIAV_GROUNDING_VERSION
    compositor.CAR_GROUNDING_METHOD = PAIAV_CAR_GROUNDING_VERSION
    compositor.frame_source_path = frame_source_path
    compositor.geometry.load_lidar_xyz = load_lidar_xyz
    compositor.animation_frames = fixed_animation_frames
    compositor.astronomical_sun_direction = contact_only_sun
    compositor.placement_request = placement_request
    compositor.annotation_context = paiav_annotation_context
    compositor.annotation_sample = paiav_annotation_sample
    compositor.fit_ground_sequence_best_effort = paiav_sequence_consistent_ground
    return asset_root


def select_tokens(clip_ids: Sequence[str], max_clips: int | None, seed: int) -> list[str]:
    return compositor.seeded_random_subset(list(clip_ids), max_clips, seed)


def balanced_directions(tokens: Sequence[str], seed: int) -> Dict[str, str]:
    ranked = sorted(
        tokens,
        key=lambda token: (compositor.token_uniform(token, seed, "direction"), token),
    )
    return {
        token: ("left" if index % 2 == 0 else "right")
        for index, token in enumerate(ranked)
    }


def configure_directions(
    args: argparse.Namespace,
    tokens: Sequence[str],
    run_dir: Path,
) -> dict:
    if getattr(args, "standing", False) or getattr(args, "car", False):
        args.direction_classifications = None
        args.direction_classification_labels = {}
        args.direction_classifications_sha256 = None
        return {
            "source": "not_applicable_stationary_actor",
            "source_path": None,
            "resolved_path": None,
            "sha256": None,
            "policy": None,
            "counts": {},
        }
    if args.direction_classifications is None:
        labels = balanced_directions(tokens, args.seed)
        source = "generated"
        source_path = None
    else:
        source_path = args.direction_classifications.expanduser().resolve()
        labels = compositor.load_direction_classifications(source_path)
        source = "external"
    compositor.validate_direction_classifications(tokens, labels)
    output_path = run_dir / "direction_classifications.json"
    compositor.write_json(output_path, labels)
    args.direction_classifications = output_path
    args.direction_classification_labels = labels
    args.direction_classifications_sha256 = compositor.file_sha256(output_path)
    return {
        "source": source,
        "source_path": str(source_path) if source_path else None,
        "resolved_path": str(output_path),
        "sha256": args.direction_classifications_sha256,
        "policy": (
            PAIAV_DIRECTION_POLICY_VERSION
            if source == "generated"
            else "external_clip_direction_classifications_v1"
        ),
        "counts": {
            label: sum(value == label for value in labels.values())
            for label in ("left", "right")
        },
    }


def configure_engine_args(args: argparse.Namespace, run_dir: Path, asset_root: Path) -> None:
    args.data_root = args.data_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.asset_dir = args.asset_dir.expanduser().resolve()
    args.blender = args.blender.expanduser().resolve()
    args.asset_root = asset_root.resolve()
    args.depth_python = args.depth_python.expanduser().resolve()
    args.depth_model_cache = args.depth_model_cache.expanduser().resolve()
    if args.depth_prediction_cache is None:
        args.depth_prediction_cache = (
            args.output_root / "_cache" / "depth_anything_v2_small_hf"
        ).resolve()
    else:
        args.depth_prediction_cache = args.depth_prediction_cache.expanduser().resolve()
    args.scene_filter = args.manifest
    args.split = "paiav"
    args.max_samples = args.max_clips
    args.gait_frame_advance = 23
    args.gait_timestep_s = 0.5
    args.shadow_label_metadata = compositor.contact_only_shadow_metadata()
    # A completed token is already self-contained and has a completion
    # manifest.  Preserve it on a resumed launch instead of needlessly
    # rebuilding all of its views; explicit --overwrite remains available to
    # the Python entry point when regeneration is intentional.
    args.refinalize = False
    args.depth_fail_open = not args.strict_depth_alignment
    args.protect_lower_actor_without_lidar = True
    args.atomic_token_publication = True
    args.minimum_projected_bbox_area_px2 = float(
        args.min_projected_bbox_area
    )


def validate_runtime(args: argparse.Namespace) -> None:
    if getattr(args, "stage", "all") != "render" and importlib.util.find_spec("DracoPy") is None:
        raise RuntimeError(
            "DracoPy is not installed in the selected Python environment. "
            "Install it once with `python -m pip install DracoPy`; it is required "
            "to decode the downloaded PAI-AV LiDAR point clouds."
        )
    if not args.prepare_only:
        if not args.blender.is_file():
            raise FileNotFoundError(f"Blender runtime not found: {args.blender}")
        if args.occlusion_mode == "dense":
            if not args.depth_python.is_file():
                raise FileNotFoundError(
                    f"Depth-inference Python not found: {args.depth_python}"
                )
            if not args.depth_model_cache.is_dir():
                raise FileNotFoundError(
                    f"Depth-Anything model cache not found: {args.depth_model_cache}"
                )


def ego_mask_compositing_config(mask_selection: dict | None = None) -> dict:
    return {
        "version": compositor.EGO_MASK_COMPOSITING_VERSION,
        "source_field": "per_camera_applied_ego_mask" if mask_selection is not None else ego_masks.DATASET_SOURCE,
        **({"selection": mask_selection} if mask_selection is not None else {}),
        "protected_layers": [
            "actor_alpha",
            "directional_shadow",
            "ambient_occlusion",
            "combined_edit_mask",
            "pctnet_foreground_mask",
        ],
        "final_source_pixel_restore": True,
    }


def run_config_document(
    args: argparse.Namespace,
    selection,
    tokens: Sequence[str],
    assets: Sequence[dict],
    directions: dict,
    asset_profile: str,
) -> dict:
    actor_mode = (
        "car"
        if getattr(args, "car", False)
        else "standing"
        if getattr(args, "standing", False)
        else "crossing"
    )
    effective_speed_mps = compositor.effective_pedestrian_speed_mps(args)
    selected_clip_chunk_ids = {
        token: selection.chunk_id_for(token) for token in tokens
    }
    selected_chunk_ids = sorted(set(selected_clip_chunk_ids.values()))
    dataset_config = {
        "repo_id": selection.manifest.get("dataset", {}).get("repo_id"),
        "data_root": str(args.data_root),
        "selection_manifest": str(selection.manifest_path),
        "selection_manifest_role": selection.manifest_role,
        "selection_manifest_sha256": compositor.file_sha256(
            selection.manifest_path
        ),
        "selected_chunk_ids": selected_chunk_ids,
        "selected_clip_chunk_ids": selected_clip_chunk_ids,
    }
    if len(selected_chunk_ids) == 1:
        dataset_config["selected_chunk_id"] = selected_chunk_ids[0]
    if selection.catalog_index_path is not None:
        dataset_config["catalog_clip_index"] = str(selection.catalog_index_path)
        dataset_config["catalog_clip_index_sha256"] = (
            selection.catalog_index_sha256
        )
    document = {
        "document_role": "paiav_edit_run_config",
        "schema_version": 2,
        "dataset": dataset_config,
        "run_name": args.run_name,
        "selected_clip_ids": list(tokens),
        "selected_clip_ids_sha256": compositor.canonical_hash(list(tokens)),
        "selected_clip_count": len(tokens),
        "output_root": str(args.output_root),
        "source_policy": "read_chunk_zip_members_without_full_extraction_v1",
        "history": {
            "anchor_time_s": args.anchor_time_s,
            "frame_count": HISTORY_FRAME_COUNT,
            "interval_s": HISTORY_INTERVAL_US / 1e6,
            "offsets_s": [-0.3, -0.2, -0.1, 0.0],
            "source_frame_selection": "nearest_camera_frame_to_nominal_timestamp",
            "camera_timing_compensation": (
                "actual_frame_timestamp_pose_plus_top_to_bottom_row_poses_"
                "with_center_row_assumption_v2"
            ),
        },
        **placement_request(args),
        "actor_mode": actor_mode,
        "speed_mps": effective_speed_mps,
        "cameras": list(CAMERA_NAMES),
        "camera_model": "native_paiav_ftheta_dense_ideal_pinhole_remap_v1",
        "ego_mask_compositing": ego_mask_compositing_config(getattr(args, "ego_mask_selection", None)),
        "egomotion": "egomotion.offline_interpolated_at_nominal_timestamps",
        "lidar": {
            "source": "lidar_top_360fov_nearest_spin",
            "decoder": "DracoPy",
            "alignment": "per_point_capture_pose_to_nominal_rig_v2",
            "per_point_motion_compensation": True,
        },
        "asset_profile": asset_profile,
        "asset_directory": str(args.asset_dir),
        "asset_selection": "blake2b_clip_id_seed_v1",
        "asset_pool": [compositor.asset_config(asset) for asset in assets],
        "directions": directions,
        "shadow_policy": "uniform_contact_only",
        "shadow_direction_labels": False,
        "render_pipeline_version": compositor.RENDER_PIPELINE_VERSION,
        "grounding_method": (
            compositor.CAR_GROUNDING_METHOD
            if actor_mode == "car"
            else compositor.GROUNDING_METHOD
        ),
        "depth_model": compositor.DEPTH_MODEL_ID,
        "depth_prediction_cache": str(args.depth_prediction_cache),
        "occlusion": {
            "requested_mode": args.occlusion_mode,
            "dense_alignment_fail_open": args.depth_fail_open,
            "fallback_mode": (
                "none_preserve_actor"
                if args.occlusion_mode == "dense"
                else "coherent_lidar_surface"
            ),
            "lidar_policy_in_dense_mode": "seed_only_never_direct_matte",
            "surface_reconstruction": (
                compositor.DENSE_OCCLUSION_VERSION
                if args.occlusion_mode == "dense"
                else compositor.LIDAR_FOREGROUND_SUPPORT_VERSION
            ),
            "protect_lower_actor_without_lidar": True,
            "max_unconfirmed_fraction": (
                args.max_unconfirmed_occlusion_fraction
            ),
        },
        "publication": {
            "atomic_per_token": True,
            "expected_views_per_token": (
                HISTORY_FRAME_COUNT * len(CAMERA_NAMES)
            ),
            "completion_manifest_tree": "completed_tokens",
        },
        "render": {
            "samples": args.samples,
            "scale": args.render_scale,
            "motion_blur": (
                False
                if actor_mode in {"standing", "car"}
                else not args.disable_motion_blur
            ),
            "actor_blur_sigma": args.actor_blur_sigma,
            "max_actor_noise_std": args.max_actor_noise_std,
            "harmonization_strength": args.harmonization_strength,
            "max_ground_tilt_deg": args.max_ground_tilt_deg,
            "collision_check": not args.disable_collision_check,
            "render_batch_size": args.render_batch_size,
            "render_workers": args.render_workers,
            "finalize_workers": args.finalize_workers,
            "minimum_projected_bbox_area_px2": args.min_projected_bbox_area,
        },
        "source_files_modified": False,
    }
    if actor_mode == "crossing":
        document["animation"] = {
            "policy": "fixed_ready_pose_sequence_v1",
            "pose_sequence": list(POSE_SEQUENCE),
            "nominal_interval_s": 0.1,
            "stride_length_m": args.stride_length_m,
            "derived_stride_reference": {
                "gait_frame_advance": args.gait_frame_advance,
                "gait_timestep_s": args.gait_timestep_s,
            },
        }
    elif actor_mode == "standing":
        document["animation"] = {
            "policy": "procedural_natural_standing_static_v1",
            "standing_pose_version": compositor.STANDING_POSE_VERSION,
            "animation_frame": compositor.STANDING_ANIMATION_FRAME,
        }
    else:
        document["animation"] = {
            "policy": "rigid_static_vehicle_v1",
            "animation_frame": compositor.CAR_STATIC_ANIMATION_FRAME,
        }
    return document


def _known_occlusion_config_migration(existing: dict, config: dict) -> bool:
    differing = {
        key
        for key in set(existing) | set(config)
        if existing.get(key) != config.get(key)
    }
    if differing != {"occlusion"}:
        return False
    previous = existing.get("occlusion")
    current = config.get("occlusion")
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    mode = previous.get("requested_mode")
    if mode not in {"none", "lidar", "dense"} or mode != current.get(
        "requested_mode"
    ):
        return False
    previous_surface = (
        compositor.PREVIOUS_DENSE_OCCLUSION_VERSION
        if mode == "dense"
        else compositor.PREVIOUS_LIDAR_FOREGROUND_SUPPORT_VERSION
    )
    current_surface = (
        compositor.DENSE_OCCLUSION_VERSION
        if mode == "dense"
        else compositor.LIDAR_FOREGROUND_SUPPORT_VERSION
    )
    if (
        previous.get("surface_reconstruction") != previous_surface
        or current.get("surface_reconstruction") != current_surface
        or previous.get("fallback_mode") != "coherent_lidar_surface"
        or current.get("fallback_mode")
        != (
            "none_preserve_actor"
            if mode == "dense"
            else "coherent_lidar_surface"
        )
        or "lidar_policy_in_dense_mode" in previous
        or current.get("lidar_policy_in_dense_mode")
        != "seed_only_never_direct_matte"
    ):
        return False
    migrated_fields = {
        "fallback_mode",
        "surface_reconstruction",
        "lidar_policy_in_dense_mode",
    }
    return (
        {key: value for key, value in previous.items() if key not in migrated_fields}
        == {key: value for key, value in current.items() if key not in migrated_fields}
    )


def _known_ego_mask_config_migration(existing: dict, config: dict) -> bool:
    """Allow explicit regeneration of a run made before calibrated self masking."""
    if "ego_mask_compositing" in existing:
        return False
    ego_mask_config = config.get("ego_mask_compositing")
    if ego_mask_config != ego_mask_compositing_config():
        return False
    config_without_ego_mask = {
        key: value
        for key, value in config.items()
        if key != "ego_mask_compositing"
    }
    return existing == config_without_ego_mask or _known_occlusion_config_migration(
        existing, config_without_ego_mask
    )


def write_or_validate_run_config(
    run_dir: Path,
    config: dict,
    *,
    allow_occlusion_migration: bool = False,
) -> None:
    path = run_dir / "run_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            # Scheduling resources may differ between CPU preparation and a
            # render-only retry without changing the generated sample.
            def sample_config(document):
                result = dict(document)
                result["render"] = {key: value for key, value in document.get("render", {}).items()
                                    if key not in {"render_workers", "render_batch_size", "finalize_workers"}}
                return result
            if sample_config(existing) == sample_config(config):
                return
            if allow_occlusion_migration and (
                _known_occlusion_config_migration(existing, config)
                or _known_ego_mask_config_migration(existing, config)
            ):
                compositor.write_json(path, config)
                return
            differing = sorted(
                key
                for key in set(existing) | set(config)
                if existing.get(key) != config.get(key)
            )
            raise RuntimeError(
                f"Run configuration conflicts with {path}; differing top-level "
                f"keys={differing}. Choose a new --run-name."
            )
    else:
        compositor.write_json(path, config)


def enrich_success_metadata(
    run_dir: Path,
    tokens: Sequence[str],
    loader: PAIAVSceneLoader,
    selection,
    navigation_commands: dict[str, str | None],
) -> None:
    for token in tokens:
        path = run_dir / "metadata" / f"{token}.json"
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        annotation = document.get("annotation_sample")
        if not isinstance(annotation, dict):
            raise RuntimeError(f"Successful metadata lacks annotation_sample: {path}")
        alpamayo_ego_history = loader.alpamayo_ego_history(token)
        ego_vehicle_geometry = loader.ego_vehicle_geometry(token)
        annotation["alpamayo_ego_history"] = alpamayo_ego_history
        annotation["ego_vehicle_geometry"] = ego_vehicle_geometry
        annotation["navigation_command"] = navigation_commands[token]
        document["dataset"] = "nvidia/PhysicalAI-Autonomous-Vehicles"
        document["selection_manifest"] = str(selection.manifest_path)
        document["paiav_source"] = loader.clip_metadata(token)
        document["ego_vehicle_geometry"] = ego_vehicle_geometry
        if document.get("actor_mode", "crossing") == "crossing":
            document["animation_pose_sequence"] = list(POSE_SEQUENCE)
        else:
            document.pop("animation_pose_sequence", None)
        document["history_interval_s"] = 0.1
        document["shadow_policy"] = "uniform_contact_only"
        compositor.write_json(path, document)


def prepared_metadata(loader, token, selection, actor_mode):
    metadata = {
        "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles",
        "selection_manifest": str(selection.manifest_path),
        "paiav_source": loader.clip_metadata(token),
        "ego_vehicle_geometry": loader.ego_vehicle_geometry(token),
        "history_interval_s": 0.1,
        "shadow_policy": "uniform_contact_only",
    }
    if actor_mode == "crossing":
        metadata["animation_pose_sequence"] = list(POSE_SEQUENCE)
    return metadata


def render_prepared(args, run_dir, prepared_tokens, tokens):
    if not args.overwrite and not args.refinalize:
        prepared_tokens = [item for item in prepared_tokens if not (
            (run_dir / "metadata" / f"{item['token']}.json").is_file()
            and compositor.token_completion_path(run_dir, item["token"]).is_file()
        )]
    jobs = edit_preparation.refresh_render_jobs(args, prepared_tokens, compositor)
    compositor.configure_finalization_native_threads(args.finalize_workers)
    compositor.run_depth_batch(args, run_dir, prepared_tokens)
    compositor.run_blender_with_streaming_finalization(args, None, prepared_tokens, run_dir, jobs)
    successful, failed = compositor.aggregate_results(run_dir, tokens)
    print(f"Finished: {successful} successful, {failed} failed")
    print(f"Output: {run_dir}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    stage = getattr(args, "stage", "prepare" if args.prepare_only else "all")
    run_dir = args.output_root.expanduser().resolve() / args.run_name
    if (run_dir / "compaction_report.json").exists():
        raise RuntimeError("This run has entered evaluation-only compaction; choose a new run name")
    request = edit_preparation.request_identity(args, [
        __file__, compositor.__file__, HERE / "paiav_io.py", HERE / "ego_masks.py",
        HERE / "_ego_mask_refinement.py",
        compositor.geometry.__file__, compositor.render_utils.__file__, edit_preparation.__file__,
    ]) if stage != "all" else None
    store = edit_preparation.PreparationStore(run_dir, "paiav", request) if request else None
    if stage == "render":
        args, prepared_tokens, index = store.load(args)
        install_paiav_adapter(index["extra"]["asset_profile"], index["extra"]["actor_mode"])
        validate_runtime(args)
        render_prepared(args, run_dir, prepared_tokens, index["tokens"])
        return 0
    selection = load_selection_manifest(args.manifest)
    tokens = select_tokens(selection.clip_ids, args.max_clips, args.seed)
    if not tokens:
        raise ValueError("The requested selection contains no clips")
    navigation_commands = load_navigation_commands(tokens)

    asset_profile = detect_asset_profile(args.asset_dir, car=args.car)
    validate_actor_asset_profile(args, asset_profile)
    actor_mode = (
        "car" if args.car else "standing" if args.standing else "crossing"
    )
    asset_root = install_paiav_adapter(asset_profile, actor_mode)
    run_dir = args.output_root.expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_engine_args(args, run_dir, asset_root)
    validate_runtime(args)
    finalization_threads = compositor.configure_finalization_native_threads(
        args.finalize_workers
    )
    print(
        f"Finalization CPU budget: {compositor.allocated_cpu_count()} allocated "
        f"CPU(s), {args.finalize_workers} worker(s), "
        f"{finalization_threads} OpenCV thread(s) per operation"
    )

    directions = configure_directions(args, tokens, run_dir)
    assets = compositor.load_asset_pool(args)
    loader = PAIAVSceneLoader(
        args.data_root,
        selection,
        tokens,
        run_dir,
        anchor_time_s=args.anchor_time_s,
        future_timestep=args.timestep,
        **ego_masks.loader_options(args),
    )
    args.ego_mask_selection = loader.ego_masks.run_identity(tokens)
    loader.navigation_commands = navigation_commands
    run_config = run_config_document(
        args, selection, tokens, assets, directions, asset_profile
    )
    write_or_validate_run_config(
        run_dir,
        run_config,
        allow_occlusion_migration=bool(args.overwrite or args.refinalize),
    )
    if store is not None:
        store.begin(tokens, args, extra={"asset_profile": asset_profile, "actor_mode": actor_mode},
                    inputs=[path for paths in loader.chunk_feature_paths.values() for path in paths.values()])

    print("Starting PAI-AV edit pipeline")
    print(f"RUN_DIR={run_dir}")
    print(f"CLIPS={len(tokens)}")
    print(f"CAMERAS={','.join(CAMERA_NAMES)}")
    print(f"ACTOR_MODE={actor_mode}")
    if actor_mode == "crossing":
        print(f"POSES={','.join(str(value) for value in POSE_SEQUENCE)}")
    elif actor_mode == "standing":
        print(f"POSE_POLICY={compositor.STANDING_POSE_VERSION}")
    else:
        print("POSE_POLICY=rigid_static_vehicle")
    print("HISTORY_INTERVAL_S=0.1")
    print("SHADOW_POLICY=uniform_contact_only")

    prepared_tokens = []
    jobs = []
    prepared_count = 0
    render_job_count = 0
    for index, token in enumerate(tokens):
        metadata_path = run_dir / "metadata" / f"{token}.json"
        completion_path = compositor.token_completion_path(run_dir, token)
        failure_path = run_dir / "failures" / f"{token}.json"
        if (
            metadata_path.is_file()
            and completion_path.is_file()
            and not args.overwrite
            and not args.refinalize
        ):
            print(f"[{index + 1}/{len(tokens)}] cached success {token}")
            if store is not None:
                store.skip(token, "cached_success", [metadata_path, completion_path])
            continue
        if metadata_path.is_file() and not completion_path.is_file():
            print(
                f"[{index + 1}/{len(tokens)}] incomplete legacy publication "
                f"{token}; rebuilding all views"
            )
        if failure_path.is_file() and not args.retry_failures and not args.overwrite:
            print(f"[{index + 1}/{len(tokens)}] cached failure {token}")
            if store is not None:
                store.skip(token, "failed", [failure_path])
            continue
        try:
            prepared = None
            if store is not None and not args.overwrite:
                try:
                    prepared = store.restore(token)
                except edit_preparation.PreparationError as exc:
                    print(f"Rebuilding {token}: {exc}")
            asset = compositor.select_asset(token, args.seed, assets)
            if prepared is None:
                prepared = compositor.prepare_token(
                    args, loader, token, run_dir, loader.front_bumper_offset_m(token), asset,
                )
                if store is not None:
                    prepared["annotation_context"] = compositor.annotation_context(loader, token)
                    prepared["annotation_context"]["actor_mode"] = actor_mode
                    prepared["metadata_extra"] = prepared_metadata(loader, token, selection, actor_mode)
            potential_failure_path = run_dir / "potential_failures" / f"{token}.json"
            if potential_failure_path.exists():
                potential_failure_path.unlink()
            for warning in prepared["potential_failures"]:
                compositor.record_potential_failure(run_dir, warning)
            if store is None:
                prepared_tokens.append(prepared)
                jobs.extend(prepared["jobs"])
            print(
                f"[{index + 1}/{len(tokens)}] prepared {token}: "
                f"{prepared['direction']}, asset={asset['id']}, "
                f"{len(prepared['jobs'])} renders"
            )
        except compositor.TokenFailure as exc:
            compositor.record_failure(run_dir, token, exc.code, str(exc), args)
            prepared = None
        except OSError as exc:
            if store is not None:
                raise
            compositor.record_failure(run_dir, token, "preparation_failed", repr(exc), args)
            prepared = None
        except Exception as exc:
            compositor.record_failure(
                run_dir, token, "preparation_failed", repr(exc), args
            )
            prepared = None
        if prepared is not None:
            prepared_count += 1
            render_job_count += len(prepared["jobs"])
        if store is not None:
            if prepared is None:
                store.skip(token, "failed", [failure_path])
            else:
                store.save(prepared, edit_preparation.prepared_input_paths(prepared, args.data_root, args.split))
            loader.release_clip(token)

    preparation_summary = {
        "selected_clips": len(tokens),
        "prepared_clips": prepared_count,
        "uncached_render_jobs": render_job_count,
        "prepare_only": bool(args.prepare_only),
    }
    compositor.write_json(run_dir / "preparation_summary.json", preparation_summary)
    print(
        f"Prepared {prepared_count}/{len(tokens)} clips with "
        f"{render_job_count} uncached views"
    )
    if args.prepare_only:
        store.finish()
        print(f"Preparation-only output: {run_dir}")
        return 0

    compositor.run_depth_batch(args, run_dir, prepared_tokens)
    compositor.run_blender_with_streaming_finalization(
        args, loader, prepared_tokens, run_dir, jobs
    )
    enrich_success_metadata(
        run_dir,
        tokens,
        loader,
        selection,
        navigation_commands,
    )
    successful, failed = compositor.aggregate_results(run_dir, tokens)
    print(f"Finished: {successful} successful, {failed} failed")
    print(f"Output: {run_dir}")
    # Per-clip failures are an expected, recoverable batch outcome and are
    # recorded in failures.jsonl/summary.json.  Exit successfully so that PCTNet
    # can harmonize every clip that did complete; process-level exceptions still
    # exit with an error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
