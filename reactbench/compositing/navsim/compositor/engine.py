#!/usr/bin/env python3
"""Create receiver previews or render final NAVSIM actor composites."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# OpenCV ships its OpenEXR codec disabled-by-default on some distributions.
# This must be set before importing cv2 so the coordinator can read the
# scene-linear standalone catcher passes emitted by Blender.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image
from pyproj import Geod, Proj, Transformer
import yaml

from . import benign_placements, benign_sides, front_visibility, geometry, render_utils
from ... import edit_preparation, locations
from ....scoring.pedestrian_geometry import normalize_pedestrian_geometry


HERE = Path(__file__).resolve().parent

DEFAULT_BLENDER = locations.BLENDER
# Runs are written to <output root>/<scene filter stem>/<run name>.
DEFAULT_OUTPUT_ROOT = locations.REACTNAV.parent
CAMERA_KEYS = geometry.CAMERA_KEYS
CAMERA_ORDER = ("CAM_L0", "CAM_F0", "CAM_R0")
NAVSIM_POSE_SEQUENCE = (24, 15, 6, 29)
EXPECTED_HISTORY_FRAMES = len(NAVSIM_POSE_SEQUENCE)
HISTORY_FRAME_INDICES = tuple(range(EXPECTED_HISTORY_FRAMES))
GROUNDING_METHOD = "strict_then_four_frame_best_effort_v1"
# Cycles' transparent actor render can leave a broad 1--3/255 alpha haze.
# Removing that numerically insignificant tail prevents background pixels from
# changing by one code value; the float catcher pass has its own threshold.
MIN_COMPOSITE_ALPHA = 4
RENDER_PIPELINE_VERSION = (
    "photoreal_counterfactual_v20_measured_pedestrian_geometry"
)
BASE_RENDER_PASS_POLICY_VERSION = "label_aware_base_actor_sun_v1"
BASE_RENDER_CORE_PASSES = (
    "actor_color",
    "actor_depth",
    "actor_sky",
    "ambient_occlusion",
)
BASE_RENDER_ACTOR_SUN_PASS = "actor_sun"
BASE_RENDER_FULL_PASSES = (*BASE_RENDER_CORE_PASSES, BASE_RENDER_ACTOR_SUN_PASS)
FUTURE_TIMESTEP_PLACEMENT_VERSION = (
    "future_ego_front_bumper_timed_interception_v2"
)
STANDING_PLACEMENT_VERSION = (
    "future_ego_front_bumper_stationary_obstacle_v1"
)
CAR_PLACEMENT_VERSION = "future_ego_front_to_stationary_car_rear_contact_v1"
CAR_GEOMETRY_VERSION = "openx_xoma_vehicle_geometry_v1"
CAR_CONTACT_TOLERANCE_M = 0.01
CAR_STATIC_ANIMATION_FRAME = 1
CAR_ROOT_OBJECT_NAME = "Grp_Root"
CAR_CONTACT_PROFILE = "openx_vehicle_wheels"
STANDING_POSE_VERSION = "procedural_natural_standing_v2"
STANDING_ANIMATION_FRAME = 1
# Rocketbox characters face local -Y. A -pi/2 rotation makes local -Y point
# along road-basis -X, toward the ego approaching the interception point.
STANDING_ACTOR_HEADING_RAD = -math.pi / 2.0
STANDING_FACING_POLICY = "opposite_future_ego_forward"
NAVSIM_TIMESTEP_S = 0.5
PEDESTRIAN_SPEED_MPS = 1.4
DIRECTION_CLASSIFICATION_POLICY_VERSION = "external_token_direction_classifications_v1"
DIRECTION_CLASSIFICATION_SPECS = {
    "left": {
        "crossing_name": "right_to_left",
        "lateral_sign": 1.0,
        "actor_heading_rad": math.pi,
    },
    "right": {
        "crossing_name": "left_to_right",
        "lateral_sign": -1.0,
        "actor_heading_rad": 0.0,
    },
}
DIRECTION_CLASSIFICATION_ALLOWED_LABELS = {
    *DIRECTION_CLASSIFICATION_SPECS,
    "failure",
}
PREVIEW_PLACEMENT_FLOAT_ATOL = 1e-9
RAW_RENDER_STORAGE_VERSION = "cropped_render_region_v1"
FINALIZATION_CACHE_VERSION = "label_independent_sensor_intermediates_v2"
RENDER_REGION_PADDING_PX = 256
RENDER_REGION_ALIGNMENT_PX = 8
SHADOW_RECEIVER_HALF_EXTENT_M = 100.0
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# The cache stores the exact full-resolution float32 array previously written
# below each run directory.  Bump this namespace if model preprocessing,
# interpolation, dtype, or array semantics change.
DEPTH_PREDICTION_CACHE_VERSION = "fullres_float32_v1"
ACTOR_DEPTH_ENCODING_MAX_M = 100.0
ACTOR_CORE_ALPHA = 32
ACTOR_DEPTH_COVERAGE_WARNING_FRACTION = 0.50
SHADOW_RECEIVER_MAX_DEPTH_M = 120.0
# Suppress sub-1.2% Monte-Carlo/denoiser variation on the otherwise uniform
# receiver.  This is a direct attenuation cutoff, not a spatial blur, so broad
# legitimate penumbrae remain intact above the numerically invisible tail.
SHADOW_MIN_AMOUNT = 0.012
AMBIENT_OCCLUSION_MIN_AMOUNT = 0.002
COMBINED_MASK_SHADOW_MAX = 249
RECEIVER_MATTE_ACTOR_EDGE_RADIUS_PX = 2
RECEIVER_MATTE_OPAQUE_ALPHA = 250
DIRECT_SUN_FADE_START = 0.15
DIRECT_SUN_FADE_END = 0.35
CLEAR_SUN_ANGLE_MIN_DEGREES = 0.50
CLEAR_SUN_ANGLE_MAX_DEGREES = 2.00
FOOT_CONTACT_DISTANCE_M = 0.030
FOOT_STANCE_RELATIVE_DISTANCE_M = 0.012
FOOT_SOLE_BAND_M = 0.018
# Label-driven finalization treats the VLM recommendation as the sole
# photometric shadow decision.  These fixed lookups translate its categorical
# output into deterministic compositor/render parameters.
SHADOW_LABEL_COMPOSITION_VERSION = (
    "per_timestep_vlm_shadow_v8_gated_receiver_depth_recovery"
)
CONTACT_ONLY_SHADOW_POLICY_VERSION = "uniform_contact_only_cli_v1"
SHADOW_LABEL_DIRECTIONAL_RENDER_VERSION = "selected_sun_only_v1"
SHADOW_LABEL_COUNTERFACTUAL_SUN_ENERGY = 2.0
DIRECTIONAL_SHADOW_RECOVERY_VERSION = (
    "gated_local_receiver_depth_bias_v1"
)
SHADOW_LABEL_DIRECTIONAL_GAINS = {
    "contact_only": 0.0,
    "medium": 0.35,
    "strong": 0.70,
}
SHADOW_LABEL_AMBIENT_OCCLUSION_GAINS = {
    # With no direct-light component, the Sun-off catcher represents the full
    # local illumination loss and should not be attenuated a second time.
    "contact_only": 1.0,
    # In direct-light scenes this remains a restrained contribution alongside
    # the independently scaled directional visibility matte.
    "medium": 0.35,
    "strong": 0.35,
}
# Full-body sky occlusion is much weaker than the stance-contact core.  Use the
# same cutoff as directional shadows to reject the 32-sample catcher's
# low-amplitude far-field variation while retaining the lower threshold for the
# explicitly measured stance-foot component.
DIFFUSE_BODY_OCCLUSION_MIN_AMOUNT = SHADOW_MIN_AMOUNT
SHADOW_LABEL_ACTOR_SUN_GAINS = {
    "contact_only": 0.0,
    "medium": 0.70,
    "strong": 1.0,
}
SHADOW_LABEL_SUN_ANGLES_DEGREES = {
    "soft": 2.0,
    "medium": 1.0,
    "hard": 0.53,
}
WGS84_GEOD = Geod(ellps="WGS84")
MAP_UTM_CRS = {
    "us-ma-boston": "EPSG:32619",
    "us-nv-las-vegas-strip": "EPSG:32611",
    "las_vegas": "EPSG:32611",
    "us-pa-pittsburgh-hazelwood": "EPSG:32617",
    "sg-one-north": "EPSG:32648",
}


class TokenFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def resolve_placement(
    scene_frames: Sequence[dict],
    timestep: int,
    front_bumper_offset_m: float,
    *,
    current_frame_index: int = EXPECTED_HISTORY_FRAMES - 1,
) -> dict:
    """Resolve a future front-bumper interception for one NAVSIM sample."""
    if (
        not isinstance(timestep, (int, np.integer))
        or isinstance(timestep, (bool, np.bool_))
        or int(timestep) < 1
    ):
        raise ValueError("--timestep must be a positive integer")
    if not math.isfinite(front_bumper_offset_m):
        raise ValueError("front-bumper offset must be finite")
    future_frame_index = int(current_frame_index) + int(timestep)
    if future_frame_index >= len(scene_frames):
        raise TokenFailure(
            "future_timestep_unavailable",
            f"requested future timestep {int(timestep)} requires frame index "
            f"{future_frame_index}, but the sample has {len(scene_frames)} frames",
        )

    current_timestamp_us = float(
        scene_frames[int(current_frame_index)].get("timestamp", math.nan)
    )
    future_frame = scene_frames[future_frame_index]
    future_timestamp_us = float(future_frame.get("timestamp", math.nan))
    if not (
        math.isfinite(current_timestamp_us)
        and math.isfinite(future_timestamp_us)
        and future_timestamp_us > current_timestamp_us
    ):
        raise TokenFailure(
            "invalid_future_timestamp",
            "future-timestep placement requires a finite timestamp later than "
            "the final history frame",
        )
    future_pose = np.asarray(
        future_frame.get("ego2global", []), dtype=np.float64
    )
    if future_pose.shape != (4, 4) or not np.isfinite(future_pose).all():
        raise TokenFailure(
            "invalid_future_ego_pose",
            f"future frame index {future_frame_index} must contain a finite "
            "4x4 ego2global transform",
        )
    future_front_bumper_global_xy = transform_xy_planar(
        future_pose,
        [float(front_bumper_offset_m), 0.0],
    )
    future_offset_s = (future_timestamp_us - current_timestamp_us) / 1e6
    return {
        "placement_mode": "future_timestep",
        "placement_version": FUTURE_TIMESTEP_PLACEMENT_VERSION,
        "future_timestep": int(timestep),
        "future_frame_index": future_frame_index,
        "current_timestamp_us": int(current_timestamp_us),
        "future_timestamp_us": int(future_timestamp_us),
        "expected_future_offset_s": float(int(timestep) * NAVSIM_TIMESTEP_S),
        "future_offset_s": float(future_offset_s),
        "front_bumper_offset_m": float(front_bumper_offset_m),
        "future_front_bumper_global_xy_m": (
            future_front_bumper_global_xy.tolist()
        ),
    }


def placement_request(args: argparse.Namespace) -> dict:
    """Validate and describe the requested future timestep."""
    timestep = getattr(args, "timestep", None)
    if (
        not isinstance(timestep, (int, np.integer))
        or isinstance(timestep, (bool, np.bool_))
        or int(timestep) < 1
    ):
        raise ValueError("--timestep must be a positive integer")
    request = {
        "placement_mode": "future_timestep",
        "placement_version": FUTURE_TIMESTEP_PLACEMENT_VERSION,
        "future_timestep": int(timestep),
        "expected_future_offset_s": float(int(timestep) * NAVSIM_TIMESTEP_S),
    }
    if getattr(args, "car", False):
        request.update(
            {
                "placement_version": CAR_PLACEMENT_VERSION,
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
                "placement_version": STANDING_PLACEMENT_VERSION,
                "pedestrian_behavior": "standing",
                "standing_pose_version": STANDING_POSE_VERSION,
                "standing_position_policy": (
                    "stationary_at_future_ego_front_bumper"
                ),
                "standing_facing_policy": STANDING_FACING_POLICY,
            }
        )
    if getattr(args, "benign", False):
        request.update(benign_placements.configuration(args))
        if hasattr(args, "benign_selection"):
            request["benign_selection"] = args.benign_selection
        request.pop("vehicle_position_policy", None)
        request.pop("standing_position_policy", None)
    visibility = front_visibility.configuration(args)
    if visibility["min_fraction"] > 0:
        request["front_visibility_requirement"] = visibility
    return request


def effective_pedestrian_speed_mps(args: argparse.Namespace) -> float:
    """Return zero for the opt-in stationary actor and the legacy speed otherwise."""
    if getattr(args, "standing", False) or getattr(args, "car", False):
        return 0.0
    return float(args.speed_mps)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    benign_placements.add_arguments(parser)
    benign_sides.add_arguments(parser)
    # ReactNAV has no front-visibility minimum; ReactTrain and the standing animals use 0.25 (their launchers pass it).
    front_visibility.add_arguments(parser, default=0.0)
    parser.add_argument("--scene-filter", type=Path, required=True)
    parser.add_argument(
        "--direction-classifications",
        type=Path,
        help=(
            "JSON object mapping NAVSIM tokens to left, right, or failure. "
            "Required for walking-pedestrian runs. Stationary pedestrian and "
            "car runs do not read this file."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=int,
        required=True,
        help=(
            "Use the ego front bumper at this future NAVSIM timestep as the "
            "interception/contact point."
        ),
    )
    actor_mode = parser.add_mutually_exclusive_group()
    actor_mode.add_argument(
        "--standing",
        action="store_true",
        help=(
            "Place a naturally posed stationary pedestrian at the selected "
            "future ego front-bumper point, facing the approaching ego."
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
    parser.set_defaults(speed_mps=PEDESTRIAN_SPEED_MPS)
    parser.add_argument("--data-root", type=Path, default=locations.NAVSIM_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "render"), default="all",
                        help="CPU preparation, render saved preparation, or the combined pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stride-length-m",
        type=float,
        help="Distance per gait cycle; by default it is derived to advance the requested number of frames per gait timestep.",
    )
    parser.add_argument(
        "--gait-frame-advance",
        type=int,
        default=23,
        help="Animation-frame advance per gait timestep when stride length is derived (default: 23).",
    )
    parser.add_argument(
        "--gait-timestep-s",
        type=float,
        default=0.5,
        help="Reference timestep used to derive stride length (default: 0.5 seconds).",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--render-scale", type=int, choices=(1, 2), default=2)
    parser.add_argument("--actor-blur-sigma", type=float, default=0.55)
    parser.add_argument("--max-actor-noise-std", type=float, default=1.5)
    parser.add_argument("--disable-motion-blur", action="store_true")
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--harmonization-strength", type=float, default=0.65)
    parser.add_argument(
        "--depth-python",
        type=Path,
        default=locations.DEPTH_PYTHON,
        help="Python environment with torch>=2.1 and transformers for depth inference.",
    )
    parser.add_argument(
        "--depth-model-cache",
        type=Path,
        default=locations.MODEL_CACHE,
        help=(
            "Hugging Face cache containing Depth-Anything-V2-Small-hf. "
            "Defaults to the cache in 3d_compositing_fast_v2."
        ),
    )
    parser.add_argument(
        "--depth-prediction-cache",
        type=Path,
        help=(
            "Persistent cache for full-resolution Depth-Anything predictions. "
            "Defaults to DATA_ROOT/../cache/depth_anything_v2_small_hf/fullres_float32_v1."
        ),
    )
    parser.add_argument(
        "--disable-collision-check",
        action="store_true",
        help="Suppress trajectory-collision potential-failure reporting (debugging only).",
    )
    parser.add_argument(
        "--max-ground-tilt-deg",
        type=float,
        default=2.0,
        help="Maximum actor pitch/roll away from gravity toward the fitted road normal.",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--asset-root", type=Path, default=HERE / "blender_asset")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        required=True,
        help=(
            "Directory of packed actor .blend files. Car mode discovers OpenX "
            ".blend/.xoma pairs recursively; pedestrian modes use direct "
            "children. One asset is selected reproducibly per token."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Select this many tokens without replacement using the deterministic --seed.",
    )
    parser.add_argument("--render-batch-size", type=int, default=96)
    parser.add_argument(
        "--render-workers",
        type=int,
        default=1,
        help=(
            "Maximum concurrent Blender subprocesses. GPU workers are isolated "
            "to separate CUDA_VISIBLE_DEVICES entries (default: 1)."
        ),
    )
    parser.add_argument(
        "--finalize-workers",
        type=int,
        default=4,
        help=(
            "Maximum concurrent token finalizers. Each worker owns its lens-remap "
            "cache, bounding the memory retained by parallel finalization "
            "(default: 4)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refinalize",
        action="store_true",
        help="Rebuild PNGs/metadata from cached Blender renders without rerendering them.",
    )
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args(argv)


def validate_actor_mode(args: argparse.Namespace) -> None:
    if getattr(args, "benign", False):
        benign_placements.configuration(args)
        if getattr(args, "disable_collision_check", False):
            raise ValueError("Benign placement requires collision checks")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=jsonable).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_direction_classifications(path: Path) -> Dict[str, str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Direction-classification file does not exist: {path}"
        )
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read direction-classification JSON {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError("Direction-classification JSON root must be an object")

    invalid = [
        (token, label)
        for token, label in document.items()
        if (
            not isinstance(token, str)
            or not token
            or not isinstance(label, str)
            or label not in DIRECTION_CLASSIFICATION_ALLOWED_LABELS
        )
    ]
    if invalid:
        raise ValueError(
            "Direction classifications must map non-empty token strings to "
            f"{sorted(DIRECTION_CLASSIFICATION_ALLOWED_LABELS)}; "
            f"invalid examples={invalid[:5]}"
        )
    return dict(document)


def configure_direction_classifications(args: argparse.Namespace) -> None:
    if getattr(args, "standing", False) or getattr(args, "car", False) or getattr(args, "benign", False):
        args.direction_classifications = None
        args.direction_classification_labels = {}
        args.direction_classifications_sha256 = None
        return
    if args.direction_classifications is None:
        raise ValueError(
            "--direction-classifications is required for walking pedestrians"
        )
    path = Path(args.direction_classifications).expanduser().resolve()
    args.direction_classifications = path
    args.direction_classification_labels = load_direction_classifications(path)
    args.direction_classifications_sha256 = file_sha256(path)


def direction_classification_mapping() -> dict:
    return {
        label: {
            "crossing_name": str(spec["crossing_name"]),
            "lateral_sign": float(spec["lateral_sign"]),
            "actor_heading_rad": float(spec["actor_heading_rad"]),
        }
        for label, spec in DIRECTION_CLASSIFICATION_SPECS.items()
    }


def direction_classification_provenance(args: argparse.Namespace) -> dict:
    if getattr(args, "standing", False) or getattr(args, "car", False) or getattr(args, "benign", False):
        return {}
    return {
        "direction_classifications_path": str(
            args.direction_classifications.expanduser().resolve()
        ),
        "direction_classifications_sha256": args.direction_classifications_sha256,
        "direction_policy": DIRECTION_CLASSIFICATION_POLICY_VERSION,
        "direction_classification_mapping": direction_classification_mapping(),
    }


def validate_direction_classifications(
    tokens: Sequence[str],
    classifications: Dict[str, str],
) -> None:
    missing = [token for token in tokens if token not in classifications]
    non_renderable = [
        (token, classifications[token])
        for token in tokens
        if token in classifications
        and classifications[token] not in DIRECTION_CLASSIFICATION_SPECS
    ]
    if missing or non_renderable:
        raise ValueError(
            "Every selected scene token must have a renderable left/right "
            "direction classification; "
            f"missing={missing[:5]}, non_renderable={non_renderable[:5]}"
        )


def render_storage_region(manifest: dict) -> Tuple[int, int, int, int]:
    """Return the ideal-canvas rectangle physically stored by a render.

    Version-12 renders use Blender's crop-to-border output.  The legacy branch
    deliberately remains readable for unit tests and old diagnostic data, but
    production manifests are cache-incompatible because the pipeline version
    is part of their render key.
    """
    canvas_width, canvas_height = [int(value) for value in manifest["resolution"]]
    if manifest.get("raw_render_storage") == RAW_RENDER_STORAGE_VERSION:
        region = [int(value) for value in manifest.get("render_region", [])]
        if len(region) != 4:
            raise TokenFailure(
                "render_failed",
                "cropped render manifest must contain a four-coordinate "
                f"render_region, got {region}",
            )
    else:
        region = [0, 0, canvas_width, canvas_height]
    x_min, y_min, x_max, y_max = region
    if not (
        0 <= x_min < x_max <= canvas_width
        and 0 <= y_min < y_max <= canvas_height
    ):
        raise TokenFailure(
            "render_failed",
            f"render storage region {region} is outside "
            f"{canvas_width}x{canvas_height}",
        )
    return x_min, y_min, x_max, y_max


def render_storage_resolution(manifest: dict) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = render_storage_region(manifest)
    return x_max - x_min, y_max - y_min


def render_storage_ideal_origin(manifest: dict) -> Tuple[float, float]:
    """Map a physically cropped ideal render into the global ideal camera."""
    overscan_x, overscan_y = [float(value) for value in manifest["ideal_origin"]]
    x_min, y_min, _x_max, _y_max = render_storage_region(manifest)
    return overscan_x + x_min, overscan_y + y_min


def contact_only_shadow_label() -> dict:
    """The shadow decision of the contact-only policy, recorded with every view."""
    return {
        "scene_decision": {
            "direct_sunlight_condition": None,
        },
        "observations": {
            "receiver_visibility": None,
        },
        "direct_shadow_class": "contact_only_shadow",
        "recommended_shadow": {
            "strength": "contact_only",
            "softness": "soft",
        },
        "confidence": None,
        "evidence": [],
        "reference_camera": None,
        "scene_confidence": None,
        "receiver_confidence": None,
        "scene_evidence": [],
        "receiver_evidence": [],
        "decision_source": CONTACT_ONLY_SHADOW_POLICY_VERSION,
    }


def contact_only_shadow_metadata() -> dict:
    """Describe the command-line policy without pretending it came from labels."""
    return {
        "document_role": "uniform_contact_only_shadow_policy",
        "decision_policy": CONTACT_ONLY_SHADOW_POLICY_VERSION,
        "source": "fixed_contact_only_policy",
        "scope": "all_selected_tokens_timesteps_and_cameras",
        "sha256": None,
    }


def preview_placement_mismatch(
    preview_value: Any,
    current_value: Any,
    path: str = "placement",
) -> Optional[str]:
    """Return the first material preview/final placement difference.

    Preview and final jobs may execute on different compute nodes. Their
    otherwise identical matrix operations can differ in the last few binary
    digits, so derived floats must not be compared through an exact JSON hash.
    Schema, discrete values, and container structure remain exact; only finite
    floating-point values receive a physically negligible absolute tolerance.
    """
    if isinstance(preview_value, dict) or isinstance(current_value, dict):
        if not isinstance(preview_value, dict) or not isinstance(
            current_value, dict
        ):
            return (
                f"{path}: container type differs "
                f"({type(preview_value).__name__} != "
                f"{type(current_value).__name__})"
            )
        preview_keys = set(preview_value)
        current_keys = set(current_value)
        if preview_keys != current_keys:
            missing = sorted(preview_keys - current_keys)
            extra = sorted(current_keys - preview_keys)
            return f"{path}: keys differ (missing={missing}, extra={extra})"
        for key in sorted(preview_keys):
            mismatch = preview_placement_mismatch(
                preview_value[key],
                current_value[key],
                f"{path}.{key}",
            )
            if mismatch is not None:
                return mismatch
        return None

    sequence_types = (list, tuple)
    if isinstance(preview_value, sequence_types) or isinstance(
        current_value, sequence_types
    ):
        if not isinstance(preview_value, sequence_types) or not isinstance(
            current_value, sequence_types
        ):
            return (
                f"{path}: container type differs "
                f"({type(preview_value).__name__} != "
                f"{type(current_value).__name__})"
            )
        if len(preview_value) != len(current_value):
            return (
                f"{path}: sequence length differs "
                f"({len(preview_value)} != {len(current_value)})"
            )
        for index, (preview_item, current_item) in enumerate(
            zip(preview_value, current_value)
        ):
            mismatch = preview_placement_mismatch(
                preview_item,
                current_item,
                f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None

    float_types = (float, np.floating)
    if isinstance(preview_value, float_types) or isinstance(
        current_value, float_types
    ):
        if not isinstance(preview_value, float_types) or not isinstance(
            current_value, float_types
        ):
            return (
                f"{path}: numeric type differs "
                f"({type(preview_value).__name__} != "
                f"{type(current_value).__name__})"
            )
        preview_float = float(preview_value)
        current_float = float(current_value)
        if not math.isfinite(preview_float) or not math.isfinite(current_float):
            return (
                f"{path}: non-finite float "
                f"({preview_value!r} != {current_value!r})"
            )
        if not math.isclose(
            preview_float,
            current_float,
            rel_tol=0.0,
            abs_tol=PREVIEW_PLACEMENT_FLOAT_ATOL,
        ):
            return (
                f"{path}: float differs "
                f"({preview_value!r} != {current_value!r}, "
                f"atol={PREVIEW_PLACEMENT_FLOAT_ATOL})"
            )
        return None

    if type(preview_value) is not type(current_value):
        return (
            f"{path}: value type differs "
            f"({type(preview_value).__name__} != "
            f"{type(current_value).__name__})"
        )
    if preview_value != current_value:
        return f"{path}: value differs ({preview_value!r} != {current_value!r})"
    return None


def shadow_label_render_policy(label: dict) -> dict:
    recommendation = label["recommended_shadow"]
    strength = recommendation["strength"]
    softness = recommendation["softness"]
    return {
        "strength": strength,
        "softness": softness,
        "directional_gain": SHADOW_LABEL_DIRECTIONAL_GAINS[strength],
        "ambient_occlusion_gain": SHADOW_LABEL_AMBIENT_OCCLUSION_GAINS[
            strength
        ],
        "actor_sun_gain": SHADOW_LABEL_ACTOR_SUN_GAINS[strength],
        "sun_angle_degrees": SHADOW_LABEL_SUN_ANGLES_DEGREES[softness],
        "requires_directional_render": strength != "contact_only",
    }


def depth_prediction_cache_root(args: argparse.Namespace) -> Path:
    if args.depth_prediction_cache is not None:
        return args.depth_prediction_cache.resolve()
    return (
        args.data_root.resolve().parent
        / "cache"
        / "depth_anything_v2_small_hf"
        / DEPTH_PREDICTION_CACHE_VERSION
    )


def depth_prediction_cache_path(
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
) -> Path:
    """Return a stable cache path for one canonical NAVSIM source image."""
    source_path = source_path.resolve()
    sensor_root = (args.data_root / "sensor_blobs").resolve()
    try:
        relative = source_path.relative_to(sensor_root)
        return depth_prediction_cache_root(args) / relative.with_suffix(".npz")
    except ValueError:
        # The normal dataset path is human-readable.  Retain safe content-based
        # behavior if a caller supplies an image outside that hierarchy.
        return (
            depth_prediction_cache_root(args)
            / "external"
            / source_sha256[:2]
            / f"{source_sha256}.npz"
        )


def _npz_scalar(data: Any, key: str) -> Optional[str]:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        return None
    return str(value.reshape(()).item())


def depth_prediction_matches(
    path: Path,
    source_sha256: str,
) -> bool:
    """Validate a cached prediction without decompressing its dense array."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if "relative_depth" not in data.files:
                return False
            if _npz_scalar(data, "model_id") != DEPTH_MODEL_ID:
                return False
            return (
                _npz_scalar(data, "source_sha256") == source_sha256
                and _npz_scalar(data, "cache_version")
                == DEPTH_PREDICTION_CACHE_VERSION
            )
    except (OSError, ValueError, TypeError):
        return False


def materialize_depth_prediction(
    cache_path: Path,
    output_path: Path,
    source_sha256: str,
) -> None:
    """Expose a persistent cache entry at the unchanged per-run path."""
    if depth_prediction_matches(
        output_path,
        source_sha256,
    ):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(cache_path.resolve())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def token_uniform(token: str, seed: int, namespace: str) -> float:
    payload = f"{namespace}:{seed}:{token}".encode()
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return integer / float(2**64)


def seeded_random_subset(
    tokens: Sequence[str],
    maximum: Optional[int],
    seed: int,
) -> List[str]:
    """Select a reproducible random subset while retaining source-list order."""
    tokens = list(tokens)
    if maximum is None or maximum >= len(tokens):
        return tokens
    if maximum <= 0:
        return []
    ranked_indices = sorted(
        range(len(tokens)),
        key=lambda index: (
            token_uniform(tokens[index], seed, "max_samples"),
            tokens[index],
            index,
        ),
    )
    selected_indices = set(ranked_indices[:maximum])
    return [token for index, token in enumerate(tokens) if index in selected_indices]


def _finite_metadata_interval(
    value: Any,
    label: str,
    metadata_path: Path,
) -> Tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} in {metadata_path} must be a two-value list")
    lower, upper = (float(value[0]), float(value[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"Invalid {label} interval in {metadata_path}: {value!r}")
    return lower, upper


def _metadata_string_list(
    value: Any,
    label: str,
    metadata_path: Path,
) -> List[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{label} in {metadata_path} must be a string list")
    return [item.strip() for item in value]


def local_bbox_corners(local_bounds: dict) -> np.ndarray:
    """Return the eight corners of normalized actor-local metric bounds."""
    return np.asarray(
        [
            [x, y, z]
            for x in local_bounds["x"]
            for y in local_bounds["y"]
            for z in local_bounds["z"]
        ],
        dtype=np.float64,
    )


def _vehicle_contact_patches(
    vehicle_data: dict,
    local_bounds: dict,
    metadata_path: Path,
) -> List[dict]:
    axles = vehicle_data.get("axles")
    if not isinstance(axles, dict):
        raise ValueError(f"Vehicle metadata {metadata_path} has no axles object")
    patches = []
    for axle_name in ("frontAxle", "rearAxle"):
        axle = axles.get(axle_name)
        if not isinstance(axle, dict):
            raise ValueError(
                f"Vehicle metadata {metadata_path} has no {axle_name} object"
            )
        position_x = float(axle.get("positionX", math.nan))
        track_width = float(axle.get("trackWidth", math.nan))
        wheel_diameter = float(axle.get("wheelDiameter", math.nan))
        values = (position_x, track_width, wheel_diameter)
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"Invalid {axle_name} dimensions in {metadata_path}")
        if track_width <= 0.0 or wheel_diameter <= 0.0:
            raise ValueError(f"Non-positive {axle_name} dimensions in {metadata_path}")
        if not local_bounds["x"][0] - 0.5 <= position_x <= local_bounds["x"][1] + 0.5:
            raise ValueError(f"{axle_name} lies outside the vehicle bounds in {metadata_path}")
        half_length = float(np.clip(0.12 * wheel_diameter, 0.06, 0.16))
        half_width = float(np.clip(0.08 * wheel_diameter, 0.04, 0.11))
        for side_name, side_sign in (("left", 1.0), ("right", -1.0)):
            center_y = side_sign * 0.5 * track_width
            points = [
                [position_x + dx, center_y + dy, local_bounds["z"][0]]
                for dx in (-half_length, half_length)
                for dy in (-half_width, half_width)
            ]
            patches.append(
                {
                    "name": f"{axle_name}_{side_name}",
                    "points_actor_m": points,
                    "wheel_diameter_m": wheel_diameter,
                }
            )
    return patches


def load_car_asset(path: Path) -> dict:
    """Load one normalized OpenX vehicle from a packed blend/XOMA pair."""
    blend_file = path.resolve()
    if not blend_file.is_file():
        raise FileNotFoundError(f"Car asset not found: {blend_file}")
    metadata_path = blend_file.with_suffix(".xoma")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"OpenX vehicle metadata sidecar not found: {metadata_path}"
        )
    try:
        document = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read XOMA metadata {metadata_path}: {exc}") from exc
    metadata = document.get("metadata") if isinstance(document, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError(f"XOMA metadata {metadata_path} has no metadata object")
    if metadata.get("assetType") != "object" or metadata.get("objectClass") != "vehicle":
        raise ValueError(f"XOMA asset is not a vehicle object: {metadata_path}")

    bounding_box = metadata.get("boundingBox")
    if not isinstance(bounding_box, dict):
        raise ValueError(f"XOMA vehicle has no boundingBox: {metadata_path}")
    local_bounds = {
        axis: list(_finite_metadata_interval(bounding_box.get(axis), f"boundingBox.{axis}", metadata_path))
        for axis in ("x", "y", "z")
    }
    length_m = local_bounds["x"][1] - local_bounds["x"][0]
    width_m = local_bounds["y"][1] - local_bounds["y"][0]
    height_m = local_bounds["z"][1] - local_bounds["z"][0]
    dimensions = (length_m, width_m, height_m)
    if not (
        1.5 <= length_m <= 15.0
        and 0.8 <= width_m <= 4.0
        and 0.5 <= height_m <= 5.0
    ):
        raise ValueError(
            f"Unreasonable vehicle dimensions {dimensions!r} in {metadata_path}"
        )
    if not local_bounds["x"][0] < 0.0 < local_bounds["x"][1]:
        raise ValueError(
            f"Vehicle root must lie between rear/front bumpers in {metadata_path}"
        )
    if not local_bounds["y"][0] < 0.0 < local_bounds["y"][1]:
        raise ValueError(
            f"Vehicle root must lie between lateral bounds in {metadata_path}"
        )
    if abs(float(local_bounds["z"][0])) > 0.10:
        raise ValueError(
            f"Vehicle ground plane must be near local z=0 in {metadata_path}"
        )

    vehicle_data = metadata.get("vehicleClassData")
    if not isinstance(vehicle_data, dict):
        raise ValueError(f"XOMA vehicle has no vehicleClassData: {metadata_path}")
    contact_patches = _vehicle_contact_patches(
        vehicle_data,
        local_bounds,
        metadata_path,
    )
    asset_id = str(metadata.get("uuid") or blend_file.stem).strip()
    if not asset_id:
        raise ValueError(f"XOMA vehicle has no usable identity: {metadata_path}")
    license_identifier = str(metadata.get("license") or "").strip()
    if not license_identifier:
        raise ValueError(f"XOMA vehicle has no license identifier: {metadata_path}")
    authors = _metadata_string_list(
        metadata.get("authors"), "authors", metadata_path
    )
    copyrights = _metadata_string_list(
        metadata.get("copyrights"), "copyrights", metadata_path
    )
    geometry_profile = {
        "profile_version": CAR_GEOMETRY_VERSION,
        "local_bounds_m": local_bounds,
        "length_m": length_m,
        "width_m": width_m,
        "height_m": height_m,
        "rear_bumper_x_m": local_bounds["x"][0],
        "front_bumper_x_m": local_bounds["x"][1],
        "contact_profile": CAR_CONTACT_PROFILE,
        "wheel_contact_patches": contact_patches,
    }
    attribution = {
        "name": metadata.get("name"),
        "authors": authors,
        "license": license_identifier,
        "copyrights": copyrights,
        "source_metadata_format": "OpenMaterial3D/XOMA",
        "open_material_3d_version": metadata.get("openMaterial3dVersion"),
        "asset_version": metadata.get("assetVersion"),
    }
    return {
        "path": blend_file,
        "sha256": file_sha256(blend_file),
        "metadata_path": metadata_path.resolve(),
        "metadata_sha256": file_sha256(metadata_path),
        "metadata": metadata,
        "id": asset_id,
        "actor_type": "car",
        "actor_root_name": CAR_ROOT_OBJECT_NAME,
        "actor_height_m": height_m,
        "actor_geometry": geometry_profile,
        "attribution": attribution,
    }


def load_asset(path: Path, args: argparse.Namespace) -> dict:
    if getattr(args, "car", False):
        return load_car_asset(path)
    blend_file = path.resolve()
    if not blend_file.is_file():
        raise FileNotFoundError(f"Pedestrian asset not found: {blend_file}")
    sidecar_path = blend_file.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Generated pedestrian metadata sidecar not found: {sidecar_path}"
        )
    metadata = json.loads(sidecar_path.read_text())
    required_metadata = {
        "id",
        "character_height_m",
        "animation_version",
        "animation_frame_start",
        "walk_cycle_frames",
    }
    missing_metadata = sorted(required_metadata - metadata.keys())
    if missing_metadata:
        raise ValueError(
            f"Asset metadata {sidecar_path} is missing {missing_metadata}"
        )
    actor_height_m = float(metadata["character_height_m"])
    if not 0.5 <= actor_height_m <= 2.3:
        raise ValueError(f"Unreasonable asset height {actor_height_m:.3f} m in {sidecar_path}")
    walk_cycle_frames = int(metadata["walk_cycle_frames"])
    if walk_cycle_frames <= 0:
        raise ValueError(f"Invalid walk-cycle length {walk_cycle_frames} in {sidecar_path}")
    action_start = int(metadata["animation_frame_start"])
    action_range = (action_start, action_start + walk_cycle_frames)
    unavailable_pose_frames = [
        frame
        for frame in NAVSIM_POSE_SEQUENCE
        if frame < action_range[0] or frame >= action_range[1]
    ]
    if unavailable_pose_frames:
        raise ValueError(
            f"Requested pose frames {unavailable_pose_frames} are outside asset action range "
            f"{action_range[0]}..{action_range[1] - 1} in {blend_file}"
        )
    if args.stride_length_m is None:
        stride_length_m = (
            walk_cycle_frames
            * args.speed_mps
            * args.gait_timestep_s
            / args.gait_frame_advance
        )
        stride_length_source = "derived_from_frame_advance"
    else:
        stride_length_m = args.stride_length_m
        stride_length_source = "explicit"
    if stride_length_m <= 0:
        raise ValueError("--stride-length-m must be positive")
    declared_actor_type = str(metadata.get("actor_type", "pedestrian")).lower()
    is_pedestrian = declared_actor_type in {"", "human", "pedestrian"}
    actor_geometry = None
    if is_pedestrian:
        actor_geometry = normalize_pedestrian_geometry(
            metadata.get("actor_geometry"),
            expected_asset_id=str(metadata["id"]),
        )
        if metadata.get("local_forward_axis") != actor_geometry["local_forward_axis"]:
            raise ValueError(
                f"Asset metadata {sidecar_path} has inconsistent local_forward_axis"
            )
    metadata_sha256 = file_sha256(sidecar_path)
    result = {
        "path": blend_file,
        "sha256": file_sha256(blend_file),
        "metadata_path": sidecar_path.resolve(),
        "metadata_sha256": metadata_sha256,
        "metadata": metadata,
        "id": metadata["id"],
        "actor_height_m": actor_height_m,
        "animation_version": metadata["animation_version"],
        "action_range": action_range,
        "walk_cycle_frames": walk_cycle_frames,
        "stride_length_m": stride_length_m,
        "stride_length_source": stride_length_source,
        "actor_type": "pedestrian" if is_pedestrian else declared_actor_type,
        "actor_root_name": str(metadata.get("renderer_root", "PED_RIG")),
    }
    if actor_geometry is not None:
        result["actor_geometry"] = actor_geometry
    return result


def load_asset_pool(args: argparse.Namespace) -> List[dict]:
    asset_dir = args.asset_dir.resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Actor asset directory not found: {asset_dir}")
    discovery = asset_dir.rglob("*.blend") if getattr(args, "car", False) else asset_dir.glob("*.blend")
    paths = sorted(path for path in discovery if path.is_file())
    if not paths:
        scope = "recursively under" if getattr(args, "car", False) else "directly under"
        raise FileNotFoundError(f"No .blend assets found {scope}: {asset_dir}")
    return [load_asset(path, args) for path in paths]


def select_asset(token: str, seed: int, assets: Sequence[dict]) -> dict:
    index = min(int(token_uniform(token, seed, "asset") * len(assets)), len(assets) - 1)
    return assets[index]


def car_asset_config(asset: dict) -> dict:
    return {
        "path": str(asset["path"]),
        "sha256": asset["sha256"],
        "metadata_path": str(asset["metadata_path"]),
        "metadata_sha256": asset["metadata_sha256"],
        "id": asset["id"],
        "actor_type": "car",
        "actor_root_name": asset["actor_root_name"],
        "actor_height_m": asset["actor_height_m"],
        "actor_geometry": asset["actor_geometry"],
        "attribution": asset["attribution"],
    }


def asset_config(asset: dict) -> dict:
    if asset.get("actor_type") == "car":
        return car_asset_config(asset)
    config = {
        "path": str(asset["path"]),
        "sha256": asset["sha256"],
        "metadata_path": str(asset["metadata_path"]),
        "metadata_sha256": asset["metadata_sha256"],
        "id": asset["id"],
        "actor_type": asset.get("actor_type", "pedestrian"),
        "actor_root_name": asset["actor_root_name"],
        "actor_height_m": asset["actor_height_m"],
        "animation_version": asset["animation_version"],
        "action_range": list(asset["action_range"]),
        "walk_cycle_frames": asset["walk_cycle_frames"],
        "stride_length_m": asset["stride_length_m"],
        "stride_length_source": asset["stride_length_source"],
    }
    if "actor_geometry" in asset:
        config["actor_geometry"] = asset["actor_geometry"]
    return config


def crossing_policy(
    token: str,
    classifications: Dict[str, str],
) -> Tuple[str, float, float]:
    classification = classifications.get(token)
    spec = DIRECTION_CLASSIFICATION_SPECS.get(classification)
    if spec is None:
        raise TokenFailure(
            "invalid_direction_classification",
            f"token {token} has non-renderable direction classification "
            f"{classification!r}",
        )
    return (
        str(spec["crossing_name"]),
        float(spec["lateral_sign"]),
        float(spec["actor_heading_rad"]),
    )


def standing_facing(
    args: argparse.Namespace,
    token: str,
) -> Tuple[str, float, float]:
    """Return a standing actor's direction name, lateral sign, and heading.

    The default faces the approaching ego (road-basis -X). Actor adapters may
    replace this hook: a lateral sign of +1/-1 faces road-basis +Y/-Y with the
    heading convention of ``DIRECTION_CLASSIFICATION_SPECS``.
    """
    return "standing_facing_ego", 0.0, STANDING_ACTOR_HEADING_RAD


def configure_navsim(data_root: Path):
    os.environ["OPENSCENE_DATA_ROOT"] = str(data_root)
    os.environ["NUPLAN_MAPS_ROOT"] = str(data_root / "maps")
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import navsim.common.dataclasses as dc

    dc.OPENSCENE_DATA_ROOT = str(data_root)
    dc.NUPLAN_MAPS_ROOT = str(data_root / "maps")


def select_filter_tokens(config: dict, args: argparse.Namespace) -> Optional[List[str]]:
    configured = config.get("tokens")
    if configured is None:
        return None
    return seeded_random_subset(list(configured), args.max_samples, args.seed)


def build_loader(args: argparse.Namespace, filter_config: dict):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    selected_tokens = select_filter_tokens(filter_config, args)
    scene_filter = SceneFilter(
        num_history_frames=int(filter_config["num_history_frames"]),
        num_future_frames=int(filter_config["num_future_frames"]),
        frame_interval=filter_config.get("frame_interval"),
        has_route=bool(filter_config.get("has_route", True)),
        max_scenes=None,
        log_names=filter_config.get("log_names"),
        tokens=selected_tokens,
    )
    if scene_filter.num_history_frames != EXPECTED_HISTORY_FRAMES:
        raise ValueError(
            "This pipeline requires exactly "
            f"{EXPECTED_HISTORY_FRAMES} history frames, got "
            f"{scene_filter.num_history_frames}"
        )
    timestep = getattr(args, "timestep", None)
    if timestep is not None and int(timestep) > scene_filter.num_future_frames:
        raise ValueError(
            f"--timestep {int(timestep)} exceeds the scene filter's "
            f"{scene_filter.num_future_frames} future frames"
        )
    loader = SceneLoader(
        data_path=args.data_root / "navsim_logs" / args.split,
        sensor_blobs_path=args.data_root / "sensor_blobs" / args.split,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
        load_image_path=True,
    )
    if selected_tokens is None:
        tokens = seeded_random_subset(loader.tokens, args.max_samples, args.seed)
    else:
        available = set(loader.tokens)
        tokens = [token for token in selected_tokens if token in available]
    return loader, tokens


def frame_source_path(data_root: Path, split: str, frame: dict, camera: str) -> Path:
    return data_root / "sensor_blobs" / split / frame["cams"][camera]["data_path"]


def planar_yaw(transform: np.ndarray) -> float:
    """Extract the heading of an approximately gravity-aligned SE(3) pose."""
    transform = np.asarray(transform, dtype=np.float64)
    return math.atan2(float(transform[1, 0]), float(transform[0, 0]))


def transform_xy_planar(transform: np.ndarray, xy: Sequence[float]) -> np.ndarray:
    """Apply only the horizontal SE(2) part of a pose to an XY point."""
    yaw = planar_yaw(transform)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return rotation @ np.asarray(xy, dtype=np.float64) + np.asarray(transform[:2, 3], dtype=np.float64)


def inverse_transform_xy_planar(transform: np.ndarray, xy: Sequence[float]) -> np.ndarray:
    """Apply the inverse horizontal SE(2) part of a pose to an XY point."""
    yaw = planar_yaw(transform)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return rotation.T @ (
        np.asarray(xy, dtype=np.float64) - np.asarray(transform[:2, 3], dtype=np.float64)
    )


def ground_point_lidar_for_global_xy(
    lidar_to_global: np.ndarray,
    plane_lidar: Sequence[float],
    global_xy: Sequence[float],
) -> np.ndarray:
    """Intersect a fitted LiDAR plane with an exact requested global XY."""
    transform = np.asarray(lidar_to_global, dtype=np.float64)
    plane = np.asarray(plane_lidar, dtype=np.float64).reshape(-1)
    target = np.asarray(global_xy, dtype=np.float64).reshape(-1)
    if (
        transform.shape != (4, 4)
        or plane.shape != (3,)
        or target.shape != (2,)
        or not np.isfinite(transform).all()
        or not np.isfinite(plane).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("ground/global intersection inputs must be finite")
    slope_x, slope_y, intercept = [float(value) for value in plane]
    rotation = transform[:3, :3]
    system = np.asarray(
        [
            [
                rotation[0, 0] + rotation[0, 2] * slope_x,
                rotation[0, 1] + rotation[0, 2] * slope_y,
            ],
            [
                rotation[1, 0] + rotation[1, 2] * slope_x,
                rotation[1, 1] + rotation[1, 2] * slope_y,
            ],
        ],
        dtype=np.float64,
    )
    right_hand_side = target - transform[:2, 3] - np.asarray(
        [rotation[0, 2] * intercept, rotation[1, 2] * intercept],
        dtype=np.float64,
    )
    try:
        lidar_xy = np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError as exc:
        raise TokenFailure(
            "invalid_lidar_pose",
            "cannot intersect the fitted road plane with the requested global XY",
        ) from exc
    lidar_point = np.asarray(
        [
            lidar_xy[0],
            lidar_xy[1],
            geometry.ground_z(plane, lidar_xy[0], lidar_xy[1]),
        ],
        dtype=np.float64,
    )
    realized_global = geometry.transform_points(
        transform, lidar_point[None]
    )[0]
    if not np.allclose(realized_global[:2], target, rtol=0.0, atol=1e-8):
        raise TokenFailure(
            "invalid_lidar_pose",
            "road-plane intersection did not preserve the requested global XY",
        )
    return lidar_point


def resolve_crossing_trajectory(
    scene_frames: Sequence[dict],
    timestamps_s: Sequence[float],
    *,
    timestep: int,
    front_bumper_offset_m: float,
    lateral_sign: float,
    speed_mps: float,
) -> dict:
    """Resolve a pedestrian trajectory timed to a future bumper interception.

    The future ego front-bumper point is the interception point, not a rendered
    history-frame position.  At every rendered timestamp ``t`` the pedestrian
    is placed at ``interception + velocity * (t - interception_time)``.
    """
    if len(scene_frames) < EXPECTED_HISTORY_FRAMES:
        raise TokenFailure(
            "missing_sensor",
            f"expected at least {EXPECTED_HISTORY_FRAMES} scene frames, got "
            f"{len(scene_frames)}",
        )
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if (
        timestamps.size != EXPECTED_HISTORY_FRAMES
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise TokenFailure(
            "invalid_timestamps",
            "crossing trajectory requires finite increasing timestamps for "
            f"exactly {EXPECTED_HISTORY_FRAMES} history frames",
        )
    if not math.isfinite(front_bumper_offset_m):
        raise ValueError("front-bumper offset must be finite")
    if not math.isfinite(lateral_sign) or abs(float(lateral_sign)) != 1.0:
        raise ValueError("lateral crossing sign must be either -1 or +1")
    if not math.isfinite(speed_mps) or speed_mps <= 0:
        raise ValueError("pedestrian speed must be finite and positive")

    current_frame_index = EXPECTED_HISTORY_FRAMES - 1
    current_frame = scene_frames[current_frame_index]
    placement = resolve_placement(
        scene_frames,
        timestep,
        front_bumper_offset_m,
        current_frame_index=current_frame_index,
    )
    reference_frame_index = int(placement["future_frame_index"])
    anchor_reference_xy = np.array(
        [float(front_bumper_offset_m), 0.0], dtype=np.float64
    )

    current_pose = np.asarray(
        current_frame.get("ego2global", []), dtype=np.float64
    )
    reference_frame = scene_frames[reference_frame_index]
    reference_pose = np.asarray(
        reference_frame.get("ego2global", []), dtype=np.float64
    )
    for label, pose in (
        ("final history", current_pose),
        ("placement reference", reference_pose),
    ):
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise TokenFailure(
                "invalid_ego_pose",
                f"{label} frame must contain a finite 4x4 ego2global transform",
            )
    current_map = str(current_frame.get("map_location", ""))
    reference_map = str(reference_frame.get("map_location", ""))
    if current_map and reference_map and current_map != reference_map:
        raise TokenFailure(
            "inconsistent_map_location",
            "placement reference and final history frames use different maps: "
            f"{current_map!r} versus {reference_map!r}",
        )

    reference_yaw = planar_yaw(reference_pose)
    reference_forward_global_xy = np.array(
        [math.cos(reference_yaw), math.sin(reference_yaw)],
        dtype=np.float64,
    )
    reference_left_global_xy = np.array(
        [-math.sin(reference_yaw), math.cos(reference_yaw)],
        dtype=np.float64,
    )
    anchor_global_xy = transform_xy_planar(
        reference_pose, anchor_reference_xy
    )
    anchor_current_ego_xy = inverse_transform_xy_planar(
        current_pose, anchor_global_xy
    )
    walking_direction_global_xy = (
        float(lateral_sign) * reference_left_global_xy
    )
    interception_timestamp_s = float(placement["future_timestamp_us"]) / 1e6
    crossing_offsets_m = (
        float(lateral_sign)
        * float(speed_mps)
        * (timestamps - interception_timestamp_s)
    )
    actor_global_xys = np.asarray(
        [
            anchor_global_xy + offset * reference_left_global_xy
            for offset in crossing_offsets_m
        ],
        dtype=np.float64,
    )
    actor_current_xys = np.asarray(
        [
            inverse_transform_xy_planar(current_pose, actor_global_xy)
            for actor_global_xy in actor_global_xys
        ],
        dtype=np.float64,
    )
    time_to_interception_s = interception_timestamp_s - float(timestamps[-1])
    distance_to_interception_m = float(speed_mps) * time_to_interception_s

    placement = {
        **placement,
        "reference_frame_index": int(reference_frame_index),
        "reference_timestamp_us": int(reference_frame["timestamp"]),
        "reference_vehicle_heading_global_rad": float(reference_yaw),
        "reference_vehicle_forward_global_xy": (
            reference_forward_global_xy.tolist()
        ),
        "reference_vehicle_left_global_xy": (
            reference_left_global_xy.tolist()
        ),
        "interception_timestamp_s": interception_timestamp_s,
        "interception_global_xy_m": anchor_global_xy.tolist(),
        "pedestrian_time_to_interception_s": time_to_interception_s,
        "pedestrian_distance_to_interception_m": distance_to_interception_m,
        "final_history_actor_global_xy_m": actor_global_xys[-1].tolist(),
        "walking_direction_global_xy": walking_direction_global_xy.tolist(),
        # Retain the anchor aliases so existing rendering/preview plumbing can
        # consume v2 metadata without duplicating coordinate transforms.
        "anchor_global_xy_m": anchor_global_xy.tolist(),
        "anchor_current_ego_xy_m": anchor_current_ego_xy.tolist(),
        "crossing_axis_global_xy": reference_left_global_xy.tolist(),
    }
    return {
        "placement": placement,
        "reference_frame_index": int(reference_frame_index),
        "reference_vehicle_forward_global_xy": (
            reference_forward_global_xy
        ),
        "reference_vehicle_left_global_xy": reference_left_global_xy,
        "walking_direction_global_xy": walking_direction_global_xy,
        "walking_velocity_global_xy_mps": (
            float(speed_mps) * walking_direction_global_xy
        ),
        "anchor_global_xy": anchor_global_xy,
        "anchor_current_ego_xy": anchor_current_ego_xy,
        "crossing_offsets_m": crossing_offsets_m,
        "actor_global_xys": actor_global_xys,
        "actor_current_xys": actor_current_xys,
    }


def resolve_standing_trajectory(
    scene_frames: Sequence[dict],
    timestamps_s: Sequence[float],
    *,
    timestep: int,
    front_bumper_offset_m: float,
    lateral_sign: float = 0.0,
) -> dict:
    """Resolve a stationary pedestrian at the future bumper interception.

    The position is fixed at the future ego front-bumper point for every
    rendered history frame. By default the pedestrian's forward direction is
    the negative of the ego's forward direction at that future frame, so the
    pedestrian looks toward the approaching vehicle. A lateral sign of +1/-1
    instead faces the ego's left/right at that frame (side-on to the ego).
    """
    if float(lateral_sign) not in (0.0, 1.0, -1.0):
        raise ValueError(f"standing lateral sign must be 0, 1, or -1, got {lateral_sign}")
    if len(scene_frames) < EXPECTED_HISTORY_FRAMES:
        raise TokenFailure(
            "missing_sensor",
            f"expected at least {EXPECTED_HISTORY_FRAMES} scene frames, got "
            f"{len(scene_frames)}",
        )
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if (
        timestamps.size != EXPECTED_HISTORY_FRAMES
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise TokenFailure(
            "invalid_timestamps",
            "standing trajectory requires finite increasing timestamps for "
            f"exactly {EXPECTED_HISTORY_FRAMES} history frames",
        )
    if not math.isfinite(front_bumper_offset_m):
        raise ValueError("front-bumper offset must be finite")

    current_frame_index = EXPECTED_HISTORY_FRAMES - 1
    current_frame = scene_frames[current_frame_index]
    placement = resolve_placement(
        scene_frames,
        timestep,
        front_bumper_offset_m,
        current_frame_index=current_frame_index,
    )
    reference_frame_index = int(placement["future_frame_index"])
    current_pose = np.asarray(
        current_frame.get("ego2global", []), dtype=np.float64
    )
    reference_frame = scene_frames[reference_frame_index]
    reference_pose = np.asarray(
        reference_frame.get("ego2global", []), dtype=np.float64
    )
    for label, pose in (
        ("final history", current_pose),
        ("placement reference", reference_pose),
    ):
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise TokenFailure(
                "invalid_ego_pose",
                f"{label} frame must contain a finite 4x4 ego2global transform",
            )
    current_map = str(current_frame.get("map_location", ""))
    reference_map = str(reference_frame.get("map_location", ""))
    if current_map and reference_map and current_map != reference_map:
        raise TokenFailure(
            "inconsistent_map_location",
            "placement reference and final history frames use different maps: "
            f"{current_map!r} versus {reference_map!r}",
        )

    reference_yaw = planar_yaw(reference_pose)
    reference_forward_global_xy = np.array(
        [math.cos(reference_yaw), math.sin(reference_yaw)],
        dtype=np.float64,
    )
    reference_left_global_xy = np.array(
        [-math.sin(reference_yaw), math.cos(reference_yaw)],
        dtype=np.float64,
    )
    anchor_global_xy = transform_xy_planar(
        reference_pose,
        [float(front_bumper_offset_m), 0.0],
    )
    anchor_current_ego_xy = inverse_transform_xy_planar(
        current_pose, anchor_global_xy
    )
    facing_direction_global_xy = (
        -reference_forward_global_xy
        if float(lateral_sign) == 0.0
        else float(lateral_sign) * reference_left_global_xy
    )
    actor_global_xys = np.repeat(
        anchor_global_xy[None, :], EXPECTED_HISTORY_FRAMES, axis=0
    )
    actor_current_xys = np.repeat(
        anchor_current_ego_xy[None, :], EXPECTED_HISTORY_FRAMES, axis=0
    )
    crossing_offsets_m = np.zeros(EXPECTED_HISTORY_FRAMES, dtype=np.float64)
    zero_velocity = np.zeros(2, dtype=np.float64)
    interception_timestamp_s = float(placement["future_timestamp_us"]) / 1e6
    time_to_interception_s = interception_timestamp_s - float(timestamps[-1])

    placement = {
        **placement,
        "placement_version": STANDING_PLACEMENT_VERSION,
        "pedestrian_behavior": "standing",
        "standing_pose_version": STANDING_POSE_VERSION,
        "standing_position_policy": "stationary_at_future_ego_front_bumper",
        "standing_facing_policy": STANDING_FACING_POLICY,
        **(
            {"standing_facing_lateral_sign": float(lateral_sign)}
            if float(lateral_sign) != 0.0
            else {}
        ),
        "reference_frame_index": int(reference_frame_index),
        "reference_timestamp_us": int(reference_frame["timestamp"]),
        "reference_vehicle_heading_global_rad": float(reference_yaw),
        "reference_vehicle_forward_global_xy": (
            reference_forward_global_xy.tolist()
        ),
        "reference_vehicle_left_global_xy": reference_left_global_xy.tolist(),
        "interception_timestamp_s": interception_timestamp_s,
        "interception_global_xy_m": anchor_global_xy.tolist(),
        "pedestrian_time_to_interception_s": time_to_interception_s,
        "pedestrian_distance_to_interception_m": 0.0,
        "final_history_actor_global_xy_m": anchor_global_xy.tolist(),
        "standing_facing_direction_global_xy": (
            facing_direction_global_xy.tolist()
        ),
        "anchor_global_xy_m": anchor_global_xy.tolist(),
        "anchor_current_ego_xy_m": anchor_current_ego_xy.tolist(),
    }
    return {
        "placement": placement,
        "reference_frame_index": int(reference_frame_index),
        "reference_vehicle_forward_global_xy": (
            reference_forward_global_xy
        ),
        "reference_vehicle_left_global_xy": reference_left_global_xy,
        "standing_facing_direction_global_xy": facing_direction_global_xy,
        # Retain the existing velocity key for downstream schema compatibility;
        # it is exactly zero for a stationary actor.
        "walking_velocity_global_xy_mps": zero_velocity,
        "actor_velocity_global_xy_mps": zero_velocity,
        "anchor_global_xy": anchor_global_xy,
        "anchor_current_ego_xy": anchor_current_ego_xy,
        "crossing_offsets_m": crossing_offsets_m,
        "actor_global_xys": actor_global_xys,
        "actor_current_xys": actor_current_xys,
    }


def oriented_box_corners_xy(
    origin_xy: Sequence[float],
    yaw_rad: float,
    x_bounds: Sequence[float],
    y_bounds: Sequence[float],
) -> np.ndarray:
    """Return counter-clockwise metric corners for an oriented local box."""
    local = np.asarray(
        [
            [float(x_bounds[0]), float(y_bounds[0])],
            [float(x_bounds[1]), float(y_bounds[0])],
            [float(x_bounds[1]), float(y_bounds[1])],
            [float(x_bounds[0]), float(y_bounds[1])],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.asarray(origin_xy, dtype=np.float64)


def oriented_box_axis_overlap_m(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the minimum SAT interval overlap (negative means separation)."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4, 2) or second.shape != (4, 2):
        raise ValueError("oriented footprints must each contain four XY corners")
    overlaps = []
    for polygon in (first, second):
        for index in range(4):
            edge = polygon[(index + 1) % 4] - polygon[index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-9:
                raise ValueError("oriented footprint has a degenerate edge")
            axis /= norm
            first_projection = first @ axis
            second_projection = second @ axis
            overlaps.append(
                min(float(first_projection.max()), float(second_projection.max()))
                - max(float(first_projection.min()), float(second_projection.min()))
            )
    return float(min(overlaps))


def stationary_vehicle_first_contact_diagnostics(
    scene_frames: Sequence[dict],
    *,
    current_frame_index: int,
    timestep: int,
    car_origin_global_xy: Sequence[float],
    car_heading_global_rad: float,
    car_local_bounds: dict,
    ego_front_length_m: float,
    ego_rear_length_m: float,
    ego_width_m: float,
) -> dict:
    """Validate that discrete ego/car footprint contact first occurs at N."""
    car_corners = oriented_box_corners_xy(
        car_origin_global_xy,
        car_heading_global_rad,
        car_local_bounds["x"],
        car_local_bounds["y"],
    )
    frame_checks = []
    for relative_timestep in range(
        -int(current_frame_index), int(timestep) + 1
    ):
        frame_index = int(current_frame_index) + relative_timestep
        pose = np.asarray(
            scene_frames[frame_index].get("ego2global", []),
            dtype=np.float64,
        )
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise TokenFailure(
                "invalid_future_ego_pose",
                f"future frame index {frame_index} must contain a finite 4x4 "
                "ego2global transform",
            )
        ego_corners = oriented_box_corners_xy(
            pose[:2, 3],
            planar_yaw(pose),
            (-float(ego_rear_length_m), float(ego_front_length_m)),
            (-0.5 * float(ego_width_m), 0.5 * float(ego_width_m)),
        )
        overlap = oriented_box_axis_overlap_m(ego_corners, car_corners)
        record = {
            "relative_timestep": relative_timestep,
            "frame_index": frame_index,
            "minimum_axis_overlap_m": overlap,
            "contact_within_tolerance": abs(overlap) <= CAR_CONTACT_TOLERANCE_M,
            "interior_overlap": overlap > CAR_CONTACT_TOLERANCE_M,
        }
        if relative_timestep >= 0:
            record.update(
                {
                    "future_timestep": relative_timestep,
                    "future_frame_index": frame_index,
                }
            )
        frame_checks.append(record)

    earlier_contact = [
        item
        for item in frame_checks[:-1]
        if item["minimum_axis_overlap_m"] >= -CAR_CONTACT_TOLERANCE_M
    ]
    if earlier_contact:
        first = earlier_contact[0]
        raise TokenFailure(
            "vehicle_contact_before_target_timestep",
            "stationary car touches or overlaps the ego footprint before the "
            f"requested timestep: relative_timestep={first['relative_timestep']}, "
            f"axis_overlap={first['minimum_axis_overlap_m']:.6f} m",
        )
    target = frame_checks[-1]
    if abs(float(target["minimum_axis_overlap_m"])) > CAR_CONTACT_TOLERANCE_M:
        raise TokenFailure(
            "vehicle_target_contact_failed",
            "ego/car footprints do not touch at the requested timestep: "
            f"axis_overlap={target['minimum_axis_overlap_m']:.6f} m",
        )
    return {
        "method": "oriented_box_discrete_first_contact_v1",
        "tolerance_m": CAR_CONTACT_TOLERANCE_M,
        "first_contact_timestep": int(timestep),
        "no_contact_before_target": True,
        "target_contact": True,
        "history_frame_checks": [
            item for item in frame_checks if item["relative_timestep"] < 0
        ],
        "per_timestep": [
            item for item in frame_checks if item["relative_timestep"] >= 0
        ],
    }


def resolve_stationary_vehicle_trajectory(
    scene_frames: Sequence[dict],
    timestamps_s: Sequence[float],
    *,
    timestep: int,
    front_bumper_offset_m: float,
    car_local_bounds: dict,
    ego_rear_length_m: float,
    ego_width_m: float,
) -> dict:
    """Place a fixed car so ego-front/rear-car contact first occurs at N."""
    if len(scene_frames) < EXPECTED_HISTORY_FRAMES:
        raise TokenFailure(
            "missing_sensor",
            f"expected at least {EXPECTED_HISTORY_FRAMES} scene frames, got "
            f"{len(scene_frames)}",
        )
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if (
        timestamps.size != EXPECTED_HISTORY_FRAMES
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise TokenFailure(
            "invalid_timestamps",
            "stationary car placement requires finite increasing timestamps "
            f"for exactly {EXPECTED_HISTORY_FRAMES} history frames",
        )
    for name, value in (
        ("ego front length", front_bumper_offset_m),
        ("ego rear length", ego_rear_length_m),
        ("ego width", ego_width_m),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    bbox = local_bbox_corners(car_local_bounds)
    if bbox.shape != (8, 3) or not np.isfinite(bbox).all():
        raise ValueError("car local bounds must define a finite 3-D box")

    current_frame_index = EXPECTED_HISTORY_FRAMES - 1
    current_frame = scene_frames[current_frame_index]
    placement = resolve_placement(
        scene_frames,
        timestep,
        front_bumper_offset_m,
        current_frame_index=current_frame_index,
    )
    reference_frame_index = int(placement["future_frame_index"])
    reference_frame = scene_frames[reference_frame_index]
    current_pose = np.asarray(current_frame.get("ego2global", []), dtype=np.float64)
    reference_pose = np.asarray(reference_frame.get("ego2global", []), dtype=np.float64)
    for label, pose in (("final history", current_pose), ("contact", reference_pose)):
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise TokenFailure(
                "invalid_ego_pose",
                f"{label} frame must contain a finite 4x4 ego2global transform",
            )
    current_map = str(current_frame.get("map_location", ""))
    reference_map = str(reference_frame.get("map_location", ""))
    if current_map and reference_map and current_map != reference_map:
        raise TokenFailure(
            "inconsistent_map_location",
            "contact and final history frames use different maps: "
            f"{current_map!r} versus {reference_map!r}",
        )

    reference_yaw = planar_yaw(reference_pose)
    forward = np.asarray(
        [math.cos(reference_yaw), math.sin(reference_yaw)],
        dtype=np.float64,
    )
    left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    ego_front_global_xy = np.asarray(
        placement["future_front_bumper_global_xy_m"], dtype=np.float64
    )
    rear_bumper_x_m = float(car_local_bounds["x"][0])
    car_origin_global_xy = ego_front_global_xy - rear_bumper_x_m * forward
    rear_bumper_global_xy = car_origin_global_xy + rear_bumper_x_m * forward
    bumper_contact_error_m = float(
        np.linalg.norm(rear_bumper_global_xy - ego_front_global_xy)
    )
    if bumper_contact_error_m > CAR_CONTACT_TOLERANCE_M:
        raise TokenFailure(
            "vehicle_target_contact_failed",
            f"constructed bumper contact error is {bumper_contact_error_m:.6f} m",
        )
    lateral_overlap_m = (
        min(0.5 * float(ego_width_m), float(car_local_bounds["y"][1]))
        - max(-0.5 * float(ego_width_m), float(car_local_bounds["y"][0]))
    )
    if lateral_overlap_m <= CAR_CONTACT_TOLERANCE_M:
        raise TokenFailure(
            "vehicle_target_contact_failed",
            "ego front and inserted-car rear bumpers have no lateral overlap",
        )
    first_contact = stationary_vehicle_first_contact_diagnostics(
        scene_frames,
        current_frame_index=current_frame_index,
        timestep=timestep,
        car_origin_global_xy=car_origin_global_xy,
        car_heading_global_rad=reference_yaw,
        car_local_bounds=car_local_bounds,
        ego_front_length_m=front_bumper_offset_m,
        ego_rear_length_m=ego_rear_length_m,
        ego_width_m=ego_width_m,
    )

    actor_global_xys = np.repeat(
        car_origin_global_xy[None, :], EXPECTED_HISTORY_FRAMES, axis=0
    )
    car_origin_current_xy = inverse_transform_xy_planar(
        current_pose, car_origin_global_xy
    )
    actor_current_xys = np.repeat(
        car_origin_current_xy[None, :], EXPECTED_HISTORY_FRAMES, axis=0
    )
    zero_offsets = np.zeros(EXPECTED_HISTORY_FRAMES, dtype=np.float64)
    zero_velocity = np.zeros(2, dtype=np.float64)
    contact_timestamp_s = float(placement["future_timestamp_us"]) / 1e6
    time_to_contact_s = contact_timestamp_s - float(timestamps[-1])
    placement = {
        **placement,
        "placement_version": CAR_PLACEMENT_VERSION,
        "actor_type": "car",
        "vehicle_behavior": "stationary",
        "vehicle_position_policy": "rear_bumper_at_future_ego_front_bumper",
        "vehicle_heading_policy": "future_ego_forward",
        "reference_frame_index": reference_frame_index,
        "reference_timestamp_us": int(reference_frame["timestamp"]),
        "reference_vehicle_heading_global_rad": float(reference_yaw),
        "reference_vehicle_forward_global_xy": forward.tolist(),
        "reference_vehicle_left_global_xy": left.tolist(),
        "contact_timestamp_s": contact_timestamp_s,
        "time_to_contact_s": time_to_contact_s,
        "ego_front_bumper_contact_global_xy_m": ego_front_global_xy.tolist(),
        "vehicle_rear_bumper_contact_global_xy_m": rear_bumper_global_xy.tolist(),
        "vehicle_origin_global_xy_m": car_origin_global_xy.tolist(),
        "vehicle_rear_bumper_local_x_m": rear_bumper_x_m,
        "bumper_contact_error_m": bumper_contact_error_m,
        "bumper_lateral_overlap_m": lateral_overlap_m,
        "first_contact_validation": first_contact,
        "final_history_actor_global_xy_m": car_origin_global_xy.tolist(),
        "anchor_global_xy_m": car_origin_global_xy.tolist(),
        "anchor_current_ego_xy_m": car_origin_current_xy.tolist(),
    }
    return {
        "placement": placement,
        "reference_frame_index": reference_frame_index,
        "reference_vehicle_forward_global_xy": forward,
        "reference_vehicle_left_global_xy": left,
        "walking_velocity_global_xy_mps": zero_velocity,
        "actor_velocity_global_xy_mps": zero_velocity,
        "anchor_global_xy": car_origin_global_xy,
        "anchor_current_ego_xy": car_origin_current_xy,
        "crossing_offsets_m": zero_offsets,
        "actor_global_xys": actor_global_xys,
        "actor_current_xys": actor_current_xys,
    }


def road_basis_aligned_to_global_forward(
    plane: np.ndarray,
    lidar_to_global: np.ndarray,
    reference_forward_global_xy: Sequence[float],
    max_tilt_degrees: float,
) -> np.ndarray:
    """Align a road-tangent actor basis to the selected vehicle heading."""
    lidar_to_global = np.asarray(lidar_to_global, dtype=np.float64)
    if lidar_to_global.shape != (4, 4) or not np.isfinite(
        lidar_to_global
    ).all():
        raise TokenFailure(
            "invalid_lidar_pose",
            "frame must contain a finite 4x4 lidar2global transform",
        )
    gravity_up_lidar = lidar_to_global[:3, :3].T @ np.array(
        [0.0, 0.0, 1.0], dtype=np.float64
    )
    base_basis = render_utils.road_basis(
        plane,
        gravity_up=gravity_up_lidar,
        max_tilt_degrees=max_tilt_degrees,
    )
    normal = base_basis[:, 2]
    forward_global = np.array(
        [
            float(reference_forward_global_xy[0]),
            float(reference_forward_global_xy[1]),
            0.0,
        ],
        dtype=np.float64,
    )
    forward_lidar = lidar_to_global[:3, :3].T @ forward_global
    forward_lidar -= normal * float(np.dot(normal, forward_lidar))
    norm = float(np.linalg.norm(forward_lidar))
    if not math.isfinite(norm) or norm < 1e-6:
        raise TokenFailure(
            "invalid_reference_heading",
            "reference vehicle heading cannot be projected onto the road plane",
        )
    forward_lidar /= norm
    left_lidar = np.cross(normal, forward_lidar)
    left_lidar /= np.linalg.norm(left_lidar)
    return np.column_stack([forward_lidar, left_lidar, normal])


def placement_metadata_fields(placement: dict) -> dict:
    """Return top-level aliases for the timed future interception."""
    keys = (
        "placement_mode",
        "future_timestep",
        "future_frame_index",
        "expected_future_offset_s",
        "future_offset_s",
        "future_front_bumper_global_xy_m",
        "interception_timestamp_s",
        "interception_global_xy_m",
        "pedestrian_time_to_interception_s",
        "pedestrian_distance_to_interception_m",
        "final_history_actor_global_xy_m",
        "walking_direction_global_xy",
        "anchor_global_xy_m",
        "anchor_current_ego_xy_m",
        "reference_frame_index",
        "reference_timestamp_us",
        "reference_vehicle_heading_global_rad",
        "reference_vehicle_forward_global_xy",
        "crossing_axis_global_xy",
        "pedestrian_behavior",
        "standing_pose_version",
        "standing_position_policy",
        "standing_facing_policy",
        "standing_facing_direction_global_xy",
        "actor_type",
        "vehicle_behavior",
        "vehicle_position_policy",
        "vehicle_heading_policy",
        "contact_timestamp_s",
        "time_to_contact_s",
        "ego_front_bumper_contact_global_xy_m",
        "vehicle_rear_bumper_contact_global_xy_m",
        "vehicle_origin_global_xy_m",
        "vehicle_rear_bumper_local_x_m",
        "bumper_contact_error_m",
        "bumper_lateral_overlap_m",
        "first_contact_validation",
    )
    return {
        key: placement[key]
        for key in keys
        if key in placement
    }


def solar_position_utc(
    timestamp_s: float,
    latitude_degrees: float,
    longitude_degrees: float,
) -> Tuple[float, float]:
    """Return geometric solar azimuth/elevation for a Unix UTC timestamp.

    This is the NOAA fractional-year/Julian-century solar-position model.  The
    azimuth is clockwise from true north.  Atmospheric refraction is
    deliberately excluded: a cast-shadow ray should follow the geometric
    direction to the centre of the solar disc.
    """
    julian_day = float(timestamp_s) / 86400.0 + 2440587.5
    julian_century = (julian_day - 2451545.0) / 36525.0

    mean_longitude = (
        280.46646
        + julian_century * (36000.76983 + 0.0003032 * julian_century)
    ) % 360.0
    mean_anomaly = 357.52911 + julian_century * (
        35999.05029 - 0.0001537 * julian_century
    )
    eccentricity = 0.016708634 - julian_century * (
        0.000042037 + 0.0000001267 * julian_century
    )
    anomaly_radians = math.radians(mean_anomaly)
    equation_of_center = (
        math.sin(anomaly_radians)
        * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century))
        + math.sin(2.0 * anomaly_radians) * (0.019993 - 0.000101 * julian_century)
        + math.sin(3.0 * anomaly_radians) * 0.000289
    )
    true_longitude = mean_longitude + equation_of_center
    omega = 125.04 - 1934.136 * julian_century
    apparent_longitude = (
        true_longitude - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    )
    mean_obliquity = 23.0 + (
        26.0
        + (
            21.448
            - julian_century
            * (46.815 + julian_century * (0.00059 - 0.001813 * julian_century))
        )
        / 60.0
    ) / 60.0
    corrected_obliquity = mean_obliquity + 0.00256 * math.cos(math.radians(omega))
    obliquity_radians = math.radians(corrected_obliquity)
    declination_radians = math.asin(
        math.sin(obliquity_radians) * math.sin(math.radians(apparent_longitude))
    )

    y = math.tan(obliquity_radians / 2.0) ** 2
    longitude_radians = math.radians(mean_longitude)
    equation_of_time_minutes = 4.0 * math.degrees(
        y * math.sin(2.0 * longitude_radians)
        - 2.0 * eccentricity * math.sin(anomaly_radians)
        + 4.0
        * eccentricity
        * y
        * math.sin(anomaly_radians)
        * math.cos(2.0 * longitude_radians)
        - 0.5 * y * y * math.sin(4.0 * longitude_radians)
        - 1.25 * eccentricity * eccentricity * math.sin(2.0 * anomaly_radians)
    )

    utc = datetime.fromtimestamp(float(timestamp_s), tz=timezone.utc)
    utc_minutes = (
        utc.hour * 60.0
        + utc.minute
        + utc.second / 60.0
        + utc.microsecond / 60_000_000.0
    )
    true_solar_minutes = (
        utc_minutes + equation_of_time_minutes + 4.0 * float(longitude_degrees)
    ) % 1440.0
    hour_angle_degrees = true_solar_minutes / 4.0 - 180.0
    latitude_radians = math.radians(latitude_degrees)
    hour_angle_radians = math.radians(hour_angle_degrees)
    cosine_zenith = (
        math.sin(latitude_radians) * math.sin(declination_radians)
        + math.cos(latitude_radians)
        * math.cos(declination_radians)
        * math.cos(hour_angle_radians)
    )
    zenith_degrees = math.degrees(math.acos(float(np.clip(cosine_zenith, -1.0, 1.0))))
    elevation_degrees = 90.0 - zenith_degrees
    azimuth_degrees = (
        math.degrees(
            math.atan2(
                math.sin(hour_angle_radians),
                math.cos(hour_angle_radians) * math.sin(latitude_radians)
                - math.tan(declination_radians) * math.cos(latitude_radians),
            )
        )
        + 180.0
    ) % 360.0
    return float(azimuth_degrees), float(elevation_degrees)


def astronomical_sun_direction(frame: dict) -> dict:
    """Recover one token-level sun ray in NAVSIM's projected global frame.

    A short true-geodesic step at the solar azimuth is transformed back to the
    map UTM CRS.  Its projected displacement is the grid-frame horizontal
    direction, so UTM meridian convergence is handled without manually adding
    an angle with an easy-to-invert sign convention.
    """
    map_location = str(frame.get("map_location", ""))
    projected_crs = MAP_UTM_CRS.get(map_location)
    if projected_crs is None:
        raise TokenFailure(
            "unsupported_map_crs",
            f"no UTM CRS is configured for NAVSIM map {map_location!r}",
        )
    timestamp_us = int(frame["timestamp"])
    timestamp_s = timestamp_us / 1e6
    ego_to_global = np.asarray(frame["ego2global"], dtype=np.float64)
    easting, northing = [float(value) for value in ego_to_global[:2, 3]]
    to_wgs84 = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    from_wgs84 = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    longitude, latitude = to_wgs84.transform(easting, northing)
    azimuth_true, elevation = solar_position_utc(timestamp_s, latitude, longitude)

    step_longitude, step_latitude, _ = WGS84_GEOD.fwd(
        longitude, latitude, azimuth_true, 1000.0
    )
    step_easting, step_northing = from_wgs84.transform(step_longitude, step_latitude)
    horizontal_grid = np.array(
        [step_easting - easting, step_northing - northing], dtype=np.float64
    )
    horizontal_norm = float(np.linalg.norm(horizontal_grid))
    if not np.isfinite(horizontal_norm) or horizontal_norm < 1e-6:
        raise TokenFailure(
            "sun_direction_failed",
            f"invalid projected geodesic at ({latitude:.7f}, {longitude:.7f})",
        )
    horizontal_grid /= horizontal_norm
    elevation_radians = math.radians(elevation)
    direction_to_sun_global = np.array(
        [
            math.cos(elevation_radians) * horizontal_grid[0],
            math.cos(elevation_radians) * horizontal_grid[1],
            math.sin(elevation_radians),
        ],
        dtype=np.float64,
    )
    direction_to_sun_global /= np.linalg.norm(direction_to_sun_global)
    sun_ray_global = -direction_to_sun_global
    grid_azimuth = math.degrees(
        math.atan2(float(horizontal_grid[0]), float(horizontal_grid[1]))
    ) % 360.0
    convergence = float(
        Proj(projected_crs).get_factors(longitude, latitude).meridian_convergence
    )
    timestamp_utc = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
    return {
        "method": "noaa_geometric_sun_geodesic_utm_v1",
        "timestamp_us": timestamp_us,
        "timestamp_utc": timestamp_utc.isoformat().replace("+00:00", "Z"),
        "map_location": map_location,
        "projected_crs": projected_crs,
        "utm_easting_m": easting,
        "utm_northing_m": northing,
        "latitude_degrees": float(latitude),
        "longitude_degrees": float(longitude),
        "azimuth_true_north_degrees": azimuth_true,
        "azimuth_grid_north_degrees": float(grid_azimuth),
        "elevation_geometric_degrees": elevation,
        "utm_meridian_convergence_degrees": convergence,
        "direction_to_sun_global_utm": direction_to_sun_global.tolist(),
        "sun_ray_direction_global_utm": sun_ray_global.tolist(),
        "direct_sun_enabled": bool(elevation > 0.0),
    }


def global_direction_to_actor(
    direction_global: Sequence[float],
    lidar_to_global: np.ndarray,
    lidar_to_actor: np.ndarray,
) -> np.ndarray:
    """Rotate a UTM/global direction into a frame's Blender actor axes."""
    direction_global = np.asarray(direction_global, dtype=np.float64)
    global_to_lidar_rotation = np.asarray(lidar_to_global, dtype=np.float64)[:3, :3].T
    lidar_to_actor_rotation = np.asarray(lidar_to_actor, dtype=np.float64)[:3, :3]
    direction_actor = lidar_to_actor_rotation @ global_to_lidar_rotation @ direction_global
    norm = float(np.linalg.norm(direction_actor))
    if not np.isfinite(norm) or norm < 1e-9:
        raise TokenFailure("sun_direction_failed", "invalid per-frame actor sun direction")
    return direction_actor / norm


def fit_frame_ground(points: np.ndarray, actor_xy: Sequence[float]) -> Tuple[np.ndarray, dict]:
    """Fit and validate the support surface at one actor timestamp.

    Each history frame has its own LiDAR/ego pose.  Grounding every pose from
    the final frame's plane makes earlier actors hover or penetrate the road.
    This fit is deliberately local to the actor's feet at the same timestamp.
    """
    actor_x, actor_y = (float(actor_xy[0]), float(actor_xy[1]))
    x_range = (actor_x - 4.0, actor_x + 4.0)
    y_range = (actor_y - 4.0, actor_y + 4.0)
    try:
        plane = geometry.fit_road_plane(points, x_range=x_range, y_range=y_range)
    except Exception as exc:
        raise TokenFailure("ground_fit_failed", str(exc)) from exc

    candidate_mask = (
        (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1])
        & (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
        & (points[:, 2] > -1.5) & (points[:, 2] < 0.5)
    )
    candidates = points[candidate_mask]
    predicted = plane[0] * candidates[:, 0] + plane[1] * candidates[:, 1] + plane[2]
    residual = candidates[:, 2] - predicted
    inliers = np.abs(residual) < 0.12
    support = (
        (np.abs(candidates[:, 0] - actor_x) < 1.6)
        & (np.abs(candidates[:, 1] - actor_y) < 1.6)
        & inliers
    )
    inlier_count = int(np.count_nonzero(inliers))
    support_count = int(np.count_nonzero(support))
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2))) if inlier_count else float("inf")
    slope = float(np.linalg.norm(plane[:2]))
    if inlier_count < 80 or support_count < 12 or rmse > 0.09 or slope > 0.22:
        raise TokenFailure(
            "ground_fit_failed",
            "unreliable per-frame support fit: "
            f"inliers={inlier_count}, local_support={support_count}, "
            f"rmse={rmse:.3f}, slope={slope:.3f}",
        )
    local_residual = residual[
        (np.abs(candidates[:, 0] - actor_x) < 1.6)
        & (np.abs(candidates[:, 1] - actor_y) < 1.6)
    ]
    local_p10 = float(np.percentile(local_residual, 10)) if len(local_residual) else None
    local_p90 = float(np.percentile(local_residual, 90)) if len(local_residual) else None
    return plane, {
        "actor_xy_lidar": [actor_x, actor_y],
        "x_range": list(x_range),
        "y_range": list(y_range),
        "candidate_count": int(len(candidates)),
        "inlier_count": inlier_count,
        "local_support_count": support_count,
        "rmse_m": rmse,
        "slope": slope,
        "local_surface_residual_p10_m": local_p10,
        "local_surface_residual_p90_m": local_p90,
        "local_surface_residual_spread_m": (
            float(local_p90 - local_p10) if local_p10 is not None and local_p90 is not None else None
        ),
    }


def _plane_as_general(plane: Sequence[float]) -> np.ndarray:
    """Convert ``z = ax + by + c`` to a homogeneous plane equation."""
    a, b, c = np.asarray(plane, dtype=np.float64)
    return np.array([-a, -b, 1.0, -c], dtype=np.float64)


def _general_as_plane(coefficients: Sequence[float]) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("invalid homogeneous plane coefficients")
    if abs(float(coefficients[2])) < 1e-6:
        raise ValueError("ground plane is vertical in the destination frame")
    normalized = coefficients / float(coefficients[2])
    return np.array(
        [-normalized[0], -normalized[1], -normalized[3]],
        dtype=np.float64,
    )


def _common_frame_from_final_ego(final_ego_to_global: np.ndarray) -> np.ndarray:
    """Return a gravity-aligned global-to-final-yaw coordinate transform."""
    final_ego_to_global = np.asarray(final_ego_to_global, dtype=np.float64)
    yaw = planar_yaw(final_ego_to_global)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    common_to_global_rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    global_to_common = np.eye(4, dtype=np.float64)
    global_to_common[:3, :3] = common_to_global_rotation.T
    origin = np.asarray(final_ego_to_global[:3, 3], dtype=np.float64)
    global_to_common[:3, 3] = -common_to_global_rotation.T @ origin
    return global_to_common


def _plane_common_to_lidar(
    plane_common: Sequence[float],
    lidar_to_common: np.ndarray,
) -> np.ndarray:
    return _general_as_plane(
        np.asarray(lidar_to_common, dtype=np.float64).T
        @ _plane_as_general(plane_common)
    )


def _plane_lidar_to_common(
    plane_lidar: Sequence[float],
    lidar_to_common: np.ndarray,
) -> np.ndarray:
    return _general_as_plane(
        np.linalg.inv(np.asarray(lidar_to_common, dtype=np.float64)).T
        @ _plane_as_general(plane_lidar)
    )


def fit_relaxed_ground_plane(
    points: np.ndarray,
    actor_xy: Sequence[float],
    half_extent_m: float,
) -> Tuple[np.ndarray, dict]:
    """Robust low-surface fit used only after the strict production fit fails."""
    points = np.asarray(points, dtype=np.float64)
    actor_x, actor_y = [float(value) for value in actor_xy]
    finite = np.isfinite(points[:, :3]).all(axis=1)
    candidates = points[
        finite
        & (np.abs(points[:, 0] - actor_x) < half_extent_m)
        & (np.abs(points[:, 1] - actor_y) < half_extent_m)
        & (points[:, 2] > -3.0)
        & (points[:, 2] < 1.0)
    ]
    if len(candidates) < 12:
        raise RuntimeError(
            f"only {len(candidates)} candidate returns in "
            f"{2.0 * half_extent_m:.1f} m window"
        )

    grid_size = 0.9
    grid_x = np.floor((candidates[:, 0] - actor_x) / grid_size).astype(int)
    grid_y = np.floor((candidates[:, 1] - actor_y) / grid_size).astype(int)
    cells = np.column_stack([grid_x, grid_y])
    samples = []
    for cell in np.unique(cells, axis=0):
        cell_points = candidates[
            (grid_x == int(cell[0])) & (grid_y == int(cell[1]))
        ]
        count = len(cell_points)
        if count == 0:
            continue
        retain_count = max(1, int(math.ceil(0.35 * count)))
        low = cell_points[np.argsort(cell_points[:, 2])[:retain_count]]
        samples.append(np.median(low, axis=0))
    samples_array = np.asarray(samples, dtype=np.float64)
    if len(samples_array) < 6:
        raise RuntimeError(
            f"only {len(samples_array)} occupied low-surface grid cells"
        )

    centered_design = np.column_stack(
        [
            samples_array[:, 0] - actor_x,
            samples_array[:, 1] - actor_y,
            np.ones(len(samples_array), dtype=np.float64),
        ]
    )
    keep = np.ones(len(samples_array), dtype=bool)
    centered_coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(8):
        if int(np.count_nonzero(keep)) < 6:
            raise RuntimeError("robust plane retained fewer than six grid cells")
        centered_coefficients = np.linalg.lstsq(
            centered_design[keep],
            samples_array[keep, 2],
            rcond=None,
        )[0]
        residual = samples_array[:, 2] - centered_design @ centered_coefficients
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center))) + 1e-6
        next_keep = np.abs(residual - center) < max(
            0.10,
            3.0 * 1.4826 * mad,
        )
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep

    residual = samples_array[:, 2] - centered_design @ centered_coefficients
    retained_count = int(np.count_nonzero(keep))
    rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    slope = float(np.linalg.norm(centered_coefficients[:2]))
    if not np.isfinite(centered_coefficients).all():
        raise RuntimeError("robust plane coefficients are non-finite")
    if retained_count < 6 or rmse > 0.25 or slope > 0.35:
        raise RuntimeError(
            "relaxed plane remains implausible: "
            f"grid_inliers={retained_count}, rmse={rmse:.3f}, "
            f"slope={slope:.3f}"
        )

    a, b, centered_height = centered_coefficients
    plane = np.array(
        [a, b, centered_height - a * actor_x - b * actor_y],
        dtype=np.float64,
    )
    horizontal_distance = np.linalg.norm(
        candidates[:, :2] - np.array([actor_x, actor_y], dtype=np.float64),
        axis=1,
    )
    support_count = int(np.count_nonzero(horizontal_distance < 2.5))
    return plane, {
        "actor_xy": [actor_x, actor_y],
        "half_extent_m": float(half_extent_m),
        "candidate_count": int(len(candidates)),
        "grid_cell_count": int(len(samples_array)),
        "grid_inlier_count": retained_count,
        "local_support_count_2p5m": support_count,
        "nearest_candidate_distance_m": float(np.min(horizontal_distance)),
        "rmse_m": rmse,
        "slope": slope,
    }


def _retry_relaxed_ground(
    points: np.ndarray,
    actor_xy: Sequence[float],
    windows_m: Sequence[float],
) -> Tuple[np.ndarray, dict]:
    attempts = []
    for half_extent in windows_m:
        try:
            plane, diagnostics = fit_relaxed_ground_plane(
                points,
                actor_xy,
                half_extent,
            )
        except Exception as exc:
            attempts.append(
                {
                    "half_extent_m": float(half_extent),
                    "error": str(exc),
                }
            )
            continue
        diagnostics["failed_attempts"] = attempts
        return plane, diagnostics
    raise RuntimeError(
        "; ".join(
            f"{item['half_extent_m']:.1f}m: {item['error']}"
            for item in attempts
        )
    )


def _emergency_horizontal_common_plane(
    common_points: np.ndarray,
    actor_xy: Sequence[float],
) -> Tuple[np.ndarray, dict]:
    """Last-resort height estimate; always flagged for manual inspection."""
    points = np.asarray(common_points, dtype=np.float64)
    finite = np.isfinite(points[:, :3]).all(axis=1)
    points = points[
        finite & (points[:, 2] > -4.0) & (points[:, 2] < 1.5)
    ]
    if len(points) == 0:
        raise RuntimeError("all accumulated LiDAR points are invalid")
    actor_xy_array = np.asarray(actor_xy, dtype=np.float64)
    distances = np.linalg.norm(points[:, :2] - actor_xy_array[None], axis=1)
    retain_count = min(len(points), 512)
    nearest_indices = np.argpartition(
        distances,
        retain_count - 1,
    )[:retain_count]
    nearest = points[nearest_indices]
    low_cut = float(np.quantile(nearest[:, 2], 0.30))
    low = nearest[nearest[:, 2] <= low_cut]
    height = float(np.median(low[:, 2]))
    return np.array([0.0, 0.0, height], dtype=np.float64), {
        "method": "nearest_accumulated_low_surface_horizontal_emergency",
        "candidate_count": int(len(points)),
        "nearest_count": int(retain_count),
        "low_surface_count": int(len(low)),
        "nearest_distance_m": float(np.min(distances)),
        "farthest_retained_distance_m": float(
            np.max(distances[nearest_indices])
        ),
        "estimated_height_common_m": height,
    }


def fit_ground_sequence_best_effort(
    frames: Sequence[dict],
    frame_points: Sequence[np.ndarray],
    actor_lidar_xy: Sequence[Sequence[float]],
    actor_reference_current_xy: Sequence[Sequence[float]],
) -> Tuple[List[np.ndarray], List[dict], List[dict]]:
    """Ground all four actor poses while preserving strict-fit failures as flags."""
    if not (
        len(frames)
        == len(frame_points)
        == len(actor_lidar_xy)
        == len(actor_reference_current_xy)
        == EXPECTED_HISTORY_FRAMES
    ):
        raise ValueError(
            "ground sequence inputs must contain exactly "
            f"{EXPECTED_HISTORY_FRAMES} frames"
        )

    global_to_common = _common_frame_from_final_ego(
        np.asarray(frames[-1]["ego2global"], dtype=np.float64)
    )
    lidar_to_common = [
        global_to_common
        @ np.asarray(frame["lidar2global"], dtype=np.float64)
        for frame in frames
    ]
    common_points_by_frame = [
        geometry.transform_points(transform, points)
        for transform, points in zip(lidar_to_common, frame_points)
    ]
    accumulated_common = np.concatenate(common_points_by_frame, axis=0)

    planes: List[Optional[np.ndarray]] = [None] * EXPECTED_HISTORY_FRAMES
    diagnostics: List[Optional[dict]] = [None] * EXPECTED_HISTORY_FRAMES
    strict_failures = []
    unresolved = []
    for frame_index in HISTORY_FRAME_INDICES:
        points = frame_points[frame_index]
        local_xy = actor_lidar_xy[frame_index]
        reference_xy = actor_reference_current_xy[frame_index]
        try:
            plane, strict_diagnostics = fit_frame_ground(points, local_xy)
        except Exception as strict_exc:
            strict_error = str(strict_exc)
            strict_failures.append(
                {
                    "frame_index": frame_index,
                    "code": (
                        strict_exc.code
                        if isinstance(strict_exc, TokenFailure)
                        else "ground_fit_failed"
                    ),
                    "message": strict_error,
                }
            )
        else:
            strict_diagnostics.update(
                {
                    "frame_index": frame_index,
                    "method": "strict_same_timestamp_local_lidar",
                    "first_attempt_failed": False,
                }
            )
            planes[frame_index] = plane
            diagnostics[frame_index] = strict_diagnostics
            continue

        fallback_errors = []
        try:
            plane, relaxed_diagnostics = _retry_relaxed_ground(
                points,
                local_xy,
                (4.0, 8.0, 12.0),
            )
        except Exception as exc:
            fallback_errors.append(
                {"method": "relaxed_same_frame", "error": str(exc)}
            )
        else:
            relaxed_diagnostics.update(
                {
                    "frame_index": frame_index,
                    "method": "relaxed_same_timestamp_local_lidar",
                    "first_attempt_failed": True,
                    "first_attempt_error": strict_error,
                    "fallback_errors": fallback_errors,
                }
            )
            planes[frame_index] = plane
            diagnostics[frame_index] = relaxed_diagnostics
            continue

        try:
            plane_common, accumulated_diagnostics = _retry_relaxed_ground(
                accumulated_common,
                reference_xy,
                (4.0, 8.0, 12.0, 20.0),
            )
            plane = _plane_common_to_lidar(
                plane_common,
                lidar_to_common[frame_index],
            )
        except Exception as exc:
            fallback_errors.append(
                {"method": "accumulated_four_frame", "error": str(exc)}
            )
            unresolved.append(
                {
                    "frame_index": frame_index,
                    "strict_error": strict_error,
                    "fallback_errors": fallback_errors,
                }
            )
        else:
            accumulated_diagnostics.update(
                {
                    "frame_index": frame_index,
                    "method": "accumulated_four_frame_local_lidar",
                    "coordinate_frame": "gravity_aligned_final_ego_yaw",
                    "accumulated_frame_count": EXPECTED_HISTORY_FRAMES,
                    "first_attempt_failed": True,
                    "first_attempt_error": strict_error,
                    "fallback_errors": fallback_errors,
                    "plane_common": plane_common.tolist(),
                }
            )
            planes[frame_index] = plane
            diagnostics[frame_index] = accumulated_diagnostics

    resolved_indices = [
        index for index, plane in enumerate(planes) if plane is not None
    ]
    for item in unresolved:
        frame_index = int(item["frame_index"])
        reference_xy = np.asarray(
            actor_reference_current_xy[frame_index],
            dtype=np.float64,
        )
        if resolved_indices:
            source_index = min(
                resolved_indices,
                key=lambda index: (
                    abs(index - frame_index),
                    float(
                        np.linalg.norm(
                            np.asarray(actor_reference_current_xy[index])
                            - reference_xy
                        )
                    ),
                ),
            )
            source_common = _plane_lidar_to_common(
                planes[source_index],
                lidar_to_common[source_index],
            )
            plane = _plane_common_to_lidar(
                source_common,
                lidar_to_common[frame_index],
            )
            diagnostics[frame_index] = {
                "frame_index": frame_index,
                "method": "nearest_resolved_timestep_plane_transfer",
                "first_attempt_failed": True,
                "first_attempt_error": item["strict_error"],
                "fallback_errors": item["fallback_errors"],
                "source_frame_index": int(source_index),
                "source_method": diagnostics[source_index]["method"],
                "actor_reference_distance_from_source_m": float(
                    np.linalg.norm(
                        np.asarray(actor_reference_current_xy[source_index])
                        - reference_xy
                    )
                ),
                "plane_common": source_common.tolist(),
            }
        else:
            plane_common, emergency_diagnostics = (
                _emergency_horizontal_common_plane(
                    accumulated_common,
                    reference_xy,
                )
            )
            plane = _plane_common_to_lidar(
                plane_common,
                lidar_to_common[frame_index],
            )
            diagnostics[frame_index] = {
                **emergency_diagnostics,
                "frame_index": frame_index,
                "first_attempt_failed": True,
                "first_attempt_error": item["strict_error"],
                "fallback_errors": item["fallback_errors"],
                "plane_common": plane_common.tolist(),
            }
        planes[frame_index] = plane

    if any(plane is None for plane in planes) or any(
        item is None for item in diagnostics
    ):
        raise RuntimeError(
            "internal error: best-effort grounding left a frame unresolved"
        )
    return (
        [np.asarray(plane, dtype=np.float64) for plane in planes],
        [dict(item) for item in diagnostics],
        strict_failures,
    )


def ground_fit_potential_failure(
    token: str,
    strict_failures: Sequence[dict],
    ground_diagnostics: Sequence[dict],
) -> Optional[dict]:
    if not strict_failures:
        return None
    failed_indices = [int(item["frame_index"]) for item in strict_failures]
    return {
        "status": "potential_failure",
        "token": token,
        "code": "ground_fit_failed",
        "message": (
            "Strict same-frame ground fitting failed and best-effort grounding "
            f"was used for frame(s) {failed_indices}"
        ),
        "failed_frame_indices": failed_indices,
        "strict_failures": list(strict_failures),
        "fallbacks": [
            {
                "frame_index": int(item["frame_index"]),
                "method": item["method"],
                "diagnostics": item,
            }
            for item in ground_diagnostics
            if item.get("first_attempt_failed")
        ],
    }


def frame_collision_diagnostics(
    points: np.ndarray,
    lidar_to_actor: np.ndarray,
    actor_heading: float,
    local_bounds_m: dict,
) -> dict:
    """Check measured pedestrian x/y bounds against non-ground returns."""
    points_actor = geometry.transform_points(lidar_to_actor, points)
    cosine, sine = math.cos(float(actor_heading)), math.sin(float(actor_heading))
    local_to_actor = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points_local = points_actor @ local_to_actor
    x_bounds = local_bounds_m["x"]
    y_bounds = local_bounds_m["y"]
    z_bounds = local_bounds_m["z"]
    ground_clearance_m = 0.15
    occupied = (
        (points_local[:, 0] > float(x_bounds[0]))
        & (points_local[:, 0] < float(x_bounds[1]))
        & (points_local[:, 1] > float(y_bounds[0]))
        & (points_local[:, 1] < float(y_bounds[1]))
        & (points_local[:, 2] > float(z_bounds[0]) + ground_clearance_m)
        & (points_local[:, 2] < float(z_bounds[1]))
    )
    count = int(np.count_nonzero(occupied))
    return {
        "method": "oriented_measured_pedestrian_local_bounds_v1",
        "local_bounds_m": local_bounds_m,
        "ground_exclusion_height_m": ground_clearance_m,
        "obstacle_points": count,
        "collision_threshold_points": 5,
        "occupied": count >= 5,
    }


def stationary_vehicle_collision_diagnostics(
    points_lidar: np.ndarray,
    lidar_to_actor: np.ndarray,
    local_bounds: dict,
) -> dict:
    """Count existing non-ground LiDAR returns inside a car's oriented box."""
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError("LiDAR points must be an Nx3 array")
    points_actor = geometry.transform_points(lidar_to_actor, points_lidar)
    x_bounds = local_bounds["x"]
    y_bounds = local_bounds["y"]
    z_bounds = local_bounds["z"]
    occupied = (
        (points_actor[:, 0] > float(x_bounds[0]))
        & (points_actor[:, 0] < float(x_bounds[1]))
        & (points_actor[:, 1] > float(y_bounds[0]))
        & (points_actor[:, 1] < float(y_bounds[1]))
        & (points_actor[:, 2] > float(z_bounds[0]) + 0.15)
        & (points_actor[:, 2] < float(z_bounds[1]) + 0.15)
    )
    count = int(np.count_nonzero(occupied))
    return {
        "method": "oriented_vehicle_local_bounds_v1",
        "obstacle_points": count,
        "maximum_obstacle_points": count,
        "collision_threshold_points": 5,
        "occupied": count >= 5,
        "local_bounds_m": local_bounds,
        "sample_count": 1,
    }


def trajectory_collision_diagnostics(
    points: np.ndarray,
    plane: np.ndarray,
    trajectory_xy: Any,
    *,
    actor_basis_lidar: np.ndarray,
    actor_heading: float,
    local_bounds_m: dict,
) -> dict:
    """Count non-ground LiDAR support inside the swept pedestrian volume.

    The trajectory is sampled more densely than the four image timestamps so a
    narrow pole or parked object between two recorded actor positions is not
    skipped. Sparse returns are treated conservatively: at least five points
    must support an obstruction before the token is flagged for review.
    """
    trajectory_xy = np.asarray(trajectory_xy, dtype=np.float64)
    if (
        trajectory_xy.ndim != 2
        or trajectory_xy.shape[1] != 2
        or len(trajectory_xy) == 0
        or not np.isfinite(trajectory_xy).all()
    ):
        raise ValueError("trajectory collision path must be a finite Nx2 array")

    if len(trajectory_xy) == 1:
        samples = np.repeat(trajectory_xy, 2, axis=0)
    else:
        segment_lengths = np.linalg.norm(
            np.diff(trajectory_xy, axis=0), axis=1
        )
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = float(cumulative[-1])
        sample_count = max(2, int(math.ceil(total_length / 0.25)) + 1)
        if total_length < 1e-9:
            samples = np.repeat(trajectory_xy[:1], sample_count, axis=0)
        else:
            distances = np.linspace(0.0, total_length, sample_count)
            samples = np.column_stack(
                [
                    np.interp(distances, cumulative, trajectory_xy[:, axis])
                    for axis in range(2)
                ]
            )
    sample_count = int(len(samples))
    counts = []
    for actor_x, actor_y in samples:
        actor_position = np.array(
            [
                float(actor_x),
                float(actor_y),
                geometry.ground_z(plane, float(actor_x), float(actor_y)),
            ],
            dtype=np.float64,
        )
        diagnostics = frame_collision_diagnostics(
            points,
            np.linalg.inv(
                render_utils.matrix4(actor_basis_lidar, actor_position)
            ),
            actor_heading,
            local_bounds_m,
        )
        counts.append(int(diagnostics["obstacle_points"]))
    maximum = max(counts, default=0)
    return {
        "method": "swept_oriented_measured_pedestrian_local_bounds_v1",
        "local_bounds_m": local_bounds_m,
        "sample_count": sample_count,
        "path_start_xy_m": trajectory_xy[0].tolist(),
        "path_end_xy_m": trajectory_xy[-1].tolist(),
        "maximum_obstacle_points": maximum,
        "collision_threshold_points": 5,
        "occupied": maximum >= 5,
    }


def trajectory_collision_potential_failure(
    args: argparse.Namespace,
    token: str,
    collision_diagnostics: dict,
    placement: dict,
) -> Optional[dict]:
    if not collision_diagnostics["occupied"] or args.disable_collision_check:
        return None
    occupied_frames = [
        item["frame_index"]
        for item in collision_diagnostics["per_frame"]
        if item["occupied"]
    ]
    swept_collision = collision_diagnostics["swept_final_frame"]
    actor_label = "vehicle" if getattr(args, "car", False) else "pedestrian"
    return {
        "status": "potential_failure",
        "token": token,
        "code": "trajectory_collision",
        "message": (
            f"LiDAR-supported obstacle intersects the requested {actor_label} placement: "
            f"per_frame={occupied_frames}, "
            f"swept_max_points={swept_collision['maximum_obstacle_points']}"
        ),
        "placement": placement,
        "speed_mps": (
            0.0 if getattr(args, "car", False) else args.speed_mps
        ),
        "collision_diagnostics": collision_diagnostics,
    }


def actor_camera_depth(camera_to_actor: np.ndarray) -> float:
    """Return actor-root depth in OpenCV camera coordinates."""
    actor_to_camera = np.linalg.inv(camera_to_actor)
    return float(actor_to_camera[2, 3])


def actor_shadow_render_region(
    actor_bbox_actor: np.ndarray,
    camera_to_actor: np.ndarray,
    intrinsic: np.ndarray,
    resolution: Sequence[int],
    sun_ray_direction_actor: Sequence[float],
    sun_energy: float,
    sun_angle_degrees: float,
    padding_px: int = RENDER_REGION_PADDING_PX,
) -> List[int]:
    """Return a conservative ideal-image border for the actor and its shadow.

    The coordinates use the same top-left image convention as the render
    manifest and are ``[x_min, y_min, x_max, y_max]`` with exclusive maxima.
    Blender stores only this rectangle through crop-to-border; finalization
    carries its ideal-canvas origin into lens distortion. Falling back to the
    full canvas is intentional whenever shadow projection is geometrically
    unsafe.
    """
    width, height = [int(value) for value in resolution]
    full_region = [0, 0, width, height]
    bbox = np.asarray(actor_bbox_actor, dtype=np.float64)
    if bbox.shape != (8, 3) or not np.isfinite(bbox).all():
        return full_region

    # The NAVSIM visibility box is deliberately tight.  Expand it in world
    # space before projection so animated hands, clothing, antialiasing, and
    # the soft-Sun penumbra cannot touch the render border.
    bbox_min = bbox.min(axis=0) - np.array([0.16, 0.16, 0.03])
    bbox_max = bbox.max(axis=0) + np.array([0.16, 0.16, 0.12])
    actor_points = np.array(
        [
            [x, y, z]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=np.float64,
    )
    points = [actor_points]

    direct_sun = float(sun_energy) > 1e-6
    if direct_sun:
        ray = np.asarray(sun_ray_direction_actor, dtype=np.float64)
        ray_norm = float(np.linalg.norm(ray))
        if not np.isfinite(ray_norm) or ray_norm < 1e-8:
            return full_region
        ray /= ray_norm
        # A ray that is horizontal or points upward has no stable finite
        # receiver intersection.  Rendering the full frame is the safe case.
        if ray[2] >= -1e-3:
            return full_region
        ground_z = -0.004
        shadow_points = []
        for point in actor_points:
            distance = (ground_z - point[2]) / ray[2]
            if distance < 0.0:
                continue
            receiver_point = point + distance * ray
            # The Blender catcher is a 200 m square.  Clamping here is
            # conservative at its boundary and avoids an unbounded low-Sun
            # projection from disabling the optimization unnecessarily.
            receiver_point[0] = np.clip(
                receiver_point[0],
                -SHADOW_RECEIVER_HALF_EXTENT_M,
                SHADOW_RECEIVER_HALF_EXTENT_M,
            )
            receiver_point[1] = np.clip(
                receiver_point[1],
                -SHADOW_RECEIVER_HALF_EXTENT_M,
                SHADOW_RECEIVER_HALF_EXTENT_M,
            )
            receiver_point[2] = ground_z
            shadow_points.append(receiver_point)
        if not shadow_points:
            return full_region
        shadow_points = np.asarray(shadow_points, dtype=np.float64)
        penumbra_m = max(
            0.16,
            float(bbox_max[2] - ground_z)
            * math.tan(math.radians(max(0.0, float(sun_angle_degrees)))),
        )
        shadow_points = np.concatenate(
            [
                shadow_points + np.array([dx, dy, 0.0], dtype=np.float64)
                for dx in (-penumbra_m, penumbra_m)
                for dy in (-penumbra_m, penumbra_m)
            ],
            axis=0,
        )
        points.append(shadow_points)

    actor_to_camera = np.linalg.inv(np.asarray(camera_to_actor, dtype=np.float64))
    actor_camera = geometry.transform_points(actor_to_camera, actor_points)
    if not np.isfinite(actor_camera).all() or np.any(actor_camera[:, 2] <= 0.05):
        return full_region

    render_points = np.concatenate(points, axis=0)
    camera_points = geometry.transform_points(actor_to_camera, render_points)
    if not np.isfinite(camera_points).all():
        return full_region
    # A directional shadow crossing the camera plane can cover an arbitrarily
    # large image region even when its endpoints do not, so retain full-frame
    # rendering for that rare geometry.
    if direct_sun and np.any(camera_points[:, 2] <= 0.05):
        return full_region
    camera_points = camera_points[camera_points[:, 2] > 0.05]
    if len(camera_points) == 0:
        return full_region

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    pixels_x = intrinsic[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsic[0, 2]
    pixels_y = intrinsic[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsic[1, 2]
    pixels = np.column_stack([pixels_x, pixels_y])
    if not np.isfinite(pixels).all():
        return full_region

    alignment = RENDER_REGION_ALIGNMENT_PX
    x_min = math.floor((float(np.min(pixels[:, 0])) - padding_px) / alignment) * alignment
    y_min = math.floor((float(np.min(pixels[:, 1])) - padding_px) / alignment) * alignment
    x_max = math.ceil((float(np.max(pixels[:, 0])) + padding_px) / alignment) * alignment
    y_max = math.ceil((float(np.max(pixels[:, 1])) + padding_px) / alignment) * alignment
    region = [
        max(0, int(x_min)),
        max(0, int(y_min)),
        min(width, int(x_max)),
        min(height, int(y_max)),
    ]
    if region[0] >= region[2] or region[1] >= region[3]:
        return full_region
    region_area = (region[2] - region[0]) * (region[3] - region[1])
    if region_area >= 0.95 * width * height:
        return full_region
    return region


def animation_frames(
    timestamps_s: np.ndarray,
    speed: float,
    stride_length: float,
    start: int,
    end: int,
    initial_frame: int,
) -> List[int]:
    """Map travelled distance to gait phase, anchored at ``initial_frame``."""
    period = end - start
    if not start <= initial_frame < end:
        raise ValueError(f"Initial animation frame {initial_frame} is outside {start}..{end - 1}")
    if len(timestamps_s) == 0:
        return []
    frames = []
    initial_phase = initial_frame - start
    for timestamp in timestamps_s:
        travelled = speed * (timestamp - timestamps_s[0])
        phase = (initial_phase + travelled / stride_length * period) % period
        frames.append(start + int(round(phase)) % period)
    return frames


def estimate_lighting(image: np.ndarray, calibration: dict, camera_to_actor: np.ndarray) -> dict:
    rgb = image.astype(np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    height, width = luma.shape
    upper_height = max(1, int(height * 0.48))
    upper = rgb[:upper_height]
    upper_luma = luma[:upper_height]
    tile_rows, tile_cols = 2, 8
    tile_scores = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            y0, y1 = row * upper_height // tile_rows, (row + 1) * upper_height // tile_rows
            x0, x1 = col * width // tile_cols, (col + 1) * width // tile_cols
            tile_scores.append((float(np.percentile(upper_luma[y0:y1, x0:x1], 90)), x0, x1, y0, y1))
    tile_scores.sort(reverse=True)
    brightest, x0, x1, y0, y1 = tile_scores[0]
    median_tile = float(np.median([item[0] for item in tile_scores]))
    directionality = float(np.clip((brightest - median_tile) / 0.35, 0.0, 1.0))
    u, v = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    intrinsic = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    ray_camera = np.linalg.inv(intrinsic) @ np.array([u, v, 1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)
    direction_to_bright_sky = camera_to_actor[:3, :3] @ ray_camera
    direction_to_bright_sky /= np.linalg.norm(direction_to_bright_sky)
    sun_ray = -direction_to_bright_sky
    if sun_ray[2] > -0.12:
        sun_ray[2] = -0.55
        sun_ray /= np.linalg.norm(sun_ray)

    sky_color = np.percentile(upper.reshape(-1, 3), 65, axis=0)
    sky_color /= max(float(np.max(sky_color)), 1e-4)
    sky_color = np.clip(sky_color, 0.42, 1.0)
    median_luma = float(np.median(luma))
    exposure = float(np.clip(math.log2((median_luma + 0.08) / 0.48), -1.5, 1.0))
    return {
        "method": "image_auto",
        "world_color": sky_color.tolist(),
        "world_strength": float(np.clip(0.45 + 1.0 * median_luma, 0.45, 1.35)),
        "sun_ray_direction": sun_ray.tolist(),
        "sun_energy": float(0.35 + 2.15 * directionality),
        "sun_angle_degrees": float(6.0 + 20.0 * (1.0 - directionality)),
        "exposure": exposure,
        "image_statistics": {
            "median_luma": median_luma,
            "brightest_upper_tile": brightest,
            "median_upper_tile": median_tile,
            "directionality": directionality,
            "brightest_tile_center_px": [u, v],
        },
    }


def connected_sky_mask(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    height, width = luma.shape
    blue_or_gray_sky = (
        (rgb[..., 2] >= 0.88 * rgb[..., 0])
        & (rgb[..., 2] >= 0.90 * rgb[..., 1])
        & (luma > 0.20)
    )
    vertical_prior = np.indices((height, width))[0] < int(0.58 * height)
    candidate = (blue_or_gray_sky & vertical_prior).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    sky = np.zeros(candidate.shape, dtype=bool)
    for label in range(1, count):
        component = labels == label
        touches_top = np.any(component[: max(2, int(0.015 * height))])
        if touches_top and stats[label, cv2.CC_STAT_AREA] > 0.004 * height * width:
            sky |= component
    return sky


def smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def estimate_cast_shadow_evidence(image: np.ndarray) -> dict:
    """Estimate conservative direct-light cues and real-edge softness.

    Sky gradients alone are not reliable evidence of direct sunlight: cloud
    banks, haze, and exposure roll-off can all make one sky tile brighter than
    another. Unequivocally blue, top-connected sky supplies one vote; hard
    scene contrast is returned as an independent cue that the multiview caller
    accepts only with directional-sky corroboration. Generic same-material
    edges are used for penumbra measurement, not as sufficient weather
    evidence, because façades and lane paint also contain strong edges.
    """
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    if height < 32 or width < 32:
        return {
            "method": "blue_sky_hard_contrast_and_edge_softness_v2",
            "probability": 0.0,
            "confidence": 0.0,
            "candidate_fraction": 0.0,
            "candidate_count": 0,
            "edge_width_px": None,
        }

    analysis_width = min(width, 640)
    scale = float(width) / float(analysis_width)
    analysis_height = max(1, int(round(height / scale)))
    if (analysis_width, analysis_height) != (width, height):
        rgb = cv2.resize(
            rgb,
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_AREA,
        )
    height, width = rgb.shape[:2]
    luma = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)
    smooth_luma = cv2.GaussianBlur(luma, (0, 0), 2.0)
    smooth_rgb = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    gradient_x = cv2.Sobel(smooth_luma, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(smooth_luma, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    magnitude = np.hypot(gradient_x, gradient_y)
    normal_x = gradient_x / np.maximum(magnitude, 1e-5)
    normal_y = gradient_y / np.maximum(magnitude, 1e-5)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    sample_distance = 6.0

    def remap(values: np.ndarray, sign: float) -> np.ndarray:
        return cv2.remap(
            values,
            grid_x + sign * sample_distance * normal_x,
            grid_y + sign * sample_distance * normal_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    positive_luma = remap(smooth_luma, 1.0)
    negative_luma = remap(smooth_luma, -1.0)
    positive_rgb = remap(smooth_rgb, 1.0)
    negative_rgb = remap(smooth_rgb, -1.0)
    bright_luma = np.maximum(positive_luma, negative_luma)
    dark_luma = np.minimum(positive_luma, negative_luma)
    contrast = bright_luma - dark_luma
    relative_contrast = contrast / np.maximum(bright_luma, 0.08)
    positive_chroma = positive_rgb / np.maximum(
        np.sum(positive_rgb, axis=2, keepdims=True), 0.06
    )
    negative_chroma = negative_rgb / np.maximum(
        np.sum(negative_rgb, axis=2, keepdims=True), 0.06
    )
    chroma_delta = np.linalg.norm(positive_chroma - negative_chroma, axis=2)

    scene_region = np.zeros((height, width), dtype=bool)
    scene_region[int(round(0.28 * height)) : int(round(0.96 * height))] = True
    margin = max(8, int(round(sample_distance + 3.0)))
    scene_region[:margin] = False
    scene_region[-margin:] = False
    scene_region[:, :margin] = False
    scene_region[:, -margin:] = False
    candidate = (
        scene_region
        & (magnitude >= 0.008)
        & (contrast >= 0.055)
        & (relative_contrast >= 0.18)
        & (chroma_delta <= 0.115)
        & (bright_luma >= 0.12)
        & (bright_luma <= 0.96)
        & (dark_luma >= 0.025)
    )
    candidate_count = int(np.count_nonzero(candidate))
    region_count = max(1, int(np.count_nonzero(scene_region)))
    candidate_fraction = float(candidate_count / region_count)
    if candidate_count:
        candidate_contrast = relative_contrast[candidate]
        median_relative_contrast = float(np.median(candidate_contrast))
        strong_relative_contrast = float(np.percentile(candidate_contrast, 85))
    else:
        median_relative_contrast = 0.0
        strong_relative_contrast = 0.0
    edge_coverage_score = float(np.clip(candidate_fraction / 0.010, 0.0, 1.0))
    edge_contrast_score = float(
        np.clip((strong_relative_contrast - 0.18) / 0.34, 0.0, 1.0)
    )
    # Generic edges are retained only as diagnostics/penumbra samples.  They
    # are not sufficient evidence of sunlight: façades, cars, and lane paint
    # made the old edge count saturate even in fog and overcast scenes.
    sky = connected_sky_mask(np.rint(rgb * 255.0).astype(np.uint8))
    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 0.05)
    blue_delta = rgb[..., 2] - rgb[..., 0]
    blue_sky = (
        sky
        & (blue_delta >= 0.025)
        & (rgb[..., 2] >= rgb[..., 1] - 0.01)
        & (saturation >= 0.065)
        & (luma >= 0.18)
    )
    blue_sky_fraction = float(np.count_nonzero(blue_sky) / (height * width))
    median_blue_delta = (
        float(np.median(blue_delta[blue_sky])) if np.any(blue_sky) else 0.0
    )
    sky_coverage_score = smoothstep01(
        (blue_sky_fraction - 0.015) / 0.11
    )
    sky_chroma_score = smoothstep01((median_blue_delta - 0.075) / 0.075)
    probability = float(sky_coverage_score * sky_chroma_score)
    luma_range = float(np.percentile(luma, 95) - np.percentile(luma, 5))
    hard_contrast_probability = smoothstep01((luma_range - 0.58) / 0.16)
    confidence = float(
        np.clip(
            max(
                np.count_nonzero(blue_sky) / max(150.0, 0.02 * height * width),
                hard_contrast_probability,
            ),
            0.0,
            1.0,
        )
    )

    edge_width_px: float | None = None
    if candidate_count >= 24:
        y, x = np.nonzero(candidate)
        score = relative_contrast[y, x] * np.maximum(magnitude[y, x], 1e-5)
        keep = min(768, len(score))
        selected = np.argsort(score)[-keep:]
        y = y[selected]
        x = x[selected]
        nx = normal_x[y, x]
        ny = normal_y[y, x]
        profile_source = cv2.GaussianBlur(luma, (0, 0), 0.8)
        offsets = np.arange(-12.0, 13.0, dtype=np.float32)
        profiles = []
        for offset in offsets:
            sample_x = (x.astype(np.float32) + offset * nx).reshape(-1, 1)
            sample_y = (y.astype(np.float32) + offset * ny).reshape(-1, 1)
            profiles.append(
                cv2.remap(
                    profile_source,
                    sample_x,
                    sample_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101,
                ).reshape(-1)
            )
        profile = np.stack(profiles, axis=1)
        endpoint_delta = profile[:, -1] - profile[:, 0]
        valid_profile = endpoint_delta >= 0.055
        if np.any(valid_profile):
            normalized = (
                profile[valid_profile] - profile[valid_profile, :1]
            ) / np.maximum(endpoint_delta[valid_profile, None], 1e-4)
            monotonic = np.maximum.accumulate(normalized, axis=1)
            low_index = np.argmax(monotonic >= 0.20, axis=1)
            high_index = np.argmax(monotonic >= 0.80, axis=1)
            valid_width = high_index > low_index
            if np.any(valid_width):
                widths = (high_index[valid_width] - low_index[valid_width]).astype(
                    np.float32
                )
                edge_width_px = float(np.median(widths) * scale)

    return {
        "method": "blue_sky_hard_contrast_and_edge_softness_v2",
        "probability": probability,
        "confidence": confidence,
        "candidate_fraction": candidate_fraction,
        "candidate_count": candidate_count,
        "median_relative_contrast": median_relative_contrast,
        "strong_relative_contrast": strong_relative_contrast,
        "generic_edge_coverage_score": edge_coverage_score,
        "generic_edge_contrast_score": edge_contrast_score,
        "connected_blue_sky_fraction": blue_sky_fraction,
        "median_blue_minus_red": median_blue_delta,
        "hard_illumination_contrast_probability": hard_contrast_probability,
        "global_luma_p95_minus_p5": luma_range,
        "edge_width_px": edge_width_px,
        "analysis_resolution": [width, height],
    }


def direct_sun_visibility_from_probability(probability: float) -> float:
    normalized = (
        float(probability) - DIRECT_SUN_FADE_START
    ) / (DIRECT_SUN_FADE_END - DIRECT_SUN_FADE_START)
    return smoothstep01(normalized)


def sun_angle_from_shadow_edges(edge_widths_px: Sequence[float]) -> float:
    finite = np.asarray(
        [float(value) for value in edge_widths_px if math.isfinite(float(value))],
        dtype=np.float32,
    )
    if not len(finite):
        return 0.80
    edge_width = float(np.median(finite))
    angle = 0.50 + 0.12 * max(0.0, edge_width - 2.0)
    return float(
        np.clip(
            angle,
            CLEAR_SUN_ANGLE_MIN_DEGREES,
            CLEAR_SUN_ANGLE_MAX_DEGREES,
        )
    )


def estimate_multiview_lighting(views: Sequence[dict]) -> dict:
    tile_candidates = []
    sky_samples = []
    image_medians = []
    sky_counts = {}
    shadow_evidence_by_camera = {}
    for view in views:
        image = view["image"]
        shadow_evidence_by_camera[view["camera"]] = estimate_cast_shadow_evidence(
            image
        )
        rgb = image.astype(np.float32) / 255.0
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        image_medians.append(float(np.median(luma)))
        sky = connected_sky_mask(image)
        sky_counts[view["camera"]] = int(np.count_nonzero(sky))
        if np.any(sky):
            sky_samples.append(rgb[sky])
        height, width = luma.shape
        upper_height = int(0.58 * height)
        for row in range(3):
            for column in range(12):
                y0, y1 = row * upper_height // 3, (row + 1) * upper_height // 3
                x0, x1 = column * width // 12, (column + 1) * width // 12
                tile_sky = sky[y0:y1, x0:x1]
                if np.mean(tile_sky) < 0.28:
                    continue
                values = luma[y0:y1, x0:x1][tile_sky]
                score = float(np.percentile(values, 92))
                tile_candidates.append((score, view, (x0 + x1) * 0.5, (y0 + y1) * 0.5))

    if not tile_candidates or not sky_samples:
        first = views[0]
        fallback = estimate_lighting(
            first["image"], first["calibration"], first["camera_to_actor"]
        )
        shadow_probabilities = sorted(
            (
                float(evidence["probability"])
                for evidence in shadow_evidence_by_camera.values()
            ),
            reverse=True,
        )
        shadow_consensus = (
            shadow_probabilities[1]
            if len(shadow_probabilities) >= 2
            else shadow_probabilities[0]
            if shadow_probabilities
            else 0.0
        )
        evidence_confidences = [
            float(evidence["confidence"])
            for evidence in shadow_evidence_by_camera.values()
        ]
        directionality = float(
            fallback.get("image_statistics", {}).get("directionality", 0.0)
        )
        hard_probabilities = sorted(
            (
                float(evidence["hard_illumination_contrast_probability"])
                for evidence in shadow_evidence_by_camera.values()
            ),
            reverse=True,
        )
        hard_consensus = (
            hard_probabilities[1]
            if len(hard_probabilities) >= 2
            else hard_probabilities[0]
            if hard_probabilities
            else 0.0
        )
        directionality_reliability = smoothstep01(
            (directionality - 0.35) / 0.35
        )
        global_direct_probability = float(
            np.clip(
                max(
                    shadow_consensus,
                    hard_consensus * directionality_reliability,
                ),
                0.0,
                1.0,
            )
        )
        direct_sun_visibility = direct_sun_visibility_from_probability(
            global_direct_probability
        )
        ungated_sun_energy = float(fallback["sun_energy"])
        edge_widths = [
            float(evidence["edge_width_px"])
            for evidence in shadow_evidence_by_camera.values()
            if evidence.get("edge_width_px") is not None
            and float(evidence["probability"]) >= 0.35
        ]
        fallback.update(
            {
                "method": "multiview_cast_shadow_fallback_no_connected_sky",
                "sun_energy": float(ungated_sun_energy * direct_sun_visibility),
                "ungated_sun_energy": ungated_sun_energy,
                "sun_angle_degrees": sun_angle_from_shadow_edges(edge_widths),
                "global_direct_probability": global_direct_probability,
                "global_direct_confidence": float(
                    np.clip(
                        0.75
                        * (
                            np.median(evidence_confidences)
                            if evidence_confidences
                            else 0.0
                        )
                        + 0.25 * min(1.0, len(shadow_probabilities) / 2.0),
                        0.0,
                        1.0,
                    )
                ),
                "direct_sun_visibility": direct_sun_visibility,
                "direct_sun_evidence_method": (
                    "second_view_blue_sky_or_hard_contrast_consensus_v2"
                ),
            }
        )
        fallback["sky_pixel_counts"] = sky_counts
        fallback["image_statistics"].update(
            {
                "cast_shadow_evidence_by_camera": shadow_evidence_by_camera,
                "cast_shadow_consensus": shadow_consensus,
                "hard_contrast_consensus": hard_consensus,
                "directionality_reliability": directionality_reliability,
                "measured_shadow_edge_widths_px": edge_widths,
            }
        )
        return fallback

    tile_candidates.sort(key=lambda item: item[0], reverse=True)
    brightest, best_view, u, v = tile_candidates[0]
    tile_median = float(np.median([item[0] for item in tile_candidates]))
    directionality = float(np.clip((brightest - tile_median) / 0.28, 0.0, 1.0))
    intrinsic = np.asarray(best_view["calibration"]["cam_intrinsic"], dtype=np.float64)
    ray_camera = np.linalg.inv(intrinsic) @ np.array([u, v, 1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)
    direction_to_sky = best_view["camera_to_actor"][:3, :3] @ ray_camera
    direction_to_sky /= np.linalg.norm(direction_to_sky)
    sun_ray = -direction_to_sky
    if sun_ray[2] > -0.12:
        sun_ray[2] = -0.55
        sun_ray /= np.linalg.norm(sun_ray)

    combined_sky = np.concatenate(sky_samples, axis=0)
    sky_color = np.percentile(combined_sky, 60, axis=0)
    sky_color /= max(float(np.max(sky_color)), 1e-4)
    sky_color = np.clip(sky_color, 0.38, 1.0)
    median_luma = float(np.median(image_medians))
    exposure = float(np.clip(math.log2((median_luma + 0.08) / 0.48), -1.5, 1.0))
    shadow_probabilities = sorted(
        (
            float(evidence["probability"])
            for evidence in shadow_evidence_by_camera.values()
        ),
        reverse=True,
    )
    if len(shadow_probabilities) >= 2:
        # The second-strongest view prevents one high-contrast facade or road
        # marking from declaring the whole token sunny.
        shadow_consensus = shadow_probabilities[1]
    elif shadow_probabilities:
        shadow_consensus = shadow_probabilities[0]
    else:
        shadow_consensus = 0.0
    evidence_confidences = [
        float(evidence["confidence"])
        for evidence in shadow_evidence_by_camera.values()
    ]
    shadow_consensus_confidence = float(
        np.median(evidence_confidences) if evidence_confidences else 0.0
    )
    hard_probabilities = sorted(
        (
            float(evidence["hard_illumination_contrast_probability"])
            for evidence in shadow_evidence_by_camera.values()
        ),
        reverse=True,
    )
    if len(hard_probabilities) >= 2:
        hard_consensus = hard_probabilities[1]
    elif hard_probabilities:
        hard_consensus = hard_probabilities[0]
    else:
        hard_consensus = 0.0
    directionality_reliability = smoothstep01(
        (directionality - 0.35) / 0.35
    )
    global_direct_probability = float(
        np.clip(
            max(
                shadow_consensus,
                hard_consensus * directionality_reliability,
            ),
            0.0,
            1.0,
        )
    )
    global_direct_confidence = float(
        np.clip(
            0.75 * shadow_consensus_confidence
            + 0.25 * min(1.0, len(shadow_probabilities) / 2.0),
            0.0,
            1.0,
        )
    )
    direct_sun_visibility = direct_sun_visibility_from_probability(
        global_direct_probability
    )
    edge_widths = [
        float(evidence["edge_width_px"])
        for evidence in shadow_evidence_by_camera.values()
        if evidence.get("edge_width_px") is not None
        and float(evidence["probability"]) >= 0.35
    ]
    sun_angle_degrees = sun_angle_from_shadow_edges(edge_widths)
    ungated_sun_energy = float(0.30 + 2.0 * directionality)
    return {
        "method": "multiview_connected_sky_and_hard_contrast_consensus",
        "world_color": sky_color.tolist(),
        "world_strength": float(np.clip(0.42 + median_luma, 0.42, 1.30)),
        "sun_ray_direction": sun_ray.tolist(),
        "sun_energy": float(ungated_sun_energy * direct_sun_visibility),
        "ungated_sun_energy": ungated_sun_energy,
        "sun_angle_degrees": sun_angle_degrees,
        "global_direct_probability": global_direct_probability,
        "global_direct_confidence": global_direct_confidence,
        "direct_sun_visibility": direct_sun_visibility,
        "direct_sun_evidence_method": (
            "second_view_blue_sky_or_hard_contrast_consensus_v2"
        ),
        "exposure": exposure,
        "image_statistics": {
            "median_luma_across_cameras": median_luma,
            "brightest_sky_tile": brightest,
            "median_sky_tile": tile_median,
            "directionality": directionality,
            "brightest_camera": best_view["camera"],
            "brightest_tile_center_px": [u, v],
            "sky_pixel_counts": sky_counts,
            "cast_shadow_evidence_by_camera": shadow_evidence_by_camera,
            "cast_shadow_consensus": shadow_consensus,
            "cast_shadow_consensus_confidence": shadow_consensus_confidence,
            "hard_contrast_consensus": hard_consensus,
            "directionality_reliability": directionality_reliability,
            "measured_shadow_edge_widths_px": edge_widths,
        },
    }


def vector2(value: Any) -> List[float]:
    values = np.asarray(value).tolist()
    return [float(values[0]), float(values[1])]


def pose3(value: Any) -> List[float]:
    values = np.asarray(value).tolist()
    return [float(values[0]), float(values[1]), float(values[2])]


def annotation_context(loader, token: str) -> dict:
    agent_input = loader.get_agent_input_from_token(token)
    return {
        "velocity": [vector2(status.ego_velocity) for status in agent_input.ego_statuses],
        "acceleration": [vector2(status.ego_acceleration) for status in agent_input.ego_statuses],
        "past_trajectory": [pose3(status.ego_pose) for status in agent_input.ego_statuses],
        "navigation_command": jsonable(agent_input.ego_statuses[-1].driving_command),
    }


def annotation_sample(token: str, context: dict, output_paths: dict) -> dict:
    return {
        "token": token,
        "front_camera": [str(output_paths[(index, "CAM_F0")]) for index in range(4)],
        "front_left_camera": [str(output_paths[(index, "CAM_L0")]) for index in range(4)],
        "front_right_camera": [str(output_paths[(index, "CAM_R0")]) for index in range(4)],
        **context,
    }


def save_png(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, compress_level=3)


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=jsonable))
    os.replace(temporary, path)


def cache_matches(
    raw_path: Path,
    info_path: Path,
    cache_key: str,
    required_paths: Sequence[Path] = (),
) -> bool:
    if (
        not raw_path.is_file()
        or not info_path.is_file()
        or any(not path.is_file() for path in required_paths)
    ):
        return False
    try:
        return json.loads(info_path.read_text()).get("cache_key") == cache_key
    except Exception:
        return False


def render_manifest_passes(manifest: dict, field: str) -> Tuple[str, ...]:
    """Read the explicit pass set of a base render manifest."""
    if field not in {"required_passes", "produced_passes"}:
        raise ValueError(f"Unsupported render pass field: {field}")
    render_mode = str(manifest.get("render_mode", ""))
    if render_mode != "base":
        raise ValueError(f"Unsupported render_mode for pass manifest: {render_mode!r}")
    allowed = BASE_RENDER_FULL_PASSES
    value = manifest.get(field)
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or len(value) != len(set(value))
        or any(item not in allowed for item in value)
    ):
        raise ValueError(f"Invalid {field} for {render_mode}: {value!r}")
    return tuple(value)


def required_render_paths(
    manifest: dict,
    paths_by_pass: dict[str, Path],
) -> Tuple[Path, ...]:
    """Map a manifest's required non-primary passes to cache artifacts."""
    required = render_manifest_passes(manifest, "required_passes")
    return tuple(paths_by_pass[name] for name in required if name in paths_by_pass)


def _pass_aware_render_manifests_match(existing: dict, current: dict) -> bool:
    """A cached base render whose passes cover the current request."""
    if existing.get("render_mode") != "base" or current.get("render_mode") != "base":
        return False
    if not existing.get("render_pipeline_version") == current.get("render_pipeline_version") == RENDER_PIPELINE_VERSION:
        return False
    if (current.get("render_pass_policy_version") != BASE_RENDER_PASS_POLICY_VERSION
            or existing.get("render_pass_policy_version") != BASE_RENDER_PASS_POLICY_VERSION):
        return False
    try:
        required = set(render_manifest_passes(current, "required_passes"))
        produced = set(render_manifest_passes(existing, "produced_passes"))
    except ValueError:
        return False
    return required <= produced


def compatible_render_cache_key(
    manifest_path: Path,
    manifest: dict,
    excluded_fields: Sequence[str],
    *,
    reuse_existing: bool,
) -> str:
    """Preserve a valid cross-node render key for numerically equal inputs."""
    excluded = {*excluded_fields, "cache_key"}
    if reuse_existing and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            existing_key = existing.get("cache_key")
            pass_aware_match = _pass_aware_render_manifests_match(
                existing,
                manifest,
            )
            comparison_excluded = set(excluded)
            if pass_aware_match:
                comparison_excluded.update(
                    {
                        "render_pipeline_version",
                        "render_pass_policy_version",
                        "required_passes",
                        "produced_passes",
                    }
                )
            existing_payload = {
                key: value
                for key, value in existing.items()
                if key not in comparison_excluded
            }
            current_payload = {
                key: value
                for key, value in manifest.items()
                if key not in comparison_excluded
            }
            if (
                isinstance(existing_key, str)
                and existing_key
                and (
                    pass_aware_match
                    or existing.get("render_mode") != "base"
                )
                and preview_placement_mismatch(
                    existing_payload,
                    current_payload,
                    path="render_manifest",
                )
                is None
            ):
                if pass_aware_match:
                    # Record what the cache actually contains. The current
                    # required set remains label-specific and may be smaller.
                    manifest["produced_passes"] = list(
                        render_manifest_passes(existing, "produced_passes")
                    )
                return existing_key
    current_payload = {
        key: value for key, value in manifest.items() if key not in excluded
    }
    return canonical_hash(current_payload)


def visible_cuda_devices() -> List[str]:
    """Return the GPU identifiers assigned to this process by Slurm."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return []
    devices = []
    for item in value.split(","):
        device = item.strip()
        if device and device != "-1" and device not in devices:
            devices.append(device)
    return devices


def render_worker_devices(device_mode: str, worker_count: int) -> List[Optional[str]]:
    """Assign at most one visible GPU to each concurrent Blender worker."""
    if device_mode == "cpu":
        return [None] * worker_count
    visible = visible_cuda_devices()
    if worker_count > 1 and len(visible) < worker_count:
        visible_value = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            f"Requested {worker_count} concurrent GPU render workers, but "
            f"CUDA_VISIBLE_DEVICES={visible_value!r} exposes only {len(visible)} device(s)"
        )
    if visible:
        return [visible[index] for index in range(worker_count)]
    # Preserve existing single-worker behavior outside a Slurm GPU allocation.
    return [None]


def allocated_cpu_count() -> int:
    """Return CPUs granted to this task, not the compute node's total CPUs."""
    for variable in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = 0
    return max(1, affinity_count or os.cpu_count() or 1)


def configure_finalization_native_threads(finalize_workers: int) -> int:
    """Divide the actual Slurm CPU allocation across token workers."""
    allocated_cpus = allocated_cpu_count()
    native_threads = max(1, allocated_cpus // max(1, finalize_workers))
    # OpenCV otherwise sees the whole host (not just the Slurm allocation) and
    # can start dozens of native threads from every Python finalizer.
    cv2.setNumThreads(native_threads)
    return native_threads


def blender_worker_environment(
    gpu_device: Optional[str],
    native_threads: int,
) -> dict:
    environment = os.environ.copy()
    if gpu_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_device
    thread_count = str(max(1, native_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = thread_count
    return environment


def run_blender_batch(
    args: argparse.Namespace,
    renderer: Path,
    batch: dict,
    gpu_device: Optional[str],
    native_threads: int,
) -> dict:
    command = [
        str(args.blender), "-b", batch["asset_path"],
        "--python", str(renderer), "--",
        "--jobs", str(batch["job_path"]), "--device", args.device,
    ]
    environment = blender_worker_environment(gpu_device, native_threads)
    with batch["log_path"].open("w") as log:
        log.write(
            f"RENDER_WORKER_GPU={gpu_device if gpu_device is not None else 'inherited'} "
            f"NATIVE_THREADS={native_threads}\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    return {
        "batch_index": batch["batch_index"],
        "log_path": batch["log_path"],
        "returncode": completed.returncode,
        "gpu_device": gpu_device,
    }


def run_blender_batches(
    args: argparse.Namespace,
    run_dir: Path,
    jobs: List[dict],
    on_batches_submitted: Optional[Callable[[], None]] = None,
    on_batch_complete: Optional[Callable[[dict], None]] = None,
) -> List[dict]:
    if not jobs:
        if on_batches_submitted is not None:
            on_batches_submitted()
        return []
    renderer = args.asset_root / "scripts/render_batch.py"
    temporary_jobs = tempfile.TemporaryDirectory(prefix="navsim-render-jobs-")
    job_dir = Path(temporary_jobs.name)
    log_dir = run_dir / "render_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs_by_asset: Dict[str, List[dict]] = {}
    for job in jobs:
        jobs_by_asset.setdefault(job["asset_path"], []).append(job)
    batches = []
    for asset_path, asset_jobs in jobs_by_asset.items():
        for start in range(0, len(asset_jobs), args.render_batch_size):
            batch = asset_jobs[start : start + args.render_batch_size]
            batch_index = len(batches)
            job_path = job_dir / f"batch_{batch_index:05d}.jsonl"
            job_path.write_text("".join(json.dumps(job, default=jsonable) + "\n" for job in batch))
            log_path = log_dir / f"batch_{batch_index:05d}.log"
            batches.append(
                {
                    "batch_index": batch_index,
                    "asset_path": asset_path,
                    "job_path": job_path,
                    "log_path": log_path,
                    "job_ids": [job["job_id"] for job in batch],
                }
            )

    worker_count = min(args.render_workers, len(batches))
    gpu_devices = render_worker_devices(args.device, worker_count)
    allocated_cpus = allocated_cpu_count()
    native_threads = max(1, allocated_cpus // worker_count)
    print(
        f"Running {len(batches)} Blender batch(es) with {worker_count} render worker(s); "
        f"GPU slots={gpu_devices}, native threads per worker={native_threads}"
    )

    available_devices: Queue[Optional[str]] = Queue()
    for gpu_device in gpu_devices:
        available_devices.put(gpu_device)

    def run_with_available_device(batch: dict) -> dict:
        gpu_device = available_devices.get()
        try:
            return run_blender_batch(
                args,
                renderer,
                batch,
                gpu_device,
                native_threads,
            )
        finally:
            available_devices.put(gpu_device)

    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_with_available_device, batch): batch
            for batch in batches
        }
        # Every render is now queued before the bounded finalization pool is
        # fed, so callbacks cannot delay a Blender worker's next batch.
        if on_batches_submitted is not None:
            on_batches_submitted()
        for future in as_completed(futures):
            batch = futures[future]
            result = future.result()
            result["job_ids"] = list(batch["job_ids"])
            results.append(result)
            if result["returncode"] != 0:
                print(
                    f"WARNING: Blender batch {result['batch_index']} returned "
                    f"{result['returncode']}; see {result['log_path']}",
                    file=sys.stderr,
                )
            if on_batch_complete is not None:
                on_batch_complete(result)
    temporary_jobs.cleanup()
    return results


def run_depth_batch(args: argparse.Namespace, run_dir: Path, prepared_tokens: Sequence[dict]):
    if not args.depth_python.is_file():
        raise RuntimeError(f"Depth inference Python is missing: {args.depth_python}")
    requests_by_source: Dict[Path, List[Path]] = {}
    records_by_source: Dict[Path, List[dict]] = {}
    for prepared in prepared_tokens:
        for record in prepared["view_records"]:
            if not record["projected_visible"]:
                continue
            source = Path(record["source_path"]).resolve()
            output = Path(record["relative_depth_path"]).resolve()
            outputs = requests_by_source.setdefault(source, [])
            if output not in outputs:
                outputs.append(output)
            records_by_source.setdefault(source, []).append(record)
    if not requests_by_source:
        return

    log_path = run_dir / "depth_inference.log"
    jobs = []
    cache_records = []
    cache_hits = 0
    for source, output_paths in requests_by_source.items():
        source_sha256 = file_sha256(source)
        for record in records_by_source[source]:
            record["source_sha256"] = source_sha256
        cache_path = depth_prediction_cache_path(args, source, source_sha256)
        cache_valid = depth_prediction_matches(
            cache_path,
            source_sha256,
        )
        if cache_valid:
            cache_hits += 1
        else:
            jobs.append(
                {
                    "input_path": str(source),
                    "output_path": str(cache_path),
                    "source_sha256": source_sha256,
                    "cache_version": DEPTH_PREDICTION_CACHE_VERSION,
                }
            )
        cache_records.append((source_sha256, cache_path, output_paths))

    print(
        f"Depth cache: {depth_prediction_cache_root(args)} "
        f"({len(requests_by_source)} unique sources, {cache_hits} hits, "
        f"{len(jobs)} to infer)"
    )
    if jobs:
        with tempfile.TemporaryDirectory(prefix="navsim-depth-jobs-") as job_dir:
            jobs_path = Path(job_dir) / "depth_jobs.jsonl"
            jobs_path.write_text(
                "".join(json.dumps(job) + "\n" for job in jobs)
            )
            command = [
                str(args.depth_python), str(HERE.parents[1] / "estimate_depth_anything.py"),
                "--jobs", str(jobs_path), "--local-files-only",
            ]
            environment = os.environ.copy()
            environment["HF_HOME"] = str(args.depth_model_cache.resolve())
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
        if completed.returncode != 0:
            raise RuntimeError(f"Depth inference failed; see {log_path}")
    else:
        log_path.write_text(
            "DEPTH_BATCH_SUMMARY="
            + json.dumps(
                {
                    "completed": 0,
                    "cached_unique_sources": cache_hits,
                    "device": "not_started",
                }
            )
            + "\n"
        )

    for source_sha256, cache_path, output_paths in cache_records:
        if not depth_prediction_matches(
            cache_path,
            source_sha256,
        ):
            raise RuntimeError(
                f"Depth inference did not produce a valid cache entry: {cache_path}"
            )
        for output_path in output_paths:
            materialize_depth_prediction(
                cache_path,
                output_path,
                source_sha256,
            )


def robust_affine_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(x) & np.isfinite(y)
    coefficients = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(8):
        if np.count_nonzero(keep) < 80:
            break
        design = np.column_stack([x[keep], np.ones(np.count_nonzero(keep))])
        coefficients = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (coefficients[0] * x + coefficients[1])
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center)) + 1e-9
        threshold = max(3.0 * 1.4826 * mad, float(np.percentile(np.abs(residual[keep]), 55)))
        updated = np.isfinite(residual) & (np.abs(residual - center) < threshold)
        if np.array_equal(updated, keep):
            break
        keep = updated
    return coefficients, keep


def align_relative_depth_to_lidar(
    relative_depth: np.ndarray,
    lidar_xyz: np.ndarray,
    calibration: dict,
    required_depth_m: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """Align Depth-Anything inverse-depth predictions to metric camera depth.

    Depth Anything predicts an inverse-depth-like quantity.  A direct affine
    depth fit can have a slightly smaller LiDAR residual while mapping every
    image pixel in front of a pedestrian (the false full-occlusion failure).
    Keep the direct fit only as a diagnostic and validate the physically
    appropriate inverse-depth model before it is allowed to affect pixels.
    """
    projected, camera_depth = geometry.project_distorted(lidar_xyz, calibration)
    height, width = relative_depth.shape
    valid = (
        np.isfinite(projected).all(axis=1)
        & np.isfinite(camera_depth)
        & (camera_depth > 1.0) & (camera_depth < 80.0)
        & (projected[:, 0] >= 2) & (projected[:, 0] < width - 2)
        & (projected[:, 1] >= 2) & (projected[:, 1] < height - 2)
    )
    projected = projected[valid]
    camera_depth = camera_depth[valid]
    pixels = np.rint(projected).astype(np.int32)
    linear = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((camera_depth, linear))
    sorted_linear = linear[order]
    first = np.concatenate([[True], sorted_linear[1:] != sorted_linear[:-1]])
    selected = order[first]
    pixels = pixels[selected]
    camera_depth = camera_depth[selected]
    relative_samples = relative_depth[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    finite = np.isfinite(relative_samples) & (relative_samples > 0)
    relative_samples = relative_samples[finite]
    camera_depth = camera_depth[finite]
    if len(camera_depth) < 200:
        raise TokenFailure("depth_alignment_failed", f"only {len(camera_depth)} depth correspondences")

    inverse_coefficients, inverse_keep = robust_affine_fit(relative_samples, 1.0 / camera_depth)
    inverse_prediction = inverse_coefficients[0] * relative_samples + inverse_coefficients[1]
    inverse_metric = 1.0 / np.maximum(inverse_prediction, 1.0 / 200.0)
    inverse_error = float(np.median(np.abs(inverse_metric[inverse_keep] - camera_depth[inverse_keep])))

    direct_coefficients, direct_keep = robust_affine_fit(relative_samples, camera_depth)
    direct_metric = direct_coefficients[0] * relative_samples + direct_coefficients[1]
    direct_error = float(np.median(np.abs(direct_metric[direct_keep] - camera_depth[direct_keep])))

    inlier_count = int(np.count_nonzero(inverse_keep))
    slope = float(inverse_coefficients[0])
    full_inverse = slope * relative_depth.astype(np.float64) + float(inverse_coefficients[1])
    valid_inverse = np.isfinite(full_inverse) & (full_inverse > 1.0 / 200.0)
    valid_fraction = float(np.mean(valid_inverse))
    maximum_allowed_error = max(2.5, 0.15 * float(np.median(camera_depth[inverse_keep])))
    if not np.isfinite(slope) or slope <= 0.0:
        raise TokenFailure("depth_alignment_failed", f"non-positive inverse-depth slope {slope:.6g}")
    if inlier_count < 80 or not np.isfinite(inverse_error) or inverse_error > maximum_allowed_error:
        raise TokenFailure(
            "depth_alignment_failed",
            f"unreliable inverse-depth fit: inliers={inlier_count}, median_error={inverse_error:.3f} m",
        )
    # Sky/very-far predictions can legitimately extrapolate past zero inverse
    # depth.  They are clamped to the far plane and are safe for occlusion; a
    # majority-invalid mapping, however, indicates a failed fit.
    if valid_fraction < 0.50:
        raise TokenFailure(
            "depth_alignment_failed",
            f"inverse-depth mapping is invalid for {100.0 * (1.0 - valid_fraction):.1f}% of pixels",
        )

    metric = 1.0 / np.maximum(full_inverse, 1.0 / 200.0)
    metric = np.clip(metric, 0.5, 200.0).astype(np.float32)
    valid_metric = metric[valid_inverse]
    mapped_valid_p99 = float(np.percentile(valid_metric, 99))
    if required_depth_m is not None and mapped_valid_p99 < float(required_depth_m) - 1.0:
        raise TokenFailure(
            "depth_alignment_failed",
            f"aligned valid-depth range ends at {mapped_valid_p99:.2f} m "
            f"before actor depth {required_depth_m:.2f} m",
        )
    return np.clip(metric, 0.5, 200.0).astype(np.float32), {
        "method": "affine_inverse_depth",
        "coefficients": inverse_coefficients.tolist(),
        "correspondence_count": int(len(camera_depth)),
        "inlier_count": inlier_count,
        "median_lidar_error_m": inverse_error,
        "inverse_candidate_error_m": inverse_error,
        "direct_candidate_error_m": direct_error,
        "direct_candidate_coefficients": direct_coefficients.tolist(),
        "direct_candidate_rejected": True,
        "valid_mapping_fraction": valid_fraction,
        "mapped_valid_depth_p99_m": mapped_valid_p99,
        "required_actor_depth_m": float(required_depth_m) if required_depth_m is not None else None,
    }


def lidar_foreground_support(
    lidar_xyz: np.ndarray,
    calibration: dict,
    ground_plane: np.ndarray,
    actor_alpha: np.ndarray,
    actor_depth_m: float,
) -> Tuple[np.ndarray, dict]:
    """Rasterize closer, above-ground LiDAR returns as occluder evidence."""
    height, width = actor_alpha.shape
    projected, camera_depth = geometry.project_distorted(lidar_xyz, calibration)
    point_ground = (
        ground_plane[0] * lidar_xyz[:, 0]
        + ground_plane[1] * lidar_xyz[:, 1]
        + ground_plane[2]
    )
    height_above_ground = lidar_xyz[:, 2] - point_ground
    margin_m = max(0.45, 0.035 * actor_depth_m)
    valid = (
        np.isfinite(projected).all(axis=1)
        & np.isfinite(camera_depth)
        & (camera_depth > 0.5)
        & (camera_depth < actor_depth_m - margin_m)
        & (height_above_ground > 0.12)
        & (height_above_ground < 3.5)
        & (projected[:, 0] >= 0) & (projected[:, 0] < width)
        & (projected[:, 1] >= 0) & (projected[:, 1] < height)
    )
    pixels = np.rint(projected[valid]).astype(np.int32)
    support = np.zeros((height, width), dtype=np.uint8)
    if len(pixels):
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        support[pixels[:, 1], pixels[:, 0]] = 1
        support = cv2.dilate(
            support,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
    actor_mask = actor_alpha > 0
    return support.astype(bool), {
        "closer_above_ground_lidar_points": int(np.count_nonzero(valid)),
        "supported_actor_pixels": int(np.count_nonzero((support > 0) & actor_mask)),
        "foreground_margin_m": float(margin_m),
    }


def dense_depth_occlusion(
    actor_alpha: np.ndarray,
    source_metric_depth: np.ndarray,
    actor_depth_m: float,
    road_metric_depth: np.ndarray,
    actor_metric_depth: Optional[np.ndarray] = None,
    foreground_support: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    actor_mask = actor_alpha > 0
    if actor_metric_depth is None:
        actor_depth = np.full(source_metric_depth.shape, actor_depth_m, dtype=np.float32)
        actor_depth_method = "root_fallback"
    else:
        actor_depth = np.asarray(actor_metric_depth, dtype=np.float32).copy()
        valid_actor_depth = (
            np.isfinite(actor_depth) & (actor_depth > 0.5) & (actor_depth < ACTOR_DEPTH_ENCODING_MAX_M - 0.5)
        )
        actor_depth[~valid_actor_depth] = actor_depth_m
        actor_depth_method = "blender_z_pass_with_root_fallback"
    margin_m = np.maximum(0.55, 0.045 * actor_depth)
    transition_m = np.maximum(0.20, 0.015 * actor_depth)
    delta = actor_depth - margin_m - source_metric_depth
    strength = np.clip(delta / transition_m, 0.0, 1.0)
    strength[~actor_mask] = 0.0
    road_tolerance_m = max(0.8, 0.06 * actor_depth_m)
    road_like = (
        np.isfinite(road_metric_depth)
        & (np.abs(source_metric_depth - road_metric_depth) < road_tolerance_m)
    )
    strength[road_like] = 0.0

    # Monocular depth often follows a curb/road transition imperfectly exactly
    # where shoes meet the support surface.  Give the lower silhouette a wider
    # road-consistency guard, unless closer LiDAR explicitly supports an object.
    lower_actor = np.zeros(actor_mask.shape, dtype=bool)
    actor_yx = np.argwhere(actor_mask)
    if len(actor_yx):
        y0 = int(actor_yx[:, 0].min())
        y1 = int(actor_yx[:, 0].max()) + 1
        lower_actor[max(y0, int(round(y0 + 0.68 * (y1 - y0)))):y1] = actor_mask[
            max(y0, int(round(y0 + 0.68 * (y1 - y0)))):y1
        ]
    support_tolerance_m = max(3.75, 0.18 * actor_depth_m)
    support_surface_like = (
        lower_actor
        & np.isfinite(road_metric_depth)
        & (np.abs(source_metric_depth - road_metric_depth) < support_tolerance_m)
        & (road_metric_depth >= actor_depth - support_tolerance_m)
    )
    if foreground_support is None:
        foreground_support = np.zeros(actor_mask.shape, dtype=bool)
    else:
        foreground_support = np.asarray(foreground_support, dtype=bool)
    protected_support_surface = support_surface_like & ~foreground_support
    strength[protected_support_surface] = 0.0

    binary = (strength > 0.5).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    strength[binary == 0] = 0.0
    actor_pixel_count = int(np.count_nonzero(actor_mask))
    requested_occluded_pixels = int(np.count_nonzero(strength > 0.5))
    requested_fraction = requested_occluded_pixels / max(actor_pixel_count, 1)
    supported_actor_pixels = int(np.count_nonzero(foreground_support & actor_mask))
    supported_fraction = supported_actor_pixels / max(actor_pixel_count, 1)
    fail_open = requested_fraction > 0.85 and supported_fraction < 0.01
    if fail_open:
        strength.fill(0.0)
    return strength.astype(np.float32), {
        "actor_pixels": actor_pixel_count,
        "occluded_pixels": int(np.count_nonzero(strength > 0.5)),
        "partial_occlusion_pixels": int(np.count_nonzero(strength > 0.0)),
        "requested_occluded_pixels_before_fail_safe": requested_occluded_pixels,
        "requested_occlusion_fraction_before_fail_safe": float(requested_fraction),
        "depth_margin_m_at_root": float(max(0.55, 0.045 * actor_depth_m)),
        "transition_m_at_root": float(max(0.20, 0.015 * actor_depth_m)),
        "actor_root_depth_m": float(actor_depth_m),
        "actor_depth_method": actor_depth_method,
        "road_depth_tolerance_m": float(road_tolerance_m),
        "support_surface_tolerance_m": float(support_tolerance_m),
        "protected_road_pixels": int(np.count_nonzero(road_like & actor_mask)),
        "protected_lower_support_pixels": int(np.count_nonzero(protected_support_surface)),
        "lidar_supported_actor_pixels": supported_actor_pixels,
        "lidar_supported_actor_fraction": float(supported_fraction),
        "near_total_occlusion_fail_open": bool(fail_open),
    }


def load_shadow_catcher_amount(
    shadow_path: Path,
    manifest: dict,
    minimum_amount: float = SHADOW_MIN_AMOUNT,
) -> Tuple[np.ndarray, dict]:
    """Read the scene-linear catcher EXR and recover receiver attenuation.

    The receiver is a uniform Lambertian plane, so its robust upper luminance
    is the unoccluded reference from the same render.  This avoids both a
    second ``L_clear`` render and the old large-radius local normalization that
    erased broad cast shadows.  The untouched float pass remains on disk; this
    scalar amount is only the naive preview/virtual-shadow indicator.
    """
    catcher = cv2.imread(str(shadow_path), cv2.IMREAD_UNCHANGED)
    if catcher is None:
        raise TokenFailure("render_failed", f"unreadable shadow catcher EXR: {shadow_path}")
    if catcher.ndim != 3 or catcher.shape[2] < 3:
        raise TokenFailure(
            "render_failed",
            f"shadow catcher must contain RGB(A), got shape {catcher.shape}: {shadow_path}",
        )
    catcher = catcher.astype(np.float32)
    # OpenCV decodes color images as BGR(A).
    rgb = catcher[..., [2, 1, 0]]
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    finite = np.isfinite(luma)
    if catcher.shape[2] >= 4:
        finite &= np.isfinite(catcher[..., 3]) & (catcher[..., 3] > 1e-4)

    ideal_width, ideal_height = [int(value) for value in manifest["resolution"]]
    render_region = manifest.get("render_region")
    if render_region is None:
        raise TokenFailure(
            "render_failed",
            f"render manifest has no render_region: {shadow_path}",
        )
    region = [int(value) for value in render_region]
    if len(region) != 4:
        raise TokenFailure(
            "render_failed",
            f"render_region must have four coordinates, got {region}: {shadow_path}",
        )
    x_min, y_min, x_max, y_max = region
    if not (
        0 <= x_min < x_max <= ideal_width
        and 0 <= y_min < y_max <= ideal_height
    ):
        raise TokenFailure(
            "render_failed",
            f"render_region {region} is outside {ideal_width}x{ideal_height}: "
            f"{shadow_path}",
        )
    source_height, source_width = luma.shape
    storage_x_min, storage_y_min, storage_x_max, storage_y_max = (
        render_storage_region(manifest)
    )
    storage_width = storage_x_max - storage_x_min
    storage_height = storage_y_max - storage_y_min
    cropped_storage = (
        manifest.get("raw_render_storage") == RAW_RENDER_STORAGE_VERSION
    )
    if cropped_storage:
        # Crop-to-border files contain only the sampled rectangle. Their local
        # pixel (0, 0) corresponds to render_region's ideal-canvas origin.
        source_x_min = 0
        source_y_min = 0
        source_x_max = source_width
        source_y_max = source_height
    else:
        # Compatibility for pre-v12 full-canvas intermediates.
        scale_x = source_width / float(ideal_width)
        scale_y = source_height / float(ideal_height)
        source_x_min = int(round(x_min * scale_x))
        source_y_min = int(round(y_min * scale_y))
        source_x_max = int(round(x_max * scale_x))
        source_y_max = int(round(y_max * scale_y))
    source_x_min = int(np.clip(source_x_min, 0, source_width))
    source_x_max = int(np.clip(source_x_max, 0, source_width))
    source_y_min = int(np.clip(source_y_min, 0, source_height))
    source_y_max = int(np.clip(source_y_max, 0, source_height))
    if source_x_min >= source_x_max or source_y_min >= source_y_max:
        raise TokenFailure(
            "render_failed",
            f"render_region {region} maps to an empty EXR region: {shadow_path}",
        )

    # Blender's normalized border can leave one unrendered black source pixel
    # on a partial region boundary.
    full_region = region == [0, 0, ideal_width, ideal_height]
    region_inset = 0 if full_region else 1
    source_x_min += region_inset
    source_y_min += region_inset
    source_x_max -= region_inset
    source_y_max -= region_inset
    rendered = np.zeros(luma.shape, dtype=bool)
    if source_x_min < source_x_max and source_y_min < source_y_max:
        rendered[source_y_min:source_y_max, source_x_min:source_x_max] = True
    valid = finite & rendered
    validity_method = (
        "cropped_render_region_zero_luminance_valid_v3"
        if cropped_storage
        else "manifest_render_region_zero_luminance_valid_v2"
    )
    source_region = [source_x_min, source_y_min, source_x_max, source_y_max]

    # A zero-luminance pixel inside an explicitly rendered receiver region is a
    # legitimate fully occluded shadow pixel. Exclude it only from estimating
    # the unshadowed reference; retain it for the attenuation calculation.
    reference_valid = valid & (luma > 1e-6)
    if np.count_nonzero(reference_valid) < 100:
        raise TokenFailure(
            "render_failed",
            "shadow catcher has only "
            f"{np.count_nonzero(reference_valid)} positive reference pixels: "
            f"{shadow_path}",
        )
    unshadowed_luma = float(np.percentile(luma[reference_valid], 99.0))
    if not np.isfinite(unshadowed_luma) or unshadowed_luma <= 1e-6:
        raise TokenFailure(
            "render_failed",
            f"invalid unshadowed catcher luminance {unshadowed_luma}: {shadow_path}",
        )
    shadow_amount = np.clip(1.0 - luma / unshadowed_luma, 0.0, 1.0)
    shadow_amount[~valid] = 0.0

    if shadow_amount.shape != (storage_height, storage_width):
        shadow_amount = cv2.resize(
            shadow_amount,
            (storage_width, storage_height),
            interpolation=cv2.INTER_AREA,
        )
    # Average the oversampled float catcher before applying the numerical
    # cutoff. This avoids turning individual noisy source samples into hard
    # pinholes in the final ideal-resolution matte.
    shadow_amount[shadow_amount < minimum_amount] = 0.0
    nonzero = shadow_amount[shadow_amount > 0.0]
    return shadow_amount.astype(np.float32), {
        "method": "scene_linear_uniform_receiver_ratio_v2",
        "path": str(shadow_path),
        "source_dtype": str(catcher.dtype),
        "source_resolution": [int(catcher.shape[1]), int(catcher.shape[0])],
        "ideal_resolution": [storage_width, storage_height],
        "ideal_canvas_resolution": [ideal_width, ideal_height],
        "ideal_storage_origin": [storage_x_min, storage_y_min],
        "validity_method": validity_method,
        "render_region_ideal": [int(value) for value in render_region],
        "render_region_source": source_region,
        "render_region_boundary_inset_source_px": region_inset,
        "valid_receiver_pixel_count": int(np.count_nonzero(valid)),
        "valid_zero_luminance_pixel_count": int(
            np.count_nonzero(valid & (luma <= 1e-6))
        ),
        "unshadowed_receiver_luma_scene_linear": unshadowed_luma,
        "minimum_retained_shadow_amount": minimum_amount,
        "retained_attenuation_pixel_count": int(len(nonzero)),
        "maximum_retained_attenuation": float(nonzero.max()) if len(nonzero) else 0.0,
        "mean_nonzero_retained_attenuation": (
            float(nonzero.mean()) if len(nonzero) else 0.0
        ),
    }


def load_linear_actor_exr(path: Path, manifest: dict) -> Tuple[np.ndarray, dict]:
    """Load a scene-linear actor EXR as premultiplied RGBA.

    Blender's Render Layers image is premultiplied internally.  Keeping that
    representation through resizing and lens remapping prevents bright color
    fringes at antialiased silhouette pixels.
    """
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise TokenFailure("render_failed", f"unreadable actor lighting EXR: {path}")
    if encoded.ndim != 3 or encoded.shape[2] < 4:
        raise TokenFailure(
            "render_failed",
            f"actor lighting EXR must contain RGBA, got {encoded.shape}: {path}",
        )
    rgba = encoded[..., [2, 1, 0, 3]].astype(np.float32)
    rgba = np.nan_to_num(rgba, nan=0.0, posinf=0.0, neginf=0.0)
    rgba[..., 3] = np.clip(rgba[..., 3], 0.0, 1.0)
    canvas_width, canvas_height = [int(value) for value in manifest["resolution"]]
    ideal_width, ideal_height = render_storage_resolution(manifest)
    storage_x, storage_y, _storage_x_max, _storage_y_max = (
        render_storage_region(manifest)
    )
    source_shape = rgba.shape[:2]
    if source_shape != (ideal_height, ideal_width):
        rgba = cv2.resize(
            rgba,
            (ideal_width, ideal_height),
            interpolation=cv2.INTER_AREA,
        )
    return rgba, {
        "path": str(path),
        "encoding": "scene_linear_half_float_premultiplied_rgba",
        "source_resolution": [int(source_shape[1]), int(source_shape[0])],
        "ideal_resolution": [ideal_width, ideal_height],
        "ideal_canvas_resolution": [canvas_width, canvas_height],
        "ideal_storage_origin": [storage_x, storage_y],
    }


def distort_linear_actor_rgba(
    ideal_premultiplied_rgba: np.ndarray,
    calibration: dict,
    remap_cache: dict,
    output_size: Tuple[int, int],
    ideal_origin: Tuple[float, float],
) -> np.ndarray:
    """Apply NAVSIM lens distortion and return straight scene-linear RGBA."""
    remapped = np.stack(
        [
            render_utils.distort_scalar_map(
                ideal_premultiplied_rgba[..., channel],
                calibration,
                remap_cache,
                output_size=output_size,
                ideal_origin=ideal_origin,
                border_value=0.0,
                interpolation=cv2.INTER_LINEAR,
            )
            for channel in range(4)
        ],
        axis=2,
    )
    alpha = np.clip(remapped[..., 3:4], 0.0, 1.0)
    straight_rgb = np.divide(
        np.clip(remapped[..., :3], 0.0, None),
        np.maximum(alpha, 1e-6),
        out=np.zeros_like(remapped[..., :3]),
        where=alpha > 1e-6,
    )
    return np.concatenate([straight_rgb, alpha], axis=2).astype(np.float32)


def partition_actor_lighting(
    actor_rgba: np.ndarray,
    actor_sun_raw: np.ndarray,
    actor_sky_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Partition the finalized actor into additive Sun and sky terms.

    Camera degradation/local color matching operate on the display actor.  We
    therefore use the raw linear renders only for their per-channel relative
    contribution and partition the finalized actor radiance by that ratio.
    At full Sun visibility, the two terms add back to the exact finalized
    full-light actor.
    """
    alpha = actor_rgba[..., 3].astype(np.float32) / 255.0
    actor_linear = render_utils.srgb_to_linear(
        actor_rgba[..., :3].astype(np.float32) / 255.0
    )
    sun = np.clip(actor_sun_raw[..., :3], 0.0, None)
    sky = np.clip(actor_sky_raw[..., :3], 0.0, None)
    total = sun + sky
    fraction = np.divide(
        sun,
        np.maximum(total, 1e-6),
        out=np.zeros_like(sun),
        where=total > 1e-6,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    actor_sun = actor_linear * fraction
    actor_sky = actor_linear - actor_sun
    outside = alpha <= 0.0
    actor_sun[outside] = 0.0
    actor_sky[outside] = 0.0
    active = alpha > 0.0
    return actor_sun.astype(np.float32), actor_sky.astype(np.float32), {
        "method": "final_actor_partition_by_raw_scene_linear_ratio_v1",
        "active_pixel_count": int(np.count_nonzero(active)),
        "mean_sun_fraction": float(np.mean(fraction[active])) if np.any(active) else 0.0,
        "maximum_reconstruction_error": float(
            np.max(np.abs((actor_sun + actor_sky) - actor_linear))
        ),
    }


def compose_actor_lighting_rgba(
    actor_rgba: np.ndarray,
    actor_sun_linear: np.ndarray,
    actor_sky_linear: np.ndarray,
    sun_gain: float,
) -> np.ndarray:
    """Recompose a finalized actor from its additive sky and Sun terms."""
    actor_rgba = np.asarray(actor_rgba)
    actor_sun_linear = np.asarray(actor_sun_linear, dtype=np.float32)
    actor_sky_linear = np.asarray(actor_sky_linear, dtype=np.float32)
    expected = (*actor_rgba.shape[:2], 3)
    if actor_rgba.ndim != 3 or actor_rgba.shape[2] != 4:
        raise ValueError(f"actor RGBA must have shape HxWx4: {actor_rgba.shape}")
    if actor_sun_linear.shape != expected or actor_sky_linear.shape != expected:
        raise ValueError(
            "actor Sun/sky RGB must match actor RGBA spatial dimensions: "
            f"{actor_sun_linear.shape}, {actor_sky_linear.shape}, {actor_rgba.shape}"
        )
    if not 0.0 <= float(sun_gain) <= 1.0:
        raise ValueError(f"actor Sun gain must be in [0, 1]: {sun_gain}")
    combined_linear = np.clip(
        actor_sky_linear + float(sun_gain) * actor_sun_linear,
        0.0,
        None,
    )
    composited = actor_rgba.copy()
    composited[..., :3] = np.rint(
        render_utils.linear_to_srgb(combined_linear) * 255.0
    ).astype(np.uint8)
    composited[composited[..., 3] == 0, :3] = 0
    return composited


def compose_labeled_shadow_alpha(
    directional_shadow_amount: np.ndarray,
    ambient_occlusion_amount: np.ndarray,
    directional_gain: float,
    ambient_occlusion_gain: float,
) -> Tuple[np.ndarray, dict]:
    """Combine Vdir and Sun-off sky occlusion in linear-light attenuation."""
    directional = np.clip(
        np.nan_to_num(
            np.asarray(directional_shadow_amount, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    ambient = np.clip(
        np.nan_to_num(
            np.asarray(ambient_occlusion_amount, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    if directional.shape != ambient.shape:
        raise ValueError(
            f"directional and AO mattes must have the same shape: "
            f"{directional.shape}, {ambient.shape}"
        )
    if not 0.0 <= float(directional_gain) <= 1.0:
        raise ValueError(f"directional shadow gain must be in [0, 1]: {directional_gain}")
    if not 0.0 <= float(ambient_occlusion_gain) <= 1.0:
        raise ValueError(
            "ambient occlusion gain must be in [0, 1]: "
            f"{ambient_occlusion_gain}"
        )
    directional_attenuation = float(directional_gain) * directional
    ambient_attenuation = float(ambient_occlusion_gain) * ambient
    combined = 1.0 - (1.0 - directional_attenuation) * (1.0 - ambient_attenuation)
    combined = np.clip(combined, 0.0, 1.0)
    alpha = np.rint(combined * 255.0).astype(np.uint8)
    active = (directional_attenuation > 0.0) | (ambient_attenuation > 0.0)
    alpha[active & (alpha == 0)] = 1
    return alpha, {
        "method": "multiplicative_vdir_and_sky_occlusion_linear_v2",
        "directional_gain": float(directional_gain),
        "ambient_occlusion_gain": float(ambient_occlusion_gain),
        "directional_effective_pixel_count": int(
            np.count_nonzero(directional_attenuation > 0.0)
        ),
        "ambient_occlusion_effective_pixel_count": int(
            np.count_nonzero(ambient_attenuation > 0.0)
        ),
        "combined_effective_pixel_count": int(np.count_nonzero(alpha)),
        "maximum_effective_attenuation": float(combined.max())
        if combined.size
        else 0.0,
    }


def merge_sky_occlusion_components(
    diffuse_body_occlusion: np.ndarray,
    stance_contact_occlusion: np.ndarray,
) -> np.ndarray:
    """Merge two estimates of one Sun-off visibility field without double count."""
    diffuse = np.clip(
        np.nan_to_num(
            np.asarray(diffuse_body_occlusion, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    contact = np.clip(
        np.nan_to_num(
            np.asarray(stance_contact_occlusion, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    if diffuse.shape != contact.shape:
        raise ValueError(
            "diffuse-body and stance-contact occlusion must have the same "
            f"shape: {diffuse.shape}, {contact.shape}"
        )
    return np.maximum(diffuse, contact).astype(np.float32)


def shadow_receiver_visibility(
    road_metric_depth: np.ndarray,
    source_metric_depth: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """Build a soft receiver mask and reject recorded foreground geometry."""
    road_metric_depth = np.asarray(road_metric_depth, dtype=np.float32)
    geometric = (
        np.isfinite(road_metric_depth)
        & (road_metric_depth > 0.2)
        & (road_metric_depth <= SHADOW_RECEIVER_MAX_DEPTH_M)
    )
    visibility = geometric.astype(np.float32)
    diagnostics = {
        "method": "projected_receiver_only_v1",
        "maximum_receiver_depth_m": SHADOW_RECEIVER_MAX_DEPTH_M,
        "geometric_receiver_pixels": int(np.count_nonzero(geometric)),
    }
    if source_metric_depth is not None:
        source_metric_depth = np.asarray(source_metric_depth, dtype=np.float32)
        depth_consistency = np.zeros(road_metric_depth.shape, dtype=np.float32)
        comparable = geometric & np.isfinite(source_metric_depth)
        tolerance = np.maximum(1.0, 0.07 * road_metric_depth[comparable])
        transition = np.maximum(0.50, 0.035 * road_metric_depth[comparable])
        residual = np.abs(
            source_metric_depth[comparable] - road_metric_depth[comparable]
        )
        depth_consistency[comparable] = 1.0 - np.clip(
            (residual - tolerance) / np.maximum(transition, 1e-4), 0.0, 1.0
        )
        visibility *= depth_consistency.astype(np.float32)
        diagnostics.update(
            {
                "method": "projected_receiver_with_dense_depth_consistency_v1",
                "depth_tolerance_at_20m_m": float(max(1.0, 0.07 * 20.0)),
                "depth_transition_at_20m_m": float(max(0.50, 0.035 * 20.0)),
                "depth_consistent_receiver_pixels": int(
                    np.count_nonzero(visibility >= 0.5)
                ),
                "partially_visible_receiver_pixels": int(
                    np.count_nonzero(visibility > 0.0)
                ),
            }
        )
    return np.clip(visibility, 0.0, 1.0), diagnostics


def retain_grounded_shadow_component(
    shadow_amount: np.ndarray,
    actor_alpha: np.ndarray,
    anchor_region: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """Discard isolated catcher noise while preserving the foot-anchored cast shadow.

    This is a connectivity filter, not a blur or shadow-shape heuristic. The
    raw float EXR remains untouched. An explicit stance region can anchor
    diffuse sky occlusion; otherwise a small actor dilation tolerates the
    subpixel gap between the rendered silhouette and receiver contact.
    """
    shadow_amount = np.asarray(shadow_amount, dtype=np.float32)
    actor_mask = np.asarray(actor_alpha) > 0
    actor_yx = np.argwhere(actor_mask)
    if not len(actor_yx) or not np.any(shadow_amount > 0.0):
        return shadow_amount, {
            "method": "foot_anchored_connected_component_v1",
            "status": "empty_actor_or_shadow",
            "retained_components": 0,
        }
    actor_height = int(actor_yx[:, 0].max() - actor_yx[:, 0].min() + 1)
    if anchor_region is not None:
        anchor_values = np.asarray(anchor_region)
        if anchor_values.shape != shadow_amount.shape:
            raise ValueError(
                "shadow anchor region must share the shadow shape: "
                f"{anchor_values.shape}, {shadow_amount.shape}"
            )
        anchor = anchor_values > 0.0
        anchor_radius = 0
        anchor_method = "explicit_stance_contact_region"
    else:
        anchor_radius = max(3, int(round(0.025 * actor_height)))
        anchor = cv2.dilate(
            actor_mask.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * anchor_radius + 1, 2 * anchor_radius + 1),
            ),
            iterations=1,
        ).astype(bool)
        anchor_method = "dilated_actor_silhouette"
    binary = (shadow_amount > 0.0).astype(np.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    anchored_labels = np.unique(labels[anchor & (binary > 0)])
    anchored_labels = anchored_labels[anchored_labels != 0]
    status = "anchored_components"
    if not len(anchored_labels) and component_count > 1:
        # Defensive fallback for a rare antialiased contact gap: retain the
        # largest physical shadow rather than every disconnected noise pixel.
        anchored_labels = np.array(
            [1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))], dtype=np.int32
        )
        status = "largest_component_fallback"
    keep = np.isin(labels, anchored_labels)
    filtered = shadow_amount.copy()
    filtered[~keep] = 0.0
    return filtered, {
        "method": "foot_anchored_connected_component_v1",
        "status": status,
        "anchor_method": anchor_method,
        "anchor_dilation_radius_px": anchor_radius,
        "candidate_components": max(0, int(component_count - 1)),
        "retained_components": int(len(anchored_labels)),
        "pixels_before": int(np.count_nonzero(binary)),
        "pixels_after": int(np.count_nonzero(filtered)),
    }


def ideal_foot_contact_region(
    render_info: dict,
    actor_alpha_ideal: np.ndarray,
    ideal_storage_origin: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, dict]:
    """Rasterize measured stance soles in the locally stored ideal rectangle."""
    geometry_info = render_info.get("foot_contact_geometry")
    if not isinstance(geometry_info, dict):
        raise TokenFailure("render_failed", "render info has no foot-contact geometry")
    feet = geometry_info.get("feet", {})
    if not isinstance(feet, dict):
        raise TokenFailure("render_failed", "render info has malformed foot geometry")
    measured = any(
        str(details.get("status", "")).startswith("measured_from_")
        for details in feet.values()
        if isinstance(details, dict)
    )
    if not measured:
        raise TokenFailure(
            "render_failed",
            "actor render has no measured ground-contact geometry",
        )

    height, width = actor_alpha_ideal.shape
    contact = np.zeros((height, width), dtype=np.uint8)
    contact_sides = []
    projected_point_count = 0
    for side, details in feet.items():
        if not isinstance(details, dict) or not details.get("contact", False):
            continue
        points = np.asarray(details.get("ideal_contact_points_px", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        points[:, 0] -= float(ideal_storage_origin[0])
        points[:, 1] -= float(ideal_storage_origin[1])
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        integer_points = np.rint(points).astype(np.int32)
        projected_point_count += len(integer_points)
        contact_sides.append(str(side))
        if len(integer_points) >= 3:
            hull = cv2.convexHull(integer_points.reshape(-1, 1, 2))
            cv2.fillConvexPoly(contact, hull, 1)
        else:
            for x, y in integer_points:
                cv2.circle(contact, (int(x), int(y)), 1, 1, thickness=-1)

    actor_y, _actor_x = np.nonzero(np.asarray(actor_alpha_ideal) > 0)
    actor_height = int(actor_y.max() - actor_y.min() + 1) if len(actor_y) else 0
    # The footprint comes from the physical sole.  A small screen-space pad
    # includes the immediately adjacent receiver where contact AO lives, but
    # remains far tighter than the former lower-18%-of-actor heuristic.
    dilation_radius = max(2, int(round(0.035 * actor_height))) if actor_height else 2
    if np.any(contact):
        contact = cv2.dilate(
            contact,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilation_radius + 1, 2 * dilation_radius + 1),
            ),
            iterations=1,
        )
    return contact.astype(np.float32), {
        "method": "evaluated_foot_contact_region_v1",
        "status": "measured_contact" if np.any(contact) else "measured_no_contact",
        "contact_sides": contact_sides,
        "projected_sole_point_count": projected_point_count,
        "contact_dilation_radius_px": dilation_radius,
        "contact_region_pixel_count": int(np.count_nonzero(contact)),
        "ideal_storage_origin": [
            float(ideal_storage_origin[0]),
            float(ideal_storage_origin[1]),
        ],
        "source_geometry": {
            "method": geometry_info.get("method"),
            "receiver_plane_z_m": geometry_info.get("receiver_plane_z_m"),
            "contact_distance_m": geometry_info.get("contact_distance_m"),
            "stance_relative_distance_m": geometry_info.get(
                "stance_relative_distance_m"
            ),
            "feet": {
                side: {
                    "contact": bool(details.get("contact", False)),
                    "status": details.get("status"),
                    "minimum_receiver_distance_m": details.get(
                        "minimum_receiver_distance_m"
                    ),
                    "projected_sole_point_count": len(
                        details.get("ideal_contact_points_px", [])
                    ),
                }
                for side, details in feet.items()
                if isinstance(details, dict)
            },
        },
    }


def overlap_receiver_matte_with_actor_edge(
    receiver_amount: np.ndarray,
    actor_alpha: np.ndarray,
    receiver_visibility: np.ndarray,
    eligible_region: np.ndarray,
    radius_px: int = RECEIVER_MATTE_ACTOR_EDGE_RADIUS_PX,
    opaque_alpha: int = RECEIVER_MATTE_OPAQUE_ALPHA,
) -> Tuple[np.ndarray, dict]:
    """Extrapolate a receiver matte beneath only the actor's soft outer edge.

    The final Porter-Duff composite still exposes the background in proportion
    to ``1 - actor_alpha`` at antialiased silhouette pixels. Clearing the
    receiver matte for every nonzero alpha therefore exposes a bright,
    unshadowed road rim around dark shoes. Values beneath the opaque/interior
    actor are irrelevant and remain zero; a receiver-capped two-pixel
    extrapolation supplies the hidden background only where it can contribute.
    """
    amount = np.clip(
        np.nan_to_num(
            np.asarray(receiver_amount, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    alpha = np.asarray(actor_alpha)
    visibility = np.clip(
        np.nan_to_num(
            np.asarray(receiver_visibility, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    if amount.shape != alpha.shape or amount.shape != visibility.shape:
        raise ValueError(
            "receiver amount, actor alpha, and receiver visibility must share "
            f"one shape: {amount.shape}, {alpha.shape}, {visibility.shape}"
        )
    eligible_values = np.asarray(eligible_region)
    if eligible_values.shape != amount.shape:
        raise ValueError(
            "eligible actor-edge region must share the receiver shape: "
            f"{eligible_values.shape}, {amount.shape}"
        )
    eligible = eligible_values > 0.0
    eligibility_method = "measured_stance_foot_region"
    if radius_px < 0:
        raise ValueError(f"actor-edge overlap radius must be non-negative: {radius_px}")
    if not 1 <= opaque_alpha <= 255:
        raise ValueError(
            f"opaque actor alpha threshold must be in [1, 255]: {opaque_alpha}"
        )

    actor = alpha > 0
    result = amount.copy()
    result[actor] = 0.0
    if not np.any(actor) or radius_px == 0 or not np.any(result > 0.0):
        return result, {
            "method": "receiver_capped_contact_edge_extrapolation_v2",
            "status": "empty_actor_radius_or_receiver_matte",
            "radius_px": int(radius_px),
            "opaque_alpha_threshold": int(opaque_alpha),
            "eligible_soft_edge_pixels": 0,
            "filled_soft_edge_pixels": 0,
            "outside_actor_changed_pixels": 0,
            "eligibility_method": eligibility_method,
        }

    distance_inside = cv2.distanceTransform(
        actor.astype(np.uint8), cv2.DIST_L2, 5
    )
    soft_outer_edge = (
        actor
        & (alpha < opaque_alpha)
        & (distance_inside <= float(radius_px) + 0.5)
        & (visibility > 0.0)
        & eligible
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius_px + 1, 2 * radius_px + 1),
    )
    extrapolated = cv2.dilate(result, kernel, iterations=1)
    extrapolated = np.minimum(extrapolated, visibility)
    result[soft_outer_edge] = extrapolated[soft_outer_edge]
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    outside = ~actor
    outside_changed = np.abs(result[outside] - amount[outside]) > 1e-7
    return result, {
        "method": "receiver_capped_contact_edge_extrapolation_v2",
        "status": "success",
        "radius_px": int(radius_px),
        "opaque_alpha_threshold": int(opaque_alpha),
        "input_nonzero_under_actor_pixels": int(
            np.count_nonzero((amount > 0.0) & actor)
        ),
        "eligible_soft_edge_pixels": int(np.count_nonzero(soft_outer_edge)),
        "filled_soft_edge_pixels": int(
            np.count_nonzero(result[soft_outer_edge] > 0.0)
        ),
        "opaque_or_interior_nonzero_pixels": int(
            np.count_nonzero((result > 0.0) & actor & ~soft_outer_edge)
        ),
        "outside_actor_changed_pixels": int(np.count_nonzero(outside_changed)),
        "eligibility_method": eligibility_method,
        "eligible_region_pixel_count": int(np.count_nonzero(eligible)),
    }


def retain_contact_ambient_occlusion(
    occlusion_amount: np.ndarray,
    actor_alpha: np.ndarray,
    receiver_visibility: np.ndarray,
    contact_region: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Keep only foot-local sky occlusion and reject catcher noise elsewhere."""
    amount = np.asarray(occlusion_amount, dtype=np.float32)
    actor = np.asarray(actor_alpha) > 0
    y, _x = np.nonzero(actor)
    if not len(y):
        return np.zeros_like(amount), {
            "method": "stance_sole_ambient_occlusion_v2",
            "status": "empty_actor",
        }
    contact_values = np.asarray(contact_region)
    if contact_values.shape != amount.shape:
        raise ValueError(
            "contact region must share the AO shape: "
            f"{contact_values.shape}, {amount.shape}"
        )
    contact = contact_values > 0.0
    far_values = amount[(~contact) & (amount > 0.0)]
    noise_floor = (
        float(np.percentile(far_values, 99.5)) if len(far_values) else 0.0
    )
    retained_threshold = max(AMBIENT_OCCLUSION_MIN_AMOUNT, noise_floor)
    filtered = amount.copy()
    filtered[~contact] = 0.0
    filtered[filtered < retained_threshold] = 0.0
    filtered, actor_edge_diagnostics = overlap_receiver_matte_with_actor_edge(
        filtered,
        actor_alpha,
        receiver_visibility,
        eligible_region=contact,
    )
    return filtered, {
        "method": "stance_sole_ambient_occlusion_v2",
        "status": "success",
        "contact_method": "evaluated_stance_sole",
        "far_field_noise_percentile": 99.5,
        "estimated_noise_floor": noise_floor,
        "retained_threshold": retained_threshold,
        "candidate_pixel_count": int(np.count_nonzero(amount > 0.0)),
        "contact_region_pixel_count": int(np.count_nonzero(contact)),
        "retained_pixel_count": int(np.count_nonzero(filtered > 0.0)),
        "actor_edge_receiver_overlap": actor_edge_diagnostics,
    }


def degrade_actor_rgba(
    rgba: np.ndarray,
    source: np.ndarray,
    blur_sigma: float,
    max_noise_std: float,
    random_seed: int,
) -> Tuple[np.ndarray, dict]:
    degraded = render_utils.blur_rgba_premultiplied(rgba, blur_sigma)
    alpha = degraded[..., 3]
    yx = np.argwhere(alpha >= 32)
    estimated_noise = 0.0
    if len(yx) and max_noise_std > 0:
        y0, x0 = yx.min(axis=0)
        y1, x1 = yx.max(axis=0) + 1
        pad = max(12, int(0.2 * (y1 - y0)))
        crop = source[max(0, y0 - pad):min(source.shape[0], y1 + pad),
                      max(0, x0 - pad):min(source.shape[1], x1 + pad)]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        residual = gray - cv2.GaussianBlur(gray, (0, 0), 0.9)
        estimated_noise = float(np.clip(1.4826 * np.median(np.abs(residual)), 0.0, max_noise_std))
        if estimated_noise > 0.05:
            rng = np.random.default_rng(random_seed)
            luma_noise = rng.normal(0.0, estimated_noise, alpha.shape).astype(np.float32)
            chroma_noise = rng.normal(0.0, estimated_noise * 0.28, degraded[..., :3].shape).astype(np.float32)
            noise = luma_noise[..., None] + chroma_noise
            rgb = degraded[..., :3].astype(np.float32)
            rgb += noise * (alpha.astype(np.float32) / 255.0)[..., None]
            degraded[..., :3] = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    return degraded, {
        "blur_sigma_px": float(blur_sigma),
        "estimated_noise_std_code_values": estimated_noise,
    }


def retain_primary_actor_component(rgba: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Remove disconnected alpha residue while retaining the actor's soft edge."""
    alpha = rgba[..., 3]
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        (alpha >= 32).astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return rgba, {"removed_disconnected_alpha_pixels": 0}

    largest = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    # The opaque/core silhouette identifies the actor. A modest dilation keeps
    # its antialiasing and transparent hair/clothing fringe without admitting
    # disconnected denoiser or lens-remapping residue at the image boundary.
    support = cv2.dilate(
        (labels == largest).astype(np.uint8),
        np.ones((11, 11), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    cleaned = rgba.copy()
    removed = int(np.count_nonzero((alpha > 0) & ~support))
    cleaned[~support] = 0
    return cleaned, {"removed_disconnected_alpha_pixels": removed}


def harmonize_actor_local(
    rgba: np.ndarray,
    source: np.ndarray,
    strength: float,
) -> Tuple[np.ndarray, dict]:
    """Apply only bounded low-frequency camera/illuminant corrections."""
    alpha = rgba[..., 3]
    core = alpha >= 32
    if not np.any(core) or strength <= 0:
        return rgba, {"method": "disabled_or_empty"}
    yx = np.argwhere(core)
    y0, x0 = yx.min(axis=0)
    y1, x1 = yx.max(axis=0) + 1
    actor_height = y1 - y0
    pad = max(10, int(round(0.16 * actor_height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    neighborhood = cv2.dilate(core.astype(np.uint8), kernel) > 0
    ring = neighborhood & ~core
    source_values = source.astype(np.float32) / 255.0
    ring_values = source_values[ring]
    if len(ring_values) < 200:
        return rgba, {"method": "insufficient_ring", "ring_pixels": int(len(ring_values))}

    ring_max = np.max(ring_values, axis=1)
    ring_min = np.min(ring_values, axis=1)
    saturation = (ring_max - ring_min) / np.maximum(ring_max, 1e-4)
    gray_reference = ring_values[saturation < 0.22]
    if len(gray_reference) < 100:
        gray_reference = ring_values
    reference_rgb = np.median(gray_reference, axis=0)
    reference_chroma = reference_rgb / max(float(np.mean(reference_rgb)), 1e-4)
    color_gain = np.clip(reference_chroma, 0.88, 1.12) ** strength

    foreground_values = rgba[..., :3].astype(np.float32) / 255.0
    foreground_luma = (
        0.2126 * foreground_values[..., 0]
        + 0.7152 * foreground_values[..., 1]
        + 0.0722 * foreground_values[..., 2]
    )
    ring_luma = 0.2126 * ring_values[:, 0] + 0.7152 * ring_values[:, 1] + 0.0722 * ring_values[:, 2]
    foreground_high = float(np.percentile(foreground_luma[core], 98))
    local_high = float(np.percentile(ring_luma, 99))
    highlight_gain = min(1.0, local_high / max(foreground_high, 1e-4)) ** strength

    foreground_max = np.max(foreground_values, axis=2)
    foreground_min = np.min(foreground_values, axis=2)
    foreground_saturation = (foreground_max - foreground_min) / np.maximum(foreground_max, 1e-4)
    foreground_sat_high = float(np.percentile(foreground_saturation[core], 90))
    local_sat_high = float(np.percentile(saturation, 90))
    saturation_gain = float(
        np.clip(local_sat_high / max(foreground_sat_high, 1e-4), 0.78, 1.0) ** strength
    )

    linear = render_utils.srgb_to_linear(foreground_values)
    linear *= color_gain[None, None, :] * highlight_gain
    # NAVSIM camera imagery is often less chromatic than a clean texture
    # render. Compress only the excess saturation, preserving linear luma and
    # retaining at least 78% of the asset's original chroma before strength.
    linear_luma = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )
    linear = linear_luma[..., None] + saturation_gain * (linear - linear_luma[..., None])
    harmonized = rgba.copy()
    harmonized[..., :3] = np.rint(
        render_utils.linear_to_srgb(linear) * 255.0
    ).astype(np.uint8)
    harmonized[~(alpha > 0), :3] = rgba[~(alpha > 0), :3]
    return harmonized, {
        "method": "bounded_local_color_camera_match",
        "strength": float(strength),
        "ring_pixels": int(len(ring_values)),
        "gray_reference_pixels": int(len(gray_reference)),
        "reference_rgb": reference_rgb.tolist(),
        "color_gain": color_gain.tolist(),
        "foreground_highlight": foreground_high,
        "local_highlight": local_high,
        "highlight_gain": float(highlight_gain),
        "foreground_saturation_p90": foreground_sat_high,
        "local_saturation_p90": local_sat_high,
        "saturation_gain": saturation_gain,
    }


def road_depth_from_plane(
    calibration: dict,
    plane: np.ndarray,
    width: int,
    height: int,
    remap_cache: dict,
) -> np.ndarray:
    """Intersect distorted camera rays with z=ax+by+c in LiDAR coordinates."""
    intrinsic = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion"], dtype=np.float64)
    key = (width, height, intrinsic.tobytes(), distortion.tobytes())
    if key not in remap_cache:
        remap_cache[key] = render_utils.distortion_maps(width, height, intrinsic, distortion)
    ideal_x, ideal_y = remap_cache[key]
    rays_camera = np.stack(
        [
            (ideal_x - intrinsic[0, 2]) / intrinsic[0, 0],
            (ideal_y - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones((height, width), dtype=np.float32),
        ],
        axis=-1,
    )
    camera_to_lidar = render_utils.camera_to_lidar(calibration)
    rotation = camera_to_lidar[:3, :3]
    origin = camera_to_lidar[:3, 3]
    rays_lidar = rays_camera @ rotation.T
    normal = np.array([-plane[0], -plane[1], 1.0], dtype=np.float64)
    numerator = -(float(np.dot(normal, origin)) - float(plane[2]))
    denominator = rays_lidar @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        camera_depth = numerator / denominator
    camera_depth[(camera_depth <= 0.2) | ~np.isfinite(camera_depth)] = np.inf
    return camera_depth.astype(np.float32)


def load_actor_metric_depth(
    depth_path: Path,
    manifest: dict,
    calibration: dict,
    output_width: int,
    output_height: int,
    remap_cache: dict,
) -> Tuple[np.ndarray, dict]:
    """Decode and lens-remap Blender's per-pixel actor distance pass."""
    encoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise TokenFailure("render_failed", f"unreadable actor depth pass: {depth_path}")
    if encoded.ndim != 3 or encoded.shape[2] < 4:
        raise TokenFailure(
            "render_failed",
            f"actor depth pass must be RGBA, got shape {encoded.shape}: {depth_path}",
        )
    alpha_channel = encoded[..., 3]
    if np.issubdtype(alpha_channel.dtype, np.integer):
        code_max = float(np.iinfo(alpha_channel.dtype).max)
    else:
        code_max = 1.0
    encoding_max_m = float(manifest["actor_depth_encoding_max_m"])
    encoded_camera_z = alpha_channel.astype(np.float32) / code_max * encoding_max_m

    render_width, render_height = render_storage_resolution(manifest)
    if encoded_camera_z.shape != (render_height, render_width):
        encoded_camera_z = cv2.resize(
            encoded_camera_z,
            (render_width, render_height),
            interpolation=cv2.INTER_NEAREST,
        )

    # Cycles' Z pass is camera-space optical-axis depth (verified against the
    # manifest root depth, including strongly off-axis side-camera renders), so
    # it already matches the OpenCV/LiDAR camera-z convention used below.
    ideal_camera_z = encoded_camera_z
    ideal_origin = render_storage_ideal_origin(manifest)
    camera_z = render_utils.distort_scalar_map(
        ideal_camera_z,
        calibration,
        remap_cache,
        output_size=(output_width, output_height),
        ideal_origin=ideal_origin,
        border_value=encoding_max_m,
        interpolation=cv2.INTER_NEAREST,
    )
    finite_actor = np.isfinite(camera_z) & (camera_z > 0.5) & (camera_z < encoding_max_m - 0.5)
    values = camera_z[finite_actor]
    diagnostics = {
        "path": str(depth_path),
        "encoded_dtype": str(alpha_channel.dtype),
        "encoding_max_m": encoding_max_m,
        "ideal_storage_resolution": [render_width, render_height],
        "ideal_storage_origin": list(ideal_origin),
        "valid_depth_pixels": int(len(values)),
        "camera_z_p05_m": float(np.percentile(values, 5)) if len(values) else None,
        "camera_z_median_m": float(np.median(values)) if len(values) else None,
        "camera_z_p95_m": float(np.percentile(values, 95)) if len(values) else None,
    }
    return camera_z.astype(np.float32), diagnostics


def actor_depth_coverage_diagnostics(
    actor_core_mask: np.ndarray,
    actor_alpha: np.ndarray,
    actor_metric_depth: np.ndarray,
) -> dict:
    """Measure Z-pass coverage without turning soft edge pixels into failures.

    The reference core is captured before camera-degradation blur.  A one-pixel
    dilation tolerates the sampling difference between linearly remapped RGBA
    and the nearest-neighbor Z pass.  Pixels still lacking Z remain safe because
    ``dense_depth_occlusion`` replaces them with actor-root depth.
    """
    actor_core_mask = np.asarray(actor_core_mask, dtype=bool)
    visible_actor_mask = np.asarray(actor_alpha) > 0
    valid_depth_mask = (
        np.isfinite(actor_metric_depth)
        & (actor_metric_depth > 0.5)
        & (actor_metric_depth < ACTOR_DEPTH_ENCODING_MAX_M - 0.5)
    )
    dilated_depth_mask = cv2.dilate(
        valid_depth_mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    core_pixels = int(np.count_nonzero(actor_core_mask))
    core_overlap_pixels = int(np.count_nonzero(actor_core_mask & valid_depth_mask))
    tolerant_core_overlap_pixels = int(
        np.count_nonzero(actor_core_mask & dilated_depth_mask)
    )
    visible_pixels = int(np.count_nonzero(visible_actor_mask))
    visible_overlap_pixels = int(
        np.count_nonzero(visible_actor_mask & valid_depth_mask)
    )
    root_fallback_pixels = visible_pixels - visible_overlap_pixels
    core_overlap_fraction = core_overlap_pixels / max(core_pixels, 1)
    tolerant_core_overlap_fraction = tolerant_core_overlap_pixels / max(core_pixels, 1)
    visible_overlap_fraction = visible_overlap_pixels / max(visible_pixels, 1)

    if core_pixels == 0:
        status = "no_opaque_core_use_root_fallback"
        warning = False
    elif tolerant_core_overlap_fraction < ACTOR_DEPTH_COVERAGE_WARNING_FRACTION:
        status = "low_core_coverage_use_root_fallback"
        warning = True
    else:
        status = "core_coverage_accepted"
        warning = False
    return {
        "coverage_reference": "pre_degradation_alpha_core",
        "core_alpha_threshold": ACTOR_CORE_ALPHA,
        "z_support_dilation_px": 1,
        "core_actor_pixels": core_pixels,
        "core_overlap_pixels": core_overlap_pixels,
        "core_overlap_fraction": float(core_overlap_fraction),
        "tolerant_core_overlap_pixels": tolerant_core_overlap_pixels,
        "tolerant_core_overlap_fraction": float(tolerant_core_overlap_fraction),
        "post_degradation_actor_pixels": visible_pixels,
        "post_degradation_overlap_pixels": visible_overlap_pixels,
        "post_degradation_overlap_fraction": float(visible_overlap_fraction),
        # Backward-compatible aliases for metadata written by the first v6
        # implementation of this check.
        "valid_actor_overlap_pixels": visible_overlap_pixels,
        "valid_actor_overlap_fraction": float(visible_overlap_fraction),
        "root_depth_fallback_pixels": int(root_fallback_pixels),
        "root_depth_fallback_fraction": float(
            root_fallback_pixels / max(visible_pixels, 1)
        ),
        "coverage_status": status,
        "coverage_warning": warning,
    }


def prepare_token(
    args: argparse.Namespace,
    loader,
    token: str,
    run_dir: Path,
    front_bumper_offset: float,
    asset: dict,
) -> dict:
    args = benign_sides.token_arguments(args, sys.modules[__name__], token)
    car = bool(getattr(args, "car", False))
    action_range = asset.get(
        "action_range",
        (CAR_STATIC_ANIMATION_FRAME, CAR_STATIC_ANIMATION_FRAME + 1),
    )
    actor_height_m = asset["actor_height_m"]
    scene_frames = loader.scene_frames_dicts[token]
    frames = scene_frames[:EXPECTED_HISTORY_FRAMES]
    if len(frames) != EXPECTED_HISTORY_FRAMES:
        raise TokenFailure(
            "missing_sensor",
            f"expected {EXPECTED_HISTORY_FRAMES} history frames, got {len(frames)}",
        )
    for frame in frames:
        for camera in CAMERA_ORDER:
            path = frame_source_path(args.data_root, args.split, frame, camera)
            if not path.is_file():
                raise TokenFailure("missing_sensor", f"missing {camera} image: {path}")
    map_locations = {str(frame.get("map_location", "")) for frame in frames}
    if len(map_locations) != 1:
        raise TokenFailure(
            "inconsistent_map_location",
            f"history frames span multiple map locations: {sorted(map_locations)}",
        )

    timestamps_s = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64) / 1e6
    if np.any(np.diff(timestamps_s) <= 0):
        raise TokenFailure("invalid_timestamps", f"non-increasing timestamps: {timestamps_s.tolist()}")
    standing = bool(getattr(args, "standing", False))
    benign = bool(getattr(args, "benign", False))
    if benign:
        direction_classification = None
        direction_name = "benign_" + ("opposite_car" if car else "standing" if standing else args.benign_walking_direction)
        lateral_sign = 0.0
        trajectory = benign_placements.resolve(
            args, sys.modules[__name__], scene_frames, timestamps_s,
            front_bumper_offset, asset["actor_geometry"]["local_bounds_m"],
        )
        actor_heading = trajectory["actor_heading_rad"]
    elif car:
        direction_classification = None
        direction_name = "stationary_lead_vehicle"
        lateral_sign = 0.0
        actor_heading = 0.0
        ego_geometry = getattr(args, "ego_vehicle_geometry", None)
        if not isinstance(ego_geometry, dict):
            raise RuntimeError("car preparation requires ego vehicle geometry")
        trajectory = resolve_stationary_vehicle_trajectory(
            scene_frames,
            timestamps_s,
            timestep=args.timestep,
            front_bumper_offset_m=front_bumper_offset,
            car_local_bounds=asset["actor_geometry"]["local_bounds_m"],
            ego_rear_length_m=float(ego_geometry["rear_length_m"]),
            ego_width_m=float(ego_geometry["width_m"]),
        )
    elif standing:
        direction_classification = None
        direction_name, lateral_sign, actor_heading = standing_facing(args, token)
        trajectory = resolve_standing_trajectory(
            scene_frames,
            timestamps_s,
            timestep=args.timestep,
            front_bumper_offset_m=front_bumper_offset,
            lateral_sign=lateral_sign,
        )
    else:
        direction_classification = args.direction_classification_labels[token]
        direction_name, lateral_sign, actor_heading = crossing_policy(
            token,
            args.direction_classification_labels,
        )
        trajectory = resolve_crossing_trajectory(
            scene_frames,
            timestamps_s,
            timestep=args.timestep,
            front_bumper_offset_m=front_bumper_offset,
            lateral_sign=lateral_sign,
            speed_mps=args.speed_mps,
        )
    placement = trajectory["placement"]
    basis_forward = trajectory.get("actor_basis_forward_global_xy", trajectory["reference_vehicle_forward_global_xy"])

    current_ego_to_global = np.asarray(frames[-1]["ego2global"], dtype=np.float64)
    global_to_current = np.linalg.inv(current_ego_to_global)
    if car:
        animation = [CAR_STATIC_ANIMATION_FRAME] * EXPECTED_HISTORY_FRAMES
        animation_fps = 24.0
    elif standing:
        animation = [STANDING_ANIMATION_FRAME] * EXPECTED_HISTORY_FRAMES
        animation_fps = 24.0
    else:
        animation = animation_frames(
            timestamps_s,
            args.speed_mps,
            asset["stride_length_m"],
            action_range[0],
            action_range[1],
            NAVSIM_POSE_SEQUENCE[0],
        )
        animation_fps = (
            (action_range[1] - action_range[0])
            * args.speed_mps
            / asset["stride_length_m"]
        )

    actor_reference_current_xys = trajectory["actor_current_xys"]
    actor_global_xys = trajectory["actor_global_xys"]
    actor_lidar_xys = [
        inverse_transform_xy_planar(
            np.asarray(frame["lidar2global"], dtype=np.float64),
            actor_global_xy,
        )
        for frame, actor_global_xy in zip(frames, actor_global_xys)
    ]
    frame_lidar_points = [
        geometry.load_lidar_xyz(args.data_root, args.split, frame)
        for frame in frames
    ]
    (
        ground_planes,
        ground_diagnostics_by_frame,
        strict_ground_failures,
    ) = fit_ground_sequence_best_effort(
        frames,
        frame_lidar_points,
        actor_lidar_xys,
        actor_reference_current_xys,
    )

    canonical_actor_to_global = None
    if car:
        final_lidar_to_global = np.asarray(
            frames[-1]["lidar2global"], dtype=np.float64
        )
        final_plane = ground_planes[-1]
        final_actor_lidar = ground_point_lidar_for_global_xy(
            final_lidar_to_global,
            final_plane,
            trajectory["anchor_global_xy"],
        )
        final_basis = road_basis_aligned_to_global_forward(
            final_plane,
            final_lidar_to_global,
            basis_forward,
            args.max_ground_tilt_deg,
        )
        canonical_actor_to_global = (
            final_lidar_to_global
            @ render_utils.matrix4(final_basis, final_actor_lidar)
        )

    actor_states = []
    per_frame_collisions = []
    for index, (
        frame,
        animation_frame,
        actor_lidar_xy,
        lidar_xyz,
        plane,
        ground_diagnostics,
    ) in enumerate(
        zip(
            frames,
            animation,
            actor_lidar_xys,
            frame_lidar_points,
            ground_planes,
            ground_diagnostics_by_frame,
        )
    ):
        lidar_to_global = np.asarray(frame["lidar2global"], dtype=np.float64)
        if car:
            # A stopped vehicle has one rigid world pose. Ground it once in
            # the final history frame, then express that same SE(3) transform
            # in every synchronized LiDAR/camera frame.
            actor_to_lidar = np.linalg.inv(lidar_to_global) @ canonical_actor_to_global
            actor_lidar = actor_to_lidar[:3, 3].copy()
            basis = actor_to_lidar[:3, :3].copy()
            actor_global = canonical_actor_to_global[:3, 3].copy()
        else:
            # Preserve the requested global horizontal trajectory, but obtain
            # vertical support independently from each timestamp's LiDAR.
            actor_lidar = np.array(
                [
                    actor_lidar_xy[0],
                    actor_lidar_xy[1],
                    geometry.ground_z(
                        plane, actor_lidar_xy[0], actor_lidar_xy[1]
                    ),
                ],
                dtype=np.float64,
            )
            actor_global = geometry.transform_points(
                lidar_to_global, actor_lidar[None]
            )[0]
            basis = road_basis_aligned_to_global_forward(
                plane,
                lidar_to_global,
                basis_forward,
                args.max_ground_tilt_deg,
            )
            actor_to_lidar = render_utils.matrix4(basis, actor_lidar)
        actor_current = geometry.transform_points(global_to_current, actor_global[None])[0]
        gravity_up_lidar = lidar_to_global[:3, :3].T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        receiver_normal = basis[:, 2]
        receiver_point = actor_lidar - 0.004 * receiver_normal
        receiver_plane = np.array(
            [
                -receiver_normal[0] / receiver_normal[2],
                -receiver_normal[1] / receiver_normal[2],
                float(np.dot(receiver_normal, receiver_point) / receiver_normal[2]),
            ],
            dtype=np.float64,
        )
        lidar_to_actor = np.linalg.inv(actor_to_lidar)
        if car:
            direction_lidar = basis[:, 0]
            yaw_lidar = math.atan2(direction_lidar[1], direction_lidar[0])
            collision = stationary_vehicle_collision_diagnostics(
                lidar_xyz,
                lidar_to_actor,
                asset["actor_geometry"]["local_bounds_m"],
            )
        else:
            direction_lidar = (benign_placements.direction_lidar(basis, actor_heading) if benign else
                               -basis[:, 0] if standing and lateral_sign == 0.0 else basis[:, 1] * lateral_sign)
            yaw_lidar = (
                math.atan2(direction_lidar[1], direction_lidar[0])
                + math.pi / 2.0
            )
            collision = frame_collision_diagnostics(
                lidar_xyz,
                lidar_to_actor,
                actor_heading,
                asset["actor_geometry"]["local_bounds_m"],
            )
        collision["frame_index"] = index
        per_frame_collisions.append(collision)
        contact_residual = actor_lidar[2] - geometry.ground_z(
            plane, actor_lidar[0], actor_lidar[1]
        )
        ground_diagnostics.update(
            {
                "frame_index": index,
                "plane_z_ax_by_c_lidar": plane.tolist(),
                "gravity_up_lidar": gravity_up_lidar.tolist(),
                "actor_ground_contact_residual_m": float(contact_residual),
            }
        )
        actor_states.append(
            {
                "frame_index": index,
                "actor_current": actor_current,
                "actor_global": actor_global,
                "actor_lidar": actor_lidar,
                "actor_to_lidar": actor_to_lidar,
                "lidar_to_actor": lidar_to_actor,
                "road_plane_lidar": plane,
                "receiver_plane_lidar": receiver_plane,
                "road_basis_lidar": basis,
                "gravity_up_lidar": gravity_up_lidar,
                "ground_diagnostics": ground_diagnostics,
                "collision_diagnostics": collision,
                "yaw_lidar": yaw_lidar,
                "animation_frame": animation_frame,
            }
        )

    final_state = actor_states[-1]
    current_bbox = render_utils.actor_bbox_lidar(
        final_state["actor_lidar"], final_state["road_basis_lidar"],
        actor_heading, asset["actor_geometry"]["local_bounds_m"],
    )
    current_front_visibility = front_visibility.assess_current(
        args, sys.modules[__name__], frames, current_bbox,
    )
    front_visibility.require(current_front_visibility, sys.modules[__name__])
    if car:
        swept_collision = dict(per_frame_collisions[-1])
        swept_collision.update(
            {
                "path_start_xy_m": final_state["actor_lidar"][:2].tolist(),
                "path_end_xy_m": final_state["actor_lidar"][:2].tolist(),
                "sample_count": 1,
            }
        )
        swept_path_point_count = 1
    else:
        collision_path_global_xys = [
            *actor_global_xys,
            np.asarray(trajectory["anchor_global_xy"], dtype=np.float64),
        ]
        if benign:
            collision_path_global_xys = trajectory["scene_collision_path_global_xys"]
        final_positions_lidar = [
            inverse_transform_xy_planar(
                np.asarray(frames[-1]["lidar2global"], dtype=np.float64),
                actor_global_xy,
            )
            for actor_global_xy in collision_path_global_xys
        ]
        swept_collision = trajectory_collision_diagnostics(
            frame_lidar_points[-1],
            final_state["road_plane_lidar"],
            final_positions_lidar,
            actor_basis_lidar=final_state["road_basis_lidar"],
            actor_heading=actor_heading,
            local_bounds_m=asset["actor_geometry"]["local_bounds_m"],
        )
        swept_path_point_count = len(final_positions_lidar)
    collision_diagnostics = {
        "per_frame": per_frame_collisions,
        "swept_final_frame": swept_collision,
        "swept_path_includes_future_interception": not car and not benign,
        "swept_path_point_count": swept_path_point_count,
        "occupied": bool(
            swept_collision["occupied"] or any(item["occupied"] for item in per_frame_collisions)
        ),
    }
    if (
        car
        and collision_diagnostics["occupied"]
        and not args.disable_collision_check
    ):
        occupied_frames = [
            item["frame_index"]
            for item in per_frame_collisions
            if item["occupied"]
        ]
        raise TokenFailure(
            "vehicle_placement_occupied",
            "LiDAR-supported scene geometry intersects the inserted car at "
            f"history frame(s) {occupied_frames}",
        )
    collision_failure = trajectory_collision_potential_failure(
        args,
        token,
        collision_diagnostics,
        placement,
    )
    ground_failure = ground_fit_potential_failure(
        token,
        strict_ground_failures,
        ground_diagnostics_by_frame,
    )
    if benign and strict_ground_failures:
        raise TokenFailure("ground_fit_failed", f"Benign placement requires strict grounding: {strict_ground_failures}")
    if benign and collision_diagnostics["occupied"]:
        raise TokenFailure("benign_placement_occupied", "Scene geometry intersects the benign actor or its motion")
    potential_failures = [
        item
        for item in (ground_failure, collision_failure)
        if item is not None
    ]

    lighting_views = []
    for camera in CAMERA_ORDER:
        calibration = frames[-1]["cams"][camera]
        camera_to_actor = actor_states[-1]["lidar_to_actor"] @ render_utils.camera_to_lidar(calibration)
        image = np.asarray(
            Image.open(frame_source_path(args.data_root, args.split, frames[-1], camera)).convert("RGB")
        )
        lighting_views.append(
            {
                "camera": camera,
                "image": image,
                "calibration": calibration,
                "camera_to_actor": camera_to_actor,
            }
        )
    estimated_lighting = estimate_multiview_lighting(lighting_views)
    discarded_image_direction = estimated_lighting.pop("sun_ray_direction", None)
    astronomical_sun = astronomical_sun_direction(frames[-1])
    lighting_base = {
        **estimated_lighting,
        "method": f"{estimated_lighting['method']}_with_ephemeris_direction",
        "direction_method": astronomical_sun["method"],
        "direct_sun_policy": (
            "ephemeris_horizon_and_multiview_cast_shadow_weather_gate_v2"
        ),
    }
    if not astronomical_sun["direct_sun_enabled"]:
        lighting_base["image_estimated_sun_energy_before_horizon_gate"] = float(
            lighting_base["sun_energy"]
        )
        lighting_base["sun_energy"] = 0.0
    sun_ray_global = np.asarray(
        astronomical_sun["sun_ray_direction_global_utm"], dtype=np.float64
    )
    for state, frame in zip(actor_states, frames):
        state["sun_ray_direction_actor"] = global_direction_to_actor(
            sun_ray_global,
            np.asarray(frame["lidar2global"], dtype=np.float64),
            state["lidar_to_actor"],
        )
    lighting = {
        **lighting_base,
        "sun_ray_directions_actor_by_frame": [
            state["sun_ray_direction_actor"].tolist() for state in actor_states
        ],
        "discarded_image_sun_ray_direction_actor_final": discarded_image_direction,
        "astronomical_sun": astronomical_sun,
    }

    view_records = []
    jobs = []
    projected_views = 0
    # Receiver previews return before token preparation. Every invocation that
    # reaches Blender publishes final images and masks.
    image_tree, mask_tree = "images", "masks"
    for state, frame in zip(actor_states, frames):
        if car:
            bbox_actor = local_bbox_corners(
                asset["actor_geometry"]["local_bounds_m"]
            )
            bbox_lidar = geometry.transform_points(
                state["actor_to_lidar"], bbox_actor
            )
        else:
            bbox_lidar = render_utils.actor_bbox_lidar(
                state["actor_lidar"],
                state["road_basis_lidar"],
                actor_heading,
                asset["actor_geometry"]["local_bounds_m"],
            )
            bbox_actor = geometry.transform_points(
                state["lidar_to_actor"], bbox_lidar
            )
        for camera in CAMERA_ORDER:
            source_path = frame_source_path(args.data_root, args.split, frame, camera)
            with Image.open(source_path) as source_image:
                width, height = source_image.size
            calibration = frame["cams"][camera]
            projected_bbox_uv, projected_bbox_depth = geometry.project_distorted(
                bbox_lidar, calibration
            )
            projected_inside = (
                np.isfinite(projected_bbox_uv).all(axis=1)
                & np.isfinite(projected_bbox_depth)
                & (projected_bbox_depth > 0.1)
                & (projected_bbox_uv[:, 0] >= 0) & (projected_bbox_uv[:, 0] < width)
                & (projected_bbox_uv[:, 1] >= 0) & (projected_bbox_uv[:, 1] < height)
            )
            projected_visible = bool(
                np.any(geometry.camera_frustum_mask(bbox_lidar, calibration))
            )
            if (current_front_visibility is not None
                    and state["frame_index"] == front_visibility.FRAME_INDEX
                    and camera == front_visibility.CAMERA):
                projected_visible = current_front_visibility["passed"]
            finite_bbox = np.isfinite(projected_bbox_uv).all(axis=1) & (projected_bbox_depth > 0.1)
            projected_bounds = None
            if np.any(finite_bbox):
                projected_bounds = [
                    float(np.min(projected_bbox_uv[finite_bbox, 0])),
                    float(np.min(projected_bbox_uv[finite_bbox, 1])),
                    float(np.max(projected_bbox_uv[finite_bbox, 0])),
                    float(np.max(projected_bbox_uv[finite_bbox, 1])),
                ]
            output_path = run_dir / image_tree / token / camera / f"{state['frame_index']}.png"
            mask_path = run_dir / mask_tree / token / camera / f"{state['frame_index']}.png"
            actor_mask_path = run_dir / "actor_masks" / token / camera / f"{state['frame_index']}.png"
            raw_path = run_dir / "raw_ideal" / token / camera / f"{state['frame_index']}.png"
            actor_sun_path = run_dir / "raw_actor_sun" / token / camera / f"{state['frame_index']}.exr"
            actor_sky_path = run_dir / "raw_actor_sky" / token / camera / f"{state['frame_index']}.exr"
            ambient_occlusion_path = run_dir / "raw_ambient_occlusion" / token / camera / f"{state['frame_index']}.exr"
            actor_depth_path = run_dir / "raw_actor_depth" / token / camera / f"{state['frame_index']}.png"
            relative_depth_path = run_dir / "source_relative_depth" / token / camera / f"{state['frame_index']}.npz"
            info_path = raw_path.with_suffix(".render.json")
            manifest_path = run_dir / "manifests" / token / camera / f"{state['frame_index']}.json"
            shadow_label = contact_only_shadow_label()
            shadow_label_policy = (
                shadow_label_render_policy(shadow_label)
                if shadow_label is not None
                else None
            )
            manifest = None
            if projected_visible:
                projected_views += 1
                required_base_passes = BASE_RENDER_FULL_PASSES
                camera_to_actor = state["lidar_to_actor"] @ render_utils.camera_to_lidar(calibration)
                root_depth_m = actor_camera_depth(camera_to_actor)
                overscan = render_utils.distortion_overscan(calibration, width, height)
                # Base rendering needs only the actor and its local Sun-off AO.
                # Directional extent is added solely to the later label-selected
                # render, so the base pass does not cover an unused long shadow.
                render_region = actor_shadow_render_region(
                    bbox_actor,
                    camera_to_actor,
                    overscan["intrinsic"],
                    overscan["resolution"],
                    state["sun_ray_direction_actor"],
                    0.0,
                    0.0,
                )
                manifest = {
                    "job_id": f"{token}:{camera}:{state['frame_index']}",
                    "render_mode": "base",
                    "resolution": overscan["resolution"],
                    "sensor_resolution": [width, height],
                    "ideal_origin": overscan["origin"],
                    "ideal_bounds": overscan["ideal_bounds"],
                    "distortion_overscan_padding_px": overscan["padding_px"],
                    "animation_frame": int(state["animation_frame"]),
                    "actor_position": [0.0, 0.0, 0.0],
                    "actor_heading": actor_heading,
                    "actor_scale": 1.0,
                    "receiver_plane_z_m": -0.004,
                    "foot_contact_distance_m": FOOT_CONTACT_DISTANCE_M,
                    "foot_stance_relative_distance_m": (
                        FOOT_STANCE_RELATIVE_DISTANCE_M
                    ),
                    "foot_sole_band_m": FOOT_SOLE_BAND_M,
                    "camera_intrinsic_matrix": np.asarray(overscan["intrinsic"]).tolist(),
                    "camera_to_world": camera_to_actor.tolist(),
                    "actor_root_depth_m": root_depth_m,
                    "timestamp_us": int(frame["timestamp"]),
                    "map_location": str(frame["map_location"]),
                    "astronomical_sun": astronomical_sun,
                    "lighting": {
                        **lighting_base,
                        "sun_ray_direction": state["sun_ray_direction_actor"].tolist(),
                    },
                    "samples": args.samples,
                    "render_scale": args.render_scale,
                    "render_region": render_region,
                    "raw_render_storage": RAW_RENDER_STORAGE_VERSION,
                    "render_region_padding_px": RENDER_REGION_PADDING_PX,
                    "motion_blur": (
                        False
                        if standing or car
                        else not args.disable_motion_blur
                    ),
                    "motion_blur_shutter": 0.25,
                    "animation_fps": animation_fps,
                    "output_path": str(raw_path.resolve()),
                    "actor_sun_output_path": str(actor_sun_path.resolve()),
                    "actor_sky_output_path": str(actor_sky_path.resolve()),
                    "ambient_occlusion_output_path": str(ambient_occlusion_path.resolve()),
                    "actor_depth_output_path": str(actor_depth_path.resolve()),
                    "actor_depth_encoding_max_m": ACTOR_DEPTH_ENCODING_MAX_M,
                    "render_info_path": str(info_path.resolve()),
                    "asset_path": str(asset["path"]),
                    "asset_sha256": asset["sha256"],
                    "asset_id": asset["id"],
                    "render_pipeline_version": RENDER_PIPELINE_VERSION,
                    "render_pass_policy_version": (
                        BASE_RENDER_PASS_POLICY_VERSION
                    ),
                    "required_passes": list(required_base_passes),
                    "produced_passes": list(required_base_passes),
                }
                if car:
                    manifest.update(
                        {
                            "actor_type": "car",
                            "actor_root_name": asset["actor_root_name"],
                            "contact_profile": CAR_CONTACT_PROFILE,
                            "asset_metadata_sha256": asset[
                                "metadata_sha256"
                            ],
                            "vehicle_contact_patches": asset[
                                "actor_geometry"
                            ]["wheel_contact_patches"],
                        }
                    )
                if standing:
                    manifest.update(
                        {
                            "standing_pose": True,
                            "standing_pose_version": STANDING_POSE_VERSION,
                        }
                    )
                cache_key = compatible_render_cache_key(
                    manifest_path,
                    manifest,
                    (
                        "output_path",
                        "actor_depth_output_path",
                        "actor_sun_output_path",
                        "actor_sky_output_path",
                        "ambient_occlusion_output_path",
                        "render_info_path",
                    ),
                    reuse_existing=not args.overwrite,
                )
                manifest["cache_key"] = cache_key
                write_json(manifest_path, manifest)
                if args.overwrite or not (
                    cache_matches(
                        raw_path,
                        info_path,
                        cache_key,
                        required_paths=required_render_paths(
                            manifest,
                            {
                                "actor_depth": actor_depth_path,
                                "actor_sun": actor_sun_path,
                                "actor_sky": actor_sky_path,
                                "ambient_occlusion": ambient_occlusion_path,
                            },
                        ),
                    )
                ):
                    jobs.append(manifest)
            view_records.append(
                {
                    "frame_index": state["frame_index"],
                    "camera": camera,
                    "timestamp_us": int(frame["timestamp"]),
                    "map_location": str(frame["map_location"]),
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                    "mask_path": str(mask_path),
                    "actor_mask_path": str(actor_mask_path),
                    "raw_path": str(raw_path) if manifest else None,
                    "actor_sun_path": str(actor_sun_path) if manifest else None,
                    "actor_sky_path": str(actor_sky_path) if manifest else None,
                    "ambient_occlusion_path": str(ambient_occlusion_path) if manifest else None,
                    "actor_depth_path": str(actor_depth_path) if manifest else None,
                    "render_info_path": str(info_path) if manifest else None,
                    "manifest_path": str(manifest_path) if manifest else None,
                    "relative_depth_path": str(relative_depth_path) if manifest else None,
                    "actor_root_depth_m": root_depth_m if manifest else None,
                    "projected_visible": projected_visible,
                    "projected_bbox_inside_corner_count": int(np.count_nonzero(projected_inside)),
                    "projected_bbox_distorted_bounds": projected_bounds,
                    "front_visibility": (current_front_visibility
                        if state["frame_index"] == front_visibility.FRAME_INDEX
                        and camera == front_visibility.CAMERA else None),
                    "render_region": render_region if manifest else None,
                    "shadow_label": shadow_label,
                    "shadow_label_policy": shadow_label_policy,
                    "actor_current": state["actor_current"].tolist(),
                    "actor_global": state["actor_global"].tolist(),
                    "actor_lidar": state["actor_lidar"].tolist(),
                    "road_plane_lidar": state["road_plane_lidar"].tolist(),
                    "receiver_plane_lidar": state["receiver_plane_lidar"].tolist(),
                    "sun_ray_direction_actor": state["sun_ray_direction_actor"].tolist(),
                    "yaw_lidar": state["yaw_lidar"],
                    "animation_frame": int(state["animation_frame"]),
                }
            )
    if projected_views == 0:
        raise TokenFailure("no_camera_visibility", "actor bounding volume is outside all 12 camera views")
    return {
        "token": token,
        "frames": frames,
        "direction_classification": direction_classification,
        "direction": direction_name,
        "lateral_sign": lateral_sign,
        "velocity_y_mps": (
            0.0 if standing or car else float(lateral_sign * args.speed_mps)
        ),
        "walking_velocity_global_xy_mps": trajectory[
            "walking_velocity_global_xy_mps"
        ],
        "crossing_offsets_m": trajectory["crossing_offsets_m"],
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
        "front_bumper_offset_m": front_bumper_offset,
        "asset": asset,
        "actor_type": asset.get("actor_type", "pedestrian"),
        "actor_geometry": asset.get("actor_geometry"),
        "actor_height_m": actor_height_m,
        "road_plane": final_state["road_plane_lidar"],
        "road_basis": final_state["road_basis_lidar"],
        "gravity_up_lidar": final_state["gravity_up_lidar"],
        "ground_diagnostics": [state["ground_diagnostics"] for state in actor_states],
        "collision_diagnostics": collision_diagnostics,
        "potential_failures": potential_failures,
        "lighting": lighting,
        "timestamps_s": timestamps_s,
        "view_records": view_records,
        "jobs": jobs,
    }


FINALIZATION_CACHE_ARRAY_DTYPES = {
    "actor_rgba": np.dtype(np.uint8),
    "base_actor_sun_raw": np.dtype(np.float32),
    "actor_sky_raw": np.dtype(np.float32),
    "receiver_visibility": np.dtype(np.float32),
    "contact_region_sensor": np.dtype(np.float32),
    "contact_ambient_occlusion_amount": np.dtype(np.float32),
    "labeled_ambient_occlusion_amount": np.dtype(np.float32),
}
FINALIZATION_CACHE_REQUIRED_ARRAYS = frozenset(
    name
    for name in FINALIZATION_CACHE_ARRAY_DTYPES
    if name != "base_actor_sun_raw"
)


def finalization_cache_path(run_dir: Path, token: str, record: dict) -> Path:
    return (
        run_dir
        / "finalization_cache"
        / token
        / str(record["camera"])
        / f"{int(record['frame_index'])}.npz"
    )


def label_independent_finalization_cache_key(
    args: argparse.Namespace,
    record: dict,
    manifest: dict,
    calibration: dict,
) -> str:
    source_sha256 = record.get("source_sha256")
    if not source_sha256:
        source_sha256 = file_sha256(Path(record["source_path"]))
        record["source_sha256"] = source_sha256
    return canonical_hash(
        {
            "cache_version": FINALIZATION_CACHE_VERSION,
            "render_pipeline_version": RENDER_PIPELINE_VERSION,
            "base_render_cache_key": manifest["cache_key"],
            "base_required_passes": list(
                render_manifest_passes(manifest, "required_passes")
            ),
            "source_sha256": source_sha256,
            "depth_model_id": DEPTH_MODEL_ID,
            "depth_prediction_cache_version": DEPTH_PREDICTION_CACHE_VERSION,
            "calibration": calibration,
            "actor_root_depth_m": record["actor_root_depth_m"],
            "road_plane_lidar": record["road_plane_lidar"],
            "receiver_plane_lidar": record["receiver_plane_lidar"],
            "degradation_seed": int(args.seed),
            "actor_blur_sigma": float(args.actor_blur_sigma),
            "max_actor_noise_std": float(args.max_actor_noise_std),
            "harmonization_strength": float(args.harmonization_strength),
            "minimum_composite_alpha": MIN_COMPOSITE_ALPHA,
            "actor_core_alpha": ACTOR_CORE_ALPHA,
            "ambient_occlusion_minimum_amount": AMBIENT_OCCLUSION_MIN_AMOUNT,
            "diffuse_body_occlusion_minimum_amount": (
                DIFFUSE_BODY_OCCLUSION_MIN_AMOUNT
            ),
            "receiver_matte_actor_edge_radius_px": (
                RECEIVER_MATTE_ACTOR_EDGE_RADIUS_PX
            ),
            "receiver_matte_opaque_alpha": RECEIVER_MATTE_OPAQUE_ALPHA,
        }
    )


def load_label_independent_finalization_cache(
    cache_path: Path,
    cache_key: str,
    sensor_shape: Tuple[int, int],
    *,
    require_base_actor_sun: bool = True,
) -> Optional[dict]:
    info_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text())
        if (
            info.get("cache_key") != cache_key
            or info.get("cache_version") != FINALIZATION_CACHE_VERSION
        ):
            return None
        height, width = sensor_shape
        expected_shapes = {
            "actor_rgba": (height, width, 4),
            "base_actor_sun_raw": (height, width, 4),
            "actor_sky_raw": (height, width, 4),
            "receiver_visibility": (height, width),
            "contact_region_sensor": (height, width),
            "contact_ambient_occlusion_amount": (height, width),
            "labeled_ambient_occlusion_amount": (height, width),
        }
        required_arrays = set(FINALIZATION_CACHE_REQUIRED_ARRAYS)
        if require_base_actor_sun:
            required_arrays.add("base_actor_sun_raw")
        arrays = {}
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["cache_key"].item()) != cache_key:
                return None
            for name, dtype in FINALIZATION_CACHE_ARRAY_DTYPES.items():
                if name not in data.files:
                    if name in required_arrays:
                        return None
                    continue
                value = data[name]
                if value.shape != expected_shapes[name] or value.dtype != dtype:
                    return None
                arrays[name] = value.copy()
        diagnostics = info.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return None
        return {"arrays": arrays, "diagnostics": diagnostics}
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: ignoring unreadable finalization cache {cache_path}: {exc}",
            file=sys.stderr,
        )
        return None


def write_label_independent_finalization_cache(
    cache_path: Path,
    cache_key: str,
    arrays: dict,
    diagnostics: dict,
) -> None:
    missing_arrays = FINALIZATION_CACHE_REQUIRED_ARRAYS - set(arrays)
    if missing_arrays:
        raise ValueError(
            "Finalization cache is missing required arrays: "
            + ", ".join(sorted(missing_arrays))
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{cache_path.stem}.",
            suffix=".npz",
            dir=cache_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                cache_key=np.asarray(cache_key),
                **{
                    name: np.asarray(arrays[name], dtype=dtype)
                    for name, dtype in FINALIZATION_CACHE_ARRAY_DTYPES.items()
                    if name in arrays
                },
            )
        os.replace(temporary_path, cache_path)
        temporary_path = None
        write_json(
            cache_path.with_suffix(".json"),
            {
                "cache_version": FINALIZATION_CACHE_VERSION,
                "cache_key": cache_key,
                "array_file": str(cache_path),
                "diagnostics": diagnostics,
            },
        )
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def compute_label_independent_finalization_intermediates(
    args: argparse.Namespace,
    prepared: dict,
    record: dict,
    source: np.ndarray,
    frame: dict,
    calibration: dict,
    manifest: dict,
    render_info: dict,
    remap_cache: dict,
    frame_lidar_cache: dict,
    frame_ground_cache: dict,
    metric_depth_cache: dict,
    road_depth_cache: dict,
    receiver_depth_cache: dict,
) -> dict:
    """Build label-independent products reused during finalization."""
    token = prepared["token"]
    frame_index, camera = record["frame_index"], record["camera"]
    height, width = source.shape[:2]
    raw_path = Path(record["raw_path"])
    actor_sun_path = Path(record["actor_sun_path"])
    actor_sky_path = Path(record["actor_sky_path"])
    ambient_occlusion_path = Path(record["ambient_occlusion_path"])
    actor_depth_path = Path(record["actor_depth_path"])

    ideal = np.asarray(Image.open(raw_path).convert("RGBA"))
    canvas_width, canvas_height = [int(value) for value in manifest["resolution"]]
    render_width, render_height = render_storage_resolution(manifest)
    storage_x, storage_y, _storage_x_max, _storage_y_max = (
        render_storage_region(manifest)
    )
    if ideal.shape[:2] != (render_height, render_width):
        ideal = render_utils.resize_rgba_premultiplied(
            ideal, render_width, render_height
        )
    ideal_origin = render_storage_ideal_origin(manifest)
    ideal_contact_region, foot_contact_diagnostics = ideal_foot_contact_region(
        render_info,
        ideal[..., 3],
        ideal_storage_origin=(storage_x, storage_y),
    )
    contact_region_sensor = render_utils.distort_scalar_map(
        ideal_contact_region,
        calibration,
        remap_cache,
        output_size=(width, height),
        ideal_origin=ideal_origin,
        border_value=0.0,
        interpolation=cv2.INTER_LINEAR,
    )
    contact_region_sensor = np.clip(contact_region_sensor, 0.0, 1.0).astype(
        np.float32
    )
    distorted = render_utils.distort_rgba(
        ideal,
        calibration,
        remap_cache,
        output_size=(width, height),
        ideal_origin=ideal_origin,
    )
    distorted, component_diagnostics = retain_primary_actor_component(distorted)
    actor_core_mask = distorted[..., 3] >= ACTOR_CORE_ALPHA
    rendered_actor_pixels = int(np.count_nonzero(distorted[..., 3]))
    projected_bounds = record.get("projected_bbox_distorted_bounds")
    projected_bbox_area = 0.0
    if projected_bounds is not None:
        projected_bbox_area = max(
            0.0, projected_bounds[2] - projected_bounds[0]
        ) * max(0.0, projected_bounds[3] - projected_bounds[1])
    minimum_expected_pixels = max(20, int(round(0.015 * projected_bbox_area)))
    if (
        record.get("projected_bbox_inside_corner_count", 0) == 8
        and rendered_actor_pixels < minimum_expected_pixels
    ):
        raise TokenFailure(
            "render_visibility_mismatch",
            f"{token} frame {frame_index} {camera}: projected bbox is visible but "
            f"only {rendered_actor_pixels} actor pixels were rendered "
            f"(minimum diagnostic threshold {minimum_expected_pixels})",
        )
    component_diagnostics.update(
        {
            "rendered_actor_pixels_before_degradation": rendered_actor_pixels,
            "projected_bbox_area_px2": float(projected_bbox_area),
            "minimum_expected_actor_pixels": minimum_expected_pixels,
            "distortion_overscan_origin": [
                float(value) for value in manifest["ideal_origin"]
            ],
            "distortion_overscan_resolution": [canvas_width, canvas_height],
            "distortion_storage_origin": list(ideal_origin),
            "distortion_storage_resolution": [render_width, render_height],
        }
    )
    degradation_seed = int.from_bytes(
        hashlib.blake2b(
            f"degradation:{args.seed}:{token}:{camera}:{frame_index}".encode(),
            digest_size=8,
        ).digest(),
        "big",
    )
    distorted, degradation_diagnostics = degrade_actor_rgba(
        distorted,
        source,
        args.actor_blur_sigma,
        args.max_actor_noise_std,
        degradation_seed,
    )
    degradation_diagnostics.update(component_diagnostics)
    distorted[distorted[..., 3] < MIN_COMPOSITE_ALPHA] = 0
    actor_alpha = distorted[..., 3]
    depth_key = (frame_index, camera)
    if frame_index not in frame_lidar_cache:
        frame_lidar_cache[frame_index] = geometry.load_lidar_xyz(
            args.data_root, args.split, frame
        )
        frame_ground_cache[frame_index] = np.asarray(
            record["road_plane_lidar"], dtype=np.float64
        )
    if depth_key not in metric_depth_cache:
        relative_path = Path(record["relative_depth_path"])
        if not relative_path.is_file():
            raise TokenFailure("depth_inference_failed", f"missing {relative_path}")
        with np.load(relative_path) as depth_data:
            relative_depth = depth_data["relative_depth"]
        metric_depth_cache[depth_key] = align_relative_depth_to_lidar(
            relative_depth,
            frame_lidar_cache[frame_index],
            calibration,
            required_depth_m=float(record["actor_root_depth_m"]),
        )
        road_depth_cache[depth_key] = road_depth_from_plane(
            calibration,
            frame_ground_cache[frame_index],
            width,
            height,
            remap_cache,
        )
    source_metric_depth, depth_alignment = metric_depth_cache[depth_key]
    actor_metric_depth, actor_depth_diagnostics = load_actor_metric_depth(
        actor_depth_path,
        manifest,
        calibration,
        width,
        height,
        remap_cache,
    )
    coverage_diagnostics = actor_depth_coverage_diagnostics(
        actor_core_mask,
        actor_alpha,
        actor_metric_depth,
    )
    actor_depth_diagnostics.update(coverage_diagnostics)
    if coverage_diagnostics["coverage_warning"]:
        print(
            f"WARNING {token} frame {frame_index} {camera}: actor Z pass covers "
            f"{100.0 * coverage_diagnostics['tolerant_core_overlap_fraction']:.1f}% "
            "of the pre-degradation actor core; using root depth for uncovered pixels",
            file=sys.stderr,
        )
    foreground_support, foreground_diagnostics = lidar_foreground_support(
        frame_lidar_cache[frame_index],
        calibration,
        frame_ground_cache[frame_index],
        actor_alpha,
        float(record["actor_root_depth_m"]),
    )
    occlusion_strength, occlusion_diagnostics = dense_depth_occlusion(
        actor_alpha,
        source_metric_depth,
        float(record["actor_root_depth_m"]),
        road_depth_cache[depth_key],
        actor_metric_depth=actor_metric_depth,
        foreground_support=foreground_support,
    )
    distorted[..., 3] = np.rint(
        distorted[..., 3].astype(np.float32) * (1.0 - occlusion_strength)
    ).astype(np.uint8)
    actor_alpha = distorted[..., 3]
    occlusion_diagnostics["depth_alignment"] = depth_alignment
    occlusion_diagnostics["actor_depth_pass"] = actor_depth_diagnostics
    occlusion_diagnostics["lidar_foreground_support"] = foreground_diagnostics
    distorted, harmonization_diagnostics = harmonize_actor_local(
        distorted, source, args.harmonization_strength
    )
    actor_rgba = distorted.copy()
    actor_alpha = actor_rgba[..., 3]

    base_actor_sun_required = (
        BASE_RENDER_ACTOR_SUN_PASS
        in render_manifest_passes(manifest, "required_passes")
    )
    base_actor_sun_raw = None
    base_actor_sun_render_diagnostics = None
    if base_actor_sun_required:
        base_actor_sun_ideal, base_actor_sun_render_diagnostics = (
            load_linear_actor_exr(actor_sun_path, manifest)
        )
        base_actor_sun_raw = distort_linear_actor_rgba(
            base_actor_sun_ideal,
            calibration,
            remap_cache,
            output_size=(width, height),
            ideal_origin=ideal_origin,
        )
    actor_sky_ideal, actor_sky_render_diagnostics = load_linear_actor_exr(
        actor_sky_path, manifest
    )
    actor_sky_raw = distort_linear_actor_rgba(
        actor_sky_ideal,
        calibration,
        remap_cache,
        output_size=(width, height),
        ideal_origin=ideal_origin,
    )

    if depth_key not in receiver_depth_cache:
        receiver_depth_cache[depth_key] = road_depth_from_plane(
            calibration,
            np.asarray(record["receiver_plane_lidar"], dtype=np.float64),
            width,
            height,
            remap_cache,
        )
    receiver_visibility, receiver_diagnostics = shadow_receiver_visibility(
        receiver_depth_cache[depth_key], source_metric_depth
    )

    ambient_full_ideal, ambient_occlusion_diagnostics = (
        load_shadow_catcher_amount(
            ambient_occlusion_path,
            manifest,
            minimum_amount=AMBIENT_OCCLUSION_MIN_AMOUNT,
        )
    )
    ambient_contact_ideal = np.clip(
        ambient_full_ideal * ideal_contact_region, 0.0, 1.0
    )
    contact_ambient_occlusion_amount = render_utils.distort_scalar_map(
        ambient_contact_ideal,
        calibration,
        remap_cache,
        output_size=(width, height),
        ideal_origin=ideal_origin,
        border_value=0.0,
        interpolation=cv2.INTER_LINEAR,
    )
    contact_ambient_occlusion_amount = np.clip(
        contact_ambient_occlusion_amount * receiver_visibility, 0.0, 1.0
    )
    (
        contact_ambient_occlusion_amount,
        contact_ambient_component_diagnostics,
    ) = retain_contact_ambient_occlusion(
        contact_ambient_occlusion_amount,
        actor_alpha,
        receiver_visibility,
        contact_region_sensor,
    )

    # Cache this label-independent variant so refinalization can avoid
    # decoding/remapping every base catcher and rerunning depth.
    diffuse_body_ideal = ambient_full_ideal.copy()
    diffuse_body_ideal[
        diffuse_body_ideal < DIFFUSE_BODY_OCCLUSION_MIN_AMOUNT
    ] = 0.0
    diffuse_body_occlusion_amount = render_utils.distort_scalar_map(
        diffuse_body_ideal,
        calibration,
        remap_cache,
        output_size=(width, height),
        ideal_origin=ideal_origin,
        border_value=0.0,
        interpolation=cv2.INTER_LINEAR,
    )
    diffuse_body_occlusion_amount = np.clip(
        diffuse_body_occlusion_amount * receiver_visibility, 0.0, 1.0
    )
    (
        diffuse_body_occlusion_amount,
        diffuse_body_component_diagnostics,
    ) = retain_grounded_shadow_component(
        diffuse_body_occlusion_amount,
        actor_alpha,
        anchor_region=contact_region_sensor,
    )
    (
        diffuse_body_occlusion_amount,
        diffuse_body_actor_edge_diagnostics,
    ) = overlap_receiver_matte_with_actor_edge(
        diffuse_body_occlusion_amount,
        actor_alpha,
        receiver_visibility,
        contact_region_sensor,
    )
    diffuse_body_component_diagnostics[
        "actor_edge_receiver_overlap"
    ] = diffuse_body_actor_edge_diagnostics
    labeled_ambient_occlusion_amount = merge_sky_occlusion_components(
        diffuse_body_occlusion_amount,
        contact_ambient_occlusion_amount,
    )

    return {
        "arrays": {
            "actor_rgba": actor_rgba,
            **(
                {
                    "base_actor_sun_raw": base_actor_sun_raw.astype(
                        np.float32
                    )
                }
                if base_actor_sun_raw is not None
                else {}
            ),
            "actor_sky_raw": actor_sky_raw.astype(np.float32),
            "receiver_visibility": receiver_visibility.astype(np.float32),
            "contact_region_sensor": contact_region_sensor.astype(np.float32),
            "contact_ambient_occlusion_amount": (
                contact_ambient_occlusion_amount.astype(np.float32)
            ),
            "labeled_ambient_occlusion_amount": (
                labeled_ambient_occlusion_amount.astype(np.float32)
            ),
        },
        "diagnostics": {
            "occlusion": occlusion_diagnostics,
            "degradation": degradation_diagnostics,
            "harmonization": harmonization_diagnostics,
            "foot_contact": foot_contact_diagnostics,
            "receiver": receiver_diagnostics,
            **(
                {
                    "base_actor_sun_render": (
                        base_actor_sun_render_diagnostics
                    )
                }
                if base_actor_sun_render_diagnostics is not None
                else {}
            ),
            "actor_sky_render": actor_sky_render_diagnostics,
            "ambient_occlusion_render": ambient_occlusion_diagnostics,
            "contact_ambient_component": (
                contact_ambient_component_diagnostics
            ),
            "diffuse_body_component": diffuse_body_component_diagnostics,
            "diffuse_body_pixel_count": int(
                np.count_nonzero(diffuse_body_occlusion_amount > 0.0)
            ),
            "stance_contact_pixel_count": int(
                np.count_nonzero(contact_ambient_occlusion_amount > 0.0)
            ),
        },
    }


def finalize_token(
    args: argparse.Namespace,
    prepared: dict,
    run_dir: Path,
    remap_cache: dict,
) -> dict:
    token = prepared["token"]
    output_paths = {}
    final_records = []
    finalization_potential_failures = []
    frame_lidar_cache = {}
    frame_ground_cache = {}
    metric_depth_cache = {}
    road_depth_cache = {}
    receiver_depth_cache = {}
    finalization_cache_hits = 0
    finalization_cache_misses = 0
    for record in prepared["view_records"]:
        view_started = time.perf_counter()
        view_cache_status = "not_projected"
        frame_index, camera = record["frame_index"], record["camera"]
        shadow_label = record.get("shadow_label")
        shadow_label_policy = record.get("shadow_label_policy")
        label_composition_diagnostics = {
            "enabled": True,
            "method": "not_evaluated",
        }
        source = np.asarray(Image.open(record["source_path"]).convert("RGB"))
        height, width = source.shape[:2]
        frame = prepared["frames"][frame_index]
        calibration = frame["cams"][camera]
        if record["projected_visible"]:
            raw_path = Path(record["raw_path"])
            actor_sun_path = Path(record["actor_sun_path"])
            actor_sky_path = Path(record["actor_sky_path"])
            ambient_occlusion_path = Path(record["ambient_occlusion_path"])
            actor_depth_path = Path(record["actor_depth_path"])
            info_path = Path(record["render_info_path"])
            manifest = json.loads(Path(record["manifest_path"]).read_text())
            base_actor_sun_required = (
                BASE_RENDER_ACTOR_SUN_PASS
                in render_manifest_passes(manifest, "required_passes")
            )
            if not cache_matches(
                raw_path,
                info_path,
                manifest["cache_key"],
                required_paths=required_render_paths(
                    manifest,
                    {
                        "actor_depth": actor_depth_path,
                        "actor_sun": actor_sun_path,
                        "actor_sky": actor_sky_path,
                        "ambient_occlusion": ambient_occlusion_path,
                    },
                ),
            ):
                raise TokenFailure("render_failed", f"missing or stale render: {raw_path}")
            render_info = json.loads(info_path.read_text())
            cache_path = finalization_cache_path(run_dir, token, record)
            finalization_cache_key = label_independent_finalization_cache_key(
                args,
                record,
                manifest,
                calibration,
            )
            base_bundle = None
            if not args.overwrite:
                base_bundle = load_label_independent_finalization_cache(
                    cache_path,
                    finalization_cache_key,
                    (height, width),
                    require_base_actor_sun=base_actor_sun_required,
                )
            if base_bundle is None:
                view_cache_status = "miss"
                finalization_cache_misses += 1
                base_bundle = compute_label_independent_finalization_intermediates(
                    args,
                    prepared,
                    record,
                    source,
                    frame,
                    calibration,
                    manifest,
                    render_info,
                    remap_cache,
                    frame_lidar_cache,
                    frame_ground_cache,
                    metric_depth_cache,
                    road_depth_cache,
                    receiver_depth_cache,
                )
                write_label_independent_finalization_cache(
                    cache_path,
                    finalization_cache_key,
                    base_bundle["arrays"],
                    base_bundle["diagnostics"],
                )
            else:
                view_cache_status = "hit"
                finalization_cache_hits += 1

            base_arrays = base_bundle["arrays"]
            base_diagnostics = base_bundle["diagnostics"]
            actor_rgba = base_arrays["actor_rgba"]
            actor_alpha = actor_rgba[..., 3]
            receiver_visibility = base_arrays["receiver_visibility"]
            occlusion_diagnostics = base_diagnostics["occlusion"]
            degradation_diagnostics = base_diagnostics["degradation"]
            harmonization_diagnostics = base_diagnostics["harmonization"]
            foot_contact_diagnostics = base_diagnostics["foot_contact"]
            receiver_diagnostics = base_diagnostics["receiver"]
            degradation_diagnostics["label_independent_finalization_cache"] = {
                "version": FINALIZATION_CACHE_VERSION,
                "key": finalization_cache_key,
                "path": str(cache_path),
            }
            actor_sun_source_path = actor_sun_path
            try:
                actor_sun_render_diagnostics = base_diagnostics[
                    "base_actor_sun_render"
                ]
                actor_sun_raw = base_arrays["base_actor_sun_raw"]
            except KeyError as exc:
                raise TokenFailure(
                    "render_failed",
                    "contact-only finalization requires the base actor Sun pass",
                ) from exc
            actor_sky_raw = base_arrays["actor_sky_raw"]
            actor_sky_render_diagnostics = base_diagnostics["actor_sky_render"]
            actor_sun_linear, actor_sky_linear, actor_lighting_diagnostics = (
                partition_actor_lighting(actor_rgba, actor_sun_raw, actor_sky_raw)
            )
            actor_lighting_diagnostics.update(
                {
                    "sun_render": actor_sun_render_diagnostics,
                    "sky_render": actor_sky_render_diagnostics,
                    "raw_sun_alpha_mean_on_actor": float(
                        np.mean(actor_sun_raw[..., 3][actor_alpha > 0])
                    )
                    if np.any(actor_alpha > 0)
                    else 0.0,
                    "raw_sky_alpha_mean_on_actor": float(
                        np.mean(actor_sky_raw[..., 3][actor_alpha > 0])
                    )
                    if np.any(actor_alpha > 0)
                    else 0.0,
                }
            )
            direct_sun_energy = 0.0  # contact-only: no directional Sun render
            directional_shadow_amount = np.zeros(
                actor_alpha.shape, dtype=np.float32
            )
            directional_component_diagnostics = {
                "method": "not_evaluated",
                "status": "contact_only_no_directional_render",
            }
            directional_shadow_diagnostics = {
                "method": "contact_only",
                "path": None,
            }
            directional_recovery_diagnostics = {
                "version": DIRECTIONAL_SHADOW_RECOVERY_VERSION,
                "status": "not_evaluated_no_directional_render",
                "attempted": False,
                "recovered": False,
            }
            directional_shadow_diagnostics.update(
                {
                    "semantic": "Vdir_geometric_direct_sun_occlusion",
                    "rendered_sun_energy": direct_sun_energy,
                    "decision_source": CONTACT_ONLY_SHADOW_POLICY_VERSION,
                    "receiver_visibility": receiver_diagnostics,
                    "receiver_depth_recovery": (
                        directional_recovery_diagnostics
                    ),
                    "component_filter": directional_component_diagnostics,
                    "nonzero_pixel_count": int(
                        np.count_nonzero(directional_shadow_amount > 0.0)
                    ),
                }
            )

            ambient_occlusion_diagnostics = dict(
                base_diagnostics["ambient_occlusion_render"]
            )
            contact_ambient_component_diagnostics = base_diagnostics[
                "contact_ambient_component"
            ]
            ambient_occlusion_amount = base_arrays[
                "labeled_ambient_occlusion_amount"
            ]
            ambient_component_diagnostics = {
                "method": "diffuse_body_plus_stance_contact_max_v1",
                "diffuse_body_minimum_attenuation": (
                    DIFFUSE_BODY_OCCLUSION_MIN_AMOUNT
                ),
                "diffuse_body": base_diagnostics[
                    "diffuse_body_component"
                ],
                "stance_contact": contact_ambient_component_diagnostics,
                "diffuse_body_pixel_count": base_diagnostics[
                    "diffuse_body_pixel_count"
                ],
                "stance_contact_pixel_count": base_diagnostics[
                    "stance_contact_pixel_count"
                ],
            }
            ambient_semantic = (
                "sky_only_diffuse_body_plus_stance_contact_occlusion"
            )
            ambient_occlusion_diagnostics.update(
                {
                    "semantic": ambient_semantic,
                    "receiver_visibility": receiver_diagnostics,
                    "component_filter": ambient_component_diagnostics,
                    "foot_contact": foot_contact_diagnostics,
                    "nonzero_pixel_count": int(
                        np.count_nonzero(ambient_occlusion_amount > 0.0)
                    ),
                }
            )
            (
                labeled_shadow_alpha,
                labeled_shadow_diagnostics,
            ) = compose_labeled_shadow_alpha(
                directional_shadow_amount,
                ambient_occlusion_amount,
                shadow_label_policy["directional_gain"],
                shadow_label_policy["ambient_occlusion_gain"],
            )
            composited_actor = compose_actor_lighting_rgba(
                actor_rgba,
                actor_sun_linear,
                actor_sky_linear,
                shadow_label_policy["actor_sun_gain"],
            )
            label_composition_diagnostics = {
                "enabled": True,
                "method": SHADOW_LABEL_COMPOSITION_VERSION,
                "decision_source": CONTACT_ONLY_SHADOW_POLICY_VERSION,
                "consensus_or_smoothing_applied": False,
                "shadow_label": shadow_label,
                "render_policy": shadow_label_policy,
                "directional_shadow_applied": bool(
                    shadow_label_policy["requires_directional_render"]
                ),
                "contact_ao_applied": True,
                "diffuse_body_occlusion_applied": True,
                "actor_illumination": (
                    "Osky_plus_scaled_label_selected_Osun"
                ),
                **labeled_shadow_diagnostics,
            }
            shadowed_source = render_utils.apply_shadow_linear(
                source, labeled_shadow_alpha, 1.0
            )
            composited_shadow_alpha = labeled_shadow_alpha
            actor_lighting_diagnostics.update(
                {
                    "composition_source": CONTACT_ONLY_SHADOW_POLICY_VERSION,
                    "selected_actor_sun_gain": shadow_label_policy["actor_sun_gain"],
                    "selected_actor_sun_render": str(actor_sun_source_path),
                }
            )
            edited, actor_alpha = render_utils.alpha_composite(
                shadowed_source, composited_actor
            )
            # PCTNet's default >=250 threshold selects only actor pixels; the
            # effective shadow remains below that threshold in the edit mask.
            mask = np.maximum(
                actor_alpha,
                np.minimum(composited_shadow_alpha, COMBINED_MASK_SHADOW_MAX),
            )
        else:
            edited = source.copy()
            mask = np.zeros((height, width), dtype=np.uint8)
            actor_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            actor_alpha = np.zeros((height, width), dtype=np.uint8)
            composited_shadow_alpha = np.zeros((height, width), dtype=np.uint8)
            directional_shadow_amount = np.zeros((height, width), dtype=np.float32)
            ambient_occlusion_amount = np.zeros((height, width), dtype=np.float32)
            receiver_visibility = np.zeros((height, width), dtype=np.float32)
            occlusion_diagnostics = {
                "actor_pixels": 0,
                "occluded_pixels": 0,
                "method": "not_projected",
            }
            degradation_diagnostics = {"method": "not_projected"}
            harmonization_diagnostics = {"method": "not_projected"}
            actor_lighting_diagnostics = {"method": "not_projected"}
            directional_shadow_diagnostics = {"method": "not_projected"}
            ambient_occlusion_diagnostics = {"method": "not_projected"}
            foot_contact_diagnostics = {"method": "not_projected"}
            label_composition_diagnostics = {
                "enabled": True,
                "method": "not_projected_noop",
                "decision_source": "no_projected_actor",
                "directional_shadow_applied": False,
                "contact_ao_applied": False,
            }
        output_path, mask_path = Path(record["output_path"]), Path(record["mask_path"])
        actor_mask_path = Path(record["actor_mask_path"])
        if actor_rgba.shape != (height, width, 4):
            raise TokenFailure(
                "finalization_failed",
                f"actor RGBA has shape {actor_rgba.shape}, expected {(height, width, 4)}",
            )
        if not np.array_equal(actor_rgba[..., 3], actor_alpha):
            raise TokenFailure(
                "finalization_failed",
                "actor RGBA alpha does not match the published actor mask",
            )
        save_png(output_path, edited)
        save_png(mask_path, mask)
        published_output_path = output_path
        published_mask_path = mask_path
        published_edited = edited
        published_mask = mask
        save_png(actor_mask_path, actor_alpha)
        output_paths[(frame_index, camera)] = published_output_path
        final_record = {
                **{key: value for key, value in record.items() if key not in ("raw_path", "render_info_path")},
                "output_path": str(published_output_path),
                "mask_path": (
                    str(published_mask_path)
                    if published_mask_path is not None
                    else None
                ),
                "rendered_visible": bool(np.any(published_mask)),
                "changed_pixel_count": int(
                    np.count_nonzero(np.any(published_edited != source, axis=2))
                ),
                "mask_pixel_count": int(np.count_nonzero(published_mask)),
                "actor_mask_pixel_count": int(np.count_nonzero(actor_alpha)),
                "shadow_label": shadow_label,
                "shadow_label_policy": shadow_label_policy,
                "composited_shadow_pixel_count": int(
                    np.count_nonzero(composited_shadow_alpha)
                ),
                "label_composition_diagnostics": label_composition_diagnostics,
                "directional_shadow_pixel_count": int(
                    np.count_nonzero(directional_shadow_amount > 0.0)
                ),
                "ambient_occlusion_pixel_count": int(
                    np.count_nonzero(ambient_occlusion_amount > 0.0)
                ),
                "receiver_visibility_pixel_count": int(
                    np.count_nonzero(receiver_visibility > 0.0)
                ),
                "occlusion_diagnostics": occlusion_diagnostics,
                "directional_shadow_diagnostics": directional_shadow_diagnostics,
                "ambient_occlusion_diagnostics": ambient_occlusion_diagnostics,
                "foot_contact_diagnostics": foot_contact_diagnostics,
                "actor_lighting_diagnostics": actor_lighting_diagnostics,
                "degradation_diagnostics": degradation_diagnostics,
                "harmonization_diagnostics": harmonization_diagnostics,
            }
        final_records.append(final_record)
        print(
            f"FINALIZED_VIEW {token} {camera}/{frame_index} "
            f"cache={view_cache_status} "
            f"elapsed={time.perf_counter() - view_started:.3f}s",
            flush=True,
        )

    annotation = annotation_sample(token, prepared["annotation_context"], output_paths)
    metadata = {
        "status": "success",
        "token": token,
        "asset": asset_config(prepared["asset"]),
        **(
            {
                "actor_type": "car",
                "actor_behavior": "stationary",
                "actor_speed_mps": 0.0,
                "vehicle_geometry": prepared["actor_geometry"],
            }
            if getattr(args, "car", False)
            else {
                "actor_type": prepared.get("actor_type", "pedestrian"),
                "direction_classification": prepared[
                    "direction_classification"
                ],
                "crossing_direction": prepared["direction"],
                "pedestrian_speed_mps": effective_pedestrian_speed_mps(args),
                "final_crossing_offset_m": float(
                    prepared["crossing_offsets_m"][-1]
                ),
                "crossing_offsets_m": prepared["crossing_offsets_m"].tolist(),
                "walking_velocity_global_xy_mps": prepared[
                    "walking_velocity_global_xy_mps"
                ].tolist(),
                **(
                    {"pedestrian_geometry": prepared["actor_geometry"]}
                    if prepared.get("actor_type", "pedestrian") == "pedestrian"
                    else {}
                ),
            }
        ),
        "placement": prepared["placement"],
        **placement_metadata_fields(prepared["placement"]),
        "front_bumper_offset_m": prepared["front_bumper_offset_m"],
        "actor_height_m": prepared["actor_height_m"],
        "timestamps_s": prepared["timestamps_s"].tolist(),
        "render_pipeline_version": RENDER_PIPELINE_VERSION,
        "background_policy": "source_files_untouched_and_referenced_per_record",
        "output_composition": {
            "mode": SHADOW_LABEL_COMPOSITION_VERSION,
            "preview": False,
            "image_tree": "images",
            "mask_tree": "masks",
            "decision_policy": "apply_uniform_contact_only_to_all_selected_views",
            "directional_shadow": "disabled_by_uniform_contact_only_policy",
            "actor_illumination": "Osky_only",
            "ambient_occlusion": (
                "sun_off_diffuse_body_plus_wheel_contact_max"
                if getattr(args, "car", False)
                else "sun_off_diffuse_body_plus_stance_contact_max"
            ),
            "ambient_occlusion_gains": (
                SHADOW_LABEL_AMBIENT_OCCLUSION_GAINS
            ),
            "directional_gains": SHADOW_LABEL_DIRECTIONAL_GAINS,
            "actor_sun_gains": SHADOW_LABEL_ACTOR_SUN_GAINS,
            "sun_angles_degrees": (
                SHADOW_LABEL_SUN_ANGLES_DEGREES
            ),
            "counterfactual_sun_energy": (
                SHADOW_LABEL_COUNTERFACTUAL_SUN_ENERGY
            ),
            "directional_shadow_receiver_recovery": (
                DIRECTIONAL_SHADOW_RECOVERY_VERSION
            ),
            "shadow_policy_source": args.shadow_label_metadata,
        },
        "shadow_data_contract": {
            "raw_render_storage": RAW_RENDER_STORAGE_VERSION,
            "base_render_pass_policy": BASE_RENDER_PASS_POLICY_VERSION,
            "label_independent_finalization_cache": (
                FINALIZATION_CACHE_VERSION
            ),
            "actor_mask": "uint8_alpha_M",
            "raw_sky_catcher": "scene_linear_half_float_rgba",
            "label_directional_catcher": (
                "scene_linear_half_float_rgba_when_required_by_label"
            ),
            "directional_shadow_receiver_recovery": (
                DIRECTIONAL_SHADOW_RECOVERY_VERSION
            ),
            "receiver_matte_actor_edge_overlap": (
                f"receiver_capped_outer_{RECEIVER_MATTE_ACTOR_EDGE_RADIUS_PX}px_"
                f"alpha_below_{RECEIVER_MATTE_OPAQUE_ALPHA}"
            ),
            "combined_edit_mask": (
                "max_actor_policy_selected_vdir_and_sky_occlusion_with_"
                "shadow_capped_at_249"
            ),
            "shadow_policy": "apply_uniform_contact_only_to_all_selected_views",
        },
        "grounding_method": (
            "single_rigid_world_pose_from_final_history_ground_v1"
            if getattr(args, "car", False)
            else GROUNDING_METHOD
        ),
        "final_road_plane_z_ax_by_c_in_lidar": prepared["road_plane"].tolist(),
        "final_road_basis_columns_in_lidar": prepared["road_basis"].tolist(),
        "final_gravity_up_in_lidar": prepared["gravity_up_lidar"].tolist(),
        "max_ground_tilt_degrees": args.max_ground_tilt_deg,
        "ground_diagnostics": prepared["ground_diagnostics"],
        "collision_diagnostics": prepared["collision_diagnostics"],
        "potential_failures": [
            *prepared["potential_failures"],
            *finalization_potential_failures,
        ],
        "lighting": prepared["lighting"],
        "shadow_policy_source": args.shadow_label_metadata,
        "annotation_sample": annotation,
        "records": final_records,
    }
    write_json(run_dir / "metadata" / f"{token}.json", metadata)
    print(
        f"FINALIZATION_CACHE {token}: {finalization_cache_hits} hit(s), "
        f"{finalization_cache_misses} miss(es)"
    )
    return metadata


def record_failure(run_dir: Path, token: str, code: str, message: str, args: argparse.Namespace):
    failure = {
        "status": "failed",
        "token": token,
        "code": code,
        "message": message,
        **placement_request(args),
        "speed_mps": effective_pedestrian_speed_mps(args),
    }
    if getattr(args, "car", False):
        failure["actor_type"] = "car"
    write_json(run_dir / "failures" / f"{token}.json", failure)
    print(f"SKIP {token}: {code}: {message}", file=sys.stderr)


def normalize_potential_failures(value: Any) -> List[dict]:
    """A token's potential-failure warnings (a list of objects)."""
    if value is None:
        return []
    if isinstance(value, list) and all(
        isinstance(item, dict) for item in value
    ):
        return list(value)
    raise ValueError("potential-failure data must be a list of objects")


def deduplicate_potential_failures(warnings: Sequence[dict]) -> List[dict]:
    unique = []
    seen = set()
    for warning in warnings:
        key = canonical_hash(warning)
        if key in seen:
            continue
        seen.add(key)
        unique.append(warning)
    return unique


def record_potential_failure(run_dir: Path, warning: dict) -> None:
    token = warning["token"]
    warning_path = run_dir / "potential_failures" / f"{token}.json"
    existing = []
    if warning_path.is_file():
        existing = normalize_potential_failures(
            json.loads(warning_path.read_text())
        )
    write_json(
        warning_path,
        deduplicate_potential_failures([*existing, warning]),
    )
    print(
        f"POTENTIAL_FAILURE {token}: {warning['code']}: {warning['message']}",
        file=sys.stderr,
    )


def prepare_finalization_tokens(
    args: argparse.Namespace,
    loader,
    prepared_tokens: Sequence[dict],
    run_dir: Path,
) -> List[dict]:
    ready = []
    for prepared in prepared_tokens:
        token = prepared["token"]
        try:
            if "annotation_context" not in prepared:
                prepared["annotation_context"] = annotation_context(loader, token)
            ready.append(prepared)
        except Exception as exc:
            record_failure(
                run_dir,
                token,
                "finalization_failed",
                repr(exc),
                args,
            )
    return ready


def finalize_prepared_token(
    args: argparse.Namespace,
    prepared: dict,
    run_dir: Path,
    remap_cache: dict,
) -> None:
    token = prepared["token"]
    started = time.perf_counter()
    status = "failed"
    print(
        f"FINALIZATION_START {token} views={len(prepared['view_records'])}",
        flush=True,
    )
    try:
        finalize_token(args, prepared, run_dir, remap_cache)
        failure_path = run_dir / "failures" / f"{token}.json"
        if failure_path.exists():
            failure_path.unlink()
        status = "success"
        print(f"DONE {token}", flush=True)
    except TokenFailure as exc:
        record_failure(run_dir, token, exc.code, str(exc), args)
    except Exception as exc:
        record_failure(
            run_dir,
            token,
            "finalization_failed",
            repr(exc),
            args,
        )
    finally:
        print(
            f"FINALIZATION_END {token} status={status} "
            f"elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )


def run_blender_with_streaming_finalization(
    args: argparse.Namespace,
    loader,
    prepared_tokens: Sequence[dict],
    run_dir: Path,
    jobs: List[dict],
) -> None:
    """Finalize ready tokens in a bounded pool while Blender keeps rendering."""
    ready = prepare_finalization_tokens(args, loader, prepared_tokens, run_dir)
    ready_by_token = {prepared["token"]: prepared for prepared in ready}
    pending_job_ids = {
        token: {job["job_id"] for job in prepared["jobs"]}
        for token, prepared in ready_by_token.items()
    }
    job_owner = {
        job["job_id"]: token
        for token, prepared in ready_by_token.items()
        for job in prepared["jobs"]
    }
    submitted_tokens = set()
    finalization_futures = {}
    worker_count = min(
        len(ready),
        max(1, int(getattr(args, "finalize_workers", 4))),
    )
    if ready:
        print(
            f"Finalizing {len(ready)} tokens with {worker_count} bounded workers "
            "as their Blender renders complete"
        )

    def finalize_ready_tokens(executor: ThreadPoolExecutor) -> None:
        for prepared in ready:
            token = prepared["token"]
            if token in submitted_tokens or pending_job_ids[token]:
                continue
            # Mark before submitting so a caught token failure is not retried
            # after every subsequent render-batch completion. Each token owns
            # a remap cache: OpenCV/NumPy arrays are never shared mutably across
            # finalization workers and are released when that token completes.
            submitted_tokens.add(token)
            future = executor.submit(
                finalize_prepared_token,
                args,
                prepared,
                run_dir,
                {},
            )
            finalization_futures[future] = token

    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        def batch_complete(result: dict) -> None:
            for job_id in result["job_ids"]:
                token = job_owner.get(job_id)
                if token is not None:
                    pending_job_ids[token].discard(job_id)
            finalize_ready_tokens(executor)

        run_blender_batches(
            args,
            run_dir,
            jobs,
            on_batches_submitted=lambda: finalize_ready_tokens(executor),
            on_batch_complete=batch_complete,
        )
        finalize_ready_tokens(executor)

        # Executor context shutdown also waits, but resolving futures here
        # preserves unexpected scheduler exceptions with their token identity.
        for future in as_completed(finalization_futures):
            token = finalization_futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"uncaught finalization worker failure for {token}"
                ) from exc

    incomplete = {
        token: sorted(job_ids)
        for token, job_ids in pending_job_ids.items()
        if token not in submitted_tokens and job_ids
    }
    if incomplete:
        raise RuntimeError(
            f"Blender scheduler returned without completing token jobs: {incomplete}"
        )


def aggregate_results(run_dir: Path, tokens: Sequence[str]):
    annotations = []
    failures = []
    potential_failures = []
    for token in tokens:
        metadata_path = run_dir / "metadata" / f"{token}.json"
        failure_path = run_dir / "failures" / f"{token}.json"
        potential_failure_path = run_dir / "potential_failures" / f"{token}.json"
        metadata = None
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            annotations.append(metadata["annotation_sample"])
        elif failure_path.is_file():
            failures.append(json.loads(failure_path.read_text()))
        token_potential_failures = []
        if potential_failure_path.is_file():
            token_potential_failures.extend(
                normalize_potential_failures(
                    json.loads(potential_failure_path.read_text())
                )
            )
        if metadata is not None:
            token_potential_failures.extend(
                normalize_potential_failures(
                    metadata.get("potential_failures", [])
                )
            )
        potential_failures.extend(
            deduplicate_potential_failures(token_potential_failures)
        )
    potential_failures = deduplicate_potential_failures(
        potential_failures
    )
    write_json(run_dir / "annotations.json", annotations)
    failure_output = run_dir / "failures.jsonl"
    failure_output.write_text("".join(json.dumps(item) + "\n" for item in failures))
    write_json(run_dir / "potential_failures.json", potential_failures)
    write_json(
        run_dir / "summary.json",
        {
            "selected_tokens": len(tokens),
            "successful_tokens": len(annotations),
            "failed_tokens": len(failures),
            "potential_failure_tokens": len(
                {item["token"] for item in potential_failures}
            ),
        },
    )
    return len(annotations), len(failures)


def render_prepared(args, run_dir, prepared_tokens, tokens):
    for name in ("render_workers", "render_batch_size", "finalize_workers"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    configure_navsim(args.data_root)
    if not args.blender.is_file():
        raise FileNotFoundError(f"Blender not found: {args.blender}")
    configure_finalization_native_threads(args.finalize_workers)
    jobs = edit_preparation.refresh_render_jobs(args, prepared_tokens, sys.modules[__name__])
    run_depth_batch(args, run_dir, prepared_tokens)
    run_blender_with_streaming_finalization(args, None, prepared_tokens, run_dir, jobs)
    successful, failed = aggregate_results(run_dir, tokens)
    print(f"Finished: {successful} successful, {failed} failed")
    print(f"Output: {run_dir}")


def main():
    args = parse_args()
    benign_sides.configure(args, sys.modules[__name__])
    validate_actor_mode(args)
    placement_request(args)
    stage = getattr(args, "stage", "all")
    store = None
    if stage != "all":
        run_dir = args.output_root / args.scene_filter.stem / args.run_name
        if (run_dir / "compaction_report.json").exists():
            raise RuntimeError("This run has entered evaluation-only compaction; choose a new run name")
        request = edit_preparation.request_identity(args, [
            __file__, geometry.__file__, render_utils.__file__,
            benign_placements.__file__, front_visibility.__file__, benign_sides.__file__, edit_preparation.__file__,
            HERE.parent / "animal_composite_navsim_blender.py",
        ])
        store = edit_preparation.PreparationStore(run_dir, "navsim", request)
    if stage == "render":
        args, prepared_tokens, index = store.load(args)
        render_prepared(args, run_dir, prepared_tokens, index["tokens"])
        return
    configure_direction_classifications(args)
    # Contact-only shadows are the only shadow mode.
    args.shadow_label_metadata = contact_only_shadow_metadata()
    # Valid base renders/finalization caches may be reused, but derived products
    # are always rebuilt.
    args.refinalize = True
    if args.speed_mps <= 0:
        raise ValueError("pedestrian speed must be positive")
    if args.gait_frame_advance <= 0:
        raise ValueError("--gait-frame-advance must be positive")
    if args.gait_timestep_s <= 0:
        raise ValueError("--gait-timestep-s must be positive")
    if args.actor_blur_sigma < 0 or args.max_actor_noise_std < 0:
        raise ValueError("Camera degradation parameters must be non-negative")
    if not 0.0 <= args.harmonization_strength <= 1.0:
        raise ValueError("--harmonization-strength must be in [0, 1]")
    if not 0.0 <= args.max_ground_tilt_deg <= 15.0:
        raise ValueError("--max-ground-tilt-deg must be in [0, 15]")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.render_batch_size < 1:
        raise ValueError("--render-batch-size must be at least 1")
    if args.render_workers < 1:
        raise ValueError("--render-workers must be at least 1")
    if args.finalize_workers < 1:
        raise ValueError("--finalize-workers must be at least 1")
    finalization_native_threads = configure_finalization_native_threads(
        args.finalize_workers
    )
    print(
        f"Finalization CPU budget: {allocated_cpu_count()} allocated CPU(s), "
        f"{args.finalize_workers} token worker(s), "
        f"up to {finalization_native_threads} OpenCV thread(s) per operation"
    )
    if stage != "prepare" and not args.blender.is_file():
        raise FileNotFoundError(f"Blender not found: {args.blender}")

    configure_navsim(args.data_root)
    filter_config = yaml.safe_load(args.scene_filter.read_text())
    loader, tokens = build_loader(args, filter_config)
    if not getattr(args, "standing", False) and not getattr(args, "car", False) and not getattr(args, "benign", False):
        validate_direction_classifications(
            tokens,
            args.direction_classification_labels,
        )
    run_name = args.run_name
    run_dir = args.output_root / args.scene_filter.stem / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    assets = load_asset_pool(args)
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    ego_vehicle = get_pacifica_parameters()
    front_bumper_offset = float(ego_vehicle.front_length)
    args.ego_vehicle_geometry = {
        "front_length_m": front_bumper_offset,
        "rear_length_m": float(ego_vehicle.rear_length),
        "width_m": float(ego_vehicle.width),
    }
    run_config = {
        "scene_filter": str(args.scene_filter.resolve()),
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "asset_mode": "token_random",
        "asset_directory": str(args.asset_dir.resolve()),
        "asset_selection": "blake2b_token_seed_v1",
        "asset_pool": [asset_config(asset) for asset in assets],
        **placement_request(args),
        **direction_classification_provenance(args),
        "speed_mps": effective_pedestrian_speed_mps(args),
        "timed_interception_policy": (
            "stationary_car_rear_bumper=future_ego_front_bumper"
            if getattr(args, "car", False)
            else "position(t)=future_front_bumper"
            if getattr(args, "standing", False)
            else (
                "position(t)=future_front_bumper+walking_velocity*"
                "(t-future_timestamp)"
            )
        ),
        "seed": args.seed,
        "stride_length_m": args.stride_length_m,
        "stride_length_source": "explicit" if args.stride_length_m is not None else "per_asset_derived_from_frame_advance",
        "gait_frame_advance": args.gait_frame_advance,
        "gait_timestep_s": args.gait_timestep_s,
        "gait_initial_frame": NAVSIM_POSE_SEQUENCE[0],
        "navsim_pose_sequence": list(NAVSIM_POSE_SEQUENCE),
        "render_pipeline_version": RENDER_PIPELINE_VERSION,
        "base_render_pass_policy": BASE_RENDER_PASS_POLICY_VERSION,
        "grounding_method": GROUNDING_METHOD,
        "sun_direction": "noaa_geometric_sun_geodesic_utm_v1",
        "sun_weather_gate": (
            "second_view_blue_sky_or_hard_contrast_consensus_v2"
        ),
        "shadow_intermediate": "sky_ao_plus_label_selected_directional_exr_v1",
        "separated_lighting_layers": True,
        "actor_lighting_intermediate": (
            "sky_only_plus_sun_only_when_required_scene_linear_half_float_exr"
        ),
        "shadow_receiver_visibility": "projected_receiver_plus_dense_depth_consistency_v1",
        "directional_shadow_receiver_recovery": (
            DIRECTIONAL_SHADOW_RECOVERY_VERSION
        ),
        "shadow_receiver_max_depth_m": SHADOW_RECEIVER_MAX_DEPTH_M,
        "shadow_minimum_attenuation": SHADOW_MIN_AMOUNT,
        "ambient_occlusion_minimum_attenuation": AMBIENT_OCCLUSION_MIN_AMOUNT,
        "combined_mask_shadow_maximum": COMBINED_MASK_SHADOW_MAX,
        "receiver_matte_actor_edge_policy": (
            "receiver_capped_measured_wheel_edge_extrapolation_v1"
            if getattr(args, "car", False)
            else "receiver_capped_measured_stance_foot_edge_extrapolation_v2"
        ),
        "receiver_matte_actor_edge_radius_px": (
            RECEIVER_MATTE_ACTOR_EDGE_RADIUS_PX
        ),
        "receiver_matte_opaque_alpha_threshold": RECEIVER_MATTE_OPAQUE_ALPHA,
        "foot_contact_geometry": (
            "xoma_axle_wheel_contact_patches_v1"
            if getattr(args, "car", False)
            else "evaluated_foot_toe_vertex_groups_v1"
        ),
        "foot_contact_distance_m": FOOT_CONTACT_DISTANCE_M,
        "foot_stance_relative_distance_m": FOOT_STANCE_RELATIVE_DISTANCE_M,
        "foot_sole_band_m": FOOT_SOLE_BAND_M,
        "distortion_rendering": "exact_ideal_overscan_v1",
        "cycles_render_region": (
            "actor_ao_base_or_label_directional_crop_to_border_v3"
        ),
        "raw_render_storage": RAW_RENDER_STORAGE_VERSION,
        "label_independent_finalization_cache": FINALIZATION_CACHE_VERSION,
        "cycles_render_region_padding_px": RENDER_REGION_PADDING_PX,
        "actor_occlusion_depth": "cycles_z_pass_16bit_v1",
        "source_depth_alignment": "depth_anything_affine_inverse_depth_only_v1",
        "near_total_occlusion_policy": "fail_open_without_lidar_support_v1",
        "samples": args.samples,
        "render_scale": args.render_scale,
        "actor_blur_sigma": args.actor_blur_sigma,
        "max_actor_noise_std": args.max_actor_noise_std,
        "motion_blur": (
            False
            if getattr(args, "standing", False) or getattr(args, "car", False)
            else not args.disable_motion_blur
        ),
        "device": args.device,
        "lighting": "auto",
        "harmonization": "local",
        "harmonization_strength": args.harmonization_strength,
        "shadows": "cycles",
        "max_ground_tilt_degrees": args.max_ground_tilt_deg,
        "occlusion": "dense",
        "collision_check": not args.disable_collision_check,
        "cameras": list(CAMERA_ORDER),
        "history_frames": list(HISTORY_FRAME_INDICES),
    }
    if getattr(args, "car", False):
        run_config.update(
            {
                "actor_behavior": "stationary",
                "actor_type": "car",
                "actor_root_name": CAR_ROOT_OBJECT_NAME,
                "actor_geometry": CAR_GEOMETRY_VERSION,
                "actor_contact_profile": CAR_CONTACT_PROFILE,
                "ego_vehicle_geometry": args.ego_vehicle_geometry,
                "first_contact_tolerance_m": CAR_CONTACT_TOLERANCE_M,
                "grounding_method": "single_rigid_world_pose_from_final_history_ground_v1",
            }
        )
        for pedestrian_only_key in (
            "stride_length_m",
            "stride_length_source",
            "gait_frame_advance",
            "gait_timestep_s",
            "gait_initial_frame",
            "navsim_pose_sequence",
        ):
            run_config.pop(pedestrian_only_key, None)
    if getattr(args, "benign", False):
        run_config["timed_interception_policy"] = "lateral_offset_with_full_path_clearance"
        run_config.pop("first_contact_tolerance_m", None)
    run_config["shadow_label_composition"] = {
        **args.shadow_label_metadata,
        "composition_version": SHADOW_LABEL_COMPOSITION_VERSION,
        "directional_render_version": (
            SHADOW_LABEL_DIRECTIONAL_RENDER_VERSION
        ),
        "counterfactual_sun_energy": (
            SHADOW_LABEL_COUNTERFACTUAL_SUN_ENERGY
        ),
        "directional_gains": SHADOW_LABEL_DIRECTIONAL_GAINS,
        "ambient_occlusion_gains": (
            SHADOW_LABEL_AMBIENT_OCCLUSION_GAINS
        ),
        "actor_sun_gains": SHADOW_LABEL_ACTOR_SUN_GAINS,
        "sun_angles_degrees": SHADOW_LABEL_SUN_ANGLES_DEGREES,
        "diffuse_body_occlusion_minimum_attenuation": (
            DIFFUSE_BODY_OCCLUSION_MIN_AMOUNT
        ),
    }
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text())
        comparison_keys = set(run_config)
        if "preview" in existing:
            comparison_keys.add("preview")
        if "shadow_label_composition" in existing:
            comparison_keys.add("shadow_label_composition")
        changed_keys = {
            key
            for key in comparison_keys
            if (
                existing.get(key, False)
                if key == "preview"
                else existing.get(key)
            )
            != (
                run_config.get(key, False)
                if key == "preview"
                else run_config.get(key)
            )
        }
        refinalizable_keys = {
            "actor_blur_sigma",
            "max_actor_noise_std",
            "harmonization_strength",
            "shadow_minimum_attenuation",
            "ambient_occlusion_minimum_attenuation",
            "combined_mask_shadow_maximum",
            "receiver_matte_actor_edge_policy",
            "receiver_matte_actor_edge_radius_px",
            "receiver_matte_opaque_alpha_threshold",
            "directional_shadow_receiver_recovery",
            "shadow_label_composition",
        }
        if changed_keys and not (
            args.refinalize and changed_keys <= set(refinalizable_keys)
        ):
            raise RuntimeError(f"Run configuration conflicts with existing {config_path}; choose a different --run-name")
        if changed_keys:
            write_json(config_path, run_config)
    else:
        write_json(config_path, run_config)

    if store is not None:
        log_root = args.data_root / "navsim_logs" / args.split
        log_names = {str(loader.scene_frames_dicts[token][0].get("log_name", "")) for token in tokens}
        log_paths = [log_root / f"{name}.pkl" for name in log_names if name]
        store.begin(tokens, args, inputs=[path for path in log_paths if path.is_file()])
    prepared_tokens = []
    jobs = []
    for index, token in enumerate(tokens):
        metadata_path = run_dir / "metadata" / f"{token}.json"
        failure_path = run_dir / "failures" / f"{token}.json"
        if metadata_path.is_file() and not args.overwrite and not args.refinalize:
            print(f"[{index + 1}/{len(tokens)}] cached success {token}")
            if store is not None:
                store.skip(token, "cached_success", [metadata_path])
            continue
        if failure_path.is_file() and not args.retry_failures and not args.overwrite:
            print(f"[{index + 1}/{len(tokens)}] cached failure {token}")
            if store is not None:
                store.skip(token, "failed", [failure_path])
            continue
        try:
            asset = select_asset(token, args.seed, assets)
            prepared = None
            if store is not None and not args.overwrite:
                try:
                    prepared = store.restore(token)
                except edit_preparation.PreparationError as exc:
                    print(f"Rebuilding {token}: {exc}")
            if prepared is None:
                prepared = prepare_token(args, loader, token, run_dir, front_bumper_offset, asset)
                if store is not None:
                    prepared["annotation_context"] = annotation_context(loader, token)
            potential_failure_path = run_dir / "potential_failures" / f"{token}.json"
            # Re-preparation is the start of a fresh warning evaluation. Clear
            # stale finalization warnings, then append the current preparation
            # and finalization warnings through the list-aware writer.
            if potential_failure_path.exists():
                potential_failure_path.unlink()
            for warning in prepared["potential_failures"]:
                record_potential_failure(run_dir, warning)
            if store is None:
                prepared_tokens.append(prepared)
                jobs.extend(prepared["jobs"])
            print(
                f"[{index + 1}/{len(tokens)}] prepared {token}: {prepared['direction']}, "
                f"asset={asset['id']}, {len(prepared['jobs'])} renders"
            )
        except TokenFailure as exc:
            record_failure(run_dir, token, exc.code, str(exc), args)
            prepared = None
        except OSError as exc:
            if store is not None:
                raise
            record_failure(run_dir, token, "preparation_failed", repr(exc), args)
            prepared = None
        except Exception as exc:
            record_failure(run_dir, token, "preparation_failed", repr(exc), args)
            prepared = None
        if store is not None:
            if prepared is None:
                store.skip(token, "failed", [failure_path])
            else:
                store.save(prepared, edit_preparation.prepared_input_paths(prepared, args.data_root, args.split))

    if stage == "prepare":
        store.finish()
        return

    print(f"Rendering {len(jobs)} uncached views for {len(prepared_tokens)} tokens")
    # Depth must be available before the first completed token can finalize.
    # Once Blender starts, all render batches are submitted up front and the
    # coordinator consumes completed tokens while the workers keep rendering.
    run_depth_batch(args, run_dir, prepared_tokens)
    run_blender_with_streaming_finalization(
        args,
        loader,
        prepared_tokens,
        run_dir,
        jobs,
    )

    successful, failed = aggregate_results(run_dir, tokens)
    print(f"Finished: {successful} successful, {failed} failed")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
