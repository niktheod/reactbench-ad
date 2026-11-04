#!/usr/bin/env python3
"""Classify PAI-AV clips with strict compositor pre-render checks.

The classifier never renders or edits an image. For every selected clip it
checks a stationary car at the requested future horizons, followed by one
deterministically selected pedestrian crossing direction. The alternate
direction is checked only when the first direction fails.

The default PAI-AV horizons ``20, 15, 10, 5`` are physically equivalent to
NAVSIM horizons ``4, 3, 2, 1``: both represent 2.0, 1.5, 1.0, and 0.5 seconds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


# One native thread per process prevents NumPy/OpenCV oversubscription when the
# classifier is launched without the Slurm wrapper.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent

from . import edit as paiav_edit
from . import ego_masks
from .paiav_io import (
    CAMERA_NAMES,
    HISTORY_FRAME_COUNT,
    HISTORY_INTERVAL_US,
    PAIAVSceneLoader,
    frame_source_path,
    load_lidar_xyz,
    load_selection_manifest,
)


compositor = paiav_edit.compositor

SCHEMA_VERSION = 1
PIPELINE_VERSION = "full_paiav_preflight_classifier_v2"
DEFAULT_TIMESTEPS = (20, 15, 10, 5)
DIRECTIONS = ("left", "right")
EGO_MASK_OVERLAP_POLICIES = ("auto", "ignore", "report", "fail")
EGO_MASK_OVERLAP_METHOD = "projected_grounded_actor_bbox_corner_hull_v1"
INITIAL_DIRECTION_NAMESPACE = "paiav_sample_classifier_initial_direction"
INITIAL_DIRECTION_METHOD = "blake2b_clip_seed_uniform_v1"
DEFAULT_OUTPUT_DIR = HERE / "classification_results"
DEFAULT_SCRATCH_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
DIRECTION_SPECS = {
    label: {
        "classification": label,
        "crossing_name": str(spec["crossing_name"]),
        "lateral_sign": float(spec["lateral_sign"]),
        "actor_heading_rad": float(spec["actor_heading_rad"]),
    }
    for label, spec in compositor.DIRECTION_CLASSIFICATION_SPECS.items()
}


class ClassificationError(RuntimeError):
    """A run-level invariant failed; ordinary clip failures do not use it."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_worker_count() -> int:
    configured = os.environ.get("SLURM_CPUS_PER_TASK")
    if configured and configured.isdigit() and int(configured) > 0:
        return int(configured)
    return max(1, min(12, os.cpu_count() or 1))


def normalize_timesteps(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("--timesteps requires at least one value")
    result = []
    seen = set()
    for raw in values:
        if isinstance(raw, bool) or int(raw) < 1:
            raise ValueError("--timesteps values must be positive integers")
        value = int(raw)
        if value in seen:
            raise ValueError(f"future timestep {value} was supplied more than once")
        seen.add(value)
        result.append(value)
    return tuple(result)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=paiav_edit.DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=paiav_edit.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument(
        "--pedestrian-assets",
        type=Path,
        default=paiav_edit.DEFAULT_ASSET_DIR,
    )
    parser.add_argument(
        "--car-assets",
        type=Path,
        default=paiav_edit.DEFAULT_CAR_ASSET_DIR,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=list(DEFAULT_TIMESTEPS),
        help="Future PAI-AV 0.1-second steps, evaluated in the supplied order.",
    )
    parser.add_argument("--anchor-time-s", type=float, default=5.1)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--speed-mps", type=float, default=1.4)
    parser.add_argument("--stride-length-m", type=float)
    parser.add_argument("--gait-frame-advance", type=int, default=23)
    parser.add_argument("--gait-timestep-s", type=float, default=0.5)
    parser.add_argument("--max-ground-tilt-deg", type=float, default=2.0)
    parser.add_argument(
        "--min-projected-bbox-area",
        type=float,
        default=64.0,
        help="Minimum clipped actor-box area for a camera view to count.",
    )
    parser.add_argument(
        "--ego-mask-overlap-policy",
        choices=EGO_MASK_OVERLAP_POLICIES,
        default="auto",
        help=(
            "auto reports refined GLS overlap and fails dataset-mask overlap. "
            "How projected grounded actor bounding envelopes that intersect "
            "calibrated ego masks affect classification."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard this configuration's checkpoint and final outputs.",
    )
    ego_masks.add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        args.timesteps = normalize_timesteps(args.timesteps)
    except ValueError as exc:
        parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.max_clips is not None and args.max_clips < 0:
        parser.error("--max-clips must be non-negative")
    if not math.isfinite(args.anchor_time_s) or args.anchor_time_s < 1.5:
        parser.error("--anchor-time-s must be finite and at least 1.5")
    if not math.isfinite(args.speed_mps) or args.speed_mps <= 0:
        parser.error("--speed-mps must be finite and positive")
    if args.stride_length_m is not None and (
        not math.isfinite(args.stride_length_m) or args.stride_length_m <= 0
    ):
        parser.error("--stride-length-m must be finite and positive")
    if args.gait_frame_advance < 1:
        parser.error("--gait-frame-advance must be positive")
    if not math.isfinite(args.gait_timestep_s) or args.gait_timestep_s <= 0:
        parser.error("--gait-timestep-s must be finite and positive")
    if not 0 <= args.max_ground_tilt_deg <= 15:
        parser.error("--max-ground-tilt-deg must be in [0, 15]")
    if (
        not math.isfinite(args.min_projected_bbox_area)
        or args.min_projected_bbox_area < 0
    ):
        parser.error("--min-projected-bbox-area must be finite and non-negative")
    return args


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def json_line(value: Any) -> str:
    return json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                json_ready(value),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json_line(record))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def asset_identity(asset: Mapping[str, Any]) -> dict:
    result = {
        "id": str(asset["id"]),
        "actor_type": str(asset.get("actor_type", "unknown")),
        "path": str(Path(asset["path"]).resolve()),
        "sha256": str(asset["sha256"]),
        "metadata_path": str(Path(asset["metadata_path"]).resolve()),
        "metadata_sha256": str(asset["metadata_sha256"]),
        "actor_height_m": float(asset["actor_height_m"]),
        "actor_geometry": json_ready(asset.get("actor_geometry")),
    }
    if asset.get("actor_type") == "car":
        preparation = asset.get("preparation", {})
        result["preparation_path"] = str(preparation.get("path", ""))
        result["attribution"] = json_ready(asset.get("attribution"))
    else:
        result.update(
            {
                "stride_length_m": float(asset["stride_length_m"]),
                "action_range": list(asset["action_range"]),
                "animation_version": str(asset["animation_version"]),
            }
        )
    return result


def load_assets(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    # Installing the adapter first makes the asset pose contract use PAI-AV's
    # fixed four-frame sequence. Candidate grounding itself calls the PAI-AV
    # adapter function directly below.
    paiav_edit.install_paiav_adapter("human", "crossing")
    pedestrian_args = argparse.Namespace(
        asset_dir=args.pedestrian_assets.expanduser().resolve(),
        car=False,
        stride_length_m=args.stride_length_m,
        speed_mps=float(args.speed_mps),
        gait_timestep_s=float(args.gait_timestep_s),
        gait_frame_advance=int(args.gait_frame_advance),
    )
    car_args = argparse.Namespace(
        asset_dir=args.car_assets.expanduser().resolve(),
        car=True,
    )
    pedestrians = compositor.load_asset_pool(pedestrian_args)
    cars = compositor.load_asset_pool(car_args)
    invalid_pedestrians = [
        asset["id"]
        for asset in pedestrians
        if asset.get("actor_type") != "pedestrian"
        or "actor_geometry" not in asset
    ]
    invalid_cars = [
        asset["id"]
        for asset in cars
        if asset.get("actor_type") != "car" or "actor_geometry" not in asset
    ]
    if invalid_pedestrians:
        raise ClassificationError(
            "Pedestrian pool contains incompatible assets: "
            f"{invalid_pedestrians[:5]}"
        )
    if invalid_cars:
        raise ClassificationError(
            f"Car pool contains incompatible assets: {invalid_cars[:5]}"
        )
    return pedestrians, cars


def diagnostic(code: str, message: str, **details: Any) -> dict:
    return {"code": str(code), "message": str(message), **details}


def exception_diagnostic(exc: Exception, **details: Any) -> dict:
    code = exc.code if isinstance(exc, compositor.TokenFailure) else "evaluation_failed"
    return diagnostic(
        code,
        str(exc),
        exception_type=type(exc).__name__,
        **details,
    )


def compact_failure(value: Any) -> dict:
    if not isinstance(value, Mapping):
        return {"code": "invalid_failure_diagnostic", "message": str(value)}
    result = {
        "code": str(value.get("code") or "candidate_failed"),
        "message": str(value.get("message") or "candidate failed"),
    }
    for key in (
        "exception_type",
        "failed_frame_indices",
        "occupied_frame_indices",
        "future_timestep",
        "direction",
    ):
        if key in value:
            result[key] = json_ready(value[key])
    strict_failures = value.get("strict_failures")
    if isinstance(strict_failures, list):
        result["strict_failures"] = [
            {
                key: json_ready(item[key])
                for key in ("frame_index", "code", "message", "exception_type")
                if isinstance(item, Mapping) and key in item
            }
            for item in strict_failures
        ]
    swept = value.get("swept_collision")
    if isinstance(swept, Mapping):
        result["swept_collision"] = {
            key: json_ready(swept[key])
            for key in (
                "method",
                "occupied",
                "maximum_obstacle_points",
                "collision_threshold_points",
                "sample_count",
            )
            if key in swept
        }
    ego_mask_overlap = value.get("ego_mask_overlap")
    if isinstance(ego_mask_overlap, Mapping):
        result["ego_mask_overlap"] = json_ready(ego_mask_overlap)
    return result


def compact_candidate(candidate: Mapping[str, Any]) -> dict:
    passed = bool(candidate.get("passed", False))
    failures = [compact_failure(item) for item in candidate.get("failures", [])]
    if not passed and not failures:
        failures = [
            {
                "code": "candidate_failed_without_diagnostic",
                "message": "candidate returned passed=false without a reason",
            }
        ]
    result = {
        "evaluation_complete": bool(candidate.get("evaluation_complete", True)),
        "passed": passed,
        "failures": failures,
    }
    for key in (
        "visible_view_count",
        "fully_visible_view_count",
        "ground_support_frame_indices",
        "strict_ground_failure_frame_indices",
    ):
        if key in candidate:
            result[key] = json_ready(candidate[key])
    collision = candidate.get("collision_diagnostics")
    if isinstance(collision, Mapping):
        result["collision_occupied"] = bool(collision.get("occupied", False))
    placement = candidate.get("placement")
    if isinstance(placement, Mapping):
        result["future_offset_s"] = json_ready(placement.get("future_offset_s"))
    ego_mask_overlap = candidate.get("ego_mask_overlap")
    if isinstance(ego_mask_overlap, Mapping):
        result["ego_mask_overlap"] = json_ready(ego_mask_overlap)
    return result


def initial_direction_choice(clip_id: str, seed: int) -> tuple[str, float]:
    uniform_value = compositor.token_uniform(
        str(clip_id), int(seed), INITIAL_DIRECTION_NAMESPACE
    )
    return ("left" if uniform_value < 0.5 else "right", float(uniform_value))


def candidate_slots(timesteps: Sequence[int]) -> list[dict]:
    slots = []
    for timestep in timesteps:
        slots.append(
            {
                "candidate_id": f"car_t{int(timestep)}",
                "mode": "standing_car",
                "future_timestep": int(timestep),
                "future_offset_s": float(int(timestep) * 0.1),
                "status": "not_evaluated_short_circuit",
                "skip_reason": "not_reached",
            }
        )
    for direction in DIRECTIONS:
        for timestep in timesteps:
            slots.append(
                {
                    "candidate_id": f"crossing_{direction}_t{int(timestep)}",
                    "mode": "crossing_pedestrian",
                    "direction": direction,
                    "crossing_name": DIRECTION_SPECS[direction]["crossing_name"],
                    "future_timestep": int(timestep),
                    "future_offset_s": float(int(timestep) * 0.1),
                    "status": "not_evaluated_short_circuit",
                    "skip_reason": "not_reached",
                }
            )
    return slots


def _slot_key(mode: str, timestep: int, direction: Optional[str] = None) -> str:
    if mode == "standing_car":
        return f"car_t{int(timestep)}"
    return f"crossing_{direction}_t{int(timestep)}"


def _first_failure(candidate: Mapping[str, Any]) -> dict:
    failures = candidate.get("failures")
    if isinstance(failures, list) and failures:
        return compact_failure(failures[0])
    return {
        "code": "candidate_failed_without_diagnostic",
        "message": "candidate returned passed=false without a reason",
    }


def _failure_reason(
    failure: Mapping[str, Any],
    *,
    mode: str,
    timestep: Optional[int] = None,
    direction: Optional[str] = None,
) -> dict:
    result = compact_failure(failure)
    result["mode"] = mode
    if timestep is not None:
        result["future_timestep"] = int(timestep)
        result["future_offset_s"] = float(int(timestep) * 0.1)
    if direction is not None:
        result["direction"] = direction
    return result


def _set_remaining_skip_reason(slots_by_id: Dict[str, dict], reason: str) -> None:
    for slot in slots_by_id.values():
        if (
            slot["status"] == "not_evaluated_short_circuit"
            and slot.get("skip_reason") == "not_reached"
        ):
            slot["skip_reason"] = reason


def classify_short_circuit(
    *,
    clip_id: str,
    seed: int,
    timesteps: Sequence[int],
    evaluate_car: Callable[[int], Mapping[str, Any]],
    evaluate_crossing: Callable[[str, int], Mapping[str, Any]],
    validate_shared: Optional[
        Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]
    ] = None,
) -> dict:
    """Apply the deterministic car-first policy with injected evaluators."""
    timesteps = normalize_timesteps(timesteps)
    initial_direction, uniform_value = initial_direction_choice(clip_id, seed)
    alternate_direction = "right" if initial_direction == "left" else "left"
    slots = candidate_slots(timesteps)
    slots_by_id = {slot["candidate_id"]: slot for slot in slots}
    evaluation_order = 0
    car_candidates: list[Mapping[str, Any]] = []

    def record_candidate(
        mode: str,
        timestep: int,
        raw_candidate: Mapping[str, Any],
        direction: Optional[str] = None,
    ) -> dict:
        nonlocal evaluation_order
        evaluation_order += 1
        candidate = compact_candidate(raw_candidate)
        slot = slots_by_id[_slot_key(mode, timestep, direction)]
        slot.pop("skip_reason", None)
        slot.update(candidate)
        slot.update({"status": "evaluated", "evaluation_order": evaluation_order})
        return candidate

    for timestep in timesteps:
        raw_car = evaluate_car(timestep)
        car_candidates.append(raw_car)
        car_result = record_candidate("standing_car", timestep, raw_car)
        if not car_result["passed"]:
            failure = _failure_reason(
                _first_failure(car_result),
                mode="standing_car",
                timestep=timestep,
            )
            _set_remaining_skip_reason(slots_by_id, "standing_car_failed")
            return {
                "classification": "failure",
                "classification_reason": "standing_car_failed",
                "classification_display": (
                    f"failure (standing_car:t{timestep}:{failure['code']})"
                ),
                "initial_direction": initial_direction,
                "initial_direction_uniform_value": uniform_value,
                "shared_validation": {
                    "status": "not_evaluated_short_circuit",
                    "skip_reason": "standing_car_failed",
                },
                "final_failure_reasons": [failure],
                "candidate_evaluations": slots,
            }

    shared_result: Mapping[str, Any] = {"passed": True, "failures": []}
    if validate_shared is not None:
        shared_result = validate_shared(car_candidates)
    shared_public = {
        key: json_ready(value)
        for key, value in shared_result.items()
        if not str(key).startswith("_")
    }
    shared_public["passed"] = bool(shared_result.get("passed", False))
    shared_public["failures"] = [
        compact_failure(item) for item in shared_result.get("failures", [])
    ]
    shared_public["status"] = "evaluated"
    if not shared_public["passed"]:
        failure_value = (
            shared_public["failures"][0]
            if shared_public["failures"]
            else {
                "code": "shared_validation_failed_without_diagnostic",
                "message": "shared validation failed without a reason",
            }
        )
        failure = _failure_reason(failure_value, mode="shared_preparation")
        _set_remaining_skip_reason(slots_by_id, "shared_validation_failed")
        return {
            "classification": "failure",
            "classification_reason": "shared_preparation_failed",
            "classification_display": (
                f"failure (shared_preparation:{failure['code']})"
            ),
            "initial_direction": initial_direction,
            "initial_direction_uniform_value": uniform_value,
            "shared_validation": shared_public,
            "final_failure_reasons": [failure],
            "candidate_evaluations": slots,
        }

    observed_direction_failures = []
    for direction_index, direction in enumerate(
        (initial_direction, alternate_direction)
    ):
        direction_failed = False
        for timestep in timesteps:
            raw_crossing = evaluate_crossing(direction, timestep)
            crossing_result = record_candidate(
                "crossing_pedestrian", timestep, raw_crossing, direction
            )
            if not crossing_result["passed"]:
                observed_direction_failures.append(
                    _failure_reason(
                        _first_failure(crossing_result),
                        mode="crossing_pedestrian",
                        timestep=timestep,
                        direction=direction,
                    )
                )
                direction_failed = True
                for later_timestep in timesteps:
                    slot = slots_by_id[
                        _slot_key("crossing_pedestrian", later_timestep, direction)
                    ]
                    if slot["status"] == "not_evaluated_short_circuit":
                        slot["skip_reason"] = "direction_already_failed"
                break
        if not direction_failed:
            _set_remaining_skip_reason(
                slots_by_id, "crossing_direction_already_selected"
            )
            return {
                "classification": direction,
                "classification_reason": (
                    "initial_crossing_direction_passed_all_timesteps"
                    if direction_index == 0
                    else "alternate_crossing_direction_passed_all_timesteps"
                ),
                "classification_display": direction,
                "initial_direction": initial_direction,
                "initial_direction_uniform_value": uniform_value,
                "shared_validation": shared_public,
                "final_failure_reasons": [],
                "candidate_evaluations": slots,
            }

    _set_remaining_skip_reason(slots_by_id, "both_directions_failed")
    display_reasons = ",".join(
        f"{item['direction']}:t{item['future_timestep']}:{item['code']}"
        for item in observed_direction_failures
    )
    return {
        "classification": "failure",
        "classification_reason": "no_crossing_direction_passed_all_timesteps",
        "classification_display": f"failure ({display_reasons})",
        "initial_direction": initial_direction,
        "initial_direction_uniform_value": uniform_value,
        "shared_validation": shared_public,
        "final_failure_reasons": observed_direction_failures,
        "candidate_evaluations": slots,
    }


def input_failure_decision(
    clip_id: str,
    seed: int,
    timesteps: Sequence[int],
    exc: Exception,
) -> dict:
    direction, uniform_value = initial_direction_choice(clip_id, seed)
    failure = _failure_reason(exception_diagnostic(exc), mode="shared_input")
    slots = candidate_slots(timesteps)
    for slot in slots:
        slot["skip_reason"] = "shared_input_failed"
    return {
        "classification": "failure",
        "classification_reason": "shared_input_failed",
        "classification_display": f"failure (shared_input:{failure['code']})",
        "initial_direction": direction,
        "initial_direction_uniform_value": uniform_value,
        "shared_validation": {
            "status": "not_evaluated_short_circuit",
            "skip_reason": "shared_input_failed",
        },
        "final_failure_reasons": [failure],
        "candidate_evaluations": slots,
    }


def actor_camera_visibility(
    bbox_lidar: np.ndarray,
    calibration: dict,
    minimum_projected_area_px2: float = 0.0,
) -> dict:
    mask = np.asarray(
        compositor.geometry.camera_frustum_mask(
            bbox_lidar, calibration, margin=0.0
        ),
        dtype=bool,
    ).reshape(-1)
    if mask.size != len(bbox_lidar):
        raise ValueError("camera frustum mask does not match bounding-box corners")
    projected_area = None
    area_accepted = True
    if minimum_projected_area_px2 > 0:
        width, height = compositor.geometry.camera_resolution(calibration)
        projected_uv, projected_depth = compositor.geometry.project_distorted(
            bbox_lidar, calibration
        )
        finite = (
            np.isfinite(projected_uv).all(axis=1)
            & np.isfinite(projected_depth)
            & (projected_depth > 0.1)
        )
        projected_area = 0.0
        if np.any(finite):
            bounds = (
                float(np.min(projected_uv[finite, 0])),
                float(np.min(projected_uv[finite, 1])),
                float(np.max(projected_uv[finite, 0])),
                float(np.max(projected_uv[finite, 1])),
            )
            x0 = float(np.clip(bounds[0], 0.0, width))
            y0 = float(np.clip(bounds[1], 0.0, height))
            x1 = float(np.clip(bounds[2], 0.0, width))
            y1 = float(np.clip(bounds[3], 0.0, height))
            projected_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area_accepted = projected_area >= float(minimum_projected_area_px2)
    visible = bool(np.any(mask) and area_accepted)
    return {
        "visible": visible,
        "fully_visible": bool(mask.size and np.all(mask) and area_accepted),
        "inside_corner_count": int(np.count_nonzero(mask)),
        "corner_count": int(mask.size),
        "projected_clipped_area_px2": projected_area,
        "minimum_projected_area_px2": float(minimum_projected_area_px2),
        "projected_area_accepted": bool(area_accepted),
    }


def projected_bbox_ego_mask_overlap(
    bbox_lidar: np.ndarray,
    calibration: dict,
    *,
    frame_index: int,
    camera: str,
    mask_cache: Optional[dict] = None,
) -> dict:
    """Intersect one projected grounded actor-box envelope with an ego mask."""
    width, height = compositor.geometry.camera_resolution(calibration)
    cache_key = (
        calibration.get("ego_mask_image_sha256"),
        int(height),
        int(width),
    )
    cached = mask_cache.get(cache_key) if mask_cache is not None else None
    if cached is None:
        protected = compositor.decode_calibrated_ego_mask(
            calibration,
            (height, width),
            f"classifier frame {int(frame_index)}/{camera}",
        )
        protected_pixel_count = int(np.count_nonzero(protected))
        if mask_cache is not None:
            mask_cache[cache_key] = (protected, protected_pixel_count)
    else:
        protected, protected_pixel_count = cached
        expected_pixels = calibration.get("ego_mask_protected_pixel_count")
        if expected_pixels != protected_pixel_count:
            raise compositor.TokenFailure(
                "invalid_ego_mask",
                f"classifier frame {int(frame_index)}/{camera}: cached ego-mask "
                "pixel count does not match calibration",
            )
    result = {
        "frame_index": int(frame_index),
        "camera": str(camera),
        "protected_pixel_count": protected_pixel_count,
        "bbox_envelope_pixel_count": 0,
        "overlap_pixel_count": 0,
        "overlap_fraction_of_envelope": 0.0,
        "projected_bounds_px": None,
    }
    if result["protected_pixel_count"] == 0:
        return result

    projected_uv, projected_depth = compositor.geometry.project_distorted(
        np.asarray(bbox_lidar, dtype=np.float64), calibration
    )
    finite = (
        np.isfinite(projected_uv).all(axis=1)
        & np.isfinite(projected_depth)
        & (projected_depth > 0.1)
    )
    if int(np.count_nonzero(finite)) < 3:
        return result
    hull = compositor.cv2.convexHull(
        np.asarray(projected_uv[finite], dtype=np.float32)
    ).reshape(-1, 2)
    if len(hull) < 3 or not np.isfinite(hull).all():
        return result

    bounds = [
        float(np.min(hull[:, 0])),
        float(np.min(hull[:, 1])),
        float(np.max(hull[:, 0])),
        float(np.max(hull[:, 1])),
    ]
    result["projected_bounds_px"] = bounds
    x0 = max(0, int(math.floor(bounds[0])))
    y0 = max(0, int(math.floor(bounds[1])))
    x1 = min(width, int(math.ceil(bounds[2])) + 1)
    y1 = min(height, int(math.ceil(bounds[3])) + 1)
    if x0 >= x1 or y0 >= y1:
        return result

    local_hull = np.rint(hull - np.array([x0, y0], dtype=np.float32)).astype(
        np.int32
    )
    envelope = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    compositor.cv2.fillConvexPoly(
        envelope, local_hull.reshape(-1, 1, 2), 1
    )
    envelope_pixels = int(np.count_nonzero(envelope))
    overlap_pixels = int(
        np.count_nonzero((envelope > 0) & protected[y0:y1, x0:x1])
    )
    result.update(
        {
            "bbox_envelope_pixel_count": envelope_pixels,
            "overlap_pixel_count": overlap_pixels,
            "overlap_fraction_of_envelope": (
                float(overlap_pixels / envelope_pixels)
                if envelope_pixels
                else 0.0
            ),
        }
    )
    return result


def candidate_ego_mask_overlap_diagnostics(
    states: Sequence[Mapping[str, Any]],
    frames: Sequence[dict],
    *,
    mask_cache: Optional[dict] = None,
) -> dict:
    """Measure conservative actor-envelope overlap in every visible view."""
    if len(states) != len(frames):
        raise ValueError("candidate states and history frames must have equal length")
    overlap_views = []
    evaluated_view_count = 0
    protected_view_count = 0
    for state, frame in zip(states, frames):
        frame_index = int(state["frame_index"])
        visible_cameras = set(state.get("visible_cameras", []))
        for camera in CAMERA_NAMES:
            if camera not in visible_cameras:
                continue
            evaluated_view_count += 1
            view = projected_bbox_ego_mask_overlap(
                np.asarray(state["bbox_lidar"], dtype=np.float64),
                frame["cams"][camera],
                frame_index=frame_index,
                camera=camera,
                mask_cache=mask_cache,
            )
            view["mask_source"] = frame["cams"][camera].get("ego_mask_provenance", {}).get(
                "source_field", ego_masks.DATASET_SOURCE
            )
            if view["protected_pixel_count"] > 0:
                protected_view_count += 1
            if view["overlap_pixel_count"] > 0:
                overlap_views.append(view)

    total_overlap = sum(
        int(view["overlap_pixel_count"]) for view in overlap_views
    )
    maximum_overlap = max(
        (int(view["overlap_pixel_count"]) for view in overlap_views),
        default=0,
    )
    return {
        "method": EGO_MASK_OVERLAP_METHOD,
        "evaluated_view_count": int(evaluated_view_count),
        "protected_view_count": int(protected_view_count),
        "overlap_detected": bool(overlap_views),
        "overlap_view_count": len(overlap_views),
        "total_overlap_pixel_count": int(total_overlap),
        "maximum_overlap_pixel_count": int(maximum_overlap),
        "overlap_views": overlap_views,
    }


def apply_ego_mask_overlap_policy(
    candidate: dict,
    states: Sequence[Mapping[str, Any]],
    frames: Sequence[dict],
    *,
    policy: str,
    mask_cache: Optional[dict] = None,
) -> Optional[dict]:
    """Record conservative overlap and optionally make it a hard failure."""
    normalized = str(policy).strip().lower()
    if normalized not in EGO_MASK_OVERLAP_POLICIES:
        raise ValueError(f"unsupported ego-mask overlap policy {policy!r}")
    if normalized == "ignore":
        return None
    summary = candidate_ego_mask_overlap_diagnostics(
        states, frames, mask_cache=mask_cache
    )
    summary["policy"] = normalized
    candidate["ego_mask_overlap"] = summary
    failing_views = [view for view in summary["overlap_views"]
                     if normalized == "fail" or
                     (normalized == "auto" and view.get("mask_source") != ego_masks.SOURCE)]
    summary["failure_view_count"] = len(failing_views)
    summary["refined_mask_overlap_policy"] = "report" if normalized == "auto" else normalized
    if failing_views or (normalized == "fail" and summary["overlap_detected"]):
        candidate["failures"].append(
            diagnostic(
                "ego_mask_overlap",
                "projected grounded actor bounding envelope intersects a "
                "calibrated ego mask",
                ego_mask_overlap=summary,
            )
        )
    return summary


def collision_path_with_interception(
    actor_global_xys: Sequence[Sequence[float]],
    interception_global_xy: Sequence[float],
) -> np.ndarray:
    history = np.asarray(actor_global_xys, dtype=np.float64)
    interception = np.asarray(interception_global_xy, dtype=np.float64).reshape(-1)
    if (
        history.ndim != 2
        or history.shape[0] == 0
        or history.shape[1] != 2
        or interception.shape != (2,)
        or not np.isfinite(history).all()
        or not np.isfinite(interception).all()
    ):
        raise ValueError("collision path requires finite history and anchor XY")
    return np.concatenate([history, interception[None]], axis=0)


def validate_token_inputs(
    args: argparse.Namespace,
    scene_frames: Sequence[dict],
) -> tuple[list[dict], np.ndarray, list[np.ndarray]]:
    frames = list(scene_frames[:HISTORY_FRAME_COUNT])
    if len(frames) != HISTORY_FRAME_COUNT:
        raise compositor.TokenFailure(
            "missing_sensor",
            f"expected {HISTORY_FRAME_COUNT} history frames, got {len(frames)}",
        )
    for frame_index, frame in enumerate(frames):
        for camera in CAMERA_NAMES:
            if camera not in frame.get("cams", {}):
                raise compositor.TokenFailure(
                    "missing_sensor",
                    f"history frame {frame_index} lacks {camera} calibration",
                )
            source = frame_source_path(args.data_root, args.split, frame, camera)
            if not source.is_file():
                raise compositor.TokenFailure(
                    "missing_sensor", f"missing {camera} image: {source}"
                )
        lidar_to_global = np.asarray(frame.get("lidar2global", []), dtype=np.float64)
        ego_to_global = np.asarray(frame.get("ego2global", []), dtype=np.float64)
        if (
            lidar_to_global.shape != (4, 4)
            or ego_to_global.shape != (4, 4)
            or not np.isfinite(lidar_to_global).all()
            or not np.isfinite(ego_to_global).all()
        ):
            raise compositor.TokenFailure(
                "invalid_sensor_pose",
                f"history frame {frame_index} has an invalid pose",
            )
    map_locations = {str(frame.get("map_location", "")) for frame in frames}
    if len(map_locations) != 1:
        raise compositor.TokenFailure(
            "inconsistent_map_location",
            f"history frames span multiple maps: {sorted(map_locations)}",
        )
    timestamps_s = np.asarray(
        [frame.get("timestamp", math.nan) for frame in frames], dtype=np.float64
    ) / 1e6
    if not np.isfinite(timestamps_s).all() or np.any(np.diff(timestamps_s) <= 0):
        raise compositor.TokenFailure(
            "invalid_timestamps",
            f"history timestamps are invalid: {timestamps_s.tolist()}",
        )
    frame_lidar_points = []
    for frame_index, frame in enumerate(frames):
        points = np.asarray(load_lidar_xyz(args.data_root, args.split, frame))
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise compositor.TokenFailure(
                "invalid_lidar",
                f"history frame {frame_index} has invalid LiDAR shape/data",
            )
        frame_lidar_points.append(points)
    return frames, timestamps_s, frame_lidar_points


def evaluate_candidate(
    *,
    args: argparse.Namespace,
    scene_frames: Sequence[dict],
    frames: Sequence[dict],
    timestamps_s: np.ndarray,
    frame_lidar_points: Sequence[np.ndarray],
    global_to_current: np.ndarray,
    front_bumper_offset_m: float,
    ego_geometry: Mapping[str, Any],
    asset: dict,
    animation: Sequence[int],
    future_timestep: int,
    actor_mode: str,
    direction: Optional[str] = None,
    ego_mask_cache: Optional[dict] = None,
) -> dict:
    if actor_mode == "car":
        direction_name = "stationary"
        crossing_name = None
        lateral_sign = 0.0
        actor_heading = 0.0
    elif actor_mode == "crossing" and direction in DIRECTIONS:
        spec = DIRECTION_SPECS[str(direction)]
        direction_name = str(direction)
        crossing_name = str(spec["crossing_name"])
        lateral_sign = float(spec["lateral_sign"])
        actor_heading = float(spec["actor_heading_rad"])
    else:
        raise ValueError(f"unsupported candidate {actor_mode}/{direction}")

    candidate = {
        "future_timestep": int(future_timestep),
        "actor_mode": actor_mode,
        "direction": direction_name,
        "crossing_name": crossing_name,
        "evaluation_complete": False,
        "passed": False,
        "failures": [],
        "_states": [],
    }
    try:
        bounds = asset["actor_geometry"]["local_bounds_m"]
        if actor_mode == "car":
            dimensions = ego_geometry["dimensions_m"]
            trajectory = compositor.resolve_stationary_vehicle_trajectory(
                scene_frames,
                timestamps_s,
                timestep=int(future_timestep),
                front_bumper_offset_m=float(front_bumper_offset_m),
                car_local_bounds=bounds,
                ego_rear_length_m=float(
                    ego_geometry["rear_axle_to_rear_bumper_m"]
                ),
                ego_width_m=float(dimensions["width"]),
            )
        else:
            trajectory = compositor.resolve_crossing_trajectory(
                scene_frames,
                timestamps_s,
                timestep=int(future_timestep),
                front_bumper_offset_m=float(front_bumper_offset_m),
                lateral_sign=lateral_sign,
                speed_mps=float(args.speed_mps),
            )

        actor_global_xys = trajectory["actor_global_xys"]
        actor_lidar_xys = [
            compositor.inverse_transform_xy_planar(
                np.asarray(frame["lidar2global"], dtype=np.float64),
                actor_global_xy,
            )
            for frame, actor_global_xy in zip(frames, actor_global_xys)
        ]

        # Call the PAI-AV adapter directly. This is the hard nearby-support
        # gate: no accepted history-frame support within 2.5 m raises
        # ground_fit_failed before a candidate can pass.
        (
            ground_planes,
            ground_diagnostics,
            strict_ground_failures,
        ) = paiav_edit.paiav_sequence_consistent_ground(
            frames,
            frame_lidar_points,
            actor_lidar_xys,
            trajectory["actor_current_xys"],
        )
        support_indices = sorted(
            {
                int(index)
                for item in ground_diagnostics
                for index in item.get("sequence_support_frame_indices", [])
            }
        )
        if not support_indices:
            raise compositor.TokenFailure(
                "ground_fit_failed",
                "PAI-AV grounding returned no accepted nearby support frames",
            )
        if strict_ground_failures:
            candidate["failures"].append(
                diagnostic(
                    "ground_fit_failed",
                    "strict same-frame local ground fitting failed",
                    failed_frame_indices=[
                        int(item["frame_index"]) for item in strict_ground_failures
                    ],
                    strict_failures=strict_ground_failures,
                )
            )

        canonical_actor_to_global = None
        if actor_mode == "car":
            final_lidar_to_global = np.asarray(
                frames[-1]["lidar2global"], dtype=np.float64
            )
            final_actor_lidar = compositor.ground_point_lidar_for_global_xy(
                final_lidar_to_global,
                ground_planes[-1],
                trajectory["anchor_global_xy"],
            )
            final_basis = compositor.road_basis_aligned_to_global_forward(
                ground_planes[-1],
                final_lidar_to_global,
                trajectory["reference_vehicle_forward_global_xy"],
                float(args.max_ground_tilt_deg),
            )
            canonical_actor_to_global = (
                final_lidar_to_global
                @ compositor.render_utils.matrix4(final_basis, final_actor_lidar)
            )

        states = []
        per_frame_collisions = []
        for frame_index, (
            frame,
            lidar_xyz,
            actor_lidar_xy,
            plane,
            ground_details,
            animation_frame,
        ) in enumerate(
            zip(
                frames,
                frame_lidar_points,
                actor_lidar_xys,
                ground_planes,
                ground_diagnostics,
                animation,
            )
        ):
            lidar_to_global = np.asarray(frame["lidar2global"], dtype=np.float64)
            if actor_mode == "car":
                if canonical_actor_to_global is None:
                    raise RuntimeError("car pose was not initialized")
                actor_to_lidar = (
                    np.linalg.inv(lidar_to_global) @ canonical_actor_to_global
                )
                actor_lidar = actor_to_lidar[:3, 3].copy()
                basis = actor_to_lidar[:3, :3].copy()
            else:
                actor_lidar = np.asarray(
                    [
                        float(actor_lidar_xy[0]),
                        float(actor_lidar_xy[1]),
                        compositor.geometry.ground_z(
                            plane,
                            float(actor_lidar_xy[0]),
                            float(actor_lidar_xy[1]),
                        ),
                    ],
                    dtype=np.float64,
                )
                basis = compositor.road_basis_aligned_to_global_forward(
                    plane,
                    lidar_to_global,
                    trajectory["reference_vehicle_forward_global_xy"],
                    float(args.max_ground_tilt_deg),
                )
                actor_to_lidar = compositor.render_utils.matrix4(basis, actor_lidar)
            lidar_to_actor = np.linalg.inv(actor_to_lidar)
            if actor_mode == "car":
                collision = compositor.stationary_vehicle_collision_diagnostics(
                    lidar_xyz, lidar_to_actor, bounds
                )
                bbox_lidar = compositor.geometry.transform_points(
                    actor_to_lidar, compositor.local_bbox_corners(bounds)
                )
            else:
                collision = compositor.frame_collision_diagnostics(
                    lidar_xyz, lidar_to_actor, actor_heading, bounds
                )
                bbox_lidar = compositor.render_utils.actor_bbox_lidar(
                    actor_lidar,
                    basis,
                    actor_heading,
                    bounds,
                    scale=1.0,
                )
            collision["frame_index"] = int(frame_index)
            per_frame_collisions.append(collision)
            camera_visibility = {
                camera: actor_camera_visibility(
                    bbox_lidar,
                    frame["cams"][camera],
                    float(args.minimum_projected_bbox_area_px2),
                )
                for camera in CAMERA_NAMES
            }
            actor_global = compositor.geometry.transform_points(
                lidar_to_global, actor_lidar[None]
            )[0]
            states.append(
                {
                    "frame_index": int(frame_index),
                    "animation_frame": int(animation_frame),
                    "actor_lidar": actor_lidar,
                    "actor_global": actor_global,
                    "actor_current": compositor.geometry.transform_points(
                        global_to_current, actor_global[None]
                    )[0],
                    "actor_to_lidar": actor_to_lidar,
                    "lidar_to_actor": lidar_to_actor,
                    "road_plane_lidar": np.asarray(plane, dtype=np.float64),
                    "road_basis_lidar": basis,
                    "bbox_lidar": bbox_lidar,
                    "visible_cameras": [
                        camera
                        for camera, visibility in camera_visibility.items()
                        if visibility["visible"]
                    ],
                    "fully_visible_cameras": [
                        camera
                        for camera, visibility in camera_visibility.items()
                        if visibility["fully_visible"]
                    ],
                    "camera_visibility": camera_visibility,
                    "ground_diagnostics": ground_details,
                    "collision_diagnostics": collision,
                }
            )

        if actor_mode == "car":
            swept_collision = dict(per_frame_collisions[-1])
            swept_collision.update(
                {
                    "path_start_xy_m": states[-1]["actor_lidar"][:2].tolist(),
                    "path_end_xy_m": states[-1]["actor_lidar"][:2].tolist(),
                    "sample_count": 1,
                }
            )
            swept_point_count = 1
            swept_includes_interception = False
        else:
            collision_path_global = collision_path_with_interception(
                actor_global_xys, trajectory["anchor_global_xy"]
            )
            final_positions_lidar = [
                compositor.inverse_transform_xy_planar(
                    np.asarray(frames[-1]["lidar2global"], dtype=np.float64),
                    actor_global_xy,
                )
                for actor_global_xy in collision_path_global
            ]
            swept_collision = compositor.trajectory_collision_diagnostics(
                frame_lidar_points[-1],
                states[-1]["road_plane_lidar"],
                final_positions_lidar,
                actor_basis_lidar=states[-1]["road_basis_lidar"],
                actor_heading=actor_heading,
                local_bounds_m=bounds,
            )
            swept_point_count = len(final_positions_lidar)
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
                    (
                        "vehicle_placement_occupied"
                        if actor_mode == "car"
                        else "trajectory_collision"
                    ),
                    "LiDAR-supported geometry intersects the actor or swept path",
                    occupied_frame_indices=occupied_frames,
                    swept_collision=swept_collision,
                )
            )

        apply_ego_mask_overlap_policy(
            candidate,
            states,
            frames,
            policy=getattr(args, "ego_mask_overlap_policy", "ignore"),
            mask_cache=ego_mask_cache,
        )

        visible_view_count = sum(len(state["visible_cameras"]) for state in states)
        fully_visible_view_count = sum(
            len(state["fully_visible_cameras"]) for state in states
        )
        if visible_view_count == 0:
            candidate["failures"].append(
                diagnostic(
                    "no_camera_visibility",
                    "actor bounding volume is outside every edited camera view",
                )
            )
        candidate.update(
            {
                "evaluation_complete": True,
                "passed": not candidate["failures"],
                "placement": trajectory["placement"],
                "ground_support_frame_indices": support_indices,
                "strict_ground_failure_frame_indices": [
                    int(item["frame_index"]) for item in strict_ground_failures
                ],
                "collision_diagnostics": {
                    "per_frame": per_frame_collisions,
                    "swept_final_frame": swept_collision,
                    "swept_path_includes_future_interception": (
                        swept_includes_interception
                    ),
                    "swept_path_point_count": swept_point_count,
                    "occupied": collision_occupied,
                },
                "visible_view_count": int(visible_view_count),
                "fully_visible_view_count": int(fully_visible_view_count),
                "_states": states,
            }
        )
    except Exception as exc:
        candidate["failures"].append(
            exception_diagnostic(
                exc,
                future_timestep=int(future_timestep),
                direction=direction_name,
            )
        )
        candidate.update(
            {"evaluation_complete": False, "passed": False, "_states": []}
        )
    return candidate


def evaluate_shared_preparation(
    *,
    args: argparse.Namespace,
    frames: Sequence[dict],
    car_candidates: Sequence[Mapping[str, Any]],
) -> dict:
    """Run source-image, contact-only lighting, and transform checks once."""
    result = {"passed": False, "failures": []}
    try:
        states = car_candidates[0].get("_states", [])
        if len(states) != HISTORY_FRAME_COUNT:
            raise compositor.TokenFailure(
                "shared_validation_failed",
                "representative car candidate has no complete state sequence",
            )
        lighting_views = []
        for camera in CAMERA_NAMES:
            calibration = frames[-1]["cams"][camera]
            camera_to_actor = (
                states[-1]["lidar_to_actor"]
                @ compositor.render_utils.camera_to_lidar(calibration)
            )
            source = frame_source_path(args.data_root, args.split, frames[-1], camera)
            try:
                with Image.open(source) as source_image:
                    image = np.asarray(source_image.convert("RGB"))
            except Exception as exc:
                raise compositor.TokenFailure(
                    "missing_sensor",
                    f"cannot decode {camera} image {source}: {exc}",
                ) from exc
            lighting_views.append(
                {
                    "camera": camera,
                    "image": image,
                    "calibration": calibration,
                    "camera_to_actor": camera_to_actor,
                }
            )
        estimated = compositor.estimate_multiview_lighting(lighting_views)
        astronomical = paiav_edit.contact_only_sun(frames[-1])
        sun_ray_global = np.asarray(
            astronomical["sun_ray_direction_global_utm"], dtype=np.float64
        )
        for candidate in car_candidates:
            candidate_states = candidate.get("_states", [])
            if len(candidate_states) != HISTORY_FRAME_COUNT:
                raise compositor.TokenFailure(
                    "shared_validation_failed",
                    "car candidate has no complete state sequence",
                )
            for state, frame in zip(candidate_states, frames):
                compositor.global_direction_to_actor(
                    sun_ray_global,
                    np.asarray(frame["lidar2global"], dtype=np.float64),
                    state["lidar_to_actor"],
                )
        result.update(
            {
                "passed": True,
                "lighting_method": str(estimated.get("method", "unknown")),
                "astronomical_method": str(astronomical["method"]),
                "direct_sun_enabled": False,
                "_sun_ray_global": sun_ray_global,
            }
        )
    except Exception as exc:
        result["failures"].append(exception_diagnostic(exc))
    return result


def validate_candidate_sun(
    candidate: dict,
    frames: Sequence[dict],
    sun_ray_global: np.ndarray,
) -> dict:
    if not candidate.get("passed", False):
        return candidate
    try:
        states = candidate.get("_states", [])
        if len(states) != HISTORY_FRAME_COUNT:
            raise compositor.TokenFailure(
                "sun_direction_failed",
                "crossing candidate has no complete state sequence",
            )
        for state, frame in zip(states, frames):
            compositor.global_direction_to_actor(
                sun_ray_global,
                np.asarray(frame["lidar2global"], dtype=np.float64),
                state["lidar_to_actor"],
            )
    except Exception as exc:
        candidate["failures"].append(exception_diagnostic(exc))
        candidate["passed"] = False
    return candidate


def classify_clip(worker: Mapping[str, Any], clip_id: str) -> dict:
    started = time.perf_counter()
    args = worker["geometry_args"]
    pedestrian = compositor.select_asset(
        clip_id, worker["seed"], worker["pedestrians"]
    )
    car = compositor.select_asset(clip_id, worker["seed"], worker["cars"])
    ego_geometry: Mapping[str, Any] = {}
    try:
        scene_frames = worker["loader"].scene_frames_dicts[clip_id]
        frames, timestamps_s, lidar_points = validate_token_inputs(
            args, scene_frames
        )
        current_ego_to_global = np.asarray(
            frames[-1]["ego2global"], dtype=np.float64
        )
        global_to_current = np.linalg.inv(current_ego_to_global)
        ego_geometry = worker["loader"].ego_vehicle_geometry(clip_id)
        front_bumper_offset = float(
            ego_geometry["rear_axle_to_front_bumper_m"]
        )
        animation = compositor.animation_frames(
            timestamps_s,
            float(args.speed_mps),
            pedestrian["stride_length_m"],
            pedestrian["action_range"][0],
            pedestrian["action_range"][1],
            compositor.NAVSIM_POSE_SEQUENCE[0],
        )
        shared_cache: Dict[str, Any] = {}

        def evaluate_car(timestep: int) -> dict:
            return evaluate_candidate(
                args=args,
                scene_frames=scene_frames,
                frames=frames,
                timestamps_s=timestamps_s,
                frame_lidar_points=lidar_points,
                global_to_current=global_to_current,
                front_bumper_offset_m=front_bumper_offset,
                ego_geometry=ego_geometry,
                asset=car,
                animation=[compositor.CAR_STATIC_ANIMATION_FRAME]
                * HISTORY_FRAME_COUNT,
                future_timestep=timestep,
                actor_mode="car",
                ego_mask_cache=worker.get("ego_mask_cache"),
            )

        def validate_shared(candidates: Sequence[Mapping[str, Any]]) -> dict:
            result = evaluate_shared_preparation(
                args=args, frames=frames, car_candidates=candidates
            )
            if result.get("passed"):
                shared_cache["sun_ray_global"] = result["_sun_ray_global"]
            return result

        def evaluate_crossing(direction: str, timestep: int) -> dict:
            candidate = evaluate_candidate(
                args=args,
                scene_frames=scene_frames,
                frames=frames,
                timestamps_s=timestamps_s,
                frame_lidar_points=lidar_points,
                global_to_current=global_to_current,
                front_bumper_offset_m=front_bumper_offset,
                ego_geometry=ego_geometry,
                asset=pedestrian,
                animation=animation,
                future_timestep=timestep,
                actor_mode="crossing",
                direction=direction,
                ego_mask_cache=worker.get("ego_mask_cache"),
            )
            return validate_candidate_sun(
                candidate, frames, shared_cache["sun_ray_global"]
            )

        decision = classify_short_circuit(
            clip_id=clip_id,
            seed=worker["seed"],
            timesteps=worker["timesteps"],
            evaluate_car=evaluate_car,
            evaluate_crossing=evaluate_crossing,
            validate_shared=validate_shared,
        )
    except Exception as exc:
        decision = input_failure_decision(
            clip_id, worker["seed"], worker["timesteps"], exc
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "configuration_fingerprint": worker["configuration_fingerprint"],
        "token": clip_id,
        "clip_id": clip_id,
        "chunk_id": int(worker["selection"].chunk_id_for(clip_id)),
        "anchor_time_s": float(worker["anchor_time_s"]),
        "classification": decision["classification"],
        "classification_reason": decision["classification_reason"],
        "classification_display": decision["classification_display"],
        "initial_direction": decision["initial_direction"],
        "initial_direction_uniform_value": decision[
            "initial_direction_uniform_value"
        ],
        "initial_direction_method": INITIAL_DIRECTION_METHOD,
        "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
        "selected_assets": {
            "standing_car": asset_identity(car),
            "crossing_pedestrian": asset_identity(pedestrian),
        },
        "ego_vehicle_geometry": json_ready(ego_geometry),
        "shared_validation": decision["shared_validation"],
        "final_failure_reasons": decision["final_failure_reasons"],
        "candidate_evaluations": decision["candidate_evaluations"],
        "duration_seconds": time.perf_counter() - started,
    }


def cleanup_loader_token(loader: PAIAVSceneLoader, scratch_dir: Path, token: str) -> None:
    for attribute in (
        "_frames",
        "_agent_inputs",
        "_alpamayo_ego_histories",
        "_clip_metadata",
    ):
        mapping = getattr(loader, attribute, None)
        if isinstance(mapping, dict):
            mapping.pop(token, None)
    for tree in ("source_frames", "source_lidar"):
        parent = (scratch_dir / tree).resolve()
        target = (parent / token).resolve()
        try:
            target.relative_to(parent)
        except ValueError:
            continue
        if target.is_dir():
            shutil.rmtree(target)


_WORKER_CONTEXT: Optional[dict] = None


def initialize_worker(context: dict) -> None:
    global _WORKER_CONTEXT
    paiav_edit.install_paiav_adapter("human", "crossing")
    compositor.frame_source_path = frame_source_path
    compositor.geometry.load_lidar_xyz = load_lidar_xyz
    try:
        compositor.cv2.setNumThreads(1)
    except AttributeError:
        pass
    selection = load_selection_manifest(Path(context["manifest"]))
    loader = PAIAVSceneLoader(
        Path(context["geometry_args"].data_root),
        selection,
        context["selected_tokens"],
        Path(context["scratch_dir"]),
        anchor_time_s=float(context["anchor_time_s"]),
        future_timestep=max(context["timesteps"]),
        **ego_masks.loader_options(context["geometry_args"]),
    )
    _WORKER_CONTEXT = {
        **context,
        "selection": selection,
        "loader": loader,
        "ego_mask_cache": {},
    }


def process_token(token: str) -> dict:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context was not initialized")
    try:
        return classify_clip(_WORKER_CONTEXT, token)
    finally:
        cleanup_loader_token(
            _WORKER_CONTEXT["loader"],
            Path(_WORKER_CONTEXT["scratch_dir"]),
            token,
        )


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint": output_dir / "paiav_checkpoint.jsonl",
        "checkpoint_meta": output_dir / "paiav_checkpoint.meta.json",
        "classifications": output_dir / "paiav_direction_classifications.json",
        "details": output_dir / "paiav_classification_details.jsonl",
        "summary": output_dir / "paiav_summary.json",
    }


def recover_and_load_checkpoint(
    path: Path,
    *,
    allowed_tokens: set[str],
    configuration_fingerprint: str,
) -> Dict[str, dict]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    chunks = raw.splitlines(keepends=True)
    records: Dict[str, dict] = {}
    valid_size = 0
    for index, chunk in enumerate(chunks):
        line = chunk.rstrip(b"\r\n")
        if not line:
            raise ClassificationError(
                f"Checkpoint contains an empty line at {index + 1}: {path}"
            )
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != len(chunks) - 1:
                raise ClassificationError(
                    f"Checkpoint has invalid JSON at line {index + 1}: {path}"
                ) from exc
            with path.open("r+b") as handle:
                handle.truncate(valid_size)
            break
        valid_size += len(chunk)
        if not chunk.endswith((b"\n", b"\r")):
            with path.open("ab") as handle:
                handle.write(b"\n")
            valid_size += 1
        if not isinstance(record, dict):
            raise ClassificationError(
                f"Checkpoint line {index + 1} is not an object: {path}"
            )
        token = record.get("token")
        if not isinstance(token, str) or token not in allowed_tokens:
            raise ClassificationError(
                f"Checkpoint contains unknown token {token!r}: {path}"
            )
        if token in records:
            raise ClassificationError(
                f"Checkpoint contains duplicate token {token}: {path}"
            )
        if record.get("configuration_fingerprint") != configuration_fingerprint:
            raise ClassificationError(
                f"Checkpoint token {token} uses another configuration"
            )
        if record.get("classification") not in (*DIRECTIONS, "failure"):
            raise ClassificationError(
                f"Checkpoint token {token} has an invalid classification"
            )
        records[token] = record
    return records


def prepare_checkpoint(
    *,
    paths: Mapping[str, Path],
    configuration: dict,
    allowed_tokens: set[str],
    overwrite: bool,
) -> Dict[str, dict]:
    if overwrite:
        for path in paths.values():
            if path.exists():
                if not path.is_file():
                    raise ClassificationError(
                        f"Refusing to overwrite non-file output: {path}"
                    )
                path.unlink()
    existing = [path for path in paths.values() if path.exists()]
    meta_path = paths["checkpoint_meta"]
    checkpoint_path = paths["checkpoint"]
    fingerprint = configuration["configuration_fingerprint"]
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClassificationError(
                f"Cannot read checkpoint metadata {meta_path}: {exc}"
            ) from exc
        if meta.get("configuration_fingerprint") != fingerprint:
            raise ClassificationError(
                f"Existing outputs use another configuration: {meta_path}; "
                "pass --overwrite to start over"
            )
    elif existing:
        raise ClassificationError(
            f"Outputs exist without checkpoint metadata in {meta_path.parent}; "
            "pass --overwrite to start over"
        )
    else:
        write_json_atomic(
            meta_path,
            {"schema_version": SCHEMA_VERSION, "created_at": utc_now(), **configuration},
        )
    checkpoint_path.touch(exist_ok=True)
    return recover_and_load_checkpoint(
        checkpoint_path,
        allowed_tokens=allowed_tokens,
        configuration_fingerprint=fingerprint,
    )


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_line(record))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_configuration(
    *,
    args: argparse.Namespace,
    selection: Any,
    tokens: Sequence[str],
    pedestrians: Sequence[dict],
    cars: Sequence[dict],
) -> dict:
    base = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset": "paiav",
        "data_root": str(args.data_root),
        "manifest": str(selection.manifest_path),
        "manifest_role": selection.manifest_role,
        "manifest_sha256": compositor.file_sha256(selection.manifest_path),
        "catalog_clip_index": (
            str(selection.catalog_index_path)
            if selection.catalog_index_path is not None
            else None
        ),
        "catalog_clip_index_sha256": selection.catalog_index_sha256,
        "source_clip_count": len(selection.clip_ids),
        "selected_clip_count": len(tokens),
        "selected_clips_sha256": compositor.canonical_hash(list(tokens)),
        "selected_chunk_ids": list(selection.chunk_ids_for(tokens)) if tokens else [],
        "anchor_time_s": float(args.anchor_time_s),
        "history_interval_s": HISTORY_INTERVAL_US / 1e6,
        "timesteps": list(args.timesteps),
        "future_offsets_s": [
            float(timestep * HISTORY_INTERVAL_US / 1e6)
            for timestep in args.timesteps
        ],
        "short_circuit_order": "standing_car_then_seeded_crossing_direction_v1",
        "seed": int(args.seed),
        "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
        "asset_selection": "production_blake2b_clip_seed_asset_v1",
        "pedestrian_assets": [asset_identity(asset) for asset in pedestrians],
        "car_assets": [asset_identity(asset) for asset in cars],
        "speed_mps": float(args.speed_mps),
        "stride_length_m": args.stride_length_m,
        "max_ground_tilt_deg": float(args.max_ground_tilt_deg),
        "minimum_projected_bbox_area_px2": float(
            args.min_projected_bbox_area
        ),
        "ego_mask_selection": getattr(args, "ego_mask_selection", {"mode": getattr(args, "ego_mask_mode", ego_masks.DEFAULT_MODE)}),
        "ego_mask_overlap": {
            "policy": str(args.ego_mask_overlap_policy),
            "method": EGO_MASK_OVERLAP_METHOD,
            "failure_threshold_pixels": 1,
            "scope": "all_visible_candidate_history_views",
        },
        "grounding_policy": {
            "version": paiav_edit.PAIAV_GROUNDING_VERSION,
            "nearby_support_radius_m": 2.5,
            "requires_at_least_one_supported_history_frame": True,
            "strict_same_frame_fallback_is_classification_failure": True,
        },
        "collision_is_classification_failure": True,
        "worker_count_independent_selection": True,
    }
    return {
        **base,
        "configuration_fingerprint": compositor.canonical_hash(json_ready(base)),
    }


def summarize_records(
    *,
    records: Sequence[dict],
    configuration: dict,
    started_at: str,
    elapsed_seconds: float,
    resumed_count: int,
    workers: int,
) -> dict:
    label_counts = Counter(record["classification"] for record in records)
    reason_counts = Counter(record["classification_reason"] for record in records)
    failure_mode_counts: Counter[str] = Counter()
    failure_timestep_counts: Counter[str] = Counter()
    failure_code_counts: Counter[str] = Counter()
    evaluated_counts: Counter[str] = Counter()
    for record in records:
        for candidate in record["candidate_evaluations"]:
            if candidate["status"] == "evaluated":
                evaluated_counts[candidate["mode"]] += 1
        for failure in record["final_failure_reasons"]:
            failure_mode_counts[str(failure.get("mode", "unknown"))] += 1
            failure_code_counts[str(failure.get("code", "unknown"))] += 1
            if failure.get("future_timestep") is not None:
                failure_timestep_counts[str(int(failure["future_timestep"]))] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": float(elapsed_seconds),
        "requested_tokens": len(records),
        "completed_tokens": len(records),
        "resumed_tokens": int(resumed_count),
        "newly_evaluated_tokens": len(records) - int(resumed_count),
        "label_counts": dict(sorted(label_counts.items())),
        "classification_reason_counts": dict(sorted(reason_counts.items())),
        "final_failure_mode_counts": dict(sorted(failure_mode_counts.items())),
        "final_failure_timestep_counts": dict(
            sorted(failure_timestep_counts.items(), key=lambda item: int(item[0]))
        ),
        "final_failure_code_counts": dict(sorted(failure_code_counts.items())),
        "evaluated_candidate_counts": dict(sorted(evaluated_counts.items())),
        "execution": {
            "workers": int(workers),
            "native_threads_per_worker": 1,
            "source_cache": "ephemeral_per_job_and_cleaned_per_token",
        },
        "configuration": configuration,
    }


def validate_environment(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.scratch_root = args.scratch_root.expanduser().resolve()
    args.pedestrian_assets = args.pedestrian_assets.expanduser().resolve()
    args.car_assets = args.car_assets.expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"PAI-AV data root is missing: {args.data_root}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"PAI-AV manifest is missing: {args.manifest}")
    for label, path in (
        ("pedestrian assets", args.pedestrian_assets),
        ("car assets", args.car_assets),
        ("scratch root", args.scratch_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {path}")
    required_modules = ("av", "cv2", "pandas", "pyarrow", "scipy", "DracoPy")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        raise ClassificationError(
            f"PAI-AV Python environment lacks required modules: {missing}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict:
    started_at = utc_now()
    started = time.perf_counter()
    validate_environment(args)
    selection = load_selection_manifest(args.manifest)
    tokens = paiav_edit.select_tokens(selection.clip_ids, args.max_clips, args.seed)
    pedestrians, cars = load_assets(args)
    from .prepare_ego_masks import prepare
    preparation_args = argparse.Namespace(**vars(args))
    preparation_args.check = False
    preparation_args.workers = min(4, args.workers)
    mask_status = prepare(preparation_args)
    failures = [record for record in mask_status if record["status"] == "failed"]
    if failures:
        raise ClassificationError(f"Required ego-mask preparation failed: {failures}")
    if tokens:
        identity_loader = PAIAVSceneLoader(args.data_root, selection, tokens, args.output_dir,
                                          anchor_time_s=args.anchor_time_s,
                                          future_timestep=max(args.timesteps),
                                          **ego_masks.loader_options(args))
        args.ego_mask_selection = identity_loader.ego_masks.run_identity(tokens)
    else:
        args.ego_mask_selection = dict(mode=args.ego_mask_mode, masks={})
    configuration = build_configuration(
        args=args,
        selection=selection,
        tokens=tokens,
        pedestrians=pedestrians,
        cars=cars,
    )
    paths = output_paths(args.output_dir)
    completed = prepare_checkpoint(
        paths=paths,
        configuration=configuration,
        allowed_tokens=set(tokens),
        overwrite=args.overwrite,
    )
    resumed_count = len(completed)
    remaining = [token for token in tokens if token not in completed]
    actual_workers = min(args.workers, max(1, len(remaining)))
    print(
        f"paiav: {len(tokens):,} requested, {resumed_count:,} resumed, "
        f"{len(remaining):,} remaining; horizons={list(args.timesteps)}, "
        f"ego-mask-overlap={args.ego_mask_overlap_policy}",
        flush=True,
    )

    if remaining:
        with tempfile.TemporaryDirectory(
            prefix="paiav-classify-", dir=args.scratch_root
        ) as temporary:
            geometry_args = argparse.Namespace(
                data_root=args.data_root,
                split="paiav",
                speed_mps=float(args.speed_mps),
                max_ground_tilt_deg=float(args.max_ground_tilt_deg),
                minimum_projected_bbox_area_px2=float(
                    args.min_projected_bbox_area
                ),
                ego_mask_overlap_policy=str(args.ego_mask_overlap_policy),
                **ego_masks.loader_options(args),
            )
            context = {
                "geometry_args": geometry_args,
                "manifest": args.manifest,
                "selected_tokens": tuple(remaining),
                "scratch_dir": Path(temporary),
                "anchor_time_s": float(args.anchor_time_s),
                "timesteps": tuple(args.timesteps),
                "seed": int(args.seed),
                "pedestrians": pedestrians,
                "cars": cars,
                "configuration_fingerprint": configuration[
                    "configuration_fingerprint"
                ],
            }
            last_reported = len(completed)

            def consume(record: dict) -> None:
                nonlocal last_reported
                token = record.get("token")
                if token not in remaining or token in completed:
                    raise ClassificationError(
                        f"Worker returned unknown or duplicate token {token!r}"
                    )
                completed[token] = record
                append_checkpoint(paths["checkpoint"], record)
                current = len(completed)
                if args.progress_every and (
                    current - last_reported >= args.progress_every
                    or current == len(tokens)
                ):
                    elapsed = time.perf_counter() - started
                    rate = (current - resumed_count) / max(elapsed, 1e-9)
                    print(
                        f"paiav: {current:,}/{len(tokens):,} "
                        f"({100.0 * current / max(len(tokens), 1):.1f}%), "
                        f"{rate:.2f} new clips/s",
                        flush=True,
                    )
                    last_reported = current

            if actual_workers == 1:
                initialize_worker(context)
                for token in remaining:
                    consume(process_token(token))
            else:
                with ProcessPoolExecutor(
                    max_workers=actual_workers,
                    initializer=initialize_worker,
                    initargs=(context,),
                ) as executor:
                    futures = {
                        executor.submit(process_token, token): token
                        for token in remaining
                    }
                    for future in as_completed(futures):
                        token = futures[future]
                        try:
                            consume(future.result())
                        except Exception as exc:
                            raise ClassificationError(
                                f"Worker failed while processing {token}: {exc}"
                            ) from exc

    missing = [token for token in tokens if token not in completed]
    unknown = sorted(set(completed) - set(tokens))
    if missing or unknown:
        raise ClassificationError(
            f"Output coverage is invalid: missing={missing[:5]}, unknown={unknown[:5]}"
        )
    records = [completed[token] for token in tokens]
    for record in records:
        if record["classification"] == "failure" and not record.get(
            "final_failure_reasons"
        ):
            raise ClassificationError(
                f"Failure clip {record['token']} has no observed reason"
            )
    classifications = {
        record["token"]: record["classification"] for record in records
    }
    elapsed = time.perf_counter() - started
    summary = summarize_records(
        records=records,
        configuration=configuration,
        started_at=started_at,
        elapsed_seconds=elapsed,
        resumed_count=resumed_count,
        workers=actual_workers,
    )
    write_json_atomic(paths["classifications"], classifications)
    write_jsonl_atomic(paths["details"], records)
    write_json_atomic(paths["summary"], summary)
    print(
        f"paiav: complete in {elapsed / 3600.0:.2f} h; "
        f"labels={summary['label_counts']}",
        flush=True,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ClassificationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
