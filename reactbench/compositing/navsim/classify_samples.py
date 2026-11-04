#!/usr/bin/env python3
"""Classify official NAVSIM samples with the compositor's pre-render checks.

The classifier does not render or edit images.  For each selected token it
checks a stationary car first at future timesteps 4, 3, 2, and 1, followed by
one deterministically selected pedestrian crossing direction.  The alternate
crossing direction is checked only when the first direction fails.

With --include-benign-scenarios, additionally require hazardous standing and
all three benign actor modes at every timestep. Cars prefer the left side;
pedestrians prefer the right. Current CAM_F0 bbox visibility defaults to 25%.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

# Keep one native thread per worker process.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from PIL import Image
import yaml


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
RECOGDRIVE_ROOT = REPOSITORY_ROOT / "third_party" / "recogdrive"  # NAVSIM devkit of the scene loader
if str(RECOGDRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(RECOGDRIVE_ROOT))

from . import batch_composite_navsim_multi_trajectory as multi  # noqa: E402
from .compositor import engine as fast  # noqa: E402


SCHEMA_VERSION = 2
PIPELINE_VERSION = "full_navsim_preflight_classifier_v2"
TIMESTEPS = (4, 3, 2, 1)
DIRECTIONS = ("left", "right")
INITIAL_DIRECTION_NAMESPACE = "sample_classifier_initial_direction"
INITIAL_DIRECTION_METHOD = "blake2b_token_seed_uniform_v1"
DEFAULT_DATA_ROOT = fast.locations.NAVSIM_DATA
DEFAULT_FILTER_ROOT = (
    RECOGDRIVE_ROOT
    / "navsim"
    / "planning"
    / "script"
    / "config"
    / "common"
    / "train_test_split"
    / "scene_filter"
)
DEFAULT_PEDESTRIAN_ASSETS = fast.locations.ASSETS / "rocketbox" / "assets"
DEFAULT_CAR_ASSETS = fast.locations.ASSETS / "openx" / "assets"
DEFAULT_OUTPUT_DIR = fast.locations.paths.outputs("classification", "navsim")
SPLIT_SPECS = {
    "navtest": {
        "storage_split": "test",
        "expected_tokens": 12_146,
        "default_filter": DEFAULT_FILTER_ROOT / "navtest.yaml",
    },
    "navtrain": {
        "storage_split": "trainval",
        "expected_tokens": 103_288,
        "default_filter": DEFAULT_FILTER_ROOT / "navtrain.yaml",
    },
}


class ClassificationError(RuntimeError):
    """A run-level invariant failed; ordinary sample failures do not use it."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_worker_count() -> int:
    configured = os.environ.get("SLURM_CPUS_PER_TASK")
    if configured and configured.isdigit() and int(configured) > 0:
        return int(configured)
    return max(1, min(56, os.cpu_count() or 1))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_SPECS),
        default=list(SPLIT_SPECS),
    )
    parser.add_argument(
        "--navtest-filter",
        type=Path,
        default=SPLIT_SPECS["navtest"]["default_filter"],
    )
    parser.add_argument(
        "--navtrain-filter",
        type=Path,
        default=SPLIT_SPECS["navtrain"]["default_filter"],
    )
    parser.add_argument(
        "--pedestrian-assets",
        type=Path,
        default=DEFAULT_PEDESTRIAN_ASSETS,
    )
    parser.add_argument("--car-assets", type=Path, default=DEFAULT_CAR_ASSETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Select a deterministic subset of at most N tokens per split. "
            "Intended for smoke tests; omit for the official full run."
        ),
    )
    parser.add_argument("--lidar-cache-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--expected-car-assets", type=int, default=7)
    parser.add_argument("--speed-mps", type=float, default=fast.PEDESTRIAN_SPEED_MPS)
    parser.add_argument("--stride-length-m", type=float)
    parser.add_argument("--gait-frame-advance", type=int, default=23)
    parser.add_argument("--gait-timestep-s", type=float, default=0.5)
    parser.add_argument("--max-ground-tilt-deg", type=float, default=2.0)
    parser.add_argument("--actor-scale", type=float, default=1.0)
    parser.add_argument(
        "--include-benign-scenarios", action="store_true",
        help="Require hazardous and benign car/standing/walking placements at all timesteps.",
    )
    fast.front_visibility.add_arguments(parser, default=None)
    parser.add_argument("--benign-offset-m", type=float, default=4.0)
    parser.add_argument("--benign-clearance-m", type=float, default=1.0)
    parser.add_argument("--benign-walking-direction", choices=("forward", "backward"), default="forward")
    parser.add_argument(
        "--allow-nonofficial-counts",
        action="store_true",
        help="Permit custom filter token counts (useful only for fixtures).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard this configuration's existing checkpoint and outputs.",
    )
    args = parser.parse_args(argv)
    if args.min_front_visible_fraction is None:
        args.min_front_visible_fraction = 0.25 if args.include_benign_scenarios else 0.0
    try:
        fast.benign_placements.configuration(args)
    except ValueError as exc:
        parser.error(str(exc))
    validate_cli_args(args, parser)
    return args


def validate_cli_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if args.lidar_cache_size < 1:
        parser.error("--lidar-cache-size must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.expected_car_assets < 1:
        parser.error("--expected-car-assets must be positive")
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
    if not math.isfinite(args.actor_scale) or args.actor_scale <= 0:
        parser.error("--actor-scale must be finite and positive")
    # Preserve the user's order, while preventing accidental duplicate work.
    args.splits = list(dict.fromkeys(args.splits))


def json_ready(value: Any) -> Any:
    """Convert NumPy/path values to strict, deterministic JSON values."""
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


def load_filter(
    split_name: str,
    path: Path,
    *,
    seed: int,
    max_samples: Optional[int],
    allow_nonofficial_counts: bool,
) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} scene filter is missing: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ClassificationError(f"Cannot read scene filter {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ClassificationError(f"Scene filter must be an object: {path}")
    expected = SPLIT_SPECS[split_name]
    for key, required in (
        ("num_history_frames", fast.EXPECTED_HISTORY_FRAMES),
        ("frame_interval", 1),
    ):
        if int(document.get(key, -1)) != required:
            raise ClassificationError(
                f"{split_name} {key} must be {required}, got {document.get(key)!r}"
            )
    if int(document.get("num_future_frames", -1)) < max(TIMESTEPS):
        raise ClassificationError(
            f"{split_name} requires at least {max(TIMESTEPS)} future frames"
        )
    raw_tokens = document.get("tokens")
    raw_logs = document.get("log_names")
    if not isinstance(raw_tokens, list) or not all(
        isinstance(token, str) and token for token in raw_tokens
    ):
        raise ClassificationError(f"{split_name} filter has no valid token list")
    if not isinstance(raw_logs, list) or not all(
        isinstance(log_name, str) and log_name for log_name in raw_logs
    ):
        raise ClassificationError(f"{split_name} filter has no valid log-name list")
    if len(set(raw_tokens)) != len(raw_tokens):
        raise ClassificationError(f"{split_name} filter contains duplicate tokens")
    if len(set(raw_logs)) != len(raw_logs):
        raise ClassificationError(f"{split_name} filter contains duplicate log names")
    official_count = int(expected["expected_tokens"])
    if (
        max_samples is None
        and not allow_nonofficial_counts
        and len(raw_tokens) != official_count
    ):
        raise ClassificationError(
            f"{split_name} filter contains {len(raw_tokens):,} tokens; "
            f"expected the official {official_count:,}"
        )
    selected_tokens = fast.seeded_random_subset(raw_tokens, max_samples, seed)
    return {
        "path": path,
        "path_sha256": fast.file_sha256(path),
        "tokens": selected_tokens,
        "source_token_count": len(raw_tokens),
        "log_names": list(raw_logs),
        "num_history_frames": int(document["num_history_frames"]),
        "num_future_frames": int(document["num_future_frames"]),
        "frame_interval": int(document["frame_interval"]),
        "has_route": bool(document.get("has_route", True)),
    }


def load_assets(args: argparse.Namespace) -> tuple[List[dict], List[dict]]:
    pedestrian_args = argparse.Namespace(
        asset_dir=args.pedestrian_assets.expanduser().resolve(),
        car=False,
        stride_length_m=args.stride_length_m,
        speed_mps=args.speed_mps,
        gait_timestep_s=args.gait_timestep_s,
        gait_frame_advance=args.gait_frame_advance,
    )
    car_dir = args.car_assets.expanduser().resolve()
    if not car_dir.is_dir():
        raise FileNotFoundError(f"Car asset directory is missing: {car_dir}")
    blend_paths = sorted(path for path in car_dir.rglob("*.blend") if path.is_file())
    if len(blend_paths) != args.expected_car_assets:
        raise ClassificationError(
            f"Expected exactly {args.expected_car_assets} prepared car assets, "
            f"found {len(blend_paths)} under {car_dir}"
        )
    for blend_path in blend_paths:
        for sidecar in (
            blend_path.with_suffix(".xoma"),
            blend_path.with_suffix(".preparation.json"),
        ):
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f"Prepared car asset is missing sidecar: {sidecar}"
                )
    car_args = argparse.Namespace(asset_dir=car_dir, car=True)
    pedestrians = fast.load_asset_pool(pedestrian_args)
    cars = fast.load_asset_pool(car_args)
    if len(cars) != args.expected_car_assets:
        raise ClassificationError(
            f"Loaded {len(cars)} cars, expected {args.expected_car_assets}"
        )
    for car in cars:
        preparation_path = Path(car["path"]).with_suffix(".preparation.json")
        car["preparation_path"] = preparation_path.resolve()
        car["preparation_sha256"] = fast.file_sha256(preparation_path)
    for pedestrian in pedestrians:
        metadata_path = Path(pedestrian["metadata_path"])
        pedestrian["metadata_sha256"] = fast.file_sha256(metadata_path)
    return pedestrians, cars


def asset_identity(asset: dict) -> dict:
    result = {
        "id": str(asset["id"]),
        "actor_type": str(asset.get("actor_type", "unknown")),
        "path": str(Path(asset["path"]).resolve()),
        "sha256": str(asset["sha256"]),
        "metadata_path": str(Path(asset["metadata_path"]).resolve()),
        "metadata_sha256": asset.get("metadata_sha256"),
        "actor_height_m": float(asset["actor_height_m"]),
    }
    if asset.get("actor_type") == "car":
        geometry = asset["actor_geometry"]
        result.update(
            {
                "preparation_path": str(asset["preparation_path"]),
                "preparation_sha256": str(asset["preparation_sha256"]),
                "length_m": float(geometry["length_m"]),
                "width_m": float(geometry["width_m"]),
                "height_m": float(geometry["height_m"]),
                "local_bounds_m": geometry["local_bounds_m"],
            }
        )
    else:
        result.update(
            {
                "stride_length_m": float(asset["stride_length_m"]),
                "action_range": list(asset["action_range"]),
                "animation_version": str(asset["animation_version"]),
            }
        )
    return json_ready(result)


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
        "front_visibility",
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
    return result


def compact_candidate(candidate: Mapping[str, Any]) -> dict:
    passed = bool(candidate.get("passed", False))
    failures = [compact_failure(item) for item in candidate.get("failures", [])]
    if not passed and not failures:
        failures = [
            {
                "code": "candidate_failed_without_diagnostic",
                "message": "candidate returned passed=false without a failure reason",
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
    ):
        if key in candidate:
            result[key] = int(candidate[key])
    collision = candidate.get("collision_diagnostics")
    if isinstance(collision, Mapping):
        result["collision_occupied"] = bool(collision.get("occupied", False))
    for key in ("placement", "front_visibility", "actor_heading_rad",
                "walking_velocity_global_xy_mps", "anchor_global_xy_m"):
        if key in candidate:
            result[key] = json_ready(candidate[key])
    if candidate.get("_states"):
        result["resolved_history"] = [
            {key: json_ready(state[key]) for key in (
                "frame_index", "animation_frame", "actor_global", "actor_current",
                "road_basis_lidar", "actor_lidar", "yaw_lidar_rad",
            ) if key in state}
            for state in candidate["_states"]
        ]
    return result


def initial_direction_choice(token: str, seed: int) -> tuple[str, float]:
    uniform_value = fast.token_uniform(
        str(token), int(seed), INITIAL_DIRECTION_NAMESPACE
    )
    return ("left" if uniform_value < 0.5 else "right", float(uniform_value))


def candidate_slots() -> List[dict]:
    slots = []
    for timestep in TIMESTEPS:
        slots.append(
            {
                "candidate_id": f"car_t{timestep}",
                "mode": "standing_car",
                "future_timestep": timestep,
                "status": "not_evaluated_short_circuit",
                "skip_reason": "not_reached",
            }
        )
    for direction in DIRECTIONS:
        for timestep in TIMESTEPS:
            slots.append(
                {
                    "candidate_id": f"crossing_{direction}_t{timestep}",
                    "mode": "crossing_pedestrian",
                    "direction": direction,
                    "crossing_name": multi.DIRECTION_BY_CLASSIFICATION[direction][
                        "crossing_name"
                    ],
                    "future_timestep": timestep,
                    "status": "not_evaluated_short_circuit",
                    "skip_reason": "not_reached",
                }
            )
    return slots


def _slot_key(mode: str, timestep: int, direction: Optional[str] = None) -> str:
    if mode == "standing_car":
        return f"car_t{int(timestep)}"
    return f"crossing_{direction}_t{int(timestep)}"


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
    if direction is not None:
        result["direction"] = direction
    return result


def _first_failure(candidate: Mapping[str, Any]) -> dict:
    failures = candidate.get("failures")
    if isinstance(failures, list) and failures:
        return compact_failure(failures[0])
    return {
        "code": "candidate_failed_without_diagnostic",
        "message": "candidate returned passed=false without a failure reason",
    }


def _set_remaining_skip_reason(
    slots_by_id: Dict[str, dict],
    reason: str,
) -> None:
    for slot in slots_by_id.values():
        if (
            slot["status"] == "not_evaluated_short_circuit"
            and slot.get("skip_reason") == "not_reached"
        ):
            slot["skip_reason"] = reason


def classify_short_circuit(
    *,
    token: str,
    seed: int,
    evaluate_car: Callable[[int], Mapping[str, Any]],
    evaluate_crossing: Callable[[str, int], Mapping[str, Any]],
    validate_shared: Optional[
        Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]
    ] = None,
) -> dict:
    """Apply the deterministic car-first classification policy.

    This function intentionally accepts injected evaluators so ordering and
    short-circuit behavior can be tested without loading NAVSIM data.
    """
    initial_direction, uniform_value = initial_direction_choice(token, seed)
    alternate_direction = "right" if initial_direction == "left" else "left"
    slots = candidate_slots()
    slots_by_id = {slot["candidate_id"]: slot for slot in slots}
    evaluation_order = 0
    car_candidates: List[Mapping[str, Any]] = []

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
        slot.update(
            {
                "status": "evaluated",
                "evaluation_order": evaluation_order,
            }
        )
        return candidate

    for timestep in TIMESTEPS:
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
                "initial_direction_method": INITIAL_DIRECTION_METHOD,
                "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
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
            "initial_direction_method": INITIAL_DIRECTION_METHOD,
            "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
            "shared_validation": shared_public,
            "final_failure_reasons": [failure],
            "candidate_evaluations": slots,
        }

    observed_direction_failures = []
    for direction_index, direction in enumerate(
        (initial_direction, alternate_direction)
    ):
        direction_failed = False
        for timestep in TIMESTEPS:
            raw_crossing = evaluate_crossing(direction, timestep)
            crossing_result = record_candidate(
                "crossing_pedestrian",
                timestep,
                raw_crossing,
                direction,
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
                # The rest of this direction is unnecessary, but the alternate
                # direction is evaluated when this was the first attempt.
                for later_timestep in TIMESTEPS:
                    slot = slots_by_id[
                        _slot_key(
                            "crossing_pedestrian", later_timestep, direction
                        )
                    ]
                    if slot["status"] == "not_evaluated_short_circuit":
                        slot["skip_reason"] = "direction_already_failed"
                break
        if not direction_failed:
            _set_remaining_skip_reason(
                slots_by_id,
                "crossing_direction_already_selected",
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
                "initial_direction_method": INITIAL_DIRECTION_METHOD,
                "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
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
        "initial_direction_method": INITIAL_DIRECTION_METHOD,
        "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
        "shared_validation": shared_public,
        "final_failure_reasons": observed_direction_failures,
        "candidate_evaluations": slots,
    }


def combined_candidate_slots() -> List[dict]:
    slots = []
    for benign, modes in ((False, ("standing",)), (True, ("car", "standing", "crossing"))):
        for mode in modes:
            sides = fast.benign_placements.preferred_sides(mode) if benign else (None,)
            variant = f"{'benign' if benign else 'hazardous'}_{mode}"
            for side in sides:
                for timestep in TIMESTEPS:
                    slots.append(dict(
                        candidate_id=f"{variant}_{side or 'fixed'}_t{timestep}",
                        mode=mode, variant=variant, benign=benign, side=side,
                        future_timestep=timestep, status="not_evaluated_short_circuit",
                        skip_reason="not_reached",
                    ))
    return slots


def extend_with_benign_requirements(decision: dict, evaluate_variant: Callable) -> dict:
    """Require one complete preferred/fallback side for each benign actor mode.

    The legacy car/crossing checks run first, then hazardous standing, followed
    by benign car, standing, walking. Failure skips all remaining requirements.
    """
    slots = combined_candidate_slots()
    decision["candidate_evaluations"].extend(slots)
    decision["selected_configurations"] = {}
    by_id = {item["candidate_id"]: item for item in slots}
    if decision["classification"] == "failure":
        _set_remaining_skip_reason(by_id, "hazardous_requirements_failed")
        return decision
    selection = decision["selected_configurations"]
    selection.update(hazardous_car={}, hazardous_crossing={"direction": decision["classification"]})
    order = max(item.get("evaluation_order", 0) for item in decision["candidate_evaluations"])
    for variant in dict.fromkeys(item["variant"] for item in slots):
        family = [item for item in slots if item["variant"] == variant]
        failures = []
        for side in dict.fromkeys(item["side"] for item in family):
            attempt = [item for item in family if item["side"] == side]
            for slot in attempt:
                order += 1
                raw = evaluate_variant(slot["mode"], slot["benign"], side, slot["future_timestep"])
                slot.pop("skip_reason", None)
                slot.update(compact_candidate(raw), status="evaluated", evaluation_order=order)
                if not slot["passed"]:
                    failures.append(dict(
                        _failure_reason(_first_failure(slot), mode=slot["mode"],
                                        timestep=slot["future_timestep"]),
                        variant=variant, side=side,
                    ))
                    _set_remaining_skip_reason(
                        {item["candidate_id"]: item for item in attempt}, "configuration_already_failed",
                    )
                    break
            else:
                selection[variant] = {"side": side} if side is not None else {}
                _set_remaining_skip_reason(
                    {item["candidate_id"]: item for item in family}, "preferred_configuration_passed",
                )
                break
        else:
            decision.update(
                classification="failure", classification_reason=f"{variant}_failed_all_configurations",
                classification_display=f"failure ({variant}:{failures[-1]['code']})",
                final_failure_reasons=failures,
                selected_configurations={},
            )
            _set_remaining_skip_reason(by_id, "required_variant_failed")
            return decision
    decision["classification_reason"] = "all_hazardous_and_benign_variants_passed_all_timesteps"
    return decision


def input_failure_decision(token: str, seed: int, exc: Exception, include_benign=False) -> dict:
    direction, uniform_value = initial_direction_choice(token, seed)
    failure = _failure_reason(
        multi.exception_diagnostic(exc), mode="shared_input"
    )
    slots = candidate_slots()
    if include_benign:
        slots.extend(combined_candidate_slots())
    for slot in slots:
        slot["skip_reason"] = "shared_input_failed"
    return {
        "classification": "failure",
        "classification_reason": "shared_input_failed",
        "classification_display": f"failure (shared_input:{failure['code']})",
        "initial_direction": direction,
        "initial_direction_uniform_value": uniform_value,
        "initial_direction_method": INITIAL_DIRECTION_METHOD,
        "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
        "shared_validation": {
            "status": "not_evaluated_short_circuit",
            "skip_reason": "shared_input_failed",
        },
        "final_failure_reasons": [failure],
        "candidate_evaluations": slots,
    }


class LidarCache:
    """Bounded per-log LRU cache for decoded overlapping LiDAR frames."""

    def __init__(self, data_root: Path, storage_split: str, maximum: int):
        if maximum < 1:
            raise ValueError("LiDAR cache size must be positive")
        self.data_root = Path(data_root)
        self.storage_split = str(storage_split)
        self.maximum = int(maximum)
        self._entries: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, frame: dict) -> np.ndarray:
        lidar_path = frame.get("lidar_path")
        if not isinstance(lidar_path, str) or not lidar_path:
            raise fast.TokenFailure(
                "missing_sensor", "history frame does not contain a LiDAR path"
            )
        if lidar_path in self._entries:
            points = self._entries.pop(lidar_path)
            self._entries[lidar_path] = points
            return points
        points = fast.geometry.load_lidar_xyz(
            self.data_root, self.storage_split, frame
        )
        self._entries[lidar_path] = points
        while len(self._entries) > self.maximum:
            self._entries.popitem(last=False)
        return points


def evaluate_stationary_car_candidate(
    *,
    args: argparse.Namespace,
    scene_frames: Sequence[dict],
    frames: Sequence[dict],
    timestamps_s: np.ndarray,
    frame_lidar_points: Sequence[np.ndarray],
    global_to_current: np.ndarray,
    front_bumper_offset_m: float,
    ego_geometry: Mapping[str, float],
    asset: dict,
    future_timestep: int,
) -> dict:
    """Pure stationary-car equivalent of the production preparation branch."""
    candidate = {
        "candidate_id": f"car_t{int(future_timestep)}",
        "future_timestep": int(future_timestep),
        "mode": "standing_car",
        "evaluation_complete": False,
        "renderable": False,
        "passed": False,
        "failures": [],
        "_states": [],
    }
    try:
        local_bounds = asset["actor_geometry"]["local_bounds_m"]
        trajectory = fast.resolve_stationary_vehicle_trajectory(
            scene_frames,
            timestamps_s,
            timestep=int(future_timestep),
            front_bumper_offset_m=float(front_bumper_offset_m),
            car_local_bounds=local_bounds,
            ego_rear_length_m=float(ego_geometry["rear_length_m"]),
            ego_width_m=float(ego_geometry["width_m"]),
        )
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
            trajectory["actor_current_xys"],
        )
        if strict_ground_failures:
            candidate["failures"].append(
                multi.diagnostic(
                    "ground_fit_failed",
                    "the strict same-frame local ground fit failed at one or "
                    "more history frames",
                    failed_frame_indices=[
                        int(item["frame_index"])
                        for item in strict_ground_failures
                    ],
                    strict_failures=strict_ground_failures,
                )
            )

        final_lidar_to_global = np.asarray(
            frames[-1].get("lidar2global", []), dtype=np.float64
        )
        if (
            final_lidar_to_global.shape != (4, 4)
            or not np.isfinite(final_lidar_to_global).all()
        ):
            raise fast.TokenFailure(
                "invalid_lidar_pose",
                "final history frame has an invalid lidar2global transform",
            )
        final_plane = ground_planes[-1]
        final_actor_lidar = fast.ground_point_lidar_for_global_xy(
            final_lidar_to_global,
            final_plane,
            trajectory["anchor_global_xy"],
        )
        final_basis = fast.road_basis_aligned_to_global_forward(
            final_plane,
            final_lidar_to_global,
            trajectory["reference_vehicle_forward_global_xy"],
            args.max_ground_tilt_deg,
        )
        canonical_actor_to_global = (
            final_lidar_to_global
            @ fast.render_utils.matrix4(final_basis, final_actor_lidar)
        )
        bbox_actor = fast.local_bbox_corners(local_bounds)
        states = []
        per_frame_collisions = []
        for frame_index, (
            frame,
            lidar_xyz,
            plane,
            ground_details,
        ) in enumerate(
            zip(frames, frame_lidar_points, ground_planes, ground_diagnostics)
        ):
            lidar_to_global = np.asarray(
                frame.get("lidar2global", []), dtype=np.float64
            )
            if (
                lidar_to_global.shape != (4, 4)
                or not np.isfinite(lidar_to_global).all()
            ):
                raise fast.TokenFailure(
                    "invalid_lidar_pose",
                    f"history frame {frame_index} has an invalid lidar2global",
                )
            actor_to_lidar = (
                np.linalg.inv(lidar_to_global) @ canonical_actor_to_global
            )
            lidar_to_actor = np.linalg.inv(actor_to_lidar)
            actor_lidar = actor_to_lidar[:3, 3].copy()
            actor_global = canonical_actor_to_global[:3, 3].copy()
            actor_current = fast.geometry.transform_points(
                global_to_current, actor_global[None]
            )[0]
            collision = fast.stationary_vehicle_collision_diagnostics(
                lidar_xyz,
                lidar_to_actor,
                local_bounds,
            )
            collision["frame_index"] = int(frame_index)
            per_frame_collisions.append(collision)
            bbox_lidar = fast.geometry.transform_points(
                actor_to_lidar, bbox_actor
            )
            camera_visibility = {
                camera: multi.actor_camera_visibility(
                    bbox_lidar, frame["cams"][camera]
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
            states.append(
                {
                    "frame_index": int(frame_index),
                    "actor_lidar": actor_lidar,
                    "actor_global": actor_global,
                    "actor_current": actor_current,
                    "actor_to_lidar": actor_to_lidar,
                    "lidar_to_actor": lidar_to_actor,
                    "road_plane_lidar": np.asarray(plane, dtype=np.float64),
                    "road_basis_lidar": actor_to_lidar[:3, :3].copy(),
                    "bbox_lidar": bbox_lidar,
                    "visible_cameras": visible_cameras,
                    "fully_visible_cameras": fully_visible_cameras,
                    "camera_visibility": camera_visibility,
                    "ground_diagnostics": ground_details,
                    "collision_diagnostics": collision,
                }
            )

        occupied_frames = [
            int(item["frame_index"])
            for item in per_frame_collisions
            if item["occupied"]
        ]
        collision_occupied = bool(occupied_frames)
        if collision_occupied:
            candidate["failures"].append(
                multi.diagnostic(
                    "vehicle_placement_occupied",
                    "LiDAR-supported scene geometry intersects the inserted "
                    "car at one or more history frames",
                    occupied_frame_indices=occupied_frames,
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
                multi.diagnostic(
                    "no_camera_visibility",
                    "the car bounding volume is outside all four-history-frame "
                    "front-camera views",
                )
            )
        candidate.update(
            {
                "evaluation_complete": True,
                "renderable": True,
                "passed": not candidate["failures"],
                "placement": trajectory["placement"],
                "strict_ground_failures": strict_ground_failures,
                "collision_diagnostics": {
                    "per_frame": per_frame_collisions,
                    "swept_final_frame": dict(per_frame_collisions[-1]),
                    "swept_path_includes_future_interception": False,
                    "swept_path_point_count": 1,
                    "occupied": collision_occupied,
                },
                "visible_view_count": int(visible_view_count),
                "fully_visible_view_count": int(fully_visible_view_count),
                "_states": states,
            }
        )
    except Exception as exc:
        candidate["failures"].append(
            multi.exception_diagnostic(
                exc,
                future_timestep=int(future_timestep),
                direction="standing_car",
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


def evaluate_shared_preparation(
    *,
    args: argparse.Namespace,
    frames: Sequence[dict],
    car_candidates: Sequence[Mapping[str, Any]],
) -> dict:
    """Run production lighting/ephemeris checks once for the whole token."""
    result = {"passed": False, "failures": []}
    try:
        representative_states = car_candidates[0].get("_states", [])
        if len(representative_states) != fast.EXPECTED_HISTORY_FRAMES:
            raise fast.TokenFailure(
                "shared_validation_failed",
                "representative car candidate has no complete state sequence",
            )
        final_state = representative_states[-1]
        lighting_views = []
        for camera in fast.CAMERA_ORDER:
            calibration = frames[-1]["cams"][camera]
            camera_to_actor = (
                final_state["lidar_to_actor"]
                @ fast.render_utils.camera_to_lidar(calibration)
            )
            source_path = fast.frame_source_path(
                args.data_root, args.split, frames[-1], camera
            )
            try:
                with Image.open(source_path) as source_image:
                    image = np.asarray(source_image.convert("RGB"))
            except Exception as exc:
                raise fast.TokenFailure(
                    "missing_sensor",
                    f"cannot decode {camera} image {source_path}: {exc}",
                ) from exc
            lighting_views.append(
                {
                    "camera": camera,
                    "image": image,
                    "calibration": calibration,
                    "camera_to_actor": camera_to_actor,
                }
            )
        estimated_lighting = fast.estimate_multiview_lighting(lighting_views)
        astronomical = fast.astronomical_sun_direction(frames[-1])
        sun_ray_global = np.asarray(
            astronomical["sun_ray_direction_global_utm"], dtype=np.float64
        )
        for candidate in car_candidates:
            states = candidate.get("_states", [])
            if len(states) != len(frames):
                raise fast.TokenFailure(
                    "shared_validation_failed",
                    "car candidate has no complete state sequence",
                )
            for state, frame in zip(states, frames):
                fast.global_direction_to_actor(
                    sun_ray_global,
                    np.asarray(frame["lidar2global"], dtype=np.float64),
                    state["lidar_to_actor"],
                )
        result.update(
            {
                "passed": True,
                "lighting_method": str(estimated_lighting.get("method", "unknown")),
                "astronomical_method": str(astronomical["method"]),
                "sun_elevation_degrees": float(
                    astronomical["elevation_geometric_degrees"]
                ),
                "direct_sun_enabled": bool(astronomical["direct_sun_enabled"]),
                "_sun_ray_global": sun_ray_global,
            }
        )
    except Exception as exc:
        result["failures"].append(multi.exception_diagnostic(exc))
    return result


def validate_crossing_sun(
    candidate: dict,
    frames: Sequence[dict],
    sun_ray_global: np.ndarray,
) -> dict:
    """Apply the cached token sun direction to one pedestrian state sequence."""
    if not candidate.get("passed", False):
        return candidate
    try:
        states = candidate.get("_states", [])
        if len(states) != len(frames):
            raise fast.TokenFailure(
                "sun_direction_failed",
                "crossing candidate has no complete state sequence",
            )
        for state, frame in zip(states, frames):
            # The evaluator retains the road basis, which defines actor
            # rotation in LiDAR before the render-time heading is applied.
            basis = np.asarray(state["road_basis_lidar"], dtype=np.float64)
            actor_to_lidar = fast.render_utils.matrix4(
                basis,
                np.asarray(state["actor_lidar"], dtype=np.float64),
            )
            lidar_to_actor = np.linalg.inv(actor_to_lidar)
            fast.global_direction_to_actor(
                sun_ray_global,
                np.asarray(frame["lidar2global"], dtype=np.float64),
                lidar_to_actor,
            )
    except Exception as exc:
        candidate["failures"].append(multi.exception_diagnostic(exc))
        candidate["passed"] = False
    return candidate


def classify_scene_frames(
    *,
    worker: Mapping[str, Any],
    token: str,
    log_name: str,
    scene_frames: Sequence[dict],
    lidar_cache: LidarCache,
) -> dict:
    started = time.perf_counter()
    args = worker["geometry_args"]
    pedestrian = fast.select_asset(token, worker["seed"], worker["pedestrians"])
    car = fast.select_asset(token, worker["seed"], worker["cars"])
    try:
        frames, timestamps_s, frame_lidar_points = multi.validate_token_inputs(
            args,
            scene_frames,
            lidar_loader=lidar_cache.get,
        )
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
        animation = fast.animation_frames(
            timestamps_s,
            args.speed_mps,
            pedestrian["stride_length_m"],
            pedestrian["action_range"][0],
            pedestrian["action_range"][1],
            fast.NAVSIM_POSE_SEQUENCE[0],
        )
        shared_cache: Dict[str, Any] = {}

        def evaluate_car(timestep: int) -> dict:
            return evaluate_stationary_car_candidate(
                args=args,
                scene_frames=scene_frames,
                frames=frames,
                timestamps_s=timestamps_s,
                frame_lidar_points=frame_lidar_points,
                global_to_current=global_to_current,
                front_bumper_offset_m=worker["front_bumper_offset_m"],
                ego_geometry=worker["ego_geometry"],
                asset=car,
                future_timestep=timestep,
            )

        def validate_shared(car_candidates: Sequence[Mapping[str, Any]]) -> dict:
            result = evaluate_shared_preparation(
                args=args,
                frames=frames,
                car_candidates=car_candidates,
            )
            if result.get("passed"):
                shared_cache["sun_ray_global"] = result["_sun_ray_global"]
            return result

        def evaluate_crossing(direction: str, timestep: int) -> dict:
            candidate = multi.evaluate_candidate(
                args=argparse.Namespace(**{**vars(args), "actor_mode": "crossing"}),
                token=token,
                scene_frames=scene_frames,
                frames=frames,
                timestamps_s=timestamps_s,
                frame_lidar_points=frame_lidar_points,
                global_to_current=global_to_current,
                front_bumper_offset_m=worker["front_bumper_offset_m"],
                asset=pedestrian,
                animation=animation,
                future_timestep=timestep,
                direction_spec=multi.DIRECTION_BY_CLASSIFICATION[direction],
            )
            return validate_crossing_sun(
                candidate,
                frames,
                shared_cache["sun_ray_global"],
            )

        decision = classify_short_circuit(
            token=token,
            seed=worker["seed"],
            evaluate_car=evaluate_car,
            evaluate_crossing=evaluate_crossing,
            validate_shared=validate_shared,
        )
        if getattr(args, "include_benign_scenarios", False):
            def evaluate_variant(mode, benign, side, timestep):
                variant_args = argparse.Namespace(**{
                    **vars(args), "actor_mode": mode, "benign": benign,
                    "benign_side": side or "left",
                    "ego_vehicle_geometry": worker["ego_geometry"],
                })
                candidate = multi.evaluate_candidate(
                    args=variant_args, token=token, scene_frames=scene_frames,
                    frames=frames, timestamps_s=timestamps_s,
                    frame_lidar_points=frame_lidar_points, global_to_current=global_to_current,
                    front_bumper_offset_m=worker["front_bumper_offset_m"],
                    asset=car if mode == "car" else pedestrian,
                    animation=(animation if mode == "crossing" else
                               [fast.CAR_STATIC_ANIMATION_FRAME if mode == "car"
                                else fast.STANDING_ANIMATION_FRAME] * len(frames)),
                    future_timestep=timestep,
                )
                return validate_crossing_sun(candidate, frames, shared_cache["sun_ray_global"])

            decision = extend_with_benign_requirements(decision, evaluate_variant)
    except Exception as exc:
        decision = input_failure_decision(
            token, worker["seed"], exc, getattr(args, "include_benign_scenarios", False),
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "configuration_fingerprint": worker["configuration_fingerprint"],
        "token": token,
        "split": worker["split_name"],
        "storage_split": worker["storage_split"],
        "log_name": log_name,
        "classification": decision["classification"],
        "classification_reason": decision["classification_reason"],
        "classification_display": decision["classification_display"],
        "initial_direction": decision["initial_direction"],
        "initial_direction_uniform_value": decision[
            "initial_direction_uniform_value"
        ],
        "initial_direction_method": decision["initial_direction_method"],
        "initial_direction_namespace": decision["initial_direction_namespace"],
        "selected_assets": {
            "standing_car": asset_identity(car),
            "crossing_pedestrian": asset_identity(pedestrian),
            **({"standing_pedestrian": asset_identity(pedestrian)}
               if getattr(args, "include_benign_scenarios", False) else {}),
        },
        "selected_configurations": decision.get("selected_configurations", {}),
        "shared_validation": decision["shared_validation"],
        "final_failure_reasons": decision["final_failure_reasons"],
        "candidate_evaluations": decision["candidate_evaluations"],
        "duration_seconds": time.perf_counter() - started,
    }


_WORKER_CONTEXT: Optional[dict] = None


def initialize_worker(context: dict) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    _WORKER_CONTEXT["selected_tokens"] = set(context["selected_tokens"])
    fast.cv2.setNumThreads(1)
    fast.configure_navsim(Path(context["geometry_args"].data_root))


def log_pickle_path(data_root: Path, storage_split: str, log_name: str) -> Path:
    filename = log_name if log_name.endswith(".pkl") else f"{log_name}.pkl"
    return data_root / "navsim_logs" / storage_split / filename


def process_log(log_name: str) -> dict:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context was not initialized")
    worker = _WORKER_CONTEXT
    path = log_pickle_path(
        Path(worker["geometry_args"].data_root),
        worker["storage_split"],
        log_name,
    )
    if not path.is_file():
        raise ClassificationError(f"NAVSIM log pickle is missing: {path}")
    try:
        with path.open("rb") as handle:
            raw_frames = pickle.load(handle)
    except Exception as exc:
        raise ClassificationError(f"Could not load NAVSIM log {path}: {exc}") from exc
    if not isinstance(raw_frames, list):
        raise ClassificationError(f"NAVSIM log is not a frame list: {path}")

    selected_tokens = worker["selected_tokens"]
    selected_positions = []
    seen_in_log = set()
    for index, frame in enumerate(raw_frames):
        if not isinstance(frame, dict):
            raise ClassificationError(
                f"NAVSIM log {path} contains a non-object frame at index {index}"
            )
        token = str(frame.get("token", ""))
        if token in selected_tokens:
            if token in seen_in_log:
                raise ClassificationError(
                    f"NAVSIM log {path} contains selected token {token} twice"
                )
            seen_in_log.add(token)
            selected_positions.append((index, token))

    lidar_cache = LidarCache(
        Path(worker["geometry_args"].data_root),
        worker["storage_split"],
        worker["lidar_cache_size"],
    )
    # The production filter carries ten future frames.  Preserve the complete
    # 14-frame window because placement/first-contact checks inspect all frames
    # through the requested future anchor.
    required_frames = int(worker["num_scene_frames"])
    records = []
    for current_index, token in selected_positions:
        start_index = current_index - (fast.EXPECTED_HISTORY_FRAMES - 1)
        end_index = start_index + required_frames
        if start_index < 0 or end_index > len(raw_frames):
            decision = input_failure_decision(
                token,
                worker["seed"],
                fast.TokenFailure(
                    "scene_window_unavailable",
                    f"cannot construct [{start_index}:{end_index}] from "
                    f"{len(raw_frames)} frames in {path.name}",
                ),
                include_benign=getattr(worker["geometry_args"], "include_benign_scenarios", False),
            )
            car = fast.select_asset(token, worker["seed"], worker["cars"])
            pedestrian = fast.select_asset(
                token, worker["seed"], worker["pedestrians"]
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "pipeline_version": PIPELINE_VERSION,
                    "configuration_fingerprint": worker[
                        "configuration_fingerprint"
                    ],
                    "token": token,
                    "split": worker["split_name"],
                    "storage_split": worker["storage_split"],
                    "log_name": log_name,
                    "classification": "failure",
                    "classification_reason": decision["classification_reason"],
                    "classification_display": decision["classification_display"],
                    "initial_direction": decision["initial_direction"],
                    "initial_direction_uniform_value": decision[
                        "initial_direction_uniform_value"
                    ],
                    "initial_direction_method": decision[
                        "initial_direction_method"
                    ],
                    "initial_direction_namespace": decision[
                        "initial_direction_namespace"
                    ],
                    "selected_assets": {
                        "standing_car": asset_identity(car),
                        "crossing_pedestrian": asset_identity(pedestrian),
                    },
                    "selected_configurations": {},
                    "shared_validation": decision["shared_validation"],
                    "final_failure_reasons": decision[
                        "final_failure_reasons"
                    ],
                    "candidate_evaluations": decision[
                        "candidate_evaluations"
                    ],
                    "duration_seconds": 0.0,
                }
            )
            continue
        scene_frames = raw_frames[start_index:end_index]
        if str(scene_frames[fast.EXPECTED_HISTORY_FRAMES - 1].get("token")) != token:
            raise ClassificationError(
                f"constructed window does not center selected token {token}"
            )
        records.append(
            classify_scene_frames(
                worker=worker,
                token=token,
                log_name=log_name,
                scene_frames=scene_frames,
                lidar_cache=lidar_cache,
            )
        )
    return {"log_name": log_name, "records": records}


def accepted_placement(record: Mapping[str, Any]) -> dict:
    """Export only the selected complete configurations, never partial attempts."""
    configurations = record["selected_configurations"]
    candidates = []
    for item in record["candidate_evaluations"]:
        if item["status"] != "evaluated" or not item.get("passed"):
            continue
        if "variant" in item:
            if item.get("side") != configurations[item["variant"]].get("side"):
                continue
        elif item["mode"] == "crossing_pedestrian" and item["direction"] != record["classification"]:
            continue
        candidates.append(item)
    if len(candidates) != 6 * len(TIMESTEPS):
        raise ClassificationError(f"Accepted token {record['token']} has incomplete selected placements")
    return dict(log_name=record["log_name"], selected_assets=record["selected_assets"],
                selected_configurations=configurations, candidates=candidates)


def accepted_benign_sides(records):
    return {record["token"]: {mode: record["selected_configurations"][f"benign_{mode}"]["side"]
                              for mode in ("car", "standing", "crossing")}
            for record in records if record["classification"] != "failure"}


def output_paths(output_dir: Path, split_name: str) -> dict:
    return {
        "checkpoint": output_dir / f"{split_name}_checkpoint.jsonl",
        "checkpoint_meta": output_dir / f"{split_name}_checkpoint.meta.json",
        "classifications": output_dir
        / f"{split_name}_direction_classifications.json",
        "details": output_dir / f"{split_name}_classification_details.jsonl",
        "summary": output_dir / f"{split_name}_summary.json",
        "placements": output_dir / f"{split_name}_accepted_placements.json",
        "benign_sides": output_dir / f"{split_name}_benign_sides.json",
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
            # A killed process may leave only its final append incomplete.
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
                f"Checkpoint token {token} was produced by another configuration"
            )
        if record.get("classification") not in (*DIRECTIONS, "failure"):
            raise ClassificationError(
                f"Checkpoint token {token} has invalid classification "
                f"{record.get('classification')!r}"
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
    existing_outputs = [path for path in paths.values() if path.exists()]
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
    elif existing_outputs:
        raise ClassificationError(
            f"Outputs exist without checkpoint metadata in {meta_path.parent}; "
            "pass --overwrite to start over"
        )
    else:
        write_json_atomic(
            meta_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                **configuration,
            },
        )
    checkpoint_path.touch(exist_ok=True)
    return recover_and_load_checkpoint(
        checkpoint_path,
        allowed_tokens=allowed_tokens,
        configuration_fingerprint=fingerprint,
    )


def summarize_records(
    *,
    records: Sequence[dict],
    requested_count: int,
    configuration: dict,
    started_at: str,
    elapsed_seconds: float,
    resumed_count: int,
) -> dict:
    label_counts = Counter(record["classification"] for record in records)
    reason_counts = Counter(record["classification_reason"] for record in records)
    failure_mode_counts: Counter[str] = Counter()
    failure_timestep_counts: Counter[str] = Counter()
    failure_code_counts: Counter[str] = Counter()
    evaluated_candidate_counts: Counter[str] = Counter()
    observed_candidate_failure_counts: Counter[str] = Counter()
    selected_side_counts: Counter[str] = Counter()
    for record in records:
        for candidate in record["candidate_evaluations"]:
            if candidate["status"] == "evaluated":
                evaluated_candidate_counts[candidate["mode"]] += 1
                for failure in candidate.get("failures", []):
                    variant = candidate.get("variant", f"hazardous_{candidate['mode']}")
                    observed_candidate_failure_counts[f"{variant}:{failure['code']}"] += 1
        if record["classification"] != "failure":
            for variant, choice in record.get("selected_configurations", {}).items():
                if choice.get("side"):
                    selected_side_counts[f"{variant}:{choice['side']}"] += 1
        for failure in record["final_failure_reasons"]:
            failure_mode_counts[str(failure.get("mode", "unknown"))] += 1
            failure_code_counts[str(failure.get("code", "unknown"))] += 1
            timestep = failure.get("future_timestep")
            if timestep is not None:
                failure_timestep_counts[str(int(timestep))] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": float(elapsed_seconds),
        "requested_tokens": int(requested_count),
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
        "evaluated_candidate_counts": dict(
            sorted(evaluated_candidate_counts.items())
        ),
        "observed_candidate_failure_counts": dict(sorted(observed_candidate_failure_counts.items())),
        "accepted_side_counts": dict(sorted(selected_side_counts.items())),
        "configuration": configuration,
    }


def append_checkpoint_batch(path: Path, records: Sequence[dict]) -> None:
    if not records:
        return
    payload = "".join(f"{json_line(record)}\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_configuration(
    *,
    args: argparse.Namespace,
    split_name: str,
    filter_data: Mapping[str, Any],
    pedestrians: Sequence[dict],
    cars: Sequence[dict],
    ego_geometry: Mapping[str, float],
) -> dict:
    selected_tokens = list(filter_data["tokens"])
    base = {
        "pipeline_version": PIPELINE_VERSION,
        "split": split_name,
        "storage_split": SPLIT_SPECS[split_name]["storage_split"],
        "data_root": str(args.data_root),
        "scene_filter": str(filter_data["path"]),
        "scene_filter_sha256": filter_data["path_sha256"],
        "source_filter_token_count": int(filter_data["source_token_count"]),
        "selected_token_count": len(selected_tokens),
        "selected_tokens_sha256": fast.canonical_hash(selected_tokens),
        "log_count": len(filter_data["log_names"]),
        "timesteps": list(TIMESTEPS),
        "short_circuit_order": "standing_car_then_seeded_crossing_direction_v1",
        "seed": int(args.seed),
        "initial_direction_namespace": INITIAL_DIRECTION_NAMESPACE,
        "asset_selection": "production_blake2b_token_seed_asset_v1",
        "pedestrian_assets": [asset_identity(asset) for asset in pedestrians],
        "car_assets": [asset_identity(asset) for asset in cars],
        "speed_mps": float(args.speed_mps),
        "stride_length_m": args.stride_length_m,
        "gait_frame_advance": int(args.gait_frame_advance),
        "gait_timestep_s": float(args.gait_timestep_s),
        "actor_scale": float(args.actor_scale),
        "max_ground_tilt_deg": float(args.max_ground_tilt_deg),
        "ego_vehicle_geometry": json_ready(ego_geometry),
        "lidar_cache_size": int(args.lidar_cache_size),
        "worker_count_independent_selection": True,
        "include_benign_scenarios": bool(args.include_benign_scenarios),
        "front_visibility_requirement": fast.front_visibility.configuration(args),
        "source_sha256": {
            Path(path).name: fast.file_sha256(Path(path))
            for path in (__file__, fast.__file__, multi.__file__,
                         fast.benign_placements.__file__, fast.front_visibility.__file__,
                         fast.geometry.__file__, fast.render_utils.__file__)
        },
    }
    if args.include_benign_scenarios:
        benign_config = fast.benign_placements.configuration(args)
        benign_config.pop("benign_side")
        base.update(
            benign_placement=benign_config,
            side_selection_policy=fast.benign_placements.SIDE_SELECTION_POLICY,
            side_preferences={mode: list(fast.benign_placements.preferred_sides(mode))
                              for mode in ("car", "standing", "crossing")},
            required_variants=[f"{variant}_{mode}" for variant in ("hazardous", "benign")
                               for mode in ("car", "standing", "crossing")],
            short_circuit_order="hazardous_car_crossing_standing_then_benign_car_standing_crossing_v1",
        )
    return {
        **base,
        "configuration_fingerprint": fast.canonical_hash(base),
    }


def run_split(
    *,
    args: argparse.Namespace,
    split_name: str,
    filter_data: Mapping[str, Any],
    pedestrians: Sequence[dict],
    cars: Sequence[dict],
    ego_geometry: Mapping[str, float],
) -> dict:
    started_at = utc_now()
    started = time.perf_counter()
    tokens = list(filter_data["tokens"])
    token_set = set(tokens)
    configuration = build_configuration(
        args=args,
        split_name=split_name,
        filter_data=filter_data,
        pedestrians=pedestrians,
        cars=cars,
        ego_geometry=ego_geometry,
    )
    paths = output_paths(args.output_dir, split_name)
    completed = prepare_checkpoint(
        paths=paths,
        configuration=configuration,
        allowed_tokens=token_set,
        overwrite=args.overwrite,
    )
    resumed_count = len(completed)
    remaining = token_set - completed.keys()
    print(
        f"{split_name}: {len(tokens):,} requested, {resumed_count:,} resumed, "
        f"{len(remaining):,} remaining",
        flush=True,
    )

    if remaining:
        geometry_args = argparse.Namespace(
            data_root=args.data_root,
            split=SPLIT_SPECS[split_name]["storage_split"],
            speed_mps=float(args.speed_mps),
            max_ground_tilt_deg=float(args.max_ground_tilt_deg),
            actor_scale=float(args.actor_scale),
            include_benign_scenarios=bool(args.include_benign_scenarios),
            min_front_visible_fraction=float(args.min_front_visible_fraction),
            benign_offset_m=float(args.benign_offset_m),
            benign_clearance_m=float(args.benign_clearance_m),
            benign_walking_direction=args.benign_walking_direction,
        )
        context = {
            "geometry_args": geometry_args,
            "split_name": split_name,
            "storage_split": SPLIT_SPECS[split_name]["storage_split"],
            "seed": int(args.seed),
            "selected_tokens": tuple(remaining),
            "pedestrians": list(pedestrians),
            "cars": list(cars),
            "front_bumper_offset_m": float(ego_geometry["front_length_m"]),
            "ego_geometry": dict(ego_geometry),
            "lidar_cache_size": int(args.lidar_cache_size),
            "num_scene_frames": int(filter_data["num_history_frames"])
            + int(filter_data["num_future_frames"]),
            "configuration_fingerprint": configuration[
                "configuration_fingerprint"
            ],
        }
        log_names = list(filter_data["log_names"])
        last_reported = len(completed)

        def consume_log_result(result: Mapping[str, Any]) -> None:
            nonlocal last_reported
            batch = result.get("records")
            if not isinstance(batch, list):
                raise ClassificationError(
                    f"Worker returned an invalid batch for {result.get('log_name')}"
                )
            for record in batch:
                token = record.get("token")
                if token not in remaining:
                    raise ClassificationError(
                        f"Worker returned unknown or already completed token {token!r}"
                    )
                if token in completed:
                    raise ClassificationError(
                        f"Worker returned duplicate token {token}"
                    )
                completed[token] = record
            append_checkpoint_batch(paths["checkpoint"], batch)
            current = len(completed)
            if (
                args.progress_every
                and current > last_reported
                and (
                    current - last_reported >= args.progress_every
                    or current == len(tokens)
                )
            ):
                elapsed = time.perf_counter() - started
                rate = (current - resumed_count) / max(elapsed, 1e-9)
                print(
                    f"{split_name}: {current:,}/{len(tokens):,} "
                    f"({100.0 * current / max(len(tokens), 1):.1f}%), "
                    f"{rate:.2f} new samples/s",
                    flush=True,
                )
                last_reported = current

        if args.workers == 1:
            initialize_worker(context)
            for log_name in log_names:
                consume_log_result(process_log(log_name))
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_worker,
                initargs=(context,),
            ) as executor:
                futures = {
                    executor.submit(process_log, log_name): log_name
                    for log_name in log_names
                }
                for future in as_completed(futures):
                    log_name = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        raise ClassificationError(
                            f"Worker failed while processing log {log_name}: {exc}"
                        ) from exc
                    consume_log_result(result)

    missing = [token for token in tokens if token not in completed]
    unknown = sorted(set(completed) - token_set)
    if missing or unknown:
        raise ClassificationError(
            f"{split_name} output coverage is invalid: "
            f"missing={len(missing)} {missing[:5]}, "
            f"unknown={len(unknown)} {unknown[:5]}"
        )
    ordered_records = [completed[token] for token in tokens]
    for record in ordered_records:
        if record["classification"] == "failure" and not record.get(
            "final_failure_reasons"
        ):
            raise ClassificationError(
                f"Failure token {record['token']} has no observed reason"
            )
    classifications = {
        record["token"]: record["classification"] for record in ordered_records
    }
    elapsed = time.perf_counter() - started
    summary = summarize_records(
        records=ordered_records,
        requested_count=len(tokens),
        configuration=configuration,
        started_at=started_at,
        elapsed_seconds=elapsed,
        resumed_count=resumed_count,
    )
    summary["execution"] = {
        "workers": int(args.workers),
        "native_threads_per_worker": 1,
        "log_partitioned": True,
    }
    write_json_atomic(paths["classifications"], classifications)
    write_jsonl_atomic(paths["details"], ordered_records)
    if args.include_benign_scenarios:
        write_json_atomic(paths["benign_sides"], accepted_benign_sides(ordered_records))
        write_json_atomic(paths["placements"], {
            "configuration": configuration,
            "samples": {record["token"]: accepted_placement(record)
                        for record in ordered_records if record["classification"] != "failure"},
        })
    write_json_atomic(paths["summary"], summary)
    print(
        f"{split_name}: complete in {elapsed / 3600.0:.2f} h; "
        f"labels={summary['label_counts']}",
        flush=True,
    )
    return summary


def validate_environment(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"NAVSIM data root is missing: {args.data_root}")
    required_relatives = ["sensor_blobs"]
    required_relatives.extend(
        f"navsim_logs/{SPLIT_SPECS[split_name]['storage_split']}"
        for split_name in args.splits
    )
    for relative in required_relatives:
        path = args.data_root / relative
        if not path.exists():
            raise FileNotFoundError(f"NAVSIM dataset path is missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_environment(args)
    fast.configure_navsim(args.data_root)
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    vehicle = get_pacifica_parameters()
    ego_geometry = {
        "front_length_m": float(vehicle.front_length),
        "rear_length_m": float(vehicle.rear_length),
        "width_m": float(vehicle.width),
    }
    pedestrians, cars = load_assets(args)
    print(
        f"Loaded {len(pedestrians)} pedestrian and {len(cars)} car assets; "
        f"workers={args.workers}",
        flush=True,
    )
    filters = {}
    for split_name in args.splits:
        filter_path = getattr(args, f"{split_name}_filter")
        filters[split_name] = load_filter(
            split_name,
            filter_path,
            seed=args.seed,
            max_samples=args.max_samples,
            allow_nonofficial_counts=args.allow_nonofficial_counts,
        )
        storage_split = SPLIT_SPECS[split_name]["storage_split"]
        log_root = args.data_root / "navsim_logs" / storage_split
        missing_logs = [
            log_name
            for log_name in filters[split_name]["log_names"]
            if not log_pickle_path(args.data_root, storage_split, log_name).is_file()
        ]
        if missing_logs:
            raise FileNotFoundError(
                f"{split_name} references {len(missing_logs)} missing logs under "
                f"{log_root}; first={missing_logs[:3]}"
            )
    summaries = []
    for split_name in args.splits:
        summaries.append(
            run_split(
                args=args,
                split_name=split_name,
                filter_data=filters[split_name],
                pedestrians=pedestrians,
                cars=cars,
                ego_geometry=ego_geometry,
            )
        )
    total = sum(summary["completed_tokens"] for summary in summaries)
    print(f"All requested splits complete: {total:,} samples", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ClassificationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
