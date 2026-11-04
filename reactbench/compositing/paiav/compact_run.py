#!/usr/bin/env python3
"""Validate and compact a completed PAI-AV edit/PCTNet run.

The edit and PCTNet stages intentionally publish a resumable working set.
This module validates that set before converting it to an evaluation-only
layout containing final images, metadata, provenance, and audit summaries.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:
    from .paiav_io import (
        PAIAVDataError,
        RECONSTRUCTION_FEATURES,
        load_selection_manifest,
        resolve_chunk_feature_paths,
    )
except ImportError:  # direct execution by compact_run.sh
    from paiav_io import (  # type: ignore
        PAIAVDataError,
        RECONSTRUCTION_FEATURES,
        load_selection_manifest,
        resolve_chunk_feature_paths,
    )


CAMERAS = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
FRAME_INDICES = tuple(range(4))
EXPECTED_VIEWS = frozenset(
    (camera, frame_index)
    for camera in CAMERAS
    for frame_index in FRAME_INDICES
)
EXPECTED_VIEW_COUNT = len(EXPECTED_VIEWS)
REPORT_FILENAME = "compaction_report.json"
REPORT_VERSION = "paiav_evaluation_only_v1"
SOURCE_INDEX_FILENAME = "source_reconstruction.json"
SOURCE_INDEX_VERSION = "paiav_source_reconstruction_v1"
MULTI_CHUNK_SOURCE_INDEX_VERSION = "paiav_source_reconstruction_v2_multichunk"
SUPPORTED_SOURCE_INDEX_VERSIONS = frozenset(
    {SOURCE_INDEX_VERSION, MULTI_CHUNK_SOURCE_INDEX_VERSION}
)
ANNOTATIONS_FILENAME = "annotations.json"
COMPACT_ANNOTATION_VERSION = "paiav_compact_evaluation_annotation_v1"
ORIGINAL_IMAGE_SOURCE_VERSION = "paiav_archive_camera_frames_v1"

# Exact reproducible run-local products. Never replace this allowlist with a
# prefix or glob based deletion policy.
CLEANUP_DIRECTORIES = (
    "prepared",
    ".finalization_staging",
    "images",
    "masks",
    "actor_masks",
    "completed_tokens",
    "finalization_cache",
    "raw_ideal",
    "raw_actor_depth",
    "raw_actor_sky",
    "raw_actor_sun",
    "raw_ambient_occlusion",
    "raw_labeled_actor_sun",
    "raw_labeled_directional_shadow",
    "manifests",
    "labeled_directional_manifests",
    "render_logs",
    "source_frames",
    "source_lidar",
    "source_relative_depth",
)
CLEANUP_FILES = ("depth_inference.log",)


class CompactionError(RuntimeError):
    """A completed-run invariant was not satisfied."""


@dataclass(frozen=True)
class FrameSpec:
    token: str
    camera: str
    frame_index: int
    source_path: Path
    source_frame_index: int
    source_timestamp_us: int
    nominal_timestamp_us: int
    metadata_mask_pixel_count: int | None
    metadata_actor_mask_pixel_count: int | None
    metadata_changed_pixel_count: int | None

    @property
    def relative_path(self) -> Path:
        return Path(self.token) / self.camera / f"{self.frame_index}.png"


@dataclass(frozen=True)
class TokenSpec:
    token: str
    frames: tuple[FrameSpec, ...]
    paiav_source: dict[str, Any]


@dataclass(frozen=True)
class FrameValidation:
    spec: FrameSpec
    width: int
    height: int
    mask_pixel_count: int
    actor_mask_pixel_count: int
    composite_changed_pixel_count: int
    pctnet_changed_pixel_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel token validators (default: min(4, available CPUs)).",
    )
    parser.add_argument(
        "--allowed-mask-threshold",
        type=int,
        default=1,
        help="Compositor changes must be inside mask values at least this large.",
    )
    parser.add_argument(
        "--pctnet-mask-threshold",
        type=int,
        default=250,
        help="PCTNet changes must be inside mask values at least this large.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate without writing provenance, reports, or deleting data.",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    for name in ("allowed_mask_threshold", "pctnet_mask_threshold"):
        value = getattr(args, name)
        if not 0 <= value <= 255:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 255]")
    if args.allowed_mask_threshold > args.pctnet_mask_threshold:
        parser.error(
            "--allowed-mask-threshold cannot exceed --pctnet-mask-threshold"
        )
    return args


def load_json(path: Path, label: str) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CompactionError(f"Could not read {label} {path}: {exc}") from exc


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_run_dir(path: Path) -> Path:
    run_dir = path.expanduser().resolve()
    if not run_dir.is_dir():
        raise CompactionError(f"Run directory does not exist: {run_dir}")
    for required in ("final_images", "metadata"):
        candidate = run_dir / required
        if not candidate.is_dir():
            raise CompactionError(f"Run is missing required directory: {candidate}")
    for required in ("run_config.json", "summary.json"):
        candidate = run_dir / required
        if not candidate.is_file():
            raise CompactionError(f"Run is missing required file: {candidate}")
    return run_dir


def optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CompactionError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise CompactionError(f"{label} must be non-negative")
    return result


def required_integer_list(
    source: dict[str, Any], key: str, expected_length: int, label: str
) -> list[int]:
    values = source.get(key)
    if not isinstance(values, list) or len(values) != expected_length:
        raise CompactionError(
            f"{label}.{key} must contain {expected_length} entries"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise CompactionError(f"{label}.{key} must contain integers")
    return [int(value) for value in values]


def load_token_spec(metadata_path: Path) -> TokenSpec:
    metadata = load_json(metadata_path, "metadata")
    if not isinstance(metadata, dict):
        raise CompactionError(f"Metadata must be an object: {metadata_path}")
    token = metadata.get("token")
    if token != metadata_path.stem or not isinstance(token, str) or not token:
        raise CompactionError(
            f"Metadata token does not match filename {metadata_path.name}"
        )
    if metadata.get("status") != "success":
        raise CompactionError(f"Metadata is not successful: {metadata_path}")

    paiav_source = metadata.get("paiav_source")
    if not isinstance(paiav_source, dict) or paiav_source.get("clip_id") != token:
        raise CompactionError(f"Missing PAI-AV source provenance: {metadata_path}")
    nominal_timestamps = required_integer_list(
        paiav_source,
        "nominal_history_timestamps_us",
        len(FRAME_INDICES),
        f"paiav_source for {token}",
    )
    frame_indices_by_camera = paiav_source.get("camera_source_frame_indices")
    timestamps_by_camera = paiav_source.get("camera_source_timestamps_us")
    if not isinstance(frame_indices_by_camera, dict) or not isinstance(
        timestamps_by_camera, dict
    ):
        raise CompactionError(f"Incomplete camera source provenance for {token}")

    camera_frame_indices = {
        camera: required_integer_list(
            frame_indices_by_camera,
            camera,
            len(FRAME_INDICES),
            f"camera_source_frame_indices for {token}",
        )
        for camera in CAMERAS
    }
    camera_timestamps = {
        camera: required_integer_list(
            timestamps_by_camera,
            camera,
            len(FRAME_INDICES),
            f"camera_source_timestamps_us for {token}",
        )
        for camera in CAMERAS
    }

    records = metadata.get("records")
    if not isinstance(records, list):
        raise CompactionError(f"Metadata records must be a list: {metadata_path}")
    frames: list[FrameSpec] = []
    observed: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CompactionError(
                f"Metadata record {index} is not an object: {metadata_path}"
            )
        camera = record.get("camera")
        frame_index = record.get("frame_index")
        if camera not in CAMERAS or frame_index not in FRAME_INDICES:
            raise CompactionError(
                f"Invalid camera/frame in {metadata_path}: {camera!r}/{frame_index!r}"
            )
        view = (str(camera), int(frame_index))
        if view in observed:
            raise CompactionError(f"Duplicate view {view} in {metadata_path}")
        observed.add(view)
        source_value = record.get("source_path")
        if not isinstance(source_value, str) or not source_value:
            raise CompactionError(f"Missing source_path for {token}/{camera}/{frame_index}")
        nominal_timestamp = nominal_timestamps[int(frame_index)]
        if record.get("timestamp_us") != nominal_timestamp:
            raise CompactionError(
                f"Nominal timestamp mismatch for {token}/{camera}/{frame_index}"
            )
        frames.append(
            FrameSpec(
                token=token,
                camera=str(camera),
                frame_index=int(frame_index),
                source_path=Path(source_value).expanduser(),
                source_frame_index=camera_frame_indices[str(camera)][int(frame_index)],
                source_timestamp_us=camera_timestamps[str(camera)][int(frame_index)],
                nominal_timestamp_us=nominal_timestamp,
                metadata_mask_pixel_count=optional_nonnegative_int(
                    record.get("mask_pixel_count"),
                    f"mask_pixel_count for {token}/{camera}/{frame_index}",
                ),
                metadata_actor_mask_pixel_count=optional_nonnegative_int(
                    record.get("actor_mask_pixel_count"),
                    f"actor_mask_pixel_count for {token}/{camera}/{frame_index}",
                ),
                metadata_changed_pixel_count=optional_nonnegative_int(
                    record.get("changed_pixel_count"),
                    f"changed_pixel_count for {token}/{camera}/{frame_index}",
                ),
            )
        )
    if observed != EXPECTED_VIEWS:
        raise CompactionError(
            f"Token {token} does not contain the complete 16-view set; "
            f"missing={sorted(EXPECTED_VIEWS - observed)}, "
            f"unexpected={sorted(observed - EXPECTED_VIEWS)}"
        )
    frames.sort(key=lambda frame: (frame.camera, frame.frame_index))
    return TokenSpec(token, tuple(frames), paiav_source)


def metadata_specs(run_dir: Path) -> dict[str, TokenSpec]:
    paths = sorted((run_dir / "metadata").glob("*.json"))
    if not paths:
        raise CompactionError(f"No metadata under {run_dir / 'metadata'}")
    result: dict[str, TokenSpec] = {}
    for path in paths:
        token_spec = load_token_spec(path)
        if token_spec.token in result:
            raise CompactionError(f"Duplicate metadata token: {token_spec.token}")
        result[token_spec.token] = token_spec
    return result


def failure_tokens(run_dir: Path) -> set[str]:
    directory_tokens: set[str] = set()
    directory = run_dir / "failures"
    if directory.is_dir():
        directory_tokens = {
            path.stem for path in directory.glob("*.json") if path.is_file()
        }
    jsonl_tokens: set[str] = set()
    jsonl = run_dir / "failures.jsonl"
    if jsonl.is_file():
        for line_number, line in enumerate(
            jsonl.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CompactionError(
                    f"Invalid failures.jsonl line {line_number}: {exc}"
                ) from exc
            token = record.get("token") if isinstance(record, dict) else None
            if not isinstance(token, str) or not token:
                raise CompactionError(
                    f"Missing token in failures.jsonl line {line_number}"
                )
            jsonl_tokens.add(token)
    if directory_tokens and jsonl_tokens and directory_tokens != jsonl_tokens:
        raise CompactionError(
            "Per-token failure files and failures.jsonl contain different tokens"
        )
    return directory_tokens or jsonl_tokens


WALKING_ONLY_METADATA_FIELDS = frozenset(
    {
        "direction_classification",
        "crossing_direction",
        "final_crossing_offset_m",
        "crossing_offsets_m",
        "walking_velocity_global_xy_mps",
        "animation_pose_sequence",
    }
)


def validate_actor_mode_contract(
    run_dir: Path,
    run_config: dict[str, Any],
    successful_tokens: set[str],
) -> None:
    """Validate mode-specific metadata before destructive compaction."""
    configured_mode = run_config.get("actor_mode")
    if configured_mode is None:
        # Preserve validation support for compacted runs created before actor
        # modes became an explicit schema field.
        return
    if configured_mode not in {"crossing", "standing", "car"}:
        raise CompactionError(f"Unsupported run actor_mode: {configured_mode!r}")
    directions = run_config.get("directions")
    if configured_mode in {"standing", "car"} and (
        not isinstance(directions, dict)
        or directions.get("source") != "not_applicable_stationary_actor"
    ):
        raise CompactionError(
            f"{configured_mode} run config must declare directions not applicable"
        )

    for token in sorted(successful_tokens):
        path = run_dir / "metadata" / f"{token}.json"
        metadata = load_json(path, "metadata")
        if not isinstance(metadata, dict):
            raise CompactionError(f"Metadata must be an object: {path}")
        if metadata.get("actor_mode") != configured_mode:
            raise CompactionError(
                f"Actor mode mismatch for {token}: "
                f"run={configured_mode!r}, metadata={metadata.get('actor_mode')!r}"
            )
        annotation = metadata.get("annotation_sample")
        if not isinstance(annotation, dict) or annotation.get("actor_mode") != (
            configured_mode
        ):
            raise CompactionError(
                f"Annotation actor mode is missing or inconsistent for {token}"
            )

        if configured_mode == "crossing":
            required = {
                "direction_classification",
                "crossing_direction",
                "crossing_offsets_m",
                "walking_velocity_global_xy_mps",
                "animation_pose_sequence",
            }
            missing = sorted(required - metadata.keys())
            if missing:
                raise CompactionError(
                    f"Crossing metadata for {token} lacks {missing}"
                )
            continue

        unexpected = sorted(WALKING_ONLY_METADATA_FIELDS & metadata.keys())
        annotation_unexpected = sorted(
            WALKING_ONLY_METADATA_FIELDS & annotation.keys()
        )
        if unexpected or annotation_unexpected:
            raise CompactionError(
                f"Stationary metadata for {token} contains walking-only fields: "
                f"metadata={unexpected}, annotation={annotation_unexpected}"
            )
        placement = metadata.get("placement")
        if not isinstance(placement, dict):
            raise CompactionError(f"Stationary metadata lacks placement for {token}")

        if configured_mode == "standing":
            if (
                metadata.get("actor_type") != "pedestrian"
                or metadata.get("actor_behavior") != "standing"
                or metadata.get("pedestrian_speed_mps") != 0.0
                or placement.get("pedestrian_behavior") != "standing"
                or not isinstance(metadata.get("pedestrian_geometry"), dict)
            ):
                raise CompactionError(
                    f"Standing metadata contract is incomplete for {token}"
                )
            continue

        asset = metadata.get("asset")
        attribution = asset.get("attribution") if isinstance(asset, dict) else None
        preparation = asset.get("preparation") if isinstance(asset, dict) else None
        if (
            metadata.get("actor_type") != "car"
            or metadata.get("actor_behavior") != "stationary"
            or metadata.get("actor_speed_mps") != 0.0
            or placement.get("vehicle_behavior") != "stationary"
            or not isinstance(metadata.get("vehicle_geometry"), dict)
            or not isinstance(metadata.get("first_contact_validation"), dict)
            or not isinstance(attribution, dict)
            or not isinstance(attribution.get("license"), str)
            or not attribution["license"].strip()
            or not isinstance(preparation, dict)
            or preparation.get("validation_passed") is not True
            or preparation.get("self_contained") is not True
        ):
            raise CompactionError(f"Car metadata contract is incomplete for {token}")


def validate_run_contract(
    run_dir: Path,
    token_specs: dict[str, TokenSpec],
    failed_tokens: set[str],
) -> dict[str, Any]:
    successful_tokens = set(token_specs)
    overlap = successful_tokens & failed_tokens
    if overlap:
        raise CompactionError(
            f"Tokens are both successful and failed: {sorted(overlap)[:10]}"
        )
    run_config = load_json(run_dir / "run_config.json", "run config")
    if not isinstance(run_config, dict) or run_config.get("document_role") != (
        "paiav_edit_run_config"
    ):
        raise CompactionError("Unexpected PAI-AV run-config contract")
    selected = run_config.get("selected_clip_ids")
    if not isinstance(selected, list) or any(
        not isinstance(token, str) for token in selected
    ):
        raise CompactionError("run_config.selected_clip_ids must be a string list")
    if len(selected) != len(set(selected)):
        raise CompactionError("run_config.selected_clip_ids contains duplicates")
    selected_set = set(selected)
    observed = successful_tokens | failed_tokens
    if observed != selected_set:
        raise CompactionError(
            "Selected, successful, and failed token sets do not agree: "
            f"missing={sorted(selected_set - observed)[:10]}, "
            f"unexpected={sorted(observed - selected_set)[:10]}"
        )
    validate_actor_mode_contract(run_dir, run_config, successful_tokens)

    summary = load_json(run_dir / "summary.json", "summary")
    if not isinstance(summary, dict):
        raise CompactionError("summary.json must contain an object")
    expected_summary = {
        "selected_tokens": len(selected_set),
        "successful_tokens": len(successful_tokens),
        "failed_tokens": len(failed_tokens),
        "expected_views_per_token": EXPECTED_VIEW_COUNT,
        "expected_selected_views": len(selected_set) * EXPECTED_VIEW_COUNT,
        "published_successful_views": len(successful_tokens) * EXPECTED_VIEW_COUNT,
        "complete_token_markers": len(successful_tokens),
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise CompactionError(f"Run summary does not match artifacts: {mismatches}")

    completion_dir = run_dir / "completed_tokens"
    if not completion_dir.is_dir():
        raise CompactionError(f"Missing completion directory: {completion_dir}")
    completion_paths = {
        path.stem: path for path in completion_dir.glob("*.json") if path.is_file()
    }
    if set(completion_paths) != successful_tokens:
        raise CompactionError(
            "Completion manifests do not match successful tokens: "
            f"missing={sorted(successful_tokens - set(completion_paths))[:10]}, "
            f"unexpected={sorted(set(completion_paths) - successful_tokens)[:10]}"
        )
    for token, path in completion_paths.items():
        document = load_json(path, "completion manifest")
        if not isinstance(document, dict) or document.get("token") != token:
            raise CompactionError(f"Invalid completion manifest: {path}")
        if document.get("status") != "complete" or document.get(
            "published_view_count"
        ) != EXPECTED_VIEW_COUNT:
            raise CompactionError(f"Incomplete completion manifest: {path}")
    return run_config


def validate_final_token_directories(
    run_dir: Path,
    successful_tokens: set[str],
    failed_tokens: set[str],
) -> set[str]:
    final_root = run_dir / "final_images"
    entries = list(final_root.iterdir())
    invalid_root_entries = [
        path.name for path in entries if not path.is_dir() or path.is_symlink()
    ]
    if invalid_root_entries:
        raise CompactionError(
            "Unexpected non-directory entries in final_images: "
            f"{sorted(invalid_root_entries)[:10]}"
        )
    final_tokens = {path.name for path in entries}
    missing = successful_tokens - final_tokens
    unknown = final_tokens - successful_tokens - failed_tokens
    if missing:
        raise CompactionError(
            f"Successful tokens missing from final_images: {sorted(missing)[:10]}"
        )
    if unknown:
        raise CompactionError(
            f"Unknown token directories in final_images: {sorted(unknown)[:10]}"
        )
    for token in sorted(successful_tokens):
        expected_paths = {
            final_root / token / camera / f"{frame_index}.png"
            for camera, frame_index in EXPECTED_VIEWS
        }
        token_root = final_root / token
        actual_paths = {
            path
            for path in token_root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual_paths != expected_paths:
            raise CompactionError(
                f"Final views differ for {token}: "
                f"missing={sorted(str(path) for path in expected_paths - actual_paths)[:4]}, "
                f"unexpected={sorted(str(path) for path in actual_paths - expected_paths)[:4]}"
            )
    return final_tokens & failed_tokens


def load_rgb(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise CompactionError(f"Could not decode {label} {path}: {exc}") from exc


def load_mask(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise CompactionError(f"Could not decode {label} {path}: {exc}") from exc


def validate_frame(
    run_dir: Path,
    spec: FrameSpec,
    allowed_mask_threshold: int,
    pctnet_mask_threshold: int,
) -> FrameValidation:
    paths = {
        "source": spec.source_path,
        "composite": run_dir / "images" / spec.relative_path,
        "mask": run_dir / "masks" / spec.relative_path,
        "actor mask": run_dir / "actor_masks" / spec.relative_path,
        "final": run_dir / "final_images" / spec.relative_path,
    }
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.is_file()]
    if missing:
        raise CompactionError(
            f"Missing frame artifacts for {spec.relative_path}: {missing}"
        )
    source = load_rgb(paths["source"], "source")
    composite = load_rgb(paths["composite"], "composite")
    final = load_rgb(paths["final"], "PCTNet final")
    mask = load_mask(paths["mask"], "combined mask")
    actor_mask = load_mask(paths["actor mask"], "actor mask")
    if source.shape != composite.shape or composite.shape != final.shape:
        raise CompactionError(
            f"Image shape mismatch for {spec.relative_path}: "
            f"source={source.shape}, composite={composite.shape}, final={final.shape}"
        )
    if mask.shape != source.shape[:2] or actor_mask.shape != source.shape[:2]:
        raise CompactionError(
            f"Mask shape mismatch for {spec.relative_path}: "
            f"combined={mask.shape}, actor={actor_mask.shape}, image={source.shape[:2]}"
        )
    if np.any(actor_mask > mask):
        raise CompactionError(
            f"Actor mask exceeds combined mask for {spec.relative_path}"
        )

    composite_difference = np.any(composite != source, axis=2)
    pctnet_difference = np.any(final != composite, axis=2)
    unexpected_composite = composite_difference & (mask < allowed_mask_threshold)
    unexpected_pctnet = pctnet_difference & (mask < pctnet_mask_threshold)
    if np.any(unexpected_composite):
        raise CompactionError(
            f"Composite changes {int(np.count_nonzero(unexpected_composite))} "
            f"pixel(s) outside its mask for {spec.relative_path}"
        )
    if np.any(unexpected_pctnet):
        raise CompactionError(
            f"PCTNet changes {int(np.count_nonzero(unexpected_pctnet))} "
            f"pixel(s) outside its foreground mask for {spec.relative_path}"
        )

    mask_pixels = int(np.count_nonzero(mask))
    actor_mask_pixels = int(np.count_nonzero(actor_mask))
    composite_changed = int(np.count_nonzero(composite_difference))
    pctnet_changed = int(np.count_nonzero(pctnet_difference))
    expected_counts = (
        ("mask_pixel_count", spec.metadata_mask_pixel_count, mask_pixels),
        (
            "actor_mask_pixel_count",
            spec.metadata_actor_mask_pixel_count,
            actor_mask_pixels,
        ),
        (
            "changed_pixel_count",
            spec.metadata_changed_pixel_count,
            composite_changed,
        ),
    )
    for label, expected, actual in expected_counts:
        if expected is not None and expected != actual:
            raise CompactionError(
                f"{label} mismatch for {spec.relative_path}: "
                f"metadata={expected}, actual={actual}"
            )
    if mask_pixels == 0 and (composite_changed != 0 or pctnet_changed != 0):
        raise CompactionError(
            f"Zero-mask view is not unchanged for {spec.relative_path}"
        )
    return FrameValidation(
        spec=spec,
        width=int(source.shape[1]),
        height=int(source.shape[0]),
        mask_pixel_count=mask_pixels,
        actor_mask_pixel_count=actor_mask_pixels,
        composite_changed_pixel_count=composite_changed,
        pctnet_changed_pixel_count=pctnet_changed,
    )


def validate_token(
    run_dir: Path,
    token_spec: TokenSpec,
    allowed_mask_threshold: int,
    pctnet_mask_threshold: int,
) -> list[FrameValidation]:
    return [
        validate_frame(
            run_dir,
            frame,
            allowed_mask_threshold,
            pctnet_mask_threshold,
        )
        for frame in token_spec.frames
    ]


def validate_all(
    run_dir: Path,
    token_specs: dict[str, TokenSpec],
    workers: int,
    allowed_mask_threshold: int,
    pctnet_mask_threshold: int,
) -> list[FrameValidation]:
    validations: list[FrameValidation] = []
    errors: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(token_specs))) as executor:
        futures = {
            executor.submit(
                validate_token,
                run_dir,
                token_spec,
                allowed_mask_threshold,
                pctnet_mask_threshold,
            ): token
            for token, token_spec in token_specs.items()
        }
        for future in as_completed(futures):
            token = futures[future]
            try:
                validations.extend(future.result())
            except Exception as exc:  # collect all token failures before mutation
                errors.append(f"{token}: {exc}")
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"Validated {completed}/{len(futures)} token(s)", flush=True)
    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:20])
        if len(errors) > 20:
            preview += f"\n  ... and {len(errors) - 20} more token error(s)"
        raise CompactionError(
            f"Validation failed for {len(errors)} token(s); no files were "
            f"deleted:\n{preview}"
        )
    validations.sort(
        key=lambda item: (
            item.spec.token,
            item.spec.camera,
            item.spec.frame_index,
        )
    )
    return validations


def source_reconstruction_document(
    run_dir: Path,
    run_config: dict[str, Any],
    token_specs: dict[str, TokenSpec],
    failed_tokens: set[str],
) -> dict[str, Any]:
    dataset = run_config.get("dataset")
    if not isinstance(dataset, dict):
        raise CompactionError("run_config.dataset must be an object")
    data_root_value = dataset.get("data_root")
    manifest_value = dataset.get("selection_manifest")
    manifest_sha256 = dataset.get("selection_manifest_sha256")
    if not isinstance(data_root_value, str) or not isinstance(manifest_value, str):
        raise CompactionError("Run config lacks dataset source paths")
    data_root = Path(data_root_value).expanduser().resolve()
    manifest_path = Path(manifest_value).expanduser().resolve()
    if not data_root.is_dir() or not manifest_path.is_file():
        raise CompactionError(
            f"Source dataset or selection manifest is unavailable: "
            f"{data_root}, {manifest_path}"
        )
    actual_manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != actual_manifest_sha256:
        raise CompactionError("Selection-manifest SHA-256 no longer matches run config")
    try:
        selection = load_selection_manifest(manifest_path)
    except (OSError, ValueError, PAIAVDataError) as exc:
        raise CompactionError(f"Invalid PAI-AV source manifest: {exc}") from exc
    configured_role = dataset.get("selection_manifest_role")
    if configured_role is not None and configured_role != selection.manifest_role:
        raise CompactionError(
            "Manifest role differs between run config and source manifest"
        )
    configured_catalog_sha256 = dataset.get("catalog_clip_index_sha256")
    if (
        configured_catalog_sha256 is not None
        and configured_catalog_sha256 != selection.catalog_index_sha256
    ):
        raise CompactionError(
            "Catalog clip-index SHA-256 differs between run config and manifest"
        )
    run_tokens = set(token_specs) | failed_tokens
    missing_manifest_tokens = sorted(run_tokens - set(selection.clip_ids))
    if missing_manifest_tokens:
        raise CompactionError(
            "Run tokens are absent from the PAI-AV manifest: "
            f"{missing_manifest_tokens[:10]}"
        )

    token_chunk_ids = {
        token: selection.chunk_id_for(token) for token in sorted(run_tokens)
    }
    selected_chunk_ids = sorted(set(token_chunk_ids.values()))
    configured_mapping = dataset.get("selected_clip_chunk_ids")
    if configured_mapping is not None:
        if not isinstance(configured_mapping, dict):
            raise CompactionError("Run config selected_clip_chunk_ids must be an object")
        mismatched = sorted(
            token
            for token, chunk_id in token_chunk_ids.items()
            if configured_mapping.get(token) != chunk_id
        )
        if mismatched:
            raise CompactionError(
                "Clip/chunk mappings differ between run config and source manifest: "
                f"{mismatched[:10]}"
            )
    configured_chunks = dataset.get("selected_chunk_ids")
    if configured_chunks is not None and configured_chunks != selected_chunk_ids:
        raise CompactionError(
            "Selected chunks differ between run config and source manifest"
        )
    configured_chunk = dataset.get("selected_chunk_id")
    if configured_chunk is not None and (
        len(selected_chunk_ids) != 1 or configured_chunk != selected_chunk_ids[0]
    ):
        raise CompactionError(
            "Selected chunk differs between run config and source manifest"
        )

    try:
        chunk_paths = resolve_chunk_feature_paths(
            data_root,
            selection,
            selected_chunk_ids,
            RECONSTRUCTION_FEATURES,
        )
    except (OSError, ValueError, PAIAVDataError) as exc:
        raise CompactionError(f"Invalid PAI-AV source features: {exc}") from exc
    resolved_chunks: dict[str, dict[str, dict[str, Any]]] = {}
    for chunk_id, feature_paths in sorted(chunk_paths.items()):
        resolved_chunks[str(chunk_id)] = {
            feature: {
                "relative_path": selection.chunks[chunk_id]
                .feature_files[feature]
                .relative_path,
                "resolved_path": str(path),
                "size_bytes": path.stat().st_size,
            }
            for feature, path in sorted(feature_paths.items())
        }

    token_sources: dict[str, Any] = {}
    for token, token_spec in sorted(token_specs.items()):
        source = token_spec.paiav_source
        reconstruction = source.get("source_reconstruction")
        if reconstruction is not None and not isinstance(reconstruction, dict):
            raise CompactionError(
                f"source_reconstruction must be an object for {token}"
            )
        reconstruction = reconstruction or {}
        token_chunk_id = token_chunk_ids[token]
        recorded_chunk_id = reconstruction.get("selected_chunk_id")
        if recorded_chunk_id is not None and recorded_chunk_id != token_chunk_id:
            raise CompactionError(
                f"Source chunk provenance differs for {token}: "
                f"metadata={recorded_chunk_id}, manifest={token_chunk_id}"
            )
        recorded_features = reconstruction.get("feature_files")
        if recorded_features is not None:
            if not isinstance(recorded_features, dict):
                raise CompactionError(
                    f"Invalid feature-file provenance for {token}"
                )
            expected_features = resolved_chunks[str(token_chunk_id)]
            mismatched_features = sorted(
                feature
                for feature, record in expected_features.items()
                if recorded_features.get(feature) != record["resolved_path"]
            )
            if mismatched_features:
                raise CompactionError(
                    f"Source feature provenance differs for {token}: "
                    f"{mismatched_features}"
                )
        camera_members = reconstruction.get("camera_members", {})
        if not isinstance(camera_members, dict):
            raise CompactionError(f"Invalid camera-member provenance for {token}")

        def camera_member(camera: str, key: str, fallback: str) -> str:
            member_record = camera_members.get(camera, {})
            if not isinstance(member_record, dict):
                raise CompactionError(
                    f"Invalid member provenance for {token}/{camera}"
                )
            value = member_record.get(key, fallback)
            if not isinstance(value, str) or not value:
                raise CompactionError(
                    f"Invalid {key} archive member for {token}/{camera}"
                )
            return value

        lidar_member = reconstruction.get(
            "lidar_member", f"{token}.lidar_top_360fov.parquet"
        )
        egomotion_members = reconstruction.get("egomotion_members", {})
        ego_vehicle_geometry = source.get("ego_vehicle_geometry")
        if not isinstance(lidar_member, str) or not lidar_member:
            raise CompactionError(f"Invalid LiDAR member provenance for {token}")
        if not isinstance(egomotion_members, dict):
            raise CompactionError(f"Invalid egomotion provenance for {token}")
        if ego_vehicle_geometry is not None and not isinstance(
            ego_vehicle_geometry, dict
        ):
            raise CompactionError(f"Invalid ego vehicle geometry for {token}")
        token_sources[token] = {
            "source_reconstruction_version": reconstruction.get("version"),
            "selected_chunk_id": token_chunk_id,
            "nominal_history_timestamps_us": source[
                "nominal_history_timestamps_us"
            ],
            "camera_sources": {
                camera: {
                    "archive_feature": camera,
                    "video_member": camera_member(
                        camera, "video", f"{token}.{camera}.mp4"
                    ),
                    "timestamps_member": camera_member(
                        camera,
                        "timestamps",
                        f"{token}.{camera}.timestamps.parquet",
                    ),
                    "frame_indices": source["camera_source_frame_indices"][camera],
                    "frame_timestamps_us": source[
                        "camera_source_timestamps_us"
                    ][camera],
                    "exposure_intervals_us": source.get(
                        "camera_exposure_intervals_us", {}
                    ).get(camera),
                }
                for camera in CAMERAS
            },
            "lidar_source": {
                "archive_feature": "lidar_top_360fov",
                "member": lidar_member,
                "spin_indices": source.get("lidar_spin_indices"),
                "midpoint_timestamps_us": source.get(
                    "lidar_midpoint_timestamps_us"
                ),
                "motion_compensation": source.get("lidar_motion_compensation"),
            },
            "egomotion_members": {
                "egomotion.offline": egomotion_members.get(
                    "egomotion.offline", f"{token}.egomotion.offline.parquet"
                ),
                "egomotion": egomotion_members.get(
                    "egomotion", f"{token}.egomotion.parquet"
                ),
            },
            **(
                {"ego_vehicle_geometry": ego_vehicle_geometry}
                if ego_vehicle_geometry is not None
                else {}
            ),
        }
    source_index = {
        "document_role": "paiav_source_reconstruction_index",
        "version": (
            SOURCE_INDEX_VERSION
            if len(selected_chunk_ids) == 1
            else MULTI_CHUNK_SOURCE_INDEX_VERSION
        ),
        "run_dir": str(run_dir),
        "dataset_repo_id": dataset.get("repo_id"),
        "data_root": str(data_root),
        "selection_manifest": str(manifest_path),
        "selection_manifest_role": selection.manifest_role,
        "selection_manifest_sha256": actual_manifest_sha256,
        "selected_chunk_ids": selected_chunk_ids,
        "chunk_feature_files": resolved_chunks,
        "successful_tokens": token_sources,
        "failed_tokens": sorted(failed_tokens),
    }
    if len(selected_chunk_ids) == 1:
        only_chunk_id = selected_chunk_ids[0]
        source_index["selected_chunk_id"] = only_chunk_id
        source_index["feature_files"] = resolved_chunks[str(only_chunk_id)]
    if selection.catalog_index_path is not None:
        source_index["catalog_clip_index"] = str(selection.catalog_index_path)
        source_index["catalog_clip_index_sha256"] = (
            selection.catalog_index_sha256
        )
    return source_index


def _stale_image_references(value: Any, run_dir: Path) -> list[str]:
    """Find path strings that resolve inside the images tree deleted on compaction."""
    images_root = (run_dir / "images").resolve()
    stale: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            candidate = Path(item).expanduser()
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            try:
                if candidate.resolve().is_relative_to(images_root):
                    stale.append(item)
            except (OSError, RuntimeError):
                pass

    visit(value)
    return stale


def compact_annotations_document(
    run_dir: Path,
    token_specs: dict[str, TokenSpec],
    source_index: dict[str, Any],
    annotations: Any | None = None,
) -> list[dict[str, Any]]:
    """Replace disposable composite paths with retained edits and source archives."""
    if annotations is None:
        annotations = load_json(
            run_dir / ANNOTATIONS_FILENAME,
            "annotations",
        )
    if not isinstance(annotations, list):
        raise CompactionError("annotations.json must contain a list")
    token_sources = source_index.get("successful_tokens")
    feature_files = source_index.get("feature_files")
    chunk_feature_files = source_index.get("chunk_feature_files")
    if not isinstance(token_sources, dict) or (
        not isinstance(feature_files, dict)
        and not isinstance(chunk_feature_files, dict)
    ):
        raise CompactionError("Source reconstruction index is incomplete")

    observed_tokens: set[str] = set()
    compacted: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise CompactionError(f"Annotation {index} must be an object")
        token = annotation.get("token")
        if not isinstance(token, str) or token not in token_specs:
            raise CompactionError(f"Annotation {index} has unknown token {token!r}")
        if token in observed_tokens:
            raise CompactionError(f"Duplicate annotation token: {token}")
        observed_tokens.add(token)
        for camera in CAMERAS:
            paths = annotation.get(camera)
            if (
                not isinstance(paths, list)
                or len(paths) != len(FRAME_INDICES)
                or any(not isinstance(path, str) or not path for path in paths)
            ):
                raise CompactionError(
                    f"Annotation camera field {token}/{camera} must contain "
                    f"{len(FRAME_INDICES)} paths"
                )

        token_source = token_sources.get(token)
        if not isinstance(token_source, dict):
            raise CompactionError(f"Missing reconstruction source for {token}")
        token_chunk_id = token_source.get("selected_chunk_id")
        if isinstance(chunk_feature_files, dict):
            token_feature_files = chunk_feature_files.get(str(token_chunk_id))
        else:
            token_feature_files = feature_files
        if not isinstance(token_feature_files, dict):
            raise CompactionError(
                f"Missing chunk feature files for {token}/{token_chunk_id}"
            )
        nominal_timestamps = required_integer_list(
            token_source,
            "nominal_history_timestamps_us",
            len(FRAME_INDICES),
            f"source reconstruction for {token}",
        )
        camera_sources = token_source.get("camera_sources")
        if not isinstance(camera_sources, dict):
            raise CompactionError(f"Missing camera reconstruction sources for {token}")

        original_cameras: dict[str, Any] = {}
        compacted_annotation = dict(annotation)
        for camera in CAMERAS:
            camera_source = camera_sources.get(camera)
            if not isinstance(camera_source, dict):
                raise CompactionError(
                    f"Missing camera reconstruction source for {token}/{camera}"
                )
            archive_feature = camera_source.get("archive_feature")
            video_member = camera_source.get("video_member")
            timestamps_member = camera_source.get("timestamps_member")
            feature = token_feature_files.get(archive_feature)
            if (
                not isinstance(archive_feature, str)
                or not isinstance(video_member, str)
                or not video_member
                or not isinstance(timestamps_member, str)
                or not timestamps_member
                or not isinstance(feature, dict)
                or not isinstance(feature.get("resolved_path"), str)
            ):
                raise CompactionError(
                    f"Invalid archive reconstruction source for {token}/{camera}"
                )
            frame_indices = required_integer_list(
                camera_source,
                "frame_indices",
                len(FRAME_INDICES),
                f"camera reconstruction for {token}/{camera}",
            )
            frame_timestamps = required_integer_list(
                camera_source,
                "frame_timestamps_us",
                len(FRAME_INDICES),
                f"camera reconstruction for {token}/{camera}",
            )
            compacted_annotation[camera] = [
                str(
                    (
                        run_dir
                        / "final_images"
                        / token
                        / camera
                        / f"{frame_index}.png"
                    ).resolve()
                )
                for frame_index in FRAME_INDICES
            ]
            original_cameras[camera] = {
                "archive_feature": archive_feature,
                "archive_path": feature["resolved_path"],
                "video_member": video_member,
                "timestamps_member": timestamps_member,
                "frame_indices": frame_indices,
                "frame_timestamps_us": frame_timestamps,
                "nominal_timestamps_us": nominal_timestamps,
            }

        compacted_annotation["compact_evaluation_contract"] = {
            "version": COMPACT_ANNOTATION_VERSION,
            "edited_image_stage": "pctnet_final",
            "edited_image_tree": "final_images",
            "camera_fields": list(CAMERAS),
            "path_kind": "absolute",
        }
        compacted_annotation["original_image_sources"] = {
            "version": ORIGINAL_IMAGE_SOURCE_VERSION,
            "source_reconstruction_file": SOURCE_INDEX_FILENAME,
            "source_reconstruction_token": token,
            "selected_chunk_id": token_chunk_id,
            "frame_order": "oldest_to_current",
            "cameras": original_cameras,
        }
        stale = _stale_image_references(compacted_annotation, run_dir)
        if stale:
            raise CompactionError(
                f"Compacted annotation for {token} still references deleted "
                f"images: {stale[:4]}"
            )
        compacted.append(compacted_annotation)

    expected_tokens = set(token_specs)
    if observed_tokens != expected_tokens:
        raise CompactionError(
            "Annotation tokens differ from successful metadata: "
            f"missing={sorted(expected_tokens - observed_tokens)[:10]}, "
            f"unexpected={sorted(observed_tokens - expected_tokens)[:10]}"
        )
    return compacted


def validate_compact_annotations(
    run_dir: Path,
    token_specs: dict[str, TokenSpec],
    source_index: dict[str, Any],
    expected_sha256: str,
) -> None:
    path = run_dir / ANNOTATIONS_FILENAME
    annotations = load_json(path, "compact annotations")
    expected = compact_annotations_document(
        run_dir,
        token_specs,
        source_index,
        annotations,
    )
    if annotations != expected:
        raise CompactionError("Compact annotations do not match their source contract")
    if file_sha256(path) != expected_sha256:
        raise CompactionError("Compact annotations SHA-256 mismatch")


def validate_compact_final_tree(
    run_dir: Path,
    token_specs: dict[str, TokenSpec],
) -> int:
    final_root = run_dir / "final_images"
    expected_tokens = set(token_specs)
    actual_tokens = {path.name for path in final_root.iterdir() if path.is_dir()}
    if actual_tokens != expected_tokens:
        raise CompactionError(
            "Compact final token set differs from metadata: "
            f"missing={sorted(expected_tokens - actual_tokens)[:10]}, "
            f"unexpected={sorted(actual_tokens - expected_tokens)[:10]}"
        )
    count = 0
    for token_spec in token_specs.values():
        expected_paths = {
            run_dir / "final_images" / frame.relative_path
            for frame in token_spec.frames
        }
        actual_paths = {
            path
            for path in (final_root / token_spec.token).glob("*/*")
            if path.is_file()
        }
        if actual_paths != expected_paths:
            raise CompactionError(
                f"Compact final views differ for {token_spec.token}: "
                f"missing={sorted(str(path) for path in expected_paths - actual_paths)[:4]}, "
                f"unexpected={sorted(str(path) for path in actual_paths - expected_paths)[:4]}"
            )
        for path in sorted(expected_paths):
            try:
                with Image.open(path) as image:
                    image.verify()
            except OSError as exc:
                raise CompactionError(f"Unreadable final image {path}: {exc}") from exc
            count += 1
    return count


def remove_path(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


def cleanup_targets_present(run_dir: Path) -> list[str]:
    return [
        name
        for name in (*CLEANUP_DIRECTORIES, *CLEANUP_FILES)
        if (run_dir / name).exists() or (run_dir / name).is_symlink()
    ]


def cleanup_intermediates(run_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in (*CLEANUP_DIRECTORIES, *CLEANUP_FILES):
        target = run_dir / name
        if target.parent != run_dir or target.name != name:
            raise CompactionError(f"Unsafe cleanup target: {target}")
        if remove_path(target):
            removed.append(name)
            print(f"Removed {target}", flush=True)
    return removed


def validate_source_index(
    run_dir: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    path = run_dir / SOURCE_INDEX_FILENAME
    if not path.is_file():
        raise CompactionError(f"Missing source reconstruction index: {path}")
    if report.get("source_reconstruction_sha256") != file_sha256(path):
        raise CompactionError("Source reconstruction index SHA-256 mismatch")
    document = load_json(path, "source reconstruction index")
    if (
        not isinstance(document, dict)
        or document.get("version") not in SUPPORTED_SOURCE_INDEX_VERSIONS
    ):
        raise CompactionError("Unsupported source reconstruction index")
    return document


def resume_or_confirm_compaction(
    run_dir: Path,
    token_specs: dict[str, TokenSpec],
    report: dict[str, Any],
) -> dict[str, Any]:
    if report.get("version") != REPORT_VERSION:
        raise CompactionError(
            f"Unsupported compaction report version: {report.get('version')!r}"
        )
    status = report.get("status")
    if status not in {"cleanup_pending", "complete"}:
        raise CompactionError(f"Invalid compaction status: {status!r}")
    frame_count = validate_compact_final_tree(run_dir, token_specs)
    if frame_count != report.get("total_final_frames"):
        raise CompactionError("Final-image count differs from compaction report")
    source_index = validate_source_index(run_dir, report)
    annotation_report_fields = (
        "annotations_file",
        "annotations_sha256",
        "compact_annotation_version",
    )
    if any(report.get(field) is not None for field in annotation_report_fields):
        if report.get("annotations_file") != ANNOTATIONS_FILENAME:
            raise CompactionError("Unsupported compact annotations file")
        if report.get("compact_annotation_version") != COMPACT_ANNOTATION_VERSION:
            raise CompactionError("Unsupported compact annotation version")
        annotations_sha256 = report.get("annotations_sha256")
        if not isinstance(annotations_sha256, str) or not annotations_sha256:
            raise CompactionError("Compaction report lacks annotations SHA-256")
        validate_compact_annotations(
            run_dir,
            token_specs,
            source_index,
            annotations_sha256,
        )
    remaining = cleanup_targets_present(run_dir)
    if status == "complete":
        if remaining:
            raise CompactionError(
                "A completed compact run contains new intermediates; refusing "
                f"unvalidated deletion: {remaining}"
            )
        print(f"Run is already compact and valid: {run_dir}")
        return report

    removed = cleanup_intermediates(run_dir)
    complete = {
        **report,
        "status": "complete",
        "completed_at": utc_now(),
        "removed_artifacts": sorted(
            set(report.get("cleanup_candidates", []))
            | set(report.get("removed_artifacts", []))
            | set(removed)
        ),
    }
    write_json_atomic(run_dir / REPORT_FILENAME, complete)
    print(f"Resumed and completed cleanup: {run_dir}")
    return complete


def compact_run(
    run_dir_value: Path,
    *,
    workers: int = 1,
    allowed_mask_threshold: int = 1,
    pctnet_mask_threshold: int = 250,
    validate_only: bool = False,
) -> dict[str, Any]:
    run_dir = resolve_run_dir(run_dir_value)
    token_specs = metadata_specs(run_dir)
    failed_tokens = failure_tokens(run_dir)
    report_path = run_dir / REPORT_FILENAME
    if report_path.is_file():
        if validate_only:
            raise CompactionError("--validate-only is not valid after compaction starts")
        report = load_json(report_path, "compaction report")
        if not isinstance(report, dict):
            raise CompactionError("Compaction report must be an object")
        return resume_or_confirm_compaction(run_dir, token_specs, report)

    run_config = validate_run_contract(run_dir, token_specs, failed_tokens)
    partial_failed_tokens = validate_final_token_directories(
        run_dir, set(token_specs), failed_tokens
    )
    print(
        f"Validating {len(token_specs)} successful token(s), "
        f"{len(failed_tokens)} failed token(s), workers={workers}",
        flush=True,
    )
    validations = validate_all(
        run_dir,
        token_specs,
        workers,
        allowed_mask_threshold,
        pctnet_mask_threshold,
    )
    source_index = source_reconstruction_document(
        run_dir, run_config, token_specs, failed_tokens
    )
    compact_annotations = compact_annotations_document(
        run_dir,
        token_specs,
        source_index,
    )
    base = {
        "version": REPORT_VERSION,
        "run_dir": str(run_dir),
        "successful_tokens": len(token_specs),
        "failed_tokens": len(failed_tokens),
        "total_final_frames": len(validations),
        "zero_mask_frames": sum(
            item.mask_pixel_count == 0 for item in validations
        ),
        "compositor_changed_frames": sum(
            item.composite_changed_pixel_count > 0 for item in validations
        ),
        "pctnet_changed_frames": sum(
            item.pctnet_changed_pixel_count > 0 for item in validations
        ),
        "allowed_mask_threshold": allowed_mask_threshold,
        "pctnet_mask_threshold": pctnet_mask_threshold,
        "partial_failed_token_directories": len(partial_failed_tokens),
    }
    if validate_only:
        result = {**base, "status": "validated_only", "validated_at": utc_now()}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    source_index_path = run_dir / SOURCE_INDEX_FILENAME
    write_json_atomic(source_index_path, source_index)
    annotations_path = run_dir / ANNOTATIONS_FILENAME
    write_json_atomic(annotations_path, compact_annotations)
    annotations_sha256 = file_sha256(annotations_path)
    validate_compact_annotations(
        run_dir,
        token_specs,
        source_index,
        annotations_sha256,
    )
    for token in sorted(partial_failed_tokens):
        remove_path(run_dir / "final_images" / token)
    published_count = validate_compact_final_tree(run_dir, token_specs)
    if published_count != len(validations):
        raise CompactionError("Final-image count changed after validation")

    pending = {
        **base,
        "status": "cleanup_pending",
        "validated_at": utc_now(),
        "source_reconstruction_file": SOURCE_INDEX_FILENAME,
        "source_reconstruction_sha256": file_sha256(source_index_path),
        "annotations_file": ANNOTATIONS_FILENAME,
        "annotations_sha256": annotations_sha256,
        "compact_annotation_version": COMPACT_ANNOTATION_VERSION,
        "cleanup_candidates": cleanup_targets_present(run_dir),
        "removed_artifacts": [],
    }
    write_json_atomic(report_path, pending)
    removed = cleanup_intermediates(run_dir)
    complete = {
        **pending,
        "status": "complete",
        "completed_at": utc_now(),
        "removed_artifacts": sorted(set(pending["cleanup_candidates"]) | set(removed)),
    }
    write_json_atomic(report_path, complete)
    print(
        f"Compaction complete: final_frames={len(validations)}, "
        f"removed={len(removed)} artifact trees/files",
        flush=True,
    )
    return complete


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        compact_run(
            args.run_dir,
            workers=args.workers,
            allowed_mask_threshold=args.allowed_mask_threshold,
            pctnet_mask_threshold=args.pctnet_mask_threshold,
            validate_only=args.validate_only,
        )
        return 0
    except (CompactionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
