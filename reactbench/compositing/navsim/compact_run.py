#!/usr/bin/env python3
"""Validate and compact a completed NAVSIM compositor run.

The compositor and PCTNet intentionally publish a fully materialized working
set so both stages remain resumable.  This script converts that working set to
the smaller evaluation contract only after validating every successful frame:

* the compositor may change pixels only where the combined mask is nonzero;
* PCTNet may change pixels only where the combined mask reaches its threshold;
* completely unchanged frames must decode identically to their source JPEG;
* every successful token must contain all twelve camera/history frames.

After all validation succeeds, unchanged final PNGs are replaced by symlinks
to their original JPEGs and render-only intermediates are removed.  A pending
report is written before deletion so an interrupted cleanup can be resumed
without requiring already-deleted masks or composites.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image


CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0")
FRAME_INDICES = tuple(range(4))
EXPECTED_VIEWS = frozenset(
    (camera, frame_index)
    for camera in CAMERAS
    for frame_index in FRAME_INDICES
)
SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SOURCE_JPEG_SUFFIXES = (".jpg", ".jpeg")
REPORT_FILENAME = "compaction_report.json"
REPORT_VERSION = "evaluation_only_v1"

# These are all reproducible run-local products.  Keep the list explicit so a
# future, unrelated directory beginning with ``raw_`` is never deleted merely
# because of its name.
CLEANUP_DIRECTORIES = (
    "prepared",
    "images",
    "masks",
    "actor_masks",
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
    "source_relative_depth",
)
CLEANUP_FILES = (
    "annotations.json",
    "depth_inference.log",
)


class CompactionError(RuntimeError):
    """A validation or publication invariant was not satisfied."""


@dataclass(frozen=True)
class FrameSpec:
    token: str
    camera: str
    frame_index: int
    source_path: Path
    metadata_mask_pixel_count: int | None
    metadata_changed_pixel_count: int | None

    @property
    def relative_stem(self) -> Path:
        return Path(self.token) / self.camera / str(self.frame_index)


@dataclass(frozen=True)
class FramePlan:
    spec: FrameSpec
    final_path: Path
    unchanged: bool
    mask_pixel_count: int
    changed_pixel_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed NAVSIM compositor/PCTNet run, link fully "
            "unchanged frames to their source JPEGs, and delete render-only "
            "intermediates."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, os.cpu_count() or 1)),
        help="Parallel token validators (default: min(12, allocated CPUs)).",
    )
    parser.add_argument(
        "--allowed-mask-threshold",
        type=int,
        default=1,
        help="Compositor pixels may differ from source at or above this value.",
    )
    parser.add_argument(
        "--pctnet-mask-threshold",
        type=int,
        default=250,
        help="PCTNet pixels may differ from the composite at or above this value.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every pre-cleanup check without linking or deleting anything.",
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
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


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
    if not (run_dir / "run_config.json").is_file():
        raise CompactionError(f"Run is missing run_config.json: {run_dir}")
    return run_dir


def optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CompactionError(f"{label} must be a non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise CompactionError(f"{label} must be non-negative, got {result}")
    return result


def load_frame_specs(metadata_path: Path) -> tuple[str, list[FrameSpec]]:
    metadata = load_json(metadata_path, "metadata")
    if not isinstance(metadata, dict):
        raise CompactionError(f"Metadata must be a JSON object: {metadata_path}")
    token = metadata.get("token")
    if token != metadata_path.stem or not isinstance(token, str) or not token:
        raise CompactionError(
            f"Metadata token does not match filename {metadata_path.name}: {token!r}"
        )
    if metadata.get("status") != "success":
        raise CompactionError(
            f"Metadata is not a successful finalized token: {metadata_path}"
        )
    records = metadata.get("records")
    if not isinstance(records, list):
        raise CompactionError(f"Metadata records must be a list: {metadata_path}")

    specs: list[FrameSpec] = []
    observed: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CompactionError(
                f"Metadata record {index} is not an object: {metadata_path}"
            )
        camera = record.get("camera")
        frame_index = record.get("frame_index")
        if camera not in CAMERAS or not isinstance(frame_index, int):
            raise CompactionError(
                f"Invalid camera/frame in {metadata_path}: {camera!r}/{frame_index!r}"
            )
        view = (camera, frame_index)
        if view in observed:
            raise CompactionError(f"Duplicate view {view} in {metadata_path}")
        observed.add(view)
        source_value = record.get("source_path")
        if not isinstance(source_value, str) or not source_value:
            raise CompactionError(
                f"Missing source_path for {token}/{camera}/{frame_index}"
            )
        specs.append(
            FrameSpec(
                token=token,
                camera=camera,
                frame_index=frame_index,
                source_path=Path(source_value).expanduser(),
                metadata_mask_pixel_count=optional_nonnegative_int(
                    record.get("mask_pixel_count"),
                    f"mask_pixel_count for {token}/{camera}/{frame_index}",
                ),
                metadata_changed_pixel_count=optional_nonnegative_int(
                    record.get("changed_pixel_count"),
                    f"changed_pixel_count for {token}/{camera}/{frame_index}",
                ),
            )
        )
    if observed != EXPECTED_VIEWS:
        missing = sorted(EXPECTED_VIEWS - observed)
        unexpected = sorted(observed - EXPECTED_VIEWS)
        raise CompactionError(
            f"Token {token} does not contain the complete 12-view set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return token, sorted(specs, key=lambda item: (item.camera, item.frame_index))


def failure_tokens(run_dir: Path) -> set[str]:
    directory = run_dir / "failures"
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.json") if path.is_file()}


def validate_summary(
    run_dir: Path,
    successful_tokens: set[str],
    failed_tokens: set[str],
) -> None:
    overlap = successful_tokens & failed_tokens
    if overlap:
        raise CompactionError(
            f"Tokens are recorded as both successful and failed: {sorted(overlap)[:10]}"
        )
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise CompactionError(f"Run is missing summary.json: {summary_path}")
    summary = load_json(summary_path, "summary")
    if not isinstance(summary, dict):
        raise CompactionError(f"Summary must be a JSON object: {summary_path}")
    expected = {
        "successful_tokens": len(successful_tokens),
        "failed_tokens": len(failed_tokens),
        "selected_tokens": len(successful_tokens) + len(failed_tokens),
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise CompactionError(f"Run summary counts do not match artifacts: {mismatches}")


def image_candidates(final_root: Path, stem: Path) -> list[Path]:
    return [
        (final_root / stem).with_suffix(suffix)
        for suffix in SUPPORTED_IMAGE_SUFFIXES
        if (final_root / stem).with_suffix(suffix).is_file()
    ]


def symlink_points_to(path: Path, target: Path) -> bool:
    if not path.is_symlink():
        return False
    try:
        return path.resolve(strict=True) == target.resolve(strict=True)
    except OSError:
        return False


def select_final_for_validation(
    final_root: Path,
    spec: FrameSpec,
) -> Path:
    candidates = image_candidates(final_root, spec.relative_stem)
    if not candidates:
        raise CompactionError(f"Missing final image: {spec.relative_stem}")
    if len(candidates) == 1:
        return candidates[0]

    # An interrupted or repeated compaction can briefly leave the materialized
    # PNG beside an already-published source-JPEG link.  Prefer the PNG for the
    # full validation pass; publication will retain only the correct format.
    pngs = [path for path in candidates if path.suffix.lower() == ".png"]
    source_links = [
        path for path in candidates if symlink_points_to(path, spec.source_path)
    ]
    if len(pngs) == 1 and len(source_links) == len(candidates) - 1:
        return pngs[0]
    raise CompactionError(
        f"Multiple final images for {spec.relative_stem}: {candidates}"
    )


def load_rgb(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise CompactionError(f"Could not decode {label} {path}: {exc}") from exc


def load_mask(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise CompactionError(f"Could not decode mask {path}: {exc}") from exc


def validate_frame(
    run_dir: Path,
    spec: FrameSpec,
    allowed_mask_threshold: int,
    pctnet_mask_threshold: int,
) -> FramePlan:
    source_path = spec.source_path
    if not source_path.is_file():
        raise CompactionError(
            f"Source image is missing for {spec.relative_stem}: {source_path}"
        )
    if source_path.suffix.lower() not in SOURCE_JPEG_SUFFIXES:
        raise CompactionError(
            f"Unchanged-frame linking requires a JPEG source for "
            f"{spec.relative_stem}, got {source_path}"
        )

    composite_path = (run_dir / "images" / spec.relative_stem).with_suffix(".png")
    mask_path = (run_dir / "masks" / spec.relative_stem).with_suffix(".png")
    if not composite_path.is_file():
        raise CompactionError(f"Missing pre-PCTNet composite: {composite_path}")
    if not mask_path.is_file():
        raise CompactionError(f"Missing combined mask: {mask_path}")
    final_path = select_final_for_validation(run_dir / "final_images", spec)

    source = load_rgb(source_path, "source image")
    composite = load_rgb(composite_path, "composite image")
    final = load_rgb(final_path, "final image")
    mask = load_mask(mask_path)
    if source.shape != composite.shape or composite.shape != final.shape:
        raise CompactionError(
            f"Image shape mismatch for {spec.relative_stem}: "
            f"source={source.shape}, composite={composite.shape}, final={final.shape}"
        )
    if mask.shape != source.shape[:2]:
        raise CompactionError(
            f"Mask shape mismatch for {spec.relative_stem}: "
            f"mask={mask.shape}, image={source.shape[:2]}"
        )

    composite_difference = np.any(composite != source, axis=2)
    final_difference = np.any(final != composite, axis=2)
    unexpected_composite = composite_difference & (mask < allowed_mask_threshold)
    unexpected_pctnet = final_difference & (mask < pctnet_mask_threshold)
    if np.any(unexpected_composite):
        raise CompactionError(
            f"Composite changes pixels outside its mask for {spec.relative_stem}: "
            f"{int(np.count_nonzero(unexpected_composite))} pixel(s)"
        )
    if np.any(unexpected_pctnet):
        raise CompactionError(
            f"PCTNet changes pixels outside its foreground mask for "
            f"{spec.relative_stem}: "
            f"{int(np.count_nonzero(unexpected_pctnet))} pixel(s)"
        )

    mask_pixel_count = int(np.count_nonzero(mask))
    changed_pixel_count = int(np.count_nonzero(composite_difference))
    if (
        spec.metadata_mask_pixel_count is not None
        and spec.metadata_mask_pixel_count != mask_pixel_count
    ):
        raise CompactionError(
            f"mask_pixel_count mismatch for {spec.relative_stem}: "
            f"metadata={spec.metadata_mask_pixel_count}, actual={mask_pixel_count}"
        )
    if (
        spec.metadata_changed_pixel_count is not None
        and spec.metadata_changed_pixel_count != changed_pixel_count
    ):
        raise CompactionError(
            f"changed_pixel_count mismatch for {spec.relative_stem}: "
            f"metadata={spec.metadata_changed_pixel_count}, "
            f"actual={changed_pixel_count}"
        )

    unchanged = mask_pixel_count == 0
    if unchanged and (changed_pixel_count != 0 or np.any(final_difference)):
        raise CompactionError(
            f"Zero-mask frame does not decode identically to its source for "
            f"{spec.relative_stem}"
        )
    if not unchanged and final_path.suffix.lower() != ".png":
        raise CompactionError(
            f"Edited frame must remain lossless PNG: {spec.relative_stem} "
            f"resolved to {final_path}"
        )
    return FramePlan(
        spec=spec,
        final_path=final_path,
        unchanged=unchanged,
        mask_pixel_count=mask_pixel_count,
        changed_pixel_count=changed_pixel_count,
    )


def validate_token(
    run_dir: Path,
    specs: Sequence[FrameSpec],
    allowed_mask_threshold: int,
    pctnet_mask_threshold: int,
) -> list[FramePlan]:
    return [
        validate_frame(
            run_dir,
            spec,
            allowed_mask_threshold,
            pctnet_mask_threshold,
        )
        for spec in specs
    ]


def metadata_specs(run_dir: Path) -> dict[str, list[FrameSpec]]:
    paths = sorted((run_dir / "metadata").glob("*.json"))
    if not paths:
        raise CompactionError(f"No finalized metadata under {run_dir / 'metadata'}")
    result: dict[str, list[FrameSpec]] = {}
    for path in paths:
        token, specs = load_frame_specs(path)
        if token in result:
            raise CompactionError(f"Duplicate metadata token: {token}")
        result[token] = specs
    return result


def validate_token_directories(
    run_dir: Path,
    successful_tokens: set[str],
    failed_tokens: set[str],
) -> set[str]:
    final_root = run_dir / "final_images"
    final_tokens = {
        path.name for path in final_root.iterdir() if path.is_dir()
    }
    missing = successful_tokens - final_tokens
    unknown = final_tokens - successful_tokens - failed_tokens
    if missing:
        raise CompactionError(
            f"Successful tokens missing from final_images: {sorted(missing)[:10]}"
        )
    if unknown:
        raise CompactionError(
            "final_images contains token directories with neither successful "
            f"metadata nor failure records: {sorted(unknown)[:10]}"
        )
    return final_tokens & failed_tokens


def validate_all(
    run_dir: Path,
    specs_by_token: dict[str, list[FrameSpec]],
    workers: int,
    allowed_mask_threshold: int,
    pctnet_mask_threshold: int,
) -> list[FramePlan]:
    plans: list[FramePlan] = []
    errors: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(specs_by_token))) as executor:
        futures = {
            executor.submit(
                validate_token,
                run_dir,
                specs,
                allowed_mask_threshold,
                pctnet_mask_threshold,
            ): token
            for token, specs in specs_by_token.items()
        }
        for future in as_completed(futures):
            token = futures[future]
            try:
                plans.extend(future.result())
            except Exception as exc:  # collect all token failures before mutation
                errors.append(f"{token}: {exc}")
            completed += 1
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"Validated {completed}/{len(futures)} token(s)",
                    flush=True,
                )
    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:20])
        remainder = len(errors) - 20
        if remainder:
            preview += f"\n  ... and {remainder} more token error(s)"
        raise CompactionError(
            f"Validation failed for {len(errors)} token(s); no files were "
            f"deleted:\n{preview}"
        )
    return sorted(
        plans,
        key=lambda plan: (
            plan.spec.token,
            plan.spec.camera,
            plan.spec.frame_index,
        ),
    )


def original_link_path(final_root: Path, spec: FrameSpec) -> Path:
    suffix = spec.source_path.suffix.lower()
    if suffix not in SOURCE_JPEG_SUFFIXES:
        raise CompactionError(f"Source is not JPEG: {spec.source_path}")
    return (final_root / spec.relative_stem).with_suffix(suffix)


def publish_original_link(final_root: Path, plan: FramePlan) -> Path:
    destination = original_link_path(final_root, plan.spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-link.{os.getpid()}"
    )
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(plan.spec.source_path.resolve())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()

    stem = final_root / plan.spec.relative_stem
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = stem.with_suffix(suffix)
        if candidate != destination and (candidate.exists() or candidate.is_symlink()):
            candidate.unlink()
    return destination


def retain_edited_png(final_root: Path, plan: FramePlan) -> Path:
    destination = (final_root / plan.spec.relative_stem).with_suffix(".png")
    if not destination.is_file():
        raise CompactionError(f"Edited PNG disappeared before publication: {destination}")
    stem = final_root / plan.spec.relative_stem
    for suffix in SOURCE_JPEG_SUFFIXES:
        candidate = stem.with_suffix(suffix)
        if candidate.exists() or candidate.is_symlink():
            if not symlink_points_to(candidate, plan.spec.source_path):
                raise CompactionError(
                    f"Refusing to remove unexpected alternate final image: {candidate}"
                )
            candidate.unlink()
    return destination


def publish_compact_final_images(
    run_dir: Path,
    plans: Sequence[FramePlan],
) -> tuple[int, int]:
    final_root = run_dir / "final_images"
    linked = 0
    retained = 0
    for plan in plans:
        if plan.unchanged:
            publish_original_link(final_root, plan)
            linked += 1
        else:
            retain_edited_png(final_root, plan)
            retained += 1
    return linked, retained


def validate_published_tree(
    run_dir: Path,
    specs_by_token: dict[str, list[FrameSpec]],
) -> tuple[int, int]:
    final_root = run_dir / "final_images"
    linked = 0
    retained = 0
    for token, specs in specs_by_token.items():
        for spec in specs:
            candidates = image_candidates(final_root, spec.relative_stem)
            if len(candidates) != 1:
                raise CompactionError(
                    f"Published frame must resolve to exactly one image for "
                    f"{spec.relative_stem}, got {candidates}"
                )
            path = candidates[0]
            if path.suffix.lower() in SOURCE_JPEG_SUFFIXES:
                if not symlink_points_to(path, spec.source_path):
                    raise CompactionError(
                        f"Published JPEG is not a link to its recorded source: {path}"
                    )
                linked += 1
            elif path.suffix.lower() == ".png":
                try:
                    with Image.open(path) as image:
                        image.verify()
                except OSError as exc:
                    raise CompactionError(
                        f"Published edited PNG is unreadable: {path}: {exc}"
                    ) from exc
                retained += 1
            else:  # guarded by image_candidates, retained for clarity
                raise CompactionError(f"Unsupported published image: {path}")
    final_tokens = {
        path.name for path in final_root.iterdir() if path.is_dir()
    }
    unexpected = final_tokens - set(specs_by_token)
    if unexpected:
        raise CompactionError(
            f"Published tree contains non-successful token directories: "
            f"{sorted(unexpected)[:10]}"
        )
    return linked, retained


def remove_path(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


def cleanup_intermediates(run_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in (*CLEANUP_DIRECTORIES, *CLEANUP_FILES):
        target = run_dir / name
        # Do not weaken this guard: cleanup is intentionally limited to exact
        # direct children of the already-resolved run directory.
        if target.parent != run_dir or target.name != name:
            raise CompactionError(f"Unsafe cleanup target: {target}")
        if remove_path(target):
            removed.append(name)
            print(f"Removed {target}", flush=True)
    return removed


def cleanup_targets_present(run_dir: Path) -> list[str]:
    return [
        name
        for name in (*CLEANUP_DIRECTORIES, *CLEANUP_FILES)
        if (run_dir / name).exists() or (run_dir / name).is_symlink()
    ]


def report_base(
    run_dir: Path,
    successful_tokens: int,
    failed_tokens_count: int,
    linked_frames: int,
    retained_frames: int,
    allowed_mask_threshold: int,
    pctnet_mask_threshold: int,
) -> dict[str, Any]:
    return {
        "version": REPORT_VERSION,
        "run_dir": str(run_dir),
        "successful_tokens": successful_tokens,
        "failed_tokens": failed_tokens_count,
        "total_final_frames": linked_frames + retained_frames,
        "linked_original_jpeg_frames": linked_frames,
        "retained_edited_png_frames": retained_frames,
        "allowed_mask_threshold": allowed_mask_threshold,
        "pctnet_mask_threshold": pctnet_mask_threshold,
    }


def resume_or_confirm_compaction(
    run_dir: Path,
    specs_by_token: dict[str, list[FrameSpec]],
    report: dict[str, Any],
) -> bool:
    if report.get("version") != REPORT_VERSION:
        raise CompactionError(
            f"Unsupported compaction report version: {report.get('version')!r}"
        )
    status = report.get("status")
    if status not in {"cleanup_pending", "complete"}:
        raise CompactionError(f"Invalid compaction report status: {status!r}")
    linked, retained = validate_published_tree(run_dir, specs_by_token)
    if linked != report.get("linked_original_jpeg_frames") or retained != report.get(
        "retained_edited_png_frames"
    ):
        raise CompactionError("Published final-image counts differ from report")
    remaining = cleanup_targets_present(run_dir)
    if status == "complete":
        if remaining:
            raise CompactionError(
                "A completed compact run contains new intermediate artifacts; "
                f"refusing to delete them without revalidation: {remaining}"
            )
        print(f"Run is already compact and valid: {run_dir}")
        return True

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
    return True


def compact_run(
    run_dir_value: Path,
    *,
    workers: int = 1,
    allowed_mask_threshold: int = 1,
    pctnet_mask_threshold: int = 250,
    validate_only: bool = False,
) -> dict[str, Any]:
    run_dir = resolve_run_dir(run_dir_value)
    specs_by_token = metadata_specs(run_dir)
    successful = set(specs_by_token)
    failed = failure_tokens(run_dir)
    validate_summary(run_dir, successful, failed)

    report_path = run_dir / REPORT_FILENAME
    publication_report: dict[str, Any] | None = None
    if report_path.is_file():
        report = load_json(report_path, "compaction report")
        if not isinstance(report, dict):
            raise CompactionError(f"Compaction report must be an object: {report_path}")
        if report.get("status") == "publication_pending":
            if report.get("version") != REPORT_VERSION:
                raise CompactionError(
                    "Unsupported compaction report version: "
                    f"{report.get('version')!r}"
                )
            publication_report = report
            print("Resuming interrupted compact-image publication", flush=True)
        elif resume_or_confirm_compaction(run_dir, specs_by_token, report):
            return load_json(report_path, "compaction report")

    partial_failure_tokens = validate_token_directories(
        run_dir,
        successful,
        failed,
    )
    print(
        f"Validating {len(successful)} successful token(s), "
        f"{len(failed)} failed token(s), workers={workers}",
        flush=True,
    )
    plans = validate_all(
        run_dir,
        specs_by_token,
        workers,
        allowed_mask_threshold,
        pctnet_mask_threshold,
    )
    linked = sum(plan.unchanged for plan in plans)
    retained = len(plans) - linked
    base = report_base(
        run_dir,
        len(successful),
        len(failed),
        linked,
        retained,
        allowed_mask_threshold,
        pctnet_mask_threshold,
    )
    if validate_only:
        result = {
            **base,
            "status": "validated_only",
            "validated_at": utc_now(),
            "partial_failed_token_directories": len(partial_failure_tokens),
        }
        print(json.dumps(result, indent=2))
        return result

    publication_pending = {
        **base,
        "status": "publication_pending",
        "validated_at": (
            publication_report.get("validated_at")
            if publication_report is not None
            else utc_now()
        ),
        "removed_artifacts": [],
    }
    write_json_atomic(report_path, publication_pending)

    actual_linked, actual_retained = publish_compact_final_images(run_dir, plans)
    if (actual_linked, actual_retained) != (linked, retained):
        raise CompactionError(
            "Published final-image counts differ from validation plan: "
            f"planned={(linked, retained)}, "
            f"published={(actual_linked, actual_retained)}"
        )

    # Failed tokens can contain partial frames from a finalization failure.
    # They are excluded only after every successful frame has validated.
    for token in sorted(partial_failure_tokens):
        remove_path(run_dir / "final_images" / token)

    published_linked, published_retained = validate_published_tree(
        run_dir,
        specs_by_token,
    )
    pending = {
        **base,
        "status": "cleanup_pending",
        "validated_at": publication_pending["validated_at"],
        "published_at": utc_now(),
        "partial_failed_token_directories_removed": len(partial_failure_tokens),
        "cleanup_candidates": cleanup_targets_present(run_dir),
        "removed_artifacts": [],
    }
    if (published_linked, published_retained) != (linked, retained):
        raise CompactionError("Published tree validation returned inconsistent counts")
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
        f"Compaction complete: linked_original_jpegs={linked}, "
        f"retained_edited_pngs={retained}, removed={len(removed)} artifact trees/files",
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
