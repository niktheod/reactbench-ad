#!/usr/bin/env python3
"""Run guarded libcom PCTNet harmonization over one PAI-AV image folder.

The sibling mask tree is inferred by replacing the ``images`` directory in the
input path with ``masks``. Only near-white mask pixels are used as foreground,
which excludes the lower-alpha shadow stored in the combined masks.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_UPSTREAM = SCRIPT_DIR.parent / "harmonization"
if LOCAL_UPSTREAM.is_dir():
    # The release's libcom copy (PCTNet), ahead of any installed libcom.
    sys.path.insert(0, str(LOCAL_UPSTREAM))
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "PCTNet_images"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
FICLONE = 0x40049409


@dataclass(frozen=True)
class WorkItem:
    image_path: Path
    mask_path: Path
    output_path: Path
    display_path: Path


@dataclass(frozen=True)
class MaskStats:
    height: int
    width: int
    nonzero_pixels: int
    foreground_pixels: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonize every image below a NAVSIM token folder with libcom "
            "PCTNet and its corresponding near-white pedestrian mask."
        )
    )
    parser.add_argument(
        "image_folder",
        type=Path,
        help=(
            "Token image folder, for example .../iteration_stage0_baseline/"
            "images/b3899f957ebe5684."
        ),
    )
    parser.add_argument(
        "--mask-folder",
        type=Path,
        help=(
            "Corresponding token mask folder. By default it is inferred from "
            "the sibling masks/ tree."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output tree root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=250,
        metavar="0-255",
        help=(
            "Treat mask values at or above this value as pedestrian pixels "
            "(default: 250)."
        ),
    )
    parser.add_argument(
        "--device",
        default="0",
        help="libcom device: a CUDA index such as 0, or a value such as cpu/cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs that already exist instead of skipping them.",
    )
    parser.add_argument(
        "--completion-dir",
        type=Path,
        help=(
            "Optional directory of per-token completion JSON files. When set, "
            "only explicitly complete tokens are processed."
        ),
    )
    parser.add_argument("--roi-padding", type=float, default=0.0)
    parser.add_argument("--min-foreground-pixels", type=int, default=0)
    parser.add_argument("--min-bbox-width", type=int, default=0)
    parser.add_argument("--min-bbox-height", type=int, default=0)
    parser.add_argument("--blend-strength", type=float, default=1.0)
    parser.add_argument("--max-mean-color-shift", type=float, default=255.0)
    parser.add_argument("--min-contrast-ratio", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate path mapping and masks without loading PCTNet or writing outputs.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.mask_threshold <= 255:
        parser.error("--mask-threshold must be in [0, 255]")
    if args.roi_padding < 0:
        parser.error("--roi-padding must be non-negative")
    for name in ("min_foreground_pixels", "min_bbox_width", "min_bbox_height"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 <= args.blend_strength <= 1:
        parser.error("--blend-strength must be in [0, 1]")
    if args.max_mean_color_shift < 0:
        parser.error("--max-mean-color-shift must be non-negative")
    if not 0 <= args.min_contrast_ratio <= 1:
        parser.error("--min-contrast-ratio must be in [0, 1]")
    return args


def load_completed_tokens(completion_dir: Path) -> set[str]:
    completion_dir = completion_dir.expanduser().resolve()
    if not completion_dir.is_dir():
        raise ValueError(
            f"Completion directory does not exist: {completion_dir}"
        )
    tokens = set()
    for path in sorted(completion_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unreadable completion manifest {path}: {exc}") from exc
        token = str(document.get("token", ""))
        expected = int(document.get("expected_view_count", -1))
        published = int(document.get("published_view_count", -2))
        if (
            document.get("status") != "complete"
            or not token
            or token != path.stem
            or expected < 1
            or published != expected
        ):
            raise ValueError(f"Invalid completion manifest: {path}")
        tokens.add(token)
    if not tokens:
        raise ValueError(f"No completed tokens found in: {completion_dir}")
    return tokens


def find_named_ancestor(path: Path, directory_name: str) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == directory_name:
            return candidate
    return None


def resolve_mask_path(candidate: Path) -> Path:
    if candidate.is_file():
        return candidate
    png_candidate = candidate.with_suffix(".png")
    if png_candidate.is_file():
        return png_candidate
    return candidate


def build_work_items(
    image_folder: Path,
    mask_folder: Path | None,
    output_root: Path,
    completed_tokens: set[str] | None = None,
) -> list[WorkItem]:
    image_folder = image_folder.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not image_folder.is_dir():
        raise ValueError(f"Image folder does not exist or is not a directory: {image_folder}")

    images_root = find_named_ancestor(image_folder, "images")
    if images_root is not None:
        output_relative_root = images_root
    else:
        output_relative_root = image_folder.parent

    if mask_folder is None:
        if images_root is None:
            raise ValueError(
                "Could not infer the mask tree because the input path has no "
                "'images' ancestor; pass --mask-folder explicitly."
            )
        masks_root = images_root.parent / "masks"
        if not masks_root.is_dir():
            raise ValueError(f"Inferred masks directory does not exist: {masks_root}")

        def corresponding_mask(image_path: Path) -> Path:
            return masks_root / image_path.relative_to(images_root)

    else:
        explicit_mask_folder = mask_folder.expanduser().resolve()
        if not explicit_mask_folder.is_dir():
            raise ValueError(
                f"Mask folder does not exist or is not a directory: {explicit_mask_folder}"
            )

        def corresponding_mask(image_path: Path) -> Path:
            return explicit_mask_folder / image_path.relative_to(image_folder)

    image_paths = sorted(
        path
        for path in image_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found below: {image_folder}")

    work_items: list[WorkItem] = []
    missing_masks: list[Path] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(output_relative_root)
        if completed_tokens is not None:
            if not relative_path.parts or relative_path.parts[0] not in completed_tokens:
                continue
        mask_path = resolve_mask_path(corresponding_mask(image_path))
        if not mask_path.is_file():
            missing_masks.append(mask_path)
        work_items.append(
            WorkItem(
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_root / relative_path,
                display_path=relative_path,
            )
        )

    if missing_masks:
        preview = "\n".join(f"  - {path}" for path in missing_masks[:10])
        remainder = len(missing_masks) - 10
        if remainder > 0:
            preview += f"\n  ... and {remainder} more"
        raise ValueError(f"Missing masks for {len(missing_masks)} image(s):\n{preview}")
    if not work_items:
        raise ValueError(
            "No input images matched the requested completed-token filter"
        )
    return work_items


def load_image_and_binary_mask(
    item: WorkItem,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray, MaskStats]:
    image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {item.image_path}")
    mask = cv2.imread(str(item.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"OpenCV could not read mask: {item.mask_path}")
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Image/mask shape mismatch for {item.display_path}: "
            f"image={image.shape[:2]}, mask={mask.shape}"
        )

    binary_mask = np.where(mask >= threshold, 255, 0).astype(np.uint8)
    stats = MaskStats(
        height=mask.shape[0],
        width=mask.shape[1],
        nonzero_pixels=int(np.count_nonzero(mask)),
        foreground_pixels=int(np.count_nonzero(binary_mask)),
    )
    return image, binary_mask, stats


def parse_device(value: str) -> int | str:
    stripped = value.strip()
    if stripped.isdecimal():
        return int(stripped)
    return stripped


def load_pctnet(device: int | str):
    try:
        from libcom import ImageHarmonizationModel
    except ImportError as exc:
        raise RuntimeError(
            "Could not import libcom. Create the environment described in "
            f"{SCRIPT_DIR / 'README.md'} and run this script from that environment."
        ) from exc
    return ImageHarmonizationModel(device=device, model_type="PCTNet")


def normalize_model_result(result: object, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(result)
    if array.shape != expected_shape:
        raise ValueError(
            f"PCTNet returned shape {array.shape}; expected {expected_shape}."
        )
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    if array.size and float(array.max()) <= 1.0:
        array *= 255.0
    return np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)


def foreground_bbox(binary_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(binary_mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def foreground_gate_reason(
    binary_mask: np.ndarray,
    stats: MaskStats,
    args: argparse.Namespace,
) -> str | None:
    bbox = foreground_bbox(binary_mask)
    if bbox is None:
        return "empty_mask"
    x0, y0, x1, y1 = bbox
    if stats.foreground_pixels < args.min_foreground_pixels:
        return "foreground_too_small"
    if x1 - x0 < args.min_bbox_width or y1 - y0 < args.min_bbox_height:
        return "bbox_too_small"
    return None


def padded_foreground_roi(
    binary_mask: np.ndarray,
    padding: float,
) -> tuple[int, int, int, int]:
    bbox = foreground_bbox(binary_mask)
    if bbox is None:
        raise ValueError("Cannot calculate an ROI for an empty foreground mask")
    x0, y0, x1, y1 = bbox
    if padding <= 0:
        return 0, 0, binary_mask.shape[1], binary_mask.shape[0]
    pad = int(round(max(x1 - x0, y1 - y0) * padding))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(binary_mask.shape[1], x1 + pad),
        min(binary_mask.shape[0], y1 + pad),
    )


def harmonize_with_quality_gate(
    model,
    image: np.ndarray,
    binary_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    x0, y0, x1, y1 = padded_foreground_roi(
        binary_mask, args.roi_padding
    )
    crop_image = image[y0:y1, x0:x1]
    crop_mask = binary_mask[y0:y1, x0:x1]
    model_result = normalize_model_result(
        model(crop_image, crop_mask), crop_image.shape
    )
    candidate = image.copy()
    crop_foreground = crop_mask > 0
    original_pixels = crop_image[crop_foreground].astype(np.float32)
    harmonized_pixels = model_result[crop_foreground].astype(np.float32)
    blended_pixels = (
        (1.0 - args.blend_strength) * original_pixels
        + args.blend_strength * harmonized_pixels
    )
    blended_pixels = np.rint(np.clip(blended_pixels, 0.0, 255.0)).astype(
        np.uint8
    )
    candidate_crop = candidate[y0:y1, x0:x1]
    candidate_crop[crop_foreground] = blended_pixels

    raw_mean_color_shift = float(
        np.mean(np.abs(harmonized_pixels - original_pixels))
    )
    mean_color_shift = float(
        np.mean(np.abs(blended_pixels.astype(np.float32) - original_pixels))
    )
    original_luma = cv2.cvtColor(
        original_pixels.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)
    candidate_luma = cv2.cvtColor(
        blended_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    original_contrast = float(np.std(original_luma))
    candidate_contrast = float(np.std(candidate_luma))
    contrast_ratio = candidate_contrast / max(original_contrast, 1.0)
    rejection_reasons = []
    if raw_mean_color_shift > args.max_mean_color_shift:
        rejection_reasons.append("excessive_mean_color_shift")
    if contrast_ratio < args.min_contrast_ratio:
        rejection_reasons.append("foreground_contrast_collapse")
    accepted = not rejection_reasons
    if not accepted:
        candidate = image.copy()
    candidate[binary_mask == 0] = image[binary_mask == 0]
    return candidate, {
        "accepted": accepted,
        "roi": [x0, y0, x1, y1],
        "foreground_pixels": int(np.count_nonzero(binary_mask)),
        "raw_mean_color_shift": raw_mean_color_shift,
        "mean_color_shift": mean_color_shift,
        "original_luma_std": original_contrast,
        "candidate_luma_std": candidate_contrast,
        "contrast_ratio": contrast_ratio,
        "rejection_reasons": rejection_reasons,
    }


def write_image_atomic(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary_path), image):
        raise OSError(f"OpenCV failed to write output: {temporary_path}")
    os.replace(temporary_path, path)


def copy_image_atomic(source: Path, destination: Path) -> str:
    """Publish a byte-identical independent copy, preferring a COW reflink."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        method = "reflink"
        try:
            with source.open("rb") as source_stream, temporary_path.open(
                "wb"
            ) as destination_stream:
                fcntl.ioctl(
                    destination_stream.fileno(),
                    FICLONE,
                    source_stream.fileno(),
                )
            shutil.copystat(source, temporary_path)
        except OSError:
            method = "copy2"
            shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
        return method
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        completed_tokens = (
            load_completed_tokens(args.completion_dir)
            if args.completion_dir is not None
            else None
        )
        work_items = build_work_items(
            args.image_folder,
            args.mask_folder,
            args.output_root,
            completed_tokens,
        )

        print(f"Images:         {len(work_items)}")
        print(f"Mask threshold: >= {args.mask_threshold}")
        print(f"Output root:    {args.output_root.expanduser().resolve()}")
        if completed_tokens is not None:
            print(f"Completed tokens: {len(completed_tokens)}")

        if args.dry_run:
            for item in work_items:
                _, binary_mask, stats = load_image_and_binary_mask(
                    item,
                    args.mask_threshold,
                )
                action = (
                    "harmonize"
                    if stats.foreground_pixels
                    and foreground_gate_reason(binary_mask, stats, args) is None
                    else "copy (mask gate)"
                )
                print(
                    f"[{action:17}] {item.display_path}  "
                    f"foreground={stats.foreground_pixels} "
                    f"mask_nonzero={stats.nonzero_pixels}"
                )
            print("Dry run complete; PCTNet was not loaded and no files were written.")
            return 0

        model = None
        harmonized_count = 0
        copied_count = 0
        gated_count = 0
        rejected_count = 0
        skipped_count = 0
        for index, item in enumerate(work_items, start=1):
            if item.output_path.exists() and not args.overwrite:
                print(
                    f"[{index:03}/{len(work_items):03}] skip       "
                    f"{item.display_path}"
                )
                skipped_count += 1
                continue

            image, binary_mask, stats = load_image_and_binary_mask(
                item,
                args.mask_threshold,
            )
            if not stats.foreground_pixels:
                copy_method = copy_image_atomic(
                    item.image_path,
                    item.output_path,
                )
                print(
                    f"[{index:03}/{len(work_items):03}] copy       "
                    f"{item.display_path} ({copy_method})"
                )
                copied_count += 1
                continue

            gate_reason = foreground_gate_reason(binary_mask, stats, args)
            if gate_reason is not None:
                copy_method = copy_image_atomic(
                    item.image_path, item.output_path
                )
                print(
                    f"[{index:03}/{len(work_items):03}] gate       "
                    f"{item.display_path} ({gate_reason}, {copy_method})"
                )
                gated_count += 1
                continue

            if model is None:
                print(f"Loading libcom PCTNet on device {args.device}...")
                model = load_pctnet(parse_device(args.device))
            result, quality = harmonize_with_quality_gate(
                model,
                image,
                binary_mask,
                args,
            )
            if quality["accepted"]:
                write_image_atomic(item.output_path, result)
                print(
                    f"[{index:03}/{len(work_items):03}] harmonized "
                    f"{item.display_path} shift={quality['mean_color_shift']:.2f} "
                    f"contrast={quality['contrast_ratio']:.2f}"
                )
                harmonized_count += 1
            else:
                copy_method = copy_image_atomic(
                    item.image_path, item.output_path
                )
                print(
                    f"[{index:03}/{len(work_items):03}] reject     "
                    f"{item.display_path} "
                    f"reasons={','.join(quality['rejection_reasons'])} "
                    f"({copy_method})"
                )
                rejected_count += 1

        print(
            "Complete: "
            f"harmonized={harmonized_count}, copied={copied_count}, "
            f"gated={gated_count}, rejected={rejected_count}, "
            f"skipped={skipped_count}"
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
