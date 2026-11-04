#!/usr/bin/env python3
"""Run libcom PCTNet harmonization over one NAVSIM token image folder.

The sibling mask tree is inferred by replacing the ``images`` directory in the
input path with ``masks``. Only near-white mask pixels are used as foreground,
which excludes the lower-alpha shadow stored in the combined masks.
"""

from __future__ import annotations

import argparse
import fcntl
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
        "--dry-run",
        action="store_true",
        help="Validate path mapping and masks without loading PCTNet or writing outputs.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.mask_threshold <= 255:
        parser.error("--mask-threshold must be in [0, 255]")
    return args


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
        work_items = build_work_items(
            args.image_folder,
            args.mask_folder,
            args.output_root,
        )

        print(f"Images:         {len(work_items)}")
        print(f"Mask threshold: >= {args.mask_threshold}")
        print(f"Output root:    {args.output_root.expanduser().resolve()}")

        if args.dry_run:
            for item in work_items:
                _, _, stats = load_image_and_binary_mask(
                    item,
                    args.mask_threshold,
                )
                action = (
                    "harmonize"
                    if stats.foreground_pixels
                    else "copy (empty mask)"
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

            if model is None:
                print(f"Loading libcom PCTNet on device {args.device}...")
                model = load_pctnet(parse_device(args.device))
            result = normalize_model_result(
                model(image, binary_mask),
                image.shape,
            )
            # Dataset safety: the background and excluded shadow pixels must be
            # identical to the input, even if a model backend changes them.
            result[binary_mask == 0] = image[binary_mask == 0]
            write_image_atomic(item.output_path, result)
            print(
                f"[{index:03}/{len(work_items):03}] harmonized "
                f"{item.display_path}"
            )
            harmonized_count += 1

        print(
            "Complete: "
            f"harmonized={harmonized_count}, copied={copied_count}, "
            f"skipped={skipped_count}"
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
