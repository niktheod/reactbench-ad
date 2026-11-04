#!/usr/bin/env python3
"""Validate and pack OpenX vehicles for the NAVSIM and PAI-AV car compositors.

Run this through ``scripts/assets.sh cars``.  Source assets
are never modified.  Each prepared output preserves the source ``.xoma`` byte
for byte and writes a self-contained ``.blend`` plus preparation metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable, Sequence

try:  # Pure metadata helpers are intentionally importable outside Blender.
    import bpy
    from mathutils import Vector
except ModuleNotFoundError:  # pragma: no cover - exercised by normal Python tests.
    bpy = None
    Vector = None


SCHEMA_VERSION = 1
PREPARER_VERSION = "openx_navsim_vehicle_v2_ignore_append_weak_refs"
ROOT_OBJECT_NAME = "Grp_Root"
ASSET_KIND = "stationary_car"
DEFAULT_BOUNDS_TOLERANCE_M = 0.01
GIT_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
BLENDER_RAW_HEADER = b"BLENDER"
BLENDER_COMPRESSED_HEADERS = (b"\x1f\x8b", b"\x28\xb5\x2f\xfd")
OPENX_REPOSITORY_URL = "https://github.com/bounverif/openx-assets"
DEFAULT_EXCLUDED_ASSET_PATTERNS = (
    "m1_audi_q7_2015",
    "m1_mercedes_sl65amg_2008",
)
REMOVED_OBJECT_TYPES = {"CAMERA", "LIGHT", "LIGHT_PROBE", "SPEAKER"}
GEOMETRY_OBJECT_TYPES = {
    "CURVE",
    "CURVES",
    "FONT",
    "GREASEPENCIL",
    "MESH",
    "META",
    "POINTCLOUD",
    "SURFACE",
    "VOLUME",
}
SPDX_OPERATORS = {"AND", "OR", "WITH"}


@dataclass(frozen=True)
class SourceAsset:
    blend: Path
    xoma: Path
    relative_blend: Path

    @property
    def stem(self) -> str:
        return self.blend.stem


def arguments_after_double_dash() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    # This makes metadata-only dry runs and unit tests usable with CPython.
    return sys.argv[1:] if bpy is None else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openx-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="NAME_OR_GLOB",
        help=(
            "Prepare only matching source stems or relative paths; repeatable. "
            "For example: --asset 'm1_audi_*'"
        ),
    )
    parser.add_argument(
        "--exclude-asset",
        action="append",
        default=[],
        metavar="NAME_OR_GLOB",
        help=(
            "Exclude matching source stems or relative paths after --asset filtering; "
            "repeatable."
        ),
    )
    parser.add_argument(
        "--include-known-nonconforming",
        action="store_true",
        help=(
            "Disable the built-in exclusions for known nonconforming OpenX assets. "
            "Use with --asset for diagnostic validation."
        ),
    )
    parser.add_argument("--limit", type=int, help="Prepare at most this many selected assets.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild current outputs too.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run XOMA, licence, and Git LFS preflight without opening Blender files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Open and validate source scenes without creating prepared outputs.",
    )
    parser.add_argument(
        "--bounds-tolerance-m",
        type=float,
        default=DEFAULT_BOUNDS_TOLERANCE_M,
        help="Maximum per-bound difference between evaluated geometry and XOMA.",
    )
    args = parser.parse_args(arguments_after_double_dash())
    args.openx_root = args.openx_root.expanduser().resolve()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not math.isfinite(args.bounds_tolerance_m) or args.bounds_tolerance_m <= 0:
        parser.error("--bounds-tolerance-m must be positive and finite")
    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only are mutually exclusive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def discover_assets(source_dir: Path, openx_root: Path) -> list[SourceAsset]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"OpenX vehicle source directory not found: {source_dir}")
    blends = sorted(path.resolve() for path in source_dir.rglob("*.blend") if path.is_file())
    if not blends:
        raise FileNotFoundError(f"No .blend files found recursively under: {source_dir}")

    missing_xoma = [path for path in blends if not path.with_suffix(".xoma").is_file()]
    if missing_xoma:
        raise FileNotFoundError(
            "OpenX Blender files are missing matching .xoma sidecars: "
            + ", ".join(str(path) for path in missing_xoma)
        )
    orphan_xoma = sorted(
        path.resolve()
        for path in source_dir.rglob("*.xoma")
        if path.is_file() and not path.with_suffix(".blend").is_file()
    )
    if orphan_xoma:
        raise FileNotFoundError(
            "OpenX XOMA files are missing matching .blend files: "
            + ", ".join(str(path) for path in orphan_xoma)
        )

    stems = [path.stem for path in blends]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(
            "Prepared outputs are flat, so source Blender stems must be unique; duplicates: "
            + ", ".join(duplicates)
        )
    invalid_stems = [stem for stem in stems if re.fullmatch(r"[A-Za-z0-9_.-]+", stem) is None]
    if invalid_stems:
        raise ValueError(f"Source assets have unsafe output stems: {invalid_stems}")

    assets = []
    for blend in blends:
        try:
            relative_blend = blend.relative_to(openx_root)
        except ValueError:
            relative_blend = blend.relative_to(source_dir)
        assets.append(
            SourceAsset(
                blend=blend,
                xoma=blend.with_suffix(".xoma").resolve(),
                relative_blend=relative_blend,
            )
        )
    return assets


def asset_matches_pattern(asset: SourceAsset, pattern: str) -> bool:
    candidates = (
        asset.stem,
        asset.blend.name,
        asset.relative_blend.as_posix(),
    )
    lowered_pattern = pattern.lower()
    return any(
        fnmatch.fnmatchcase(candidate.lower(), lowered_pattern)
        for candidate in candidates
    )


def default_exclusions_for_assets(assets: Sequence[SourceAsset]) -> list[str]:
    """Return built-in exclusions that exist in this particular source tree."""
    return [
        pattern
        for pattern in DEFAULT_EXCLUDED_ASSET_PATTERNS
        if any(asset_matches_pattern(asset, pattern) for asset in assets)
    ]


def select_assets(
    assets: Sequence[SourceAsset],
    patterns: Sequence[str],
    limit: int | None,
    excluded_patterns: Sequence[str] = (),
) -> list[SourceAsset]:
    if not patterns:
        selected = list(assets)
    else:
        selected = [
            asset
            for asset in assets
            if any(asset_matches_pattern(asset, pattern) for pattern in patterns)
        ]
        unmatched = [
            pattern
            for pattern in patterns
            if not any(asset_matches_pattern(asset, pattern) for asset in assets)
        ]
        if unmatched:
            raise ValueError(f"No OpenX asset matched: {', '.join(unmatched)}")

    unmatched_exclusions = [
        pattern
        for pattern in excluded_patterns
        if not any(asset_matches_pattern(asset, pattern) for asset in assets)
    ]
    if unmatched_exclusions:
        raise ValueError(
            "No selected OpenX asset matched exclusion: "
            + ", ".join(unmatched_exclusions)
        )
    selected = [
        asset
        for asset in selected
        if not any(
            asset_matches_pattern(asset, pattern) for pattern in excluded_patterns
        )
    ]
    return selected[:limit] if limit is not None else selected


def finite_interval(value: Any, label: str, path: Path) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} in {path} must contain exactly two numbers")
    try:
        lower, upper = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} in {path} must contain exactly two numbers") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"Invalid {label} interval in {path}: {value!r}")
    return [lower, upper]


def required_string(value: Any, label: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} in {path} must be a non-empty string")
    return value.strip()


def required_string_list(value: Any, label: str, path: Path) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{label} in {path} must be a non-empty string list")
    return [item.strip() for item in value]


def finite_number(value: Any, label: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} in {path} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} in {path} must be finite")
    return parsed


def spdx_license_ids(expression: str) -> list[str]:
    identifiers = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", expression)
        if token.upper() not in SPDX_OPERATORS
    ]
    if not identifiers:
        raise ValueError(f"No SPDX licence identifiers found in expression: {expression!r}")
    return sorted(set(identifiers))


def load_xoma(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read XOMA metadata {path}: {exc}") from exc
    metadata = document.get("metadata") if isinstance(document, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError(f"XOMA file has no metadata object: {path}")
    if metadata.get("assetType") != "object" or metadata.get("objectClass") != "vehicle":
        raise ValueError(f"XOMA asset is not a vehicle object: {path}")

    name = required_string(metadata.get("name"), "name", path)
    asset_uuid = required_string(metadata.get("uuid"), "uuid", path)
    license_expression = required_string(metadata.get("license"), "license", path)
    authors = required_string_list(metadata.get("authors"), "authors", path)
    copyrights = required_string_list(metadata.get("copyrights"), "copyrights", path)
    bounds_value = metadata.get("boundingBox")
    if not isinstance(bounds_value, dict):
        raise ValueError(f"XOMA vehicle has no boundingBox object: {path}")
    bounds = {
        axis: finite_interval(bounds_value.get(axis), f"boundingBox.{axis}", path)
        for axis in ("x", "y", "z")
    }
    dimensions = {
        axis: bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")
    }
    if not (
        1.5 <= dimensions["x"] <= 15.0
        and 0.8 <= dimensions["y"] <= 4.0
        and 0.5 <= dimensions["z"] <= 5.0
    ):
        raise ValueError(f"Unreasonable vehicle dimensions {dimensions!r} in {path}")
    if not bounds["x"][0] < 0.0 < bounds["x"][1]:
        raise ValueError(f"Vehicle origin must lie between rear/front X bounds in {path}")
    if not bounds["y"][0] < 0.0 < bounds["y"][1]:
        raise ValueError(f"Vehicle origin must lie between lateral Y bounds in {path}")
    if abs(bounds["z"][0]) > 0.10:
        raise ValueError(f"Vehicle ground plane must be near local z=0 in {path}")
    if metadata.get("animated") is True:
        raise ValueError(f"Stationary-car assets cannot be marked animated: {path}")

    vehicle_data = metadata.get("vehicleClassData")
    if not isinstance(vehicle_data, dict):
        raise ValueError(f"XOMA vehicle has no vehicleClassData object: {path}")
    axles_value = vehicle_data.get("axles")
    if not isinstance(axles_value, dict):
        raise ValueError(f"XOMA vehicle has no vehicleClassData.axles object: {path}")
    axles = {}
    for axle_name in ("frontAxle", "rearAxle"):
        axle = axles_value.get(axle_name)
        if not isinstance(axle, dict):
            raise ValueError(f"XOMA vehicle has no {axle_name} object: {path}")
        values = {
            field: finite_number(axle.get(field), f"{axle_name}.{field}", path)
            for field in ("positionX", "trackWidth", "wheelDiameter")
        }
        values["positionZ"] = (
            finite_number(axle.get("positionZ"), f"{axle_name}.positionZ", path)
            if axle.get("positionZ") is not None
            else None
        )
        if values["trackWidth"] <= 0.0 or values["wheelDiameter"] <= 0.0:
            raise ValueError(f"{axle_name} has non-positive wheel dimensions in {path}")
        if not bounds["x"][0] - 0.5 <= values["positionX"] <= bounds["x"][1] + 0.5:
            raise ValueError(f"{axle_name} lies outside the vehicle X bounds in {path}")
        if values["positionZ"] is not None and not (
            bounds["z"][0] - 0.25
            <= values["positionZ"]
            <= bounds["z"][1] + 0.25
        ):
            raise ValueError(f"{axle_name} lies outside the vehicle Z bounds in {path}")
        if values["trackWidth"] > dimensions["y"] + 0.5:
            raise ValueError(f"{axle_name} track width exceeds the vehicle width in {path}")
        axles[axle_name] = values
    if axles["frontAxle"]["positionX"] <= axles["rearAxle"]["positionX"]:
        raise ValueError(f"frontAxle must be ahead of rearAxle along local +X in {path}")

    return {
        "document": document,
        "metadata": metadata,
        "name": name,
        "uuid": asset_uuid,
        "license": license_expression,
        "license_ids": spdx_license_ids(license_expression),
        "authors": authors,
        "copyrights": copyrights,
        "bounds_m": bounds,
        "dimensions_m": dimensions,
        "axles": axles,
    }


def git_lfs_pointer(path: Path) -> dict[str, Any] | None:
    with path.open("rb") as handle:
        header = handle.read(1024)
    if not header.startswith(GIT_LFS_HEADER):
        return None
    text = header.decode("utf-8", errors="replace")
    oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", text, flags=re.MULTILINE)
    size_match = re.search(r"^size ([0-9]+)$", text, flags=re.MULTILINE)
    return {
        "oid_sha256": oid_match.group(1) if oid_match else "unknown",
        "expected_size_bytes": int(size_match.group(1)) if size_match else None,
    }


def validate_blend_payload(path: Path, openx_root: Path) -> None:
    pointer = git_lfs_pointer(path)
    if pointer is not None:
        size = pointer["expected_size_bytes"]
        detail = f", expected payload size {size} bytes" if size is not None else ""
        raise ValueError(
            f"{path} is a Git LFS pointer "
            f"(oid {pointer['oid_sha256']}{detail}), not a Blender file. "
            f"Fetch payloads first with git-lfs in {openx_root}."
        )
    with path.open("rb") as handle:
        magic = handle.read(7)
    if not (
        magic.startswith(BLENDER_RAW_HEADER)
        or any(magic.startswith(header) for header in BLENDER_COMPRESSED_HEADERS)
    ):
        raise ValueError(
            f"Source file is neither a raw nor a supported compressed Blender file: {path}"
        )


def validate_license_files(openx_root: Path, license_ids: Sequence[str]) -> list[Path]:
    paths = []
    for identifier in license_ids:
        path = openx_root / "LICENSES" / f"{identifier}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"OpenX licence text for {identifier} is missing: {path}"
            )
        paths.append(path.resolve())
    return paths


def preflight_asset(asset: SourceAsset, openx_root: Path) -> dict[str, Any]:
    xoma = load_xoma(asset.xoma)
    validate_license_files(openx_root, xoma["license_ids"])
    validate_blend_payload(asset.blend, openx_root)
    return {
        "asset": asset,
        "xoma": xoma,
        "source_blend_sha256": sha256_file(asset.blend),
        "source_xoma_sha256": sha256_file(asset.xoma),
    }


def preflight_assets(
    assets: Sequence[SourceAsset], openx_root: Path
) -> dict[str, dict[str, Any]]:
    results = {}
    failures = []
    for asset in assets:
        try:
            results[asset.stem] = preflight_asset(asset, openx_root)
        except Exception as exc:
            failures.append(f"{asset.stem}: {exc}")
    if failures:
        raise RuntimeError(
            "OpenX source preflight failed; no outputs were created:\n  - "
            + "\n  - ".join(failures)
        )
    return results


def require_blender() -> None:
    if bpy is None or Vector is None:
        raise RuntimeError(
            "Blender Python is required after metadata preflight; run prepare_openx_assets.sh"
        )


def object_has_animation(obj: Any) -> bool:
    owners: list[Any] = [obj]
    if obj.data is not None:
        owners.append(obj.data)
    if obj.type == "MESH" and obj.data is not None and obj.data.shape_keys is not None:
        owners.append(obj.data.shape_keys)
    for owner in owners:
        animation = getattr(owner, "animation_data", None)
        if animation is None:
            continue
        if animation.action is not None or animation.drivers or animation.nla_tracks:
            return True
    return False


def is_descendant_of(obj: Any, root: Any) -> bool:
    current = obj
    while current is not None:
        if current == root:
            return True
        current = current.parent
    return False


def geometry_objects(scene: Any) -> list[Any]:
    return [
        obj
        for obj in scene.objects
        if obj.type in GEOMETRY_OBJECT_TYPES and not obj.hide_render
    ]


def evaluated_geometry_stats(scene: Any) -> dict[str, Any]:
    require_blender()
    objects = geometry_objects(scene)
    if not objects:
        raise ValueError("Scene contains no renderable vehicle geometry")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    vertex_count = 0
    triangle_count = 0
    object_names = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        try:
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        except RuntimeError:
            mesh = None
        if mesh is not None and mesh.vertices:
            object_names.append(obj.name)
            vertex_count += len(mesh.vertices)
            mesh.calc_loop_triangles()
            triangle_count += len(mesh.loop_triangles)
            for vertex in mesh.vertices:
                world = evaluated.matrix_world @ vertex.co
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], world[axis])
                    maximum[axis] = max(maximum[axis], world[axis])
            evaluated.to_mesh_clear()
            continue
        if mesh is not None:
            evaluated.to_mesh_clear()
        corners = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
        if not corners:
            continue
        object_names.append(obj.name)
        for world in corners:
            for axis in range(3):
                minimum[axis] = min(minimum[axis], world[axis])
                maximum[axis] = max(maximum[axis], world[axis])
    if not object_names or not all(math.isfinite(value) for value in (*minimum, *maximum)):
        raise ValueError("Unable to compute finite bounds for renderable vehicle geometry")
    dimensions = maximum - minimum
    if min(dimensions) <= 0:
        raise ValueError(f"Vehicle geometry has degenerate bounds: {list(dimensions)}")
    return {
        "minimum": minimum,
        "maximum": maximum,
        "dimensions": dimensions,
        "objects": sorted(object_names),
        "object_count": len(object_names),
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
    }


def vector_list(vector: Iterable[float]) -> list[float]:
    return [float(value) for value in vector]


def bounds_dict(stats: dict[str, Any]) -> dict[str, list[float]]:
    return {
        axis: [float(stats["minimum"][index]), float(stats["maximum"][index])]
        for index, axis in enumerate(("x", "y", "z"))
    }


def matrix_identity_error(matrix: Any) -> float:
    return max(
        abs(float(matrix[row][column]) - (1.0 if row == column else 0.0))
        for row in range(4)
        for column in range(4)
    )


def normalized_blender_path(path: str, data_block: Any | None = None) -> str:
    require_blender()
    library = getattr(data_block, "library", None) if data_block is not None else None
    absolute = bpy.path.abspath(path, library=library)
    return str(Path(absolute).resolve())


def library_weak_reference_records() -> list[dict[str, str]]:
    """Return append/copy provenance that is not a runtime file dependency."""
    require_blender()
    records = []
    for property_definition in bpy.data.bl_rna.properties:
        if property_definition.type != "COLLECTION":
            continue
        collection_name = property_definition.identifier
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue
        for data_block in collection:
            weak_reference = getattr(data_block, "library_weak_reference", None)
            if weak_reference is None or not weak_reference.filepath:
                continue
            records.append(
                {
                    "data_collection": collection_name,
                    "data_block": str(getattr(data_block, "name", "")),
                    "filepath": str(weak_reference.filepath),
                    "absolute_filepath": normalized_blender_path(
                        weak_reference.filepath, data_block
                    ),
                }
            )
    return sorted(
        records,
        key=lambda item: (
            item["absolute_filepath"],
            item["data_collection"],
            item["data_block"],
        ),
    )


def subtract_path_occurrences(
    paths: Sequence[str], ignored_paths: Sequence[str]
) -> list[str]:
    """Subtract only matching occurrences, preserving a real duplicate reference."""
    remaining = Counter(paths)
    for path, count in Counter(ignored_paths).items():
        remaining[path] = max(0, remaining[path] - count)
    return sorted(remaining.elements())


def external_path_report() -> dict[str, Any]:
    require_blender()
    raw_paths = [
        normalized_blender_path(path)
        for path in bpy.utils.blend_paths(absolute=True, packed=True)
    ]
    weak_references = library_weak_reference_records()
    required_occurrences = subtract_path_occurrences(
        raw_paths,
        [item["absolute_filepath"] for item in weak_references],
    )
    return {
        "external_paths": sorted(set(required_occurrences)),
        "external_path_reference_count": len(required_occurrences),
        "ignored_library_weak_references": weak_references,
        "ignored_library_weak_reference_count": len(weak_references),
    }


def validate_scene(
    xoma: dict[str, Any], bounds_tolerance_m: float, *, prepared: bool
) -> dict[str, Any]:
    require_blender()
    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.update()
    root = bpy.data.objects.get(ROOT_OBJECT_NAME)
    failures = []
    if root is None or root.name not in scene.objects:
        failures.append(f"{ROOT_OBJECT_NAME} is missing from the active scene")
    elif root.type != "EMPTY":
        failures.append(f"{ROOT_OBJECT_NAME} must be an EMPTY object")
    else:
        if root.parent is not None:
            failures.append(f"{ROOT_OBJECT_NAME} must not have a parent")
        identity_error = matrix_identity_error(root.matrix_world)
        if identity_error > 1e-6:
            failures.append(
                f"{ROOT_OBJECT_NAME} must have an identity world transform; "
                f"max error={identity_error}"
            )

    objects = geometry_objects(scene)
    if root is not None:
        outside_root = [obj.name for obj in objects if not is_descendant_of(obj, root)]
        if outside_root:
            failures.append(
                f"renderable geometry is outside {ROOT_OBJECT_NAME}: {outside_root}"
            )
        animated = [
            obj.name
            for obj in scene.objects
            if is_descendant_of(obj, root) and object_has_animation(obj)
        ]
        if animated:
            failures.append(f"stationary vehicle hierarchy contains animation: {animated}")
    external_report = None
    if prepared:
        unexpected = [obj.name for obj in scene.objects if obj.type in REMOVED_OBJECT_TYPES]
        if unexpected:
            failures.append(f"non-asset scene objects remain: {unexpected}")
        linked = [library.filepath for library in bpy.data.libraries if library.filepath]
        if linked:
            failures.append(f"linked Blender libraries remain: {linked}")
        external_report = external_path_report()
        unpacked = external_report["external_paths"]
        if unpacked:
            failures.append(f"unpacked external resources remain: {unpacked}")

    try:
        stats = evaluated_geometry_stats(scene)
    except Exception as exc:
        failures.append(str(exc))
        stats = None
    bound_errors = {}
    if stats is not None:
        actual_bounds = bounds_dict(stats)
        for axis in ("x", "y", "z"):
            expected = xoma["bounds_m"][axis]
            actual = actual_bounds[axis]
            error = max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1]))
            bound_errors[axis] = error
            if error > bounds_tolerance_m:
                failures.append(
                    f"evaluated {axis.upper()} bounds {actual} differ from XOMA {expected} "
                    f"by {error:.6f} m (tolerance {bounds_tolerance_m:.6f} m)"
                )
    if failures:
        raise RuntimeError("Vehicle scene validation failed: " + "; ".join(failures))

    rear_marker = bpy.data.objects.get("Grp_Rear_Axle_Center")
    rear_marker_position = None
    if rear_marker is not None and root is not None and is_descendant_of(rear_marker, root):
        rear_marker_position = vector_list(
            root.matrix_world.inverted() @ rear_marker.matrix_world.translation
        )
    return {
        "bounds_m": bounds_dict(stats),
        "bound_errors_m": bound_errors,
        "bounds_tolerance_m": float(bounds_tolerance_m),
        "object_names": stats["objects"],
        "object_count": stats["object_count"],
        "vertex_count_evaluated": stats["vertex_count"],
        "triangle_count_evaluated": stats["triangle_count"],
        "root_identity_error": matrix_identity_error(root.matrix_world),
        "rear_axle_marker_position_m": rear_marker_position,
        "source_unit_system": scene.unit_settings.system,
        "source_unit_scale_length": float(scene.unit_settings.scale_length),
        "ignored_library_weak_reference_count": (
            external_report["ignored_library_weak_reference_count"]
            if external_report is not None
            else 0
        ),
    }


def validate_external_resources() -> dict[str, Any]:
    require_blender()
    linked = [library.filepath for library in bpy.data.libraries if library.filepath]
    if linked:
        raise ValueError(f"Linked Blender libraries are unsupported: {linked}")
    report = external_path_report()
    missing = [path for path in report["external_paths"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing external Blender resources: " + "; ".join(missing))
    return report


def image_is_packed(image: Any) -> bool:
    if image.packed_file is not None:
        return True
    return bool(getattr(image, "packed_files", ()))


def pack_external_resources() -> dict[str, Any]:
    require_blender()
    source_report = validate_external_resources()
    bpy.ops.file.pack_all()
    prepared_report = external_path_report()
    unpacked = prepared_report["external_paths"]
    if unpacked:
        raise RuntimeError(f"External resources remain unpacked: {unpacked}")
    packed_images = sorted(
        image.name
        for image in bpy.data.images
        if image.source == "FILE" and image.filepath and image_is_packed(image)
    )
    return {
        "source_external_paths": source_report["external_paths"],
        "ignored_library_weak_references": source_report[
            "ignored_library_weak_references"
        ],
        "packed_image_names": packed_images,
    }


def remove_non_asset_scene_data() -> None:
    require_blender()
    scene = bpy.context.scene
    for other_scene in list(bpy.data.scenes):
        if other_scene != scene:
            bpy.data.scenes.remove(other_scene)
    scene.camera = None
    scene.world = None
    scene.use_nodes = False
    for obj in list(bpy.data.objects):
        if obj.type in REMOVED_OBJECT_TYPES:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_set(1)
    bpy.context.view_layer.update()


def set_asset_properties(
    preflight: dict[str, Any], source_revision: str, diagnostics: dict[str, Any]
) -> None:
    require_blender()
    scene = bpy.context.scene
    root = bpy.data.objects[ROOT_OBJECT_NAME]
    xoma = preflight["xoma"]
    properties: dict[str, Any] = {
        "navsim_asset_schema_version": SCHEMA_VERSION,
        "navsim_asset_preparer_version": PREPARER_VERSION,
        "navsim_asset_kind": ASSET_KIND,
        "navsim_actor_root": ROOT_OBJECT_NAME,
        "navsim_asset_uuid": xoma["uuid"],
        "navsim_asset_name": xoma["name"],
        "navsim_asset_license": xoma["license"],
        "navsim_source_repository": OPENX_REPOSITORY_URL,
        "navsim_source_revision": source_revision,
        "navsim_source_blend_sha256": preflight["source_blend_sha256"],
        "navsim_source_xoma_sha256": preflight["source_xoma_sha256"],
        "navsim_local_forward_axis": "+X",
        "navsim_local_lateral_axis": "+Y",
        "navsim_local_up_axis": "+Z",
        "navsim_resources_packed": True,
        "navsim_bounds_tolerance_m": diagnostics["bounds_tolerance_m"],
    }
    for key, value in properties.items():
        scene[key] = value
        root[key] = value
    # Geometry coordinates already match metric XOMA bounds; this declares the
    # output units without rescaling or otherwise repairing the source model.
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def preparation_paths(output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"{stem}.blend",
        output_dir / f"{stem}.xoma",
        output_dir / f"{stem}.preparation.json",
    )


def existing_output_is_current(
    output_dir: Path, stem: str, preflight: dict[str, Any]
) -> bool:
    output_blend, output_xoma, output_metadata = preparation_paths(output_dir, stem)
    if not all(path.is_file() for path in (output_blend, output_xoma, output_metadata)):
        return False
    try:
        metadata = json.loads(output_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("preparer_version") == PREPARER_VERSION
        and metadata.get("preparer_script_sha256")
        == sha256_file(Path(__file__).resolve())
        and metadata.get("source", {}).get("blend_sha256")
        == preflight["source_blend_sha256"]
        and metadata.get("source", {}).get("xoma_sha256")
        == preflight["source_xoma_sha256"]
        and metadata.get("output", {}).get("xoma_sha256") == sha256_file(output_xoma)
        and metadata.get("output", {}).get("blend_sha256") == sha256_file(output_blend)
        and metadata.get("validation", {}).get("passed") is True
        and metadata.get("output", {}).get("self_contained") is True
    )


def material_names() -> list[str]:
    return sorted(
        {
            slot.material.name
            for obj in geometry_objects(bpy.context.scene)
            for slot in obj.material_slots
            if slot.material is not None
        }
    )


def prepare_asset(
    asset: SourceAsset,
    preflight: dict[str, Any],
    openx_root: Path,
    output_dir: Path,
    source_revision: str,
    overwrite: bool,
    bounds_tolerance_m: float,
) -> dict[str, Any]:
    require_blender()
    output_blend, output_xoma, output_metadata = preparation_paths(output_dir, asset.stem)
    if not overwrite and existing_output_is_current(output_dir, asset.stem, preflight):
        print(f"SKIP current {asset.stem}: {output_blend}", flush=True)
        return {"asset": asset.stem, "status": "skipped", "blend": str(output_blend)}
    if any(path.exists() for path in (output_blend, output_xoma, output_metadata)):
        print(f"REBUILD stale or incomplete {asset.stem}: {output_blend}", flush=True)

    print(f"PREPARE {asset.relative_blend} -> {output_blend.name}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(asset.blend), load_ui=False)
    source_diagnostics = validate_scene(
        preflight["xoma"], bounds_tolerance_m, prepared=False
    )
    remove_non_asset_scene_data()
    packing = pack_external_resources()
    prepared_diagnostics = validate_scene(
        preflight["xoma"], bounds_tolerance_m, prepared=True
    )
    set_asset_properties(preflight, source_revision, prepared_diagnostics)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    prepared_diagnostics = validate_scene(
        preflight["xoma"], bounds_tolerance_m, prepared=True
    )

    temporary_blend = output_dir / f".{asset.stem}.{os.getpid()}.tmp.blend"
    temporary_xoma = output_dir / f".{asset.stem}.{os.getpid()}.tmp.xoma"
    temporary_metadata = output_dir / f".{asset.stem}.{os.getpid()}.tmp.preparation.json"
    for path in (temporary_blend, temporary_xoma, temporary_metadata):
        path.unlink(missing_ok=True)
    try:
        save_result = bpy.ops.wm.save_as_mainfile(
            filepath=str(temporary_blend), check_existing=False, compress=True
        )
        if "FINISHED" not in save_result or not temporary_blend.is_file():
            raise RuntimeError(f"Blender did not save candidate asset: {temporary_blend}")
        bpy.ops.wm.open_mainfile(filepath=str(temporary_blend), load_ui=False)
        reloaded_diagnostics = validate_scene(
            preflight["xoma"], bounds_tolerance_m, prepared=True
        )
        output_blend_sha256 = sha256_file(temporary_blend)
        shutil.copyfile(asset.xoma, temporary_xoma)
        output_xoma_sha256 = sha256_file(temporary_xoma)
        if output_xoma_sha256 != preflight["source_xoma_sha256"]:
            raise RuntimeError("Candidate XOMA is not byte-identical to the OpenX source")

        xoma = preflight["xoma"]
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "preparer_version": PREPARER_VERSION,
            "kind": ASSET_KIND,
            "id": xoma["uuid"],
            "name": xoma["name"],
            "root_object": ROOT_OBJECT_NAME,
            "source": {
                "repository": OPENX_REPOSITORY_URL,
                "repository_root": str(openx_root),
                "revision": source_revision,
                "blend": asset.relative_blend.as_posix(),
                "blend_sha256": preflight["source_blend_sha256"],
                "xoma": asset.xoma.name,
                "xoma_sha256": preflight["source_xoma_sha256"],
            },
            "attribution": {
                "authors": xoma["authors"],
                "license": xoma["license"],
                "copyrights": xoma["copyrights"],
            },
            "output": {
                "blend": output_blend.name,
                "blend_sha256": output_blend_sha256,
                "xoma": output_xoma.name,
                "xoma_sha256": output_xoma_sha256,
                "self_contained": True,
            },
            "geometry": {
                "coordinate_system": "right-handed +X forward, +Y lateral, +Z up",
                "meters_per_blender_unit": 1.0,
                "xoma_bounds_m": xoma["bounds_m"],
                "evaluated_bounds_m": reloaded_diagnostics["bounds_m"],
                "object_names": reloaded_diagnostics["object_names"],
                "object_count": reloaded_diagnostics["object_count"],
                "vertex_count_evaluated": reloaded_diagnostics["vertex_count_evaluated"],
                "triangle_count_evaluated": reloaded_diagnostics["triangle_count_evaluated"],
            },
            "materials": {
                "material_names": material_names(),
                "source_external_paths": packing["source_external_paths"],
                "ignored_library_weak_references": packing[
                    "ignored_library_weak_references"
                ],
                "packed_image_names": packing["packed_image_names"],
                "all_external_resources_packed": True,
            },
            "validation": {
                "passed": True,
                "source_scene": source_diagnostics,
                "prepared_scene": prepared_diagnostics,
                "reloaded_scene": reloaded_diagnostics,
                "reopened_after_save": True,
                "geometry_transformation_applied": False,
                "xoma_preserved_byte_for_byte": True,
            },
            "blender_version": bpy.app.version_string,
            "preparer_script_sha256": sha256_file(Path(__file__).resolve()),
        }
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The preparation sidecar is the commit marker and is replaced last.
        os.replace(temporary_xoma, output_xoma)
        os.replace(temporary_blend, output_blend)
        os.replace(temporary_metadata, output_metadata)
    finally:
        for path in (temporary_blend, temporary_xoma, temporary_metadata):
            path.unlink(missing_ok=True)

    print(
        f"DONE {asset.stem}: dimensions_m={list(xoma['dimensions_m'].values())} "
        f"triangles={reloaded_diagnostics['triangle_count_evaluated']} "
        f"packed_images={len(packing['packed_image_names'])}",
        flush=True,
    )
    return {
        "asset": asset.stem,
        "status": "prepared",
        "blend": str(output_blend),
        "xoma": str(output_xoma),
        "preparation_metadata": str(output_metadata),
    }


def validate_source_asset(
    asset: SourceAsset, preflight: dict[str, Any], bounds_tolerance_m: float
) -> dict[str, Any]:
    require_blender()
    bpy.ops.wm.open_mainfile(filepath=str(asset.blend), load_ui=False)
    diagnostics = validate_scene(preflight["xoma"], bounds_tolerance_m, prepared=False)
    remove_non_asset_scene_data()
    external_report = validate_external_resources()
    print(
        f"VALID {asset.stem}: bounds_m={diagnostics['bounds_m']} "
        f"external_resources={len(external_report['external_paths'])} "
        "ignored_library_weak_references="
        f"{external_report['ignored_library_weak_reference_count']}",
        flush=True,
    )
    return {
        "asset": asset.stem,
        "status": "valid",
        "diagnostics": diagnostics,
        "external_resources": external_report,
    }


def copy_required_licenses(
    openx_root: Path, preflights: Iterable[dict[str, Any]], destination_root: Path
) -> list[str]:
    identifiers = sorted(
        {
            identifier
            for preflight in preflights
            for identifier in preflight["xoma"]["license_ids"]
        }
    )
    license_dir = destination_root / "LICENSES"
    license_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for identifier in identifiers:
        source = openx_root / "LICENSES" / f"{identifier}.txt"
        destination = license_dir / source.name
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        copied.append(str(destination))
    return copied


def markdown_text(value: str) -> str:
    return " ".join(value.split())


def write_attribution(output_dir: Path, destination_root: Path) -> Path:
    entries = []
    for xoma_path in sorted(output_dir.glob("*.xoma")):
        if not xoma_path.with_suffix(".blend").is_file():
            continue
        xoma = load_xoma(xoma_path)
        entries.append((xoma_path.stem, xoma))
    lines = [
        "# OpenX vehicle attribution",
        "",
        (
            "Prepared from [bounverif/openx-assets]"
            f"({OPENX_REPOSITORY_URL}). The original XOMA sidecar beside each vehicle is "
            "the authoritative per-asset provenance record."
        ),
        "",
        (
            "Licence texts are copied into `LICENSES/`. Preparation removes non-asset "
            "scene objects, packs resources, and resaves the Blender file; it does not "
            "transform vehicle geometry."
        ),
        "",
    ]
    for stem, xoma in entries:
        lines.extend(
            [
                f"## {markdown_text(xoma['name'])} (`{stem}`)",
                "",
                f"- UUID: `{xoma['uuid']}`",
                f"- Licence: `{markdown_text(xoma['license'])}`",
                "- Authors: " + "; ".join(markdown_text(item) for item in xoma["authors"]),
                "- Copyrights: "
                + "; ".join(markdown_text(item) for item in xoma["copyrights"]),
                "",
            ]
        )
    path = destination_root / "OPENX_ATTRIBUTION.md"
    atomic_text(path, "\n".join(lines).rstrip() + "\n")
    return path


def main() -> None:
    args = parse_args()
    if not args.openx_root.is_dir():
        raise FileNotFoundError(f"OpenX repository root not found: {args.openx_root}")
    assets = discover_assets(args.source_dir, args.openx_root)
    default_exclusions = (
        []
        if args.include_known_nonconforming
        else default_exclusions_for_assets(assets)
    )
    excluded_patterns = list(dict.fromkeys([*default_exclusions, *args.exclude_asset]))
    selected = select_assets(
        assets,
        args.asset,
        args.limit,
        excluded_patterns=excluded_patterns,
    )
    if not selected:
        raise ValueError(
            "No OpenX assets selected after exclusions; use "
            "--include-known-nonconforming to opt in to built-in exclusions"
        )
    print(f"OPENX_ROOT={args.openx_root}", flush=True)
    print(f"SOURCE_DIR={args.source_dir}", flush=True)
    print(f"OUTPUT_DIR={args.output_dir}", flush=True)
    print(
        f"DEFAULT_EXCLUSIONS_ENABLED={not args.include_known_nonconforming}",
        flush=True,
    )
    for pattern in excluded_patterns:
        print(f"EXCLUDE_PATTERN {pattern}", flush=True)
    print(f"SELECTED_ASSETS={len(selected)}", flush=True)
    for asset in selected:
        print(f"SELECT {asset.relative_blend}", flush=True)

    preflights = preflight_assets(selected, args.openx_root)
    for asset in selected:
        xoma = preflights[asset.stem]["xoma"]
        print(
            f"PREFLIGHT {asset.stem}: name={xoma['name']!r} license={xoma['license']!r} "
            f"dimensions_m={list(xoma['dimensions_m'].values())}",
            flush=True,
        )
    if args.dry_run:
        print(f"OPENX_PREFLIGHT_COMPLETE selected={len(selected)}", flush=True)
        return

    require_blender()
    if args.validate_only:
        failures = []
        for asset in selected:
            try:
                validate_source_asset(
                    asset, preflights[asset.stem], args.bounds_tolerance_m
                )
            except Exception as exc:
                failures.append({"asset": asset.stem, "error": str(exc)})
                print(f"FAILED {asset.stem}: {exc}", file=sys.stderr, flush=True)
        print(
            f"OPENX_VALIDATION_COMPLETE valid={len(selected) - len(failures)} "
            f"failed={len(failures)}",
            flush=True,
        )
        if failures:
            raise SystemExit(1)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination_root = args.output_dir.parent
    source_revision = git_revision(args.openx_root)
    copied_licenses = copy_required_licenses(
        args.openx_root, preflights.values(), destination_root
    )
    results = []
    failures = []
    for asset in selected:
        try:
            results.append(
                prepare_asset(
                    asset,
                    preflights[asset.stem],
                    args.openx_root,
                    args.output_dir,
                    source_revision,
                    args.overwrite,
                    args.bounds_tolerance_m,
                )
            )
        except Exception as exc:
            failure = {
                "asset": asset.stem,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"FAILED {asset.stem}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()

    attribution_path = write_attribution(args.output_dir, destination_root)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "preparer_version": PREPARER_VERSION,
        "openx_root": str(args.openx_root),
        "openx_revision": source_revision,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "bounds_tolerance_m": args.bounds_tolerance_m,
        "included_asset_patterns": list(args.asset),
        "default_exclusions_enabled": not args.include_known_nonconforming,
        "default_excluded_asset_patterns": default_exclusions,
        "additional_excluded_asset_patterns": list(args.exclude_asset),
        "excluded_asset_patterns": excluded_patterns,
        "selected_count": len(selected),
        "prepared_count": sum(result["status"] == "prepared" for result in results),
        "skipped_count": sum(result["status"] == "skipped" for result in results),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
        "copied_license_files": copied_licenses,
        "attribution_file": str(attribution_path),
        "blender_version": bpy.app.version_string,
        "preparer_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = destination_root / "generation_manifest.json"
    atomic_json(manifest_path, summary)
    print(
        "OPENX_PREPARATION_COMPLETE "
        f"prepared={summary['prepared_count']} skipped={summary['skipped_count']} "
        f"failed={summary['failed_count']} output={args.output_dir}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
