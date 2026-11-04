#!/usr/bin/env python3
"""Measure full-gait pedestrian bounds and validate or update asset sidecars.

Run with Blender, for example::

    blender --background --python scripts/measure_pedestrian_geometry.py -- \
      --asset-dir generated_assets/rocketbox/assets --write-sidecars
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import bpy


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reactbench.scoring.pedestrian_geometry import (  # noqa: E402
    make_pedestrian_geometry,
    normalize_pedestrian_geometry,
    union_local_bounds,
)


def arguments_after_double_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="NAME_OR_GLOB",
        help="Measure only matching asset stems; repeatable.",
    )
    parser.add_argument(
        "--write-sidecars",
        action="store_true",
        help="Atomically add/replace actor_geometry instead of checking it.",
    )
    args = parser.parse_args(arguments_after_double_dash())
    args.asset_dir = args.asset_dir.expanduser().resolve()
    return args


def _is_descendant_of(obj: bpy.types.Object, root: bpy.types.Object) -> bool:
    current = obj
    while current is not None:
        if current == root:
            return True
        current = current.parent
    return False


def render_visible_meshes(
    root: bpy.types.Object,
) -> list[bpy.types.Object]:
    meshes = sorted(
        (
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH"
            and not obj.hide_render
            and _is_descendant_of(obj, root)
        ),
        key=lambda obj: obj.name,
    )
    if not meshes:
        raise RuntimeError(
            f"No render-visible mesh descendants found under {root.name!r}"
        )
    return meshes


def evaluated_frame_local_bounds(
    meshes: Sequence[bpy.types.Object],
    root: bpy.types.Object,
) -> tuple[dict[str, list[float]], int]:
    """Measure evaluated vertices in the renderer root's local frame."""

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_root = root.evaluated_get(depsgraph)
    world_to_root = evaluated_root.matrix_world.inverted()
    minima = [float("inf")] * 3
    maxima = [float("-inf")] * 3
    vertex_count = 0
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            mesh_to_root = world_to_root @ evaluated.matrix_world
            for vertex in evaluated_mesh.vertices:
                point = mesh_to_root @ vertex.co
                for axis in range(3):
                    coordinate = float(point[axis])
                    if not math.isfinite(coordinate):
                        raise RuntimeError(
                            f"Non-finite evaluated vertex in {obj.name!r}"
                        )
                    minima[axis] = min(minima[axis], coordinate)
                    maxima[axis] = max(maxima[axis], coordinate)
                vertex_count += 1
        finally:
            evaluated.to_mesh_clear()
    if vertex_count == 0:
        raise RuntimeError("No evaluated mesh vertices found")
    return (
        {
            axis: [minima[index], maxima[index]]
            for index, axis in enumerate(("x", "y", "z"))
        },
        vertex_count,
    )


def measure_gait_union_geometry(
    *,
    asset_id: str,
    renderer_root: str,
    local_forward_axis: str,
    frame_start: int,
    frame_end_exclusive: int,
) -> tuple[dict[str, Any], dict[int, dict[str, list[float]]]]:
    """Measure every integer pose in one complete stationary gait cycle."""

    if frame_end_exclusive <= frame_start:
        raise ValueError(
            f"Invalid gait frame range [{frame_start}, {frame_end_exclusive})"
        )
    root = bpy.data.objects.get(renderer_root)
    if root is None:
        raise RuntimeError(f"Renderer root {renderer_root!r} is missing")
    meshes = render_visible_meshes(root)
    scene = bpy.context.scene
    original_frame = scene.frame_current
    per_frame: dict[int, dict[str, list[float]]] = {}
    evaluated_vertex_samples = 0
    try:
        for frame in range(frame_start, frame_end_exclusive):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            bounds, vertex_count = evaluated_frame_local_bounds(meshes, root)
            per_frame[frame] = bounds
            evaluated_vertex_samples += vertex_count
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()
    geometry = make_pedestrian_geometry(
        asset_id=asset_id,
        local_bounds_m=union_local_bounds(per_frame.values()),
        local_forward_axis=local_forward_axis,
        frame_start=frame_start,
        frame_end_exclusive=frame_end_exclusive,
        measured_mesh_objects=[mesh.name for mesh in meshes],
        evaluated_vertex_samples=evaluated_vertex_samples,
        blender_version=bpy.app.version_string,
    )
    return geometry, per_frame


def geometry_from_sidecar(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        asset_id = str(metadata["id"])
        renderer_root = str(metadata["renderer_root"])
        local_forward_axis = str(metadata["local_forward_axis"])
        frame_start = int(metadata["animation_frame_start"])
        frame_count = int(metadata["walk_cycle_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Sidecar must contain id, renderer_root, local_forward_axis, "
            "animation_frame_start, and walk_cycle_frames"
        ) from exc
    geometry, _per_frame = measure_gait_union_geometry(
        asset_id=asset_id,
        renderer_root=renderer_root,
        local_forward_axis=local_forward_axis,
        frame_start=frame_start,
        frame_end_exclusive=frame_start + frame_count,
    )
    return geometry


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selected(paths: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    if not patterns:
        return list(paths)
    return [
        path
        for path in paths
        if any(fnmatch.fnmatchcase(path.stem, pattern) for pattern in patterns)
    ]


def main() -> None:
    args = parse_args()
    if not args.asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {args.asset_dir}")
    assets = _selected(
        sorted(path for path in args.asset_dir.glob("*.blend") if path.is_file()),
        args.asset,
    )
    if not assets:
        raise FileNotFoundError("No matching pedestrian .blend assets found")

    failures = []
    for blend_path in assets:
        sidecar_path = blend_path.with_suffix(".json")
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
            bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
            measured = geometry_from_sidecar(metadata)
            if args.write_sidecars:
                metadata["actor_geometry"] = measured
                _atomic_json(sidecar_path, metadata)
                status = "UPDATED"
            else:
                recorded = normalize_pedestrian_geometry(
                    metadata.get("actor_geometry"),
                    expected_asset_id=str(metadata.get("id", "")),
                )
                if recorded != measured:
                    raise ValueError(
                        "recorded actor_geometry differs from a fresh full-gait "
                        "measurement"
                    )
                status = "VALID"
            print(
                f"{status} {blend_path.name}: "
                f"bounds={measured['local_bounds_m']}",
                flush=True,
            )
        except Exception as exc:
            failures.append((str(blend_path), str(exc)))
            print(f"INVALID {blend_path}: {exc}", file=sys.stderr, flush=True)
    print(
        "PEDESTRIAN_GEOMETRY_SUMMARY="
        + json.dumps(
            {
                "asset_count": len(assets),
                "failure_count": len(failures),
                "write_sidecars": bool(args.write_sidecars),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
