"""Full-gait pedestrian bounds measured in Blender (``measure_gait_union_geometry``).

Used by the Rocketbox asset builders (``create_rocketbox_assets.py``, ``create_walk_cycle.py``), which record the
measurement in each asset's sidecar; it runs inside Blender's Python.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Sequence

import bpy


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reactbench.scoring.pedestrian_geometry import (  # noqa: E402
    make_pedestrian_geometry,
    union_local_bounds,
)


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
