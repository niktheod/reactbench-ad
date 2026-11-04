"""Render several linked copies of one pedestrian into one transparent image.

The opened .blend file supplies the pedestrian asset.  Each JSONL job supplies
one calibrated camera plus a list of actor transforms.  The script adds no
ground receiver, Sun, shadow pass, depth pass, or scene-derived correction.
Materials are preserved and illuminated only by a fixed neutral world.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Matrix
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CV_TO_BLENDER = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
HELPER_PREFIX = "NAVSIM_MULTI_TRAJECTORY_"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
    )
    return parser.parse_args(argv)


def load_device_module():
    path = ROOT / "scripts/configure_cycles_device.py"
    spec = importlib.util.spec_from_file_location(
        "cycles_config", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_camera(camera, intrinsic, width: int, height: int) -> None:
    fx, fy = float(intrinsic[0][0]), float(intrinsic[1][1])
    cx, cy = float(intrinsic[0][2]), float(intrinsic[1][2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = fx * camera.data.sensor_width / width
    camera.data.shift_x = (width * 0.5 - cx) / width
    camera.data.shift_y = (cy - height * 0.5) * (fx / fy) / width
    camera.data.clip_start = 0.05
    camera.data.clip_end = 500.0
    scene = bpy.context.scene
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy


def actor_collection_and_root():
    root = bpy.data.objects.get("PED_RIG")
    if root is None:
        raise RuntimeError("PED_RIG is missing")
    for name in ("PED_ACTOR", "PEDESTRIAN_ASSET"):
        collection = bpy.data.collections.get(name)
        if collection is not None and root in set(collection.all_objects):
            objects = list(collection.all_objects)
            if objects:
                return collection, root, objects
    raise RuntimeError(
        "PED_RIG is not contained in PED_ACTOR or PEDESTRIAN_ASSET"
    )


def remap_object_references(duplicate, mapping) -> None:
    for modifier in duplicate.modifiers:
        target = getattr(modifier, "object", None)
        if target in mapping:
            modifier.object = mapping[target]
    for constraint in duplicate.constraints:
        target = getattr(constraint, "target", None)
        if target in mapping:
            constraint.target = mapping[target]
    if duplicate.pose is not None:
        for pose_bone in duplicate.pose.bones:
            for constraint in pose_bone.constraints:
                target = getattr(constraint, "target", None)
                if target in mapping:
                    constraint.target = mapping[target]


def duplicate_actor(source_objects, source_root, index: int) -> dict:
    scene = bpy.context.scene
    collection = bpy.data.collections.new(
        f"{HELPER_PREFIX}ACTOR_{index:02d}"
    )
    scene.collection.children.link(collection)
    mapping = {}
    for original in source_objects:
        duplicate = original.copy()
        duplicate.name = (
            f"{HELPER_PREFIX}{index:02d}_{original.name}"
        )
        duplicate.hide_render = True
        duplicate.hide_viewport = False
        if hasattr(duplicate, "visible_shadow"):
            duplicate.visible_shadow = False
        collection.objects.link(duplicate)
        mapping[original] = duplicate

    for original, duplicate in mapping.items():
        duplicate.parent = mapping.get(original.parent)
        duplicate.matrix_parent_inverse = (
            original.matrix_parent_inverse.copy()
        )
        remap_object_references(duplicate, mapping)
    root = mapping.get(source_root)
    if root is None:
        raise RuntimeError("duplicated actor does not contain PED_RIG")
    return {
        "collection": collection,
        "objects": list(mapping.values()),
        "root": root,
        # Generated assets can carry a small static root offset that places
        # the shoe soles exactly on local z=0. Apply placement around that
        # authored transform instead of replacing it.
        "rest_matrix": source_root.matrix_world.copy(),
    }


def create_instances(count: int) -> list:
    _collection, source_root, source_objects = actor_collection_and_root()
    for obj in bpy.context.scene.objects:
        obj.hide_render = True
    return [
        duplicate_actor(source_objects, source_root, index)
        for index in range(count)
    ]


def create_camera():
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new(
        f"{HELPER_PREFIX}CAMERA_DATA"
    )
    camera = bpy.data.objects.new(
        f"{HELPER_PREFIX}CAMERA", camera_data
    )
    scene.collection.objects.link(camera)
    camera.hide_render = False
    scene.camera = camera
    return camera


def configure_fixed_world(strength: float) -> None:
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new(
        f"{HELPER_PREFIX}WORLD"
    )
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    background = nodes.get("Background")
    if background is None:
        background = nodes.new("ShaderNodeBackground")
        output = nodes.get("World Output")
        if output is None:
            output = nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(
            background.outputs["Background"],
            output.inputs["Surface"],
        )
    background.inputs["Color"].default_value = (
        0.8,
        0.8,
        0.8,
        1.0,
    )
    background.inputs["Strength"].default_value = float(strength)
    scene.view_settings.exposure = 0.0
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass


def set_instance_enabled(instance: dict, enabled: bool) -> None:
    for obj in instance["objects"]:
        obj.hide_render = not enabled


def configure_output(scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.use_nodes = False


def configure_render_region(scene, job: dict, width: int, height: int) -> list:
    """Configure the optional experimental ideal-image border.

    Even with ``use_crop_to_border`` disabled, Cycles border rendering has
    produced small RGB differences from full-frame rendering in equivalence
    tests. Production manifests therefore request the full frame by default.
    When explicitly enabled, the coordinator losslessly crops the full-sized
    encoded PNG after Blender exits.
    """
    region = [int(value) for value in job["render_region"]]
    if len(region) != 4:
        raise ValueError(
            f"render_region must contain four coordinates, got {region}"
        )
    x_min, y_min, x_max, y_max = region
    if not (
        0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    ):
        raise ValueError(
            f"render_region {region} is outside the "
            f"{width}x{height} ideal canvas"
        )
    if region == [0, 0, width, height]:
        scene.render.use_border = False
        scene.render.use_crop_to_border = False
        return region

    scene.render.use_border = True
    scene.render.use_crop_to_border = False
    scene.render.border_min_x = x_min / width
    scene.render.border_max_x = x_max / width
    scene.render.border_min_y = 1.0 - y_max / height
    scene.render.border_max_y = 1.0 - y_min / height
    return region


def alpha_touches_artificial_crop_boundary(
    output: Path,
    region: list,
    width: int,
    height: int,
    guard_px: int = 2,
) -> bool:
    """Detect an underestimated experimental border for full-frame fallback."""
    x_min, y_min, x_max, y_max = region
    if region == [0, 0, width, height]:
        return False
    image = bpy.data.images.load(str(output), check_existing=False)
    try:
        stored_width, stored_height = [int(value) for value in image.size]
        if (stored_width, stored_height) != (width, height):
            return True
        channels = int(image.channels)
        if channels < 4:
            return True
        pixels = np.empty(
            stored_width * stored_height * channels, dtype=np.float32
        )
        image.pixels.foreach_get(pixels)
        alpha = pixels.reshape(
            stored_height, stored_width, channels
        )[..., 3]
        guard_x = min(int(guard_px), x_max - x_min)
        guard_y = min(int(guard_px), y_max - y_min)
        boundary_values = []
        if x_min > 0:
            boundary_values.append(
                alpha[:, x_min : x_min + guard_x]
            )
        if x_max < width:
            boundary_values.append(
                alpha[:, x_max - guard_x : x_max]
            )
        # Blender image buffers are bottom-up. Convert the top-left manifest
        # coordinates to row ranges immediately inside each render boundary.
        if y_max < height:
            lower_row = height - y_max
            boundary_values.append(
                alpha[lower_row : lower_row + guard_y, :]
            )
        if y_min > 0:
            upper_row = height - y_min
            boundary_values.append(
                alpha[upper_row - guard_y : upper_row, :]
            )
        return any(np.any(values > 1e-6) for values in boundary_values)
    finally:
        bpy.data.images.remove(image)


def render_job(job: dict, camera, instances: list) -> None:
    actors = list(job["actors"])
    if len(actors) > len(instances):
        raise ValueError(
            f"job contains {len(actors)} actors but only "
            f"{len(instances)} instances were created"
        )
    scene = bpy.context.scene
    width, height = [int(value) for value in job["resolution"]]
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid render resolution {width}x{height}")
    intrinsic = [
        [float(value) for value in row]
        for row in job["camera_intrinsic_matrix"]
    ]
    camera.matrix_world = (
        Matrix(job["camera_to_world"]) @ CV_TO_BLENDER
    )
    configure_camera(camera, intrinsic, width, height)
    configure_fixed_world(float(job["world_strength"]))

    scale = float(job["actor_scale"])
    if scale <= 0.0:
        raise ValueError("actor_scale must be positive")
    scale_matrix = Matrix.Diagonal((scale, scale, scale, 1.0))
    for index, instance in enumerate(instances):
        enabled = index < len(actors)
        set_instance_enabled(instance, enabled)
        if not enabled:
            continue
        transform = Matrix(actors[index]["root_to_world"])
        instance["root"].matrix_world = (
            transform @ instance["rest_matrix"] @ scale_matrix
        )

    scene.frame_set(int(job["animation_frame"]))
    bpy.context.view_layer.update()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    requested_region = configure_render_region(scene, job, width, height)
    scene.render.use_motion_blur = False
    scene.cycles.samples = int(job["samples"])
    output = Path(job["output_path"]).resolve()
    configure_output(scene, output)
    bpy.ops.render.render(write_still=True)
    if not output.is_file():
        raise RuntimeError(f"actor render was not written: {output}")
    full_region = [0, 0, width, height]
    rendered_region = requested_region
    crop_fallback_reason = None
    if alpha_touches_artificial_crop_boundary(
        output, requested_region, width, height
    ):
        crop_fallback_reason = "alpha_touched_crop_boundary"
        scene.render.use_border = False
        scene.render.use_crop_to_border = False
        bpy.ops.render.render(write_still=True)
        if not output.is_file():
            raise RuntimeError(
                f"full-frame crop fallback was not written: {output}"
            )
        rendered_region = full_region
    pending_storage_crop_region = (
        requested_region
        if rendered_region != full_region
        else None
    )

    info_path = Path(job["render_info_path"]).resolve()
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        json.dumps(
            {
                "cache_key": job["cache_key"],
                "pipeline_version": job["pipeline_version"],
                "job_id": job["job_id"],
                "asset_id": job["asset_id"],
                "actor_count": len(actors),
                "actor_ids": [
                    str(actor["actor_id"]) for actor in actors
                ],
                "animation_frame": int(job["animation_frame"]),
                "resolution": [width, height],
                "requested_render_region": requested_region,
                "rendered_region": rendered_region,
                "stored_render_region": full_region,
                "pending_storage_crop_region": (
                    pending_storage_crop_region
                ),
                "raw_render_storage": job["raw_render_storage"],
                "crop_fallback_reason": crop_fallback_reason,
                "samples": int(job["samples"]),
                "world_strength": float(job["world_strength"]),
                "shadows": "disabled",
                "receiver_geometry": False,
                "output_path": str(output),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    jobs = [
        json.loads(line)
        for line in Path(args.jobs).read_text().splitlines()
        if line.strip()
    ]
    if not jobs:
        print(
            "BATCH_RENDER_SUMMARY="
            + json.dumps({"completed": 0, "failed": 0})
        )
        return

    maximum_actor_count = max(len(job["actors"]) for job in jobs)
    if maximum_actor_count < 1:
        raise RuntimeError("render batch contains no actor instances")
    device = load_device_module().configure(args.device)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = (
        "CPU" if device["backend"] == "CPU" else "GPU"
    )
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True

    instances = create_instances(maximum_actor_count)
    camera = create_camera()
    completed = 0
    failures = []
    for index, job in enumerate(jobs):
        try:
            render_job(job, camera, instances)
            completed += 1
            print(
                f"BATCH_JOB_DONE={index}:{job.get('job_id', '')}"
            )
            sys.stdout.flush()
        except Exception as exc:
            failures.append(
                {
                    "job_id": job.get("job_id"),
                    "error": str(exc),
                }
            )
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()

    print(
        "BATCH_RENDER_SUMMARY="
        + json.dumps(
            {
                "completed": completed,
                "failed": len(failures),
                "failures": failures,
            }
        )
    )
    if failures:
        os._exit(2)


if __name__ == "__main__":
    main()
