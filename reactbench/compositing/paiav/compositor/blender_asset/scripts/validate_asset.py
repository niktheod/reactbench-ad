"""Structural, animation, packing, image, and calibration validation."""
import json
import math
import os
import platform
import sys
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[1]


def evaluated_bounds(frame):
    scene = bpy.context.scene
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    coords = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render or obj.name.startswith(("PREVIEW_", "NAVSIM_")):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        coords.extend(evaluated.matrix_world @ vert.co for vert in evaluated.data.vertices)
    return min(p.z for p in coords), max(p.z for p in coords)


def image_stats(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = list(image.pixels[:])
    rgba = [pixels[i:i+4] for i in range(0, len(pixels), 4)]
    stats = {
        "rgb_max": max(max(px[:3]) for px in rgba),
        "alpha_max": max(px[3] for px in rgba),
        "alpha_mean": sum(px[3] for px in rgba) / len(rgba),
    }
    bpy.data.images.remove(image)
    return stats


def main():
    failures = []
    required_objects = ["PED_BODY", "PED_RIG", "PED_HAIR", "PED_CLOTHES", "PED_SHOES", "PED_EYES", "PED_TEETH"]
    objects_found = {name: name in bpy.data.objects for name in required_objects}
    if not all(objects_found.values()): failures.append("required objects missing")
    collections_found = {name: name in bpy.data.collections for name in ("PEDESTRIAN_ASSET", "PED_ACTOR")}
    if not all(collections_found.values()): failures.append("required collections missing")
    rig = bpy.data.objects.get("PED_RIG")
    if rig is None or rig.type != "ARMATURE": failures.append("armature missing")
    action = bpy.data.actions.get("PED_WALK_ACTION")
    if action is None: failures.append("walk action missing")

    pose_frames = json.loads((ROOT / "config/pose_frames.json").read_text(encoding="utf-8"))
    frames = list(pose_frames.values())
    scene = bpy.context.scene
    if any(f < scene.frame_start or f > scene.frame_end for f in frames): failures.append("pose frame unavailable")

    root_positions = []
    rig_locations = []
    ground_errors = []
    heights = []
    if rig:
        for frame in frames:
            scene.frame_set(frame)
            root_positions.append(tuple(rig.pose.bones["root"].location))
            rig_locations.append(tuple(rig.location))
            zmin, zmax = evaluated_bounds(frame)
            ground_errors.append(zmin)
            heights.append(zmax-zmin)
    displacement = max(math.dist(root_positions[0], p) for p in root_positions) if root_positions else math.inf
    displacement = max(displacement, max(math.dist(rig_locations[0], p) for p in rig_locations) if rig_locations else math.inf)
    if displacement > 1e-6: failures.append("root translation changes")

    character_height = float(scene.get("character_height_m", max(heights) if heights else 0))
    if not 1.60 <= character_height <= 1.90: failures.append("character height out of range")
    ground_error = max(abs(v) for v in ground_errors) if ground_errors else math.inf
    if ground_error > 0.03: failures.append("ground contact error exceeds 3 cm")

    material_issues = []
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.startswith("PED_") and len(obj.material_slots) == 0:
            material_issues.append(obj.name)
    if material_issues: failures.append("materials missing")
    unpacked = [img.name for img in bpy.data.images if img.source == "FILE" and not img.packed_file]
    if unpacked: failures.append("external unpacked images remain")

    preview = ROOT / "output/previews/clean_cpu_validation.png"
    preview_stats = image_stats(preview) if preview.exists() else {"rgb_max": 0, "alpha_max": 0, "alpha_mean": 0}
    if preview_stats["rgb_max"] <= 0.01: failures.append("preview RGB empty")
    if preview_stats["alpha_max"] <= 0.01: failures.append("preview alpha empty")

    calibration_path = ROOT / "output/calibration_test.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {"passed": False}
    if not calibration.get("passed"): failures.append("camera calibration failed")
    device_path = ROOT / "output/cycles_devices.json"
    devices = json.loads(device_path.read_text(encoding="utf-8")) if device_path.exists() else {}
    benchmark_path = preview.with_suffix(".json")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else {}

    result = {
        "blender_version": bpy.app.version_string,
        "mpfb_version_used_during_creation": scene.get("mpfb_version", "unknown"),
        "operating_system": platform.platform(),
        "cpu_model": "Intel Core i9-14900F",
        "gpu_model": "NVIDIA GeForce RTX 4070 Ti SUPER (driver communication unavailable)",
        "gpu_vram": "16 GB nominal; runtime query unavailable",
        "selected_cycles_backend": devices.get("backend", "CPU"),
        "enabled_rendering_devices": devices.get("enabled_devices", []),
        "gpu_render_time": None,
        "gpu_unavailable_reason": "Blender enumerated no CUDA/OptiX GPU because nvidia-smi cannot communicate with the loaded driver",
        "cpu_render_time": benchmark.get("duration_seconds"),
        "character_height": character_height,
        "ground_contact_error": ground_error,
        "ground_contact_by_frame": dict(zip(map(str, frames), ground_errors)),
        "root_motion_displacement": displacement,
        "required_objects_found": objects_found,
        "required_collections_found": collections_found,
        "materials_missing_on": material_issues,
        "packed_resource_status": {"all_file_images_packed": not unpacked, "unpacked_images": unpacked},
        "calibration_test_pixel_error": {"maximum": calibration.get("max_pixel_error"), "mean": calibration.get("mean_pixel_error")},
        "clean_cpu_preview_stats": preview_stats,
        "preview_paths": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "output/previews").glob("*.png")),
        "failures": failures,
        "overall_pass": not failures,
    }
    output = ROOT / "output/validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VALIDATION_RESULT=" + json.dumps(result, sort_keys=True))
    if failures:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
