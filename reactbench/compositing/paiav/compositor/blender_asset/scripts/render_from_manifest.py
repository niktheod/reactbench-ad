"""Render the packed pedestrian using a calibration/placement JSON manifest.

OpenCV camera axes (+x right, +y down, +z forward) are converted to Blender
camera axes (+x right, +y up, -z forward) with diag(1,-1,-1,1).
Lens distortion is deliberately not modeled; distort all output passes later.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Matrix


ROOT = Path(__file__).resolve().parents[1]
CV_TO_BLENDER = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    return p.parse_args(argv)


def configure_camera_from_k(camera, K, width, height):
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = fx * camera.data.sensor_width / width
    scene = bpy.context.scene
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy
    camera.data.shift_x = (width * 0.5 - cx) / width
    camera.data.shift_y = (cy - height * 0.5) * (fx / fy) / width


def camera_to_world(manifest):
    if "camera_to_world" in manifest:
        return Matrix(manifest["camera_to_world"])
    if "camera_to_lidar" in manifest:
        c2l = Matrix(manifest["camera_to_lidar"])
        l2w = Matrix(manifest.get("lidar_to_world", Matrix.Identity(4)))
        return l2w @ c2l
    raise ValueError("Manifest needs camera_to_world or camera_to_lidar")


def add_stage(manifest):
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.003))
    ground = bpy.context.object
    ground.name = "NAVSIM_GROUND_SHADOW_CATCHER"
    ground.is_shadow_catcher = True
    ground.pass_index = 0
    mat = bpy.data.materials.new("NAVSIM_GROUND_MATERIAL")
    mat.diffuse_color = (0.18, 0.18, 0.18, 1)
    ground.data.materials.append(mat)

    sun_data = bpy.data.lights.new("NAVSIM_DAYLIGHT", "SUN")
    sun = bpy.data.objects.new("NAVSIM_DAYLIGHT", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = tuple(math.radians(v) for v in manifest.get("sun_rotation_degrees", [35, -20, -30]))
    sun_data.energy = float(manifest.get("sun_energy", 2.0))
    sun_data.angle = math.radians(6.0)


def setup_outputs(paths):
    scene = bpy.context.scene
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_object_index = True
    if hasattr(layer, "cycles"):
        layer.cycles.use_pass_shadow_catcher = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    links = scene.node_tree.links
    rl = nodes.new("CompositorNodeRLayers")
    comp = nodes.new("CompositorNodeComposite")
    links.new(rl.outputs["Image"], comp.inputs["Image"])

    generated = []
    specs = (("Alpha", "alpha", "PNG"), ("Depth", "depth", "OPEN_EXR"),
             ("IndexOB", "object_mask", "OPEN_EXR"), ("Shadow Catcher", "shadow", "OPEN_EXR"))
    for socket, key, file_format in specs:
        if key not in paths or socket not in rl.outputs:
            continue
        target = Path(paths[key]).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        node = nodes.new("CompositorNodeOutputFile")
        node.base_path = str(target.parent)
        node.file_slots[0].path = target.stem + "_"
        node.format.file_format = file_format
        node.format.color_mode = "BW" if key != "shadow" else "RGBA"
        links.new(rl.outputs[socket], node.inputs[0])
        generated.append((target, file_format))
    return generated


def normalize_compositor_names(generated, frame):
    for target, file_format in generated:
        ext = ".png" if file_format == "PNG" else ".exr"
        candidate = target.parent / f"{target.stem}_{frame:04d}{ext}"
        if candidate.exists() and candidate != target:
            os.replace(candidate, target)


def main():
    a = args()
    with open(a.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    width, height = manifest["resolution"]
    K = manifest["camera_intrinsic_matrix"]
    frame = int(manifest["animation_frame"])

    rig = bpy.data.objects.get("PED_RIG")
    if rig is None:
        raise RuntimeError("PED_RIG is missing")
    # The packed asset's rest location contains a small vertical offset which
    # places its shoe soles exactly at z=0. Preserve it when positioning the
    # actor at the manifest's ground-contact point.
    rest_location = rig.location.copy()
    position = manifest.get("actor_position", [0, 0, 0])
    rig.location = (
        float(position[0]) + rest_location.x,
        float(position[1]) + rest_location.y,
        float(position[2]) + rest_location.z,
    )
    rig.rotation_euler[2] = float(manifest.get("actor_heading", 0.0))
    scale = float(manifest.get("actor_scale", 1.0))
    rig.scale = (scale, scale, scale)

    cam_data = bpy.data.cameras.new("NAVSIM_CAMERA_DATA")
    cam = bpy.data.objects.new("NAVSIM_CAMERA", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.matrix_world = camera_to_world(manifest) @ CV_TO_BLENDER
    configure_camera_from_k(cam, K, width, height)
    bpy.context.scene.camera = cam
    add_stage(manifest)

    device_path = ROOT / "scripts/configure_cycles_device.py"
    spec = importlib.util.spec_from_file_location("cycles_config", device_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = module.configure(manifest.get("rendering_device", "auto"))

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU" if device["backend"] == "CPU" else "GPU"
    scene.cycles.samples = int(manifest.get("samples", 128))
    scene.cycles.use_denoising = True
    scene.render.use_motion_blur = bool(manifest.get("motion_blur", False))
    scene.render.film_transparent = True
    scene.render.resolution_x, scene.render.resolution_y = int(width), int(height)
    scene.render.resolution_percentage = 100
    scene.frame_set(frame)

    paths = manifest["output_paths"]
    rgb = Path(paths["rgb"]).resolve()
    rgb.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(rgb)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    generated = setup_outputs(paths)
    bpy.ops.render.render(write_still=True)
    normalize_compositor_names(generated, frame)
    print("MANIFEST_RENDER=" + json.dumps({"rgb": str(rgb), "frame": frame, "device": device}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(2)
