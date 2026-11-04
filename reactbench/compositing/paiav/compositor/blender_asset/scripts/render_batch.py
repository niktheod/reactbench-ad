"""Render multiple calibrated actor views while loading Blender once.

Input is a JSONL file with one manifest per line. Each completed render writes
its own cache-info JSON, allowing the coordinator to resume interrupted runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector


ROOT = Path(__file__).resolve().parents[1]
CV_TO_BLENDER = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
STANDING_POSE_VERSION = "procedural_natural_standing_v2"
BIP_PREFIX_RE = re.compile(r"^Bip\d+\s*")
STANDING_ARM_OUTWARD_COMPONENT = 0.12
STANDING_LEG_OUTWARD_COMPONENT = 0.02


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    return parser.parse_args(argv)


def load_device_module():
    path = ROOT / "scripts/configure_cycles_device.py"
    spec = importlib.util.spec_from_file_location("cycles_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_camera(camera, intrinsic, width, height):
    fx, fy = float(intrinsic[0][0]), float(intrinsic[1][1])
    cx, cy = float(intrinsic[0][2]), float(intrinsic[1][2])
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = fx * camera.data.sensor_width / width
    camera.data.shift_x = (width * 0.5 - cx) / width
    camera.data.shift_y = (cy - height * 0.5) * (fx / fy) / width
    scene = bpy.context.scene
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy


def create_stage():
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("NAVSIM_BATCH_CAMERA_DATA")
    camera = bpy.data.objects.new("NAVSIM_BATCH_CAMERA", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    sun_data = bpy.data.lights.new("NAVSIM_DAYLIGHT", "SUN")
    sun = bpy.data.objects.new("NAVSIM_DAYLIGHT", sun_data)
    scene.collection.objects.link(sun)

    stage_collection = bpy.data.collections.new("NAVSIM_SHADOW_STAGE")
    scene.collection.children.link(stage_collection)
    ground_mesh = bpy.data.meshes.new("NAVSIM_SHADOW_GROUND_MESH")
    ground = bpy.data.objects.new("NAVSIM_SHADOW_GROUND", ground_mesh)
    stage_collection.objects.link(ground)
    vertices = [(-100.0, -100.0, -0.004), (100.0, -100.0, -0.004),
                (100.0, 100.0, -0.004), (-100.0, 100.0, -0.004)]
    ground_mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    ground_mesh.update()
    ground.is_shadow_catcher = True
    material = bpy.data.materials.new("NAVSIM_SHADOW_GROUND_MATERIAL")
    material.diffuse_color = (0.18, 0.18, 0.18, 1.0)
    ground.data.materials.append(material)

    actor_layer = scene.view_layers[0]
    actor_layer.name = "NAVSIM_ACTOR_LAYER"
    actor_layer.use_pass_z = True
    actor_layer.layer_collection.children[stage_collection.name].exclude = True
    shadow_layer = scene.view_layers.new("NAVSIM_SHADOW_LAYER")
    if hasattr(shadow_layer, "cycles"):
        shadow_layer.cycles.use_pass_shadow_catcher = True
    return camera, sun, actor_layer, shadow_layer


def configure_compositor(actor_layer, shadow_layer):
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    links = scene.node_tree.links
    actor_render = nodes.new("CompositorNodeRLayers")
    actor_render.layer = actor_layer.name
    composite = nodes.new("CompositorNodeComposite")
    links.new(actor_render.outputs["Image"], composite.inputs["Image"])

    # Keep the current color-managed transparent actor PNG for the downstream
    # image pipeline. All required products are emitted by explicit File Output
    # nodes; the scene's primary output is not written.
    actor_output = nodes.new("CompositorNodeOutputFile")
    actor_output.name = "NAVSIM_ACTOR_OUTPUT"
    actor_output.format.file_format = "PNG"
    actor_output.format.color_mode = "RGBA"
    actor_output.format.color_depth = "8"
    actor_output.file_slots[0].save_as_render = True
    links.new(actor_render.outputs["Image"], actor_output.inputs[0])

    # The scene-aware compositor needs additive lighting ingredients, not
    # display-referred PNGs.  This output is enabled only for the sky-only and
    # Sun-only renders below.  ``save_as_render = False`` preserves the
    # scene-linear values instead of applying the AgX view transform.
    actor_linear_output = nodes.new("CompositorNodeOutputFile")
    actor_linear_output.name = "NAVSIM_ACTOR_LINEAR_OUTPUT"
    actor_linear_output.format.file_format = "OPEN_EXR"
    actor_linear_output.format.color_mode = "RGBA"
    actor_linear_output.format.color_depth = "16"
    actor_linear_output.format.exr_codec = "ZIP"
    actor_linear_output.file_slots[0].save_as_render = False
    actor_linear_output.mute = True
    links.new(actor_render.outputs["Image"], actor_linear_output.inputs[0])

    # Store metric actor distance in the PNG alpha channel.  RGB channels are
    # color-managed, whereas alpha remains linear.  Encoding 0..100 m into a
    # 16-bit alpha channel gives enough precision for per-pixel occlusion while
    # retaining a portable format readable by OpenCV in the coordinator.
    depth_scale = nodes.new("CompositorNodeMath")
    depth_scale.operation = "DIVIDE"
    depth_scale.inputs[1].default_value = 100.0
    links.new(actor_render.outputs["Depth"], depth_scale.inputs[0])
    depth_clamp = nodes.new("CompositorNodeMath")
    depth_clamp.operation = "MINIMUM"
    depth_clamp.inputs[1].default_value = 1.0
    links.new(depth_scale.outputs[0], depth_clamp.inputs[0])
    depth_rgba = nodes.new("CompositorNodeSetAlpha")
    depth_rgba.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(depth_clamp.outputs[0], depth_rgba.inputs[1])
    depth_output = nodes.new("CompositorNodeOutputFile")
    depth_output.name = "NAVSIM_ACTOR_DEPTH_OUTPUT"
    depth_output.format.file_format = "PNG"
    depth_output.format.color_mode = "RGBA"
    depth_output.format.color_depth = "16"
    links.new(depth_rgba.outputs[0], depth_output.inputs[0])

    shadow_render = nodes.new("CompositorNodeRLayers")
    shadow_render.layer = shadow_layer.name
    if "Shadow Catcher" not in shadow_render.outputs:
        raise RuntimeError("Cycles Shadow Catcher pass is unavailable")
    shadow_output = nodes.new("CompositorNodeOutputFile")
    shadow_output.name = "NAVSIM_SHADOW_OUTPUT"
    shadow_output.format.file_format = "OPEN_EXR"
    shadow_output.format.color_mode = "RGBA"
    shadow_output.format.color_depth = "16"
    shadow_output.format.exr_codec = "ZIP"
    shadow_output.file_slots[0].save_as_render = False
    links.new(shadow_render.outputs["Shadow Catcher"], shadow_output.inputs[0])
    return actor_output, actor_linear_output, shadow_output, depth_output


def normalize_output_file(target, frame):
    candidate = target.parent / f"{target.stem}_{frame:04d}{target.suffix}"
    if candidate.exists() and candidate != target:
        os.replace(candidate, target)


def configure_file_output(node, target):
    """Point a compositor File Output node at one exact per-job target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    node.base_path = str(target.parent)
    node.file_slots[0].path = target.stem + "_"


def configure_world(lighting):
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("NAVSIM_WORLD")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    color = lighting["world_color"]
    background.inputs["Color"].default_value = (*[float(v) for v in color], 1.0)
    background.inputs["Strength"].default_value = float(lighting["world_strength"])
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = float(lighting["exposure"])


def configure_sun(sun, lighting):
    direction = Vector(lighting["sun_ray_direction"])
    if direction.length < 1e-6:
        raise ValueError("sun_ray_direction must be nonzero")
    direction.normalize()
    sun.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    sun.data.energy = float(lighting["sun_energy"])
    sun.data.angle = math.radians(float(lighting["sun_angle_degrees"]))


def configure_render_region(scene, job, width, height):
    """Limit Cycles sampling and physically store only the sampled rectangle."""
    region = [int(value) for value in job["render_region"]]
    if len(region) != 4:
        raise ValueError(f"render_region must contain four coordinates, got {region}")
    x_min, y_min, x_max, y_max = region
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            f"render_region {region} is outside the {width}x{height} ideal canvas"
        )
    if region == [0, 0, width, height]:
        scene.render.use_border = False
        scene.render.use_crop_to_border = False
        return region

    scene.render.use_border = True
    scene.render.use_crop_to_border = True
    scene.render.border_min_x = x_min / width
    scene.render.border_max_x = x_max / width
    # Manifest coordinates start at the top-left; Blender's normalized render
    # border starts at the bottom-left.
    scene.render.border_min_y = 1.0 - y_max / height
    scene.render.border_max_y = 1.0 - y_min / height
    return region


def normalized_group_name(name):
    return " ".join(
        str(name)
        .lower()
        .replace(".", " ")
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def foot_vertex_group_indices(mesh_object, side):
    side_token = "l" if side == "left" else "r"
    selected = []
    for group in mesh_object.vertex_groups:
        tokens = normalized_group_name(group.name).split()
        if side_token not in tokens:
            continue
        if any(token.startswith("foot") or token.startswith("toe") for token in tokens):
            selected.append((group.index, group.name))
    return selected


def project_world_points(scene, camera, points, width, height):
    projected = []
    for point in points:
        normalized = world_to_camera_view(scene, camera, point)
        if normalized.z <= 0.0:
            continue
        x = float(normalized.x * width)
        y = float((1.0 - normalized.y) * height)
        if -2.0 <= x <= width + 2.0 and -2.0 <= y <= height + 2.0:
            projected.append([x, y])
    return projected


def collect_foot_contact_geometry(scene, camera, width, height, job):
    """Measure animated sole-to-receiver contact without asset annotations.

    Rocketbox assets already carry Foot/Toe deformation weights.  Evaluating
    those vertices after the animation frame and actor transform have been set
    lets us distinguish the stance sole from the airborne shoe.  Coordinates
    are projected onto the ideal (pre-render-scale) image canvas so the
    coordinator can distort the footprint with the rest of the render.
    """
    receiver_z = float(job["receiver_plane_z_m"])
    contact_distance = float(job["foot_contact_distance_m"])
    relative_distance = float(job["foot_stance_relative_distance_m"])
    sole_band = float(job["foot_sole_band_m"])
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    per_side_points = {"left": [], "right": []}
    per_side_groups = {"left": [], "right": []}

    for mesh_object in bpy.data.objects:
        if mesh_object.type != "MESH" or not mesh_object.vertex_groups:
            continue
        evaluated = mesh_object.evaluated_get(depsgraph)
        evaluated_vertices = evaluated.data.vertices
        for side in ("left", "right"):
            groups = foot_vertex_group_indices(mesh_object, side)
            if not groups:
                continue
            group_indices = {index for index, _ in groups}
            per_side_groups[side].extend(
                f"{mesh_object.name}:{name}" for _, name in groups
            )
            for vertex in mesh_object.data.vertices:
                weight = max(
                    (
                        membership.weight
                        for membership in vertex.groups
                        if membership.group in group_indices
                    ),
                    default=0.0,
                )
                if weight < 0.15 or vertex.index >= len(evaluated_vertices):
                    continue
                per_side_points[side].append(
                    evaluated.matrix_world @ evaluated_vertices[vertex.index].co
                )

    feet = {}
    for side in ("left", "right"):
        points = per_side_points[side]
        if not points:
            feet[side] = {
                "contact": False,
                "status": "foot_vertex_groups_unavailable",
                "group_names": per_side_groups[side],
                "weighted_vertex_count": 0,
                "minimum_receiver_distance_m": None,
                "ideal_contact_points_px": [],
            }
            continue
        minimum_z = min(float(point.z) for point in points)
        signed_distance = minimum_z - receiver_z
        in_contact = -contact_distance <= signed_distance <= contact_distance
        sole_points = [
            point for point in points if float(point.z) <= minimum_z + sole_band
        ]
        projected = (
            project_world_points(scene, camera, sole_points, width, height)
            if in_contact
            else []
        )
        feet[side] = {
            "contact": bool(in_contact and projected),
            "status": "measured_from_evaluated_foot_vertices",
            "group_names": sorted(set(per_side_groups[side])),
            "weighted_vertex_count": len(points),
            "minimum_world_z_m": minimum_z,
            "minimum_receiver_distance_m": signed_distance,
            "sole_band_m": sole_band,
            "ideal_contact_points_px": projected,
        }

    measured_distances = [
        abs(float(details["minimum_receiver_distance_m"]))
        for details in feet.values()
        if details.get("minimum_receiver_distance_m") is not None
    ]
    stance_reference = min(measured_distances) if measured_distances else None
    if stance_reference is not None:
        for details in feet.values():
            distance = details.get("minimum_receiver_distance_m")
            if distance is None:
                continue
            relative_contact = (
                abs(float(distance)) <= stance_reference + relative_distance
            )
            details["relative_to_lowest_sole_m"] = (
                abs(float(distance)) - stance_reference
            )
            if details["contact"] and not relative_contact:
                details["contact"] = False
                details["ideal_contact_points_px"] = []
                details["status"] = "measured_swing_foot_excluded"

    return {
        "method": "evaluated_foot_toe_vertex_groups_v1",
        "receiver_plane_z_m": receiver_z,
        "contact_distance_m": contact_distance,
        "stance_relative_distance_m": relative_distance,
        "stance_reference_distance_m": stance_reference,
        "feet": feet,
        "contact_sides": [
            side for side, details in feet.items() if details["contact"]
        ],
    }


def collect_vehicle_contact_geometry(
    scene,
    camera,
    width,
    height,
    job,
    actor_root,
):
    """Project XOMA-derived wheel contact patches onto the receiver."""
    patches = job.get("vehicle_contact_patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError("Car render has no vehicle_contact_patches")
    receiver_z = float(job["receiver_plane_z_m"])
    contact_distance = float(job["foot_contact_distance_m"])
    bpy.context.view_layer.update()
    contacts = {}
    for patch in patches:
        if not isinstance(patch, dict):
            raise ValueError("Vehicle contact patch must be an object")
        name = str(patch.get("name", "")).strip()
        raw_points = patch.get("points_actor_m")
        if not name or not isinstance(raw_points, list) or len(raw_points) < 3:
            raise ValueError(f"Malformed vehicle contact patch: {patch!r}")
        points = []
        for raw_point in raw_points:
            if not isinstance(raw_point, list) or len(raw_point) != 3:
                raise ValueError(
                    f"Malformed point in vehicle contact patch {name}"
                )
            point = Vector(tuple(float(value) for value in raw_point))
            points.append(actor_root.matrix_world @ point)
        minimum_z = min(float(point.z) for point in points)
        signed_distance = minimum_z - receiver_z
        in_contact = -contact_distance <= signed_distance <= contact_distance
        projected = (
            project_world_points(scene, camera, points, width, height)
            if in_contact
            else []
        )
        contacts[name] = {
            "contact": bool(in_contact and projected),
            "status": "measured_from_xoma_wheel_patch",
            "group_names": [],
            "weighted_vertex_count": len(points),
            "minimum_world_z_m": minimum_z,
            "minimum_receiver_distance_m": signed_distance,
            "ideal_contact_points_px": projected,
            "wheel_diameter_m": float(patch.get("wheel_diameter_m", 0.0)),
        }
    measured_distances = [
        abs(float(details["minimum_receiver_distance_m"]))
        for details in contacts.values()
    ]
    return {
        "method": "xoma_axle_wheel_contact_patches_v1",
        "contact_profile": str(job.get("contact_profile", "")),
        "receiver_plane_z_m": receiver_z,
        "contact_distance_m": contact_distance,
        "stance_relative_distance_m": 0.0,
        "stance_reference_distance_m": min(measured_distances),
        "feet": contacts,
        "contact_sides": [
            name for name, details in contacts.items() if details["contact"]
        ],
    }


def collect_actor_contact_geometry(
    scene,
    camera,
    width,
    height,
    job,
    actor_root,
):
    profile = str(job.get("contact_profile", "biped_feet"))
    if profile == "openx_vehicle_wheels":
        return collect_vehicle_contact_geometry(
            scene, camera, width, height, job, actor_root
        )
    return collect_foot_contact_geometry(scene, camera, width, height, job)


def pose_bones_by_suffix(rig):
    bones = {}
    for pose_bone in rig.pose.bones:
        suffix = BIP_PREFIX_RE.sub("", pose_bone.name)
        if suffix in bones:
            raise RuntimeError(
                f"Duplicate biped bone suffix {suffix!r} in {rig.name}"
            )
        bones[suffix] = pose_bone
    return bones


def reset_pose(rig):
    for pose_bone in rig.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = Quaternion()
        pose_bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def direction_from_down(angle_degrees, outward_component=0.0):
    angle = math.radians(float(angle_degrees))
    sagittal_scale = math.sqrt(1.0 - float(outward_component) ** 2)
    return Vector(
        (
            float(outward_component),
            math.sin(angle) * sagittal_scale,
            -math.cos(angle) * sagittal_scale,
        )
    )


def direction_from_up(lean_degrees):
    angle = math.radians(float(lean_degrees))
    return Vector((0.0, -math.sin(angle), math.cos(angle)))


def apply_armature_rotation_at_joint(pose_bone, rotation):
    desired_matrix = rotation.to_matrix().to_4x4()
    desired_matrix.translation = pose_bone.head.copy()
    pose_bone.matrix = desired_matrix
    bpy.context.view_layer.update()


def align_joint_vector(actor_rig, deform_rig, parent, child, actor_direction):
    """Aim one anatomical segment in the outer actor's local coordinates."""
    current_armature_direction = (child.head - parent.head).normalized()
    desired_world_direction = (
        actor_rig.matrix_world.to_3x3() @ actor_direction
    ).normalized()
    desired_armature_direction = (
        deform_rig.matrix_world.to_3x3().inverted() @ desired_world_direction
    ).normalized()
    correction = current_armature_direction.rotation_difference(
        desired_armature_direction
    )
    corrected_rotation = correction @ parent.matrix.to_quaternion()
    apply_armature_rotation_at_joint(parent, corrected_rotation)


def evaluated_actor_minimum_z(actor_rig):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    minimum_z = math.inf
    mesh_count = 0
    vertex_count = 0
    for mesh_object in bpy.data.objects:
        if mesh_object.type != "MESH":
            continue
        ancestor = mesh_object.parent
        belongs_to_actor = False
        while ancestor is not None:
            if ancestor == actor_rig:
                belongs_to_actor = True
                break
            ancestor = ancestor.parent
        if not belongs_to_actor:
            continue
        evaluated = mesh_object.evaluated_get(depsgraph)
        mesh_count += 1
        vertex_count += len(evaluated.data.vertices)
        for vertex in evaluated.data.vertices:
            minimum_z = min(
                minimum_z, float((evaluated.matrix_world @ vertex.co).z)
            )
    if mesh_count == 0 or vertex_count == 0 or not math.isfinite(minimum_z):
        raise RuntimeError("Standing pose has no evaluated actor mesh vertices")
    return minimum_z, mesh_count, vertex_count


def configure_natural_standing_pose(actor_rig):
    """Replace the walk action with a relaxed, grounded neutral stance."""
    deform_rig = bpy.data.objects.get("ROCKETBOX_DEFORM_RIG")
    ground_correction = bpy.data.objects.get("PED_GROUND_CORRECTION")
    if deform_rig is None or deform_rig.type != "ARMATURE":
        raise RuntimeError(
            "Standing mode requires the Rocketbox ROCKETBOX_DEFORM_RIG armature"
        )
    if ground_correction is None:
        raise RuntimeError(
            "Standing mode requires the PED_GROUND_CORRECTION object"
        )
    if deform_rig.animation_data is not None:
        deform_rig.animation_data.action = None
    if ground_correction.animation_data is not None:
        ground_correction.animation_data.action = None
    ground_correction.location.z = 0.0
    reset_pose(deform_rig)

    bones = pose_bones_by_suffix(deform_rig)
    required = {
        "Spine",
        "Spine1",
        *(
            f"{side} {part}"
            for side in ("L", "R")
            for part in (
                "Thigh",
                "Calf",
                "Foot",
                "Toe0",
                "UpperArm",
                "Forearm",
                "Hand",
            )
        ),
    }
    missing = sorted(required - bones.keys())
    if missing:
        raise RuntimeError(
            "Standing mode requires Rocketbox biped bones: " + ", ".join(missing)
        )

    actor_world_to_local = actor_rig.matrix_world.to_3x3().inverted()
    deform_to_world = deform_rig.matrix_world.to_3x3()
    neutral_foot_directions = {
        side: (
            actor_world_to_local
            @ deform_to_world
            @ (bones[f"{side} Toe0"].head - bones[f"{side} Foot"].head)
        ).normalized()
        for side in ("L", "R")
    }

    align_joint_vector(
        actor_rig,
        deform_rig,
        bones["Spine"],
        bones["Spine1"],
        direction_from_up(1.0),
    )
    arm_angles = {"L": (2.0, -8.0), "R": (-3.0, 6.0)}
    for side in ("L", "R"):
        sign = 1.0 if side == "L" else -1.0
        align_joint_vector(
            actor_rig,
            deform_rig,
            bones[f"{side} Thigh"],
            bones[f"{side} Calf"],
            direction_from_down(1.0, sign * STANDING_LEG_OUTWARD_COMPONENT),
        )
        align_joint_vector(
            actor_rig,
            deform_rig,
            bones[f"{side} Calf"],
            bones[f"{side} Foot"],
            direction_from_down(-1.0, sign * STANDING_LEG_OUTWARD_COMPONENT),
        )
        align_joint_vector(
            actor_rig,
            deform_rig,
            bones[f"{side} Foot"],
            bones[f"{side} Toe0"],
            neutral_foot_directions[side],
        )
        upper_arm, forearm = arm_angles[side]
        align_joint_vector(
            actor_rig,
            deform_rig,
            bones[f"{side} UpperArm"],
            bones[f"{side} Forearm"],
            direction_from_down(
                upper_arm, sign * STANDING_ARM_OUTWARD_COMPONENT
            ),
        )
        align_joint_vector(
            actor_rig,
            deform_rig,
            bones[f"{side} Forearm"],
            bones[f"{side} Hand"],
            direction_from_down(
                forearm, sign * STANDING_ARM_OUTWARD_COMPONENT
            ),
        )

    bpy.context.view_layer.update()
    ungrounded_minimum_z, mesh_count, vertex_count = evaluated_actor_minimum_z(
        actor_rig
    )
    actor_ground_z = float(
        (actor_rig.matrix_world @ Vector((0.0, 0.0, 0.0))).z
    )
    correction_world_z = float(
        (actor_rig.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).z
    )
    if not math.isfinite(correction_world_z) or abs(correction_world_z) < 1e-9:
        raise RuntimeError("Standing ground-correction axis is degenerate")
    ground_correction.location.z += (
        actor_ground_z - ungrounded_minimum_z
    ) / correction_world_z
    bpy.context.view_layer.update()
    grounded_minimum_z, _mesh_count, _vertex_count = evaluated_actor_minimum_z(
        actor_rig
    )
    residual = grounded_minimum_z - actor_ground_z
    if not math.isfinite(residual) or abs(residual) > 1e-5:
        raise RuntimeError(
            f"Standing pose failed ground correction: residual={residual:.9f} m"
        )
    return {
        "version": STANDING_POSE_VERSION,
        "pose_source": "runtime_procedural_rocketbox_biped",
        "walk_action_disabled": True,
        "ground_action_disabled": True,
        "ground_correction_local_z_m": float(ground_correction.location.z),
        "ground_contact_residual_m": float(residual),
        "evaluated_mesh_count": int(mesh_count),
        "evaluated_vertex_count": int(vertex_count),
    }


def render_job(
    job,
    camera,
    sun,
    rig,
    rig_rest_location,
    actor_output,
    actor_linear_output,
    shadow_output,
    depth_output,
):
    scene = bpy.context.scene
    output = Path(job["output_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    camera.matrix_world = Matrix(job["camera_to_world"]) @ CV_TO_BLENDER
    width, height = [int(v) for v in job["resolution"]]
    render_scale = int(job["render_scale"])
    render_width, render_height = width * render_scale, height * render_scale
    intrinsic = [[float(value) for value in row] for row in job["camera_intrinsic_matrix"]]
    for row in (0, 1):
        for column in range(3):
            intrinsic[row][column] *= render_scale
    configure_camera(camera, intrinsic, render_width, render_height)
    configure_world(job["lighting"])
    configure_sun(sun, job["lighting"])

    position = job["actor_position"]
    rig.location = (
        float(position[0]) + rig_rest_location.x,
        float(position[1]) + rig_rest_location.y,
        float(position[2]) + rig_rest_location.z,
    )
    rig.rotation_euler = (0.0, 0.0, float(job["actor_heading"]))
    scale = float(job["actor_scale"])
    rig.scale = (scale, scale, scale)
    scene.frame_set(int(job["animation_frame"]))
    standing_pose_diagnostics = None
    if bool(job.get("standing_pose", False)):
        requested_pose_version = str(job.get("standing_pose_version", ""))
        if requested_pose_version != STANDING_POSE_VERSION:
            raise ValueError(
                "Unsupported standing pose version: "
                f"{requested_pose_version!r}"
            )
        if bool(job["motion_blur"]):
            raise ValueError("Standing renders must disable motion blur")
        standing_pose_diagnostics = configure_natural_standing_pose(rig)

    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.resolution_percentage = 100
    render_region = configure_render_region(scene, job, width, height)
    stored_render_width = (
        int(render_region[2]) - int(render_region[0])
    ) * render_scale
    stored_render_height = (
        int(render_region[3]) - int(render_region[1])
    ) * render_scale
    foot_contact_geometry = collect_actor_contact_geometry(
        scene,
        camera,
        width,
        height,
        job,
        rig,
    )
    scene.cycles.samples = int(job["samples"])
    scene.render.use_motion_blur = bool(job["motion_blur"])
    scene.render.fps = max(1, int(round(float(job["animation_fps"]))))
    if hasattr(scene.render, "motion_blur_shutter"):
        scene.render.motion_blur_shutter = float(job["motion_blur_shutter"])
    configure_file_output(actor_output, output)
    depth_path = Path(job["actor_depth_output_path"]).resolve()
    configure_file_output(depth_output, depth_path)

    actor_sun_path = Path(job["actor_sun_output_path"]).resolve()

    frame = int(job["animation_frame"])
    world_background = scene.world.node_tree.nodes.get("Background")
    if world_background is None:
        raise RuntimeError("NAVSIM world has no Background node")
    original_world_strength = float(world_background.inputs["Strength"].default_value)
    original_sun_energy = float(sun.data.energy)
    actor_layer = scene.view_layers["NAVSIM_ACTOR_LAYER"]
    shadow_layer = scene.view_layers["NAVSIM_SHADOW_LAYER"]

    render_mode = str(job["render_mode"])
    if render_mode == "directional_only":
        required_passes = set(
            job.get("required_passes", ["actor_sun", "directional_shadow"])
        )
        produced_passes = set(job.get("produced_passes", required_passes))
        if required_passes != {"actor_sun", "directional_shadow"}:
            raise ValueError(
                f"Invalid directional required_passes: {sorted(required_passes)}"
            )
        if not required_passes <= produced_passes:
            raise ValueError("Directional produced_passes do not satisfy required_passes")
        # The VLM preview has already cached geometry, sky illumination, AO,
        # actor depth, and the display actor.  Render only the selected Sun
        # angle now, emitting both its actor contribution and Vdir in one pass.
        actor_output.mute = True
        actor_linear_output.mute = False
        shadow_output.mute = False
        depth_output.mute = True
        actor_layer.use = True
        shadow_layer.use = True
        configure_file_output(actor_linear_output, actor_sun_path)
        directional_shadow_path = Path(
            job["directional_shadow_output_path"]
        ).resolve()
        configure_file_output(shadow_output, directional_shadow_path)
        world_background.inputs["Strength"].default_value = 0.0
        sun.data.energy = original_sun_energy
        try:
            bpy.ops.render.render(write_still=False)
            normalize_output_file(actor_sun_path, frame)
            normalize_output_file(directional_shadow_path, frame)
        finally:
            sun.data.energy = original_sun_energy
            world_background.inputs["Strength"].default_value = original_world_strength
            actor_output.mute = False
            actor_linear_output.mute = True
            shadow_output.mute = False
            depth_output.mute = False
            actor_layer.use = True
            shadow_layer.use = True
        if not actor_sun_path.is_file():
            raise RuntimeError(
                f"Label-selected Sun-only actor pass was not written: {actor_sun_path}"
            )
        if not directional_shadow_path.is_file():
            raise RuntimeError(
                "Label-selected directional shadow pass was not written: "
                f"{directional_shadow_path}"
            )
        info_path = Path(job["render_info_path"]).resolve()
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(
            json.dumps(
                {
                    "cache_key": job["cache_key"],
                    "render_mode": render_mode,
                    "required_passes": sorted(required_passes),
                    "produced_passes": sorted(produced_passes),
                    "samples": int(job["samples"]),
                    "animation_frame": int(job["animation_frame"]),
                    "render_region": render_region,
                    "ideal_canvas_resolution": [width, height],
                    "full_render_output_resolution": [render_width, render_height],
                    "render_output_resolution": [
                        stored_render_width,
                        stored_render_height,
                    ],
                    "raw_render_storage": job["raw_render_storage"],
                    "actor_sun_output_path": str(actor_sun_path),
                    "directional_shadow_output_path": str(
                        directional_shadow_path
                    ),
                    "foot_contact_geometry": foot_contact_geometry,
                    **(
                        {"standing_pose": standing_pose_diagnostics}
                        if standing_pose_diagnostics is not None
                        else {}
                    ),
                    "selected_shadow_softness": job[
                        "shadow_label_policy"
                    ]["softness"],
                    "sun_angle_degrees": float(
                        job["lighting"]["sun_angle_degrees"]
                    ),
                    "sun_energy": float(
                        job["lighting"]["sun_energy"]
                    ),
                    "label_directional_render_version": job[
                        "label_directional_render_version"
                    ],
                },
                indent=2,
            )
        )
        return
    if render_mode != "base":
        raise ValueError(f"Unsupported render_mode: {render_mode}")

    required_passes = set(
        job.get(
            "required_passes",
            [
                "actor_color",
                "actor_depth",
                "actor_sky",
                "ambient_occlusion",
                "actor_sun",
            ],
        )
    )
    produced_passes = set(job.get("produced_passes", required_passes))
    core_passes = {
        "actor_color",
        "actor_depth",
        "actor_sky",
        "ambient_occlusion",
    }
    allowed_passes = {*core_passes, "actor_sun"}
    if not core_passes <= required_passes or not required_passes <= allowed_passes:
        raise ValueError(f"Invalid base required_passes: {sorted(required_passes)}")
    if not required_passes <= produced_passes or not produced_passes <= allowed_passes:
        raise ValueError(f"Invalid base produced_passes: {sorted(produced_passes)}")
    produce_actor_sun = "actor_sun" in produced_passes

    actor_sky_path = Path(job["actor_sky_output_path"]).resolve()
    ambient_occlusion_path = Path(job["ambient_occlusion_output_path"]).resolve()

    # Pass 1: color-managed actor and metric Z. The catcher view layer is
    # disabled because this pass has no consumed shadow product.
    actor_output.mute = False
    actor_linear_output.mute = True
    shadow_output.mute = True
    depth_output.mute = False
    actor_layer.use = True
    shadow_layer.use = False
    bpy.ops.render.render(write_still=False)
    normalize_output_file(output, frame)
    normalize_output_file(depth_path, frame)

    try:
        # Pass 2: diffuse environment illumination only.  The actor output is
        # O_sky; the catcher output contains the short ambient/contact term.
        actor_output.mute = True
        actor_linear_output.mute = False
        shadow_output.mute = False
        depth_output.mute = True
        actor_layer.use = True
        shadow_layer.use = True
        configure_file_output(actor_linear_output, actor_sky_path)
        configure_file_output(shadow_output, ambient_occlusion_path)
        sun.data.energy = 0.0
        world_background.inputs["Strength"].default_value = original_world_strength
        bpy.ops.render.render(write_still=False)
        normalize_output_file(actor_sky_path, frame)
        normalize_output_file(ambient_occlusion_path, frame)

        if produce_actor_sun:
            # Pass 3: direct Sun actor only. Directional labels replace this
            # layer with their selected Sun pass, so they omit it entirely.
            configure_file_output(actor_linear_output, actor_sun_path)
            shadow_output.mute = True
            shadow_layer.use = False
            sun.data.energy = original_sun_energy
            world_background.inputs["Strength"].default_value = 0.0
            bpy.ops.render.render(write_still=False)
            normalize_output_file(actor_sun_path, frame)
    finally:
        sun.data.energy = original_sun_energy
        world_background.inputs["Strength"].default_value = original_world_strength
        actor_output.mute = False
        actor_linear_output.mute = True
        shadow_output.mute = False
        depth_output.mute = False
        actor_layer.use = True
        shadow_layer.use = True
    if not output.is_file():
        raise RuntimeError(f"Actor render was not written: {output}")
    if not depth_path.is_file():
        raise RuntimeError(f"Actor depth pass was not written: {depth_path}")
    expected_outputs = [
        ("sky-only actor", actor_sky_path),
        ("ambient-occlusion", ambient_occlusion_path),
    ]
    if produce_actor_sun:
        expected_outputs.append(("Sun-only actor", actor_sun_path))
    for label, path in expected_outputs:
        if not path.is_file():
            raise RuntimeError(f"{label} pass was not written: {path}")

    info_path = Path(job["render_info_path"]).resolve()
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        json.dumps(
            {
                "cache_key": job["cache_key"],
                "render_mode": render_mode,
                "required_passes": sorted(required_passes),
                "produced_passes": sorted(produced_passes),
                "samples": int(job["samples"]),
                "animation_frame": int(job["animation_frame"]),
                "render_region": render_region,
                "ideal_canvas_resolution": [width, height],
                "full_render_output_resolution": [render_width, render_height],
                "render_output_resolution": [
                    stored_render_width,
                    stored_render_height,
                ],
                "raw_render_storage": job["raw_render_storage"],
                "output_path": str(output),
                "actor_depth_output_path": str(depth_path),
                "actor_sun_output_path": (
                    str(actor_sun_path) if produce_actor_sun else None
                ),
                "actor_sky_output_path": str(actor_sky_path),
                "ambient_occlusion_output_path": str(ambient_occlusion_path),
                "actor_depth_encoding_max_m": float(job["actor_depth_encoding_max_m"]),
                "foot_contact_geometry": foot_contact_geometry,
                **(
                    {"standing_pose": standing_pose_diagnostics}
                    if standing_pose_diagnostics is not None
                    else {}
                ),
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    if not jobs:
        print("BATCH_RENDER_SUMMARY=" + json.dumps({"completed": 0, "failed": 0}))
        return

    device = load_device_module().configure(args.device)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU" if device["backend"] == "CPU" else "GPU"
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"  # Added by me
    scene.cycles.denoising_use_gpu = device["backend"] != "CPU"  # Added by me
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True

    root_names = {
        str(job.get("actor_root_name", "PED_RIG")) for job in jobs
    }
    if len(root_names) != 1:
        raise RuntimeError(
            f"One Blender batch cannot mix actor roots: {sorted(root_names)}"
        )
    root_name = next(iter(root_names))
    rig = bpy.data.objects.get(root_name)
    if rig is None:
        raise RuntimeError(f"Actor root {root_name!r} is missing")
    rig_rest_location = rig.location.copy()
    camera, sun, actor_layer, shadow_layer = create_stage()
    actor_output, actor_linear_output, shadow_output, depth_output = configure_compositor(
        actor_layer, shadow_layer
    )

    completed = 0
    failures = []
    for index, job in enumerate(jobs):
        try:
            render_job(
                job,
                camera,
                sun,
                rig,
                rig_rest_location,
                actor_output,
                actor_linear_output,
                shadow_output,
                depth_output,
            )
            completed += 1
            print(f"BATCH_JOB_DONE={index}:{job.get('job_id', '')}")
            sys.stdout.flush()
        except Exception as exc:
            failures.append({"job_id": job.get("job_id"), "error": str(exc)})
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()

    print("BATCH_RENDER_SUMMARY=" + json.dumps({"completed": completed, "failed": len(failures), "failures": failures}))
    if failures:
        os._exit(2)


if __name__ == "__main__":
    main()
