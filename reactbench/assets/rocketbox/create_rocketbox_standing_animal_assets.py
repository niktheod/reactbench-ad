#!/usr/bin/env python3
"""Build packed, standing-only Blender assets for Rocketbox animals.

This is deliberately separate from ``create_rocketbox_animal_assets.py`` so
the dog walk assets cannot change. Each asset keeps the source rest pose (a
four-leg stance) and holds it in a constant 32-frame action, so the
compositor's asset contract (``animation_frame_start``/``walk_cycle_frames``)
still holds. The sidecar records the grounded rest-pose bounds about the
renderer root; they need not be centred on it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_rocketbox_assets as common


CONVERTER_VERSION = "rocketbox_standing_animal_navsim_v1"
REST_POSE_VERSION = "rocketbox_source_rest_pose_v1"
GEOMETRY_PROFILE_VERSION = "standing_animal_rest_pose_bounds_v1"
LOCAL_FORWARD_AXIS = "-Y"
MAX_FORWARD_AXIS_ERROR_DEGREES = 20.0
MAX_POSE_DRIFT_M = 1e-6
MAX_GROUND_ERROR_M = 0.002
# Every hoof of a four-leg stance must reach the ground.
MAX_CONTACT_HEIGHT_M = 0.05
CONTACT_WEIGHT_THRESHOLD = 0.15
# The quadruped contact renderer selects these deformation groups by side
# token and name prefix (dog_blender_asset/scripts/render_batch.py).
CONTACT_GROUPS = {
    "left_front": ("l", ("hand", "finger")),
    "right_front": ("r", ("hand", "finger")),
    "left_hind": ("l", ("foot", "toe")),
    "right_hind": ("r", ("foot", "toe")),
}


REST_POSE_ADJUSTMENT_VERSION = "hanging_tail_baked_into_rest_v1"
# Per tail segment, degrees back from straight down: a natural hanging tail.
HANGING_TAIL_ANGLES_FROM_DOWN = (55.0, 15.0, 5.0)
MAX_TAIL_DIRECTION_ERROR_DEGREES = 0.1


@dataclass(frozen=True)
class AnimalSpec:
    name: str
    species: str
    height_range_m: tuple[float, float]
    # The renderer resets standing animals to their rest pose, so a tail held
    # straight out in the source rest pose is fixed in the rest pose itself.
    hanging_tail: bool = False


# Rocketbox animal directory, species, and plausible measured standing height.
ANIMAL_SPECS = (
    AnimalSpec("Camel_OneHump_01", "camel", (1.6, 3.0)),
    AnimalSpec("Horse_Brown_01", "horse", (1.4, 2.8), hanging_tail=True),
    AnimalSpec("Cow_Black_01", "cow", (1.0, 2.2)),
    AnimalSpec("Deer_Roe_Tabby_01", "deer", (0.6, 1.6)),
    AnimalSpec("Sheep_White_01", "sheep", (0.6, 1.5)),
)


@dataclass(frozen=True)
class AnimalAsset:
    spec: AnimalSpec
    fbx: Path
    texture_dir: Path

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def output_stem(self) -> str:
        return "rocketbox_" + self.spec.name.lower()

    @property
    def contact_profile(self) -> str:
        return "quadruped_" + self.spec.species


def arguments_after_double_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocketbox-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="NAME_OR_GLOB",
        help="Convert only matching animal asset names; repeatable.",
    )
    parser.add_argument("--limit", type=int, help="Convert at most this many animals.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(arguments_after_double_dash())
    args.rocketbox_root = args.rocketbox_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def discover_animals(rocketbox_root: Path) -> list[AnimalAsset]:
    animal_root = rocketbox_root / "Assets" / "Animals"
    if not animal_root.is_dir():
        raise FileNotFoundError(f"Rocketbox animal directory not found: {animal_root}")
    assets = []
    for spec in ANIMAL_SPECS:
        asset_dir = animal_root / spec.name
        fbx = asset_dir / "Export" / f"{spec.name}.fbx"
        texture_dir = asset_dir / "Textures"
        if not fbx.is_file():
            raise FileNotFoundError(f"Animal FBX is missing: {fbx}")
        if not texture_dir.is_dir():
            raise FileNotFoundError(f"Animal texture directory is missing: {texture_dir}")
        assets.append(
            AnimalAsset(spec=spec, fbx=fbx.resolve(), texture_dir=texture_dir.resolve())
        )
    return assets


def select_assets(
    assets: Sequence[AnimalAsset], patterns: Sequence[str], limit: int | None
) -> list[AnimalAsset]:
    if not patterns:
        selected = list(assets)
    else:
        lowered_patterns = [pattern.lower() for pattern in patterns]
        selected = [
            asset
            for asset in assets
            if any(
                fnmatch.fnmatchcase(asset.name.lower(), pattern)
                for pattern in lowered_patterns
            )
        ]
        unmatched = [
            pattern
            for pattern in patterns
            if not any(
                fnmatch.fnmatchcase(asset.name.lower(), pattern.lower())
                for asset in assets
            )
        ]
        if unmatched:
            raise ValueError(f"No Rocketbox standing animal matched: {', '.join(unmatched)}")
    return selected[:limit] if limit is not None else selected


def existing_output_is_current(
    blend_path: Path,
    sidecar_path: Path,
    asset: AnimalAsset,
    revision: str,
) -> bool:
    if not blend_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    common_script = Path(common.__file__).resolve()
    return (
        blend_path.stat().st_size > 0
        and metadata.get("animation_version") == CONVERTER_VERSION
        and metadata.get("source_animation_version") == REST_POSE_VERSION
        and metadata.get("source_revision") == revision
        and metadata.get("source_character_id") == asset.name
        and metadata.get("species") == asset.spec.species
        and metadata.get("source_character_fbx_sha256") == common.sha256_file(asset.fbx)
        and metadata.get("generator_script_sha256")
        == common.sha256_file(Path(__file__).resolve())
        and metadata.get("common_asset_helper_sha256")
        == common.sha256_file(common_script)
        and metadata.get("resources_packed") is True
        and metadata.get("standing_only") is True
        and metadata.get("walk_cycle_frames") == common.WALK_CYCLE_FRAMES
    )


def bones_by_suffix(rig: bpy.types.Object) -> dict[str, bpy.types.PoseBone]:
    """Map Rocketbox ``<animal> <suffix>`` bone names to their suffix."""
    result = {}
    for pose_bone in rig.pose.bones:
        parts = pose_bone.name.split(" ", 1)
        if len(parts) != 2:
            continue
        if parts[1] in result:
            raise RuntimeError(f"Duplicate animal bone suffix {parts[1]!r} in {rig.name}")
        result[parts[1]] = pose_bone
    return result


def clear_imported_animation(rig: bpy.types.Object) -> None:
    matrix_basis = rig.matrix_basis.copy()
    rig.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    rig.matrix_basis = matrix_basis
    common.reset_pose(rig)


def bake_hanging_tail(
    rig: bpy.types.Object,
    meshes: Sequence[bpy.types.Object],
    bones: dict[str, bpy.types.PoseBone],
) -> dict:
    """Pose the tail to hang down, then make that pose the rest pose.

    Each skinned mesh first receives the posed shape (its armature modifier is
    applied), the posed armature becomes the new rest pose, and the modifier is
    restored, so the rest pose shows the hanging tail.
    """
    chain = [name for name in ("Tail", "Tail1", "Tail2", "Tail3") if name in bones]
    segments = list(zip(chain, chain[1:]))
    if len(segments) != len(HANGING_TAIL_ANGLES_FROM_DOWN):
        raise RuntimeError(f"Expected a four-bone tail, found {chain}")
    for mesh in meshes:
        if mesh.data.shape_keys is not None:
            raise RuntimeError(f"Cannot bake a rest pose into shape-keyed mesh {mesh.name}")

    common.reset_pose(rig)
    targets = {}
    for (parent, child), angle in zip(segments, HANGING_TAIL_ANGLES_FROM_DOWN):
        desired = common.direction_from_down(angle)
        common.align_joint_vector(rig, bones[parent], bones[child], desired)
        targets[f"{parent}->{child}"] = (parent, child, desired, angle)
    bpy.context.view_layer.update()
    errors = {
        name: math.degrees(
            common.joint_world_direction(rig, bones[parent], bones[child]).angle(desired)
        )
        for name, (parent, child, desired, _angle) in targets.items()
    }
    if max(errors.values()) > MAX_TAIL_DIRECTION_ERROR_DEGREES:
        raise RuntimeError(f"Hanging tail missed its target directions: {errors}")

    restored = []
    for mesh in meshes:
        modifiers = [
            modifier
            for modifier in mesh.modifiers
            if modifier.type == "ARMATURE" and modifier.object == rig
        ]
        if len(modifiers) != 1:
            raise RuntimeError(f"Expected one armature modifier on {mesh.name}")
        modifier = modifiers[0]
        restored.append(
            (
                mesh,
                modifier.name,
                list(mesh.modifiers).index(modifier),
                modifier.use_vertex_groups,
                modifier.use_bone_envelopes,
                modifier.use_deform_preserve_volume,
            )
        )
        with bpy.context.temp_override(
            object=mesh,
            active_object=mesh,
            selected_objects=[mesh],
            selected_editable_objects=[mesh],
        ):
            result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        if "FINISHED" not in result:
            raise RuntimeError(f"Could not apply the armature modifier of {mesh.name}")

    bpy.context.view_layer.objects.active = rig
    with bpy.context.temp_override(
        object=rig,
        active_object=rig,
        selected_objects=[rig],
        selected_editable_objects=[rig],
    ):
        bpy.ops.object.mode_set(mode="POSE")
        result = bpy.ops.pose.armature_apply(selected=False)
        bpy.ops.object.mode_set(mode="OBJECT")
    if "FINISHED" not in result:
        raise RuntimeError("Could not apply the hanging-tail pose as the rest pose")

    for mesh, name, index, vertex_groups, envelopes, preserve_volume in restored:
        modifier = mesh.modifiers.new(name, "ARMATURE")
        modifier.object = rig
        modifier.use_vertex_groups = vertex_groups
        modifier.use_bone_envelopes = envelopes
        modifier.use_deform_preserve_volume = preserve_volume
        with bpy.context.temp_override(object=mesh, active_object=mesh):
            bpy.ops.object.modifier_move_to_index(modifier=name, index=index)
    common.reset_pose(rig)
    return {
        "version": REST_POSE_ADJUSTMENT_VERSION,
        "tail_angles_from_down_degrees": {
            name: angle for name, (_p, _c, _d, angle) in targets.items()
        },
        "maximum_target_direction_error_degrees": max(errors.values()),
    }


def install_rest_pose_action(rig: bpy.types.Object) -> bpy.types.Action:
    """Replace imported animation with the rest pose held over one cycle."""
    clear_imported_animation(rig)
    action = common.new_rig_action(rig, "PED_WALK_ACTION")
    samples = common.basis_samples(rig, {})
    for frame in (common.FRAME_START, common.FRAME_END_CLOSURE):
        common.key_basis_samples(rig, frame, samples)
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "CONSTANT"
    action.use_frame_range = True
    action.frame_start = common.FRAME_START
    action.frame_end = common.FRAME_END_CLOSURE
    return action


def evaluated_world_bounds(
    meshes: Sequence[bpy.types.Object],
) -> tuple[list[float], list[float], int]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum = [math.inf] * 3
    maximum = [-math.inf] * 3
    count = 0
    for mesh in meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            matrix = evaluated.matrix_world
            for vertex in evaluated_mesh.vertices:
                point = matrix @ vertex.co
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], float(point[axis]))
                    maximum[axis] = max(maximum[axis], float(point[axis]))
                count += 1
        finally:
            evaluated.to_mesh_clear()
    if count == 0:
        raise RuntimeError("No evaluated animal mesh vertices found")
    return minimum, maximum, count


def rest_pose_drift_m(meshes: Sequence[bpy.types.Object], reference: tuple) -> float:
    """Largest bound change over the cycle; the held rest pose must not move."""
    scene = bpy.context.scene
    drift = 0.0
    for frame in sorted({*common.POSE_FRAMES.values(), common.WALK_CYCLE_FRAMES}):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        minimum, maximum, _count = evaluated_world_bounds(meshes)
        drift = max(
            drift,
            *(abs(a - b) for a, b in zip(minimum, reference[0])),
            *(abs(a - b) for a, b in zip(maximum, reference[1])),
        )
    scene.frame_set(common.FRAME_START)
    bpy.context.view_layer.update()
    return drift


def forward_axis_diagnostics(
    rig: bpy.types.Object, bones: dict[str, bpy.types.PoseBone]
) -> dict:
    """Verify that the head lies along local -Y from the pelvis."""
    head = rig.matrix_world @ bones["Head"].head
    pelvis = rig.matrix_world @ bones["Pelvis"].head
    direction = Vector((head.x - pelvis.x, head.y - pelvis.y))
    if direction.length < 1e-6:
        raise RuntimeError("Head and pelvis coincide in plan view")
    error = math.degrees(direction.angle(Vector((0.0, -1.0))))
    if error > MAX_FORWARD_AXIS_ERROR_DEGREES:
        raise RuntimeError(
            f"Animal does not face {LOCAL_FORWARD_AXIS}: pelvis-to-head "
            f"direction is {error:.2f} degrees away"
        )
    return {
        "method": "pelvis_to_head_plan_direction",
        "head_xyz_m": [float(value) for value in head],
        "pelvis_xyz_m": [float(value) for value in pelvis],
        "forward_axis_error_degrees": error,
    }


def normalized_group_tokens(name: str) -> list[str]:
    return (
        str(name).lower().replace(".", " ").replace("_", " ").replace("-", " ").split()
    )


def contact_diagnostics(meshes: Sequence[bpy.types.Object]) -> dict:
    """Require weighted vertices for every hoof, each touching the ground."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for contact_name, (side_token, prefixes) in CONTACT_GROUPS.items():
        group_names = []
        count = 0
        minimum_z = math.inf
        for mesh in meshes:
            indices = {
                group.index
                for group in mesh.vertex_groups
                if side_token in normalized_group_tokens(group.name)
                and any(
                    token.startswith(prefix)
                    for token in normalized_group_tokens(group.name)
                    for prefix in prefixes
                )
            }
            if not indices:
                continue
            group_names.extend(
                f"{mesh.name}:{group.name}"
                for group in mesh.vertex_groups
                if group.index in indices
            )
            evaluated = mesh.evaluated_get(depsgraph)
            evaluated_mesh = evaluated.to_mesh()
            try:
                for vertex in mesh.data.vertices:
                    if vertex.index >= len(evaluated_mesh.vertices) or not any(
                        membership.group in indices
                        and membership.weight >= CONTACT_WEIGHT_THRESHOLD
                        for membership in vertex.groups
                    ):
                        continue
                    point = evaluated.matrix_world @ evaluated_mesh.vertices[vertex.index].co
                    minimum_z = min(minimum_z, float(point.z))
                    count += 1
            finally:
                evaluated.to_mesh_clear()
        if count == 0:
            raise RuntimeError(f"No weighted contact vertices for {contact_name}")
        if minimum_z > MAX_CONTACT_HEIGHT_M:
            raise RuntimeError(
                f"{contact_name} does not reach the ground in the rest pose: "
                f"lowest weighted vertex at {minimum_z:.4f} m"
            )
        result[contact_name] = {
            "group_names": sorted(group_names),
            "weighted_vertex_count": count,
            "minimum_z_m": minimum_z,
        }
    return result


def build_one(
    asset: AnimalAsset,
    rocketbox_root: Path,
    output_dir: Path,
    revision: str,
    overwrite: bool,
) -> dict:
    blend_path = output_dir / f"{asset.output_stem}.blend"
    sidecar_path = blend_path.with_suffix(".json")
    if not overwrite and existing_output_is_current(
        blend_path, sidecar_path, asset, revision
    ):
        print(f"SKIP complete {asset.name}: {blend_path}", flush=True)
        return {"asset": asset.name, "status": "skipped", "blend": str(blend_path)}
    if not overwrite and (blend_path.exists() or sidecar_path.exists()):
        print(f"REBUILD stale or incomplete {asset.name}: {blend_path}", flush=True)

    common.reset_blender()
    print(f"BUILD {asset.name} (Animals, {asset.spec.species}, standing)", flush=True)
    character_objects = common.imported_objects(asset.fbx)
    deform_rig = common.exactly_one_armature(character_objects, asset.fbx)
    meshes = [obj for obj in character_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No meshes imported from {asset.fbx}")
    print(
        "IMPORTED_MESHES "
        + json.dumps({mesh.name: len(mesh.data.vertices) for mesh in meshes}, sort_keys=True),
        flush=True,
    )
    bones = bones_by_suffix(deform_rig)
    missing = sorted({"Pelvis", "Head"} - bones.keys())
    if missing:
        raise RuntimeError(f"Animal rig is missing required bones: {missing}")

    texture_sources = common.repair_and_pack_textures(
        meshes, asset.texture_dir, rocketbox_root
    )
    clear_imported_animation(deform_rig)
    rest_pose_adjustment = (
        bake_hanging_tail(deform_rig, meshes, bones) if asset.spec.hanging_tail else None
    )
    bones = bones_by_suffix(deform_rig)
    walk_action = install_rest_pose_action(deform_rig)

    for obj in list(character_objects):
        if obj.type not in {"MESH", "ARMATURE"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    character_objects = list(bpy.context.scene.objects)
    collection = common.move_to_asset_collection(character_objects)
    wrapper, ground = common.create_stationary_root(deform_rig, collection)
    common.rename_meshes(meshes)
    ground_action, ground_diagnostics = common.add_ground_correction(ground, meshes)
    transform_diagnostics = common.object_transform_diagnostics(deform_rig)

    scene = bpy.context.scene
    scene.frame_set(common.FRAME_START)
    bpy.context.view_layer.update()
    minimum, maximum, vertex_samples = evaluated_world_bounds(meshes)
    drift = rest_pose_drift_m(meshes, (minimum, maximum))
    forward_diagnostics = forward_axis_diagnostics(deform_rig, bones)
    contacts = contact_diagnostics(meshes)

    height = maximum[2] - minimum[2]
    width = maximum[0] - minimum[0]
    length = maximum[1] - minimum[1]
    scene["character_height_m"] = float(height)
    scene["actor_type"] = asset.spec.species
    scene["source"] = "Microsoft Rocketbox"
    scene["source_revision"] = revision
    scene["license"] = "MIT"
    scene["conversion_version"] = CONVERTER_VERSION

    if wrapper.animation_data is not None and wrapper.animation_data.action is not None:
        raise RuntimeError("PED_RIG must not have object animation")
    xy_displacement = common.action_xy_displacement(deform_rig)
    if xy_displacement > 1e-6:
        raise RuntimeError(f"Horizontal root motion remains: {xy_displacement:.9f} m")
    if transform_diagnostics["location_displacement_m"] > 1e-7:
        raise RuntimeError("Animal armature object translation is animated")
    if transform_diagnostics["rotation_displacement_radians"] > 1e-7:
        raise RuntimeError("Animal armature object rotation is animated")
    if transform_diagnostics["scale_displacement"] > 1e-7:
        raise RuntimeError("Animal armature object scale is animated")
    if ground_diagnostics["maximum_ground_error_m"] > MAX_GROUND_ERROR_M:
        raise RuntimeError(
            "Animal ground correction failed: "
            f"{ground_diagnostics['maximum_ground_error_m']:.6f} m"
        )
    if drift > MAX_POSE_DRIFT_M:
        raise RuntimeError(f"Held rest pose moves over the cycle: {drift:.9f} m")
    low, high = asset.spec.height_range_m
    if not low <= height <= high:
        raise RuntimeError(
            f"Generated {asset.spec.species} height is outside {low}..{high} m: "
            f"{height:.3f} m"
        )
    if not length > width:
        raise RuntimeError(
            f"Animal body is not longest along Y: length={length:.3f} m, "
            f"width={width:.3f} m"
        )
    if not (minimum[0] < 0.0 < maximum[0] and minimum[1] < 0.0 < maximum[1]):
        raise RuntimeError(
            "Renderer root lies outside the animal footprint: "
            f"x={minimum[0]:.3f}..{maximum[0]:.3f}, y={minimum[1]:.3f}..{maximum[1]:.3f}"
        )
    common.pack_check(meshes)

    common_script = Path(common.__file__).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    actor_geometry = {
        "profile_version": GEOMETRY_PROFILE_VERSION,
        "actor_type": "animal",
        "species": asset.spec.species,
        "asset_id": asset.output_stem,
        "source": "generated_asset_sidecar",
        "units": "m",
        "local_forward_axis": LOCAL_FORWARD_AXIS,
        "local_bounds_m": {
            "x": [minimum[0], maximum[0]],
            "y": [minimum[1], maximum[1]],
            "z": [minimum[2], maximum[2]],
        },
        "measurement": "evaluated mesh vertices of the grounded rest pose",
        "measurement_frame": common.FRAME_START,
        "measured_mesh_objects": sorted(mesh.name for mesh in meshes),
        "evaluated_vertex_samples": vertex_samples,
        "measurement_blender_version": bpy.app.version_string,
    }
    metadata = {
        "id": asset.output_stem,
        "asset_path": str(blend_path),
        "actor_type": "animal",
        "species": asset.spec.species,
        "contact_profile": asset.contact_profile,
        "standing_only": True,
        "source": "Microsoft Rocketbox",
        "source_repository": "https://github.com/microsoft/Microsoft-Rocketbox",
        "source_revision": revision,
        "source_category": "Animals",
        "source_character_id": asset.name,
        "source_character_fbx": str(asset.fbx.relative_to(rocketbox_root)),
        "source_character_fbx_sha256": common.sha256_file(asset.fbx),
        "source_animation": "source rest pose held over one cycle (standing only)",
        "source_animation_version": REST_POSE_VERSION,
        "rest_pose_adjustment": rest_pose_adjustment,
        "generator_script": str(Path(__file__).resolve().relative_to(repository_root)),
        "generator_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_asset_helper": str(common_script.relative_to(repository_root)),
        "common_asset_helper_sha256": common.sha256_file(common_script),
        "source_texture_files": texture_sources,
        "license": "MIT",
        "license_file": "../ROCKETBOX_LICENSE.md",
        "procedural_gait": False,
        "character_height_m": height,
        "actor_width_m": width,
        "actor_length_m": length,
        "actor_geometry": actor_geometry,
        "walk_action": walk_action.name,
        "ground_action": ground_action.name,
        "animation_version": CONVERTER_VERSION,
        "animation_frame_start": common.FRAME_START,
        "walk_cycle_frames": common.WALK_CYCLE_FRAMES,
        "pose_frames": common.POSE_FRAMES,
        "local_forward_axis": LOCAL_FORWARD_AXIS,
        "ground_contact_z_m": 0.0,
        "root_motion": "stationary_xy",
        "resources_packed": True,
        "deformation_rig": deform_rig.name,
        "renderer_root": wrapper.name,
        "actor_collection": collection.name,
        "mesh_objects": sorted(mesh.name for mesh in meshes),
        "mesh_vertices": sum(len(mesh.data.vertices) for mesh in meshes),
        "deformation_bones": len(deform_rig.data.bones),
        "rest_pose_drift_m": drift,
        "forward_axis_diagnostics": forward_diagnostics,
        "contact_diagnostics": contacts,
        "object_transform_diagnostics": transform_diagnostics,
        "ground_diagnostics": ground_diagnostics,
        "horizontal_root_displacement_m": xy_displacement,
        "blender_version": bpy.app.version_string,
    }

    temporary_blend = output_dir / f".{asset.output_stem}.tmp.blend"
    temporary_blend.unlink(missing_ok=True)
    save_result = bpy.ops.wm.save_as_mainfile(
        filepath=str(temporary_blend), check_existing=False, compress=True
    )
    if "FINISHED" not in save_result or not temporary_blend.is_file():
        raise RuntimeError(f"Blender did not save candidate animal asset: {temporary_blend}")
    os.replace(temporary_blend, blend_path)
    common.atomic_json(sidecar_path, metadata)
    print(
        f"DONE {asset.name}: height={height:.3f}m length={length:.3f}m "
        f"width={width:.3f}m x={minimum[0]:.3f}..{maximum[0]:.3f} "
        f"y={minimum[1]:.3f}..{maximum[1]:.3f} output={blend_path}",
        flush=True,
    )
    return {
        "asset": asset.name,
        "status": "completed",
        "blend": str(blend_path),
        "metadata": str(sidecar_path),
        "character_height_m": height,
    }


def main() -> None:
    args = parse_args()
    assets = discover_animals(args.rocketbox_root)
    selected = select_assets(assets, args.asset, args.limit)
    print(
        f"ROCKETBOX_STANDING_ANIMAL_SELECTION total={len(assets)} "
        f"selected={len(selected)} output={args.output_dir}",
        flush=True,
    )
    for asset in selected:
        print(f"SELECT Animals/{asset.name} ({asset.spec.species})", flush=True)
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    license_source = args.rocketbox_root / "LICENSE.md"
    if not license_source.is_file():
        raise FileNotFoundError(f"Rocketbox MIT license file missing: {license_source}")
    shutil.copyfile(license_source, args.output_dir.parent / "ROCKETBOX_LICENSE.md")
    revision = common.git_revision(args.rocketbox_root)

    results = []
    failures = []
    for asset in selected:
        try:
            results.append(
                build_one(
                    asset,
                    rocketbox_root=args.rocketbox_root,
                    output_dir=args.output_dir,
                    revision=revision,
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "asset": asset.name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAILED {asset.name}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()

    summary = {
        "converter_version": CONVERTER_VERSION,
        "rest_pose_version": REST_POSE_VERSION,
        "rocketbox_root": str(args.rocketbox_root),
        "rocketbox_revision": revision,
        "output_dir": str(args.output_dir),
        "selected_count": len(selected),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "skipped_count": sum(result["status"] == "skipped" for result in results),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
    }
    common.atomic_json(args.output_dir.parent / "generation_manifest.json", summary)
    print(
        "ROCKETBOX_STANDING_ANIMAL_GENERATION_COMPLETE "
        f"completed={summary['completed_count']} skipped={summary['skipped_count']} "
        f"failed={summary['failed_count']} output={args.output_dir}",
        flush=True,
    )
    if failures:
        raise RuntimeError(
            f"Standing animal generation failed for {len(failures)} selected asset(s)"
        )


if __name__ == "__main__":
    main()
