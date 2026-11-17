#!/usr/bin/env python3
"""Build packed, stationary-walk Blender assets from Microsoft Rocketbox.

Run this through ``scripts/assets.sh humans``.
The generated .blend/.json pairs are directly consumable by
``reactbench/compositing/navsim/compositor/engine.py``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import bpy
from mathutils import Matrix, Quaternion, Vector


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent  # measure_pedestrian_geometry; Blender does not add it to sys.path
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reactbench.scoring.pedestrian_geometry import (  # noqa: E402
    PEDESTRIAN_GEOMETRY_PROFILE_VERSION,
    normalize_pedestrian_geometry,
)
from measure_pedestrian_geometry import (  # noqa: E402
    measure_gait_union_geometry,
)


CONVERTER_VERSION = "rocketbox_navsim_v8_male_torso_bind_preserved"
REFERENCE_GAIT_VERSION = "procedural_natural_walk_v7_male_torso_bind_preserved"
FRAME_START = 1
FRAME_END_CLOSURE = 33
WALK_CYCLE_FRAMES = 32
POSE_FRAMES = {
    "oldest": 24,
    "history_2": 15,
    "history_1": 6,
    "current": 29,
}
NAVSIM_REFERENCE_SPEED_MPS = 2.0
NAVSIM_GAIT_TIMESTEP_S = 0.5
NAVSIM_GAIT_FRAME_ADVANCE = 23
BIP_PREFIX_RE = re.compile(r"^Bip\d+\s*")
CORE_AVATAR_NAME_RE = re.compile(r"^(?:Female|Male)_(?:Adult|Child)_[0-9]+$")
CORE_AVATAR_CATEGORIES = {"Adults", "Children"}
EXPECTED_CORE_AVATARS = 42
REFERENCE_GAIT_KEYFRAMES = {
    # Each side stores global sagittal angles in degrees for
    # (thigh, calf, foot, upper arm, forearm).  These were measured from the
    # evaluated seed_42 procedural gait rather than copied from local bone
    # rotations, so they remain meaningful on Rocketbox's differently rolled
    # Biped bones and on child proportions.
    1: {
        "L": (18.748, 32.404, 45.155, -21.850, -54.780),
        "R": (-21.930, -3.122, 32.155, 31.318, 5.757),
        "torso_lean": 2.899,
    },
    5: {
        "L": (5.474, 20.397, 39.155, -9.466, -40.798),
        "R": (-7.787, 24.848, 60.155, 18.376, -8.829),
        "torso_lean": 2.688,
    },
    9: {
        "L": (-11.322, 17.855, 45.155, 13.691, -13.829),
        "R": (8.127, 23.397, 43.155, -5.184, -36.191),
        "torso_lean": 2.548,
    },
    13: {
        "L": (-17.509, 3.374, 36.155, 23.985, -2.246),
        "R": (13.436, 27.901, 44.155, -15.093, -47.376),
        "torso_lean": 2.689,
    },
    17: {
        "L": (-21.930, -3.122, 32.155, 31.318, 5.757),
        "R": (18.748, 32.404, 45.155, -21.850, -54.780),
        "torso_lean": 2.899,
    },
    21: {
        "L": (-7.787, 24.848, 60.155, 18.376, -8.829),
        "R": (5.474, 20.397, 39.155, -9.466, -40.798),
        "torso_lean": 2.688,
    },
    25: {
        "L": (8.127, 23.397, 43.155, -5.184, -36.191),
        "R": (-11.322, 17.855, 45.155, 13.691, -13.829),
        "torso_lean": 2.548,
    },
    29: {
        "L": (13.436, 27.901, 44.155, -15.093, -47.376),
        "R": (-17.509, 3.374, 36.155, 23.985, -2.246),
        "torso_lean": 2.689,
    },
    33: {
        "L": (18.748, 32.404, 45.155, -21.850, -54.780),
        "R": (-21.930, -3.122, 32.155, 31.318, 5.757),
        "torso_lean": 2.899,
    },
}
# Retain a narrow lateral component to prevent the arms or legs from passing
# through the torso while keeping the side silhouette dominated by fore/aft
# motion, as in the seed_42 reference assets.
ARM_OUTWARD_COMPONENT = 0.12
LEG_OUTWARD_COMPONENT = 0.02
MIN_ANIMATED_ARM_OUTWARD_COMPONENT = 0.10
MAX_REFERENCE_DIRECTION_ERROR_DEGREES = 0.1
# The female and male Rocketbox rigs use different bind-pose relationships
# between their first two torso segments.  Offset the male's lower-torso target
# by that measured difference.  Its visible Spine1 -> Spine2 chest segment then
# matches the female direction while every internal male spine/neck relationship
# remains untouched; rotating Spine1 itself would introduce an artificial arch.
FEMALE_CHEST_AXIS_OFFSET_DEGREES = -4.933292
MALE_CHEST_AXIS_OFFSET_DEGREES = -1.567598
MALE_TORSO_LEAN_CORRECTION_DEGREES = (
    FEMALE_CHEST_AXIS_OFFSET_DEGREES - MALE_CHEST_AXIS_OFFSET_DEGREES
)
MIN_ANATOMICAL_KNEE_ANGLE_DEGREES = 140.0
MAX_ANATOMICAL_KNEE_ANGLE_DEGREES = 175.0
MIN_ANATOMICAL_ELBOW_ANGLE_DEGREES = 140.0
MAX_ANATOMICAL_ELBOW_ANGLE_DEGREES = 160.0
ARM_CLEARANCE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SourceAsset:
    name: str
    category: str
    sex: str
    fbx: Path
    texture_dir: Path

    @property
    def output_stem(self) -> str:
        return "rocketbox_" + self.name.lower()


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
        help="Convert only matching source asset names; repeatable.",
    )
    parser.add_argument("--limit", type=int, help="Convert at most this many selected assets.")
    parser.add_argument("--overwrite", action="store_true", help="Replace complete existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="List selected inputs without converting.")
    args = parser.parse_args(arguments_after_double_dash())
    args.rocketbox_root = args.rocketbox_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def infer_sex(name: str) -> str:
    lowered = name.lower()
    if "female" in lowered:
        return "female"
    if "male" in lowered:
        return "male"
    raise ValueError(f"Cannot infer sex from Rocketbox asset name: {name}")


def discover_assets(rocketbox_root: Path) -> list[SourceAsset]:
    avatar_root = rocketbox_root / "Assets" / "Avatars"
    if not avatar_root.is_dir():
        raise FileNotFoundError(f"Rocketbox avatar directory not found: {avatar_root}")

    assets = []
    for fbx in sorted(avatar_root.glob("*/*/Export/*.fbx")):
        if fbx.name.endswith("_facial.fbx"):
            continue
        asset_dir = fbx.parent.parent
        if fbx.stem != asset_dir.name:
            continue
        category = asset_dir.parent.name
        # Keep only the 38 generic adults and four generic children. Rocketbox
        # stores Female_Party avatars under Adults, so category filtering alone
        # is insufficient; the canonical name must also say Adult or Child.
        if (
            category not in CORE_AVATAR_CATEGORIES
            or CORE_AVATAR_NAME_RE.fullmatch(asset_dir.name) is None
        ):
            continue
        texture_dir = asset_dir / "Textures"
        if not texture_dir.is_dir():
            raise FileNotFoundError(f"Texture directory missing for {fbx}: {texture_dir}")
        assets.append(
            SourceAsset(
                name=asset_dir.name,
                category=category,
                sex=infer_sex(asset_dir.name),
                fbx=fbx.resolve(),
                texture_dir=texture_dir.resolve(),
            )
        )

    names = [asset.name for asset in assets]
    if len(assets) != EXPECTED_CORE_AVATARS or len(set(names)) != EXPECTED_CORE_AVATARS:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE_AVATARS} unique adult/child Rocketbox FBX avatars, found "
            f"{len(assets)} files and {len(set(names))} unique names"
        )
    return assets


def select_assets(
    assets: Sequence[SourceAsset], patterns: Sequence[str], limit: int | None
) -> list[SourceAsset]:
    if patterns:
        lowered_patterns = [pattern.lower() for pattern in patterns]
        selected = [
            asset
            for asset in assets
            if any(fnmatch.fnmatchcase(asset.name.lower(), pattern) for pattern in lowered_patterns)
        ]
        unmatched = [
            pattern
            for pattern in patterns
            if not any(fnmatch.fnmatchcase(asset.name.lower(), pattern.lower()) for asset in assets)
        ]
        if unmatched:
            raise ValueError(f"No Rocketbox asset matched: {', '.join(unmatched)}")
    else:
        selected = list(assets)
    return selected[:limit] if limit is not None else selected


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prune_excluded_generated_outputs(
    output_dir: Path, assets: Sequence[SourceAsset]
) -> list[str]:
    """Remove generator-owned non-core outputs left by older converter versions."""
    allowed_stems = {asset.output_stem for asset in assets}
    removed = []
    for path in sorted(output_dir.iterdir()):
        if (
            not path.is_file()
            or path.suffix not in {".blend", ".json"}
            or not path.stem.startswith("rocketbox_")
            or path.stem in allowed_stems
        ):
            continue
        path.unlink()
        removed.append(str(path))
        print(f"REMOVE excluded non-core output: {path}", flush=True)
    return removed


def navsim_gait_timing() -> dict:
    """Describe and validate the gait phase used by the NAVSIM compositor."""
    names = ("oldest", "history_2", "history_1", "current")
    sequence = [POSE_FRAMES[name] for name in names]
    expected = [sequence[0]]
    for _ in range(1, len(sequence)):
        phase = (expected[-1] - FRAME_START + NAVSIM_GAIT_FRAME_ADVANCE) % WALK_CYCLE_FRAMES
        expected.append(FRAME_START + phase)
    if sequence != expected:
        raise RuntimeError(
            "NAVSIM pose frames do not follow the configured gait advance: "
            f"sequence={sequence}, expected={expected}, advance={NAVSIM_GAIT_FRAME_ADVANCE}"
        )

    stride_length_m = (
        WALK_CYCLE_FRAMES
        * NAVSIM_REFERENCE_SPEED_MPS
        * NAVSIM_GAIT_TIMESTEP_S
        / NAVSIM_GAIT_FRAME_ADVANCE
    )
    return {
        "reference_speed_mps": NAVSIM_REFERENCE_SPEED_MPS,
        "timestep_s": NAVSIM_GAIT_TIMESTEP_S,
        "frame_advance": NAVSIM_GAIT_FRAME_ADVANCE,
        "pose_sequence": sequence,
        "stride_length_m": stride_length_m,
        "cycle_duration_s": stride_length_m / NAVSIM_REFERENCE_SPEED_MPS,
    }


def reset_blender() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = WALK_CYCLE_FRAMES
    scene.render.fps = 24
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def imported_objects(fbx: Path) -> set[bpy.types.Object]:
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.fbx(filepath=str(fbx), use_anim=True)
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender failed to import FBX: {fbx}")
    return set(bpy.data.objects) - before


def exactly_one_armature(objects: Iterable[bpy.types.Object], source: Path) -> bpy.types.Object:
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature in {source}, found {len(armatures)}")
    return armatures[0]


def image_nodes(meshes: Sequence[bpy.types.Object]) -> list[bpy.types.ShaderNodeTexImage]:
    nodes = []
    seen_materials = set()
    for mesh in meshes:
        for slot in mesh.material_slots:
            material = slot.material
            if material is None or material in seen_materials:
                continue
            seen_materials.add(material)
            if not material.use_nodes or material.node_tree is None:
                continue
            nodes.extend(
                node
                for node in material.node_tree.nodes
                if node.bl_idname == "ShaderNodeTexImage" and node.image is not None
            )
    return nodes


def repair_and_pack_textures(
    meshes: Sequence[bpy.types.Object], texture_dir: Path, rocketbox_root: Path
) -> list[str]:
    available = {}
    for texture in texture_dir.rglob("*"):
        if texture.is_file():
            available.setdefault(texture.name.casefold(), texture.resolve())

    nodes = image_nodes(meshes)
    if not nodes:
        raise RuntimeError(f"No imported image texture nodes found for {texture_dir.parent.name}")

    canonical_by_path: dict[Path, bpy.types.Image] = {}
    source_paths: set[Path] = set()
    missing = []
    for node in nodes:
        imported_path = node.image.filepath.replace("\\", "/")
        basename = Path(imported_path).name or node.image.name
        source = available.get(basename.casefold())
        if source is None:
            missing.append(basename)
            continue

        if source in canonical_by_path:
            node.image = canonical_by_path[source]
            continue

        image = node.image
        image.filepath = str(source)
        image.source = "FILE"
        image.reload()
        lowered = source.name.lower()
        if "_normal" in lowered or "_specular" in lowered:
            image.colorspace_settings.name = "Non-Color"
        canonical_by_path[source] = image
        source_paths.add(source)

    if missing:
        raise FileNotFoundError(
            f"Missing {len(set(missing))} textures in {texture_dir}: "
            + ", ".join(sorted(set(missing)))
        )

    # Remove duplicate FBX image datablocks after remapping their material nodes.
    used_images = {node.image for node in nodes}
    for image in list(bpy.data.images):
        if image not in used_images and image.users == 0:
            bpy.data.images.remove(image)

    for source, image in canonical_by_path.items():
        image.pack()
        if image.packed_file is None:
            raise RuntimeError(f"Texture did not pack into the blend: {source}")
        image.filepath = f"//textures/{source.name}"

    unpacked = [
        image.name
        for image in {node.image for node in nodes}
        if image.source == "FILE" and image.packed_file is None
    ]
    if unpacked:
        raise RuntimeError(f"Unpacked texture images remain: {', '.join(sorted(unpacked))}")

    return sorted(str(path.relative_to(rocketbox_root)) for path in source_paths)


def pose_bones_by_suffix(rig: bpy.types.Object) -> dict[str, bpy.types.PoseBone]:
    result = {}
    for pose_bone in rig.pose.bones:
        suffix = BIP_PREFIX_RE.sub("", pose_bone.name)
        if suffix in result:
            raise RuntimeError(
                f"Duplicate biped bone suffix {suffix!r} in armature {rig.name}"
            )
        result[suffix] = pose_bone
    return result


def knee_angles_degrees(
    bones_by_suffix: dict[str, bpy.types.PoseBone],
) -> dict[str, float]:
    result = {}
    for side in ("L", "R"):
        names = [f"{side} {part}" for part in ("Thigh", "Calf", "Foot")]
        try:
            thigh, calf, foot = (bones_by_suffix[name] for name in names)
        except KeyError as exc:
            raise RuntimeError(f"Required leg bone is missing: {exc.args[0]}") from exc
        hip = thigh.head
        knee = calf.head
        ankle = foot.head
        result[side] = math.degrees((hip - knee).angle(ankle - knee))
    return result


def apply_armature_rotation_at_joint(
    pose_bone: bpy.types.PoseBone,
    rotation: Quaternion,
) -> None:
    """Apply an armature-space rotation without changing the joint position."""
    desired_matrix = rotation.to_matrix().to_4x4()
    desired_matrix.translation = pose_bone.head.copy()
    pose_bone.matrix = desired_matrix
    bpy.context.view_layer.update()


def arm_clearance_diagnostics(
    rig: bpy.types.Object,
    bones_by_suffix: dict[str, bpy.types.PoseBone],
) -> dict[str, float]:
    """Measure arm-joint clearance in the pedestrian's world-space lateral axis."""
    result = {}
    for side in ("L", "R"):
        sign = 1.0 if side == "L" else -1.0
        shoulder = rig.matrix_world @ bones_by_suffix[f"{side} UpperArm"].head
        elbow = rig.matrix_world @ bones_by_suffix[f"{side} Forearm"].head
        wrist = rig.matrix_world @ bones_by_suffix[f"{side} Hand"].head
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        result[f"{side}_upper_arm_outward_component"] = (
            sign * upper_arm.normalized().x
        )
        result[f"{side}_elbow_outward_clearance_m"] = sign * (
            elbow.x - shoulder.x
        )
        result[f"{side}_wrist_outward_clearance_m"] = sign * (
            wrist.x - shoulder.x
        )
        result[f"{side}_wrist_outward_clearance_ratio"] = (
            result[f"{side}_wrist_outward_clearance_m"]
            / (upper_arm.length + forearm.length)
        )
    return result


def elbow_angles_degrees(
    bones_by_suffix: dict[str, bpy.types.PoseBone],
) -> dict[str, float]:
    result = {}
    for side in ("L", "R"):
        names = [f"{side} {part}" for part in ("UpperArm", "Forearm", "Hand")]
        try:
            upper_arm, forearm, hand = (bones_by_suffix[name] for name in names)
        except KeyError as exc:
            raise RuntimeError(f"Required arm bone is missing: {exc.args[0]}") from exc
        shoulder = upper_arm.head
        elbow = forearm.head
        wrist = hand.head
        result[side] = math.degrees((shoulder - elbow).angle(wrist - elbow))
    return result


def direction_from_down(
    angle_degrees: float, outward_component: float = 0.0
) -> Vector:
    angle = math.radians(angle_degrees)
    sagittal_scale = math.sqrt(1.0 - outward_component**2)
    return Vector(
        (
            outward_component,
            math.sin(angle) * sagittal_scale,
            -math.cos(angle) * sagittal_scale,
        )
    )


def direction_from_forward(angle_degrees: float) -> Vector:
    angle = math.radians(angle_degrees)
    return Vector((0.0, -math.cos(angle), -math.sin(angle)))


def direction_from_up(lean_degrees: float) -> Vector:
    angle = math.radians(lean_degrees)
    return Vector((0.0, -math.sin(angle), math.cos(angle)))


def joint_world_direction(
    rig: bpy.types.Object,
    parent: bpy.types.PoseBone,
    child: bpy.types.PoseBone,
) -> Vector:
    direction = child.head - parent.head
    if direction.length_squared < 1e-12:
        raise RuntimeError(
            f"Coincident joints prevent retargeting {parent.name} -> {child.name}"
        )
    return (rig.matrix_world.to_3x3() @ direction).normalized()


def align_joint_vector(
    rig: bpy.types.Object,
    parent: bpy.types.PoseBone,
    child: bpy.types.PoseBone,
    desired_world_direction: Vector,
) -> None:
    """Aim an anatomical joint segment while preserving Rocketbox bone roll."""
    current_armature_direction = (child.head - parent.head).normalized()
    desired_armature_direction = (
        rig.matrix_world.to_3x3().inverted() @ desired_world_direction
    ).normalized()
    correction = current_armature_direction.rotation_difference(
        desired_armature_direction
    )
    corrected_rotation = correction @ parent.matrix.to_quaternion()
    apply_armature_rotation_at_joint(parent, corrected_rotation)


def reset_pose(rig: bpy.types.Object) -> None:
    for pose_bone in rig.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = Quaternion()
        pose_bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def required_reference_gait_bones() -> set[str]:
    return {
        "Spine",
        "Spine1",
        "Spine2",
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


def apply_reference_gait_pose(
    rig: bpy.types.Object,
    bones_by_suffix: dict[str, bpy.types.PoseBone],
    specification: dict,
    sex: str,
) -> None:
    reset_pose(rig)
    torso_lean = float(specification["torso_lean"])
    if sex == "male":
        torso_target = torso_lean + MALE_TORSO_LEAN_CORRECTION_DEGREES
    elif sex == "female":
        torso_target = torso_lean
    else:
        raise ValueError(f"Unsupported Rocketbox sex for gait authoring: {sex}")
    align_joint_vector(
        rig,
        bones_by_suffix["Spine"],
        bones_by_suffix["Spine1"],
        direction_from_up(torso_target),
    )

    for side in ("L", "R"):
        sign = 1.0 if side == "L" else -1.0
        thigh, calf, foot, upper_arm, forearm = specification[side]
        align_joint_vector(
            rig,
            bones_by_suffix[f"{side} Thigh"],
            bones_by_suffix[f"{side} Calf"],
            direction_from_down(thigh, sign * LEG_OUTWARD_COMPONENT),
        )
        align_joint_vector(
            rig,
            bones_by_suffix[f"{side} Calf"],
            bones_by_suffix[f"{side} Foot"],
            direction_from_down(calf, sign * LEG_OUTWARD_COMPONENT),
        )
        align_joint_vector(
            rig,
            bones_by_suffix[f"{side} Foot"],
            bones_by_suffix[f"{side} Toe0"],
            direction_from_forward(foot),
        )
        align_joint_vector(
            rig,
            bones_by_suffix[f"{side} UpperArm"],
            bones_by_suffix[f"{side} Forearm"],
            direction_from_down(upper_arm, sign * ARM_OUTWARD_COMPONENT),
        )
        align_joint_vector(
            rig,
            bones_by_suffix[f"{side} Forearm"],
            bones_by_suffix[f"{side} Hand"],
            direction_from_down(forearm, sign * ARM_OUTWARD_COMPONENT),
        )


def new_rig_action(rig: bpy.types.Object, name: str) -> bpy.types.Action:
    action = bpy.data.actions.new(name)
    rig.animation_data_create()
    rig.animation_data.action = action
    if len(action.slots) == 0:
        action.slots.new("OBJECT", rig.name)
    if len(action.slots) != 1:
        raise RuntimeError(
            f"Expected one object slot in {name}, found {len(action.slots)}"
        )
    rig.animation_data.action_slot = action.slots[0]
    return action


def basis_samples(
    rig: bpy.types.Object,
    previous_quaternions: dict[str, Quaternion],
) -> dict[str, tuple[Vector, Quaternion, Vector]]:
    samples = {}
    for pose_bone in rig.pose.bones:
        location, rotation, scale = pose_bone.matrix_basis.decompose()
        rotation.normalize()
        previous = previous_quaternions.get(pose_bone.name)
        if previous is not None and previous.dot(rotation) < 0.0:
            rotation.negate()
        previous_quaternions[pose_bone.name] = rotation.copy()
        samples[pose_bone.name] = (
            location.copy(),
            rotation.copy(),
            scale.copy(),
        )
    return samples


def key_basis_samples(
    rig: bpy.types.Object,
    frame: int,
    samples: dict[str, tuple[Vector, Quaternion, Vector]],
) -> None:
    for name, (location, rotation, scale) in samples.items():
        pose_bone = rig.pose.bones[name]
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.location = location
        pose_bone.rotation_quaternion = rotation
        pose_bone.scale = scale
        pose_bone.keyframe_insert(
            data_path="location", frame=frame, group=name
        )
        pose_bone.keyframe_insert(
            data_path="rotation_quaternion", frame=frame, group=name
        )
        pose_bone.keyframe_insert(
            data_path="scale", frame=frame, group=name
        )


def reference_pose_direction_errors(
    rig: bpy.types.Object,
    bones_by_suffix: dict[str, bpy.types.PoseBone],
    specification: dict,
    sex: str,
) -> dict[str, float]:
    torso_lean = float(specification["torso_lean"])
    torso_target = (
        torso_lean + MALE_TORSO_LEAN_CORRECTION_DEGREES
        if sex == "male"
        else torso_lean
    )
    desired_segments = {
        "torso": (
            "Spine",
            "Spine1",
            direction_from_up(torso_target),
        ),
        "chest": (
            "Spine1",
            "Spine2",
            direction_from_up(
                torso_lean + FEMALE_CHEST_AXIS_OFFSET_DEGREES
            ),
        ),
    }
    for side in ("L", "R"):
        sign = 1.0 if side == "L" else -1.0
        thigh, calf, foot, upper_arm, forearm = specification[side]
        desired_segments.update(
            {
                f"{side}_thigh": (
                    f"{side} Thigh",
                    f"{side} Calf",
                    direction_from_down(thigh, sign * LEG_OUTWARD_COMPONENT),
                ),
                f"{side}_calf": (
                    f"{side} Calf",
                    f"{side} Foot",
                    direction_from_down(calf, sign * LEG_OUTWARD_COMPONENT),
                ),
                f"{side}_foot": (
                    f"{side} Foot",
                    f"{side} Toe0",
                    direction_from_forward(foot),
                ),
                f"{side}_upper_arm": (
                    f"{side} UpperArm",
                    f"{side} Forearm",
                    direction_from_down(
                        upper_arm, sign * ARM_OUTWARD_COMPONENT
                    ),
                ),
                f"{side}_forearm": (
                    f"{side} Forearm",
                    f"{side} Hand",
                    direction_from_down(forearm, sign * ARM_OUTWARD_COMPONENT),
                ),
            }
        )

    return {
        name: math.degrees(
            joint_world_direction(
                rig, bones_by_suffix[parent_name], bones_by_suffix[child_name]
            ).angle(desired.normalized())
        )
        for name, (parent_name, child_name, desired) in desired_segments.items()
    }


def bake_reference_gait(
    target_rig: bpy.types.Object,
    sex: str,
) -> tuple[bpy.types.Action, dict]:
    """Author the measured seed_42 gait on Rocketbox anatomical joint chains."""
    bones_by_suffix = pose_bones_by_suffix(target_rig)
    missing = sorted(required_reference_gait_bones() - bones_by_suffix.keys())
    if missing:
        raise RuntimeError(
            f"Cannot author reference gait; required bone suffixes are missing: {missing}"
        )

    scene = bpy.context.scene
    key_action = new_rig_action(target_rig, "PED_WALK_REFERENCE_KEYS")
    previous_quaternions: dict[str, Quaternion] = {}
    for frame in sorted(REFERENCE_GAIT_KEYFRAMES):
        scene.frame_set(frame)
        apply_reference_gait_pose(
            target_rig,
            bones_by_suffix,
            REFERENCE_GAIT_KEYFRAMES[frame],
            sex,
        )
        key_basis_samples(
            target_rig,
            frame,
            basis_samples(target_rig, previous_quaternions),
        )

    for curve in key_action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"
            point.handle_left_type = "AUTO_CLAMPED"
            point.handle_right_type = "AUTO_CLAMPED"
    key_action.use_frame_range = True
    key_action.frame_start = FRAME_START
    key_action.frame_end = FRAME_END_CLOSURE

    samples = {}
    previous_quaternions = {}
    for frame in range(FRAME_START, FRAME_END_CLOSURE + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples[frame] = basis_samples(target_rig, previous_quaternions)

    walk_action = new_rig_action(target_rig, "PED_WALK_ACTION")
    for frame, frame_samples in samples.items():
        key_basis_samples(target_rig, frame, frame_samples)
    for curve in walk_action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    walk_action.use_frame_range = True
    walk_action.frame_start = FRAME_START
    walk_action.frame_end = FRAME_END_CLOSURE
    bpy.data.actions.remove(key_action)

    knee_angles = {}
    elbow_angles = {}
    arm_clearances = {}
    reference_errors = {}
    for frame in range(FRAME_START, FRAME_END_CLOSURE + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        knee_angles[frame] = knee_angles_degrees(bones_by_suffix)
        elbow_angles[frame] = elbow_angles_degrees(bones_by_suffix)
        arm_clearances[frame] = arm_clearance_diagnostics(
            target_rig, bones_by_suffix
        )
        if frame in REFERENCE_GAIT_KEYFRAMES:
            reference_errors[frame] = reference_pose_direction_errors(
                target_rig,
                bones_by_suffix,
                REFERENCE_GAIT_KEYFRAMES[frame],
                sex,
            )

    minimum_knee_angle = min(
        value for values in knee_angles.values() for value in values.values()
    )
    maximum_knee_angle = max(
        value for values in knee_angles.values() for value in values.values()
    )
    minimum_elbow_angle = min(
        value for values in elbow_angles.values() for value in values.values()
    )
    maximum_elbow_angle = max(
        value for values in elbow_angles.values() for value in values.values()
    )
    maximum_reference_direction_error = max(
        value for values in reference_errors.values() for value in values.values()
    )
    minimum_upper_arm_outward_component = min(
        values[f"{side}_upper_arm_outward_component"]
        for values in arm_clearances.values()
        for side in ("L", "R")
    )
    minimum_elbow_outward_clearance = min(
        values[f"{side}_elbow_outward_clearance_m"]
        for values in arm_clearances.values()
        for side in ("L", "R")
    )
    minimum_wrist_outward_clearance = min(
        values[f"{side}_wrist_outward_clearance_m"]
        for values in arm_clearances.values()
        for side in ("L", "R")
    )

    if maximum_reference_direction_error > MAX_REFERENCE_DIRECTION_ERROR_DEGREES:
        raise RuntimeError(
            "Rocketbox pose does not reproduce the seed_42 reference directions: "
            f"maximum error={maximum_reference_direction_error:.6f} degrees"
        )
    if not (
        MIN_ANATOMICAL_KNEE_ANGLE_DEGREES
        <= minimum_knee_angle
        <= maximum_knee_angle
        <= MAX_ANATOMICAL_KNEE_ANGLE_DEGREES
    ):
        raise RuntimeError(
            "Procedural gait produced a non-anatomical knee range: "
            f"{minimum_knee_angle:.3f}..{maximum_knee_angle:.3f} degrees"
        )
    if not (
        MIN_ANATOMICAL_ELBOW_ANGLE_DEGREES
        <= minimum_elbow_angle
        <= maximum_elbow_angle
        <= MAX_ANATOMICAL_ELBOW_ANGLE_DEGREES
    ):
        raise RuntimeError(
            "Procedural gait produced a non-anatomical elbow range: "
            f"{minimum_elbow_angle:.3f}..{maximum_elbow_angle:.3f} degrees"
        )
    if (
        minimum_upper_arm_outward_component
        < MIN_ANIMATED_ARM_OUTWARD_COMPONENT - ARM_CLEARANCE_TOLERANCE
    ):
        raise RuntimeError(
            "Upper-arm torso clearance was not preserved: "
            f"minimum outward component={minimum_upper_arm_outward_component:.6f}"
        )
    if (
        minimum_elbow_outward_clearance <= 0.0
        or minimum_wrist_outward_clearance <= 0.0
    ):
        raise RuntimeError(
            "An elbow or wrist crossed inward through the shoulder line: "
            f"minimum elbow clearance={minimum_elbow_outward_clearance:.6f} m, "
            f"minimum wrist clearance={minimum_wrist_outward_clearance:.6f} m"
        )

    closure_location_error = max(
        (samples[FRAME_START][name][0] - samples[FRAME_END_CLOSURE][name][0]).length
        for name in samples[FRAME_START]
    )
    closure_rotation_error = max(
        samples[FRAME_START][name][1]
        .rotation_difference(samples[FRAME_END_CLOSURE][name][1])
        .angle
        for name in samples[FRAME_START]
    )
    if closure_location_error > 1e-7 or closure_rotation_error > 1e-7:
        raise RuntimeError(
            "Reference gait does not close cleanly: "
            f"location={closure_location_error:.9f}, "
            f"rotation={closure_rotation_error:.9f} radians"
        )

    diagnostic_frames = sorted(set(POSE_FRAMES.values()))
    scene.frame_set(FRAME_START)
    return walk_action, {
        "gait_method": "seed_42_anatomical_joint_direction_reference",
        "torso_normalization": (
            "male_bind_preserving_chest_alignment"
            if sex == "male"
            else "native_female_bind_pose"
        ),
        "female_chest_axis_offset_degrees": (
            FEMALE_CHEST_AXIS_OFFSET_DEGREES
        ),
        "male_chest_axis_offset_degrees": MALE_CHEST_AXIS_OFFSET_DEGREES,
        "male_torso_lean_correction_degrees": (
            MALE_TORSO_LEAN_CORRECTION_DEGREES
        ),
        "reference_gait_version": REFERENCE_GAIT_VERSION,
        "reference_keyframes": sorted(REFERENCE_GAIT_KEYFRAMES),
        "baked_pose_frames": FRAME_END_CLOSURE - FRAME_START + 1,
        "baked_fcurves": len(walk_action.fcurves),
        "maximum_reference_direction_error_degrees": (
            maximum_reference_direction_error
        ),
        "minimum_knee_angle_degrees": minimum_knee_angle,
        "maximum_knee_angle_degrees": maximum_knee_angle,
        "minimum_elbow_angle_degrees": minimum_elbow_angle,
        "maximum_elbow_angle_degrees": maximum_elbow_angle,
        "minimum_upper_arm_outward_component": (
            minimum_upper_arm_outward_component
        ),
        "minimum_elbow_outward_clearance_m": minimum_elbow_outward_clearance,
        "minimum_wrist_outward_clearance_m": minimum_wrist_outward_clearance,
        "closure_location_error_m": closure_location_error,
        "closure_rotation_error_radians": closure_rotation_error,
        "pose_knee_angles_degrees": {
            frame: knee_angles[frame] for frame in diagnostic_frames
        },
        "pose_elbow_angles_degrees": {
            frame: elbow_angles[frame] for frame in diagnostic_frames
        },
        "pose_arm_clearance": {
            frame: arm_clearances[frame] for frame in diagnostic_frames
        },
    }


def install_walk_action(
    target_rig: bpy.types.Object,
    sex: str,
) -> tuple[bpy.types.Action, dict]:
    # Preserve the avatar FBX's import transform. World-space anatomical
    # directions are converted through this matrix while the pose is authored.
    target_matrix_basis = target_rig.matrix_basis.copy()
    target_rig.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    target_rig.matrix_basis = target_matrix_basis
    walk_action, diagnostics = bake_reference_gait(target_rig, sex)

    for action in list(bpy.data.actions):
        if action != walk_action:
            bpy.data.actions.remove(action)
    return walk_action, diagnostics


def move_to_asset_collection(objects: Sequence[bpy.types.Object]) -> bpy.types.Collection:
    scene = bpy.context.scene
    collection = bpy.data.collections.new("PEDESTRIAN_ASSET")
    scene.collection.children.link(collection)
    for obj in objects:
        for current in list(obj.users_collection):
            current.objects.unlink(obj)
        collection.objects.link(obj)
    return collection


def create_stationary_root(
    deform_rig: bpy.types.Object,
    collection: bpy.types.Collection,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    wrapper_data = bpy.data.armatures.new("PED_RIG_ARMATURE")
    wrapper = bpy.data.objects.new("PED_RIG", wrapper_data)
    collection.objects.link(wrapper)

    ground = bpy.data.objects.new("PED_GROUND_CORRECTION", None)
    collection.objects.link(ground)
    ground.parent = wrapper
    ground.matrix_parent_inverse = Matrix.Identity(4)

    deform_rig.name = "ROCKETBOX_DEFORM_RIG"
    deform_rig.data.name = "ROCKETBOX_DEFORM_ARMATURE"
    deform_rig.parent = ground
    deform_rig.matrix_parent_inverse = Matrix.Identity(4)
    return wrapper, ground


def evaluated_bounds(meshes: Sequence[bpy.types.Object]) -> tuple[float, float]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum = float("inf")
    maximum = float("-inf")
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            matrix = evaluated.matrix_world
            for vertex in evaluated_mesh.vertices:
                z = (matrix @ vertex.co).z
                minimum = min(minimum, z)
                maximum = max(maximum, z)
        finally:
            evaluated.to_mesh_clear()
    if minimum == float("inf"):
        raise RuntimeError("No evaluated mesh vertices found")
    return minimum, maximum


def add_ground_correction(
    ground: bpy.types.Object, meshes: Sequence[bpy.types.Object]
) -> tuple[bpy.types.Action, dict]:
    scene = bpy.context.scene
    ground.location = (0.0, 0.0, 0.0)
    corrections = {}
    uncorrected_bounds = {}
    for frame in range(FRAME_START, FRAME_END_CLOSURE + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        z_min, z_max = evaluated_bounds(meshes)
        corrections[frame] = -z_min
        uncorrected_bounds[frame] = (z_min, z_max)

    action = bpy.data.actions.new("PED_GROUND_ACTION")
    ground.animation_data_create()
    ground.animation_data.action = action
    for frame, correction in corrections.items():
        ground.location.z = correction
        ground.keyframe_insert(data_path="location", index=2, frame=frame, group="Ground contact")
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"
            point.handle_left_type = "AUTO_CLAMPED"
            point.handle_right_type = "AUTO_CLAMPED"
    action.use_frame_range = True
    action.frame_start = FRAME_START
    action.frame_end = FRAME_END_CLOSURE

    corrected_bounds = {}
    for frame in range(FRAME_START, WALK_CYCLE_FRAMES + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        corrected_bounds[frame] = evaluated_bounds(meshes)

    ground_error = max(abs(bounds[0]) for bounds in corrected_bounds.values())
    character_height = max(bounds[1] - bounds[0] for bounds in corrected_bounds.values())
    closure_error = abs(corrections[FRAME_START] - corrections[FRAME_END_CLOSURE])
    return action, {
        "character_height_m": character_height,
        "maximum_ground_error_m": ground_error,
        "ground_correction_closure_error_m": closure_error,
        "ground_correction_min_m": min(corrections.values()),
        "ground_correction_max_m": max(corrections.values()),
        "uncorrected_ground_min_m": min(bounds[0] for bounds in uncorrected_bounds.values()),
        "uncorrected_ground_max_m": max(bounds[0] for bounds in uncorrected_bounds.values()),
    }


def action_xy_displacement(rig: bpy.types.Object) -> float:
    scene = bpy.context.scene
    points = []
    for frame in range(FRAME_START, WALK_CYCLE_FRAMES + 1):
        scene.frame_set(frame)
        points.append((float(rig.location.x), float(rig.location.y)))
    x0, y0 = points[0]
    return max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for x, y in points)


def object_transform_diagnostics(rig: bpy.types.Object) -> dict:
    """Verify that gait animation does not tilt, turn, translate, or scale the avatar object."""
    scene = bpy.context.scene
    original_frame = scene.frame_current
    samples = []
    try:
        for frame in range(FRAME_START, WALK_CYCLE_FRAMES + 1):
            scene.frame_set(frame)
            location, rotation, scale = rig.matrix_basis.decompose()
            samples.append((location.copy(), rotation.copy(), scale.copy()))
    finally:
        scene.frame_set(original_frame)

    reference_location, reference_rotation, reference_scale = samples[0]
    location_displacement = max(
        (location - reference_location).length for location, _, _ in samples
    )
    rotation_displacement = max(
        min(
            reference_rotation.rotation_difference(rotation).angle,
            math.tau - reference_rotation.rotation_difference(rotation).angle,
        )
        for _, rotation, _ in samples
    )
    scale_displacement = max(
        (scale - reference_scale).length for _, _, scale in samples
    )
    return {
        "location_displacement_m": float(location_displacement),
        "rotation_displacement_radians": float(rotation_displacement),
        "scale_displacement": float(scale_displacement),
        "static_location": [float(value) for value in reference_location],
        "static_rotation_quaternion": [float(value) for value in reference_rotation],
        "static_scale": [float(value) for value in reference_scale],
    }


def rename_meshes(meshes: Sequence[bpy.types.Object]) -> None:
    ordered = sorted(meshes, key=lambda obj: len(obj.data.vertices), reverse=True)
    for index, mesh in enumerate(ordered):
        name = "PED_BODY" if index == 0 else f"PED_BODY_PART_{index:02d}"
        mesh.name = name
        mesh.data.name = name + "_MESH"


def pack_check(meshes: Sequence[bpy.types.Object]) -> None:
    unpacked = sorted(
        {
            node.image.name
            for node in image_nodes(meshes)
            if node.image.source == "FILE" and node.image.packed_file is None
        }
    )
    if unpacked:
        raise RuntimeError(f"Unpacked resources remain: {', '.join(unpacked)}")


def existing_output_is_current(
    blend_path: Path,
    sidecar_path: Path,
    asset: SourceAsset,
    revision: str,
) -> bool:
    if not blend_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        normalize_pedestrian_geometry(
            metadata.get("actor_geometry"),
            expected_asset_id=metadata.get("id"),
        )
    except ValueError:
        return False
    return (
        blend_path.stat().st_size > 0
        and metadata.get("animation_version") == CONVERTER_VERSION
        and metadata.get("source_revision") == revision
        and metadata.get("source_character_id") == asset.name
        and metadata.get("resources_packed") is True
        and metadata.get("walk_cycle_frames") == WALK_CYCLE_FRAMES
    )


def build_one(
    asset: SourceAsset,
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

    reset_blender()
    print(f"BUILD {asset.name} ({asset.category}, {asset.sex})", flush=True)
    character_objects = imported_objects(asset.fbx)
    deform_rig = exactly_one_armature(character_objects, asset.fbx)
    meshes = [obj for obj in character_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No meshes imported from {asset.fbx}")

    texture_sources = repair_and_pack_textures(meshes, asset.texture_dir, rocketbox_root)
    reference_gait_script = Path(__file__).with_name("create_walk_cycle.py").resolve()
    if not reference_gait_script.is_file():
        raise FileNotFoundError(
            f"Reference gait authoring script is missing: {reference_gait_script}"
        )
    walk_action, motion_diagnostics = install_walk_action(deform_rig, asset.sex)

    # Drop imported footsteps/helper empties; the skinned meshes only need the armature.
    for obj in list(character_objects):
        if obj.type not in {"MESH", "ARMATURE"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    character_objects = [obj for obj in bpy.context.scene.objects]
    collection = move_to_asset_collection(character_objects)
    wrapper, ground = create_stationary_root(deform_rig, collection)
    rename_meshes(meshes)
    ground_action, ground_diagnostics = add_ground_correction(ground, meshes)
    gait_timing = navsim_gait_timing()
    transform_diagnostics = object_transform_diagnostics(deform_rig)

    scene = bpy.context.scene
    scene.frame_set(FRAME_START)
    scene["character_height_m"] = float(ground_diagnostics["character_height_m"])
    scene["source"] = "Microsoft Rocketbox"
    scene["source_revision"] = revision
    scene["license"] = "MIT"
    scene["conversion_version"] = CONVERTER_VERSION
    scene["navsim_gait_timing_json"] = json.dumps(gait_timing, sort_keys=True)

    if wrapper.animation_data is not None and wrapper.animation_data.action is not None:
        raise RuntimeError("PED_RIG must not have object animation")
    xy_displacement = action_xy_displacement(deform_rig)
    if xy_displacement > 1e-6:
        raise RuntimeError(f"Horizontal root motion remains: {xy_displacement:.9f} m")
    if transform_diagnostics["location_displacement_m"] > 1e-7:
        raise RuntimeError(
            "Avatar object translation is animated: "
            f"{transform_diagnostics['location_displacement_m']:.9f} m"
        )
    if transform_diagnostics["rotation_displacement_radians"] > 1e-7:
        raise RuntimeError(
            "Avatar object rotation is animated: "
            f"{transform_diagnostics['rotation_displacement_radians']:.9f} radians"
        )
    if transform_diagnostics["scale_displacement"] > 1e-7:
        raise RuntimeError(
            "Avatar object scale is animated: "
            f"{transform_diagnostics['scale_displacement']:.9f}"
        )
    if ground_diagnostics["maximum_ground_error_m"] > 0.002:
        raise RuntimeError(
            "Ground correction failed: "
            f"{ground_diagnostics['maximum_ground_error_m']:.6f} m"
        )
    height = float(ground_diagnostics["character_height_m"])
    if not 0.5 <= height <= 2.3:
        raise RuntimeError(f"Generated character height is unreasonable: {height:.3f} m")
    pack_check(meshes)
    actor_geometry, _per_frame_geometry = measure_gait_union_geometry(
        asset_id=asset.output_stem,
        renderer_root=wrapper.name,
        local_forward_axis="-Y",
        frame_start=FRAME_START,
        frame_end_exclusive=FRAME_START + WALK_CYCLE_FRAMES,
    )
    scene["pedestrian_geometry_profile_version"] = (
        PEDESTRIAN_GEOMETRY_PROFILE_VERSION
    )
    scene["pedestrian_geometry_json"] = json.dumps(
        actor_geometry, sort_keys=True
    )

    metadata = {
        "id": asset.output_stem,
        "asset_path": str(blend_path),
        "source": "Microsoft Rocketbox",
        "source_repository": "https://github.com/microsoft/Microsoft-Rocketbox",
        "source_revision": revision,
        "source_category": asset.category,
        "source_character_id": asset.name,
        "source_character_fbx": str(asset.fbx.relative_to(rocketbox_root)),
        "source_character_fbx_sha256": sha256_file(asset.fbx),
        "source_animation": "seed_42 procedural anatomical reference gait",
        "source_animation_version": REFERENCE_GAIT_VERSION,
        "source_animation_reference_script": str(
            reference_gait_script.relative_to(Path(__file__).resolve().parents[2])
        ),
        "source_animation_reference_script_sha256": sha256_file(
            reference_gait_script
        ),
        "source_texture_files": texture_sources,
        "license": "MIT",
        "license_file": "../ROCKETBOX_LICENSE.md",
        "procedural_reference_gait": True,
        "character_height_m": height,
        "walk_action": walk_action.name,
        "ground_action": ground_action.name,
        "animation_version": CONVERTER_VERSION,
        "animation_frame_start": FRAME_START,
        "walk_cycle_frames": WALK_CYCLE_FRAMES,
        "pose_frames": POSE_FRAMES,
        "navsim_gait_timing": gait_timing,
        "local_forward_axis": "-Y",
        "actor_geometry": actor_geometry,
        "ground_contact_z_m": 0.0,
        "root_motion": "stationary_xy",
        "resources_packed": True,
        "deformation_rig": deform_rig.name,
        "renderer_root": wrapper.name,
        "mesh_objects": sorted(mesh.name for mesh in meshes),
        "mesh_vertices": sum(len(mesh.data.vertices) for mesh in meshes),
        "deformation_bones": len(deform_rig.data.bones),
        "motion_diagnostics": motion_diagnostics,
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
        raise RuntimeError(f"Blender did not save candidate asset: {temporary_blend}")
    os.replace(temporary_blend, blend_path)
    atomic_json(sidecar_path, metadata)
    print(
        f"DONE {asset.name}: height={height:.3f}m vertices={metadata['mesh_vertices']} "
        f"output={blend_path}",
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
    assets = discover_assets(args.rocketbox_root)
    selected = select_assets(assets, args.asset, args.limit)
    print(
        f"ROCKETBOX_SELECTION total={len(assets)} selected={len(selected)} "
        f"output={args.output_dir}",
        flush=True,
    )
    for asset in selected:
        print(f"SELECT {asset.category}/{asset.name} ({asset.sex})", flush=True)
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pruned_outputs = prune_excluded_generated_outputs(args.output_dir, assets)
    license_source = args.rocketbox_root / "LICENSE.md"
    if not license_source.is_file():
        raise FileNotFoundError(f"Rocketbox MIT license file missing: {license_source}")
    shutil.copyfile(license_source, args.output_dir.parent / "ROCKETBOX_LICENSE.md")
    revision = git_revision(args.rocketbox_root)

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
        except Exception as exc:  # Continue so one malformed source does not waste the batch.
            failure = {
                "asset": asset.name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"FAILED {asset.name}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()

    summary = {
        "converter_version": CONVERTER_VERSION,
        "rocketbox_root": str(args.rocketbox_root),
        "rocketbox_revision": revision,
        "output_dir": str(args.output_dir),
        "selected_count": len(selected),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "skipped_count": sum(result["status"] == "skipped" for result in results),
        "failed_count": len(failures),
        "pruned_excluded_output_count": len(pruned_outputs),
        "pruned_excluded_outputs": pruned_outputs,
        "results": results,
        "failures": failures,
    }
    atomic_json(args.output_dir.parent / "generation_manifest.json", summary)
    print(
        "ROCKETBOX_GENERATION_COMPLETE "
        f"completed={summary['completed_count']} skipped={summary['skipped_count']} "
        f"failed={summary['failed_count']} pruned={len(pruned_outputs)} "
        f"output={args.output_dir}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
