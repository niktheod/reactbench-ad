"""Author a stationary-root 32-frame urgent walk cycle at 24 fps.

Input: work/pedestrian_unanimated.blend (normally passed on Blender's CLI)
Output: assets/pedestrian_template.blend
Local character forward is -Y; the armature object and root bone never translate.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import sys

import bpy
from mathutils import Quaternion, Vector


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent  # measure_pedestrian_geometry; Blender does not add it to sys.path
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reactbench.scoring.pedestrian_geometry import PEDESTRIAN_GEOMETRY_PROFILE_VERSION  # noqa: E402
from measure_pedestrian_geometry import measure_gait_union_geometry  # noqa: E402

RIG_NAME = "PED_RIG"
ANIMATION_VERSION = "procedural_natural_walk_v5_collision_checked_8phase"
POSE_CORRECTION_PROFILES = (
    {"name": "original", "arm_fold_degrees": 24.0, "elbow_extension_degrees": 0.0},
    {"name": "reduced_fold", "arm_fold_degrees": 18.0, "elbow_extension_degrees": 0.0},
    {"name": "wide_arm_clearance", "arm_fold_degrees": 12.0, "elbow_extension_degrees": 0.0},
    {"name": "wide_extended", "arm_fold_degrees": 8.0, "elbow_extension_degrees": 6.0},
    {"name": "maximum_clearance", "arm_fold_degrees": 4.0, "elbow_extension_degrees": 10.0},
)


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "assets/pedestrian_template.blend")
    parser.add_argument(
        "--metadata-asset-path",
        type=Path,
        help="Final asset path recorded in the JSON when --output is a temporary candidate path.",
    )
    return parser.parse_args(argv)


def load_collision_module():
    path = ROOT / "scripts/pose_collision.py"
    spec = importlib.util.spec_from_file_location("pose_collision", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pose-collision module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_axis(bone, world_axis):
    return (bone.bone.matrix_local.to_3x3().inverted() @ Vector(world_axis)).normalized()


def set_world_axis_rotation(pbone, world_axis, degrees):
    pbone.rotation_mode = "QUATERNION"
    pbone.rotation_quaternion = Quaternion(local_axis(pbone, world_axis), math.radians(degrees))


def apply_world_axis_rotation(pbone, world_axis, degrees):
    rotation = Quaternion(local_axis(pbone, world_axis), math.radians(degrees))
    pbone.rotation_quaternion = rotation @ pbone.rotation_quaternion


def set_world_translation(pbone, world_offset):
    pbone.location = pbone.bone.matrix_local.to_3x3().inverted() @ Vector(world_offset)


def visible_min_z():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum = float("inf")
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        minimum = min(
            minimum,
            min((evaluated.matrix_world @ vertex.co).z for vertex in evaluated.data.vertices),
        )
    return minimum


def key_pose(rig, frame, spec, arm_fold_degrees, elbow_extension_degrees):
    bpy.context.scene.frame_set(frame)
    for pbone in rig.pose.bones:
        pbone.rotation_mode = "QUATERNION"
        pbone.rotation_quaternion = Quaternion()
        pbone.location = (0.0, 0.0, 0.0)

    # Split rotations over MPFB's two-segment long bones for smooth deformation.
    for side in ("L", "R"):
        hip = spec[f"hip_{side}"]
        knee = spec[f"knee_{side}"]
        ankle = spec[f"ankle_{side}"]
        arm = spec[f"arm_{side}"]
        elbow = spec[f"elbow_{side}"] + elbow_extension_degrees
        for name in (f"upperleg01.{side}", f"upperleg02.{side}"):
            set_world_axis_rotation(rig.pose.bones[name], (1, 0, 0), hip * 0.5)
        for name in (f"lowerleg01.{side}", f"lowerleg02.{side}"):
            set_world_axis_rotation(rig.pose.bones[name], (1, 0, 0), knee * 0.5)
        set_world_axis_rotation(rig.pose.bones[f"foot.{side}"], (1, 0, 0), ankle)
        for name in (f"upperarm01.{side}", f"upperarm02.{side}"):
            set_world_axis_rotation(rig.pose.bones[name], (1, 0, 0), arm * 0.5)
            # Fold MPFB's A-pose arms toward the torso after applying fore/aft
            # swing. The rest skeleton already bends both forearms forward, so
            # elbow values below counteract that bend rather than increasing it.
            fold = arm_fold_degrees if side == "L" else -arm_fold_degrees
            apply_world_axis_rotation(rig.pose.bones[name], (0, 1, 0), fold)
        for name in (f"lowerarm01.{side}", f"lowerarm02.{side}"):
            set_world_axis_rotation(rig.pose.bones[name], (1, 0, 0), elbow * 0.5)

    pelvis = rig.pose.bones["spine05"]
    for name in ("pelvis.L", "pelvis.R", "spine05"):
        set_world_translation(rig.pose.bones[name], (0, 0, spec["bob"]))
    set_world_axis_rotation(pelvis, (0, 0, 1), spec["pelvis_yaw"])
    set_world_axis_rotation(rig.pose.bones["spine03"], (0, 0, 1), -spec["pelvis_yaw"] * 0.65)
    set_world_axis_rotation(rig.pose.bones["spine02"], (1, 0, 0), spec["lean"])
    set_world_axis_rotation(rig.pose.bones["head"], (0, 0, 1), spec["head_yaw"])

    # Rotating the feet and long leg chains can move the shoe sole slightly
    # below or above the asset's z=0 convention. Correct the pelvis hierarchy at
    # every authored phase so interpolation starts from grounded key poses.
    bpy.context.view_layer.update()
    ground_correction = -visible_min_z()
    for name in ("pelvis.L", "pelvis.R", "spine05"):
        pbone = rig.pose.bones[name]
        pbone.location += pbone.bone.matrix_local.to_3x3().inverted() @ Vector((0, 0, ground_correction))
    bpy.context.view_layer.update()

    keyed = [
        "pelvis.L", "pelvis.R", "spine05", "spine03", "spine02", "head",
        *[f"{base}.{side}" for side in ("L", "R") for base in (
            "upperleg01", "upperleg02", "lowerleg01", "lowerleg02", "foot",
            "upperarm01", "upperarm02", "lowerarm01", "lowerarm02")],
    ]
    for name in keyed:
        pbone = rig.pose.bones[name]
        pbone.keyframe_insert("rotation_quaternion", frame=frame, group=name)
        if name in {"pelvis.L", "pelvis.R", "spine05"}:
            pbone.keyframe_insert("location", frame=frame, group=name)


def remove_walk_actions(rig):
    if rig.animation_data is not None:
        rig.animation_data.action = None
    for action in list(bpy.data.actions):
        if action.name.startswith("PED_WALK_ACTION"):
            bpy.data.actions.remove(action)


def author_action(rig, poses, profile):
    remove_walk_actions(rig)
    action = bpy.data.actions.new("PED_WALK_ACTION")
    rig.animation_data_create()
    rig.animation_data.action = action
    for frame, spec in poses.items():
        key_pose(
            rig,
            frame,
            spec,
            arm_fold_degrees=float(profile["arm_fold_degrees"]),
            elbow_extension_degrees=float(profile["elbow_extension_degrees"]),
        )

    for fcurve in action.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = "BEZIER"
        modifier = fcurve.modifiers.new("CYCLES")
        modifier.mode_before = "REPEAT"
        modifier.mode_after = "REPEAT"
    return action


def main():
    args = arguments()
    collision_module = load_collision_module()
    rig = bpy.data.objects.get(RIG_NAME)
    if rig is None or rig.type != "ARMATURE":
        raise RuntimeError("PED_RIG armature is missing; open work/pedestrian_unanimated.blend")
    body = bpy.data.objects.get("PED_BODY")
    if body is None or body.type != "MESH":
        raise RuntimeError("PED_BODY mesh is missing")

    # Contact, passing, opposite contact, passing, loop closure.
    poses = {
        1:  dict(hip_L=24, knee_L=4, ankle_L=-10, hip_R=-22, knee_R=18, ankle_R=9,
                 arm_L=-34, elbow_L=30, arm_R=34, elbow_R=26, bob=-0.0266,
                 pelvis_yaw=-3.5, lean=2.0, head_yaw=1.5),
        5:  dict(hip_L=9, knee_L=8, ankle_L=-5, hip_R=-6, knee_R=34, ankle_R=5,
                 arm_L=-18, elbow_L=29, arm_R=18, elbow_R=27, bob=-0.002,
                 pelvis_yaw=-1.0, lean=1.7, head_yaw=0.5),
        9:  dict(hip_L=-10, knee_L=30, ankle_L=-2, hip_R=12, knee_R=8, ankle_R=-4,
                 arm_L=12, elbow_L=28, arm_R=-12, elbow_R=28, bob=0.0188,
                 pelvis_yaw=1.5, lean=1.5, head_yaw=-1.0),
        13: dict(hip_L=-17, knee_L=20, ankle_L=6, hip_R=18, knee_R=6, ankle_R=-7,
                 arm_L=25, elbow_L=27, arm_R=-25, elbow_R=29, bob=-0.004,
                 pelvis_yaw=2.5, lean=1.7, head_yaw=-1.2),
        17: dict(hip_L=-22, knee_L=18, ankle_L=9, hip_R=24, knee_R=4, ankle_R=-10,
                 arm_L=34, elbow_L=26, arm_R=-34, elbow_R=30, bob=-0.0266,
                 pelvis_yaw=3.5, lean=2.0, head_yaw=-1.5),
        21: dict(hip_L=-6, knee_L=34, ankle_L=5, hip_R=9, knee_R=8, ankle_R=-5,
                 arm_L=18, elbow_L=27, arm_R=-18, elbow_R=29, bob=-0.002,
                 pelvis_yaw=1.0, lean=1.7, head_yaw=-0.5),
        25: dict(hip_L=12, knee_L=8, ankle_L=-4, hip_R=-10, knee_R=30, ankle_R=-2,
                 arm_L=-12, elbow_L=28, arm_R=12, elbow_R=28, bob=0.0188,
                 pelvis_yaw=-1.5, lean=1.5, head_yaw=1.0),
        29: dict(hip_L=18, knee_L=6, ankle_L=-7, hip_R=-17, knee_R=20, ankle_R=6,
                 arm_L=-25, elbow_L=29, arm_R=25, elbow_R=27, bob=-0.004,
                 pelvis_yaw=-2.5, lean=1.7, head_yaw=1.2),
        33: dict(hip_L=24, knee_L=4, ankle_L=-10, hip_R=-22, knee_R=18, ankle_R=9,
                 arm_L=-34, elbow_L=30, arm_R=34, elbow_R=26, bob=-0.0266,
                 pelvis_yaw=-3.5, lean=2.0, head_yaw=1.5),
    }
    action = None
    collision_report = None
    selected_profile = None
    collision_attempts = []
    for profile in POSE_CORRECTION_PROFILES:
        action = author_action(rig, poses, profile)
        report = collision_module.validate_pose_frames(rig, body)
        minimum_clearance = min(
            hand["minimum_clearance_m"]
            for frame_report in report["frames"]
            for hand in frame_report["hands"]
        )
        attempt = {
            "profile": dict(profile),
            "passed": bool(report["passed"]),
            "minimum_clearance_m": float(minimum_clearance),
        }
        collision_attempts.append(attempt)
        print("POSE_COLLISION_ATTEMPT=" + json.dumps(attempt, sort_keys=True))
        if report["passed"]:
            collision_report = report
            selected_profile = dict(profile)
            break

    if collision_report is None or selected_profile is None or action is None:
        remove_walk_actions(rig)
        raise RuntimeError(
            "Unable to produce collision-free designated poses: "
            + json.dumps(collision_attempts, sort_keys=True)
        )

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 33
    scene.render.fps = 24
    scene.frame_set(17)
    scene["pose_frames"] = "config/pose_frames.json"
    scene["animation_frame_start"] = 1
    scene["walk_cycle_frames"] = 32
    scene["animation_version"] = ANIMATION_VERSION
    scene["root_translation"] = "fixed at (0,0,0); only pelvis-relative bob is animated"
    scene["pose_collision_validation_json"] = json.dumps(collision_report, sort_keys=True)
    scene["pose_correction_profile_json"] = json.dumps(selected_profile, sort_keys=True)
    rig["root_location_reference"] = list(rig.pose.bones["root"].location)

    # Pack every texture so the result opens without MPFB or the asset repository.
    bpy.ops.file.pack_all()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_asset_path = (
        args.metadata_asset_path.resolve() if args.metadata_asset_path is not None else output
    )
    generation_config = json.loads(scene.get("generation_config_json", "{}"))
    asset_id = str(generation_config.get("id", output.stem))
    actor_geometry, _per_frame_geometry = measure_gait_union_geometry(
        asset_id=asset_id,
        renderer_root=RIG_NAME,
        local_forward_axis="-Y",
        frame_start=1,
        frame_end_exclusive=33,
    )
    scene["pedestrian_geometry_profile_version"] = (
        PEDESTRIAN_GEOMETRY_PROFILE_VERSION
    )
    scene["pedestrian_geometry_json"] = json.dumps(
        actor_geometry, sort_keys=True
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    metadata = {
        **generation_config,
        "id": asset_id,
        "asset_path": str(metadata_asset_path),
        "character_height_m": float(scene.get("character_height_m", 0.0)),
        "walk_action": action.name,
        "animation_version": ANIMATION_VERSION,
        "animation_frame_start": 1,
        "walk_cycle_frames": 32,
        "pose_frames": {"oldest": 24, "history_2": 15, "history_1": 6, "current": 29},
        "pose_correction_profile": selected_profile,
        "collision_validation": collision_report,
        "collision_attempts": collision_attempts,
        "local_forward_axis": "-Y",
        "actor_geometry": actor_geometry,
        "ground_contact_z_m": 0.0,
        "root_motion": "stationary",
        "resources_packed": True,
        "renderer_root": RIG_NAME,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"CREATED_WALK_CYCLE={output}")
    print("ROOT_TRANSLATION_DISPLACEMENT=0.0")


if __name__ == "__main__":
    main()
