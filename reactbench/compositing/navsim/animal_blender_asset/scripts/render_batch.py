#!/usr/bin/env python3
"""Quadruped contact and standing adapter for the pedestrian renderer."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PEDESTRIAN_RENDERER = (
    HERE.parents[1] / "compositor" / "blender_asset" / "scripts" / "render_batch.py"
).resolve()
spec = importlib.util.spec_from_file_location(
    "navsim_pedestrian_render_batch",
    PEDESTRIAN_RENDERER,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load the pedestrian renderer: {PEDESTRIAN_RENDERER}")
pedestrian = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pedestrian)


CONTACT_GROUPS = {
    "left_front": {"side_token": "l", "parts": ("hand", "finger")},
    "right_front": {"side_token": "r", "parts": ("hand", "finger")},
    "left_hind": {"side_token": "l", "parts": ("foot", "toe")},
    "right_hind": {"side_token": "r", "parts": ("foot", "toe")},
}
QUADRUPED_CONTACT_PROFILES = frozenset(
    {
        "quadruped_camel",
        "quadruped_horse",
        "quadruped_cow",
        "quadruped_deer",
        "quadruped_sheep",
    }
)
# Must match animal_composite_navsim_blender.QUADRUPED_STANDING_POSE_VERSION.
QUADRUPED_STANDING_POSE_VERSION = "rocketbox_quadruped_rest_pose_v1"


def contact_vertex_group_indices(mesh_object, spec):
    side_token = str(spec["side_token"])
    prefixes = tuple(str(value) for value in spec["parts"])
    selected = []
    for group in mesh_object.vertex_groups:
        tokens = pedestrian.normalized_group_name(group.name).split()
        if side_token not in tokens:
            continue
        if any(
            token.startswith(prefix)
            for token in tokens
            for prefix in prefixes
        ):
            selected.append((group.index, group.name))
    return selected


def collect_quadruped_contact_geometry(scene, camera, width, height, job):
    profile = str(job.get("contact_profile", ""))
    if profile not in QUADRUPED_CONTACT_PROFILES:
        raise ValueError(
            "Quadruped renderer requires a contact_profile in "
            f"{sorted(QUADRUPED_CONTACT_PROFILES)}, got {profile!r}"
        )
    receiver_z = float(job["receiver_plane_z_m"])
    contact_distance = float(job["foot_contact_distance_m"])
    relative_distance = float(job["foot_stance_relative_distance_m"])
    sole_band = float(job["foot_sole_band_m"])
    depsgraph = pedestrian.bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    per_contact_points = {name: [] for name in CONTACT_GROUPS}
    per_contact_groups = {name: [] for name in CONTACT_GROUPS}

    for mesh_object in pedestrian.bpy.data.objects:
        if mesh_object.type != "MESH" or not mesh_object.vertex_groups:
            continue
        evaluated = mesh_object.evaluated_get(depsgraph)
        evaluated_vertices = evaluated.data.vertices
        for contact_name, group_spec in CONTACT_GROUPS.items():
            groups = contact_vertex_group_indices(mesh_object, group_spec)
            if not groups:
                continue
            group_indices = {index for index, _name in groups}
            per_contact_groups[contact_name].extend(
                f"{mesh_object.name}:{name}" for _index, name in groups
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
                per_contact_points[contact_name].append(
                    evaluated.matrix_world
                    @ evaluated_vertices[vertex.index].co
                )

    contacts = {}
    for contact_name in CONTACT_GROUPS:
        points = per_contact_points[contact_name]
        if not points:
            contacts[contact_name] = {
                "contact": False,
                "status": "contact_vertex_groups_unavailable",
                "group_names": per_contact_groups[contact_name],
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
            pedestrian.project_world_points(
                scene,
                camera,
                sole_points,
                width,
                height,
            )
            if in_contact
            else []
        )
        contacts[contact_name] = {
            "contact": bool(in_contact and projected),
            "status": "measured_from_evaluated_contact_vertices",
            "group_names": sorted(set(per_contact_groups[contact_name])),
            "weighted_vertex_count": len(points),
            "minimum_world_z_m": minimum_z,
            "minimum_receiver_distance_m": signed_distance,
            "sole_band_m": sole_band,
            "ideal_contact_points_px": projected,
        }

    measured_distances = [
        abs(float(details["minimum_receiver_distance_m"]))
        for details in contacts.values()
        if details.get("minimum_receiver_distance_m") is not None
    ]
    stance_reference = min(measured_distances) if measured_distances else None
    if stance_reference is not None:
        for details in contacts.values():
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
        "method": "evaluated_quadruped_contact_vertex_groups_v1",
        "contact_profile": profile,
        "receiver_plane_z_m": receiver_z,
        "contact_distance_m": contact_distance,
        "stance_relative_distance_m": relative_distance,
        "stance_reference_distance_m": stance_reference,
        # Keep the legacy key because the frozen downstream rasterizer already
        # treats its contents generically; it now contains four paw records.
        "feet": contacts,
        "contact_sides": [
            name for name, details in contacts.items() if details["contact"]
        ],
    }


def configure_quadruped_standing_pose(actor_rig):
    """Hold the Rocketbox rest pose, a four-leg stance, and reground it.

    The human routine would pose biped limbs: quadruped rigs share the bone
    suffixes, so it would bend the legs like a person's without failing.
    """
    bpy = pedestrian.bpy
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
    pedestrian.reset_pose(deform_rig)

    ungrounded_minimum_z, mesh_count, vertex_count = (
        pedestrian.evaluated_actor_minimum_z(actor_rig)
    )
    actor_ground_z = float(
        (actor_rig.matrix_world @ pedestrian.Vector((0.0, 0.0, 0.0))).z
    )
    correction_world_z = float(
        (actor_rig.matrix_world.to_3x3() @ pedestrian.Vector((0.0, 0.0, 1.0))).z
    )
    if not math.isfinite(correction_world_z) or abs(correction_world_z) < 1e-9:
        raise RuntimeError("Standing ground-correction axis is degenerate")
    ground_correction.location.z += (
        actor_ground_z - ungrounded_minimum_z
    ) / correction_world_z
    bpy.context.view_layer.update()
    grounded_minimum_z, _mesh_count, _vertex_count = (
        pedestrian.evaluated_actor_minimum_z(actor_rig)
    )
    residual = grounded_minimum_z - actor_ground_z
    if not math.isfinite(residual) or abs(residual) > 1e-5:
        raise RuntimeError(
            f"Standing pose failed ground correction: residual={residual:.9f} m"
        )
    return {
        "version": QUADRUPED_STANDING_POSE_VERSION,
        "pose_source": "rocketbox_quadruped_rest_pose",
        "walk_action_disabled": True,
        "ground_action_disabled": True,
        "ground_correction_local_z_m": float(ground_correction.location.z),
        "ground_contact_residual_m": float(residual),
        "evaluated_mesh_count": int(mesh_count),
        "evaluated_vertex_count": int(vertex_count),
    }


pedestrian.collect_foot_contact_geometry = collect_quadruped_contact_geometry
pedestrian.configure_natural_standing_pose = configure_quadruped_standing_pose
pedestrian.STANDING_POSE_VERSION = QUADRUPED_STANDING_POSE_VERSION


if __name__ == "__main__":
    pedestrian.main()
