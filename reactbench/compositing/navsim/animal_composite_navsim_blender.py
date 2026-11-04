#!/usr/bin/env python3
"""Animal adapter around the pedestrian compositor (the standing animals of the unseen-objects set).

The imported pedestrian module is kept free of animal logic.  This adapter
installs animal-specific behavior only in this process: measured geometry,
oriented collision envelopes, quadruped contact manifests and side-on standing
for standing-only animals (such as the camel). Animal runs use --standing and
--contact-only. Internal names keep the original "dog" prefix, which the
recorded render pipeline version shares.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import batch_composite_navsim_blender as pedestrian  # noqa: E402


DOG_RENDER_PIPELINE_VERSION = "photoreal_counterfactual_dog_v1"
DOG_GEOMETRY_PROFILE_VERSION = "quadruped_dog_geometry_v1"
DOG_COLLISION_WIDTH_MARGIN_M = 0.12
DOG_COLLISION_LENGTH_MARGIN_M = 0.28
DOG_COLLISION_HEIGHT_MARGIN_M = 0.15
DOG_CONTACT_GEOMETRY_VERSION = (
    "evaluated_quadruped_contact_vertex_groups_v1"
)
DOG_RENDERER = (
    HERE / "animal_blender_asset" / "scripts" / "render_batch.py"
).resolve()
ANIMAL_SPECIES = frozenset({"dog", "camel", "horse", "cow", "deer", "sheep"})
QUADRUPED_CONTACT_PROFILES = frozenset(f"quadruped_{species}" for species in ANIMAL_SPECIES)
# Sidecar profile written by create_rocketbox_standing_animal_assets.py.
STANDING_ANIMAL_SIDECAR_GEOMETRY_PROFILE = "standing_animal_rest_pose_bounds_v1"
MEASURED_GEOMETRY_PROFILE_VERSION = "quadruped_measured_bounds_geometry_v1"
# Must match animal_blender_asset/scripts/render_batch.py.
QUADRUPED_STANDING_POSE_VERSION = "rocketbox_quadruped_rest_pose_v1"
STANDING_SIDE_ON_FACING_POLICY = (
    "perpendicular_to_future_ego_forward_seeded_side_v1"
)
STANDING_SIDE_HASH_NAMESPACE = "standing_facing_side"


_ORIGINAL_LOAD_ASSET = pedestrian.load_asset
_ORIGINAL_ASSET_CONFIG = pedestrian.asset_config
_ORIGINAL_PREPARE_TOKEN = pedestrian.prepare_token
_ORIGINAL_CROSSING_POLICY = pedestrian.crossing_policy
_ORIGINAL_ROAD_BASIS = pedestrian.road_basis_aligned_to_global_forward
_ORIGINAL_CACHE_KEY = pedestrian.compatible_render_cache_key
_ORIGINAL_WRITE_JSON = pedestrian.write_json
_ORIGINAL_ACTOR_BBOX = pedestrian.render_utils.actor_bbox_lidar
_ORIGINAL_STANDING_FACING = pedestrian.standing_facing

_ACTIVE_GEOMETRY: Optional[dict] = None
_ACTIVE_LATERAL_SIGN: Optional[float] = None
_ACTIVE_YAW: Optional[float] = None


def _sha256(path: Path) -> str:
    return pedestrian.file_sha256(path)


DOG_EXTENSION_SHA256 = _sha256(Path(__file__).resolve())
DOG_RENDERER_SHA256 = _sha256(DOG_RENDERER)


def measured_animal_geometry(
    metadata: dict,
    sidecar_path: Path,
    species: str,
    contact_profile: str,
) -> dict:
    """Normalize measured rest-pose bounds, which need not be root-centred."""
    value = metadata.get("actor_geometry")
    if (
        not isinstance(value, dict)
        or value.get("profile_version") != STANDING_ANIMAL_SIDECAR_GEOMETRY_PROFILE
        or value.get("species") != species
        or value.get("asset_id") != metadata.get("id")
        or value.get("local_forward_axis") != "-Y"
    ):
        raise ValueError(
            f"Animal asset metadata {sidecar_path} has no supported measured "
            "actor_geometry"
        )
    try:
        bounds = {
            axis: [
                float(value["local_bounds_m"][axis][0]),
                float(value["local_bounds_m"][axis][1]),
            ]
            for axis in ("x", "y", "z")
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid measured animal bounds in {sidecar_path}") from exc
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = (
        bounds["x"],
        bounds["y"],
        bounds["z"],
    )
    if not all(
        math.isfinite(item) for item in (x_min, x_max, y_min, y_max, z_min, z_max)
    ) or not (x_min < 0.0 < x_max and y_min < 0.0 < y_max and z_min < z_max):
        raise ValueError(
            f"Measured animal bounds in {sidecar_path} must contain the root: {bounds}"
        )
    dimensions = {
        "height_m": z_max - z_min,
        "width_m": x_max - x_min,
        "length_m": y_max - y_min,
    }
    if any(item > 5.0 for item in dimensions.values()):
        raise ValueError(f"Invalid animal dimensions in {sidecar_path}: {dimensions}")
    if not math.isclose(
        dimensions["height_m"],
        float(metadata.get("character_height_m", math.nan)),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"Measured animal height disagrees with character_height_m in {sidecar_path}"
        )
    return {
        "profile_version": MEASURED_GEOMETRY_PROFILE_VERSION,
        "actor_type": "animal",
        "species": species,
        "asset_id": str(metadata["id"]),
        "source": "generated_asset_sidecar",
        "sidecar_profile_version": value["profile_version"],
        "local_forward_axis": "-Y",
        "standing_only": metadata.get("standing_only") is True,
        **dimensions,
        "collision_width_m": dimensions["width_m"] + DOG_COLLISION_WIDTH_MARGIN_M,
        "collision_length_m": dimensions["length_m"] + DOG_COLLISION_LENGTH_MARGIN_M,
        "collision_center_xy_m": [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)],
        "collision_min_height_m": 0.15,
        "collision_max_height_m": z_max + DOG_COLLISION_HEIGHT_MARGIN_M,
        "local_bounds_m": bounds,
        "contact_profile": contact_profile,
    }


def dog_geometry_from_metadata(metadata: dict, sidecar_path: Path) -> dict:
    """Validate and normalize one generated Rocketbox animal sidecar."""
    actor_type = str(metadata.get("actor_type", "")).strip().lower()
    species = str(metadata.get("species", "")).strip().lower()
    if species not in ANIMAL_SPECIES or actor_type not in {"animal", "dog"}:
        raise ValueError(
            "Animal compositor requires a homogeneous animal asset pool; "
            f"got actor_type={actor_type!r}, species={species!r} in "
            f"{sidecar_path}"
        )
    # Dog sidecars generated before contact profiles were recorded imply theirs.
    contact_profile = str(metadata.get("contact_profile") or f"quadruped_{species}")
    if (
        contact_profile != f"quadruped_{species}"
        or contact_profile not in QUADRUPED_CONTACT_PROFILES
    ):
        raise ValueError(
            f"Unsupported animal contact_profile {contact_profile!r} in {sidecar_path}"
        )
    if "actor_geometry" in metadata:
        return measured_animal_geometry(
            metadata, sidecar_path, species, contact_profile
        )
    if species != "dog":
        raise ValueError(
            f"{species} asset metadata {sidecar_path} requires measured actor_geometry"
        )
    required = {
        "character_height_m",
        "actor_width_m",
        "actor_length_m",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Dog asset metadata {sidecar_path} is missing {missing}")
    height = float(metadata["character_height_m"])
    width = float(metadata["actor_width_m"])
    length = float(metadata["actor_length_m"])
    dimensions = {"height_m": height, "width_m": width, "length_m": length}
    if any(
        not math.isfinite(value) or value <= 0.0 or value > 5.0
        for value in dimensions.values()
    ):
        raise ValueError(f"Invalid dog dimensions in {sidecar_path}: {dimensions}")
    return {
        "profile_version": DOG_GEOMETRY_PROFILE_VERSION,
        "actor_type": "dog",
        "species": "dog",
        **dimensions,
        "collision_width_m": width + DOG_COLLISION_WIDTH_MARGIN_M,
        "collision_length_m": length + DOG_COLLISION_LENGTH_MARGIN_M,
        "collision_min_height_m": 0.15,
        "collision_max_height_m": height + DOG_COLLISION_HEIGHT_MARGIN_M,
        "local_bounds_m": {
            "x": [-width * 0.5, width * 0.5],
            "y": [-length * 0.5, length * 0.5],
            "z": [0.0, height],
        },
        "contact_profile": "quadruped_dog",
    }


def load_dog_asset(path: Path, args: argparse.Namespace) -> dict:
    asset = _ORIGINAL_LOAD_ASSET(path, args)
    asset["actor_geometry"] = dog_geometry_from_metadata(
        asset["metadata"], asset["metadata_path"]
    )
    standing_only = asset["actor_geometry"].get("standing_only") is True
    if bool(getattr(args, "standing", False)) != standing_only:
        raise ValueError(
            "Standing-only animal assets require --standing, and --standing "
            f"requires standing-only animal assets: {path}"
        )
    return asset


def dog_asset_config(asset: dict) -> dict:
    config = _ORIGINAL_ASSET_CONFIG(asset)
    config["actor_geometry"] = dict(asset["actor_geometry"])
    return config


def _require_active_geometry() -> dict:
    if _ACTIVE_GEOMETRY is None:
        raise RuntimeError("Dog geometry was requested outside token preparation")
    return _ACTIVE_GEOMETRY


def dog_actor_bbox_lidar(
    position: np.ndarray,
    actor_basis_lidar: np.ndarray,
    actor_heading: float,
    local_bounds_m: Optional[dict] = None,
    scale: float = 1.0,
) -> np.ndarray:
    profile = _require_active_geometry()
    return _ORIGINAL_ACTOR_BBOX(
        position,
        actor_basis_lidar,
        actor_heading,
        profile["local_bounds_m"],
        scale=scale,
    )


def _local_xy(
    points_xy: np.ndarray,
    actor_xy: Sequence[float],
    yaw: float,
) -> np.ndarray:
    offsets = np.asarray(points_xy, dtype=np.float64) - np.asarray(
        actor_xy, dtype=np.float64
    )[None]
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return offsets @ np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )


def _active_yaw() -> float:
    if _ACTIVE_YAW is None:
        raise RuntimeError("Dog collision yaw was not captured during preparation")
    return float(_ACTIVE_YAW)


def _collision_envelope(profile: dict) -> dict:
    envelope = {
        key: float(profile[key])
        for key in (
            "collision_width_m",
            "collision_length_m",
            "collision_min_height_m",
            "collision_max_height_m",
        )
    }
    if "collision_center_xy_m" in profile:
        envelope["collision_center_xy_m"] = [
            float(item) for item in profile["collision_center_xy_m"]
        ]
    return envelope


def dog_frame_collision_diagnostics(
    points: np.ndarray,
    lidar_to_actor: np.ndarray,
    actor_heading: float,
    local_bounds_m: dict,
) -> dict:
    profile = _require_active_geometry()
    yaw = _active_yaw()
    points_actor = pedestrian.geometry.transform_points(lidar_to_actor, points)
    cosine, sine = math.cos(float(actor_heading)), math.sin(float(actor_heading))
    local_to_actor = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local = points_actor @ local_to_actor
    # Measured bounds need not be centred on the root; dog bounds are.
    center_x, center_y = (
        float(item) for item in profile.get("collision_center_xy_m", (0.0, 0.0))
    )
    occupied = (
        (np.abs(local[:, 0] - center_x) < float(profile["collision_width_m"]) * 0.5)
        & (np.abs(local[:, 1] - center_y) < float(profile["collision_length_m"]) * 0.5)
        & (local[:, 2] > float(profile["collision_min_height_m"]))
        & (local[:, 2] < float(profile["collision_max_height_m"]))
    )
    count = int(np.count_nonzero(occupied))
    return {
        "obstacle_points": count,
        "collision_threshold_points": 5,
        "occupied": count >= 5,
        "actor_yaw_rad": yaw,
        "collision_envelope": _collision_envelope(profile),
    }


def dog_trajectory_collision_diagnostics(
    points: np.ndarray,
    plane: np.ndarray,
    trajectory_xy_or_longitudinal: Any,
    lateral_positions: Optional[Sequence[float]] = None,
    *,
    actor_basis_lidar: np.ndarray,
    actor_heading: float,
    local_bounds_m: dict,
) -> dict:
    if lateral_positions is not None:
        trajectory_xy = np.asarray(
            [
                [float(trajectory_xy_or_longitudinal), float(lateral)]
                for lateral in lateral_positions
            ],
            dtype=np.float64,
        )
    else:
        trajectory_xy = np.asarray(trajectory_xy_or_longitudinal, dtype=np.float64)
    if (
        trajectory_xy.ndim != 2
        or trajectory_xy.shape[1] != 2
        or len(trajectory_xy) == 0
        or not np.isfinite(trajectory_xy).all()
    ):
        raise ValueError("trajectory collision path must be a finite Nx2 array")
    if len(trajectory_xy) == 1:
        samples = np.repeat(trajectory_xy, 2, axis=0)
    else:
        segment_lengths = np.linalg.norm(np.diff(trajectory_xy, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = float(cumulative[-1])
        sample_count = max(2, int(math.ceil(total_length / 0.25)) + 1)
        if total_length < 1e-9:
            samples = np.repeat(trajectory_xy[:1], sample_count, axis=0)
        else:
            distances = np.linspace(0.0, total_length, sample_count)
            samples = np.column_stack(
                [
                    np.interp(distances, cumulative, trajectory_xy[:, axis])
                    for axis in range(2)
                ]
            )
    profile = _require_active_geometry()
    yaw = _active_yaw()
    counts = []
    for actor_x, actor_y in samples:
        actor_position = np.array(
            [
                float(actor_x),
                float(actor_y),
                pedestrian.geometry.ground_z(
                    plane, float(actor_x), float(actor_y)
                ),
            ],
            dtype=np.float64,
        )
        diagnostics = dog_frame_collision_diagnostics(
            points,
            np.linalg.inv(
                pedestrian.render_utils.matrix4(
                    actor_basis_lidar, actor_position
                )
            ),
            actor_heading,
            local_bounds_m,
        )
        counts.append(int(diagnostics["obstacle_points"]))
    maximum = max(counts, default=0)
    return {
        "sample_count": int(len(samples)),
        "path_start_xy_m": trajectory_xy[0].tolist(),
        "path_end_xy_m": trajectory_xy[-1].tolist(),
        "maximum_obstacle_points": maximum,
        "collision_threshold_points": 5,
        "occupied": maximum >= 5,
        "actor_yaw_rad": yaw,
        "collision_envelope": _collision_envelope(profile),
    }


def capture_crossing_policy(token: str, classifications: dict):
    global _ACTIVE_LATERAL_SIGN
    result = _ORIGINAL_CROSSING_POLICY(token, classifications)
    _ACTIVE_LATERAL_SIGN = float(result[1])
    return result


def animal_standing_facing(
    args: argparse.Namespace,
    token: str,
) -> tuple[str, float, float]:
    """Stand the animal side-on to the ego path, its head to a seeded side."""
    global _ACTIVE_LATERAL_SIGN
    side = (
        "left"
        if pedestrian.token_uniform(
            token, int(args.seed), STANDING_SIDE_HASH_NAMESPACE
        )
        < 0.5
        else "right"
    )
    spec = pedestrian.DIRECTION_CLASSIFICATION_SPECS[side]
    _ACTIVE_LATERAL_SIGN = float(spec["lateral_sign"])
    return (
        f"standing_side_on_head_{side}",
        float(spec["lateral_sign"]),
        float(spec["actor_heading_rad"]),
    )


def capture_road_basis(*args, **kwargs) -> np.ndarray:
    global _ACTIVE_YAW
    basis = _ORIGINAL_ROAD_BASIS(*args, **kwargs)
    if _ACTIVE_GEOMETRY is not None and _ACTIVE_LATERAL_SIGN is not None:
        direction = basis[:, 1] * float(_ACTIVE_LATERAL_SIGN)
        _ACTIVE_YAW = (
            math.atan2(float(direction[1]), float(direction[0])) + math.pi / 2.0
        )
    return basis


def dog_cache_key(
    manifest_path: Path,
    manifest: dict,
    excluded_fields: Sequence[str],
    *,
    reuse_existing: bool,
) -> str:
    if _ACTIVE_GEOMETRY is not None and manifest.get("render_mode"):
        manifest["actor_geometry"] = dict(_ACTIVE_GEOMETRY)
        manifest["contact_profile"] = str(_ACTIVE_GEOMETRY["contact_profile"])
        manifest["dog_extension_source_sha256"] = DOG_EXTENSION_SHA256
        manifest["dog_renderer_source_sha256"] = DOG_RENDERER_SHA256
    return _ORIGINAL_CACHE_KEY(
        manifest_path,
        manifest,
        excluded_fields,
        reuse_existing=reuse_existing,
    )


def prepare_dog_token(*args, **kwargs) -> dict:
    global _ACTIVE_GEOMETRY, _ACTIVE_LATERAL_SIGN, _ACTIVE_YAW
    asset = kwargs.get("asset")
    if asset is None and len(args) >= 6:
        asset = args[5]
    if not isinstance(asset, dict) or "actor_geometry" not in asset:
        raise ValueError("Dog preparation requires normalized actor geometry")
    _ACTIVE_GEOMETRY = asset["actor_geometry"]
    _ACTIVE_LATERAL_SIGN = None
    _ACTIVE_YAW = None
    try:
        return _ORIGINAL_PREPARE_TOKEN(*args, **kwargs)
    finally:
        _ACTIVE_GEOMETRY = None
        _ACTIVE_LATERAL_SIGN = None
        _ACTIVE_YAW = None










def _recorded_contact_profile(value: dict) -> str:
    """Return the quadruped contact profile of a manifest, metadata, or run config.

    A run config whose asset pool mixes species records ``quadruped_mixed``.
    """
    if isinstance(value.get("contact_profile"), str):
        return value["contact_profile"]
    profiles = set()
    for asset in (value.get("asset"), *(value.get("asset_pool") or ())):
        geometry = asset.get("actor_geometry") if isinstance(asset, dict) else None
        if isinstance(geometry, dict) and isinstance(
            geometry.get("contact_profile"), str
        ):
            profiles.add(geometry["contact_profile"])
    if len(profiles) > 1:
        return "quadruped_mixed"
    return profiles.pop() if profiles else "quadruped_dog"


def dog_write_json(path: Path, value: Any):
    if (
        isinstance(value, dict)
        and value.get("render_pipeline_version")
        == DOG_RENDER_PIPELINE_VERSION
    ):
        value.setdefault("actor_profile", _recorded_contact_profile(value))
        value.setdefault("dog_extension_source_sha256", DOG_EXTENSION_SHA256)
        value.setdefault("dog_renderer_source_sha256", DOG_RENDERER_SHA256)
        if value.get("foot_contact_geometry") == (
            "evaluated_foot_toe_vertex_groups_v1"
        ):
            value["foot_contact_geometry"] = DOG_CONTACT_GEOMETRY_VERSION
    return _ORIGINAL_WRITE_JSON(path, value)


def install_dog_adapter() -> None:
    if not DOG_RENDERER.is_file():
        raise FileNotFoundError(f"Dog Blender renderer is missing: {DOG_RENDERER}")
    pedestrian.RENDER_PIPELINE_VERSION = DOG_RENDER_PIPELINE_VERSION
    # Standing animals keep their rest pose (quadruped renderer) and stand
    # side-on to the ego path; walking dogs never reach the standing branch.
    pedestrian.STANDING_POSE_VERSION = QUADRUPED_STANDING_POSE_VERSION
    pedestrian.STANDING_FACING_POLICY = STANDING_SIDE_ON_FACING_POLICY
    pedestrian.standing_facing = animal_standing_facing
    pedestrian.load_asset = load_dog_asset
    pedestrian.asset_config = dog_asset_config
    pedestrian.prepare_token = prepare_dog_token
    pedestrian.crossing_policy = capture_crossing_policy
    pedestrian.road_basis_aligned_to_global_forward = capture_road_basis
    pedestrian.frame_collision_diagnostics = dog_frame_collision_diagnostics
    pedestrian.trajectory_collision_diagnostics = (
        dog_trajectory_collision_diagnostics
    )
    pedestrian.render_utils.actor_bbox_lidar = dog_actor_bbox_lidar
    pedestrian.compatible_render_cache_key = dog_cache_key
    pedestrian.write_json = dog_write_json




def main() -> None:
    install_dog_adapter()
    pedestrian.main()


if __name__ == "__main__":
    main()
