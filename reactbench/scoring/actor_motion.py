"""Motion of the inserted object, reconstructed from the compositor's per-sample metadata.

Every inserted object moves at constant velocity in the current ego frame (crossing pedestrians
walk at their placement speed; standing pedestrians, cars and animals stand still). The returned
"motion contract" holds its initial centre, velocity, heading and asymmetric footprint extents.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from .geometry import finite_xy, oriented_box_corners
from .pedestrian_geometry import (
    local_forward_angle_rad,
    local_forward_vector,
    normalize_pedestrian_geometry,
    pedestrian_footprint_extents,
)

# Written by the NAVSIM compositor for standing animals (unseen-objects test set).
ANIMAL_GEOMETRY_PROFILE_VERSION = "quadruped_measured_bounds_geometry_v1"


def placement_mapping(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    placement = metadata.get("placement")
    return placement if isinstance(placement, Mapping) else {}


def current_front_record(
    metadata: Mapping[str, Any],
    camera_name: Optional[str] = "CAM_F0",
) -> Mapping[str, Any]:
    """Return one current actor record, optionally independent of camera names.

    Existing NAVSIM callers retain the historical ``CAM_F0`` default.  Dataset
    adapters whose metadata repeats one actor state across differently named
    cameras can pass ``camera_name=None``.
    """

    records = metadata.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Timed-placement metadata has no frame records")
    frame_records = [record for record in records if isinstance(record, Mapping)]
    if not frame_records:
        raise ValueError("Timed-placement metadata has no valid frame records")
    current_frame_index = max(
        int(record.get("frame_index", -1)) for record in frame_records
    )
    current_records = [
        record
        for record in frame_records
        if int(record.get("frame_index", -1)) == current_frame_index
        and (camera_name is None or record.get("camera") == camera_name)
    ]
    if camera_name is not None and len(current_records) != 1:
        raise ValueError(
            f"Expected exactly one current {camera_name} compositor record, got "
            f"{len(current_records)}"
        )
    if not current_records:
        raise ValueError("Timed-placement metadata has no current actor record")

    reference = current_records[0]
    if camera_name is None and len(current_records) > 1:
        for field in ("actor_current", "yaw_lidar"):
            reference_value = reference.get(field)
            if reference_value is None:
                continue
            reference_array = np.asarray(reference_value, dtype=np.float64)
            for record in current_records[1:]:
                candidate = np.asarray(record.get(field), dtype=np.float64)
                if candidate.shape != reference_array.shape or not np.allclose(
                    candidate,
                    reference_array,
                    rtol=1e-6,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "Current actor state differs across camera records for "
                        f"field {field!r}"
                    )
    return reference


def pedestrian_motion_from_metadata(
    metadata: Mapping[str, Any],
    *,
    camera_name: Optional[str] = "CAM_F0",
) -> dict[str, Any]:
    """Build the canonical local pedestrian-motion contract from v2 metadata."""

    placement = placement_mapping(metadata)
    placement_mode = metadata.get("placement_mode", placement.get("placement_mode"))
    if placement_mode != "future_timestep":
        raise ValueError(
            "Geometry evaluation requires placement_mode='future_timestep', "
            f"got {placement_mode!r}"
        )

    geometry_value = metadata.get("pedestrian_geometry")
    if not isinstance(geometry_value, Mapping):
        raise ValueError(
            "Edited pedestrian metadata has no recorded pedestrian_geometry; "
            "legacy fixed-footprint samples are unsupported"
        )
    pedestrian_geometry = normalize_pedestrian_geometry(geometry_value)
    asset_value = metadata.get("asset")
    if isinstance(asset_value, Mapping):
        asset_id = asset_value.get("id")
        if asset_id is not None and str(asset_id) != pedestrian_geometry["asset_id"]:
            raise ValueError(
                "pedestrian_geometry asset_id disagrees with sample asset id: "
                f"{pedestrian_geometry['asset_id']!r} != {str(asset_id)!r}"
            )
        nested_geometry = asset_value.get("actor_geometry")
        if nested_geometry is not None and normalize_pedestrian_geometry(
            nested_geometry,
            expected_asset_id=pedestrian_geometry["asset_id"],
        ) != pedestrian_geometry:
            raise ValueError(
                "Top-level pedestrian_geometry disagrees with asset.actor_geometry"
            )

    current_record = current_front_record(metadata, camera_name)
    actor_current = np.asarray(
        current_record.get("actor_current"), dtype=np.float64
    )
    if actor_current.ndim != 1 or actor_current.size < 2:
        raise ValueError(
            "current CAM_F0 actor_current must contain at least x and y, "
            f"got {current_record.get('actor_current')!r}"
        )
    initial_center = finite_xy(
        actor_current[:2], "current pedestrian center"
    )
    interception_center = finite_xy(
        metadata.get(
            "anchor_current_ego_xy_m",
            placement.get("anchor_current_ego_xy_m"),
        ),
        "interception center in current-ego coordinates",
    )
    interception_time_s = float(
        metadata.get(
            "pedestrian_time_to_interception_s",
            placement.get("pedestrian_time_to_interception_s", math.nan),
        )
    )
    if not math.isfinite(interception_time_s) or interception_time_s <= 0:
        raise ValueError(
            "pedestrian_time_to_interception_s must be finite and positive, "
            f"got {interception_time_s}"
        )

    displacement = interception_center - initial_center
    derived_velocity = displacement / interception_time_s
    derived_speed = float(np.linalg.norm(derived_velocity))
    stored_speed = float(metadata.get("pedestrian_speed_mps", derived_speed))
    if not math.isfinite(stored_speed) or stored_speed < 0:
        raise ValueError(f"Invalid pedestrian_speed_mps: {stored_speed}")

    yaw_lidar = float(current_record.get("yaw_lidar", math.nan))
    if not math.isfinite(yaw_lidar):
        raise ValueError(
            "Pedestrian metadata requires a finite current actor yaw_lidar, "
            f"got {current_record.get('yaw_lidar')!r}"
        )
    rendered_heading = yaw_lidar + local_forward_angle_rad(
        pedestrian_geometry["local_forward_axis"]
    )
    rendered_heading = math.atan2(
        math.sin(rendered_heading), math.cos(rendered_heading)
    )

    if stored_speed == 0.0:
        # A standing pedestrian remains at the rendered current-frame origin.
        # Its facing direction comes from the declared asset-local forward axis
        # transformed by the renderer root yaw.
        velocity = np.zeros(2, dtype=np.float64)
        heading = rendered_heading
    else:
        if not math.isfinite(derived_speed) or derived_speed <= 0:
            raise ValueError(
                "Derived pedestrian velocity must be finite and non-zero"
            )
        if not math.isclose(
            derived_speed, stored_speed, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise ValueError(
                "Derived pedestrian speed disagrees with compositor metadata: "
                f"{derived_speed:.6f} vs {stored_speed:.6f} m/s"
            )
        velocity = derived_velocity
        # The compositor owns actor-orientation validation. Its road-tangent
        # rendered heading can legitimately differ from the planar velocity
        # heading on sloped or banked roads.
        heading = rendered_heading

    footprint = pedestrian_footprint_extents(pedestrian_geometry)

    return {
        "actor_type": str(metadata.get("actor_type", "pedestrian")),
        "behavior": metadata.get(
            "actor_behavior", metadata.get("pedestrian_behavior", "moving")
        ),
        "initial_center_xy_m": initial_center.tolist(),
        "velocity_xy_mps": velocity.tolist(),
        "heading_rad": float(heading),
        **footprint,
        "geometry": pedestrian_geometry,
        "geometry_source": pedestrian_geometry["source"],
        "geometry_profile_version": pedestrian_geometry["profile_version"],
        "geometry_asset_id": pedestrian_geometry["asset_id"],
        "target_time_s": interception_time_s,
        "interception_time_s": interception_time_s,
        "interception_center_xy_m": interception_center.tolist(),
        "future_timestep": int(
            metadata.get("future_timestep", placement.get("future_timestep"))
        ),
        "placement_version": metadata.get(
            "placement_version", placement.get("placement_version")
        ),
    }


def stationary_car_motion_from_metadata(
    metadata: Mapping[str, Any],
    *,
    camera_name: Optional[str] = "CAM_F0",
) -> dict[str, Any]:
    """Build a stationary inserted-car contract from finalized car metadata."""

    placement = placement_mapping(metadata)
    placement_mode = metadata.get("placement_mode", placement.get("placement_mode"))
    if placement_mode != "future_timestep":
        raise ValueError(
            "Geometry evaluation requires placement_mode='future_timestep', "
            f"got {placement_mode!r}"
        )
    actor_type = str(metadata.get("actor_type", placement.get("actor_type", "")))
    if actor_type.lower() != "car":
        raise ValueError(f"Expected actor_type='car', got {actor_type!r}")
    behavior = metadata.get(
        "vehicle_behavior",
        metadata.get(
            "actor_behavior", placement.get("vehicle_behavior", "stationary")
        ),
    )
    if behavior != "stationary":
        raise ValueError(
            "Car geometry currently supports vehicle_behavior='stationary', "
            f"got {behavior!r}"
        )

    initial_center = finite_xy(
        metadata.get(
            "anchor_current_ego_xy_m",
            placement.get("anchor_current_ego_xy_m"),
        ),
        "stationary car origin in current-ego coordinates",
    )
    current_record = current_front_record(metadata, camera_name)
    heading = float(current_record.get("yaw_lidar", math.nan))
    if not math.isfinite(heading):
        raise ValueError(
            "Stationary car metadata requires a finite current CAM_F0 "
            f"yaw_lidar, got {current_record.get('yaw_lidar')!r}"
        )

    geometry_value = metadata.get("vehicle_geometry")
    if not isinstance(geometry_value, Mapping):
        asset_value = metadata.get("asset")
        asset = asset_value if isinstance(asset_value, Mapping) else {}
        geometry_value = asset.get("actor_geometry")
    if not isinstance(geometry_value, Mapping):
        raise ValueError("Stationary car metadata has no vehicle geometry")
    bounds_value = geometry_value.get("local_bounds_m")
    if not isinstance(bounds_value, Mapping):
        raise ValueError("Stationary car geometry has no local_bounds_m")
    x_bounds = finite_xy(bounds_value.get("x"), "car local x bounds")
    y_bounds = finite_xy(bounds_value.get("y"), "car local y bounds")
    if not x_bounds[0] < 0.0 < x_bounds[1]:
        raise ValueError(f"Car local x bounds must straddle zero, got {x_bounds}")
    if not y_bounds[0] < 0.0 < y_bounds[1]:
        raise ValueError(f"Car local y bounds must straddle zero, got {y_bounds}")

    target_time_s = float(
        metadata.get("time_to_contact_s", placement.get("time_to_contact_s", math.nan))
    )
    if not math.isfinite(target_time_s) or target_time_s <= 0:
        raise ValueError(
            "time_to_contact_s must be finite and positive, "
            f"got {target_time_s}"
        )
    future_timestep_value = metadata.get(
        "future_timestep", placement.get("future_timestep")
    )
    try:
        future_timestep = int(future_timestep_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"future_timestep must be a positive integer, got {future_timestep_value!r}"
        ) from error
    if future_timestep <= 0 or future_timestep != future_timestep_value:
        raise ValueError(
            f"future_timestep must be a positive integer, got {future_timestep_value!r}"
        )

    front_length = float(x_bounds[1])
    rear_length = float(-x_bounds[0])
    left_width = float(y_bounds[1])
    right_width = float(-y_bounds[0])
    return {
        "actor_type": "car",
        "behavior": "stationary",
        "initial_center_xy_m": initial_center.tolist(),
        "velocity_xy_mps": [0.0, 0.0],
        "heading_rad": heading,
        "length_m": front_length + rear_length,
        "width_m": left_width + right_width,
        "front_length_m": front_length,
        "rear_length_m": rear_length,
        "left_width_m": left_width,
        "right_width_m": right_width,
        "target_time_s": target_time_s,
        "time_to_contact_s": target_time_s,
        "future_timestep": future_timestep,
        "placement_version": metadata.get(
            "placement_version", placement.get("placement_version")
        ),
        "geometry_profile_version": geometry_value.get("profile_version"),
    }


def actor_motion_from_metadata(
    metadata: Mapping[str, Any],
    *,
    camera_name: Optional[str] = "CAM_F0",
    ego_heading_rad: Optional[float] = None,
) -> dict[str, Any]:
    """Dispatch compositor metadata; moving benign actors need the source ego heading."""

    placement = placement_mapping(metadata)
    if placement.get("benign") is True:
        return _benign_actor_motion_from_metadata(metadata, camera_name, ego_heading_rad)
    actor_type = str(
        metadata.get("actor_type", placement.get("actor_type", "pedestrian"))
    ).lower()
    if actor_type == "car":
        return stationary_car_motion_from_metadata(
            metadata, camera_name=camera_name
        )
    return pedestrian_motion_from_metadata(metadata, camera_name=camera_name)


def _benign_actor_motion_from_metadata(metadata, camera_name, ego_heading_rad):
    """Read a benign placement without assigning a fictitious interception/contact.

    The compositor records walking velocity in world coordinates. Transform it
    with the original scene's planar ego heading, never the pedestrian's facing
    direction (which can differ on banked roads).
    """
    placement = placement_mapping(metadata)
    if metadata.get("placement_mode", placement.get("placement_mode")) != "future_timestep":
        raise ValueError("Benign metadata requires placement_mode='future_timestep'")
    record = current_front_record(metadata, camera_name)
    actor_type = metadata.get("actor_type", "pedestrian")
    heading = float(record.get("yaw_lidar", math.nan))
    if not math.isfinite(heading):
        raise ValueError("Benign metadata requires finite yaw_lidar")
    velocity_global = finite_xy(placement.get("walking_velocity_global_xy_mps"),
                                 "benign world velocity")
    if "walking_velocity_global_xy_mps" in metadata and not np.allclose(
        finite_xy(metadata["walking_velocity_global_xy_mps"], "world velocity"),
        velocity_global, rtol=0, atol=1e-9,
    ):
        raise ValueError("Benign world velocity disagrees across metadata fields")
    speed = float(np.linalg.norm(velocity_global))
    behavior = placement.get("actor_behavior")
    if behavior not in ("stationary", "parallel_walking"):
        raise ValueError(f"Unsupported benign actor behavior: {behavior!r}")
    if (behavior == "stationary" and speed != 0) or (behavior == "parallel_walking" and speed <= 0):
        raise ValueError("Benign velocity disagrees with actor behavior")
    velocity = np.zeros(2)
    if speed > 0:
        if ego_heading_rad is None or not math.isfinite(ego_heading_rad):
            raise ValueError("Moving benign actors require the source scene's ego_heading_rad")
        c, s = math.cos(ego_heading_rad), math.sin(ego_heading_rad)
        velocity = np.array([[c, s], [-s, c]]) @ velocity_global
    geometry_fields = {}
    if actor_type == "pedestrian":
        geometry = normalize_pedestrian_geometry(metadata.get("pedestrian_geometry"))
        asset = metadata.get("asset", {})
        if isinstance(asset, Mapping):
            if asset.get("id") is not None and str(asset["id"]) != geometry["asset_id"]:
                raise ValueError("pedestrian_geometry asset_id disagrees with sample asset id")
            if asset.get("actor_geometry") is not None and normalize_pedestrian_geometry(
                asset["actor_geometry"], expected_asset_id=geometry["asset_id"]
            ) != geometry:
                raise ValueError("pedestrian_geometry disagrees with asset.actor_geometry")
        stored_speed = float(metadata.get("pedestrian_speed_mps", math.nan))
        if not math.isfinite(stored_speed) or not math.isclose(speed, stored_speed, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("Benign pedestrian speed disagrees with compositor metadata")
        initial = finite_xy(record.get("actor_current", [])[:2], "benign pedestrian origin")
        heading += local_forward_angle_rad(geometry["local_forward_axis"])
        footprint = pedestrian_footprint_extents(geometry)
        geometry_fields = {"geometry": geometry, "geometry_source": geometry["source"],
                           "geometry_profile_version": geometry["profile_version"],
                           "geometry_asset_id": geometry["asset_id"]}
    elif actor_type == "car":
        if behavior != "stationary":
            raise ValueError("Benign car must be stationary")
        initial = finite_xy(metadata.get("anchor_current_ego_xy_m", placement.get("anchor_current_ego_xy_m")),
                             "benign car origin")
        geometry = metadata.get("vehicle_geometry", metadata.get("asset", {}).get("actor_geometry", {}))
        bounds = geometry.get("local_bounds_m", {})
        x, y = finite_xy(bounds.get("x"), "car x bounds"), finite_xy(bounds.get("y"), "car y bounds")
        if not x[0] < 0 < x[1] or not y[0] < 0 < y[1]:
            raise ValueError("Benign car bounds must straddle zero")
        footprint = dict(front_length_m=x[1], rear_length_m=-x[0], left_width_m=y[1],
                         right_width_m=-y[0], length_m=x[1]-x[0], width_m=y[1]-y[0])
        geometry_fields = {"geometry_profile_version": geometry.get("profile_version")}
    else:
        raise ValueError(f"Unsupported benign actor type: {actor_type!r}")
    timestep = metadata.get("future_timestep", placement.get("future_timestep"))
    if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep <= 0:
        raise ValueError("Benign future_timestep must be a positive integer")
    return validated_motion({"actor_type": actor_type, "behavior": behavior, "benign": True,
        "initial_center_xy_m": initial.tolist(), "velocity_xy_mps": velocity.tolist(),
        "heading_rad": math.atan2(math.sin(heading), math.cos(heading)), **footprint, **geometry_fields,
        "future_timestep": timestep, "placement_version": placement.get("placement_version")})


def _motion_extents(
    motion: Mapping[str, Any],
    *,
    first_key: str,
    second_key: str,
    total_key: str,
) -> tuple[float, float, float]:
    first_value = motion.get(first_key)
    second_value = motion.get(second_key)
    if (first_value is None) != (second_value is None):
        raise ValueError(f"{first_key} and {second_key} must be provided together")
    if first_value is None:
        total = float(motion.get(total_key, math.nan))
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"{total_key} must be finite and positive, got {total}")
        return total / 2.0, total / 2.0, total

    first = float(first_value)
    second = float(second_value)
    if not math.isfinite(first) or first <= 0:
        raise ValueError(f"{first_key} must be finite and positive, got {first}")
    if not math.isfinite(second) or second <= 0:
        raise ValueError(f"{second_key} must be finite and positive, got {second}")
    total = first + second
    stored_total = motion.get(total_key)
    if stored_total is not None and not math.isclose(
        float(stored_total), total, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError(
            f"{total_key} disagrees with {first_key} + {second_key}: "
            f"{stored_total} vs {total}"
        )
    return first, second, total


def validated_motion(motion: Mapping[str, Any]) -> dict[str, Any]:
    initial_center = finite_xy(
        motion.get("initial_center_xy_m"), "initial_center_xy_m"
    )
    velocity = finite_xy(motion.get("velocity_xy_mps"), "velocity_xy_mps")
    heading = float(motion.get("heading_rad", math.atan2(velocity[1], velocity[0])))
    if not math.isfinite(heading):
        raise ValueError(f"heading_rad must be finite, got {heading}")
    front_length, rear_length, length = _motion_extents(
        motion,
        first_key="front_length_m",
        second_key="rear_length_m",
        total_key="length_m",
    )
    left_width, right_width, width = _motion_extents(
        motion,
        first_key="left_width_m",
        second_key="right_width_m",
        total_key="width_m",
    )
    result = {
        **dict(motion),
        "initial_center_xy_m": initial_center.tolist(),
        "velocity_xy_mps": velocity.tolist(),
        "heading_rad": heading,
        "length_m": length,
        "width_m": width,
        "front_length_m": front_length,
        "rear_length_m": rear_length,
        "left_width_m": left_width,
        "right_width_m": right_width,
    }
    geometry_value = motion.get("geometry")
    if geometry_value is not None:
        geometry = normalize_pedestrian_geometry(geometry_value)
        expected = pedestrian_footprint_extents(geometry)
        for key, expected_value in expected.items():
            if not math.isclose(
                float(result[key]),
                float(expected_value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"Pedestrian motion {key} disagrees with recorded geometry: "
                    f"{result[key]} vs {expected_value}"
                )
        result["geometry"] = geometry
    return result


def actor_center_at_time(
    actor_motion: Mapping[str, Any], time_s: float
) -> np.ndarray:
    """Return the inserted actor origin in current-ego coordinates."""

    motion = validated_motion(actor_motion)
    return np.asarray(motion["initial_center_xy_m"], dtype=np.float64) + float(
        time_s
    ) * np.asarray(motion["velocity_xy_mps"], dtype=np.float64)


def actor_footprint_corners(
    actor_motion: Mapping[str, Any], time_s: float
) -> np.ndarray:
    """Return the inserted actor's oriented footprint at ``time_s``."""

    motion = validated_motion(actor_motion)
    center = actor_center_at_time(motion, time_s)
    return oriented_box_corners(
        center,
        float(motion["heading_rad"]),
        float(motion["front_length_m"]),
        float(motion["rear_length_m"]),
        float(motion["left_width_m"]),
        float(motion["right_width_m"]),
    )


def is_animal_metadata(metadata: Mapping[str, Any]) -> bool:
    placement = placement_mapping(metadata)
    actor_type = metadata.get("actor_type", placement.get("actor_type", "pedestrian"))
    return str(actor_type).lower() == "animal"


def inserted_actor_motion(
    metadata: Mapping[str, Any],
    *,
    camera_name: Optional[str] = "CAM_F0",
    ego_heading_rad: Optional[float] = None,
) -> dict[str, Any]:
    """Return the motion contract of an inserted animal, pedestrian, or car."""
    if is_animal_metadata(metadata):
        return animal_motion_from_metadata(metadata, camera_name=camera_name)
    return actor_motion_from_metadata(
        metadata, camera_name=camera_name, ego_heading_rad=ego_heading_rad
    )


def normalize_animal_geometry(
    value: Any, *, expected_asset_id: Optional[str] = None
) -> dict[str, Any]:
    """Validate measured animal bounds about the renderer root."""
    if not isinstance(value, Mapping):
        raise ValueError("Inserted animal metadata has no asset actor_geometry")
    if value.get("profile_version") != ANIMAL_GEOMETRY_PROFILE_VERSION:
        raise ValueError(
            "Unsupported animal geometry profile_version "
            f"{value.get('profile_version')!r}; expected "
            f"{ANIMAL_GEOMETRY_PROFILE_VERSION!r}"
        )
    if str(value.get("actor_type", "")).lower() != "animal":
        raise ValueError("animal geometry actor_type must be 'animal'")
    species = value.get("species")
    asset_id = value.get("asset_id")
    if not isinstance(species, str) or not species.strip():
        raise ValueError("animal geometry species must be non-empty")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError("animal geometry asset_id must be non-empty")
    if expected_asset_id is not None and asset_id != str(expected_asset_id):
        raise ValueError(
            f"animal geometry asset_id disagrees with sample asset id: "
            f"{asset_id!r} != {str(expected_asset_id)!r}"
        )
    forward_axis = str(value.get("local_forward_axis", ""))
    local_forward_vector(forward_axis)
    bounds_value = value.get("local_bounds_m")
    if not isinstance(bounds_value, Mapping):
        raise ValueError("animal geometry local_bounds_m must be a mapping")
    bounds = {}
    for axis in ("x", "y", "z"):
        interval = finite_xy(bounds_value.get(axis), f"animal local {axis} bounds")
        if not interval[0] < interval[1]:
            raise ValueError(f"animal local {axis} bounds must increase, got {interval}")
        bounds[axis] = [float(interval[0]), float(interval[1])]
    if not (bounds["x"][0] < 0.0 < bounds["x"][1] and bounds["y"][0] < 0.0 < bounds["y"][1]):
        raise ValueError("animal asset root must lie inside its x/y footprint")
    return {
        "profile_version": ANIMAL_GEOMETRY_PROFILE_VERSION,
        "actor_type": "animal",
        "species": species,
        "asset_id": asset_id,
        "source": str(value.get("source", "")),
        "local_forward_axis": forward_axis,
        "local_bounds_m": bounds,
    }


def animal_footprint_extents(geometry: Mapping[str, Any]) -> dict[str, float]:
    """Project local x/y bounds onto forward/left footprint axes."""
    forward_x, forward_y = local_forward_vector(geometry["local_forward_axis"])
    left_x, left_y = -forward_y, forward_x
    bounds = geometry["local_bounds_m"]
    corners = [(x, y) for x in bounds["x"] for y in bounds["y"]]
    forward = [x * forward_x + y * forward_y for x, y in corners]
    left = [x * left_x + y * left_y for x, y in corners]
    front, rear = max(forward), -min(forward)
    left_width, right_width = max(left), -min(left)
    return {
        "front_length_m": float(front),
        "rear_length_m": float(rear),
        "left_width_m": float(left_width),
        "right_width_m": float(right_width),
        "length_m": float(front + rear),
        "width_m": float(left_width + right_width),
    }


def animal_motion_from_metadata(
    metadata: Mapping[str, Any],
    *,
    camera_name: Optional[str] = "CAM_F0",
) -> dict[str, Any]:
    """Build the motion contract of a standing inserted animal."""
    placement = placement_mapping(metadata)
    if placement.get("benign") is True:
        raise ValueError("Benign animal insertions are not supported")
    placement_mode = metadata.get("placement_mode", placement.get("placement_mode"))
    if placement_mode != "future_timestep":
        raise ValueError(
            "Geometry evaluation requires placement_mode='future_timestep', "
            f"got {placement_mode!r}"
        )
    asset = metadata.get("asset")
    if not isinstance(asset, Mapping):
        raise ValueError("Inserted animal metadata has no asset record")
    geometry = normalize_animal_geometry(
        asset.get("actor_geometry"), expected_asset_id=asset.get("id")
    )

    behavior = metadata.get("actor_behavior", metadata.get("pedestrian_behavior"))
    speed = float(metadata.get("pedestrian_speed_mps", math.nan))
    walking_velocity = finite_xy(
        metadata.get(
            "walking_velocity_global_xy_mps",
            placement.get("walking_velocity_global_xy_mps"),
        ),
        "animal walking velocity",
    )
    if behavior != "standing" or speed != 0.0 or np.any(walking_velocity != 0.0):
        raise ValueError(
            "Only standing inserted animals are supported, got "
            f"behavior={behavior!r}, speed={speed!r}"
        )

    current_record = current_front_record(metadata, camera_name)
    actor_current = np.asarray(current_record.get("actor_current"), dtype=np.float64)
    if actor_current.ndim != 1 or actor_current.size < 2:
        raise ValueError(
            "current CAM_F0 actor_current must contain at least x and y, "
            f"got {current_record.get('actor_current')!r}"
        )
    initial_center = finite_xy(actor_current[:2], "current animal root")
    yaw_lidar = float(current_record.get("yaw_lidar", math.nan))
    if not math.isfinite(yaw_lidar):
        raise ValueError(
            "Animal metadata requires a finite current actor yaw_lidar, "
            f"got {current_record.get('yaw_lidar')!r}"
        )
    # The same rendered-heading convention as the pedestrian contract.
    heading = yaw_lidar + local_forward_angle_rad(geometry["local_forward_axis"])
    heading = math.atan2(math.sin(heading), math.cos(heading))

    interception_center = finite_xy(
        metadata.get(
            "anchor_current_ego_xy_m", placement.get("anchor_current_ego_xy_m")
        ),
        "interception center in current-ego coordinates",
    )
    interception_time_s = float(
        metadata.get(
            "pedestrian_time_to_interception_s",
            placement.get("pedestrian_time_to_interception_s", math.nan),
        )
    )
    if not math.isfinite(interception_time_s) or interception_time_s <= 0:
        raise ValueError(
            "pedestrian_time_to_interception_s must be finite and positive, "
            f"got {interception_time_s}"
        )
    future_timestep = metadata.get("future_timestep", placement.get("future_timestep"))
    if (
        isinstance(future_timestep, bool)
        or not isinstance(future_timestep, int)
        or future_timestep <= 0
    ):
        raise ValueError(
            f"future_timestep must be a positive integer, got {future_timestep!r}"
        )
    z_min, z_max = geometry["local_bounds_m"]["z"]
    return validated_motion(
        {
            "actor_type": "animal",
            "species": geometry["species"],
            "behavior": "standing",
            "initial_center_xy_m": initial_center.tolist(),
            "velocity_xy_mps": [0.0, 0.0],
            "heading_rad": float(heading),
            **animal_footprint_extents(geometry),
            "height_m": float(z_max - z_min),
            "animal_geometry": geometry,
            "geometry_source": geometry["source"],
            "geometry_profile_version": geometry["profile_version"],
            "geometry_asset_id": geometry["asset_id"],
            "target_time_s": interception_time_s,
            "interception_time_s": interception_time_s,
            "interception_center_xy_m": interception_center.tolist(),
            "future_timestep": future_timestep,
            "placement_version": metadata.get(
                "placement_version", placement.get("placement_version")
            ),
        }
    )
