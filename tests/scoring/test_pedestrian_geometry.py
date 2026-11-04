from __future__ import annotations

import copy
import math

import pytest

from reactbench.scoring.actor_motion import pedestrian_motion_from_metadata
from reactbench.scoring.pedestrian_geometry import (
    PEDESTRIAN_GEOMETRY_PROFILE_VERSION,
    make_pedestrian_geometry,
    pedestrian_footprint_extents,
    union_local_bounds,
)



def measured_geometry(local_forward_axis: str = "-Y") -> dict:
    return make_pedestrian_geometry(
        asset_id="pedestrian_test_asset",
        local_bounds_m={
            "x": [-0.31, 0.35],
            "y": [-0.22, 0.27],
            "z": [0.0, 1.81],
        },
        local_forward_axis=local_forward_axis,
        frame_start=1,
        frame_end_exclusive=33,
        measured_mesh_objects=["PED_BODY", "PED_HAIR"],
        evaluated_vertex_samples=6400,
        blender_version="test",
    )


def motion_metadata(*, direction: str = "left", standing: bool = False) -> dict:
    geometry = measured_geometry()
    if standing:
        anchor = [8.0, 0.0]
        speed = 0.0
        yaw = math.pi / 2.0
    elif direction == "left":
        anchor = [8.0, 1.4]
        speed = 1.4
        yaw = math.pi
    else:
        anchor = [8.0, -1.4]
        speed = 1.4
        yaw = 0.0
    return {
        "actor_type": "pedestrian",
        "placement_mode": "future_timestep",
        "placement_version": "test",
        "future_timestep": 2,
        "pedestrian_time_to_interception_s": 1.0,
        "pedestrian_speed_mps": speed,
        "pedestrian_behavior": "standing" if standing else "moving",
        "anchor_current_ego_xy_m": anchor,
        "pedestrian_geometry": geometry,
        "asset": {
            "id": geometry["asset_id"],
            "actor_geometry": geometry,
        },
        "records": [
            {
                "frame_index": 3,
                "camera": "CAM_F0",
                "actor_current": [8.0, 0.0, 0.0],
                "yaw_lidar": yaw,
            }
        ],
    }


def test_union_bounds_cover_every_measured_gait_pose() -> None:
    frame_bounds = [
        {"x": [-0.20, 0.21], "y": [-0.30, 0.25], "z": [0.0, 1.75]},
        {"x": [-0.31, 0.19], "y": [-0.22, 0.40], "z": [-0.01, 1.81]},
        {"x": [-0.18, 0.35], "y": [-0.41, 0.24], "z": [0.01, 1.77]},
    ]

    union = union_local_bounds(frame_bounds)

    assert union == {
        "x": [-0.31, 0.35],
        "y": [-0.41, 0.40],
        "z": [-0.01, 1.81],
    }
    for bounds in frame_bounds:
        for axis in ("x", "y", "z"):
            assert union[axis][0] <= bounds[axis][0]
            assert union[axis][1] >= bounds[axis][1]


def test_local_forward_axis_maps_asymmetric_bounds_to_footprint() -> None:
    extents = pedestrian_footprint_extents(measured_geometry("-Y"))

    assert extents == pytest.approx(
        {
            "front_length_m": 0.22,
            "rear_length_m": 0.27,
            "left_width_m": 0.35,
            "right_width_m": 0.31,
            "length_m": 0.49,
            "width_m": 0.66,
        }
    )


def test_motion_adapter_preserves_left_right_and_standing_orientation() -> None:
    adapters = (pedestrian_motion_from_metadata,)  # the one adapter every runner uses
    for metadata, expected_heading in (
        (motion_metadata(direction="left"), math.pi / 2.0),
        (motion_metadata(direction="right"), -math.pi / 2.0),
        (motion_metadata(standing=True), 0.0),
    ):
        motions = [adapter(metadata) for adapter in adapters]
        assert all(motion == motions[0] for motion in motions[1:])
        assert motions[0]["heading_rad"] == pytest.approx(expected_heading)
        assert motions[0]["geometry_profile_version"] == (
            PEDESTRIAN_GEOMETRY_PROFILE_VERSION
        )

    paiav_metadata = motion_metadata(direction="left")
    paiav_metadata["records"] = [
        {**record, "camera": camera}
        for camera in ("camera_front_wide_120fov", "camera_cross_left_120fov")
        for record in paiav_metadata["records"]
    ]
    paiav_motion = pedestrian_motion_from_metadata(paiav_metadata, camera_name=None)
    assert paiav_motion == adapters[0](motion_metadata(direction="left"))


def test_motion_adapter_accepts_terrain_aligned_heading_offset() -> None:
    metadata = motion_metadata(direction="left")
    metadata["records"][0]["yaw_lidar"] += 0.002
    adapters = (pedestrian_motion_from_metadata,)  # the one adapter every runner uses

    motions = [adapter(metadata) for adapter in adapters]
    paiav_metadata = copy.deepcopy(metadata)
    paiav_metadata["records"] = [
        {**record, "camera": camera}
        for camera in ("camera_front_wide_120fov", "camera_cross_left_120fov")
        for record in paiav_metadata["records"]
    ]
    motions.append(
        pedestrian_motion_from_metadata(paiav_metadata, camera_name=None)
    )

    assert all(motion == motions[0] for motion in motions[1:])
    assert motions[0]["velocity_xy_mps"] == pytest.approx([0.0, 1.4])
    assert motions[0]["heading_rad"] == pytest.approx(math.pi / 2.0 + 0.002)


def test_legacy_metadata_without_measured_geometry_is_rejected() -> None:
    metadata = motion_metadata()
    metadata.pop("pedestrian_geometry")

    with pytest.raises(ValueError, match="legacy fixed-footprint samples"):
        pedestrian_motion_from_metadata(metadata)




def test_unknown_geometry_version_is_rejected() -> None:
    metadata = motion_metadata()
    metadata["pedestrian_geometry"] = copy.deepcopy(
        metadata["pedestrian_geometry"]
    )
    metadata["pedestrian_geometry"]["profile_version"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported pedestrian geometry"):
        pedestrian_motion_from_metadata(metadata)


def test_noninteger_measurement_count_is_rejected() -> None:
    metadata = motion_metadata()
    metadata["pedestrian_geometry"] = copy.deepcopy(
        metadata["pedestrian_geometry"]
    )
    metadata["pedestrian_geometry"]["measurement_frame_count"] = 32.5

    with pytest.raises(ValueError, match="must contain integers"):
        pedestrian_motion_from_metadata(metadata)
