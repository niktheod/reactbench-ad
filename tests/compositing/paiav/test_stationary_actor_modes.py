from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from reactbench.scoring.actor_motion import actor_motion_from_metadata
from reactbench.compositing.paiav import edit as paiav_edit
from reactbench.compositing.paiav import compact_run as paiav_compact


compositor = paiav_edit.compositor
CAR_BOUNDS = {
    "x": [-2.5, 2.5],
    "y": [-1.0, 1.0],
    "z": [0.0, 1.8],
}


def ego_pose(x: float, y: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
    pose[:2, 3] = [x, y]
    return pose


def scene_frames(poses: list[np.ndarray]) -> list[dict]:
    return [
        {
            "timestamp": index * 100_000,
            "ego2global": pose,
            "map_location": "test-map",
        }
        for index, pose in enumerate(poses)
    ]


def xoma_document() -> dict:
    return {
        "metadata": {
            "name": "Fixture Car",
            "uuid": "fixture-car-uuid",
            "authors": ["Fixture Author"],
            "license": "MPL-2.0 AND CC-BY-4.0",
            "copyrights": ["Copyright Fixture Author"],
            "openMaterial3dVersion": "1.0.0",
            "assetVersion": "test",
            "assetType": "object",
            "objectClass": "vehicle",
            "boundingBox": CAR_BOUNDS,
            "vehicleClassData": {
                "vehicleCategory": "car",
                "axles": {
                    "frontAxle": {
                        "positionX": 1.5,
                        "trackWidth": 1.7,
                        "wheelDiameter": 0.8,
                    },
                    "rearAxle": {
                        "positionX": -1.4,
                        "trackWidth": 1.7,
                        "wheelDiameter": 0.8,
                    },
                },
            },
            "animated": False,
        }
    }


def make_car_asset(root: Path) -> Path:
    asset_dir = root / "nested" / "fixture_car"
    asset_dir.mkdir(parents=True)
    blend = asset_dir / "fixture_car.blend"
    blend.write_bytes(b"BLENDER-v300 fixture")
    xoma = blend.with_suffix(".xoma")
    document = xoma_document()
    xoma.write_text(json.dumps(document), encoding="utf-8")
    metadata = document["metadata"]
    preparation = {
        "schema_version": compositor.CAR_PREPARATION_SCHEMA_VERSION,
        "preparer_version": "openx_paiav_test_fixture_v1",
        "kind": "stationary_car",
        "id": metadata["uuid"],
        "name": metadata["name"],
        "root_object": compositor.CAR_ROOT_OBJECT_NAME,
        "attribution": {
            key: metadata[key] for key in ("authors", "license", "copyrights")
        },
        "output": {
            "blend": blend.name,
            "blend_sha256": compositor.file_sha256(blend),
            "xoma": xoma.name,
            "xoma_sha256": compositor.file_sha256(xoma),
            "self_contained": True,
        },
        "geometry": {"xoma_bounds_m": metadata["boundingBox"]},
        "validation": {
            "passed": True,
            "geometry_transformation_applied": False,
        },
        "source": {"repository": "fixture"},
    }
    blend.with_suffix(".preparation.json").write_text(
        json.dumps(preparation), encoding="utf-8"
    )
    return blend


def test_cli_modes_are_mutually_exclusive_and_choose_local_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["edit.py", "--run-name", "standing", "--timestep", "3", "--standing"],
    )
    standing = paiav_edit.parse_args()
    paiav_edit.validate_args(standing)
    assert standing.standing is True
    assert standing.car is False
    assert standing.asset_dir == paiav_edit.DEFAULT_ASSET_DIR

    monkeypatch.setattr(
        sys,
        "argv",
        ["edit.py", "--run-name", "car", "--timestep", "3", "--car"],
    )
    car = paiav_edit.parse_args()
    paiav_edit.validate_args(car)
    assert car.car is True
    assert car.standing is False
    assert car.occlusion_mode == "dense"
    assert car.asset_dir == paiav_edit.DEFAULT_CAR_ASSET_DIR

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edit.py",
            "--run-name",
            "invalid",
            "--timestep",
            "3",
            "--standing",
            "--car",
        ],
    )
    with pytest.raises(SystemExit):
        paiav_edit.parse_args()


@pytest.mark.parametrize("mode", ["standing", "car"])
def test_stationary_modes_skip_direction_files(tmp_path: Path, mode: str) -> None:
    args = SimpleNamespace(
        standing=mode == "standing",
        car=mode == "car",
        direction_classifications=tmp_path / "does-not-exist.json",
    )

    provenance = paiav_edit.configure_directions(args, ["clip-a"], tmp_path)

    assert args.direction_classifications is None
    assert args.direction_classification_labels == {}
    assert args.direction_classifications_sha256 is None
    assert provenance == {
        "source": "not_applicable_stationary_actor",
        "source_path": None,
        "resolved_path": None,
        "sha256": None,
        "policy": None,
        "counts": {},
    }
    assert not (tmp_path / "direction_classifications.json").exists()


def test_paiav_placement_requests_use_ten_hertz_stationary_contracts() -> None:
    standing = paiav_edit.placement_request(
        SimpleNamespace(timestep=3, standing=True, car=False)
    )
    car = paiav_edit.placement_request(
        SimpleNamespace(timestep=3, standing=False, car=True)
    )

    assert standing["expected_future_offset_s"] == pytest.approx(0.3)
    assert standing["placement_version"] == paiav_edit.PAIAV_STANDING_PLACEMENT_VERSION
    assert standing["pedestrian_behavior"] == "standing"
    assert car["expected_future_offset_s"] == pytest.approx(0.3)
    assert car["placement_version"] == paiav_edit.PAIAV_CAR_PLACEMENT_VERSION
    assert car["vehicle_behavior"] == "stationary"




def test_recursive_car_pool_loads_geometry_and_attribution(tmp_path: Path) -> None:
    make_car_asset(tmp_path)

    assert paiav_edit.detect_asset_profile(tmp_path, car=True) == "car"
    assets = compositor.load_asset_pool(
        SimpleNamespace(asset_dir=tmp_path, car=True)
    )

    assert len(assets) == 1
    asset = assets[0]
    assert asset["id"] == "fixture-car-uuid"
    assert asset["actor_root_name"] == compositor.CAR_ROOT_OBJECT_NAME
    assert asset["actor_geometry"]["rear_bumper_x_m"] == -2.5
    assert len(asset["actor_geometry"]["wheel_contact_patches"]) == 4
    assert asset["attribution"]["license"] == "MPL-2.0 AND CC-BY-4.0"
    assert asset["preparation"]["validation_passed"] is True


def test_car_wheel_contacts_are_accepted_by_finalization() -> None:
    render_info = {
        "foot_contact_geometry": {
            "method": "xoma_axle_wheel_contact_patches_v1",
            "contact_profile": compositor.CAR_CONTACT_PROFILE,
            "receiver_plane_z_m": 0.0,
            "contact_distance_m": 0.03,
            "feet": {
                "rear_left": {
                    "contact": True,
                    "status": "measured_from_xoma_wheel_patch",
                    "minimum_receiver_distance_m": 0.002,
                    "ideal_contact_points_px": [
                        [28.0, 44.0],
                        [36.0, 44.0],
                        [36.0, 48.0],
                        [28.0, 48.0],
                    ],
                },
                "rear_right": {
                    "contact": True,
                    "status": "measured_from_xoma_wheel_patch",
                    "minimum_receiver_distance_m": 0.003,
                    "ideal_contact_points_px": [
                        [64.0, 44.0],
                        [72.0, 44.0],
                        [72.0, 48.0],
                        [64.0, 48.0],
                    ],
                },
            },
        }
    }
    alpha = np.zeros((60, 100), dtype=np.uint8)
    alpha[8:50, 20:80] = 255

    contact, diagnostics = compositor.ideal_actor_contact_region(
        render_info, alpha
    )

    assert np.count_nonzero(contact) > 0
    assert diagnostics["status"] == "measured_contact"
    assert diagnostics["contact_sides"] == ["rear_left", "rear_right"]
    assert diagnostics["source_geometry"]["method"] == (
        "xoma_axle_wheel_contact_patches_v1"
    )


def test_car_pool_rejects_missing_xoma_and_incomplete_attribution(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "car.blend").write_bytes(b"fixture")
    with pytest.raises(FileNotFoundError, match="XOMA sidecar"):
        paiav_edit.detect_asset_profile(missing, car=True)

    missing_preparation = tmp_path / "missing-preparation"
    missing_preparation.mkdir()
    (missing_preparation / "car.blend").write_bytes(b"fixture")
    (missing_preparation / "car.xoma").write_text(
        json.dumps(xoma_document()), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="preparation sidecar"):
        paiav_edit.detect_asset_profile(missing_preparation, car=True)

    blend = make_car_asset(tmp_path / "incomplete")
    document = xoma_document()
    document["metadata"]["authors"] = []
    blend.with_suffix(".xoma").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string list"):
        compositor.load_car_asset(blend)


def test_standing_actor_is_world_stationary_and_faces_future_ego() -> None:
    frames = scene_frames([ego_pose(2.0 * index) for index in range(5)])
    trajectory = compositor.resolve_standing_trajectory(
        frames,
        [0.0, 0.1, 0.2, 0.3],
        timestep=1,
        front_bumper_offset_m=4.0,
    )

    np.testing.assert_allclose(
        trajectory["actor_global_xys"],
        np.repeat([[12.0, 0.0]], compositor.EXPECTED_HISTORY_FRAMES, axis=0),
    )
    np.testing.assert_allclose(
        trajectory["standing_facing_direction_global_xy"], [-1.0, 0.0]
    )
    np.testing.assert_array_equal(
        trajectory["actor_velocity_global_xy_mps"], [0.0, 0.0]
    )
    placement = trajectory["placement"]
    assert placement["future_offset_s"] == pytest.approx(0.1)
    assert placement["pedestrian_time_to_interception_s"] == pytest.approx(0.1)
    assert placement["standing_pose_version"] == compositor.STANDING_POSE_VERSION


def test_car_rear_bumper_first_touches_at_requested_paiav_step() -> None:
    frames = scene_frames([ego_pose(2.0 * index) for index in range(6)])
    trajectory = compositor.resolve_stationary_vehicle_trajectory(
        frames,
        [0.0, 0.1, 0.2, 0.3],
        timestep=2,
        front_bumper_offset_m=4.0,
        car_local_bounds=CAR_BOUNDS,
        ego_rear_length_m=1.1,
        ego_width_m=2.3,
    )

    np.testing.assert_allclose(
        trajectory["actor_global_xys"], np.repeat([[16.5, 0.0]], 4, axis=0)
    )
    placement = trajectory["placement"]
    assert placement["vehicle_rear_bumper_contact_global_xy_m"] == [14.0, 0.0]
    assert placement["time_to_contact_s"] == pytest.approx(0.2)
    assert placement["bumper_contact_error_m"] == pytest.approx(0.0)
    checks = placement["first_contact_validation"]["per_timestep"]
    assert [item["future_timestep"] for item in checks] == [0, 1, 2]
    assert checks[-1]["contact_within_tolerance"] is True
    assert all(item["interior_overlap"] is False for item in checks)


def test_car_rejects_contact_before_requested_step() -> None:
    frames = scene_frames([ego_pose(0.0) for _ in range(6)])
    with pytest.raises(compositor.TokenFailure, match="before the requested timestep"):
        compositor.resolve_stationary_vehicle_trajectory(
            frames,
            [0.0, 0.1, 0.2, 0.3],
            timestep=2,
            front_bumper_offset_m=4.0,
            car_local_bounds=CAR_BOUNDS,
            ego_rear_length_m=1.1,
            ego_width_m=2.3,
        )


def test_car_collision_uses_oriented_vehicle_bounds() -> None:
    inside = np.asarray([[1.8, 0.8, 0.8]] * 6, dtype=np.float64)
    outside = np.asarray(
        [[2.8, 0.0, 0.8], [0.0, 1.3, 0.8], [0.0, 0.0, 0.05]],
        dtype=np.float64,
    )

    diagnostics = compositor.stationary_vehicle_collision_diagnostics(
        np.concatenate([inside, outside]), np.eye(4), CAR_BOUNDS
    )

    assert diagnostics["method"] == "oriented_vehicle_local_bounds_v1"
    assert diagnostics["obstacle_points"] == 6
    assert diagnostics["occupied"] is True


@pytest.mark.parametrize("mode", ["standing", "car"])
def test_stationary_metadata_omits_walking_only_fields(mode: str) -> None:
    geometry = {"profile_version": "fixture_geometry_v1"}
    prepared = {"actor_mode": mode, "actor_geometry": geometry}

    fields = compositor.actor_metadata_fields(
        SimpleNamespace(speed_mps=1.4, standing=mode == "standing", car=mode == "car"),
        prepared,
    )

    expected_behavior = "stationary" if mode == "car" else "standing"
    assert fields["actor_behavior"] == expected_behavior
    assert {
        "direction_classification",
        "crossing_direction",
        "final_crossing_offset_m",
        "crossing_offsets_m",
        "walking_velocity_global_xy_mps",
    }.isdisjoint(fields)
    if mode == "car":
        assert fields["vehicle_geometry"] is geometry
    else:
        assert fields["pedestrian_geometry"] is geometry
        assert fields["pedestrian_speed_mps"] == 0.0


def test_stationary_annotation_omits_walking_pose_sequence() -> None:
    output_paths = {
        (index, camera): f"{camera}/{index}.png"
        for camera in paiav_edit.CAMERA_NAMES
        for index in range(4)
    }
    annotation = paiav_edit.paiav_annotation_sample(
        "clip-a", {"actor_mode": "standing"}, output_paths
    )
    crossing = paiav_edit.paiav_annotation_sample(
        "clip-b", {"actor_mode": "crossing"}, output_paths
    )

    assert "animation_pose_sequence" not in annotation
    assert crossing["animation_pose_sequence"] == list(paiav_edit.POSE_SEQUENCE)


def test_evaluator_accepts_new_standing_metadata_without_walking_fields() -> None:
    asset_sidecar = paiav_edit.DEFAULT_ASSET_DIR / "rocketbox_female_adult_01.json"
    if not asset_sidecar.is_file():
        pytest.skip("needs the generated Rocketbox assets (scripts/assets.sh humans)")
    geometry = json.loads(asset_sidecar.read_text(encoding="utf-8"))[
        "actor_geometry"
    ]
    metadata = {
        "actor_mode": "standing",
        "actor_type": "pedestrian",
        "actor_behavior": "standing",
        "placement_mode": "future_timestep",
        "future_timestep": 3,
        "pedestrian_time_to_interception_s": 0.3,
        "pedestrian_speed_mps": 0.0,
        "pedestrian_geometry": geometry,
        "anchor_current_ego_xy_m": [10.0, 0.0],
        "records": [
            {
                "camera": camera,
                "frame_index": 3,
                "actor_current": [10.0, 0.0, 0.0],
                "yaw_lidar": -math.pi / 2.0,
            }
            for camera in paiav_edit.CAMERA_NAMES
        ],
    }

    motion = actor_motion_from_metadata(metadata, camera_name=None)

    assert motion["actor_type"] == "pedestrian"
    assert motion["behavior"] == "standing"
    np.testing.assert_array_equal(motion["velocity_xy_mps"], [0.0, 0.0])
    assert motion["target_time_s"] == pytest.approx(0.3)






def test_renderer_source_supports_generic_roots_and_static_pose() -> None:
    renderer = (
        Path(paiav_edit.__file__).resolve().parent
        / "compositor"
        / "blender_asset"
        / "scripts"
        / "render_batch.py"
    ).read_text(encoding="utf-8")

    assert 'job.get("actor_root_name", "PED_RIG")' in renderer
    assert "collect_vehicle_contact_geometry" in renderer
    assert "configure_natural_standing_pose" in renderer
    assert 'if bool(job.get("standing_pose", False))' in renderer


def test_compactor_validates_stationary_metadata_contract(tmp_path: Path) -> None:
    token = "clip-a"
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata = {
        "status": "success",
        "token": token,
        "actor_mode": "standing",
        "actor_type": "pedestrian",
        "actor_behavior": "standing",
        "pedestrian_speed_mps": 0.0,
        "pedestrian_geometry": {"profile_version": "fixture"},
        "placement": {"pedestrian_behavior": "standing"},
        "annotation_sample": {"token": token, "actor_mode": "standing"},
    }
    path = metadata_dir / f"{token}.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    run_config = {
        "actor_mode": "standing",
        "directions": {"source": "not_applicable_stationary_actor"},
    }

    paiav_compact.validate_actor_mode_contract(tmp_path, run_config, {token})

    metadata["crossing_direction"] = "left_to_right"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(
        paiav_compact.CompactionError, match="walking-only fields"
    ):
        paiav_compact.validate_actor_mode_contract(tmp_path, run_config, {token})


def test_compactor_requires_car_contact_and_licence_provenance(
    tmp_path: Path,
) -> None:
    token = "clip-a"
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata = {
        "status": "success",
        "token": token,
        "actor_mode": "car",
        "actor_type": "car",
        "actor_behavior": "stationary",
        "actor_speed_mps": 0.0,
        "vehicle_geometry": {"profile_version": "fixture"},
        "first_contact_validation": {"target_contact": True},
        "placement": {"vehicle_behavior": "stationary"},
        "asset": {
            "attribution": {"license": "CC-BY-4.0"},
            "preparation": {
                "validation_passed": True,
                "self_contained": True,
            },
        },
        "annotation_sample": {"token": token, "actor_mode": "car"},
    }
    path = metadata_dir / f"{token}.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    run_config = {
        "actor_mode": "car",
        "directions": {"source": "not_applicable_stationary_actor"},
    }

    paiav_compact.validate_actor_mode_contract(tmp_path, run_config, {token})

    metadata["asset"]["attribution"]["license"] = ""
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(
        paiav_compact.CompactionError, match="Car metadata contract"
    ):
        paiav_compact.validate_actor_mode_contract(tmp_path, run_config, {token})
