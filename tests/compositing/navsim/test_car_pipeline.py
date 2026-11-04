from __future__ import annotations

from argparse import Namespace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as compositor  # noqa: E402


def ego_pose(x: float, y: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
    pose[:2, 3] = [x, y]
    return pose


def scene_frames(poses: list[np.ndarray]) -> list[dict]:
    return [
        {
            "timestamp": index * 500_000,
            "ego2global": pose,
            "map_location": "test-map",
        }
        for index, pose in enumerate(poses)
    ]


CAR_BOUNDS = {
    "x": [-2.5, 2.5],
    "y": [-1.0, 1.0],
    "z": [0.0, 1.8],
}


class CarCliTest(unittest.TestCase):
    @staticmethod
    def argv(*extra: str) -> list[str]:
        return [
            "batch_composite_navsim_blender.py",
            "--scene-filter",
            "filter.yaml",
            "--timestep",
            "2",
            "--run-name",
            "car-run",
            "--asset-dir",
            "assets",
            *extra,
        ]

    def test_car_is_contact_only_and_needs_no_directions(self) -> None:
        with patch.object(sys, "argv", self.argv("--car", "--contact-only")):
            args = compositor.parse_args()

        compositor.validate_actor_mode(args)
        compositor.configure_direction_classifications(args)

        self.assertTrue(args.car)
        self.assertFalse(args.standing)
        self.assertIsNone(args.direction_classifications)
        self.assertEqual(args.direction_classification_labels, {})
        self.assertEqual(
            compositor.placement_request(args)["placement_version"],
            compositor.CAR_PLACEMENT_VERSION,
        )

    def test_car_rejects_non_contact_output_modes(self) -> None:
        with patch.object(sys, "argv", self.argv("--car", "--preview")):
            args = compositor.parse_args()
        with self.assertRaisesRegex(ValueError, "requires --contact-only"):
            compositor.validate_actor_mode(args)

    def test_car_and_standing_are_mutually_exclusive(self) -> None:
        with patch.object(
            sys,
            "argv",
            self.argv("--car", "--standing", "--contact-only"),
        ):
            with self.assertRaises(SystemExit):
                compositor.parse_args()


class OpenXAssetTest(unittest.TestCase):
    @staticmethod
    def xoma_document() -> dict:
        return {
            "metadata": {
                "name": "Test Car",
                "uuid": "test-car-uuid",
                "authors": ["Asset Author"],
                "license": "MPL-2.0 AND CC-BY-4.0",
                "copyrights": ["Copyright Asset Author"],
                "openMaterial3dVersion": "1.0.0",
                "assetVersion": "1.0",
                "assetType": "object",
                "objectClass": "vehicle",
                "boundingBox": CAR_BOUNDS,
                "vehicleClassData": {
                    "vehicleCategory": "car",
                    "axles": {
                        "frontAxle": {
                            "positionX": 1.5,
                            "positionZ": 0.4,
                            "trackWidth": 1.7,
                            "wheelDiameter": 0.8,
                        },
                        "rearAxle": {
                            "positionX": -1.4,
                            "positionZ": 0.4,
                            "trackWidth": 1.7,
                            "wheelDiameter": 0.8,
                        },
                    },
                },
            }
        }

    def test_recursive_pool_loads_xoma_geometry_and_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_dir = root / "nested" / "test_car"
            asset_dir.mkdir(parents=True)
            blend = asset_dir / "test_car.blend"
            blend.write_bytes(b"packed blender fixture")
            blend.with_suffix(".xoma").write_text(
                json.dumps(self.xoma_document())
            )

            assets = compositor.load_asset_pool(
                Namespace(asset_dir=root, car=True)
            )

        self.assertEqual(len(assets), 1)
        asset = assets[0]
        self.assertEqual(asset["id"], "test-car-uuid")
        self.assertEqual(asset["actor_root_name"], "Grp_Root")
        self.assertEqual(
            asset["actor_geometry"]["rear_bumper_x_m"], -2.5
        )
        self.assertEqual(
            len(asset["actor_geometry"]["wheel_contact_patches"]), 4
        )
        self.assertEqual(
            asset["attribution"]["license"],
            "MPL-2.0 AND CC-BY-4.0",
        )


class StationaryCarPlacementTest(unittest.TestCase):
    def test_straight_path_first_touches_at_requested_timestep(self) -> None:
        frames = scene_frames([ego_pose(2.0 * index) for index in range(6)])
        trajectory = compositor.resolve_stationary_vehicle_trajectory(
            frames,
            [0.0, 0.5, 1.0, 1.5],
            timestep=2,
            front_bumper_offset_m=4.0,
            car_local_bounds=CAR_BOUNDS,
            ego_rear_length_m=1.1,
            ego_width_m=2.3,
        )

        np.testing.assert_allclose(
            trajectory["actor_global_xys"],
            np.repeat([[16.5, 0.0]], 4, axis=0),
        )
        placement = trajectory["placement"]
        self.assertEqual(
            placement["vehicle_rear_bumper_contact_global_xy_m"],
            [14.0, 0.0],
        )
        self.assertAlmostEqual(placement["bumper_contact_error_m"], 0.0)
        contact_validation = placement["first_contact_validation"]
        self.assertEqual(len(contact_validation["history_frame_checks"]), 3)
        checks = contact_validation["per_timestep"]
        self.assertEqual([item["future_timestep"] for item in checks], [0, 1, 2])
        self.assertTrue(checks[-1]["contact_within_tolerance"])
        self.assertTrue(all(not item["interior_overlap"] for item in checks))

    def test_asset_rear_bound_controls_origin_but_not_contact_point(self) -> None:
        frames = scene_frames([ego_pose(2.0 * index) for index in range(6)])
        for rear_x in (-1.5, -3.0):
            with self.subTest(rear_x=rear_x):
                bounds = {**CAR_BOUNDS, "x": [rear_x, 2.5]}
                trajectory = compositor.resolve_stationary_vehicle_trajectory(
                    frames,
                    [0.0, 0.5, 1.0, 1.5],
                    timestep=2,
                    front_bumper_offset_m=4.0,
                    car_local_bounds=bounds,
                    ego_rear_length_m=1.1,
                    ego_width_m=2.3,
                )
                np.testing.assert_allclose(
                    trajectory["anchor_global_xy"],
                    [14.0 - rear_x, 0.0],
                )
                self.assertEqual(
                    trajectory["placement"][
                        "vehicle_rear_bumper_contact_global_xy_m"
                    ],
                    [14.0, 0.0],
                )

    def test_car_heading_tracks_turning_future_ego(self) -> None:
        poses = [
            ego_pose(-6.0),
            ego_pose(-4.0),
            ego_pose(-2.0),
            ego_pose(0.0),
            ego_pose(10.0, yaw=math.pi / 2.0),
        ]
        trajectory = compositor.resolve_stationary_vehicle_trajectory(
            scene_frames(poses),
            [0.0, 0.5, 1.0, 1.5],
            timestep=1,
            front_bumper_offset_m=4.0,
            car_local_bounds=CAR_BOUNDS,
            ego_rear_length_m=1.1,
            ego_width_m=2.3,
        )

        np.testing.assert_allclose(
            trajectory["reference_vehicle_forward_global_xy"],
            [0.0, 1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory["anchor_global_xy"],
            [10.0, 6.5],
            atol=1e-12,
        )

    def test_rejects_contact_before_requested_timestep(self) -> None:
        frames = scene_frames([ego_pose(0.0) for _ in range(6)])
        with self.assertRaisesRegex(
            compositor.TokenFailure,
            "before the requested timestep",
        ):
            compositor.resolve_stationary_vehicle_trajectory(
                frames,
                [0.0, 0.5, 1.0, 1.5],
                timestep=2,
                front_bumper_offset_m=4.0,
                car_local_bounds=CAR_BOUNDS,
                ego_rear_length_m=1.1,
                ego_width_m=2.3,
            )


class StationaryCarCollisionTest(unittest.TestCase):
    def test_uses_oriented_vehicle_bounds_instead_of_pedestrian_radius(self) -> None:
        inside = np.asarray(
            [[1.8, 0.8, 0.8]] * 6,
            dtype=np.float64,
        )
        outside = np.asarray(
            [[2.8, 0.0, 0.8], [0.0, 1.3, 0.8], [0.0, 0.0, 0.05]],
            dtype=np.float64,
        )
        diagnostics = compositor.stationary_vehicle_collision_diagnostics(
            np.concatenate([inside, outside], axis=0),
            np.eye(4, dtype=np.float64),
            CAR_BOUNDS,
        )

        self.assertEqual(diagnostics["obstacle_points"], 6)
        self.assertTrue(diagnostics["occupied"])
        self.assertEqual(
            diagnostics["method"], "oriented_vehicle_local_bounds_v1"
        )

    def test_ground_intersection_preserves_global_xy_with_sensor_tilt(self) -> None:
        roll = math.radians(4.0)
        pitch = math.radians(-3.0)
        roll_rotation = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(roll), -math.sin(roll)],
                [0.0, math.sin(roll), math.cos(roll)],
            ],
            dtype=np.float64,
        )
        pitch_rotation = np.asarray(
            [
                [math.cos(pitch), 0.0, math.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch), 0.0, math.cos(pitch)],
            ],
            dtype=np.float64,
        )
        lidar_to_global = np.eye(4, dtype=np.float64)
        lidar_to_global[:3, :3] = pitch_rotation @ roll_rotation
        lidar_to_global[:3, 3] = [11.0, -4.0, 2.0]
        target_xy = np.asarray([18.25, 3.5], dtype=np.float64)

        point_lidar = compositor.ground_point_lidar_for_global_xy(
            lidar_to_global,
            [0.04, -0.03, -1.2],
            target_xy,
        )
        realized = compositor.geometry.transform_points(
            lidar_to_global,
            point_lidar[None],
        )[0]

        np.testing.assert_allclose(realized[:2], target_xy, atol=1e-8)




if __name__ == "__main__":
    unittest.main()
