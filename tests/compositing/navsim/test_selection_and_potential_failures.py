from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as compositor  # noqa: E402


class TimestepOnlyPlacementTest(unittest.TestCase):
    def test_final_placement_must_match_labeled_preview(self) -> None:
        frames = [
            {
                "timestamp": index * 500_000,
                "ego2global": np.eye(4, dtype=np.float64),
            }
            for index in range(compositor.EXPECTED_HISTORY_FRAMES + 1)
        ]
        placement = compositor.resolve_placement(
            frames,
            timestep=1,
            front_bumper_offset_m=4.0,
        )
        args = Namespace(
            shadow_label_lookup={("token_a", 2): {"placement": placement}}
        )
        compositor.validate_preview_placement(args, "token_a", placement)

        changed = {**placement, "future_timestep": 2}
        with self.assertRaises(compositor.TokenFailure) as context:
            compositor.validate_preview_placement(args, "token_a", changed)
        self.assertEqual(
            context.exception.code, "preview_final_placement_mismatch"
        )

    def test_preview_placement_accepts_last_bit_float_drift(self) -> None:
        preview_placement = {
            "placement_mode": "future_timestep",
            "future_timestep": 4,
            "anchor_current_ego_xy_m": [
                25.588824900688067,
                -0.8731280269889646,
            ],
        }
        current_placement = {
            **preview_placement,
            "anchor_current_ego_xy_m": [
                25.588824900688067,
                -0.8731280269889643,
            ],
        }
        args = Namespace(
            shadow_label_lookup={
                ("token_a", 2): {"placement": preview_placement}
            }
        )

        compositor.validate_preview_placement(
            args,
            "token_a",
            current_placement,
        )

    def test_preview_placement_rejects_material_float_drift(self) -> None:
        preview_placement = {
            "placement_mode": "future_timestep",
            "future_timestep": 4,
            "anchor_current_ego_xy_m": [25.0, -1.0],
        }
        current_placement = {
            **preview_placement,
            "anchor_current_ego_xy_m": [25.0, -0.999],
        }
        args = Namespace(
            shadow_label_lookup={
                ("token_a", 2): {"placement": preview_placement}
            }
        )

        with self.assertRaises(compositor.TokenFailure) as context:
            compositor.validate_preview_placement(
                args,
                "token_a",
                current_placement,
            )
        self.assertEqual(
            context.exception.code,
            "preview_final_placement_mismatch",
        )
        self.assertIn(
            "placement.anchor_current_ego_xy_m[1]",
            str(context.exception),
        )


class FutureTimestepPlacementTest(unittest.TestCase):
    @staticmethod
    def pose(x: float, y: float, yaw: float) -> np.ndarray:
        cosine, sine = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4, dtype=np.float64)
        pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
        pose[:2, 3] = [x, y]
        return pose

    @classmethod
    def scene_frames(
        cls,
        future_pose: np.ndarray,
    ) -> list[dict]:
        frames = [
            {
                "timestamp": index * 500_000,
                "ego2global": cls.pose(0.0, 0.0, 0.0),
                "map_location": "test-map",
            }
            for index in range(compositor.EXPECTED_HISTORY_FRAMES)
        ]
        frames.append(
            {
                "timestamp": compositor.EXPECTED_HISTORY_FRAMES * 500_000,
                "ego2global": future_pose,
                "map_location": "test-map",
            }
        )
        return frames

    def test_one_timestep_anchors_at_straight_future_bumper(self) -> None:
        frames = self.scene_frames(self.pose(2.0, 0.0, 0.0))
        trajectory = compositor.resolve_crossing_trajectory(
            frames,
            [0.0, 0.5, 1.0, 1.5],
            timestep=1,
            front_bumper_offset_m=4.0,
            lateral_sign=-1.0,
            speed_mps=2.0,
        )

        placement = trajectory["placement"]
        self.assertEqual(placement["placement_mode"], "future_timestep")
        self.assertEqual(placement["future_frame_index"], 4)
        self.assertAlmostEqual(placement["future_offset_s"], 0.5)
        np.testing.assert_allclose(
            placement["future_front_bumper_global_xy_m"],
            [6.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory["actor_global_xys"][-1],
            [6.0, 1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory["walking_direction_global_xy"],
            [0.0, -1.0],
            atol=1e-12,
        )
        self.assertAlmostEqual(
            placement["pedestrian_distance_to_interception_m"], 1.0
        )

    def test_actual_timestamp_interval_controls_default_speed_offset(
        self,
    ) -> None:
        frames = self.scene_frames(self.pose(2.0, 0.0, 0.0))
        frames[-1]["timestamp"] = 2_300_000
        trajectory = compositor.resolve_crossing_trajectory(
            frames,
            [0.0, 0.5, 1.0, 1.5],
            timestep=1,
            front_bumper_offset_m=4.0,
            lateral_sign=1.0,
            speed_mps=compositor.PEDESTRIAN_SPEED_MPS,
        )

        self.assertAlmostEqual(
            trajectory["placement"]["future_offset_s"], 0.8
        )
        self.assertAlmostEqual(
            trajectory["placement"][
                "pedestrian_distance_to_interception_m"
            ],
            1.12,
        )
        np.testing.assert_allclose(
            trajectory["actor_global_xys"][-1],
            [6.0, -1.12],
            atol=1e-12,
        )

    def test_regular_half_second_timesteps_give_requested_two_mps_offsets(
        self,
    ) -> None:
        frames = [
            {
                "timestamp": index * 500_000,
                "ego2global": self.pose(0.0, 0.0, 0.0),
                "map_location": "test-map",
            }
            for index in range(compositor.EXPECTED_HISTORY_FRAMES)
        ]
        frames.extend(
            [
                {
                    "timestamp": 2_000_000,
                    "ego2global": self.pose(2.0, 0.0, 0.0),
                    "map_location": "test-map",
                },
                {
                    "timestamp": 2_500_000,
                    "ego2global": self.pose(3.0, 0.0, 0.0),
                    "map_location": "test-map",
                },
            ]
        )

        for timestep, expected_distance_m in ((1, 1.0), (2, 2.0)):
            with self.subTest(timestep=timestep):
                trajectory = compositor.resolve_crossing_trajectory(
                    frames,
                    [0.0, 0.5, 1.0, 1.5],
                    timestep=timestep,
                    front_bumper_offset_m=4.0,
                    lateral_sign=1.0,
                    speed_mps=2.0,
                )
                remaining = (
                    trajectory["anchor_global_xy"]
                    - trajectory["actor_global_xys"][-1]
                )
                self.assertAlmostEqual(
                    float(np.linalg.norm(remaining)),
                    expected_distance_m,
                )
                self.assertAlmostEqual(
                    float(
                        np.dot(
                            remaining,
                            trajectory["walking_direction_global_xy"],
                        )
                    ),
                    expected_distance_m,
                )

    def test_turning_future_pose_rotates_anchor_and_crossing_axis(self) -> None:
        frames = self.scene_frames(self.pose(10.0, 20.0, np.pi / 2.0))
        trajectory = compositor.resolve_crossing_trajectory(
            frames,
            [0.0, 0.5, 1.0, 1.5],
            timestep=1,
            front_bumper_offset_m=4.0,
            lateral_sign=1.0,
            speed_mps=2.0,
        )

        np.testing.assert_allclose(
            trajectory["anchor_global_xy"],
            [10.0, 24.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory["reference_vehicle_forward_global_xy"],
            [0.0, 1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory["walking_direction_global_xy"],
            [-1.0, 0.0],
            atol=1e-12,
        )
        self.assertAlmostEqual(
            float(
                np.dot(
                    trajectory["reference_vehicle_forward_global_xy"],
                    trajectory["walking_direction_global_xy"],
                )
            ),
            0.0,
        )
        np.testing.assert_allclose(
            trajectory["actor_global_xys"][-1],
            [11.0, 24.0],
            atol=1e-12,
        )

    def test_standing_actor_remains_at_future_bumper_and_faces_ego(self) -> None:
        frames = self.scene_frames(self.pose(2.0, 0.0, 0.0))
        trajectory = compositor.resolve_standing_trajectory(
            frames,
            [0.0, 0.5, 1.0, 1.5],
            timestep=1,
            front_bumper_offset_m=4.0,
        )

        np.testing.assert_allclose(
            trajectory["actor_global_xys"],
            np.repeat([[6.0, 0.0]], compositor.EXPECTED_HISTORY_FRAMES, axis=0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory["standing_facing_direction_global_xy"],
            [-1.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            trajectory["walking_velocity_global_xy_mps"],
            [0.0, 0.0],
        )
        np.testing.assert_array_equal(
            trajectory["crossing_offsets_m"],
            np.zeros(compositor.EXPECTED_HISTORY_FRAMES),
        )
        placement = trajectory["placement"]
        self.assertEqual(placement["pedestrian_behavior"], "standing")
        self.assertEqual(
            placement["standing_facing_policy"],
            "opposite_future_ego_forward",
        )
        self.assertEqual(
            placement["standing_pose_version"],
            compositor.STANDING_POSE_VERSION,
        )
        self.assertEqual(placement["pedestrian_distance_to_interception_m"], 0.0)

    def test_standing_facing_direction_tracks_turning_future_ego(self) -> None:
        frames = self.scene_frames(self.pose(10.0, 20.0, np.pi / 2.0))
        trajectory = compositor.resolve_standing_trajectory(
            frames,
            [0.0, 0.5, 1.0, 1.5],
            timestep=1,
            front_bumper_offset_m=4.0,
        )

        np.testing.assert_allclose(
            trajectory["actor_global_xys"],
            np.repeat(
                [[10.0, 24.0]], compositor.EXPECTED_HISTORY_FRAMES, axis=0
            ),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory["standing_facing_direction_global_xy"],
            [0.0, -1.0],
            atol=1e-12,
        )

    def test_side_on_standing_actor_faces_future_ego_left_or_right(self) -> None:
        frames = self.scene_frames(self.pose(10.0, 20.0, np.pi / 2.0))
        timestamps = [0.0, 0.5, 1.0, 1.5]
        default = compositor.resolve_standing_trajectory(
            frames, timestamps, timestep=1, front_bumper_offset_m=4.0
        )
        self.assertNotIn("standing_facing_lateral_sign", default["placement"])
        changed_keys = {
            "standing_facing_direction_global_xy",
            "standing_facing_lateral_sign",
        }

        for lateral_sign, facing in ((1.0, [-1.0, 0.0]), (-1.0, [1.0, 0.0])):
            trajectory = compositor.resolve_standing_trajectory(
                frames,
                timestamps,
                timestep=1,
                front_bumper_offset_m=4.0,
                lateral_sign=lateral_sign,
            )
            np.testing.assert_array_equal(
                trajectory["actor_global_xys"], default["actor_global_xys"]
            )
            np.testing.assert_allclose(
                trajectory["standing_facing_direction_global_xy"],
                facing,
                atol=1e-12,
            )
            placement = trajectory["placement"]
            self.assertEqual(placement["standing_facing_lateral_sign"], lateral_sign)
            self.assertEqual(
                {k: v for k, v in placement.items() if k not in changed_keys},
                {
                    k: v
                    for k, v in default["placement"].items()
                    if k not in changed_keys
                },
            )

    def test_standing_rejects_non_unit_lateral_sign(self) -> None:
        frames = self.scene_frames(self.pose(2.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "lateral sign"):
            compositor.resolve_standing_trajectory(
                frames,
                [0.0, 0.5, 1.0, 1.5],
                timestep=1,
                front_bumper_offset_m=4.0,
                lateral_sign=0.5,
            )

    def test_rejects_nonpositive_or_unavailable_future_timestep(self) -> None:
        frames = self.scene_frames(self.pose(2.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compositor.resolve_placement(
                frames,
                timestep=0,
                front_bumper_offset_m=4.0,
            )
        with self.assertRaises(compositor.TokenFailure) as context:
            compositor.resolve_placement(
                frames,
                timestep=2,
                front_bumper_offset_m=4.0,
            )
        self.assertEqual(context.exception.code, "future_timestep_unavailable")

    def test_road_basis_tracks_future_vehicle_heading(self) -> None:
        basis = compositor.road_basis_aligned_to_global_forward(
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.eye(4, dtype=np.float64),
            [0.0, 1.0],
            max_tilt_degrees=2.0,
        )

        np.testing.assert_allclose(basis[:, 0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(basis[:, 1], [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(basis[:, 2], [0.0, 0.0, 1.0])


class BestEffortGroundingTest(unittest.TestCase):
    @staticmethod
    def road_points() -> np.ndarray:
        points = []
        for x in np.linspace(-2.0, 14.0, 33):
            for y in np.linspace(-10.0, 10.0, 41):
                for offset in (-0.02, 0.0, 0.02):
                    points.append(
                        [
                            x + offset,
                            y - offset,
                            0.01 * x - 0.005 * y - 0.9 + offset * 0.1,
                        ]
                    )
        return np.asarray(points, dtype=np.float64)

    @staticmethod
    def frames() -> list[dict]:
        return [
            {
                "ego2global": np.eye(4, dtype=np.float64),
                "lidar2global": np.eye(4, dtype=np.float64),
            }
            for _ in range(compositor.EXPECTED_HISTORY_FRAMES)
        ]

    def test_relaxed_fit_recovers_a_finite_local_plane(self) -> None:
        plane, diagnostics = compositor.fit_relaxed_ground_plane(
            self.road_points(),
            actor_xy=[6.0, 2.0],
            half_extent_m=4.0,
        )

        self.assertTrue(np.isfinite(plane).all())
        self.assertLess(diagnostics["rmse_m"], 0.05)
        self.assertGreater(diagnostics["grid_inlier_count"], 20)

    def test_strict_failures_are_flagged_while_all_frames_are_grounded(
        self,
    ) -> None:
        points = [
            self.road_points()
            for _ in range(compositor.EXPECTED_HISTORY_FRAMES)
        ]
        actor_xys = [
            [6.0, float(index - (compositor.EXPECTED_HISTORY_FRAMES - 1))]
            for index in compositor.HISTORY_FRAME_INDICES
        ]

        with patch.object(
            compositor,
            "fit_frame_ground",
            side_effect=compositor.TokenFailure(
                "ground_fit_failed",
                "strict support unavailable",
            ),
        ):
            planes, diagnostics, strict_failures = (
                compositor.fit_ground_sequence_best_effort(
                    self.frames(),
                    points,
                    actor_xys,
                    actor_xys,
                )
            )

        warning = compositor.ground_fit_potential_failure(
            "token",
            strict_failures,
            diagnostics,
        )
        self.assertEqual(
            len(planes),
            compositor.EXPECTED_HISTORY_FRAMES,
        )
        self.assertTrue(all(np.isfinite(plane).all() for plane in planes))
        self.assertEqual(
            len(strict_failures),
            compositor.EXPECTED_HISTORY_FRAMES,
        )
        self.assertIsNotNone(warning)
        self.assertEqual(warning["code"], "ground_fit_failed")
        self.assertEqual(
            warning["failed_frame_indices"],
            list(compositor.HISTORY_FRAME_INDICES),
        )
        self.assertTrue(
            all(item["first_attempt_failed"] for item in diagnostics)
        )
        self.assertTrue(
            all(
                item["method"] == "relaxed_same_timestamp_local_lidar"
                for item in diagnostics
            )
        )

    def test_strict_success_produces_no_review_warning(self) -> None:
        plane = np.array([0.01, -0.005, -0.9], dtype=np.float64)
        points = [
            self.road_points()
            for _ in range(compositor.EXPECTED_HISTORY_FRAMES)
        ]
        actor_xys = [
            [6.0, float(index)]
            for index in compositor.HISTORY_FRAME_INDICES
        ]

        with patch.object(
            compositor,
            "fit_frame_ground",
            return_value=(plane, {"rmse_m": 0.01}),
        ):
            planes, diagnostics, strict_failures = (
                compositor.fit_ground_sequence_best_effort(
                    self.frames(),
                    points,
                    actor_xys,
                    actor_xys,
                )
            )

        self.assertEqual(strict_failures, [])
        self.assertIsNone(
            compositor.ground_fit_potential_failure(
                "token",
                strict_failures,
                diagnostics,
            )
        )
        self.assertTrue(
            all(
                item["method"] == "strict_same_timestamp_local_lidar"
                for item in diagnostics
            )
        )
        for fitted in planes:
            np.testing.assert_allclose(fitted, plane)

    def test_accumulated_four_frame_fit_is_used_after_same_frame_fallbacks_fail(
        self,
    ) -> None:
        frames = self.frames()
        frame_points = [
            np.array([[float(index), 0.0, -1.0]], dtype=np.float64)
            for index in compositor.HISTORY_FRAME_INDICES
        ]
        actor_xys = [
            [float(index), 0.0]
            for index in compositor.HISTORY_FRAME_INDICES
        ]

        def retry(points, _actor_xy, _windows):
            if len(points) == 1:
                raise RuntimeError("same-frame support unavailable")
            return (
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
                {"half_extent_m": 4.0},
            )

        with patch.object(
            compositor,
            "fit_frame_ground",
            side_effect=compositor.TokenFailure(
                "ground_fit_failed",
                "strict support unavailable",
            ),
        ), patch.object(
            compositor,
            "_retry_relaxed_ground",
            side_effect=retry,
        ):
            planes, diagnostics, strict_failures = (
                compositor.fit_ground_sequence_best_effort(
                    frames,
                    frame_points,
                    actor_xys,
                    actor_xys,
                )
            )

        self.assertEqual(
            len(strict_failures),
            compositor.EXPECTED_HISTORY_FRAMES,
        )
        self.assertTrue(all(np.isfinite(plane).all() for plane in planes))
        self.assertTrue(
            all(
                item["method"] == "accumulated_four_frame_local_lidar"
                for item in diagnostics
            )
        )
        self.assertTrue(
            all(
                item["accumulated_frame_count"]
                == compositor.EXPECTED_HISTORY_FRAMES
                for item in diagnostics
            )
        )

    def test_unresolved_frames_transfer_the_nearest_resolved_plane(
        self,
    ) -> None:
        plane = np.array([0.01, -0.005, -0.9], dtype=np.float64)
        points = [
            self.road_points()
            for _ in range(compositor.EXPECTED_HISTORY_FRAMES)
        ]
        actor_xys = [
            [6.0, float(index)]
            for index in compositor.HISTORY_FRAME_INDICES
        ]
        strict_results = [
            (plane, {"rmse_m": 0.01}),
            *[
                compositor.TokenFailure(
                    "ground_fit_failed",
                    f"strict frame {index} unavailable",
                )
                for index in range(1, compositor.EXPECTED_HISTORY_FRAMES)
            ],
        ]

        with patch.object(
            compositor,
            "fit_frame_ground",
            side_effect=strict_results,
        ), patch.object(
            compositor,
            "_retry_relaxed_ground",
            side_effect=RuntimeError("fallback support unavailable"),
        ):
            planes, diagnostics, strict_failures = (
                compositor.fit_ground_sequence_best_effort(
                    self.frames(),
                    points,
                    actor_xys,
                    actor_xys,
                )
            )

        self.assertEqual(
            [item["frame_index"] for item in strict_failures],
            [1, 2, 3],
        )
        self.assertEqual(
            diagnostics[0]["method"],
            "strict_same_timestamp_local_lidar",
        )
        self.assertTrue(
            all(
                item["method"]
                == "nearest_resolved_timestep_plane_transfer"
                for item in diagnostics[1:]
            )
        )
        self.assertTrue(
            all(item["source_frame_index"] == 0 for item in diagnostics[1:])
        )
        for fitted in planes:
            np.testing.assert_allclose(fitted, plane)


class SeededSampleSelectionTest(unittest.TestCase):
    def test_subset_is_seeded_unique_and_keeps_source_order(self) -> None:
        tokens = [f"token_{index:03d}" for index in range(100)]

        selected = compositor.seeded_random_subset(tokens, 12, seed=42)
        repeated = compositor.seeded_random_subset(tokens, 12, seed=42)
        different_seed = compositor.seeded_random_subset(tokens, 12, seed=43)

        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(selected)), 12)
        self.assertNotEqual(selected, tokens[:12])
        self.assertNotEqual(selected, different_seed)
        self.assertEqual(
            [tokens.index(token) for token in selected],
            sorted(tokens.index(token) for token in selected),
        )

    def test_zero_and_oversized_limits(self) -> None:
        tokens = ["a", "b", "c"]
        self.assertEqual(compositor.seeded_random_subset(tokens, 0, 42), [])
        self.assertEqual(compositor.seeded_random_subset(tokens, 10, 42), tokens)
        self.assertEqual(compositor.seeded_random_subset(tokens, None, 42), tokens)

    def test_scene_filter_integration_does_not_take_prefix(self) -> None:
        tokens = [f"token_{index:03d}" for index in range(40)]
        args = Namespace(
            max_samples=5,
            seed=42,
        )

        selected = compositor.select_filter_tokens({"tokens": tokens}, args)

        self.assertEqual(len(selected), 5)
        self.assertNotEqual(selected, tokens[:5])


class SweptTrajectoryCollisionTest(unittest.TestCase):
    def test_detects_obstacle_along_diagonal_path(self) -> None:
        obstacle = np.tile(
            np.array([[0.5, 0.5, 1.0]], dtype=np.float64),
            (6, 1),
        )

        diagnostics = compositor.trajectory_collision_diagnostics(
            obstacle,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            actor_basis_lidar=np.eye(3, dtype=np.float64),
            actor_heading=0.0,
            local_bounds_m={
                "x": [-0.3, 0.3],
                "y": [-0.3, 0.3],
                "z": [0.0, 1.8],
            },
        )

        self.assertTrue(diagnostics["occupied"])
        self.assertGreater(diagnostics["sample_count"], 2)
        self.assertEqual(diagnostics["path_start_xy_m"], [0.0, 0.0])
        self.assertEqual(diagnostics["path_end_xy_m"], [1.0, 1.0])


class PotentialFailureTest(unittest.TestCase):
    def collision_diagnostics(self) -> dict:
        return {
            "per_frame": [
                {"frame_index": 0, "occupied": False, "obstacle_points": 0},
                {"frame_index": 1, "occupied": True, "obstacle_points": 8},
            ],
            "swept_final_frame": {
                "occupied": True,
                "maximum_obstacle_points": 74,
            },
            "occupied": True,
        }

    def test_collision_becomes_review_warning(self) -> None:
        args = Namespace(
            disable_collision_check=False,
            speed_mps=2.0,
        )
        placement = {
            "future_timestep": 1,
            "pedestrian_distance_to_interception_m": 0.7,
        }
        warning = compositor.trajectory_collision_potential_failure(
            args,
            "token_a",
            self.collision_diagnostics(),
            placement,
        )

        self.assertIsNotNone(warning)
        self.assertEqual(warning["status"], "potential_failure")
        self.assertEqual(warning["code"], "trajectory_collision")
        self.assertIn("per_frame=[1]", warning["message"])
        self.assertEqual(
            warning["collision_diagnostics"], self.collision_diagnostics()
        )
        self.assertEqual(warning["placement"], placement)

    def test_disabled_reporting_returns_no_warning(self) -> None:
        args = Namespace(
            disable_collision_check=True,
            speed_mps=2.0,
        )
        self.assertIsNone(
            compositor.trajectory_collision_potential_failure(
                args,
                "token_a",
                self.collision_diagnostics(),
                {"future_timestep": 1},
            )
        )

    def test_aggregate_writes_potential_failures_as_json_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            warning = {
                "status": "potential_failure",
                "token": "token_a",
                "code": "trajectory_collision",
                "message": "review me",
            }
            compositor.write_json(
                run_dir / "metadata" / "token_a.json",
                {
                    "annotation_sample": {"token": "token_a"},
                    "potential_failures": [warning],
                },
            )
            compositor.write_json(
                run_dir / "potential_failures" / "token_a.json",
                warning,
            )
            compositor.write_json(
                run_dir / "failures" / "token_b.json",
                {
                    "status": "failed",
                    "token": "token_b",
                    "code": "missing_sensor",
                    "message": "still a normal failure",
                },
            )

            successful, failed = compositor.aggregate_results(
                run_dir,
                ["token_a", "token_b"],
            )

            self.assertEqual((successful, failed), (1, 1))
            potential = json.loads(
                (run_dir / "potential_failures.json").read_text()
            )
            self.assertEqual(potential, [warning])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["potential_failure_tokens"], 1)

    def test_per_token_warning_file_retains_distinct_warning_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            collision = {
                "status": "potential_failure",
                "token": "token_a",
                "code": "trajectory_collision",
                "message": "collision review",
            }
            shadow = {
                "status": "potential_failure",
                "token": "token_a",
                "code": (
                    "directional_shadow_suppressed_by_receiver_depth"
                ),
                "message": "shadow review",
                "camera": "CAM_F0",
                "frame_index": 1,
            }

            compositor.record_potential_failure(run_dir, collision)
            compositor.record_potential_failure(run_dir, collision)
            compositor.record_potential_failure(run_dir, shadow)
            stored = json.loads(
                (
                    run_dir
                    / "potential_failures"
                    / "token_a.json"
                ).read_text()
            )

            self.assertEqual(stored, [collision, shadow])
            compositor.write_json(
                run_dir / "metadata" / "token_a.json",
                {
                    "annotation_sample": {"token": "token_a"},
                    "potential_failures": [collision, shadow],
                },
            )
            compositor.aggregate_results(run_dir, ["token_a"])
            aggregate = json.loads(
                (run_dir / "potential_failures.json").read_text()
            )
            self.assertEqual(aggregate, [collision, shadow])


if __name__ == "__main__":
    unittest.main()
