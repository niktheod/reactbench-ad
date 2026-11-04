from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as compositor  # noqa: E402
import receiver_previews  # noqa: E402


class ReceiverMarkerRenderingTest(unittest.TestCase):
    def test_marker_contains_only_ground_outline_and_center_cross(self) -> None:
        source = np.zeros((80, 80, 3), dtype=np.uint8)
        polygon = np.asarray(
            [[30, 50], [40, 47], [50, 50], [50, 60], [40, 63], [30, 60]],
            dtype=np.int32,
        )

        preview = receiver_previews._draw_marker(source, polygon, [40.0, 55.0])

        self.assertFalse(np.array_equal(preview, source))
        self.assertGreater(int(preview[55, 40].sum()), 0)
        self.assertTrue(np.array_equal(preview[:35], source[:35]))


class ReceiverGroundingTest(unittest.TestCase):
    def test_preview_uses_shared_fallback_and_retains_review_warning(
        self,
    ) -> None:
        frames = [
            {
                "timestamp": index * 500_000,
                "ego2global": np.eye(4, dtype=np.float64),
                "lidar2global": np.eye(4, dtype=np.float64),
            }
            for index in range(compositor.EXPECTED_HISTORY_FRAMES + 1)
        ]
        loader = SimpleNamespace(
            scene_frames_dicts={"token": frames},
        )
        args = Namespace(
            data_root=Path("/unused"),
            split="test",
            seed=42,
            speed_mps=1.4,
            timestep=1,
            max_ground_tilt_deg=2.0,
            direction_classification_labels={"token": "left"},
        )
        asset = {
            "stride_length_m": 1.0,
            "action_range": (0, 40),
        }
        planes = [
            np.array([0.0, 0.0, -1.0], dtype=np.float64)
            for _ in compositor.HISTORY_FRAME_INDICES
        ]
        diagnostics = [
            {
                "frame_index": index,
                "method": (
                    "relaxed_same_timestamp_local_lidar"
                    if index == 1
                    else "strict_same_timestamp_local_lidar"
                ),
                "first_attempt_failed": index == 1,
            }
            for index in compositor.HISTORY_FRAME_INDICES
        ]
        strict_failures = [
            {
                "frame_index": 1,
                "code": "ground_fit_failed",
                "message": "strict support unavailable",
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            sensor_path = Path(temporary_directory) / "sensor.jpg"
            sensor_path.write_bytes(b"preview path validation only")
            with patch.object(
                compositor,
                "frame_source_path",
                return_value=sensor_path,
            ), patch.object(
                compositor.geometry,
                "load_lidar_xyz",
                return_value=np.zeros((1, 3), dtype=np.float64),
            ), patch.object(
                compositor,
                "fit_ground_sequence_best_effort",
                return_value=(planes, diagnostics, strict_failures),
            ) as fit_sequence:
                prepared = receiver_previews._prepare_geometry(
                    args,
                    compositor,
                    loader,
                    "token",
                    front_bumper=4.0,
                    asset=asset,
                )

        fit_sequence.assert_called_once()
        self.assertEqual(len(prepared["states"]), compositor.EXPECTED_HISTORY_FRAMES)
        self.assertEqual(len(prepared["potential_failures"]), 1)
        warning = prepared["potential_failures"][0]
        self.assertEqual(warning["status"], "potential_failure")
        self.assertEqual(warning["code"], "ground_fit_failed")
        self.assertEqual(warning["failed_frame_indices"], [1])
        self.assertEqual(
            warning["fallbacks"][0]["method"],
            "relaxed_same_timestamp_local_lidar",
        )

    def test_preview_uses_shared_future_timestep_geometry(self) -> None:
        future_pose = np.eye(4, dtype=np.float64)
        future_pose[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
        future_pose[:2, 3] = [10.0, 20.0]
        frames = [
            {
                "timestamp": index * 500_000,
                "ego2global": np.eye(4, dtype=np.float64),
                "lidar2global": np.eye(4, dtype=np.float64),
                "map_location": "test-map",
            }
            for index in compositor.HISTORY_FRAME_INDICES
        ]
        frames.append(
            {
                "timestamp": compositor.EXPECTED_HISTORY_FRAMES * 500_000,
                "ego2global": future_pose,
                "map_location": "test-map",
            }
        )
        loader = SimpleNamespace(scene_frames_dicts={"token": frames})
        args = Namespace(
            data_root=Path("/unused"),
            split="test",
            seed=42,
            speed_mps=1.4,
            timestep=1,
            max_ground_tilt_deg=2.0,
            direction_classification_labels={"token": "left"},
        )
        asset = {
            "stride_length_m": 1.0,
            "action_range": (0, 40),
        }
        planes = [
            np.array([0.0, 0.0, 0.0], dtype=np.float64)
            for _ in compositor.HISTORY_FRAME_INDICES
        ]
        diagnostics = [
            {
                "frame_index": index,
                "method": "strict_same_timestamp_local_lidar",
                "first_attempt_failed": False,
            }
            for index in compositor.HISTORY_FRAME_INDICES
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            sensor_path = Path(temporary_directory) / "sensor.jpg"
            sensor_path.write_bytes(b"preview path validation only")
            with patch.object(
                compositor,
                "frame_source_path",
                return_value=sensor_path,
            ), patch.object(
                compositor.geometry,
                "load_lidar_xyz",
                return_value=np.zeros((1, 3), dtype=np.float64),
            ), patch.object(
                compositor,
                "fit_ground_sequence_best_effort",
                return_value=(planes, diagnostics, []),
            ):
                prepared = receiver_previews._prepare_geometry(
                    args,
                    compositor,
                    loader,
                    "token",
                    front_bumper=4.0,
                    asset=asset,
                )

        _, lateral_sign, _ = compositor.crossing_policy(
            "token",
            args.direction_classification_labels,
        )
        expected = compositor.resolve_crossing_trajectory(
            frames,
            [0.0, 0.5, 1.0, 1.5],
            timestep=1,
            front_bumper_offset_m=4.0,
            lateral_sign=lateral_sign,
            speed_mps=1.4,
        )
        self.assertEqual(
            compositor.canonical_hash(prepared["placement"]),
            compositor.canonical_hash(expected["placement"]),
        )
        np.testing.assert_allclose(
            prepared["states"][-1]["actor_lidar"][:2],
            expected["actor_global_xys"][-1],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            prepared["states"][-1]["road_basis_lidar"][:, 0],
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )

    def test_standing_preview_is_stationary_and_faces_future_ego(self) -> None:
        future_pose = np.eye(4, dtype=np.float64)
        future_pose[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
        future_pose[:2, 3] = [10.0, 20.0]
        frames = [
            {
                "timestamp": index * 500_000,
                "ego2global": np.eye(4, dtype=np.float64),
                "lidar2global": np.eye(4, dtype=np.float64),
                "map_location": "test-map",
            }
            for index in compositor.HISTORY_FRAME_INDICES
        ]
        frames.append(
            {
                "timestamp": compositor.EXPECTED_HISTORY_FRAMES * 500_000,
                "ego2global": future_pose,
                "map_location": "test-map",
            }
        )
        args = Namespace(
            data_root=Path("/unused"),
            split="test",
            speed_mps=1.4,
            timestep=1,
            standing=True,
            max_ground_tilt_deg=2.0,
            direction_classification_labels={"token": "right"},
        )
        loader = SimpleNamespace(scene_frames_dicts={"token": frames})
        planes = [
            np.zeros(3, dtype=np.float64)
            for _ in compositor.HISTORY_FRAME_INDICES
        ]
        diagnostics = [
            {
                "frame_index": index,
                "method": "strict_same_timestamp_local_lidar",
                "first_attempt_failed": False,
            }
            for index in compositor.HISTORY_FRAME_INDICES
        ]
        asset = {"stride_length_m": 1.0, "action_range": (0, 40)}

        with tempfile.TemporaryDirectory() as temporary_directory:
            sensor_path = Path(temporary_directory) / "sensor.jpg"
            sensor_path.write_bytes(b"preview path validation only")
            with patch.object(
                compositor, "frame_source_path", return_value=sensor_path
            ), patch.object(
                compositor.geometry,
                "load_lidar_xyz",
                return_value=np.zeros((1, 3), dtype=np.float64),
            ), patch.object(
                compositor,
                "fit_ground_sequence_best_effort",
                return_value=(planes, diagnostics, []),
            ):
                prepared = receiver_previews._prepare_geometry(
                    args,
                    compositor,
                    loader,
                    "token",
                    front_bumper=4.0,
                    asset=asset,
                )

        self.assertEqual(prepared["direction"], "standing_facing_ego")
        self.assertTrue(prepared["standing"])
        self.assertEqual(prepared["placement"]["pedestrian_behavior"], "standing")
        np.testing.assert_allclose(
            [state["actor_lidar"][:2] for state in prepared["states"]],
            np.repeat([[10.0, 24.0]], compositor.EXPECTED_HISTORY_FRAMES, axis=0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [state["yaw_lidar"] for state in prepared["states"]],
            np.zeros(compositor.EXPECTED_HISTORY_FRAMES),
            atol=1e-12,
        )
        self.assertEqual(
            [state["animation_frame"] for state in prepared["states"]],
            [compositor.STANDING_ANIMATION_FRAME]
            * compositor.EXPECTED_HISTORY_FRAMES,
        )


class ReceiverCameraSelectionTest(unittest.TestCase):
    @staticmethod
    def records(body_counts: dict[str, int], receiver_count: int = 10) -> dict:
        return {
            camera: {
                "projected_body_pixel_count": body_counts.get(camera, 0),
                "receiver_marker_pixel_count": receiver_count,
            }
            for camera in receiver_previews.CAMERA_TIE_ORDER
        }

    def test_largest_projected_body_is_selected(self) -> None:
        selected = receiver_previews._select_camera(
            self.records({"CAM_F0": 10, "CAM_L0": 30, "CAM_R0": 20})
        )
        self.assertEqual(selected, "CAM_L0")

    def test_front_camera_breaks_an_exact_tie(self) -> None:
        selected = receiver_previews._select_camera(
            self.records({"CAM_F0": 20, "CAM_L0": 20, "CAM_R0": 20})
        )
        self.assertEqual(selected, "CAM_F0")

    def test_camera_without_receiver_marker_is_ineligible(self) -> None:
        records = self.records({"CAM_F0": 40, "CAM_L0": 20, "CAM_R0": 10})
        records["CAM_F0"]["receiver_marker_pixel_count"] = 0
        self.assertEqual(receiver_previews._select_camera(records), "CAM_L0")

    def test_valid_off_image_receiver_selects_contact_only_fallback(self) -> None:
        records = self.records(
            {"CAM_F0": 40, "CAM_L0": 20, "CAM_R0": 10},
            receiver_count=0,
        )
        for record in records.values():
            record["receiver_projection_valid"] = True

        selected = receiver_previews._select_contact_only_fallback_camera(
            records
        )

        self.assertEqual(selected, "CAM_F0")

    def test_invalid_receiver_projection_does_not_hide_geometry_failure(
        self,
    ) -> None:
        records = self.records({"CAM_F0": 40}, receiver_count=0)
        for record in records.values():
            record["receiver_projection_valid"] = False

        selected = receiver_previews._select_contact_only_fallback_camera(
            records
        )

        self.assertIsNone(selected)


class DeterministicFallbackTemplateTest(unittest.TestCase):
    def test_all_body_visible_receiver_clipped_frames_remain_successful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            from PIL import Image

            Image.new("RGB", (64, 64), (10, 20, 30)).save(source)
            cameras = ("CAM_L0", "CAM_F0", "CAM_R0")
            frames = [
                {"cams": {camera: {} for camera in cameras}}
                for _ in range(4)
            ]
            states = [
                {
                    "frame_index": index,
                    "actor_lidar": np.zeros(3),
                    "road_basis_lidar": np.eye(3),
                    "yaw_lidar": 0.0,
                    "road_plane_lidar": np.zeros(3),
                    "animation_frame": index,
                    "ground_diagnostics": {},
                }
                for index in range(4)
            ]
            geometry = {
                "frames": frames,
                "states": states,
                "lateral_sign": 1.0,
                "actor_heading": 0.0,
                "direction_classification": "left",
                "direction": "right_to_left",
                "lateral_positions_m": np.zeros(4),
                "crossing_offsets_m": np.zeros(4),
                "walking_velocity_global_xy_mps": np.zeros(2),
                "placement": {"mode": "future_timestep", "timestep": 1},
                "timestamps_s": np.arange(4, dtype=np.float64),
                "potential_failures": [],
            }
            asset = {
                "preview_asset_identity": {},
                "id": "asset",
                "actor_height_m": 1.7,
                "animation_version": "v1",
                "action_range": (0, 4),
                "walk_cycle_frames": 4,
                "stride_length_m": 1.0,
                "stride_length_source": "test",
                "actor_geometry": {
                    "local_bounds_m": {
                        "x": [-0.3, 0.3],
                        "y": [-0.2, 0.2],
                        "z": [0.0, 1.7],
                    }
                },
            }
            fast = SimpleNamespace(
                CAMERA_ORDER=cameras,
                GROUNDING_METHOD="test",
                frame_source_path=lambda *_args: source,
                file_sha256=compositor.file_sha256,
                canonical_hash=compositor.canonical_hash,
                placement_metadata_fields=lambda _placement: {},
                render_utils=SimpleNamespace(
                    actor_bbox_lidar=lambda *_args, **_kwargs: np.zeros((8, 3))
                ),
            )
            args = Namespace(data_root=root, split="test", speed_mps=1.4)

            with patch.object(
                receiver_previews,
                "_polygon_mask",
                return_value=(
                    np.zeros((64, 64), dtype=np.uint8),
                    0,
                    np.zeros((4, 2), dtype=np.int32),
                    np.asarray(
                        [[10.0, 80.0], [20.0, 80.0], [20.0, 90.0], [10.0, 90.0]]
                    ),
                    True,
                ),
            ), patch.object(
                receiver_previews,
                "_body_proxy_mask",
                return_value=(
                    np.zeros((64, 64), dtype=np.uint8),
                    100,
                    np.zeros((4, 2), dtype=np.int32),
                ),
            ), patch.object(
                receiver_previews,
                "_project_point",
                return_value=[15.0, 85.0],
            ):
                metadata = receiver_previews._token_metadata(
                    args,
                    fast,
                    "token",
                    geometry,
                    asset,
                    root,
                    {"placement_config_sha256": "config"},
                )

        self.assertEqual(
            metadata["pedestrian_geometry"], asset["actor_geometry"]
        )
        self.assertEqual(
            {
                selection["status"]
                for selection in metadata["preview_selections"].values()
            },
            {receiver_previews.CONTACT_ONLY_FALLBACK_STATUS},
        )
        self.assertTrue(
            all(
                selection["deterministic_shadow_override"]
                == "contact_only_shadow"
                for selection in metadata["preview_selections"].values()
            )
        )

    def test_template_publishes_contact_only_without_preview_image(self) -> None:
        metadata = {
            "preview_selections": {
                "0": {
                    "status": receiver_previews.CONTACT_ONLY_FALLBACK_STATUS,
                    "reason": receiver_previews.CONTACT_ONLY_FALLBACK_REASON,
                    "reference_camera": "CAM_F0",
                    "source_image": {"image_sha256": "source-hash"},
                    "preview_selection_sha256": "selection-hash",
                    "deterministic_shadow_override": "contact_only_shadow",
                    "decision_source": (
                        receiver_previews.CONTACT_ONLY_FALLBACK_POLICY_VERSION
                    ),
                }
            },
            "scene_input_images": [],
        }
        fast = SimpleNamespace(
            file_sha256=lambda _path: "metadata-hash",
            EXPECTED_HISTORY_FRAMES=4,
        )

        record = receiver_previews._template_record(
            metadata,
            Path("metadata.json"),
            fast,
        )

        fallback = record["timesteps"]["0"]
        self.assertEqual(fallback["recommended_shadow"], "contact_only_shadow")
        self.assertEqual(
            fallback["decision_source"],
            receiver_previews.CONTACT_ONLY_FALLBACK_POLICY_VERSION,
        )
        self.assertNotIn("relative_path", fallback)
        self.assertNotIn("image_sha256", fallback)


class ReceiverLabelContractTest(unittest.TestCase):
    def test_canonical_module_lives_in_fast_directory(self) -> None:
        self.assertEqual(Path(receiver_previews.__file__).resolve().parent, MODULE_ROOT)

    def test_loader_rejects_legacy_shadow_label_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = root / "shadow_labels.json"
            labels_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "document_role": "shadow_labels",
                    }
                )
            )
            args = Namespace(
                output_root=root / "outputs",
                scene_filter=root / "filter.yaml",
                run_name="test-run",
            )
            with self.assertRaisesRegex(
                ValueError,
                "Invalid direct shadow-label file",
            ):
                receiver_previews.load_receiver_shadow_labels(
                    labels_path,
                    args,
                    compositor,
                    MODULE_ROOT,
                )

    def test_direct_classes_map_to_existing_render_presets(self) -> None:
        expected = {
            "contact_only_shadow": {
                "strength": "contact_only",
                "softness": "soft",
            },
            "moderate_directional_shadow": {
                "strength": "medium",
                "softness": "medium",
            },
            "clear_directional_shadow": {
                "strength": "strong",
                "softness": "hard",
            },
        }

        self.assertEqual(receiver_previews.DIRECT_SHADOW_PRESETS, expected)
        for shadow_class, preset in expected.items():
            with self.subTest(shadow_class=shadow_class):
                policy = compositor.shadow_label_render_policy(
                    {
                        "direct_shadow_class": shadow_class,
                        "recommended_shadow": dict(preset),
                    }
                )
                self.assertEqual(policy["strength"], preset["strength"])
                self.assertEqual(policy["softness"], preset["softness"])

    def test_loader_injects_validated_geometry_contact_only_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "outputs"
            scene_filter = root / "filter.yaml"
            scene_filter.write_text("scene_filter: test\n")
            directions = root / "directions.json"
            directions.write_text(json.dumps({"token": "left"}))
            data_root = root / "dataset"
            data_root.mkdir()
            asset_dir = root / "assets"
            asset_dir.mkdir()
            run_name = "fallback-run"
            run_dir = output_root / scene_filter.stem / run_name
            preview_root = run_dir / "preview_images"
            preview_root.mkdir(parents=True)
            metadata_root = run_dir / "receiver_preview_metadata"
            metadata_root.mkdir()
            source_image = root / "source.png"
            from PIL import Image

            Image.new("RGB", (64, 64), (20, 30, 40)).save(source_image)
            source_sha256 = compositor.file_sha256(source_image)
            from nuplan.common.actor_state.vehicle_parameters import (
                get_pacifica_parameters,
            )

            args = Namespace(
                output_root=output_root,
                scene_filter=scene_filter,
                run_name=run_name,
                direction_classifications=directions,
                direction_classifications_sha256=compositor.file_sha256(
                    directions
                ),
                data_root=data_root,
                split="test",
                asset_dir=asset_dir,
                timestep=1,
                speed_mps=1.4,
                seed=42,
                stride_length_m=None,
                gait_frame_advance=4,
                gait_timestep_s=0.5,
                max_ground_tilt_deg=2.0,
                max_samples=1,
            )
            config = {
                "pipeline_version": receiver_previews.PIPELINE_VERSION,
                "fast_pipeline_source_sha256": compositor.file_sha256(
                    Path(compositor.__file__)
                ),
                "scene_filter": str(scene_filter.resolve()),
                "scene_filter_sha256": compositor.file_sha256(scene_filter),
                "direction_classifications_path": str(directions.resolve()),
                "direction_classifications_sha256": (
                    args.direction_classifications_sha256
                ),
                "data_root": str(data_root.resolve()),
                "split": "test",
                "output_root": str(output_root.resolve()),
                "run_name": run_name,
                "asset_directory": str(asset_dir.resolve()),
                "asset_pool_preview_identity": [],
                **compositor.placement_request(args),
                "front_bumper_offset_m": float(
                    get_pacifica_parameters().front_length
                ),
                "speed_mps": 1.4,
                "seed": 42,
                "stride_length_m": None,
                "gait_frame_advance": 4,
                "gait_timestep_s": 0.5,
                "max_ground_tilt_degrees": 2.0,
                "grounding_method": compositor.GROUNDING_METHOD,
                "max_samples": 1,
                "direction_policy": (
                    compositor.DIRECTION_CLASSIFICATION_POLICY_VERSION
                ),
                "direction_classification_mapping": (
                    compositor.direction_classification_mapping()
                ),
                "deterministic_contact_only_fallback": {
                    "policy_version": (
                        receiver_previews.CONTACT_ONLY_FALLBACK_POLICY_VERSION
                    ),
                    "trigger": receiver_previews.CONTACT_ONLY_FALLBACK_REASON,
                    "scope": "per_token_history_frame",
                },
                "selected_tokens": ["token"],
            }
            config["placement_config_sha256"] = compositor.canonical_hash(
                receiver_previews._placement_core(config)
            )
            config_path = run_dir / "receiver_preview_config.json"
            config_path.write_text(json.dumps(config))
            fallback_selection = {
                "status": receiver_previews.CONTACT_ONLY_FALLBACK_STATUS,
                "reason": receiver_previews.CONTACT_ONLY_FALLBACK_REASON,
                "frame_index": 3,
                "reference_camera": "CAM_F0",
                "source_image": {
                    "camera": "CAM_F0",
                    "frame_index": 3,
                    "source_path": str(source_image.resolve()),
                    "image_sha256": source_sha256,
                    "width": 64,
                    "height": 64,
                },
                "camera_projection_counts": {},
                "selected_projected_body_pixel_count": 100,
                "selected_receiver_marker_pixel_count": 0,
                "receiver_center_pixel": [32.0, 80.0],
                "receiver_projected_bounds": [20.0, 75.0, 40.0, 90.0],
                "deterministic_shadow_override": "contact_only_shadow",
                "decision_source": (
                    receiver_previews.CONTACT_ONLY_FALLBACK_POLICY_VERSION
                ),
                "selection_version": receiver_previews.SELECTION_VERSION,
            }
            fallback_selection["preview_selection_sha256"] = (
                compositor.canonical_hash(fallback_selection)
            )
            metadata = {
                "schema_version": receiver_previews.SCHEMA_VERSION,
                "pipeline_version": receiver_previews.PIPELINE_VERSION,
                "scene_input_version": receiver_previews.SCENE_INPUT_VERSION,
                "receiver_input_version": receiver_previews.RECEIVER_INPUT_VERSION,
                "preview_selection_version": receiver_previews.SELECTION_VERSION,
                "token": "token",
                "placement_config_sha256": config[
                    "placement_config_sha256"
                ],
                "crossing": {"placement": {"future_timestep": 1}},
                "scene_input_images": [
                    {
                        "camera": camera,
                        "frame_index": 3,
                        "source_path": str(source_image.resolve()),
                        "image_sha256": source_sha256,
                    }
                    for camera in compositor.CAMERA_ORDER
                ],
                "preview_selections": {"3": fallback_selection},
            }
            metadata_path = metadata_root / "token.json"
            metadata_path.write_text(json.dumps(metadata))
            source_record = receiver_previews._template_record(
                metadata,
                metadata_path,
                compositor,
            )
            template = {
                "schema_version": receiver_previews.SCHEMA_VERSION,
                "document_role": "shadow_labels",
                "scene_input_version": receiver_previews.SCENE_INPUT_VERSION,
                "receiver_input_version": receiver_previews.RECEIVER_INPUT_VERSION,
                "preview_selection_version": receiver_previews.SELECTION_VERSION,
                "deterministic_contact_only_fallback_policy": (
                    receiver_previews.CONTACT_ONLY_FALLBACK_POLICY_VERSION
                ),
                "input_root": str(preview_root.resolve()),
                "metadata_root": str(metadata_root.resolve()),
                "placement_config_path": str(config_path.resolve()),
                "placement_config_sha256": config[
                    "placement_config_sha256"
                ],
                "labels": {"token": source_record},
                "failures": {},
            }
            template_path = run_dir / "receiver_label_template.json"
            template_path.write_text(json.dumps(template))
            direct_record = json.loads(json.dumps(source_record))
            direct_record["timesteps"]["3"][
                "experimental_receiver_visibility"
            ] = None
            direct_document = {
                "schema_version": receiver_previews.SCHEMA_VERSION,
                "document_role": "direct_shadow_labels",
                "source_template_path": str(template_path.resolve()),
                "source_template_sha256": compositor.file_sha256(template_path),
                "labels": {"token": direct_record},
                "failures": {},
            }
            direct_path = run_dir / "shadow_labels.json"
            direct_path.write_text(json.dumps(direct_document))

            with patch.object(
                receiver_previews,
                "_load_light_assets",
                return_value=[],
            ):
                lookup, label_metadata = (
                    receiver_previews.load_receiver_shadow_labels(
                        direct_path,
                        args,
                        compositor,
                        MODULE_ROOT,
                    )
                )

        fallback = lookup[("token", 3)]
        self.assertEqual(fallback["direct_shadow_class"], "contact_only_shadow")
        self.assertEqual(
            fallback["decision_source"],
            receiver_previews.CONTACT_ONLY_FALLBACK_POLICY_VERSION,
        )
        self.assertFalse(
            compositor.shadow_label_render_policy(fallback)[
                "requires_directional_render"
            ]
        )
        self.assertEqual(label_metadata["label_count"], 1)
        self.assertEqual(
            label_metadata["deterministic_contact_only_fallback_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
