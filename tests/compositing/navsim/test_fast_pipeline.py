from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as compositor  # noqa: E402


class ActorShadowRenderRegionTest(unittest.TestCase):
    @staticmethod
    def actor_bbox() -> np.ndarray:
        return np.array(
            [
                [x, y, z]
                for x in (-0.31, 0.35)
                for y in (-0.22, 0.27)
                for z in (-0.01, 1.81)
            ],
            dtype=np.float64,
        )

    @staticmethod
    def camera_to_actor() -> np.ndarray:
        actor_to_camera = np.eye(4, dtype=np.float64)
        actor_to_camera[2, 3] = 10.0
        return np.linalg.inv(actor_to_camera)

    def test_region_is_smaller_than_canvas_and_contains_actor_projection(self) -> None:
        intrinsic = np.array(
            [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        region = compositor.actor_shadow_render_region(
            self.actor_bbox(),
            self.camera_to_actor(),
            intrinsic,
            (1920, 1080),
            (0.25, 0.10, -0.96),
            sun_energy=1.5,
            sun_angle_degrees=8.0,
            padding_px=32,
        )

        self.assertNotEqual(region, [0, 0, 1920, 1080])
        self.assertLess(region[0], 960)
        self.assertLess(region[1], 540)
        self.assertGreater(region[2], 960)
        self.assertGreater(region[3], 540)

    def test_nearly_horizontal_direct_sun_falls_back_to_full_canvas(self) -> None:
        region = compositor.actor_shadow_render_region(
            self.actor_bbox(),
            self.camera_to_actor(),
            np.array(
                [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            (1920, 1080),
            (1.0, 0.0, -1e-4),
            sun_energy=1.5,
            sun_angle_degrees=8.0,
        )

        self.assertEqual(region, [0, 0, 1920, 1080])


class CroppedDistortionEquivalenceTest(unittest.TestCase):
    def test_cropped_and_zero_padded_full_render_distort_identically(self) -> None:
        height, width = 16, 20
        full = np.zeros((height, width, 4), dtype=np.uint8)
        rng = np.random.default_rng(9)
        full[4:12, 5:15] = rng.integers(
            0, 256, size=(8, 10, 4), dtype=np.uint8
        )
        crop = full[4:12, 5:15].copy()
        calibration = {
            "cam_intrinsic": [
                [18.0, 0.0, 9.5],
                [0.0, 18.0, 7.5],
                [0.0, 0.0, 1.0],
            ],
            "distortion": [0.03, -0.01, 0.001, -0.001, 0.0],
        }

        full_result = compositor.render_utils.distort_rgba(
            full,
            calibration,
            {},
            output_size=(width, height),
            ideal_origin=(0.0, 0.0),
        )
        crop_result = compositor.render_utils.distort_rgba(
            crop,
            calibration,
            {},
            output_size=(width, height),
            ideal_origin=(5.0, 4.0),
        )

        np.testing.assert_array_equal(crop_result, full_result)


class SparseAlphaCompositeEquivalenceTest(unittest.TestCase):
    @staticmethod
    def dense_reference(
        background: np.ndarray,
        foreground_rgba: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = (
            foreground_rgba[..., 3:4].astype(np.float32) / 255.0
        )
        changed = alpha[..., 0] > 0
        background_linear = compositor.render_utils.srgb_to_linear(
            background.astype(np.float32) / 255.0
        )
        foreground_linear = compositor.render_utils.srgb_to_linear(
            foreground_rgba[..., :3].astype(np.float32) / 255.0
        )
        composite_linear = (
            background_linear * (1.0 - alpha)
            + foreground_linear * alpha
        )
        composite = np.rint(
            compositor.render_utils.linear_to_srgb(composite_linear)
            * 255.0
        ).astype(np.uint8)
        composite[~changed] = background[~changed]
        return composite, foreground_rgba[..., 3]

    def test_sparse_path_is_pixel_identical_to_dense_reference(self) -> None:
        rng = np.random.default_rng(17)
        background = rng.integers(
            0, 256, size=(120, 180, 3), dtype=np.uint8
        )
        foreground = np.zeros((120, 180, 4), dtype=np.uint8)
        foreground[43:67, 71:96] = rng.integers(
            0, 256, size=(24, 25, 4), dtype=np.uint8
        )

        expected, expected_mask = self.dense_reference(
            background, foreground
        )
        actual, actual_mask = compositor.render_utils.alpha_composite(
            background, foreground
        )

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_mask, expected_mask)


class StreamingFinalizationTest(unittest.TestCase):


    def test_allocated_cpu_count_falls_back_to_process_affinity(self) -> None:
        with patch.dict(
            compositor.os.environ,
            {},
            clear=True,
        ), patch.object(
            compositor.os,
            "sched_getaffinity",
            return_value=set(range(8)),
        ):
            self.assertEqual(compositor.allocated_cpu_count(), 8)

    def test_tokens_finalize_only_when_their_own_jobs_are_complete(self) -> None:
        prepared = [
            {"token": "token_a", "jobs": [{"job_id": "a1"}, {"job_id": "a2"}]},
            {"token": "token_b", "jobs": [{"job_id": "b1"}]},
            {"token": "token_cached", "jobs": []},
        ]
        finalized = []
        completed_jobs = set()
        required_jobs = {
            item["token"]: {job["job_id"] for job in item["jobs"]}
            for item in prepared
        }
        lock = threading.Lock()

        def fake_render_batches(
            args,
            run_dir,
            jobs,
            on_batches_submitted=None,
            on_batch_complete=None,
        ):
            on_batches_submitted()
            with lock:
                completed_jobs.add("a1")
            on_batch_complete({"job_ids": ["a1"]})
            with lock:
                completed_jobs.add("b1")
            on_batch_complete({"job_ids": ["b1"]})
            with lock:
                completed_jobs.add("a2")
            on_batch_complete({"job_ids": ["a2"]})
            return []

        def fake_finalize(args, item, run_dir, cache):
            with lock:
                self.assertTrue(
                    required_jobs[item["token"]].issubset(completed_jobs)
                )
                finalized.append(item["token"])

        with patch.object(
            compositor,
            "prepare_finalization_tokens",
            return_value=prepared,
        ), patch.object(
            compositor,
            "finalize_prepared_token",
            side_effect=fake_finalize,
        ), patch.object(
            compositor,
            "run_blender_batches",
            side_effect=fake_render_batches,
        ):
            compositor.run_blender_with_streaming_finalization(
                Namespace(finalize_workers=2),
                object(),
                prepared,
                Path("/unused"),
                [job for item in prepared for job in item["jobs"]],
            )

        self.assertCountEqual(finalized, ["token_cached", "token_b", "token_a"])

    def test_finalization_pool_is_bounded_and_parallel(self) -> None:
        prepared = [
            {"token": f"token_{index}", "jobs": []}
            for index in range(4)
        ]
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def fake_finalize(args, item, run_dir, cache):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1

        def fake_render_batches(
            args,
            run_dir,
            jobs,
            on_batches_submitted=None,
            on_batch_complete=None,
        ):
            on_batches_submitted()
            return []

        with patch.object(
            compositor,
            "prepare_finalization_tokens",
            return_value=prepared,
        ), patch.object(
            compositor,
            "finalize_prepared_token",
            side_effect=fake_finalize,
        ), patch.object(
            compositor,
            "run_blender_batches",
            side_effect=fake_render_batches,
        ):
            compositor.run_blender_with_streaming_finalization(
                Namespace(finalize_workers=2),
                object(),
                prepared,
                Path("/unused"),
                [],
            )

        self.assertEqual(maximum_active, 2)


class FastOutputRootTest(unittest.TestCase):
    @staticmethod
    def base_argv() -> list[str]:
        return [
            "batch_composite_navsim_blender.py",
            "--scene-filter",
            "filter.yaml",
            "--direction-classifications",
            "directions.json",
            "--timestep",
            "1",
            "--run-name",
            "test-run",
            "--asset-dir",
            "assets",
        ]

    def test_batch_default_is_data_filesystem(self) -> None:
        argv = [*self.base_argv(), "--preview"]
        with patch.object(sys, "argv", argv):
            args = compositor.parse_args()
        self.assertEqual(args.timestep, 1)
        self.assertEqual(args.speed_mps, 1.4)
        self.assertEqual(args.output_root, compositor.locations.REACTNAV.parent)
        self.assertEqual(args.render_batch_size, 96)
        self.assertEqual(args.finalize_workers, 4)
        self.assertFalse(args.standing)

    def test_standing_is_an_opt_in_boolean_mode(self) -> None:
        argv = [*self.base_argv(), "--standing", "--preview"]
        with patch.object(sys, "argv", argv):
            args = compositor.parse_args()

        self.assertTrue(args.standing)
        self.assertEqual(compositor.effective_pedestrian_speed_mps(args), 0.0)
        self.assertEqual(
            compositor.placement_request(args),
            {
                "placement_mode": "future_timestep",
                "placement_version": compositor.STANDING_PLACEMENT_VERSION,
                "future_timestep": 1,
                "expected_future_offset_s": 0.5,
                "pedestrian_behavior": "standing",
                "standing_pose_version": compositor.STANDING_POSE_VERSION,
                "standing_position_policy": (
                    "stationary_at_future_ego_front_bumper"
                ),
                "standing_facing_policy": "opposite_future_ego_forward",
                "front_visibility_requirement": compositor.front_visibility.configuration(args),
            },
        )

    def test_standing_does_not_load_direction_classifications(self) -> None:
        argv = [
            "batch_composite_navsim_blender.py",
            "--scene-filter",
            "filter.yaml",
            "--direction-classifications",
            "does-not-exist.json",
            "--timestep",
            "1",
            "--run-name",
            "standing-run",
            "--asset-dir",
            "assets",
            "--standing",
            "--preview",
        ]
        with patch.object(sys, "argv", argv):
            args = compositor.parse_args()

        compositor.configure_direction_classifications(args)

        self.assertIsNone(args.direction_classifications)
        self.assertEqual(args.direction_classification_labels, {})
        self.assertIsNone(args.direction_classifications_sha256)
        self.assertEqual(compositor.direction_classification_provenance(args), {})

    def test_default_placement_request_remains_the_walking_contract(self) -> None:
        self.assertEqual(
            compositor.placement_request(Namespace(timestep=2)),
            {
                "placement_mode": "future_timestep",
                "placement_version": (
                    compositor.FUTURE_TIMESTEP_PLACEMENT_VERSION
                ),
                "future_timestep": 2,
                "expected_future_offset_s": 1.0,
            },
        )

    def test_output_mode_is_required_and_exclusive(self) -> None:
        base_argv = self.base_argv()
        with patch.object(sys, "argv", base_argv):
            with self.assertRaises(SystemExit):
                compositor.parse_args()
        with patch.object(sys, "argv", [*base_argv, "--preview"]):
            self.assertTrue(compositor.parse_args().preview)
        with patch.object(sys, "argv", [*base_argv, "--contact-only"]):
            self.assertTrue(compositor.parse_args().contact_only)
        with patch.object(
            sys,
            "argv",
            [
                *base_argv,
                "--preview",
                "--shadow-labels",
                "shadow_labels.json",
            ],
        ):
            with self.assertRaises(SystemExit):
                compositor.parse_args()
        with patch.object(
            sys,
            "argv",
            [
                *base_argv,
                "--contact-only",
                "--shadow-labels",
                "shadow_labels.json",
            ],
        ):
            with self.assertRaises(SystemExit):
                compositor.parse_args()

    def test_shadow_labels_select_final_mode(self) -> None:
        argv = [
            *self.base_argv(),
            "--shadow-labels",
            "vlm_inference/shadow_labels.json",
        ]
        with patch.object(sys, "argv", argv):
            args = compositor.parse_args()
        self.assertEqual(
            args.shadow_labels,
            Path("vlm_inference/shadow_labels.json"),
        )
        self.assertFalse(args.contact_only)

    def test_contact_only_selects_final_mode_without_labels(self) -> None:
        with patch.object(
            sys,
            "argv",
            [*self.base_argv(), "--contact-only"],
        ):
            args = compositor.parse_args()

        self.assertTrue(args.contact_only)
        self.assertIsNone(args.shadow_labels)

    def test_timestep_is_required_and_legacy_modes_are_absent(self) -> None:
        timestep_argv = self.base_argv()
        placement_index = timestep_argv.index("--timestep")
        without_timestep = [
            *timestep_argv[:placement_index],
            *timestep_argv[placement_index + 2 :],
            "--preview",
        ]
        with patch.object(sys, "argv", without_timestep):
            with self.assertRaises(SystemExit):
                compositor.parse_args()
        for legacy_option in ("--distance-m", "--msd-multiplier"):
            legacy_argv = [
                *without_timestep[:-1],
                legacy_option,
                "1",
                "--preview",
            ]
            with patch.object(sys, "argv", legacy_argv):
                with self.assertRaises(SystemExit):
                    compositor.parse_args()

    def test_timestep_request_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compositor.placement_request(
                Namespace(timestep=0)
            )








    def test_preview_dispatches_receiver_pipeline_before_blender_checks(self) -> None:
        args = Namespace(
            preview=True,
            timestep=1,
            direction_classifications=Path("directions.json"),
        )
        with patch.object(compositor, "parse_args", return_value=args), patch.object(
            compositor,
            "configure_direction_classifications",
        ) as configure_directions, patch.object(
            compositor.receiver_previews,
            "run_receiver_preview",
        ) as run_receiver_preview:
            compositor.main()

        configure_directions.assert_called_once_with(args)
        run_receiver_preview.assert_called_once_with(
            args,
            compositor,
            compositor.HERE,
        )

    def test_final_mode_dispatches_receiver_label_loader(self) -> None:
        labels_path = Path("receiver-shadow-labels.json")
        args = Namespace(
            preview=False,
            shadow_labels=labels_path,
            timestep=1,
            direction_classifications=Path("directions.json"),
        )
        with patch.object(compositor, "parse_args", return_value=args), patch.object(
            compositor,
            "configure_direction_classifications",
        ), patch.object(
            compositor.receiver_previews,
            "load_receiver_shadow_labels",
            side_effect=RuntimeError("loader reached"),
        ) as load_receiver_shadow_labels:
            with self.assertRaisesRegex(RuntimeError, "loader reached"):
                compositor.main()

        load_receiver_shadow_labels.assert_called_once_with(
            labels_path,
            args,
            compositor,
            compositor.HERE,
        )

    def test_contact_only_bypasses_receiver_label_loader(self) -> None:
        args = Namespace(
            preview=False,
            shadow_labels=None,
            contact_only=True,
            timestep=1,
            speed_mps=-1.0,
            direction_classifications=Path("directions.json"),
        )
        with patch.object(
            compositor,
            "parse_args",
            return_value=args,
        ), patch.object(
            compositor,
            "configure_direction_classifications",
        ), patch.object(
            compositor.receiver_previews,
            "load_receiver_shadow_labels",
        ) as load_receiver_shadow_labels:
            with self.assertRaisesRegex(
                ValueError,
                "pedestrian speed must be positive",
            ):
                compositor.main()

        load_receiver_shadow_labels.assert_not_called()
        self.assertIsNone(args.shadow_label_lookup)
        self.assertEqual(
            args.shadow_label_metadata["decision_policy"],
            compositor.CONTACT_ONLY_SHADOW_POLICY_VERSION,
        )
        self.assertTrue(args.refinalize)


class DirectionClassificationTest(unittest.TestCase):
    def test_left_and_right_map_to_requested_crossing_directions(self) -> None:
        self.assertEqual(
            compositor.crossing_policy("left-token", {"left-token": "left"}),
            ("right_to_left", 1.0, np.pi),
        )
        self.assertEqual(
            compositor.crossing_policy("right-token", {"right-token": "right"}),
            ("left_to_right", -1.0, 0.0),
        )

    def test_loader_accepts_failure_but_selected_tokens_must_be_renderable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "directions.json"
            path.write_text(
                json.dumps(
                    {
                        "left-token": "left",
                        "right-token": "right",
                        "failed-token": "failure",
                    }
                )
            )
            loaded = compositor.load_direction_classifications(path)

        compositor.validate_direction_classifications(
            ["left-token", "right-token"],
            loaded,
        )
        with self.assertRaisesRegex(ValueError, "non_renderable"):
            compositor.validate_direction_classifications(
                ["failed-token"],
                loaded,
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            compositor.validate_direction_classifications(
                ["missing-token"],
                loaded,
            )

    def test_loader_rejects_unknown_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "directions.json"
            path.write_text(json.dumps({"token": "TBD"}))
            with self.assertRaisesRegex(ValueError, "invalid examples"):
                compositor.load_direction_classifications(path)


class ShadowCatcherLoadingTest(unittest.TestCase):
    def test_zero_luminance_inside_render_region_is_full_occlusion(self) -> None:
        catcher = np.zeros((24, 24, 4), dtype=np.float32)
        catcher[..., 3] = 1.0
        catcher[3:21, 3:21, :3] = 1.0
        catcher[10:13, 11:14, :3] = 0.0
        manifest = {
            "resolution": [24, 24],
            "render_region": [3, 3, 21, 21],
            "render_scale": 1,
        }

        with patch.object(compositor.cv2, "imread", return_value=catcher):
            amount, diagnostics = compositor.load_shadow_catcher_amount(
                Path("unused.exr"),
                manifest,
            )

        np.testing.assert_array_equal(
            amount[10:13, 11:14],
            np.ones((3, 3), dtype=np.float32),
        )
        self.assertEqual(float(amount[5, 5]), 0.0)
        self.assertEqual(float(amount[0, 0]), 0.0)
        self.assertEqual(
            diagnostics["validity_method"],
            "manifest_render_region_zero_luminance_valid_v2",
        )
        self.assertEqual(diagnostics["valid_zero_luminance_pixel_count"], 9)

    def test_crop_to_border_catcher_stays_local_to_its_ideal_region(self) -> None:
        catcher = np.ones((72, 72, 4), dtype=np.float32)
        catcher[28:44, 28:44, :3] = 0.0
        manifest = {
            "resolution": [24, 24],
            "render_region": [3, 3, 21, 21],
            "render_scale": 4,
            "raw_render_storage": compositor.RAW_RENDER_STORAGE_VERSION,
            "ideal_origin": [-5, -7],
        }

        with patch.object(compositor.cv2, "imread", return_value=catcher):
            amount, diagnostics = compositor.load_shadow_catcher_amount(
                Path("unused.exr"),
                manifest,
            )

        self.assertEqual(amount.shape, (18, 18))
        self.assertEqual(
            diagnostics["validity_method"],
            "cropped_render_region_zero_luminance_valid_v3",
        )
        self.assertEqual(diagnostics["ideal_storage_origin"], [3, 3])
        self.assertEqual(float(amount[8:10, 8:10].min()), 1.0)
        self.assertEqual(
            compositor.render_storage_ideal_origin(manifest),
            (-2.0, -4.0),
        )

    def test_foot_geometry_is_translated_into_crop_local_coordinates(self) -> None:
        render_info = {
            "foot_contact_geometry": {
                "method": "test",
                "receiver_plane_z_m": 0.0,
                "contact_distance_m": 0.03,
                "stance_relative_distance_m": 0.01,
                "feet": {
                    "left": {
                        "contact": True,
                        "status": "measured_from_evaluated_foot_vertices",
                        "ideal_contact_points_px": [
                            [11.0, 21.0],
                            [14.0, 21.0],
                            [12.0, 24.0],
                        ],
                    }
                },
            }
        }
        alpha = np.zeros((8, 8), dtype=np.uint8)
        alpha[1:7, 1:6] = 255

        contact, diagnostics = compositor.ideal_foot_contact_region(
            render_info,
            alpha,
            ideal_storage_origin=(10.0, 20.0),
        )

        self.assertGreater(int(np.count_nonzero(contact)), 0)
        self.assertEqual(diagnostics["ideal_storage_origin"], [10.0, 20.0])
        self.assertGreater(float(contact[:7, :7].max()), 0.0)

    def test_diffuse_component_is_anchored_to_stance_region(self) -> None:
        amount = np.zeros((12, 12), dtype=np.float32)
        amount[7:10, 4:8] = 0.25
        amount[1:3, 1:3] = 0.8
        actor_alpha = np.zeros((12, 12), dtype=np.uint8)
        actor_alpha[4:10, 5:7] = 255
        stance = np.zeros((12, 12), dtype=np.uint8)
        stance[8:11, 5:8] = 1

        filtered, diagnostics = compositor.retain_grounded_shadow_component(
            amount,
            actor_alpha,
            anchor_region=stance,
        )

        self.assertTrue(np.all(filtered[7:10, 4:8] == 0.25))
        self.assertTrue(np.all(filtered[1:3, 1:3] == 0.0))
        self.assertEqual(
            diagnostics["anchor_method"],
            "explicit_stance_contact_region",
        )


class FinalizationCacheTest(unittest.TestCase):
    def test_directional_policy_omits_only_base_actor_sun(self) -> None:
        directional = compositor.base_render_required_passes(
            {"requires_directional_render": True}
        )
        contact = compositor.base_render_required_passes(
            {"requires_directional_render": False}
        )

        self.assertEqual(directional, compositor.BASE_RENDER_CORE_PASSES)
        self.assertEqual(contact, compositor.BASE_RENDER_FULL_PASSES)

    def test_legacy_full_pass_cache_satisfies_directional_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "render_mode": "base",
                        "camera_to_world": [[1.0, 2.0]],
                        "render_pipeline_version": (
                            compositor.LEGACY_RENDER_PIPELINE_VERSION
                        ),
                        "cache_key": "legacy-full-pass-key",
                    }
                )
            )
            current = {
                "render_mode": "base",
                "camera_to_world": [[1.0 + 5e-10, 2.0]],
                "render_pipeline_version": compositor.RENDER_PIPELINE_VERSION,
                "render_pass_policy_version": (
                    compositor.BASE_RENDER_PASS_POLICY_VERSION
                ),
                "required_passes": list(compositor.BASE_RENDER_CORE_PASSES),
                "produced_passes": list(compositor.BASE_RENDER_CORE_PASSES),
            }

            cache_key = compositor.compatible_render_cache_key(
                manifest_path,
                current,
                (),
                reuse_existing=True,
            )

        self.assertEqual(cache_key, "legacy-full-pass-key")
        self.assertEqual(
            current["produced_passes"],
            list(compositor.BASE_RENDER_FULL_PASSES),
        )

    def test_reduced_directional_cache_cannot_satisfy_contact_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "manifest.json"
            shared = {
                "render_mode": "base",
                "camera_to_world": [[1.0, 2.0]],
                "render_pipeline_version": compositor.RENDER_PIPELINE_VERSION,
                "render_pass_policy_version": (
                    compositor.BASE_RENDER_PASS_POLICY_VERSION
                ),
            }
            manifest_path.write_text(
                json.dumps(
                    {
                        **shared,
                        "required_passes": list(
                            compositor.BASE_RENDER_CORE_PASSES
                        ),
                        "produced_passes": list(
                            compositor.BASE_RENDER_CORE_PASSES
                        ),
                        "cache_key": "reduced-directional-key",
                    }
                )
            )
            current = {
                **shared,
                "required_passes": list(compositor.BASE_RENDER_FULL_PASSES),
                "produced_passes": list(compositor.BASE_RENDER_FULL_PASSES),
            }

            cache_key = compositor.compatible_render_cache_key(
                manifest_path,
                current,
                (),
                reuse_existing=True,
            )

        self.assertNotEqual(cache_key, "reduced-directional-key")

    def test_render_cache_key_reuses_cross_node_last_bit_float_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "camera_to_world": [[1.0, 2.0]],
                        "output_path": "/old/output.png",
                        "cache_key": "existing-render-key",
                    }
                )
            )
            current = {
                "camera_to_world": [[1.0 + 5e-10, 2.0]],
                "output_path": "/new/output.png",
            }

            cache_key = compositor.compatible_render_cache_key(
                manifest_path,
                current,
                ("output_path",),
                reuse_existing=True,
            )

        self.assertEqual(cache_key, "existing-render-key")

    def test_render_cache_key_rejects_material_float_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "camera_to_world": [[1.0, 2.0]],
                        "cache_key": "existing-render-key",
                    }
                )
            )
            current = {"camera_to_world": [[1.001, 2.0]]}

            cache_key = compositor.compatible_render_cache_key(
                manifest_path,
                current,
                (),
                reuse_existing=True,
            )

        self.assertNotEqual(cache_key, "existing-render-key")

    def test_cache_round_trip_is_lossless_and_keyed(self) -> None:
        height, width = 5, 7
        rng = np.random.default_rng(3)
        arrays = {
            "actor_rgba": rng.integers(
                0, 256, size=(height, width, 4), dtype=np.uint8
            ),
            "base_actor_sun_raw": rng.random(
                (height, width, 4), dtype=np.float32
            ),
            "actor_sky_raw": rng.random(
                (height, width, 4), dtype=np.float32
            ),
            "receiver_visibility": rng.random(
                (height, width), dtype=np.float32
            ),
            "contact_region_sensor": rng.random(
                (height, width), dtype=np.float32
            ),
            "contact_ambient_occlusion_amount": rng.random(
                (height, width), dtype=np.float32
            ),
            "labeled_ambient_occlusion_amount": rng.random(
                (height, width), dtype=np.float32
            ),
        }
        diagnostics = {"occlusion": {"method": "test"}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "view.npz"
            compositor.write_label_independent_finalization_cache(
                cache_path,
                "cache-key",
                arrays,
                diagnostics,
            )
            loaded = compositor.load_label_independent_finalization_cache(
                cache_path,
                "cache-key",
                (height, width),
            )
            stale = compositor.load_label_independent_finalization_cache(
                cache_path,
                "different-key",
                (height, width),
            )

        self.assertIsNotNone(loaded)
        self.assertIsNone(stale)
        self.assertEqual(loaded["diagnostics"], diagnostics)
        for name, value in arrays.items():
            np.testing.assert_array_equal(loaded["arrays"][name], value)

    def test_directional_cache_does_not_require_base_actor_sun(self) -> None:
        height, width = 3, 4
        arrays = {
            "actor_rgba": np.zeros((height, width, 4), dtype=np.uint8),
            "actor_sky_raw": np.zeros((height, width, 4), dtype=np.float32),
            "receiver_visibility": np.zeros((height, width), dtype=np.float32),
            "contact_region_sensor": np.zeros((height, width), dtype=np.float32),
            "contact_ambient_occlusion_amount": np.zeros(
                (height, width), dtype=np.float32
            ),
            "labeled_ambient_occlusion_amount": np.zeros(
                (height, width), dtype=np.float32
            ),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "view.npz"
            compositor.write_label_independent_finalization_cache(
                cache_path,
                "directional-key",
                arrays,
                {},
            )
            directional = compositor.load_label_independent_finalization_cache(
                cache_path,
                "directional-key",
                (height, width),
                require_base_actor_sun=False,
            )
            contact = compositor.load_label_independent_finalization_cache(
                cache_path,
                "directional-key",
                (height, width),
                require_base_actor_sun=True,
            )

        self.assertIsNotNone(directional)
        self.assertNotIn("base_actor_sun_raw", directional["arrays"])
        self.assertIsNone(contact)


class DirectionalShadowReceiverRecoveryTest(unittest.TestCase):
    @staticmethod
    def scene_depths(
        foreground_in_shadow: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width = 80, 100
        row = np.arange(height, dtype=np.float32)[:, None]
        receiver = np.broadcast_to(
            12.0 + 0.01 * row, (height, width)
        ).copy()
        source = receiver - 1.8
        raw = np.zeros((height, width), dtype=np.float32)
        raw[48:56, 42:58] = 0.9
        actor_alpha = np.zeros((height, width), dtype=np.uint8)
        actor_alpha[26:52, 48:53] = 255
        if foreground_in_shadow:
            # The surrounding road has a small, smooth depth bias, while the
            # shadow footprint contains genuinely closer source geometry.
            source = receiver - 0.2
            source[raw > 0.0] = receiver[raw > 0.0] - 3.0
        return raw, actor_alpha, source, receiver

    def test_normal_receiver_mask_does_not_enter_recovery(self) -> None:
        raw, actor_alpha, _source, _receiver = self.scene_depths()
        current = raw.copy()
        prepared = {"collision_diagnostics": {"occupied": False}}

        with patch.object(
            compositor,
            "ensure_directional_shadow_depth_context",
        ) as ensure_depth:
            selected, diagnostics = (
                compositor.maybe_recover_collapsed_directional_shadow(
                    Namespace(),
                    prepared,
                    {"frame_index": 1, "camera": "CAM_F0"},
                    {},
                    {},
                    raw,
                    current,
                    actor_alpha,
                    raw.shape[1],
                    raw.shape[0],
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                )
            )

        ensure_depth.assert_not_called()
        np.testing.assert_array_equal(selected, current)
        self.assertEqual(diagnostics["status"], "not_needed")
        self.assertFalse(diagnostics["attempted"])

    def test_smooth_local_road_bias_recovers_collapsed_shadow(self) -> None:
        raw, actor_alpha, source, receiver = self.scene_depths()
        receiver_visibility, _diagnostics = (
            compositor.shadow_receiver_visibility(receiver, source)
        )
        current = raw * receiver_visibility

        collapse = compositor.directional_shadow_collapse_diagnostics(
            raw, current, actor_alpha
        )
        recovered, local_diagnostics = (
            compositor.local_bias_corrected_directional_shadow(
                raw,
                actor_alpha,
                source,
                receiver,
            )
        )

        self.assertTrue(collapse["collapsed"])
        self.assertIsNotNone(recovered)
        self.assertEqual(local_diagnostics["status"], "success")
        self.assertAlmostEqual(
            local_diagnostics["signed_depth_bias_m"], 1.8, places=4
        )
        self.assertGreaterEqual(
            local_diagnostics["recovered_maximum_ratio"],
            compositor.DIRECTIONAL_SHADOW_RECOVERY_MIN_RETAINED_RATIO,
        )
        self.assertAlmostEqual(float(recovered.max()), 0.9, places=5)

    def test_collision_gate_keeps_original_receiver_mask(self) -> None:
        raw, actor_alpha, source, receiver = self.scene_depths()
        receiver_visibility, _diagnostics = (
            compositor.shadow_receiver_visibility(receiver, source)
        )
        current = raw * receiver_visibility
        prepared = {"collision_diagnostics": {"occupied": True}}

        with patch.object(
            compositor,
            "ensure_directional_shadow_depth_context",
        ) as ensure_depth:
            selected, diagnostics = (
                compositor.maybe_recover_collapsed_directional_shadow(
                    Namespace(),
                    prepared,
                    {"frame_index": 1, "camera": "CAM_F0"},
                    {},
                    {},
                    raw,
                    current,
                    actor_alpha,
                    raw.shape[1],
                    raw.shape[0],
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                )
            )

        ensure_depth.assert_not_called()
        np.testing.assert_array_equal(selected, current)
        self.assertEqual(
            diagnostics["status"], "blocked_trajectory_collision"
        )

    def test_local_foreground_is_not_reinterpreted_as_road_bias(self) -> None:
        raw, actor_alpha, source, receiver = self.scene_depths(
            foreground_in_shadow=True
        )
        receiver_visibility, _diagnostics = (
            compositor.shadow_receiver_visibility(receiver, source)
        )
        current = raw * receiver_visibility
        recovered, local_diagnostics = (
            compositor.local_bias_corrected_directional_shadow(
                raw,
                actor_alpha,
                source,
                receiver,
            )
        )
        recovered_evaluation = (
            compositor.directional_shadow_collapse_diagnostics(
                raw, recovered, actor_alpha
            )
        )

        self.assertEqual(local_diagnostics["status"], "success")
        self.assertLess(
            recovered_evaluation["retained_maximum_ratio"],
            compositor.DIRECTIONAL_SHADOW_RECOVERY_MIN_RETAINED_RATIO,
        )
        np.testing.assert_array_equal(current, recovered)

    def test_lidar_support_blocks_local_bias_recovery(self) -> None:
        raw, actor_alpha, source, receiver = self.scene_depths()
        receiver_visibility, _diagnostics = (
            compositor.shadow_receiver_visibility(receiver, source)
        )
        current = raw * receiver_visibility
        support = np.zeros(raw.shape, dtype=bool)
        support[50, 45] = True
        depth_context = {
            "source_metric_depth": source,
            "depth_alignment": {"method": "test"},
            "road_metric_depth": receiver,
            "receiver_metric_depth": receiver,
            "lidar_xyz": np.empty((0, 3), dtype=np.float32),
            "ground_plane": np.array([0.0, 0.0, 0.0]),
        }

        with patch.object(
            compositor,
            "ensure_directional_shadow_depth_context",
            return_value=depth_context,
        ), patch.object(
            compositor,
            "lidar_foreground_support",
            return_value=(
                support,
                {
                    "closer_above_ground_lidar_points": 1,
                    "supported_actor_pixels": 1,
                    "foreground_margin_m": 0.45,
                },
            ),
        ):
            selected, diagnostics = (
                compositor.maybe_recover_collapsed_directional_shadow(
                    Namespace(),
                    {"collision_diagnostics": {"occupied": False}},
                    {
                        "frame_index": 1,
                        "camera": "CAM_F0",
                        "actor_root_depth_m": 12.0,
                    },
                    {},
                    {},
                    raw,
                    current,
                    actor_alpha,
                    raw.shape[1],
                    raw.shape[0],
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                )
            )

        np.testing.assert_array_equal(selected, current)
        self.assertEqual(
            diagnostics["status"],
            "blocked_lidar_foreground_support",
        )
        self.assertEqual(
            diagnostics["lidar_foreground_support"][
                "supported_footprint_pixels"
            ],
            1,
        )


class ShadowLabelCompositionTest(unittest.TestCase):
    @staticmethod
    def label(strength: str, softness: str) -> dict:
        return {
            "scene_decision": {
                "direct_sunlight_condition": "clear_direct_sunlight",
            },
            "observations": {
                "receiver_visibility": "lit",
            },
            "recommended_shadow": {
                "strength": strength,
                "softness": softness,
            },
            "confidence": 0.9,
            "evidence": ["test evidence"],
        }

    def test_policy_uses_fixed_category_mapping(self) -> None:
        policy = compositor.shadow_label_render_policy(
            self.label("medium", "medium")
        )

        self.assertEqual(policy["strength"], "medium")
        self.assertEqual(policy["softness"], "medium")
        self.assertAlmostEqual(policy["directional_gain"], 0.35)
        self.assertAlmostEqual(policy["ambient_occlusion_gain"], 0.35)
        self.assertAlmostEqual(policy["actor_sun_gain"], 0.70)
        self.assertAlmostEqual(policy["sun_angle_degrees"], 1.0)
        self.assertTrue(policy["requires_directional_render"])

        contact = compositor.shadow_label_render_policy(
            self.label("contact_only", "soft")
        )
        self.assertEqual(contact["directional_gain"], 0.0)
        self.assertEqual(contact["ambient_occlusion_gain"], 1.0)
        self.assertEqual(contact["actor_sun_gain"], 0.0)
        self.assertFalse(contact["requires_directional_render"])

    def test_contact_only_mode_synthesizes_one_policy_for_every_view(self) -> None:
        args = Namespace(
            contact_only=True,
            shadow_labels=None,
            shadow_label_lookup=None,
        )

        label = compositor.shadow_label_for(args, "token", "CAM_F0", 3)
        self.assertEqual(label["direct_shadow_class"], "contact_only_shadow")
        self.assertEqual(
            label["recommended_shadow"],
            {"strength": "contact_only", "softness": "soft"},
        )
        self.assertEqual(
            label["decision_source"],
            compositor.CONTACT_ONLY_SHADOW_POLICY_VERSION,
        )
        self.assertTrue(compositor.uses_shadow_composition(args))
        self.assertFalse(
            compositor.shadow_label_render_policy(label)[
                "requires_directional_render"
            ]
        )

    def test_scene_level_derives_strength_and_softness(self) -> None:
        expected_lit = {
            "no_direct_sunlight": {
                "strength": "contact_only",
                "softness": "soft",
            },
            "moderate_direct_sunlight": {
                "strength": "medium",
                "softness": "medium",
            },
            "clear_direct_sunlight": {"strength": "strong", "softness": "hard"},
        }
        self.assertEqual(
            compositor.SHADOW_LABEL_DIRECT_SUNLIGHT_CONDITIONS,
            set(expected_lit),
        )
        self.assertEqual(
            compositor.SHADOW_LABEL_DIRECT_SUNLIGHT_PRESETS,
            expected_lit,
        )
        expected_strengths = {"contact_only", "medium", "strong"}
        self.assertEqual(
            set(compositor.SHADOW_LABEL_DIRECTIONAL_GAINS),
            expected_strengths,
        )
        self.assertEqual(
            set(compositor.SHADOW_LABEL_AMBIENT_OCCLUSION_GAINS),
            expected_strengths,
        )
        self.assertEqual(
            set(compositor.SHADOW_LABEL_ACTOR_SUN_GAINS),
            expected_strengths,
        )
        for level, expected in expected_lit.items():
            with self.subTest(level=level):
                self.assertEqual(
                    compositor.expected_timestep_shadow_recommendation(
                        level, "lit"
                    ),
                    expected,
                )
        self.assertEqual(
            compositor.expected_timestep_shadow_recommendation(
                "clear_direct_sunlight", "shaded"
            ),
            {"strength": "contact_only", "softness": "soft"},
        )

    def test_labeled_vdir_and_contact_ao_are_combined_multiplicatively(self) -> None:
        directional = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
        ambient = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

        alpha, diagnostics = compositor.compose_labeled_shadow_alpha(
            directional,
            ambient,
            directional_gain=0.35,
            ambient_occlusion_gain=0.35,
        )

        np.testing.assert_array_equal(
            alpha,
            np.array([[89, 81, 89]], dtype=np.uint8),
        )
        self.assertEqual(diagnostics["directional_gain"], 0.35)
        self.assertEqual(diagnostics["ambient_occlusion_gain"], 0.35)

    def test_diffuse_body_and_contact_occlusion_use_maximum_not_sum(self) -> None:
        diffuse = np.array([[0.4, 0.8, np.nan]], dtype=np.float32)
        contact = np.array([[0.7, 0.3, 0.2]], dtype=np.float32)

        merged = compositor.merge_sky_occlusion_components(diffuse, contact)

        np.testing.assert_allclose(
            merged,
            np.array([[0.7, 0.8, 0.2]], dtype=np.float32),
        )

    def test_actor_recomposition_preserves_alpha_and_scales_only_sun(self) -> None:
        actor = np.array([[[100, 110, 120, 255], [1, 2, 3, 0]]], dtype=np.uint8)
        sky = np.array(
            [[[0.10, 0.20, 0.30], [0.30, 0.30, 0.30]]], dtype=np.float32
        )
        sun = np.array(
            [[[0.20, 0.10, 0.05], [0.20, 0.20, 0.20]]], dtype=np.float32
        )

        no_sun = compositor.compose_actor_lighting_rgba(
            actor, sun, sky, sun_gain=0.0
        )
        full_sun = compositor.compose_actor_lighting_rgba(
            actor, sun, sky, sun_gain=1.0
        )

        expected_no_sun = np.rint(
            compositor.render_utils.linear_to_srgb(sky[0, 0]) * 255.0
        ).astype(np.uint8)
        expected_full_sun = np.rint(
            compositor.render_utils.linear_to_srgb(sky[0, 0] + sun[0, 0])
            * 255.0
        ).astype(np.uint8)
        np.testing.assert_array_equal(no_sun[0, 0, :3], expected_no_sun)
        np.testing.assert_array_equal(full_sun[0, 0, :3], expected_full_sun)
        np.testing.assert_array_equal(no_sun[..., 3], actor[..., 3])
        np.testing.assert_array_equal(full_sun[..., 3], actor[..., 3])
        np.testing.assert_array_equal(no_sun[0, 1, :3], np.zeros(3, dtype=np.uint8))
        np.testing.assert_array_equal(full_sun[0, 1, :3], np.zeros(3, dtype=np.uint8))


class DirectLightEvidenceTest(unittest.TestCase):
    @staticmethod
    def view(camera: str, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        return {
            "camera": camera,
            "image": image,
            "calibration": {
                "cam_intrinsic": [
                    [500.0, 0.0, width / 2.0],
                    [0.0, 500.0, height / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            },
            "camera_to_actor": np.eye(4),
        }

    def test_gray_overcast_does_not_create_directional_sun(self) -> None:
        image = np.full((360, 640, 3), 155, dtype=np.uint8)
        views = [self.view(camera, image) for camera in compositor.CAMERA_ORDER]

        lighting = compositor.estimate_multiview_lighting(views)

        self.assertLess(lighting["global_direct_probability"], 0.05)
        self.assertEqual(lighting["direct_sun_visibility"], 0.0)
        self.assertEqual(lighting["sun_energy"], 0.0)

    def test_top_connected_clear_blue_sky_is_direct_light_evidence(self) -> None:
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image[:210] = [95, 155, 225]
        image[210:] = [125, 125, 125]

        evidence = compositor.estimate_cast_shadow_evidence(image)

        self.assertGreater(evidence["probability"], 0.75)
        self.assertGreater(evidence["connected_blue_sky_fraction"], 0.20)

    def test_clear_sun_angle_fallback_stays_physically_tight(self) -> None:
        self.assertAlmostEqual(compositor.sun_angle_from_shadow_edges([]), 0.8)
        self.assertGreaterEqual(compositor.sun_angle_from_shadow_edges([1.0]), 0.5)
        self.assertLessEqual(compositor.sun_angle_from_shadow_edges([30.0]), 2.0)


if __name__ == "__main__":
    unittest.main()
