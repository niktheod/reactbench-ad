from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"))
import batch_composite_navsim_blender as fast
import front_visibility as visibility
import batch_composite_navsim_multi_trajectory as multi
import receiver_previews
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import classify_samples as classifier


class FrontVisibilityTest(unittest.TestCase):
    def rectangle(self, x0, y0, x1, y1, minimum=.25):
        uv = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return visibility.projected_fraction(uv, np.ones(4), 100, 100, minimum)

    def test_exact_threshold_is_inclusive(self):
        result = self.rectangle(-75, 0, 25, 100)
        self.assertEqual(result["fraction"], .25)
        self.assertTrue(result["passed"])
        self.assertFalse(self.rectangle(-75.01, 0, 24.99, 100)["passed"])

    def test_full_hidden_and_two_axis_cropping(self):
        self.assertEqual(self.rectangle(10, 10, 90, 90)["fraction"], 1)
        self.assertEqual(self.rectangle(100, 0, 200, 100)["fraction"], 0)
        self.assertEqual(self.rectangle(-50, -50, 50, 50)["fraction"], .25)

    def test_box_enclosing_image_passes_without_inside_corners(self):
        result = self.rectangle(-50, -50, 150, 150)
        self.assertEqual(result["fraction"], .25)
        self.assertTrue(result["passed"])
        uv = np.array([[-50., -50.], [150., -50.], [150., 150.], [-50., 150.]])
        with patch.object(fast.geometry, "camera_frustum_mask", return_value=np.zeros(4, dtype=bool)), \
             patch.object(fast.geometry, "project_distorted", return_value=(uv, np.ones(4))):
            mask, count, _ = receiver_previews._body_proxy_mask(
                np.zeros((4, 3)), {}, 100, 100, fast, allow_bbox_intersection=True,
            )
        self.assertEqual(count, 10000)

    def test_degenerate_behind_camera_and_nonfinite_fail(self):
        self.assertFalse(self.rectangle(0, 0, 0, 100)["passed"])
        for depth in ([1, -1], [1, .1], [1, float("nan")]):
            self.assertFalse(visibility.projected_fraction([[0, 0], [100, 100]], depth, 100, 100, .25)["passed"])
        self.assertFalse(self.rectangle(float("nan"), 0, 100, 100)["passed"])

    def test_invalid_threshold_is_rejected(self):
        for value in (-1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                visibility.fraction_argument(value)

    def test_current_front_only_and_shared_rejection(self):
        args = Namespace(min_front_visible_fraction=.25, data_root=Path("/unused"), split="test")
        frames = [{"cams": {"CAM_F0": {"frame": i}}} for i in range(4)]
        states = [{"bbox_lidar": np.zeros((8, 3)), "visible_cameras": ["CAM_L0"]} for _ in frames]
        with patch.object(fast, "frame_source_path", return_value=Path("/front.jpg")) as path, \
             patch.object(visibility, "image_size", return_value=(100, 100)), \
             patch.object(fast.geometry, "project_distorted", return_value=(np.array([[-80, 0], [20, 100]]), np.ones(2))) as project:
            candidate = {"failures": []}
            result = visibility.check_candidate(args, fast, frames, states, candidate)
        self.assertIs(path.call_args.args[2], frames[3])
        self.assertEqual(path.call_args.args[3], "CAM_F0")
        self.assertEqual(project.call_args.args[1], {"frame": 3})
        self.assertEqual(result["fraction"], .2)
        self.assertEqual(candidate["failures"][0]["code"], "insufficient_front_visibility")
        with self.assertRaises(fast.TokenFailure) as error:
            visibility.require(result, fast)
        self.assertEqual(error.exception.code, candidate["failures"][0]["code"])

    def test_disabled_gate_never_reads_images(self):
        self.assertIsNone(visibility.assess_current(Namespace(), fast, [], None))

    def test_visibility_requirement_changes_placement_cache_identity(self):
        args = Namespace(timestep=2, min_front_visible_fraction=0)
        previous = fast.placement_request(args)
        args.min_front_visible_fraction = .25
        self.assertNotEqual(previous, fast.placement_request(args))

    def test_all_variants_use_same_current_box_in_preflight_and_production(self):
        """Exercise real trajectories/bounds; isolate sensors and grounding."""
        scene = []
        for i in range(14):
            pose = np.eye(4)
            pose[0, 3] = 2 * i
            scene.append(dict(timestamp=i * 500_000, ego2global=pose, lidar2global=pose,
                              map_location="test", cams={cam: {} for cam in fast.CAMERA_ORDER}))
        frames = scene[:4]
        points = [np.zeros((0, 3)) for _ in frames]
        ego = dict(front_length_m=3., rear_length_m=1., width_m=2.)
        with tempfile.TemporaryDirectory(prefix="front-visibility-test-") as directory, ExitStack() as stack:
            root = Path(directory)
            source = root / "sensor.jpg"
            Image.new("RGB", (100, 100)).save(source)
            stack.enter_context(patch.object(fast, "frame_source_path", return_value=source))
            stack.enter_context(patch.object(fast.geometry, "load_lidar_xyz", return_value=points[0]))
            stack.enter_context(patch.object(fast, "fit_ground_sequence_best_effort",
                side_effect=lambda *a: ([np.zeros(3) for _ in frames], [{} for _ in frames], [])))
            stack.enter_context(patch.object(fast.geometry, "camera_frustum_mask", return_value=np.ones(8, dtype=bool)))
            stack.enter_context(patch.object(fast.geometry, "project_distorted",
                return_value=(np.array([[-80., 0], [20., 100]]), np.ones(2))))
            for benign in (False, True):
                for mode in ("car", "standing", "crossing"):
                    with self.subTest(benign=benign, mode=mode):
                        bounds = (dict(x=[-1., 4.], y=[-.8, 1.2], z=[0., 1.8]) if mode == "car" else
                                  dict(x=[-.4, .4], y=[-.3, .3], z=[0., 1.8]))
                        asset = dict(actor_geometry={"local_bounds_m": bounds}, actor_height_m=1.8,
                                     stride_length_m=1., action_range=(0, 40))
                        args = Namespace(data_root=root, split="test", timestep=2, speed_mps=1.4,
                                         max_ground_tilt_deg=2., actor_scale=1., actor_mode=mode,
                                         car=mode == "car", standing=mode == "standing", benign=benign,
                                         benign_side="left" if mode == "car" else "right",
                                         benign_walking_direction="forward", ego_vehicle_geometry=ego,
                                         min_front_visible_fraction=.25, direction_classification_labels={"token": "left"})
                        inputs = dict(args=args, scene_frames=scene, frames=frames,
                                      timestamps_s=np.arange(4) * .5, frame_lidar_points=points,
                                      global_to_current=np.linalg.inv(frames[-1]["ego2global"]),
                                      front_bumper_offset_m=3., asset=asset, future_timestep=2)
                        with patch.object(visibility, "assess_current", wraps=visibility.assess_current) as assess:
                            if mode == "car" and not benign:
                                candidate = classifier.evaluate_stationary_car_candidate(**inputs, ego_geometry=ego)
                            else:
                                candidate = multi.evaluate_candidate(**inputs, token="token", animation=[1] * 4,
                                    direction_spec=multi.DIRECTION_BY_CLASSIFICATION["left"])
                            self.assertTrue(candidate["evaluation_complete"], candidate["failures"])
                            self.assertEqual(candidate["front_visibility"]["fraction"], .2)
                            self.assertFalse(candidate["passed"])
                            expected_bbox = assess.call_args.args[3].copy()
                        loader = Namespace(scene_frames_dicts={"token": scene})
                        with patch.object(visibility, "assess_current", wraps=visibility.assess_current) as assess:
                            with self.assertRaises(fast.TokenFailure) as failure:
                                fast.prepare_token(args, loader, "token", root / "run", 3., asset)
                            self.assertEqual(failure.exception.code, "insufficient_front_visibility")
                            np.testing.assert_allclose(assess.call_args.args[3], expected_bbox, atol=1e-8)
                        if mode != "car":
                            with patch.object(visibility, "assess_current", wraps=visibility.assess_current) as assess:
                                with self.assertRaises(fast.TokenFailure) as failure:
                                    receiver_previews._prepare_geometry(args, fast, loader, "token", 3., asset)
                                self.assertEqual(failure.exception.code, "insufficient_front_visibility")
                                np.testing.assert_allclose(assess.call_args.args[3], expected_bbox, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
