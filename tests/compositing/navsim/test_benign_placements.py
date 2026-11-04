from argparse import Namespace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"))
import batch_composite_navsim_blender as fast
import benign_placements as benign
import receiver_previews


EGO = dict(front_length_m=3.0, rear_length_m=1.0, width_m=2.0)
CAR = dict(x=[-1.0, 4.0], y=[-0.8, 1.2], z=[0, 1.8])
PED = dict(x=[-0.4, 0.4], y=[-0.3, 0.3], z=[0, 1.8])


def scene():
    frames = []
    for i in range(12):
        pose = np.eye(4)
        pose[0, 3] = 2 * i
        frames.append(dict(timestamp=i * 500_000, ego2global=pose, map_location="test"))
    return frames


def args(**kwargs):
    values = dict(benign=True, benign_side="left", benign_offset_m=4.0,
                  benign_clearance_m=1.0, benign_walking_direction="forward",
                  timestep=2, speed_mps=1.4, ego_vehicle_geometry=EGO)
    return Namespace(**(values | kwargs))


def resolve(mode, **kwargs):
    frames = scene()
    return benign.resolve(args(**kwargs), fast, frames, np.arange(4) * .5, 3.0,
                          CAR if mode == "car" else PED, actor_mode=mode)


class BenignGeometryTest(unittest.TestCase):
    def test_car_reverses_about_asymmetric_footprint_centre(self):
        result = resolve("car")
        bumper = np.array([13.0, 0.0])
        original_origin = bumper - CAR["x"][0] * np.array([1., 0.])
        expected = Polygon(benign.corners(CAR) + original_origin + [0, 4])
        actual = Polygon(benign.corners(CAR) @ benign.rotation(result["actor_root_yaw_global_rad"]).T + result["anchor_global_xy"])
        self.assertLess(actual.symmetric_difference(expected).area, 1e-8)
        np.testing.assert_allclose(result["actor_basis_forward_global_xy"], [-1, 0])
        np.testing.assert_allclose(result["actor_global_xys"], np.tile(result["anchor_global_xy"], (4, 1)))
        self.assertNotIn("first_contact_validation", result["placement"])

    def test_standing_and_walking_use_four_metre_offset(self):
        for mode in ("standing", "crossing"):
            for side, sign in (("left", 1), ("right", -1)):
                result = resolve(mode, benign_side=side)
                np.testing.assert_allclose(result["actor_global_xys"][:, 1], sign * 4)
                np.testing.assert_allclose(result["actor_global_xys"][-1], [13, sign * 4])

    def test_parallel_motion_and_renderer_heading_agree(self):
        for direction, sign in (("forward", 1), ("backward", -1)):
            result = resolve("crossing", benign_walking_direction=direction)
            np.testing.assert_allclose(np.diff(result["actor_global_xys"], axis=0), np.tile([sign * .7, 0], (3, 1)))
            np.testing.assert_allclose(benign.direction_lidar(np.eye(3), result["actor_heading_rad"]), [sign, 0, 0], atol=1e-8)
            points, center = receiver_previews._marker_points(np.zeros(3), np.eye(3), 0,
                                                            actor_heading=result["actor_heading_rad"])
            self.assertTrue(np.isfinite(points).all())

    def test_later_turn_can_reject_locally_clear_placement(self):
        frames = scene()
        frames[8]["ego2global"][:2, 3] = [13, 4]
        with self.assertRaises(fast.TokenFailure) as raised:
            benign.resolve(args(), fast, frames, np.arange(4) * .5, 3., PED, actor_mode="standing")
        self.assertEqual(raised.exception.code, "benign_ego_clearance")

    def test_sweeps_catch_collision_between_recorded_poses(self):
        frames = scene()
        frames[4]["ego2global"][:2, 3] = [13, 0]
        frames[5]["ego2global"][:2, 3] = [13, 8]
        result = benign.clearance_diagnostics(frames, 3, np.array([13, 4]), np.zeros(2),
                                             0., PED, EGO, 1.)
        self.assertFalse(result["passed"])

    def test_footprint_width_changes_clearance(self):
        wide = dict(CAR, y=[-3.5, 3.5])
        with self.assertRaises(fast.TokenFailure):
            benign.resolve(args(), fast, scene(), np.arange(4) * .5, 3., wide, actor_mode="car")

    def test_invalid_parameters_and_missing_future_fail(self):
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                benign.configuration(args(benign_offset_m=value))
        with self.assertRaises(ValueError):
            benign.clearance_diagnostics(scene()[:4], 3, np.zeros(2), np.zeros(2), 0, PED, EGO, 1)

    def test_cli_needs_no_crossing_direction_file_and_fingerprints_side(self):
        argv = ["compositor", "--scene-filter", "filter.yaml", "--timestep", "2", "--run-name", "pilot",
                "--asset-dir", "assets", "--benign", "--contact-only"]
        with patch.object(sys, "argv", argv):
            parsed = fast.parse_args()
        fast.configure_direction_classifications(parsed)
        fast.validate_actor_mode(parsed)
        self.assertIsNone(parsed.direction_classifications)
        first = fast.placement_request(parsed)
        parsed.benign_side = "right"
        self.assertNotEqual(first, fast.placement_request(parsed))



if __name__ == "__main__":
    unittest.main()
