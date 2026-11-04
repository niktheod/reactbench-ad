"""Braking feasibility: GT path, braking kinematics, overlap test, and first-contact classification."""

import math
import unittest

import numpy as np

from reactbench.scoring import feasibility as bf

NAVSIM_EGO = {"x_min": -1.127, "x_max": 4.049, "y_min": -1.1485, "y_max": 1.1485}
FRONT = NAVSIM_EGO["x_max"]


def box_motion(actor_type, centre, velocity=(0., 0.), heading=0., length=1., width=1.):
    return {"actor_type": actor_type, "initial_center_xy_m": list(centre), "velocity_xy_mps": list(velocity),
            "heading_rad": heading, "front_length_m": length / 2, "rear_length_m": length / 2,
            "left_width_m": width / 2, "right_width_m": width / 2, "length_m": length, "width_m": width}


def straight_path(length=60.):
    return bf.GTPath(np.column_stack((np.linspace(length / 8, length, 8), np.zeros(8))))


class PathTest(unittest.TestCase):
    def test_merge_drops_backward_and_short_steps(self):
        points = np.array([[0., 0.], [1., 0.], [1.02, 0.], [0.5, 0.], [2., 0.]])
        np.testing.assert_allclose(bf.merge_waypoints(points), [[0., 0.], [1., 0.], [2., 0.]])

    def test_straight_path_and_extension(self):
        path = bf.GTPath(np.column_stack((np.arange(1, 9) * 1.5, np.zeros(8))))
        np.testing.assert_allclose(path.poses([0., 3.5, 12., 20.]),
                                   [[0., 0., 0.], [3.5, 0., 0.], [12., 0., 0.], [20., 0., 0.]], atol=1e-6)

    def test_curved_path_arc_length_and_heading(self):
        angles = np.linspace(0., math.pi / 2, 9)[1:]
        radius = 20.
        path = bf.GTPath(np.column_stack((radius * np.sin(angles), radius * (1 - np.cos(angles)))))
        pose = path.poses([radius * math.pi / 4])
        self.assertAlmostEqual(path.length, radius * math.pi / 2, delta=0.01)
        np.testing.assert_allclose(pose[0], [radius * math.sin(math.pi / 4), radius * (1 - math.cos(math.pi / 4)),
                                             math.pi / 4], atol=0.01)


class KinematicsTest(unittest.TestCase):
    def test_braking_holds_after_stop(self):
        arcs, speeds = bf.braking_motion(10., [0., 5.], [0., 1., 2., 3.])
        np.testing.assert_allclose(arcs, [[0., 10., 20., 30.], [0., 7.5, 10., 10.]])
        np.testing.assert_allclose(speeds, [[10., 10., 10., 10.], [10., 5., 0., 0.]])

    def test_overlap_matches_scorer_clearance(self):
        from reactbench.scoring.geometry import COLLISION_EPSILON_M
        from reactbench.scoring.collisions import rectangle_corners, signed_clearances
        rng = np.random.default_rng(0)
        poses = np.column_stack((rng.uniform(-4, 4, (2000, 2)), rng.uniform(-math.pi, math.pi, 2000)))
        first = rectangle_corners(poses, rng.uniform(0.3, 5., (2000, 2)))
        second = rectangle_corners(np.zeros((2000, 3)), np.tile([4.6, 2.], (2000, 1)))
        expected = signed_clearances(first, second) <= COLLISION_EPSILON_M
        self.assertTrue(expected.any() and not expected.all())
        np.testing.assert_array_equal(bf.footprints_overlap(first, second), expected)


class OracleTest(unittest.TestCase):
    def test_stationary_box_threshold(self):
        # Car whose rear bumper is a gap g ahead of the ego's front bumper: stopping needs v0^2 / (2 g).
        v0 = 10.
        for gap in (5., 7.):
            motion = box_motion("car", (FRONT + gap + 2., 0.), length=4., width=2.)
            result = bf.BrakingOracle(straight_path(), v0, motion, NAVSIM_EGO).evaluate()
            self.assertAlmostEqual(result["a_min_avoid_mps2"], v0 ** 2 / (2 * gap), delta=0.01)
            self.assertEqual(result["feasible"], v0 ** 2 / (2 * gap) <= bf.A_MAX_MPS2)
            self.assertTrue(result["contact_at_zero"] and result["monotone"])
            self.assertEqual(result["kind_at_zero"], "stopped_actor")

    def test_contact_with_stopped_ego_is_exempt(self):
        # A pedestrian crossing 1.2 m ahead of the front bumper reaches the ego's corridor at ~1.5 s. Constant
        # speed hits it; braking at >= ~2.65 m/s^2 stops inside its path before it arrives (exempt contact), and
        # braking above 16 / (2 * 1.2) = 6.7 m/s^2 stops short of it.
        motion = box_motion("pedestrian", (FRONT + 1.2 + 0.25, 3.5), velocity=(0., -1.4), heading=-math.pi / 2,
                            length=0.5, width=0.5)
        oracle = bf.BrakingOracle(straight_path(), 4., motion, NAVSIM_EGO)
        at_fault, kind, _ = oracle.contacts([0., 2., 4., 8.])
        np.testing.assert_array_equal(at_fault, [True, True, False, False])
        self.assertEqual(list(kind), ["moving_actor", "moving_actor", "stopped_ego", "none"])
        result = oracle.evaluate()
        self.assertTrue(result["feasible"] and result["monotone"])
        self.assertGreater(result["a_min_avoid_mps2"], 2.6)
        self.assertLess(result["a_min_avoid_mps2"], 2.75)

    def test_rear_contact_is_exempt(self):
        # An actor catching up from behind touches the rear bumper: exempt, as in at_fault rule 3.
        motion = box_motion("pedestrian", (-4., 0.), velocity=(3., 0.), length=0.5, width=0.5)
        at_fault, kind, _ = bf.BrakingOracle(straight_path(), 1., motion, NAVSIM_EGO).contacts([0.])
        self.assertFalse(at_fault[0])
        self.assertEqual(kind[0], "rear")

    def test_anchor_of_car_is_rear_bumper(self):
        motion = box_motion("car", (8., 1.), heading=math.pi / 2, length=4., width=2.)
        np.testing.assert_allclose(bf.anchor_xy(motion), [8., -1.], atol=1e-12)
        np.testing.assert_allclose(bf.front_bumper([1., 2., math.pi / 2], NAVSIM_EGO), [1., 2. + FRONT])


if __name__ == "__main__":
    unittest.main()
