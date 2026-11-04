from __future__ import annotations

import unittest
import numpy as np

from reactbench.scoring.geometry import footprint_signed_clearance
from tests.scoring.reference import evaluate_trajectory_geometry
from reactbench.scoring.collisions import (
    first_contact_speed, paired_outcome, rectangle_corners, score_trajectory, signed_clearances, track_corners, validate_trajectory,
)

GEOMETRY = {"x_min": -1., "x_max": 3., "y_min": -1., "y_max": 1.}


def motion(x=12., y=0., velocity=(0., 0.)):
    return {"actor_type": "car", "initial_center_xy_m": [x, y], "velocity_xy_mps": list(velocity),
            "heading_rad": 0., "length_m": 2., "width_m": 2.}


def trajectory(speed=5., lateral=0.):
    times = np.arange(1, 41) * .1
    return np.column_stack((speed * times, lateral * np.minimum(times, 1.), np.zeros(40)))


def track(x=15., y=4., actor_id="background"):
    times = np.arange(41) * .1
    return {"actor_id": actor_id, "classes": ["automobile"], "times_s": times.tolist(),
            "poses": np.tile([x, y, 0.], (41, 1)).tolist(), "sizes": np.tile([2., 2.], (41, 1)).tolist()}


def world(*tracks):
    return {"horizon_s": 4., "tracks": list(tracks)}


class CollisionMetricsTests(unittest.TestCase):
    def test_contact_speed_uses_planar_incoming_segment_including_origin(self):
        poses = np.zeros((40, 3))
        poses[0, :2] = [.3, .4]  # 5 m/s from the current rear axle to t=0.1.
        poses[1:, :2] = [.3, .6]  # 2 m/s to t=0.2, then stopped.
        def at(time):
            return first_contact_speed(poses, [{"scope": "inserted", "actor_id": "inserted", "time_s": time}])
        for time in (.01, .05, .1):
            self.assertAlmostEqual(at(time)["ego_speed_mps"], 5.)
            self.assertAlmostEqual(at(time)["speed_interval_end_s"], .1)
        for time in (.10001, .15, .2):
            self.assertAlmostEqual(at(time)["ego_speed_mps"], 2.)
        self.assertEqual(at(4.)["ego_speed_mps"], 0.)
        self.assertIsNone(at(0.)["ego_speed_mps"])
        self.assertIsNone(first_contact_speed(poses, []))
        with self.assertRaisesRegex(ValueError, "horizon"):
            at(4.1)

    def test_contact_speed_compares_own_first_contact_even_with_different_actors(self):
        original = score_trajectory(trajectory(speed=5), motion(), GEOMETRY, world(track()))
        edited = score_trajectory(trajectory(speed=3, lateral=4), motion(), GEOMETRY, world(track()))
        self.assertEqual(original["first_contact"]["actors"][0]["actor_id"], "inserted")
        self.assertEqual(edited["first_contact"]["actors"][0]["actor_id"], "background")
        self.assertGreater(edited["first_contact"]["time_s"], original["first_contact"]["time_s"])
        value = paired_outcome(original, edited)["contact_speed"]
        self.assertEqual(value["status"], "scored")
        self.assertFalse(value["shared_first_contact_actor"])
        self.assertAlmostEqual(value["reduction_mps"], 2.)
        self.assertAlmostEqual(paired_outcome(edited, original)["contact_speed"]["reduction_mps"], -2.)

    def test_contact_speed_excludes_avoidance_initial_overlap_and_missing_speed(self):
        collide = score_trajectory(trajectory(), motion(), GEOMETRY, world())
        clear = score_trajectory(trajectory(speed=0), motion(), GEOMETRY, world())
        initial = score_trajectory(trajectory(), motion(x=0), GEOMETRY, world())
        for before, after, status in ((collide, clear, "not_both_collide"), (clear, collide, "not_both_collide"),
                                      (clear, clear, "not_both_collide"), (collide, initial, "initial_contact"),
                                      (collide, {**collide, "first_contact": None}, "missing_contact_speed")):
            with self.subTest(status=status):
                value = paired_outcome(before, after)["contact_speed"]
                self.assertEqual(value["status"], status)
                self.assertIsNone(value["reduction_mps"])
        same = paired_outcome(collide, collide)["contact_speed"]
        self.assertEqual(same["reduction_mps"], 0.)
        self.assertTrue(same["shared_first_contact_actor"])

    def test_vectorized_geometry_matches_existing_helper(self):
        rng = np.random.default_rng(42)
        a = rectangle_corners(rng.normal(size=(100, 3)), rng.uniform(.1, 3, (100, 2)))
        b = rectangle_corners(rng.normal(size=(100, 3)), rng.uniform(.1, 3, (100, 2)))
        np.testing.assert_allclose(signed_clearances(a, b), [footprint_signed_clearance(x, y) for x, y in zip(a, b)], atol=1e-12)

    def test_inserted_only_agrees_with_legacy_for_moving_and_stationary_objects(self):
        for actor in (motion(), motion(y=20), motion(y=-10, velocity=(0, 5)), motion(x=-2), motion(x=4)):
            for poses in (trajectory(), trajectory(lateral=4), trajectory(speed=0)):
                old = evaluate_trajectory_geometry(poses, actor, .1, GEOMETRY)
                new = score_trajectory(poses, actor, GEOMETRY, world())
                self.assertEqual(new["inserted"], old["collision"])
                self.assertEqual(new["all"], new["inserted"])
                self.assertFalse(new["background"])
                if new["inserted"]:
                    self.assertAlmostEqual(new["contacts"][0]["time_s"], old["time_to_collision_s"], places=5)

    def test_avoidance_that_hits_background_still_counts_as_collision(self):
        original = score_trajectory(trajectory(), motion(), GEOMETRY, world(track()))
        edited = score_trajectory(trajectory(lateral=4), motion(), GEOMETRY, world(track()))
        pair = paired_outcome(original, edited)
        self.assertEqual(pair["inserted"]["transition"], "avoided")
        self.assertEqual(pair["background"]["transition"], "introduced")
        self.assertEqual(pair["all"]["transition"], "persistent_collision")
        self.assertEqual(pair["all"]["benefit"], 0)
        safe = score_trajectory(trajectory(speed=1), motion(), GEOMETRY, world(track()))
        self.assertEqual(paired_outcome(original, safe)["all"]["benefit"], 1)

    def test_simultaneous_contacts_and_rear_touch_at_initial_time(self):
        score = score_trajectory(trajectory(), motion(x=-2), GEOMETRY, world(track(x=-2, y=0)))
        self.assertTrue(score["inserted"] and score["background"] and score["all"])
        self.assertEqual({c["actor_id"] for c in score["contacts"]}, {"inserted", "background"})
        self.assertTrue(all(c["time_s"] == 0 for c in score["contacts"]))
        self.assertEqual(len(score["first_contact"]["actors"]), 2)
        self.assertEqual(score["first_contact"]["speed_status"], "initial_contact")

    def test_contact_between_geometry_samples_and_heading_wrap(self):
        tiny = {"x_min": -.001, "x_max": .001, "y_min": -.001, "y_max": .001}
        crossing = track(x=1, y=0)
        crossing.update(times_s=[0., 4.], poses=[[1., -10.05, 0.], [1., 29.95, 0.]], sizes=[[.002, .002]] * 2)
        poses = trajectory(speed=0)
        poses[:, 0] = 1.
        score = score_trajectory(poses, motion(y=100), tiny, world(crossing))
        self.assertTrue(score["background"])
        self.assertAlmostEqual(score["contacts"][0]["time_s"], 1.0048, places=5)
        crossing.update(poses=[[1, 0, 3.13], [1, 0, -3.13]], sizes=[[4, 1]] * 2)
        expected = rectangle_corners(np.array([[1, 0, np.pi]]), np.array([[4, 1]]))[0]
        np.testing.assert_allclose(track_corners(crossing, [2])[0], expected, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "extrapolate"):
            track_corners(crossing, [-.1])

    def test_invalid_predictions_are_rejected(self):
        for poses, dt in ((trajectory()[:39], .1), (trajectory(), .5), (np.full((40, 3), np.nan), .1)):
            with self.assertRaises(ValueError):
                validate_trajectory(poses, dt)


if __name__ == "__main__":
    unittest.main()
