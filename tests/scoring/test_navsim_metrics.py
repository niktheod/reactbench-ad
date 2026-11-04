from __future__ import annotations
import unittest
import numpy as np
from shapely.geometry import box
from tests.scoring.reference import score_prediction
from reactbench.scoring.collisions import first_contact_speed, paired_outcome, sample_pose
from reactbench.scoring.reporting import comparison_summary, contact_summary, summarize_original
from tests.scoring.test_collisions import GEOMETRY, motion, track


def world(*tracks):
    return {"horizon_s": 4., "tracks": list(tracks), "geometry": GEOMETRY, "origin": [0., 0., 0.],
            "map_status": "available", "map_polygons_wkb": [box(-100, -100, 100, 100).wkb_hex]}


class RawMetricTests(unittest.TestCase):
    def test_original_missing_map_is_excluded_even_when_collision_fails(self):
        poses = np.zeros((8, 3))
        clear = score_prediction(poses, .5, None, world())
        missing = score_prediction(poses, .5, None, {**world(track(x=0, y=0)), "map_status": "unavailable"})
        self.assertEqual(missing["nc_all"], 0)
        rows = [{"status": status, "scores": {"original": score}}
                for status, score in (("scored", clear), ("partial", missing), ("failed", None))]
        overall = summarize_original(rows)["overall"]
        self.assertEqual((overall["valid_count"], overall["excluded_count"]), (1, 2))
        self.assertEqual(overall["mean_nc_dac"], 1.)
        self.assertEqual(overall["mean_nc"], 1.)
        self.assertIsNone(summarize_original(rows[1:])["overall"]["mean_nc_dac"])

    def test_native_half_second_contact_and_impossible_stationary_prediction(self):
        poses = np.column_stack((5*np.arange(1, 9)*.5, np.zeros((8, 2))))
        score = score_prediction(poses, .5, motion(), world())
        self.assertEqual((score["nc_all"], score["nc_inserted"], score["dac"], score["nc_dac"]), (0, 0, 1, 0))
        self.assertAlmostEqual(score["first_contact"]["ego_speed_mps"], 5.)
        self.assertAlmostEqual(score["first_contact"]["time_s"], 1.6, places=5)
        # No measured initial speed is allowed to move a stationary raw prediction.
        stopped = score_prediction(np.zeros((8, 3)), .5, motion(), {**world(), "initial_velocity": [100, 0]})
        self.assertEqual(stopped["nc_dac"], 1)

    def test_background_crossing_between_waypoints_and_initial_contact(self):
        crossing = track(x=0, y=0)
        crossing.update(times_s=[0, .5, 4], poses=[[0,-5,0],[0,5,0],[0,5,0]], sizes=[[.1,.1]]*3)
        value = score_prediction(np.zeros((8,3)), .5, motion(y=30), world(crossing))
        self.assertTrue(value["background"])
        self.assertFalse(value["inserted"])
        self.assertGreater(value["first_contact"]["time_s"], 0)
        self.assertLess(value["first_contact"]["time_s"], .5)
        self.assertEqual(value["first_contact"]["ego_speed_mps"], 0)
        initial = score_prediction(np.zeros((8,3)), .5, motion(x=0), world())
        self.assertEqual(initial["first_contact"]["speed_status"], "initial_contact")

    def test_dac_corners_boundary_missing_map_and_product_before_average(self):
        poses = np.zeros((8,3))
        w = world()
        w["map_polygons_wkb"] = [box(-1,-1,3,1).wkb_hex]
        boundary = score_prediction(poses, .5, motion(y=30), w)
        self.assertEqual((boundary["nc_all"], boundary["dac"], boundary["nc_dac"]), (1,0,0))
        # Different corners may lie inside different overlapping polygons.
        w["map_polygons_wkb"] = [box(-2,-2,1.1,2).wkb_hex, box(.9,-2,4,2).wkb_hex]
        clear = score_prediction(poses, .5, motion(y=30), w)
        self.assertEqual(clear["dac"], 1)
        missing = score_prediction(poses, .5, motion(y=30), {**w,"map_status":"unavailable"})
        self.assertIsNone(missing["nc_dac"])
        collide = score_prediction(poses, .5, motion(x=0), world())
        rows=[]
        for a,b in ((boundary,collide),(collide,boundary),(missing,clear)):
            rows.append({"status":"scored","scores":{"original":a,"edited":b},"primary_pair":"edited",
                         "pairs":{"edited":{"from_condition":"original","to_condition":"edited",**paired_outcome(a,b)}}})
        s=comparison_summary(rows,"nc_dac")
        self.assertEqual(s["paired_count"],2)
        self.assertEqual(s["primary"]["from_rate"],0)
        self.assertEqual(s["primary"]["transitions"]["both_fail"],2)
        self.assertEqual(s["contact_speed"]["count"],0)

    def test_incoming_native_derivative_and_heading_wrap(self):
        p=np.array([[3,4,3.13],[3,5,-3.13]]+[[3,5,-3.13]]*6)
        self.assertAlmostEqual(first_contact_speed(p,[{"time_s":.5}],.5)["ego_speed_mps"],10)
        self.assertAlmostEqual(first_contact_speed(p,[{"time_s":.50001}],.5)["ego_speed_mps"],2)
        self.assertAlmostEqual(sample_pose(p,.75,.5)[2],np.pi)

    def test_delta_percentage_is_mean_of_ratios_and_zero_reference_is_separate(self):
        pairs=[{"contact_speed":{"status":"scored","from_speed_mps":a,"to_speed_mps":b,
                                   "shared_first_contact_actor":True}} for a,b in [(10,5),(2,0),(0,1)]]
        s=contact_summary(pairs)
        self.assertEqual((s["count"],s["percentage_count"],s["zero_reference_speed_count"]),(3,2,1))
        self.assertEqual(s["mean_delta_mps"],-2)
        self.assertEqual(s["median_delta_mps"],-2)
        self.assertEqual(s["mean_percentage_delta"],-75)


if __name__ == "__main__":
    unittest.main()
