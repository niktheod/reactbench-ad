"""Stopped-vehicle cohort: ego lane chain, gap along the lane, first in path, strip fraction, target motion."""

from types import SimpleNamespace
import unittest

import numpy as np
from shapely.geometry import LineString, Point, box

from reactbench.selection import real_scenes as cohort

FRONT, HALF = cohort.EGO["front_m"], cohort.EGO["half_width_m"]
POINT = lambda x, y: SimpleNamespace(x=x, y=y)  # noqa: E731


def lane(lane_id, roadblock, x0, x1, y=0., width=3.5, heading=0.):
    """Straight lane along +x (or -x with heading pi) between x0 and x1 at lateral offset y."""
    xs = np.arange(x0, x1 + 1e-9, 1.)
    points = [SimpleNamespace(x=float(x), y=y, heading=heading) for x in (xs if heading == 0. else xs[::-1])]
    return SimpleNamespace(id=lane_id, polygon=box(x0, y - width / 2, x1, y + width / 2), outgoing_edges=[],
                           baseline_path=SimpleNamespace(discrete_path=points, length=float(x1 - x0)),
                           get_roadblock_id=lambda: roadblock)


class FakeMap:
    def __init__(self, lanes):
        self.lanes = lanes

    def get_all_map_objects(self, point, layer):
        return [l for l in self.lanes if layer == "lane" and l.polygon.contains(Point(point.x, point.y))]


def chain(map_api, xy=(10., 0.), heading=0., route=("r",), future=()):
    return cohort.ego_lane_chain(map_api, np.asarray(xy), heading, set(route), [np.asarray(p) for p in future],
                                 layers=("lane",), point=POINT)


class EgoLaneChainTest(unittest.TestCase):
    def test_follows_successors_until_the_lookahead(self):
        a, b, c, d = (lane(k, "r", 30 * i, 30 * (i + 1)) for i, k in enumerate("abcd"))
        a.outgoing_edges, b.outgoing_edges, c.outgoing_edges = [b], [c], [d]
        lanes, corridor = chain(FakeMap([a, b, c, d]))
        self.assertEqual([l.id for l in lanes], ["a", "b", "c"])  # 20 + 30 + 30 m ahead >= 60 m
        self.assertTrue(corridor.contains(Point(85., 0.)))

    def test_prefers_the_on_route_lane_and_skips_opposing_lanes(self):
        off = lane("off", "x", 0, 100)
        on = lane("on", "r", 0, 100)
        opposite = lane("opp", "r", 0, 100, heading=np.pi)
        lanes, _ = chain(FakeMap([off, opposite, on]))
        self.assertEqual(lanes[0].id, "on")
        lanes, _ = chain(FakeMap([opposite]))
        self.assertIsNone(lanes)

    def test_branch_follows_the_logged_future(self):
        a = lane("a", "r", 0, 30)
        straight = lane("straight", "r", 30, 100)
        shifted = lane("shifted", "r", 30, 100, y=3.5)
        a.outgoing_edges = [straight, shifted]
        lanes, _ = chain(FakeMap([a, straight, shifted]), future=[(40., 3.5), (50., 3.5)])
        self.assertEqual([l.id for l in lanes], ["a", "shifted"])
        lanes, _ = chain(FakeMap([a, straight, shifted]), future=[(40., 0.), (50., 0.)])
        self.assertEqual([l.id for l in lanes], ["a", "straight"])


def vehicle(gap, y=0., length=4.5, width=2., s_bumper=FRONT):
    x0 = s_bumper + gap
    return box(x0, y - width / 2, x0 + length, y + width / 2)


class SelectTargetTest(unittest.TestCase):
    line = LineString([(0., 0.), (100., 0.)])
    corridor = box(-10., -1.75, 100., 1.75)

    def select(self, objects, corridor=None):
        return cohort.select_target(objects, corridor or self.corridor, self.line, FRONT, HALF)

    def test_keeps_a_stopped_vehicle_in_the_lane(self):
        target, flags = self.select([("car", "vehicle", vehicle(8.), 0.)])
        self.assertAlmostEqual(target["gap_lanepath_m"], 8.)
        self.assertAlmostEqual(target["strip_fraction"], 1.)
        self.assertEqual(flags, {"in_lane_candidate": True, "first_in_path": True, "kept": True})

    def test_rejects_moving_too_close_too_far_or_protruding_vehicles(self):
        for objects in ([("car", "vehicle", vehicle(8.), .6)], [("car", "vehicle", vehicle(3.), 0.)],
                        [("car", "vehicle", vehicle(15.), 0.)], [("car", "vehicle", vehicle(8., y=1.), 0.)]):
            target, flags = self.select(objects)
            self.assertIsNone(target)
            self.assertFalse(flags["in_lane_candidate"])

    def test_anything_nearer_in_the_strip_blocks_the_target(self):
        objects = [("car", "vehicle", vehicle(8.), 0.), ("ped", "pedestrian", box(9., -.3, 9.6, .3), 1.2)]
        target, flags = self.select(objects)
        self.assertIsNone(target)
        self.assertEqual((flags["in_lane_candidate"], flags["first_in_path"]), (True, False))

    def test_objects_beside_the_strip_or_behind_the_ego_do_not_block(self):
        objects = [("car", "vehicle", vehicle(8.), 0.), ("ped", "pedestrian", box(9., 1.3, 9.6, 1.7), 1.2),
                   ("follower", "vehicle", box(-8., -1., -3.5, 1.), 3.)]
        target, _ = self.select(objects)
        self.assertEqual(target["track_token"], "car")

    def test_picks_the_nearest_unblocked_vehicle(self):
        far, near = ("far", "vehicle", vehicle(12., y=.2), 0.), ("near", "vehicle", vehicle(6., y=-.1, length=1.), 0.)
        target, _ = self.select([far, near])
        self.assertEqual(target["track_token"], "near")

    def test_needs_half_of_the_vehicle_inside_the_ego_width(self):
        wide = box(-10., -3.5, 100., 3.5)
        target, flags = self.select([("edge", "vehicle", vehicle(8., y=2.2), 0.)], corridor=wide)
        self.assertIsNone(target)
        self.assertEqual(flags, {"in_lane_candidate": True, "first_in_path": True, "kept": False})
        target, _ = self.select([("half", "vehicle", vehicle(8., y=1.1), 0.)], corridor=wide)
        self.assertGreaterEqual(target["strip_fraction"], .5)

    def test_gap_is_measured_along_a_curved_lane(self):
        radius = 20.
        angles = np.linspace(-.5, 1.5, 400)
        line = LineString(np.column_stack((radius * np.sin(angles), radius * (1 - np.cos(angles)))))
        s_ego = line.project(Point(0., 0.))
        s_car = s_ego + FRONT + 10.
        c = line.interpolate(s_car)
        car = Point(c.x, c.y).buffer(.5, cap_style=3)  # 1 m square around the centreline point
        target, _ = cohort.select_target([("car", "vehicle", car, 0.)], line.buffer(1.75), line, s_ego + FRONT, HALF)
        # The square's nearest corner lies at most 0.5 * (|cos| + |sin|) of the tangent before its centre.
        self.assertTrue(10. - .75 < target["gap_lanepath_m"] < 10. - .5, target["gap_lanepath_m"])
        self.assertGreater(target["gap_lanepath_m"], c.x - FRONT - .5)  # longer than the straight-ahead distance


class HelpersTest(unittest.TestCase):
    def test_box_corners(self):
        corners = cohort.box_corners([1., 2., 0., 4., 2., 1.5, np.pi / 2])
        np.testing.assert_allclose(sorted(map(tuple, np.round(corners, 9))), [(0., 0.), (0., 4.), (2., 0.), (2., 4.)])

    def test_target_motion(self):
        def frame(x):
            return {"ego2global": np.eye(4), "anns": {"track_tokens": ["car"], "gt_boxes": [[x, 0., 0., 4., 2., 1., 0.]]}}
        frames = [frame(10.), frame(10.), frame(10.4), frame(11.)] + [frame(11.)] * 5 + [frame(20.)]
        self.assertAlmostEqual(cohort.target_motion(frames, 0, "car"), 1.)  # the frame at 4.5 s is ignored


if __name__ == "__main__":
    unittest.main()
