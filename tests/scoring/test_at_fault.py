"""At-fault classification (protocol V2) and its NAVSIM/PAI-AV scoring integration."""
from __future__ import annotations

import unittest

import numpy as np
from shapely.geometry import box

from reactbench.scoring.at_fault import (
    classify_contact, classify_score, segment_box_distance, spans_several_lanes,
)
from tests.scoring.test_collisions import motion

GEOMETRY = {"x_min": -1., "x_max": 3., "y_min": -1., "y_max": 1.}  # rear axle at the origin


def straight(speed, dt=.5, count=8):
    """Constant-speed forward motion along +x (negative speed: reversing, heading still forward)."""
    times = np.arange(1, count + 1) * dt
    return np.column_stack((speed * times, np.zeros(count), np.zeros(count)))


def rect(cx, cy, length=2., width=2.):
    return np.array([[cx + length / 2, cy + width / 2], [cx + length / 2, cy - width / 2],
                     [cx - length / 2, cy - width / 2], [cx - length / 2, cy + width / 2]])


def classify(poses, time_s, corners, *, stopped=False, vulnerable=False, map_context=None, dt=.5):
    return classify_contact(poses, dt, time_s, GEOMETRY, corners, other_stopped=stopped,
                            vulnerable=vulnerable, map_context=map_context)


def lanes_map(lanes, drivable=(box(-100, -100, 100, 100),)):
    return {"origin": (0., 0., 0.), "drivable": list(drivable), "lanes": list(lanes)}


class RuleTests(unittest.TestCase):
    # Ego at 2 m/s: at t=1 s the rear axle is at x=2, footprint x in [1, 5], y in [-1, 1].
    moving = straight(2.)

    def test_follower_rear_ending_a_moving_ego_is_exempt(self):
        self.assertEqual(classify(self.moving, 1., rect(0., 0.)), ("rear", False))

    def test_front_bumper_contact_is_at_fault(self):
        self.assertEqual(classify(self.moving, 1., rect(6., 0.)), ("front", True))
        # The bumper corner belongs to the front edge.
        self.assertEqual(classify(self.moving, 1., rect(6., 2.)), ("front", True))

    def test_side_contacts(self):
        pedestrian = rect(3., 1.25, .5, .5)
        self.assertEqual(classify(self.moving, 1., pedestrian, vulnerable=True), ("lateral_vulnerable", True))
        # PAI-AV: no map, so a vehicle side contact is exempt.
        self.assertEqual(classify(self.moving, 1., pedestrian), ("lateral", False))

    def test_navsim_lateral_uses_lanes_and_drivable_area(self):
        car = rect(3., 2., 2., 2.)
        one_lane = lanes_map([box(-100, -2, 100, 2)])
        self.assertEqual(classify(self.moving, 1., car, map_context=one_lane), ("lateral", False))
        two_lanes = lanes_map([box(-100, -2, 100, 0), box(-100, 0, 100, 2)])
        self.assertEqual(classify(self.moving, 1., car, map_context=two_lanes), ("lateral_several_lanes", True))
        # A lane holding all four corners keeps the ego in a single lane, even if another overlaps it.
        overlap = lanes_map([box(-100, -2, 100, 2), box(-100, 0, 100, 4)])
        self.assertEqual(classify(self.moving, 1., car, map_context=overlap), ("lateral", False))
        off_road = lanes_map([box(-100, -2, 100, 2)], drivable=[box(-100, -.5, 100, 3)])
        self.assertEqual(classify(self.moving, 1., car, map_context=off_road), ("lateral_off_road", True))

    def test_stopped_ego_and_stopped_actor(self):
        self.assertEqual(classify(np.zeros((8, 3)), 1., rect(4., 0.)), ("stopped_ego", False))
        self.assertEqual(classify(np.zeros((8, 3)), 1., rect(4., 0.), stopped=True), ("stopped_ego", False))
        self.assertEqual(classify(self.moving, 1., rect(3., 2.), stopped=True), ("stopped_actor", True))

    def test_reversing_into_a_follower_is_at_fault(self):
        backwards = straight(-1.)  # rear axle at x=-1 at t=1, rear bumper at x=-2
        self.assertEqual(classify(backwards, 1., rect(-3., 0.)), ("rear_reversing", True))

    def test_speed_uses_the_incoming_segment_and_the_first_segment_at_t0(self):
        stop_then_go = np.array([[0, 0, 0]] * 2 + [[k - 1., 0, 0] for k in range(2, 8)], dtype=float)
        self.assertEqual(classify(stop_then_go, 1., rect(-2., 0.))[0], "stopped_ego")
        self.assertEqual(classify(stop_then_go, 1.01, rect(-1., 0.))[0], "rear")
        self.assertEqual(classify(self.moving, 0., rect(4., 0.)), ("front", True))

    def test_segment_box_distance(self):
        corners = rect(0., 0.)
        self.assertEqual(segment_box_distance(np.array([-.5, 0]), np.array([.5, 0]), corners), 0.)
        self.assertEqual(segment_box_distance(np.array([1., -3.]), np.array([1., 3.]), corners), 0.)
        self.assertAlmostEqual(segment_box_distance(np.array([2., -3.]), np.array([2., 3.]), corners), 1.)
        self.assertAlmostEqual(segment_box_distance(np.array([2., 2.]), np.array([3., 3.]), corners), np.sqrt(2))

    def test_spans_several_lanes_ignores_boundaries(self):
        corners = np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]])
        self.assertFalse(spans_several_lanes(corners, []))
        self.assertTrue(spans_several_lanes(corners, [box(-5, -5, 5, 0), box(-5, 0, 5, 5)]))
        # A corner on a lane boundary is not inside that lane (as PDMS's contains test).
        self.assertFalse(spans_several_lanes(corners, [box(-5, -5, 5, 1), box(-5, 1, 5, 5)]))


def track(actor_id, x, y, classes, speed=0., times=None):
    times = np.arange(41) * .1 if times is None else np.asarray(times)
    return {"actor_id": actor_id, "classes": classes, "times_s": times.tolist(),
            "poses": [[x + speed * t, y, 0.] for t in times], "sizes": [[2., 2.]] * len(times)}


class ScoreTests(unittest.TestCase):
    def test_first_at_fault_contact_drives_collision_flags_and_contact_speed(self):
        poses = straight(2.)
        follower = track("follower", -1., 0., ["VEHICLE"], speed=1.)  # centre at x=0 at t=1: touches the rear
        car = motion(x=6., y=0.)  # stationary inserted car ahead: front contact at t=2 (front at x=7)
        physical = {"contacts": [{"scope": "background", "actor_id": "follower", "classes": ["VEHICLE"], "time_s": 1.},
                                 {"scope": "background", "actor_id": "follower", "classes": ["VEHICLE"], "time_s": 3.},
                                 {"scope": "inserted", "actor_id": "inserted", "time_s": 2.}]}
        score = classify_score(physical, poses, .5, actor_motion=car, tracks=[follower], ego_geometry=GEOMETRY,
                               dataset="navsim", map_context=None)
        self.assertEqual((score["all"], score["inserted"], score["background"]), (True, True, False))
        self.assertEqual([c["type"] for c in score["contacts"]], ["stopped_actor"])
        # Only each actor's first contact is classified, as PDMS ignores repeat contacts.
        self.assertEqual([(c["type"], c["time_s"]) for c in score["exempt_contacts"]], [("rear", 1.)])
        self.assertEqual(score["first_contact"]["time_s"], 2.)
        self.assertAlmostEqual(score["first_contact"]["ego_speed_mps"], 2.)

    def test_static_classes_are_always_stopped(self):
        poses = straight(2.)
        jitter = track("cone", 3., 2., ["TRAFFIC_CONE"], speed=.5)
        physical = {"contacts": [{"scope": "background", "actor_id": "cone", "classes": ["TRAFFIC_CONE"], "time_s": 1.}]}
        score = classify_score(physical, poses, .5, actor_motion=None, tracks=[jitter], ego_geometry=GEOMETRY,
                               dataset="navsim", map_context=None)
        self.assertEqual(score["contacts"][0]["type"], "stopped_actor")
        parked = track("parked", 3., 2., ["protruding_object"], speed=.5)
        physical["contacts"][0]["actor_id"] = "parked"
        score = classify_score(physical, poses, .5, actor_motion=None, tracks=[parked], ego_geometry=GEOMETRY,
                               dataset="pai-av")
        self.assertEqual(score["contacts"][0]["type"], "stopped_actor")

    def test_moving_pai_av_vehicle_side_contact_is_exempt_and_person_is_not(self):
        poses = straight(2.)
        for classes, expected in ((["automobile"], ("lateral", False)), (["person"], ("lateral_vulnerable", True))):
            other = track("other", 2.5, 2., classes, speed=1.)  # beside the ego, clear of its front edge
            physical = {"contacts": [{"scope": "background", "actor_id": "other", "classes": classes, "time_s": 1.}]}
            score = classify_score(physical, poses, .5, actor_motion=None, tracks=[other], ego_geometry=GEOMETRY,
                                   dataset="pai-av")
            contact = (score["contacts"] + score["exempt_contacts"])[0]
            self.assertEqual((contact["type"], contact["at_fault"]), expected)
            self.assertEqual(score["all"], expected[1])
            self.assertEqual(score["first_contact"] is None, not expected[1])


def world(*tracks):
    return {"horizon_s": 4., "tracks": list(tracks), "geometry": GEOMETRY, "origin": [0., 0., 0.],
            "map_status": "available", "map_polygons_wkb": [box(-100, -100, 100, 100).wkb_hex]}


class NavsimV2Tests(unittest.TestCase):
    def test_v2_keeps_v1_physical_contacts_and_exempts_a_rear_ending_follower(self):
        from reactbench.scoring.navsim_metrics import MapContext, score_prediction_at_fault
        from tests.scoring.reference import score_prediction
        braking = np.column_stack(([1., 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25], np.zeros((8, 2))))  # slows to 0.5 m/s
        follower = track("follower", -6., 0., ["VEHICLE"], speed=4.)  # reaches the ego's rear bumper at t=1.5 s
        w = world(follower)
        v1 = score_prediction(braking, .5, None, w)
        v2 = score_prediction_at_fault(braking, .5, None, w, MapContext("token", w, "/nonexistent"))
        self.assertEqual(v1["nc_all"], 0)
        self.assertEqual((v2["nc"], v2["nc_dac"], v2["first_contact"]), (1, 1, None))
        self.assertEqual([c["type"] for c in v2["exempt_contacts"]], ["rear"])
        strip = lambda c: {k: v for k, v in c.items() if k not in ("type", "at_fault")}
        self.assertEqual([strip(c) for c in v2["contacts"] + v2["exempt_contacts"]], v1["contacts"])
        self.assertNotIn("nc_all", v2)

    def test_batched_physical_scores_give_the_scalar_v2_result(self):
        from reactbench.scoring.navsim_batch import score_predictions
        from reactbench.scoring.navsim_metrics import MapContext, at_fault_from_physical, score_prediction_at_fault
        follower = track("follower", -6., 0., ["VEHICLE"], speed=4.)
        w = world(follower)
        poses = np.stack([straight(v) for v in (0., 1., 3.)])
        context = MapContext("token", w, "/nonexistent")
        batch = score_predictions(poses, .5, motion(x=9.), w)
        for p, physical in zip(poses, batch):
            self.assertEqual(at_fault_from_physical(physical, p, .5, motion(x=9.), w, context),
                             score_prediction_at_fault(p, .5, motion(x=9.), w, context))

    def test_lateral_contact_without_map_fails_instead_of_passing(self):
        from reactbench.scoring.navsim_metrics import MapContext, score_prediction_at_fault
        side = track("side", .5, 2.05, ["VEHICLE"], speed=2.)  # beside the rear half; the ego drifts left into it
        poses = np.column_stack((2. * np.arange(1, 9) * .5, np.minimum(np.arange(1, 9) * .5, 1.) * .5, np.zeros(8)))
        w = {**world(side), "map_status": "unavailable", "map_error": "test"}
        with self.assertRaisesRegex(ValueError, "Map unavailable"):
            score_prediction_at_fault(poses, .5, None, w, MapContext("token", w, "/nonexistent"))


class ReportingAndRecordTests(unittest.TestCase):
    def test_v2_summary_carries_contract_and_transition_tokens(self):
        from reactbench.scoring.collisions import paired_outcome
        from reactbench.scoring.reporting import summarize

        def v2(at_fault, inserted):
            return {"all": at_fault, "inserted": inserted, "background": at_fault and not inserted, "dac": 1,
                    "first_contact": None}
        rows = []
        for token, before, after in (("avoided", (True, True), (False, False)),
                                     ("introduced", (False, False), (True, False)), ("same", (True, True), (True, True))):
            a, b = v2(*before), v2(*after)
            rows.append({"token": token, "status": "scored", "groups": {}, "primary_pair": "edited",
                         "scores": {"original": a, "edited": b},
                         "pairs": {"edited": {"from_condition": "original", "to_condition": "edited", **paired_outcome(a, b)}}})
        summary = summarize(rows, "nc_dac")
        self.assertEqual((summary["version"], summary["family"], summary["contract"]["version"]), (2, "at_fault", 2))
        self.assertEqual(summary["transition_tokens"]["nc_dac"], {"improved": ["avoided"], "worsened": ["introduced"]})
        self.assertEqual(summary["transition_tokens"]["nc_inserted"], {"improved": ["avoided"], "worsened": []})

    def test_new_sample_records_hold_no_metrics(self):
        from reactbench.scoring.records import sample_record, visualization_title
        record = sample_record(token="t", output_stem="t", speed_mps=5., actor_motion=motion(),
                               original_trajectory=straight(1.), edited_trajectory=straight(2.), interval_length_s=.5)
        legacy = {"ade_m", "original_collision", "edited_collision", "original_time_to_critical_s",
                  "original_minimum_clearance_m", "original_distance_driven_m", "distance_driven_drop_percentage"}
        self.assertFalse(legacy & set(record))
        self.assertEqual((record["speed_kmh"], record["actor_type"]), (18., "car"))
        self.assertEqual(visualization_title(record, str), "Token t | 18.0 km/h")
        legacy_record = {**record, "ade_m": .5, "original_time_to_collision_s": 1., "edited_time_to_collision_s": None}
        self.assertEqual(visualization_title(legacy_record, lambda t: "none" if t is None else f"{t:.1f} s"),
                         "Token t | 18.0 km/h | ADE 0.50 m | raw inserted contact original 1.0 s | edited none")


if __name__ == "__main__":
    unittest.main()
