"""Exact batched/scalar parity, especially at geometric and temporal boundaries."""
import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from reactbench.scoring.navsim_batch import score_predictions
from tests.scoring.reference import score_prediction


GEOMETRY = {"x_min": -1., "x_max": 1., "y_min": -.5, "y_max": .5}


def track(x=4., y=0., *, times=(0., 4.), name="a", sizes=None, headings=None):
    return {"actor_id": name, "classes": ["VEHICLE"], "times_s": list(times),
            "poses": [[x, y, 0 if headings is None else headings[i]] for i in range(len(times))],
            "sizes": [[2., 1.]] * len(times) if sizes is None else sizes}


def world(tracks=()):
    return {"horizon_s": 4., "tracks": list(tracks), "geometry": GEOMETRY,
            "origin": [0., 0., 0.], "map_status": "available",
            "map_polygons_wkb": [box(-20, -5, 30, 5).wkb_hex]}


def candidates(interval=.5):
    ts = np.arange(1, round(4/interval)+1) * interval
    return np.stack([np.column_stack((ts*v, ts*lateral, np.zeros_like(ts)))
                     for v, lateral in ((0, 0), (1, 0), (2, 0), (5, 0), (10, 0), (2, 2), (2, -2), (-2, 0))])


def check(poses, w, actor=None, interval=.5):
    snapshot = copy.deepcopy(w)
    expected = [score_prediction(p, interval, actor, w) for p in poses]
    actual = score_predictions(poses, interval, actor, w)
    assert actual == expected  # Includes exact times, speeds, every contact and DAC diagnostics.
    assert w == snapshot
    return actual


@pytest.mark.parametrize("interval", [.5, .1])
@pytest.mark.parametrize("tracks", [[], [track()], [track(x=1)], [track(x=100)],
    [track(times=(.031, .071))], [track(times=(.5,))],
    [track(times=(0, .5)), track(times=(2, 4))],
    [track(times=(0., .017, .51, 1.13, 4.), sizes=[[2, 1], [3, 1], [1, 2], [4, 1], [2, 2]])]])
def test_full_dictionary_parity_and_track_lifetimes(interval, tracks):
    check(candidates(interval), world(tracks), interval=interval)


def test_fast_crossing_rotations_heading_wrap_and_irregular_grids():
    crossing = track(x=0., times=(.031, .039), name="crossing", sizes=[[.2, .2]] * 2)
    crossing["poses"] = [[0, -4, 0], [0, 4, 0]]
    # Native endpoint boxes are clear; intermediate rotation hits the ego.
    turning = track(x=2.25, times=(0, .1), name="turning", sizes=[[.4, 4.]] * 2,
                    headings=[0, np.pi])
    wrapped = track(x=6., name="wrapped", headings=[3.13, -3.13])
    poses = candidates()
    poses[1, :, 2] = [3.13, -3.13, 3.12, -3.12, 3.1, -3.1, 3.1, -3.1]
    result = check(poses, world([crossing, turning, wrapped]))
    assert result[0]["all"] and 0 < result[0]["first_contact"]["time_s"] < .04
    assert {c["actor_id"] for c in result[0]["contacts"]} == {"crossing", "turning"}


def test_changing_dimensions_can_contact_between_clear_grid_samples():
    # Both endpoints are disjoint; the intermediate 4x4 box covers the ego.
    actor = track(x=1.9, y=1.9, times=(.031, .039), sizes=[[6., 2.], [2., 6.]])
    w = world([actor])
    w["geometry"] = {"x_min": -.1, "x_max": .1, "y_min": -.1, "y_max": .1}
    result = check(np.zeros((2, 8, 3)), w)
    assert .031 < result[0]["first_contact"]["time_s"] < .039


def test_native_knot_incoming_speed_simultaneous_contacts_and_t0():
    poses = candidates()
    poses[0, :, 0] = [1., 4., 4., 4., 4., 4., 4., 4.]
    result = check(poses, world([track(x=3, name="b", times=(.5, 4)),
                                track(x=3, name="a", times=(.5, 4))]))
    first = result[0]["first_contact"]
    assert first["time_s"] == .5 and first["ego_speed_mps"] == 2
    assert [a["actor_id"] for a in first["actors"]] == ["a", "b"]
    initial = check(poses, world([track(x=1)]))
    assert all(r["first_contact"]["speed_status"] == "initial_contact" for r in initial)


@pytest.mark.parametrize("offset", [-1e-10, 0., 1e-10, 1e-8])
def test_near_touching_rotated_bounds(offset):
    w = world([track(x=2 + offset, times=(0, 4))])
    check(candidates(), w)


def test_polygon_holes_boundaries_and_global_origin():
    w = world()
    w["origin"] = [100., 100., np.pi/2]
    polygon = Polygon([(90, 90), (110, 90), (110, 120), (90, 120)],
                      holes=[[(98, 105), (102, 105), (102, 110), (98, 110)]])
    w["map_polygons_wkb"] = [polygon.wkb_hex, box(99, 105, 101, 107).wkb_hex]
    check(candidates(), w)
    w = world()
    w["map_polygons_wkb"] = [box(-1, -.5, 40, .5).wkb_hex]
    assert all(r["dac"] == 0 for r in check(candidates(), w))
    w["map_status"], w["map_error"] = "unavailable", "test"
    assert all(r["dac"] is None for r in check(candidates(), w))


def test_inserted_actor_and_background_first_contact():
    actor = {"initial_center_xy_m": [6., 0.], "velocity_xy_mps": [0., 0.],
             "heading_rad": 0., "front_length_m": 1., "rear_length_m": 1.,
             "left_width_m": .5, "right_width_m": .5}
    check(candidates(), world([track()]), actor)


@pytest.mark.parametrize("seed", range(4))
def test_randomized_curved_braking_and_moving_actors(seed):
    rng = np.random.default_rng(seed)
    poses = candidates()
    poses[:, :, :2] += rng.normal(0, .7, size=poses[:, :, :2].shape)
    poses[:, :, 2] = rng.normal(0, .4, size=poses[:, :, 2].shape)
    tracks = []
    for i in range(8):
        ts = np.sort(rng.choice(np.arange(9)*.5, size=4, replace=False))
        obj = track(times=ts, name=str(i))
        obj["poses"] = np.column_stack((rng.uniform(-5, 20) + ts*rng.uniform(-2, 4),
                                        rng.uniform(-4, 4) + ts*rng.uniform(-1, 1),
                                        rng.uniform(-np.pi, np.pi, len(ts)))).tolist()
        obj["sizes"] = rng.uniform(.3, 3, size=(len(ts), 2)).tolist()
        tracks.append(obj)
    check(poses, world(tracks))


def test_thread_safety_candidate_order_and_empty_batch():
    w, poses = world([track(), track(x=10, times=(.123, 3.51), name="b")]), candidates()
    expected = check(poses, w)
    orders = [np.arange(len(poses)), np.arange(len(poses))[::-1], np.array([2, 2, 0, 5])]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda order: score_predictions(poses[order], .5, None, w), orders))
    for order, result in zip(orders, results):
        assert result == [expected[i] for i in order]
    assert score_predictions(np.empty((0, 8, 3)), .5, None, w) == []


@pytest.mark.parametrize("poses,interval", [(np.zeros((1, 7, 3)), .5),
    (np.zeros((1, 8, 2)), .5), (np.full((1, 8, 3), np.nan), .5),
    (np.zeros((8, 3)), .5), (np.zeros((1, 8, 3)), 0), (np.zeros((1, 8, 3)), np.nan)])
def test_invalid_inputs_remain_errors(poses, interval):
    with pytest.raises(ValueError):
        score_predictions(poses, interval, None, world())
