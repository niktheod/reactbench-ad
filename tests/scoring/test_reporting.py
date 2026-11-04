"""Original-scene support preserves native contact and edited collision semantics."""
import numpy as np
import pytest

from reactbench.scoring.collisions import first_contact_speed, score_raw_trajectory, score_trajectory

GEOMETRY = {"x_min": -1., "x_max": 1., "y_min": -.5, "y_max": .5}


def motion(x, y=0):
    return {"initial_center_xy_m": [x, y], "velocity_xy_mps": [0, 0], "heading_rad": 0,
            "front_length_m": 1., "rear_length_m": 1., "left_width_m": .5, "right_width_m": .5}


def world(x=None):
    return {"horizon_s": 4., "tracks": [] if x is None else [
        {"actor_id": "background", "classes": ["VEHICLE"], "times_s": [0, 4],
         "poses": [[x, 0, 0]] * 2, "sizes": [[2, 1]] * 2}]}


def poses(interval=.5, speed=2):
    return np.column_stack((np.arange(1, round(4/interval)+1) * interval * speed,
                            np.zeros((round(4/interval), 2))))


def test_no_insertion_retains_background_collisions():
    original = score_raw_trajectory(poses(), None, GEOMETRY, world(4), interval_length_s=.5)
    # A distant actor yields the exact same existing background contact result.
    edited = score_raw_trajectory(poses(), motion(100), GEOMETRY, world(4), interval_length_s=.5)
    assert original == edited
    assert original["all"] and original["background"] and not original["inserted"]
    assert original["first_contact"]["time_s"] == pytest.approx(1., abs=1e-6)
    assert original["first_contact"]["ego_speed_mps"] == 2
    clear = score_raw_trajectory(poses(), None, GEOMETRY, world(), interval_length_s=.5)
    assert not clear["all"] and clear["first_contact"] is None


def test_inserted_actor_scoring_unchanged_and_first_contact_chooses_any_object():
    result = score_raw_trajectory(poses(), motion(6), GEOMETRY, world(4), interval_length_s=.5)
    assert result["all"] and result["background"] and result["inserted"]
    assert [c["scope"] for c in result["contacts"]] == ["background", "inserted"]
    assert [c["time_s"] for c in result["contacts"]] == pytest.approx([1, 2], abs=1e-6)
    native_paiav = score_trajectory(poses(.1), motion(6), GEOMETRY, world(4))
    assert native_paiav["first_contact"]["ego_speed_mps"] == pytest.approx(2)
    assert native_paiav["contacts"] == result["contacts"]


def test_incoming_segment_at_native_knot_and_initial_contact():
    trajectory = poses()
    trajectory[0, 0] = .5
    trajectory[1:, 0] = 3
    contact = first_contact_speed(trajectory, [{"time_s": .5, "scope": "background", "actor_id": "a"}], .5)
    assert contact["ego_speed_mps"] == 1 and contact["speed_interval_start_s"] == 0
    after = first_contact_speed(trajectory, [{"time_s": .500001}], .5)
    assert after["ego_speed_mps"] == 5
    initial = score_raw_trajectory(trajectory, None, GEOMETRY, world(1), interval_length_s=.5)
    assert initial["all"] and initial["first_contact"]["speed_status"] == "initial_contact"
    assert initial["first_contact"]["ego_speed_mps"] is None


def test_fast_crossing_between_pdms_timestamps_is_contact():
    # Present only between t=0 and 0.1 with no endpoint overlap: evaluation's
    # finer geometry, rather than the 10 Hz PDMS occupancy check, must be used.
    moving = world()
    moving["tracks"] = [{"actor_id": "crossing", "classes": ["PEDESTRIAN"], "times_s": [0., .1],
                         "poses": [[0, -4, 0], [0, 4, 0]], "sizes": [[.2, .2]] * 2}]
    result = score_raw_trajectory(poses(speed=0), None, GEOMETRY, moving, interval_length_s=.5)
    assert result["all"] and 0 < result["first_contact"]["time_s"] < .1
    assert result["first_contact"]["ego_speed_mps"] == 0
