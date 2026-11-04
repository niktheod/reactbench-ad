from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from reactbench.compositing.paiav import classify_samples as classifier


def passing_candidate() -> dict:
    return {"evaluation_complete": True, "passed": True, "failures": []}


def failing_candidate(code: str) -> dict:
    return {
        "evaluation_complete": True,
        "passed": False,
        "failures": [{"code": code, "message": f"failed with {code}"}],
    }


def test_default_horizons_are_physically_equivalent_to_navsim() -> None:
    assert classifier.DEFAULT_TIMESTEPS == (20, 15, 10, 5)
    slots = classifier.candidate_slots(classifier.DEFAULT_TIMESTEPS)
    car_slots = [item for item in slots if item["mode"] == "standing_car"]
    assert [item["future_offset_s"] for item in car_slots] == [2.0, 1.5, 1.0, 0.5]


def test_ego_mask_overlap_policy_cli_defaults_to_auto() -> None:
    assert classifier.parse_args([]).ego_mask_overlap_policy == "auto"
    for policy in classifier.EGO_MASK_OVERLAP_POLICIES:
        parsed = classifier.parse_args(["--ego-mask-overlap-policy", policy])
        assert parsed.ego_mask_overlap_policy == policy


def test_direction_labels_retain_unambiguous_crossing_meaning() -> None:
    assert classifier.DIRECTION_SPECS["left"]["crossing_name"] == "right_to_left"
    assert classifier.DIRECTION_SPECS["right"]["crossing_name"] == "left_to_right"


def test_initial_direction_is_stable_and_selection_independent() -> None:
    first = classifier.initial_direction_choice("clip-a", 42)
    second = classifier.initial_direction_choice("clip-a", 42)
    assert first == second
    assert first[0] in classifier.DIRECTIONS
    assert 0.0 <= first[1] < 1.0


@patch.object(classifier, "initial_direction_choice", return_value=("left", 0.1))
def test_short_circuit_checks_cars_then_one_passing_direction(_choice: Mock) -> None:
    calls = []

    def car(timestep: int) -> dict:
        calls.append(("car", timestep))
        return passing_candidate()

    def crossing(direction: str, timestep: int) -> dict:
        calls.append((direction, timestep))
        return passing_candidate()

    result = classifier.classify_short_circuit(
        clip_id="clip-a",
        seed=42,
        timesteps=(20, 15, 10, 5),
        evaluate_car=car,
        evaluate_crossing=crossing,
        validate_shared=lambda _cars: {"passed": True, "failures": []},
    )

    assert result["classification"] == "left"
    assert calls == [
        ("car", 20),
        ("car", 15),
        ("car", 10),
        ("car", 5),
        ("left", 20),
        ("left", 15),
        ("left", 10),
        ("left", 5),
    ]
    right = [
        item
        for item in result["candidate_evaluations"]
        if item.get("direction") == "right"
    ]
    assert all(item["status"] == "not_evaluated_short_circuit" for item in right)


@patch.object(classifier, "initial_direction_choice", return_value=("right", 0.9))
def test_car_ground_failure_stops_all_later_checks(_choice: Mock) -> None:
    car_calls = []

    def car(timestep: int) -> dict:
        car_calls.append(timestep)
        if timestep == 15:
            return failing_candidate("ground_fit_failed")
        return passing_candidate()

    crossing = Mock(side_effect=AssertionError("crossing must not run"))
    result = classifier.classify_short_circuit(
        clip_id="clip-a",
        seed=42,
        timesteps=(20, 15, 10, 5),
        evaluate_car=car,
        evaluate_crossing=crossing,
    )

    assert car_calls == [20, 15]
    assert result["classification"] == "failure"
    assert result["final_failure_reasons"][0]["code"] == "ground_fit_failed"
    assert result["final_failure_reasons"][0]["future_offset_s"] == 1.5
    crossing.assert_not_called()


@patch.object(classifier, "initial_direction_choice", return_value=("left", 0.1))
def test_alternate_direction_can_recover_from_ground_failure(_choice: Mock) -> None:
    calls = []

    def crossing(direction: str, timestep: int) -> dict:
        calls.append((direction, timestep))
        if direction == "left" and timestep == 10:
            return failing_candidate("ground_fit_failed")
        return passing_candidate()

    result = classifier.classify_short_circuit(
        clip_id="clip-a",
        seed=42,
        timesteps=(20, 15, 10, 5),
        evaluate_car=lambda _timestep: passing_candidate(),
        evaluate_crossing=crossing,
        validate_shared=lambda _cars: {"passed": True, "failures": []},
    )

    assert result["classification"] == "right"
    assert calls == [
        ("left", 20),
        ("left", 15),
        ("left", 10),
        ("right", 20),
        ("right", 15),
        ("right", 10),
        ("right", 5),
    ]


def _candidate_fixture() -> tuple[Namespace, list[dict], dict, dict]:
    frames = [
        {
            "timestamp": 4_800_000 + 100_000 * index,
            "ego2global": np.eye(4),
            "lidar2global": np.eye(4),
            "map_location": "pai-av-local",
            "cams": {camera: {} for camera in classifier.CAMERA_NAMES},
        }
        for index in range(classifier.HISTORY_FRAME_COUNT)
    ]
    trajectory = {
        "actor_global_xys": np.zeros((4, 2)),
        "actor_current_xys": np.zeros((4, 2)),
        "anchor_global_xy": np.zeros(2),
        "reference_vehicle_forward_global_xy": np.array([1.0, 0.0]),
        "placement": {"future_offset_s": 2.0},
    }
    asset = {
        "actor_geometry": {
            "local_bounds_m": {
                "x": [-0.3, 0.3],
                "y": [-0.2, 0.2],
                "z": [0.0, 1.8],
            }
        }
    }
    args = Namespace(
        speed_mps=1.4,
        max_ground_tilt_deg=2.0,
        minimum_projected_bbox_area_px2=0.0,
    )
    return args, frames, trajectory, asset


def test_candidate_hard_fails_when_nearby_lidar_support_is_absent() -> None:
    args, frames, trajectory, asset = _candidate_fixture()
    failure = classifier.compositor.TokenFailure(
        "ground_fit_failed",
        "No history frame has accepted LiDAR ground support within 2.5 m",
    )
    with patch.object(
        classifier.compositor,
        "resolve_crossing_trajectory",
        return_value=trajectory,
    ), patch.object(
        classifier.paiav_edit,
        "paiav_sequence_consistent_ground",
        side_effect=failure,
    ) as grounding:
        result = classifier.evaluate_candidate(
            args=args,
            scene_frames=frames,
            frames=frames,
            timestamps_s=np.array([4.8, 4.9, 5.0, 5.1]),
            frame_lidar_points=[np.zeros((1, 3)) for _ in frames],
            global_to_current=np.eye(4),
            front_bumper_offset_m=4.0,
            ego_geometry={},
            asset=asset,
            animation=[24, 27, 30, 1],
            future_timestep=20,
            actor_mode="crossing",
            direction="left",
        )

    grounding.assert_called_once()
    assert not result["passed"]
    assert result["failures"][0]["code"] == "ground_fit_failed"
    assert "within 2.5 m" in result["failures"][0]["message"]


def test_strict_ground_fallback_is_a_classification_failure() -> None:
    args, frames, trajectory, asset = _candidate_fixture()
    ground_diagnostics = [
        {"sequence_support_frame_indices": [0], "frame_index": index}
        for index in range(4)
    ]
    strict_failures = [
        {
            "frame_index": 2,
            "code": "ground_fit_failed",
            "message": "strict fit used fallback",
        }
    ]
    clear_collision = {"occupied": False, "obstacle_points": 0}
    clear_sweep = {
        "occupied": False,
        "maximum_obstacle_points": 0,
        "collision_threshold_points": 5,
        "sample_count": 2,
    }
    with patch.object(
        classifier.compositor,
        "resolve_crossing_trajectory",
        return_value=trajectory,
    ), patch.object(
        classifier.paiav_edit,
        "paiav_sequence_consistent_ground",
        return_value=(
            [np.zeros(3) for _ in frames],
            ground_diagnostics,
            strict_failures,
        ),
    ), patch.object(
        classifier.compositor,
        "road_basis_aligned_to_global_forward",
        return_value=np.eye(3),
    ), patch.object(
        classifier.compositor,
        "frame_collision_diagnostics",
        return_value=clear_collision,
    ), patch.object(
        classifier.compositor,
        "trajectory_collision_diagnostics",
        return_value=clear_sweep,
    ), patch.object(
        classifier.compositor.render_utils,
        "actor_bbox_lidar",
        return_value=np.zeros((8, 3)),
    ), patch.object(
        classifier.compositor.geometry,
        "camera_frustum_mask",
        return_value=np.ones(8, dtype=bool),
    ):
        result = classifier.evaluate_candidate(
            args=args,
            scene_frames=frames,
            frames=frames,
            timestamps_s=np.array([4.8, 4.9, 5.0, 5.1]),
            frame_lidar_points=[np.zeros((1, 3)) for _ in frames],
            global_to_current=np.eye(4),
            front_bumper_offset_m=4.0,
            ego_geometry={},
            asset=asset,
            animation=[24, 27, 30, 1],
            future_timestep=20,
            actor_mode="crossing",
            direction="left",
        )

    assert not result["passed"]
    assert result["ground_support_frame_indices"] == [0]
    assert result["strict_ground_failure_frame_indices"] == [2]
    assert result["failures"][0]["code"] == "ground_fit_failed"


@pytest.mark.parametrize(
    ("diagnostic", "accepted"),
    [
        ({"method": "strict_same_timestamp_local_lidar", "local_support_count": 12}, True),
        ({"method": "strict_same_timestamp_local_lidar", "local_support_count": 11}, False),
        (
            {
                "method": "relaxed_same_timestamp_local_lidar",
                "local_support_count_2p5m": 1,
                "nearest_candidate_distance_m": 2.5,
            },
            True,
        ),
        (
            {
                "method": "relaxed_same_timestamp_local_lidar",
                "local_support_count_2p5m": 1,
                "nearest_candidate_distance_m": 2.5001,
            },
            False,
        ),
        (
            {
                "method": "nearest_accumulated_low_surface_horizontal_emergency",
                "nearest_distance_m": 2.5,
                "low_surface_count": 6,
            },
            True,
        ),
    ],
)
def test_current_paiav_nearby_ground_predicate(diagnostic: dict, accepted: bool) -> None:
    assert classifier.paiav_edit._ground_diagnostic_has_local_support(diagnostic) is accepted


def test_visibility_applies_production_minimum_projected_area() -> None:
    calibration = {"resolution": [100, 100]}
    bbox = np.zeros((8, 3))
    projected_uv = np.asarray(
        [[10.0, 10.0], [15.0, 15.0]] + [[12.0, 12.0]] * 6
    )
    with patch.object(
        classifier.compositor.geometry,
        "camera_frustum_mask",
        return_value=np.ones(8, dtype=bool),
    ), patch.object(
        classifier.compositor.geometry,
        "project_distorted",
        return_value=(projected_uv, np.ones(8)),
    ):
        rejected = classifier.actor_camera_visibility(bbox, calibration, 64.0)
        accepted = classifier.actor_camera_visibility(bbox, calibration, 25.0)

    assert rejected["projected_clipped_area_px2"] == 25.0
    assert not rejected["visible"]
    assert accepted["visible"]


def test_projected_bbox_ego_mask_overlap_uses_clipped_actor_envelope() -> None:
    protected = np.zeros((100, 100), dtype=bool)
    protected[45:56, 45:56] = True
    projected_uv = np.asarray(
        [
            [40.0, 40.0],
            [40.0, 60.0],
            [60.0, 40.0],
            [60.0, 60.0],
            [42.0, 42.0],
            [42.0, 58.0],
            [58.0, 42.0],
            [58.0, 58.0],
        ]
    )
    calibration = {
        "resolution": [100, 100],
        "ego_mask_image_sha256": "mask-digest",
        "ego_mask_protected_pixel_count": 121,
    }
    cache = {}
    with patch.object(
        classifier.compositor,
        "decode_calibrated_ego_mask",
        return_value=protected,
    ) as decode, patch.object(
        classifier.compositor.geometry,
        "project_distorted",
        return_value=(projected_uv, np.ones(8)),
    ):
        diagnostic = classifier.projected_bbox_ego_mask_overlap(
            np.zeros((8, 3)),
            calibration,
            frame_index=3,
            camera="camera_front_wide_120fov",
            mask_cache=cache,
        )
        classifier.projected_bbox_ego_mask_overlap(
            np.zeros((8, 3)),
            calibration,
            frame_index=3,
            camera="camera_front_wide_120fov",
            mask_cache=cache,
        )

    decode.assert_called_once()
    assert diagnostic["overlap_pixel_count"] == 121
    assert diagnostic["bbox_envelope_pixel_count"] > 121
    assert diagnostic["projected_bounds_px"] == [40.0, 40.0, 60.0, 60.0]


def test_ego_mask_overlap_report_and_fail_policies_are_distinct() -> None:
    summary = {
        "method": classifier.EGO_MASK_OVERLAP_METHOD,
        "overlap_detected": True,
        "overlap_view_count": 1,
        "total_overlap_pixel_count": 121,
        "maximum_overlap_pixel_count": 121,
        "overlap_views": [],
    }
    with patch.object(
        classifier,
        "candidate_ego_mask_overlap_diagnostics",
        side_effect=lambda *_args, **_kwargs: dict(summary),
    ) as evaluate:
        ignored = {"failures": []}
        assert classifier.apply_ego_mask_overlap_policy(
            ignored, [], [], policy="ignore"
        ) is None
        assert ignored == {"failures": []}

        reported = {"failures": []}
        classifier.apply_ego_mask_overlap_policy(
            reported, [], [], policy="report"
        )
        assert reported["ego_mask_overlap"]["policy"] == "report"
        assert reported["failures"] == []

        failed = {"failures": []}
        classifier.apply_ego_mask_overlap_policy(
            failed, [], [], policy="fail"
        )
        assert failed["ego_mask_overlap"]["policy"] == "fail"
        assert failed["failures"][0]["code"] == "ego_mask_overlap"
        compact = classifier.compact_candidate(
            {**failed, "evaluation_complete": True, "passed": False}
        )
        assert compact["ego_mask_overlap"]["maximum_overlap_pixel_count"] == 121
        assert compact["failures"][0]["ego_mask_overlap"]["policy"] == "fail"

    assert evaluate.call_count == 2


def test_ego_mask_overlap_policy_changes_checkpoint_fingerprint(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    selection = Mock(
        manifest_path=manifest,
        manifest_role="pai_av_dataset_manifest",
        catalog_index_path=None,
        catalog_index_sha256=None,
        clip_ids=("clip-a",),
    )
    selection.chunk_ids_for.return_value = (105,)

    ignore_args = classifier.parse_args(
        ["--ego-mask-overlap-policy", "ignore"]
    )
    ignore_args.data_root = tmp_path
    report_args = classifier.parse_args(
        ["--ego-mask-overlap-policy", "report"]
    )
    report_args.data_root = tmp_path
    ignored = classifier.build_configuration(
        args=ignore_args,
        selection=selection,
        tokens=("clip-a",),
        pedestrians=(),
        cars=(),
    )
    reported = classifier.build_configuration(
        args=report_args,
        selection=selection,
        tokens=("clip-a",),
        pedestrians=(),
        cars=(),
    )

    assert ignored["ego_mask_overlap"]["policy"] == "ignore"
    assert reported["ego_mask_overlap"]["policy"] == "report"
    assert (
        ignored["configuration_fingerprint"]
        != reported["configuration_fingerprint"]
    )


def test_checkpoint_recovers_a_partial_final_append() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = classifier.output_paths(root)
        configuration = {"configuration_fingerprint": "fingerprint"}
        assert classifier.prepare_checkpoint(
            paths=paths,
            configuration=configuration,
            allowed_tokens={"a", "b"},
            overwrite=False,
        ) == {}
        classifier.append_checkpoint(
            paths["checkpoint"],
            {
                "token": "a",
                "classification": "left",
                "configuration_fingerprint": "fingerprint",
            },
        )
        with paths["checkpoint"].open("ab") as handle:
            handle.write(b'{"token":"b"')

        resumed = classifier.recover_and_load_checkpoint(
            paths["checkpoint"],
            allowed_tokens={"a", "b"},
            configuration_fingerprint="fingerprint",
        )

        assert set(resumed) == {"a"}
        assert paths["checkpoint"].read_bytes().endswith(b"\n")




