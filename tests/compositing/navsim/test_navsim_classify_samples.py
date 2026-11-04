from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import classify_samples as classifier  # noqa: E402


def passing_candidate() -> dict:
    return {
        "evaluation_complete": True,
        "passed": True,
        "failures": [],
    }


def failing_candidate(code: str) -> dict:
    return {
        "evaluation_complete": True,
        "passed": False,
        "failures": [{"code": code, "message": f"failed with {code}"}],
    }


class DirectionSemanticsTest(unittest.TestCase):
    def test_public_labels_have_unambiguous_crossing_directions(self) -> None:
        self.assertEqual(
            classifier.multi.DIRECTION_BY_CLASSIFICATION["left"][
                "crossing_name"
            ],
            "right_to_left",
        )
        self.assertEqual(
            classifier.multi.DIRECTION_BY_CLASSIFICATION["right"][
                "crossing_name"
            ],
            "left_to_right",
        )

    def test_seeded_direction_is_stable(self) -> None:
        first = classifier.initial_direction_choice("token-a", 42)
        second = classifier.initial_direction_choice("token-a", 42)
        self.assertEqual(first, second)
        self.assertIn(first[0], classifier.DIRECTIONS)
        self.assertGreaterEqual(first[1], 0.0)
        self.assertLess(first[1], 1.0)

    def test_policy_result_does_not_depend_on_worker_count(self) -> None:
        def run_as_if_with_workers(_worker_count: int) -> dict:
            return classifier.classify_short_circuit(
                token="token-worker-independent",
                seed=17,
                evaluate_car=lambda _timestep: passing_candidate(),
                evaluate_crossing=lambda _direction, _timestep: passing_candidate(),
                validate_shared=lambda _cars: {"passed": True, "failures": []},
            )

        self.assertEqual(run_as_if_with_workers(1), run_as_if_with_workers(56))


class ShortCircuitTest(unittest.TestCase):
    def run_policy(self, car, crossing, shared=None):
        return classifier.classify_short_circuit(
            token="token-a",
            seed=42,
            evaluate_car=car,
            evaluate_crossing=crossing,
            validate_shared=shared,
        )

    @patch.object(classifier, "initial_direction_choice", return_value=("left", 0.1))
    def test_car_then_first_direction_passes_without_alternate(
        self, _choice: Mock
    ) -> None:
        calls = []

        def car(timestep):
            calls.append(("car", timestep))
            return passing_candidate()

        def crossing(direction, timestep):
            calls.append((direction, timestep))
            return passing_candidate()

        shared = Mock(return_value={"passed": True, "failures": []})
        result = self.run_policy(car, crossing, shared)

        self.assertEqual(result["classification"], "left")
        self.assertEqual(
            calls,
            [
                ("car", 4),
                ("car", 3),
                ("car", 2),
                ("car", 1),
                ("left", 4),
                ("left", 3),
                ("left", 2),
                ("left", 1),
            ],
        )
        shared.assert_called_once()
        right = [
            item
            for item in result["candidate_evaluations"]
            if item.get("direction") == "right"
        ]
        self.assertTrue(
            all(item["status"] == "not_evaluated_short_circuit" for item in right)
        )

    @patch.object(classifier, "initial_direction_choice", return_value=("right", 0.9))
    def test_car_failure_stops_every_later_check(self, _choice: Mock) -> None:
        car_calls = []

        def car(timestep):
            car_calls.append(timestep)
            if timestep == 3:
                return failing_candidate("vehicle_placement_occupied")
            return passing_candidate()

        crossing = Mock(side_effect=AssertionError("crossing must not run"))
        shared = Mock(side_effect=AssertionError("shared check must not run"))
        result = self.run_policy(car, crossing, shared)

        self.assertEqual(car_calls, [4, 3])
        self.assertEqual(result["classification"], "failure")
        self.assertEqual(result["classification_reason"], "standing_car_failed")
        self.assertEqual(result["final_failure_reasons"][0]["future_timestep"], 3)
        self.assertEqual(
            result["final_failure_reasons"][0]["code"],
            "vehicle_placement_occupied",
        )
        crossing.assert_not_called()
        shared.assert_not_called()

    @patch.object(classifier, "initial_direction_choice", return_value=("left", 0.1))
    def test_alternate_direction_is_used_only_after_first_fails(
        self, _choice: Mock
    ) -> None:
        calls = []

        def crossing(direction, timestep):
            calls.append((direction, timestep))
            if direction == "left" and timestep == 2:
                return failing_candidate("trajectory_collision")
            return passing_candidate()

        result = self.run_policy(
            lambda _timestep: passing_candidate(),
            crossing,
            lambda _cars: {"passed": True, "failures": []},
        )

        self.assertEqual(result["classification"], "right")
        self.assertEqual(
            calls,
            [
                ("left", 4),
                ("left", 3),
                ("left", 2),
                ("right", 4),
                ("right", 3),
                ("right", 2),
                ("right", 1),
            ],
        )
        failed_left = next(
            item
            for item in result["candidate_evaluations"]
            if item["candidate_id"] == "crossing_left_t2"
        )
        skipped_left = next(
            item
            for item in result["candidate_evaluations"]
            if item["candidate_id"] == "crossing_left_t1"
        )
        self.assertEqual(failed_left["failures"][0]["code"], "trajectory_collision")
        self.assertEqual(skipped_left["status"], "not_evaluated_short_circuit")
        self.assertEqual(skipped_left["skip_reason"], "direction_already_failed")

    @patch.object(classifier, "initial_direction_choice", return_value=("left", 0.1))
    def test_both_failed_directions_preserve_both_reasons(
        self, _choice: Mock
    ) -> None:
        def crossing(direction, timestep):
            if direction == "left" and timestep == 4:
                return failing_candidate("ground_fit_failed")
            if direction == "right" and timestep == 3:
                return failing_candidate("trajectory_collision")
            return passing_candidate()

        result = self.run_policy(
            lambda _timestep: passing_candidate(),
            crossing,
            lambda _cars: {"passed": True, "failures": []},
        )

        self.assertEqual(result["classification"], "failure")
        self.assertEqual(
            result["classification_reason"],
            "no_crossing_direction_passed_all_timesteps",
        )
        self.assertEqual(
            [item["code"] for item in result["final_failure_reasons"]],
            ["ground_fit_failed", "trajectory_collision"],
        )
        self.assertEqual(
            [item["direction"] for item in result["final_failure_reasons"]],
            ["left", "right"],
        )

    def test_shared_failure_stops_pedestrian_checks(self) -> None:
        crossing = Mock(side_effect=AssertionError("crossing must not run"))
        result = self.run_policy(
            lambda _timestep: passing_candidate(),
            crossing,
            lambda _cars: failing_candidate("unsupported_map_crs"),
        )

        self.assertEqual(result["classification"], "failure")
        self.assertEqual(result["classification_reason"], "shared_preparation_failed")
        self.assertEqual(
            result["final_failure_reasons"][0]["code"], "unsupported_map_crs"
        )
        crossing.assert_not_called()


class CombinedPolicyTest(unittest.TestCase):
    def run_policy(self, evaluate):
        decision = classifier.classify_short_circuit(
            token="combined-test", seed=42,
            evaluate_car=lambda _t: passing_candidate(),
            evaluate_crossing=lambda _d, _t: passing_candidate(),
        )
        return classifier.extend_with_benign_requirements(decision, evaluate)

    def test_preferred_sides_and_all_six_variants_are_required(self):
        calls = []

        def evaluate(mode, benign, side, timestep):
            calls.append((mode, benign, side, timestep))
            return passing_candidate()

        result = self.run_policy(evaluate)
        self.assertNotEqual(result["classification"], "failure")
        self.assertEqual(calls, [
            (mode, benign, side, t)
            for mode, benign, side in (("standing", False, None), ("car", True, "left"),
                                      ("standing", True, "right"), ("crossing", True, "right"))
            for t in classifier.TIMESTEPS
        ])
        self.assertEqual(len(result["selected_configurations"]), 6)
        self.assertEqual(sum(item.get("passed", False) for item in result["candidate_evaluations"]), 24)
        skipped = [item for item in result["candidate_evaluations"] if item["status"] != "evaluated"]
        self.assertTrue(all("passed" not in item for item in skipped))

    def test_fallback_restarts_all_timesteps_and_exports_no_partial_attempts(self):
        calls = []

        def evaluate(mode, benign, side, timestep):
            calls.append((mode, benign, side, timestep))
            preferred = classifier.fast.benign_placements.preferred_sides(mode)[0]
            return (failing_candidate("insufficient_front_visibility")
                    if benign and side == preferred and timestep == 2 else passing_candidate())

        result = self.run_policy(evaluate)
        self.assertNotEqual(result["classification"], "failure")
        for mode, side in (("car", "right"), ("standing", "left"), ("crossing", "left")):
            self.assertEqual(result["selected_configurations"][f"benign_{mode}"]["side"], side)
            self.assertEqual([t for m, b, s, t in calls if m == mode and b and s == side], list(classifier.TIMESTEPS))
        record = dict(result, token="combined-test", log_name="log", selected_assets={})
        exported = classifier.accepted_placement(record)
        self.assertEqual(len(exported["candidates"]), 24)
        self.assertEqual(len([c for c in exported["candidates"] if c.get("variant") == "benign_car"]), 4)
        self.assertEqual(classifier.accepted_benign_sides([
            record, dict(token="rejected", classification="failure"),
        ]), {"combined-test": {"car": "right", "standing": "left", "crossing": "left"}})

    def test_mixing_sides_between_timesteps_cannot_pass(self):
        def evaluate(mode, benign, side, timestep):
            if benign and mode == "car" and ((side == "left" and timestep == 2)
                                             or (side == "right" and timestep == 3)):
                return failing_candidate("benign_ego_clearance")
            return passing_candidate()

        result = self.run_policy(evaluate)
        self.assertEqual(result["classification"], "failure")
        self.assertEqual(result["selected_configurations"], {})
        self.assertEqual(len(result["final_failure_reasons"]), 2)
        self.assertTrue(all(item["status"] != "evaluated" for item in result["candidate_evaluations"]
                            if item.get("variant") == "benign_crossing"))

    def test_hazardous_standing_is_a_required_variant(self):
        result = self.run_policy(lambda mode, benign, side, t:
                                 passing_candidate() if benign else failing_candidate("ground_fit_failed"))
        self.assertEqual(result["classification"], "failure")
        self.assertEqual(result["classification_reason"], "hazardous_standing_failed_all_configurations")

    def test_input_failure_has_all_skipped_candidates(self):
        result = classifier.input_failure_decision("token", 42, ValueError("bad input"), True)
        self.assertEqual(len(result["candidate_evaluations"]), 40)
        self.assertTrue(all(c["status"] == "not_evaluated_short_circuit" for c in result["candidate_evaluations"]))

    def test_combined_default_and_legacy_visibility(self):
        self.assertEqual(classifier.parse_args([]).min_front_visible_fraction, 0.0)
        args = classifier.parse_args(["--include-benign-scenarios", "--splits", "navtrain"])
        self.assertEqual(args.min_front_visible_fraction, 0.25)
        self.assertEqual(args.splits, ["navtrain"])

    def test_policy_and_visibility_change_fingerprint_but_workers_do_not(self):
        args = classifier.parse_args(["--include-benign-scenarios"])
        def config():
            return classifier.build_configuration(
                args=args, split_name="navtrain", pedestrians=[], cars=[], ego_geometry={},
                filter_data=dict(tokens=["a"], path="filter", path_sha256="abc",
                                 source_token_count=1, log_names=["log"]),
            )
        first = config()
        args.workers = 2
        self.assertEqual(first, config())
        args.min_front_visible_fraction = .5
        self.assertNotEqual(first["configuration_fingerprint"], config()["configuration_fingerprint"])
        args.min_front_visible_fraction = .25
        with patch.object(classifier.fast.benign_placements, "SIDE_SELECTION_POLICY", "changed"):
            changed = config()
        self.assertNotEqual(first["configuration_fingerprint"], changed["configuration_fingerprint"])
        with tempfile.TemporaryDirectory() as directory:
            paths = classifier.output_paths(Path(directory), "navtrain")
            classifier.prepare_checkpoint(paths=paths, configuration=first, allowed_tokens={"a"}, overwrite=False)
            with self.assertRaisesRegex(classifier.ClassificationError, "another configuration"):
                classifier.prepare_checkpoint(paths=paths, configuration=changed, allowed_tokens={"a"}, overwrite=False)


class CacheAndCheckpointTest(unittest.TestCase):
    def test_lidar_cache_reuses_and_evicts_decoded_frames(self) -> None:
        loader = Mock(side_effect=lambda _root, _split, frame: np.array([[len(frame["lidar_path"]), 0, 0]]))
        cache = classifier.LidarCache(Path("/dataset"), "test", maximum=2)
        with patch.object(classifier.fast.geometry, "load_lidar_xyz", loader):
            first = cache.get({"lidar_path": "a"})
            self.assertIs(first, cache.get({"lidar_path": "a"}))
            cache.get({"lidar_path": "bb"})
            cache.get({"lidar_path": "ccc"})
            cache.get({"lidar_path": "a"})
        self.assertEqual(loader.call_count, 4)

    def test_checkpoint_resumes_and_recovers_trailing_partial_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = classifier.output_paths(root, "navtest")
            configuration = {"configuration_fingerprint": "fingerprint"}
            completed = classifier.prepare_checkpoint(
                paths=paths,
                configuration=configuration,
                allowed_tokens={"a", "b"},
                overwrite=False,
            )
            self.assertEqual(completed, {})
            record = {
                "token": "a",
                "classification": "left",
                "configuration_fingerprint": "fingerprint",
            }
            classifier.append_checkpoint_batch(paths["checkpoint"], [record])
            with paths["checkpoint"].open("ab") as handle:
                handle.write(b'{"token":"b"')

            resumed = classifier.recover_and_load_checkpoint(
                paths["checkpoint"],
                allowed_tokens={"a", "b"},
                configuration_fingerprint="fingerprint",
            )

            self.assertEqual(set(resumed), {"a"})
            self.assertTrue(paths["checkpoint"].read_bytes().endswith(b"\n"))

    def test_checkpoint_rejects_duplicate_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            record = {
                "token": "a",
                "classification": "failure",
                "configuration_fingerprint": "fingerprint",
            }
            path.write_text(
                classifier.json_line(record)
                + "\n"
                + classifier.json_line(record)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(classifier.ClassificationError, "duplicate"):
                classifier.recover_and_load_checkpoint(
                    path,
                    allowed_tokens={"a"},
                    configuration_fingerprint="fingerprint",
                )


class StationaryCarEvaluatorTest(unittest.TestCase):
    def test_vehicle_occupancy_is_a_hard_failure_using_asset_bounds(self) -> None:
        local_bounds = {
            "x": [-2.0, 2.0],
            "y": [-1.0, 1.0],
            "z": [0.0, 1.5],
        }
        asset = {
            "actor_geometry": {"local_bounds_m": local_bounds},
        }
        frames = [
            {
                "lidar2global": np.eye(4),
                "cams": {camera: {} for camera in classifier.fast.CAMERA_ORDER},
            }
            for _ in range(4)
        ]
        trajectory = {
            "actor_global_xys": np.zeros((4, 2)),
            "actor_current_xys": np.zeros((4, 2)),
            "anchor_global_xy": np.zeros(2),
            "reference_vehicle_forward_global_xy": np.array([1.0, 0.0]),
            "placement": {},
        }
        collisions = [
            {"occupied": False},
            {"occupied": False},
            {"occupied": True},
            {"occupied": False},
        ]
        with patch.object(
            classifier.fast,
            "resolve_stationary_vehicle_trajectory",
            return_value=trajectory,
        ), patch.object(
            classifier.fast,
            "fit_ground_sequence_best_effort",
            return_value=(
                [np.zeros(3) for _ in range(4)],
                [{} for _ in range(4)],
                [],
            ),
        ), patch.object(
            classifier.fast,
            "ground_point_lidar_for_global_xy",
            return_value=np.zeros(3),
        ), patch.object(
            classifier.fast,
            "road_basis_aligned_to_global_forward",
            return_value=np.eye(3),
        ), patch.object(
            classifier.fast,
            "stationary_vehicle_collision_diagnostics",
            side_effect=collisions,
        ) as collision_check, patch.object(
            classifier.multi,
            "actor_camera_visibility",
            return_value={
                "visible": True,
                "fully_visible": True,
                "inside_corner_count": 8,
                "corner_count": 8,
            },
        ):
            result = classifier.evaluate_stationary_car_candidate(
                args=Namespace(max_ground_tilt_deg=2.0),
                scene_frames=frames,
                frames=frames,
                timestamps_s=np.array([0.0, 0.5, 1.0, 1.5]),
                frame_lidar_points=[np.zeros((1, 3)) for _ in range(4)],
                global_to_current=np.eye(4),
                front_bumper_offset_m=4.0,
                ego_geometry={"rear_length_m": 1.1, "width_m": 2.3},
                asset=asset,
                future_timestep=4,
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["code"], "vehicle_placement_occupied")
        self.assertEqual(result["failures"][0]["occupied_frame_indices"], [2])
        self.assertEqual(collision_check.call_count, 4)
        self.assertTrue(
            all(call.args[2] is local_bounds for call in collision_check.call_args_list)
        )


class FilterAndShellTest(unittest.TestCase):
    def test_filter_subset_is_deterministic_and_keeps_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter.yaml"
            path.write_text(
                "num_history_frames: 4\n"
                "num_future_frames: 10\n"
                "frame_interval: 1\n"
                "has_route: true\n"
                "log_names: [log-a]\n"
                "tokens: [a, b, c, d]\n",
                encoding="utf-8",
            )
            first = classifier.load_filter(
                "navtest",
                path,
                seed=7,
                max_samples=2,
                allow_nonofficial_counts=True,
            )
            second = classifier.load_filter(
                "navtest",
                path,
                seed=7,
                max_samples=2,
                allow_nonofficial_counts=True,
            )
        self.assertEqual(first["tokens"], second["tokens"])
        self.assertEqual(
            first["tokens"],
            [token for token in ("a", "b", "c", "d") if token in first["tokens"]],
        )




if __name__ == "__main__":
    unittest.main()
