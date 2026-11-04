from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_multi_trajectory as multi  # noqa: E402


def candidate(
    timestep: int,
    direction: str,
    *,
    passed: bool,
    renderable: bool = True,
    evaluation_complete: bool = True,
    failures=None,
) -> dict:
    return {
        "candidate_id": f"t{timestep}_{direction}",
        "future_timestep": timestep,
        "direction": direction,
        "crossing_name": (
            "right_to_left" if direction == "left" else "left_to_right"
        ),
        "lateral_sign": 1.0 if direction == "left" else -1.0,
        "evaluation_complete": evaluation_complete,
        "renderable": renderable,
        "passed": passed,
        "failures": list(failures or []),
        "_states": [],
    }


class TimestepValidationTest(unittest.TestCase):
    def test_preserves_unique_positive_timesteps(self) -> None:
        self.assertEqual(
            multi.normalize_timesteps([1, 2, 4]),
            (1, 2, 4),
        )

    def test_rejects_empty_nonpositive_and_duplicate_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            multi.normalize_timesteps([])
        with self.assertRaisesRegex(ValueError, "positive"):
            multi.normalize_timesteps([0])
        with self.assertRaisesRegex(ValueError, "more than once"):
            multi.normalize_timesteps([1, 2, 1])


class ActorModeConfigurationTest(unittest.TestCase):
    def parse(self, *extra: str):
        argv = [
            "batch_composite_navsim_multi_trajectory.py",
            "--scene-filter",
            "filter.yaml",
            "--asset-source",
            "assets",
            "--run-name",
            "run",
            "--timesteps",
            "1",
            "2",
            "3",
            "4",
            *extra,
        ]
        with patch.object(sys, "argv", argv):
            return multi.parse_args()

    def test_actor_mode_is_explicit_and_crossing_is_default(self) -> None:
        self.assertEqual(self.parse().actor_mode, "crossing")
        self.assertEqual(
            self.parse("--actor-mode", "standing").actor_mode,
            "standing",
        )
        self.assertEqual(
            self.parse("--actor-mode", "car").actor_mode,
            "car",
        )

    def test_external_crossing_label_becomes_run_classification(
        self,
    ) -> None:
        args = Namespace(
            actor_mode="crossing",
            direction_classification_labels={"token": "right"},
        )
        result = multi.requested_mode_classification(args, "token")

        self.assertEqual(result["classification"], "right")
        self.assertEqual(result["direction_label"], "right")
        self.assertEqual(
            result["classification_reason"], "external_direction_label"
        )

    def test_all_four_timestep_candidates_form_one_complete_group(self) -> None:
        candidates = [
            candidate(timestep, "left", passed=True)
            for timestep in (1, 2, 3, 4)
        ]
        results = multi.build_timestep_results(
            candidates,
            (1, 2, 3, 4),
        )

        self.assertTrue(results["all_requested_timesteps_present"])
        self.assertTrue(results["all_requested_timesteps_renderable"])
        self.assertEqual(
            set(results["by_timestep"]),
            {"1", "2", "3", "4"},
        )

    def test_one_unrenderable_timestep_rejects_the_combined_group(self) -> None:
        candidates = [
            candidate(
                timestep,
                "left",
                passed=timestep != 3,
                renderable=timestep != 3,
            )
            for timestep in (1, 2, 3, 4)
        ]
        results = multi.build_timestep_results(
            candidates,
            (1, 2, 3, 4),
        )

        self.assertFalse(results["all_requested_timesteps_renderable"])


class PerformanceConfigurationTest(unittest.TestCase):
    def test_defaults_use_one_renderer_large_batches_and_bounded_finalizers(
        self,
    ) -> None:
        argv = [
            "batch_composite_navsim_multi_trajectory.py",
            "--scene-filter",
            "filter.yaml",
            "--asset-source",
            "assets",
            "--run-name",
            "run",
            "--timesteps",
            "1",
        ]
        with patch.object(sys, "argv", argv):
            args = multi.parse_args()

        self.assertEqual(args.render_workers, 1)
        self.assertEqual(args.render_batch_size, 96)
        self.assertEqual(args.finalize_workers, 4)
        self.assertEqual(args.speed_mps, 1.4)
        self.assertFalse(args.experimental_render_region)

    def test_actor_regions_are_unioned_conservatively(self) -> None:
        states = [
            {"bbox_lidar": multi.np.zeros((8, 3))},
            {"bbox_lidar": multi.np.ones((8, 3))},
        ]
        with patch.object(
            multi.fast.render_utils,
            "camera_to_lidar",
            return_value=multi.np.eye(4),
        ), patch.object(
            multi.fast,
            "actor_shadow_render_region",
            side_effect=(
                [10, 20, 40, 60],
                [30, 10, 80, 50],
            ),
        ) as actor_region:
            region = multi.multi_actor_render_region(
                states,
                {},
                multi.np.eye(3),
                (100, 100),
                padding_px=32,
            )

        self.assertEqual(region, [10, 10, 80, 60])
        self.assertEqual(actor_region.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["padding_px"] == 32
                for call in actor_region.call_args_list
            )
        )

    def test_actor_scale_applies_to_measured_bounds(self) -> None:
        self.assertEqual(
            multi.scaled_local_bounds(
                {
                    "x": [-0.31, 0.35],
                    "y": [-0.22, 0.27],
                    "z": [0.0, 1.81],
                },
                1.25,
            ),
            {
                "x": [-0.3875, 0.4375],
                "y": [-0.275, 0.3375],
                "z": [0.0, 2.2625],
            },
        )

    def test_finalization_starts_when_each_tokens_jobs_finish(self) -> None:
        prepared = [
            {"token": "a", "jobs": [{"job_id": "a1"}, {"job_id": "a2"}]},
            {"token": "b", "jobs": [{"job_id": "b1"}]},
            {"token": "cached", "jobs": []},
        ]
        required = {
            item["token"]: {job["job_id"] for job in item["jobs"]}
            for item in prepared
        }
        completed = set()
        finalized = []
        lock = threading.Lock()

        def fake_render(
            args,
            run_dir,
            jobs,
            on_batches_submitted=None,
            on_batch_complete=None,
        ):
            on_batches_submitted()
            for job_id in ("a1", "b1", "a2"):
                with lock:
                    completed.add(job_id)
                on_batch_complete({"job_ids": [job_id]})
            return []

        def fake_finalize(args, run_dir, item):
            with lock:
                self.assertTrue(
                    required[item["token"]].issubset(completed)
                )
                finalized.append(item["token"])
            return {"token": item["token"]}

        with patch.object(
            multi,
            "run_blender_batches",
            side_effect=fake_render,
        ), patch.object(
            multi,
            "finalize_prepared_token",
            side_effect=fake_finalize,
        ):
            metadata = multi.run_blender_with_streaming_finalization(
                Namespace(finalize_workers=2),
                Path("/unused"),
                prepared,
                [job for item in prepared for job in item["jobs"]],
            )

        self.assertEqual(set(finalized), {"a", "b", "cached"})
        self.assertEqual(set(metadata), {"a", "b", "cached"})


class CollisionSweepTest(unittest.TestCase):
    def test_unrendered_interception_segment_is_included(self) -> None:
        history = multi.np.array(
            [[0.0, 0.0], [0.0, 0.7], [0.0, 1.4]]
        )
        path = multi.collision_path_with_interception(
            history, [0.0, 2.1]
        )

        multi.np.testing.assert_array_equal(path[:-1], history)
        multi.np.testing.assert_array_equal(path[-1], [0.0, 2.1])
        self.assertEqual(path.shape, (4, 2))


class CameraSelectionTest(unittest.TestCase):
    ACTORS = ("a", "b", "c")

    def select(
        self,
        *,
        fully_visible=None,
        visible=None,
        **coverage,
    ):
        fully_visible = (
            coverage if fully_visible is None else fully_visible
        )
        visible = coverage if visible is None else visible
        return multi.select_cameras_for_actor_visibility(
            self.ACTORS,
            fully_visible,
            visible,
        )

    def test_only_complete_camera_is_selected(self) -> None:
        selection = self.select(
            CAM_L0=self.ACTORS,
            CAM_F0=("a", "b"),
            CAM_R0=("c",),
        )

        self.assertEqual(selection["selected_cameras"], ["CAM_L0"])
        self.assertEqual(selection["mode"], "single_camera_all_actors")
        self.assertEqual(selection["reason"], "only_complete_camera")

    def test_front_camera_wins_between_complete_views(self) -> None:
        selection = self.select(
            CAM_L0=self.ACTORS,
            CAM_F0=self.ACTORS,
            CAM_R0=self.ACTORS,
        )

        self.assertEqual(selection["selected_cameras"], ["CAM_F0"])
        self.assertEqual(
            selection["reason"],
            "front_camera_preferred_among_complete_views",
        )

    def test_best_two_camera_union_is_selected(self) -> None:
        selection = self.select(
            CAM_L0=("b",),
            CAM_F0=("a",),
            CAM_R0=("b", "c"),
        )

        self.assertEqual(
            selection["selected_cameras"],
            ["CAM_F0", "CAM_R0"],
        )
        self.assertTrue(
            selection["all_actors_fully_visible_by_selection"]
        )
        self.assertEqual(
            selection["actors_not_fully_visible_in_selection"],
            [],
        )

    def test_front_pair_wins_equal_union_coverage(self) -> None:
        selection = self.select(
            CAM_L0=("b",),
            CAM_F0=("a",),
            CAM_R0=("c",),
        )

        self.assertEqual(
            selection["selected_cameras"],
            ["CAM_F0", "CAM_L0"],
        )
        self.assertEqual(
            selection["mode"],
            "two_camera_maximum_full_containment_union",
        )
        self.assertFalse(
            selection["all_actors_fully_visible_by_selection"]
        )
        self.assertEqual(
            selection["actors_not_fully_visible_in_selection"],
            ["c"],
        )

    def test_second_view_recovers_actor_clipped_in_first(self) -> None:
        selection = self.select(
            fully_visible={
                "CAM_L0": (),
                "CAM_F0": ("a", "b"),
                "CAM_R0": ("c",),
            },
            visible={
                "CAM_L0": (),
                "CAM_F0": self.ACTORS,
                "CAM_R0": ("c",),
            },
        )

        self.assertEqual(
            selection["selected_cameras"],
            ["CAM_F0", "CAM_R0"],
        )
        self.assertTrue(
            selection["all_actors_fully_visible_by_selection"]
        )
        self.assertEqual(
            selection["fully_visible_actor_ids_by_camera"]["CAM_F0"],
            ["a", "b"],
        )

    def test_empty_second_view_is_not_selected(self) -> None:
        selection = self.select(
            fully_visible={
                "CAM_L0": (),
                "CAM_F0": ("a", "b"),
                "CAM_R0": (),
            },
            visible={
                "CAM_L0": (),
                "CAM_F0": self.ACTORS,
                "CAM_R0": (),
            },
        )

        self.assertEqual(selection["selected_cameras"], ["CAM_F0"])
        self.assertEqual(
            selection["mode"],
            "single_useful_camera_no_second_view",
        )
        self.assertEqual(
            selection["reason"],
            "no_other_camera_contains_any_actor",
        )

    def test_bbox_must_have_every_corner_inside_to_be_full(self) -> None:
        with patch.object(
            multi.fast.geometry,
            "camera_frustum_mask",
            return_value=multi.np.array(
                [True, True, True, True, True, True, True, False]
            ),
        ) as frustum_mask:
            visibility = multi.actor_camera_visibility(
                multi.np.zeros((8, 3)),
                {},
            )

        self.assertTrue(visibility["visible"])
        self.assertFalse(visibility["fully_visible"])
        self.assertEqual(visibility["inside_corner_count"], 7)
        frustum_mask.assert_called_once()
        self.assertEqual(frustum_mask.call_args.kwargs["margin"], 0.0)


class DirectionClassificationTest(unittest.TestCase):
    TIMESTEPS = (1, 2, 4)

    def aggregate(self, candidates):
        results = multi.build_direction_results(
            candidates, self.TIMESTEPS
        )
        return results, multi.classify_direction_results(
            results,
            token="test-token",
            seed=42,
        )

    def test_left_is_unique_all_timestep_winner(self) -> None:
        candidates = []
        for timestep in self.TIMESTEPS:
            candidates.append(
                candidate(timestep, "left", passed=True)
            )
            candidates.append(
                candidate(
                    timestep,
                    "right",
                    passed=timestep != 2,
                    failures=(
                        []
                        if timestep != 2
                        else [{"code": "trajectory_collision"}]
                    ),
                )
            )

        results, classification = self.aggregate(candidates)

        self.assertTrue(
            results["left"]["passes_all_requested_timesteps"]
        )
        self.assertFalse(
            results["right"]["passes_all_requested_timesteps"]
        )
        self.assertEqual(classification["classification"], "left")

    def test_right_is_unique_all_timestep_winner(self) -> None:
        candidates = []
        for timestep in self.TIMESTEPS:
            candidates.append(
                candidate(
                    timestep,
                    "left",
                    passed=timestep != 4,
                    failures=(
                        []
                        if timestep != 4
                        else [{"code": "ground_fit_failed"}]
                    ),
                )
            )
            candidates.append(
                candidate(timestep, "right", passed=True)
            )

        _results, classification = self.aggregate(candidates)

        self.assertEqual(classification["classification"], "right")

    def test_both_passing_uses_seeded_random_left_or_right(self) -> None:
        candidates = [
            candidate(timestep, direction, passed=True)
            for timestep in self.TIMESTEPS
            for direction in ("left", "right")
        ]

        _results, classification = self.aggregate(candidates)

        self.assertIn(classification["classification"], ("left", "right"))
        self.assertEqual(
            classification["classification_reason"],
            "both_directions_pass_seeded_random_choice",
        )
        self.assertTrue(
            classification["classification_tie_break"]["applied"]
        )
        self.assertEqual(
            classification["classification_tie_break"][
                "eligible_directions"
            ],
            ["left", "right"],
        )

    def test_both_passing_random_choice_covers_both_labels(self) -> None:
        results = {
            "left": {"passes_all_requested_timesteps": True},
            "right": {"passes_all_requested_timesteps": True},
        }

        with patch.object(
            multi.fast, "token_uniform", return_value=0.25
        ):
            left = multi.classify_direction_results(
                results,
                token="test-token",
                seed=42,
            )
        with patch.object(
            multi.fast, "token_uniform", return_value=0.75
        ):
            right = multi.classify_direction_results(
                results,
                token="test-token",
                seed=42,
            )

        self.assertEqual(left["classification"], "left")
        self.assertEqual(right["classification"], "right")

    def test_both_having_any_failure_is_tbd(self) -> None:
        candidates = [
            candidate(
                timestep,
                direction,
                passed=not (
                    (direction == "left" and timestep == 1)
                    or (direction == "right" and timestep == 4)
                ),
            )
            for timestep in self.TIMESTEPS
            for direction in ("left", "right")
        ]

        _results, classification = self.aggregate(candidates)

        self.assertEqual(classification["classification"], "TBD")
        self.assertEqual(
            classification["classification_reason"],
            "no_direction_passes_all_requested_timesteps",
        )

    def test_fallback_grounded_candidate_remains_failed(self) -> None:
        candidates = [
            candidate(timestep, "left", passed=True)
            for timestep in self.TIMESTEPS
        ]
        candidates.extend(
            candidate(
                timestep,
                "right",
                passed=timestep != 2,
                renderable=True,
                failures=(
                    []
                    if timestep != 2
                    else [{"code": "ground_fit_failed"}]
                ),
            )
            for timestep in self.TIMESTEPS
        )

        results, classification = self.aggregate(candidates)

        failed = results["right"]["by_timestep"]["2"]
        self.assertTrue(failed["renderable"])
        self.assertFalse(failed["passed"])
        self.assertEqual(
            failed["failures"][0]["code"], "ground_fit_failed"
        )
        self.assertEqual(classification["classification"], "left")

    def test_complete_sample_failure_overrides_direction_results(self) -> None:
        results = {
            "left": {"passes_all_requested_timesteps": True},
            "right": {"passes_all_requested_timesteps": False},
        }

        classification = multi.classify_direction_results(
            results,
            token="test-token",
            seed=42,
            sample_failed=True,
            failure_reason="no_edited_images_produced",
        )

        self.assertEqual(classification["classification"], "failure")
        self.assertEqual(
            classification["classification_reason"],
            "no_edited_images_produced",
        )

    def test_public_left_and_right_mean_walking_destination(self) -> None:
        self.assertEqual(
            multi.DIRECTION_BY_CLASSIFICATION["left"][
                "crossing_name"
            ],
            "right_to_left",
        )
        self.assertEqual(
            multi.DIRECTION_BY_CLASSIFICATION["right"][
                "crossing_name"
            ],
            "left_to_right",
        )


if __name__ == "__main__":
    unittest.main()
