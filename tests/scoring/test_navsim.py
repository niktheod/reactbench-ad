from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reactbench.io import file_digest, read_json, write_json
from reactbench.scoring.collisions import paired_outcome
from reactbench.scoring.navsim import load_config, process_result
from reactbench.scoring.reporting import score_value


def scored_row(token="sample"):
    """A scored augmented-world row: the edited plan collides at a lower speed than the original."""
    a = {"all": True, "inserted": True, "background": False, "dac": 1,
         "first_contact": {"time_s": 1., "ego_speed_mps": 10., "actors": [{"scope": "inserted", "actor_id": "inserted"}]}}
    b = {**a, "first_contact": {**a["first_contact"], "ego_speed_mps": 5.}}
    for score in (a, b):
        for key in ("nc", "nc_inserted", "nc_dac"):
            score[key] = score_value(score, key)
    return {"token": token, "world": "augmented", "status": "scored", "world_fingerprint": "scene", "groups": {},
            "scores": {"original": a, "edited": b}, "primary_pair": "edited",
            "pairs": {"edited": {"from_condition": "original", "to_condition": "edited", **paired_outcome(a, b)}}}


class NavsimScoringTests(unittest.TestCase):
    def test_original_clean_scene_scoring_summary_resume_and_missing_prediction(self):
        from reactbench.scoring.navsim import score_task
        from tests.scoring.test_navsim_metrics import world
        from tests.scoring.test_collisions import track
        from shapely.geometry import box
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory(prefix="test-navsim-original-") as tmp:
            root = Path(tmp)
            cfg = load_config(overrides={"work_root": str(root / "work")})
            (root / "trajectories").mkdir()
            tokens = ["clear", "collision", "offroad"]
            for token in tokens:
                # A stopped ego is never at fault, so the colliding prediction drives into a parked car.
                poses = np.column_stack((5 * np.arange(1, 9) * .5, np.zeros((8, 2)))) if token == "collision" else np.zeros((8, 3))
                np.savez(root / f"trajectories/{token}.npz", trajectory=poses, interval_length_s=.5)
            manifest = {"mode": "original", "selected_tokens": tokens,
                        "signature": {"evaluator": "autovla-navsim", "interval_length_s": .5}}
            write_json(root / "evaluation_manifest.json", manifest)

            def load_world(token, radius):
                value = world(track(x=10, y=0)) if token == "collision" else world()
                if token == "offroad":
                    value["map_polygons_wkb"] = [box(20, 20, 30, 30).wkb_hex]
                return {**value, "source": {"token": token}, "contract": {},
                        "initial_velocity": [0, 0], "initial_acceleration": [0, 0]}

            def backend(inventory, config):
                self.assertEqual(inventory["mode"], "original")
                for task in inventory["tasks"]:
                    self.assertEqual(task["world"], "clean")
                    self.assertIsNone(task["actor_motion"])
                    self.assertEqual(task["pairs"], {})
                rows = [score_task(t, SimpleNamespace(load=load_world), config, "test")
                        for t in inventory["tasks"]]
                for row in rows:
                    if row["token"] != "missing":
                        self.assertEqual(row["status"], "scored", row)
                return {"rows": rows}

            summary = process_result(root, cfg, backend=backend)
            overall = summary["overall"]
            self.assertEqual((overall["valid_count"], overall["excluded_count"]), (3, 0))
            self.assertAlmostEqual(overall["mean_nc"], 2 / 3)
            self.assertAlmostEqual(overall["mean_dac"], 2 / 3)
            self.assertAlmostEqual(overall["mean_nc_dac"], 1 / 3)
            self.assertEqual(read_json(root / "metrics/navsim_at_fault_summary.json"), summary)
            process_result(root, cfg, backend=lambda *_: self.fail("completed original run must resume"))

            manifest["selected_tokens"].append("missing")
            write_json(root / "evaluation_manifest.json", manifest)
            overall = process_result(root, cfg, backend=backend)["overall"]
            self.assertEqual((overall["sample_count"], overall["valid_count"], overall["excluded_count"]), (4, 3, 1))
            self.assertAlmostEqual(overall["mean_nc_dac"], 1 / 3)
            self.assertEqual(overall["coverage"], {"scored": 3, "failed": 1})

    def test_saved_rollouts_pair_by_action_token_across_changed_ranks(self):
        from reactbench.scoring.results import inventory_result, load_prediction
        with tempfile.TemporaryDirectory(prefix="test-navsim-branches-") as tmp:
            root = Path(tmp)
            poses = np.zeros((2, 8, 3))
            for name, ids in (("original", [100, 200]), ("edited", [200, 300])):
                path = root / name / "rollouts/t.npz"
                path.parent.mkdir(parents=True)
                np.savez(path, trajectories=poses, interval_length_s=.5, candidate_ranks=[1, 2], candidate_token_ids=ids)
            write_json(root / "edited/metrics/summary.json", {"interval_length_s": .5, "original_result_dir": str(root / "original")})
            write_json(root / "edited/metrics/per_sample.json", [{"token": "t", "actor_motion": {
                "actor_type": "car", "initial_center_xy_m": [20., 0.], "velocity_xy_mps": [0., 0.],
                "heading_rad": 0., "length_m": 4., "width_m": 2.}}])
            task = inventory_result(root / "edited")["tasks"][0]
            self.assertEqual(task["pairs"], {"edited/rank_1": ["original/rank_2", "edited/rank_1"]})
            self.assertIsNone(task["primary_condition"])
            source = task["predictions"]["edited/rank_1"]
            load_prediction(source)
            Path(source["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since inventory"):
                load_prediction(source)

    def test_invalid_inventory_is_not_reported_as_inapplicable(self):
        inventory = {"dataset": "navsim", "mode": "paired", "tasks": [{"status": "invalid"}]}
        with patch("reactbench.scoring.navsim.inventory_result", return_value=inventory):
            with self.assertRaisesRegex(ValueError, "No scoreable"):
                process_result(Path("/tmp"), load_config(), backend=lambda *_: self.fail("invalid inventory"))

    def test_scoring_publishes_metrics_resumes_and_keeps_scores_on_failure(self):
        with tempfile.TemporaryDirectory(prefix="test-navsim-results-") as tmp:
            root = Path(tmp)
            cfg = load_config(overrides={"work_root": str(root / "work")})
            inventory = {"result_dir": str(root), "dataset": "navsim", "mode": "paired", "sources": {},
                         "tasks": [{"status": "ready"}]}
            with patch("reactbench.scoring.navsim.inventory_result", return_value=inventory):
                value = process_result(root, cfg, backend=lambda *_: {"rows": [scored_row()]})
                self.assertEqual(value["overall"]["paired_count"], 1)
                self.assertTrue((root / "metrics/navsim_at_fault_manifest.json").is_file())
                with patch("reactbench.scoring.navsim.backend_call", side_effect=AssertionError("no rescoring")):
                    process_result(root, cfg, backend=lambda *_: self.fail("no rescoring"))
                original = file_digest(root / "metrics/navsim_at_fault_per_sample.json")
                inventory["tasks"].append({"status": "invalid"})
                with self.assertRaisesRegex(RuntimeError, "No predictions could be scored"):
                    process_result(root, cfg, backend=lambda *_: {"rows": [{"status": "failed", "scores": {}}]})
                self.assertEqual(file_digest(root / "metrics/navsim_at_fault_per_sample.json"), original)

    def test_output_root_keeps_the_run_untouched(self):
        with tempfile.TemporaryDirectory(prefix="test-navsim-output-") as tmp:
            root, output = Path(tmp) / "run", Path(tmp) / "scored"
            root.mkdir()
            cfg = load_config(overrides={"work_root": str(Path(tmp) / "work")})
            inventory = {"result_dir": str(root), "dataset": "navsim", "mode": "paired", "sources": {},
                         "tasks": [{"status": "ready"}]}
            with patch("reactbench.scoring.navsim.inventory_result", return_value=inventory):
                process_result(root, cfg, output_root=output, backend=lambda *_: {"rows": [scored_row()]})
            self.assertEqual(list(root.iterdir()), [])
            self.assertTrue((output / "metrics/navsim_at_fault_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
