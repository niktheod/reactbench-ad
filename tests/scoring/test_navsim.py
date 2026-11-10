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
    def test_original_scene_runs_are_inputs_not_scored(self):
        with tempfile.TemporaryDirectory(prefix="test-navsim-original-") as tmp:
            root = Path(tmp)
            (root / "trajectories").mkdir()
            np.savez(root / "trajectories/t.npz", trajectory=np.zeros((8, 3)), interval_length_s=.5)
            write_json(root / "evaluation_manifest.json", {"mode": "original", "selected_tokens": ["t"],
                                                           "signature": {"interval_length_s": .5}})
            summary = process_result(root, load_config(), backend=lambda *_: self.fail("original runs are not scored"))
            self.assertEqual(summary["status"], "inapplicable")
            self.assertFalse((root / "metrics").exists())

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
