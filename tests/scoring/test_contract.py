"""The output format of a user's model: predictions alone, without the runners' bookkeeping (README)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reactbench.io import write_json

MOTION = {"actor_type": "car", "initial_center_xy_m": [20., 0.], "velocity_xy_mps": [0., 0.],
          "heading_rad": 0., "length_m": 4., "width_m": 2.}


def predictions(root, dataset, run, dt):
    """<model>/<dataset>/{original,<run>}/trajectories/t.npz and a benchmark run with t's metadata."""
    poses = np.column_stack((np.arange(1, int(round(4 / dt)) + 1) * dt * 5, np.zeros((round(4 / dt), 2))))
    for name in ("original", run):
        path = root / "mymodel" / dataset / name / "trajectories" / "t.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, trajectory=poses, interval_length_s=dt)
    write_json(root / "benchmark" / run / "metadata" / "t.json", {"token": "t"})
    return root / "mymodel" / dataset / run, root / "benchmark" / run


class OutputFormatTests(unittest.TestCase):
    def test_reactnav_predictions_without_bookkeeping(self):
        from reactbench.scoring.navsim import inventory_result
        with tempfile.TemporaryDirectory(prefix="test-contract-") as tmp:
            run, bench = predictions(Path(tmp), "reactnav", "car_tau2.0s", .5)
            with patch("reactbench.scoring.actor_motion.inserted_actor_motion", return_value=MOTION):
                inventory = inventory_result(run, run_dir=bench)
            self.assertEqual(inventory["dataset"], "navsim")
            task, = inventory["tasks"]
            self.assertEqual(task["status"], "ready", task.get("reason"))
            self.assertEqual(task["pairs"], {"edited": ["original", "edited"]})
            self.assertEqual(Path(task["predictions"]["original"]["path"]),
                             (run.parent / "original" / "trajectories" / "t.npz").resolve())
            self.assertEqual(task["predictions"]["edited"]["interval_length_s"], .5)
            self.assertEqual(Path(task["metadata_source"]["path"]), (bench / "metadata" / "t.json").resolve())

    def test_reactpai_predictions_without_bookkeeping(self):
        from reactbench.scoring.paiav import inventory_result
        with tempfile.TemporaryDirectory(prefix="test-contract-") as tmp:
            run, bench = predictions(Path(tmp), "reactpai", "car_tau2.0s", .1)
            inventory = inventory_result(run, run_dir=bench)
            self.assertEqual(inventory["dataset"], "pai-av")
            self.assertEqual(Path(inventory["run_dir"]), bench.resolve())
            task, = inventory["tasks"]
            self.assertEqual(task["status"], "ready", task.get("reason"))
            self.assertEqual(task["pairs"], {"edited": ["original", "edited"]})

    def test_short_predictions_are_rejected(self):
        from reactbench.scoring.results import inventory_result
        with tempfile.TemporaryDirectory(prefix="test-contract-") as tmp:
            run, bench = predictions(Path(tmp), "reactnav", "car_tau2.0s", .5)
            np.savez(run / "trajectories" / "t.npz", trajectory=np.zeros((4, 3)), interval_length_s=.5)
            with patch("reactbench.scoring.actor_motion.inserted_actor_motion", return_value=MOTION):
                task, = inventory_result(run, run_dir=bench)["tasks"]
            self.assertIn("shorter than four seconds", task["predictions"]["edited"]["error"])


if __name__ == "__main__":
    unittest.main()
