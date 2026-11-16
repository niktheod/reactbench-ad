"""Summary tables added for the paper's appendix and section 5: plan motion, real-scenes control, side cameras."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reactbench.summarize import markdown, training


class PlanMotionTest(unittest.TestCase):
    def test_constant_speed(self):
        poses = np.array([[5. * (i + 1), 0., 0.] for i in range(10)])  # 10 m/s at 0.5 s, 5 s long
        distance, deceleration = training.plan_motion(poses, 0.5, 10.)
        self.assertAlmostEqual(distance, 40.)  # only the first 4 s
        self.assertEqual(deceleration, 0.)

    def test_braking_from_the_start_speed(self):
        # Mean speeds 8, 6, 4, ... m/s per 0.5 s interval after a measured start speed of 10 m/s.
        speeds = [8., 6., 4., 2., 0., 0., 0., 0.]
        xs = np.cumsum([0.5 * v for v in speeds])
        distance, deceleration = training.plan_motion(np.c_[xs, np.zeros(8), np.zeros(8)], 0.5, 10.)
        self.assertAlmostEqual(distance, xs[-1])
        self.assertAlmostEqual(deceleration, 8.)  # (10 - 8) / 0.25 s at the first interval centre


class RealSceneControlTest(unittest.TestCase):
    def test_records(self):
        per = {h: {"n": 308, "p_frozen": p} for h, p in (("0.5", 0.), ("1.0", 0.), ("1.5", 1.), ("2.0", 22.7))}
        summary = {"summary": {"recogdrive": per, "recogdrive_il": per},
                   "mean_over_headways": {"recogdrive": {"p_frozen": 5.9, "cs_kmh_frozen": 22.3},
                                          "recogdrive_il": {"p_frozen": 6.8, "cs_kmh_frozen": 21.4}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary))
            records = training.real_scene_control(path)
            self.assertEqual([r["model"] for r in records], ["recogdrive"])
            self.assertEqual(records[0]["p_pct"]["2.0"], 22.7)
            self.assertEqual([r["checkpoint"] for r in training.real_scenes(path)], ["recogdrive_il"])
        table = markdown.real_scene_control_table(records)
        self.assertIn("| ReCogDrive | 0.0 | 0.0 | 1.0 | 22.7 | 5.9 |", table)


class SideCameraTableTest(unittest.TestCase):
    def test_rows(self):
        cells = [{"model": m, "object": "car", "tau_s": t, "delta_p_pp": 1., "delta_p_ci95_pp": [-.1, 2.],
                  "delta_cs_kmh": -.5, "delta_cs_ci95_kmh": [-1., -.1]}
                 for m in ("autovla", "autovla_l0r0") for t in (0.5, 1.0)]
        tir = [{"model": m, "object": "car", "tir_0.05_pct": v} for m, v in (("autovla", 17.3), ("autovla_l0r0", 38.2))]
        table = markdown.side_camera_table({"cells": cells, "tir": tir})
        self.assertIn("| CAM_L1/R1 | ΔP | +1.00* | +1.00* |", table)
        self.assertIn("|  | ΔCS | -0.50 | -0.50 |", table)
        self.assertIn("|  | TIR | 38.2 |  |", table)


if __name__ == "__main__":
    unittest.main()
