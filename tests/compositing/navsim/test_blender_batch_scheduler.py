from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as compositor  # noqa: E402


class BlenderBatchSchedulerTest(unittest.TestCase):

    def test_multiple_gpu_workers_require_matching_visible_devices(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "exposes only 1 device"):
                compositor.render_worker_devices("gpu", 2)

    def test_single_worker_preserves_non_slurm_behavior(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(compositor.render_worker_devices("gpu", 1), [None])


if __name__ == "__main__":
    unittest.main()
