"""ADE over four seconds and the trajectory invariance rate (TIR) threshold semantics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reactbench.io import file_digest
from reactbench.scoring.ade import paired_ade, statistics


class AdeThresholdTests(unittest.TestCase):
    def test_strict_boundaries_and_invalid_denominator(self):
        result = statistics([0, .05, .1, .2, .5, None, float("nan"), -1, True])
        self.assertEqual(result["valid_count"], 5)
        self.assertEqual(result["excluded_count"], 4)
        self.assertEqual([r["count"] for r in result["thresholds"]], [1, 2, 3, 4])
        self.assertEqual([r["percentage"] for r in result["thresholds"]], [20, 40, 60, 80])
        self.assertIsNone(statistics([])["thresholds"][0]["percentage"])

    def test_paired_ade_uses_only_four_seconds_and_rejects_short_trajectories(self):
        with tempfile.TemporaryDirectory(prefix="test-ade-") as tmp:
            root = Path(tmp)
            sources = {}
            for name, offset in (("original", 0), ("edited", .03)):
                poses = np.zeros((10, 3))
                poses[:8, 0] = offset
                poses[8:, 0] = offset * 1000  # Must not use the final extra second.
                path = root / f"{name}.npz"
                np.savez(path, trajectory=poses, interval_length_s=.5)
                sources[name] = {"path": str(path), "sha256": file_digest(path), "interval_length_s": .5}
            self.assertAlmostEqual(paired_ade(sources), .03)
            np.savez(root / "edited.npz", trajectory=np.zeros((7, 3)), interval_length_s=.5)
            sources["edited"]["sha256"] = file_digest(root / "edited.npz")
            with self.assertRaisesRegex(ValueError, "four-second"):
                paired_ade(sources)


if __name__ == "__main__":
    unittest.main()
