from __future__ import annotations

import unittest

from reactbench.scoring.motion_bins import (
    acceleration_bin_label,
    observed_motion_labels,
    project_longitudinal_acceleration,
    speed_bin_label,
)


class CanonicalMotionCalculationTest(unittest.TestCase):
    def test_fixed_width_labels_cover_low_high_and_boundary_values_without_merging(self):
        for value, expected in ((0, "[0,5)"), (4.999, "[0,5)"), (5, "[5,10)"),
                                (19.999, "[15,20)"), (20, "[20,25)"), (105, "[105,110)")):
            self.assertEqual(speed_bin_label(value), expected)
        for value, expected in ((-4, "[-4,-3.5)"), (-.001, "[-0.5,0)"),
                                (0, "[0,0.5)"), (.5, "[0.5,1)"), (5, "[5,5.5)")):
            self.assertEqual(acceleration_bin_label(value), expected)
        self.assertIsNone(acceleration_bin_label(None))
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                speed_bin_label(value)
        rows = [{"groups": {"speed_bin_kmh": label}} for label in
                ("[105,110)", "[5,10)", None, "[20,30)", "[5,10)")]
        self.assertEqual(observed_motion_labels(rows, "speed_bin_kmh"),
                         ["[5,10)", "[20,30)", "[105,110)"])

    def test_projection_uses_shared_low_speed_threshold(self) -> None:
        self.assertIsNone(
            project_longitudinal_acceleration([0.01, 0.0], [1.0, 0.0])
        )
        self.assertAlmostEqual(
            project_longitudinal_acceleration([3.0, 4.0], [2.0, 1.0]),
            2.0,
        )

if __name__ == "__main__":
    unittest.main()
