from __future__ import annotations

import unittest

from reactbench.scoring.motion_bins import project_longitudinal_acceleration, speed_bin_label, speed_bin_sort_key


class CanonicalMotionCalculationTest(unittest.TestCase):
    def test_fixed_width_labels_cover_low_high_and_boundary_values_without_merging(self):
        for value, expected in ((0, "[0,5)"), (4.999, "[0,5)"), (5, "[5,10)"),
                                (19.999, "[15,20)"), (20, "[20,25)"), (105, "[105,110)")):
            self.assertEqual(speed_bin_label(value), expected)
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                speed_bin_label(value)
        self.assertEqual(sorted(["[105,110)", "[5,10)", "[20,25)"], key=speed_bin_sort_key),
                         ["[5,10)", "[20,25)", "[105,110)"])
        with self.assertRaises(ValueError):
            speed_bin_sort_key("5-10")

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
