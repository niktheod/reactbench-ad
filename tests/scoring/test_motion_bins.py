from __future__ import annotations

import unittest

from reactbench.scoring.motion_bins import (
    NAVSIM_BIN_SPEC_ID,
    PAIAV_BIN_SPEC_ID,
    acceleration_bin_label,
    load_motion_bin_spec,
    occupied_speed_bin_groups,
    observed_motion_labels,
    project_longitudinal_acceleration,
    speed_bin_label,
)


class FrozenMotionBinSpecTest(unittest.TestCase):
    def test_loads_reviewed_dataset_artifacts(self) -> None:
        navsim = load_motion_bin_spec(NAVSIM_BIN_SPEC_ID)
        paiav = load_motion_bin_spec(PAIAV_BIN_SPEC_ID)

        self.assertEqual(navsim.dataset, "navsim")
        self.assertEqual(paiav.dataset, "pai-av")
        self.assertEqual(
            [item.label for item in navsim.speed_bins],
            [
                "[5,10)",
                "[10,15)",
                "[15,20)",
                "[20,25)",
                "[25,30)",
                "[30,35)",
                "[35,40)",
            ],
        )
        self.assertEqual(paiav.speed_bins[0].label, "[20,30)")
        self.assertEqual(paiav.speed_bins[-1].label, "[90,105)")
        self.assertEqual(
            sum(item.source_count for item in navsim.acceleration_bins), 3228
        )
        self.assertEqual(
            sum(item.source_count for item in paiav.acceleration_bins), 932
        )
        self.assertEqual(len(navsim.artifact_sha256), 64)

    def test_frozen_labels_use_merged_dataset_boundaries(self) -> None:
        self.assertEqual(
            speed_bin_label(5.0, bin_spec=NAVSIM_BIN_SPEC_ID), "[5,10)"
        )
        self.assertEqual(
            speed_bin_label(29.999, bin_spec=PAIAV_BIN_SPEC_ID), "[20,30)"
        )
        self.assertEqual(
            acceleration_bin_label(-2.2, bin_spec=NAVSIM_BIN_SPEC_ID),
            "[-2.5,-1.5)",
        )
        with self.assertRaisesRegex(ValueError, "outside frozen coverage"):
            speed_bin_label(4.99, bin_spec=NAVSIM_BIN_SPEC_ID)


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

    def test_dynamic_derivation_processes_lowest_sparse_bin_first(self) -> None:
        records = []
        for speed, count in ((2.0, 2), (7.0, 8), (12.0, 1), (17.0, 9)):
            records.extend({"speed_kmh": speed} for _ in range(count))

        groups, unbinned = occupied_speed_bin_groups(records)

        self.assertEqual(unbinned, 0)
        self.assertEqual(list(groups), ["[0,10)", "[10,20)"])
        self.assertEqual([len(group) for group in groups.values()], [10, 10])


if __name__ == "__main__":
    unittest.main()
