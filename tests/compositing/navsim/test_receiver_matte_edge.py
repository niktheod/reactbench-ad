from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as compositor  # noqa: E402


class ReceiverMatteActorEdgeTest(unittest.TestCase):
    def test_soft_outer_edge_is_filled_without_changing_visible_receiver(self) -> None:
        amount = np.zeros((11, 11), dtype=np.float32)
        amount[5:, :] = 0.65
        alpha = np.zeros((11, 11), dtype=np.uint8)
        alpha[3:9, 3:8] = 96
        alpha[4:8, 4:7] = 255
        visibility = np.ones_like(amount)

        repaired, diagnostics = compositor.overlap_receiver_matte_with_actor_edge(
            amount,
            alpha,
            visibility,
            np.ones_like(amount),
        )

        actor = alpha > 0
        np.testing.assert_array_equal(repaired[~actor], amount[~actor])
        self.assertGreater(float(repaired[8, 3]), 0.0)
        self.assertEqual(float(repaired[6, 5]), 0.0)
        self.assertEqual(diagnostics["outside_actor_changed_pixels"], 0)
        self.assertEqual(diagnostics["opaque_or_interior_nonzero_pixels"], 0)

    def test_fill_is_receiver_capped_and_does_not_reach_translucent_interior(self) -> None:
        amount = np.full((11, 11), 0.8, dtype=np.float32)
        alpha = np.zeros((11, 11), dtype=np.uint8)
        alpha[1:10, 1:10] = 100
        visibility = np.ones_like(amount)
        visibility[1, 5] = 0.25

        repaired, _ = compositor.overlap_receiver_matte_with_actor_edge(
            amount,
            alpha,
            visibility,
            np.ones_like(amount),
        )

        self.assertAlmostEqual(float(repaired[1, 5]), 0.25, places=6)
        self.assertEqual(float(repaired[5, 5]), 0.0)

    def test_no_nearby_receiver_shadow_does_not_create_one(self) -> None:
        amount = np.zeros((9, 9), dtype=np.float32)
        amount[0, 0] = 0.7
        alpha = np.zeros((9, 9), dtype=np.uint8)
        alpha[4:7, 4:7] = 64

        repaired, diagnostics = compositor.overlap_receiver_matte_with_actor_edge(
            amount,
            alpha,
            np.ones_like(amount),
            np.ones_like(amount),
        )

        self.assertEqual(int(np.count_nonzero(repaired)), 1)
        self.assertEqual(diagnostics["filled_soft_edge_pixels"], 0)

    def test_measured_contact_region_excludes_airborne_actor_edge(self) -> None:
        amount = np.zeros((15, 15), dtype=np.float32)
        amount[10:, :] = 0.7
        alpha = np.zeros((15, 15), dtype=np.uint8)
        alpha[3:13, 4:11] = 96
        alpha[4:12, 5:10] = 255
        contact = np.zeros_like(amount)
        contact[9:, 3:12] = 1.0

        repaired, diagnostics = compositor.overlap_receiver_matte_with_actor_edge(
            amount,
            alpha,
            np.ones_like(amount),
            eligible_region=contact,
        )

        self.assertEqual(float(repaired[3, 4]), 0.0)
        self.assertGreater(float(repaired[12, 4]), 0.0)
        self.assertEqual(
            diagnostics["eligibility_method"], "measured_stance_foot_region"
        )

    def test_contact_raster_uses_only_feet_marked_in_contact(self) -> None:
        actor = np.zeros((80, 100), dtype=np.uint8)
        actor[10:70, 42:58] = 255
        render_info = {
            "foot_contact_geometry": {
                "method": "evaluated_foot_toe_vertex_groups_v1",
                "feet": {
                    "left": {
                        "contact": True,
                        "status": "measured_from_evaluated_foot_vertices",
                        "minimum_receiver_distance_m": 0.004,
                        "ideal_contact_points_px": [[43, 68], [49, 69], [46, 66]],
                    },
                    "right": {
                        "contact": False,
                        "status": "measured_swing_foot_excluded",
                        "minimum_receiver_distance_m": 0.08,
                        "ideal_contact_points_px": [[80, 30], [86, 31]],
                    },
                },
            }
        }

        contact, diagnostics = compositor.ideal_foot_contact_region(
            render_info,
            actor,
        )

        self.assertEqual(diagnostics["contact_sides"], ["left"])
        self.assertGreater(float(contact[68, 46]), 0.0)
        self.assertEqual(float(contact[30, 83]), 0.0)


if __name__ == "__main__":
    unittest.main()
