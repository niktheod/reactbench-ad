"""AutoVLA's ReactNAV inputs: the rendered cameras are replaced, its CAM_L1/R1 views stay original or become L0/R0."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from reactbench.models.autovla import eval as autovla

KEYS = ("front_camera_paths", "front_left_camera_paths", "front_right_camera_paths", "left_camera_paths",
        "right_camera_paths")


def sample():
    return {key: [f"original/{key}/{i}.jpg" for i in range(4)] for key in KEYS}


LOADER = SimpleNamespace(edited_image=lambda token, camera, frame: f"edited/{camera}/{frame}.png")


class EditedInputTest(unittest.TestCase):
    def test_side_cameras_stay_original(self):
        edited = autovla.edited_input(sample(), LOADER, "t")
        self.assertEqual(edited["front_camera_paths"], [f"edited/CAM_F0/{i}.png" for i in range(4)])
        self.assertEqual(edited["front_left_camera_paths"], [f"edited/CAM_L0/{i}.png" for i in range(4)])
        for key in ("left_camera_paths", "right_camera_paths"):
            self.assertEqual(edited[key], sample()[key])

    def test_l0r0_feeds_the_rendered_views(self):
        original = autovla.with_side_cameras(sample(), "l0r0")
        self.assertEqual(original["left_camera_paths"], sample()["front_left_camera_paths"])
        edited = autovla.with_side_cameras(autovla.edited_input(sample(), LOADER, "t"), "l0r0")
        self.assertEqual(edited["right_camera_paths"], [f"edited/CAM_R0/{i}.png" for i in range(4)])
        self.assertIs(autovla.with_side_cameras(original, "l1r1"), original)
        with self.assertRaises(ValueError):
            autovla.with_side_cameras(sample(), "f0")


class ExclusionTest(unittest.TestCase):
    def test_released_list(self):
        counts = {f"crossing_pedestrian_timestep_{k}": len(autovla.side_camera_exclusions(
            f"crossing_pedestrian_timestep_{k}")) for k in (1, 2, 3, 4)}
        self.assertEqual(counts, {"crossing_pedestrian_timestep_1": 0, "crossing_pedestrian_timestep_2": 1,
                                  "crossing_pedestrian_timestep_3": 4, "crossing_pedestrian_timestep_4": 9})
        self.assertEqual(autovla.side_camera_exclusions("crossing_pedestrian_timestep_4_contact_only"),
                         autovla.side_camera_exclusions("crossing_pedestrian_timestep_4"))
        self.assertEqual(autovla.side_camera_exclusions("car_timestep_4"), {})
        self.assertEqual(autovla.side_camera_exclusions("original"), {})

    def test_file_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fov.json"
            path.write_text(json.dumps({"runs": {"standing_pedestrian_timestep_2": ["a", "b"]}}))
            self.assertEqual(sorted(autovla.side_camera_exclusions("standing_pedestrian_timestep_2", path)),
                             ["a", "b"])


if __name__ == "__main__":
    unittest.main()
