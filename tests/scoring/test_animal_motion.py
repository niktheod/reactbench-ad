"""Inserted-animal motion contract and its raw (at-fault) scoring path."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reactbench.io import write_json
from reactbench.scoring import actor_motion as animal_motion
from reactbench.scoring.actor_motion import inserted_actor_motion
from reactbench.scoring.at_fault import classify_score
from reactbench.scoring.results import actor_box_state, inventory_result

GEOMETRY = {"x_min": -1., "x_max": 3., "y_min": -1., "y_max": 1.}  # ego rear axle at the origin


def camel_metadata(yaw_lidar=math.pi):
    """Compositor metadata for a standing side-on camel (head toward +Y for yaw pi)."""
    geometry = {
        "profile_version": "quadruped_measured_bounds_geometry_v1",
        "actor_type": "animal",
        "species": "camel",
        "asset_id": "rocketbox_camel_onehump_01",
        "source": "generated_asset_sidecar",
        "local_forward_axis": "-Y",
        "standing_only": True,
        "local_bounds_m": {"x": [-0.32, 0.32], "y": [-1.84, 1.40], "z": [0.0, 2.14]},
        "contact_profile": "quadruped_camel",
    }
    return {
        "status": "success",
        "token": "t",
        "actor_type": "animal",
        "pedestrian_behavior": "standing",
        "pedestrian_speed_mps": 0.0,
        "walking_velocity_global_xy_mps": [0.0, 0.0],
        "placement_mode": "future_timestep",
        "future_timestep": 4,
        "pedestrian_time_to_interception_s": 2.0,
        "anchor_current_ego_xy_m": [14.3, 1.7],
        "placement": {"placement_mode": "future_timestep",
                      "placement_version": "future_ego_front_bumper_stationary_obstacle_v1",
                      "future_timestep": 4, "standing_facing_lateral_sign": 1.0},
        "asset": {"id": "rocketbox_camel_onehump_01", "actor_type": "animal", "actor_geometry": geometry},
        "records": [{"camera": "CAM_F0", "frame_index": index, "actor_current": [14.3, 1.7, -0.3],
                     "yaw_lidar": yaw_lidar} for index in range(4)],
    }


class AnimalMotionTests(unittest.TestCase):
    def test_standing_camel_footprint_follows_measured_bounds(self):
        motion = inserted_actor_motion(camel_metadata())
        self.assertEqual(motion["actor_type"], "animal")
        self.assertEqual(motion["behavior"], "standing")
        self.assertEqual(motion["velocity_xy_mps"], [0.0, 0.0])
        # Local -Y is forward: the head side extends 1.84 m, the tail side 1.40 m.
        self.assertAlmostEqual(motion["front_length_m"], 1.84)
        self.assertAlmostEqual(motion["rear_length_m"], 1.40)
        self.assertAlmostEqual(motion["left_width_m"], 0.32)
        self.assertAlmostEqual(motion["right_width_m"], 0.32)
        self.assertAlmostEqual(motion["height_m"], 2.14)
        self.assertAlmostEqual(motion["heading_rad"], math.pi / 2)  # head toward the ego's left
        self.assertEqual(motion["future_timestep"], 4)
        self.assertEqual(motion["target_time_s"], 2.0)
        self.assertNotIn("geometry", motion)  # reserved for the measured pedestrian contract
        self.assertEqual(motion["animal_geometry"]["asset_id"], "rocketbox_camel_onehump_01")

    def test_box_is_offset_toward_the_head(self):
        motion = inserted_actor_motion(camel_metadata())
        state = actor_box_state(motion, [0.0, 0.0, 0.0], 0.0)
        np.testing.assert_allclose(state["pose"][:2], [14.3, 1.7 + 0.22], atol=1e-9)
        self.assertAlmostEqual(state["length"], 3.24)
        self.assertAlmostEqual(state["width"], 0.64)
        self.assertAlmostEqual(state["height"], 2.14)
        self.assertEqual(state["actor_type"], "animal")
        corners = animal_motion.actor_footprint_corners(motion, 3.0)
        np.testing.assert_allclose(corners[:, 1].min(), 1.7 - 1.40, atol=1e-9)
        np.testing.assert_allclose(corners[:, 1].max(), 1.7 + 1.84, atol=1e-9)

    def test_moving_ego_contact_with_the_camel_is_at_fault(self):
        motion = inserted_actor_motion(camel_metadata())
        poses = np.column_stack((2. * np.arange(1, 9) * .5, np.zeros(8), np.zeros(8)))
        physical = {"contacts": [{"scope": "inserted", "actor_id": "inserted", "time_s": 2.}]}
        score = classify_score(physical, poses, .5, actor_motion=motion, tracks=[],
                               ego_geometry=GEOMETRY, dataset="navsim", map_context=None)
        self.assertEqual((score["all"], score["inserted"]), (True, True))
        self.assertEqual([c["type"] for c in score["contacts"]], ["stopped_actor"])

    def test_pedestrians_and_cars_are_delegated_unchanged(self):
        metadata = {"actor_type": "car", "placement": {}}
        with patch.object(animal_motion, "actor_motion_from_metadata", return_value="sentinel") as original:
            self.assertEqual(inserted_actor_motion(metadata), "sentinel")
            self.assertEqual(inserted_actor_motion({}, camera_name=None, ego_heading_rad=.5), "sentinel")
        self.assertEqual(original.call_args_list[0].args, (metadata,))
        self.assertEqual(original.call_args_list[0].kwargs, {"camera_name": "CAM_F0", "ego_heading_rad": None})
        self.assertEqual(original.call_args_list[1].kwargs, {"camera_name": None, "ego_heading_rad": .5})

    def test_unsupported_animal_metadata_is_rejected(self):
        walking = camel_metadata()
        walking["pedestrian_speed_mps"] = 1.4
        dog_profile = camel_metadata()
        dog_profile["asset"]["actor_geometry"]["profile_version"] = "quadruped_dog_geometry_v1"
        wrong_asset = camel_metadata()
        wrong_asset["asset"]["id"] = "other"
        root_outside = camel_metadata()
        root_outside["asset"]["actor_geometry"]["local_bounds_m"]["y"] = [0.1, 3.0]
        for metadata, message in ((walking, "Only standing"), (dog_profile, "profile_version"),
                                  (wrong_asset, "asset_id"), (root_outside, "root must lie inside")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                inserted_actor_motion(metadata)


class RawInventoryTests(unittest.TestCase):
    def test_raw_inventory_parses_camel_metadata(self):
        with tempfile.TemporaryDirectory(prefix="test-animal-raw-", dir="/tmp") as tmp:
            root = Path(tmp)
            poses = np.column_stack((np.arange(1., 9.), np.zeros((8, 2))))
            for name in ("original", "edited"):
                path = root / name / "trajectories/t.npz"
                path.parent.mkdir(parents=True)
                np.savez(path, trajectory=poses, interval_length_s=.5)
            metadata = camel_metadata()
            metadata_path = root / "metadata/t.json"
            write_json(metadata_path, metadata)
            write_json(root / "original/evaluation_manifest.json", {"mode": "original", "selected_tokens": ["t"],
                       "signature": {"interval_length_s": .5, "evaluator": "recogdrive-navsim"}})
            write_json(root / "edited/metrics/summary.json", {"interval_length_s": .5,
                       "original_result_dir": str(root / "original")})
            saved = inserted_actor_motion(copy.deepcopy(metadata))
            write_json(root / "edited/metrics/per_sample.json",
                       [{"token": "t", "actor_motion": saved, "metadata_path": str(metadata_path)}])
            task = inventory_result(root / "edited")["tasks"][0]
        self.assertEqual(task["world"], "augmented")
        self.assertEqual(task["groups"]["actor_type"], "animal")
        self.assertEqual(task["groups"]["placement_timestep"], 4)
        self.assertEqual(task["actor_motion"]["animal_geometry"]["species"], "camel")
        self.assertAlmostEqual(task["actor_motion"]["front_length_m"], 1.84)


if __name__ == "__main__":
    unittest.main()
