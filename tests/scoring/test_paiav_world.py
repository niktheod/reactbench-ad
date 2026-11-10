from __future__ import annotations

import tempfile
import io
import zipfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from reactbench.scoring.collisions import score_trajectory
from reactbench.scoring.paiav_world import BackgroundWorlds, build_background_world, pose_matrices
from reactbench.scoring.paiav_world import (
    ANNOTATION_TRACK_EXCLUSIONS, annotation_track_exclusions, apply_annotation_track_exclusions,
)
from reactbench.io import read_json, write_json


def fixtures():
    times = np.arange(61) * 100_000
    yaw = times / 1e6 * .1
    xyz = np.column_stack((times / 1e6 * 2, np.zeros((len(times), 2))))
    q = Rotation.from_euler("z", yaw).as_quat()
    ego = pd.DataFrame(np.column_stack((times, xyz, q)), columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
    # Distinct box and reference times: the transform must use reference time.
    refs = times[1:-1] - 50_000
    transforms = pose_matrices(ego, refs)
    position = np.array([20., 3., 1.])
    centers = np.einsum("nji,nj->ni", transforms[:, :3, :3], position - transforms[:, :3, 3])
    orientations = Rotation.from_matrix(transforms[:, :3, :3].transpose(0, 2, 1)).as_quat()
    obstacles = pd.DataFrame({"timestamp_us": times[1:-1], "reference_frame_timestamp_us": refs,
        "track_id": "a", "label_class": "automobile", "source": "scene:obstacles:autolabels:v2", "reference_frame": "rig",
        "center_x": centers[:, 0], "center_y": centers[:, 1], "center_z": centers[:, 2],
        "size_x": 4., "size_y": 2., "size_z": 1.5,
        **{f"orientation_{axis}": orientations[:, i] for i, axis in enumerate("xyzw")}})
    return obstacles, ego


class CollisionWorldTests(unittest.TestCase):
    def test_self_annotation_exclusion_is_exact_and_preserves_other_tracks(self):
        entry = ANNOTATION_TRACK_EXCLUSIONS[0]
        self.assertEqual(annotation_track_exclusions(entry), [entry])
        for key in ("repo_id", "revision", "token", "obstacle_member_sha256"):
            with self.subTest(key=key):
                self.assertEqual(annotation_track_exclusions({**entry, key: "different"}), [])
        obstacles, ego = fixtures()
        self_track = obstacles.assign(track_id="250")
        other = obstacles.assign(track_id="251")
        value = build_background_world(pd.concat([self_track, other]), ego, 1_000_000)
        original_other = value["tracks"][1]
        apply_annotation_track_exclusions(value, [entry])
        self.assertEqual(value["tracks"], [original_other])
        self.assertEqual(value["coverage"]["track_count"], 1)
        self.assertEqual(value["coverage"]["excluded_annotation_track_count"], 1)
        self.assertEqual(value["coverage"]["annotation_track_exclusions"], [entry])
        value = build_background_world(self_track, ego, 1_000_000)
        apply_annotation_track_exclusions(value, [])
        self.assertEqual(value["tracks"][0]["actor_id"], "250")

    def test_moving_rotating_rig_and_distinct_reference_timestamps(self):
        obstacles, ego = fixtures()
        value = build_background_world(obstacles, ego, 1_000_000)
        track = value["tracks"][0]
        initial = pose_matrices(ego, [1_000_000])[0]
        expected = np.linalg.inv(initial) @ [20., 3., 1., 1.]
        np.testing.assert_allclose(np.asarray(track["poses"])[:, :2], np.tile(expected[:2], (len(track["poses"]), 1)), atol=1e-12)
        np.testing.assert_allclose(np.asarray(track["poses"])[:, 2], -.1, atol=1e-12)
        self.assertEqual(value["coverage"]["partial_lifetime_track_count"], 0)

    def test_gaps_duplicates_missing_columns_and_invalid_geometry_fail(self):
        obstacles, ego = fixtures()
        variants = [pd.concat([obstacles, obstacles.iloc[[15]]]),
                    obstacles.drop(index=range(20, 25)),
                    obstacles.drop(columns="size_x"), obstacles.iloc[:0].drop(columns="size_x")]
        for column, value in (("reference_frame", "world"), ("source", "unknown"), ("size_x", 0), ("orientation_w", 0), ("center_x", np.nan)):
            changed = obstacles.copy()
            changed.loc[15, column] = value
            variants.append(changed)
        for frame in variants:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                build_background_world(frame, ego, 1_000_000)

    def test_empty_annotations_and_windows_are_valid_backgrounds(self):
        obstacles, ego = fixtures()
        for frame in (obstacles.iloc[:0], obstacles[obstacles.timestamp_us <= 500_000],
                      obstacles[obstacles.timestamp_us >= 5_500_000],
                      obstacles[(obstacles.timestamp_us <= 500_000) |
                                (obstacles.timestamp_us >= 5_500_000)]):
            with self.subTest(timestamps=frame.timestamp_us.tolist()):
                value = build_background_world(frame, ego, 1_000_000)
                self.assertEqual(value["tracks"], [])
                self.assertEqual(value["coverage"]["track_count"], 0)
                self.assertEqual(value["coverage"]["partial_lifetime_track_count"], 0)

    def test_empty_background_still_scores_the_inserted_object(self):
        obstacles, ego = fixtures()
        world = build_background_world(obstacles.iloc[:0], ego, 1_000_000)
        geometry = {"x_min": -1., "x_max": 3., "y_min": -1., "y_max": 1.}
        for x, collision in ((0., True), (100., False)):
            with self.subTest(inserted_x=x):
                actor = {"actor_type": "car", "initial_center_xy_m": [x, 0.],
                         "velocity_xy_mps": [0., 0.], "heading_rad": 0.,
                         "length_m": 2., "width_m": 2.}
                score = score_trajectory(np.zeros((40, 3)), actor, geometry, world)
                self.assertFalse(score["background"])
                self.assertEqual(score["inserted"], collision)
                self.assertEqual(score["all"], collision)

    def test_nonfinite_timestamps_cannot_be_treated_as_an_empty_window(self):
        obstacles, ego = fixtures()
        for timestamp in (np.nan, np.inf, -np.inf):
            with self.subTest(timestamp=timestamp):
                frame = obstacles.assign(timestamp_us=timestamp)
                with self.assertRaisesRegex(ValueError, "Nonfinite obstacle timestamps"):
                    build_background_world(frame, ego, 1_000_000)

    def test_empty_annotations_still_require_valid_egomotion(self):
        obstacles, ego = fixtures()
        ego.loc[15, "x"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite offline egomotion"):
            build_background_world(obstacles.iloc[:0], ego, 1_000_000)

    def test_first_and_last_objects_need_not_span_the_window(self):
        obstacles, ego = fixtures()
        for start, end in ((2_000_000, 5_200_000), (800_000, 3_000_000),
                           (2_000_000, 3_000_000)):
            with self.subTest(start=start, end=end):
                frame = obstacles[obstacles.timestamp_us.between(start, end)]
                value = build_background_world(frame, ego, 1_000_000)
                self.assertEqual(value["coverage"]["track_count"], 1)
                self.assertEqual(value["coverage"]["partial_lifetime_track_count"], 1)
                track = value["tracks"][0]
                self.assertAlmostEqual(track["times_s"][0], (start - 1_000_000) / 1e6)
                self.assertAlmostEqual(track["times_s"][-1], (end - 1_000_000) / 1e6)

    def test_partial_lifetimes_are_disclosed_without_extrapolation(self):
        obstacles, ego = fixtures()
        partial = obstacles.iloc[20:30].copy()
        partial["track_id"] = "b"
        value = build_background_world(pd.concat([obstacles, partial]), ego, 1_000_000)
        self.assertEqual(value["coverage"]["partial_lifetime_track_count"], 1)
        self.assertAlmostEqual(value["tracks"][1]["times_s"][0], 1.1)

    def test_world_cache_checks_payload_and_input_members(self):
        with tempfile.TemporaryDirectory(prefix="test-paiav-world-cache-", dir="/tmp") as scratch:
            root = Path(scratch)
            obstacles, ego = fixtures()
            def archive(path, member, frame):
                data = io.BytesIO()
                frame.to_parquet(data)
                with zipfile.ZipFile(path, "w") as value:
                    value.writestr(member, data.getvalue())
            path = root / "obstacles.zip"
            archive(path, "t.parquet", obstacles)
            archive(root / "ego.zip", "ego.parquet", ego)
            context = {"repo_id": "repo", "revision": "abc", "obstacle_relative_path": "obstacles.zip",
                       "obstacle_member": "t.parquet", "offline_egomotion_path": str(root / "ego.zip"),
                       "offline_egomotion_member": "ego.parquet", "current_timestamp_us": 1_000_000}
            config = {"obstacle_root": str(root), "work_root": str(root / "work"), "horizon_s": 4., "max_track_gap_s": .25}
            worlds = BackgroundWorlds(config)
            try:
                first, fingerprint = worlds.load(context)
                self.assertEqual(worlds.load(context), (first, fingerprint))
                cache = root / "work/worlds" / f"{fingerprint}.json"
                bad = read_json(cache)
                bad["world"]["tracks"] = []
                write_json(cache, bad)
                self.assertEqual(worlds.load(context)[0], first)
            finally:
                worlds.close()
            obstacles["size_x"] = 5.
            archive(path, "t.parquet", obstacles)
            worlds = BackgroundWorlds(config)
            try:
                changed, changed_fingerprint = worlds.load(context)
                self.assertNotEqual(changed_fingerprint, fingerprint)
                self.assertEqual(changed["tracks"][0]["sizes"][0][0], 5.)
            finally:
                worlds.close()


if __name__ == "__main__":
    unittest.main()
