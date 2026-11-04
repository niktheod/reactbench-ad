from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import math
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as pedestrian  # noqa: E402
import animal_composite_navsim_blender as dog  # noqa: E402


class PedestrianIsolationTest(unittest.TestCase):
    def test_importing_dog_adapter_does_not_patch_pedestrian_pipeline(self) -> None:
        self.assertEqual(
            pedestrian.RENDER_PIPELINE_VERSION,
            "photoreal_counterfactual_v20_measured_pedestrian_geometry",
        )
        self.assertIs(pedestrian.load_asset, dog._ORIGINAL_LOAD_ASSET)
        self.assertIs(pedestrian.prepare_token, dog._ORIGINAL_PREPARE_TOKEN)
        self.assertIs(
            pedestrian.render_utils.actor_bbox_lidar,
            dog._ORIGINAL_ACTOR_BBOX,
        )
        self.assertIs(pedestrian.standing_facing, dog._ORIGINAL_STANDING_FACING)
        self.assertEqual(
            pedestrian.STANDING_POSE_VERSION, "procedural_natural_standing_v2"
        )
        self.assertEqual(
            pedestrian.STANDING_FACING_POLICY, "opposite_future_ego_forward"
        )

    def test_default_standing_hook_faces_the_ego(self) -> None:
        self.assertEqual(
            pedestrian.standing_facing(Namespace(seed=42), "token"),
            ("standing_facing_ego", 0.0, pedestrian.STANDING_ACTOR_HEADING_RAD),
        )


class DogGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = dog.dog_geometry_from_metadata(
            {
                "actor_type": "animal",
                "species": "dog",
                "character_height_m": 0.9,
                "actor_width_m": 0.31,
                "actor_length_m": 1.41,
            },
            Path("dog.json"),
        )

    def tearDown(self) -> None:
        dog._ACTIVE_GEOMETRY = None
        dog._ACTIVE_YAW = None

    def test_profile_uses_measured_dimensions_and_quadruped_contacts(self) -> None:
        self.assertEqual(self.profile["width_m"], 0.31)
        self.assertEqual(self.profile["length_m"], 1.41)
        self.assertAlmostEqual(self.profile["collision_width_m"], 0.43)
        self.assertAlmostEqual(self.profile["collision_length_m"], 1.69)
        self.assertEqual(self.profile["contact_profile"], "quadruped_dog")

    def test_dog_profile_keeps_root_centred_envelope(self) -> None:
        self.assertNotIn("collision_center_xy_m", self.profile)
        self.assertNotIn("standing_only", self.profile)
        self.assertEqual(self.profile["profile_version"], dog.DOG_GEOMETRY_PROFILE_VERSION)

    def test_human_sidecar_is_rejected_by_dog_adapter(self) -> None:
        with self.assertRaisesRegex(ValueError, "homogeneous animal asset pool"):
            dog.dog_geometry_from_metadata(
                {"character_height_m": 1.8},
                Path("human.json"),
            )

    def test_dog_bbox_uses_measured_dimensions(self) -> None:
        dog._ACTIVE_GEOMETRY = self.profile
        bbox = dog.dog_actor_bbox_lidar(
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            0.0,
            self.profile["local_bounds_m"],
        )

        self.assertAlmostEqual(float(np.ptp(bbox[:, 0])), 0.31)
        self.assertAlmostEqual(float(np.ptp(bbox[:, 1])), 1.41)
        self.assertAlmostEqual(float(np.ptp(bbox[:, 2])), 0.9)

    def test_collision_envelope_rotates_with_dog(self) -> None:
        dog._ACTIVE_GEOMETRY = self.profile
        points = np.tile(
            np.array([[0.65, 0.0, 0.5]], dtype=np.float64),
            (6, 1),
        )
        lidar_to_actor = np.eye(4, dtype=np.float64)

        dog._ACTIVE_YAW = 0.0
        narrow_axis = dog.dog_frame_collision_diagnostics(
            points,
            lidar_to_actor,
            0.0,
            self.profile["local_bounds_m"],
        )
        dog._ACTIVE_YAW = np.pi / 2.0
        long_axis = dog.dog_frame_collision_diagnostics(
            points,
            lidar_to_actor,
            np.pi / 2.0,
            self.profile["local_bounds_m"],
        )

        self.assertFalse(narrow_axis["occupied"])
        self.assertTrue(long_axis["occupied"])


def camel_metadata(**overrides) -> dict:
    """A standing camel sidecar shaped like the generator's output."""
    metadata = {
        "id": "rocketbox_camel_onehump_01",
        "actor_type": "animal",
        "species": "camel",
        "contact_profile": "quadruped_camel",
        "standing_only": True,
        "character_height_m": 2.14,
        "actor_width_m": 0.64,
        "actor_length_m": 3.24,
        "actor_geometry": {
            "profile_version": "standing_animal_rest_pose_bounds_v1",
            "actor_type": "animal",
            "species": "camel",
            "asset_id": "rocketbox_camel_onehump_01",
            "source": "generated_asset_sidecar",
            "local_forward_axis": "-Y",
            "local_bounds_m": {
                "x": [-0.32, 0.32],
                "y": [-1.84, 1.40],
                "z": [0.0, 2.14],
            },
        },
    }
    metadata.update(overrides)
    return metadata


def standing_animal_metadata(species: str, name: str) -> dict:
    """``camel_metadata`` carrying another standing species' identity."""
    asset_id = f"rocketbox_{name}"
    metadata = camel_metadata(
        id=asset_id, species=species, contact_profile=f"quadruped_{species}"
    )
    metadata["actor_geometry"] = {
        **metadata["actor_geometry"],
        "species": species,
        "asset_id": asset_id,
    }
    return metadata


class StandingSpeciesTest(unittest.TestCase):
    def test_each_standing_species_uses_its_own_contact_profile(self) -> None:
        for species, name in (
            ("horse", "horse_brown_01"),
            ("cow", "cow_black_01"),
            ("deer", "deer_roe_tabby_01"),
            ("sheep", "sheep_white_01"),
        ):
            with self.subTest(species=species):
                profile = dog.dog_geometry_from_metadata(
                    standing_animal_metadata(species, name), Path(f"{name}.json")
                )
                self.assertEqual(profile["species"], species)
                self.assertEqual(profile["contact_profile"], f"quadruped_{species}")

    def test_unlisted_species_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "homogeneous animal asset pool"):
            dog.dog_geometry_from_metadata(
                standing_animal_metadata("goose", "goose_grey_01"), Path("goose.json")
            )

    def test_run_config_of_a_mixed_pool_is_labelled_mixed(self) -> None:
        camel = dog.dog_geometry_from_metadata(camel_metadata(), Path("camel.json"))
        deer = dog.dog_geometry_from_metadata(
            standing_animal_metadata("deer", "deer_roe_tabby_01"), Path("deer.json")
        )
        written = []
        with patch.object(
            dog,
            "_ORIGINAL_WRITE_JSON",
            side_effect=lambda path, value: written.append(value),
        ):
            for pool in ([camel, deer], [deer, deer]):
                dog.dog_write_json(
                    Path("run_config.json"),
                    {
                        "render_pipeline_version": dog.DOG_RENDER_PIPELINE_VERSION,
                        "asset_pool": [{"actor_geometry": item} for item in pool],
                    },
                )

        self.assertEqual(
            [value["actor_profile"] for value in written],
            ["quadruped_mixed", "quadruped_deer"],
        )


class CamelGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = dog.dog_geometry_from_metadata(
            camel_metadata(), Path("camel.json")
        )

    def tearDown(self) -> None:
        dog._ACTIVE_GEOMETRY = None
        dog._ACTIVE_YAW = None

    def test_measured_bounds_are_kept_off_centre(self) -> None:
        self.assertEqual(
            self.profile["profile_version"], dog.MEASURED_GEOMETRY_PROFILE_VERSION
        )
        self.assertEqual(self.profile["local_bounds_m"]["y"], [-1.84, 1.40])
        self.assertAlmostEqual(self.profile["length_m"], 3.24)
        self.assertAlmostEqual(self.profile["width_m"], 0.64)
        self.assertAlmostEqual(self.profile["height_m"], 2.14)
        self.assertAlmostEqual(self.profile["collision_length_m"], 3.52)
        np.testing.assert_allclose(
            self.profile["collision_center_xy_m"], [0.0, -0.22], atol=1e-12
        )
        self.assertEqual(self.profile["contact_profile"], "quadruped_camel")
        self.assertEqual(self.profile["asset_id"], "rocketbox_camel_onehump_01")
        self.assertTrue(self.profile["standing_only"])

    def test_collision_envelope_follows_off_centre_bounds(self) -> None:
        dog._ACTIVE_GEOMETRY = self.profile
        dog._ACTIVE_YAW = 0.0

        def diagnostics(y: float) -> dict:
            points = np.tile(np.array([[0.0, y, 1.0]], dtype=np.float64), (6, 1))
            return dog.dog_frame_collision_diagnostics(
                points,
                np.eye(4, dtype=np.float64),
                0.0,
                self.profile["local_bounds_m"],
            )

        # Envelope y spans [-1.98, 1.54]; a root-centred one would span +/-1.76.
        self.assertTrue(diagnostics(-1.95)["occupied"])
        self.assertFalse(diagnostics(1.55)["occupied"])
        np.testing.assert_allclose(
            diagnostics(0.0)["collision_envelope"]["collision_center_xy_m"],
            [0.0, -0.22],
            atol=1e-12,
        )

    def test_bbox_uses_measured_bounds(self) -> None:
        dog._ACTIVE_GEOMETRY = self.profile
        bbox = dog.dog_actor_bbox_lidar(
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            0.0,
            self.profile["local_bounds_m"],
        )

        self.assertAlmostEqual(float(bbox[:, 1].min()), -1.84)
        self.assertAlmostEqual(float(bbox[:, 1].max()), 1.40)
        self.assertAlmostEqual(float(np.ptp(bbox[:, 2])), 2.14)

    def test_camel_without_measured_geometry_is_rejected(self) -> None:
        metadata = camel_metadata()
        del metadata["actor_geometry"]
        with self.assertRaisesRegex(ValueError, "requires measured actor_geometry"):
            dog.dog_geometry_from_metadata(metadata, Path("camel.json"))

    def test_contact_profile_must_match_species(self) -> None:
        with self.assertRaisesRegex(ValueError, "contact_profile"):
            dog.dog_geometry_from_metadata(
                camel_metadata(contact_profile="quadruped_dog"), Path("camel.json")
            )

    def test_bounds_must_agree_with_recorded_height(self) -> None:
        with self.assertRaisesRegex(ValueError, "character_height_m"):
            dog.dog_geometry_from_metadata(
                camel_metadata(character_height_m=2.5), Path("camel.json")
            )

    def test_bounds_must_contain_the_root(self) -> None:
        metadata = camel_metadata()
        metadata["actor_geometry"] = {
            **metadata["actor_geometry"],
            "local_bounds_m": {
                "x": [0.1, 0.7],
                "y": [-1.84, 1.40],
                "z": [0.0, 2.14],
            },
        }
        with self.assertRaisesRegex(ValueError, "contain the root"):
            dog.dog_geometry_from_metadata(metadata, Path("camel.json"))

    def test_standing_flag_must_match_asset_capability(self) -> None:
        camel_asset = {"metadata": camel_metadata(), "metadata_path": Path("camel.json")}
        dog_asset = {
            "metadata": {
                "actor_type": "animal",
                "species": "dog",
                "character_height_m": 0.9,
                "actor_width_m": 0.31,
                "actor_length_m": 1.41,
            },
            "metadata_path": Path("dog.json"),
        }
        with patch.object(
            dog, "_ORIGINAL_LOAD_ASSET", side_effect=lambda path, args: dict(camel_asset)
        ):
            loaded = dog.load_dog_asset(Path("camel.blend"), Namespace(standing=True))
            self.assertEqual(loaded["actor_geometry"]["species"], "camel")
            with self.assertRaisesRegex(ValueError, "require --standing"):
                dog.load_dog_asset(Path("camel.blend"), Namespace(standing=False))
        with patch.object(
            dog, "_ORIGINAL_LOAD_ASSET", side_effect=lambda path, args: dict(dog_asset)
        ):
            dog.load_dog_asset(Path("dog.blend"), Namespace(standing=False))
            with self.assertRaisesRegex(ValueError, "requires standing-only"):
                dog.load_dog_asset(Path("dog.blend"), Namespace(standing=True))

    def test_written_json_records_the_contact_profile(self) -> None:
        written = []
        with patch.object(
            dog,
            "_ORIGINAL_WRITE_JSON",
            side_effect=lambda path, value: written.append(value),
        ):
            dog.dog_write_json(
                Path("metadata.json"),
                {
                    "render_pipeline_version": dog.DOG_RENDER_PIPELINE_VERSION,
                    "asset": {"actor_geometry": self.profile},
                },
            )
            dog.dog_write_json(
                Path("manifest.json"),
                {"render_pipeline_version": dog.DOG_RENDER_PIPELINE_VERSION},
            )

        self.assertEqual(written[0]["actor_profile"], "quadruped_camel")
        self.assertEqual(written[1]["actor_profile"], "quadruped_dog")


class StandingSideOnTest(unittest.TestCase):
    def tearDown(self) -> None:
        dog._ACTIVE_GEOMETRY = None
        dog._ACTIVE_LATERAL_SIGN = None
        dog._ACTIVE_YAW = None

    def test_side_is_seeded_per_token_and_balanced(self) -> None:
        args = Namespace(seed=42)
        tokens = [f"token_{index}" for index in range(200)]
        first = [dog.animal_standing_facing(args, token) for token in tokens]

        self.assertEqual(
            first, [dog.animal_standing_facing(args, token) for token in tokens]
        )
        left = sum(name.endswith("_left") for name, _sign, _heading in first)
        self.assertTrue(70 <= left <= 130, left)
        for name, sign, heading in first:
            side = "left" if name.endswith("_left") else "right"
            spec = pedestrian.DIRECTION_CLASSIFICATION_SPECS[side]
            self.assertEqual(
                (sign, heading), (spec["lateral_sign"], spec["actor_heading_rad"])
            )

    def test_head_and_collision_yaw_point_to_the_seeded_side(self) -> None:
        for token in ("token_a", "token_b", "token_c", "token_d"):
            _name, sign, heading = dog.animal_standing_facing(
                Namespace(seed=42), token
            )
            self.assertEqual(dog._ACTIVE_LATERAL_SIGN, sign)
            # Rocketbox assets face local -Y; the renderer rotates them by heading.
            head = np.array([math.sin(heading), -math.cos(heading)])
            np.testing.assert_allclose(head, [0.0, sign], atol=1e-12)
            dog._ACTIVE_GEOMETRY = {"contact_profile": "quadruped_camel"}
            with patch.object(
                dog, "_ORIGINAL_ROAD_BASIS", return_value=np.eye(3, dtype=np.float64)
            ):
                dog.capture_road_basis()
            # yaw - pi/2 is the head direction that scoring reads from yaw_lidar.
            self.assertAlmostEqual(
                dog._ACTIVE_YAW - math.pi / 2.0, math.atan2(head[1], head[0])
            )


class AdapterInstallTest(unittest.TestCase):
    def test_install_routes_standing_through_animal_hooks(self) -> None:
        names = (
            "RENDER_PIPELINE_VERSION",
            "STANDING_POSE_VERSION",
            "STANDING_FACING_POLICY",
            "standing_facing",
            "load_asset",
            "asset_config",
            "prepare_token",
            "crossing_policy",
            "road_basis_aligned_to_global_forward",
            "frame_collision_diagnostics",
            "trajectory_collision_diagnostics",
            "compatible_render_cache_key",
            "write_json",
            "build_loader",
        )
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(
                    patch.object(pedestrian, name, getattr(pedestrian, name))
                )
            stack.enter_context(
                patch.object(
                    pedestrian.render_utils,
                    "actor_bbox_lidar",
                    pedestrian.render_utils.actor_bbox_lidar,
                )
            )
            stack.enter_context(
                patch.object(
                    pedestrian.receiver_previews,
                    "load_receiver_shadow_labels",
                    pedestrian.receiver_previews.load_receiver_shadow_labels,
                )
            )
            dog.install_dog_adapter()
            self.assertIs(pedestrian.standing_facing, dog.animal_standing_facing)
            self.assertEqual(
                pedestrian.STANDING_POSE_VERSION, dog.QUADRUPED_STANDING_POSE_VERSION
            )
            self.assertEqual(
                pedestrian.STANDING_FACING_POLICY, dog.STANDING_SIDE_ON_FACING_POLICY
            )

        self.assertIs(pedestrian.standing_facing, dog._ORIGINAL_STANDING_FACING)
        self.assertEqual(
            pedestrian.STANDING_POSE_VERSION, "procedural_natural_standing_v2"
        )

    def test_renderer_accepts_the_adapter_contract(self) -> None:
        renderer = dog.DOG_RENDERER.read_text()
        self.assertIn(
            "QUADRUPED_STANDING_POSE_VERSION = "
            f'"{dog.QUADRUPED_STANDING_POSE_VERSION}"',
            renderer,
        )
        for profile in dog.QUADRUPED_CONTACT_PROFILES:
            self.assertIn(f'"{profile}"', renderer)






if __name__ == "__main__":
    unittest.main()
