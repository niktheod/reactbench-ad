from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from reactbench.compositing.paiav import edit as paiav_edit
from reactbench.compositing.paiav import paiav_io


geometry = paiav_edit.compositor.geometry
lens = paiav_edit.compositor.render_utils




def test_compositor_and_runtime_defaults_come_from_the_configured_locations() -> None:
    from reactbench.compositing import locations
    package_root = Path(paiav_edit.__file__).resolve().parent
    compositor_path = Path(paiav_edit.compositor.__file__).resolve()
    assert compositor_path.is_relative_to(package_root)
    assert paiav_edit.DEFAULT_ASSET_DIR == locations.ASSETS / "rocketbox" / "assets"
    assert paiav_edit.DEFAULT_BLENDER == locations.BLENDER
    assert paiav_edit.DEFAULT_DEPTH_CACHE == locations.MODEL_CACHE


def test_posix_spawn_command_redirects_output_environment_and_status(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "spawn.log"
    environment = os.environ.copy()
    environment["PAIAV_SPAWN_TEST_VALUE"] = "environment-ok"
    command = [
        sys.executable,
        "-c",
        (
            "import os, sys; "
            "print(os.environ['PAIAV_SPAWN_TEST_VALUE']); "
            "print('stderr-ok', file=sys.stderr); "
            "raise SystemExit(7)"
        ),
    ]

    with log_path.open("w", encoding="utf-8") as log:
        returncode = paiav_edit.compositor.run_command_with_posix_spawn(
            command,
            environment,
            log.fileno(),
        )

    output = log_path.read_text(encoding="utf-8")
    assert returncode == 7
    assert "environment-ok" in output
    assert "stderr-ok" in output


def test_blender_batch_uses_fork_free_launcher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_spawn(command, environment, output_fd):
        captured["command"] = command
        captured["environment"] = environment
        captured["output_fd"] = output_fd
        return 17

    def reject_subprocess_run(*args, **kwargs):
        raise AssertionError("Blender launch must not call subprocess.run")

    monkeypatch.setattr(
        paiav_edit.compositor,
        "run_command_with_posix_spawn",
        fake_spawn,
    )
    monkeypatch.setattr(
        paiav_edit.compositor.subprocess,
        "run",
        reject_subprocess_run,
    )
    log_path = tmp_path / "batch.log"
    batch = {
        "batch_index": 4,
        "asset_path": tmp_path / "actor.blend",
        "job_path": tmp_path / "jobs.jsonl",
        "log_path": log_path,
    }
    args = SimpleNamespace(blender=tmp_path / "blender", device="gpu")

    result = paiav_edit.compositor.run_blender_batch(
        args,
        tmp_path / "render_batch.py",
        batch,
        "2",
        3,
    )

    assert result == {
        "batch_index": 4,
        "log_path": log_path,
        "returncode": 17,
        "gpu_device": "2",
    }
    assert captured["command"] == [
        str(args.blender),
        "-b",
        str(batch["asset_path"]),
        "--python",
        str(tmp_path / "render_batch.py"),
        "--",
        "--jobs",
        str(batch["job_path"]),
        "--device",
        "gpu",
    ]
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert captured["environment"][variable] == "3"
    assert captured["output_fd"] >= 0
    assert log_path.read_text(encoding="utf-8") == (
        "RENDER_WORKER_GPU=2 NATIVE_THREADS=3\n"
    )








def test_overwrite_allows_only_known_occlusion_config_migration(
    tmp_path: Path,
) -> None:
    previous = {
        "run_name": "existing",
        "actor_mode": "car",
        "occlusion": {
            "requested_mode": "dense",
            "dense_alignment_fail_open": True,
            "fallback_mode": "coherent_lidar_surface",
            "surface_reconstruction": (
                paiav_edit.compositor.PREVIOUS_DENSE_OCCLUSION_VERSION
            ),
            "protect_lower_actor_without_lidar": True,
            "max_unconfirmed_fraction": 0.2,
        },
    }
    current = json.loads(json.dumps(previous))
    current["occlusion"].update(
        {
            "fallback_mode": "none_preserve_actor",
            "lidar_policy_in_dense_mode": "seed_only_never_direct_matte",
            "surface_reconstruction": (
                paiav_edit.compositor.DENSE_OCCLUSION_VERSION
            ),
        }
    )
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(previous), encoding="utf-8")

    paiav_edit.write_or_validate_run_config(
        tmp_path,
        current,
        allow_occlusion_migration=True,
    )

    assert json.loads(path.read_text(encoding="utf-8")) == current


def test_known_occlusion_migration_rejects_unrelated_changes() -> None:
    previous = {
        "run_name": "existing",
        "occlusion": {
            "requested_mode": "dense",
            "fallback_mode": "coherent_lidar_surface",
            "surface_reconstruction": (
                paiav_edit.compositor.PREVIOUS_DENSE_OCCLUSION_VERSION
            ),
        },
    }
    current = json.loads(json.dumps(previous))
    current["run_name"] = "different"
    current["occlusion"].update(
        {
            "fallback_mode": "none_preserve_actor",
            "lidar_policy_in_dense_mode": "seed_only_never_direct_matte",
            "surface_reconstruction": (
                paiav_edit.compositor.DENSE_OCCLUSION_VERSION
            ),
        }
    )

    assert not paiav_edit._known_occlusion_config_migration(previous, current)


def test_known_ego_mask_migration_requires_the_exact_new_policy() -> None:
    existing = {"run_name": "existing", "occlusion": {"requested_mode": "dense"}}
    current = json.loads(json.dumps(existing))
    current["ego_mask_compositing"] = paiav_edit.ego_mask_compositing_config()

    assert paiav_edit._known_ego_mask_config_migration(existing, current)
    current["ego_mask_compositing"]["version"] = "unexpected"
    assert not paiav_edit._known_ego_mask_config_migration(existing, current)




def ftheta_calibration(width: int = 65, height: int = 49, focal: float = 40.0) -> dict:
    principal_point = [(width - 1) / 2, (height - 1) / 2]
    return {
        "camera_model": "ftheta",
        "resolution": [width, height],
        "cam_intrinsic": [
            [focal, 0.0, principal_point[0]],
            [0.0, focal, principal_point[1]],
            [0.0, 0.0, 1.0],
        ],
        "principal_point": principal_point,
        # Exact equidistant model and inverse for a compact synthetic camera.
        "angle_to_pixeldist_poly": [0.0, focal],
        "pixeldist_to_angle_poly": [0.0, 1.0 / focal],
        "max_angle_rad": 1.4,
        "sensor2lidar_rotation": np.eye(3).tolist(),
        "sensor2lidar_translation": [0.0, 0.0, 0.0],
    }


def ego_mask_png(mask: np.ndarray) -> bytes:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask, 2] = 255
    payload = io.BytesIO()
    Image.fromarray(rgb).save(payload, format="PNG")
    return payload.getvalue()


def test_paiav_intrinsic_loads_and_propagates_calibrated_ego_mask() -> None:
    assert (
        paiav_io.EGO_MASK_ENCODING_VERSION
        == paiav_edit.compositor.EGO_MASK_ENCODING_VERSION
    )
    assert (
        paiav_io.EGO_MASK_IMAGE_BYTES_KEY
        == paiav_edit.compositor.EGO_MASK_IMAGE_BYTES_KEY
    )
    width, height = 8, 6
    protected = np.zeros((height, width), dtype=bool)
    protected[4:, 1:7] = True
    payload = ego_mask_png(protected)
    parameters = {
        "resolution": [width, height],
        "principal_point": [3.5, 2.5],
        "angle_to_pixeldist_poly": [0.0, 5.0],
        "pixeldist_to_angle_poly": [0.0, 0.2],
        "max_angle": 1.4,
    }
    loader = object.__new__(paiav_io.PAIAVSceneLoader)
    loader.camera_intrinsics = pd.DataFrame(
        {
            "model_type": ["ftheta"],
            "model_parameters": [json.dumps(parameters)],
            "ego_mask_image_png": [payload],
        },
        index=pd.MultiIndex.from_tuples(
            [("clip", "camera")], names=["clip_id", "camera_name"]
        ),
    )

    calibration = loader._camera_intrinsic("clip", "camera")

    assert calibration[paiav_io.EGO_MASK_IMAGE_BYTES_KEY] == payload
    assert calibration["ego_mask_encoding"] == paiav_io.EGO_MASK_ENCODING_VERSION
    assert calibration["ego_mask_protected_pixel_count"] == 12
    assert calibration["ego_mask_image_sha256"] == hashlib.sha256(payload).hexdigest()


def test_ego_mask_protects_actor_shadows_pctnet_mask_and_source_pixels() -> None:
    width, height = 8, 6
    protected = np.zeros((height, width), dtype=bool)
    protected[3:, 1:7] = True
    payload = ego_mask_png(protected)
    calibration = ftheta_calibration(width, height, focal=5.0)
    calibration.update(
        {
            paiav_edit.compositor.EGO_MASK_IMAGE_BYTES_KEY: payload,
            "ego_mask_image_sha256": hashlib.sha256(payload).hexdigest(),
            "ego_mask_encoding": paiav_edit.compositor.EGO_MASK_ENCODING_VERSION,
            "ego_mask_protected_pixel_count": int(np.count_nonzero(protected)),
            "ego_mask_provenance": {"source_field": "refined_gls_ego_mask", "input_hash": "test"},
        }
    )
    cache = {}
    decoded = paiav_edit.compositor.decode_calibrated_ego_mask(
        calibration, (height, width), "clip/camera/3.png", cache
    )
    assert decoded is paiav_edit.compositor.decode_calibrated_ego_mask(
        calibration, (height, width), "clip/camera/3.png", cache
    )
    np.testing.assert_array_equal(decoded, protected)
    assert (
        paiav_edit.compositor.calibration_without_ego_mask(calibration)
        == ftheta_calibration(width, height, focal=5.0)
    )

    actor_rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    shadow_alpha = np.full((height, width), 120, dtype=np.uint8)
    actor_rgba = paiav_edit.compositor.zero_pixels_in_ego_mask(
        actor_rgba, decoded
    )
    shadow_alpha = paiav_edit.compositor.zero_pixels_in_ego_mask(
        shadow_alpha, decoded
    )
    assert not np.any(actor_rgba[protected])
    assert not np.any(shadow_alpha[protected])

    source = np.full((height, width, 3), 80, dtype=np.uint8)
    edited = np.full((height, width, 3), 160, dtype=np.uint8)
    combined_mask = np.full((height, width), 255, dtype=np.uint8)
    # Exercise the final safeguard independently of the earlier layer masking.
    actor_rgba[protected] = 255
    shadow_alpha[protected] = 120
    (
        edited,
        combined_mask,
        actor_rgba,
        actor_alpha,
        shadow_alpha,
        diagnostics,
    ) = paiav_edit.compositor.apply_ego_mask_final_guard(
        source,
        edited,
        combined_mask,
        actor_rgba,
        shadow_alpha,
        decoded,
    )

    np.testing.assert_array_equal(edited[protected], source[protected])
    assert not np.any(combined_mask[protected])
    assert not np.any(actor_rgba[protected])
    assert not np.any(actor_alpha[protected])
    assert not np.any(shadow_alpha[protected])
    assert diagnostics["source_pixels_restored_by_final_guard"] == 18
    assert diagnostics["pctnet_excluded_protected_pixel_count"] == 18


def test_fixed_animation_uses_exact_ready_poses() -> None:
    timestamps = np.array([4.8, 4.9, 5.0, 5.1])
    assert paiav_edit.fixed_animation_frames(
        timestamps, 1.4, 1.2, 1, 31, 24
    ) == [24, 27, 30, 1]

    with pytest.raises(ValueError, match="requires 4 history timestamps"):
        paiav_edit.fixed_animation_frames(timestamps[:3], 1.4, 1.2, 1, 31, 24)
    with pytest.raises(ValueError, match="outside action range"):
        paiav_edit.fixed_animation_frames(timestamps, 1.4, 1.2, 2, 31, 24)


def test_ftheta_pixel_ray_round_trip() -> None:
    calibration = ftheta_calibration()
    pixels = np.array(
        [[32.0, 24.0], [44.0, 24.0], [17.0, 31.0], [56.0, 9.0]],
        dtype=np.float64,
    )
    rays = geometry.pixels_to_camera_rays(pixels, calibration)
    recovered = geometry.camera_rays_to_pixels(rays, calibration)

    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(recovered, pixels, atol=1e-10)


def test_ftheta_projection_and_dense_lens_map_agree() -> None:
    calibration = ftheta_calibration()
    pixels = np.array([[32.0, 24.0], [42.0, 29.0], [12.0, 17.0]])
    rays = geometry.pixels_to_camera_rays(pixels, calibration)
    projected, depth = geometry.project_distorted(rays * 8.0, calibration)
    map_x, map_y = lens.lens_maps(65, 49, calibration)

    np.testing.assert_allclose(projected, pixels, atol=1e-10)
    assert np.all(depth > 0)
    assert map_x.shape == (49, 65)
    assert map_y.shape == (49, 65)
    assert np.isfinite(map_x).all()
    assert np.isfinite(map_y).all()
    assert map_x[24, 32] == pytest.approx(32.0)
    assert map_y[24, 32] == pytest.approx(24.0)


def test_existing_brown_projection_is_unchanged() -> None:
    calibration = {
        "cam_intrinsic": [[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0, 0, 1]],
        "distortion": [0.01, -0.005, 0.001, -0.002, 0.0001],
        "sensor2lidar_rotation": np.eye(3).tolist(),
        "sensor2lidar_translation": [0.0, 0.0, 0.0],
    }
    points = np.array([[1.0, 0.5, 10.0], [-0.8, 0.2, 7.0], [0.1, -0.5, 4.0]])
    expected, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        np.asarray(calibration["cam_intrinsic"]),
        np.asarray(calibration["distortion"]),
    )
    actual, depth = geometry.project_distorted(points, calibration)

    np.testing.assert_allclose(actual, expected.reshape(-1, 2), atol=1e-10)
    np.testing.assert_allclose(depth, points[:, 2])


def test_balanced_directions_are_deterministic_and_nearly_even() -> None:
    tokens = [f"clip-{index:03d}" for index in range(11)]
    first = paiav_edit.balanced_directions(tokens, seed=42)
    second = paiav_edit.balanced_directions(list(reversed(tokens)), seed=42)

    assert first == second
    assert set(first) == set(tokens)
    counts = [sum(value == label for value in first.values()) for label in ("left", "right")]
    assert abs(counts[0] - counts[1]) == 1


def test_selection_manifest_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "selection.json"
    document = {
        "document_role": "pai_av_clip_selection_manifest",
        "selection": {
            "selected_clip_count": 2,
            "selected_chunk_id": 726,
            "camera_features": list(paiav_io.CAMERA_NAMES),
        },
        "selected_clip_ids": ["clip-a", "clip-b"],
    }
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    selection = paiav_io.load_selection_manifest(manifest_path)

    assert selection.clip_ids == ("clip-a", "clip-b")
    assert selection.chunk_id == 726
    assert selection.chunk_ids == (726,)
    assert selection.chunk_id_for("clip-a") == 726
    assert selection.manifest_path == manifest_path.resolve()


def write_dataset_manifest(root: Path) -> Path:
    clip_chunks = {"clip-a": 11, "clip-b": 22}
    clip_index_path = root / "clip_index.parquet"
    pd.DataFrame(
        {"chunk": list(clip_chunks.values())},
        index=pd.Index(clip_chunks, name="clip_id"),
    ).to_parquet(clip_index_path)
    chunk_records = []
    for chunk_id in sorted(set(clip_chunks.values())):
        feature_files = {}
        for feature in paiav_io.RECONSTRUCTION_FEATURES:
            relative_path = Path(f"chunk_{chunk_id}") / feature
            source_path = root / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(f"{chunk_id}:{feature}".encode())
            feature_files[feature] = {
                "path": str(relative_path),
                "size_bytes": source_path.stat().st_size,
            }
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "clip_count": 1,
                "feature_files": feature_files,
            }
        )
    manifest_path = root / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "document_role": "pai_av_dataset_manifest",
                "schema_version": 1,
                "dataset": {
                    "repo_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
                    "catalog_inputs": {
                        "clip_index.parquet": {
                            "path": str(clip_index_path),
                            "size_bytes": clip_index_path.stat().st_size,
                            "sha256": hashlib.sha256(
                                clip_index_path.read_bytes()
                            ).hexdigest(),
                        }
                    },
                },
                "samples": {
                    "count": len(clip_chunks),
                    "clip_ids": list(clip_chunks),
                    "required_features": list(paiav_io.RECONSTRUCTION_FEATURES),
                    "chunk_count": len(chunk_records),
                    "chunks": chunk_records,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_dataset_manifest_routes_clips_across_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = write_dataset_manifest(tmp_path)
    selection = paiav_io.load_selection_manifest(manifest_path)

    assert selection.manifest_role == "pai_av_dataset_manifest"
    assert selection.clip_ids == ("clip-a", "clip-b")
    assert selection.chunk_ids == (11, 22)
    assert selection.chunk_id is None
    assert selection.chunk_id_for("clip-a") == 11
    assert selection.chunk_id_for("clip-b") == 22
    paths = paiav_io.resolve_chunk_feature_paths(
        tmp_path, selection, selection.chunk_ids
    )
    assert paths[11]["egomotion"].parent.name == "chunk_11"
    assert paths[22]["egomotion"].parent.name == "chunk_22"

    parquet_reads = []

    def fake_read_parquet(path, columns=None, filters=None):
        parquet_reads.append((Path(path), columns, filters))
        return pd.DataFrame()

    monkeypatch.setattr(paiav_io.pd, "read_parquet", fake_read_parquet)
    loader = paiav_io.PAIAVSceneLoader(
        tmp_path,
        selection,
        selection.clip_ids,
        tmp_path / "run",
        anchor_time_s=5.1,
        future_timestep=10,
    )

    assert loader._feature_paths_for_token("clip-a")["egomotion"].parent.name == (
        "chunk_11"
    )
    assert loader._feature_paths_for_token("clip-b")["egomotion"].parent.name == (
        "chunk_22"
    )
    assert len(parquet_reads) == 6
    assert all(read[2] is not None for read in parquet_reads)
    assert {
        tuple(read[2][0][2])
        for read in parquet_reads
        if read[0].parent.name == "chunk_11"
    } == {("clip-a",)}
    camera_read = next(
        read
        for read in parquet_reads
        if read[0].name == "camera_intrinsics.offline"
    )
    assert "ego_mask_image_png" in camera_read[1]


def test_decoded_draco_attribute_flattens_named_data() -> None:
    decoded = SimpleNamespace(
        get_attribute_by_name=lambda name: {
            "name": name,
            "data": np.array([[10], [20], [30]], dtype=np.uint32),
        }
    )

    values = paiav_io._decoded_draco_attribute(decoded, "timestamp")

    np.testing.assert_array_equal(values, [10, 20, 30])


def test_lidar_motion_compensation_uses_each_point_timestamp() -> None:
    egomotion = pd.DataFrame(
        {
            "timestamp": [0, 100_000],
            "qx": [0.0, 0.0],
            "qy": [0.0, 0.0],
            "qz": [0.0, 0.0],
            "qw": [1.0, 1.0],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )
    nominal_pose = np.eye(4)
    nominal_pose[0, 3] = 5.0

    compensated = paiav_io._motion_compensate_lidar_points(
        np.zeros((2, 3)),
        np.array([0, 100_000]),
        np.eye(4),
        nominal_pose,
        egomotion,
    )

    np.testing.assert_allclose(
        compensated,
        [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
    )


def test_alpamayo_history_preserves_native_3d_poses_at_10hz() -> None:
    timestamps = np.arange(16, dtype=np.int64) * 100_000
    euler_xyz = np.column_stack(
        [
            np.linspace(0.0, 0.15, 16),
            np.linspace(0.0, -0.09, 16),
            np.linspace(0.0, 0.3, 16),
        ]
    )
    quaternions = paiav_io.Rotation.from_euler("xyz", euler_xyz).as_quat()
    translations = np.column_stack(
        [
            np.linspace(0.0, 15.0, 16),
            np.linspace(0.0, 3.0, 16),
            np.linspace(0.0, 1.5, 16),
        ]
    )
    egomotion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "qx": quaternions[:, 0],
            "qy": quaternions[:, 1],
            "qz": quaternions[:, 2],
            "qw": quaternions[:, 3],
            "x": translations[:, 0],
            "y": translations[:, 1],
            "z": translations[:, 2],
        }
    )

    history = paiav_io.build_alpamayo_ego_history(egomotion, timestamps[-1])

    world_from_rig = np.repeat(np.eye(4)[None], 16, axis=0)
    world_from_rig[:, :3, :3] = paiav_io.Rotation.from_quat(
        quaternions
    ).as_matrix()
    world_from_rig[:, :3, 3] = translations
    expected = np.linalg.inv(world_from_rig[-1])[None] @ world_from_rig
    assert history["version"] == paiav_io.ALPAMAYO_EGO_HISTORY_VERSION
    assert history["source_feature"] == "egomotion"
    assert history["coordinate_frame"] == "current_ego"
    assert history["history_interval_us"] == 100_000
    assert history["timestamps_us"] == timestamps.tolist()
    assert np.asarray(history["ego_history_xyz"]).shape == (16, 3)
    assert np.asarray(history["ego_history_rot"]).shape == (16, 3, 3)
    np.testing.assert_allclose(history["ego_history_xyz"][:-1], expected[:-1, :3, 3])
    np.testing.assert_allclose(
        history["ego_history_rot"][:-1], expected[:-1, :3, :3]
    )
    np.testing.assert_allclose(history["ego_history_xyz"][-1], 0.0)
    np.testing.assert_allclose(history["ego_history_rot"][-1], np.eye(3))
    assert not np.allclose(np.asarray(history["ego_history_rot"])[0, 2, :2], 0.0)


def test_ego_vehicle_geometry_preserves_all_native_dimensions() -> None:
    dimensions = pd.Series(
        {
            "length": 5.393,
            "width": 2.109,
            "height": 1.503,
            "rear_axle_to_bbox_center": 1.3965,
            "wheelbase": 3.216,
            "track_width": 1.669,
        }
    )

    geometry = paiav_io.build_ego_vehicle_geometry(dimensions)

    assert geometry["version"] == paiav_io.EGO_VEHICLE_GEOMETRY_VERSION
    assert geometry["source_feature"] == "vehicle_dimensions"
    assert geometry["units"] == "meters"
    assert geometry["dimensions_m"] == dimensions.to_dict()
    assert geometry["rear_axle_to_front_bumper_m"] == pytest.approx(4.093)
    assert geometry["rear_axle_to_rear_bumper_m"] == pytest.approx(1.3)
    assert geometry["footprint_rear_axle_m"] == pytest.approx(
        {
            "x_min": -1.3,
            "x_max": 4.093,
            "y_min": -1.0545,
            "y_max": 1.0545,
        }
    )


@pytest.mark.parametrize("navigation_command", [None])
def test_paiav_annotation_context_adds_history_without_replacing_four_steps(
    navigation_command,
) -> None:
    statuses = [
        SimpleNamespace(
            ego_velocity=np.array([index, index + 0.5]),
            ego_acceleration=np.array([index + 1.0, index + 1.5]),
            ego_pose=np.array([index + 2.0, index + 2.5, index + 3.0]),
            driving_command="unknown",
        )
        for index in range(4)
    ]
    native_history = {
        "ego_history_xyz": np.zeros((16, 3)).tolist(),
        "ego_history_rot": np.repeat(np.eye(3)[None], 16, axis=0).tolist(),
    }
    vehicle_geometry = {"version": paiav_io.EGO_VEHICLE_GEOMETRY_VERSION}
    loader = SimpleNamespace(
        get_agent_input_from_token=lambda _token: SimpleNamespace(
            ego_statuses=statuses
        ),
        alpamayo_ego_history=lambda _token: native_history,
        ego_vehicle_geometry=lambda _token: vehicle_geometry,
        navigation_commands={"clip-a": navigation_command},
    )

    context = paiav_edit.paiav_annotation_context(loader, "clip-a")

    assert len(context["velocity"]) == 4
    assert len(context["acceleration"]) == 4
    assert len(context["past_trajectory"]) == 4
    assert context["navigation_command"] == navigation_command
    assert context["alpamayo_ego_history"] is native_history
    assert context["ego_vehicle_geometry"] is vehicle_geometry


def test_success_metadata_backfills_paiav_evaluation_fields(tmp_path: Path) -> None:
    token = "clip-a"
    metadata_path = tmp_path / "metadata" / f"{token}.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        json.dumps({"status": "success", "annotation_sample": {"token": token}}),
        encoding="utf-8",
    )
    native_history = {"version": paiav_io.ALPAMAYO_EGO_HISTORY_VERSION}
    vehicle_geometry = {"version": paiav_io.EGO_VEHICLE_GEOMETRY_VERSION}
    clip_metadata = {
        "clip_id": token,
        "ego_vehicle_geometry": vehicle_geometry,
    }
    loader = SimpleNamespace(
        alpamayo_ego_history=lambda _token: native_history,
        ego_vehicle_geometry=lambda _token: vehicle_geometry,
        clip_metadata=lambda _token: clip_metadata,
    )
    selection = SimpleNamespace(manifest_path=tmp_path / "selection.json")

    paiav_edit.enrich_success_metadata(
        tmp_path,
        [token],
        loader,
        selection,
        {token: "Continue straight"},
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["annotation_sample"]["alpamayo_ego_history"] == native_history
    assert metadata["annotation_sample"]["ego_vehicle_geometry"] == (
        vehicle_geometry
    )
    assert metadata["ego_vehicle_geometry"] == vehicle_geometry
    assert metadata["paiav_source"] == clip_metadata
    assert metadata["annotation_sample"]["navigation_command"] == (
        "Continue straight"
    )


def test_rolling_shutter_pose_interpolation_is_row_dependent() -> None:
    calibration = ftheta_calibration()
    calibration.update(
        {
            "shutter_type": "ROLLING_TOP_TO_BOTTOM",
            "rolling_shutter_sensor2lidar_start_rotation": np.eye(3).tolist(),
            "rolling_shutter_sensor2lidar_start_translation": [-1.0, 0.0, 0.0],
            "rolling_shutter_sensor2lidar_end_rotation": np.eye(3).tolist(),
            "rolling_shutter_sensor2lidar_end_translation": [1.0, 0.0, 0.0],
        }
    )

    rotations, translations = geometry.rolling_shutter_sensor_poses(
        calibration, np.array([0.0, 0.5, 1.0])
    )

    np.testing.assert_allclose(rotations, np.repeat(np.eye(3)[None], 3, axis=0))
    np.testing.assert_allclose(translations[:, 0], [-1.0, 0.0, 1.0])


def test_paiav_grounding_uses_one_supported_common_plane(monkeypatch) -> None:
    planes = [np.array([0.0, 0.0, value]) for value in (0.0, 0.1, 0.2, 0.3)]
    diagnostics = [
        {
            "frame_index": index,
            "method": "strict_same_timestamp_local_lidar",
            "local_support_count": 20,
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        paiav_edit,
        "_BASE_FIT_GROUND_SEQUENCE",
        lambda *_args: (planes, diagnostics, []),
    )
    frames = [
        {"ego2global": np.eye(4), "lidar2global": np.eye(4)}
        for _ in range(4)
    ]

    consistent, output_diagnostics, failures = (
        paiav_edit.paiav_sequence_consistent_ground(
            frames,
            [np.empty((0, 3))] * 4,
            [[0.0, 0.0]] * 4,
            [[0.0, 0.0]] * 4,
        )
    )

    assert failures == []
    for plane in consistent:
        np.testing.assert_allclose(plane, [0.0, 0.0, 0.15])
    assert all(item["local_support_accepted"] for item in output_diagnostics)


def test_paiav_grounding_hard_fails_without_nearby_support(monkeypatch) -> None:
    diagnostics = [
        {
            "frame_index": index,
            "method": "relaxed_same_timestamp_local_lidar",
            "local_support_count_2p5m": 0,
            "nearest_candidate_distance_m": 3.5,
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        paiav_edit,
        "_BASE_FIT_GROUND_SEQUENCE",
        lambda *_args: ([np.zeros(3)] * 4, diagnostics, []),
    )
    frames = [
        {"ego2global": np.eye(4), "lidar2global": np.eye(4)}
        for _ in range(4)
    ]

    with pytest.raises(
        paiav_edit.compositor.TokenFailure,
        match="No history frame has accepted LiDAR ground support within 2.5 m",
    ) as failure:
        paiav_edit.paiav_sequence_consistent_ground(
            frames,
            [np.empty((0, 3))] * 4,
            [[0.0, 0.0]] * 4,
            [[0.0, 0.0]] * 4,
        )

    assert failure.value.code == "ground_fit_failed"
    assert '"nearest_candidate_distance_m":3.5' in str(failure.value)


def test_paiav_grounding_keeps_sequence_with_one_nearby_support(
    monkeypatch,
) -> None:
    planes = [np.array([0.0, 0.0, value]) for value in (0.0, 0.1, 0.2, 0.3)]
    diagnostics = [
        {
            "frame_index": index,
            "method": "relaxed_same_timestamp_local_lidar",
            "local_support_count_2p5m": 1 if index == 2 else 0,
            "nearest_candidate_distance_m": 2.0 if index == 2 else 3.5,
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        paiav_edit,
        "_BASE_FIT_GROUND_SEQUENCE",
        lambda *_args: (planes, diagnostics, []),
    )
    frames = [
        {"ego2global": np.eye(4), "lidar2global": np.eye(4)}
        for _ in range(4)
    ]

    consistent, output_diagnostics, failures = (
        paiav_edit.paiav_sequence_consistent_ground(
            frames,
            [np.empty((0, 3))] * 4,
            [[0.0, 0.0]] * 4,
            [[0.0, 0.0]] * 4,
        )
    )

    assert failures == []
    for plane in consistent:
        np.testing.assert_allclose(plane, [0.0, 0.0, 0.2])
    assert [
        item["local_support_accepted"] for item in output_diagnostics
    ] == [False, False, True, False]


def test_conservative_dense_occlusion_preserves_unsupported_lower_body() -> None:
    alpha = np.zeros((20, 10), dtype=np.uint8)
    alpha[2:18, 2:8] = 255
    source_depth = np.full(alpha.shape, 5.0, dtype=np.float32)
    road_depth = np.full(alpha.shape, 100.0, dtype=np.float32)

    strength, diagnostics = paiav_edit.compositor.dense_depth_occlusion(
        alpha,
        source_depth,
        actor_depth_m=20.0,
        road_metric_depth=road_depth,
        foreground_support=np.zeros(alpha.shape, dtype=bool),
        maximum_unconfirmed_fraction=0.20,
        protect_lower_without_lidar=True,
    )

    assert not np.any(strength)
    assert "protected_lower_actor_without_lidar_support" in diagnostics[
        "actor_integrity_actions"
    ]
    assert "fell_back_to_no_occlusion" in diagnostics[
        "actor_integrity_actions"
    ]


def test_dense_occlusion_never_uses_lidar_surface_as_direct_matte() -> None:
    alpha = np.zeros((50, 60), dtype=np.uint8)
    alpha[5:45, 5:55] = 255
    source_depth = np.full(alpha.shape, 20.0, dtype=np.float32)
    road_depth = np.full(alpha.shape, 100.0, dtype=np.float32)
    lidar_surface = np.zeros(alpha.shape, dtype=bool)
    lidar_surface[15:35, 15:45] = True

    strength, diagnostics = paiav_edit.compositor.dense_depth_occlusion(
        alpha,
        source_depth,
        actor_depth_m=10.0,
        road_metric_depth=road_depth,
        foreground_support=lidar_surface,
        maximum_unconfirmed_fraction=0.20,
        protect_lower_without_lidar=True,
    )

    assert not np.any(strength)
    assert diagnostics["requested_occluded_pixels_before_fail_safe"] == 0
    assert diagnostics["lidar_supported_actor_pixels"] == 600
    assert diagnostics["reconstructed_surface_pixels_before_growth_guard"] == 0
    assert diagnostics["lidar_support_policy"] == "seed_only_never_direct_matte"


def test_lidar_foreground_support_uses_per_pixel_actor_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_alpha = np.full((20, 20), 255, dtype=np.uint8)
    actor_depth = np.full(actor_alpha.shape, 4.0, dtype=np.float32)
    projected = np.asarray(
        [[x, 6.0] for x in range(2, 7)]
        + [[x, 12.0] for x in range(10, 15)],
        dtype=np.float64,
    )
    camera_depth = np.asarray([5.0] * 5 + [3.0] * 5, dtype=np.float64)
    lidar_xyz = np.zeros((len(projected), 3), dtype=np.float64)
    lidar_xyz[:, 2] = 1.0

    monkeypatch.setattr(
        geometry,
        "project_distorted_rolling_shutter",
        lambda _points, _calibration: (projected, camera_depth),
    )
    support, diagnostics = paiav_edit.compositor.lidar_foreground_support(
        lidar_xyz,
        {},
        np.zeros(3, dtype=np.float64),
        actor_alpha,
        actor_depth_m=10.0,
        actor_metric_depth=actor_depth,
    )

    assert diagnostics["actor_intersecting_above_ground_lidar_points"] == 10
    assert diagnostics["rejected_by_actor_surface_depth_points"] == 5
    assert diagnostics["closer_above_ground_lidar_points"] == 5
    assert diagnostics["z_pass_gated_point_count"] == 5
    assert diagnostics["root_depth_fallback_point_count"] == 0
    assert diagnostics["actor_depth_gate"] == (
        "per_pixel_cycles_z_with_root_fallback"
    )
    assert not np.any(support[4:9, :8])
    assert np.any(support[10:15, 8:17])


def test_lidar_surface_reconstruction_rejects_dots_and_fills_a_cluster() -> None:
    actor = np.zeros((300, 400), dtype=bool)
    actor[20:280, 20:380] = True
    cluster = np.array(
        [
            [x, y]
            for y in range(90, 171, 8)
            for x in range(120, 201, 8)
        ],
        dtype=np.float32,
    )
    isolated = np.array([[40, 40], [350, 250]], dtype=np.float32)
    pixels = np.concatenate([cluster, isolated], axis=0)
    depths = np.concatenate(
        [
            np.full(len(cluster), 6.0, dtype=np.float32),
            np.array([5.0, 5.5], dtype=np.float32),
        ]
    )

    support, diagnostics = (
        paiav_edit.compositor.coherent_lidar_surface_support(
            pixels,
            depths,
            actor,
            actor_depth_m=12.0,
        )
    )

    assert diagnostics["retained_component_count"] == 1
    assert diagnostics["rejected_sparse_component_count"] == 2
    assert support[130, 160]
    assert not support[40, 40]
    assert not support[250, 350]
    component_count, _labels = cv2.connectedComponents(
        support.astype(np.uint8), connectivity=8
    )
    assert component_count == 2
    assert np.count_nonzero(support) > len(cluster)


def test_dense_occlusion_keeps_only_lidar_seeded_depth_component() -> None:
    alpha = np.zeros((80, 120), dtype=np.uint8)
    alpha[5:75, 5:115] = 255
    source_depth = np.full(alpha.shape, 20.0, dtype=np.float32)
    source_depth[15:60, 15:65] = 5.0
    source_depth[20:50, 85:105] = 5.0
    road_depth = np.full(alpha.shape, 100.0, dtype=np.float32)
    support = np.zeros(alpha.shape, dtype=bool)
    support[30:42, 28:40] = True

    strength, diagnostics = paiav_edit.compositor.dense_depth_occlusion(
        alpha,
        source_depth,
        actor_depth_m=12.0,
        road_metric_depth=road_depth,
        foreground_support=support,
        maximum_unconfirmed_fraction=0.80,
        protect_lower_without_lidar=False,
    )

    assert diagnostics["surface_reconstruction"]["retained_component_count"] == 1
    assert np.count_nonzero(strength[15:60, 15:65]) > np.count_nonzero(support)
    assert not np.any(strength[20:50, 85:105])
    assert "removed_unseeded_depth_components" in diagnostics[
        "actor_integrity_actions"
    ]


def test_occlusion_sequence_consistency_flags_large_fraction_jump() -> None:
    records = [
        {
            "camera": "camera_front",
            "frame_index": frame_index,
            "occlusion_diagnostics": {
                "actor_pixels": 100,
                "occluded_pixels": occluded_pixels,
            },
        }
        for frame_index, occluded_pixels in enumerate((10, 12, 80, 82))
    ]

    diagnostics = paiav_edit.compositor.occlusion_sequence_consistency(records)

    assert diagnostics["status"] == "review"
    assert diagnostics["review_cameras"] == ["camera_front"]
    assert diagnostics["cameras"]["camera_front"][
        "maximum_adjacent_fraction_change"
    ] == pytest.approx(0.68)


def test_atomic_publication_commits_only_a_complete_token(tmp_path: Path, monkeypatch) -> None:
    compositor = paiav_edit.compositor
    monkeypatch.setattr(compositor, "CAMERA_ORDER", ("camera",))
    monkeypatch.setattr(compositor, "HISTORY_FRAME_INDICES", (0,))
    staging = compositor.initialize_atomic_token_publication(tmp_path, "clip")
    staged_to_final = {}
    for tree in compositor.TOKEN_PUBLICATION_TREES:
        final = tmp_path / tree / "clip" / "camera" / "0.png"
        staged = compositor.staged_token_output_path(tmp_path, staging, final)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(tree.encode())
        staged_to_final[staged] = final

    compositor.commit_atomic_token_publication(
        tmp_path, "clip", staged_to_final
    )

    for final in staged_to_final.values():
        assert final.is_file()
    assert not staging.exists()


def test_atomic_publication_cleans_metadata_only_legacy_output(tmp_path: Path) -> None:
    compositor = paiav_edit.compositor
    token = "legacy-partial"
    legacy_image = tmp_path / "images" / token / "camera" / "0.png"
    legacy_final = tmp_path / "final_images" / token / "camera" / "0.png"
    metadata = tmp_path / "metadata" / f"{token}.json"
    for path in (legacy_image, legacy_final, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")

    staging = compositor.initialize_atomic_token_publication(tmp_path, token)

    assert staging.is_dir()
    assert not legacy_image.exists()
    assert not legacy_final.exists()
    assert not metadata.exists()
