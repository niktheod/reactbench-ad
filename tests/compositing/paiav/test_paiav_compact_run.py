from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from reactbench.compositing.paiav import compact_run


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run(root: Path) -> Path:
    run_dir = root / "output" / "synthetic"
    data_root = root / "data"
    data_root.mkdir(parents=True)
    feature_names = (
        *compact_run.CAMERAS,
        "lidar_top_360fov",
        "camera_intrinsics.offline",
        "lidar_intrinsics.offline",
        "sensor_extrinsics.offline",
        "vehicle_dimensions",
        "egomotion.offline",
        "egomotion",
    )
    feature_files = {}
    for feature in feature_names:
        relative = f"features/{feature}.source"
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(feature.encode())
        feature_files[feature] = relative

    manifest_path = data_root / "selection.json"
    manifest_path.write_text(
        json.dumps(
            {
                "document_role": "pai_av_clip_selection_manifest",
                "selection": {
                    "selected_clip_count": 1,
                    "selected_chunk_id": 726,
                    "camera_features": list(compact_run.CAMERAS),
                },
                "chunk_feature_files": feature_files,
                "selected_clip_ids": ["clip-a"],
            }
        ),
        encoding="utf-8",
    )
    token = "clip-a"
    run_config = {
        "document_role": "paiav_edit_run_config",
        "schema_version": 2,
        "dataset": {
            "repo_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
            "data_root": str(data_root),
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": sha256(manifest_path),
            "selected_chunk_id": 726,
        },
        "selected_clip_ids": [token],
        "selected_clip_count": 1,
    }
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(run_config))

    nominal_timestamps = [4_800_000 + 100_000 * index for index in range(4)]
    camera_indices = {
        camera: [100 + index for index in range(4)]
        for camera in compact_run.CAMERAS
    }
    camera_timestamps = {
        camera: [timestamp + 7_500 for timestamp in nominal_timestamps]
        for camera in compact_run.CAMERAS
    }
    records = []
    for camera in compact_run.CAMERAS:
        for frame_index in compact_run.FRAME_INDICES:
            relative = Path(token) / camera / f"{frame_index}.png"
            source = np.full((8, 12, 3), 20, dtype=np.uint8)
            composite = source.copy()
            final = source.copy()
            mask = np.zeros((8, 12), dtype=np.uint8)
            actor_mask = np.zeros((8, 12), dtype=np.uint8)
            if camera == "camera_front_wide_120fov":
                mask[3, 5] = 255
                actor_mask[3, 5] = 255
                composite[3, 5] = [60, 70, 80]
                final[3, 5] = [65, 75, 85]
            source_path = run_dir / "source_frames" / relative
            write_png(source_path, source)
            write_png(run_dir / "images" / relative, composite)
            write_png(run_dir / "masks" / relative, mask)
            write_png(run_dir / "actor_masks" / relative, actor_mask)
            write_png(run_dir / "final_images" / relative, final)
            records.append(
                {
                    "camera": camera,
                    "frame_index": frame_index,
                    "timestamp_us": nominal_timestamps[frame_index],
                    "source_path": str(source_path),
                    "mask_pixel_count": int(np.count_nonzero(mask)),
                    "actor_mask_pixel_count": int(np.count_nonzero(actor_mask)),
                    "changed_pixel_count": int(
                        np.count_nonzero(np.any(composite != source, axis=2))
                    ),
                }
            )

    ego_vehicle_geometry = {
        "version": "paiav_ego_vehicle_geometry_v1",
        "source_feature": "vehicle_dimensions",
        "units": "meters",
        "dimensions_m": {
            "length": 5.393,
            "width": 2.109,
            "height": 1.503,
            "rear_axle_to_bbox_center": 1.3965,
            "wheelbase": 3.216,
            "track_width": 1.669,
        },
        "rear_axle_to_front_bumper_m": 4.093,
        "rear_axle_to_rear_bumper_m": 1.3,
        "footprint_rear_axle_m": {
            "x_min": -1.3,
            "x_max": 4.093,
            "y_min": -1.0545,
            "y_max": 1.0545,
        },
    }
    metadata = {
        "status": "success",
        "token": token,
        "records": records,
        "paiav_source": {
            "clip_id": token,
            "nominal_history_timestamps_us": nominal_timestamps,
            "camera_source_frame_indices": camera_indices,
            "camera_source_timestamps_us": camera_timestamps,
            "camera_exposure_intervals_us": {
                camera: [[timestamp - 10, timestamp + 10] for timestamp in values]
                for camera, values in camera_timestamps.items()
            },
            "lidar_spin_indices": [40, 41, 42, 43],
            "lidar_midpoint_timestamps_us": nominal_timestamps,
            "lidar_motion_compensation": [],
            "ego_vehicle_geometry": ego_vehicle_geometry,
        },
    }
    metadata_path = run_dir / "metadata" / f"{token}.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    completion_path = run_dir / "completed_tokens" / f"{token}.json"
    completion_path.parent.mkdir(parents=True)
    completion_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "token": token,
                "published_view_count": 16,
                "expected_view_count": 16,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "selected_tokens": 1,
                "successful_tokens": 1,
                "failed_tokens": 0,
                "expected_views_per_token": 16,
                "expected_selected_views": 16,
                "published_successful_views": 16,
                "complete_token_markers": 1,
                "potential_failure_tokens": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "failures.jsonl").write_text("", encoding="utf-8")
    annotation = {
        "token": token,
        "clip_id": token,
        **{
            camera: [
                str(run_dir / "images" / token / camera / f"{index}.png")
                for index in compact_run.FRAME_INDICES
            ]
            for camera in compact_run.CAMERAS
        },
        "history_interval_s": 0.1,
        "past_trajectory": [[0.0, 0.0, 0.0]] * 4,
        "navigation_command": "Turn right",
        "ego_vehicle_geometry": ego_vehicle_geometry,
    }
    (run_dir / "annotations.json").write_text(
        json.dumps([annotation]),
        encoding="utf-8",
    )
    for filename in (
        "direction_classifications.json",
        "potential_failures.json",
        "preparation_summary.json",
    ):
        (run_dir / filename).write_text("{}", encoding="utf-8")
    for directory in compact_run.CLEANUP_DIRECTORIES:
        path = run_dir / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "temporary.txt").write_text("temporary", encoding="utf-8")
    for filename in compact_run.CLEANUP_FILES:
        (run_dir / filename).write_text("temporary", encoding="utf-8")
    return run_dir


def test_compaction_validates_then_removes_only_working_set(tmp_path: Path) -> None:
    run_dir = build_run(tmp_path)

    report = compact_run.compact_run(run_dir, workers=2)

    assert report["status"] == "complete"
    assert report["total_final_frames"] == 16
    assert report["zero_mask_frames"] == 12
    assert report["compositor_changed_frames"] == 4
    assert report["pctnet_changed_frames"] == 4
    for name in (*compact_run.CLEANUP_DIRECTORIES, *compact_run.CLEANUP_FILES):
        assert not (run_dir / name).exists()
    assert len(list((run_dir / "final_images").glob("*/*/*.png"))) == 16
    assert (run_dir / "metadata" / "clip-a.json").is_file()
    assert (run_dir / "annotations.json").is_file()

    annotations = json.loads(
        (run_dir / "annotations.json").read_text(encoding="utf-8")
    )
    annotation = annotations[0]
    assert annotation["navigation_command"] == "Turn right"
    assert annotation["compact_evaluation_contract"]["version"] == (
        compact_run.COMPACT_ANNOTATION_VERSION
    )
    for camera in compact_run.CAMERAS:
        assert all("/images/" not in path for path in annotation[camera])
        assert all(Path(path).is_file() for path in annotation[camera])
    original = annotation["original_image_sources"]
    assert original["version"] == compact_run.ORIGINAL_IMAGE_SOURCE_VERSION
    front_source = original["cameras"]["camera_front_wide_120fov"]
    assert front_source["video_member"] == (
        "clip-a.camera_front_wide_120fov.mp4"
    )
    assert front_source["frame_indices"] == [100, 101, 102, 103]
    assert Path(front_source["archive_path"]).is_file()
    assert report["annotations_sha256"] == sha256(run_dir / "annotations.json")

    source_index = json.loads(
        (run_dir / compact_run.SOURCE_INDEX_FILENAME).read_text(encoding="utf-8")
    )
    assert source_index["version"] == compact_run.SOURCE_INDEX_VERSION
    assert source_index["successful_tokens"]["clip-a"]["camera_sources"][
        "camera_front_wide_120fov"
    ]["video_member"] == "clip-a.camera_front_wide_120fov.mp4"
    source_geometry = source_index["successful_tokens"]["clip-a"][
        "ego_vehicle_geometry"
    ]
    assert source_geometry["version"] == "paiav_ego_vehicle_geometry_v1"
    assert annotation["ego_vehicle_geometry"] == source_geometry

    repeated = compact_run.compact_run(run_dir, workers=2)
    assert repeated == report


def test_source_reconstruction_routes_each_token_to_its_dataset_chunk(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    token_chunks = {"clip-a": 11, "clip-b": 22}
    clip_index_path = data_root / "clip_index.parquet"
    pd.DataFrame(
        {"chunk": list(token_chunks.values())},
        index=pd.Index(token_chunks, name="clip_id"),
    ).to_parquet(clip_index_path)

    chunks = []
    resolved_by_chunk = {}
    for chunk_id in token_chunks.values():
        feature_files = {}
        resolved_by_chunk[chunk_id] = {}
        for feature in compact_run.RECONSTRUCTION_FEATURES:
            relative = Path(f"chunk_{chunk_id}") / feature
            path = data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{chunk_id}:{feature}".encode())
            feature_files[feature] = {
                "path": str(relative),
                "size_bytes": path.stat().st_size,
            }
            resolved_by_chunk[chunk_id][feature] = str(path.resolve())
        chunks.append(
            {
                "chunk_id": chunk_id,
                "clip_count": 1,
                "feature_files": feature_files,
            }
        )

    manifest_path = data_root / "dataset.json"
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
                            "sha256": sha256(clip_index_path),
                        }
                    },
                },
                "samples": {
                    "count": 2,
                    "clip_ids": list(token_chunks),
                    "required_features": list(
                        compact_run.RECONSTRUCTION_FEATURES
                    ),
                    "chunk_count": 2,
                    "chunks": chunks,
                },
            }
        ),
        encoding="utf-8",
    )
    run_config = {
        "dataset": {
            "repo_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
            "data_root": str(data_root),
            "selection_manifest": str(manifest_path),
            "selection_manifest_role": "pai_av_dataset_manifest",
            "selection_manifest_sha256": sha256(manifest_path),
            "selected_chunk_ids": [11, 22],
            "selected_clip_chunk_ids": token_chunks,
        }
    }
    nominal_timestamps = [4_800_000 + 100_000 * index for index in range(4)]
    token_specs = {}
    for token, chunk_id in token_chunks.items():
        source = {
            "source_reconstruction": {
                "selected_chunk_id": chunk_id,
                "feature_files": resolved_by_chunk[chunk_id],
            },
            "nominal_history_timestamps_us": nominal_timestamps,
            "camera_source_frame_indices": {
                camera: [0, 1, 2, 3] for camera in compact_run.CAMERAS
            },
            "camera_source_timestamps_us": {
                camera: nominal_timestamps for camera in compact_run.CAMERAS
            },
            "camera_exposure_intervals_us": {
                camera: [[value, value] for value in nominal_timestamps]
                for camera in compact_run.CAMERAS
            },
            "lidar_spin_indices": [0, 1, 2, 3],
            "lidar_midpoint_timestamps_us": nominal_timestamps,
            "lidar_motion_compensation": [],
        }
        token_specs[token] = compact_run.TokenSpec(token, (), source)

    source_index = compact_run.source_reconstruction_document(
        tmp_path / "run",
        run_config,
        token_specs,
        set(),
    )

    assert source_index["version"] == (
        compact_run.MULTI_CHUNK_SOURCE_INDEX_VERSION
    )
    assert source_index["selected_chunk_ids"] == [11, 22]
    assert "selected_chunk_id" not in source_index
    assert "feature_files" not in source_index
    assert source_index["successful_tokens"]["clip-a"]["selected_chunk_id"] == 11
    assert source_index["successful_tokens"]["clip-b"]["selected_chunk_id"] == 22

    annotations = [
        {
            "token": token,
            **{
                camera: [f"working/{token}/{camera}/{index}.png" for index in range(4)]
                for camera in compact_run.CAMERAS
            },
        }
        for token in token_chunks
    ]
    compacted = compact_run.compact_annotations_document(
        tmp_path / "run", token_specs, source_index, annotations
    )
    originals = {
        item["token"]: item["original_image_sources"] for item in compacted
    }
    assert originals["clip-a"]["selected_chunk_id"] == 11
    assert originals["clip-b"]["selected_chunk_id"] == 22
    assert "/chunk_11/" in originals["clip-a"]["cameras"][
        compact_run.CAMERAS[0]
    ]["archive_path"]
    assert "/chunk_22/" in originals["clip-b"]["cameras"][
        compact_run.CAMERAS[0]
    ]["archive_path"]


def test_validation_failure_preserves_every_intermediate(tmp_path: Path) -> None:
    run_dir = build_run(tmp_path)
    final_path = (
        run_dir
        / "final_images"
        / "clip-a"
        / "camera_cross_left_120fov"
        / "0.png"
    )
    final = np.asarray(Image.open(final_path).convert("RGB")).copy()
    final[0, 0] = [200, 10, 10]
    write_png(final_path, final)

    with pytest.raises(compact_run.CompactionError, match="Validation failed"):
        compact_run.compact_run(run_dir, workers=2)

    for name in (*compact_run.CLEANUP_DIRECTORIES, *compact_run.CLEANUP_FILES):
        assert (run_dir / name).exists()
    assert not (run_dir / compact_run.REPORT_FILENAME).exists()
    assert not (run_dir / compact_run.SOURCE_INDEX_FILENAME).exists()


def test_unexpected_final_artifact_fails_before_any_mutation(tmp_path: Path) -> None:
    run_dir = build_run(tmp_path)
    unexpected = run_dir / "final_images" / "clip-a" / "debug.txt"
    unexpected.write_text("not an evaluation image", encoding="utf-8")

    with pytest.raises(compact_run.CompactionError, match="Final views differ"):
        compact_run.compact_run(run_dir, workers=2)

    assert unexpected.is_file()
    for name in (*compact_run.CLEANUP_DIRECTORIES, *compact_run.CLEANUP_FILES):
        assert (run_dir / name).exists()
    assert not (run_dir / compact_run.REPORT_FILENAME).exists()
    assert not (run_dir / compact_run.SOURCE_INDEX_FILENAME).exists()


def test_validate_only_does_not_mutate_run(tmp_path: Path) -> None:
    run_dir = build_run(tmp_path)
    annotations_sha256 = sha256(run_dir / "annotations.json")

    report = compact_run.compact_run(run_dir, workers=2, validate_only=True)

    assert report["status"] == "validated_only"
    assert (run_dir / "images").is_dir()
    assert sha256(run_dir / "annotations.json") == annotations_sha256
    assert not (run_dir / compact_run.REPORT_FILENAME).exists()
    assert not (run_dir / compact_run.SOURCE_INDEX_FILENAME).exists()


def test_cleanup_pending_report_resumes_without_revalidating_deleted_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = build_run(tmp_path)
    real_cleanup = compact_run.cleanup_intermediates

    def interrupted_cleanup(path: Path) -> list[str]:
        assert compact_run.remove_path(path / compact_run.CLEANUP_DIRECTORIES[0])
        raise OSError("synthetic cleanup interruption")

    monkeypatch.setattr(compact_run, "cleanup_intermediates", interrupted_cleanup)
    with pytest.raises(OSError, match="synthetic cleanup interruption"):
        compact_run.compact_run(run_dir, workers=2)

    pending = json.loads(
        (run_dir / compact_run.REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert pending["status"] == "cleanup_pending"

    monkeypatch.setattr(compact_run, "cleanup_intermediates", real_cleanup)
    completed = compact_run.compact_run(run_dir, workers=2)
    assert completed["status"] == "complete"
    for name in (*compact_run.CLEANUP_DIRECTORIES, *compact_run.CLEANUP_FILES):
        assert not (run_dir / name).exists()
