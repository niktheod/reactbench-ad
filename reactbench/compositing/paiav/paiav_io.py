#!/usr/bin/env python3
"""Lazy, local-only access to the selected PAI-AV clips.

The dataset remains in its downloaded chunk archives.  This adapter extracts
only four selected images and four selected LiDAR spins per clip into the run
directory, then presents the records in the small frame-dictionary contract
used by the existing Blender compositor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


CAMERA_NAMES = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
POSE_SEQUENCE = (24, 27, 30, 1)
HISTORY_INTERVAL_US = 100_000
HISTORY_FRAME_COUNT = 4
ALPAMAYO_HISTORY_FRAME_COUNT = 16
ALPAMAYO_EGO_HISTORY_VERSION = "paiav_alpamayo_ego_history_v1"
EGO_VEHICLE_GEOMETRY_VERSION = "paiav_ego_vehicle_geometry_v1"
VEHICLE_DIMENSION_FIELDS = (
    "length",
    "width",
    "height",
    "rear_axle_to_bbox_center",
    "wheelbase",
    "track_width",
)
LIDAR_NAME = "lidar_top_360fov"
LOCAL_MAP_NAME = "pai-av-local"
LIDAR_CACHE_VERSION = "paiav_lidar_per_point_egomotion_v2"
ROLLING_SHUTTER_TIME_POLICY = "frame_timestamp_assumed_center_row_v1"
SOURCE_RECONSTRUCTION_VERSION = "paiav_archive_member_index_v1"
EGO_MASK_ENCODING_VERSION = "paiav_camera_ego_mask_nonzero_rgb_v1"
EGO_MASK_IMAGE_BYTES_KEY = "_paiav_ego_mask_image_png"
RECONSTRUCTION_FEATURES = (
    *CAMERA_NAMES,
    LIDAR_NAME,
    "camera_intrinsics.offline",
    "lidar_intrinsics.offline",
    "sensor_extrinsics.offline",
    "vehicle_dimensions",
    "egomotion.offline",
    "egomotion",
)


class PAIAVDataError(RuntimeError):
    """Raised when a selected clip cannot satisfy the editing contract."""


def decode_ego_mask_image_png(
    value: Any,
    resolution: tuple[int, int],
    context: str,
) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    """Decode PAI-AV's calibrated ego/self mask into a protected pixel mask."""
    if isinstance(value, memoryview):
        payload = value.tobytes()
    elif isinstance(value, bytearray):
        payload = bytes(value)
    elif isinstance(value, bytes):
        payload = value
    else:
        raise PAIAVDataError(
            f"Missing or invalid ego_mask_image_png for {context}: "
            f"expected bytes, got {type(value).__name__}"
        )
    if not payload:
        raise PAIAVDataError(f"Empty ego_mask_image_png for {context}")

    expected_width, expected_height = [int(item) for item in resolution]
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise PAIAVDataError(
                    f"ego_mask_image_png for {context} is {image.format!r}, not PNG"
                )
            if image.size != (expected_width, expected_height):
                raise PAIAVDataError(
                    f"ego_mask_image_png resolution mismatch for {context}: "
                    f"mask={image.size}, calibration={(expected_width, expected_height)}"
                )
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except PAIAVDataError:
        raise
    except (OSError, SyntaxError, ValueError) as exc:
        raise PAIAVDataError(
            f"Cannot decode ego_mask_image_png for {context}: {exc}"
        ) from exc

    protected = np.any(rgb != 0, axis=2)
    diagnostics = {
        "source_field": "camera_intrinsics.offline.ego_mask_image_png",
        "encoding": EGO_MASK_ENCODING_VERSION,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "resolution": [expected_width, expected_height],
        "protected_pixel_count": int(np.count_nonzero(protected)),
    }
    return payload, protected, diagnostics


@dataclass(frozen=True)
class FeatureFileSource:
    relative_path: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class ChunkSource:
    chunk_id: int
    feature_files: Mapping[str, FeatureFileSource]
    clip_count: int | None = None


@dataclass(frozen=True)
class SelectedClips:
    manifest_path: Path
    manifest: dict
    clip_ids: tuple[str, ...]
    clip_chunk_ids: Mapping[str, int]
    chunks: Mapping[int, ChunkSource]
    catalog_index_path: Path | None = None
    catalog_index_sha256: str | None = None

    @property
    def manifest_role(self) -> str:
        return str(self.manifest.get("document_role"))

    @property
    def chunk_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.chunks))

    @property
    def chunk_id(self) -> int | None:
        """Retain the legacy scalar accessor for single-chunk manifests."""
        if len(self.chunks) == 1:
            return next(iter(self.chunks))
        return None

    def chunk_id_for(self, clip_id: str) -> int:
        try:
            return int(self.clip_chunk_ids[str(clip_id)])
        except KeyError as exc:
            raise PAIAVDataError(
                f"Clip {clip_id!r} has no source-chunk mapping"
            ) from exc

    def chunk_ids_for(self, clip_ids: Sequence[str]) -> tuple[int, ...]:
        return tuple(sorted({self.chunk_id_for(clip_id) for clip_id in clip_ids}))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_reference(manifest_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PAIAVDataError(f"{label} path must be a nonempty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _manifest_clip_ids(values: Any, count: Any, context: str) -> tuple[str, ...]:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise PAIAVDataError(f"{context} clip IDs must be nonempty strings")
    clip_ids = tuple(values)
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise PAIAVDataError(f"{context} must contain unique clip IDs")
    if isinstance(count, bool):
        recorded_count = -1
    else:
        try:
            recorded_count = int(count)
        except (TypeError, ValueError):
            recorded_count = -1
    if recorded_count != len(clip_ids):
        raise PAIAVDataError(
            f"{context} count={recorded_count}, but contains {len(clip_ids)} IDs"
        )
    return clip_ids


def _chunk_id(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise PAIAVDataError(f"{context} chunk_id must be an integer")
    try:
        chunk_id = int(value)
    except (TypeError, ValueError) as exc:
        raise PAIAVDataError(f"{context} chunk_id must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PAIAVDataError(f"{context} chunk_id must be an integer")
    if chunk_id < 0:
        raise PAIAVDataError(f"{context} chunk_id must be nonnegative")
    return chunk_id


def _feature_file_source(value: Any, context: str) -> FeatureFileSource:
    if isinstance(value, str):
        relative_path = value
        size_bytes = None
    elif isinstance(value, Mapping):
        relative_path = value.get("path")
        size_bytes = value.get("size_bytes")
    else:
        raise PAIAVDataError(f"{context} must be a path or feature-file object")
    if not isinstance(relative_path, str) or not relative_path:
        raise PAIAVDataError(f"{context}.path must be a nonempty string")
    if size_bytes is not None and (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise PAIAVDataError(f"{context}.size_bytes must be a nonnegative integer")
    return FeatureFileSource(relative_path, size_bytes)


def _catalog_clip_chunk_ids(
    path: Path,
    document: dict,
    clip_ids: Sequence[str],
) -> tuple[dict[str, int], Path, str]:
    dataset = document.get("dataset")
    catalog_inputs = dataset.get("catalog_inputs") if isinstance(dataset, Mapping) else None
    index_record = (
        catalog_inputs.get("clip_index.parquet")
        if isinstance(catalog_inputs, Mapping)
        else None
    )
    if not isinstance(index_record, Mapping):
        raise PAIAVDataError(
            "Dataset manifest must reference dataset.catalog_inputs/clip_index.parquet"
        )
    index_path = _manifest_reference(
        path, index_record.get("path"), "Dataset clip index"
    )
    if not index_path.is_file():
        raise FileNotFoundError(f"PAI-AV catalog clip index not found: {index_path}")
    declared_size = index_record.get("size_bytes")
    if declared_size is not None and (
        not isinstance(declared_size, int)
        or isinstance(declared_size, bool)
        or declared_size != index_path.stat().st_size
    ):
        raise PAIAVDataError(
            "Dataset clip-index size differs from its manifest record"
        )
    declared_sha256 = index_record.get("sha256")
    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        raise PAIAVDataError("Dataset clip index must have a SHA-256 record")
    actual_sha256 = _file_sha256(index_path)
    if actual_sha256 != declared_sha256:
        raise PAIAVDataError(
            "Dataset clip-index SHA-256 differs from its manifest record"
        )
    try:
        frame = pd.read_parquet(index_path, columns=["chunk"])
    except Exception as exc:
        raise PAIAVDataError(f"Could not read dataset clip index: {index_path}") from exc
    if frame.index.name != "clip_id":
        if "clip_id" not in frame.columns:
            raise PAIAVDataError("Dataset clip index has no clip_id index")
        frame = frame.set_index("clip_id")
    frame.index = frame.index.map(str)
    if frame.index.has_duplicates:
        raise PAIAVDataError("Dataset clip index contains duplicate clip IDs")
    missing = sorted(set(clip_ids) - set(frame.index))
    if missing:
        raise PAIAVDataError(
            f"Dataset clips are absent from the catalog clip index: {missing[:10]}"
        )
    mapping = {
        clip_id: _chunk_id(frame.at[clip_id, "chunk"], f"Clip {clip_id}")
        for clip_id in clip_ids
    }
    return mapping, index_path, actual_sha256


def _dataset_manifest(path: Path, document: dict) -> SelectedClips:
    samples = document.get("samples")
    if not isinstance(samples, Mapping):
        raise PAIAVDataError("Dataset manifest has no samples object")
    clip_ids = _manifest_clip_ids(
        samples.get("clip_ids"), samples.get("count"), "Dataset manifest"
    )
    required_features = samples.get("required_features")
    if not isinstance(required_features, list) or any(
        not isinstance(value, str) for value in required_features
    ):
        raise PAIAVDataError("Dataset manifest required_features must be a list")
    missing_features = sorted(set(RECONSTRUCTION_FEATURES) - set(required_features))
    if missing_features:
        raise PAIAVDataError(
            f"Dataset manifest does not require features: {missing_features}"
        )
    raw_chunks = samples.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise PAIAVDataError("Dataset manifest has no chunk records")
    if samples.get("chunk_count") != len(raw_chunks):
        raise PAIAVDataError("Dataset manifest chunk_count is inconsistent")
    chunks: dict[int, ChunkSource] = {}
    for index, raw_chunk in enumerate(raw_chunks):
        context = f"Dataset chunk {index}"
        if not isinstance(raw_chunk, Mapping):
            raise PAIAVDataError(f"{context} must be an object")
        chunk_id = _chunk_id(raw_chunk.get("chunk_id"), context)
        if chunk_id in chunks:
            raise PAIAVDataError(f"Dataset manifest repeats chunk {chunk_id}")
        raw_features = raw_chunk.get("feature_files")
        if not isinstance(raw_features, Mapping):
            raise PAIAVDataError(f"{context} has no feature_files object")
        feature_files = {
            str(name): _feature_file_source(
                value, f"{context} feature {name!r}"
            )
            for name, value in raw_features.items()
            if isinstance(name, str) and name
        }
        if len(feature_files) != len(raw_features):
            raise PAIAVDataError(f"{context} feature names must be nonempty strings")
        missing = sorted(set(RECONSTRUCTION_FEATURES) - set(feature_files))
        if missing:
            raise PAIAVDataError(f"{context} lacks features: {missing}")
        clip_count = raw_chunk.get("clip_count")
        if not isinstance(clip_count, int) or isinstance(clip_count, bool) or clip_count < 1:
            raise PAIAVDataError(f"{context}.clip_count must be positive")
        chunks[chunk_id] = ChunkSource(chunk_id, feature_files, clip_count)

    clip_chunk_ids, index_path, index_sha256 = _catalog_clip_chunk_ids(
        path, document, clip_ids
    )
    undeclared = sorted(set(clip_chunk_ids.values()) - set(chunks))
    if undeclared:
        raise PAIAVDataError(
            f"Dataset clip index refers to undeclared chunks: {undeclared}"
        )
    observed_counts: dict[int, int] = {chunk_id: 0 for chunk_id in chunks}
    for chunk_id in clip_chunk_ids.values():
        observed_counts[chunk_id] += 1
    inconsistent_counts = {
        chunk_id: (chunks[chunk_id].clip_count, observed_counts[chunk_id])
        for chunk_id in chunks
        if chunks[chunk_id].clip_count != observed_counts[chunk_id]
    }
    if inconsistent_counts:
        raise PAIAVDataError(
            "Dataset chunk clip counts disagree with the catalog index: "
            f"{inconsistent_counts}"
        )
    return SelectedClips(
        path,
        document,
        clip_ids,
        clip_chunk_ids,
        chunks,
        index_path,
        index_sha256,
    )


def load_selection_manifest(path: Path) -> SelectedClips:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PAI-AV manifest not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PAIAVDataError("PAI-AV manifest must contain an object")
    role = document.get("document_role")
    if role != "pai_av_dataset_manifest":
        raise PAIAVDataError(f"Unsupported PAI-AV manifest role in {path}: {role!r}")
    return _dataset_manifest(path, document)


def resolve_chunk_feature_paths(
    data_root: Path,
    selection: SelectedClips,
    chunk_ids: Sequence[int],
    required_features: Sequence[str] = RECONSTRUCTION_FEATURES,
) -> dict[int, dict[str, Path]]:
    """Resolve and validate the declared source files for selected chunks."""
    data_root = data_root.expanduser().resolve()
    resolved: dict[int, dict[str, Path]] = {}
    for chunk_id in sorted(set(int(value) for value in chunk_ids)):
        try:
            chunk = selection.chunks[chunk_id]
        except KeyError as exc:
            raise PAIAVDataError(f"Manifest does not declare chunk {chunk_id}") from exc
        missing = sorted(set(required_features) - set(chunk.feature_files))
        if missing:
            raise PAIAVDataError(
                f"Manifest chunk {chunk_id} lacks source features: {missing}"
            )
        feature_paths: dict[str, Path] = {}
        for feature in required_features:
            source = chunk.feature_files[feature]
            relative = Path(source.relative_path)
            path = (
                relative.resolve()
                if relative.is_absolute()
                else (data_root / relative).resolve()
            )
            try:
                path.relative_to(data_root)
            except ValueError as exc:
                raise PAIAVDataError(
                    f"Chunk {chunk_id} feature path escapes the data root: "
                    f"{source.relative_path}"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(
                    f"Downloaded PAI-AV feature file is missing: {path}"
                )
            if source.size_bytes is not None and path.stat().st_size != source.size_bytes:
                raise PAIAVDataError(
                    f"Chunk {chunk_id} feature {feature} size differs from the manifest"
                )
            feature_paths[feature] = path
        resolved[chunk_id] = feature_paths
    return resolved


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PAIAVDataError(f"{source} is missing columns: {missing}")


def _read_zip_parquet(zip_path: Path, member: str, columns=None) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            payload = archive.read(member)
    except KeyError as exc:
        raise PAIAVDataError(f"Missing {member} in {zip_path}") from exc
    return pd.read_parquet(io.BytesIO(payload), columns=columns)


def _nearest_unique_indices(reference: np.ndarray, targets: np.ndarray, label: str) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if reference.ndim != 1 or reference.size == 0 or np.any(np.diff(reference) < 0):
        raise PAIAVDataError(f"{label} timestamps must be nonempty and sorted")
    right = np.searchsorted(reference, targets, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    choose_right = np.abs(reference[right] - targets) < np.abs(reference[left] - targets)
    selected = np.where(choose_right, right, left).astype(np.int64)
    if len(np.unique(selected)) != len(selected):
        raise PAIAVDataError(
            f"{label} cannot provide {len(targets)} distinct 10 Hz samples"
        )
    return selected


def _interpolate_pose_matrices(
    egomotion: pd.DataFrame,
    timestamps_us: np.ndarray,
    *,
    source: str = "egomotion.offline",
) -> np.ndarray:
    _require_columns(
        egomotion,
        ("timestamp", "qx", "qy", "qz", "qw", "x", "y", "z"),
        source,
    )
    ordered = egomotion.sort_values("timestamp").drop_duplicates("timestamp")
    source_timestamps = ordered["timestamp"].to_numpy(dtype=np.float64)
    queries = np.asarray(timestamps_us, dtype=np.float64)
    if queries.min() < source_timestamps[0] or queries.max() > source_timestamps[-1]:
        raise PAIAVDataError(
            "Requested nominal history/future timestamps fall outside "
            f"{source} coverage"
        )
    translations = np.column_stack(
        [
            np.interp(queries, source_timestamps, ordered[axis].to_numpy(dtype=float))
            for axis in ("x", "y", "z")
        ]
    )
    rotations = Rotation.from_quat(
        ordered[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    )
    interpolated_rotations = Slerp(source_timestamps, rotations)(queries).as_matrix()
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(queries), axis=0)
    matrices[:, :3, :3] = interpolated_rotations
    matrices[:, :3, 3] = translations
    return matrices


def build_alpamayo_ego_history(
    raw_egomotion: pd.DataFrame,
    current_timestamp_us: int,
) -> dict[str, Any]:
    """Build Alpamayo's native 16-step, 10 Hz history in the current ego frame."""
    current_timestamp_us = int(current_timestamp_us)
    timestamps_us = current_timestamp_us + HISTORY_INTERVAL_US * np.arange(
        -(ALPAMAYO_HISTORY_FRAME_COUNT - 1), 1, dtype=np.int64
    )
    world_from_rig = _interpolate_pose_matrices(
        raw_egomotion,
        timestamps_us,
        source="egomotion",
    )
    current_from_world = np.linalg.inv(world_from_rig[-1])
    current_from_history = current_from_world[None] @ world_from_rig
    xyz = current_from_history[:, :3, 3].copy()
    rotations = current_from_history[:, :3, :3].copy()
    if not np.isfinite(xyz).all() or not np.isfinite(rotations).all():
        raise PAIAVDataError("Alpamayo ego history contains non-finite poses")

    # Avoid retaining insignificant numerical inversion error at the current pose.
    xyz[-1] = 0.0
    rotations[-1] = np.eye(3, dtype=np.float64)
    return {
        "version": ALPAMAYO_EGO_HISTORY_VERSION,
        "source_feature": "egomotion",
        "coordinate_frame": "current_ego",
        "frame_order": "oldest_to_current",
        "history_interval_us": HISTORY_INTERVAL_US,
        "current_timestamp_us": current_timestamp_us,
        "timestamps_us": timestamps_us.tolist(),
        "ego_history_xyz": xyz.tolist(),
        "ego_history_rot": rotations.tolist(),
    }


def build_ego_vehicle_geometry(dimensions: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve native PAI-AV dimensions and derive its rear-axle footprint."""
    try:
        values = {
            name: float(dimensions[name])
            for name in VEHICLE_DIMENSION_FIELDS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PAIAVDataError(
            "vehicle_dimensions does not contain the complete numeric geometry"
        ) from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise PAIAVDataError("vehicle_dimensions contains non-finite geometry")
    for name in ("length", "width", "height", "wheelbase", "track_width"):
        if values[name] <= 0:
            raise PAIAVDataError(f"vehicle_dimensions.{name} must be positive")

    half_length = 0.5 * values["length"]
    half_width = 0.5 * values["width"]
    front_distance = values["rear_axle_to_bbox_center"] + half_length
    rear_distance = half_length - values["rear_axle_to_bbox_center"]
    if front_distance <= 0 or rear_distance <= 0:
        raise PAIAVDataError(
            "rear axle must lie inside the vehicle's longitudinal bounds"
        )
    return {
        "version": EGO_VEHICLE_GEOMETRY_VERSION,
        "source_feature": "vehicle_dimensions",
        "units": "meters",
        "dimensions_m": values,
        "rear_axle_to_front_bumper_m": front_distance,
        "rear_axle_to_rear_bumper_m": rear_distance,
        "footprint_rear_axle_m": {
            "x_min": -rear_distance,
            "x_max": front_distance,
            "y_min": -half_width,
            "y_max": half_width,
        },
    }


def _interpolate_columns(
    frame: pd.DataFrame,
    timestamps_us: np.ndarray,
    columns: Sequence[str],
) -> np.ndarray:
    _require_columns(frame, ("timestamp", *columns), "egomotion")
    ordered = frame.sort_values("timestamp").drop_duplicates("timestamp")
    source_timestamps = ordered["timestamp"].to_numpy(dtype=np.float64)
    queries = np.asarray(timestamps_us, dtype=np.float64)
    return np.column_stack(
        [
            np.interp(queries, source_timestamps, ordered[name].to_numpy(dtype=float))
            for name in columns
        ]
    )


def _pose_matrix(extrinsics_row: pd.Series) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        extrinsics_row[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    ).as_matrix()
    matrix[:3, 3] = extrinsics_row[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return matrix


def _decoded_draco_attribute(decoded: Any, name: str) -> np.ndarray:
    """Return one named Draco attribute as a one-dimensional array."""
    try:
        attribute = decoded.get_attribute_by_name(name)
    except (AttributeError, KeyError, RuntimeError) as exc:
        raise PAIAVDataError(
            f"Decoded PAI-AV LiDAR has no {name!r} attribute"
        ) from exc
    if isinstance(attribute, Mapping):
        attribute = attribute.get("data")
    values = np.asarray(attribute)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise PAIAVDataError(
            f"Decoded LiDAR attribute {name!r} has invalid shape {values.shape}"
        )
    return values


def _motion_compensate_lidar_points(
    points_lidar: np.ndarray,
    point_timestamps_us: np.ndarray,
    rig_from_lidar: np.ndarray,
    nominal_world_from_rig: np.ndarray,
    offline_egomotion: pd.DataFrame,
) -> np.ndarray:
    """Express asynchronously captured LiDAR returns in one nominal rig frame."""
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    point_timestamps_us = np.asarray(point_timestamps_us, dtype=np.int64)
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise PAIAVDataError(
            f"Decoded LiDAR points have invalid shape {points_lidar.shape}"
        )
    if point_timestamps_us.shape != (len(points_lidar),):
        raise PAIAVDataError(
            "LiDAR point/timestamp count mismatch: "
            f"points={len(points_lidar)}, timestamps={point_timestamps_us.shape}"
        )
    finite = np.isfinite(points_lidar).all(axis=1)
    if not np.any(finite):
        return np.empty((0, 3), dtype=np.float32)
    points_lidar = points_lidar[finite]
    point_timestamps_us = point_timestamps_us[finite]

    points_rig_at_capture = (
        points_lidar @ rig_from_lidar[:3, :3].T + rig_from_lidar[:3, 3]
    )
    world_from_rig_at_point = _interpolate_pose_matrices(
        offline_egomotion,
        point_timestamps_us,
    )
    points_global = np.einsum(
        "nij,nj->ni",
        world_from_rig_at_point[:, :3, :3],
        points_rig_at_capture,
    ) + world_from_rig_at_point[:, :3, 3]

    nominal_world_from_rig = np.asarray(
        nominal_world_from_rig, dtype=np.float64
    )
    points_nominal_rig = (
        points_global - nominal_world_from_rig[:3, 3]
    ) @ nominal_world_from_rig[:3, :3]
    return points_nominal_rig[
        np.isfinite(points_nominal_rig).all(axis=1)
    ].astype(np.float32)


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_png_atomic(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    try:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            temporary, compress_level=3
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _decode_video_indices(
    zip_path: Path,
    video_member: str,
    requested_indices: Sequence[int],
    output_paths: Mapping[int, Path],
) -> None:
    missing = sorted(
        int(index) for index in requested_indices if not output_paths[int(index)].is_file()
    )
    if not missing:
        return
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to decode PAI-AV camera MP4 members. "
            "Run with the configured compositing environment."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="paiav-video-") as temporary_dir:
        video_path = Path(temporary_dir) / "clip.mp4"
        try:
            with zipfile.ZipFile(zip_path) as archive, archive.open(video_member) as source:
                with video_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        except KeyError as exc:
            raise PAIAVDataError(f"Missing {video_member} in {zip_path}") from exc

        wanted = set(missing)
        decoded = set()
        with av.open(str(video_path), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in wanted:
                    _save_png_atomic(
                        output_paths[frame_index], frame.to_ndarray(format="rgb24")
                    )
                    decoded.add(frame_index)
                    if decoded == wanted:
                        break
        absent = sorted(wanted - decoded)
        if absent:
            raise PAIAVDataError(
                f"Video {video_member} did not decode requested frame indices {absent}"
            )


class _LazySceneFrames(Mapping[str, List[dict]]):
    def __init__(self, loader: "PAIAVSceneLoader") -> None:
        self.loader = loader

    def __getitem__(self, token: str) -> List[dict]:
        return self.loader.load_clip(token)

    def __iter__(self):
        return iter(self.loader.tokens)

    def __len__(self) -> int:
        return len(self.loader.tokens)


class PAIAVSceneLoader:
    """Materialize only the records needed by one editing run."""

    def __init__(
        self,
        data_root: Path,
        selection: SelectedClips,
        tokens: Sequence[str],
        run_dir: Path,
        *,
        anchor_time_s: float,
        future_timestep: int,
        ego_mask_mode: str = "refined-gls",
        ego_mask_cache: Path | None = None,
        ego_mask_model: Path | None = None,
    ) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.selection = selection
        self.tokens = tuple(str(token) for token in tokens)
        self.run_dir = run_dir.resolve()
        self.anchor_time_s = float(anchor_time_s)
        self.future_timestep = int(future_timestep)
        minimum_anchor_time_s = (
            ALPAMAYO_HISTORY_FRAME_COUNT - 1
        ) * HISTORY_INTERVAL_US / 1_000_000
        if self.anchor_time_s < minimum_anchor_time_s:
            raise ValueError(
                "--anchor-time-s must be at least "
                f"{minimum_anchor_time_s:g} to record Alpamayo ego history"
            )
        if self.future_timestep < 1:
            raise ValueError("--timestep must be positive")
        unknown = sorted(set(self.tokens) - set(selection.clip_ids))
        if unknown:
            raise PAIAVDataError(f"Tokens are not in the PAI-AV manifest: {unknown}")

        self.token_chunk_ids = {
            token: selection.chunk_id_for(token) for token in self.tokens
        }
        used_chunk_ids = selection.chunk_ids_for(self.tokens)
        self.chunk_feature_paths = resolve_chunk_feature_paths(
            self.data_root,
            selection,
            used_chunk_ids,
        )
        # Kept for callers inspecting a legacy/single-chunk loader. Internal
        # reads always route through the token-specific mapping below.
        self.feature_paths = (
            self.chunk_feature_paths[used_chunk_ids[0]]
            if len(used_chunk_ids) == 1
            else {}
        )
        tokens_by_chunk = {
            chunk_id: [
                token
                for token in self.tokens
                if self.token_chunk_ids[token] == chunk_id
            ]
            for chunk_id in used_chunk_ids
        }

        def read_calibration(feature: str, columns=None) -> pd.DataFrame:
            frames = [
                pd.read_parquet(
                    self.chunk_feature_paths[chunk_id][feature],
                    columns=columns,
                    filters=[("clip_id", "in", tokens_by_chunk[chunk_id])],
                )
                for chunk_id in used_chunk_ids
            ]
            return frames[0] if len(frames) == 1 else pd.concat(frames)

        self.camera_intrinsics = read_calibration(
            "camera_intrinsics.offline",
            columns=[
                "model_type",
                "model_parameters",
                "ego_mask_image_png",
            ],
        )
        self.sensor_extrinsics = read_calibration("sensor_extrinsics.offline")
        self.vehicle_dimensions = read_calibration("vehicle_dimensions")
        self._frames: Dict[str, List[dict]] = {}
        self._agent_inputs: Dict[str, Any] = {}
        self._alpamayo_ego_histories: Dict[str, dict] = {}
        self._clip_metadata: Dict[str, dict] = {}
        from .ego_masks import Resolver, DEFAULT_CACHE, DEFAULT_MODEL

        self.ego_masks = Resolver(
            self, ego_mask_mode,
            ego_mask_cache or DEFAULT_CACHE,
            ego_mask_model or DEFAULT_MODEL,
        )
        self.scene_frames_dicts = _LazySceneFrames(self)

    def _feature_paths_for_token(self, token: str) -> Mapping[str, Path]:
        try:
            chunk_id = self.token_chunk_ids[str(token)]
        except KeyError as exc:
            raise PAIAVDataError(f"Unknown PAI-AV clip: {token}") from exc
        return self.chunk_feature_paths[chunk_id]

    @property
    def nominal_history_timestamps_us(self) -> np.ndarray:
        anchor = int(round(self.anchor_time_s * 1_000_000))
        return anchor + HISTORY_INTERVAL_US * np.arange(
            -(HISTORY_FRAME_COUNT - 1), 1, dtype=np.int64
        )

    @property
    def nominal_future_timestamps_us(self) -> np.ndarray:
        anchor = int(round(self.anchor_time_s * 1_000_000))
        return anchor + HISTORY_INTERVAL_US * np.arange(
            1, self.future_timestep + 1, dtype=np.int64
        )

    def front_bumper_offset_m(self, token: str) -> float:
        return float(
            self.ego_vehicle_geometry(token)["rear_axle_to_front_bumper_m"]
        )

    def ego_vehicle_geometry(self, token: str) -> dict[str, Any]:
        try:
            row = self.vehicle_dimensions.loc[token]
        except KeyError as exc:
            raise PAIAVDataError(f"Missing vehicle dimensions for {token}") from exc
        if not isinstance(row, pd.Series):
            raise PAIAVDataError(f"Vehicle dimensions are not unique for {token}")
        return build_ego_vehicle_geometry(row)

    def _camera_intrinsic(self, token: str, camera: str) -> dict:
        try:
            row = self.camera_intrinsics.loc[(token, camera)]
        except KeyError as exc:
            raise PAIAVDataError(f"Missing camera intrinsics for {token}/{camera}") from exc
        if str(row["model_type"]).lower() != "ftheta":
            raise PAIAVDataError(
                f"Unsupported PAI-AV camera model {row['model_type']!r} for {camera}"
            )
        parameters = json.loads(row["model_parameters"])
        width, height = [int(value) for value in parameters["resolution"]]
        ego_mask_payload, _ego_mask, ego_mask_diagnostics = (
            decode_ego_mask_image_png(
                row["ego_mask_image_png"],
                (width, height),
                f"{token}/{camera}",
            )
        )
        if hasattr(self, "ego_masks"):
            ego_mask_payload, ego_mask_diagnostics = self.ego_masks.resolve(
                token, camera, ego_mask_payload, ego_mask_diagnostics
            )
        principal_point = np.asarray(parameters["principal_point"], dtype=np.float64)
        angle_to_pixel = np.asarray(
            parameters["angle_to_pixeldist_poly"], dtype=np.float64
        )
        # The linear coefficient gives a stable tangent-plane focal length at
        # the optical axis.  The dense remap applies the full F-theta model.
        focal = float(angle_to_pixel[1])
        intrinsic = np.array(
            [
                [focal, 0.0, principal_point[0]],
                [0.0, focal, principal_point[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return {
            "camera_model": "ftheta",
            "resolution": [width, height],
            "width": width,
            "height": height,
            "cam_intrinsic": intrinsic.tolist(),
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "principal_point": principal_point.tolist(),
            "angle_to_pixeldist_poly": angle_to_pixel.tolist(),
            "pixeldist_to_angle_poly": list(parameters["pixeldist_to_angle_poly"]),
            "max_angle_rad": float(parameters["max_angle"]),
            "shutter_type": parameters.get("shutter_type"),
            "shutter_delay_us": parameters.get("shutter_delay_us"),
            EGO_MASK_IMAGE_BYTES_KEY: ego_mask_payload,
            "ego_mask_image_sha256": ego_mask_diagnostics["sha256"],
            "ego_mask_encoding": ego_mask_diagnostics["encoding"],
            "ego_mask_provenance": ego_mask_diagnostics,
            "ego_mask_protected_pixel_count": ego_mask_diagnostics[
                "protected_pixel_count"
            ],
        }

    def _sensor_pose(self, token: str, sensor: str) -> np.ndarray:
        try:
            row = self.sensor_extrinsics.loc[(token, sensor)]
        except KeyError as exc:
            raise PAIAVDataError(f"Missing sensor extrinsics for {token}/{sensor}") from exc
        return _pose_matrix(row)

    def _read_egomotion(self, token: str, offline: bool) -> pd.DataFrame:
        feature = "egomotion.offline" if offline else "egomotion"
        member = f"{token}.{feature}.parquet"
        return _read_zip_parquet(self._feature_paths_for_token(token)[feature], member)

    def _camera_samples(
        self,
        token: str,
        camera: str,
        nominal_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[Path]]:
        zip_path = self._feature_paths_for_token(token)[camera]
        timestamp_member = f"{token}.{camera}.timestamps.parquet"
        timestamps_frame = _read_zip_parquet(zip_path, timestamp_member)
        _require_columns(timestamps_frame, ("timestamp",), timestamp_member)
        timestamps = timestamps_frame["timestamp"].to_numpy(dtype=np.int64)
        positions = _nearest_unique_indices(timestamps, nominal_timestamps, camera)
        if timestamps_frame.index.name == "frame_index":
            frame_indices = timestamps_frame.index.to_numpy(dtype=np.int64)[positions]
        else:
            frame_indices = positions
        actual_timestamps = timestamps[positions]
        paths = [
            self.run_dir
            / "source_frames"
            / token
            / camera
            / f"{history_index}.png"
            for history_index in range(HISTORY_FRAME_COUNT)
        ]
        outputs_by_source_index = {
            int(source_index): path
            for source_index, path in zip(frame_indices, paths)
        }
        _decode_video_indices(
            zip_path,
            f"{token}.{camera}.mp4",
            frame_indices,
            outputs_by_source_index,
        )
        return frame_indices, actual_timestamps, paths

    def _lidar_samples(
        self,
        token: str,
        nominal_timestamps: np.ndarray,
        world_from_rig_nominal: np.ndarray,
        offline_egomotion: pd.DataFrame,
    ) -> tuple[list[Path], np.ndarray, np.ndarray, list[dict]]:
        try:
            import DracoPy
        except ImportError as exc:
            raise RuntimeError(
                "DracoPy is required to decode PAI-AV LiDAR. Install it in "
                "the environment selected by PAIAV_PYTHON (pip install DracoPy)."
            ) from exc

        zip_path = self._feature_paths_for_token(token)[LIDAR_NAME]
        member = f"{token}.{LIDAR_NAME}.parquet"
        with tempfile.TemporaryDirectory(prefix="paiav-lidar-") as temporary_dir:
            parquet_path = Path(temporary_dir) / "lidar.parquet"
            try:
                with zipfile.ZipFile(zip_path) as archive, archive.open(member) as source:
                    with parquet_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            except KeyError as exc:
                raise PAIAVDataError(f"Missing {member} in {zip_path}") from exc
            lidar = pd.read_parquet(
                parquet_path,
                columns=[
                    "spin_index",
                    "spin_start_timestamp",
                    "spin_end_timestamp",
                    "draco_encoded_pointcloud",
                ],
            )

        _require_columns(
            lidar,
            (
                "spin_index",
                "spin_start_timestamp",
                "spin_end_timestamp",
                "draco_encoded_pointcloud",
            ),
            member,
        )
        midpoints = np.rint(
            0.5
            * (
                lidar["spin_start_timestamp"].to_numpy(dtype=np.float64)
                + lidar["spin_end_timestamp"].to_numpy(dtype=np.float64)
            )
        ).astype(np.int64)
        positions = _nearest_unique_indices(midpoints, nominal_timestamps, LIDAR_NAME)
        selected_midpoints = midpoints[positions]
        selected_spin_indices = lidar["spin_index"].to_numpy(dtype=np.int64)[positions]
        rig_from_lidar = self._sensor_pose(token, LIDAR_NAME)
        output_paths = [
            self.run_dir / "source_lidar" / token / f"{index}.npy"
            for index in range(HISTORY_FRAME_COUNT)
        ]
        diagnostics = []
        for history_index, (row_position, output_path) in enumerate(
            zip(positions, output_paths)
        ):
            row = lidar.iloc[int(row_position)]
            spin_index = int(row["spin_index"])
            nominal_timestamp_us = int(nominal_timestamps[history_index])
            sidecar_path = output_path.with_suffix(".json")
            cached = None
            if output_path.is_file() and sidecar_path.is_file():
                try:
                    cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    cached = None
            if (
                isinstance(cached, dict)
                and cached.get("cache_version") == LIDAR_CACHE_VERSION
                and cached.get("spin_index") == spin_index
                and cached.get("nominal_timestamp_us") == nominal_timestamp_us
            ):
                diagnostics.append(cached)
                continue
            encoded = row["draco_encoded_pointcloud"]
            decoded = DracoPy.decode(bytes(encoded))
            points_lidar = np.asarray(decoded.points, dtype=np.float64)
            point_timestamps_us = _decoded_draco_attribute(decoded, "timestamp")
            if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
                raise PAIAVDataError(
                    f"Decoded LiDAR has invalid shape {points_lidar.shape} for {token}"
                )
            if len(point_timestamps_us) != len(points_lidar):
                raise PAIAVDataError(
                    f"Decoded LiDAR point/timestamp count mismatch for {token}"
                )
            spin_start_us = int(row["spin_start_timestamp"])
            spin_end_us = int(row["spin_end_timestamp"])
            timestamp_min_us = int(np.min(point_timestamps_us))
            timestamp_max_us = int(np.max(point_timestamps_us))
            if (
                timestamp_min_us < spin_start_us - 5_000
                or timestamp_max_us > spin_end_us + 5_000
            ):
                raise PAIAVDataError(
                    "Decoded LiDAR timestamps fall outside their spin interval: "
                    f"points={timestamp_min_us}..{timestamp_max_us}, "
                    f"spin={spin_start_us}..{spin_end_us}"
                )
            points_nominal_rig = _motion_compensate_lidar_points(
                points_lidar,
                point_timestamps_us,
                rig_from_lidar,
                world_from_rig_nominal[history_index],
                offline_egomotion,
            )
            if len(points_nominal_rig) < 100:
                raise PAIAVDataError(
                    f"Decoded LiDAR spin has only {len(points_nominal_rig)} finite points"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(f".{output_path.name}.tmp")
            try:
                with temporary.open("wb") as handle:
                    np.save(handle, points_nominal_rig, allow_pickle=False)
                temporary.replace(output_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            diagnostic = {
                "cache_version": LIDAR_CACHE_VERSION,
                "motion_compensation": "egomotion_interpolated_per_point_timestamp",
                "spin_index": spin_index,
                "spin_start_timestamp_us": spin_start_us,
                "spin_end_timestamp_us": spin_end_us,
                "point_timestamp_min_us": timestamp_min_us,
                "point_timestamp_max_us": timestamp_max_us,
                "nominal_timestamp_us": nominal_timestamp_us,
                "point_count": int(len(points_nominal_rig)),
            }
            _write_json_atomic(sidecar_path, diagnostic)
            diagnostics.append(diagnostic)
        return output_paths, selected_spin_indices, selected_midpoints, diagnostics

    def _make_agent_input(
        self,
        nominal_timestamps: np.ndarray,
        world_from_rig: np.ndarray,
        raw_egomotion: pd.DataFrame,
    ) -> Any:
        velocities = _interpolate_columns(
            raw_egomotion, nominal_timestamps, ("vx", "vy", "vz")
        )
        accelerations = _interpolate_columns(
            raw_egomotion, nominal_timestamps, ("ax", "ay", "az")
        )
        final_from_world = np.linalg.inv(world_from_rig[-1])
        relative = final_from_world[None] @ world_from_rig
        statuses = []
        for index in range(HISTORY_FRAME_COUNT):
            yaw = math.atan2(relative[index, 1, 0], relative[index, 0, 0])
            statuses.append(
                SimpleNamespace(
                    ego_velocity=velocities[index, :2],
                    ego_acceleration=accelerations[index, :2],
                    ego_pose=np.array(
                        [relative[index, 0, 3], relative[index, 1, 3], yaw],
                        dtype=np.float64,
                    ),
                    driving_command="unknown",
                )
            )
        return SimpleNamespace(ego_statuses=statuses)

    def load_clip(self, token: str) -> List[dict]:
        if token in self._frames:
            return self._frames[token]
        if token not in self.tokens:
            raise KeyError(token)

        history_timestamps = self.nominal_history_timestamps_us
        future_timestamps = self.nominal_future_timestamps_us
        all_timestamps = np.concatenate([history_timestamps, future_timestamps])
        offline_egomotion = self._read_egomotion(token, offline=True)
        raw_egomotion = self._read_egomotion(token, offline=False)
        world_from_rig = _interpolate_pose_matrices(offline_egomotion, all_timestamps)

        camera_samples = {}
        for camera in CAMERA_NAMES:
            frame_indices, actual_timestamps, paths = self._camera_samples(
                token, camera, history_timestamps
            )
            actual_camera_poses = _interpolate_pose_matrices(
                offline_egomotion, actual_timestamps
            )
            intrinsic = self._camera_intrinsic(token, camera)
            rig_from_camera = self._sensor_pose(token, camera)
            shutter_delay_us = int(intrinsic.get("shutter_delay_us") or 0)
            rolling_shutter = (
                str(intrinsic.get("shutter_type", "")).upper()
                == "ROLLING_TOP_TO_BOTTOM"
                and shutter_delay_us > 0
            )
            if rolling_shutter:
                # PAI-AV publishes one timestamp per image. Treat it as the
                # center-row time and model the calibrated readout symmetrically
                # around it; the static Blender camera also uses this center pose.
                exposure_start_timestamps = (
                    actual_timestamps - shutter_delay_us // 2
                )
                exposure_end_timestamps = (
                    exposure_start_timestamps + shutter_delay_us
                )
                exposure_start_poses = _interpolate_pose_matrices(
                    offline_egomotion, exposure_start_timestamps
                )
                exposure_end_poses = _interpolate_pose_matrices(
                    offline_egomotion, exposure_end_timestamps
                )
            else:
                exposure_start_timestamps = actual_timestamps.copy()
                exposure_end_timestamps = actual_timestamps.copy()
                exposure_start_poses = actual_camera_poses
                exposure_end_poses = actual_camera_poses
            records = []
            for history_index in range(HISTORY_FRAME_COUNT):
                # Express the camera at its real exposure time in the nominal
                # rig frame used for this 10 Hz timestep.
                nominal_rig_from_camera = (
                    np.linalg.inv(world_from_rig[history_index])
                    @ actual_camera_poses[history_index]
                    @ rig_from_camera
                )
                nominal_rig_from_camera_start = (
                    np.linalg.inv(world_from_rig[history_index])
                    @ exposure_start_poses[history_index]
                    @ rig_from_camera
                )
                nominal_rig_from_camera_end = (
                    np.linalg.inv(world_from_rig[history_index])
                    @ exposure_end_poses[history_index]
                    @ rig_from_camera
                )
                records.append(
                    {
                        **intrinsic,
                        "sensor2lidar_rotation": nominal_rig_from_camera[
                            :3, :3
                        ].tolist(),
                        "sensor2lidar_translation": nominal_rig_from_camera[
                            :3, 3
                        ].tolist(),
                        "data_path": str(paths[history_index].resolve()),
                        "source_frame_index": int(frame_indices[history_index]),
                        "source_timestamp_us": int(actual_timestamps[history_index]),
                        "nominal_timestamp_us": int(history_timestamps[history_index]),
                        "rolling_shutter_time_policy": ROLLING_SHUTTER_TIME_POLICY,
                        "exposure_start_timestamp_us": int(
                            exposure_start_timestamps[history_index]
                        ),
                        "exposure_end_timestamp_us": int(
                            exposure_end_timestamps[history_index]
                        ),
                        "rolling_shutter_sensor2lidar_start_rotation": (
                            nominal_rig_from_camera_start[:3, :3].tolist()
                        ),
                        "rolling_shutter_sensor2lidar_start_translation": (
                            nominal_rig_from_camera_start[:3, 3].tolist()
                        ),
                        "rolling_shutter_sensor2lidar_end_rotation": (
                            nominal_rig_from_camera_end[:3, :3].tolist()
                        ),
                        "rolling_shutter_sensor2lidar_end_translation": (
                            nominal_rig_from_camera_end[:3, 3].tolist()
                        ),
                    }
                )
            camera_samples[camera] = records

        (
            lidar_paths,
            spin_indices,
            lidar_midpoints,
            lidar_motion_diagnostics,
        ) = self._lidar_samples(
            token,
            history_timestamps,
            world_from_rig[:HISTORY_FRAME_COUNT],
            offline_egomotion,
        )

        frames = []
        for history_index in range(HISTORY_FRAME_COUNT):
            frames.append(
                {
                    "timestamp": int(history_timestamps[history_index]),
                    "ego2global": world_from_rig[history_index],
                    # The adapter uses the nominal vehicle/rig frame as the
                    # compositor's common LiDAR coordinate frame.
                    "lidar2global": world_from_rig[history_index],
                    "map_location": LOCAL_MAP_NAME,
                    "cams": {
                        camera: camera_samples[camera][history_index]
                        for camera in CAMERA_NAMES
                    },
                    "_paiav_lidar_xyz_path": str(lidar_paths[history_index]),
                    "_paiav_lidar_spin_index": int(spin_indices[history_index]),
                    "_paiav_lidar_timestamp_us": int(lidar_midpoints[history_index]),
                }
            )
        for future_index, timestamp in enumerate(future_timestamps):
            pose_index = HISTORY_FRAME_COUNT + future_index
            frames.append(
                {
                    "timestamp": int(timestamp),
                    "ego2global": world_from_rig[pose_index],
                    "lidar2global": world_from_rig[pose_index],
                    "map_location": LOCAL_MAP_NAME,
                    "cams": {},
                }
            )

        self._agent_inputs[token] = self._make_agent_input(
            history_timestamps,
            world_from_rig[:HISTORY_FRAME_COUNT],
            raw_egomotion,
        )
        self._alpamayo_ego_histories[token] = build_alpamayo_ego_history(
            raw_egomotion,
            int(history_timestamps[-1]),
        )
        ego_vehicle_geometry = self.ego_vehicle_geometry(token)
        token_chunk_id = self.token_chunk_ids[token]
        token_feature_paths = self._feature_paths_for_token(token)
        self._clip_metadata[token] = {
            "clip_id": token,
            "source_reconstruction": {
                "version": SOURCE_RECONSTRUCTION_VERSION,
                "selection_manifest": str(self.selection.manifest_path),
                "manifest_role": self.selection.manifest_role,
                "selected_chunk_id": token_chunk_id,
                "feature_files": {
                    feature: str(path)
                    for feature, path in sorted(token_feature_paths.items())
                },
                "camera_members": {
                    camera: {
                        "video": f"{token}.{camera}.mp4",
                        "timestamps": f"{token}.{camera}.timestamps.parquet",
                    }
                    for camera in CAMERA_NAMES
                },
                "lidar_member": f"{token}.{LIDAR_NAME}.parquet",
                "egomotion_members": {
                    "egomotion.offline": f"{token}.egomotion.offline.parquet",
                    "egomotion": f"{token}.egomotion.parquet",
                },
            },
            "nominal_history_timestamps_us": history_timestamps.tolist(),
            "nominal_future_timestamps_us": future_timestamps.tolist(),
            "history_interval_us": HISTORY_INTERVAL_US,
            "camera_source_frame_indices": {
                camera: [
                    int(record["source_frame_index"])
                    for record in camera_samples[camera]
                ]
                for camera in CAMERA_NAMES
            },
            "camera_source_timestamps_us": {
                camera: [
                    int(record["source_timestamp_us"])
                    for record in camera_samples[camera]
                ]
                for camera in CAMERA_NAMES
            },
            "camera_exposure_intervals_us": {
                camera: [
                    [
                        int(record["exposure_start_timestamp_us"]),
                        int(record["exposure_end_timestamp_us"]),
                    ]
                    for record in camera_samples[camera]
                ]
                for camera in CAMERA_NAMES
            },
            "camera_ego_masks": {
                camera: camera_samples[camera][0]["ego_mask_provenance"]
                for camera in CAMERA_NAMES
            },
            "rolling_shutter_time_policy": ROLLING_SHUTTER_TIME_POLICY,
            "lidar_spin_indices": spin_indices.tolist(),
            "lidar_midpoint_timestamps_us": lidar_midpoints.tolist(),
            "lidar_motion_compensation": lidar_motion_diagnostics,
            "ego_vehicle_geometry": ego_vehicle_geometry,
            "front_bumper_offset_m": ego_vehicle_geometry[
                "rear_axle_to_front_bumper_m"
            ],
        }
        self._frames[token] = frames
        return frames

    def release_clip(self, token: str) -> None:
        """Release state once a complete CPU preparation checkpoint is saved."""
        for cache in (self._frames, self._agent_inputs, self._alpamayo_ego_histories,
                      self._clip_metadata, self.ego_masks.entries):
            cache.pop(token, None)

    def get_agent_input_from_token(self, token: str) -> Any:
        self.load_clip(token)
        return self._agent_inputs[token]

    def alpamayo_ego_history(self, token: str) -> dict:
        self.load_clip(token)
        return self._alpamayo_ego_histories[token]

    def clip_metadata(self, token: str) -> dict:
        self.load_clip(token)
        return self._clip_metadata[token]


def load_lidar_xyz(_data_root: Path, _split: str, frame: dict) -> np.ndarray:
    """Load a selected, nominal-rig-aligned point cloud for the compositor."""
    path_value = frame.get("_paiav_lidar_xyz_path")
    if not path_value:
        raise PAIAVDataError("PAI-AV history frame has no materialized LiDAR path")
    path = Path(path_value)
    if not path.is_file():
        raise PAIAVDataError(f"Materialized PAI-AV LiDAR does not exist: {path}")
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise PAIAVDataError(f"Invalid materialized LiDAR shape {points.shape}: {path}")
    return np.asarray(points, dtype=np.float64)


def frame_source_path(_data_root: Path, _split: str, frame: dict, camera: str) -> Path:
    try:
        return Path(frame["cams"][camera]["data_path"])
    except KeyError as exc:
        raise PAIAVDataError(f"Frame has no source image for camera {camera}") from exc
