"""Loader for compact ReactPAI compositor runs (edited images, annotations, original sources).

Original camera histories are decoded on demand from the downloaded PAI-AV
chunk archives into a temporary directory. Edited histories always come from
the compact run's ``final_images`` tree; compositor intermediates are never
used as model inputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image

from ..scoring.geometry import project_acceleration_onto_velocity


CAMERA_SPECS = (
    ("camera_cross_left_120fov", 0),
    ("camera_front_wide_120fov", 1),
    ("camera_cross_right_120fov", 2),
    ("camera_front_tele_30fov", 6),
)
CAMERA_NAMES = tuple(name for name, _ in CAMERA_SPECS)
FRAMES_PER_CAMERA = 4
HISTORY_STEPS = 16
HISTORY_INTERVAL_US = 100_000
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
SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SOURCE_INDEX_V1 = "paiav_source_reconstruction_v1"
SOURCE_INDEX_MULTICHUNK_V2 = "paiav_source_reconstruction_v2_multichunk"
SUPPORTED_SOURCE_INDEX_VERSIONS = frozenset(
    {SOURCE_INDEX_V1, SOURCE_INDEX_MULTICHUNK_V2}
)


class PAIAVContractError(RuntimeError):
    """Raised when a compact run cannot satisfy the evaluation contract."""


@dataclass(frozen=True)
class CameraSource:
    camera: str
    archive_feature: str
    archive_path: Path
    video_member: str
    timestamps_member: str
    frame_indices: tuple[int, ...]
    frame_timestamps_us: tuple[int, ...]
    nominal_timestamps_us: tuple[int, ...]


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise PAIAVContractError(f"Invalid JSON in {label} {path}: {exc}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PAIAVContractError(f"{label} must be an object")
    return value


def _integer_tuple(value: Any, length: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise PAIAVContractError(f"{label} must contain {length} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise PAIAVContractError(f"{label} must contain only integers")
    return tuple(int(item) for item in value)


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    try:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            temporary, compress_level=3
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def decode_archive_video_frames(
    archive_path: Path,
    video_member: str,
    frame_indices: Sequence[int],
    output_paths: Sequence[Path],
) -> None:
    """Decode only selected MP4 frames from one PAI-AV camera archive."""

    if len(frame_indices) != len(output_paths):
        raise ValueError("frame_indices and output_paths must have equal length")
    outputs_by_index = {
        int(frame_index): Path(output_path)
        for frame_index, output_path in zip(frame_indices, output_paths)
    }
    if len(outputs_by_index) != len(frame_indices):
        raise PAIAVContractError(
            f"Duplicate source frame indices requested from {video_member}"
        )
    missing = {
        frame_index
        for frame_index, output_path in outputs_by_index.items()
        if not output_path.is_file()
    }
    if not missing:
        return
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to decode PAI-AV camera archives in the "
            "Alpamayo environment"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="alpamayo-paiav-video-") as temp:
        video_path = Path(temp) / "clip.mp4"
        try:
            with zipfile.ZipFile(archive_path) as archive:
                with archive.open(video_member) as source:
                    with video_path.open("wb") as destination:
                        shutil.copyfileobj(
                            source, destination, length=8 * 1024 * 1024
                        )
        except KeyError as exc:
            raise PAIAVContractError(
                f"Missing {video_member} in {archive_path}"
            ) from exc

        decoded: set[int] = set()
        with av.open(str(video_path), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in missing:
                    _save_png(
                        outputs_by_index[frame_index],
                        frame.to_ndarray(format="rgb24"),
                    )
                    decoded.add(frame_index)
                    if decoded == missing:
                        break
        absent = sorted(missing - decoded)
        if absent:
            raise PAIAVContractError(
                f"Video {video_member} did not decode requested frames {absent}"
            )


def _pose_matrices(egomotion: Any, timestamps_us: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required to reconstruct legacy PAI-AV ego history"
        ) from exc
    required = {"timestamp", "qx", "qy", "qz", "qw", "x", "y", "z"}
    missing = sorted(required - set(egomotion.columns))
    if missing:
        raise PAIAVContractError(f"Raw egomotion is missing columns: {missing}")
    ordered = egomotion.sort_values("timestamp").drop_duplicates("timestamp")
    source_times = ordered["timestamp"].to_numpy(dtype=np.float64)
    queries = np.asarray(timestamps_us, dtype=np.float64)
    if (
        source_times.size < 2
        or queries.min() < source_times[0]
        or queries.max() > source_times[-1]
    ):
        raise PAIAVContractError(
            "Raw egomotion does not cover Alpamayo's 1.5-second history"
        )
    translations = np.column_stack(
        [
            np.interp(
                queries,
                source_times,
                ordered[axis].to_numpy(dtype=np.float64),
            )
            for axis in ("x", "y", "z")
        ]
    )
    rotations = Rotation.from_quat(
        ordered[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    )
    matrices = np.repeat(np.eye(4)[None], len(queries), axis=0)
    matrices[:, :3, :3] = Slerp(source_times, rotations)(queries).as_matrix()
    matrices[:, :3, 3] = translations
    return matrices


def build_alpamayo_ego_history(
    raw_egomotion: Any, current_timestamp_us: int
) -> dict[str, Any]:
    """Reconstruct the native history used by legacy compact runs."""

    current_timestamp_us = int(current_timestamp_us)
    timestamps = current_timestamp_us + HISTORY_INTERVAL_US * np.arange(
        -(HISTORY_STEPS - 1), 1, dtype=np.int64
    )
    world_from_rig = _pose_matrices(raw_egomotion, timestamps)
    current_from_history = np.linalg.inv(world_from_rig[-1])[None] @ world_from_rig
    xyz = current_from_history[:, :3, 3].copy()
    rotations = current_from_history[:, :3, :3].copy()
    xyz[-1] = 0.0
    rotations[-1] = np.eye(3)
    return {
        "version": ALPAMAYO_EGO_HISTORY_VERSION,
        "source_feature": "egomotion",
        "coordinate_frame": "current_ego",
        "frame_order": "oldest_to_current",
        "history_interval_us": HISTORY_INTERVAL_US,
        "current_timestamp_us": current_timestamp_us,
        "timestamps_us": timestamps.tolist(),
        "ego_history_xyz": xyz.tolist(),
        "ego_history_rot": rotations.tolist(),
    }


def build_ego_vehicle_geometry(dimensions: Mapping[str, Any]) -> dict[str, Any]:
    """Build a rear-axle footprint while retaining all native dimensions."""

    try:
        values = {
            field: float(dimensions[field]) for field in VEHICLE_DIMENSION_FIELDS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PAIAVContractError(
            "vehicle_dimensions lacks the complete numeric geometry"
        ) from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise PAIAVContractError("vehicle_dimensions contains non-finite values")
    for field in ("length", "width", "height", "wheelbase", "track_width"):
        if values[field] <= 0:
            raise PAIAVContractError(f"vehicle_dimensions.{field} must be positive")
    half_length = values["length"] / 2.0
    half_width = values["width"] / 2.0
    front = values["rear_axle_to_bbox_center"] + half_length
    rear = half_length - values["rear_axle_to_bbox_center"]
    if front <= 0 or rear <= 0:
        raise PAIAVContractError("Rear axle must lie inside the vehicle footprint")
    return {
        "version": EGO_VEHICLE_GEOMETRY_VERSION,
        "source_feature": "vehicle_dimensions",
        "units": "meters",
        "dimensions_m": values,
        "rear_axle_to_front_bumper_m": front,
        "rear_axle_to_rear_bumper_m": rear,
        "footprint_rear_axle_m": {
            "x_min": -rear,
            "x_max": front,
            "y_min": -half_width,
            "y_max": half_width,
        },
    }


def validate_ego_history(history: Mapping[str, Any]) -> dict[str, Any]:
    if history.get("version") != ALPAMAYO_EGO_HISTORY_VERSION:
        raise PAIAVContractError(
            f"Unsupported Alpamayo ego-history version: {history.get('version')!r}"
        )
    if history.get("coordinate_frame") != "current_ego" or history.get(
        "frame_order"
    ) != "oldest_to_current":
        raise PAIAVContractError("Ego history has an invalid frame contract")
    if int(history.get("history_interval_us", -1)) != HISTORY_INTERVAL_US:
        raise PAIAVContractError("Ego history is not sampled at 10 Hz")
    timestamps = np.asarray(history.get("timestamps_us"), dtype=np.int64)
    xyz = np.asarray(history.get("ego_history_xyz"), dtype=np.float64)
    rotations = np.asarray(history.get("ego_history_rot"), dtype=np.float64)
    if timestamps.shape != (HISTORY_STEPS,) or not np.all(
        np.diff(timestamps) == HISTORY_INTERVAL_US
    ):
        raise PAIAVContractError("Ego history must contain sixteen 10 Hz timestamps")
    if xyz.shape != (HISTORY_STEPS, 3) or rotations.shape != (
        HISTORY_STEPS,
        3,
        3,
    ):
        raise PAIAVContractError("Ego history has invalid translation/rotation shape")
    if not np.isfinite(xyz).all() or not np.isfinite(rotations).all():
        raise PAIAVContractError("Ego history contains non-finite values")
    if not np.allclose(xyz[-1], 0.0, atol=1e-5) or not np.allclose(
        rotations[-1], np.eye(3), atol=1e-5
    ):
        raise PAIAVContractError("Current ego-history pose is not zero/identity")
    if int(history.get("current_timestamp_us", -1)) != int(timestamps[-1]):
        raise PAIAVContractError("Current ego-history timestamp is inconsistent")
    return dict(history)


def validate_ego_vehicle_geometry(
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    if geometry.get("version") != EGO_VEHICLE_GEOMETRY_VERSION:
        raise PAIAVContractError(
            f"Unsupported ego-geometry version: {geometry.get('version')!r}"
        )
    rebuilt = build_ego_vehicle_geometry(
        _mapping(geometry.get("dimensions_m"), "geometry.dimensions_m")
    )
    for field in (
        "rear_axle_to_front_bumper_m",
        "rear_axle_to_rear_bumper_m",
    ):
        if not math.isclose(
            float(geometry.get(field, math.nan)),
            float(rebuilt[field]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise PAIAVContractError(f"Ego geometry has inconsistent {field}")
    footprint = _mapping(
        geometry.get("footprint_rear_axle_m"), "geometry footprint"
    )
    for field, expected in rebuilt["footprint_rear_axle_m"].items():
        if not math.isclose(
            float(footprint.get(field, math.nan)),
            float(expected),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise PAIAVContractError(
                f"Ego geometry footprint has inconsistent {field}"
            )
    return dict(geometry)


FrameDecoder = Callable[[Path, str, Sequence[int], Sequence[Path]], None]


class PAIAVCompactRun:
    """Validated access to one compact PAI-AV compositor run."""

    def __init__(
        self,
        run_dir: Path,
        *,
        frame_decoder: FrameDecoder = decode_archive_video_frames,
        validate_archive_members: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        if not self.run_dir.is_dir():
            raise NotADirectoryError(f"PAI-AV run directory does not exist: {self.run_dir}")
        self.frame_decoder = frame_decoder
        self.final_images_dir = self.run_dir / "final_images"
        self.metadata_dir = self.run_dir / "metadata"
        if not self.final_images_dir.is_dir() or not self.metadata_dir.is_dir():
            raise PAIAVContractError(
                "Compact run must contain final_images/ and metadata/"
            )

        self.run_config_path = self.run_dir / "run_config.json"
        self.annotations_path = self.run_dir / "annotations.json"
        self.source_index_path = self.run_dir / "source_reconstruction.json"
        self.compaction_report_path = self.run_dir / "compaction_report.json"
        self.run_config = _mapping(
            _load_json(self.run_config_path, "run configuration"),
            "run configuration",
        )
        self.compaction_report = _mapping(
            _load_json(self.compaction_report_path, "compaction report"),
            "compaction report",
        )
        self.source_index = _mapping(
            _load_json(self.source_index_path, "source reconstruction index"),
            "source reconstruction index",
        )
        annotations_value = _load_json(self.annotations_path, "compact annotations")
        if not isinstance(annotations_value, list) or not annotations_value:
            raise PAIAVContractError("annotations.json must be a nonempty list")
        if self.compaction_report.get("status") != "complete":
            raise PAIAVContractError("Compaction report status must be complete")
        self.source_index_version = self.source_index.get("version")
        if self.source_index.get("document_role") != (
            "paiav_source_reconstruction_index"
        ) or self.source_index_version not in SUPPORTED_SOURCE_INDEX_VERSIONS:
            raise PAIAVContractError("Unsupported source reconstruction index")
        self._verify_manifest_hashes()

        self.annotations: dict[str, dict[str, Any]] = {}
        for index, annotation_value in enumerate(annotations_value):
            annotation = _mapping(annotation_value, f"annotation {index}")
            token = annotation.get("token")
            if not isinstance(token, str) or not token or token in self.annotations:
                raise PAIAVContractError(
                    f"Annotation {index} has an invalid or duplicate token"
                )
            self.annotations[token] = dict(annotation)
        self.token_sources = dict(
            _mapping(
                self.source_index.get("successful_tokens"),
                "source successful_tokens",
            )
        )
        if set(self.annotations) != set(self.token_sources):
            raise PAIAVContractError(
                "Annotation tokens differ from source-reconstruction tokens"
            )
        final_tokens = {
            path.name for path in self.final_images_dir.iterdir() if path.is_dir()
        }
        if final_tokens != set(self.annotations):
            raise PAIAVContractError(
                "final_images token directories differ from compact annotations"
            )
        configured = self.run_config.get("selected_clip_ids")
        order = (
            [token for token in configured if token in self.annotations]
            if isinstance(configured, list)
            else []
        )
        order.extend(sorted(set(self.annotations) - set(order)))
        self.tokens = tuple(order)

        self.feature_records: dict[str, Any] = {}
        self.chunk_feature_records: dict[str, Any] = {}
        if self.source_index_version == SOURCE_INDEX_V1:
            self.feature_records = dict(
                _mapping(
                    self.source_index.get("feature_files"),
                    "source feature_files",
                )
            )
        else:
            self.chunk_feature_records = dict(
                _mapping(
                    self.source_index.get("chunk_feature_files"),
                    "source chunk_feature_files",
                )
            )
        self._feature_path_cache: dict[tuple[int | None, str], Path] = {}
        self.feature_paths = (
            {
                feature: self._resolve_feature_path(feature)
                for feature in (*CAMERA_NAMES, "egomotion", "vehicle_dimensions")
            }
            if self.source_index_version == SOURCE_INDEX_V1
            else {}
        )
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._history_cache: dict[str, tuple[dict[str, Any], str]] = {}
        self._geometry_cache: dict[str, tuple[dict[str, Any], str]] = {}
        self._vehicle_dimensions: Any = None
        self._vehicle_dimensions_by_path: dict[Path, Any] = {}
        self._camera_source_cache: dict[str, dict[str, CameraSource]] = {}
        for token in self.tokens:
            for feature in (*CAMERA_NAMES, "egomotion", "vehicle_dimensions"):
                self._resolve_feature_path(feature, token=token)
        if validate_archive_members:
            self._validate_camera_archives()

    def _verify_manifest_hashes(self) -> None:
        source_hash = self.compaction_report.get("source_reconstruction_sha256")
        if not isinstance(source_hash, str) or source_hash != file_sha256(
            self.source_index_path
        ):
            raise PAIAVContractError("source_reconstruction.json SHA-256 mismatch")
        annotation_hash = self.compaction_report.get("annotations_sha256")
        if annotation_hash is not None and annotation_hash != file_sha256(
            self.annotations_path
        ):
            raise PAIAVContractError("annotations.json SHA-256 mismatch")

    def _token_chunk_id(self, token: str) -> int | None:
        source = _mapping(self.token_sources[token], f"source for {token}")
        value = source.get("selected_chunk_id")
        if self.source_index_version == SOURCE_INDEX_V1 and value is None:
            value = self.source_index.get("selected_chunk_id")
        if value is None and self.source_index_version == SOURCE_INDEX_V1:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise PAIAVContractError(f"Invalid source chunk for {token}: {value!r}")
        return value

    def _resolve_feature_path(self, feature: str, *, token: str | None = None) -> Path:
        chunk_id = self._token_chunk_id(token) if token is not None else None
        cache_key = (chunk_id, feature)
        cached = self._feature_path_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.source_index_version == SOURCE_INDEX_MULTICHUNK_V2:
            if token is None or chunk_id is None:
                raise PAIAVContractError(
                    f"Multi-chunk feature {feature} requires a source token"
                )
            chunk_records = _mapping(
                self.chunk_feature_records.get(str(chunk_id)),
                f"chunk_feature_files.{chunk_id}",
            )
            feature_records = chunk_records
            record_label = f"chunk_feature_files.{chunk_id}.{feature}"
        else:
            feature_records = self.feature_records
            record_label = f"feature_files.{feature}"
        record = _mapping(
            feature_records.get(feature), record_label
        )
        candidates: list[Path] = []
        resolved_value = record.get("resolved_path")
        if isinstance(resolved_value, str):
            candidates.append(Path(resolved_value).expanduser())
        relative_value = record.get("relative_path")
        data_root_value = self.source_index.get("data_root")
        if isinstance(relative_value, str) and isinstance(data_root_value, str):
            candidates.append(Path(data_root_value).expanduser() / relative_value)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                self._feature_path_cache[cache_key] = resolved
                return resolved
        raise FileNotFoundError(
            f"Local PAI-AV source feature is missing for {feature}: {candidates}"
        )

    def _archive(self, path: Path) -> zipfile.ZipFile:
        archive = self._archives.get(path)
        if archive is None:
            archive = zipfile.ZipFile(path)
            self._archives[path] = archive
        return archive

    def camera_sources(self, token: str) -> dict[str, CameraSource]:
        if token in self._camera_source_cache:
            return self._camera_source_cache[token]
        annotation = self.annotation(token)
        token_source = _mapping(self.token_sources[token], f"source for {token}")
        index_sources = _mapping(
            token_source.get("camera_sources"), f"camera sources for {token}"
        )
        improved_value = annotation.get("original_image_sources")
        improved_cameras: Mapping[str, Any] = {}
        if improved_value is not None:
            improved = _mapping(improved_value, f"original_image_sources for {token}")
            improved_cameras = _mapping(
                improved.get("cameras"),
                f"original_image_sources.cameras for {token}",
            )
            improved_chunk_id = improved.get("selected_chunk_id")
            token_chunk_id = self._token_chunk_id(token)
            if (
                improved_chunk_id is not None
                and improved_chunk_id != token_chunk_id
            ):
                raise PAIAVContractError(
                    f"Annotation/source chunk mismatch for {token}"
                )
        result: dict[str, CameraSource] = {}
        for camera in CAMERA_NAMES:
            indexed = _mapping(index_sources.get(camera), f"source {token}/{camera}")
            improved = (
                _mapping(improved_cameras.get(camera), f"annotation source {token}/{camera}")
                if improved_cameras
                else {}
            )
            if improved:
                for field in (
                    "archive_feature",
                    "video_member",
                    "timestamps_member",
                    "frame_indices",
                    "frame_timestamps_us",
                ):
                    if improved.get(field) != indexed.get(field):
                        raise PAIAVContractError(
                            "Annotation/source reconstruction mismatch for "
                            f"{token}/{camera}/{field}"
                        )
            feature = str(
                improved.get("archive_feature", indexed.get("archive_feature", camera))
            )
            if feature != camera:
                raise PAIAVContractError(
                    f"Camera archive feature mismatch for {token}/{camera}: {feature}"
                )
            video_member = str(
                improved.get("video_member", indexed.get("video_member", ""))
            )
            timestamps_member = str(
                improved.get(
                    "timestamps_member", indexed.get("timestamps_member", "")
                )
            )
            frame_indices = _integer_tuple(
                improved.get("frame_indices", indexed.get("frame_indices")),
                FRAMES_PER_CAMERA,
                f"frame indices for {token}/{camera}",
            )
            frame_times = _integer_tuple(
                improved.get(
                    "frame_timestamps_us", indexed.get("frame_timestamps_us")
                ),
                FRAMES_PER_CAMERA,
                f"frame timestamps for {token}/{camera}",
            )
            nominal = _integer_tuple(
                improved.get(
                    "nominal_timestamps_us",
                    token_source.get("nominal_history_timestamps_us"),
                ),
                FRAMES_PER_CAMERA,
                f"nominal timestamps for {token}/{camera}",
            )
            if len(set(frame_indices)) != FRAMES_PER_CAMERA:
                raise PAIAVContractError(
                    f"Source frame indices are not unique for {token}/{camera}"
                )
            if not video_member or not timestamps_member:
                raise PAIAVContractError(
                    f"Archive member provenance is incomplete for {token}/{camera}"
                )
            archive_path = self._resolve_feature_path(camera, token=token)
            annotation_archive = improved.get("archive_path")
            if isinstance(annotation_archive, str) and Path(
                annotation_archive
            ).expanduser().is_file():
                recorded_archive = Path(annotation_archive).expanduser().resolve()
                if recorded_archive != archive_path:
                    raise PAIAVContractError(
                        "Annotation archive path differs from canonical source "
                        f"reconstruction for {token}/{camera}"
                    )
            result[camera] = CameraSource(
                camera=camera,
                archive_feature=feature,
                archive_path=archive_path,
                video_member=video_member,
                timestamps_member=timestamps_member,
                frame_indices=frame_indices,
                frame_timestamps_us=frame_times,
                nominal_timestamps_us=nominal,
            )
        self._camera_source_cache[token] = result
        return result

    def _validate_camera_archives(self) -> None:
        for token in self.tokens:
            for source in self.camera_sources(token).values():
                archive = self._archive(source.archive_path)
                for member in (source.video_member, source.timestamps_member):
                    try:
                        archive.getinfo(member)
                    except KeyError as exc:
                        raise PAIAVContractError(
                            f"Missing {member} in {source.archive_path}"
                        ) from exc

    def annotation(self, token: str) -> dict[str, Any]:
        try:
            return self.annotations[token]
        except KeyError as exc:
            raise KeyError(f"Unknown compact-run token {token!r}") from exc

    def navigation_command(self, token: str) -> str | None:
        annotation = self.annotation(token)
        if "navigation_command" not in annotation:
            raise PAIAVContractError(
                f"Annotation {token} has no navigation_command"
            )
        command = annotation["navigation_command"]
        if command is None:
            return None
        if not isinstance(command, str):
            raise PAIAVContractError(
                f"Annotation {token} has an invalid navigation_command: "
                f"{command!r}"
            )
        command = command.strip()
        if not command:
            raise PAIAVContractError(
                f"Annotation {token} has an empty navigation_command"
            )
        return None if command.lower() == "unknown" else command

    def edited_camera_paths(self, token: str) -> dict[str, list[Path]]:
        annotation = self.annotation(token)
        improved_contract = isinstance(
            annotation.get("compact_evaluation_contract"), Mapping
        )
        result: dict[str, list[Path]] = {}
        for camera in CAMERA_NAMES:
            recorded = annotation.get(camera)
            if not isinstance(recorded, list) or len(recorded) != FRAMES_PER_CAMERA:
                raise PAIAVContractError(
                    f"Annotation {token}/{camera} must contain four edited paths"
                )
            paths: list[Path] = []
            for frame_index in range(FRAMES_PER_CAMERA):
                stem = self.final_images_dir / token / camera / str(frame_index)
                matches = [
                    stem.with_suffix(suffix)
                    for suffix in SUPPORTED_IMAGE_SUFFIXES
                    if stem.with_suffix(suffix).is_file()
                ]
                if len(matches) != 1:
                    raise PAIAVContractError(
                        f"Expected one final image for {token}/{camera}/{frame_index}, "
                        f"got {matches}"
                    )
                if improved_contract:
                    recorded_path = Path(recorded[frame_index]).expanduser()
                    if (
                        recorded_path.parent.name != camera
                        or recorded_path.parent.parent.name != token
                        or recorded_path.parent.parent.parent.name
                        != "final_images"
                        or recorded_path.stem != str(frame_index)
                    ):
                        raise PAIAVContractError(
                            "Improved compact annotation does not point to its "
                            f"final image for {token}/{camera}/{frame_index}: "
                            f"{recorded_path}"
                        )
                paths.append(matches[0])
            result[camera] = paths
        return result

    @contextmanager
    def materialize_original_images(
        self, token: str
    ) -> Iterator[dict[str, list[Path]]]:
        """Yield archive-decoded originals and delete them after paired inference."""

        temporary_root_value = os.environ.get("ALPAMAYO_PAIAV_TMPDIR")
        temporary_root = (
            Path(temporary_root_value).expanduser().resolve()
            if temporary_root_value
            else None
        )
        if temporary_root is not None:
            temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"alpamayo-paiav-{token}-", dir=temporary_root
        ) as temporary:
            root = Path(temporary)
            result: dict[str, list[Path]] = {}
            for camera, source in self.camera_sources(token).items():
                outputs = [
                    root / camera / f"{frame_index}.png"
                    for frame_index in range(FRAMES_PER_CAMERA)
                ]
                self.frame_decoder(
                    source.archive_path,
                    source.video_member,
                    source.frame_indices,
                    outputs,
                )
                missing = [path for path in outputs if not path.is_file()]
                if missing:
                    raise PAIAVContractError(
                        f"Original decoder did not create paths: {missing}"
                    )
                result[camera] = outputs
            yield result

    def original_source_contract(self, token: str) -> dict[str, Any]:
        contract: dict[str, Any] = {
            "version": "paiav_archive_camera_frames_v1",
            "frame_order": "oldest_to_current",
            "source_reconstruction_file": str(self.source_index_path),
            "source_reconstruction_token": token,
            "cameras": {
                camera: {
                    "archive_feature": source.archive_feature,
                    "archive_path": str(source.archive_path),
                    "video_member": source.video_member,
                    "timestamps_member": source.timestamps_member,
                    "frame_indices": list(source.frame_indices),
                    "frame_timestamps_us": list(source.frame_timestamps_us),
                    "nominal_timestamps_us": list(source.nominal_timestamps_us),
                }
                for camera, source in self.camera_sources(token).items()
            },
        }
        chunk_id = self._token_chunk_id(token)
        if chunk_id is not None:
            contract["selected_chunk_id"] = chunk_id
        return contract

    def _read_zip_parquet(self, token: str, feature: str, member: str) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "pandas/pyarrow are required for legacy PAI-AV reconstruction"
            ) from exc
        feature_path = self._resolve_feature_path(feature, token=token)
        archive = self._archive(feature_path)
        try:
            payload = archive.read(member)
        except KeyError as exc:
            raise PAIAVContractError(
                f"Missing {member} in {feature_path}"
            ) from exc
        return pd.read_parquet(io.BytesIO(payload))

    def ego_history(self, token: str) -> tuple[dict[str, Any], str]:
        if token in self._history_cache:
            return self._history_cache[token]
        annotation = self.annotation(token)
        stored = annotation.get("alpamayo_ego_history")
        if isinstance(stored, Mapping):
            result = (validate_ego_history(stored), "annotations.json")
        else:
            source = _mapping(self.token_sources[token], f"source for {token}")
            members = _mapping(
                source.get("egomotion_members"), f"egomotion members for {token}"
            )
            member = members.get("egomotion", f"{token}.egomotion.parquet")
            if not isinstance(member, str) or not member:
                raise PAIAVContractError(f"Invalid egomotion member for {token}")
            nominal = _integer_tuple(
                source.get("nominal_history_timestamps_us"),
                FRAMES_PER_CAMERA,
                f"nominal history for {token}",
            )
            raw = self._read_zip_parquet(token, "egomotion", member)
            result = (
                validate_ego_history(build_alpamayo_ego_history(raw, nominal[-1])),
                "source_reconstruction.json+egomotion",
            )
        self._history_cache[token] = result
        return result

    def ego_vehicle_geometry(self, token: str) -> tuple[dict[str, Any], str]:
        if token in self._geometry_cache:
            return self._geometry_cache[token]
        annotation = self.annotation(token)
        stored = annotation.get("ego_vehicle_geometry")
        source_label = "annotations.json"
        if not isinstance(stored, Mapping):
            token_source = _mapping(self.token_sources[token], f"source for {token}")
            stored = token_source.get("ego_vehicle_geometry")
            source_label = "source_reconstruction.json"
        if isinstance(stored, Mapping):
            result = (validate_ego_vehicle_geometry(stored), source_label)
        else:
            if self.source_index_version == SOURCE_INDEX_V1:
                if self._vehicle_dimensions is None:
                    try:
                        import pandas as pd
                    except ImportError as exc:
                        raise RuntimeError(
                            "pandas/pyarrow are required for legacy PAI-AV geometry"
                        ) from exc
                    self._vehicle_dimensions = pd.read_parquet(
                        self._resolve_feature_path(
                            "vehicle_dimensions", token=token
                        )
                    )
                vehicle_dimensions = self._vehicle_dimensions
            else:
                dimensions_path = self._resolve_feature_path(
                    "vehicle_dimensions", token=token
                )
                vehicle_dimensions = self._vehicle_dimensions_by_path.get(
                    dimensions_path
                )
                if vehicle_dimensions is None:
                    try:
                        import pandas as pd
                    except ImportError as exc:
                        raise RuntimeError(
                            "pandas/pyarrow are required for legacy PAI-AV geometry"
                        ) from exc
                    vehicle_dimensions = pd.read_parquet(dimensions_path)
                    self._vehicle_dimensions_by_path[dimensions_path] = (
                        vehicle_dimensions
                    )
            try:
                row = vehicle_dimensions.loc[token]
            except KeyError as exc:
                raise PAIAVContractError(
                    f"vehicle_dimensions has no row for {token}"
                ) from exc
            if getattr(row, "ndim", 1) != 1:
                raise PAIAVContractError(
                    f"vehicle_dimensions row is not unique for {token}"
                )
            result = (
                validate_ego_vehicle_geometry(build_ego_vehicle_geometry(row)),
                "source_reconstruction.json+vehicle_dimensions",
            )
        self._geometry_cache[token] = result
        return result

    def _selection_manifest(self, path: Path, expected: Any) -> Mapping[str, Any]:
        """The run's recorded selection manifest, or a declared replacement of it.

        A replacement (e.g. after re-downloading PAI-AV) has a new hash, so it must name
        the recorded hash under ``restoration.replaces_sha256`` and select this run's clips/chunks.
        """
        if not expected:
            raise PAIAVContractError("Selection manifest checksum mismatch for collision scoring")
        manifest = _mapping(_load_json(path, "selection manifest"), "selection manifest")
        if file_sha256(path) == expected:
            return manifest
        restoration = manifest.get("restoration")
        samples = manifest.get("samples")
        if not isinstance(restoration, Mapping) or restoration.get("replaces_sha256") != expected:
            raise PAIAVContractError("Selection manifest checksum mismatch for collision scoring")
        clips = samples.get("clip_ids") if isinstance(samples, Mapping) else None
        chunks = samples.get("chunks") if isinstance(samples, Mapping) else None
        run_clips = self.run_config.get("selected_clip_ids")
        run_chunks = _mapping(self.run_config.get("dataset", {}), "run dataset").get(
            "selected_chunk_ids"
        )
        if (
            not isinstance(clips, list)
            or not isinstance(chunks, list)
            or set(clips) != set(run_clips if isinstance(run_clips, list) else self.tokens)
            or (run_chunks is not None
                and {chunk.get("chunk_id") for chunk in chunks if isinstance(chunk, Mapping)}
                != set(run_chunks))
        ):
            raise PAIAVContractError("Restored selection manifest does not match this run")
        return manifest

    def collision_context(self, token: str) -> dict[str, Any]:
        """Source provenance for CPU scene scoring; no image decoding or inference."""
        from ..scoring.paiav_world import checked_catalog, pinned_dataset
        if not hasattr(self, "_collision_catalog"):
            import pandas as pd
            manifest_path = Path(self.source_index["selection_manifest"])
            expected = self.source_index.get("selection_manifest_sha256")
            manifest = self._selection_manifest(manifest_path, expected)
            repo, revision = pinned_dataset(manifest)
            data_root = Path(self.source_index["data_root"])
            features = pd.read_csv(checked_catalog(manifest, "features.csv", data_root)).set_index("feature")
            index = pd.read_parquet(checked_catalog(manifest, "clip_index.parquet", data_root))
            self._collision_catalog = (repo, revision, features.loc["obstacle.offline"], index, expected)
        repo, revision, feature, index, manifest_sha = self._collision_catalog
        chunk = int(index.loc[token, "chunk"])
        declared = self._token_chunk_id(token)
        if declared is not None and declared != chunk:
            raise PAIAVContractError("Source reconstruction and catalog disagree on clip chunk")
        history, _ = self.ego_history(token)
        geometry, _ = self.ego_vehicle_geometry(token)
        relative = str(feature["chunk_path"]).format(chunk_id=chunk)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PAIAVContractError("Unsafe obstacle path in catalog")
        return {"token": token, "repo_id": repo, "revision": revision, "chunk_id": chunk,
                "current_timestamp_us": int(history["current_timestamp_us"]),
                "ego_geometry": geometry, "selection_manifest_sha256": manifest_sha,
                "obstacle_relative_path": relative,
                "obstacle_member": json.loads(feature["clip_files_in_zip"])["obstacle.offline"].format(clip_id=token),
                "offline_egomotion_path": str(self._resolve_feature_path("egomotion.offline", token=token)),
                "offline_egomotion_member": f"{token}.egomotion.offline.parquet"}

    def metadata(self, token: str) -> tuple[dict[str, Any], Path]:
        path = self.metadata_dir / f"{token}.json"
        metadata = _mapping(_load_json(path, f"metadata for {token}"), "metadata")
        if metadata.get("status") != "success" or metadata.get("token") != token:
            raise PAIAVContractError(f"Metadata is not successful for {token}")
        return dict(metadata), path

    def current_speed_mps(self, token: str) -> float:
        velocities = self.annotation(token).get("velocity")
        if not isinstance(velocities, list) or not velocities:
            raise PAIAVContractError(f"Annotation {token} has no velocity history")
        current = np.asarray(velocities[-1], dtype=np.float64)
        if current.ndim != 1 or current.size < 2 or not np.isfinite(current[:2]).all():
            raise PAIAVContractError(f"Annotation {token} has invalid velocity")
        return float(np.linalg.norm(current[:2]))

    def current_longitudinal_acceleration_mps2(
        self, token: str
    ) -> float | None:
        annotation = self.annotation(token)
        velocities = annotation.get("velocity")
        accelerations = annotation.get("acceleration")
        if not isinstance(velocities, list) or not velocities:
            raise PAIAVContractError(f"Annotation {token} has no velocity history")
        if not isinstance(accelerations, list) or not accelerations:
            raise PAIAVContractError(
                f"Annotation {token} has no acceleration history"
            )
        try:
            return project_acceleration_onto_velocity(
                velocities[-1], accelerations[-1]
            )
        except ValueError as exc:
            raise PAIAVContractError(
                f"Annotation {token} has invalid ego dynamics: {exc}"
            ) from exc

    def preflight_report(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "token_count": len(self.tokens),
            "camera_count": len(CAMERA_NAMES),
            "frames_per_camera": FRAMES_PER_CAMERA,
            "source_reconstruction_version": self.source_index.get("version"),
            "compaction_report_version": self.compaction_report.get("version"),
            "compaction_status": self.compaction_report.get("status"),
            "annotations_sha256_verified": self.compaction_report.get(
                "annotations_sha256"
            )
            is not None,
            "source_reconstruction_sha256_verified": True,
            "original_policy": "archive_decoded_temporary_files",
            "edited_policy": "final_images_only",
        }

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self) -> "PAIAVCompactRun":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
