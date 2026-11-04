"""Reuse compact-run archive decoding and native PAI-AV egomotion/geometry.

Inputs follow NVIDIA's six-camera trajectory profile: the four edited cameras
plus rear-left/right frames decoded from the unedited source archives for both
pair members. Rear frames carry no insertion, so edited samples whose actor
would appear in a rear view are excluded instead of being shown inconsistent
views.
"""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ...io import digest
from ...datasets.paiav import (
    CAMERA_NAMES, CameraSource, PAIAVCompactRun, PAIAVContractError, _pose_matrices,
)
from ...scoring.actor_motion import actor_motion_from_metadata
from ...scoring.actor_motion import actor_footprint_corners
from ...compositing.paiav.paiav_io import _nearest_unique_indices, _pose_matrix


REAR_CAMERAS = ("camera_rear_left_70fov", "camera_rear_right_70fov")
# NVIDIA's trajectory input profile, in the ascending camera-ID order upstream requires.
MODEL_CAMERAS = (("camera_cross_left_120fov", 0), ("camera_front_wide_120fov", 1),
                 ("camera_cross_right_120fov", 2), ("camera_rear_left_70fov", 3),
                 ("camera_rear_right_70fov", 5), ("camera_front_tele_30fov", 6))
FUTURE_STEPS, FUTURE_INTERVAL_US = 64, 100_000
# Conservative actor volume around the rig ground plane, sampled on a grid so a
# box crossing an image edge is caught even when no corner projects inside.
ACTOR_Z_BELOW_M, ACTOR_Z_ABOVE_M, BOX_SAMPLES = 1.0, 0.5, 5
VISIBILITY_MARGIN = 0.04  # Same angular/pixel margin as the compositor's frustum test.
# Runs prepared separately store the same ego history up to ~1e-14 float noise;
# the provenance hash compares it at micrometre/microradian precision.
EGO_HISTORY_HASH_DECIMALS = 6


def ftheta_visible(points_rig: np.ndarray, rig_from_camera: np.ndarray, intrinsic: dict,
                   margin: float = VISIBILITY_MARGIN) -> np.ndarray:
    """F-theta frustum test equivalent to the compositor's ``camera_frustum_mask``.

    Reimplemented here because the compositor geometry module needs OpenCV,
    which the upstream Alpamayo 2 environment does not install.
    """
    camera = (np.asarray(points_rig, dtype=np.float64) - rig_from_camera[:3, 3]) @ rig_from_camera[:3, :3]
    depth = camera[:, 2]
    xy_norm = np.linalg.norm(camera[:, :2], axis=1)
    theta = np.arctan2(xy_norm, depth)
    radius = np.polynomial.polynomial.polyval(theta, intrinsic["angle_to_pixeldist_poly"])
    radial = np.divide(camera[:, :2], xy_norm[:, None], out=np.zeros_like(camera[:, :2]),
                       where=xy_norm[:, None] > 1e-12)
    pixels = np.asarray(intrinsic["principal_point"], dtype=np.float64) + radius[:, None] * radial
    width, height = intrinsic["resolution"]
    pad = margin * max(width, height)
    return ((depth > 0.2) & (theta <= intrinsic["max_angle"] + margin)
            & (pixels[:, 0] >= -pad) & (pixels[:, 0] < width + pad)
            & (pixels[:, 1] >= -pad) & (pixels[:, 1] < height + pad))


def actor_volume_points(footprint: np.ndarray, height_m: float) -> np.ndarray:
    """Grid over an ordered footprint rectangle and a padded height range."""
    a, b, c, d = np.asarray(footprint, dtype=np.float64)
    steps = np.linspace(0.0, 1.0, BOX_SAMPLES)
    u, v = np.meshgrid(steps, steps, indexing="ij")
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)
    xy = (1 - u) * (1 - v) * a + u * (1 - v) * b + u * v * c + (1 - u) * v * d
    z = np.linspace(-ACTOR_Z_BELOW_M, height_m + ACTOR_Z_ABOVE_M, BOX_SAMPLES)
    return np.column_stack((np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))))


def _actor_height(motion: dict) -> float:
    bounds = (motion.get("geometry") or {}).get("local_bounds_m") or {}
    if bounds.get("z") is not None:
        return float(bounds["z"][1] - bounds["z"][0])
    return float(motion.get("height_m", 2.0))


class PAIAVRun(PAIAVCompactRun):
    dataset = "pai-av"
    camera_profile = {"camera_names": [name for name, _ in MODEL_CAMERAS],
                      "camera_ids": [index for _, index in MODEL_CAMERAS],
                      "edited_cameras": list(CAMERA_NAMES),
                      "frames_per_camera": 4, "image_interval_s": 0.1,
                      "ego_history": "16 native 3D poses at 10 Hz",
                      "navigation": None,
                      "timestamp_source": "decoded archive camera timestamps",
                      "rear_cameras": "unedited source frames shared by both pair members; edited samples "
                                      "whose inserted actor projects into a rear view are excluded",
                      "shift": "none: NVIDIA's six-camera trajectory profile"}

    def __init__(self, run_dir, **kwargs):
        super().__init__(run_dir, **kwargs)
        self._rear_source_cache: dict[str, dict[str, CameraSource]] = {}
        self._rear_calibration_cache: dict[str, dict] = {}

    @staticmethod
    def _rear_archive_path(front_path: Path, front_camera: str, camera: str) -> Path:
        """Rear archives share PAI-AV's camera/<feature>/<feature>.chunk_<id>.zip layout."""
        if front_camera not in front_path.name:
            raise PAIAVContractError(f"Cannot derive {camera} archive from {front_path}")
        path = Path(*(part.replace(front_camera, camera) for part in front_path.parts))
        if not path.is_file():
            raise FileNotFoundError(f"Missing {camera} archive for this chunk: {path}")
        return path

    def rear_camera_sources(self, token: str) -> dict[str, CameraSource]:
        """Select rear frames nearest the nominal history times, as the compositor did for edits."""
        if token in self._rear_source_cache:
            return self._rear_source_cache[token]
        import pandas as pd
        front_camera = CAMERA_NAMES[0]
        front = self.camera_sources(token)[front_camera]
        nominal = np.asarray(front.nominal_timestamps_us, dtype=np.int64)
        result = {}
        for camera in REAR_CAMERAS:
            archive_path = self._rear_archive_path(front.archive_path, front_camera, camera)
            video_member, timestamps_member = f"{token}.{camera}.mp4", f"{token}.{camera}.timestamps.parquet"
            archive = self._archive(archive_path)
            try:
                archive.getinfo(video_member)
                frame = pd.read_parquet(io.BytesIO(archive.read(timestamps_member)))
            except KeyError as exc:
                raise PAIAVContractError(f"Missing {camera} members for {token} in {archive_path}") from exc
            if "timestamp" not in frame.columns:
                raise PAIAVContractError(f"{timestamps_member} has no timestamp column")
            timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
            positions = _nearest_unique_indices(timestamps, nominal, camera)
            indices = (frame.index.to_numpy(dtype=np.int64)[positions]
                       if frame.index.name == "frame_index" else positions)
            result[camera] = CameraSource(
                camera=camera, archive_feature=camera, archive_path=archive_path,
                video_member=video_member, timestamps_member=timestamps_member,
                frame_indices=tuple(int(value) for value in indices),
                frame_timestamps_us=tuple(int(value) for value in timestamps[positions]),
                nominal_timestamps_us=tuple(int(value) for value in nominal))
        self._rear_source_cache[token] = result
        return result

    def all_camera_sources(self, token: str) -> dict[str, CameraSource]:
        sources = {**self.camera_sources(token), **self.rear_camera_sources(token)}
        return {name: sources[name] for name, _ in MODEL_CAMERAS}

    def model_input(self, token, camera_paths):
        history, _ = self.ego_history(token)
        return {"camera_paths": camera_paths, "alpamayo_ego_history": history,
                "navigation_command": None,
                "image_timestamps_us": [list(source.frame_timestamps_us)
                                        for source in self.all_camera_sources(token).values()]}

    @contextmanager
    def materialize_inputs(self, token, *, original: bool):
        """Yield six cameras' frames; decoded archive frames are deleted afterwards.

        Rear frames always come from the source archives. Front frames are the
        archive originals or the edited final images.
        """
        root_value = os.environ.get("ALPAMAYO_PAIAV_TMPDIR")
        root = Path(root_value).expanduser().resolve() if root_value else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        edited = {} if original else self.edited_camera_paths(token)
        sources = self.all_camera_sources(token)
        with tempfile.TemporaryDirectory(prefix=f"alpamayo2-paiav-{token}-", dir=root) as temporary:
            paths = {}
            for camera, source in sources.items():
                if camera in edited:
                    paths[camera] = edited[camera]
                    continue
                outputs = [Path(temporary) / camera / f"{index}.png" for index in range(4)]
                self.frame_decoder(source.archive_path, source.video_member, source.frame_indices, outputs)
                missing = [path for path in outputs if not path.is_file()]
                if missing:
                    raise PAIAVContractError(f"Archive decoder did not create paths: {missing}")
                paths[camera] = outputs
            yield paths

    def original_source_contract(self, token):
        contract = super().original_source_contract(token)
        history, _ = self.ego_history(token)
        geometry, _ = self.ego_vehicle_geometry(token)
        # Only the provenance hash is rounded; the model receives the exact history.
        hashed = {**history, **{key: (np.round(np.asarray(history[key], dtype=np.float64),
                                               EGO_HISTORY_HASH_DECIMALS) + 0.0).tolist()  # +0.0 drops -0.0
                                for key in ("ego_history_xyz", "ego_history_rot")}}
        contract.update(ego_history_sha256=digest(hashed), ego_geometry_sha256=digest(geometry),
                        camera_ids=[index for _, index in MODEL_CAMERAS])
        for camera, source in self.camera_sources(token).items():
            stat = source.archive_path.stat()
            contract["cameras"][camera]["archive_stat"] = [stat.st_size, stat.st_mtime_ns]
        contract["rear_cameras"] = {}
        for camera, source in self.rear_camera_sources(token).items():
            stat = source.archive_path.stat()
            contract["rear_cameras"][camera] = {
                "archive_feature": source.archive_feature, "archive_path": str(source.archive_path),
                "video_member": source.video_member, "timestamps_member": source.timestamps_member,
                "frame_indices": list(source.frame_indices),
                "frame_timestamps_us": list(source.frame_timestamps_us),
                "nominal_timestamps_us": list(source.nominal_timestamps_us),
                "archive_stat": [stat.st_size, stat.st_mtime_ns]}
        return contract

    def _rear_calibration(self, token: str) -> dict:
        if token in self._rear_calibration_cache:
            return self._rear_calibration_cache[token]
        import pandas as pd
        filters = [("clip_id", "==", token)]
        intrinsics = pd.read_parquet(self._resolve_feature_path("camera_intrinsics.offline", token=token),
                                     columns=["model_type", "model_parameters"], filters=filters)
        extrinsics = pd.read_parquet(self._resolve_feature_path("sensor_extrinsics.offline", token=token),
                                     filters=filters)
        result = {}
        for camera in REAR_CAMERAS:
            try:
                row, pose = intrinsics.loc[(token, camera)], extrinsics.loc[(token, camera)]
            except KeyError as exc:
                raise PAIAVContractError(f"Missing calibration for {token}/{camera}") from exc
            if not isinstance(row, pd.Series) or not isinstance(pose, pd.Series):
                raise PAIAVContractError(f"Calibration is not unique for {token}/{camera}")
            if str(row["model_type"]).lower() != "ftheta":
                raise PAIAVContractError(f"Unsupported camera model for {camera}: {row['model_type']!r}")
            parameters = json.loads(row["model_parameters"])
            intrinsic = {"resolution": [int(value) for value in parameters["resolution"]],
                         "principal_point": [float(value) for value in parameters["principal_point"]],
                         "angle_to_pixeldist_poly": [float(value) for value in parameters["angle_to_pixeldist_poly"]],
                         "max_angle": float(parameters["max_angle"])}
            result[camera] = (intrinsic, _pose_matrix(pose))
        self._rear_calibration_cache[token] = result
        return result

    def rear_visibility(self, token: str) -> dict[str, list[int]]:
        """Count inserted-actor sample points inside each rear image, per history frame."""
        metadata, _ = self.metadata(token)
        motion = actor_motion_from_metadata(metadata, camera_name=None)
        history, _ = self.ego_history(token)
        times = np.asarray(history["timestamps_us"], dtype=np.int64)
        current = int(history["current_timestamp_us"])
        xyz = np.asarray(history["ego_history_xyz"], dtype=np.float64)
        rotations = np.asarray(history["ego_history_rot"], dtype=np.float64)
        calibration = self._rear_calibration(token)
        height = _actor_height(motion)
        counts = {camera: [] for camera in REAR_CAMERAS}
        for timestamp in self.camera_sources(token)[CAMERA_NAMES[0]].nominal_timestamps_us:
            index = np.flatnonzero(times == timestamp)
            if len(index) != 1:
                raise PAIAVContractError("Rear-view check needs an ego pose at every image timestamp")
            footprint = actor_footprint_corners(motion, (timestamp - current) / 1e6)
            # Current-ego points into the rig frame at this image: R^T (p - t).
            points = (actor_volume_points(footprint, height) - xyz[index[0]]) @ rotations[index[0]]
            for camera, (intrinsic, rig_from_camera) in calibration.items():
                counts[camera].append(int(np.count_nonzero(ftheta_visible(points, rig_from_camera, intrinsic))))
        return counts

    def check_edited_inputs(self, token):
        """Reject insertions the unedited rear cameras should have seen."""
        counts = self.rear_visibility(token)
        visible = [f"{camera} frame {frame}" for camera, values in counts.items()
                   for frame, count in enumerate(values) if count]
        if visible:
            raise PAIAVContractError("Inserted actor projects into unedited rear views (" + ", ".join(visible)
                                     + "); excluded because both members show original rear frames")
        return {"rear_visible_point_counts": counts}

    def recorded_future(self, token):
        """Recorded rear-axle future (64 x/y/yaw poses at 0.1 s) for the memorisation check."""
        history, _ = self.ego_history(token)
        members = self.token_sources[token].get("egomotion_members") or {}
        raw = self._read_zip_parquet(token, "egomotion", members.get("egomotion", f"{token}.egomotion.parquet"))
        times = int(history["current_timestamp_us"]) + FUTURE_INTERVAL_US * np.arange(FUTURE_STEPS + 1)
        world_from_rig = _pose_matrices(raw, times)
        current_from_future = np.linalg.inv(world_from_rig[0])[None] @ world_from_rig[1:]
        yaw = np.arctan2(current_from_future[:, 1, 0], current_from_future[:, 0, 0])
        return np.column_stack((current_from_future[:, 0, 3], current_from_future[:, 1, 3], yaw))

    def preflight_report(self):
        return {**super().preflight_report(), "rear_policy": "archive_decoded_for_both_members",
                "camera_ids": [index for _, index in MODEL_CAMERAS]}

    def metric_context(self, token):
        metadata, metadata_path = self.metadata(token)
        geometry, geometry_source = self.ego_vehicle_geometry(token)
        return {"actor_motion": actor_motion_from_metadata(metadata, camera_name=None),
                "vehicle_geometry": geometry, "speed_mps": self.current_speed_mps(token),
                "longitudinal_acceleration_mps2": self.current_longitudinal_acceleration_mps2(token),
                "placement_metadata_path": str(metadata_path),
                "ego_geometry_source": geometry_source, "ego_origin": "rear_axle"}
