"""Recorded NAVSIM camera histories and four-pose ego-history adaptation."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import pickle

import numpy as np

from ..alpamayo1_5.agent_navsim import (
    CAMERA_SPECS, navigation_text, resample_ego_history,
)
from ...scoring.actor_motion import inserted_actor_motion as actor_motion_from_metadata
from ...scoring.motion_bins import project_longitudinal_acceleration
from ...io import digest
from .. import common




def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def checked_token(token):
    if not isinstance(token, str) or not token or token in (".", "..") or Path(token).name != token:
        raise ValueError(f"Invalid sample token: {token!r}")
    return token


class NAVSIMRun:
    dataset = "navsim"
    camera_profile = {"camera_names": ["CAM_L0", "CAM_F0", "CAM_R0"],
                      "camera_ids": [0, 1, 2], "frames_per_camera": 4,
                      "image_interval_s": 0.5,
                      "ego_history": "4 poses at nominal 2 Hz interpolated to 16 poses at 10 Hz",
                      "navigation": "translated native four-way command",
                      "timestamp_source": "source-log frame timestamps; exposure timestamps unavailable",
                      "shift": "three cameras at 2 Hz versus six cameras at 10 Hz in the public notebook"}

    def __init__(self, run_dir, *, annotations_path=None):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.final_images_dir = self.run_dir / "final_images"
        if not self.final_images_dir.is_dir() or not (self.run_dir / "metadata").is_dir():
            raise ValueError("NAVSIM run must contain final_images/ and metadata/")
        self.run_config_path = next((self.run_dir / name for name in
            ("receiver_preview_config.json", "run_config.json") if (self.run_dir / name).is_file()), None)
        if self.run_config_path is None:
            raise FileNotFoundError("Missing NAVSIM compositor run configuration")
        self.run_config = read_json(self.run_config_path)
        if not isinstance(self.run_config, dict):
            raise ValueError("NAVSIM run configuration must be a mapping")
        self.annotations_path = Path(annotations_path) if annotations_path else (
            common.annotations_path(self.run_config))
        self.annotations_path = self.annotations_path.expanduser().resolve()
        records = read_json(self.annotations_path)
        if not isinstance(records, list):
            raise ValueError("NAVSIM annotations must be a list")
        self.annotations = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("NAVSIM annotation must be a mapping")
            token = checked_token(record.get("token"))
            if token in self.annotations:
                raise ValueError(f"Duplicate annotation: {token}")
            self.annotations[token] = record
        available = {checked_token(path.name) for path in self.final_images_dir.iterdir() if path.is_dir()}
        if not available:
            raise ValueError("No edited samples found")
        configured = self.run_config.get("selected_tokens", [])
        if not isinstance(configured, list) or len(configured) != len(set(configured)):
            raise ValueError("selected_tokens must be a list without duplicates")
        ordered = [checked_token(token) for token in configured if token in available]
        ordered.extend(sorted(available - set(ordered)))
        self.tokens = tuple(ordered)
        if available - set(self.annotations):
            raise ValueError("Edited samples missing from NAVSIM annotations")
        split = str(self.run_config.get("data_split", "test"))
        self.log_root = Path(self.run_config["data_root"]).expanduser().resolve() / "navsim_logs" / split
        self._log_name = None
        self._frames = self._frame_index = None
        self._history = {}

    def _original_paths(self, token):
        annotation = self.annotations[token]
        paths = {}
        for key, _, camera in CAMERA_SPECS:
            values = annotation.get(key)
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"Expected four original paths for {token}/{key}")
            paths[key] = [Path(value).expanduser().resolve() for value in values]
            for path in paths[key]:
                if not path.is_file() or path.parent.name != camera:
                    raise ValueError(f"Missing or mislabelled original image: {path}")
        return paths

    def _recorded_history(self, token):
        if token in self._history:
            return self._history[token]
        paths = self._original_paths(token)
        log_names = {p.parent.parent.name for values in paths.values() for p in values}
        if len(log_names) != 1:
            raise ValueError("Original camera histories span multiple logs")
        log_name = next(iter(log_names))
        log_path = self.log_root / f"{log_name}.pkl"
        if self._log_name != log_name:
            with log_path.open("rb") as handle:
                self._frames = pickle.load(handle)
            self._frame_index = {}
            for index, frame in enumerate(self._frames):
                if frame["token"] in self._frame_index:
                    raise ValueError("Duplicate token in NAVSIM source log")
                self._frame_index[frame["token"]] = index
            self._log_name = log_name
        current = self._frame_index[token]
        if current < 3:
            raise ValueError(f"{token} lacks four real history observations")
        frames = self._frames[current - 3:current + 1]
        times = np.asarray([frame["timestamp"] for frame in frames], dtype=np.int64)
        if np.any(np.abs(np.diff(times) - 500_000) > 10_000):
            raise ValueError("Recorded history is not approximately 2 Hz; interpolation would be invalid")
        for key, _, camera in CAMERA_SPECS:
            for frame, path in zip(frames, paths[key]):
                expected = Path(frame["cams"][camera]["data_path"])
                if tuple(path.parts[-3:]) != tuple(expected.parts[-3:]):
                    raise ValueError("Annotation camera order disagrees with source-log history")
        result = (times.tolist(), log_path)
        self._history[token] = result
        return result

    @contextmanager
    def materialize_original_images(self, token):
        yield self._original_paths(token)

    def edited_camera_paths(self, token):
        result = {}
        for key, _, camera in CAMERA_SPECS:
            paths = []
            for index in range(4):
                stem = self.final_images_dir / token / camera / str(index)
                matches = [stem.with_suffix(suffix) for suffix in (".png", ".jpg", ".jpeg")
                           if stem.with_suffix(suffix).is_file()]
                if len(matches) != 1:
                    raise ValueError(f"Expected one edited image for {token}/{camera}/{index}")
                paths.append(matches[0])
            result[key] = paths
        return result

    @contextmanager
    def materialize_inputs(self, token, *, original):
        yield self._original_paths(token) if original else self.edited_camera_paths(token)

    def check_edited_inputs(self, token):
        """All recorded cameras are edited; no unedited view can contradict the insertion."""
        return {}

    def recorded_future(self, token):
        """Annotations carry no future trajectory; the memorisation check is PAI-AV only."""
        return None

    def model_input(self, token, camera_paths):
        annotation = self.annotations[token]
        poses = np.asarray(annotation["past_trajectory"])
        if poses.shape != (4, 3):
            raise ValueError("NAVSIM history requires exactly four x/y/yaw poses")
        xyz, rot = resample_ego_history(poses)
        times, _ = self._recorded_history(token)
        current = times[-1]
        history = {"coordinate_frame": "current_ego", "frame_order": "oldest_to_current",
                   "history_interval_us": 100_000, "current_timestamp_us": current,
                   "timestamps_us": list(range(current - 1_500_000, current + 1, 100_000)),
                   "ego_history_xyz": xyz[0, 0].tolist(), "ego_history_rot": rot[0, 0].tolist()}
        return {"camera_paths": camera_paths, "alpamayo_ego_history": history,
                "image_timestamps_us": [list(times) for _ in CAMERA_SPECS],
                "navigation_command": navigation_text(annotation["navigation_command"])}

    def original_source_contract(self, token):
        times, log_path = self._recorded_history(token)
        paths = self._original_paths(token)
        stat = log_path.stat()
        return {"version": "alpamayo2_navsim_recorded_history_v1",
                "annotation_sha256": digest(self.annotations[token]),
                "source_log": str(log_path), "source_log_stat": [stat.st_size, stat.st_mtime_ns],
                "timestamps_us": times,
                "images": {key: [[str(path), path.stat().st_size, path.stat().st_mtime_ns]
                                 for path in values] for key, values in paths.items()}}

    def metric_context(self, token):
        path = self.run_dir / "metadata" / f"{token}.json"
        metadata = read_json(path)
        if metadata.get("status") != "success" or metadata.get("token") != token:
            raise ValueError(f"Invalid finalized insertion metadata for {token}")
        annotation = self.annotations[token]
        velocity = np.asarray(annotation["velocity"][-1], dtype=float)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError("Invalid current ego velocity")
        return {"actor_motion": actor_motion_from_metadata(metadata),
                "speed_mps": float(np.linalg.norm(velocity)),
                "longitudinal_acceleration_mps2": project_longitudinal_acceleration(
                    velocity, annotation["acceleration"][-1]),
                "placement_metadata_path": str(path), "ego_origin": "rear_axle"}

    def preflight_report(self):
        return {"run_dir": str(self.run_dir), "token_count": len(self.tokens),
                "original_policy": "recorded source images", "edited_policy": "final_images_only"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._frames = self._frame_index = None
