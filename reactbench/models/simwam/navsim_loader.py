"""Current CAM_F0 frame and ego state for SimWAM NAVSIM evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from ...io import digest
from ...scoring.actor_motion import inserted_actor_motion as actor_motion_from_metadata
from ...scoring.motion_bins import project_longitudinal_acceleration
from .. import common


CAMERA = "CAM_F0"
CURRENT_INDEX = 3
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SOURCE_CONTRACT_VERSION = "simwam_navsim_current_front_v1"


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def checked_token(token):
    if not isinstance(token, str) or not token or token in (".", "..") or Path(token).name != token:
        raise ValueError(f"Invalid sample token: {token!r}")
    return token


def ego_state(annotation):
    """Upstream 8-D state: current ego velocity(2), acceleration(2), one-hot command(4)."""
    velocity = np.asarray(annotation["velocity"][-1], dtype=np.float64)
    acceleration = np.asarray(annotation["acceleration"][-1], dtype=np.float64)
    command = np.asarray(annotation["navigation_command"], dtype=np.float64)
    if velocity.shape != (2,) or acceleration.shape != (2,) or command.shape != (4,):
        raise ValueError("Expected 2-D velocity/acceleration and a 4-way navigation command")
    if not (np.isfinite(velocity).all() and np.isfinite(acceleration).all() and np.isfinite(command).all()):
        raise ValueError("Ego state contains non-finite values")
    if not np.isin(command, (0.0, 1.0)).all() or command.sum() != 1:
        raise ValueError(f"Navigation command must be one-hot, got {command.tolist()}")
    return np.concatenate([velocity, acceleration, command]).astype(np.float32)


class NAVSIMRun:
    dataset = "navsim"
    input_profile = {"camera": CAMERA, "frame": "current (history index 3)",
                     "state": "current ego velocity(2), acceleration(2), driving command(4)",
                     "prompt": "fixed upstream prompt embedding, identical for every sample",
                     "history_inputs": "none; earlier edited frames are not used"}

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
        self.annotations_path = Path(annotations_path) if annotations_path else common.annotations_path(self.run_config)
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
        split = str(self.run_config.get("split", self.run_config.get("data_split", "test")))
        self.log_root = Path(self.run_config["data_root"]).expanduser().resolve() / "navsim_logs" / split
        self._log_name = None
        self._frames = self._frame_index = None
        self._sources = {}

    def original_image_path(self, token):
        values = self.annotations[token].get("front_camera")
        if not isinstance(values, list) or len(values) != CURRENT_INDEX + 1:
            raise ValueError(f"Expected four original {CAMERA} paths for {token}")
        path = Path(values[CURRENT_INDEX]).expanduser().resolve()
        if not path.is_file() or path.parent.name != CAMERA:
            raise ValueError(f"Missing or mislabelled original image: {path}")
        return path

    def edited_image_path(self, token):
        stem = self.final_images_dir / token / CAMERA / str(CURRENT_INDEX)
        matches = [stem.with_suffix(suffix) for suffix in IMAGE_SUFFIXES if stem.with_suffix(suffix).is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected one edited current {CAMERA} image for {token}, found {len(matches)}")
        return matches[0]

    def image_path(self, token, *, original):
        return self.original_image_path(token) if original else self.edited_image_path(token)

    def _source_frame(self, token):
        """Check that the annotation's current image is the source log's current CAM_F0 frame."""
        if token in self._sources:
            return self._sources[token]
        path = self.original_image_path(token)
        log_name = path.parent.parent.name
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
        if token not in self._frame_index:
            raise ValueError(f"{token} is absent from source log {log_path}")
        frame = self._frames[self._frame_index[token]]
        expected = Path(frame["cams"][CAMERA]["data_path"])
        if tuple(path.parts[-3:]) != tuple(expected.parts[-3:]):
            raise ValueError("Annotation current image is not the source log's current CAM_F0 frame")
        result = (int(frame["timestamp"]), log_path)
        self._sources[token] = result
        return result

    def check_edited_inputs(self, token):
        """The edited current frame must be the same sensor image size as its source."""
        with Image.open(self.original_image_path(token)) as original, \
                Image.open(self.edited_image_path(token)) as edited:
            if original.size != edited.size:
                raise ValueError(f"Edited image size {edited.size} differs from source {original.size}")
            return {"edited_image_size": list(edited.size)}

    def model_input(self, token, image_path):
        timestamp, _ = self._source_frame(token)
        return {"image_path": Path(image_path), "state": ego_state(self.annotations[token]),
                "current_timestamp_us": timestamp}

    def original_source_contract(self, token):
        timestamp, log_path = self._source_frame(token)
        path = self.original_image_path(token)
        stat, image_stat = log_path.stat(), path.stat()
        return {"version": SOURCE_CONTRACT_VERSION,
                "annotation_sha256": digest(self.annotations[token]),
                "source_log": str(log_path), "source_log_stat": [stat.st_size, stat.st_mtime_ns],
                "current_timestamp_us": timestamp,
                "image": [str(path), image_stat.st_size, image_stat.st_mtime_ns]}

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
                "original_policy": "recorded source current CAM_F0 image",
                "edited_policy": f"final_images/<token>/{CAMERA}/{CURRENT_INDEX}.* only"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._frames = self._frame_index = None
