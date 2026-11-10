"""Loader for ReactNAV runs: edited frames, inserted objects and the NAVSIM annotations of their scenes.

A ReactNAV run (``reactbench.datasets.reactnav``) holds ``final_images/<token>/{CAM_L0,CAM_F0,CAM_R0}/{0..3}.{png,jpg}``
(0.5 s apart, frame 3 is the current frame), ``metadata/<token>.json`` (the inserted object) and
``run_config.json``. The original images, ego history and driving command of each token come from the NAVSIM
annotations of the run's scene filter (``reactbench.datasets.navsim_annotations``), the NAVSIM logs from
``configs/paths.yaml``. The ReactPAI counterpart is ``reactbench.datasets.paiav.PAIAVCompactRun``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .. import paths
from ..io import read_json

# Annotation key of each camera, in the model's order (front-left, front, front-right).
CAMERAS = {"front_left_camera": "CAM_L0", "front_camera": "CAM_F0", "front_right_camera": "CAM_R0"}
FRAMES = 4
CURRENT = FRAMES - 1
SUFFIXES = (".png", ".jpg", ".jpeg")


def checked_token(token):
    if not isinstance(token, str) or not token or token in (".", "..") or Path(token).name != token:
        raise ValueError(f"Invalid sample token: {token!r}")
    return token


def annotations_path(run_config):
    """The NAVSIM annotations of a run's scene filter: ``<annotations>/<scene filter stem>.json``."""
    path = paths.get("annotations", paths.outputs("annotations")) / f"{Path(run_config['scene_filter']).stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"NAVSIM annotations do not exist: {path} (reactbench.datasets.navsim_annotations)")
    return path.resolve()


class NAVSIMRun:
    """Validated access to one ReactNAV run (a context manager, like ``PAIAVCompactRun``)."""

    def __init__(self, run_dir, *, annotations=None):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.final_images_dir = (self.run_dir / "final_images").resolve()
        self.metadata_dir = self.run_dir / "metadata"
        if not self.final_images_dir.is_dir() or not self.metadata_dir.is_dir():
            raise NotADirectoryError(f"A ReactNAV run needs final_images/ and metadata/: {self.run_dir}")
        self.run_config_path = self.run_dir / "run_config.json"
        self.run_config = read_json(self.run_config_path)
        if not isinstance(self.run_config, dict) or not self.run_config.get("scene_filter"):
            raise ValueError(f"Expected a JSON object with a scene_filter: {self.run_config_path}")
        available = {checked_token(path.name) for path in self.final_images_dir.iterdir() if path.is_dir()}
        if not available:
            raise ValueError(f"No token directories under {self.final_images_dir}")
        configured = self.run_config.get("selected_tokens") or []
        if not isinstance(configured, list) or len(configured) != len(set(configured)):
            raise ValueError("selected_tokens must be a list without duplicates")
        # The compositor's token order, then any other token, sorted.
        ordered = [token for token in configured if token in available]
        self.tokens = tuple(ordered + sorted(available - set(ordered)))
        self.annotations_path = Path(annotations).resolve() if annotations else None
        self._annotations = None
        self._log = (None, None)

    # --- inputs --------------------------------------------------------------------------------------------

    @property
    def annotations(self):
        """{token: NAVSIM annotation} (read on first use; runners on other inputs never need it)."""
        if self._annotations is None:
            self.annotations_path = self.annotations_path or annotations_path(self.run_config)
            records = read_json(self.annotations_path)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON list in {self.annotations_path}")
            by_token = {}
            for record in records:
                token = checked_token(record.get("token") if isinstance(record, dict) else None)
                if token in by_token:
                    raise ValueError(f"Duplicate annotation {token} in {self.annotations_path}")
                by_token[token] = record
            missing = sorted(set(self.tokens) - set(by_token))
            if missing:
                raise ValueError(f"{len(missing)} tokens of the run are not in {self.annotations_path}: {missing[:5]}")
            self._annotations = by_token
        return self._annotations

    def annotation(self, token):
        return self.annotations[token]

    def edited_image(self, token, camera="CAM_F0", frame=CURRENT):
        """``final_images/<token>/<camera>/<frame>.{png,jpg,jpeg}``; exactly one must exist."""
        stem = self.final_images_dir / token / camera / str(frame)
        matches = [stem.with_suffix(suffix) for suffix in SUFFIXES if stem.with_suffix(suffix).is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one edited image {stem}.{{png,jpg,jpeg}}, found {len(matches)}")
        return matches[0]

    def edited_images(self, token):
        """{annotation key: the four edited frames} of the three cameras."""
        return {key: [self.edited_image(token, camera, frame) for frame in range(FRAMES)]
                for key, camera in CAMERAS.items()}

    def original_images(self, token):
        """{annotation key: the four original navtest frames}, as the annotations record them."""
        annotation = self.annotation(token)
        result = {}
        for key in CAMERAS:
            values = annotation.get(key)
            if not isinstance(values, list) or len(values) != FRAMES:
                raise ValueError(f"Expected {FRAMES} {key} paths for {token}")
            result[key] = [Path(value) for value in values]
        return result

    def source_log(self, token, log_name):
        """The NAVSIM log frames of ``log_name`` and the index of ``token`` in them (the last log is kept)."""
        if self._log[0] != log_name:
            split = str(self.run_config.get("split", self.run_config.get("data_split", "test")))
            path = paths.navsim_logs(split) / f"{log_name}.pkl"
            with path.open("rb") as handle:
                frames = pickle.load(handle)
            index = {}
            for position, frame in enumerate(frames):
                if frame["token"] in index:
                    raise ValueError(f"Duplicate token in NAVSIM log {path}")
                index[frame["token"]] = position
            self._log = (log_name, (path, frames, index))
        path, frames, index = self._log[1]
        if token not in index:
            raise ValueError(f"{token} is absent from the NAVSIM log {path}")
        return path, frames, index[token]

    # --- bookkeeping ---------------------------------------------------------------------------------------

    def metadata(self, token):
        path = self.metadata_dir / f"{token}.json"
        metadata = read_json(path)
        if not isinstance(metadata, dict) or metadata.get("status") != "success" or metadata.get("token") != token:
            raise ValueError(f"Invalid insertion metadata for {token}: {path}")
        return metadata, path

    def metric_context(self, token):
        """What a sample's bookkeeping needs: the inserted object's motion and the ego speed and acceleration."""
        from ..scoring.actor_motion import inserted_actor_motion
        from ..scoring.geometry import project_acceleration_onto_velocity
        metadata, path = self.metadata(token)
        annotation = self.annotation(token)
        velocity = np.asarray(annotation["velocity"][-1], dtype=np.float64)
        if velocity.ndim != 1 or velocity.size < 2 or not np.isfinite(velocity[:2]).all():
            raise ValueError(f"Invalid current ego velocity for {token}")
        return {"actor_motion": inserted_actor_motion(metadata), "speed_mps": float(np.linalg.norm(velocity[:2])),
                "longitudinal_acceleration_mps2": project_acceleration_onto_velocity(
                    annotation["velocity"][-1], annotation["acceleration"][-1]),
                "placement_metadata_path": str(path), "ego_origin": "rear_axle"}

    def close(self):
        self._log = (None, None)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
