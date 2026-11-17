"""SimWAM's inputs from a ReactNAV variant: the current CAM_F0 frame and the 8-D ego state."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ...datasets import navsim
from ...io import digest

CAMERA = "CAM_F0"
SOURCE_CONTRACT_VERSION = "simwam_navsim_current_front_v1"


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


class NAVSIMVariant(navsim.NAVSIMVariant):
    input_profile = {"camera": CAMERA, "frame": "current (history index 3)",
                     "state": "current ego velocity(2), acceleration(2), driving command(4)",
                     "prompt": "fixed upstream prompt embedding, identical for every sample",
                     "history_inputs": "none; earlier edited frames are not used"}

    def __init__(self, variant_dir, **kwargs):
        super().__init__(variant_dir, **kwargs)
        self._sources = {}

    def original_image_path(self, token):
        path = self.original_images(token)["front_camera"][navsim.CURRENT].expanduser().resolve()
        if not path.is_file() or path.parent.name != CAMERA:
            raise ValueError(f"Missing or mislabelled original image: {path}")
        return path

    def image_path(self, token, *, original):
        return self.original_image_path(token) if original else self.edited_image(token, CAMERA)

    def _source_frame(self, token):
        """Check that the annotation's current image is the source log's current CAM_F0 frame."""
        if token not in self._sources:
            path = self.original_image_path(token)
            log_path, frames, index = self.source_log(token, path.parent.parent.name)
            frame = frames[index]
            expected = Path(frame["cams"][CAMERA]["data_path"])
            if tuple(path.parts[-3:]) != tuple(expected.parts[-3:]):
                raise ValueError("Annotation current image is not the source log's current CAM_F0 frame")
            self._sources[token] = (int(frame["timestamp"]), log_path)
        return self._sources[token]

    def check_edited_inputs(self, token):
        """The edited current frame must be the same sensor image size as its source."""
        with Image.open(self.original_image_path(token)) as original, \
                Image.open(self.image_path(token, original=False)) as edited:
            if original.size != edited.size:
                raise ValueError(f"Edited image size {edited.size} differs from source {original.size}")
            return {"edited_image_size": list(edited.size)}

    def model_input(self, token, image_path):
        timestamp, _ = self._source_frame(token)
        return {"image_path": Path(image_path), "state": ego_state(self.annotation(token)),
                "current_timestamp_us": timestamp}

    def original_source_contract(self, token):
        timestamp, log_path = self._source_frame(token)
        path = self.original_image_path(token)
        stat, image_stat = log_path.stat(), path.stat()
        return {"version": SOURCE_CONTRACT_VERSION,
                "annotation_sha256": digest(self.annotation(token)),
                "source_log": str(log_path), "source_log_stat": [stat.st_size, stat.st_mtime_ns],
                "current_timestamp_us": timestamp,
                "image": [str(path), image_stat.st_size, image_stat.st_mtime_ns]}
