"""Alpamayo 2's inputs from a ReactNAV variant: the recorded 2 Hz camera history and the interpolated ego history."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ...datasets import navsim
from ...io import digest
from ..alpamayo1_5.inputs import navigation_text, navsim_ego_history


class NAVSIMVariant(navsim.NAVSIMVariant):
    camera_profile = {"camera_names": ["CAM_L0", "CAM_F0", "CAM_R0"],
                      "camera_ids": [0, 1, 2], "frames_per_camera": 4,
                      "image_interval_s": 0.5,
                      "ego_history": "4 poses at nominal 2 Hz interpolated to 16 poses at 10 Hz",
                      "navigation": "translated native four-way command",
                      "timestamp_source": "source-log frame timestamps; exposure timestamps unavailable",
                      "shift": "three cameras at 2 Hz versus six cameras at 10 Hz in the public notebook"}

    def __init__(self, variant_dir, **kwargs):
        super().__init__(variant_dir, **kwargs)
        self._history = {}

    def _original_paths(self, token):
        paths = {}
        for key, values in self.original_images(token).items():
            paths[key] = [value.expanduser().resolve() for value in values]
            for path in paths[key]:
                if not path.is_file() or path.parent.name != navsim.CAMERAS[key]:
                    raise ValueError(f"Missing or mislabelled original image: {path}")
        return paths

    def _recorded_history(self, token):
        """Timestamps of the four history frames, checked against the source log."""
        if token in self._history:
            return self._history[token]
        paths = self._original_paths(token)
        log_names = {p.parent.parent.name for values in paths.values() for p in values}
        if len(log_names) != 1:
            raise ValueError("Original camera histories span multiple logs")
        log_path, frames, current = self.source_log(token, next(iter(log_names)))
        if current < 3:
            raise ValueError(f"{token} lacks four real history observations")
        frames = frames[current - 3:current + 1]
        times = np.asarray([frame["timestamp"] for frame in frames], dtype=np.int64)
        if np.any(np.abs(np.diff(times) - 500_000) > 10_000):
            raise ValueError("Recorded history is not approximately 2 Hz; interpolation would be invalid")
        for key, camera in navsim.CAMERAS.items():
            for frame, path in zip(frames, paths[key]):
                expected = Path(frame["cams"][camera]["data_path"])
                if tuple(path.parts[-3:]) != tuple(expected.parts[-3:]):
                    raise ValueError("Annotation camera order disagrees with source-log history")
        self._history[token] = (times.tolist(), log_path)
        return self._history[token]

    @contextmanager
    def materialize_inputs(self, token, *, original):
        yield self._original_paths(token) if original else self.edited_images(token)

    def model_input(self, token, camera_paths):
        annotation = self.annotation(token)
        times, _ = self._recorded_history(token)
        return {"camera_paths": camera_paths,
                "alpamayo_ego_history": navsim_ego_history(annotation["past_trajectory"], times[-1]),
                "image_timestamps_us": [list(times) for _ in navsim.CAMERAS],
                "navigation_command": navigation_text(annotation["navigation_command"])}

    def original_source_contract(self, token):
        times, log_path = self._recorded_history(token)
        paths = self._original_paths(token)
        stat = log_path.stat()
        return {"version": "alpamayo2_navsim_recorded_history_v1",
                "annotation_sha256": digest(self.annotation(token)),
                "source_log": str(log_path), "source_log_stat": [stat.st_size, stat.st_mtime_ns],
                "timestamps_us": times,
                "images": {key: [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in values]
                           for key, values in paths.items()}}
