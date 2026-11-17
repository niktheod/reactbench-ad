"""Alpamayo inputs of a ReactNAV or ReactPAI sample (Alpamayo 2 reuses the ReactNAV part).

ReactNAV has three cameras (front-left, front, front-right) with four frames 0.5 s apart instead of the model's
0.1 s; its four ego poses (2 Hz) are linearly interpolated to the 16 poses at 10 Hz the model expects, and its
driving command is passed as a navigation instruction. ReactPAI supplies the model's native inputs: four cameras at
10 Hz and the 16-pose ego history, without navigation text; its original frames are decoded from the PAI-AV
archives for the duration of one rollout.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np

from ...datasets.navsim import NAVSIMVariant

NAVIGATION_TEXT = {
    (1, 0, 0, 0): "Turn left at the next opportunity.",
    (0, 1, 0, 0): "Continue straight.",
    (0, 0, 1, 0): "Turn right at the next opportunity.",
    (0, 0, 0, 1): None,
}


def navigation_text(navigation_command: Any) -> str | None:
    """NAVSIM's four-way one-hot driving command as Alpamayo route text."""
    command = np.asarray(navigation_command)
    if command.shape != (4,):
        raise ValueError(f"navigation_command must contain four values, got shape {command.shape}")
    if not np.isfinite(command).all():
        raise ValueError("navigation_command contains non-finite values")
    rounded = np.rint(command).astype(np.int64)
    if not np.allclose(command, rounded):
        raise ValueError(f"navigation_command must be a one-hot vector, got {navigation_command}")
    key = tuple(int(value) for value in rounded)
    if key not in NAVIGATION_TEXT:
        raise ValueError(f"Unsupported NAVSIM navigation command: {navigation_command}")
    return NAVIGATION_TEXT[key]


def resample_ego_history(history_poses: Any, *, source_interval_s: float = 0.5, target_interval_s: float = 0.1,
                         target_steps: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """NAVSIM x/y/yaw history as Alpamayo xyz (16, 3) and rotations (16, 3, 3), float32, current ego frame.

    NAVSIM's four poses cover 1.5 s at 2 Hz; Alpamayo uses sixteen poses over the same 1.5 s at 10 Hz. Positions
    and unwrapped headings are linearly interpolated after expressing the poses in the current ego frame.
    """
    poses = np.asarray(history_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] < 2 or poses.shape[1] < 3:
        raise ValueError(f"history_poses must have shape (N, >=3) with at least two poses, got {poses.shape}")
    poses = poses[:, :3]
    if not np.isfinite(poses).all():
        raise ValueError("history_poses contains non-finite values")
    if source_interval_s <= 0 or target_interval_s <= 0 or target_steps < 2:
        raise ValueError("history intervals must be positive and target_steps at least two")
    source_duration_s = (poses.shape[0] - 1) * float(source_interval_s)
    target_duration_s = (target_steps - 1) * float(target_interval_s)
    if not np.isclose(source_duration_s, target_duration_s, atol=1e-6):
        raise ValueError(f"Source and target history durations differ: {source_duration_s:g}s != "
                         f"{target_duration_s:g}s")
    current_xy, current_yaw = poses[-1, :2], float(poses[-1, 2])
    delta_xy = poses[:, :2] - current_xy
    cosine, sine = np.cos(current_yaw), np.sin(current_yaw)
    local_x = cosine * delta_xy[:, 0] + sine * delta_xy[:, 1]
    local_y = -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1]
    unwrapped_yaw = np.unwrap(poses[:, 2])
    local_yaw = unwrapped_yaw - unwrapped_yaw[-1]
    source_times = np.linspace(-source_duration_s, 0.0, poses.shape[0])
    target_times = np.linspace(-target_duration_s, 0.0, target_steps)
    interpolated_yaw = np.interp(target_times, source_times, local_yaw)
    xyz = np.zeros((target_steps, 3), dtype=np.float32)
    xyz[:, 0] = np.interp(target_times, source_times, local_x)
    xyz[:, 1] = np.interp(target_times, source_times, local_y)
    xyz[-1] = 0.0
    rotations = np.zeros((target_steps, 3, 3), dtype=np.float32)
    cosine, sine = np.cos(interpolated_yaw), np.sin(interpolated_yaw)
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    rotations[-1] = np.eye(3, dtype=np.float32)
    return xyz, rotations


def navsim_ego_history(past_trajectory: Any, current_timestamp_us: int = 0) -> dict[str, Any]:
    """Alpamayo's ego history from NAVSIM's four past poses (timestamps relative to the current frame)."""
    poses = np.asarray(past_trajectory)
    if poses.shape != (4, 3):
        raise ValueError("NAVSIM history requires exactly four x/y/yaw poses")
    xyz, rotations = resample_ego_history(poses)
    current = int(current_timestamp_us)
    return {"coordinate_frame": "current_ego", "frame_order": "oldest_to_current",
            "history_interval_us": 100_000, "current_timestamp_us": current,
            "timestamps_us": list(range(current - 1_500_000, current + 1, 100_000)),
            "ego_history_xyz": xyz.tolist(), "ego_history_rot": rotations.tolist()}


@contextmanager
def model_input(loader, token: str, *, original: bool):
    """The Alpamayo 1.5 input of one sample of a ``NAVSIMVariant`` or ``PAIAVCompactVariant``."""
    if isinstance(loader, NAVSIMVariant):
        annotation = loader.annotation(token)
        yield {"camera_paths": loader.original_images(token) if original else loader.edited_images(token),
               "alpamayo_ego_history": navsim_ego_history(annotation["past_trajectory"]),
               "navigation_command": navigation_text(annotation["navigation_command"])}
        return
    history, _ = loader.ego_history(token)
    cameras = loader.materialize_original_images(token) if original else \
        nullcontext(loader.edited_camera_paths(token))
    with cameras as camera_paths:
        yield {"camera_paths": camera_paths, "navigation_command": None, "alpamayo_ego_history": dict(history)}
