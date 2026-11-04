"""Workspace-side adapter for Alpamayo 1.5 inference on NAVSIM samples."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ... import paths


ALPAMAYO_REPO_ROOT = Path(
    os.environ.get(
        "ALPAMAYO_REPO_ROOT",
        str(paths.REPO_ROOT / "third_party" / "alpamayo1.5"),
    )
).expanduser()

CAMERA_SPECS = (
    ("front_left_camera", 0, "CAM_L0"),
    ("front_camera", 1, "CAM_F0"),
    ("front_right_camera", 2, "CAM_R0"),
)

NAVIGATION_TEXT = {
    (1, 0, 0, 0): "Turn left at the next opportunity.",
    (0, 1, 0, 0): "Continue straight.",
    (0, 0, 1, 0): "Turn right at the next opportunity.",
    (0, 0, 0, 1): None,
}


@dataclass
class AlpamayoPrediction:
    """One trajectory and its model-generated textual outputs."""

    poses: np.ndarray
    extra: dict[str, Any]


def navigation_text(navigation_command: Any) -> str | None:
    """Translate NAVSIM's four-way one-hot command to Alpamayo route text."""

    command = np.asarray(navigation_command)
    if command.shape != (4,):
        raise ValueError(
            "navigation_command must contain four values, "
            f"got shape {command.shape}"
        )
    if not np.isfinite(command).all():
        raise ValueError("navigation_command contains non-finite values")
    rounded = np.rint(command).astype(np.int64)
    if not np.allclose(command, rounded):
        raise ValueError(
            f"navigation_command must be a one-hot vector, got {navigation_command}"
        )
    key = tuple(int(value) for value in rounded)
    if key not in NAVIGATION_TEXT:
        raise ValueError(
            f"Unsupported NAVSIM navigation command: {navigation_command}"
        )
    return NAVIGATION_TEXT[key]


def resample_ego_history(
    history_poses: Any,
    *,
    source_interval_s: float = 0.5,
    target_interval_s: float = 0.1,
    target_steps: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert NAVSIM x/y/yaw history to Alpamayo xyz/rotation tensors.

    NAVSIM's standard four poses cover 1.5 seconds at 2 Hz. Alpamayo uses
    sixteen poses over the same duration at 10 Hz. Positions and unwrapped
    headings are linearly interpolated after normalizing the final pose to the
    current ego frame.
    """

    poses = np.asarray(history_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] < 2 or poses.shape[1] < 3:
        raise ValueError(
            "history_poses must have shape (N, >=3) with at least two poses, "
            f"got {poses.shape}"
        )
    poses = poses[:, :3]
    if not np.isfinite(poses).all():
        raise ValueError("history_poses contains non-finite values")
    if source_interval_s <= 0 or target_interval_s <= 0:
        raise ValueError("history intervals must be positive")
    if target_steps < 2:
        raise ValueError("target_steps must be at least two")

    source_duration_s = (poses.shape[0] - 1) * float(source_interval_s)
    target_duration_s = (target_steps - 1) * float(target_interval_s)
    if not np.isclose(source_duration_s, target_duration_s, atol=1e-6):
        raise ValueError(
            "Source and target history durations differ: "
            f"{source_duration_s:g}s != {target_duration_s:g}s"
        )

    current_xy = poses[-1, :2]
    current_yaw = float(poses[-1, 2])
    delta_xy = poses[:, :2] - current_xy
    cosine = np.cos(current_yaw)
    sine = np.sin(current_yaw)
    local_x = cosine * delta_xy[:, 0] + sine * delta_xy[:, 1]
    local_y = -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1]

    unwrapped_yaw = np.unwrap(poses[:, 2])
    local_yaw = unwrapped_yaw - unwrapped_yaw[-1]
    source_times = np.linspace(-source_duration_s, 0.0, poses.shape[0])
    target_times = np.linspace(-target_duration_s, 0.0, target_steps)
    interpolated_x = np.interp(target_times, source_times, local_x)
    interpolated_y = np.interp(target_times, source_times, local_y)
    interpolated_yaw = np.interp(target_times, source_times, local_yaw)

    xyz = np.zeros((target_steps, 3), dtype=np.float32)
    xyz[:, 0] = interpolated_x
    xyz[:, 1] = interpolated_y
    xyz[-1] = 0.0

    rotations = np.zeros((target_steps, 3, 3), dtype=np.float32)
    cosine = np.cos(interpolated_yaw)
    sine = np.sin(interpolated_yaw)
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    rotations[-1] = np.eye(3, dtype=np.float32)

    return (
        torch.from_numpy(xyz).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(rotations).unsqueeze(0).unsqueeze(0),
    )


def load_camera_tensor(
    camera_paths: dict[str, list[Path]],
    *,
    frames_per_camera: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ordered front-left/front/front-right RGB histories."""

    camera_tensors: list[torch.Tensor] = []
    camera_indices: list[int] = []
    expected_shape: tuple[int, int, int] | None = None

    for annotation_key, camera_index, _ in CAMERA_SPECS:
        paths = camera_paths.get(annotation_key)
        if not isinstance(paths, list) or len(paths) != frames_per_camera:
            raise ValueError(
                f"Expected {frames_per_camera} paths for {annotation_key!r}"
            )

        frame_tensors: list[torch.Tensor] = []
        for path_value in paths:
            path = Path(path_value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Camera image does not exist: {path}")
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            shape = tuple(int(value) for value in tensor.shape)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    "All Alpamayo camera frames must share one shape; "
                    f"expected {expected_shape}, got {shape} for {path}"
                )
            frame_tensors.append(tensor)

        camera_tensors.append(torch.stack(frame_tensors, dim=0))
        camera_indices.append(camera_index)

    return (
        torch.stack(camera_tensors, dim=0),
        torch.tensor(camera_indices, dtype=torch.int64),
    )


def extract_trajectory(
    pred_xyz: torch.Tensor, pred_rot: torch.Tensor
) -> np.ndarray:
    """Extract a single Alpamayo sample as NAVSIM-style x/y/yaw poses."""

    if pred_xyz.ndim != 5 or tuple(pred_xyz.shape[:3]) != (1, 1, 1):
        raise ValueError(
            "Expected pred_xyz shape (1, 1, 1, T, 3), "
            f"got {tuple(pred_xyz.shape)}"
        )
    if pred_xyz.shape[-1] < 2:
        raise ValueError("pred_xyz must contain at least x/y coordinates")
    if pred_rot.ndim != 6 or tuple(pred_rot.shape[:3]) != (1, 1, 1):
        raise ValueError(
            "Expected pred_rot shape (1, 1, 1, T, 3, 3), "
            f"got {tuple(pred_rot.shape)}"
        )
    if pred_rot.shape[3] != pred_xyz.shape[3] or tuple(pred_rot.shape[-2:]) != (
        3,
        3,
    ):
        raise ValueError(
            "pred_rot does not align with pred_xyz: "
            f"{tuple(pred_rot.shape)} vs {tuple(pred_xyz.shape)}"
        )

    xyz = pred_xyz[0, 0, 0].detach().float().cpu().numpy()
    rotations = pred_rot[0, 0, 0].detach().float().cpu().numpy()
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    poses = np.column_stack((xyz[:, 0], xyz[:, 1], yaw)).astype(np.float32)
    if not np.isfinite(poses).all():
        raise ValueError("Alpamayo predicted non-finite trajectory values")
    return poses


class AlpamayoAgent:
    """Adapter around Alpamayo 1.5's official VLM-plus-diffusion rollout."""

    def __init__(
        self,
        model_name_or_path: str = "nvidia/Alpamayo-1.5-10B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
        seed: int = 42,
        diffusion_seed_base: int = 42,
        local_files_only: bool = False,
        source_history_interval_s: float = 0.5,
        target_history_interval_s: float = 0.1,
        history_steps: int = 16,
        frames_per_camera: int = 4,
        use_navigation: bool = True,
        diffusion_kwargs: dict[str, Any] | None = None,
        trajectory_sampling: dict[str, Any] | None = None,
        metric_horizon_s: float | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = str(device)
        self.dtype_name = str(dtype)
        self.attn_implementation = str(attn_implementation)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.temperature = float(temperature)
        self.num_traj_samples = int(num_traj_samples)
        self.max_generation_length = int(max_generation_length)
        self.seed = int(seed)
        self.diffusion_seed_base = int(diffusion_seed_base)
        self.local_files_only = bool(local_files_only)
        self.source_history_interval_s = float(source_history_interval_s)
        self.target_history_interval_s = float(target_history_interval_s)
        self.history_steps = int(history_steps)
        self.frames_per_camera = int(frames_per_camera)
        self.use_navigation = bool(use_navigation)
        self.diffusion_kwargs = dict(diffusion_kwargs or {})
        self.trajectory_sampling = trajectory_sampling
        self.metric_horizon_s = metric_horizon_s

        self.model: Any = None
        self.processor: Any = None
        self.helper: Any = None
        self.dtype: torch.dtype | None = None

    def _resolve_dtype(self) -> torch.dtype:
        supported = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype_name not in supported:
            raise ValueError(
                f"Unsupported dtype {self.dtype_name!r}; "
                f"choose one of {sorted(supported)}"
            )
        return supported[self.dtype_name]

    def initialize(self) -> None:
        if self.num_traj_samples != 1:
            raise ValueError(
                "The paired evaluator currently requires num_traj_samples=1"
            )
        if self.frames_per_camera <= 0:
            raise ValueError("frames_per_camera must be positive")
        if self.max_generation_length <= 0:
            raise ValueError("max_generation_length must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.diffusion_seed_base <= (1 << 63) - 1:
            raise ValueError(
                "diffusion_seed_base must be between 0 and 2**63 - 1, "
                f"got {self.diffusion_seed_base}"
            )
        if not ALPAMAYO_REPO_ROOT.is_dir():
            raise FileNotFoundError(
                f"Alpamayo repository not found at {ALPAMAYO_REPO_ROOT}"
            )

        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Alpamayo evaluation requires a CUDA GPU."
            )
        if self.attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Flash Attention was requested but could not be imported. "
                    "Install a compatible build or use attn_implementation=sdpa."
                ) from exc

        source_root = ALPAMAYO_REPO_ROOT / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        self.dtype = self._resolve_dtype()
        self.model = Alpamayo1_5.from_pretrained(
            self.model_name_or_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self.model.eval()
        self.processor = helper.get_processor(self.model.tokenizer)
        self.helper = helper

    def _seed_all(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _make_model_input(self, agent_input: dict[str, Any]) -> dict[str, Any]:
        if self.processor is None or self.helper is None:
            raise RuntimeError("AlpamayoAgent.initialize() must be called first")

        image_frames, camera_indices = load_camera_tensor(
            agent_input["camera_paths"],
            frames_per_camera=self.frames_per_camera,
        )
        history_xyz, history_rot = resample_ego_history(
            agent_input["history_poses"],
            source_interval_s=self.source_history_interval_s,
            target_interval_s=self.target_history_interval_s,
            target_steps=self.history_steps,
        )
        nav_text = (
            navigation_text(agent_input["navigation_command"])
            if self.use_navigation
            else None
        )
        messages = self.helper.create_message(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=self.frames_per_camera,
            nav_text=nav_text,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_input = {
            "tokenized_data": tokenized,
            "ego_history_xyz": history_xyz,
            "ego_history_rot": history_rot,
        }
        return self.helper.to_device(model_input, self.device)

    @torch.inference_mode()
    def compute_trajectory(
        self,
        agent_input: dict[str, Any],
        *,
        diffusion_seed: int | None = None,
    ) -> AlpamayoPrediction:
        if self.model is None or self.dtype is None:
            raise RuntimeError("AlpamayoAgent.initialize() must be called first")
        model_input = self._make_model_input(agent_input)
        self._seed_all()
        diffusion_kwargs = dict(self.diffusion_kwargs)
        if diffusion_seed is not None:
            diffusion_seed = int(diffusion_seed)
            if not 0 <= diffusion_seed <= (1 << 63) - 1:
                raise ValueError(
                    "diffusion_seed must be between 0 and 2**63 - 1, "
                    f"got {diffusion_seed}"
                )
            diffusion_kwargs["diffusion_seed"] = diffusion_seed
        device_type = torch.device(self.device).type
        with torch.autocast(device_type=device_type, dtype=self.dtype, enabled=device_type == "cuda"):
            pred_xyz, pred_rot, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_input,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    num_traj_samples=self.num_traj_samples,
                    max_generation_length=self.max_generation_length,
                    diffusion_kwargs=diffusion_kwargs,
                    return_extra=True,
                )
            )
        poses = extract_trajectory(pred_xyz, pred_rot)
        return AlpamayoPrediction(poses=poses, extra=extra)
