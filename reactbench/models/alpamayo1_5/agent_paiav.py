"""Workspace-side adapter for Alpamayo 1.5 inference on PAI-AV samples."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from ... import paths
from .forced_cot import forced_cot_prefix


ALPAMAYO_REPO_ROOT = Path(
    os.environ.get(
        "ALPAMAYO_REPO_ROOT",
        str(paths.REPO_ROOT / "third_party" / "alpamayo1.5"),
    )
).expanduser()

# Alpamayo's native semantic camera order and IDs for PAI-AV.
CAMERA_SPECS = (
    ("camera_cross_left_120fov", 0),
    ("camera_front_wide_120fov", 1),
    ("camera_cross_right_120fov", 2),
    ("camera_front_tele_30fov", 6),
)


@dataclass
class AlpamayoPrediction:
    """One trajectory and its model-generated textual outputs."""

    poses: np.ndarray
    extra: dict[str, Any]


def load_camera_tensor(
    camera_paths: Mapping[str, list[Path]],
    *,
    frames_per_camera: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the four PAI-AV camera histories in Alpamayo semantic order."""

    camera_tensors: list[torch.Tensor] = []
    camera_indices: list[int] = []
    expected_shape: tuple[int, int, int] | None = None
    for camera_name, camera_index in CAMERA_SPECS:
        paths = camera_paths.get(camera_name)
        if not isinstance(paths, list) or len(paths) != frames_per_camera:
            raise ValueError(
                f"Expected {frames_per_camera} paths for {camera_name!r}"
            )

        frames: list[torch.Tensor] = []
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
            frames.append(tensor)
        camera_tensors.append(torch.stack(frames, dim=0))
        camera_indices.append(camera_index)

    return (
        torch.stack(camera_tensors, dim=0),
        torch.tensor(camera_indices, dtype=torch.int64),
    )


def native_ego_history_tensors(
    history: Mapping[str, Any],
    *,
    history_steps: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and tensorize PAI-AV's native current-ego 3D pose history."""

    if not isinstance(history, Mapping):
        raise ValueError("alpamayo_ego_history must be a mapping")
    if history.get("coordinate_frame") != "current_ego":
        raise ValueError("Alpamayo ego history must use coordinate_frame=current_ego")
    if history.get("frame_order") != "oldest_to_current":
        raise ValueError("Alpamayo ego history must be ordered oldest_to_current")
    interval_us = int(history.get("history_interval_us", -1))
    if interval_us != 100_000:
        raise ValueError(
            "Alpamayo ego history must be sampled at 10 Hz "
            f"(100000 us), got {interval_us}"
        )

    xyz = np.asarray(history.get("ego_history_xyz"), dtype=np.float32)
    rotations = np.asarray(history.get("ego_history_rot"), dtype=np.float32)
    if xyz.shape != (history_steps, 3):
        raise ValueError(
            f"Expected ego_history_xyz shape {(history_steps, 3)}, got {xyz.shape}"
        )
    if rotations.shape != (history_steps, 3, 3):
        raise ValueError(
            "Expected ego_history_rot shape "
            f"{(history_steps, 3, 3)}, got {rotations.shape}"
        )
    if not np.isfinite(xyz).all() or not np.isfinite(rotations).all():
        raise ValueError("Alpamayo ego history contains non-finite values")
    if not np.allclose(xyz[-1], 0.0, atol=1e-5):
        raise ValueError("Current ego-history translation must be zero")
    if not np.allclose(rotations[-1], np.eye(3), atol=1e-5):
        raise ValueError("Current ego-history rotation must be identity")
    identities = rotations @ np.swapaxes(rotations, -1, -2)
    if not np.allclose(identities, np.eye(3), atol=2e-4):
        raise ValueError("ego_history_rot contains a non-orthonormal rotation")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=2e-4):
        raise ValueError("ego_history_rot contains an improper rotation")

    timestamps = np.asarray(history.get("timestamps_us"), dtype=np.int64)
    if timestamps.shape != (history_steps,):
        raise ValueError(
            f"Expected {history_steps} ego-history timestamps, got {timestamps.shape}"
        )
    if not np.all(np.diff(timestamps) == interval_us):
        raise ValueError("Ego-history timestamps are not uniformly spaced at 10 Hz")
    if int(history.get("current_timestamp_us", -1)) != int(timestamps[-1]):
        raise ValueError("Current ego-history timestamp does not match final pose")

    return (
        torch.from_numpy(xyz.copy()).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(rotations.copy()).unsqueeze(0).unsqueeze(0),
    )


def extract_trajectory(
    pred_xyz: torch.Tensor, pred_rot: torch.Tensor
) -> np.ndarray:
    """Extract a single Alpamayo sample as rear-axle x/y/yaw poses."""

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
        history_steps: int = 16,
        frames_per_camera: int = 4,
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
        self.history_steps = int(history_steps)
        self.frames_per_camera = int(frames_per_camera)
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

    def _prepare_runtime(self) -> None:
        """Validate shared input/runtime settings without constructing a model."""
        if self.num_traj_samples != 1:
            raise ValueError(
                "The paired evaluator currently requires num_traj_samples=1"
            )
        if self.frames_per_camera != 4 or self.history_steps != 16:
            raise ValueError(
                "PAI-AV Alpamayo evaluation requires four camera frames and "
                "sixteen ego-history poses"
            )
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
        self.dtype = self._resolve_dtype()

    def initialize(self) -> None:
        self._prepare_runtime()
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

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

    def _make_model_input(
        self,
        agent_input: dict[str, Any],
        *,
        forced_cot: str | None = None,
    ) -> dict[str, Any]:
        if self.processor is None or self.helper is None:
            raise RuntimeError("AlpamayoAgent.initialize() must be called first")
        image_frames, camera_indices = load_camera_tensor(
            agent_input["camera_paths"],
            frames_per_camera=self.frames_per_camera,
        )
        history_xyz, history_rot = native_ego_history_tensors(
            agent_input["alpamayo_ego_history"],
            history_steps=self.history_steps,
        )
        messages = self.helper.create_message(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=self.frames_per_camera,
            nav_text=agent_input.get("navigation_command"),
        )
        if forced_cot is not None:
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(
                    "Alpamayo message template has no final assistant prefix"
                )
            messages[-1]["content"] = [
                {"type": "text", "text": forced_cot_prefix(forced_cot)}
            ]
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        return self.helper.to_device(
            {
                "tokenized_data": tokenized,
                "ego_history_xyz": history_xyz,
                "ego_history_rot": history_rot,
            },
            self.device,
        )

    @torch.inference_mode()
    def compute_trajectory(
        self,
        agent_input: dict[str, Any],
        *,
        forced_cot: str | None = None,
        diffusion_seed: int | None = None,
    ) -> AlpamayoPrediction:
        if self.model is None or self.dtype is None:
            raise RuntimeError("AlpamayoAgent.initialize() must be called first")
        model_input = self._make_model_input(
            agent_input,
            forced_cot=forced_cot,
        )
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
        with torch.autocast(
            device_type=device_type,
            dtype=self.dtype,
            enabled=device_type == "cuda",
        ):
            pred_xyz, pred_rot, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_input,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    num_traj_samples=self.num_traj_samples,
                    max_generation_length=(
                        # The forced prompt already ends in traj_future_start.
                        # One throwaway generation step makes Transformers
                        # return the complete prompt KV cache expected by the
                        # unchanged Alpamayo diffusion code.
                        1
                        if forced_cot is not None
                        else self.max_generation_length
                    ),
                    diffusion_kwargs=diffusion_kwargs,
                    return_extra=True,
                )
            )
        return AlpamayoPrediction(
            poses=extract_trajectory(pred_xyz, pred_rot),
            extra=extra,
        )
