"""Adapter around DriveWAM's official NAVSIM rollout evaluator (third_party/DriveWAM)."""
import copy
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ... import paths

DRIVEWAM_REPO_ROOT = Path(os.environ.get("DRIVEWAM_REPO_ROOT", paths.REPO_ROOT / "third_party" / "DriveWAM")).expanduser()


def navigation_command_to_index(navigation_command):
    command = np.asarray(navigation_command)
    if command.shape != (4,):
        raise ValueError(
            f"navigation_command must contain four values, got shape {command.shape}"
        )
    if not np.isfinite(command).all() or np.count_nonzero(command) != 1:
        raise ValueError(
            "navigation_command must be a finite one-hot vector with four values, "
            f"got {navigation_command}"
        )
    return int(np.argmax(command))


class DriveWAMAgent:
    """Small adapter around DriveWAM's official NAVSIM rollout evaluator."""

    def __init__(
        self,
        checkpoint_path,
        base_model_path,
        config_name="navsim_cfg",
        attn_mode="flashattn",
        device="cuda:0",
        num_video_steps=3,
        num_action_steps=10,
        video_denoise_fraction=0.6,
        seed=0,
        trajectory_sampling=None,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.base_model_path = Path(base_model_path).expanduser()
        self.config_name = config_name
        self.attn_mode = attn_mode
        self.device = device
        self.num_video_steps = int(num_video_steps)
        self.num_action_steps = int(num_action_steps)
        self.video_denoise_fraction = float(video_denoise_fraction)
        self.seed = int(seed)
        self.trajectory_sampling = trajectory_sampling

        self.config = None
        self.evaluator = None
        self._checkpoint_shim = None

    @staticmethod
    def _is_transformer_dir(path):
        if not (path / "config.json").is_file():
            return False
        weight_patterns = (
            "diffusion_pytorch_model*.safetensors",
            "diffusion_pytorch_model*.bin",
            "pytorch_model*.safetensors",
            "pytorch_model*.bin",
        )
        return any(any(path.glob(pattern)) for pattern in weight_patterns)

    def _resolve_transformer_dir(self):
        candidates = (
            self.checkpoint_path / "transformer",
            self.checkpoint_path / "navsim",
            self.checkpoint_path,
        )
        for candidate in candidates:
            if self._is_transformer_dir(candidate):
                return candidate.resolve()

        raise FileNotFoundError(
            "Could not find the DriveWAM NAVSIM transformer. Expected config.json and "
            "diffusion_pytorch_model*.safetensors in one of: "
            + ", ".join(str(path) for path in candidates)
            + ". Run scripts/download_checkpoints.sh or set checkpoint_path in "
            "reactbench/models/drivewam/config.yaml (or an alternate --config file)."
        )

    def _validate_base_model(self):
        required_subdirs = ("vae", "tokenizer", "text_encoder")
        missing = [
            name
            for name in required_subdirs
            if not (self.base_model_path / name).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"LingBot-VA base model is incomplete at {self.base_model_path}; "
                f"missing: {', '.join(missing)}. Run scripts/download_checkpoints.sh "
                "or set base_model_path in reactbench/models/drivewam/config.yaml (or an "
                "alternate --config file)."
            )

    def _checkpoint_parent_for_official_evaluator(self, transformer_dir):
        if transformer_dir.name == "transformer":
            return transformer_dir.parent

        # The official evaluator always appends '/transformer'. The released HF
        # checkpoint is under 'navsim', so present that directory through a
        # temporary compatibility layout without copying the weights.
        self._checkpoint_shim = tempfile.TemporaryDirectory(
            prefix="drivewam_checkpoint_"
        )
        shim_root = Path(self._checkpoint_shim.name)
        (shim_root / "transformer").symlink_to(transformer_dir, target_is_directory=True)
        return shim_root

    def initialize(self):
        if self.num_video_steps <= 0 or self.num_action_steps <= 0:
            raise ValueError("num_video_steps and num_action_steps must be positive")
        if not 0.0 <= self.video_denoise_fraction <= 1.0:
            raise ValueError("video_denoise_fraction must be between 0 and 1")
        if not 0 <= self.seed <= (1 << 63) - 1:
            raise ValueError(
                "seed must be between 0 and 2**63 - 1, "
                f"got {self.seed}"
            )
        if self.attn_mode not in {"torch", "flashattn"}:
            raise ValueError("attn_mode must be either 'torch' or 'flashattn'")
        if not DRIVEWAM_REPO_ROOT.is_dir():
            raise FileNotFoundError(
                f"DriveWAM repository not found at {DRIVEWAM_REPO_ROOT}. "
                "Run third_party/setup.sh or set DRIVEWAM_REPO_ROOT."
            )

        self._validate_base_model()
        transformer_dir = self._resolve_transformer_dir()

        # DriveWAM's release uses both ``src.*`` and top-level imports such as
        # ``configs``/``modules``. Put both roots ahead of the adapter modules.
        sys.path.insert(0, str(DRIVEWAM_REPO_ROOT))
        sys.path.insert(0, str(DRIVEWAM_REPO_ROOT / "src"))
        try:
            from src.configs import VA_CONFIGS
            from src.navsim.eval import NavSimEvaluator
        except ImportError as exc:
            raise ImportError(
                "Failed to import DriveWAM. Create the drivewam conda environment "
                "from envs/drivewam.yml before evaluation."
            ) from exc

        if self.config_name not in VA_CONFIGS:
            raise KeyError(
                f"Unknown DriveWAM config {self.config_name!r}; "
                f"available configs: {sorted(VA_CONFIGS)}"
            )

        self.config = copy.deepcopy(VA_CONFIGS[self.config_name])
        self.config.wan22_pretrained_model_name_or_path = str(
            self.base_model_path.resolve()
        )
        checkpoint_parent = self._checkpoint_parent_for_official_evaluator(
            transformer_dir
        )
        self.evaluator = NavSimEvaluator(
            self.config,
            checkpoint_path=str(checkpoint_parent),
            attn_mode=self.attn_mode,
            device=self.device,
        )

    def _load_image_tensor(self, image_path):
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Front-camera image does not exist: {image_path}")

        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB")).copy()

        # Match src/navsim/process_data.py: halve the camera frame with area
        # interpolation before the evaluator's final 256x448 resize.
        height, width = image_array.shape[:2]
        image_array = cv2.resize(
            image_array,
            (width // 2, height // 2),
            interpolation=cv2.INTER_AREA,
        )
        image_tensor = (
            torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        )

        # DriveWAM uses only frame 0 as observed context. The remaining slots
        # establish the one-step future-video rollout used by official NAVSIM
        # inference; their pixel values are never consumed.
        num_frames = 1 + len(self.config.future_image_indices)
        return image_tensor.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

    def _make_batch(self, agent_input):
        command_id = navigation_command_to_index(agent_input["navigation_command"])

        velocity = torch.as_tensor(agent_input["velocity"], dtype=torch.float32)
        acceleration = torch.as_tensor(
            agent_input["acceleration"], dtype=torch.float32
        )
        if velocity.numel() < 2 or acceleration.numel() < 2:
            raise ValueError("velocity and acceleration must each contain x/y values")

        history_poses = torch.as_tensor(
            agent_input["history_poses"], dtype=torch.float32
        )
        if history_poses.ndim != 2 or history_poses.shape[1] < 3:
            raise ValueError(
                "history_poses must have shape (frames, >=3) for x/y/heading"
            )
        history_poses = history_poses[:, :3]
        history_dim = int(self.config.history_num_frames)
        history_poses = history_poses[-history_dim:]
        if history_poses.shape[0] < history_dim:
            padding = torch.zeros(
                history_dim - history_poses.shape[0], 3, dtype=torch.float32
            )
            history_poses = torch.cat([padding, history_poses], dim=0)

        actions = torch.zeros(
            1,
            int(self.config.action_dim),
            int(self.config.action_chunk_count),
            int(self.config.action_chunk_steps),
            1,
            dtype=torch.float32,
        )
        actions_mask = torch.zeros_like(actions, dtype=torch.bool)
        action_channels = list(self.config.used_action_channel_ids)
        actions_mask[:, action_channels, 1, :, :] = True

        return {
            "images": self._load_image_tensor(agent_input["image_path"]),
            "prompt": [self.config.command_prompts[command_id]],
            "command_id": torch.tensor([command_id], dtype=torch.long),
            "ego_velocity_acc": torch.cat(
                [velocity[:2], acceleration[:2]], dim=0
            ).unsqueeze(0),
            "history_poses": history_poses.unsqueeze(0),
            "actions": actions,
            "actions_mask": actions_mask,
        }

    @torch.inference_mode()
    def compute_trajectory(self, agent_input, *, rollout_seed=None):
        """Return the predicted poses of one rollout (video and action noise drawn from ``rollout_seed``)."""
        if self.evaluator is None:
            raise RuntimeError("DriveWAMAgent.initialize() must be called first")
        rollout_seed = self.seed if rollout_seed is None else int(rollout_seed)
        if not 0 <= rollout_seed <= (1 << 63) - 1:
            raise ValueError(
                "rollout_seed must be between 0 and 2**63 - 1, "
                f"got {rollout_seed}"
            )
        batch = self._make_batch(agent_input)
        predicted_actions, _ = self.evaluator.predict(
            batch,
            num_video_steps=self.num_video_steps,
            num_action_steps=self.num_action_steps,
            seed=rollout_seed,
            video_denoise_fraction=self.video_denoise_fraction,
        )
        predicted_trajectory = self.evaluator.actions_to_trajectory(predicted_actions)
        return self.evaluator.pred_traj_to_poses(predicted_trajectory)
