"""Standalone ReCogDrive text planning, without the diffusion action head."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


PROMPT_VERSION = "upstream_base_single_v1"
DEFAULT_GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 512,
    "min_new_tokens": 50,
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POINT = rf"\(\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*\)"


class TrajectoryParseError(ValueError):
    """A generated answer cannot be used as an eight-pose trajectory."""


def parse_trajectory(response: str) -> np.ndarray:
    if not isinstance(response, str):
        raise TrajectoryParseError("Expected a text response")
    starts = list(re.finditer(r"\[\s*PT\b", response))
    if len(starts) != 1:
        raise TrajectoryParseError("Expected exactly one [PT, ...] block")
    start = starts[0].end()
    end = response.find("]", start)
    if end < 0:
        raise TrajectoryParseError("Incomplete trajectory block (missing closing ])")
    body = response[start:end]
    if not re.fullmatch(rf"\s*,\s*{POINT}(?:\s*,\s*{POINT})*\s*", body):
        raise TrajectoryParseError("Malformed trajectory: expected numeric (x, y, heading) points")
    coordinates = re.findall(POINT, body)
    if len(coordinates) != 8:
        raise TrajectoryParseError(f"Expected 8 trajectory points, got {len(coordinates)}")
    with np.errstate(over="ignore", invalid="ignore"):
        poses = np.asarray(coordinates, dtype=np.float32)
    if not np.isfinite(poses).all():
        raise TrajectoryParseError("Trajectory contains non-finite or overflowing values")
    return poses


def format_number(value: float) -> str:
    return f"{value:+.2f}" if abs(round(float(value), 2)) > 1e-2 else "0.0"


def build_base_prompt(agent_input: Any) -> str:
    statuses = agent_input.ego_statuses
    if len(statuses) != 4:
        raise ValueError("The base prompt requires four historical ego poses")
    history = " ".join(
        f"   - t-{3-i}: ({', '.join(format_number(v) for v in status.ego_pose)})"
        for i, status in enumerate(statuses)
    )
    commands = ["turn left", "go straight", "turn right"]
    active = np.flatnonzero(np.asarray(statuses[-1].driving_command) == 1)
    if len(active) > 1:
        raise ValueError("Expected at most one active navigation command")
    command = commands[active[0]] if len(active) and active[0] < len(commands) else "unknown"
    # Match upstream InternVLAgent's base/single prompt, including formatting.
    return (
        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{history}\n"
        f"3. Active navigation command: [{command.upper()}]"
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )


class ReCogDriveVLMAgent:
    """Duck-typed evaluation agent; heavy imports/loading happen in initialize."""

    def __init__(
        self,
        trajectory_sampling: Any,
        checkpoint_path: str,
        prompt_type: str = "base",
        cam_type: str = "single",
        generation_config: dict[str, Any] | None = None,
        device: str = "cuda",
    ):
        if prompt_type != "base" or cam_type != "single":
            raise ValueError("VLM baseline supports only prompt_type=base, cam_type=single")
        if trajectory_sampling.num_poses != 8 or not np.isclose(trajectory_sampling.interval_length, 0.5):
            raise ValueError("VLM baseline requires eight poses at 0.5-second intervals")
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = device
        self.generation_config = {**DEFAULT_GENERATION, **(generation_config or {})}
        if set(self.generation_config) != set(DEFAULT_GENERATION):
            raise ValueError("Only greedy decoding and min/max_new_tokens are configurable")
        if self.generation_config["do_sample"] is not False or self.generation_config["num_beams"] != 1:
            raise ValueError("VLM baseline requires greedy decoding (do_sample=False, num_beams=1)")
        minimum = self.generation_config["min_new_tokens"]
        maximum = self.generation_config["max_new_tokens"]
        if type(minimum) is not int or type(maximum) is not int or not 0 <= minimum < maximum:
            raise ValueError("Require integer token bounds: 0 <= min_new_tokens < max_new_tokens")
        self.model = None
        self.tokenizer = None
        self.last_generation: dict[str, Any] | None = None

    def provenance(self) -> dict[str, Any]:
        """Fingerprint local code/config and weight metadata without reading GBs."""
        from navsim.agents.recogdrive import recogdrive_backbone
        from navsim.agents.recogdrive.utils import internvl_preprocess

        checkpoint = self.checkpoint_path
        model_config = json.loads((checkpoint / "config.json").read_text())
        if model_config.get("force_image_size", 448) != 448:
            raise ValueError("Expected the ReCogDrive checkpoint's 448-pixel preprocessing")
        weights = sorted(checkpoint.glob("*.safetensors"))
        if not weights:
            raise FileNotFoundError(f"No safetensors weights found in {checkpoint}")
        files = sorted(
            path for path in checkpoint.iterdir()
            if path.is_file() and path.suffix in {".json", ".py", ".txt"}
            and path.name not in {"trainer_state.json", "train_results.json", "all_results.json"}
        )
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        implementation = {
            "vlm_agent": Path(__file__),
            "backbone": Path(recogdrive_backbone.__file__),
            "preprocessing": Path(internvl_preprocess.__file__),
        }
        return {
            "backend": "vlm",
            "prompt_version": PROMPT_VERSION,
            "checkpoint_path": str(checkpoint),
            "checkpoint_files_sha256": hashes,
            "weight_identity_method": "filename_size_mtime_ns",
            "weights": [
                {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                for p in weights
            ],
            "implementation_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in implementation.items()
            },
            "generation_config": dict(self.generation_config),
            "preprocessing": {"input_size": 448, "max_num": 12, "use_thumbnail": True},
            "dtype": "bfloat16",
            "device": self.device,
            "package_versions": {
                name: version(name)
                for name in ("torch", "transformers", "torchvision", "Pillow")
            },
        }

    def initialize(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from navsim.agents.recogdrive.recogdrive_backbone import system_message
        from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

        self.model = AutoModel.from_pretrained(
            str(self.checkpoint_path), torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
            use_flash_attn=True, device_map=self.device, local_files_only=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_path), trust_remote_code=True,
            use_fast=False, local_files_only=True,
        )
        self.model.system_message = system_message
        self._load_image = load_image

    def compute_trajectory(self, agent_input: Any) -> np.ndarray:
        import torch

        self.last_generation = None
        if self.model is None:
            raise RuntimeError("Call initialize() before VLM inference")
        question = build_base_prompt(agent_input)
        image_path = str(agent_input.cameras[-1].cam_f0.image)
        self.last_generation = {
            "prompt_version": PROMPT_VERSION,
            "prompt": question,
            "image_path": image_path,
            "generation_config": dict(self.generation_config),
            "response": None,
            "parse_status": "not_generated",
        }
        pixels = self._load_image(image_path, input_size=448, max_num=12).to(
            device=self.device, dtype=torch.bfloat16,
        )
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer, pixels, question, dict(self.generation_config),
                history=None, num_patches_list=[pixels.shape[0]],
            )
        self.last_generation["response"] = response
        try:
            poses = parse_trajectory(response)
        except TrajectoryParseError as exc:
            self.last_generation.update(parse_status="invalid", parse_error=str(exc))
            raise
        self.last_generation["parse_status"] = "valid"
        return poses
