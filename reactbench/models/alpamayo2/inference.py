"""Pinned official Alpamayo 2 inference and independent paired random streams."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ... import paths
from ..alpamayo1_5.agent_paiav import native_ego_history_tensors


UPSTREAM_REVISION = "5e7975f4a2100ee8ac1a62b79239bbabefddcbef"
MODEL_REVISION = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
MODEL_ID = "nvidia/Alpamayo2-Super"
SEED_POLICY = "SHA-256 of alpamayo2:<text|diffusion>:<base>:<token>; first 63 bits"
# Free memory kept on every GPU for activations, the KV cache and the CUDA context.
DEFAULT_GPU_RESERVE_GIB = 4.0


def paired_seed(token: str, base_seed: int, stream: str) -> int:
    if not isinstance(token, str) or not token or token in (".", "..") or Path(token).name != token:
        raise ValueError("token must be a nonempty filename component")
    if stream not in ("text", "diffusion"):
        raise ValueError("Unknown paired random stream")
    if not 0 <= int(base_seed) <= (1 << 63) - 1:
        raise ValueError("seed must be between 0 and 2**63 - 1")
    material = f"alpamayo2:{stream}:{int(base_seed)}:{token}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 1


def _cuda_indices(devices) -> list[int]:
    if isinstance(devices, (str, torch.device)):
        devices = [devices]
    indices = []
    for value in devices:
        device = torch.device(value)
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index not in indices:
                indices.append(index)
    return indices


@contextmanager
def seeded_rng(seed: int, devices):
    """Restore caller RNG state and seed the CPU and every selected CUDA device.

    A model split across GPUs may consume randomness on any device holding its
    layers, so every such device is seeded and restored.
    """
    cuda_devices = _cuda_indices(devices)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % (1 << 32))
            torch.random.default_generator.manual_seed(seed)
            for index in cuda_devices:
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def seeded_diffusion(diffusion: Any, seed: int, devices):
    """Seed at the official sampler boundary, independent of VLM token count.

    Only this model instance's bound method is temporarily wrapped. The sampler
    algorithm and upstream source are unchanged; restore the original attribute
    even when inference fails. Evaluation is sequential, one sample at a time.
    """
    original = diffusion.sample
    previous = vars(diffusion).get("sample")
    had_attribute = "sample" in vars(diffusion)

    def sample(*args, **kwargs):
        with seeded_rng(seed, devices):
            return original(*args, **kwargs)

    diffusion.sample = sample
    try:
        yield
    finally:
        if had_attribute:
            diffusion.sample = previous
        else:
            delattr(diffusion, "sample")


def hardware_profile(device: str | torch.device) -> dict[str, Any]:
    """Describe the GPUs a run uses; paired runs must share one numerics class.

    A seed reproduces the same rollout only on the same GPU architecture, so the
    class (CUDA compute capability) is part of the original-cache signature.
    Splitting layers across more GPUs of one architecture does not change it.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return {"class": device.type, "gpu_count": 0, "gpus": []}
    if not torch.cuda.is_available():
        raise RuntimeError("Alpamayo2 inference requires a CUDA GPU; CPU preflight needs no weights")
    indices = [device.index] if device.index is not None else list(range(torch.cuda.device_count()))
    gpus = []
    for index in indices:
        properties = torch.cuda.get_device_properties(index)
        gpus.append({"index": index, "name": properties.name,
                     "compute_capability": [properties.major, properties.minor],
                     "memory_bytes": int(properties.total_memory)})
    capabilities = {tuple(gpu["compute_capability"]) for gpu in gpus}
    if len(capabilities) != 1:
        raise RuntimeError(f"Visible GPUs must share one architecture, got {sorted(capabilities)}")
    major, minor = capabilities.pop()
    return {"class": f"cuda-sm{major}{minor}", "gpu_count": len(gpus), "gpus": gpus}


def _module_bytes(module: torch.nn.Module, dtype: torch.dtype) -> int:
    """Parameter and buffer bytes once floating tensors are loaded in ``dtype``."""
    floating = torch.empty((), dtype=dtype).element_size()
    return sum(tensor.numel() * (floating if tensor.is_floating_point() else tensor.element_size())
               for tensor in (*module.parameters(), *module.buffers()))


def _decoder_layers(model: torch.nn.Module, prefix: str) -> tuple[str, int]:
    found = [name for name, module in model.named_modules()
             if name.startswith(prefix) and name.rsplit(".", 1)[-1] == "layers"
             and isinstance(module, torch.nn.ModuleList)]
    if len(found) != 1:
        raise RuntimeError(f"Expected one decoder layer list under {prefix!r}, found {found}")
    return found[0], len(model.get_submodule(found[0]))


def layer_aligned_device_map(
    model: torch.nn.Module, capacities_bytes: Sequence[int], dtype: torch.dtype,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Split decoder layers across GPUs, keeping VLM layer i with expert layer i.

    The diffusion expert attends to the VLM's per-layer KV cache, so each expert
    layer must share a GPU with the VLM layer whose cache it reads. All other
    modules (vision tower, embeddings, heads, action projections) stay on GPU 0,
    where inputs and diffusion noise live. Contiguous layer ranges minimise the
    largest per-GPU load. Keys never overlap: accelerate hooks a mapped module
    as one block, so a root ("") entry would disable the split.
    """
    vlm_layers, vlm_count = _decoder_layers(model, "vlm.")
    expert_layers, expert_count = _decoder_layers(model, "expert.")
    if expert_count > vlm_count:
        raise RuntimeError("The expert has more layers than the VLM cache it reads")
    units = [[f"{vlm_layers}.{index}"] + ([f"{expert_layers}.{index}"] if index < expert_count else [])
             for index in range(vlm_count)]
    sizes = [sum(_module_bytes(model.get_submodule(name), dtype) for name in unit) for unit in units]
    head = _module_bytes(model, dtype) - sum(sizes)
    gpu_count = len(capacities_bytes)

    def assign(limit):
        devices, gpu, load = [], 0, head
        for size in sizes:
            if load + size > limit:
                gpu, load = gpu + 1, 0
                if gpu == gpu_count:
                    return None
            devices.append(gpu)
            load += size
        return devices

    low, high = max([head, *sizes]), head + sum(sizes)
    while low < high:
        middle = (low + high) // 2
        low, high = (low, middle) if assign(middle) is not None else (middle + 1, high)
    devices = assign(low)
    loads = [head] + [0] * (gpu_count - 1)
    for size, gpu in zip(sizes, devices):
        loads[gpu] += size
    if any(load > capacity for load, capacity in zip(loads, capacities_bytes)):
        raise RuntimeError(
            f"Alpamayo2 needs about {max(loads) / 2**30:.1f} GiB per GPU on {gpu_count} GPUs, but only "
            f"{min(capacities_bytes) / 2**30:.1f} GiB per GPU is available after headroom; allocate more GPUs")
    assignments = {name: gpu for unit, gpu in zip(units, devices) for name in unit}
    device_map: dict[str, int] = {}

    def cover(name: str, module: torch.nn.Module) -> None:
        if name in assignments:
            device_map[name] = assignments[name]
            return
        prefix = f"{name}." if name else ""
        if not any(key.startswith(prefix) for key in assignments):
            device_map[name] = 0
            return
        for tensor_name, _ in (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)):
            device_map[prefix + tensor_name] = 0
        for child_name, child in module.named_children():
            cover(prefix + child_name, child)

    cover("", model)
    return device_map, {"vlm_layers": vlm_layers, "expert_layers": expert_layers,
                        "layers_per_gpu": [devices.count(gpu) for gpu in range(gpu_count)],
                        "estimated_bytes_per_gpu": loads}


@dataclass
class Alpamayo2Prediction:
    poses: np.ndarray
    xyz: np.ndarray
    rotations: np.ndarray
    extra: dict[str, Any]


def extract_prediction(pred_xyz, pred_rot, extra) -> Alpamayo2Prediction:
    if tuple(pred_xyz.shape) != (1, 1, 1, 64, 3):
        raise ValueError(f"Expected pred_xyz (1,1,1,64,3), got {tuple(pred_xyz.shape)}")
    if tuple(pred_rot.shape) != (1, 1, 1, 64, 3, 3):
        raise ValueError(f"Expected pred_rot (1,1,1,64,3,3), got {tuple(pred_rot.shape)}")
    xyz = pred_xyz[0, 0, 0].detach().float().cpu().numpy().copy()
    rotations = pred_rot[0, 0, 0].detach().float().cpu().numpy().copy()
    if not np.isfinite(xyz).all() or not np.isfinite(rotations).all():
        raise ValueError("Prediction contains non-finite XYZ or rotations")
    if not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=5e-3):
        raise ValueError("Prediction contains non-orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=5e-3):
        raise ValueError("Prediction contains improper rotations")
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    poses = np.column_stack((xyz[:, :2], yaw)).astype(np.float32)
    return Alpamayo2Prediction(poses, xyz, rotations, dict(extra))


def input_tensors(agent_input: Mapping[str, Any], camera_loader):
    images, camera_ids = camera_loader(agent_input["camera_paths"], frames_per_camera=4)
    xyz, rot = native_ego_history_tensors(agent_input["alpamayo_ego_history"])
    timestamps = np.asarray(agent_input["image_timestamps_us"])
    if timestamps.shape != (images.shape[0], 4) or timestamps.dtype.kind not in "iu":
        raise ValueError("image_timestamps_us must contain four integer timestamps per camera")
    if not np.all(np.diff(timestamps, axis=1) > 0):
        raise ValueError("Camera timestamps must be ordered oldest to current")
    current = int(agent_input["alpamayo_ego_history"]["current_timestamp_us"])
    # Archive decoders select the nearest real camera exposure, as upstream does.
    # Allow at most half a native 10 Hz interval around the current ego timestamp.
    if np.any(np.abs(timestamps[:, -1] - current) > 50_000):
        raise ValueError("Current camera frames are not aligned with the current ego pose")
    return images, camera_ids, xyz, rot


class Alpamayo2Inference:
    """Dataset-independent VLM plus diffusion rollout using NVIDIA's API."""

    def __init__(
        self, *, model_name_or_path=MODEL_ID, model_revision=MODEL_REVISION,
        upstream_revision=UPSTREAM_REVISION, upstream_root=None, device="cuda",
        dtype="bfloat16", attn_implementation="sdpa",
        local_files_only=False, top_p=0.98, top_k=None, temperature=0.6,
        max_generation_length=256, seed=42, diffusion_seed_base=42,
        diffusion_kwargs=None, history_steps=16, frames_per_camera=4,
        num_traj_samples=1, trajectory_sampling=None, metric_horizon_s=4.0,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.model_revision, self.upstream_revision = str(model_revision), str(upstream_revision)
        self.upstream_root = Path(upstream_root or os.environ.get(
            "ALPAMAYO2_REPO_ROOT", paths.REPO_ROOT / "third_party" / "alpamayo2")).expanduser().resolve()
        self.device, self.dtype_name = str(device), str(dtype)
        self.attn_implementation = str(attn_implementation)
        self.local_files_only = bool(local_files_only)
        self.top_p, self.top_k, self.temperature = float(top_p), top_k, float(temperature)
        self.max_generation_length = int(max_generation_length)
        self.seed, self.diffusion_seed_base = int(seed), int(diffusion_seed_base)
        self.diffusion_kwargs = dict(diffusion_kwargs or {"inference_step": 10})
        self.model = self.processor = self.helper = self.conversation_builder = None
        if not re.fullmatch(r"cuda(:\d+)?|cpu", self.device):
            raise ValueError("device must be cuda (all visible GPUs), cuda:N, or cpu for tests")
        # "cuda" uses every visible GPU, splitting layers when one cannot hold the model.
        self.input_device, self.rng_devices = torch.device(self.device), [self.device]
        self.placement = self._checkpoint = self._hardware = None
        if history_steps != 16 or frames_per_camera != 4 or num_traj_samples != 1:
            raise ValueError("Basic evaluation requires 16 ego poses, four camera frames, one trajectory")
        sampling = trajectory_sampling or {"time_horizon": 6.4, "interval_length": 0.1}
        if (not np.isclose(sampling.get("time_horizon", 0), 6.4)
                or not np.isclose(sampling.get("interval_length", 0), 0.1)
                or not np.isclose(metric_horizon_s, 4.0)):
            raise ValueError("Alpamayo2 saves 64 poses at 0.1 s and scores the four-second prefix")
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported dtype: {dtype}")
        if not (np.isfinite(self.temperature) and self.temperature > 0
                and 0 < self.top_p <= 1 and self.max_generation_length > 0):
            raise ValueError("Invalid generation settings")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be null or positive")
        for revision in (self.model_revision, self.upstream_revision):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Model and upstream revisions must be full commit hashes")
        allowed = {"inference_step", "int_method", "temperature"}
        if set(self.diffusion_kwargs) - allowed:
            raise ValueError("Unsupported diffusion settings; seed isolation is managed by the evaluator")
        steps = self.diffusion_kwargs.get("inference_step", 10)
        if (type(steps) is not int or steps < 1
                or self.diffusion_kwargs.get("int_method", "euler") != "euler"
                or not np.isfinite(self.diffusion_kwargs.get("temperature", 1.0))
                or self.diffusion_kwargs.get("temperature", 1.0) <= 0):
            raise ValueError("Invalid diffusion settings")
        paired_seed("validation", self.seed, "text")
        paired_seed("validation", self.diffusion_seed_base, "diffusion")

    def _check_loaded_modules(self) -> None:
        source = self.upstream_root / "src"
        for name, module in tuple(sys.modules.items()):
            if name == "alpamayo2_super" or name.startswith("alpamayo2_super."):
                filename = getattr(module, "__file__", None)
                if filename and not Path(filename).resolve().is_relative_to(source):
                    raise ValueError(f"Upstream module {name} was loaded from another checkout: {filename}")

    def identity(self) -> dict[str, Any]:
        """Validate provenance before cache reuse or model/weight downloads."""
        if not (self.upstream_root / "src/alpamayo2_super").is_dir():
            raise FileNotFoundError(f"Initialize the Alpamayo2 submodule at {self.upstream_root}")
        commit = subprocess.check_output(
            ["git", "-C", str(self.upstream_root), "rev-parse", "HEAD"], text=True).strip()
        if commit != self.upstream_revision:
            raise ValueError(f"Upstream revision mismatch: {commit} != {self.upstream_revision}")
        changed = subprocess.run(["git", "-C", str(self.upstream_root), "diff", "--quiet",
                                  "HEAD", "--", "src", "pyproject.toml", "uv.lock"], check=False)
        if changed.returncode:
            raise ValueError("Pinned upstream inference source has local changes")
        self._check_loaded_modules()
        identity = {"model_name_or_path": self.model_name_or_path,
                    "model_revision": self.model_revision, "upstream_revision": commit,
                    "seed_policy": SEED_POLICY, "adapter_version": 1}
        local = Path(self.model_name_or_path).expanduser()
        if local.is_dir():
            from ..original_cache import file_sha256
            config = local / "config.json"
            shards = sorted(local.glob("*.safetensors"))
            if not config.is_file() or not shards:
                raise ValueError("Local checkpoint requires config.json and safetensors shards")
            identity["local_checkpoint"] = {
                "path": str(local.resolve()), "config_sha256": file_sha256(config),
                "files": [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                          for p in sorted(local.iterdir()) if p.is_file()],
            }
        return identity

    def hardware(self) -> dict[str, Any]:
        if self._hardware is None:
            self._hardware = hardware_profile(self.device)
        return self._hardware

    def placement_report(self) -> dict[str, Any] | None:
        """GPU placement without the per-module device map."""
        if self.placement is None:
            return None
        return {key: value for key, value in self.placement.items() if key != "device_map"}

    def prepare(self) -> None:
        """Validate the environment, resolve the pinned snapshot, and plan GPU placement.

        Weights are not loaded, so an unusable allocation fails before a result
        directory is claimed.
        """
        if self._checkpoint is not None:
            return
        self.identity()
        hardware = self.hardware()
        if not hardware["gpu_count"]:
            raise RuntimeError("Alpamayo2 inference requires a CUDA GPU; CPU preflight needs no weights")
        if self.attn_implementation == "flash_attention_2":
            import flash_attn  # noqa: F401
        if paths.get("hf_home", required=False):
            os.environ.setdefault("HF_HOME", str(paths.get("hf_home")))
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
        source = str(self.upstream_root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from huggingface_hub import snapshot_download
        import alpamayo2_super.models.alpamayo2_super  # noqa: F401
        self._check_loaded_modules()

        local = Path(self.model_name_or_path).expanduser()
        checkpoint = str(local.resolve()) if local.is_dir() else snapshot_download(
            repo_id=self.model_name_or_path, revision=self.model_revision,
            cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=self.local_files_only)
        self.placement = self._plan_placement(Path(checkpoint), hardware)
        self._checkpoint = checkpoint

    def _plan_placement(self, checkpoint: Path, hardware: Mapping[str, Any]) -> dict[str, Any]:
        reserve = int(float(os.environ.get("ALPAMAYO2_GPU_RESERVE_GIB", DEFAULT_GPU_RESERVE_GIB)) * 2**30)
        gpus = hardware["gpus"]
        capacities = [gpu["memory_bytes"] - reserve for gpu in gpus]
        if len(gpus) == 1:
            weights = sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))
            if weights > capacities[0]:
                raise RuntimeError(
                    f"{weights / 2**30:.1f} GiB of weights plus {reserve / 2**30:.1f} GiB headroom exceed one "
                    f"{gpus[0]['name']}; allocate several GPUs")
            return {"strategy": "single_gpu", "device_map": f"cuda:{gpus[0]['index']}",
                    "gpu_indices": [gpus[0]["index"]], "estimated_bytes_per_gpu": [weights],
                    "reserve_bytes": reserve}
        from accelerate import init_empty_weights
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
        config = Alpamayo2Super.config_class.from_pretrained(checkpoint, local_files_only=True)
        config.name_or_path = str(checkpoint)  # Tokenizer files come from the pinned snapshot.
        with init_empty_weights():
            skeleton = Alpamayo2Super(config)
        device_map, summary = layer_aligned_device_map(skeleton, capacities, getattr(torch, self.dtype_name))
        return {"strategy": "layer_aligned", "device_map": device_map,
                "gpu_indices": [gpu["index"] for gpu in gpus], "reserve_bytes": reserve, **summary}

    def load(self) -> None:
        self.prepare()
        if self.model is not None:
            return
        from alpamayo2_super import helper
        from alpamayo2_super.chat_template.conversation import build_conversation
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

        # The snapshot was resolved first: upstream loads its tokenizer/processor
        # from config._name_or_path, which otherwise could point at a moving HF branch.
        model = Alpamayo2Super.from_pretrained(
            self._checkpoint, dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            local_files_only=True, device_map=self.placement["device_map"])
        model.eval()
        self.input_device = model.get_input_embeddings().weight.device
        self.rng_devices = [torch.device("cuda", index) for index in self.placement["gpu_indices"]]
        self.processor = helper.get_processor(model.tokenizer, model.config)
        self.helper, self.conversation_builder = helper, build_conversation
        self.model = model

    def initialize(self):
        self.prepare()
        self.load()

    def _make_model_input(self, agent_input):
        images, camera_ids, xyz, rot = input_tensors(agent_input, self.camera_loader)
        data = {"image_frames": images, "camera_indices": camera_ids}
        nav = agent_input.get("navigation_command")
        if nav is None:
            messages = self.helper.create_messages(data, self.model.config)
        else:
            if not isinstance(nav, str) or not nav.strip():
                raise ValueError("Navigation text must be a nonempty string or null")
            # The same guided prompt construction as NVIDIA's navigation demo;
            # ordinary single-prompt inference, with no CFG intervention.
            data["nav_text"] = [nav]
            config = self.model.config
            messages = self.conversation_builder(
                data=data, num_tokens_per_history_traj=config.tokens_per_history_traj,
                num_tokens_per_future_traj=config.tokens_per_future_traj,
                components_order=["image", "traj_history", "nav_instruction", "prompt"],
                components_prompt=["cot", "traj_future"], generation_mode=True,
                include_camera_ids=config.include_camera_ids, camera_ids=camera_ids,
                include_frame_nums=config.frame_label == "frame_num")
            if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
                messages = messages[:-1]
        has_assistant = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not has_assistant,
            add_vision_id=False, continue_final_message=has_assistant)
        tokenized = dict(self.processor(
            text=text, images=images.flatten(0, 1).float() / 255.0, videos=None,
            padding=False, return_tensors="pt", do_rescale=False))
        if tokenized["input_ids"].shape[0] != 1:
            raise ValueError("Only one sample is supported")
        return self.helper.to_device(
            {"tokenized_data": tokenized, "ego_history_xyz": xyz, "ego_history_rot": rot},
            self.input_device)

    @torch.inference_mode()
    def compute_trajectory(self, agent_input, *, text_seed: int, diffusion_seed: int):
        if self.model is None or self.processor is None:
            raise RuntimeError("initialize() must be called before inference")
        for seed in (text_seed, diffusion_seed):
            if not 0 <= int(seed) <= (1 << 63) - 1:
                raise ValueError("Invalid rollout seed")
        model_input = self._make_model_input(agent_input)
        device_type = self.input_device.type
        with seeded_rng(text_seed, self.rng_devices), seeded_diffusion(
                self.model.expert.diffusion, diffusion_seed, self.rng_devices), torch.autocast(
                    device_type=device_type, dtype=getattr(torch, self.dtype_name),
                    enabled=device_type == "cuda"):
            xyz, rotations, _logprob, extra = self.model.sample_trajectories_from_data(
                data=model_input, top_p=self.top_p, top_k=self.top_k,
                temperature=self.temperature, num_traj_samples=1, num_traj_sets=1,
                max_generation_length=self.max_generation_length,
                diffusion_kwargs=dict(self.diffusion_kwargs), return_extra=True)
        prediction = extract_prediction(xyz, rotations, extra)
        prediction.extra.update({"text_seed": text_seed, "diffusion_seed": diffusion_seed,
                                 "image_timestamps_us": agent_input["image_timestamps_us"],
                                 "navigation_command": agent_input.get("navigation_command")})
        return prediction
