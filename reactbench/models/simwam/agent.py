"""Adapter around SimWAM's official action-only NAVSIM inference (no video generation)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image


from ... import paths
from ...io import file_digest
from ..seeds import MAX_SEED, paired_seed

SIMWAM_REPO_ROOT = Path(os.environ.get("SIMWAM_REPO_ROOT", paths.REPO_ROOT / "third_party" / "SimWAM")).expanduser()
VAE_FILE = Path("DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors")
SUPPORTED_DATA = {"camera_layout": "front", "video_frame_mode": "current_plus_future",
                  "trajectory_mode": "absolute", "normalize_action": True, "use_dynamic_prompt": False}


def upstream_dataset_class():
    for path in (SIMWAM_REPO_ROOT / "navsim", SIMWAM_REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from simwam.datasets.navsim.navsim_dataset import NavSimVideoDataset
    return NavSimVideoDataset


def compose_upstream_config(task):
    """The same Hydra composition as experiments/navsim/eval_navsim.py (sim_navsim + task)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    upstream_dataset_class()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(SIMWAM_REPO_ROOT / "configs"), version_base="1.3"):
        return compose(config_name="sim_navsim", overrides=[f"task={task}"])


def data_settings(cfg):
    """Read only the data fields used by inference, without resolving dataset path variables."""
    train = cfg.data.train
    settings = {key: train[key] for key in ("camera_layout", "video_frame_mode", "trajectory_mode",
                "normalize_action", "use_dynamic_prompt", "use_current_frame_prompt", "context_len",
                "action_dim", "state_dim", "future_action_horizon")}
    settings["video_size"] = [int(value) for value in train.video_size]
    settings["text_encoder_id"] = train.get("text_encoder_id", "wan22ti2v5b")
    for key, expected in SUPPORTED_DATA.items():
        if settings[key] != expected:
            raise ValueError(f"Unsupported upstream data setting {key}={settings[key]!r}; expected {expected!r}")
    return settings


def preprocess_image(dataset_class, image_path, video_size):
    """Upstream front-camera frame tensor ([1, 3, H, W] in [-1, 1]) for one image file."""
    with Image.open(image_path) as image:
        array = np.array(image.convert("RGB"))
    frame = SimpleNamespace(cameras=SimpleNamespace(cam_f0=SimpleNamespace(image=array)))
    helper = SimpleNamespace(camera_layout="front", _video_size_hw=[int(v) for v in video_size])
    helper._extract_frame_image = lambda cameras: dataset_class._extract_frame_image(helper, cameras)
    video = dataset_class._build_video_tensor(helper, SimpleNamespace(frames=[frame]), [0])
    return video[:, 0].unsqueeze(0)


def fixed_prompt(dataset_class, settings):
    """The single fixed prompt cached by upstream scripts/precompute_navsim_text_embeds.py."""
    import torch

    return dataset_class.build_prompt_fixed(
        hist_xyh=torch.zeros((4, 3), dtype=torch.float32), high_cmd_one_hot=torch.zeros(4),
        speed_mps=0.0, acc_mps2=0.0, use_dynamic_prompt=False,
        use_current_frame_prompt=bool(settings["use_current_frame_prompt"]))


def prompt_embedding_path(prompt, directory, settings):
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return Path(directory) / f"{hashed}.t5_len{settings['context_len']}.{settings['text_encoder_id']}.pt"


class SimWAMAgent:
    def __init__(self, checkpoint_path, diffsynth_model_root, prompt_embedding_dir, *,
                 checkpoint_repo, checkpoint_revision, upstream_revision,
                 upstream_task="navsim_uncond_front_384x672_1e-4", device="cuda", mixed_precision="bf16",
                 num_inference_steps=10, sigma_shift=None, rand_device="cpu", tiled=False, seed=0,
                 per_token_seed=True, load_pretrained_init=False, trajectory_sampling=None):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.diffsynth_model_root = Path(diffsynth_model_root).expanduser()
        self.prompt_embedding_dir = Path(prompt_embedding_dir).expanduser()
        self.checkpoint_repo = str(checkpoint_repo)
        self.checkpoint_revision = str(checkpoint_revision)
        self.upstream_revision = str(upstream_revision)
        self.upstream_task = str(upstream_task)
        self.device = str(device)
        self.mixed_precision = str(mixed_precision)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = None if sigma_shift is None else float(sigma_shift)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.seed = int(seed)
        self.per_token_seed = bool(per_token_seed)
        self.load_pretrained_init = bool(load_pretrained_init)
        self.trajectory_sampling = dict(trajectory_sampling or {"time_horizon": 4, "interval_length": 0.5})
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not 0 <= self.seed <= MAX_SEED:
            raise ValueError("seed must be between 0 and 2**63 - 1")
        if self.mixed_precision not in ("bf16", "fp16", "no"):
            raise ValueError("mixed_precision must be bf16, fp16 or no")
        self.model = self.settings = self.dataset_class = None
        self.context = self.context_mask = None
        self.load_report = {}

    @property
    def interval_length_s(self):
        return float(self.trajectory_sampling["interval_length"])

    def rollout_seed(self, token):
        return paired_seed("simwam-action-noise", self.seed, token) if self.per_token_seed else self.seed

    def identity(self):
        stat = self.checkpoint_path.stat() if self.checkpoint_path.is_file() else None
        return {"backend": "simwam-official-infer_action", "checkpoint_repo": self.checkpoint_repo,
                "checkpoint_revision": self.checkpoint_revision, "checkpoint_file": self.checkpoint_path.name,
                "checkpoint_size": None if stat is None else stat.st_size,
                "upstream_revision": self.upstream_revision, "upstream_task": self.upstream_task,
                "mixed_precision": self.mixed_precision, "num_inference_steps": self.num_inference_steps,
                "sigma_shift": self.sigma_shift, "rand_device": self.rand_device, "tiled": self.tiled,
                "seed_policy": "sha256_per_token" if self.per_token_seed else "fixed",
                "load_pretrained_init": self.load_pretrained_init,
                "trajectory_sampling": self.trajectory_sampling}

    def prepare(self):
        """CPU checks that the upstream source, configuration and assets are usable."""
        if not (SIMWAM_REPO_ROOT / "src/simwam").is_dir():
            raise FileNotFoundError(f"SimWAM source not found at {SIMWAM_REPO_ROOT}; "
                                    "run third_party/setup.sh")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing SimWAM checkpoint {self.checkpoint_path}; run scripts/download_checkpoints.sh")
        if not (self.diffsynth_model_root / VAE_FILE).is_file():
            raise FileNotFoundError(f"Missing Wan2.2 VAE under {self.diffsynth_model_root}; "
                                    "run scripts/download_checkpoints.sh")
        self.dataset_class = upstream_dataset_class()
        self.settings = data_settings(compose_upstream_config(self.upstream_task))
        if self.settings["future_action_horizon"] * self.interval_length_s != self.trajectory_sampling["time_horizon"]:
            raise ValueError("trajectory_sampling disagrees with the upstream action horizon")
        prompt = fixed_prompt(self.dataset_class, self.settings)
        path = prompt_embedding_path(prompt, self.prompt_embedding_dir, self.settings)
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixed-prompt embedding {path}; run python -m reactbench.models.simwam.prompt_embedding")
        self.load_report.update(prompt=prompt, prompt_embedding=str(path),
                                prompt_embedding_sha256=file_digest(path), data_settings=self.settings)
        return self.load_report

    def load(self):
        import torch
        from hydra.utils import instantiate

        if self.settings is None:
            self.prepare()
        cfg = compose_upstream_config(self.upstream_task)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(self.diffsynth_model_root)
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[self.mixed_precision]
        # Without the pretrained initialisation every weight comes from the checkpoint,
        # so loading below insists that the checkpoint covers every parameter.
        model = instantiate(cfg.model, model_dtype=dtype, device=self.device,
                            skip_dit_load_from_pretrain=not self.load_pretrained_init)
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if "mot" not in payload or "proprio_encoder" not in payload:
            raise ValueError(f"Checkpoint lacks mot/proprio_encoder weights: {self.checkpoint_path}")
        mot = model.mot.load_state_dict(payload["mot"], strict=False)
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if mot.unexpected_keys or (mot.missing_keys and not self.load_pretrained_init):
            raise ValueError(f"Checkpoint does not cover the model: missing={mot.missing_keys[:10]} "
                             f"unexpected={mot.unexpected_keys[:10]}")
        self.load_report.update(checkpoint_step=payload.get("step"), missing_keys=list(mot.missing_keys),
                                model_paths={k: str(v) for k, v in getattr(model, "model_paths", {}).items()})
        del payload
        self.model = model.to(self.device).eval()
        # Exact upstream cached-context post-processing (bf16, zeroed padding, all-ones mask).
        helper = SimpleNamespace(_text_context_cache={}, text_embedding_cache_dir=str(self.prompt_embedding_dir),
                                 context_len=int(self.settings["context_len"]),
                                 text_encoder_id=self.settings["text_encoder_id"])
        self.context, self.context_mask = self.dataset_class._get_cached_text_context(
            helper, self.load_report["prompt"])

    def compute_trajectory(self, model_input, *, rollout_seed):
        """Return native (8, 3) x/y/heading poses in the current rear-axle frame."""
        import torch

        if self.model is None:
            raise RuntimeError("SimWAMAgent.load() must be called first")
        image = preprocess_image(self.dataset_class, model_input["image_path"], self.settings["video_size"])
        state = torch.as_tensor(np.asarray(model_input["state"], dtype=np.float32))
        if state.shape != (int(self.settings["state_dim"]),):
            raise ValueError(f"Expected an {self.settings['state_dim']}-D ego state, got {tuple(state.shape)}")
        with torch.no_grad():
            output = self.model.infer_action(
                prompt=None, context=self.context, context_mask=self.context_mask, input_image=image,
                action_horizon=int(self.settings["future_action_horizon"]), proprio=state,
                negative_prompt="", text_cfg_scale=1.0, num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift, seed=int(rollout_seed), rand_device=self.rand_device,
                tiled=self.tiled)
        action = output["action"]
        poses = self.dataset_class.denorm_odo(action.unsqueeze(0).float())[0].numpy()
        expected = (int(self.settings["future_action_horizon"]), 3)
        if poses.shape != expected or not np.isfinite(poses).all():
            raise ValueError(f"Expected finite {expected} poses, got {poses.shape}")
        return poses.astype(np.float32)
