# Copyright 2025 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import hashlib
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from timm.models.layers import Mlp
from torch import nn
from torch.distributions import Beta, Normal, kl_divergence
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_scores
from navsim.evaluate.raw_pdm_score import (
    ReferenceProgressCache, executed_reference_progress, raw_pdm_scores, training_reward_protocol,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from .blocks.encoder import (
    ActionEncoder,
    SinusoidalPositionalEncoding,
    StateAttentionEncoder,
    SwiGLUFFN,
)
from .recogdrive_dit import LightningDiT

MOTION_DROPOUT_PROTOCOL = "recogdrive-motion-dropout-seed-rank-forward-v1"
JOINT_DROPOUT_PROTOCOL = "recogdrive-joint-dropout-seed-rank-forward-v1"
PDM_REWARD_COMPONENTS = {
    "nc": "no_at_fault_collisions",
    "dac": "drivable_area_compliance",
    "ttc": "time_to_collision_within_bound",
    "ep": "ego_progress",
    "comfort": "comfort",
}
RAW_SAFETY_DIAGNOSTICS = (
    "executed_reward", "raw_nc_all", "raw_dac", "raw_safety", "raw_contact_penalty", "total_reward",
    "raw_contact_speed_sum", "raw_contact_speed_count", "raw_initial_contact_count",
)


def planner_state_from_checkpoint(planner, checkpoint):
    """Load student weights, allowing only new null state to be absent in legacy files."""
    from .recogdrive_conditioning import validate_conditioning_checkpoint
    validate_conditioning_checkpoint(getattr(planner, "config", None), checkpoint)
    source = {}
    for name, value in checkpoint["state_dict"].items():
        name = name.removeprefix("agent.").removeprefix("action_head.")
        source[name] = value
    target = {k: v for k, v in planner.state_dict().items() if not k.startswith("old_policy.")}
    if "velacc_dropout" not in checkpoint and "null_motion_hidden" in target:
        source.setdefault("null_motion_hidden", torch.zeros_like(target["null_motion_hidden"]))
    if "joint_motion_dropout" not in checkpoint and "null_history" in target:
        source.setdefault("null_history", torch.zeros_like(target["null_history"]))
    missing = [k for k, v in target.items() if k not in source or source[k].shape != v.shape]
    if missing:
        raise ValueError(f"Checkpoint missing/incompatible inference weights: {missing[:12]}")
    return {k: source[k] for k in target}

@dataclass
class FlowConfig:
    """Configuration specific to Flow Matching."""
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    mean_variance_net: bool = False

@dataclass
class DDPMConfig:
    """Configuration specific to DDPM."""
    num_train_timesteps: int = 100

@dataclass
class DDIMConfig:
    """Configuration specific to DDIM."""
    num_train_timesteps: int = 100
    ddim_eta: float = 0.0

@dataclass
class GRPOConfig:
    """Configuration specific to GRPO training."""
    denoised_clip_value: float = 1.0
    eval_randn_clip_value: float = 1.0
    randn_clip_value: float = 5.0
    final_action_clip_value: float = 1.0
    eps_clip_value: Optional[float] = None
    eval_min_sampling_denoising_std: float = 0.0001
    min_sampling_denoising_std: float = 0.04
    min_logprob_denoising_std: float = 0.1
    clip_advantage_lower_quantile: float = 0.0
    clip_advantage_upper_quantile: float = 1.0
    gamma_denoising: float = 0.6
    
    metric_cache_path: str = "/path/to/metric_cache_train"
    metric_cache_staging_path: Optional[str] = None
    reference_policy_checkpoint: str = "/path/to/IL_Model.ckpt"
    planner_initialization: Literal["stage2", "random"] = "stage2"
    pdm_reward_workers: int = 1
    training_reward_mode: Literal["executed", "raw", "executed_raw_safety", "minimum_jerk"] = "executed"
    training_raw_safety_weight: float = 0.2
    training_raw_contact_speed_weight: float = 0.05
    training_raw_safety_inputs: Optional[str] = None
    training_minimum_jerk: Optional[dict] = None
    # Opt-in: no exploration noise at the last denoising step of rewarded rollouts.
    deterministic_final_step: bool = False
    # Weight of the reward-ranked speed-sibling denoising loss (minimum_jerk only); 0 disables it.
    sibling_weight: float = 0.0
    training_siblings: Optional[dict] = None
    scorer_config: PDMScorerConfig = field(default_factory=lambda: PDMScorerConfig(
        progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0
    ))


@dataclass
class ReCogDriveDiffusionPlannerConfig(PretrainedConfig):
    """A refined configuration for the ReCogDriveDiffusionPlanner."""
    # --- Core Architecture ---
    diffusion_model_cfg: dict = field(default_factory=dict)
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    action_dim: int = 3
    action_horizon: int = 8
    add_pos_embed: bool = True
    max_seq_len: int = 8
    ego_status_encoder_type: Literal['mlp', 'attention'] = 'mlp'
    velacc_dropout_probability: float = 0.0
    velacc_dropout_seed: int = 0
    joint_motion_dropout_probability: float = 0.0
    conditioning_mode: str = "hidden_states"
    mask_vlm_padding: bool = False
    vlm_kv_layers: list = field(default_factory=list)
    vlm_identity: Optional[dict] = None

    sampling_method: Literal['flow', 'ddpm', 'ddim'] = 'ddim'
    num_inference_steps: int = 5
    model_dtype: str = "float16"
    grpo: bool = False
    vlm_size: str = 'large'
    
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    
    flow_cfg: FlowConfig = field(default_factory=FlowConfig)
    ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
    ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
    grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)


class ReCogDriveDiffusionPlanner(nn.Module):
    config_class = ReCogDriveDiffusionPlannerConfig

    def __init__(self, config: ReCogDriveDiffusionPlannerConfig):
        super().__init__()
        self.config = config
        if config.conditioning_mode not in ("hidden_states", "vlm_kv"):
            raise ValueError("conditioning_mode must be hidden_states or vlm_kv")
        if config.grpo and (config.conditioning_mode == "vlm_kv" or config.mask_vlm_padding):
            raise ValueError("VLM KV/masked conditioning is implemented for stage 2 only; GRPO is pending")
        if config.conditioning_mode == "vlm_kv":
            if not config.mask_vlm_padding or config.sampling_method != "ddim":
                raise ValueError("The KV experiment requires masked DDIM conditioning")
            count = sum(i % 2 != 0 or not config.diffusion_model_cfg.get("interleave_attention", True)
                        for i in range(config.diffusion_model_cfg["num_layers"]))
            if len(config.vlm_kv_layers) != count or any(type(i) is not int or i < 0 for i in config.vlm_kv_layers):
                raise ValueError("vlm_kv_layers must select one nonnegative layer per cross-attention block")
            config.diffusion_model_cfg = {**config.diffusion_model_cfg, "conditioning_mode": "vlm_kv"}
        if config.grpo:
            training_reward_protocol(config.grpo_cfg.training_reward_mode,
                raw_safety_weight=config.grpo_cfg.training_raw_safety_weight,
                raw_contact_speed_weight=config.grpo_cfg.training_raw_contact_speed_weight,
                minimum_jerk=config.grpo_cfg.training_minimum_jerk)
        probability = config.velacc_dropout_probability
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("velacc_dropout_probability must be finite and in [0, 1]")
        if type(config.velacc_dropout_seed) is not int or not 0 <= config.velacc_dropout_seed < 2**32:
            raise ValueError("velacc_dropout_seed must be an integer in [0, 2**32)")
        joint = config.joint_motion_dropout_probability
        if (isinstance(joint, bool) or not isinstance(joint, (int, float))
                or not math.isfinite(joint) or not 0 <= joint <= 1):
            raise ValueError("joint_motion_dropout_probability must be finite and in [0, 1]")
        if joint > 0 and probability > 0:
            raise ValueError("Joint and velocity/acceleration-only dropout are mutually exclusive")
        if max(probability, joint) > 0 and config.ego_status_encoder_type != "mlp":
            raise ValueError("Velocity/acceleration dropout requires the MLP ego status encoder")
        self._motion_dropout_forwards = 0
        self._joint_dropout_forwards = 0
        
        self.model = LightningDiT(**config.diffusion_model_cfg)

        self.his_traj_encoder = Mlp(
            in_features=12,
            hidden_features=config.hidden_size,
            out_features=config.input_embedding_dim,
            norm_layer=nn.LayerNorm
        )

        if config.ego_status_encoder_type == 'attention':
            self.ego_status_encoder = StateAttentionEncoder(
                state_dim=8,
                embed_dim=config.input_embedding_dim,
                num_kinematic_states=4 
            )
        else: 
            self.ego_status_encoder = Mlp(
                in_features=8,
                hidden_features=config.hidden_size,
                out_features=config.input_embedding_dim,
                norm_layer=nn.LayerNorm
            )

        # Zero initialization consumes no RNG. Inactive state is a buffer so the
        # p=0 optimizer retains exactly the legacy parameter order and membership.
        null_motion = torch.zeros(config.hidden_size)
        if config.grpo and max(probability, joint) > 0:
            self.null_motion_hidden = nn.Parameter(null_motion)
        else:
            self.register_buffer("null_motion_hidden", null_motion)
        null_history = torch.zeros(1, 1, config.input_embedding_dim)
        if config.grpo and joint > 0:
            self.null_history = nn.Parameter(null_history)
        else:
            self.register_buffer("null_history", null_history)

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.input_embedding_dim,
        )
        if config.vlm_size == "large":
            self.feature_encoder = nn.Linear(3584, config.input_embedding_dim)
        else:
            self.feature_encoder = nn.Linear(1536, config.input_embedding_dim)
            
        self.fusion_projector = nn.Linear(config.input_embedding_dim * 3, config.input_embedding_dim)

        output_dim = 2 * config.action_dim if (
            config.sampling_method == 'flow' and config.flow_cfg.mean_variance_net
        ) else config.action_dim
        
        self.action_decoder = Mlp(
            in_features=self.model.output_dim,
            hidden_features=config.hidden_size,
            out_features=output_dim,
            norm_layer=nn.LayerNorm
        )
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.config.sampling_method == 'flow':
            self._init_flow_sampler(config.flow_cfg)
        elif self.config.sampling_method == 'ddpm':
            self._init_ddpm_sampler(config.ddpm_cfg)
        elif self.config.sampling_method == 'ddim':
            self._init_ddim_sampler(config.ddim_cfg)

        if config.grpo:
            self._init_grpo(config.grpo_cfg)

    def _init_flow_sampler(self, cfg: FlowConfig):
        """Initializes components required for Flow Matching."""
        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)
        self.num_timestep_buckets = cfg.num_timestep_buckets

    def _init_ddpm_sampler(self, cfg: DDPMConfig):
        """Initializes buffers required for DDPM, using original naming."""
        ddpm_betas = self.cosine_beta_schedule(cfg.num_train_timesteps)
        self.register_buffer('ddpm_betas', ddpm_betas)

        ddpm_alphas = 1.0 - ddpm_betas
        self.register_buffer('ddpm_alphas', ddpm_alphas)

        ddpm_alphas_cumprod = torch.cumprod(ddpm_alphas, dim=0)
        self.register_buffer('ddpm_alphas_cumprod', ddpm_alphas_cumprod)

        ddpm_alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), ddpm_alphas_cumprod[:-1]])
        self.register_buffer('ddpm_alphas_cumprod_prev', ddpm_alphas_cumprod_prev)

        self.register_buffer('ddpm_sqrt_alphas_cumprod', torch.sqrt(ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod - 1.0))

        ddpm_var = ddpm_betas * (1.0 - ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod)
        self.register_buffer('ddpm_var', ddpm_var)
        self.register_buffer('ddpm_logvar_clipped', torch.log(ddpm_var.clamp(min=1e-20)))
        
        self.register_buffer('ddpm_mu_coef1', ddpm_betas * torch.sqrt(ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_mu_coef2', (1.0 - ddpm_alphas_cumprod_prev) * torch.sqrt(ddpm_alphas) / (1.0 - ddpm_alphas_cumprod))

    def _init_ddim_sampler(self, cfg: DDIMConfig):
        """Initializes buffers required for DDIM sampling, using original naming."""
        self._init_ddpm_sampler(DDPMConfig(num_train_timesteps=cfg.num_train_timesteps))

        self.eta = EtaFixed(base_eta=1.0).to(self.device)
        for param in self.eta.parameters():
            param.requires_grad = False
        ddim_steps = self.config.num_inference_steps
        self.ddim_steps = ddim_steps
        self.ft_denoising_steps = ddim_steps
        ddim_eta = cfg.ddim_eta 

        self.ddpm_num_train_timesteps = cfg.num_train_timesteps
        step_ratio = self.ddpm_num_train_timesteps // ddim_steps
        ddim_t = torch.arange(0, ddim_steps) * step_ratio
        self.register_buffer('ddim_t_schedule', ddim_t.long()) # Die originalen Zeitpunkte

        ddim_alphas = self.ddpm_alphas_cumprod[self.ddim_t_schedule].clone().to(torch.float32)
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float32),
            self.ddpm_alphas_cumprod[self.ddim_t_schedule[:-1]]
        ])
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5

        ddim_sigmas = ddim_eta * (
            (1 - ddim_alphas_prev) / (1 - ddim_alphas) *
            (1 - ddim_alphas / ddim_alphas_prev)
        )**0.5

        # Flip all for sampling order (T -> 0)
        def flip_buffer(name, tensor):
            self.register_buffer(name, torch.flip(tensor, [0]))

        flip_buffer('ddim_t', self.ddim_t_schedule)
        flip_buffer('ddim_alphas', ddim_alphas)
        flip_buffer('ddim_alphas_sqrt', torch.sqrt(ddim_alphas))
        flip_buffer('ddim_alphas_prev', ddim_alphas_prev)
        flip_buffer('ddim_sqrt_one_minus_alphas', ddim_sqrt_one_minus_alphas)
        flip_buffer('ddim_sigmas', ddim_sigmas)

    def _init_grpo(self, cfg: GRPOConfig):
        """Initializes components and hyperparameters for GRPO training."""
        if cfg.planner_initialization not in ("stage2", "random"):
            raise ValueError("planner_initialization must be stage2 or random.")
        if cfg.planner_initialization == "random" and cfg.reference_policy_checkpoint:
            raise ValueError("Random planner initialization requires an empty reference_policy_checkpoint.")
        self.denoised_clip_value = cfg.denoised_clip_value
        self.eval_randn_clip_value = cfg.eval_randn_clip_value
        self.randn_clip_value = cfg.randn_clip_value
        self.final_action_clip_value = cfg.final_action_clip_value
        self.eps_clip_value = cfg.eps_clip_value
        self.eval_min_sampling_denoising_std = cfg.eval_min_sampling_denoising_std
        self.min_sampling_denoising_std = cfg.min_sampling_denoising_std
        self.min_logprob_denoising_std = cfg.min_logprob_denoising_std
        self.clip_advantage_lower_quantile = cfg.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = cfg.clip_advantage_upper_quantile
        self.gamma_denoising = cfg.gamma_denoising
        
        if cfg.pdm_reward_workers <= 0:
            raise ValueError("pdm_reward_workers must be a positive integer.")
        metric_cache_staging_path = (
            Path(cfg.metric_cache_staging_path)
            if cfg.metric_cache_staging_path
            else None
        )
        self.metric_cache_loader = MetricCacheLoader(
            Path(cfg.metric_cache_path),
            local_staging_path=metric_cache_staging_path,
        )
        self.pdm_reward_workers = cfg.pdm_reward_workers
        self._pdm_scorer_config = cfg.scorer_config
        self._pdm_proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self._reward_executor: Optional[ThreadPoolExecutor] = None
        self._training_reward_protocol = training_reward_protocol(cfg.training_reward_mode,
            raw_safety_weight=cfg.training_raw_safety_weight, raw_contact_speed_weight=cfg.training_raw_contact_speed_weight)
        
        if cfg.planner_initialization == "stage2":
            checkpoint = torch.load(cfg.reference_policy_checkpoint, map_location="cpu", weights_only=False)
            self.load_state_dict(planner_state_from_checkpoint(self, checkpoint), strict=True)

            self.old_policy = copy.deepcopy(self)
            self.old_policy.eval()
            for param in self.old_policy.parameters():
                param.requires_grad = False

        # Runtime cache belongs to the student, after the teacher deepcopy.
        self._reference_progress_cache = ReferenceProgressCache() if cfg.training_reward_mode == "raw" else None
        self._raw_safety_inputs = None
        if cfg.training_reward_mode == "executed_raw_safety":
            raise NotImplementedError("executed_raw_safety (the raw-safety RL reward) is not part of this release; use executed or minimum_jerk")
        if type(cfg.deterministic_final_step) is not bool:
            raise ValueError("deterministic_final_step must be a boolean")
        self._minimum_jerk_config = None
        if cfg.training_reward_mode == "minimum_jerk":
            from navsim.evaluate.minimum_jerk_pdm_score import MinimumJerkRewardConfig
            if not cfg.deterministic_final_step:
                raise ValueError("minimum_jerk requires deterministic_final_step: final-step exploration noise "
                                 "makes every reconstructed candidate infeasible")
            self._minimum_jerk_config = MinimumJerkRewardConfig.from_mapping(cfg.training_minimum_jerk)
            self._training_reward_protocol = training_reward_protocol(
                cfg.training_reward_mode, minimum_jerk=self._minimum_jerk_config)
        weight = cfg.sibling_weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("sibling_weight must be finite and nonnegative")
        self._sibling_settings = None
        if weight > 0:
            if cfg.training_reward_mode != "minimum_jerk":
                raise ValueError("Sibling targets require the minimum_jerk training reward")
            from navsim.evaluate.sibling_targets import SiblingSettings
            self._sibling_settings = SiblingSettings.from_mapping(cfg.training_siblings)
        logging.getLogger(__name__).info("Stage-3 training reward protocol: %s", self._training_reward_protocol)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "old_policy"):
            self.old_policy.eval()
        return self

    def encode_ego_status(self, status, motion_drop_mask=None):
        """Replace only the projected vx/vy/ax/ay contribution; preserve the command."""
        if status.ndim != 2 or status.shape[1] != 8:
            raise ValueError("Expected status [B, 8]: command[4], velocity[2], acceleration[2]")
        encoder = self.ego_status_encoder
        if motion_drop_mask is None:
            return encoder(status)
        if (motion_drop_mask.shape != (status.shape[0], 1) or motion_drop_mask.dtype != torch.bool
                or motion_drop_mask.device != status.device):
            raise ValueError("motion_drop_mask must be boolean [B, 1] on the status device")
        if self.config.ego_status_encoder_type != "mlp":
            raise ValueError("Velocity/acceleration dropout requires the MLP ego status encoder")
        full_hidden = encoder.fc1(status)
        missing_hidden = F.linear(status[:, :4], encoder.fc1.weight[:, :4], encoder.fc1.bias)
        missing_hidden = missing_hidden + self.null_motion_hidden.to(full_hidden)
        # Both branches stay in the graph, including when an entire rank retains
        # motion. The null then receives a zero gradient instead of being unused.
        hidden = torch.where(motion_drop_mask, missing_hidden, full_hidden)
        hidden = encoder.act(hidden)
        hidden = encoder.drop1(hidden)
        hidden = encoder.norm(hidden)
        return encoder.drop2(encoder.fc2(hidden))

    def sample_motion_drop_mask(self, status):
        if not self.training or not self.config.grpo or self.config.velacc_dropout_probability == 0:
            return None
        rank = dist.get_rank() if dist.is_initialized() else 0
        material = f"{MOTION_DROPOUT_PROTOCOL}:{self.config.velacc_dropout_seed}:{rank}:{self._motion_dropout_forwards}"
        seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") >> 1
        generator = torch.Generator(device="cpu").manual_seed(seed)
        mask = torch.rand((status.shape[0], 1), generator=generator, device="cpu", dtype=torch.float32)
        mask = mask < self.config.velacc_dropout_probability
        self._motion_dropout_forwards += 1
        return mask.to(status.device)

    def encode_history(self, history, joint_drop_mask=None):
        encoded = self.his_traj_encoder(history.unsqueeze(1))
        if joint_drop_mask is not None:
            if (joint_drop_mask.shape != (history.shape[0], 1) or joint_drop_mask.dtype != torch.bool
                    or joint_drop_mask.device != history.device):
                raise ValueError("joint_drop_mask must be boolean [B, 1] on the history device")
            encoded = torch.where(joint_drop_mask.unsqueeze(-1), self.null_history.to(encoded), encoded)
        return encoded.repeat(1, self.config.action_horizon, 1)

    def sample_joint_drop_mask(self, status):
        if not self.training or not self.config.grpo or self.config.joint_motion_dropout_probability == 0:
            return None
        rank = dist.get_rank() if dist.is_initialized() else 0
        material = f"{JOINT_DROPOUT_PROTOCOL}:{self.config.velacc_dropout_seed}:{rank}:{self._joint_dropout_forwards}"
        seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") >> 1
        generator = torch.Generator(device="cpu").manual_seed(seed)
        mask = torch.rand((status.shape[0], 1), generator=generator) < self.config.joint_motion_dropout_probability
        self._joint_dropout_forwards += 1
        return mask.to(status.device)

    def joint_dropout_state(self):
        return {"version": 1, "protocol": JOINT_DROPOUT_PROTOCOL,
                "probability": self.config.joint_motion_dropout_probability,
                "seed": self.config.velacc_dropout_seed, "next_forward": self._joint_dropout_forwards,
                "null_shape": list(self.null_history.shape), "torch_version": torch.__version__}

    def restore_joint_dropout_state(self, state):
        current = self.joint_dropout_state()
        if any(state.get(k) != v for k, v in current.items() if k != "next_forward"):
            raise ValueError("Incompatible joint motion dropout state on resume")
        if type(state.get("next_forward")) is not int or state["next_forward"] < 0:
            raise ValueError("Invalid joint motion dropout counter")
        self._joint_dropout_forwards = state["next_forward"]

    def motion_dropout_state(self):
        return {"version": 1, "protocol": MOTION_DROPOUT_PROTOCOL,
                "probability": self.config.velacc_dropout_probability,
                "seed": self.config.velacc_dropout_seed, "next_forward": self._motion_dropout_forwards,
                "null_dimension": self.null_motion_hidden.numel(), "torch_version": torch.__version__}

    def restore_motion_dropout_state(self, state):
        current = self.motion_dropout_state()
        if any(state.get(k) != current[k] for k in (
                "version", "protocol", "probability", "seed", "null_dimension", "torch_version")):
            raise ValueError("Incompatible velocity/acceleration dropout state on resume")
        if type(state.get("next_forward")) is not int or state["next_forward"] < 0:
            raise ValueError("Invalid motion-dropout forward counter")
        self._motion_dropout_forwards = state["next_forward"]

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Calculates a cosine noise schedule as proposed in the iDDPM paper.
        
        This method is static as it does not depend on the instance's state.
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return torch.tensor(betas_clipped, dtype=dtype)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extracts values from tensor `a` at indices `t` and reshapes them
        to be broadcastable with a tensor of shape `x_shape`.
        
        This method is static as it does not depend on the instance's state.
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def make_timesteps(batch_size: int, i: int, device: torch.device) -> torch.Tensor:
        """
        Creates a tensor of a constant value `i` for a given batch size and device.
        
        This method is static as it does not depend on the instance's state.
        """
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        return t


    def set_frozen_modules_to_eval_mode(self):
        """
        Sets frozen parts of the model to evaluation mode during training.
        This is necessary to disable behaviors like dropout in the frozen layers.
        """
        if self.training:

            if not self.config.tune_projector:
                self.his_traj_encoder.eval()
                self.ego_status_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.feature_encoder.eval()
                self.fusion_projector.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            
            if not self.config.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Samples time for training based on the sampling method."""
        if self.config.sampling_method == 'flow':
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            return (self.config.flow_cfg.noise_s - sample) / self.config.flow_cfg.noise_s
        elif self.config.sampling_method in ['ddpm', 'ddim']:
            return torch.randint(0, self.ddpm_num_train_timesteps, (batch_size,), device=device).long()
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")


    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True,
        vlm_conditioning=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + self.position_embedding(pos_ids)

        vl_features_mean = vl_features.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(
            torch.cat((his_traj_features, vl_features_mean, action_features), dim=2)
        )

        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=vl_features,
            conditioning_features=ego_status_features,
            timesteps=t,
            **({"vlm_conditioning": vlm_conditioning} if vlm_conditioning is not None else {}),
        )
        pred_noise = self.action_decoder(model_output)

        if self.config.sampling_method == 'ddpm':
            x_recon = self.extract(self.ddpm_sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                      self.extract(self.ddpm_sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        elif self.config.sampling_method == 'ddim':
            alpha_t = self.extract(self.ddim_alphas, index, x.shape)
            sqrt_one_minus_alpha_t = self.extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
            x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t**0.5)
        else:
             raise ValueError(f"p_mean_variance not supported for method: {self.config.sampling_method}")

        denoised_clip_value = getattr(self, 'denoised_clip_value', 1.0)
        x_recon.clamp_(-denoised_clip_value, denoised_clip_value)

        if self.config.sampling_method == 'ddpm':
            model_mean = self.extract(self.ddpm_mu_coef1, t, x.shape) * x_recon + \
                         self.extract(self.ddpm_mu_coef2, t, x.shape) * x
            model_log_variance = self.extract(self.ddpm_logvar_clipped, t, x.shape)
        elif self.config.sampling_method == 'ddim':
            alpha_prev = self.extract(self.ddim_alphas_prev, index, x.shape)
            
            pred_noise = (x - (alpha_t**0.5) * x_recon) / sqrt_one_minus_alpha_t

            eps_clip_value = getattr(self, 'eps_clip_value', None)
            if eps_clip_value is not None:
                pred_noise.clamp_(-eps_clip_value, eps_clip_value)

            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(x).unsqueeze(1)

            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)

            pred_dir_xt = (1.0 - alpha_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
            model_mean = (alpha_prev**0.5) * x_recon + pred_dir_xt
            model_log_variance = torch.log(sigma**2 + 1e-20)

        return model_mean, model_log_variance, x_recon

    def forward(self, vl_features: torch.Tensor, action_input: BatchFeature, *, vlm_conditioning=None) -> BatchFeature:
        """
        Computes the training loss for a given batch.

        Args:
            vl_features (torch.Tensor): The vision-language features from the backbone.
            action_input (BatchFeature): A batch containing ground truth actions and other conditioning.

        Returns:
            BatchFeature: A batch containing the computed loss.
        """
        
        vl_embeds = self.feature_encoder(vl_features)
        his_traj_features = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(action_input.status_feature)
        
        gt_actions = self.norm_odo(action_input.action)

        if self.config.sampling_method == 'flow':
            noise = torch.randn_like(gt_actions)
            t_cont = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            t_cont_reshaped = t_cont[:, None, None]
            
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target = gt_actions - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()

            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
            )

            model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete,
                                     **({"vlm_conditioning": vlm_conditioning} if vlm_conditioning is not None else {}))
            pred_velocity = self.action_decoder(model_output)
            loss = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
        else: 
            noise = torch.randn_like(gt_actions)
            t_discrete = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            
            noisy_actions = (
                self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape) * gt_actions +
                self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape) * noise
            )
            
            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
            )
            
            model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete,
                                     **({"vlm_conditioning": vlm_conditioning} if vlm_conditioning is not None else {}))
            pred_noise = self.action_decoder(model_output)
            loss = F.mse_loss(pred_noise, noise, reduction='mean')

        return BatchFeature(data={"loss": loss})

    def denoising_losses(self, vl_features, his_traj, status_feature, actions, motion_drop_mask=None,
                         joint_drop_mask=None) -> torch.Tensor:
        """Per-sample stage-2 objective (random timestep, epsilon MSE) toward ``actions``.

        Same computation as ``forward`` for DDPM/DDIM, with the stage-3 student
        conditioning (motion/joint dropout masks), used by the sibling loss.
        """
        if self.config.sampling_method not in ("ddpm", "ddim"):
            raise ValueError("Sibling denoising targets require DDPM/DDIM sampling")
        vl_embeds = self.feature_encoder(vl_features)
        his_traj_features = self.encode_history(his_traj, joint_drop_mask)
        ego_status_features = self.encode_ego_status(status_feature, motion_drop_mask)
        target_actions = self.norm_odo(actions)
        noise = torch.randn_like(target_actions)
        t_discrete = self.sample_time(target_actions.shape[0], device=target_actions.device, dtype=target_actions.dtype)
        noisy_actions = (
            self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, target_actions.shape) * target_actions +
            self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, target_actions.shape) * noise
        )
        action_features = self.action_encoder(noisy_actions, t_discrete)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=target_actions.device)
            action_features += self.position_embedding(pos_ids)
        vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2))
        model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete)
        pred_noise = self.action_decoder(model_output)
        return F.mse_loss(pred_noise, noise, reduction="none").mean(dim=(1, 2))

    def get_action(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        noise_seeds: Optional[List[int]] = None,
        vlm_conditioning=None,
    ) -> BatchFeature:
        """
        Generates action trajectories via the configured sampling method.

        This method strictly preserves the original logic for each sampler,
        including specific clipping and noise handling for DDPM and DDIM.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            action_input (BatchFeature): Input containing conditioning features like
                historical trajectory and ego status.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start
                the denoising from. If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic (eta=0).

        Returns:
            BatchFeature: A batch containing the final predicted trajectory.
        """
        vl_embeds = self.feature_encoder(vl_features)
        
        history_embeds = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)

        ego_embeds = self.ego_status_encoder(
            action_input.status_feature
        )

        B, D = vl_embeds.shape[0], self.config.action_dim
        device, dtype = vl_embeds.device, vl_embeds.dtype
        
        # Independent CPU streams couple each source across edits, batches and ranks.
        # Keep legacy global-RNG sampling byte-for-byte when no seeds are supplied.
        generators = None
        if noise_seeds is not None:
            if len(noise_seeds) != B or any(type(s) is not int or not 0 <= s < 2**63 for s in noise_seeds):
                raise ValueError("noise_seeds must contain one integer seed per trajectory")
            generators = [torch.Generator(device="cpu").manual_seed(s) for s in noise_seeds]

        def draw_noise():
            if generators is None:
                return torch.randn((B, self.config.action_horizon, D), device=device, dtype=dtype)
            return torch.stack([torch.randn((self.config.action_horizon, D), generator=g,
                                            dtype=torch.float32) for g in generators]).to(device=device, dtype=dtype)

        current_actions = init_actions if init_actions is not None else draw_noise()

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t = torch.full((B,), idx, device=device, dtype=torch.long)

                action_features = self.action_encoder(current_actions, t)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, vl_embeds_mean, action_features), dim=2)
                )
                
                model_output = self.model(fused_input, vl_embeds, ego_embeds, t,
                                         **({"vlm_conditioning": vlm_conditioning} if vlm_conditioning is not None else {}))
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow

        elif self.config.sampling_method == 'ddpm':
            step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
            timesteps_to_iterate = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            
            for i, t_int in enumerate(timesteps_to_iterate):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_embeds, history_embeds, ego_embeds, deterministic,
                    **({"vlm_conditioning": vlm_conditioning} if vlm_conditioning is not None else {}),
                )

                noise_sample = torch.randn_like(current_actions) if generators is None else draw_noise()
                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                if t_int == 0:
                    std.zero_()
                else:
                    std = torch.clamp(std, min=1e-3)

                if hasattr(self, 'eval_randn_clip_value') and self.eval_randn_clip_value is not None:
                    noise_sample.clamp_(-self.eval_randn_clip_value, self.eval_randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None and i == len(timesteps_to_iterate) - 1:
                    current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)

        elif self.config.sampling_method == 'ddim':
            eval_min_sampling_denoising_std = getattr(self, 'eval_min_sampling_denoising_std', 0.0001)
            eval_randn_clip_value = getattr(self, 'eval_randn_clip_value', 1.0)
            for i in range(self.ddim_steps):
                t_batch = self.make_timesteps(B, self.ddim_t[i], device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_embeds, history_embeds, ego_embeds, deterministic,
                    **({"vlm_conditioning": vlm_conditioning} if vlm_conditioning is not None else {}),
                )

                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                noise_sample = torch.randn_like(current_actions) if generators is None else draw_noise()
                
                if deterministic:
                    std.zero_()
                else:
                    std = std.clamp(min=eval_min_sampling_denoising_std)
                
                noise_sample.clamp_(-eval_randn_clip_value, eval_randn_clip_value)
                
                current_actions = mean + std * noise_sample
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
        if final_action_clip_value is not None:
            current_actions.clamp_(-final_action_clip_value, final_action_clip_value)

        final_actions = self.denorm_odo(current_actions)

        return BatchFeature(data={"pred_traj": final_actions})

    def sample_chain(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        motion_drop_mask: Optional[torch.Tensor] = None,
        joint_drop_mask: Optional[torch.Tensor] = None,
        deterministic_final_step: bool = False,
    ):
        """
        Generates the full denoising chain and the final trajectory.
        This method reuses the logic from get_action but stores intermediate steps.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            his_traj_features (torch.Tensor): Raw historical poses, flattened to 12 values.
            ego_status_features (torch.Tensor): Raw command/velocity/acceleration, eight values.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start from.
                If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic.
            motion_drop_mask (Optional[torch.Tensor]): Shared sample-level mask, already
                expanded over GRPO candidates. The teacher leaves this unset.
            deterministic_final_step (bool): Sample earlier steps as usual, but take the
                mean at the last step (no per-waypoint exploration noise).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The full denoising chain as a tensor of shape (B, K+1, H, D).
                - The final, denormalized trajectory of shape (B, H, D).
        """
        B, D = vl_features.shape[0], self.config.action_dim
        device, dtype = vl_features.device, vl_features.dtype
        
        vl_features = self.feature_encoder(vl_features)
        
        his_traj_features = self.encode_history(his_traj_features, joint_drop_mask)

        ego_status_features = self.encode_ego_status(ego_status_features, motion_drop_mask)

        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )
        denoising_chain = [current_actions.clone()]

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t_batch = torch.full((B,), idx, device=device, dtype=torch.long)
                
                action_features = self.action_encoder(current_actions, t_batch)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                vl_features_mean = vl_features.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, vl_features_mean, action_features), dim=2)
                )
                
                model_output = self.model(fused_input, vl_features, ego_status_features, t_batch)
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow
                denoising_chain.append(current_actions.clone())

        elif self.config.sampling_method in ['ddpm', 'ddim']:
            if self.config.sampling_method == 'ddpm':
                step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
                timesteps = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            else: 
                timesteps = self.ddim_t
            
            for i, t_int in enumerate(timesteps):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device) if self.config.sampling_method == 'ddim' else t_batch

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_features, his_traj_features, ego_status_features, deterministic
                )

                std = torch.exp(0.5 * logvar).to(dtype)
                noise_sample = torch.randn_like(current_actions)

                if self.config.sampling_method == 'ddim':
                    if deterministic:
                        std.zero_()
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                else: # ddpm
                    if deterministic and t_int == 0:
                        std = torch.zeros_like(std)
                    elif deterministic:
                        std = std.clamp(min=1e-3)
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                if deterministic_final_step and i == len(timesteps) - 1:
                    std = torch.zeros_like(std)
                
                if hasattr(self, 'randn_clip_value') and self.randn_clip_value is not None:
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if i == len(timesteps) - 1 and hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None:
                    current_actions = current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)
                
                denoising_chain.append(current_actions.clone())
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_actions = self.denorm_odo(current_actions)
        chain_tensor = torch.stack(denoising_chain, dim=1)
        
        return chain_tensor.detach(), final_actions.detach()

    def get_logprobs(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        deterministic: bool = False,
        motion_drop_mask: Optional[torch.Tensor] = None,
        joint_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates the log probability of a full denoising chain."""
        B, K1, H, D = chains.shape
        num_denoising_steps = K1 - 1
        
        vl_features = self.feature_encoder(vl_features)

        his_traj_features = self.encode_history(his_traj_features, joint_drop_mask)

        ego_status_features = self.encode_ego_status(ego_status_features, motion_drop_mask)

        conditioning_embeds = {
            'vl_features': vl_features,
            'his_traj_features': his_traj_features,
            'ego_status_features': ego_status_features
        }
        
        batched_conditioning = {}
        for key, value in conditioning_embeds.items():
            batched_conditioning[key] = value.unsqueeze(1).repeat(
                1, num_denoising_steps, *(1,) * (value.ndim - 1)
            ).flatten(0, 1)

        if self.config.sampling_method == 'ddim':
            t_single = self.ddim_t[-num_denoising_steps:]
            indices_single = torch.arange(
                start=self.ddim_steps - num_denoising_steps,
                end=self.ddim_steps,
                device=chains.device
            )
            indices_batch = indices_single.repeat(B)
        else:
            t_single = torch.arange(start=K1 - 2, end=-1, step=-1, device=chains.device)
            indices_batch = None 
            
        t_batch = t_single.repeat(B)

        x_t = chains[:, :-1].reshape(-1, H, D)
        x_t_minus_1 = chains[:, 1:].reshape(-1, H, D)

        mean, logvar, _ = self.p_mean_variance(
            x_t, t_batch, indices_batch,
            batched_conditioning['vl_features'],
            batched_conditioning['his_traj_features'],
            batched_conditioning['ego_status_features'],
            deterministic=deterministic
        )

        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x_t_minus_1)
        
        return log_prob

    def forward_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: int = 8,
        deterministic=False,
        bc_coeff: float = 0.1,
        use_bc_loss: bool = True,
        joint_drop_mask: Optional[torch.Tensor] = None,
        teacher_vl_features: Optional[torch.Tensor] = None,
        reward_targets: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """Computes the Diffusion-GRPO loss.

        ``reward_targets`` is the clean GT in its original precision, used only by
        the minimum-jerk reward.
        """
        if self.config.grpo_cfg.planner_initialization == "random" and (bc_coeff != 0 or use_bc_loss):
            raise ValueError("Random planner initialization requires bc_coeff=0 and use_bc_loss=False.")
        self.set_frozen_modules_to_eval_mode()
        B = vl_features.shape[0]
        G = sample_time 

        joint_enabled = self.config.joint_motion_dropout_probability > 0
        if joint_enabled and (joint_drop_mask is None or (use_bc_loss and teacher_vl_features is None)):
            raise ValueError("Joint dropout requires the agent's shared mask and full-input teacher features")
        if not joint_enabled and joint_drop_mask is not None:
            raise ValueError("Joint mask supplied when joint dropout is disabled")
        motion_drop_mask = joint_drop_mask if joint_enabled else self.sample_motion_drop_mask(action_input.status_feature)
        group_mask = motion_drop_mask.repeat_interleave(G, 0) if motion_drop_mask is not None else None

        joint_kwargs = {"joint_drop_mask": group_mask} if joint_enabled else {}
        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)

        # Rollouts are detached; get_logprobs below rebuilds student conditioning
        # with gradients for both the policy and BC losses, including the null.
        minimum_jerk = self.config.grpo_cfg.training_reward_mode == "minimum_jerk"
        if minimum_jerk != (reward_targets is not None):
            raise ValueError("Clean GT reward targets are required by, and only by, the minimum_jerk reward")
        reward_kwargs = {}
        if minimum_jerk:
            targets = reward_targets.detach().cpu().numpy().astype(np.float64)
            if targets.shape != (B, self.config.action_horizon, 3):
                raise ValueError("Reward targets must be (batch, 8, 3) clean GT poses")
            reward_kwargs["reward_targets"] = {}
            for token, target in zip(tokens_list, targets):
                if not np.array_equal(reward_kwargs["reward_targets"].setdefault(token, target), target):
                    raise ValueError(f"Different clean GT targets for repeated token {token}")
        with torch.no_grad():
            final_step = {"deterministic_final_step": True} if self.config.grpo_cfg.deterministic_final_step else {}
            chains, trajs = self.sample_chain(
                vl_features_rep, his_traj_rep, status_feature_rep, deterministic=False, motion_drop_mask=group_mask,
                **joint_kwargs, **final_step
            )
            # Raw positions are already in metres. Compare matching future
            # timestamps over unique candidate pairs, excluding headings.
            xy = trajs.float().reshape(B, G, *trajs.shape[1:])[..., :2]
            pair_i, pair_j = torch.triu_indices(G, G, offset=1, device=trajs.device)
            group_pairwise_ade_m = torch.linalg.vector_norm(
                xy[:, pair_i] - xy[:, pair_j], dim=-1
            ).mean()
            siblings = getattr(self, "_sibling_settings", None)
            if siblings is not None:
                # One deterministic student sample per scene, with the rollouts' conditioning masks.
                _, bases = self.sample_chain(
                    vl_features, action_input.his_traj, action_input.status_feature, deterministic=True,
                    motion_drop_mask=motion_drop_mask, **({"joint_drop_mask": joint_drop_mask} if joint_enabled else {}))
                reward_kwargs["sibling_bases"] = list(zip(tokens_list, bases.float().cpu().numpy().astype(np.float64)))

        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        rewards, reward_components = self.reward_fn(trajs, tokens_rep, return_components=True, **reward_kwargs)
        raw_diagnostics = {name: reward_components.pop(name) for name in RAW_SAFETY_DIAGNOSTICS if name in reward_components}
        candidate_safety = reward_components.pop("candidate_safety", None)
        candidate_contact_speed = reward_components.pop("candidate_contact_speed", None)
        minimum_jerk_diagnostics = reward_components.pop("minimum_jerk", None)
        sibling_outcomes = reward_components.pop("sibling_outcomes", None)

        rewards_matrix = rewards.view(B, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        reward_std = rewards_matrix.std(dim=1, keepdim=True)
        std_r = reward_std + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()
        
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (self.gamma_denoising ** (num_denoising_steps - denoising_indices - 1))
    
        
        adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)  
        discount = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)  
        adv_weighted_flat = (adv_steps * discount).reshape(-1)             

        log_probs = self.get_logprobs(vl_features_rep, his_traj_rep, status_feature_rep, chains,
                                     deterministic=False, motion_drop_mask=group_mask, **joint_kwargs)
        log_probs = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        
        policy_loss = -torch.mean(log_probs * adv_weighted_flat)
        total_loss = policy_loss

        bc_loss = 0.0
        if use_bc_loss:
            with torch.no_grad():
                teacher_chains, _ = self.old_policy.sample_chain(
                    teacher_vl_features if joint_enabled else vl_features,
                    action_input.his_traj, action_input.status_feature, deterministic=False
                )
            bc_logp = self.get_logprobs(vl_features, action_input.his_traj, action_input.status_feature, teacher_chains,
                                       deterministic=False, motion_drop_mask=motion_drop_mask,
                                       **({"joint_drop_mask": joint_drop_mask} if joint_enabled else {}))
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + bc_coeff * bc_loss

        sibling_outputs = {}
        if siblings is not None:
            from navsim.evaluate.sibling_targets import SIBLING_FIELDS, diagnostic_row
            chosen = [i for i, outcome in enumerate(sibling_outcomes) if outcome.target is not None]
            rows = torch.as_tensor(np.stack([diagnostic_row(outcome) for outcome in sibling_outcomes]),
                                   dtype=torch.float64, device=vl_features.device)
            sibling_loss = vl_features.new_zeros(())
            if chosen:
                index = torch.tensor(chosen, device=vl_features.device)
                masks = {} if motion_drop_mask is None else {"motion_drop_mask": motion_drop_mask[index]}
                if joint_enabled:
                    masks["joint_drop_mask"] = joint_drop_mask[index]
                targets = torch.as_tensor(np.stack([sibling_outcomes[i].target for i in chosen]),
                                          dtype=action_input.action.dtype, device=vl_features.device)
                losses = self.denoising_losses(vl_features[index], action_input.his_traj[index],
                                               action_input.status_feature[index], targets, **masks)
                # Mean over the micro-batch: scenes without a target contribute zero.
                sibling_loss = losses.sum() / B
                total_loss = total_loss + self.config.grpo_cfg.sibling_weight * sibling_loss
                rows[index, SIBLING_FIELDS.index("loss_sum")] = losses.detach().to(torch.float64)
            sibling_outputs = {"sibling_loss": sibling_loss.detach(), "sibling_diagnostics": rows}

        dropped = motion_drop_mask.sum() if motion_drop_mask is not None else vl_features.new_zeros((), dtype=torch.long)
        group_safety = {}
        if candidate_safety is not None:
            # Columns: collision-free and NC x DAC = 1 under this reward's own definitions.
            safety = candidate_safety.view(B, G, 2)
            group_safety["safety_group_diagnostics"] = {
                "any_collision_free": safety[..., 0].any(dim=1),
                "all_collision_free": safety[..., 0].all(dim=1),
                "any_safe": safety[..., 1].any(dim=1),
            }
        if candidate_contact_speed is not None:
            # First counted contact speed per candidate (NaN without one), summed per scene.
            speeds = candidate_contact_speed.view(B, G)
            contacts = torch.isfinite(speeds)
            group_safety["contact_group_diagnostics"] = {
                "contact_speed_sum": torch.where(contacts, speeds, torch.zeros_like(speeds)).sum(dim=1),
                "contact_count": contacts.sum(dim=1).to(speeds.dtype),
            }
        if minimum_jerk_diagnostics is not None:
            group_safety["minimum_jerk_reward_diagnostics"] = minimum_jerk_diagnostics
        return BatchFeature(data={**group_safety, **sibling_outputs, "loss": total_loss, "reward": rewards.mean(), "policy_loss": policy_loss, "bc_loss": bc_loss,
                                  "reward_components": reward_components,
                                  "raw_reward_diagnostics": raw_diagnostics,
                                  "reward_candidate_count": len(tokens_rep),
                                  "reward_group_diagnostics": {
                                      "equal": (rewards_matrix == rewards_matrix[:, :1]).all(dim=1).detach(),
                                      "zero": (rewards_matrix == 0).all(dim=1).detach(),
                                      "std": reward_std.detach().float().squeeze(1),
                                  },
                                  "reward_group_std": reward_std.detach().float().mean(),
                                  "group_pairwise_ade_m": group_pairwise_ade_m,
                                  "equal_reward_group_fraction": (rewards_matrix == rewards_matrix[:, :1]).all(dim=1).float().mean(),
                                  "velacc_dropped_samples": dropped.new_zeros(()) if joint_enabled else dropped,
                                  "velacc_total_samples": vl_features.new_tensor(B, dtype=torch.long),
                                  "velacc_dropout_probability": vl_features.new_tensor(self.config.velacc_dropout_probability),
                                  "velacc_null_norm": self.null_motion_hidden.detach().float().norm(),
                                  "joint_dropped_samples": dropped if joint_enabled else dropped.new_zeros(()),
                                  "joint_total_samples": vl_features.new_tensor(B, dtype=torch.long),
                                  "joint_dropout_probability": vl_features.new_tensor(self.config.joint_motion_dropout_probability),
                                  "joint_null_history_norm": self.null_history.detach().float().norm()})

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)
    
    def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list,
        *,
        return_components: bool = False,
        reward_targets: Optional[Dict[str, np.ndarray]] = None,
        sibling_bases: Optional[List[Tuple[str, np.ndarray]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Score once per scene group; optionally return candidate-mean subscores.

        The reward tensor retains its original dtype/order. Logging means use
        full precision aggregation before conversion to float32, independently
        of policy mixed precision, and never include the PDM reference itself.
        The minimum-jerk reward takes the clean GT per token and returns its
        diagnostic sums instead of PDM component means. Every mode also returns
        per-candidate safety flags, and contact speeds where the reward has them.
        ``sibling_bases`` (one (token, poses) per scene) adds each scene's
        sibling outcome, scored in the same per-token task.
        """
        pred_np = pred_traj.detach().cpu().numpy()
        token_groups: Dict[str, List[Tuple[int, Trajectory]]] = {}
        for i, token in enumerate(tokens_list):
            token_groups.setdefault(token, []).append((i, Trajectory(pred_np[i])))

        tasks = [
            (token, [trajectory for _, trajectory in indexed_trajectories])
            for token, indexed_trajectories in token_groups.items()
        ]
        minimum_jerk = self.config.grpo_cfg.training_reward_mode == "minimum_jerk"
        if sibling_bases is not None and (not minimum_jerk or not return_components
                                          or getattr(self, "_sibling_settings", None) is None):
            raise ValueError("Sibling bases require enabled siblings, the minimum_jerk reward and components")
        if minimum_jerk:
            if reward_targets is None:
                raise ValueError("minimum_jerk scoring requires clean GT reward targets")
            tasks = [(token, trajectories, reward_targets[token]) for token, trajectories in tasks]
        if sibling_bases is not None:
            scene_bases: Dict[str, List[Tuple[int, np.ndarray]]] = {}
            for scene, (token, base) in enumerate(sibling_bases):
                scene_bases.setdefault(token, []).append((scene, base))
            if set(scene_bases) != set(token_groups):
                raise ValueError("Sibling bases must cover exactly the scored scenes")
            tasks = [(*task, [base for _, base in scene_bases[task[0]]]) for task in tasks]
        if self.pdm_reward_workers > 1 and len(tasks) > 1:
            if self._reward_executor is None:
                self._reward_executor = ThreadPoolExecutor(
                    max_workers=self.pdm_reward_workers,
                    thread_name_prefix="pdm_reward",
                )
            grouped_results = list(self._reward_executor.map(self._score_token_group, tasks))
        else:
            grouped_results = [self._score_token_group(task) for task in tasks]

        rewards = np.empty(len(tokens_list), dtype=np.float64)
        sibling_outcomes = None
        if sibling_bases is not None:
            grouped_results, grouped_siblings = [list(part) for part in zip(*grouped_results)]
            sibling_outcomes = [None] * len(sibling_bases)
            for token, outcomes in zip(token_groups, grouped_siblings):
                for (scene, _), outcome in zip(scene_bases[token], outcomes, strict=True):
                    sibling_outcomes[scene] = outcome
        if minimum_jerk:
            from navsim.evaluate.minimum_jerk_pdm_score import diagnostic_sums
            ordered = [None] * len(tokens_list)
            for indexed_trajectories, token_results in zip(token_groups.values(), grouped_results):
                if len(indexed_trajectories) != len(token_results):
                    raise ValueError("Scorer returned a different number of candidates")
                for (index, _), result in zip(indexed_trajectories, token_results):
                    rewards[index], ordered[index] = result.score, result
            if not np.isfinite(rewards).all():
                raise ValueError("Nonfinite minimum-jerk reward")
            reward_tensor = torch.as_tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()
            if not return_components:
                return reward_tensor
            collision_free = np.array([r.no_at_fault_collisions == 1 for r in ordered])
            safe = collision_free & np.array([r.drivable_area_compliance == 1 for r in ordered])
            speeds = np.array([np.nan if r.contact_speed is None else r.contact_speed for r in ordered])
            values = {
                "minimum_jerk": torch.as_tensor(diagnostic_sums(ordered), device=pred_traj.device, dtype=torch.float64),
                "candidate_safety": torch.as_tensor(np.column_stack((collision_free, safe)), device=pred_traj.device),
                "candidate_contact_speed": torch.as_tensor(speeds, device=pred_traj.device, dtype=torch.float64)}
            if sibling_outcomes is not None:
                values["sibling_outcomes"] = sibling_outcomes
            return reward_tensor, values
        components = np.empty((len(tokens_list), len(PDM_REWARD_COMPONENTS)), dtype=np.float64) if return_components else None
        hybrid = self.config.grpo_cfg.training_reward_mode == "executed_raw_safety"
        diagnostics = np.empty((len(tokens_list), len(RAW_SAFETY_DIAGNOSTICS)), dtype=np.float64) if hybrid and return_components else None
        for indexed_trajectories, token_results in zip(token_groups.values(), grouped_results):
            if len(indexed_trajectories) != len(token_results):
                raise ValueError("Scorer returned a different number of candidates")
            for (index, _), result in zip(indexed_trajectories, token_results):
                rewards[index] = result.score
                if components is not None:
                    pdm = result.executed if hybrid else result
                    components[index] = [getattr(pdm, field) for field in PDM_REWARD_COMPONENTS.values()]
                if diagnostics is not None:
                    diagnostics[index] = [result.executed.score, result.nc_all, result.dac, result.safety,
                        result.contact_penalty, result.score, result.contact_speed or 0.,
                        result.contact_speed is not None, result.initial_contact]

        if hybrid and (not np.isfinite(rewards).all() or (components is not None and not np.isfinite(components).all())):
            raise ValueError("Nonfinite raw safety reward/components")

        reward_tensor = torch.as_tensor(
            rewards,
            device=pred_traj.device,
            dtype=pred_traj.dtype,
        ).detach()
        if components is None:
            return reward_tensor
        means = torch.as_tensor(components.mean(axis=0), device=pred_traj.device, dtype=torch.float32)
        values = dict(zip(PDM_REWARD_COMPONENTS, means.unbind()))
        if diagnostics is not None:
            aggregate = diagnostics.mean(axis=0)
            aggregate[-3:] = diagnostics[:, -3:].sum(axis=0)
            values.update(zip(RAW_SAFETY_DIAGNOSTICS, torch.as_tensor(
                aggregate, device=pred_traj.device, dtype=torch.float64).unbind()))
            # Per-candidate flags, in input order, for group-level safety logs:
            # raw NC-all and raw NC-all x DAC, and the raw first-contact speed.
            flags = diagnostics[:, [1, 3]] > 0
            values["candidate_contact_speed"] = torch.as_tensor(
                np.where(diagnostics[:, 7] > 0, diagnostics[:, 6], np.nan), device=pred_traj.device, dtype=torch.float64)
        else:
            # At-fault collision-free, and also drivable-area compliant, for the scored trajectory.
            collision_free = components[:, list(PDM_REWARD_COMPONENTS).index("nc")] == 1
            flags = np.column_stack((collision_free, collision_free & (components[:, list(PDM_REWARD_COMPONENTS).index("dac")] == 1)))
        values["candidate_safety"] = torch.as_tensor(flags, device=pred_traj.device)
        return reward_tensor, values

    def _score_token_group(
        self,
        task: Tuple[str, List[Trajectory]],
    ) -> List[PDMResults]:
        """Score one scene's sampled trajectories with thread-local PDM state."""
        token, trajectories = task[:2]
        metric_cache = self.metric_cache_loader.get_from_token(token)
        if self.config.grpo_cfg.training_reward_mode == "minimum_jerk":
            from navsim.evaluate.minimum_jerk_pdm_score import minimum_jerk_scores
            results = minimum_jerk_scores(metric_cache, trajectories, task[2], self._pdm_scorer_config,
                                          self._minimum_jerk_config)
            if len(task) < 4:
                return results
            from navsim.evaluate.sibling_targets import sibling_outcome
            # Sibling families of this token's scenes, sharing the loaded metric cache.
            return results, [sibling_outcome(metric_cache, base, task[2], self._pdm_scorer_config,
                                             self._minimum_jerk_config, self._sibling_settings) for base in task[3]]
        scorer = PDMScorer(self._pdm_proposal_sampling, self._pdm_scorer_config)
        if self.config.grpo_cfg.training_reward_mode == "raw":
            source = Path(self.metric_cache_loader.metric_cache_paths[token]).expanduser().resolve()
            stat = source.stat()
            key = (token, str(source), stat.st_size, stat.st_mtime_ns,
                   tuple(sorted(self._training_reward_protocol.items())))
            reference_progress = self._reference_progress_cache.get_or_compute(
                key, lambda: executed_reference_progress(
                    metric_cache, self._pdm_proposal_sampling, PDMSimulator(self._pdm_proposal_sampling), scorer,
                ),
            )
            return raw_pdm_scores(
                metric_cache, trajectories, self._pdm_proposal_sampling, scorer, reference_progress,
            )
        simulator = PDMSimulator(self._pdm_proposal_sampling)
        results = pdm_scores(
            metric_cache=metric_cache,
            model_trajectories=trajectories,
            future_sampling=self._pdm_proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        return results

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

#https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/eta.py#L12
class EtaFixed(nn.Module):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta

        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, x):
        """Match input batch size, but do not depend on input"""
        B = len(x)
        device = x.device
        eta_normalized = torch.tanh(self.eta_logit)

        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)
