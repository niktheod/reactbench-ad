from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math
from contextlib import nullcontext

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.evaluate.raw_pdm_score import training_reward_protocol
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .recogdrive_features import ReCogDriveFeatureBuilder ,TrajectoryTargetBuilder
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_conditioning import DEFAULT_KV_LAYERS, VLMConditioning, vlm_file_identity
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
    planner_state_from_checkpoint,
)


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        lr_scheduler_epochs: int = 10,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        pdm_reward_workers: int = 1,
        vlm_size: Optional[str] = 'small', 
        train_backbone: bool = False,
        velacc_dropout_probability: float = 0.0,
        velacc_dropout_seed: int = 0,
        joint_motion_dropout_probability: float = 0.0,
        bc_coeff: float = 0.1,
        training_comfort_weight: float = 2.0,
        planner_initialization: str = "stage2",
        training_ep_weight: float = 10.0,
        training_reward_mode: str = "executed",
        training_raw_safety_weight: float = 0.2,
        training_raw_contact_speed_weight: float = 0.05,
        training_raw_safety_inputs: Optional[str] = None,
        training_minimum_jerk: Optional[Dict[str, float]] = None,
        grpo_deterministic_final_step: bool = False,
        sibling_weight: float = 0.0,
        training_siblings: Optional[Dict[str, Any]] = None,
        conditioning_mode: str = "hidden_states",
        vlm_kv_layers: Optional[List[int]] = None,
        mask_vlm_padding: bool = False,
        cross_attention_heads: int = 8,
        cross_attention_kv_heads: int = 2,
        cross_attention_head_dim: int = 128,
        cross_attention_q_norm: bool = False,
        vlm_rope_theta: float = 1_000_000.0,
        inference_precision: str = "32-true",
        per_sample_loss: bool = False,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        if conditioning_mode not in ("hidden_states", "vlm_kv"):
            raise ValueError("conditioning_mode must be hidden_states or vlm_kv")
        if conditioning_mode == "vlm_kv" and (cache_hidden_state or cache_mode or not mask_vlm_padding):
            raise ValueError("VLM KV conditioning requires online inputs and mask_vlm_padding=True")
        if (conditioning_mode == "vlm_kv" or mask_vlm_padding) and (
                grpo or train_backbone or vlm_type != "internvl" or dit_type != "small" or vlm_size != "small"):
            raise ValueError("Stage-2 conditioning requires a frozen InternVL 2B VLM and the small DiT, without GRPO")
        self.conditioning_mode = conditioning_mode
        self.vlm_kv_layers = list(DEFAULT_KV_LAYERS if vlm_kv_layers is None else vlm_kv_layers)
        self.mask_vlm_padding = mask_vlm_padding
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_kv_heads = cross_attention_kv_heads
        self.cross_attention_head_dim = cross_attention_head_dim
        self.cross_attention_q_norm = cross_attention_q_norm
        self.vlm_rope_theta = vlm_rope_theta
        if inference_precision not in ("32-true", "16-mixed", "bf16-mixed"):
            raise ValueError("Unknown inference_precision")
        self.inference_precision = inference_precision
        if conditioning_mode == "vlm_kv" and (len(self.vlm_kv_layers) != 8 or any(
                type(i) is not int or not 0 <= i < 28 for i in self.vlm_kv_layers)):
            raise ValueError("Select eight VLM layers, each in [0, 27]")
        # Kinematic SFT (ReactTrain): the same stage-2 objective, returned per sample.
        if type(per_sample_loss) is not bool:
            raise ValueError("per_sample_loss must be a boolean")
        if per_sample_loss and (grpo or conditioning_mode != "hidden_states" or mask_vlm_padding
                                or sampling_method not in ("ddim", "ddpm")):
            raise ValueError("per_sample_loss requires stage-2 training (no GRPO) with hidden-state DDIM/DDPM conditioning")
        self.per_sample_loss = per_sample_loss
        self._lr = lr
        if lr_scheduler_epochs <= 0:
            raise ValueError("lr_scheduler_epochs must be a positive integer.")
        self.lr_scheduler_epochs = lr_scheduler_epochs
        self.grpo = grpo
        training_reward_protocol(training_reward_mode, minimum_jerk=training_minimum_jerk)
        self.training_reward_mode = training_reward_mode
        self.training_minimum_jerk = None if training_minimum_jerk is None else dict(training_minimum_jerk)
        if type(grpo_deterministic_final_step) is not bool:
            raise ValueError("grpo_deterministic_final_step must be a boolean")
        self.grpo_deterministic_final_step = grpo_deterministic_final_step
        if grpo and training_reward_mode == "minimum_jerk" and not grpo_deterministic_final_step:
            raise ValueError("minimum_jerk requires grpo_deterministic_final_step=true")
        if (isinstance(sibling_weight, bool) or not isinstance(sibling_weight, (int, float))
                or not math.isfinite(sibling_weight) or sibling_weight < 0):
            raise ValueError("sibling_weight must be finite and nonnegative.")
        if grpo and sibling_weight > 0 and training_reward_mode != "minimum_jerk":
            raise ValueError("sibling_weight > 0 requires training_reward_mode=minimum_jerk")
        from navsim.evaluate.sibling_targets import SiblingSettings
        SiblingSettings.from_mapping(training_siblings)  # reject invalid settings before model setup
        self.sibling_weight = float(sibling_weight)
        self.training_siblings = None if training_siblings is None else dict(training_siblings)
        from navsim.evaluate.raw_pdm_score import validate_raw_safety_weights
        validate_raw_safety_weights(training_raw_safety_weight, training_raw_contact_speed_weight)
        self.training_raw_safety_weight = float(training_raw_safety_weight)
        self.training_raw_contact_speed_weight = float(training_raw_contact_speed_weight)
        self.training_raw_safety_inputs = training_raw_safety_inputs
        if grpo and training_reward_mode == "executed_raw_safety" and not training_raw_safety_inputs:
            raise ValueError("executed_raw_safety requires training_raw_safety_inputs (prepared manifest.json)")
        if (isinstance(bc_coeff, bool) or not isinstance(bc_coeff, (int, float))
                or not math.isfinite(bc_coeff) or bc_coeff < 0):
            raise ValueError("bc_coeff must be finite and nonnegative.")
        self.bc_coeff = bc_coeff
        if planner_initialization not in ("stage2", "random"):
            raise ValueError("planner_initialization must be stage2 or random.")
        self.planner_initialization = planner_initialization
        self._validate_planner_initialization(reference_policy_checkpoint)
        if (isinstance(training_comfort_weight, bool) or not isinstance(training_comfort_weight, (int, float))
                or not math.isfinite(training_comfort_weight) or training_comfort_weight < 0):
            raise ValueError("training_comfort_weight must be finite and nonnegative.")
        self.training_comfort_weight = float(training_comfort_weight)
        if (isinstance(training_ep_weight, bool) or not isinstance(training_ep_weight, (int, float))
                or not math.isfinite(training_ep_weight) or training_ep_weight < 0):
            raise ValueError("training_ep_weight must be finite and nonnegative.")
        self.training_ep_weight = float(training_ep_weight)
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        if pdm_reward_workers <= 0:
            raise ValueError("pdm_reward_workers must be a positive integer.")
        self.pdm_reward_workers = pdm_reward_workers
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=device
            )

            if not self.train_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True

        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method)

        cfg.vlm_size = self.vlm_size
        cfg.conditioning_mode = conditioning_mode
        cfg.mask_vlm_padding = mask_vlm_padding
        cfg.vlm_kv_layers = self.vlm_kv_layers if conditioning_mode == "vlm_kv" else []
        if conditioning_mode == "vlm_kv":
            cfg.diffusion_model_cfg.update(conditioning_mode=conditioning_mode,
                cross_attention_heads=cross_attention_heads, cross_attention_kv_heads=cross_attention_kv_heads,
                cross_attention_head_dim=cross_attention_head_dim, cross_attention_q_norm=cross_attention_q_norm,
                vlm_rope_theta=vlm_rope_theta)
            vlm_config = self.backbone.model.language_model.config
            rotary = self.backbone.rotary_contract()
            if (vlm_config.num_key_value_heads != cross_attention_kv_heads or
                    rotary != {"implementation": "Qwen2RotaryEmbedding", "head_dim": cross_attention_head_dim,
                               "theta": float(vlm_rope_theta)}):
                raise ValueError(f"Configured KV geometry/RoPE differs from the VLM runtime: {rotary}")
        if mask_vlm_padding:
            import transformers
            cfg.vlm_identity = {**vlm_file_identity(vlm_path), "transformers": transformers.__version__,
                                "torch": str(torch.__version__), "extraction": "internvl-decoder-v1"}
        cfg.velacc_dropout_probability = velacc_dropout_probability
        cfg.velacc_dropout_seed = velacc_dropout_seed
        cfg.joint_motion_dropout_probability = joint_motion_dropout_probability
        if (isinstance(joint_motion_dropout_probability, bool) or
                not isinstance(joint_motion_dropout_probability, (int, float)) or
                not math.isfinite(joint_motion_dropout_probability) or not 0 <= joint_motion_dropout_probability <= 1):
            raise ValueError("joint_motion_dropout_probability must be finite and in [0, 1]")
        if joint_motion_dropout_probability > 0 and (not grpo or not cache_hidden_state or train_backbone or vlm_type != "internvl"):
            raise ValueError("Joint dropout requires stage-3 GRPO, cached full inputs and a frozen InternVL backbone")

        if self.grpo:
            cfg.grpo_cfg.planner_initialization = self.planner_initialization
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            cfg.grpo_cfg.pdm_reward_workers = self.pdm_reward_workers
            cfg.grpo_cfg.training_reward_mode = self.training_reward_mode
            cfg.grpo_cfg.training_raw_safety_weight = self.training_raw_safety_weight
            cfg.grpo_cfg.training_raw_contact_speed_weight = self.training_raw_contact_speed_weight
            cfg.grpo_cfg.training_raw_safety_inputs = self.training_raw_safety_inputs
            cfg.grpo_cfg.training_minimum_jerk = self.training_minimum_jerk
            cfg.grpo_cfg.deterministic_final_step = self.grpo_deterministic_final_step
            cfg.grpo_cfg.sibling_weight = self.sibling_weight
            cfg.grpo_cfg.training_siblings = self.training_siblings
            cfg.grpo_cfg.scorer_config.comfortable_weight = self.training_comfort_weight
            cfg.grpo_cfg.scorer_config.progress_weight = self.training_ep_weight
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        self.joint_image_lookup = {}
        self.joint_feature_contract = None
        if joint_motion_dropout_probability > 0:
            # Loading frozen weights must not perturb the planner/diffusion RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self.backbone = RecogDriveBackbone(model_type=vlm_type, checkpoint_path=vlm_path, device=device)
            self.backbone.requires_grad_(False).eval()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None and not self.train_backbone:
            self.backbone.eval()
        return self

    def configure_joint_images(self, cache_paths, log_root, sensor_root, mixed_manifest=None):
        from .recogdrive_motion_inputs import build_image_lookup, feature_contract
        import torch.distributed as dist
        packet = [None]
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                if mixed_manifest is None:
                    lookup, sources = build_image_lookup(cache_paths, log_root, sensor_root)
                else:
                    from navsim.planning.training.recogdrive_mixed_data import mixed_image_lookup
                    lookup, sources = mixed_image_lookup(cache_paths, log_root, sensor_root, mixed_manifest)
                packet[0] = {"lookup": lookup, "contract": feature_contract(self.vlm_path, lookup, sources)}
            except Exception as exc:
                packet[0] = {"error": f"Joint image preparation: {type(exc).__name__}: {exc}"}
        if dist.is_initialized():
            dist.broadcast_object_list(packet)
        if "error" in packet[0]:
            raise ValueError(packet[0]["error"])
        self.joint_image_lookup = packet[0]["lookup"]
        self.joint_feature_contract = packet[0]["contract"]

    def _validate_planner_initialization(self, reference_policy_checkpoint) -> None:
        if self.grpo and self.planner_initialization == "random":
            if self.bc_coeff != 0:
                raise ValueError("Random planner initialization requires bc_coeff=0 throughout training.")
            if self.checkpoint_path or reference_policy_checkpoint:
                raise ValueError(
                    "Random planner initialization requires empty checkpoint_path and reference_policy_checkpoint; "
                    "use resume_checkpoint_path to restore a random-initialized training run."
                )

    def initialize(self) -> None:
        self._validate_planner_initialization(self.reference_policy_checkpoint)
        if getattr(self, "mask_vlm_padding", False) and not self.checkpoint_path:
            raise ValueError("Stage-2 inference requires checkpoint_path; select a trained checkpoint")
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            # The teacher was initialized separately from the IL reference. Weight
            # initialization loads the student; full teacher/optimizer restoration
            # belongs to Lightning's resume path.
            self.action_head.load_state_dict(planner_state_from_checkpoint(self.action_head, checkpoint), strict=False)
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in checkpoint["state_dict"].items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if not k2.startswith("action_head.") and k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        if getattr(self, "mask_vlm_padding", False):
            sensors = SensorConfig.build_no_sensors()
            sensors.cam_f0 = [3]  # Motion history is loaded independently of camera/lidar blobs.
            return sensors
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
            cache_mode=self.cache_mode,
            include_valid_mask=getattr(self, "mask_vlm_padding", False),
        )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        device = next(self.action_head.parameters()).device
        features = {key: tensor.to(device) if isinstance(tensor, torch.Tensor) and key != "image_path_tensor"
                    else tensor for key, tensor in features.items()}
        model_dtype = next(self.action_head.parameters()).dtype
        conditioning = None

        history_trajectory = features["history_trajectory"]
        high_command_one_hot = features["high_command_one_hot"]
        
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"]
            if getattr(self, "mask_vlm_padding", False):
                if "context_valid_mask" not in features:
                    raise ValueError("Masked controls require tokenizer masks; old hidden-state caches omit them")
                conditioning = VLMConditioning(features["context_valid_mask"])
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            if "pixel_values" in features:  # already preprocessed by DataLoader workers (collate_features)
                num_patches_list = features["num_patches"].tolist()
                pixel_values_cat = features["pixel_values"]
            else:
                image_path_tensor = features["image_path_tensor"]
                if image_path_tensor.ndim == 1:
                    image_path_tensor = image_path_tensor.unsqueeze(0)
                image_paths = self._decode_paths_from_tensor(image_path_tensor)

                pixel_values_list = [load_image(path) for path in image_paths]

                num_patches_list = [p.shape[0] for p in pixel_values_list]
                pixel_values_cat = torch.cat(pixel_values_list, dim=0).to(device)


            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]

                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                    f'{format_number(history_trajectory_sample[j, 1].item())}, '
                    f'{format_number(history_trajectory_sample[j, 2].item())})'
                    for j in range(history_trajectory_sample.shape[0])
                ])
                
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            if getattr(self, "mask_vlm_padding", False):
                layers = self.vlm_kv_layers if self.conditioning_mode == "vlm_kv" else ()
                last_hidden_state, conditioning = self.backbone.extract_conditioning(
                    pixel_values_cat, questions, num_patches_list, layers=layers,
                    action_horizon=self.action_head.config.action_horizon)
            else:
                with nullcontext() if self.train_backbone else torch.no_grad():
                    outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
                    last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"]
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
        conditioning_kwargs = {"vlm_conditioning": conditioning} if conditioning is not None else {}

        if self.training and not self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            if getattr(self, "per_sample_loss", False):
                # Same draws and objective as the planner forward; the mean is its loss.
                losses = self.action_head.denoising_losses(last_hidden_state, action_inputs.his_traj,
                                                           action_inputs.status_feature, action_inputs.action)
                return BatchFeature(data={"loss": losses.mean(), "sample_losses": losses})
            return self.action_head(last_hidden_state, action_inputs, **conditioning_kwargs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            joint_kwargs, joint_timings = {}, {}
            student_features = last_hidden_state
            if self.action_head.config.joint_motion_dropout_probability > 0:
                from .recogdrive_motion_inputs import replace_dropped_features
                if self.joint_feature_contract is None:
                    raise RuntimeError("Joint dropout requires the training image lookup to be configured")
                mask = self.action_head.sample_joint_drop_mask(action_inputs.status_feature)
                student_features, joint_timings = replace_dropped_features(
                    self.backbone, last_hidden_state, high_command_one_hot, tokens_list, mask, self.joint_image_lookup)
                joint_kwargs = {"joint_drop_mask": mask, "teacher_vl_features": last_hidden_state}
            if getattr(self, "training_reward_mode", "executed") == "minimum_jerk":
                # Clean GT before the policy-precision cast above, for reward EP only.
                joint_kwargs["reward_targets"] = targets["trajectory"]
            result = self.action_head.forward_grpo(
                student_features, action_inputs, tokens_list,
                bc_coeff=self.bc_coeff, use_bc_loss=self.bc_coeff > 0,
                **joint_kwargs,
            )
            result.update({k: last_hidden_state.new_tensor(v) for k, v in joint_timings.items()})
            return result
        else: 
            action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs,
                                               **conditioning_kwargs)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad(), self._inference_autocast():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad(), self._inference_autocast():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)

    def _inference_autocast(self):
        precision = getattr(self, "inference_precision", "32-true")
        device = next(self.action_head.parameters()).device
        return torch.autocast(device.type, enabled=precision != "32-true",
                              dtype=torch.bfloat16 if precision == "bf16-mixed" else torch.float16)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))

        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self._lr,
                min_lr=0.0,
                epochs=self.lr_scheduler_epochs,
                warmup_epochs=0,
            )
        elif getattr(self, "mask_vlm_padding", False):
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6,
                                    epochs=self.lr_scheduler_epochs, warmup_epochs=min(3, self.lr_scheduler_epochs - 1))
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor.detach().cpu():
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
