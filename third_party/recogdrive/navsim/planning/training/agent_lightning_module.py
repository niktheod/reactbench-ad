import pytorch_lightning as pl
import torch
import logging

from torch import Tensor
from typing import Dict, Tuple,Any

from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.raw_pdm_score import training_reward_protocol

logger = logging.getLogger(__name__)


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    REWARD_COHORTS = ("original", "hazardous", "benign")
    REWARD_GROUP_METRICS = ("equal_reward_group_fraction", "zero_reward_group_fraction", "reward_group_std")
    # Each reward's own definitions: collision-free and NC x DAC = 1 (hybrid: raw NC-all).
    SAFETY_GROUP_FLAGS = ("any_collision_free", "all_collision_free", "any_safe")
    SAFETY_GROUP_METRICS = tuple(f"{flag}_group_fraction" for flag in SAFETY_GROUP_FLAGS)
    # (logged name, numerator, denominator) over minimum_jerk diagnostic sums.
    MINIMUM_JERK_RATIOS = (
        ("infeasible_fit_fraction", "infeasible_fit", "candidates"),
        ("beta", "beta_sum", "candidates"),
        ("qp_fallback_fraction", "qp_fallback", "fitted"),
        ("nc", "nc_sum", "fitted"), ("dac", "dac_sum", "fitted"), ("ttc", "ttc_sum", "fitted"),
        ("ep", "ep_sum", "fitted"), ("comfort", "comfort_sum", "fitted"),
        ("base_reward", "base_reward_sum", "fitted"), ("contact_penalty", "contact_penalty_sum", "fitted"),
        ("max_speed_ratio", "max_speed_ratio_sum", "fitted"), ("alpha", "alpha_sum", "fitted"),
        ("feasibility_ratio", "feasibility_ratio_sum", "fitted"),
        ("beta_backward", "beta_backward_sum", "fitted"), ("backward_fraction", "backward_count", "fitted"),
        ("backward_distance_m", "backward_distance_sum", "backward_count"),
        ("total_acceleration_violation_fraction", "total_acceleration_violation", "fitted"),
        ("forward_acceleration_violation_fraction", "forward_acceleration_violation", "fitted"),
        ("curvature_violation_fraction", "curvature_violation", "fitted"),
        ("initial_contact_fraction", "initial_contact", "fitted"),
        ("vru_rule_contact_fraction", "vru_rule_contact", "fitted"),
        ("contact_fraction", "contact_count", "fitted"),
        ("contact_speed_mps", "contact_speed_sum", "contact_count"),
    )
    # (logged name, numerator, denominator) over sibling diagnostic sums.
    SIBLING_RATIOS = (
        ("target_rate", "targets", "scenes"), ("margin", "margin_sum", "feasible_base"),
        ("selected_decel", "deceleration_sum", "targets"), ("late_onset_fraction", "late_onset", "targets"),
        ("base_contact_fraction", "base_contact", "feasible_base"),
        ("target_contact_fraction", "target_contact", "targets"),
        ("base_infeasible_fraction", "base_infeasible", "scenes"),
        ("base_backward_fraction", "base_backward", "scenes"), ("loss", "loss_sum", "scenes"),
    )
    # Kinematic SFT: per-cohort mean of the per-sample stage-2 loss, and the sample count.
    SFT_COHORT_METRICS = ("sft_loss", "sft_sample_count")

    def __init__(self, agent: AbstractAgent, training_batch_size=None, training_sample_kinds=None):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.training_batch_size = training_batch_size
        self.training_sample_kinds = training_sample_kinds
        # Rank-local metric state must not be a registered DDP buffer: buffer
        # broadcasts would overwrite other ranks' running sums at each forward.
        self._raw_contact_totals = torch.zeros(3, dtype=torch.float64)
        self._has_raw_reward = False
        self._reward_cohort_totals = torch.zeros(len(self.REWARD_COHORTS), 4, dtype=torch.float64)
        self._has_reward_groups = False
        self._safety_cohort_totals = torch.zeros(len(self.REWARD_COHORTS), 4, dtype=torch.float64)
        self._has_safety_groups = False
        self._contact_cohort_totals = torch.zeros(len(self.REWARD_COHORTS), 2, dtype=torch.float64)
        self._has_contact_groups = False
        self._minimum_jerk_totals = None
        self._has_minimum_jerk = False
        self._sibling_totals = None
        self._has_siblings = False
        self._sft_cohort_totals = torch.zeros(len(self.REWARD_COHORTS), 2, dtype=torch.float64)
        self._has_sft_losses = False

    def _cohort_totals(self, columns, tokens, scene_count=True):
        """Rank-local sums and (by default) scene counts; the manifest supplies cohort labels."""
        kinds = (["original"] * len(tokens) if self.training_sample_kinds is None
                 else [self.training_sample_kinds[token] for token in tokens])
        columns = [column.detach() for column in columns]
        columns = (*columns, torch.ones_like(columns[0])) if scene_count else columns
        values = torch.stack(columns, dim=1).to(dtype=torch.float64)
        indices = torch.tensor([self.REWARD_COHORTS.index(kind) for kind in kinds], device=values.device)
        totals = values.new_zeros((len(self.REWARD_COHORTS), values.shape[1]))
        return totals.index_add_(0, indices, values)

    def _reward_group_totals(self, diagnostics, tokens):
        return self._cohort_totals([diagnostics[name] for name in ("equal", "zero", "std")], tokens)

    def _safety_group_totals(self, diagnostics, tokens):
        return self._cohort_totals([diagnostics[name] for name in self.SAFETY_GROUP_FLAGS], tokens)

    def _contact_group_totals(self, diagnostics, tokens):
        return self._cohort_totals([diagnostics["contact_speed_sum"], diagnostics["contact_count"]], tokens,
                                   scene_count=False)

    def _log_contact_cohorts(self, totals, suffix):
        """Mean first-contact speed over contacts per cohort; NaN without contacts."""
        for cohort, (speed_sum, count) in zip(self.REWARD_COHORTS, totals):
            mean = torch.where(count > 0, speed_sum / count.clamp_min(1), torch.full_like(speed_sum, float("nan")))
            for name, value in (("contact_speed_mps", mean), ("contact_count", count)):
                self.log(f"train/{cohort}/{name}_{suffix}", value, on_step=suffix == "step",
                         on_epoch=suffix == "epoch", sync_dist=False)

    def _log_reward_cohorts(self, totals, suffix, names=None):
        """Log globally reduced totals, without averaging per-rank fractions.

        The last column is the scene count; it is logged only when ``names``
        includes a name for it (the reward-group count).
        """
        names = (*self.REWARD_GROUP_METRICS, "reward_group_count") if names is None else names
        for cohort, row in zip(self.REWARD_COHORTS, totals):
            count = row[-1]
            means = row[:-1] / count.clamp_min(1)
            means = torch.where(count > 0, means, torch.full_like(means, float("nan")))
            for name, value in zip(names, (*means, count)):
                self.log(f"train/{cohort}/{name}_{suffix}", value, on_step=suffix == "step",
                         on_epoch=suffix == "epoch", sync_dist=False)

    def _log_minimum_jerk(self, totals, suffix):
        """Ratios of globally reduced sums; NaN when a denominator is zero."""
        from navsim.evaluate.minimum_jerk_pdm_score import DIAGNOSTIC_FIELDS
        values = dict(zip(DIAGNOSTIC_FIELDS, totals))
        logged = [(name, values[name]) for name in ("candidates", "fitted", "contact_count")]
        for name, numerator, denominator in self.MINIMUM_JERK_RATIOS:
            count = values[denominator]
            ratio = values[numerator] / count.clamp_min(1)
            logged.append((name, torch.where(count > 0, ratio, torch.full_like(ratio, float("nan")))))
        for name, value in logged:
            self.log(f"train/minimum_jerk/{name}_{suffix}", value, on_step=suffix == "step",
                     on_epoch=suffix == "epoch", sync_dist=False)

    def _log_siblings(self, totals, suffix):
        """Overall and per-cohort ratios of globally reduced sibling sums; NaN for empty denominators."""
        from navsim.evaluate.sibling_targets import SIBLING_FIELDS
        for prefix, row in (("", totals.sum(dim=0)), *((f"{c}/", r) for c, r in zip(self.REWARD_COHORTS, totals))):
            values = dict(zip(SIBLING_FIELDS, row))
            for name, numerator, denominator in self.SIBLING_RATIOS:
                count = values[denominator]
                ratio = torch.where(count > 0, values[numerator] / count.clamp_min(1),
                                    torch.full_like(count, float("nan")))
                self.log(f"train/sibling/{prefix}{name}_{suffix}", ratio, on_step=suffix == "step",
                         on_epoch=suffix == "epoch", sync_dist=False)

    def _training_protocol(self):
        if not getattr(self.agent, "grpo", False):
            return None
        protocol = {"version": 1, "planner_initialization": self.agent.planner_initialization,
                    "bc_coeff": float(self.agent.bc_coeff)}
        # Absent when disabled, so earlier checkpoints still resume.
        if getattr(self.agent, "grpo_deterministic_final_step", False):
            protocol["deterministic_final_step"] = True
        if getattr(self.agent, "sibling_weight", 0) > 0:
            from navsim.evaluate.sibling_targets import protocol_metadata
            protocol["siblings"] = {"weight": float(self.agent.sibling_weight),
                                    **protocol_metadata(self.agent.training_siblings)}
        return protocol

    def _training_reward_weights(self):
        """Read the resolved scorer used for GRPO rollouts; other agents have none."""
        planner = getattr(self.agent, "action_head", None)
        config = getattr(planner, "_pdm_scorer_config", None)
        if config is None:
            return None
        return {name: getattr(config, name) for name in (
            "progress_weight", "ttc_weight", "comfortable_weight", "driving_direction_weight",
        )}

    def _training_reward_protocol(self):
        planner = getattr(self.agent, "action_head", None)
        if getattr(planner, "_pdm_scorer_config", None) is None:
            return None
        cfg = planner.config.grpo_cfg
        if cfg.training_reward_mode == "executed_raw_safety":
            return training_reward_protocol(cfg.training_reward_mode,
                raw_safety_weight=cfg.training_raw_safety_weight,
                raw_contact_speed_weight=cfg.training_raw_contact_speed_weight,
                raw_safety_inputs=planner._raw_safety_inputs.protocol_metadata())
        if cfg.training_reward_mode == "minimum_jerk":
            return training_reward_protocol(cfg.training_reward_mode, minimum_jerk=cfg.training_minimum_jerk)
        return training_reward_protocol(cfg.training_reward_mode)

    def on_train_epoch_start(self):
        self._raw_contact_totals.zero_()
        self._has_raw_reward = False
        self._reward_cohort_totals.zero_()
        self._has_reward_groups = False
        self._safety_cohort_totals.zero_()
        self._has_safety_groups = False
        self._contact_cohort_totals.zero_()
        self._has_contact_groups = False
        self._minimum_jerk_totals = None
        self._has_minimum_jerk = False
        self._sibling_totals = None
        self._has_siblings = False
        self._sft_cohort_totals.zero_()
        self._has_sft_losses = False

    def on_train_epoch_end(self):
        if (self._has_raw_reward or self._has_reward_groups or self._has_safety_groups or self._has_contact_groups
                or self._has_minimum_jerk or self._has_siblings or self._has_sft_losses):
            # Rank zero's progress callback may already be reducing Lightning's
            # epoch metrics. Finish those reductions on every rank before our
            # custom collective, so their order and tensor shapes agree.
            _ = self.trainer.callback_metrics
        if self._has_raw_reward:
            totals = self.trainer.strategy.reduce(self._raw_contact_totals.clone(), reduce_op="sum")
            mean = totals[0] / totals[1] if totals[1] > 0 else totals.new_tensor(float("nan"))
            # Reduce sums/counts first: ranks and batches with zero contacts must
            # not enter the speed denominator, and a zero-speed contact must.
            self.log("train/raw_contact_speed_mps_epoch", mean, on_step=False, on_epoch=True, sync_dist=False)
        if self._has_reward_groups:
            totals = self.trainer.strategy.reduce(self._reward_cohort_totals.clone(), reduce_op="sum")
            self._log_reward_cohorts(totals, "epoch")
        if self._has_safety_groups:
            totals = self.trainer.strategy.reduce(self._safety_cohort_totals.clone(), reduce_op="sum")
            self._log_reward_cohorts(totals, "epoch", self.SAFETY_GROUP_METRICS)
        if self._has_contact_groups:
            totals = self.trainer.strategy.reduce(self._contact_cohort_totals.clone(), reduce_op="sum")
            self._log_contact_cohorts(totals, "epoch")
        if self._has_minimum_jerk:
            totals = self.trainer.strategy.reduce(self._minimum_jerk_totals.clone(), reduce_op="sum")
            self._log_minimum_jerk(totals, "epoch")
        if self._has_siblings:
            totals = self.trainer.strategy.reduce(self._sibling_totals.clone(), reduce_op="sum")
            self._log_siblings(totals, "epoch")
        if self._has_sft_losses:
            totals = self.trainer.strategy.reduce(self._sft_cohort_totals.clone(), reduce_op="sum")
            self._log_reward_cohorts(totals, "epoch", self.SFT_COHORT_METRICS)

    def _sft_step(self, features, targets, tokens_list):
        """Kinematic SFT: the stage-2 objective on each sample's target,
        logged per cohort from globally reduced per-sample sums."""
        prediction = self.agent.forward(features, targets, tokens_list)
        loss = prediction["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(tokens_list))
        totals = self._cohort_totals([prediction["sample_losses"]], tokens_list)
        self._has_sft_losses = True
        self._sft_cohort_totals = self._sft_cohort_totals.to(totals).add_(totals)
        totals = self.trainer.strategy.reduce(totals.clone(), reduce_op="sum")
        self._log_reward_cohorts(totals, "step", self.SFT_COHORT_METRICS)
        return loss

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        if logging_prefix == 'train' and getattr(self.agent, "per_sample_loss", False):
            return self._sft_step(features, targets, tokens_list)
        if logging_prefix == 'train':
            prediction = self.agent.forward(features,targets,tokens_list)
            predictions = self.agent.compute_loss(features, targets, prediction)

            loss = predictions.loss
            reward = predictions.reward
            raw = predictions.get("raw_reward_diagnostics", {})
            if raw:
                reward = raw["total_reward"]
            policy_loss = predictions.policy_loss
            bc_loss = predictions.bc_loss
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                     batch_size=predictions.get("reward_candidate_count", len(tokens_list)))
            self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            for name, value in predictions.get("reward_components", {}).items():
                self.log(f"train/pdms_{name}", value.detach(), on_step=True, on_epoch=True,
                         sync_dist=True, batch_size=predictions["reward_candidate_count"])
            if raw:
                self._has_raw_reward = True
                for name in ("executed_reward", "raw_nc_all", "raw_dac", "raw_safety", "raw_contact_penalty"):
                    key = "pdms_score" if name == "executed_reward" else name
                    self.log(f"train/{key}", raw[name].detach(), on_step=True, on_epoch=True,
                             sync_dist=True, batch_size=predictions["reward_candidate_count"])
                totals = torch.stack([raw[name].detach() for name in (
                    "raw_contact_speed_sum", "raw_contact_speed_count", "raw_initial_contact_count")])
                self._raw_contact_totals = self._raw_contact_totals.to(totals).add_(totals)
                for name in ("raw_contact_speed_count", "raw_initial_contact_count"):
                    self.log(f"train/{name}", raw[name].detach(), on_step=True, on_epoch=True,
                             sync_dist=True, reduce_fx="sum", batch_size=predictions["reward_candidate_count"])
            for name in ("reward_group_std", "equal_reward_group_fraction", "group_pairwise_ade_m"):
                if name in predictions:
                    self.log(f"train/{name}", predictions[name], on_step=True, on_epoch=True,
                             sync_dist=True, batch_size=len(tokens_list))
            if "reward_group_diagnostics" in predictions:
                self._has_reward_groups = True
                totals = self._reward_group_totals(predictions["reward_group_diagnostics"], tokens_list)
                self._reward_cohort_totals = self._reward_cohort_totals.to(totals).add_(totals)
                totals = self.trainer.strategy.reduce(totals.clone(), reduce_op="sum")
                self._log_reward_cohorts(totals, "step")
            if "safety_group_diagnostics" in predictions:
                self._has_safety_groups = True
                totals = self._safety_group_totals(predictions["safety_group_diagnostics"], tokens_list)
                self._safety_cohort_totals = self._safety_cohort_totals.to(totals).add_(totals)
                totals = self.trainer.strategy.reduce(totals.clone(), reduce_op="sum")
                self._log_reward_cohorts(totals, "step", self.SAFETY_GROUP_METRICS)
            if "contact_group_diagnostics" in predictions:
                self._has_contact_groups = True
                totals = self._contact_group_totals(predictions["contact_group_diagnostics"], tokens_list)
                self._contact_cohort_totals = self._contact_cohort_totals.to(totals).add_(totals)
                totals = self.trainer.strategy.reduce(totals.clone(), reduce_op="sum")
                self._log_contact_cohorts(totals, "step")
            if "minimum_jerk_reward_diagnostics" in predictions:
                self._has_minimum_jerk = True
                totals = predictions["minimum_jerk_reward_diagnostics"].detach().to(torch.float64)
                self._minimum_jerk_totals = (totals.clone() if self._minimum_jerk_totals is None
                                             else self._minimum_jerk_totals.to(totals).add_(totals))
                self._log_minimum_jerk(self.trainer.strategy.reduce(totals.clone(), reduce_op="sum"), "step")
            if "sibling_diagnostics" in predictions:
                self._has_siblings = True
                rows = predictions["sibling_diagnostics"].detach().to(torch.float64)
                totals = self._cohort_totals(list(rows.unbind(dim=1)), tokens_list, scene_count=False)
                self._sibling_totals = (totals.clone() if self._sibling_totals is None
                                        else self._sibling_totals.to(totals).add_(totals))
                self._log_siblings(self.trainer.strategy.reduce(totals.clone(), reduce_op="sum"), "step")
            if "velacc_dropped_samples" in predictions:
                for name in ("velacc_dropped_samples", "velacc_total_samples"):
                    self.log(f"train/{name}", predictions[name].float(), on_step=True, on_epoch=True,
                             sync_dist=True, reduce_fx="sum", batch_size=len(tokens_list))
                for name in ("velacc_dropout_probability", "velacc_null_norm"):
                    self.log(f"train/{name}", predictions[name], on_step=True, on_epoch=True,
                             sync_dist=True, batch_size=len(tokens_list))
            for name in ("joint_dropped_samples", "joint_total_samples", "joint_image_seconds", "joint_vlm_seconds"):
                if name in predictions:
                    self.log(f"train/{name}", predictions[name].float(), on_step=True, on_epoch=True,
                             sync_dist=True, reduce_fx="sum", batch_size=len(tokens_list))
            for name in ("joint_dropout_probability", "joint_null_history_norm"):
                if name in predictions:
                    self.log(f"train/{name}", predictions[name], on_step=True, on_epoch=True,
                             sync_dist=True, batch_size=len(tokens_list))
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd
        planner = getattr(self.agent, "action_head", None)
        if hasattr(planner, "joint_dropout_state"):
            checkpoint["joint_motion_dropout"] = {
                **planner.joint_dropout_state(), "batch_size": self.training_batch_size,
                "world_size": self.trainer.world_size,
                "features": getattr(self.agent, "joint_feature_contract", None),
            }
            if planner.config.joint_motion_dropout_probability > 0:
                checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items()
                                             if not k.startswith("agent.backbone.")}
        if hasattr(planner, "motion_dropout_state"):
            checkpoint["velacc_dropout"] = {
                **planner.motion_dropout_state(), "batch_size": self.training_batch_size,
                "world_size": self.trainer.world_size,
            }
        weights = self._training_reward_weights()
        if weights is not None:
            checkpoint["training_reward_weights"] = weights
        reward_protocol = self._training_reward_protocol()
        if reward_protocol is not None:
            checkpoint["training_reward_protocol"] = reward_protocol
        protocol = self._training_protocol()
        if protocol is not None:
            checkpoint["stage3_training"] = protocol
        mixed = getattr(self, "mixed_training_protocol", None)
        if mixed is not None:
            checkpoint["mixed_training"] = mixed

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if checkpoint.get("mixed_training") != getattr(self, "mixed_training_protocol", None):
            raise ValueError("Mixed-data population, update schedule or topology differs on resume; start a new run")
        reward_protocol = self._training_reward_protocol()
        if reward_protocol is not None:
            saved = checkpoint.get("training_reward_protocol", training_reward_protocol("executed"))
            if saved != reward_protocol:
                raise ValueError(
                    f"Training reward protocol differs on resume: saved={saved}, current={reward_protocol}. "
                    "Changing reward mode or protocol requires a new run."
                )
        weights = self._training_reward_weights()
        if weights is not None:
            # Older stage-3 checkpoints used fixed 10/5/2 training weights.
            saved_weights = checkpoint.get("training_reward_weights", {
                "progress_weight": 10.0, "ttc_weight": 5.0,
                "comfortable_weight": 2.0, "driving_direction_weight": 0.0,
            })
            if saved_weights != weights:
                raise ValueError(
                    f"Training reward weights differ on resume: saved={saved_weights}, current={weights}. "
                    "Changing reward weights requires a new run."
                )
        protocol = self._training_protocol()
        if protocol is not None and checkpoint.get("stage3_training") != protocol:
            raise ValueError(f"Stage-3 initialization/BC settings differ on resume: "
                             f"saved={checkpoint.get('stage3_training')}, current={protocol}. "
                             "Start a new run to change them.")
        planner = getattr(self.agent, "action_head", None)
        if hasattr(planner, "joint_dropout_state"):
            joint = checkpoint.get("joint_motion_dropout")
            if joint is None:
                if planner.config.joint_motion_dropout_probability > 0:
                    raise ValueError("Joint dropout resume requires saved joint state; initialize a new run")
                for name, value in self.state_dict().items():
                    if name.endswith(".null_history"):
                        checkpoint["state_dict"].setdefault(name, torch.zeros_like(value))
            else:
                planner.restore_joint_dropout_state(joint)
                if planner.config.joint_motion_dropout_probability > 0:
                    if (joint.get("world_size") != self.trainer.world_size or
                            joint.get("batch_size") != self.training_batch_size or
                            joint.get("features") != getattr(self.agent, "joint_feature_contract", None) or
                            joint.get("features") is None):
                        raise ValueError("Joint dropout resume requires matching topology and VLM/image/prompt identities")
                    # Frozen VLM is restored from verified external weights, never serialized per epoch.
                    checkpoint["state_dict"].update({k: v for k, v in self.state_dict().items()
                                                      if k.startswith("agent.backbone.")})
        if not hasattr(planner, "restore_motion_dropout_state"):
            return
        state = checkpoint.get("velacc_dropout")
        if state is None:
            if planner.config.velacc_dropout_probability > 0:
                raise ValueError("Dropout resume requires saved dropout state; initialize a new run")
            # Preserve legacy p=0 full-state resumes and optimizer membership.
            for name, value in self.state_dict().items():
                if name.endswith(".null_motion_hidden"):
                    checkpoint["state_dict"].setdefault(name, torch.zeros_like(value))
            return
        if planner.config.velacc_dropout_probability > 0 and (
                state.get("world_size") != self.trainer.world_size
                or state.get("batch_size") != self.training_batch_size):
            raise ValueError("Dropout resume requires the same world size and training micro-batch size")
        planner.restore_motion_dropout_state(state)

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        #print(batch_idx)
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        config = self.agent.get_optimizers()
        mixed = getattr(self, "mixed_training_protocol", None)
        if mixed is not None:
            # Leave all existing epoch-based schedules unchanged. Mixed runs
            # finish at an update limit, including a possible partial epoch.
            warmup = mixed.get("warmup_updates", 0)
            if warmup:
                from functools import partial
                from navsim.planning.training.recogdrive_mixed_training import warmup_cosine_factor
                scheduler = torch.optim.lr_scheduler.LambdaLR(config["optimizer"], partial(
                    warmup_cosine_factor, max_updates=mixed["max_updates"], warmup_updates=warmup))
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    config["optimizer"], T_max=mixed["max_updates"], eta_min=0.0)
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return config
