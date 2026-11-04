from typing import Tuple
from pathlib import Path
import datetime
import logging
import math
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule, AgentLightningDiT
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "recogdrive_stage3"
# Rank 0's dataset check stats every file of a mixed view on the shared filesystem; the other
# ranks wait for its verdict on a CPU group with this timeout instead of the NCCL default (10 min).
RANK0_CHECK_TIMEOUT = datetime.timedelta(hours=2)


def rank0_check(rank, check, timeout=RANK0_CHECK_TIMEOUT):
    """Run ``check`` on rank 0 only; every rank raises its error as a ValueError.

    The verdict travels on a temporary gloo group, so a long check does not trip the training
    (NCCL) group's timeout; that group keeps its default for hangs during training.
    """
    group = dist.new_group(backend="gloo", timeout=timeout)
    try:
        error = [None]
        if rank == 0:
            try:
                check()
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(error, group=group)
    finally:
        dist.destroy_process_group(group)
    if error[0]:
        raise ValueError(error[0])



def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()


    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
        'last_hidden_state': last_hidden_state,
    }
    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list


class Stage3ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """Preserve the reward ranking on resume and always keep last.ckpt current."""

    def load_state_dict(self, state_dict) -> None:
        # Lightning otherwise drops the ranking when a new Slurm logger version
        # changes dirpath. Reuse the saved directory, including legacy layouts.
        saved_dir = state_dict.get("dirpath")
        if saved_dir is not None:
            if not Path(saved_dir).is_absolute():
                raise ValueError(f"Checkpoint resume requires an absolute saved directory: {saved_dir}")
            missing = [path for path in state_dict.get("best_k_models", {}) if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot restore checkpoint ranking; restore the retained checkpoints "
                    f"in their original directory before resuming. Missing: {missing}"
                )
            if self.dirpath != saved_dir:
                logger.info("Resuming reward checkpoints in their saved directory: %s", saved_dir)
            self.dirpath = saved_dir
        super().load_state_dict(state_dict)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_skip_saving_checkpoint(trainer) or not self._should_save_on_train_epoch_end(trainer):
            return
        super().on_train_epoch_end(trainer, pl_module)
        # Lightning 2.6.5 skips save_last when no top-k checkpoint was saved.
        if self.save_last and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


def build_validation_callbacks(cfg, *, kinematic_sft=False):
    """Choose checkpoints independently of optional edited safety monitoring."""
    selection_mode = cfg.checkpoint_selection
    if selection_mode not in ("training_reward", "validation", "all_epochs"):
        raise ValueError(f"Unknown checkpoint_selection: {selection_mode}")
    if kinematic_sft and selection_mode != "all_epochs":
        raise ValueError("Kinematic SFT requires checkpoint_selection=all_epochs")
    if selection_mode == "all_epochs":
        return build_kinematic_sft_callbacks(cfg) if kinematic_sft else build_hazard_sft_callbacks(cfg)
    if cfg.get("paired_validation_config"):
        raise ValueError("paired_validation_config is for hazard SFT runs (checkpoint_selection=all_epochs)")
    evaluator = None
    if selection_mode == "validation" or cfg.get("edited_validation", False):
        from navsim.planning.training.recogdrive_validation import ValidationEvaluator, load_config
        if not cfg.validation_config:
            raise ValueError("validation_config is required for validation selection or edited reporting")
        config = load_config(cfg.validation_config, training_batch_size=cfg.dataloader.params.batch_size,
                             edited_protocol=cfg.get("edited_validation_protocol"))
        from navsim.planning.training.recogdrive_triplet_validation import is_triplet, TripletValidationEvaluator
        if is_triplet(config):
            if selection_mode != "training_reward":
                raise ValueError("Triplet evaluation is monitoring only; select checkpoints by training_reward")
            manifest = load_manifest_for_triplets(cfg, config)
            evaluator = TripletValidationEvaluator(config, edited_only=True)
            if evaluator.manifest["artifact_sha256"] != manifest["artifact_sha256"]:
                raise ValueError("Triplet holdout differs from the training population")
        else:
            evaluator = ValidationEvaluator(config, edited_only=selection_mode == "training_reward")
        evaluator.check_agent_config(cfg.agent)
        logger.info("Validation inference batch size per GPU: %d", config["batch_size"])
    if selection_mode == "validation":
        from navsim.planning.training.recogdrive_validation import load_baseline
        from navsim.planning.training.callbacks.recogdrive_validation_callback import RecogDriveValidationCallback
        baseline, identity = load_baseline(config, evaluator)
        callbacks = [RecogDriveValidationCallback(evaluator, baseline, identity, cfg.output_dir)]
    else:
        callbacks = [Stage3ModelCheckpoint(
            dirpath=Path(cfg.output_dir).expanduser().resolve() / "checkpoints",
            monitor="train/reward_epoch", mode="max", save_top_k=5, save_last=True,
            every_n_epochs=1, save_on_train_epoch_end=True)]
        if evaluator is not None:
            from navsim.planning.training.callbacks.recogdrive_validation_callback import RecogDriveEditedValidationCallback, RecogDriveTripletValidationCallback
            callbacks.append((RecogDriveTripletValidationCallback if is_triplet(config) else RecogDriveEditedValidationCallback)(evaluator, cfg.output_dir))
    interval = cfg.get("fixed_update_checkpoint_interval", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("fixed_update_checkpoint_interval must be a nonnegative integer")
    if interval:
        callbacks.append(Stage3ModelCheckpoint(dirpath=Path(cfg.output_dir).expanduser().resolve() / "checkpoints/updates",
            filename="update={step}", auto_insert_metric_name=False, save_top_k=-1, save_last=False,
            every_n_train_steps=interval, save_on_train_epoch_end=False))
    logger.info("Stage-3 checkpoint selection: %s; edited reporting: %s", selection_mode, evaluator is not None)
    return callbacks


def build_hazard_sft_callbacks(cfg):
    """Hazard SFT: keep every epoch checkpoint (there is no training
    reward) and monitor both the held-out triplets (with L1/V0) and the paired edited set."""
    from navsim.planning.training.recogdrive_validation import ValidationEvaluator, load_config
    from navsim.planning.training.recogdrive_triplet_validation import TripletValidationEvaluator
    from navsim.planning.training.callbacks.recogdrive_validation_callback import (
        RecogDriveEditedValidationCallback, RecogDriveTripletValidationCallback)
    if (not cfg.get("edited_validation", False) or cfg.get("edited_validation_protocol") != "triplet"
            or not cfg.get("validation_config") or not cfg.get("paired_validation_config")):
        raise ValueError("Hazard SFT requires triplet (validation_config) and paired (paired_validation_config) "
                         "edited monitoring")
    batch_size = cfg.dataloader.params.batch_size
    triplet_config = load_config(cfg.validation_config, training_batch_size=batch_size, edited_protocol="triplet")
    if triplet_config.get("hazard_sft_metrics") is not True:
        raise ValueError("Hazard SFT triplet monitoring requires hazard_sft_metrics: true")
    manifest = load_manifest_for_triplets(cfg, triplet_config)
    triplets = TripletValidationEvaluator(triplet_config, edited_only=True)
    if triplets.manifest["artifact_sha256"] != manifest["artifact_sha256"]:
        raise ValueError("Triplet holdout differs from the training population")
    paired = ValidationEvaluator(load_config(cfg.paired_validation_config, training_batch_size=batch_size,
                                             edited_protocol="paired"), edited_only=True)
    for evaluator in (triplets, paired):
        evaluator.check_agent_config(cfg.agent)
    output = Path(cfg.output_dir).expanduser().resolve()
    callbacks = [Stage3ModelCheckpoint(dirpath=output / "checkpoints", monitor=None, save_top_k=-1, save_last=True,
                                       every_n_epochs=1, save_on_train_epoch_end=True),
                 RecogDriveTripletValidationCallback(triplets, cfg.output_dir),
                 RecogDriveEditedValidationCallback(paired, cfg.output_dir)]
    interval = cfg.get("fixed_update_checkpoint_interval", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("fixed_update_checkpoint_interval must be a nonnegative integer")
    if interval:
        callbacks.append(Stage3ModelCheckpoint(dirpath=output / "checkpoints/updates",
            filename="update={step}", auto_insert_metric_name=False, save_top_k=-1, save_last=False,
            every_n_train_steps=interval, save_on_train_epoch_end=False))
    logger.info("Hazard SFT checkpoints: every epoch plus last; triplet (L1/V0) and paired edited monitoring; "
                "triplet/paired batch sizes %d/%d", triplet_config["batch_size"], paired.config["batch_size"])
    return callbacks


def build_kinematic_sft_callbacks(cfg):
    """Kinematic stop-target SFT (ReactTrain fine-tuning): keep every epoch checkpoint; val/loss is the clean
    validation split. With validation_config set, a paired edited validation set is also monitored after each
    epoch; the monitor runs on its own RNG state (recogdrive_validation.preserve_inference_state) and does not
    change training. The paper's run used it for monitoring only."""
    output = Path(cfg.output_dir).expanduser().resolve()
    callbacks = [Stage3ModelCheckpoint(dirpath=output / "checkpoints", monitor=None, save_top_k=-1, save_last=True,
                                       every_n_epochs=1, save_on_train_epoch_end=True)]
    if cfg.get("validation_config"):
        from navsim.planning.training.recogdrive_validation import ValidationEvaluator, load_config
        from navsim.planning.training.callbacks.recogdrive_validation_callback import RecogDriveEditedValidationCallback
        if (not cfg.get("edited_validation", False) or cfg.get("edited_validation_protocol") != "paired"
                or cfg.get("paired_validation_config")):
            raise ValueError("Kinematic SFT monitors a paired edited set only: edited_validation=true, "
                             "edited_validation_protocol=paired, paired_validation_config unset")
        paired = ValidationEvaluator(load_config(cfg.validation_config,
                                                 training_batch_size=cfg.dataloader.params.batch_size,
                                                 edited_protocol="paired"), edited_only=True)
        paired.check_agent_config(cfg.agent)
        callbacks.append(RecogDriveEditedValidationCallback(paired, cfg.output_dir))
        logger.info("Paired edited monitoring (batch size %d)", paired.config["batch_size"])
    interval = cfg.get("fixed_update_checkpoint_interval", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("fixed_update_checkpoint_interval must be a nonnegative integer")
    if interval:
        callbacks.append(Stage3ModelCheckpoint(dirpath=output / "checkpoints/updates",
            filename="update={step}", auto_insert_metric_name=False, save_top_k=-1, save_last=False,
            every_n_train_steps=interval, save_on_train_epoch_end=False))
    logger.info("Kinematic SFT checkpoints: every epoch plus last")
    return callbacks


def load_manifest_for_triplets(cfg, config):
    from navsim.planning.training.recogdrive_mixed_data import load_manifest, HOLDOUT_PROTOCOLS
    if not cfg.get("mixed_training_manifest"):
        raise ValueError("Triplet training monitoring requires mixed_training_manifest")
    manifest = load_manifest(cfg.mixed_training_manifest)
    if manifest["protocol"] not in HOLDOUT_PROTOCOLS:
        raise ValueError("Triplet monitoring requires an edited-only or hazard SFT population")
    return manifest


class OptimizerClockValidationCallback(pl.Callback):
    """Fail before optimization if Lightning resolves a different training clock."""

    def __init__(self, expected_steps_per_epoch: int, expected_total_steps: int):
        self.expected_steps_per_epoch = expected_steps_per_epoch
        self.expected_total_steps = expected_total_steps

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not isinstance(trainer.accumulate_grad_batches, int):
            raise RuntimeError(
                "Stage-3 optimizer-clock validation requires a fixed integer "
                "accumulate_grad_batches value."
            )
        steps_per_epoch = math.ceil(
            trainer.num_training_batches / trainer.accumulate_grad_batches
        )
        total_steps = int(trainer.estimated_stepping_batches)
        if steps_per_epoch != self.expected_steps_per_epoch:
            raise RuntimeError(
                "Resolved optimizer steps per epoch do not match the Stage-3 "
                f"contract: actual={steps_per_epoch}, "
                f"expected={self.expected_steps_per_epoch}."
            )
        if total_steps != self.expected_total_steps:
            raise RuntimeError(
                "Resolved total optimizer steps do not match the Stage-3 "
                f"contract: actual={total_steps}, expected={self.expected_total_steps}."
            )
        logger.info(
            "Validated optimizer clock: %d steps/epoch, %d total steps",
            steps_per_epoch,
            total_steps,
        )


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    mixed = None
    hazard_sft = False
    kinematic_sft = False  # ReactTrain fine-tuning: the same SFT recipe with kinematic targets and no held-out sources.
    if cfg.get("lr_warmup_updates", 0) != 0 and not cfg.get("mixed_training_manifest"):
        raise ValueError("lr_warmup_updates requires mixed_training_manifest; original-only schedules are unchanged")
    stop_after_updates = cfg.get("stop_after_updates", 0)
    if type(stop_after_updates) is not int or stop_after_updates < 0:
        raise ValueError("stop_after_updates must be a nonnegative integer")
    if stop_after_updates and not cfg.get("mixed_training_manifest"):
        raise ValueError("stop_after_updates requires mixed_training_manifest; original-only schedules are unchanged")
    if cfg.get("mixed_training_manifest"):
        from navsim.planning.training.recogdrive_mixed_data import load_manifest
        mixed = load_manifest(cfg.mixed_training_manifest)
        from navsim.planning.training.recogdrive_mixed_data import (
            EDITED_ONLY_PROTOCOL, HAZARD_SFT_PROTOCOL, KINEMATIC_SFT_PROTOCOL)
        if mixed["protocol"] == EDITED_ONLY_PROTOCOL:
            if cfg.agent.get("training_reward_mode", "executed") not in ("executed", "minimum_jerk"):
                raise NotImplementedError("Edited-only training supports the executed and minimum_jerk rewards; "
                                          "other revised rewards are not implemented")
            if cfg.get("edited_validation_protocol") != "triplet" or not cfg.get("edited_validation"):
                raise ValueError("Edited-only training requires held-out triplet monitoring")
        hazard_sft = mixed["protocol"] == HAZARD_SFT_PROTOCOL
        kinematic_sft = mixed["protocol"] == KINEMATIC_SFT_PROTOCOL
        if ((hazard_sft or kinematic_sft)
                and (cfg.agent.get("grpo") is not False or cfg.agent.get("per_sample_loss") is not True)):
            raise ValueError("Hazard SFT trains the stage-2 objective: requires agent.grpo=false and "
                             "agent.per_sample_loss=true")
        if not cfg.use_cache_without_dataset:
            raise ValueError("Mixed training requires the prepared cache-only dataset")
        with open_dict(cfg.trainer.params):
            if "max_steps" not in cfg.trainer.params:
                cfg.trainer.params.max_steps = 8325
        if stop_after_updates >= cfg.trainer.params.max_steps:
            raise ValueError("stop_after_updates must be smaller than the optimizer-update limit")
    sft_run = hazard_sft or kinematic_sft
    if sft_run != (cfg.checkpoint_selection == "all_epochs"):
        raise ValueError("checkpoint_selection=all_epochs is reserved for, and required by, hazard SFT manifests")
    if not sft_run and cfg.get("hazard_sft_originals_per_epoch") is not None:
        raise ValueError("hazard_sft_originals_per_epoch requires a hazard SFT manifest")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    logger.info("Planner initialization: %s; BC coefficient: %s", agent.planner_initialization, agent.bc_coeff)
    if agent.action_head.config.velacc_dropout_seed != cfg.seed:
        raise ValueError("Motion dropout must use the resolved training seed")
    logger.info("Velocity/acceleration dropout: %s", agent.action_head.motion_dropout_state())

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningDiT(
        agent=agent,
        training_batch_size=cfg.dataloader.params.batch_size,
        training_sample_kinds=({token: sample["kind"] for token, sample in mixed["samples"].items()
                               if sample["split"] == "train"} if mixed is not None else None),
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    expected_train_samples = cfg.get("expected_train_samples")
    expected_validation_samples = cfg.get("expected_validation_samples")
    train_sample_count = len(train_data)
    validation_sample_count = len(val_data)
    if expected_train_samples is not None and train_sample_count != int(expected_train_samples):
        raise RuntimeError(
            "Training dataset cardinality does not match the Stage-3 contract: "
            f"actual={train_sample_count}, expected={expected_train_samples}."
        )
    if (
        expected_validation_samples is not None
        and validation_sample_count != int(expected_validation_samples)
    ):
        raise RuntimeError(
            "Validation dataset cardinality does not match the Stage-3 contract: "
            f"actual={validation_sample_count}, expected={expected_validation_samples}."
        )

    logger.info("Building Datasets")
    if mixed is not None:
        from navsim.planning.training.recogdrive_mixed_data import validate_dataset
        from navsim.planning.training.recogdrive_mixed_training import training_protocol

        def check_view():
            if sft_run:  # Stage-2 planners have no reward loader; check the view's metric index itself.
                from navsim.common.dataloader import MetricCacheLoader
                metric_paths = MetricCacheLoader(Path(cfg.agent.metric_cache_path)).metric_cache_paths
            else:
                metric_paths = agent.action_head.metric_cache_loader.metric_cache_paths
            validate_dataset(mixed, train_data, val_data, metric_paths, cfg.cache_path, cfg.agent.metric_cache_path)

        rank0_check(rank, check_view)
        sft_protocol = None
        if hazard_sft:
            from navsim.planning.training import recogdrive_hazard_sft as sft
            originals_per_epoch = cfg.get("hazard_sft_originals_per_epoch")
            if originals_per_epoch is None:
                originals_per_epoch = mixed["counts"]["hazardous"]
            train_data = sft.HazardTargetDataset(
                train_data, {t: s["kind"] for t, s in mixed["samples"].items() if s["split"] == "train"},
                sft.load_targets(mixed))
            # The held-out triplet originals replace the 18,179-sample clean L1 pass; the triplet and
            # paired callbacks report the held-out edits after it.
            val_data = sft.TokenDirectoryDataset(
                {t: mixed["original_sources"][t]["feature_path"] for t in mixed["holdout"]["source_tokens"]},
                agent.get_feature_builders(), agent.get_target_builders())
            validation_sample_count = len(val_data)
            sft_sampler = sft.EpochOriginalSampler(train_data, originals_per_epoch, seed=cfg.seed,
                                                   num_replicas=world_size, rank=rank)
            sft_protocol = sft.sampler_protocol(mixed, originals_per_epoch, cfg.seed)
            logger.info("Hazard SFT epoch: %d edits + %d of %d originals (disjoint across %d epochs per pass); "
                        "val/loss on %d held-out originals", len(sft_sampler.edited), originals_per_epoch,
                        len(sft_sampler.originals), sft_sampler.disjoint_epochs, validation_sample_count)
        elif kinematic_sft:
            from navsim.planning.training import recogdrive_kinematic_sft as kinematic
            # Ablation views (training_kinds) may hold no originals: then every epoch is the view's edits.
            train_data, sft_sampler, sft_protocol = kinematic.training_inputs(
                mixed, train_data, cfg.get("hazard_sft_originals_per_epoch"), seed=cfg.seed,
                num_replicas=world_size, rank=rank)
            # No held-out sources: val/loss stays the clean validation split; the paired callback follows it.
            if mixed["counts"]["original"]:
                logger.info("Kinematic SFT epoch: %d edits + %d of %d originals (disjoint across %d epochs per pass); "
                            "val/loss on %d clean validation samples", len(sft_sampler.edited),
                            sft_sampler.originals_per_epoch, len(sft_sampler.originals), sft_sampler.disjoint_epochs,
                            validation_sample_count)
            else:
                logger.info("Kinematic SFT epoch: %d edits, no originals; val/loss on %d clean validation samples",
                            len(sft_sampler.edited), validation_sample_count)
        lightning_module.mixed_training_protocol = training_protocol(mixed, cfg.trainer.params.max_steps,
            batch_size=cfg.dataloader.params.batch_size, world_size=world_size,
            accumulation=cfg.trainer.params.accumulate_grad_batches,
            warmup_updates=cfg.get("lr_warmup_updates", 0), hazard_sft=sft_protocol)
    if agent.action_head.config.grpo_cfg.training_reward_mode == "minimum_jerk" and rank == 0:
        from reactbench.io import write_json
        write_json(Path(cfg.output_dir) / "training_reward_protocol.json", lightning_module._training_reward_protocol())
    if agent.action_head.config.joint_motion_dropout_probability > 0:
        if not isinstance(train_data, CacheOnlyDataset):
            raise ValueError("Joint dropout requires the existing CacheOnlyDataset training inputs")
        if mixed is None:
            agent.configure_joint_images(train_data._valid_cache_paths, cfg.navsim_log_path, cfg.sensor_blobs_path)
        else:
            agent.configure_joint_images(train_data._valid_cache_paths, cfg.navsim_log_path, cfg.sensor_blobs_path,
                                         mixed_manifest=mixed)
    if sft_run:
        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn, **cfg.dataloader.params,
                                      sampler=sft_sampler)
        logger.info("Hazard SFT samples per epoch: %d (all ranks)", sft_sampler.epoch_size)
    else:
        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", train_sample_count)
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", validation_sample_count)

    logger.info("Building Trainer")
    selection_mode = cfg.checkpoint_selection
    callbacks = build_validation_callbacks(cfg, kinematic_sft=kinematic_sft)
    checkpoint_callback = callbacks[0]
    if mixed is not None:
        from navsim.planning.training.recogdrive_mixed_training import MixedUpdateLimit
        callbacks.append(MixedUpdateLimit(stop_after_updates=stop_after_updates))
        if stop_after_updates:
            logger.info("Stopping after %d optimizer updates; LR schedule keeps %d", stop_after_updates,
                        cfg.trainer.params.max_steps)
    expected_steps_per_epoch = cfg.get("expected_optimizer_steps_per_epoch")
    expected_total_steps = cfg.get("expected_total_optimizer_steps")
    if (expected_steps_per_epoch is None) != (expected_total_steps is None):
        raise RuntimeError(
            "Both expected_optimizer_steps_per_epoch and "
            "expected_total_optimizer_steps must be configured together."
        )
    if expected_steps_per_epoch is not None:
        callbacks.append(
            OptimizerClockValidationCallback(
                expected_steps_per_epoch=int(expected_steps_per_epoch),
                expected_total_steps=int(expected_total_steps),
            )
        )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    resume_checkpoint_path = cfg.get("resume_checkpoint_path")
    if resume_checkpoint_path:
        resume_checkpoint_path = Path(str(resume_checkpoint_path)).expanduser().resolve()
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_checkpoint_path}"
            )
        logger.info(
            "Resuming complete Lightning state from checkpoint: %s",
            resume_checkpoint_path,
        )
    else:
        resume_checkpoint_path = None

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_checkpoint_path,
        # Stage-3 resume checkpoints are trusted local Lightning artifacts and
        # require optimizer, scheduler, and callback state in addition to weights.
        weights_only=False,
    )

    if trainer.is_global_zero:
        logger.info("Best %s checkpoint: %s", selection_mode, checkpoint_callback.best_model_path or "none")
        if selection_mode == "training_reward" and checkpoint_callback.best_model_score is not None:
            logger.info("Best training reward: %.6f", checkpoint_callback.best_model_score.item())
        logger.info("Latest checkpoint: %s", checkpoint_callback.last_model_path or "none")


if __name__ == "__main__":
    main()
