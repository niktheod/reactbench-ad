from typing import Tuple
from pathlib import Path
import datetime
import logging
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
from navsim.planning.training.agent_lightning_module import AgentLightningDiT
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
    """Keep last.ckpt current after every training epoch."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_skip_saving_checkpoint(trainer) or not self._should_save_on_train_epoch_end(trainer):
            return
        super().on_train_epoch_end(trainer, pl_module)
        # Lightning 2.6.5 skips save_last when no top-k checkpoint was saved.
        if self.save_last and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


def build_checkpoint_callbacks(cfg):
    """A checkpoint after every epoch, plus last.ckpt (the paper's ReactTrain checkpoint is epoch=2-step=276.ckpt)."""
    output = Path(cfg.output_dir).expanduser().resolve()
    return [Stage3ModelCheckpoint(dirpath=output / "checkpoints", monitor=None, save_top_k=-1, save_last=True,
                                  every_n_epochs=1, save_on_train_epoch_end=True)]


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
    kinematic_sft = False  # ReactTrain fine-tuning (a kinematic-SFT training view); otherwise upstream RL training.
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
        from navsim.planning.training.recogdrive_mixed_data import KINEMATIC_SFT_PROTOCOL
        if mixed["protocol"] != KINEMATIC_SFT_PROTOCOL:
            raise ValueError(f"Only kinematic-SFT training views are supported, got {mixed['protocol']}")
        kinematic_sft = True
        if cfg.agent.get("grpo") is not False or cfg.agent.get("per_sample_loss") is not True:
            raise ValueError("Kinematic SFT trains the stage-2 objective: requires agent.grpo=false and "
                             "agent.per_sample_loss=true")
        if not cfg.use_cache_without_dataset:
            raise ValueError("Mixed training requires the prepared cache-only dataset")
        with open_dict(cfg.trainer.params):
            if "max_steps" not in cfg.trainer.params:
                cfg.trainer.params.max_steps = 8325
        if stop_after_updates >= cfg.trainer.params.max_steps:
            raise ValueError("stop_after_updates must be smaller than the optimizer-update limit")
    if not kinematic_sft and cfg.get("kinematic_sft_originals_per_epoch") is not None:
        raise ValueError("kinematic_sft_originals_per_epoch requires a kinematic-SFT training view")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    if agent.grpo:
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

    train_sample_count = len(train_data)
    validation_sample_count = len(val_data)

    logger.info("Building Datasets")
    if mixed is not None:
        from navsim.planning.training.recogdrive_mixed_data import validate_dataset
        from navsim.planning.training.recogdrive_mixed_training import training_protocol

        def check_view():
            # Stage-2 planners have no reward loader; check the view's metric index itself.
            from navsim.common.dataloader import MetricCacheLoader
            metric_paths = MetricCacheLoader(Path(cfg.agent.metric_cache_path)).metric_cache_paths
            validate_dataset(mixed, train_data, val_data, metric_paths, cfg.cache_path, cfg.agent.metric_cache_path)

        rank0_check(rank, check_view)
        from navsim.planning.training import recogdrive_kinematic_sft as kinematic
        train_data, sft_sampler, sft_protocol = kinematic.training_inputs(
            mixed, train_data, cfg.get("kinematic_sft_originals_per_epoch"), seed=cfg.seed,
            num_replicas=world_size, rank=rank)
        # val/loss stays the clean validation split.
        logger.info("Kinematic SFT epoch: %d edits + %d of %d originals (disjoint across %d epochs per pass); "
                    "val/loss on %d clean validation samples", len(sft_sampler.edited),
                    sft_sampler.originals_per_epoch, len(sft_sampler.originals), sft_sampler.disjoint_epochs,
                    validation_sample_count)
        lightning_module.mixed_training_protocol = training_protocol(mixed, cfg.trainer.params.max_steps,
            batch_size=cfg.dataloader.params.batch_size, world_size=world_size,
            accumulation=cfg.trainer.params.accumulate_grad_batches,
            warmup_updates=cfg.get("lr_warmup_updates", 0), kinematic_sft=sft_protocol)
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
    if kinematic_sft:
        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn, **cfg.dataloader.params,
                                      sampler=sft_sampler)
        logger.info("Kinematic SFT samples per epoch: %d (all ranks)", sft_sampler.epoch_size)
    else:
        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", train_sample_count)
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", validation_sample_count)

    logger.info("Building Trainer")
    callbacks = build_checkpoint_callbacks(cfg)
    checkpoint_callback = callbacks[0]
    if mixed is not None:
        from navsim.planning.training.recogdrive_mixed_training import MixedUpdateLimit
        callbacks.append(MixedUpdateLimit(stop_after_updates=stop_after_updates))
        if stop_after_updates:
            logger.info("Stopping after %d optimizer updates; LR schedule keeps %d", stop_after_updates,
                        cfg.trainer.params.max_steps)
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
        logger.info("Latest checkpoint: %s", checkpoint_callback.last_model_path or "none")


if __name__ == "__main__":
    main()
