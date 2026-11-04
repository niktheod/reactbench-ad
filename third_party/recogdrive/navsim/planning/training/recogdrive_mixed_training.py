"""Update-count schedule of mixed-data training (ReactTrain fine-tuning): warm-up + per-update cosine LR,
the recorded training protocol and the final epoch at the update limit."""
import math

import pytorch_lightning as pl


def training_protocol(manifest, max_updates, *, batch_size, world_size, accumulation, warmup_updates=0,
                      kinematic_sft=None):
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0:
        raise ValueError("Mixed training requires a positive integer optimizer-update limit")
    if isinstance(warmup_updates, bool) or not isinstance(warmup_updates, int) or not 0 <= warmup_updates < max_updates:
        raise ValueError("LR warmup must be an integer number of updates in [0, max_updates)")
    protocol = {"version": 1, "dataset_identity": manifest["artifact_sha256"],
                "max_updates": max_updates, "scheduler": "cosine-per-optimizer-update-v1",
                "batch_size": batch_size, "world_size": world_size, "accumulation": accumulation}
    if warmup_updates:  # absent when disabled, so earlier protocols stay unchanged
        protocol.update(scheduler="linear-warmup-cosine-per-optimizer-update-v1", warmup_updates=warmup_updates)
    if kinematic_sft is not None:  # sampled population, targets and loss
        protocol["kinematic_sft"] = kinematic_sft
    return protocol


def warmup_cosine_factor(update, max_updates, warmup_updates):
    """LR multiplier for 0-based ``update``: linear (u + 1) / W to the peak, then
    cosine from the peak at update W to 0 at ``max_updates``."""
    if update < warmup_updates:
        return (update + 1) / warmup_updates
    return 0.5 * (1 + math.cos(math.pi * min(update - warmup_updates, max_updates - warmup_updates)
                               / (max_updates - warmup_updates)))


class MixedUpdateLimit(pl.Callback):
    """Make the final partial epoch run normal validation/checkpoint hooks.

    ``stop_after_updates`` (opt-in, not part of the protocol) ends the run
    earlier through the same final pass, leaving max_steps and therefore the
    LR schedule unchanged; resuming without it continues to max_steps.
    """

    def __init__(self, stop_after_updates=0):
        if isinstance(stop_after_updates, bool) or not isinstance(stop_after_updates, int) or stop_after_updates < 0:
            raise ValueError("stop_after_updates must be a nonnegative integer")
        self.stop_after_updates = stop_after_updates

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Lightning's max_steps alone ends the epoch without necessarily entering
        # its end-of-epoch validation loop. should_stop requests that final pass.
        if trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps:
            trainer.should_stop = True
        if self.stop_after_updates and trainer.global_step >= self.stop_after_updates:
            trainer.should_stop = True

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Lightning 2.6's max_steps loop resumes an epoch-end checkpoint as an
        # unfinished iteration loop, otherwise replaying an empty epoch. Store
        # the completed boundary for mixed runs only. Keep optimizer totals and
        # partial-epoch checkpoints untouched. The update scheduler has already
        # stepped, so no pending epoch scheduler work needs to be replayed.
        loop = checkpoint["loops"]["fit_loop"]
        batch = loop["epoch_loop.batch_progress"]
        progress = loop["epoch_progress"]
        current = progress["current"]
        epoch_end_hook = current["processed"] == current["started"] and current["processed"] == current["completed"] + 1
        if batch["is_last_batch"] and epoch_end_hook:
            for scope in ("current", "total"):
                progress[scope]["completed"] = progress[scope]["processed"]
            batch["current"] = {key: 0 for key in batch["current"]}
            batch["is_last_batch"] = False
        elif self.stop_after_updates and trainer.global_step >= self.stop_after_updates and epoch_end_hook:
            # An early stop ends a partial epoch through the epoch-end hooks. Store it
            # as mid-epoch progress, so a resume finishes that epoch rather than
            # counting it as complete and losing the rest of its update budget.
            for scope in ("current", "total"):
                progress[scope]["processed"] -= 1

    def on_train_start(self, trainer, pl_module):
        updates = math.ceil(trainer.num_training_batches / trainer.accumulate_grad_batches)
        if trainer.max_epochs < 0 or trainer.max_epochs * updates < trainer.max_steps:
            raise ValueError("Mixed-data max_epochs must allow reaching the optimizer-update limit")
        if trainer.check_val_every_n_epoch != 1 or trainer.val_check_interval != 1.0 or not trainer.limit_val_batches:
            raise ValueError("Mixed training requires clean validation at each epoch and at the update limit")
        if self.stop_after_updates and not trainer.global_step < self.stop_after_updates < trainer.max_steps:
            raise ValueError("stop_after_updates must lie after the resumed update and before the update limit")
