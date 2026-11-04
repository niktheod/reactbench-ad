"""Independent edited monitoring and the optional legacy validation selector."""
from __future__ import annotations

import logging
from pathlib import Path
import re

from pytorch_lightning.callbacks import Callback, Checkpoint
import torch

from navsim.planning.training.recogdrive_checkpoint_selection import CheckpointSelector, aggregate, read_json, write_json
from navsim.planning.training.recogdrive_validation import evaluate_distributed, live_checkpoint_identity

LOGGER = logging.getLogger(__name__)


class RecogDriveEditedValidationCallback(Callback):
    """Report edited safety after clean L1 validation, without selecting weights."""
    def __init__(self, evaluator, output_dir):
        self.evaluator = evaluator
        self.directory = Path(output_dir).resolve() / "validation" / "edited"
        self.identity = evaluator.artifact_context(edited_only=True)["input_identity"]
        self.last_reported_epoch = -1

    def state_dict(self):
        return {"identity": self.identity, "last_reported_epoch": self.last_reported_epoch}

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Edited reporting input/protocol mismatch on resume")
        self.last_reported_epoch = state["last_reported_epoch"]

    def on_fit_start(self, trainer, pl_module):
        if trainer.check_val_every_n_epoch != 1 or trainer.val_check_interval != 1.0 or not trainer.limit_val_batches:
            raise ValueError("Edited reporting requires a validation phase at the end of every epoch")

        def initialize():
            path = self.directory / "config.json"
            if path.exists() and read_json(path)["input_identity"] != self.identity:
                raise ValueError("Output directory belongs to different edited reporting inputs")
            write_json(path, {"config": self.evaluator.config, "purpose": "monitoring_only",
                              **self.evaluator.artifact_context(edited_only=True)})

        RecogDriveValidationCallback._rank_zero_action(trainer, initialize)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch, step = int(trainer.current_epoch), int(trainer.global_step)
        if trainer.sanity_checking or epoch <= self.last_reported_epoch:
            return
        result = evaluate_distributed(self.evaluator, pl_module.agent, trainer.global_rank,
                                      trainer.world_size, edited_only=True)
        # Each component reports its own coverage, including a known NC value
        # when DAC is unavailable. The product still requires both components.
        components = {}
        for name in ("nc_all", "dac"):
            records = []
            for record in result["records"]:
                value = record.get("components", {}).get(name)
                valid = value in (0, 1)
                records.append({"sample_id": record["sample_id"],
                                "score": 100. * value if valid else None,
                                "error": None if valid else record.get("error") or f"{name} unavailable"})
            components[name] = aggregate(records, [s["sample_id"] for s in self.evaluator.samples
                                                    if s["kind"] == "edited"])
        result.update(epoch=epoch, step=step, purpose="monitoring_only", edited_components=components)
        def publish():
            result["checkpoint_identity"] = live_checkpoint_identity(pl_module.agent, epoch, step)
            write_json(self.directory / f"epoch={epoch}-step={step}.json", result)

        RecogDriveValidationCallback._rank_zero_action(trainer, publish)
        self.last_reported_epoch = epoch
        if "paired" in result:
            paired = result["paired"]
            for key in ("original_mean", "edited_mean", "reaction_gain_pp", "rescued", "harmed",
                        "rescue_fraction", "harm_fraction", "original_failures", "rescue_rate",
                        "expected", "attempted", "complete", "failed", "diagnostics_complete"):
                value = paired[key]
                pl_module.log(f"edited/paired_{key}", float(value) if value is not None else float("nan"),
                              on_step=False, on_epoch=True, sync_dist=False, batch_size=1)
            for side, metrics in paired["trajectory_diagnostics"].items():
                for key, value in metrics.items():
                    pl_module.log(f"edited/{side}_{key}", float(value) if value is not None else float("nan"),
                                  on_step=False, on_epoch=True, sync_dist=False, batch_size=1)
            if trainer.is_global_zero:
                LOGGER.info("Epoch %d paired reaction: gain=%s pp, rescued=%d, harmed=%d, complete=%d/%d",
                            epoch, paired["reaction_gain_pp"], paired["rescued"], paired["harmed"],
                            paired["complete"], paired["expected"])
        for name, scores in {"nc_dac": result["edited"], **components}.items():
            values = {name: scores["mean"] if scores["mean"] is not None else float("nan"),
                      f"{name}_successful": scores["successful"], f"{name}_failed": scores["failed"],
                      f"{name}_expected": scores["expected"]}
            for key, value in values.items():
                # Scores were already aggregated globally by sample count.
                pl_module.log(f"edited/{key}", float(value), on_step=False, on_epoch=True,
                              sync_dist=False, batch_size=1)
        if trainer.is_global_zero:
            scores = result["edited"]
            LOGGER.info("Epoch %d edited monitoring: NC-all=%s, DAC=%s, NC-all x DAC=%s (0-100), "
                        "coverage=%d/%d, failures=%d; checkpoint selection uses training reward",
                        epoch, components["nc_all"]["mean"], components["dac"]["mean"], scores["mean"],
                        scores["successful"], scores["expected"], scores["failed"])


class RecogDriveValidationCallback(Checkpoint):
    def __init__(self, evaluator, baseline, selection_identity, output_dir):
        self.evaluator = evaluator
        self.baseline = baseline
        self.selector = CheckpointSelector(selection_identity, baseline["clean"])
        self.directory = Path(output_dir).resolve() / "validation"
        self.checkpoint_directory = Path(output_dir).resolve() / "checkpoints"
        self.last_model_path = str(self.checkpoint_directory / "last.ckpt")
        self.best_model_path = ""
        self.pending = None
        self.last_saved_epoch = -1

    @property
    def state_key(self):
        # A constant key ensures incompatible resumes are checked, never ignored.
        return "RecogDriveValidationCallback/v1"

    def state_dict(self):
        return {"selector": self.selector.state_dict(), "last_saved_epoch": self.last_saved_epoch,
                "checkpoint_directory": str(self.checkpoint_directory)}

    def _publication_context(self):
        return {**self.evaluator.artifact_context(), "seed": self.evaluator.config["inference"]["seed"],
                "baseline_checkpoint": self.evaluator.config.get("baseline_checkpoint")}

    def load_state_dict(self, state):
        self.selector.load_state_dict(state["selector"])
        if state["checkpoint_directory"] != str(self.checkpoint_directory):
            raise ValueError("Resume validation selection in the same output directory to preserve retained weights")
        self.last_saved_epoch = state["last_saved_epoch"]
        self.best_model_path = self.selector.best["checkpoint"] if self.selector.best else ""

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if self.state_key not in checkpoint.get("callbacks", {}):
            raise ValueError("Resume checkpoint has no validation ranking state; initialize a new run from weights instead")

    @staticmethod
    def _rank_zero_action(trainer, action):
        error = None
        if trainer.is_global_zero:
            try:
                action()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = trainer.strategy.broadcast(error)
        if error:
            raise RuntimeError(f"Validation artifact publication failed: {error}")

    def on_fit_start(self, trainer, pl_module):
        if trainer.check_val_every_n_epoch != 1 or trainer.val_check_interval != 1.0 or not trainer.limit_val_batches:
            raise ValueError("Validation selection requires a validation phase at the end of every epoch")

        def initialize():
            self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
            self.directory.mkdir(parents=True, exist_ok=True)
            selected = self.directory / "selected_checkpoint.json"
            if selected.exists():
                published = read_json(selected)
                if published["identity"] != self.selector.contract["identity"]:
                    raise ValueError("Output directory belongs to a different selection identity")
                if published["last_scored_epoch"] is not None and published["last_scored_epoch"] > self.last_saved_epoch:
                    raise ValueError("Resume from last.ckpt; this checkpoint predates published selection decisions")
            if not self.selector.history and any(self.checkpoint_directory.glob("*.ckpt")):
                raise ValueError("Output already contains checkpoints; resume last.ckpt or use a new output directory")
            for record in self.selector.top:
                if not Path(record["checkpoint"]).is_file():
                    raise FileNotFoundError(f"Retained checkpoint missing on resume: {record['checkpoint']}")
            # Recover a crash between candidate/last saves and pruning. Delete only
            # history-owned weights or a newer candidate carrying our exact state.
            retained = {r["checkpoint"] for r in self.selector.top}
            owned = {r["checkpoint"] for r in self.selector.history}
            for path in self.checkpoint_directory.glob("epoch=*-step=*.ckpt"):
                if str(path) in retained:
                    continue
                if str(path) not in owned:
                    match = re.fullmatch(r"epoch=(\d+)-step=(\d+)\.ckpt", path.name)
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    state = saved.get("callbacks", {}).get(self.state_key, {})
                    if (not match or state.get("selector", {}).get("contract") != self.selector.contract
                            or int(match[1]) != saved["epoch"] or int(match[2]) != saved["global_step"]
                            or saved["epoch"] <= self.last_saved_epoch):
                        raise ValueError(f"Unrecognized checkpoint in validation output: {path}")
                    del saved
                path.unlink()
            if self.selector.history:
                self.selector.publish(self.directory, self._publication_context(), self.last_model_path)
            write_json(self.directory / "baseline.json", self.baseline)
            write_json(self.directory / "config.json", {"config": self.evaluator.config,
                       **self.evaluator.artifact_context(),
                       "selection_identity": self.selector.contract["identity"]})
            reference = self.baseline["clean"]
            LOGGER.info("Baseline clean PDMS=%s, coverage=%d/%d%s", reference["mean"],
                        reference["successful"], reference["expected"],
                        " (PARTIAL COVERAGE)" if reference["partial_coverage"] else "")

        self._rank_zero_action(trainer, initialize)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if epoch <= self.last_saved_epoch:
            return
        result = evaluate_distributed(self.evaluator, pl_module.agent, trainer.global_rank, trainer.world_size)
        result.update(epoch=epoch, step=int(trainer.global_step),
                      selection_identity=self.selector.contract["identity"])
        self.pending = result
        if trainer.is_global_zero:
            for kind in ("clean", "edited"):
                scores = result[kind]
                LOGGER.info("Epoch %d validation %s: mean=%s, coverage=%d/%d, failures=%d%s",
                            epoch, kind, scores["mean"], scores["successful"], scores["expected"],
                            scores["failed"], " (PARTIAL COVERAGE)" if scores["partial_coverage"] else "")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch, step = int(trainer.current_epoch), int(trainer.global_step)
        if epoch <= self.last_saved_epoch:
            return
        if self.pending is None or (self.pending["epoch"], self.pending["step"]) != (epoch, step):
            # Never fall back to a stale previous epoch's score.
            trainer.save_checkpoint(self.last_model_path, weights_only=False)
            raise RuntimeError("No finalized current-epoch validation scores; saved latest state without ranking")
        path = str(self.checkpoint_directory / f"epoch={epoch}-step={step}.ckpt")
        reward = trainer.callback_metrics.get("train/reward_epoch")
        reward = float(reward) if reward is not None else None
        record, evicted = self.selector.add(self.pending, path, epoch, step, reward)
        self.best_model_path = self.selector.best["checkpoint"] if self.selector.best else ""
        self.last_saved_epoch = epoch
        report_error = None
        try:
            def publish_report():
                self.pending["checkpoint_identity"] = live_checkpoint_identity(pl_module.agent, epoch, step)
                write_json(self.directory / f"epoch={epoch}-step={step}.json", self.pending)

            self._rank_zero_action(trainer, publish_report)
        except RuntimeError as exc:
            report_error = exc
        # Every rank participates in Lightning's checkpoint protocol. Saving at
        # train epoch end matches Lightning's completed-epoch resume bookkeeping.
        if record in self.selector.top:
            trainer.save_checkpoint(path, weights_only=False)
        trainer.save_checkpoint(self.last_model_path, weights_only=False)
        if report_error is not None:
            raise report_error

        def publish():
            for old in evicted:
                old_path = Path(old)
                if old_path.parent != self.checkpoint_directory:
                    raise ValueError("Refusing to prune a checkpoint outside this selection run")
                old_path.unlink(missing_ok=True)
            self.selector.publish(self.directory, self._publication_context(), self.last_model_path)
            LOGGER.info("Best eligible validation checkpoint: %s; latest checkpoint: %s; training reward: %s",
                        self.best_model_path or "none", self.last_model_path, reward)
            rewards = [r["training_reward"] for r in self.selector.history if r["training_reward"] is not None]
            LOGGER.info("Best observed training reward (diagnostic only): %s", max(rewards) if rewards else "unavailable")

        self._rank_zero_action(trainer, publish)
        self.pending = None


class RecogDriveTripletValidationCallback(RecogDriveEditedValidationCallback):
    """Raw held-out reaction monitoring, independent of reward checkpoint ranking."""
    def __init__(self, evaluator, output_dir):
        super().__init__(evaluator, output_dir)
        self.directory = Path(output_dir).resolve() / "validation" / "triplets"

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if self.state_key not in checkpoint.get("callbacks", {}):
            raise ValueError("Resume checkpoint has no triplet reporting state; initialize from weights instead")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch, step = int(trainer.current_epoch), int(trainer.global_step)
        if trainer.sanity_checking or epoch <= self.last_reported_epoch:
            return
        result = evaluate_distributed(self.evaluator, pl_module.agent, trainer.global_rank,
                                      trainer.world_size, edited_only=True)
        result.update(epoch=epoch, step=step, purpose="monitoring_only")
        def publish():
            result["checkpoint_identity"] = live_checkpoint_identity(pl_module.agent, epoch, step)
            write_json(self.directory / f"epoch={epoch}-step={step}.json", result)
        RecogDriveValidationCallback._rank_zero_action(trainer, publish)
        self.last_reported_epoch = epoch
        summary = result["triplets"]
        def log(key, value):
            pl_module.log(f"triplets/{key}", float(value) if value is not None else float("nan"),
                          on_step=False, on_epoch=True, sync_dist=False, batch_size=1)
        for key in ("hazard_gain_pp", "benign_mean", "hazardous_mean", "rescued", "harmed", "rescue_fraction",
                    "harm_fraction", "benign_failures", "rescue_rate", "original_gain_pp", "original_pairs",
                    "expected", "attempted", "complete", "failed", "complete_pairs", "diagnostics_complete",
                    "early_speed_reduction_mps", "benign_emergency_only", "source_count", "log_count"):
            log(key, summary[key])
        for world, components in summary["components"].items():
            for component, coverage in components.items():
                log(f"{world}_{component}", coverage["mean"])
                log(f"{world}_{component}_complete", coverage["complete"])
        for side, metrics in summary["predicted_trajectories"].items():
            for key, value in metrics.items():
                log(f"{side}_{key}", value["mean"])
        for key, value in summary["benign_minus_original"].items():
            log(f"benign_minus_original_{key}", value["mean"])
        # Hazard SFT runs only: L1 to each input's training target, V0 and stock PDMS in its own world.
        cohorts = result.get("hazard_sft", {}).get("cohorts", {})
        for cohort, values in cohorts.items():
            for key in ("l1", "reward", "contact_fraction", "contact_speed_mps", "infeasible_fit_fraction",
                        "nc", "dac", "ep", "comfort", "pdms", "pdms_nc", "pdms_dac", "pdms_ep", "pdms_ttc",
                        "pdms_comfort", "l1_complete", "reward_complete", "pdms_complete", "failed"):
                log(f"sft_{cohort}_{key}", values[key])
        if trainer.is_global_zero:
            LOGGER.info("Epoch %d raw triplets: hazard gain=%s pp; H/B=%s/%s; rescue/harm=%d/%d; complete=%d/%d",
                        epoch, summary["hazard_gain_pp"], summary["hazardous_mean"], summary["benign_mean"],
                        summary["rescued"], summary["harmed"], summary["complete"], summary["expected"])
            for cohort, values in cohorts.items():
                LOGGER.info("Epoch %d hazard SFT %s: L1=%s, V0=%s, PDMS=%s, V0 contact=%s "
                            "(L1/V0/PDMS coverage %d/%d/%d of %d)",
                            epoch, cohort, values["l1"], values["reward"], values["pdms"], values["contact_fraction"],
                            values["l1_complete"], values["reward_complete"], values["pdms_complete"],
                            values["expected"])
