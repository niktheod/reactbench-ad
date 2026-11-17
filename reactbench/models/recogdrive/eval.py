"""Run ReCogDrive on a ReactNAV variant (original or edited front-camera image).

    python -m reactbench.models.recogdrive.eval VARIANT_DIR --original --config rl   # cache original predictions
    python -m reactbench.models.recogdrive.eval VARIANT_DIR --config rl              # edited predictions + scoring

``--config`` is one of the files in ``configs/`` (``vlm_sft``, ``il``, ``rl``, ``il_rt``) or a path.
Predictions go to ``<outputs>/predictions/<name>/reactnav/<variant>``, where ``name`` comes from the configuration.
The original and edited rollouts of a scene share all diffusion noise; the VLM (``vlm_sft``) decodes greedily
and saves its answers in ``responses/``.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .. import runner
from ..seeds import MAX_SEED, paired_seed

CONFIG_DIR = Path(__file__).with_name("configs")
VLM_TARGET = "reactbench.models.recogdrive.vlm_agent.ReCogDriveVLMAgent"
SEED_NAMESPACE = "recogdrive-diffusion"  # per-token seeds: models/seeds.py


def compute_trajectory_with_seed(agent, agent_input, seed):
    """Run one stochastic rollout without advancing surrounding Torch RNGs."""
    seed = int(seed)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and 2**63 - 1, got {seed}")
    device = torch.device(agent.device)
    rng_devices = []
    if device.type == "cuda":
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        return agent.compute_trajectory(agent_input)


def load_config(value):
    """A configuration name or path, checked: (path, configuration)."""
    path = runner.config_path(value, CONFIG_DIR)
    config = runner.read_config(path)
    if "name" not in config:
        raise ValueError(f"Expected a 'name' in {path}")
    backend = config.get("backend", "diffusion")
    if backend not in ("diffusion", "vlm") or (backend == "vlm") != (config.get("_target_") == VLM_TARGET):
        raise ValueError(f"backend={backend} does not match _target_={config.get('_target_')} in {path}")
    return path, config


def agent_config(config):
    """Only the configuration accepted by the agent constructor."""
    return {k: v for k, v in config.items() if k not in ("name", "backend", "diffusion_seed_base")}


def make_input(annotation, image_path):
    from .inputs import make_recogdrive_agent_input
    return make_recogdrive_agent_input(
        image_path=image_path,
        velocities=annotation["velocity"],
        accelerations=annotation["acceleration"],
        history_trajectories=annotation["past_trajectory"],
        navigation_command=annotation["navigation_command"],
    )


class ReCogDrive(runner.Runner):
    model = "recogdrive"
    config_dir = CONFIG_DIR
    default_config = "rl"

    def setup(self):
        from hydra.utils import instantiate
        load_config(self.config_path)
        self.vlm = self.config.get("backend", "diffusion") == "vlm"
        self.seed_base = None if self.vlm else int(self.config.get("diffusion_seed_base", 42))
        self.agent = instantiate(agent_config(self.config))
        self.provenance = self.agent.provenance() if self.vlm else None
        self.interval_length_s = float(self.config["trajectory_sampling"]["interval_length"])

    def cache_fields(self):
        if self.vlm:
            return {"evaluator": "recogdrive-vlm-navsim", "inference_options": self.provenance,
                    "seed_name": "unused_greedy", "seed_base": 0}
        return {"evaluator": "recogdrive-navsim", "seed_name": "diffusion_seed_base", "seed_base": self.seed_base}

    def seeds(self, token):
        return {} if self.vlm else {"diffusion_seed": paired_seed(SEED_NAMESPACE, self.seed_base, token)}

    def predict(self, token, seeds):
        image = self.loader.original_images(token)["front_camera"][-1] if self.original else \
            self.loader.edited_image(token)
        agent_input = make_input(self.loader.annotation(token), image)
        if not self.vlm:
            return compute_trajectory_with_seed(self.agent, agent_input, seeds["diffusion_seed"])
        self.agent.last_generation = None
        try:
            return self.agent.compute_trajectory(agent_input)
        finally:
            # The answer is kept even when it cannot be parsed into a trajectory.
            if self.agent.last_generation is not None:
                runner.save_json(self.result_dir / "responses" / f"{token}.json", self.agent.last_generation)

    def save_extras(self, token, prediction):
        return {"response_path": str(self.result_dir / "responses" / f"{token}.json")} if self.vlm else {}

    def summary_extras(self):
        return {"inference_provenance": self.provenance} if self.vlm else {"diffusion_seed_base": self.seed_base}


if __name__ == "__main__":
    raise SystemExit(ReCogDrive.main(__doc__))
