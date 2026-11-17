"""Run SimWAM on a ReactNAV variant (current front-camera frame and ego state; action-only inference).

    python -m reactbench.models.simwam.eval VARIANT_DIR --original --config rl   # cache original predictions
    python -m reactbench.models.simwam.eval VARIANT_DIR --config rl              # edited predictions + scoring

``--config`` is ``rl`` (FlowGRPO, ``simwam``), ``il`` (``simwam_il``) or a path. The original and edited rollouts
of a scene share their action noise, and both must run on the same GPU architecture (part of the cache signature).
"""
from __future__ import annotations

from pathlib import Path

from ...io import digest, file_digest
from .. import runner
from .agent import SimWAMAgent
from .navsim_loader import NAVSIMVariant

HERE = Path(__file__).parent
AGENT_TARGET = "reactbench.models.simwam.agent.SimWAMAgent"
# Sources that shape the inputs and the inference, hashed into the cache signature.
ADAPTER_SOURCES = (HERE / "eval.py", HERE / "agent.py", HERE / "navsim_loader.py",
                   HERE.parents[1] / "datasets" / "navsim.py")


def hardware_class(device):
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.device(device))


class SimWAM(runner.Runner):
    model = "simwam"
    config_dir = HERE / "configs"
    default_config = "rl"

    def open_variant(self):
        return NAVSIMVariant(self.variant_dir)

    def setup(self):
        if self.config.get("_target_") != AGENT_TARGET:
            raise ValueError(f"SimWAM configuration must have _target_: {AGENT_TARGET}")
        self.agent = SimWAMAgent(**{key: value for key, value in self.config.items() if key != "_target_"})
        self.interval_length_s = self.agent.interval_length_s
        self.prepared = self.agent.prepare()
        self.hardware = hardware_class(self.agent.device)

    def identity(self):
        return {**self.agent.identity(),
                "adapter_source_sha256": digest({path.name: file_digest(path) for path in ADAPTER_SOURCES})}

    def cache_fields(self):
        # Pairs must share one GPU architecture so that only the image differs between members.
        return {"evaluator": "simwam-navsim", "seed_name": "action_noise_seed_base", "seed_base": self.agent.seed,
                "inference_options": {"identity": self.identity(), "input_profile": self.loader.input_profile,
                                      "prompt_embedding_sha256": self.prepared["prompt_embedding_sha256"],
                                      "data_settings": self.prepared["data_settings"],
                                      "hardware_class": self.hardware}}

    def load(self):
        self.agent.load()

    def seeds(self, token):
        return {"action_noise_seed": self.agent.rollout_seed(token)}

    def check_edited_inputs(self, token):
        return self.loader.check_edited_inputs(token)

    def model_input(self, token):
        return self.loader.model_input(token, self.loader.image_path(token, original=self.original))

    def predict(self, token, seeds):
        return self.agent.compute_trajectory(self.model_input(token), rollout_seed=seeds["action_noise_seed"])

    def arrays(self, prediction):
        return {"ego_origin": "rear_axle"}

    def save_extras(self, token, prediction):
        model_input = self.model_input(token)
        return {"image_path": str(model_input["image_path"]), "ego_state": model_input["state"].tolist(),
                "current_timestamp_us": model_input["current_timestamp_us"]}

    def summary_extras(self):
        return {"checkpoint_path": self.config["checkpoint_path"], "hardware_class": self.hardware,
                "model_load": {key: self.agent.load_report.get(key) for key in
                               ("checkpoint_step", "missing_keys", "model_paths", "prompt_embedding")},
                "trajectory_convention": "8 x/y/heading poses at 0.5 s, current rear-axle frame, current pose excluded",
                "input_contract": self.loader.input_profile}


if __name__ == "__main__":
    raise SystemExit(SimWAM.main(__doc__))
