"""Run AutoVLA on a ReactNAV run (three front cameras, four history frames).

    python -m reactbench.models.autovla.eval RUN_DIR --original     # cache original predictions
    python -m reactbench.models.autovla.eval RUN_DIR                # edited predictions + scoring

Inputs are AutoVLA's own per-token no-CoT samples (``autovla_inputs`` in configs/paths.yaml, made with AutoVLA's
preprocessing); for edited runs the front, front-left and front-right images of all four history frames are
replaced by the edited ones. Its text output goes to ``cot_outputs/``. Predictions go to
``<outputs>/predictions/autovla/reactnav/<run>``.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ... import paths
from ...io import read_json
from .. import runner
from ..seeds import paired_seed

AUTOVLA_REPO_ROOT = Path(os.environ.get("AUTOVLA_REPO_ROOT", paths.REPO_ROOT / "third_party" / "AutoVLA"))
# AutoVLA's input field of each camera.
CAMERA_INPUT_KEYS = {"CAM_F0": "front_camera_paths", "CAM_L0": "front_left_camera_paths",
                     "CAM_R0": "front_right_camera_paths"}
SEED_NAMESPACE = "autovla-generation"  # per-token seeds: models/seeds.py


def edited_input(agent_input, loader, token):
    """AutoVLA's input with the four frames of the three front cameras replaced by the edited ones."""
    agent_input = copy.deepcopy(agent_input)
    for camera, key in CAMERA_INPUT_KEYS.items():
        if not isinstance(agent_input.get(key), list) or len(agent_input[key]) != 4:
            raise ValueError(f"Expected four paths in {key!r} for token {token}")
        edited = [str(loader.edited_image(token, camera, frame)) for frame in range(4)]
        agent_input[key] = edited
        for side, alias in (("CAM_L0", "left_camera_paths"), ("CAM_R0", "right_camera_paths")):
            if camera == side and alias in agent_input:
                agent_input[alias] = list(edited)
    return agent_input


def split(prediction):
    """(trajectory, text): AutoVLA returns both."""
    return prediction if isinstance(prediction, tuple) and len(prediction) == 2 else (prediction, None)


def cot_text(cot):
    if cot is None:
        return ""
    if isinstance(cot, np.ndarray):
        cot = cot.tolist()
    if isinstance(cot, (list, tuple)):
        return "\n".join(str(item).strip() for item in cot)
    return str(cot).strip()


class AutoVLA(runner.Runner):
    model = "autovla"
    config_dir = Path(__file__).parent
    default_config = "config"

    def setup(self):
        sys.path.insert(0, str(AUTOVLA_REPO_ROOT))
        sys.path.insert(0, str(AUTOVLA_REPO_ROOT / "navsim"))
        from hydra.utils import instantiate
        self.inputs = paths.get("autovla_inputs")
        if not self.inputs.is_dir():
            raise NotADirectoryError(f"AutoVLA token inputs do not exist: {self.inputs}")
        self.config.pop("metric_horizon_s", None)  # every metric uses the common four seconds
        self.seed_base = int(self.config.pop("generation_seed_base", 42))
        self.agent = instantiate(self.config)
        self.interval_length_s = float(self.config["trajectory_sampling"]["interval_length"])

    def cache_fields(self):
        return {"evaluator": "autovla-navsim", "inference_options": {"thinking": False},
                "seed_name": "generation_seed_base", "seed_base": self.seed_base,
                "source": {"autovla_inputs": str(self.inputs)}}

    def seeds(self, token):
        return {"generation_seed": paired_seed(SEED_NAMESPACE, self.seed_base, token)}

    def predict(self, token, seeds):
        agent_input = read_json(self.inputs / f"{token}.json")
        if not isinstance(agent_input, dict) or agent_input.get("token") != token:
            raise ValueError(f"Missing or mismatched AutoVLA input {self.inputs / f'{token}.json'}")
        if not self.original:
            agent_input = edited_input(agent_input, self.loader, token)
        torch.manual_seed(seeds["generation_seed"])
        return self.agent.compute_trajectory(agent_input)

    def save(self, token, prediction):
        return super().save(token, split(prediction)[0])

    def save_extras(self, token, prediction):
        path = self.result_dir / "cot_outputs" / f"{token}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cot_text(split(prediction)[1]) + "\n", encoding="utf-8")
        return {}

    def summary_extras(self):
        return {"generation_seed_base": self.seed_base}


if __name__ == "__main__":
    raise SystemExit(AutoVLA.main(__doc__))
