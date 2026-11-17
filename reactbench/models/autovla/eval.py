"""Run AutoVLA on a ReactNAV run (three cameras, four history frames).

    python -m reactbench.models.autovla.eval RUN_DIR --original     # cache original predictions
    python -m reactbench.models.autovla.eval RUN_DIR                # edited predictions + scoring

Inputs are AutoVLA's own per-token no-CoT samples (``autovla_inputs`` in configs/paths.yaml, made with AutoVLA's
preprocessing). On NAVSIM, AutoVLA's released code reads the side cameras CAM_L1/R1 (``left/right_camera_paths``)
as its front-left and front-right views. The compositor renders CAM_F0/L0/R0 only, so in edited runs only the four
CAM_F0 frames (and the unused CAM_L0/R0 fields) are replaced, and CAM_L1/R1 stay original; the few samples whose
inserted object would enter CAM_L1/R1 are skipped (``data/reactnav/autovla_side_camera_fov.json``). ``--config
l0r0`` (``autovla_l0r0``, appendix) instead feeds CAM_L0/R0 as the two views, edited in edited runs, on every
sample. The text output goes to ``cot_outputs/``. Predictions go to ``<outputs>/predictions/<name>/reactnav/<run>``.
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
from ...runs import parse_run_name
from .. import runner
from ..seeds import paired_seed

AUTOVLA_REPO_ROOT = Path(os.environ.get("AUTOVLA_REPO_ROOT", paths.REPO_ROOT / "third_party" / "AutoVLA"))
# AutoVLA's input field of each rendered camera (on NAVSIM it reads CAM_L0/R0 only with --config l0r0).
CAMERA_INPUT_KEYS = {"CAM_F0": "front_camera_paths", "CAM_L0": "front_left_camera_paths",
                     "CAM_R0": "front_right_camera_paths"}
SEED_NAMESPACE = "autovla-generation"  # per-token seeds: models/seeds.py
# The cameras of AutoVLA's front-left/right views: l1r1 (CAM_L1/R1, the released code) or l0r0 (CAM_L0/R0).
SIDE_CAMERAS = ("l1r1", "l0r0")
FOV_EXCLUSIONS = paths.REPO_ROOT / "data" / "reactnav" / "autovla_side_camera_fov.json"


def edited_input(agent_input, loader, token):
    """AutoVLA's input with the four frames of the rendered cameras replaced by the edited ones.

    AutoVLA's front-left/right views (``left/right_camera_paths``, CAM_L1/R1) are not rendered and stay original.
    """
    agent_input = copy.deepcopy(agent_input)
    for camera, key in CAMERA_INPUT_KEYS.items():
        if not isinstance(agent_input.get(key), list) or len(agent_input[key]) != 4:
            raise ValueError(f"Expected four paths in {key!r} for token {token}")
        agent_input[key] = [str(loader.edited_image(token, camera, frame)) for frame in range(4)]
    return agent_input


def with_side_cameras(agent_input, side_cameras):
    """The input whose front-left/right views are CAM_L1/R1 (l1r1, unchanged) or CAM_L0/R0 (l0r0)."""
    if side_cameras not in SIDE_CAMERAS:
        raise ValueError(f"side_cameras must be one of {SIDE_CAMERAS}, got {side_cameras!r}")
    if side_cameras == "l1r1":
        return agent_input
    agent_input = copy.deepcopy(agent_input)
    for side in ("left", "right"):
        views = agent_input.get(f"front_{side}_camera_paths")
        if not isinstance(views, list) or len(views) != 4:
            raise ValueError(f"Expected four paths in 'front_{side}_camera_paths'")
        agent_input[f"{side}_camera_paths"] = list(views)
    return agent_input


def side_camera_exclusions(run_name, path=FOV_EXCLUSIONS):
    """{token: reason} of a run's samples whose inserted object would enter CAM_L1/R1."""
    if parse_run_name(run_name, "navsim") is None:
        return {}
    tokens = read_json(path)["runs"].get(run_name, [])
    return {token: "inserted object in the field of view of CAM_L1/R1, which are not rendered" for token in tokens}


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
        self.side_cameras = self.config.pop("side_cameras", "l1r1")
        if self.side_cameras not in SIDE_CAMERAS:
            raise ValueError(f"side_cameras must be one of {SIDE_CAMERAS}, got {self.side_cameras!r}")
        self.agent = instantiate(self.config)
        self.interval_length_s = float(self.config["trajectory_sampling"]["interval_length"])

    def cache_fields(self):
        options = {"thinking": False} | ({"side_cameras": self.side_cameras} if self.side_cameras != "l1r1" else {})
        return {"evaluator": "autovla-navsim", "inference_options": options,
                "seed_name": "generation_seed_base", "seed_base": self.seed_base,
                "source": {"autovla_inputs": str(self.inputs)}}

    def seeds(self, token):
        return {"generation_seed": paired_seed(SEED_NAMESPACE, self.seed_base, token)}

    def excluded_tokens(self):
        # With CAM_L0/R0 every view that shows the object is rendered.
        return side_camera_exclusions(self.run_dir.name) if self.side_cameras == "l1r1" else {}

    def predict(self, token, seeds):
        agent_input = read_json(self.inputs / f"{token}.json")
        if not isinstance(agent_input, dict) or agent_input.get("token") != token:
            raise ValueError(f"Missing or mismatched AutoVLA input {self.inputs / f'{token}.json'}")
        if not self.original:
            agent_input = edited_input(agent_input, self.loader, token)
        agent_input = with_side_cameras(agent_input, self.side_cameras)
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
        return {"generation_seed_base": self.seed_base, "side_cameras": self.side_cameras}


if __name__ == "__main__":
    raise SystemExit(AutoVLA.main(__doc__))
