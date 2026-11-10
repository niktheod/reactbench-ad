"""Run Alpamayo 2 on a ReactNAV or ReactPAI run: original and edited scenes.

    python -m reactbench.models.alpamayo2.eval RUN_DIR [--dataset reactpai] --original   # cache originals
    python -m reactbench.models.alpamayo2.eval RUN_DIR [--dataset reactpai]              # edited + scoring

(``eval_navsim``/``eval_paiav`` are the same with the dataset preset.) ReactNAV: the three recorded cameras at 2 Hz
and the interpolated 10 Hz ego history (``navsim_loader.py``). ReactPAI: NVIDIA's six-camera profile, the two rear
cameras decoded from the unedited archives for both members; edited samples whose object would appear in a rear
view are excluded (``paiav_loader.py``). Text and diffusion seeds are paired; both members of a pair must run on
the same GPU architecture (part of the cache signature). The decoded reasoning goes to ``reasoning/``; the CoT
interventions are ``python -m reactbench.analysis.cot.run_alpamayo2``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ...io import digest, file_digest
from .. import runner
from .inference import Alpamayo2Inference, paired_seed

HERE = Path(__file__).resolve().parent
TARGET = "reactbench.models.alpamayo2.inference.Alpamayo2Inference"
NATIVE_POSES = 64
# Sources that shape the inputs and the inference, hashed into the cache signature.
ADAPTER_SOURCES = {
    "reactnav": (HERE / "eval.py", HERE / "inference.py", HERE / "navsim_loader.py",
                 HERE.parent / "alpamayo1_5" / "agent.py", HERE.parent / "alpamayo1_5" / "inputs.py",
                 HERE.parents[1] / "datasets" / "navsim.py"),
    "reactpai": (HERE / "eval.py", HERE / "inference.py", HERE / "paiav_loader.py",
                 HERE.parent / "alpamayo1_5" / "agent.py", HERE.parents[1] / "datasets" / "paiav.py"),
}


class Alpamayo2(runner.Runner):
    model = "alpamayo2"
    datasets = ("reactnav", "reactpai")
    config_dir = HERE
    default_config = "config"

    def open_run(self):
        if self.dataset == "reactnav":
            from .navsim_loader import NAVSIMRun
            return NAVSIMRun(self.run_dir)
        from .paiav_loader import PAIAVRun
        return PAIAVRun(self.run_dir)

    def setup(self):
        if self.config.get("_target_") != TARGET:
            raise ValueError(f"Alpamayo 2 configuration must have _target_: {TARGET}")
        self.agent = Alpamayo2Inference(**{key: value for key, value in self.config.items()
                                           if not key.startswith("_")})
        self.interval_length_s = 0.1
        self.hardware = self.agent.hardware()

    def identity(self):
        package = HERE.parents[1]
        return {**self.agent.identity(), "adapter_source_sha256": digest(
            {str(path.relative_to(package)): file_digest(path) for path in ADAPTER_SOURCES[self.dataset]})}

    def cache_fields(self):
        # Seeds reproduce rollouts only on one GPU architecture; pairs must share it.
        return {"evaluator": f"alpamayo2-{runner.DATASETS[self.dataset]}", "seed_name": "diffusion_seed_base",
                "seed_base": self.agent.diffusion_seed_base,
                "inference_options": {"identity": self.identity(), "camera_profile": self.loader.camera_profile,
                                      "text_seed_base": self.agent.seed, "hardware_class": self.hardware["class"]}}

    def load(self):
        self.agent.prepare()
        self.agent.load()
        self.agent.identity()  # The pinned upstream source must not have changed while it was imported.

    def seeds(self, token):
        return {"text_seed": paired_seed(token, self.agent.seed, "text"),
                "diffusion_seed": paired_seed(token, self.agent.diffusion_seed_base, "diffusion")}

    def check_edited_inputs(self, token):
        return self.loader.check_edited_inputs(token) if self.dataset == "reactpai" else {}

    @contextmanager
    def inputs(self, token, original):
        with self.loader.materialize_inputs(token, original=original) as paths:
            yield self.loader.model_input(token, paths)

    def predict(self, token, seeds):
        with self.inputs(token, self.original) as model_input:
            return self.agent.compute_trajectory(model_input, text_seed=seeds["text_seed"],
                                                 diffusion_seed=seeds["diffusion_seed"])

    def arrays(self, prediction):
        xyz, rotations = np.asarray(prediction.xyz), np.asarray(prediction.rotations)
        if (np.asarray(prediction.poses).shape != (NATIVE_POSES, 3) or xyz.shape != (NATIVE_POSES, 3)
                or rotations.shape != (NATIVE_POSES, 3, 3)):
            raise ValueError("Prediction must contain all 64 poses, XYZ positions, and rotation matrices")
        if not (np.isfinite(xyz).all() and np.isfinite(rotations).all()):
            raise ValueError("Prediction contains non-finite values")
        return {"pred_xyz": xyz, "pred_rot": rotations, "ego_origin": "rear_axle"}

    def save_extras(self, token, prediction):
        runner.save_json(self.result_dir / "reasoning" / f"{token}.json", prediction.extra)
        return {"model_name_or_path": self.agent.model_name_or_path,
                "image_timestamps_us": prediction.extra["image_timestamps_us"],
                "navigation_command": prediction.extra.get("navigation_command")}

    def summary_extras(self):
        return {"model_name_or_path": self.agent.model_name_or_path, "input_contract": self.loader.camera_profile,
                "hardware": self.hardware, "gpu_placement": self.agent.placement_report(),
                "training_clip_exclusion": ("PhysicalAI-AV is in the model's training data; clip-level split "
                                            "unverified" if self.dataset == "reactpai" else None)}


if __name__ == "__main__":
    raise SystemExit(Alpamayo2.main(__doc__))
