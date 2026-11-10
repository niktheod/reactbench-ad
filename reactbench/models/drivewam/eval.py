"""Run DriveWAM on a ReactNAV run (original or edited front-camera image).

    python -m reactbench.models.drivewam.eval RUN_DIR --original     # cache original predictions
    python -m reactbench.models.drivewam.eval RUN_DIR                # edited predictions + scoring

Predictions go to ``<outputs>/predictions/drivewam/reactnav/<run>``. The original and edited rollouts of a scene
share all video and action noise. ``models/runner.py`` holds everything that is not specific to DriveWAM.
"""
from __future__ import annotations

from pathlib import Path

from .. import runner
from ..seeds import paired_seed

SEED_NAMESPACE = "drivewam-rollout"  # per-token seeds: models/seeds.py


def make_input(annotation, image_path):
    return {
        "image_path": image_path,
        "velocity": annotation["velocity"][-1],
        "acceleration": annotation["acceleration"][-1],
        "history_poses": annotation["past_trajectory"],
        "navigation_command": annotation["navigation_command"],
    }


class DriveWAM(runner.Runner):
    model = "drivewam"
    config_dir = Path(__file__).parent
    default_config = "config"

    def setup(self):
        from hydra.utils import instantiate
        self.agent = instantiate(self.config)
        self.interval_length_s = float(self.config["trajectory_sampling"]["interval_length"])

    def cache_fields(self):
        return {"evaluator": "drivewam-navsim", "seed_name": "rollout_seed_base", "seed_base": self.agent.seed}

    def seeds(self, token):
        return {"rollout_seed": paired_seed(SEED_NAMESPACE, self.agent.seed, token)}

    def predict(self, token, seeds):
        annotation = self.loader.annotation(token)
        image = self.loader.original_images(token)["front_camera"][-1] if self.original else \
            self.loader.edited_image(token)
        return self.agent.compute_trajectory(make_input(annotation, image), rollout_seed=seeds["rollout_seed"])


if __name__ == "__main__":
    raise SystemExit(DriveWAM.main(__doc__))
