"""Run Alpamayo 1.5 on a ReactNAV or ReactPAI variant: original and edited scenes.

    python -m reactbench.models.alpamayo1_5.eval VARIANT_DIR [--dataset reactpai] --original   # cache originals
    python -m reactbench.models.alpamayo1_5.eval VARIANT_DIR [--dataset reactpai]              # edited + scoring

(``eval_navsim``/``eval_paiav`` are the same with the dataset preset.) ReactNAV: the front-left, front and
front-right cameras stand in for the cross-left, front-wide and cross-right cameras (frames 0.5 s apart instead of
0.1 s), the four ego poses are interpolated to 16 at 10 Hz and the driving command becomes a navigation
instruction. ReactPAI: the model's native inputs (four 10 Hz cameras, 16-pose ego history, no navigation text);
original frames are decoded from the PAI-AV archives. The decoded reasoning goes to ``reasoning/``. The CoT
interventions are ``python -m reactbench.analysis.cot.run_alpamayo1_5``.
"""
from __future__ import annotations

from pathlib import Path

from .. import runner
from ..seeds import paired_seed
from .inputs import model_input

SEED_NAMESPACE = "alpamayo-diffusion"  # per-token seeds: models/seeds.py


class Alpamayo15(runner.Runner):
    model = "alpamayo1_5"
    datasets = ("reactnav", "reactpai")
    config_dir = Path(__file__).parent
    default_config = "config"

    def setup(self):
        from hydra.utils import instantiate
        self.agent = instantiate(self.config)
        self.interval_length_s = float(self.config["trajectory_sampling"]["interval_length"])

    def cache_fields(self):
        return {"evaluator": "alpamayo-navsim" if self.dataset == "reactnav" else "alpamayo-paiav",
                "seed_name": "diffusion_seed_base", "seed_base": self.agent.diffusion_seed_base}

    def seeds(self, token):
        return {"diffusion_seed": paired_seed(SEED_NAMESPACE, self.agent.diffusion_seed_base, token)}

    def inputs(self, token, original):
        return model_input(self.loader, token, original=original)

    def predict(self, token, seeds):
        with self.inputs(token, self.original) as agent_input:
            return self.agent.compute_trajectory(agent_input, diffusion_seed=seeds["diffusion_seed"])

    def save_extras(self, token, prediction):
        runner.save_json(self.result_dir / "reasoning" / f"{token}.json", prediction.extra)
        return {}

    def summary_extras(self):
        return {"model_name_or_path": self.agent.model_name_or_path, "revision": self.agent.revision,
                "attn_implementation": self.agent.attn_implementation, "vlm_seed": self.agent.seed,
                "diffusion_seed_base": self.agent.diffusion_seed_base}


if __name__ == "__main__":
    raise SystemExit(Alpamayo15.main(__doc__))
