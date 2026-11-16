"""Alpamayo 2 CoT interventions on ReactPAI (paper section 4.3 and appendix): CoT swap and fixed CoTs.

    python -m reactbench.analysis.cot.run_alpamayo2 RUN_DIR --cot-swap     # O/E images x O/E CoTs
    python -m reactbench.analysis.cot.run_alpamayo2 RUN_DIR --fixed-cots   # the five CoTs of fixed_cots.yaml

Both build on the run's natural original and edited predictions (``python -m reactbench.models.alpamayo2.eval_paiav``:
same configuration, GPU architecture class and per-token seeds; ``natural.py``). The natural CoT must have been
closed by ``<|traj_future_start|>``; it is forced as the assistant prefix ``{cot}<|traj_future_start|>``
(``models/alpamayo2/forced_cot.py``). Results go to ``<outputs>/predictions/alpamayo2/reactpai/<run>_cot_swap`` and
``<run>_fixed_cots``.
"""
from __future__ import annotations

from ...models.alpamayo2 import forced_cot
from ...models.alpamayo2.eval import Alpamayo2
from .natural import main


class Alpamayo2CoT(Alpamayo2):
    forced_prefix = staticmethod(forced_cot.forced_cot_prefix)

    def donor_cot(self, document):
        return forced_cot.natural_cot(document)

    def force(self, model_input, cot, seeds):
        return forced_cot.compute_forced_trajectory(self.agent, model_input, cot, text_seed=seeds["text_seed"],
                                                    diffusion_seed=seeds["diffusion_seed"])

    def cot_extras(self):
        return {"forced_cot_adapter": forced_cot.adapter_identity()}

    def cot_identity(self):
        return {"backend": "alpamayo2-pai-av", "dataset": "pai-av", "model_identity": self.identity(),
                "forced_cot_adapter": forced_cot.adapter_identity()}


if __name__ == "__main__":
    raise SystemExit(main(Alpamayo2CoT, __doc__))
