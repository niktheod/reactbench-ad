"""Alpamayo 1.5 CoT interventions on ReactPAI (paper section 4.2 and appendix): CoT swap and fixed CoTs.

    python -m reactbench.analysis.cot.run_alpamayo1_5 RUN_DIR --cot-swap     # O/E images x O/E CoTs
    python -m reactbench.analysis.cot.run_alpamayo1_5 RUN_DIR --fixed-cots   # the five CoTs of fixed_cots.yaml

Both build on the run's natural original and edited predictions (``python -m reactbench.models.alpamayo1_5.eval_paiav``:
same configuration and per-token seeds; ``natural.py``) and force the CoT as Alpamayo 1.5's assistant prefix
``<|cot_start|>{cot}<|cot_end|><|traj_future_start|>`` (``models/alpamayo1_5/forced_cot.py``). Results go to
``<outputs>/predictions/alpamayo1_5/reactpai/<run>_cot_swap`` and ``<run>_fixed_cots``.
"""
from __future__ import annotations

from ...io import file_digest
from ...models.alpamayo1_5.eval import Alpamayo15
from ...models.alpamayo1_5.forced_cot import forced_cot_prefix
from .cot_swap import unwrap_singleton
from .natural import main


class Alpamayo15CoT(Alpamayo15):
    forced_prefix = staticmethod(forced_cot_prefix)

    def donor_cot(self, document):
        """The CoT a natural rollout decoded (without Alpamayo's control tokens)."""
        if "cot" not in document:
            raise ValueError("Saved reasoning has no 'cot' field")
        cot = unwrap_singleton(document["cot"])
        forced_cot_prefix(cot)
        return cot

    def force(self, model_input, cot, seeds):
        prediction = self.agent.compute_trajectory(model_input, forced_cot=cot, diffusion_seed=seeds["diffusion_seed"])
        if unwrap_singleton(prediction.extra.get("cot")) != cot:
            raise ValueError("Alpamayo output did not preserve the forced CoT")
        return prediction

    def cot_extras(self):
        return {}

    def cot_identity(self):
        return {"backend": "alpamayo", "source_files": {
            str(path): file_digest(path) for path in (self.loader.annotations_path, self.loader.source_index_path)}}


if __name__ == "__main__":
    raise SystemExit(main(Alpamayo15CoT, __doc__))
