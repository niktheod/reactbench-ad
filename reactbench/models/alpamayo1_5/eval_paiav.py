"""Run Alpamayo 1.5 on a ReactPAI variant: original and edited scenes (``eval.py`` with ``--dataset reactpai``).

    python -m reactbench.models.alpamayo1_5.eval_paiav VARIANT_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo1_5.eval_paiav VARIANT_DIR               # edited predictions + scoring

The CoT interventions (CoT swap, fixed CoTs) are ``python -m reactbench.analysis.cot.run_alpamayo1_5``.
"""
from .eval import Alpamayo15

if __name__ == "__main__":
    raise SystemExit(Alpamayo15.main(__doc__, dataset="reactpai"))
