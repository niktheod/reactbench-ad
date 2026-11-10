"""Run Alpamayo 2 on a ReactPAI run: original and edited scenes (``eval.py`` with ``--dataset reactpai``).

    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR               # edited predictions + scoring

The CoT interventions (CoT swap, fixed CoTs) are ``python -m reactbench.analysis.cot.run_alpamayo2``.
"""
from .eval import Alpamayo2

if __name__ == "__main__":
    raise SystemExit(Alpamayo2.main(__doc__, dataset="reactpai"))
