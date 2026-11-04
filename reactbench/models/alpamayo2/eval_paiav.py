"""Run Alpamayo 2 on a compact ReactPAI run: original and edited scenes.

    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR               # edited predictions + scoring

The CoT interventions (CoT swap, fixed CoTs) are ``python -m reactbench.analysis.cot.run_alpamayo2``.
"""
from .runner import main

if __name__ == "__main__":
    raise SystemExit(main("pai-av"))
