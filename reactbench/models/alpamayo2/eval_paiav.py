"""Run Alpamayo 2 on a compact ReactPAI run: natural pairs, CoT swaps, fixed CoTs.

    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR               # edited predictions + scoring
    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR --cot-swap    # O/E images x O/E CoTs
    python -m reactbench.models.alpamayo2.eval_paiav RUN_DIR --fixed-cots  # five fixed CoTs
"""
import sys

from .runner import main

# These flags select cot_interventions.py; every other invocation is the natural runner.
INTERVENTION_FLAGS = ("--cot-swap", "--fixed-cots")

if __name__ == "__main__":
    if any(arg in INTERVENTION_FLAGS or arg.startswith("--fixed-cots=") for arg in sys.argv[1:]):
        from .cot_interventions import main as intervention_main
        raise SystemExit(intervention_main("pai-av"))
    raise SystemExit(main("pai-av"))
