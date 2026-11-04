"""Run Alpamayo 2 on a ReactNAV run: original and edited scenes.

    python -m reactbench.models.alpamayo2.eval_navsim RUN_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo2.eval_navsim RUN_DIR               # edited predictions + scoring
"""
from .runner import main

if __name__ == "__main__":
    raise SystemExit(main("navsim"))
