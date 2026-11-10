"""Run Alpamayo 1.5 on a ReactNAV run: original and edited scenes (``eval.py`` with ``--dataset reactnav``).

    python -m reactbench.models.alpamayo1_5.eval_navsim RUN_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo1_5.eval_navsim RUN_DIR               # edited predictions + scoring
"""
from .eval import Alpamayo15

if __name__ == "__main__":
    raise SystemExit(Alpamayo15.main(__doc__, dataset="reactnav"))
