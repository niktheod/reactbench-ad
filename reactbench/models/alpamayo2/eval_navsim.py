"""Run Alpamayo 2 on a ReactNAV variant: original and edited scenes (``eval.py`` with ``--dataset reactnav``).

    python -m reactbench.models.alpamayo2.eval_navsim VARIANT_DIR --original    # cache original predictions
    python -m reactbench.models.alpamayo2.eval_navsim VARIANT_DIR               # edited predictions + scoring
"""
from .eval import Alpamayo2

if __name__ == "__main__":
    raise SystemExit(Alpamayo2.main(__doc__, dataset="reactnav"))
