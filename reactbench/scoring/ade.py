"""Original/edited ADE over four seconds and the trajectory invariance rate (TIR) thresholds."""
from __future__ import annotations

import math
import numpy as np

from ..io import digest
from .results import load_prediction

THRESHOLDS_M = (0.05, 0.10, 0.20, 0.50)


def statistics(values, excluded=0):
    valid = [float(v) for v in values if isinstance(v, (int, float))
             and not isinstance(v, bool) and math.isfinite(v) and v >= 0]
    count = len(valid)
    return {"horizon_s": 4.0, "comparison": "<", "valid_count": count,
            "excluded_count": excluded + len(values) - count,
            "values_sha256": digest(valid),
            "thresholds": [{"threshold_m": t, "count": sum(v < t for v in valid),
                            "percentage": 100 * sum(v < t for v in valid) / count if count else None}
                           for t in THRESHOLDS_M]}


def paired_ade(sources):
    original, dt = load_prediction(sources["original"])
    edited, edited_dt = load_prediction(sources["edited"])
    if not math.isclose(dt, edited_dt, abs_tol=1e-6):
        raise ValueError("Original and edited sampling intervals differ")
    count = round(4.0 / dt)
    if count < 1 or not math.isclose(count * dt, 4.0, abs_tol=1e-6):
        raise ValueError("Four seconds is not an integer number of trajectory steps")
    if min(len(original), len(edited)) < count:
        raise ValueError("Trajectory does not cover four seconds")
    return float(np.linalg.norm(original[:count, :2] - edited[:count, :2], axis=1).mean())
