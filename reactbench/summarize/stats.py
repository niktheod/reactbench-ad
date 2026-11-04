"""Bootstrap confidence intervals used for every interval in the paper.

All intervals are 95% percentile bootstrap intervals over scenes with 10,000 resamples. Paired
statistics resample the per-scene differences, so both predictions of a scene stay together.
"""
from __future__ import annotations

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
SEED = 20260911


def bootstrap_mean(values, seed=SEED, samples=BOOTSTRAP_RESAMPLES):
    """95% percentile interval of the mean, resampling the given per-scene values."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.concatenate([values[rng.integers(len(values), size=(min(50, samples - i), len(values)))].mean(axis=1)
                            for i in range(0, samples, 50)])
    return np.quantile(means, [.025, .975]).tolist()


def aggregate_bin(cells, *, min_bin_count=10, seed=SEED, samples=BOOTSTRAP_RESAMPLES):
    """Equal-weight mean over the four look-ahead times of one bin; resamples whole source scenes.

    ``cells`` holds one ``{"n", "scene_deltas": {token: delta}}`` per look-ahead time.
    """
    counts = [c.get("n", 0) for c in cells]
    if len(cells) != 4 or any(n < min_bin_count for n in counts):
        return {"n_by_timestep": counts, "visible": False, "delta": None, "ci95": [None, None]}
    tokens = sorted(set().union(*(c["scene_deltas"] for c in cells)))
    values = np.array([[c["scene_deltas"].get(token, np.nan) for c in cells] for token in tokens])
    mean = float(np.nanmean(values, axis=0).mean())
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, samples, 50):
        selected = values[rng.integers(len(tokens), size=(min(50, samples - start), len(tokens)))]
        valid_counts = np.isfinite(selected).sum(axis=1)
        sums = np.nansum(selected, axis=1)
        complete = (valid_counts > 0).all(axis=1)
        estimates.extend((sums[complete] / valid_counts[complete]).mean(axis=1).tolist())
    return {"n": len(tokens), "n_by_timestep": counts, "visible": True, "delta": mean,
            "ci95": np.quantile(estimates, [.025, .975]).tolist() if estimates else [None, None]}


def excludes_zero(interval):
    lower, upper = interval
    return lower is not None and upper is not None and (lower > 0 or upper < 0)
