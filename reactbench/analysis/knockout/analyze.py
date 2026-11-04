"""Summarize the attention knockout (appendix table, Attention Knockout).

Per token, over the first 4 s (40 poses at 10 Hz) in the ego frame: the ADE and
FDE from the natural trajectory to the knockout trajectory (KO) and to the
reseeded one (RS). The knockout shares the natural rollout's noise, so its ADE
is the CoT's effect alone (0 means no effect); the reseed ADE is the model's
sampling spread, a scale reference, not a threshold.

Per model: means with 95% percentile bootstrap CIs over tokens (10,000
resamples, seed 0). Between models: the Alpamayo 1.5 - Alpamayo 2 differences
on the same resamples of the shared tokens (a paired bootstrap). Also the
knockout ADE of the tokens whose natural CoT asks for slowing down, stopping,
yielding, braking or decelerating, and of the remaining tokens.

    python -m reactbench.analysis.knockout.analyze
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...io import write_json
from ...summarize.markdown import knockout_table
from .common import read_tokens_file
from .knockout import (
    CONDITIONS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TOKENS_FILE,
    MODEL_LABELS,
    load_result,
)

MODEL_ORDER = ("alpamayo1_5", "alpamayo2")
METRIC_POSES = 40
METRIC_HORIZON_S = 4.0
METRICS = ("ade_knockout", "ade_reseed", "fde_knockout", "fde_reseed")
# Natural CoTs that ask for slowing down, stopping, yielding or braking.
SLOW_KEYWORDS = re.compile(r"\b(slow\w*|stop\w*|yield\w*|brak\w*|decelerat\w*)\b", re.IGNORECASE)
RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95


def bootstrap_weights(n: int, resamples: int, seed: int) -> np.ndarray:
    """[resamples, n] resampling weights (counts / n) for means over n samples."""
    if n <= 0 or resamples <= 0:
        raise ValueError("Bootstrap needs samples and resamples")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, n))
    offsets = np.arange(resamples, dtype=np.int64)[:, None] * n
    counts = np.bincount((draws + offsets).ravel(), minlength=resamples * n)
    return counts.reshape(resamples, n).astype(np.float64) / n


def mean_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Mean and percentile CI of the mean over the bootstrap ``weights``."""
    tail = (1.0 - CI_LEVEL) / 2.0
    low, high = np.quantile(weights @ values, [tail, 1.0 - tail])
    return {"mean": float(values.mean()), "ci": [float(low), float(high)]}


@dataclass
class ModelData:
    model: str
    tokens: list[str]
    metrics: dict[str, np.ndarray]  # metric -> [samples]
    slow_cot: np.ndarray  # [samples] bool
    saved_cot_match: list[bool | None]


def token_metrics(poses: np.ndarray) -> dict[str, float]:
    """Metrics of one token's [natural, knockout, reseed] poses."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[0] != len(CONDITIONS) or poses.shape[1] < METRIC_POSES:
        raise ValueError(f"Expected [{len(CONDITIONS)}, >={METRIC_POSES}, 3] poses, got {poses.shape}")
    natural, knockout, reseed = poses[:, :METRIC_POSES, :2]

    def ade(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second, axis=-1).mean())

    def fde(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first[-1] - second[-1]))

    return {
        "ade_knockout": ade(natural, knockout),
        "ade_reseed": ade(natural, reseed),
        "fde_knockout": fde(natural, knockout),
        "fde_reseed": fde(natural, reseed),
    }


def load_model_data(output_root: Path, model: str, tokens: Sequence[str]) -> ModelData:
    """Load the given tokens; every one must have a valid result file."""
    if not tokens:
        raise ValueError(f"No tokens to load for {model}")
    directory = Path(output_root) / model
    metrics: dict[str, list[float]] = {metric: [] for metric in METRICS}
    slow, saved = [], []
    for token in tokens:
        result = load_result(directory / f"{token}.npz")
        meta = result["meta"]
        if meta.get("model") != model:
            raise ValueError(f"{directory / token}.npz belongs to {meta.get('model')!r}")
        for metric, value in token_metrics(result["poses"]).items():
            metrics[metric].append(value)
        slow.append(SLOW_KEYWORDS.search(meta["cot"]) is not None)
        saved.append(meta.get("saved_cot_match"))
    return ModelData(
        model=model,
        tokens=list(tokens),
        metrics={metric: np.asarray(values) for metric, values in metrics.items()},
        slow_cot=np.asarray(slow, dtype=bool),
        saved_cot_match=saved,
    )


def summarize(models: dict[str, ModelData], *, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    first, second = (models[key] for key in MODEL_ORDER)
    if first.tokens != second.tokens:
        raise ValueError("Both models must be summarized on the same ordered tokens")
    weights = bootstrap_weights(len(first.tokens), resamples, seed)
    per_model = {}
    for key in MODEL_ORDER:
        data = models[key]
        checked = [match for match in data.saved_cot_match if match is not None]
        per_model[key] = {
            "label": MODEL_LABELS[key],
            "samples": len(data.tokens),
            "metrics": {metric: mean_summary(values, weights) for metric, values in data.metrics.items()},
            "ade_knockout_by_cot": {
                name: {"samples": int(mask.sum()),
                       "mean": float(data.metrics["ade_knockout"][mask].mean()) if mask.any() else None}
                for name, mask in (("slow_keyword", data.slow_cot), ("other", ~data.slow_cot))
            },
            "saved_cot_match": {"checked": len(checked), "matched": int(sum(checked))},
        }
    return {
        "conditions": list(CONDITIONS),
        "metric_poses": METRIC_POSES,
        "metric_horizon_s": METRIC_HORIZON_S,
        "slow_keyword_pattern": SLOW_KEYWORDS.pattern,
        "bootstrap": {"resamples": resamples, "seed": seed, "ci_level": CI_LEVEL, "paired": True},
        "samples": len(first.tokens),
        "models": per_model,
        # Paired: the same resamples of the shared tokens for both models.
        "difference": {metric: mean_summary(first.metrics[metric] - second.metrics[metric], weights)
                       for metric in METRICS},
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokens-file", type=Path, default=DEFAULT_TOKENS_FILE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    tokens = read_tokens_file(args.tokens_file)
    present = {key: {token for token in tokens if (Path(args.output_root) / key / f"{token}.npz").is_file()}
               for key in MODEL_ORDER}
    shared = [token for token in tokens if all(token in present[key] for key in MODEL_ORDER)]
    if not shared:
        raise SystemExit("No selected token has results for both models")
    models = {key: load_model_data(args.output_root, key, shared) for key in MODEL_ORDER}
    summary = summarize(models, resamples=args.resamples, seed=args.seed)
    summary.update(selected_tokens=len(tokens),
                   missing_tokens={key: [t for t in tokens if t not in present[key]] for key in MODEL_ORDER})
    results_dir = Path(args.results_dir)
    write_json(results_dir / "summary.json", summary)

    print(f"Shared tokens: {len(shared)} of {len(tokens)}")
    print(knockout_table(summary))
    for key in MODEL_ORDER:
        model = summary["models"][key]
        subsets = model["ade_knockout_by_cot"]
        print(f"{model['label']}: ADE KO {subsets['slow_keyword']['mean']:.2f} m on slow-down CoTs "
              f"(n = {subsets['slow_keyword']['samples']}), {subsets['other']['mean']:.2f} m on the others; "
              f"natural CoT = main evaluation for {model['saved_cot_match']['matched']} of "
              f"{model['saved_cot_match']['checked']} checked")
    print(f"Wrote {results_dir / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
