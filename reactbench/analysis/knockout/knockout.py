"""Attention knockout on original ReactPAI scenes (appendix, Attention Knockout).

Each token gets three rollouts of the same input: ``natural``, with the seeds of
the main evaluation; ``knockout``, the same seeds, but every expert layer and
denoising step hides all generated keys (the CoT and its boundary token); and
``reseed``, a natural rollout with a different diffusion seed (the model's
sampling spread, a scale reference). All three must keep the natural CoT.

Token selection (300 of the 1,045 original ReactPAI samples, seed 0)::

    python -m reactbench.analysis.knockout.knockout select-tokens
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ... import paths
from .collector import COT, SPANS, ExpertAttentionCollector
from .common import (
    SampleContext,
    checked_token,
    read_tokens_file,
    same_prediction,
    select_tokens,
    single_text,
    write_token_selection,
)

MODEL_LABELS = {"alpamayo1_5": "Alpamayo 1.5", "alpamayo2": "Alpamayo 2"}
DEFAULT_OUTPUT_ROOT = paths.outputs("knockout", "rollouts")
DEFAULT_RESULTS_DIR = paths.outputs("knockout")
# The released selection; ``select-tokens`` regenerates it from the original predictions.
DEFAULT_TOKENS_FILE = paths.REPO_ROOT / "data" / "knockout" / "tokens.json"
# Any ReactPAI run: only its original inputs are read, which every ReactPAI run shares.
DEFAULT_RUN_DIR = paths.get("reactpai", "datasets/reactpai") / "crossing_pedestrian_timestep_20"
# The original-scene predictions of the main evaluation; their CoTs must be reproduced exactly.
ORIGINAL_PREDICTIONS = {model: paths.outputs("predictions", model, "reactpai", "original")
                        for model in MODEL_LABELS}
CONDITIONS = ("natural", "knockout", "reseed")
NUM_POSES = 64
# The reseed rollout derives its diffusion seed from the evaluator's base + 1.
RESEED_BASE_OFFSET = 1
DEFAULT_SAMPLE_COUNT = 300
DEFAULT_SELECTION_SEED = 0
HIDDEN_SPANS = (COT,)
META_INTEGERS = (
    "prompt_len", "cache_len", "generated_tokens", "hidden_keys", "num_layers", "num_steps",
)


def build_parser(*, description: str, default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_RUN_DIR,
        help="ReactPAI run supplying the tokens; only original inputs are read.",
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--tokens-file",
        type=Path,
        default=DEFAULT_TOKENS_FILE,
        help="JSON object with the exact ordered token list under 'tokens'.",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Use only the first N tokens of the tokens file."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite", action="store_true", help="Rerun tokens whose result file already exists."
    )
    parser.add_argument(
        "--check-passthrough",
        action="store_true",
        help=(
            "Also run each token's natural rollout through a collector without hidden "
            "spans and require an identical trajectory and CoT."
        ),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")


def select_run_tokens(available: Sequence[str], args: argparse.Namespace) -> list[str]:
    tokens = select_tokens(list(available), args.tokens_file)
    return tokens if args.max_samples is None else tokens[: args.max_samples]


def saved_reasoning(model: str, token: str) -> dict[str, Any] | None:
    """The token's reasoning JSON in the model's original-scene predictions, if any."""
    path = ORIGINAL_PREDICTIONS[model] / "reasoning" / f"{checked_token(token)}.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def trajectory_poses(prediction: Any) -> np.ndarray:
    poses = np.asarray(prediction.poses, dtype=np.float32)
    if poses.shape != (NUM_POSES, 3) or not np.isfinite(poses).all():
        raise ValueError(f"Expected {NUM_POSES} finite x/y/yaw poses, got {poses.shape}")
    return poses


def build_collectors(
    *, check_passthrough: bool, **kwargs: Any
) -> tuple[ExpertAttentionCollector, ExpertAttentionCollector | None]:
    """The knockout collector and, if asked, one without hidden spans; both uninstalled."""
    knockout = ExpertAttentionCollector(hidden_spans=HIDDEN_SPANS, **kwargs)
    knockout.close()
    passthrough = None
    if check_passthrough:
        passthrough = ExpertAttentionCollector(**kwargs)
        passthrough.close()
    return knockout, passthrough


def _collected_rollout(
    collector: ExpertAttentionCollector, rollout: Callable[[int], Any], seed: int
) -> tuple[Any, dict[str, Any]]:
    collector.install()
    try:
        collector.reset()
        prediction = rollout(seed)
        attention = collector.finalize(returned_cot=single_text(prediction.extra, "cot"))
    finally:
        collector.close()
    return prediction, attention


def run_conditions(
    rollout: Callable[[int], Any],
    *,
    natural_seed: int,
    reseed_seed: int,
    knockout: ExpertAttentionCollector,
    passthrough: ExpertAttentionCollector | None,
    context: SampleContext,
) -> dict[str, Any]:
    """The three rollouts of one input; ``rollout(diffusion_seed)`` runs the model once."""
    if knockout.hidden_spans != HIDDEN_SPANS:
        raise ValueError(f"The knockout collector must hide {[SPANS[span] for span in HIDDEN_SPANS]}")
    if passthrough is not None and passthrough.hidden_spans:
        raise ValueError("The passthrough collector must not hide any span")
    if natural_seed == reseed_seed:
        raise ValueError("The reseed rollout needs a different diffusion seed")

    context.stage = "natural_inference"
    natural = rollout(natural_seed)
    cot = single_text(natural.extra, "cot")
    if passthrough is not None:
        context.stage = "passthrough_inference"
        checked, _ = _collected_rollout(passthrough, rollout, natural_seed)
        if not same_prediction(natural, checked):
            raise RuntimeError("The natural rollout through the collector differs from the one without it")
    context.stage = "knockout_inference"
    knocked, attention = _collected_rollout(knockout, rollout, natural_seed)
    context.stage = "reseed_inference"
    reseeded = rollout(reseed_seed)

    context.stage = "check_rollouts"
    for name, prediction in (("knockout", knocked), ("reseed", reseeded)):
        if single_text(prediction.extra, "cot") != cot:
            raise ValueError(f"The {name} rollout generated a different CoT than the natural one")
    poses = np.stack([trajectory_poses(item) for item in (natural, knocked, reseeded)])
    collected = attention["meta"]
    meta = {
        "conditions": list(CONDITIONS),
        "cot": cot,
        "generated_text": collected["generated_text"],
        "hidden_spans": collected["hidden_spans"],
        "passthrough_checked": passthrough is not None,
        **{name: collected[name] for name in META_INTEGERS},
    }
    return {"poses": poses, "meta": meta}


def validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Check one token's poses and metadata; return them as plain values."""
    if set(result) != {"poses", "meta"}:
        raise ValueError(f"Result keys {sorted(result)} differ from ['meta', 'poses']")
    poses = np.asarray(result["poses"])
    expected = (len(CONDITIONS), NUM_POSES, 3)
    if poses.dtype != np.float32 or poses.shape != expected or not np.isfinite(poses).all():
        raise ValueError(f"poses must be finite float32 {expected}, got {poses.dtype} {poses.shape}")
    meta = result["meta"]
    if not isinstance(meta, Mapping):
        raise ValueError("meta must be a mapping")
    meta = json.loads(json.dumps(meta))
    if meta.get("conditions") != list(CONDITIONS):
        raise ValueError(f"meta conditions {meta.get('conditions')} differ from {list(CONDITIONS)}")
    if meta.get("hidden_spans") != [SPANS[span] for span in HIDDEN_SPANS]:
        raise ValueError(f"meta hidden spans {meta.get('hidden_spans')} are not the CoT")
    for name in META_INTEGERS:
        value = meta.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"meta[{name!r}] must be an integer")
    generated = meta["cache_len"] - meta["prompt_len"]
    if not 0 < generated == meta["generated_tokens"] == meta["hidden_keys"]:
        raise ValueError("The knockout must hide exactly the generated tokens")
    for name in ("token", "cot"):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            raise ValueError(f"meta[{name!r}] must be a non-empty string")
    return {"poses": poses, "meta": meta}


def load_result(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"poses", "meta"}:
            raise ValueError(f"Unexpected arrays in {path}: {sorted(data.files)}")
        meta = data["meta"]
        if meta.ndim != 0 or meta.dtype.kind != "U":
            raise ValueError(f"meta in {path} is not a JSON string")
        result = {"poses": data["poses"], "meta": json.loads(meta.item())}
    loaded = validate_result(result)
    if loaded["meta"]["token"] != Path(path).stem:
        raise ValueError(f"{path} stores token {loaded['meta']['token']!r}")
    return loaded


class KnockoutOutputManager:
    """One ``<root>/<model>/<token>.npz`` per token, written atomically."""

    def __init__(self, *, model: str, root: Path) -> None:
        if model not in MODEL_LABELS:
            raise ValueError(f"Unknown model {model!r}")
        self.model = model
        self.output_dir = Path(root).expanduser() / model
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def failures_path(self) -> Path:
        return self.output_dir / "failed_samples.json"

    def path(self, token: str) -> Path:
        return self.output_dir / f"{checked_token(token)}.npz"

    def is_complete(self, token: str) -> bool:
        path = self.path(token)
        if not path.is_file():
            return False
        try:
            load_result(path)
        except Exception:
            return False
        return True

    def save(self, token: str, result: Mapping[str, Any]) -> None:
        prepared = validate_result(result)
        if prepared["meta"]["token"] != token:
            raise ValueError(f"Result for {prepared['meta']['token']!r} saved as {token!r}")
        target = self.path(token)
        temporary = target.with_name(f".{target.name}.publishing")
        try:
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    poses=prepared["poses"],
                    meta=np.array(json.dumps(prepared["meta"], sort_keys=True)),
                )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Attention-knockout utilities.")
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser(
        "select-tokens",
        help="Draw the tokens shared by both models from their original-scene ReactPAI predictions.",
    )
    select.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    select.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT)
    select.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    select.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "tokens.json")
    select.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    document = write_token_selection(
        args.output,
        run_dir=args.run_dir,
        prediction_dirs=ORIGINAL_PREDICTIONS,
        count=args.count,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"Selected {document['count']} of {document['pool_size']} common tokens "
        f"(seed {document['seed']}) into {args.output}"
    )
    if DEFAULT_TOKENS_FILE.is_file():
        same = document["tokens"] == read_tokens_file(DEFAULT_TOKENS_FILE)
        print(f"{'Identical to' if same else 'DIFFERENT from'} the released selection {DEFAULT_TOKENS_FILE}")


if __name__ == "__main__":
    main()
