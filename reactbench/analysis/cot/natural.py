"""The natural runs the CoT interventions build on, and the command line of both CoT runners.

A CoT runner (``run_alpamayo1_5.py``, ``run_alpamayo2.py``) starts from its model's finished natural runs on a
ReactPAI variant, made by the model's runner (``models/<model>/eval.py``): ``original/`` and the edited run
``<variant>/``.
Both must carry this configuration's cache signature (``models/original_cache.py``) and, per token, the same source
contract and seeds; their decoded CoTs are the CoT-swap donors and their trajectories the natural conditions.
Forced rollouts reuse each token's natural seeds. Results go next to the natural runs::

    python -m reactbench.analysis.cot.run_<model> VARIANT_DIR --cot-swap     # <variant>_cot_swap: O/E images x O/E CoTs
    python -m reactbench.analysis.cot.run_<model> VARIANT_DIR --fixed-cots   # <variant>_fixed_cots: fixed_cots.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ...io import file_digest, read_json
from ...models import runner
from ...models.original_cache import load_cached_trajectory, read_manifest, source_contract_identity

FIXED_COTS = Path(__file__).resolve().with_name("fixed_cots.yaml")
SUFFIXES = {"cot_swap": "_cot_swap", "fixed_cots": "_fixed_cots"}


class NaturalMismatch(ValueError):
    """A finished natural run disagrees with this sample's inputs or seeds (fatal)."""


@dataclass(frozen=True)
class NaturalOutput:
    poses: np.ndarray
    document: dict
    reasoning_path: Path
    reasoning_sha256: str
    trajectory_path: Path
    trajectory_sha256: str

    def provenance(self):
        return {"inference": "natural", "trajectory_path": str(self.trajectory_path),
                "trajectory_sha256": self.trajectory_sha256, "reasoning_path": str(self.reasoning_path),
                "reasoning_sha256": self.reasoning_sha256}


class NaturalRun:
    """A finished natural run (``mode`` original or paired) of the same configuration."""

    def __init__(self, path, *, mode, signature, variant_dir=None):
        self.path, self.mode = Path(path).expanduser().resolve(), mode
        manifest_path, manifest = read_manifest(self.path, mode=mode, signature=signature)
        if variant_dir is not None and manifest.get("source_run_dir") != str(variant_dir):
            raise ValueError(f"{manifest_path} was computed from {manifest.get('source_run_dir')}, not {variant_dir}")
        self.selected, self.successful = set(manifest["selected_tokens"]), set(manifest["successful_tokens"])
        self.contracts = manifest.get("source_contracts")
        if not isinstance(self.contracts, dict):
            raise ValueError(f"Invalid source_contracts in {manifest_path}")
        self.records = {row["token"]: row for row in read_json(self.path / "metrics" / "per_sample.json", [])}
        self.interval_length_s = float(signature["interval_length_s"])
        self.status = {"path": str(self.path), "manifest_sha256": file_digest(manifest_path)}

    def unavailable(self, token):
        """Why this token has no natural output, or None."""
        if token not in self.selected:
            return f"Never attempted by the natural {self.mode} run in {self.path}"
        if token not in self.successful:
            return f"Natural {self.mode} inference failed or the sample was excluded"
        return None

    def load(self, token, contract, seeds):
        """Saved outputs of a successful token; input or seed mismatches are fatal."""
        if source_contract_identity(self.contracts.get(token)) != source_contract_identity(contract):
            raise NaturalMismatch(f"Source provenance differs from the natural {self.mode} run for {token}")
        record = self.records.get(token, {})
        if any(record.get(name) != value for name, value in seeds.items()):
            raise NaturalMismatch(f"Natural {self.mode} seeds differ for {token}")
        reasoning_path = self.path / "reasoning" / f"{token}.json"
        document = read_json(reasoning_path)
        if not isinstance(document, dict):
            raise FileNotFoundError(f"Missing natural {self.mode} reasoning: {reasoning_path}")
        trajectory_path = self.path / "trajectories" / f"{token}.npz"
        poses = load_cached_trajectory(trajectory_path, expected_interval_length_s=self.interval_length_s)
        return NaturalOutput(poses, document, reasoning_path, file_digest(reasoning_path),
                             trajectory_path, file_digest(trajectory_path))


def natural_runs(s, *, required):
    """{"original": NaturalRun, "edited": NaturalRun}; a missing run is an error if ``required``, else None."""
    runs, status = {}, {}
    for name, path, mode, variant_dir in (("original", s.original_dir, "original", None),
                                      ("edited", s.edited_dir, "paired", s.model.variant_dir)):
        try:
            runs[name] = NaturalRun(path, mode=mode, signature=s.signature, variant_dir=variant_dir)
            status[name] = runs[name].status
        except FileNotFoundError as exc:
            if required:
                raise
            runs[name], status[name] = None, {"missing": str(exc)}
    return runs, status


def parse_args(description, argv=None):
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant_dir", type=Path, help="ReactPAI variant (its natural runs must be finished)")
    parser.add_argument("--config", help="The natural runs' configuration (default: the model's config.yaml)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cot-swap", action="store_true",
                      help="Original image with the edited CoT and edited image with the original CoT")
    mode.add_argument("--fixed-cots", type=Path, metavar="PATH", nargs="?", const=FIXED_COTS,
                      help="The five fixed CoTs (default: the bundled fixed_cots.yaml)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-samples", type=int, help="Only the first N clips (a smoke test)")
    selection.add_argument("--tokens-file", type=Path, help="JSON with the exact ordered 'tokens'")
    parser.add_argument("--original-result-dir", type=Path, help="Natural original run (default reactpai/original)")
    parser.add_argument("--edited-result-dir", type=Path, help="Natural edited run (default reactpai/<variant>)")
    parser.add_argument("--scoring", choices=("score", "off"), default="score",
                        help="Score the result on CPU afterwards (default) or not")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    args.dataset, args.original = "reactpai", False
    return args


def main(model_class, description, argv=None):
    """Run one intervention with a model's runner class extended by the CoT methods (see the CoT runners)."""
    from . import cot_swap, fixed_cot
    args = parse_args(description, argv)
    model = model_class(args)
    group = runner.predictions_dir(model.model, args.dataset, "original").parent
    mode = "cot_swap" if args.cot_swap else "fixed_cots"
    s = SimpleNamespace(
        args=args, model=model, mode=mode, result_dir=group / f"{model.variant_dir.name}{SUFFIXES[mode]}",
        original_dir=Path(args.original_result_dir or group / "original").expanduser().resolve(),
        edited_dir=Path(args.edited_result_dir or group / model.variant_dir.name).expanduser().resolve())
    if s.result_dir in (s.original_dir, s.edited_dir):
        raise ValueError("Intervention results must not be written into a natural run directory")
    runner.prepare_output(s.result_dir, args.overwrite, remove=False)
    with model.open_variant() as loader:
        model.loader = loader
        model.setup()
        s.tokens = runner.select_tokens(loader.tokens, tokens_file=args.tokens_file, max_samples=args.max_samples)
        s.signature = model.signature()
        status = (cot_swap.run if args.cot_swap else fixed_cot.run)(s)
    if args.scoring == "score" and not runner.score(s.result_dir, args.dataset):
        status = 1
    return status
