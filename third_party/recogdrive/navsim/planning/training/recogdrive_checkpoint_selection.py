"""Finalized validation records and constrained checkpoint ranking (scores: 0–100)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A sibling temporary file permits atomic rename on shared filesystems.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def aggregate(records, expected_ids):
    """Require exactly one completed attempt per sample; errors are not zero scores."""
    expected_ids = list(expected_ids)
    ids = [record["sample_id"] for record in records]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate cohort IDs")
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("Validation pass is not finalized: missing, extra, or duplicate results")
    successes, failures = [], []
    for record in records:
        value = record.get("score")
        if value is None:
            if not record.get("error"):
                raise ValueError("Undefined sample score requires a failure reason")
            failures.append({"sample_id": record["sample_id"], "reason": record["error"]})
        else:
            if record.get("error") or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Primary sample scores must be finite and on the 0–100 scale")
            successes.append(value)
    total = math.fsum(successes)
    return {"sum": total, "mean": total / len(successes) if successes else None,
            "expected": len(expected_ids), "attempted": len(ids),
            "successful": len(successes), "failed": len(failures),
            "partial_coverage": bool(failures), "failures": failures}


def validate_aggregate(value):
    if value["expected"] != value["attempted"] or (
        value["successful"] + value["failed"] != value["attempted"]
    ):
        raise ValueError("Cannot rank an unfinished validation pass")
    mean = value["mean"]
    if value["successful"] == 0:
        if mean is not None:
            raise ValueError("Zero valid samples must have an undefined mean")
    elif mean is None or not math.isfinite(mean) or not 0 <= mean <= 100:
        raise ValueError("Invalid aggregate score (expected 0–100)")
    elif not math.isclose(mean * value["successful"], value["sum"], abs_tol=1e-7):
        raise ValueError("Aggregate sum/count disagree with mean")


def aggregate_contract(value):
    """Scores and coverage affect selection; diagnostic wording/order does not."""
    validate_aggregate(value)
    return {**{key: value[key] for key in (
        "sum", "mean", "expected", "attempted", "successful", "failed", "partial_coverage")},
        "failed_sample_ids": sorted(f["sample_id"] for f in value["failures"])}


class CheckpointSelector:
    """Serializable ranking state; ineligible records never enter the retained set."""

    def __init__(self, identity, baseline, tolerance=3.0, keep=5):
        validate_aggregate(baseline)
        if baseline["mean"] is None:
            raise ValueError("Baseline clean PDMS is undefined")
        if tolerance != 3.0 or keep != 5:
            raise ValueError("Stage-3 selection requires tolerance=3.0 and keep=5")
        self.contract = {"identity": identity, "baseline": aggregate_contract(baseline),
                         "tolerance": tolerance, "keep": keep}
        self.history = []
        self.top = []

    @property
    def best(self):
        return self.top[0] if self.top else None

    def add(self, result, checkpoint, epoch, step, training_reward=None):
        if not result.get("finalized") or result["selection_identity"] != self.contract["identity"]:
            raise ValueError("Unfinalized scores or selection identity mismatch")
        clean, edited = result["clean"], result["edited"]
        validate_aggregate(clean)
        validate_aggregate(edited)
        if any(r["epoch"] == epoch for r in self.history):
            raise ValueError(f"Epoch {epoch} already has finalized selection results")
        defined = clean["mean"] is not None and edited["mean"] is not None
        eligible = defined and clean["mean"] >= self.contract["baseline"]["mean"] - self.contract["tolerance"]
        record = {"checkpoint": str(checkpoint), "epoch": int(epoch), "step": int(step),
                  "eligible": bool(eligible), "clean": clean, "edited": edited,
                  "training_reward": training_reward,
                  "selection_identity": self.contract["identity"],
                  "reason": "eligible" if eligible else ("clean PDMS below threshold" if defined else "undefined aggregate"),
                  "partial_coverage": clean["partial_coverage"] or edited["partial_coverage"]}
        self.history.append(record)
        previous = list(self.top)
        if eligible:
            self.top = sorted(self.top + [record], key=lambda r: (
                -r["edited"]["mean"], -r["clean"]["mean"], r["epoch"]
            ))[:self.contract["keep"]]
        evicted = [r["checkpoint"] for r in previous if r not in self.top]
        return record, evicted

    def state_dict(self):
        return {"contract": self.contract, "history": self.history, "top": self.top}

    def load_state_dict(self, state):
        if state["contract"] != self.contract:
            raise ValueError("Selection identity mismatch on resume; start a new selection run")
        self.history, self.top = state["history"], state["top"]

    def publish(self, directory, context, latest=None):
        directory = Path(directory)
        write_json(directory / "selected_checkpoint.json", {
            **context, **self.contract, "selected": self.best,
            "eligible_top_five": self.top, "latest_checkpoint": latest,
            "last_scored_epoch": self.history[-1]["epoch"] if self.history else None,
            "status": "selected" if self.best else "no eligible best checkpoint",
        })
        columns = ["checkpoint", "epoch", "step", "eligible", "reason", "training_reward",
                   "clean_PDMS", "edited_NC_all_x_DAC", "partial_coverage", "selection_identity",
                   "baseline_clean_PDMS", "baseline_partial_coverage", "tolerance", "seed",
                   "input_identity", "manifest_identity", "retained_in_top_five"]
        for group in ("clean", "edited"):
            columns += [f"{group}_{name}" for name in ("expected", "attempted", "successful", "failed")]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for record in self.history:
            row = {key: record[key] for key in columns if key in record}
            row.update(clean_PDMS=record["clean"]["mean"], edited_NC_all_x_DAC=record["edited"]["mean"])
            row.update(baseline_clean_PDMS=self.contract["baseline"]["mean"],
                       baseline_partial_coverage=self.contract["baseline"]["partial_coverage"],
                       tolerance=self.contract["tolerance"], retained_in_top_five=record in self.top,
                       **{key: context.get(key) for key in ("seed", "input_identity", "manifest_identity")})
            for group in ("clean", "edited"):
                row.update({f"{group}_{key}": record[group][key]
                            for key in ("expected", "attempted", "successful", "failed")})
            writer.writerow(row)
        atomic_write(directory / "checkpoint_scores.csv", buffer.getvalue())
