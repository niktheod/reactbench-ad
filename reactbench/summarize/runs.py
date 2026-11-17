"""Locate scored runs.

By default the runs of a model on a dataset are the run directories under
``<outputs>/predictions/<model>/<dataset>/`` (one per variant, plus ``original``); ``--runs MODEL:DATASET=DIR``
points one (model, dataset) at another directory.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from ..variants import parse_variant_name
from ..scoring.rows import read_rows

DATASETS = {
    "reactnav": {"dataset": "navsim", "prefix": "navsim_at_fault", "primary": "nc_dac", "label": "ReactNAV"},
    "reactpai": {"dataset": "pai-av", "prefix": "paiav_at_fault", "primary": "nc", "label": "ReactPAI"},
}
MODELS = {"reactnav": ("alpamayo1_5", "alpamayo2", "autovla", "recogdrive", "drivewam", "simwam"),
          "reactpai": ("alpamayo1_5", "alpamayo2")}
LABELS = {"alpamayo1_5": "Alpamayo 1.5", "alpamayo2": "Alpamayo 2", "autovla": "AutoVLA",
          "autovla_l0r0": "AutoVLA (CAM_L0/R0)",
          "recogdrive": "ReCogDrive", "drivewam": "DriveWAM", "simwam": "SimWAM",
          "recogdrive_vlm": "ReCogDrive (VLM SFT)", "recogdrive_il": "ReCogDrive (IL)",
          "recogdrive_il_rt": "ReCogDrive (IL + RT)", "simwam_il": "SimWAM (IL)"}


class Runs:
    def __init__(self, overrides=None):
        """``overrides``: {(model, dataset): directory}."""
        self.overrides = {key: Path(value) for key, value in (overrides or {}).items()}
        self._rows = {}

    @classmethod
    def from_arguments(cls, values):
        overrides = {}
        for value in values or ():
            key, _, directory = value.partition("=")
            model, _, dataset = key.partition(":")
            if not directory or not dataset:
                raise ValueError(f"--runs expects MODEL:DATASET=DIR, got {value!r}")
            overrides[model, dataset] = directory
        return cls(overrides)

    def directory(self, model, dataset):
        return self.overrides.get((model, dataset), paths.outputs("predictions", model, dataset))

    def insertion_runs(self, model, dataset, *, frame_dataset=None):
        """{(object, tau_s): run directory} of the scored insertion runs."""
        root = self.directory(model, dataset)
        found = {}
        if not root.is_dir():
            return found
        for run in sorted(root.iterdir()):
            parsed = parse_variant_name(run.name, frame_dataset or DATASETS[dataset]["dataset"])
            if parsed and run.is_dir():
                obj, _, tau = parsed
                if (obj, tau) in found:
                    raise ValueError(f"Two runs for {obj} at tau={tau} s under {root}")
                found[obj, tau] = run
        return found

    def rows(self, run, dataset):
        """Verified per-sample rows of a scored run (cached)."""
        run = Path(run)
        if run not in self._rows:
            rows, manifest = read_rows(run, DATASETS[dataset]["prefix"])
            if rows is None:
                raise FileNotFoundError(f"{run} is not scored; run reactbench.scoring first")
            if manifest.get("partial_selection"):
                raise ValueError(f"{run} was scored on a partial token selection")
            self._rows[run] = rows
        return self._rows[run]
