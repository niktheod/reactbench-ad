"""Names of ReactBench runs: one run per (dataset, inserted object, look-ahead time).

A compositor run, and every model run evaluated on it, is named ``<object>_tau<tau>s``, e.g. ``car_tau1.0s``: the
object is inserted where the ego vehicle's front bumper is after the look-ahead time tau on the logged trajectory.
The compositors count tau in frames, k = tau / frame interval (0.5 s on NAVSIM, 0.1 s on PAI-AV). Original
(unedited) scenes are in the run ``original``; the unseen-objects set (standing animals, paper section 5) uses
``standing_animal_tau<tau>s`` runs next to the ReactNAV ones.

``python -m reactbench.runs DATASET OBJECT TAU`` prints the run name and k (used by the compositing launchers).
"""
from __future__ import annotations

import argparse
import re

OBJECTS = ("crossing_pedestrian", "standing_pedestrian", "car")
OBJECT_LABELS = {"crossing_pedestrian": "Crossing pedestrian", "standing_pedestrian": "Standing pedestrian",
                 "car": "Car", "standing_animal": "Standing animal"}
FRAME_INTERVAL_S = {"navsim": 0.5, "pai-av": 0.1}
TAUS_S = (0.5, 1.0, 1.5, 2.0)
_RUN = re.compile(r"(?P<object>crossing_pedestrian|standing_pedestrian|car|standing_animal)_tau(?P<tau>\d+\.\d)s")


def timestep(tau_s: float, dataset: str) -> int:
    """k: the look-ahead time tau in frames of ``dataset``."""
    steps = tau_s / FRAME_INTERVAL_S[dataset]
    if steps < 0.5 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"tau={tau_s} s is not a positive whole number of {dataset} frames")
    return round(steps)


def run_name(obj: str, tau_s: float, dataset: str) -> str:
    timestep(tau_s, dataset)
    return f"{obj}_tau{tau_s:.1f}s"


def parse_run_name(name: str, dataset: str):
    """(object, timestep k, tau in seconds) for an insertion run name, else None."""
    match = _RUN.fullmatch(name)
    if not match:
        return None
    tau = float(match.group("tau"))
    return match.group("object"), timestep(tau, dataset), tau


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print the run name and the timestep k of a look-ahead time.")
    parser.add_argument("dataset", choices=sorted(FRAME_INTERVAL_S))
    parser.add_argument("object", choices=sorted(OBJECT_LABELS))
    parser.add_argument("tau_s", type=float, help="look-ahead time in seconds")
    args = parser.parse_args(argv)
    print(run_name(args.object, args.tau_s, args.dataset), timestep(args.tau_s, args.dataset))


if __name__ == "__main__":
    main()
