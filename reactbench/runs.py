"""Names of ReactBench runs: one run per (dataset, inserted object, look-ahead time).

A compositor run, and every model run evaluated on it, is named ``<object>_timestep_<k>``: the
object is inserted at the ego vehicle's front-bumper position after ``k`` future frames of the
ground truth, i.e. at look-ahead time tau = k * frame interval (0.5 s on NAVSIM, 0.1 s on PAI-AV).
Original (unedited) scenes are in the run ``original``; the unseen-objects set (standing animals,
paper section 5) uses ``standing_animal_timestep_<k>`` runs next to the ReactNAV ones.
"""
from __future__ import annotations

import re

OBJECTS = ("crossing_pedestrian", "standing_pedestrian", "car")
OBJECT_LABELS = {"crossing_pedestrian": "Crossing pedestrian", "standing_pedestrian": "Standing pedestrian",
                 "car": "Car", "standing_animal": "Standing animal"}
FRAME_INTERVAL_S = {"navsim": 0.5, "pai-av": 0.1}
TAUS_S = (0.5, 1.0, 1.5, 2.0)
# "_contact_only" is accepted for runs named before contact-only shadows became the only mode.
_RUN = re.compile(r"^(?P<object>crossing_pedestrian|standing_pedestrian|car|standing_animal)_timestep_(?P<k>\d+)"
                  r"(_contact_only)?$")


def run_name(obj: str, tau_s: float, dataset: str) -> str:
    steps = tau_s / FRAME_INTERVAL_S[dataset]
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"tau={tau_s} s is not a whole number of {dataset} frames")
    return f"{obj}_timestep_{round(steps)}"


def parse_run_name(name: str, dataset: str):
    """(object, timestep k, tau in seconds) for an insertion run name, else None."""
    match = _RUN.fullmatch(name)
    if not match:
        return None
    k = int(match.group("k"))
    return match.group("object"), k, round(k * FRAME_INTERVAL_S[dataset], 3)
