"""Tables of the training analyses: training stages (section 4.3), fine-tuning on ReactTrain and
generalization (section 5), the real-scenes control of the released single-frame models and the GRPO
exploration probe (appendix), and the data statistics."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .. import paths
from ..variants import OBJECTS, TAUS_S
from ..scoring.reporting import comparison_summary, pair_records
from .runs import DATASETS

LOW_SPEED_BINS = ("[5,10)", "[10,15)", "[15,20)")
# Training-stage checkpoints and their navtest PDMS as reported by the model papers.
STAGES = {"ReCogDrive": (("VLM SFT", "recogdrive_vlm", 84.1), ("IL", "recogdrive_il", 86.5),
                         ("RL", "recogdrive", 90.8)),
          "SimWAM": (("IL", "simwam_il", 90.3), ("RL", "simwam", 91.9))}
STAGE_SEED = 20261030
# Real-scenes runs (``python -m reactbench.models.<model>.real_scenes infer --name NAME``) of the generalization
# table (section 5) and of the control with the released single-frame models (appendix).
GENERALIZATION_RUNS = ("recogdrive_il", "recogdrive_il_rt")
CONTROL_RUNS = ("recogdrive", "drivewam", "simwam")
# Original-scene plans of IL + RT that are >= 20% shorter (4 s distance) than IL's (section 5).
SHORTER = 0.8
# Object names in the ReactTrain sample IDs (<source token>__hazardous_<name>_t<k>).
PROBE_STRATA = {"crossing_pedestrian": "walking", "standing_pedestrian": "standing", "car": "car"}


def stage_bootstrap(values, reps=10_000, seed=STAGE_SEED, chunk=1000):
    """Percentile 95% interval of the mean (the training-stage table's own resampling stream)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([values[rng.integers(0, len(values), (min(chunk, reps - i), len(values)))].mean(1)
                            for i in range(0, reps, chunk)])
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def _token_deltas(rows):
    """token -> (speed bin, dP in pp) over the valid original/edited pairs."""
    result = {}
    for p in pair_records(rows, "nc_dac"):
        token = str(p["row"]["token"])
        if token in result:
            raise ValueError(f"Duplicate token {token}")
        result[token] = (p["row"]["groups"]["speed_bin_kmh"], 100.0 * (p["primary"][1] - p["primary"][0]))
    return result


def training_stages(runs, tau_s=2.0):
    """dP below 20 km/h per checkpoint; gray when the paired CI of the change from the previous stage includes 0."""
    records = []
    for model, stages in STAGES.items():
        for obj in OBJECTS:
            previous = None
            for stage, name, pdms in stages:
                run = runs.insertion_runs(name, "reactnav").get((obj, tau_s))
                if run is None:
                    previous = None
                    continue
                rows = _token_deltas(runs.rows(run, "reactnav"))
                low = [d for b, d in rows.values() if b in LOW_SPEED_BINS]
                high = [d for b, d in rows.values() if b not in LOW_SPEED_BINS]
                record = {"model": model, "stage": stage, "checkpoint": name, "object": obj, "pdms": pdms,
                          "n_low": len(low), "delta_p_low_pp": float(np.mean(low)),
                          "n_high": len(high), "delta_p_high_pp": float(np.mean(high)) if high else None,
                          "change_from_previous_pp": None, "change_ci95_pp": None}
                if previous is not None:
                    common = sorted(t for t, (b, _) in rows.items() if b in LOW_SPEED_BINS and t in previous)
                    diff = [rows[t][1] - previous[t][1] for t in common]
                    record.update(change_from_previous_pp=float(np.mean(diff)), change_ci95_pp=stage_bootstrap(diff),
                                  n_common=len(common))
                records.append(record)
                previous = rows
    return records


def _mean_over_runs(runs, name, objects, taus=TAUS_S, tokens=None):
    """Mean over the (object, tau) variants of dP (pp) and the contact-speed change (km/h)."""
    deltas, contacts, used = [], [], []
    for obj in objects:
        for tau in taus:
            run = runs.insertion_runs(name, "reactnav").get((obj, tau))
            if run is None:
                raise FileNotFoundError(f"Missing {name} run for {obj} at tau={tau} s")
            rows = runs.rows(run, "reactnav")
            if tokens is not None:
                rows = [r for r in rows if r["token"] in tokens]
            stat = comparison_summary([p["row"] for p in pair_records(rows, "nc_dac")], "nc_dac")
            deltas.append(stat["primary"]["improvement_pp"])
            contact = stat["contact_speed"]["mean_delta_mps"]
            contacts.append(None if contact is None else 3.6 * contact)
            used.append(str(run))
    return {"delta_p_pp": float(np.mean(deltas)),
            "delta_cs_kmh": float(np.mean([c for c in contacts if c is not None])) if any(
                c is not None for c in contacts) else None,
            "runs": used}


def navtest_pdms(name):
    """PDMS and sub-metrics ('average' row) of a NAVSIM navtest evaluation (CSV written by run_pdm_score)."""
    root = paths.outputs("navtest_pdms", name)
    files = sorted(root.rglob("*.csv"))
    if not files:
        return None
    import csv
    with files[-1].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    average = next((r for r in rows if r.get("token") == "average"), None)
    if average is None:
        return None
    keys = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
            "ego_progress", "comfort", "score")
    return {k: 100. * float(average[k]) for k in keys if k in average} | {"source": str(files[-1])}


def fine_tuning(runs, checkpoints=("recogdrive_il", "recogdrive_il_rt")):
    """ReactNAV dP / dCS (mean over the 12 variants) and navtest PDMS of IL and IL + RT."""
    return [{"checkpoint": name, **_mean_over_runs(runs, name, OBJECTS), "navtest": navtest_pdms(name)}
            for name in checkpoints]


def plan_motion(poses, interval_s, start_speed_mps):
    """4 s distance (m) of a plan from the ego origin, and its largest deceleration (m/s^2): mean speed per
    interval from the waypoints, decelerations between consecutive intervals and from the measured start speed."""
    xy = np.asarray(poses, dtype=float)[:int(round(4.0 / interval_s)), :2]
    steps = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], xy]), axis=0), axis=1)
    speeds = steps / interval_s
    centres = np.concatenate(([interval_s / 2], np.full(len(speeds) - 1, interval_s)))  # interval centres
    return float(steps.sum()), float(max((-np.diff(np.concatenate(([start_speed_mps], speeds))) / centres).max(), 0.))


def original_plan_shortening(runs, checkpoints=("recogdrive_il", "recogdrive_il_rt")):
    """Original ReactNAV scenes: share of scenes whose IL + RT plan covers >= 20% less distance in 4 s than IL's,
    and each checkpoint's largest deceleration. Needs the original predictions (not part of the released
    scores); None without them. Shares are given on all scenes and without the scenes whose logged future has
    irregular timestamps (``timestamp_warning``); the paper's 0.9% is the latter."""
    folders = [runs.directory(name, "reactnav") / "original" / "trajectories" for name in checkpoints]
    run = runs.insertion_runs(checkpoints[0], "reactnav").get(("car", TAUS_S[0]))
    if run is None or not all(folder.is_dir() for folder in folders):
        return None
    rows = [r for r in runs.rows(run, "reactnav") if r.get("status") == "scored"]
    tokens = sorted({r["token"] for r in rows})
    start = {r["token"]: float(r["groups"]["speed_mps"]) for r in rows}
    warned = {r["token"] for r in rows if r.get("timestamp_warning")}
    motion = {}
    for name, folder in zip(checkpoints, folders):
        for token in tokens:
            with np.load(folder / f"{token}.npz") as archive:
                motion[name, token] = plan_motion(archive["trajectory"], float(archive["interval_length_s"]),
                                                  start[token])
    il, rt = checkpoints
    shorter = {t for t in tokens if motion[rt, t][0] <= SHORTER * motion[il, t][0]}
    kept = [t for t in tokens if t not in warned]
    return {"checkpoints": list(checkpoints), "n": len(tokens), "shorter_n": len(shorter),
            "shorter_pct": 100. * len(shorter) / len(tokens),
            "n_without_timestamp_warning": len(kept), "shorter_n_without_timestamp_warning": len(shorter & set(kept)),
            "shorter_pct_without_timestamp_warning": 100. * len(shorter & set(kept)) / len(kept),
            "max_deceleration_mps2": {name: max(motion[name, t][1] for t in tokens) for name in checkpoints}}


def unseen_objects(runs, checkpoints=("recogdrive_il", "recogdrive_il_rt")):
    """Standing animals: mean over tau on the tokens present in every animal variant for every checkpoint."""
    token_sets = []
    for name in checkpoints:
        for tau in TAUS_S:
            run = runs.insertion_runs(name, "reactnav").get(("standing_animal", tau))
            if run is None:
                return []
            token_sets.append({p["row"]["token"] for p in pair_records(runs.rows(run, "reactnav"), "nc_dac")})
    common = set.intersection(*token_sets)
    return [{"checkpoint": name, "n_tokens": len(common),
             **_mean_over_runs(runs, name, ("standing_animal",), tokens=common)} for name in checkpoints]


def _real_scene_summary(summary_path=None):
    path = Path(summary_path or paths.outputs("real_scenes", "summary", "navtest_v2", "summary.json"))
    return json.loads(path.read_text()) if path.is_file() else None


def real_scenes(summary_path=None, names=GENERALIZATION_RUNS):
    """Real scenes: P and CS with the target frozen, mean over the four headways."""
    value = _real_scene_summary(summary_path)
    if value is None:
        return []
    return [{"checkpoint": name, "p_pct": value["mean_over_headways"][name]["p_frozen"],
             "cs_kmh": value["mean_over_headways"][name]["cs_kmh_frozen"]}
            for name in names if name in value["mean_over_headways"]]


def real_scene_control(summary_path=None, names=CONTROL_RUNS):
    """Real scenes, released single-frame models: P (target frozen) per headway and its mean."""
    value = _real_scene_summary(summary_path)
    if value is None:
        return []
    records = []
    for name in names:
        if name not in value["summary"]:
            continue
        per_headway = value["summary"][name]
        records.append({"model": name, "n": next(iter(per_headway.values()))["n"],
                        "p_pct": {h: v["p_frozen"] for h, v in per_headway.items()},
                        "mean_p_pct": value["mean_over_headways"][name]["p_frozen"]})
    return records


def exploration_probe(path=None):
    """Share of GRPO groups with a non-zero gradient (Signal) and with a safe, above-mean candidate (Safe signal)."""
    path = Path(path or paths.outputs("exploration_probe", "noise_0.04_reactbench.json"))
    if not path.is_file():
        return []
    groups = json.loads(path.read_text())["candidates"]
    records = []
    for label, members in [(obj, [g for g in groups if (g.get("stratum") or "").startswith(f"{PROBE_STRATA[obj]}_")])
                           for obj in OBJECTS] + [("all", groups)]:
        record = {"object": label, "n_groups": len(members)}
        for k in (8, 16, 32):
            signal = [np.ptp(g["scores"]["executed"][:k]) > 0 for g in members]
            safe = [any(p == 1 and r > np.mean(g["scores"]["executed"][:k])
                        for p, r in zip(g["reactbench"]["p"][:k], g["scores"]["executed"][:k])) for g in members]
            record[f"signal_g{k}_pct"] = 100. * float(np.mean(signal)) if members else None
            record[f"safe_signal_g{k}_pct"] = 100. * float(np.mean(safe)) if members else None
        records.append(record)
    return records


def speed_distribution(runs, dataset, model):
    """Number of samples per 5 km/h ego-speed bin at the current frame (any insertion run of one model)."""
    found = runs.insertion_runs(model, dataset)
    if not found:
        return {}
    run = found[min(found)]
    rows = runs.rows(run, dataset)
    return dict(sorted(Counter(r["groups"].get("speed_bin_kmh") for r in rows if r.get("groups")).items(),
                       key=lambda item: float(str(item[0]).strip("[").split(",")[0])))


__all__ = ["DATASETS", "training_stages", "fine_tuning", "unseen_objects", "real_scenes", "exploration_probe",
           "speed_distribution"]
