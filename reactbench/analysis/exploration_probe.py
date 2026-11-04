"""GRPO exploration probe (appendix, Exploration Probe; CPU only).

Would ReCogDrive's RL stage learn to brake from hazardous ReactTrain scenes? Starting from the IL
checkpoint (the RL initialization), it samples a group of 32 GRPO candidates per scene, with RL's
sampling settings and a stochastic final denoising step, on 300 hazardous training scenes (50 per
object x tau in {1.5, 2.0} s). Each candidate gets the RL reward (PDMS of the LQR-executed
trajectory) and ReactBench's P = NC x DAC of the raw trajectory, scored as on ReactNAV: at-fault
collisions in the source scene's recorded world plus the inserted object's motion. The groups of
8 and 16 are the first candidates of each group of 32. ``python -m reactbench.summarize --tables
exploration_probe`` turns the output into the table (Signal: unequal rewards in a group; Safe
signal: a P = 1 candidate with an above-mean reward).

    PYTHONPATH=third_party/recogdrive:. python -m reactbench.analysis.exploration_probe \
        --manifest <ReactTrain manifest.json>
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU sampling, as the paper run

import numpy as np

from .. import paths

VARIANT = re.compile(r"__(?P<kind>[a-z]+)_(?P<actor>[a-z]+)_t(?P<timestep>\d+)$")
RAW_INTERVAL_S = .5  # native waypoint spacing of the planner's candidates
NOISE_STD = 0.04  # GRPOConfig's sampling noise floor (min_sampling_denoising_std)


def default_checkpoint():
    return paths.get("checkpoints", "checkpoints") / "recogdrive" / "IL" / "ReCogDrive_Diffusion_Planner_2B_IL.ckpt"


def stratum(token, kind="hazardous"):
    """``<object>_t<timestep>`` of a ReactTrain sample ID ``<source>__<kind>_<object>_t<timestep>``, else None."""
    match = VARIANT.search(token)
    return f"{match['actor']}_t{match['timestep']}" if match and match["kind"] == kind else None


def select_tokens(population, scenes, seed, kind="hazardous"):
    """An equal seeded draw per object x timestep stratum, strata in sorted order."""
    cells = {}
    for token in population:
        name = stratum(token, kind)
        if name is None:
            raise ValueError(f"No {kind} object/timestep variant in sample ID {token}")
        cells.setdefault(name, []).append(token)
    if not cells or scenes % len(cells):
        raise ValueError(f"{scenes} scenes do not split evenly over the {len(cells)} strata {sorted(cells)}")
    per = scenes // len(cells)
    rng, tokens = np.random.default_rng(seed), []
    for name in sorted(cells):
        if len(cells[name]) < per:
            raise ValueError(f"Stratum {name} has only {len(cells[name])} samples, {per} needed")
        tokens += [cells[name][i] for i in rng.choice(len(cells[name]), per, replace=False)]
    return tokens, {name: per for name in sorted(cells)}


def load_planner(checkpoint):
    """The diffusion planner of a ReCogDrive checkpoint with GRPO's sampling settings."""
    import torch
    from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
        GRPOConfig, ReCogDriveDiffusionPlanner, planner_state_from_checkpoint,
    )
    config = make_recogdrive_config("small", action_dim=3, action_horizon=8, input_embedding_dim=384, grpo=False)
    config.vlm_size = "small"
    planner = ReCogDriveDiffusionPlanner(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    planner.load_state_dict(planner_state_from_checkpoint(planner, state), strict=False)
    grpo = GRPOConfig()
    for name in ("min_sampling_denoising_std", "randn_clip_value", "final_action_clip_value",
                 "denoised_clip_value", "eps_clip_value"):
        setattr(planner, name, getattr(grpo, name))
    return planner.eval()


def sample_candidates(planner, feature_directory, count):
    """``count`` GRPO candidates (stochastic final step) from a sample's cached VLM features."""
    import torch
    from navsim.common.dataclasses import Trajectory
    with gzip.open(Path(feature_directory) / "internvl_feature.gz", "rb") as handle:
        features = pickle.load(handle)
    repeat = lambda tensor: tensor.float()[None].repeat(count, *([1] * tensor.ndim))  # noqa: E731
    with torch.no_grad():
        _, poses = planner.sample_chain(repeat(features["last_hidden_state"]),
                                        repeat(features["history_trajectory"].reshape(-1)),
                                        repeat(features["status_feature"]), deterministic=False,
                                        deterministic_final_step=False)
    return [Trajectory(pose) for pose in poses.numpy()]


def reactbench_scores(poses, actor_motion, world, map_context):
    """NC, DAC and P = NC x DAC of every candidate's raw trajectory, as the ReactNAV evaluation scores them."""
    from ..scoring.navsim_batch import score_predictions
    from ..scoring.navsim_metrics import at_fault_from_physical, score_prediction_at_fault
    poses = np.asarray(poses, dtype=np.float64)
    if np.linalg.norm(poses[..., :2], axis=-1).max() + 10 > world["query_radius_m"]:  # the evaluation's world margin
        raise ValueError("A candidate leaves the recorded world's query radius")
    try:  # batched physical contacts equal the scalar reference; like the evaluation, fall back per candidate
        physical = score_predictions(poses, RAW_INTERVAL_S, actor_motion, world)
    except Exception:
        physical = [None] * len(poses)
    scores = [score_prediction_at_fault(q, RAW_INTERVAL_S, actor_motion, world, map_context) if p is None else
              at_fault_from_physical(p, q, RAW_INTERVAL_S, actor_motion, world, map_context)
              for p, q in zip(physical, poses)]
    if any(s["nc_dac"] is None for s in scores):
        raise ValueError("ReactBench DAC unavailable")
    return {key: [int(s[name]) for s in scores] for key, name in (("nc", "nc"), ("dac", "dac"), ("p", "nc_dac"))}


def source_metric_files(root):
    """{token: metric cache file} of the source (unedited) training scenes."""
    import csv
    found = sorted((Path(root) / "metadata").glob("*metadata_node_0.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one metric-cache index under {root}/metadata")
    with found[0].open(newline="") as handle:
        return {Path(row["file_name"]).parent.name: row["file_name"] for row in csv.DictReader(handle)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True, help="ReactTrain manifest (training samples)")
    parser.add_argument("--source-metric-cache", type=Path, default=None,
                        help="navtrain metric cache of the source scenes (lane polygons; default: "
                             "the navtrain_metric_cache path)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: the released IL checkpoint")
    parser.add_argument("--scenes", type=int, default=300)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Stop after the first N selected scenes (a check)")
    args = parser.parse_args(argv)
    from ..scoring.navsim import load_config
    config = load_config()
    os.environ.setdefault("NUPLAN_MAPS_ROOT", config["maps_root"])

    import torch
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_scores
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from ..models.recogdrive.real_scenes import split_config
    from ..scoring.navsim_lanes import extract_one
    from ..scoring.navsim_metrics import MapContext
    from ..scoring.navsim_world import RawWorlds

    # The global seed precedes the planner's construction, whose initialization draws from it.
    torch.manual_seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    population = sorted(t for t, s in manifest["samples"].items() if s["split"] == "train" and s["kind"] == "hazardous")
    tokens, strata = select_tokens(population, args.scenes, args.seed)
    loader = MetricCacheLoader(Path(manifest["metric_cache"]))
    planner = load_planner(args.checkpoint or default_checkpoint())
    planner.min_sampling_denoising_std = NOISE_STD  # read by sample_chain at every DDIM step
    scorer_config = GRPOConfig().scorer_config
    sampling = TrajectorySampling(time_horizon=4, interval_length=.1)
    worlds = RawWorlds(split_config(config, "trainval"))
    metric_files = source_metric_files(args.source_metric_cache or paths.get("navtrain_metric_cache"))
    lanes = tempfile.TemporaryDirectory()  # per-scene lane sidecars, removed at the end
    groups = []
    try:
        for index, token in enumerate(tokens[:args.limit], 1):
            sample = manifest["samples"][token]
            trajectories = sample_candidates(planner, sample["feature_path"], args.candidates)
            cache = loader.get_from_token(token)
            executed = pdm_scores(cache, trajectories, sampling, PDMSimulator(sampling), PDMScorer(sampling, scorer_config))
            source = sample.get("source_token", token)
            world = worlds.load(source)
            extract_one((source, metric_files[source], lanes.name))
            groups.append({"token": token, "stratum": stratum(token),
                           "scores": {"executed": [float(r.score) for r in executed]},
                           "reactbench": reactbench_scores([t.poses for t in trajectories], sample.get("actor_motion"),
                                                           world, MapContext(source, world, lanes.name))})
            if index % 25 == 0:
                print(f"{index}/{len(tokens)} scenes", flush=True)
    finally:
        lanes.cleanup()
    output = {"settings": {"manifest": args.manifest.name, "checkpoint": Path(args.checkpoint or default_checkpoint()).name,
                           "seed": args.seed, "candidates": args.candidates, "final_step": "stochastic",
                           "min_sampling_denoising_std": NOISE_STD, "strata": strata},
              "candidates": groups}
    target = args.output or paths.outputs("exploration_probe", "noise_0.04_reactbench.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=1) + "\n")
    print(f"Wrote {len(groups)} groups to {target}")


if __name__ == "__main__":
    main()
