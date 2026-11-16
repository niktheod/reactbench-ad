"""SimWAM on the real-scenes test set (appendix control; ``models/recogdrive/real_scenes.py``).

    python -m reactbench.models.recogdrive.real_scenes annotations        # once, recogdrive environment
    python -m reactbench.models.simwam.real_scenes infer --name simwam    # GPU (simwam environment), then scores

The scenes, run layout (``<outputs>/real_scenes/runs/<scene file>/<name>/``), predictions format and scoring are
the ReCogDrive module's. SimWAM's state has no history, so a headway replaces only the current velocity and
acceleration ([v, 0] and 0) of the real annotation record; the current front-camera image and the driving
command stay real. Every headway of a scene shares the token's action-noise seed, as an original/edited pair
does in ``eval.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...io import file_digest
from ..recogdrive import real_scenes as scenes_module
from ..runner import trajectory_poses
from .eval import AGENT_TARGET, HERE, hardware_class
from .navsim_loader import ego_state


def model_input(record, scene, condition):
    """SimWAM's input under one headway: the real current image and the injected 8-D ego state."""
    injected = scenes_module.injected_annotation(record, scene["headways"][condition])
    return {"image_path": Path(record["front_camera"][-1]), "state": ego_state(injected)}


def infer(args):
    from ..runner import config_path, read_config
    from .agent import SimWAMAgent
    output = scenes_module.run_dir(args.scenes, args.name)
    if (output / "predictions.json").exists():
        raise FileExistsError(f"{output} already holds predictions; choose a new --name")
    path = config_path(args.config, HERE / "configs")
    config = read_config(path)
    config.pop("name", None)
    if config.pop("_target_", None) != AGENT_TARGET:
        raise ValueError(f"SimWAM configuration must have _target_: {AGENT_TARGET}")
    scenes, sources, conditions = scenes_module.load_scenes([args.scenes])
    records = scenes_module.scene_annotations(scenes, sources[0])
    agent = SimWAMAgent(**config)
    agent.prepare()
    agent.load()
    predictions = {}
    for index, scene in enumerate(scenes, 1):
        token = scene["token"]
        seed = agent.rollout_seed(token)
        for condition in conditions:
            poses = trajectory_poses(agent.compute_trajectory(model_input(records[token], scene, condition),
                                                              rollout_seed=seed))
            predictions.setdefault(token, {})[condition] = np.asarray(poses, dtype=float).tolist()
        if index % 50 == 0 or index == len(scenes):
            print(f"[{index}/{len(scenes)}] scenes", flush=True)
    value = {"protocol": scenes_module.INFERENCE_PROTOCOL, "scenes": sources,
             "config": {"path": str(path), "sha256": file_digest(path)}, "model_family": "simwam",
             "diffusion_seed_base": agent.seed, "conditions": conditions,
             "checkpoints": {args.name: {"path": str(agent.checkpoint_path), "identity": agent.identity()}},
             "reference": args.name, "predictions": {args.name: predictions}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(value) + "\n")
    (output / "run_info.json").write_text(json.dumps({"gpu": hardware_class(agent.device)}, indent=1) + "\n")
    print(f"Wrote {output / 'predictions.json'}", flush=True)
    if not args.no_score:
        scenes_module.run_scoring_backend(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("infer", help="Run SimWAM on a scene file (GPU), then score")
    p.add_argument("--scenes", type=Path, default=scenes_module.default_scenes())
    p.add_argument("--name", default="simwam", help="Run name (default simwam, the RL checkpoint)")
    p.add_argument("--config", default="rl", help="SimWAM configuration: rl, il or a path")
    p.add_argument("--no-score", action="store_true")
    p.set_defaults(function=infer)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
