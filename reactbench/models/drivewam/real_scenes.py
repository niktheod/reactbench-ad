"""DriveWAM on the real-scenes test set (appendix control; ``models/recogdrive/real_scenes.py``).

    python -m reactbench.models.recogdrive.real_scenes annotations         # once, recogdrive environment
    python -m reactbench.models.drivewam.real_scenes infer --name drivewam # GPU (drivewam environment), then scores

The scenes, run layout (``<outputs>/real_scenes/runs/<scene file>/<name>/``), predictions format and scoring are
the ReCogDrive module's. DriveWAM reads the same annotation fields as ReCogDrive, so a headway replaces the
current velocity and acceleration and the four past poses with the injected ones; the current front-camera image
and the driving command stay real. Every headway of a scene shares the token's rollout seed, as an
original/edited pair does in ``eval.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...io import file_digest
from ..recogdrive import real_scenes as scenes_module
from ..runner import config_path, read_config, trajectory_poses
from ..seeds import paired_seed
from .eval import SEED_NAMESPACE, make_input

HERE = Path(__file__).parent


def infer(args):
    import torch
    from hydra.utils import instantiate
    output = scenes_module.run_dir(args.scenes, args.name)
    if (output / "predictions.json").exists():
        raise FileExistsError(f"{output} already holds predictions; choose a new --name")
    path = config_path(args.config, HERE)
    config = read_config(path)
    config.pop("name", None)
    scenes, sources, conditions = scenes_module.load_scenes([args.scenes])
    records = scenes_module.scene_annotations(scenes, sources[0])
    agent = instantiate(config)
    agent.initialize()
    predictions = {}
    for index, scene in enumerate(scenes, 1):
        token = scene["token"]
        seed = paired_seed(SEED_NAMESPACE, agent.seed, token)
        for condition in conditions:
            record = scenes_module.injected_annotation(records[token], scene["headways"][condition])
            poses = trajectory_poses(agent.compute_trajectory(make_input(record, Path(record["front_camera"][-1])),
                                                              rollout_seed=seed))
            predictions.setdefault(token, {})[condition] = np.asarray(poses, dtype=float).tolist()
        if index % 50 == 0 or index == len(scenes):
            print(f"[{index}/{len(scenes)}] scenes", flush=True)
    value = {"protocol": scenes_module.INFERENCE_PROTOCOL, "scenes": sources,
             "config": {"path": str(path), "sha256": file_digest(path)}, "model_family": "drivewam",
             "diffusion_seed_base": agent.seed, "conditions": conditions,
             "checkpoints": {args.name: {"path": str(config["checkpoint_path"])}},
             "reference": args.name, "predictions": {args.name: predictions}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(value) + "\n")
    (output / "run_info.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=1) + "\n")
    print(f"Wrote {output / 'predictions.json'}", flush=True)
    if not args.no_score:
        scenes_module.run_scoring_backend(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("infer", help="Run DriveWAM on a scene file (GPU), then score")
    p.add_argument("--scenes", type=Path, default=scenes_module.default_scenes())
    p.add_argument("--name", default="drivewam")
    p.add_argument("--config", default="config", help="DriveWAM configuration (a name in drivewam/ or a path)")
    p.add_argument("--no-score", action="store_true")
    p.set_defaults(function=infer)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
