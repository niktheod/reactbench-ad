"""NAVSIM annotations: per token, the camera paths and ego history the model runners read.

    python -m reactbench.datasets.navsim_annotations data/reactnav/reactnav.yaml

reads the scene filter (a NAVSIM SceneFilter YAML) and writes ``<outputs>/annotations/<stem>.json``,
a list of {token, front_camera, front_left_camera, front_right_camera, velocity, acceleration,
past_trajectory, navigation_command}, with four history frames each.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from .. import paths

if TYPE_CHECKING:
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader


def configure_navsim_environment() -> None:
    os.environ["NUPLAN_MAPS_ROOT"] = str(paths.nuplan_maps())
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return data


def load_scene_filter(scene_filter_path: Path) -> "SceneFilter":
    from navsim.common.dataclasses import SceneFilter

    cfg = read_yaml(scene_filter_path)
    return SceneFilter(
        num_history_frames=int(cfg["num_history_frames"]),
        num_future_frames=int(cfg["num_future_frames"]),
        frame_interval=cfg.get("frame_interval"),
        has_route=bool(cfg.get("has_route", True)),
        max_scenes=cfg.get("max_scenes"),
        log_names=cfg.get("log_names"),
        tokens=cfg.get("tokens"),
    )


def build_loader(split: str, scene_filter: "SceneFilter") -> "SceneLoader":
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    sensor_config = SensorConfig(cam_f0=True, cam_l0=True, cam_l1=False, cam_l2=False, cam_r0=True,
                                 cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)
    return SceneLoader(
        data_path=paths.navsim_logs(split),
        sensor_blobs_path=paths.navsim_sensor_blobs(split),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def vector2(value: Any) -> List[float]:
    values = as_jsonable(value)
    return [float(values[0]), float(values[1])]


def pose3(value: Any) -> List[float]:
    values = as_jsonable(value)
    return [float(values[0]), float(values[1]), float(values[2])]


def build_sample(loader: "SceneLoader", token: str) -> Dict[str, Any]:
    agent_input = loader.get_agent_input_from_token(token)
    return {
        "token": token,
        "front_camera": [str(cameras.cam_f0.image) for cameras in agent_input.cameras],
        "front_left_camera": [str(cameras.cam_l0.image) for cameras in agent_input.cameras],
        "front_right_camera": [str(cameras.cam_r0.image) for cameras in agent_input.cameras],
        "velocity": [vector2(status.ego_velocity) for status in agent_input.ego_statuses],
        "acceleration": [vector2(status.ego_acceleration) for status in agent_input.ego_statuses],
        "past_trajectory": [pose3(status.ego_pose) for status in agent_input.ego_statuses],
        "navigation_command": as_jsonable(agent_input.ego_statuses[-1].driving_command),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene_filter", type=Path, help="NAVSIM SceneFilter YAML")
    parser.add_argument("--split", default="test", help="Log split: test (navtest) or trainval (navtrain)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <outputs>/annotations")
    args = parser.parse_args(argv)
    configure_navsim_environment()
    loader = build_loader(args.split, load_scene_filter(args.scene_filter))
    samples = [build_sample(loader, token) for token in loader.tokens]
    output_dir = args.output_dir or paths.get("annotations", paths.outputs("annotations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.scene_filter.stem}.json"
    output_path.write_text(json.dumps(samples, indent=2))
    print(f"Wrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
