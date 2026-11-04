"""Locations of datasets, checkpoints and outputs.

Every location is a key in ``configs/paths.yaml`` (start from ``configs/paths.example.yaml``).
Any key can be overridden with the environment variable ``REACTBENCH_<KEY>`` (upper case), and
``REACTBENCH_PATHS`` selects a different configuration file. Relative paths resolve against the
repository root.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"


@lru_cache(maxsize=1)
def _config() -> dict:
    path = Path(os.environ.get("REACTBENCH_PATHS", DEFAULT_CONFIG)).expanduser()
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping of path keys")
    return value


def resolve(value) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def get(key: str, default=None, *, required: bool = True) -> Path | None:
    """Return the configured location for ``key``."""
    value = os.environ.get(f"REACTBENCH_{key.upper()}") or _config().get(key) or default
    if value is None:
        if required:
            raise KeyError(f"Set '{key}' in configs/paths.yaml or REACTBENCH_{key.upper()}")
        return None
    return resolve(value)


def outputs(*parts: str) -> Path:
    """A location under the outputs root (default ``outputs/``)."""
    return get("outputs", "outputs").joinpath(*parts)


# NAVSIM's standard layout under ``navsim_data`` (OPENSCENE_DATA_ROOT): navsim_logs/<split>,
# sensor_blobs/<split> and maps/. Each part can also be set on its own.
def navsim_logs(split: str = "test") -> Path:
    return get(f"navsim_logs_{split}", get("navsim_data", required=False) and get("navsim_data") / "navsim_logs" / split)


def navsim_sensor_blobs(split: str = "test") -> Path:
    return get(f"navsim_sensor_blobs_{split}",
               get("navsim_data", required=False) and get("navsim_data") / "sensor_blobs" / split)


def nuplan_maps() -> Path:
    return get("nuplan_maps", get("navsim_data", required=False) and get("navsim_data") / "maps")


def main(argv=None) -> None:
    """Print a location for shell scripts: ``python -m reactbench.paths KEY [DEFAULT]``.

    KEY is a configs/paths.yaml key, ``outputs[/SUB/PATH]``, ``navsim_logs:SPLIT``,
    ``navsim_sensor_blobs:SPLIT`` or ``nuplan_maps``.
    """
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if not 1 <= len(args) <= 2:
        raise SystemExit(main.__doc__)
    key, default = args[0], (args[1] if len(args) == 2 else None)
    name, _, part = key.partition(":")
    if name == "outputs" or name.startswith("outputs/"):
        value = outputs(*name.split("/")[1:])
    elif name in ("navsim_logs", "navsim_sensor_blobs"):
        value = (navsim_logs if name == "navsim_logs" else navsim_sensor_blobs)(part or "test")
    elif name == "nuplan_maps":
        value = nuplan_maps()
    else:
        value = get(name, default)
    print(value)


if __name__ == "__main__":
    main()
