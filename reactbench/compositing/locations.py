"""Default locations of the compositors, from configs/paths.yaml (see paths.example.yaml)."""
from __future__ import annotations

import sys
from pathlib import Path

from .. import paths


def optional(key: str, default=None) -> Path | None:
    """The configured location of ``key``, else ``default`` (None when neither is set)."""
    return paths.get(key, default, required=False)


# Blender 4.5.11 (scripts/setup_blender.sh) and the Python with torch/transformers for Depth Anything.
BLENDER = optional("blender", "third_party/blender-4.5.11-linux-x64/blender")
DEPTH_PYTHON = optional("depth_python") or Path(sys.executable)
MODEL_CACHE = optional("hf_home", paths.outputs("work", "hf_cache"))
# Generated assets (scripts/assets.sh): <assets>/{rocketbox,rocketbox_standing_animals,openx}/assets.
ASSETS = optional("assets", "assets")
NAVSIM_DATA = optional("navsim_data")
PAIAV_ROOT = optional("paiav_root")
REACTNAV = optional("reactnav", "datasets/reactnav")
REACTPAI = optional("reactpai", "datasets/reactpai")
REACTPAI_MANIFEST = optional("reactpai_manifest", "data/reactpai/selection_manifest.json")
