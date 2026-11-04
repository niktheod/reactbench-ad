"""Original-image lookup and frozen history-free features for joint dropout."""
import hashlib
import json
from pathlib import Path
import pickle
import time

import torch

from .recogdrive_backbone import system_message
from .utils.internvl_preprocess import load_image

PROMPT_PROTOCOL = "recogdrive-history-free-internvl-v1"
HISTORY_FREE_SYSTEM = system_message.replace(
    "multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands",
    "a front camera image and discrete navigation commands").replace(
    "The input provides a 2-second history, and your output should ensure",
    "Your output should ensure")


def history_free_question(command):
    names = ["turn left", "go straight", "turn right"]
    name = next((names[i] for i, value in enumerate(command[:3]) if value == 1), "unknown")
    return ("<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
            "1. Visual perception from front camera view\n"
            f"2. Active navigation command: [{name.upper()}]"
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places")


def file_identity(path, content=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        result["sha256"] = hasher.hexdigest()
    return result


def build_image_lookup(cache_paths, log_root, sensor_root):
    """Read only metadata for logs containing the exact cached training tokens."""
    groups = {}
    for token, cache_path in cache_paths.items():
        groups.setdefault(Path(cache_path).parent.name, set()).add(token)
    lookup, sources = {}, []
    for log, wanted in sorted(groups.items()):
        path = Path(log_root) / f"{log}.pkl"
        sources.append(file_identity(path))
        with path.open("rb") as handle:
            frames = pickle.load(handle)
        for frame in frames:
            token = frame["token"]
            if token not in wanted:
                continue
            if token in lookup:
                raise ValueError(f"Duplicate training image token: {token}")
            camera = next(v for k, v in frame["cams"].items() if k.lower() == "cam_f0")
            lookup[token] = str((Path(sensor_root) / camera["data_path"]).resolve())
    missing = set(cache_paths) - lookup.keys()
    if missing:
        raise ValueError(f"Missing original images for {len(missing)} training tokens: {sorted(missing)[:5]}")
    return lookup, sources


def feature_contract(vlm_path, lookup, sources):
    root = Path(vlm_path)
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("Joint dropout requires local safetensors VLM weights")
    paths = sorted(set(weights + list(root.glob("*.json")) + list(root.glob("*.py")) + list(root.glob("*.model"))))
    code_root = Path(__file__).parent
    paths += [Path(__file__), code_root / "recogdrive_backbone.py",
              code_root / "utils/internvl_preprocess.py", code_root / "utils/conversation.py"]
    return {"prompt_protocol": PROMPT_PROTOCOL, "precision": "bfloat16; autocast disabled",
            "preprocessing": "internvl 448 max_num=12 thumbnail; fixed 2800 prompt tokens",
            "prompt_sha256": hashlib.sha256((HISTORY_FREE_SYSTEM + history_free_question([0, 1, 0, 0])).encode()).hexdigest(),
            "files": [file_identity(p, content=True) for p in paths], "sources": sources,
            "image_lookup_sha256": hashlib.sha256(json.dumps(lookup, sort_keys=True).encode()).hexdigest()}


def replace_dropped_features(backbone, cached, commands, tokens, mask, lookup):
    """One detached VLM forward for the selected subset; no cache writes."""
    indices = mask.squeeze(-1).nonzero(as_tuple=False).flatten()
    if not len(indices):
        return cached, {"joint_image_seconds": 0., "joint_vlm_seconds": 0.}
    before = time.perf_counter()
    selected = indices.tolist()
    pixels = [load_image(lookup[tokens[i]], max_num=12) for i in selected]
    questions = [history_free_question(commands[i].tolist()) for i in selected]
    counts = [len(p) for p in pixels]
    pixels = torch.cat(pixels).to(cached.device)
    loading = time.perf_counter() - before
    if cached.is_cuda:
        torch.cuda.synchronize(cached.device)
    before = time.perf_counter()
    backbone.eval()
    with torch.no_grad(), torch.autocast(device_type=cached.device.type, enabled=False):
        output = backbone(pixels, questions, num_patches_list=counts, system_prompt=HISTORY_FREE_SYSTEM)
        hidden = output.hidden_states[-1].detach().to(cached)
    if cached.is_cuda:
        torch.cuda.synchronize(cached.device)
    duration = time.perf_counter() - before
    if hidden.shape != (len(indices), cached.shape[1], cached.shape[2]) or not torch.isfinite(hidden).all():
        raise ValueError("History-free features must match the fixed cached sequence shape and be finite")
    mixed = cached.clone()
    mixed.index_copy_(0, indices, hidden)
    return mixed, {"joint_image_seconds": loading, "joint_vlm_seconds": duration}
