"""Shared stage-2 conditioning, input collation and checkpoint contracts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

DEFAULT_KV_LAYERS = (3, 6, 10, 13, 17, 20, 24, 27)


@dataclass(frozen=True)
class VLMConditioning:
    valid_mask: torch.Tensor
    kv_cache: tuple = ()
    query_position_ids: torch.Tensor | None = None
    query_rope: tuple | None = None


def collate_features(samples, load_images=False):
    """CPU collation shared by stage-2 training and validation; never runs a VLM.

    ``load_images`` preprocesses online images exactly as ``forward`` would, so DataLoader workers do it;
    samples already carrying ``pixel_values`` (validation workers) are concatenated without reloading.
    """
    batch = {key: torch.stack([sample[key] for sample in samples])
             for key in ("history_trajectory", "high_command_one_hot", "status_feature")}
    online = ["image_path_tensor" in sample for sample in samples]
    if any(online) != all(online):
        raise ValueError("Cannot mix online inputs and cached hidden states in one batch")
    key = "image_path_tensor" if all(online) else "last_hidden_state"
    batch[key] = torch.nn.utils.rnn.pad_sequence([s[key] for s in samples], batch_first=True).detach()
    if all(online) and (load_images or any("pixel_values" in s for s in samples)):
        from .utils import internvl_preprocess
        pixels = [s["pixel_values"] if "pixel_values" in s else
                  internvl_preprocess.load_image("".join(map(chr, s[key].tolist()))) for s in samples]
        batch["pixel_values"] = torch.cat(pixels)
        batch["num_patches"] = torch.tensor([p.shape[0] for p in pixels])
    masked = ["context_valid_mask" in s for s in samples]
    if any(masked):
        if not all(masked):
            raise ValueError("Every cached sample must supply its tokenizer mask")
        for sample in samples:
            mask = sample["context_valid_mask"]
            if mask.dtype != torch.bool or mask.shape != sample["last_hidden_state"].shape[:1]:
                raise ValueError("Cached tokenizer mask must be boolean [S]")
        batch["context_valid_mask"] = torch.nn.utils.rnn.pad_sequence(
            [s["context_valid_mask"] for s in samples], batch_first=True, padding_value=False)
    return batch


def vlm_file_identity(path):
    """Record config/code content and weight-file identity without rereading GBs per rank."""
    root = Path(path).expanduser().resolve()
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    files = {}
    for item in sorted(root.iterdir()):
        if item.is_file() and item.suffix in (".json", ".py", ".model", ".safetensors", ".bin"):
            stat = item.stat()
            files[item.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if item.suffix not in (".safetensors", ".bin"):
                files[item.name]["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
    return {"path": str(root), "files": files}


def planner_contract(config):
    model = config.diffusion_model_cfg
    mode = getattr(config, "conditioning_mode", "hidden_states")
    return {
        "version": 1, "conditioning_mode": mode,
        "mask_vlm_padding": getattr(config, "mask_vlm_padding", False),
        "vlm_kv_layers": list(getattr(config, "vlm_kv_layers", ())) if mode == "vlm_kv" else [],
        "geometry": {k: model[k] for k in ("num_heads", "head_dim", "num_layers", "output_dim")},
        "cross_attention": {k: model.get(k) for k in
                            ("cross_attention_heads", "cross_attention_kv_heads", "cross_attention_head_dim",
                             "cross_attention_q_norm", "vlm_rope_theta")} if mode == "vlm_kv" else {},
        "pooling": "unmasked_mean", "sampling_method": config.sampling_method,
        "vlm_identity": getattr(config, "vlm_identity", None),
    }


def validate_conditioning_checkpoint(config, checkpoint):
    saved = checkpoint.get("vlm_conditioning")
    if saved is None:
        if getattr(config, "conditioning_mode", "hidden_states") != "hidden_states" or getattr(config, "mask_vlm_padding", False):
            raise ValueError("Checkpoint has no stage-2 VLM conditioning contract; incompatible conditioning")
        return
    if config is None:
        raise ValueError("A stage-2 checkpoint requires a planner with a conditioning configuration")
    expected = planner_contract(config)
    if saved != expected:
        differing = [key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key)]
        raise ValueError(f"Checkpoint VLM conditioning mismatch: {sorted(differing)}")


def stage2_agent_settings(agent):
    """Portable constructor fields used by standalone evaluators."""
    keys = ("vlm_path", "vlm_type", "vlm_size", "dit_type", "cache_hidden_state",
            "train_backbone", "conditioning_mode", "vlm_kv_layers", "mask_vlm_padding",
            "cross_attention_heads", "cross_attention_kv_heads", "cross_attention_head_dim",
            "cross_attention_q_norm", "vlm_rope_theta", "inference_precision")
    result = {key: getattr(agent, key) for key in keys}
    result["sampling_method"] = agent.action_head.config.sampling_method
    return json.loads(json.dumps(result))
