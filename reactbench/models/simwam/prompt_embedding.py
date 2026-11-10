"""Encode SimWAM's single fixed NAVSIM prompt with upstream's own T5 caching code.

Equivalent to upstream scripts/precomput_text_embed.sh for the released
configuration (use_dynamic_prompt=false), without instantiating the NAVSIM
dataset. Run once after scripts/download_checkpoints.sh.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import yaml

from . import agent as adapter
from ..runner import expand_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "rl.yaml")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, else cpu")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(expand_config(args.config.read_text()))
    import torch

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(cfg["diffsynth_model_root"]).expanduser())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    dataset_class = adapter.upstream_dataset_class()
    upstream_cfg = adapter.compose_upstream_config(cfg.get("upstream_task", "navsim_uncond_front_384x672_1e-4"))
    settings = adapter.data_settings(upstream_cfg)
    prompt = adapter.fixed_prompt(dataset_class, settings)
    output_dir = Path(cfg["prompt_embedding_dir"]).expanduser()
    target = adapter.prompt_embedding_path(prompt, output_dir, settings)

    spec = importlib.util.spec_from_file_location(
        "simwam_precompute", adapter.SIMWAM_REPO_ROOT / "scripts/precompute_navsim_text_embeds.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    model = upstream_cfg.model
    enc_id = upstream._model_id_to_enc_id(str(model.model_id))
    if enc_id != settings["text_encoder_id"]:
        raise ValueError(f"Encoder id {enc_id} disagrees with dataset cache id {settings['text_encoder_id']}")
    _, text_config, _, tokenizer_config = upstream._resolve_configs(
        model_id=str(model.model_id), tokenizer_model_id=str(model.tokenizer_model_id),
        redirect_common_files=bool(model.redirect_common_files))
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    if not text_config.path or not tokenizer_config.path:
        raise FileNotFoundError("Missing T5 encoder/tokenizer; run scripts/download_checkpoints.sh first")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = upstream._load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=device).eval()
    tokenizer = upstream.HuggingfaceTokenizer(
        name=tokenizer_config.path, seq_len=int(settings["context_len"]), clean="whitespace")
    upstream._encode_group(
        prompts_to_dirs={prompt: {output_dir}}, context_len=int(settings["context_len"]), enc_id=enc_id,
        tokenizer=tokenizer, text_encoder=text_encoder, device=device, overwrite=False,
        rank=0, world_size=1, is_distributed=False)
    if not target.is_file():
        raise RuntimeError(f"Prompt embedding was not written: {target}")
    print(f"Prompt: {prompt}\nEmbedding: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
