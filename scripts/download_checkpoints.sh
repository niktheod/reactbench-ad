#!/usr/bin/env bash
# Download the evaluated checkpoints into the `checkpoints` location of configs/paths.yaml:
#   recogdrive/{VLM,IL,RL}     ReCogDrive 2B (VLM SFT, IL and RL stages)          owl10/ReCogDrive-*
#   autovla/                   AutoVLA_PDMS_89 (SFT + RFT with the PDMS reward)    Zewei-Zhou/AutoVLA
#   drivewam/                  DriveWAM NAVSIM checkpoint and LingBot-VA parts     chenchenshi/DriveWAM, robbyant/lingbot-va-base
#   simwam/                    SimWAM (IL) and SimWAM-RL, Wan2.1/2.2 components    H-EmbodVis/SimWAM, Wan-AI, DiffSynth-Studio
# All downloads are pinned to the revisions of the paper's runs. Alpamayo 1.5 (nvidia/Alpamayo-1.5-10B) and
# Alpamayo 2 (nvidia/Alpamayo2-Super) load from the Hugging Face cache (`hf_home`) at the revisions in their
# configs. `depth` downloads Depth Anything V2 Small, which the compositors run offline, into `hf_home`. The
# fine-tuned IL + RT checkpoint is not released: scripts/reacttrain/train.sh trains it, and it goes to
# recogdrive/IL_RT/recogdrive_il_rt.ckpt.
#   scripts/download_checkpoints.sh [recogdrive autovla drivewam simwam depth]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODELS=("$@"); (( ${#MODELS[@]} )) || MODELS=(recogdrive autovla drivewam simwam)
HF_CACHE="$(location hf_home "$(location outputs/work/hf_cache)")"
export CHECKPOINTS HF_CACHE
for model in "${MODELS[@]}"; do
    "$PYTHON" - "$model" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
root, model = os.environ["CHECKPOINTS"], sys.argv[1]
if model == "recogdrive":
    snapshot_download("owl10/ReCogDrive-VLM-2B", revision="16873acca08e3c04ab229b3d973f39aeba9db68d",
                      local_dir=f"{root}/recogdrive/VLM")
    for stage, revision in (("IL", "0fa96f2159d5bc0f816ecd3818f00541db9db6ff"),
                            ("RL", "7346ede73f03ed577811024f1a5d16fb063a47b9")):
        hf_hub_download(f"owl10/ReCogDrive-2B-{stage}", f"ReCogDrive_Diffusion_Planner_2B_{stage}.ckpt",
                        revision=revision, local_dir=f"{root}/recogdrive/{stage}")
elif model == "autovla":
    hf_hub_download("Zewei-Zhou/AutoVLA", "AutoVLA_PDMS_89.ckpt", revision="a7d7ba3ed7529b248d2694c2defa31b35208340f",
                    local_dir=f"{root}/autovla")
elif model == "drivewam":
    snapshot_download("chenchenshi/DriveWAM", revision="c4fb6d2c5d26ee3018a9ee1507f36fd02b85b1ce",
                      allow_patterns=["navsim/*"], local_dir=f"{root}/drivewam/DriveWAM")
    snapshot_download("robbyant/lingbot-va-base", revision="68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c",
                      allow_patterns=["vae/*", "tokenizer/*", "text_encoder/*"],
                      local_dir=f"{root}/drivewam/lingbot-va-base")
elif model == "simwam":
    for name in ("SimWAM.pt", "SimWAM-RL.pt"):
        hf_hub_download("H-EmbodVis/SimWAM", f"weights/{name}", revision="5d4492937e357e6ca64de128cdccbd25c2b34f30",
                        local_dir=f"{root}/simwam/SimWAM")
    diffsynth = f"{root}/simwam/diffsynth"
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", revision="37ec512624d61f7aa208f7ea8140a131f93afc9a",
                      allow_patterns=["google/umt5-xxl/*"], local_dir=f"{diffsynth}/Wan-AI/Wan2.1-T2V-1.3B")
    from modelscope import snapshot_download as modelscope_download
    modelscope_download("DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                        local_dir=f"{diffsynth}/DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                        allow_file_pattern=["Wan2.2_VAE.safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"])
    print("SimWAM: next, python -m reactbench.models.simwam.prompt_embedding (text embeddings)")
elif model == "depth":  # Hugging Face cache layout: the compositors set HF_HOME to `hf_home`
    snapshot_download("depth-anything/Depth-Anything-V2-Small-hf", revision="5426e4f0f36572d16453bbda7a8389317b1bef99",
                      cache_dir=f"{os.environ['HF_CACHE']}/hub")
else:
    sys.exit(f"unknown model {model}")
print(f"{model}: done")
PY
done
