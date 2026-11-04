#!/usr/bin/env bash
# Download the evaluated checkpoints into the `checkpoints` location of configs/paths.yaml:
#   recogdrive/{VLM,IL,RL}     ReCogDrive 2B (VLM SFT, IL and RL stages)          owl10/ReCogDrive-*
#   autovla/                   AutoVLA_PDMS_89 (SFT + RFT with the PDMS reward)    Zewei-Zhou/AutoVLA
#   drivewam/                  DriveWAM NAVSIM checkpoint and LingBot-VA parts     chenchenshi/DriveWAM, robbyant/lingbot-va-base
#   simwam/                    SimWAM (IL) and SimWAM-RL, Wan2.1/2.2 components    H-EmbodVis/SimWAM, Wan-AI, DiffSynth-Studio
# Alpamayo 1.5 (nvidia/Alpamayo-1.5-10B) and Alpamayo 2 (nvidia/Alpamayo2-Super) load from the Hugging Face cache
# (`hf_home`) at the revisions in their configs; Depth Anything V2 Small loads into `hf_home` too. The fine-tuned
# IL + RT checkpoint is released separately (README) and goes to recogdrive/IL_RT/recogdrive_il_rt.ckpt.
#   scripts/download_checkpoints.sh [recogdrive autovla drivewam simwam]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODELS=("$@"); (( ${#MODELS[@]} )) || MODELS=(recogdrive autovla drivewam simwam)
export CHECKPOINTS
for model in "${MODELS[@]}"; do
    "$PYTHON" - "$model" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
root, model = os.environ["CHECKPOINTS"], sys.argv[1]
if model == "recogdrive":
    snapshot_download("owl10/ReCogDrive-VLM-2B", local_dir=f"{root}/recogdrive/VLM")
    for stage in ("IL", "RL"):
        hf_hub_download(f"owl10/ReCogDrive-2B-{stage}", f"ReCogDrive_Diffusion_Planner_2B_{stage}.ckpt",
                        local_dir=f"{root}/recogdrive/{stage}")
elif model == "autovla":
    hf_hub_download("Zewei-Zhou/AutoVLA", "AutoVLA_PDMS_89.ckpt", local_dir=f"{root}/autovla")
elif model == "drivewam":
    snapshot_download("chenchenshi/DriveWAM", allow_patterns=["navsim/*"], local_dir=f"{root}/drivewam/DriveWAM")
    snapshot_download("robbyant/lingbot-va-base", allow_patterns=["vae/*", "tokenizer/*", "text_encoder/*"],
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
else:
    sys.exit(f"unknown model {model}")
print(f"{model}: done")
PY
done
