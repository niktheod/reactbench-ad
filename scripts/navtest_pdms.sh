#!/usr/bin/env bash
# navtest PDMS of a ReCogDrive checkpoint (paper section 5, fine-tuning table), one GPU.
#   scripts/navtest_pdms.sh NAME CHECKPOINT     e.g. recogdrive_il_rt checkpoints/recogdrive/IL_RT/recogdrive_il_rt.ckpt
# Needs a navtest metric cache built with this NAVSIM devkit (first run: scripts/navtest_pdms.sh --metric-cache).
# Writes outputs/navtest_pdms/NAME/<run>/*.csv, which `python -m reactbench.summarize --tables fine_tuning` reads.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
CACHE="$(location navtest_pdms_metric_cache "$(location outputs/work/metric_cache_navtest)")"
if [[ "${1:-}" == --metric-cache ]]; then
    export NAVSIM_EXP_ROOT="$(dirname "$CACHE")"
    exec "$PYTHON" "$FORK/navsim/planning/script/run_metric_caching.py" train_test_split=navtest \
        "cache.cache_path=$CACHE" "${@:2}"
fi
NAME="${1:?NAME}"; CHECKPOINT="$(realpath "${2:?CHECKPOINT}")"
export NAVSIM_EXP_ROOT="$(location outputs/navtest_pdms)"
# Settings of the paper's runs: hidden-state conditioning, unmasked VLM padding, full precision, seed 0.
exec "$PYTHON" -m torch.distributed.run --nproc_per_node=1 --master_port="${MASTER_PORT:-29522}" \
    "$FORK/navsim/planning/script/run_pdm_score_recogdrive.py" train_test_split=navtest agent=recogdrive_agent \
    "agent.checkpoint_path='$CHECKPOINT'" "agent.vlm_path='$CHECKPOINTS/recogdrive/VLM'" \
    agent.cam_type=single agent.grpo=false agent.cache_hidden_state=false agent.mask_vlm_padding=false \
    agent.train_backbone=false agent.vlm_type=internvl agent.dit_type=small agent.vlm_size=small \
    agent.sampling_method=ddim agent.conditioning_mode=hidden_states agent.inference_precision=32-true \
    "metric_cache_path='$CACHE'" scorer.config.progress_weight=5.0 scorer.config.ttc_weight=5.0 \
    scorer.config.comfortable_weight=2.0 +inference_seed=0 "experiment_name=$NAME" "${@:3}"
