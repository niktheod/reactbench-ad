#!/usr/bin/env bash
# Caches of the ReactTrain fine-tuning under <reacttrain_work> (default outputs/reacttrain):
#   original   navtrain VLM features (GPU) and metric caches (CPU), each with its validator
#   edited     VLM features (GPU) and metric caches with the inserted object (CPU) of the ReactTrain samples,
#              for the timesteps 3/4 and 1/2 (the paper's two batches)
#   scripts/reacttrain/caches.sh original|edited      (WORKERS: CPU workers, default 16)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EXP="$REACTTRAIN_WORK"; WORKERS="${WORKERS:-16}"; CACHE="$FORK/scripts/cache_dataset"
LOGS="$(location navsim_logs:trainval)"
export NAVSIM_EXP_ROOT="$EXP" NAVSIM_DEVKIT_ROOT="$FORK"
case "${1:?original|edited}" in
    original)
        torchrun --nproc_per_node=1 "$FORK/navsim/planning/script/run_dataset_caching_multi_node.py" \
            agent=recogdrive_agent experiment_name=recogdrive_agent_cache agent.cam_type=single \
            agent.cache_hidden_state=True agent.cache_mode=True train_test_split=navtrain \
            "worker.threads_per_node=$WORKERS" "agent.vlm_path=$CHECKPOINTS/recogdrive/VLM" \
            "cache_path=$EXP/recogdrive_agent_cache_dir_train"
        "$PYTHON" "$CACHE/validate_recogdrive_hidden_state_cache.py" --cache-path "$EXP/recogdrive_agent_cache_dir_train" \
            --navsim-log-path "$LOGS" --workers "$WORKERS"
        "$PYTHON" "$FORK/navsim/planning/script/run_metric_caching.py" train_test_split=navtrain \
            "cache.cache_path=$EXP/metric_cache_train" "worker.threads_per_node=$WORKERS"
        "$PYTHON" "$CACHE/validate_recogdrive_metric_cache.py" --cache-path "$EXP/metric_cache_train" \
            --navsim-log-path "$LOGS" --workers "$WORKERS" ;;
    edited)
        T12="$EXP/recogdrive_edited_training/navtrain_selected_t12_v1"
        "$PYTHON" "$CACHE/prepare_edited_training_features.py"
        "$PYTHON" "$CACHE/prepare_edited_training_metric_cache.py" --workers "$WORKERS"
        "$PYTHON" "$CACHE/prepare_edited_training_features.py" --timesteps 1 2 --output "$T12"
        "$PYTHON" "$CACHE/prepare_edited_training_metric_cache.py" --timesteps 1 2 --output "$T12" --workers "$WORKERS" ;;
    *) echo "unknown step: $1" >&2; exit 2 ;;
esac
