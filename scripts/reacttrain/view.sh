#!/usr/bin/env bash
# The ReactTrain training view (CPU): the original/edited population of the t3/t4 edits, the kinematic stop
# targets of all hazardous edits (t1-t4), and the view of the paper's run: two look-ahead times per object
# and source scene, 11,742 hazardous + 11,742 benign edits and a pool of 35,226 originals.
#   scripts/reacttrain/view.sh        (WORKERS: CPU workers for the targets, default 24)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EXP="$REACTTRAIN_WORK"; CACHE="$FORK/scripts/cache_dataset"
"$PYTHON" "$CACHE/prepare_mixed_training_dataset.py"
"$PYTHON" "$CACHE/prepare_kinematic_sft_targets.py" --workers "${WORKERS:-24}"
"$PYTHON" "$CACHE/prepare_mixed_training_dataset.py" --population kinematic-sft \
    --mixed-manifest "$EXP/recogdrive_mixed_training/navtrain_selected_t34_v1/manifest.json" \
    --extra-edited-root "$EXP/recogdrive_edited_training/navtrain_selected_t12_v1" \
    --kinematic-targets "$EXP/recogdrive_kinematic_sft/targets_v1/targets.json" --sft-original-pool 35226
