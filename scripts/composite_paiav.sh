#!/usr/bin/env bash
# Insert one object into the selected PAI-AV clips (ReactPAI): ego-mask preparation, CPU preparation, Blender
# rendering (GPU), guarded PCTNet harmonization (GPU), then validation and compaction. Contact-only shadows.
#
#   scripts/composite_paiav.sh [--standing | --car] [--max-samples N] [--output-root DIR] [--keep-intermediates] \
#                              RUN_NAME TIMESTEP        (TIMESTEP: 0.1 s frames)
# ReactPAI:  scripts/composite_paiav.sh --car car_timestep_20 20
#            scripts/composite_paiav.sh --standing standing_pedestrian_timestep_20 20
#            scripts/composite_paiav.sh crossing_pedestrian_timestep_20 20
# Inputs: paiav_root (the PAI-AV download), reactpai_manifest (data/reactpai/selection_manifest.json),
# data/reactpai/directions.json (crossing directions); outputs: <output root>/<RUN_NAME> (default: `reactpai`).
# Environment: PYTHON (compositing environment), PCTNET_PYTHON (libcom environment), RENDER_WORKERS (1),
# RENDER_BATCH_SIZE (96), FINALIZE_WORKERS (4), COMPACT_WORKERS (4).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODE=(--direction-classifications "$REPO/data/reactpai/directions.json"); MAX=(); KEEP=0
while (( $# > 2 )); do
    case "$1" in
        --standing) MODE=(--standing); shift ;;
        --car) MODE=(--car); shift ;;
        --max-samples) MAX=(--max-clips "$2"); shift 2 ;;
        --output-root) OUTPUT_ROOT="$(realpath "$2")"; shift 2 ;;
        --keep-intermediates) KEEP=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
RUN_NAME="${1:?RUN_NAME}"; TIMESTEP="${2:?TIMESTEP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(location reactpai datasets/reactpai)}"; RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
EGO_MASKS=(--ego-mask-cache "$OUTPUT_ROOT/_cache/ego_masks")  # next to the depth cache (<output root>/_cache)
DATA=(--data-root "$(location paiav_root)" --manifest "$(location reactpai_manifest data/reactpai/selection_manifest.json)")
ASSETS="$(location assets assets)/rocketbox/assets"; [[ "${MODE[0]}" == --car ]] && ASSETS="$(location assets assets)/openx/assets"
COMMON=("${DATA[@]}" --asset-dir "$ASSETS" --output-root "$OUTPUT_ROOT" --run-name "$RUN_NAME" --timestep "$TIMESTEP"
        --anchor-time-s 5.1 --occlusion-mode dense --samples 16 --render-scale 1 --ego-mask-mode refined-gls "${EGO_MASKS[@]}"
        "${MAX[@]}" "${MODE[@]}" --render-workers "${RENDER_WORKERS:-1}" --render-batch-size "${RENDER_BATCH_SIZE:-96}"
        --finalize-workers "${FINALIZE_WORKERS:-4}")
export PYTHONUNBUFFERED=1
cd "$REPO"
echo "== ego masks"
"$PYTHON" -B -m reactbench.compositing.paiav.prepare_ego_masks "${DATA[@]}" --anchor-time-s 5.1 \
    --ego-mask-mode refined-gls "${EGO_MASKS[@]}" "${MAX[@]}" --status "$RUN_DIR/ego_mask_preparation.json"
echo "== prepare ($RUN_DIR)"; "$PYTHON" -u -m reactbench.compositing.paiav.edit "${COMMON[@]}" --stage prepare
echo "== render";             "$PYTHON" -u -m reactbench.compositing.paiav.edit "${COMMON[@]}" --stage render
echo "== harmonize"
"${PCTNET_PYTHON:-$PYTHON}" -u -m reactbench.compositing.paiav.run_pctnet_folder "$RUN_DIR/images" \
    --output-root "$RUN_DIR/final_images" --mask-threshold 250 --device 0 --completion-dir "$RUN_DIR/completed_tokens" \
    --roi-padding 1.0 --min-foreground-pixels 256 --min-bbox-width 12 --min-bbox-height 24 --blend-strength 0.5 \
    --max-mean-color-shift 24 --min-contrast-ratio 0.5 --overwrite
if (( ! KEEP )); then
    echo "== validate and compact"
    OMP_NUM_THREADS=1 "$PYTHON" -u -m reactbench.compositing.paiav.compact_run "$RUN_DIR" \
        --workers "${COMPACT_WORKERS:-4}" --pctnet-mask-threshold 250
fi
