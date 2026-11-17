#!/usr/bin/env bash
# Insert one object into NAVSIM scenes (ReactNAV, standing animals, ReactTrain): CPU preparation, Blender
# rendering (GPU), PCTNet harmonization (GPU), then validation and compaction. Contact-only shadows.
#
#   scripts/composite_navsim.sh [options] TAU        (look-ahead time in seconds: 0.5, 1.0, 1.5 or 2.0)
#
# The run is named <object>_tau<TAU>s, e.g. car_tau1.0s (--run-name NAME overrides it).
# Options: --standing | --car | --animal, --benign [benign options] (needs --run-name), --filter FILE (default
# data/reactnav/reactnav.yaml), --split test|trainval, --directions FILE (crossing pedestrians; default
# data/reactnav/directions.json), --assets DIR, --max-samples N, --output-root DIR (runs go to
# <output root>/<filter name>/<run name>; default: the parent of `reactnav`), --keep-intermediates,
# --min-front-visible-fraction F (share of the object's projected box inside the current front camera; default 0,
# 0.25 with --animal; scripts/reacttrain/composite.sh passes 0.25).
# ReactNAV:  scripts/composite_navsim.sh --car 2.0                         (car_tau2.0s)
#            scripts/composite_navsim.sh --standing 2.0                    (standing_pedestrian_tau2.0s)
#            scripts/composite_navsim.sh 2.0                               (crossing_pedestrian_tau2.0s)
# Animals:   scripts/composite_navsim.sh --animal --max-samples 300 2.0    (standing_animal_tau2.0s)
# Environment: PYTHON (compositing environment), PCTNET_PYTHON (libcom environment), RENDER_WORKERS (GPUs, 1),
# RENDER_BATCH_SIZE (96), FINALIZE_WORKERS (4), COMPACT_WORKERS (4).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
HERE="$REPO/reactbench/compositing/navsim"
FILTER="$REPO/data/reactnav/reactnav.yaml"; SPLIT=test; DIRECTIONS="$REPO/data/reactnav/directions.json"
ASSETS_ROOT="$(location assets assets)"; ASSETS=""; MODE=crossing; BENIGN=0; MAX=(); EXTRA=(); OUTPUT=(); KEEP=0
VISIBILITY=0; RUN_NAME=""
while (( $# > 1 )); do
    case "$1" in
        --standing) MODE=standing; shift ;;
        --car) MODE=car; shift ;;
        --animal) MODE=animal; shift ;;
        --benign) EXTRA+=(--benign); BENIGN=1; shift ;;
        --min-front-visible-fraction) EXTRA+=("$1" "$2"); VISIBILITY=1; shift 2 ;;
        --benign-side|--benign-offset-m|--benign-walking-direction|--benign-clearance-m|--benign-sides)
            EXTRA+=("$1" "$2"); shift 2 ;;
        --filter) FILTER="$(realpath "$2")"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --directions) DIRECTIONS="$(realpath "$2")"; shift 2 ;;
        --assets) ASSETS="$(realpath "$2")"; shift 2 ;;
        --max-samples) MAX=(--max-samples "$2"); shift 2 ;;
        --output-root) OUTPUT=(--output-root "$(realpath "$2")"); shift 2 ;;
        --keep-intermediates) KEEP=1; shift ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
TAU="${1:?TAU}"
if (( BENIGN )) && [[ -z $RUN_NAME ]]; then echo "--benign needs --run-name" >&2; exit 2; fi
[[ $MODE == animal && $VISIBILITY == 0 ]] && EXTRA+=(--min-front-visible-fraction 0.25)
ENTRY=reactbench.compositing.navsim.compositor.engine; ASSET_ROOT="$HERE/compositor/blender_asset"; MODE_ARGS=()
case "$MODE" in
    crossing) ASSETS="${ASSETS:-$ASSETS_ROOT/rocketbox/assets}"; OBJECT=crossing_pedestrian
              (( BENIGN )) || MODE_ARGS=(--direction-classifications "$DIRECTIONS") ;;  # benign walkers do not cross
    standing) ASSETS="${ASSETS:-$ASSETS_ROOT/rocketbox/assets}"; MODE_ARGS=(--standing); OBJECT=standing_pedestrian ;;
    car) ASSETS="${ASSETS:-$ASSETS_ROOT/openx/assets}"; MODE_ARGS=(--car); OBJECT=car ;;
    animal) ASSETS="${ASSETS:-$ASSETS_ROOT/rocketbox_standing_animals/assets}"; MODE_ARGS=(--standing)
            OBJECT=standing_animal
            ENTRY=reactbench.compositing.navsim.animal_composite_navsim_blender; ASSET_ROOT="$HERE/animal_blender_asset" ;;
esac
RUN="$(cd "$REPO" && "$PYTHON" -m reactbench.runs navsim "$OBJECT" "$TAU")"  # "<object>_tau<TAU>s k"
read -r NAME TIMESTEP <<< "$RUN"; RUN_NAME="${RUN_NAME:-$NAME}"
ROOT="$(dirname "$(location reactnav datasets/reactnav)")"
[[ ${#OUTPUT[@]} -gt 0 ]] && ROOT="${OUTPUT[1]}"
RUN_DIR="$ROOT/$(basename "$FILTER" .yaml)/$RUN_NAME"
COMMON=(--scene-filter "$FILTER" --split "$SPLIT" "${MODE_ARGS[@]}" --asset-root "$ASSET_ROOT" --asset-dir "$ASSETS"
        --blender "$(location blender third_party/blender-4.5.11-linux-x64/blender)" --timestep "$TIMESTEP"
        --run-name "$RUN_NAME" --render-batch-size "${RENDER_BATCH_SIZE:-96}" --render-workers "${RENDER_WORKERS:-1}"
        --finalize-workers "${FINALIZE_WORKERS:-4}" "${OUTPUT[@]}" "${MAX[@]}" "${EXTRA[@]}"
        --refinalize --retry-failures --device gpu)
export PYTHONUNBUFFERED=1
cd "$REPO"
echo "== prepare ($RUN_DIR)"; "$PYTHON" -u -m "$ENTRY" "${COMMON[@]}" --stage prepare
echo "== render";             "$PYTHON" -u -m "$ENTRY" "${COMMON[@]}" --stage render
echo "== harmonize";          "${PCTNET_PYTHON:-$PYTHON}" -u -m reactbench.compositing.navsim.run_pctnet_folder "$RUN_DIR/images" \
                                  --output-root "$RUN_DIR/final_images" --mask-threshold 250 --device 0
if (( ! KEEP )); then
    echo "== validate and compact"
    "$PYTHON" -u -m reactbench.compositing.navsim.compact_run "$RUN_DIR" --workers "${COMPACT_WORKERS:-4}" \
        --pctnet-mask-threshold 250
fi
