#!/usr/bin/env bash
# The 24 ReactTrain runs (3 objects x hazardous/benign x 4 look-ahead times) on the same 2,000 navtrain scenes,
# then the 1,957 scenes rendered in all of them (paper appendix, ReactTrain). Output: <reacttrain>/<run>.
#   scripts/reacttrain/composite.sh            (environment as scripts/composite_navsim.sh)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
ROOT="$(location reacttrain datasets/reacttrain)"
DATA="$REPO/data/reacttrain"
COMMON=(--filter "$DATA/reacttrain.yaml" --split trainval --max-samples 2000 --output-root "$(dirname "$ROOT")")
for t in 1 2 3 4; do
    "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" --directions "$DATA/directions.json" "hazardous_walking_timestep_${t}_n2000" "$t"
    "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" --standing "hazardous_standing_timestep_${t}_n2000" "$t"
    "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" --car "hazardous_car_timestep_${t}_n2000" "$t"
    for object in walking standing car; do
        mode=(); [[ $object == standing ]] && mode=(--standing); [[ $object == car ]] && mode=(--car)
        "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" "${mode[@]}" --benign --benign-sides "$DATA/benign_sides.json" \
            "benign_${object}_timestep_${t}_n2000" "$t"
    done
done
"$PYTHON" -m reactbench.selection.reacttrain intersect "$ROOT"
