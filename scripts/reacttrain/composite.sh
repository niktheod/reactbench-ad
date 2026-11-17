#!/usr/bin/env bash
# The 24 ReactTrain variants (3 objects x hazardous/benign x 4 look-ahead times) on the same 2,000 navtrain scenes,
# then the 1,957 scenes rendered in all of them (paper appendix, ReactTrain). Output: <reacttrain>/<variant>.
#   scripts/reacttrain/composite.sh            (environment as scripts/composite_navsim.sh)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
ROOT="$(location reacttrain datasets/reacttrain)"
DATA="$REPO/data/reacttrain"
COMMON=(--filter "$DATA/reacttrain.yaml" --split trainval --max-samples 2000 --output-root "$(dirname "$ROOT")"
        --min-front-visible-fraction 0.25)  # at least 25% of the object's projected box inside the front camera
TAUS=(0.5 1.0 1.5 2.0)  # the variants keep their working names, with t = tau in 0.5 s frames
for t in 1 2 3 4; do
    tau="${TAUS[t-1]}"
    "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" --directions "$DATA/directions.json" \
        --variant "hazardous_walking_timestep_${t}_n2000" "$tau"
    "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" --standing --variant "hazardous_standing_timestep_${t}_n2000" "$tau"
    "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" --car --variant "hazardous_car_timestep_${t}_n2000" "$tau"
    for object in walking standing car; do
        mode=(); [[ $object == standing ]] && mode=(--standing); [[ $object == car ]] && mode=(--car)
        "$REPO/scripts/composite_navsim.sh" "${COMMON[@]}" "${mode[@]}" --benign --benign-sides "$DATA/benign_sides.json" \
            --variant "benign_${object}_timestep_${t}_n2000" "$tau"
    done
done
"$PYTHON" -m reactbench.selection.reacttrain intersect "$ROOT"
