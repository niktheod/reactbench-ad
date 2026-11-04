#!/usr/bin/env bash
# Build the inserted-object assets with Blender 4.5.11 (once, CPU):
#   scripts/assets.sh humans  ROCKETBOX_ROOT   # the 42 generic adult/child Rocketbox avatars (pedestrians)
#   scripts/assets.sh animals ROCKETBOX_ROOT   # the 5 standing Rocketbox animals (unseen objects, section 5)
#   scripts/assets.sh cars    OPENX_ROOT       # the 7 OpenX vehicles (cars)
# ROCKETBOX_ROOT: a clone of https://github.com/microsoft/Microsoft-Rocketbox at 0943055 (MIT);
# OPENX_ROOT: a clone of https://github.com/bounverif/openx-assets at 463ea3d (MPL-2.0 AND CC-BY-4.0).
# Outputs go to <assets>/{rocketbox,rocketbox_standing_animals,openx}/assets (`assets` in configs/paths.yaml).
# Extra arguments are passed to the builder (e.g. --limit 1 --overwrite).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
KIND="${1:?humans|animals|cars}"; SOURCE="$(realpath "${2:?source repository}")"; shift 2
BLENDER="$(location blender third_party/blender-4.5.11-linux-x64/blender)"
ASSETS="$(location assets assets)"
SCRIPTS="$REPO/reactbench/assets"
run() { "$BLENDER" --background --factory-startup --python-exit-code 2 --python "$1" -- "${@:2}"; }
case "$KIND" in
    humans)  run "$SCRIPTS/rocketbox/create_rocketbox_assets.py" --rocketbox-root "$SOURCE" \
                 --output-dir "$ASSETS/rocketbox/assets" "$@" ;;
    animals) run "$SCRIPTS/rocketbox/create_rocketbox_standing_animal_assets.py" --rocketbox-root "$SOURCE" \
                 --output-dir "$ASSETS/rocketbox_standing_animals/assets" "$@" ;;
    cars)    run "$SCRIPTS/openx/prepare_openx_assets.py" --openx-root "$SOURCE" \
                 --source-dir "$SOURCE/src/vehicles/main" --output-dir "$ASSETS/openx/assets" "$@" ;;
    *) echo "unknown asset kind: $KIND" >&2; exit 2 ;;
esac
