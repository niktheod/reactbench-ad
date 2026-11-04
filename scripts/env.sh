# Sourced by the launchers: repository paths (configs/paths.yaml) and the NAVSIM environment variables.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
location() { (cd "$REPO" && "$PYTHON" -m reactbench.paths "$@"); }
export PYTHONPATH="$REPO/third_party/recogdrive:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$(location navsim_data)}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$(location nuplan_maps)}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
CHECKPOINTS="$(location checkpoints checkpoints)"
REACTTRAIN_WORK="$(location reacttrain_work "$(location outputs/reacttrain)")"
FORK="$REPO/third_party/recogdrive"
