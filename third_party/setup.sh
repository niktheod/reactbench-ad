#!/usr/bin/env bash
# Fetch the code of the evaluated models at the commits used in the paper, apply the local patches
# (patches/*.patch), and install Blender 4.5.11 LTS for the compositors. Safe to rerun.
#   third_party/setup.sh [--no-blender]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

fetch() {  # directory url commit [patch]
    local directory="$1" url="$2" commit="$3" patch="${4:-}"
    [[ -d "$directory/.git" ]] || git clone --filter=blob:none "$url" "$directory"
    if [[ -n "$patch" ]] && git -C "$directory" apply --reverse --check "$PWD/patches/$patch" 2>/dev/null; then
        echo "$directory: $patch already applied"
    else
        git -C "$directory" checkout -q "$commit"
        [[ -z "$patch" ]] || git -C "$directory" apply "$PWD/patches/$patch"
    fi
    echo "$directory: $(git -C "$directory" rev-parse --short HEAD)${patch:+ + $patch}"
}

# Alpamayo 1.5: a dedicated generator for the initial diffusion noise (diffusion_seed), so that the original
# and edited scene of a pair share it.
fetch alpamayo1.5 https://github.com/NVlabs/alpamayo1.5.git f42e594aaf8b50dcd2cbb359d62e3ffc7b12fcf8 alpamayo1.5.patch
fetch alpamayo2 https://github.com/NVlabs/alpamayo2.git 5e7975f4a2100ee8ac1a62b79239bbabefddcbef
# AutoVLA: the Qwen2.5-VL weights from the Hugging Face hub, FlashAttention-2, and an error instead of a
# wrong-length trajectory when the generated action tokens are incomplete.
fetch AutoVLA https://github.com/ucla-mobility/AutoVLA.git ba34eed74ce6729e7986592d0e66cbaca397b4fa AutoVLA.patch
# AutoVLA reads its action codebook through this config entry; point it at this checkout.
sed -i "s#^  codebook_cache_path: .*#  codebook_cache_path: \"$PWD/AutoVLA/codebook_cache/agent_vocab.pkl\"#" \
    AutoVLA/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml
# DriveWAM: works without FlashAttention (PyTorch SDPA, cuDNN backend off), nuplan-devkit from its repository.
fetch DriveWAM https://github.com/chenshi3/DriveWAM.git abbefb407c9c553f221522fc26077d307620adb4 DriveWAM.patch
fetch SimWAM https://github.com/H-EmbodVis/SimWAM.git 1433ee7e06a62bd8b89da74455df0d82bda7b726

if [[ "${1:-}" != --no-blender && ! -x blender-4.5.11-linux-x64/blender ]]; then
    archive=blender-4.5.11-linux-x64.tar.xz
    curl -fL -o "$archive" "https://download.blender.org/release/Blender4.5/$archive"
    echo "05ed7bd41bf3e61ae4f4a7cdc364c43088bf8b3fed702c2269c018fdf63a2188  $archive" | sha256sum -c -
    tar -xf "$archive" && rm "$archive"
fi
