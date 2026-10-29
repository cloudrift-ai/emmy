#!/bin/bash
# Bake a recipe's native lane into its serving image, on the target card:
#   1. prepare the pack from the recipe golden alone under strict evidence (`emmy serve --native --export`), with the
#      lane's preparation flags and environment (emmy/serving/native/image.py reads them from the recipe);
#   2. build `emmy-server` and the runtime worker from the same checkout;
#   3. split the pack into sub-10 GB parts and build the image, tagged
#      cloudriftai/emmy-native-<slug>:<emmy version>-<short sha> and labelled for `emmy publish`.
# Nothing is pushed. Usage: docker/emmy-native/bake.sh RECIPE_DIR [GPU_NAME]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RECIPE=$1
GPU=${2:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)}
WARM=$ROOT/docker/emmy-native/warm
cd "$ROOT"

LANE=$("$ROOT"/venv/bin/python -m emmy.serving.native.image "$RECIPE" "$GPU" "$ROOT/venv/bin/emmy")
eval "$LANE"
echo "[bake] $MODEL on $GPU -> $IMAGE"
echo "[bake] prepare: ${PREPARE[*]}"

rm -rf "$WARM"
mkdir -p "$WARM/bin"
start=$(date +%s)
"${PREPARE[@]}" --export "$WARM/pack"
echo "[bake] pack prepared in $(( $(date +%s) - start )) s: $(du -sh "$WARM/pack" | cut -f1)"

cargo build --release --locked -p emmy-server -p emmy-runtime
cp target/release/emmy-server target/release/emmy-runtime-worker "$WARM/bin/"

docker/vllm-emmy-serve/split_hf.sh "$WARM/pack" "$WARM/pack_parts"
docker build -f docker/emmy-native/Dockerfile \
    --build-arg PUBLISH_VERSION="$VERSION" \
    --build-arg PUBLISH_REVISION="$(git rev-parse HEAD)" \
    --build-arg MODEL="$MODEL" \
    --build-arg TARGET_GPU="$GPU" \
    -t "$IMAGE" docker/emmy-native
echo "[bake] built $IMAGE"
