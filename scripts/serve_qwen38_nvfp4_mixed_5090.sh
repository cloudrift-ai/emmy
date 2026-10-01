#!/bin/sh
# Pinned BF16 mixed lane on one RTX 5090. Re-audit piece identities after graph changes.
set -eu

. "$(dirname "$0")/qwen38_nvfp4_mixed_5090_knobs.sh"

exec emmy serve Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462 \
    --runner generate --compile-scope mlp --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 64 \
    --language-model-only --gpu-memory-utilization 0.97 "$@"
