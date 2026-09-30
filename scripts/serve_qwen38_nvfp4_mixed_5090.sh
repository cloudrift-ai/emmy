#!/bin/sh
# Initial pinned BF16 mixed lane on one RTX 5090. The per-piece identities match
# the M=16 decode graph of the revision below; re-audit them after graph changes.
set -eu

export EMMY_KNOBS='FAST_MATH=true,PLACE@map.1/map=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.3/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,WORK=w1x4,TILE=,STAGE=,REDUCE=,RASTER='
STATIC='WORK@place_6b4be893d5=w1x2,TILE@place_6b4be893d5=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_6b4be893d5=d2/smem-async'
STATIC="$STATIC,WORK@place_743937bec0=w1x1,TILE@place_743937bec0=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_743937bec0=d2/smem-async"
STATIC="$STATIC,WORK@place_b5f468b49f=w1x1,TILE@place_b5f468b49f=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_b5f468b49f=d2/smem-async"
STATIC="$STATIC,WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async"
export EMMY_MLP_STATIC_KNOBS="$STATIC"
PREFILL='WORK@place_66b5682eed=w1x2,TILE@place_66b5682eed=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_66b5682eed=d2/smem-async'
PREFILL="$PREFILL,WORK@place_2cedf62283=w1x1,TILE@place_2cedf62283=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_2cedf62283=d2/smem-async"
PREFILL="$PREFILL,WORK@place_2c71f28601=w1x1,TILE@place_2c71f28601=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_2c71f28601=d2/smem-async"
PREFILL="$PREFILL,WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async"
export EMMY_MLP_PREFILL_KNOBS="$PREFILL"
export EMMY_FAST_MATH=1

exec emmy serve Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462 \
    --runner generate --compile-scope mlp --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 64 \
    --language-model-only --gpu-memory-utilization 0.97 "$@"
