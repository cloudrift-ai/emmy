#!/bin/sh
# Pinned BF16 mixed lane on one RTX 5090. Re-audit piece identities after graph changes.
set -eu

export EMMY_KNOBS='FAST_MATH=true,PLACE@map.1/map=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.3/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,WORK=,TILE=,STAGE=,REDUCE=,RASTER='
STATIC='PLACE@place_643aecc968/map.1/inner=cut,PLACE@place_643aecc968/map.2/inner=cut'
STATIC="$STATIC,WORK@place_532c520dc2=w1x2,TILE@place_532c520dc2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_532c520dc2=d2/smem-async"
STATIC="$STATIC,WORK@place_4b5e95ec28=w1x1,TILE@place_4b5e95ec28=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_4b5e95ec28=d2/smem-async"
STATIC="$STATIC,WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async"
export EMMY_MLP_STATIC_KNOBS="$STATIC"
PREFILL='PLACE@place_f688369f74/map.1/inner=cut,PLACE@place_f688369f74/map.2/inner=cut'
PREFILL="$PREFILL,WORK@place_c0904cfc6e=w1x1,TILE@place_c0904cfc6e=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_c0904cfc6e=d2/smem-async"
PREFILL="$PREFILL,WORK@place_8f9ed3f314=w1x1,TILE@place_8f9ed3f314=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_8f9ed3f314=d2/smem-async"
PREFILL="$PREFILL,WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async"
export EMMY_MLP_PREFILL_KNOBS="$PREFILL"
export EMMY_FAST_MATH=1
