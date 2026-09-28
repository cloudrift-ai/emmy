# Qwen3.8 attention repeats probability computation; encode launches a block per byte

## Summary

Two forms of repeated work appear in the inspected Qwen3.8 NVFP4 layer: attention recomputes each softmax
probability for every output column, and activation encoding recomputes each group's maximum for every packed byte.
A bounded code and IR check found no practical route to fused online-softmax attention for this layer.

The checkpoint has 16 full-attention layers out of 64, at indices 3, 7, 11, and so on. The examples below use layer 3
at 16 tokens after cuts split the fused kernel into smaller kernels, called pieces, exposing its projections to fp4
tensor-core schedules.

## Reading the repeated work

These normalized sketches omit batch/head axes and abbreviate the IR. The expected forms illustrate reuse; they are
not compiler output or prescribed thread layouts.

```text
Observed P·V:                      Expected fused attention:
for query, column:                 Fold[key] contraction ⟨twist=softmax⟩
    sum over key:                      carry max, sum, weighted_output[columns]
        p = exp(score - max) / sum      reuse probabilities across a column tile
        output += p * V

Observed encode:                   Expected encode:
for byte:                          for group of 16 values:
    max = reduce(group_of(byte))        max = reduce(group)
    write one byte                     reuse max for eight bytes and the scale
```

The output-column loop repeats each probability calculation; the byte loop repeats each group's maximum eight times.
Online softmax carries a running maximum, sum and weighted output while visiting keys in blocks, avoiding a stored
score matrix. It improves reuse and data movement while retaining dense attention's quadratic query/key work.

## Observed details

1. **No practical flash route found; P·V recomputes softmax probabilities.** None of the three cut sets tried (the
   default, a projection-only cut, and the full-projection cut) keeps scaled dot-product attention as one kernel with
   an online softmax. P·V becomes a reduce over keys with one 128-thread block per output element (batch × head ×
   query × head-dim column, 98,304 blocks at 16 tokens). Each block recomputes `exp(score − max) / sum` for every key,
   so probability evaluation runs once per head-dim column (256 times per score) instead of being reused. The max and
   sum have already been materialized; this is not a fresh reduction of the entire softmax in each block.
   - Four pieces compute this product. Two are printed as `reduce` (`…__place_1108b7f44e`, `…__place_d20db92cbe`), at
     111–149 µs each.
   - Two are printed as `contraction` (`…__place_129c440787`, `…__place_54c95851bf`); with the fp4 TILE pinned, the
     scheduler leaves them without a schedule.
   - The four-fold copy is likely the same recompute as in [the producer-duplication
     report](fp4-encode-recomputes-producer.md); this link between causes is unverified. The cut
     prints two of them as `reduce`, not `contraction`; that may be why the scheduler offers them no tensor-core
     schedule.
   - This repeats work within attention; it does not change its query/key arithmetic complexity.
2. **Activation-encode kernels launch one block per packed output byte.** A 4-bit re-encode of a matmul output runs
   `WORK=t128, REDUCE=coop`, with one 128-thread block per byte cooperatively reducing its 16-value group. The group's
   index is `byte_index / 8`, so eight neighboring byte outputs repeat the same maximum reduction.
   - For the gate/up output at 16 tokens that is 139,264 blocks, 151.7 µs, for a job that moves about 0.5 MB.
   - At 512 tokens, o_proj's encode takes 1.6 ms, and gate/up's takes 7.2 ms with the fp4 TILE pinned (0.65 ms under
     the default pick).

## Reproduce

All commands run from the repository root inside `nix develop`, with a fresh tune DB.

```sh
M=Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462   # ~26 GB download on first use
rm -f /tmp/q38.db; export EMMY_TUNE_DB=/tmp/q38.db
./venv/bin/emmy trace $M --layer 3 --seq-len 16 --target sm_120 -o /tmp/l3_s16.golden.json

# 1. Attention kernel (q/k/v projections, q/k norm, RoPE, SDPA, output gate, encode): full-projection cut + fp4 cell.
#    The compile takes about 14 s.
ATTN_CUT='PLACE@map.1/map=cut,PLACE@map.1/map.1/inner=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.2/inner.1/map.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner.1/map.11/map.1/reduce=cut,PLACE@map.1/map.2/inner.2/map.1/inner.1/map.11/map.1/reduce.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner.2/map.11/map.1/reduce=cut,PLACE@map.1/map.2/inner.2/map.1/inner.2/map.11/map.1/reduce.1/inner=cut,PLACE@map.1/map.2/inner.2/map.2/reduce=cut,PLACE@map.1/map.2/inner.2/map.3/reduce=cut,PLACE@map.1/map.3/inner=cut,PLACE@map.1/map.4/reduce.1/inner=cut,PLACE@map.1/map.4/reduce.2/inner=cut,PLACE@map.1/map.4/reduce.2/inner.1/map.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.1/reduce.1/inner=cut,PLACE@map.2/map.1/reduce.2/inner=cut,PLACE@map.2/map.1/reduce.2/inner.1/map.1/inner=cut'
EMMY_KNOBS="$ATTN_CUT,TILE=mma_m16n8k64_e2m1_f32/f1x1/k8,STAGE=d3/smem-async" \
  ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_sdpa_linear_mean_reduce_c6c239 --ir tile
# the same knobs with `emmy run … --bench --no-record-evidence` give per-piece times and block counts

# 2. Gate/up + SiLU + encode, full-projection cut + fp4 cell; the bench shows the encode piece's launch shape:
EMMY_KNOBS='PLACE@map.1/map=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.3/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,TILE=mma_m16n8k64_e2m1_f32/f1x2/k8,STAGE=d3/smem-async' \
  ./venv/bin/emmy run --golden /tmp/l3_s16.golden.json --realization k_linear_reduce_2d1c79 --bench --no-record-evidence
```

Kernel names are the ones this trace gives at commit `a98fd4f8`. `emmy golden kernels /tmp/l3_s16.golden.json` prints
each kernel's Loop IR as one JSON line, and its `"name"` fields list the current names.

## IR and launch evidence

1. One of the P·V `reduce` pieces. `free` lists its free axes; `grid` lists the axes spread over the launch:

```
=== 13: k_sdpa_linear_mean_reduce_c6c239__place_1108b7f44e ===
    place  free=(a1, a0, a2, a3, a4)  grid=(a1, a0, a2, a3, a4)
    Fold[a5 in 0..16] reduce
    ├─ operand[in6]: load mul_10_static_fp4_scale_bits__place_5b87de7ca7_0[a0, a3, a4, a5]   ‹materialized›
    ├─ operand[v5]: Fold  free   ‹computed›
    │  ├─ operand[in5]: load …__place_ff1a5c8a30_0[a1, ((6 * a0) + a2), a5]   ‹materialized›
    │  ├─ operand[…]: Fold  free  ‹pointwise›
    │  │       in3 = load …__place_19eeb2b186_0[a1, ((6 * a0) + a2)]
    │  │       in4 = load …__place_3814a75921_0[a1, ((6 * a0) + a2)]
    │       v4 = exp(v3)
    │       v5 = divide(v4, in3)
    outputs
    └─ …__place_1108b7f44e_0[a1, ((6 * a0) + a2), a3, a4] = acc0
```

   `mul_10_static_fp4_scale_bits__place_…` are names the cut generated for its workspaces. They hold scores, sums and
   V, not scale data. The piece has no TILE.

2. The gate/up encode piece in the bench table:

```
Kernel                                       us      %    grid  block   …  WORK  …  REDUCE
k_linear_reduce_2d1c79__place_03cba86402  151.7  25.9%  139264    128   …  t128  …  coop
```

   Here `grid` is the thread-block count and `block` is threads per block. 139,264 = 16 tokens × 8,704 packed bytes
   per row (17,408 values, two per byte).

## Bounded code and IR check (2026-09-28)

At `a98fd4f8`, flash formation is an automatic algebraic rewrite, not a separate knob to enable. The `020_twisted`
pass calls `rewrite_twisted`; `Fold.fuse` tries the `SOFTMAX` recipe from `emmy/compiler/ir/pure/twist.py`, including
hoisting the key-invariant divisor. It
requires matching score expressions and compatible reduction axes. A tensor-core TILE pin schedules the resulting
tree; it does not manufacture a missing online-softmax carrier.

A CPU-only check of the saved layer-3, 16-token Loop IR found no twisted carrier after `tile/lift`: the rewrite
reported unmatched sibling reductions. Repeating the projection-only cut with `--passes tp`, stopping before
scheduling, still produced no `twist=softmax`. As a control, plain causal GQA with the same 24 query heads, 4 KV
heads, 16 tokens and head dimension 256 produced one softmax carrier with both denominator and weighted-value
channels. Thus these dimensions and GQA alone do not prevent recognition; the fused checkpoint expression is the
failing case.

The successful control contains this Tile IR line (the checkpoint dump has no `twist=softmax` line):

```text
├─ operand[acc1, acc3, acc5__sum]: Fold[a2 in 0..16] contraction  ⟨twist=softmax⟩   ‹computed›
```

The lift and control can be checked without a GPU or an exhaustive schedule search:

```sh
EMMY_KNOBS= ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json \
  --realization k_sdpa_linear_mean_reduce_c6c239 --target sm_120 --passes t --ir tile -vv
EMMY_KNOBS= ./venv/bin/emmy compile --target sm_120 --passes dolfnst --ir tile \
  -c 'F.scaled_dot_product_attention(torch.randn(1,24,16,256).half(), torch.randn(1,4,16,256).half(), torch.randn(1,4,16,256).half(), is_causal=True, enable_gqa=True)'
```

No practical flash route emerged from this bounded check. Keep this as a compiler bug: finding a usable route should
not require an exhaustive search. The exact expression mismatch and whether another cut would expose a matching
carrier remain to be diagnosed. The control proves carrier formation, not successful CUDA execution or speed.

## Compare

Not checked: whether shipped goldens reach a flash form for full attention, and by which route. Candidates are
`recipes/gemma-4-12B-it/golden/rtx5090_sm120.json` and the Qwen3-0.6B goldens under `experiments/Qwen3-0.6B/*/golden/`
and `experiments/golden-bench-2026/kernels/golden/`.

## Fix criteria

1. **Attention.** With some cut set, at 16 and 512 tokens:
   - A documented cut and schedule reaches fused attention for this checkpoint's grouped-query layout. Tile IR carries
     the running max, sum and weighted output across key blocks, and both score and P·V contractions carry compatible
     tensor-core TILEs (`mma_…`). The number of kernels or query tiles is not prescribed; legal tiling and split-KV
     remain available.
   - No piece reduces over keys with `exp`/`divide` in its body while it grids the head-dim axis.
   - P·V is computed once, not four times.
   - Compare attention and whole-kernel-set latency before and after at 16 and 512 tokens, on the same hardware and
     inputs with correctness checked. Check that probability evaluation is reused across output columns. Attention
     still requires work proportional to queries × keys × head dimension; flash changes data movement and reuse, not
     that arithmetic bound.
2. **Encode.**
   - Compute each 16-value group's maximum once and reuse it for its eight packed bytes and stored scale. Distribute
     groups across threads or warps without dedicating a mostly idle 128-thread block to every byte. Do not require
     one exact block count or layout.
   - Compare encode and total kernel-set time at 16 and 512 tokens under matching measurement conditions. Record
     launch geometry, work per block, and correctness. Treat the reported microseconds as baselines; explain any
     latency regression rather than imposing an unmeasured absolute target.
- **Correctness:** the outputs match the current kernel set on the same inputs (`emmy run … --ab`, whose wrong-answer
  check compares both). This check drops out when the reference compile disagrees with itself (the bench prints
  "wrong-answer reference unusable"). A fixer must confirm the check actually ran.
