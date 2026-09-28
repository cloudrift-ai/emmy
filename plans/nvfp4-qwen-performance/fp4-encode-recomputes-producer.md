# FP4 re-encode cuts duplicate producer projections

## Summary

In a W4A4 NVFP4 model, a matmul whose output feeds another quantized matmul is followed by a 4-bit re-encode of that
output. The encode reads the matmul's output three times, in two index layouts:

- pairs of neighbouring values, to pack two 4-bit codes per byte (`2*a0`, `2*a0 + 1`),
- 16-value blocks, for the absolute maximum that sets the codes' scale (`16*a0 + a2`),
- 16-value blocks again, for the stored 8-bit block scale.

The full-projection cut splits such a fused kernel so that each matmul can run on the fp4 tensor-core instruction. It
then materializes the producer matmul once per read, not once in total. Two of the three copies even use the same
layout and have byte-identical bodies.

In `Inferact/Qwen3.8-27B-NVFP4` layer 3 at 16 tokens this gives:

- gate/up: 3 copies
- o_proj: 4 copies
- q_proj: 4 copies
- v_proj: 3 copies

Each copy streams the full weight again.

## Terms

- **Full-projection cut:** one decision of the cut pass (`full_projection_seams`,
  `emmy/compiler/pipeline/passes/tile/_cut.py`, line 410; "Full-projection cut" in `GLOSSARY.md`). It is offered where
  a fused kernel owns outputs its projection cannot bind. It cuts every contraction occurrence, every reduce hoisted
  ahead of an output sweep, and every output-owning branch. In `EMMY_KNOBS` it is spelled as a set of
  `PLACE@<site>=cut` keys. The resulting kernels are called *pieces*.
- **fp4 cell:** the native fp4 tensor-core instruction `mma_m16n8k64_e2m1_f32`, called "the block-scaled fp4 cell" in
  `emmy/compiler/ir/atom.py`. It multiplies packed 4-bit codes and applies the 16-value block scales in hardware.
- **Seam:** a point in a kernel's Fold tree where the cut pass can place a kernel boundary. The knob site
  (`map.1/map.2/inner`) spells that point.

## Reproduce

All commands run from the repository root inside `nix develop`, with a fresh tune DB.

```sh
M=Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462   # ~26 GB download on first use
rm -f /tmp/q38.db; export EMMY_TUNE_DB=/tmp/q38.db
./venv/bin/emmy trace $M --layer 3 --seq-len 16 --target sm_120 -o /tmp/l3_s16.golden.json

# gate/up + SiLU + encode, full-projection cut:
GATEUP_CUT='PLACE@map.1/map=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.3/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut'
EMMY_KNOBS="$GATEUP_CUT" ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_linear_reduce_2d1c79 --ir tile
# timing with the fp4 cell pinned:
EMMY_KNOBS="$GATEUP_CUT,TILE=mma_m16n8k64_e2m1_f32/f1x2/k8,STAGE=d3/smem-async" \
  ./venv/bin/emmy run --golden /tmp/l3_s16.golden.json --realization k_linear_reduce_2d1c79 --bench --no-record-evidence

# o_proj + residual + RMSNorm + encode, full-projection cut:
EMMY_KNOBS='PLACE@map.1/map=cut,PLACE@map.1/map.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,PLACE@map.2/map.2/reduce.4/map.1/reduce=cut,PLACE@map.3/map=cut,PLACE@map.3/map.2/inner=cut,PLACE@map.3/map.3/reduce.1/inner=cut' \
  ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_linear_mean_reduce_fd2717 --ir tile
```

Kernel names are the ones this trace gives at commit `a98fd4f8`. `emmy golden kernels /tmp/l3_s16.golden.json` prints
each kernel's Loop IR as one JSON line, and its `"name"` fields list the current names.

## Observed

Tile IR of the gate/up kernel under the cut. Three pieces each contract both weights over K = 5120:

```
=== 0: k_linear_reduce_2d1c79__place_4458599e17 ===          (16-value blocks)
    Fold[a3 in 0..5120] contraction   ⟨TILE=mma_m16n8k64_e2m1_f32/f1x2/k4 …⟩
    │  ├─ operand[in4]: load p_mlp_up_proj_weight_bits[((16 * a0) + a2), …]
    │  ├─ operand[in3]: load p_mlp_gate_proj_weight_bits[((16 * a0) + a2), …]
=== 2: k_linear_reduce_2d1c79__place_ca3a5e6d66 ===          (code pairs)
    ├─ operand[acc0, acc1, acc2, acc3]: Fold[a2 in 0..5120] contraction   ⟨TILE=mma_m16n8k64_e2m1_f32/f1x1/k8 …⟩
    │  │  ├─ operand[in3]: load p_mlp_gate_proj_weight_bits[(2 * a0), …]
    │  │  ├─ operand[in4]: load p_mlp_up_proj_weight_bits[(2 * a0), …]
    │  │  ├─ operand[in8]: load p_mlp_up_proj_weight_bits[((2 * a0) + 1), …]
    │  │  ├─ operand[in7]: load p_mlp_gate_proj_weight_bits[((2 * a0) + 1), …]
=== 3: k_linear_reduce_2d1c79__place_f98562e225 ===          (16-value blocks again)
    Fold[a3 in 0..5120] contraction   ⟨TILE=mma_m16n8k64_e2m1_f32/f1x2/k4 …⟩
    │  ├─ operand[in3]: load p_mlp_gate_proj_weight_bits[((16 * a0) + a2), …]
    │  ├─ operand[in4]: load p_mlp_up_proj_weight_bits[((16 * a0) + a2), …]
```

Pieces 0 and 3 have identical bodies and differ only in their workspace names.

The o_proj kernel under its cut streams `p_attn_o_proj_weight_bits` in four contractions:

- two independent block-layout contractions, `[((16 * a2) + a3), …]`, each split four ways over K with its own
  finishing kernel;
- one pair-layout contraction, `[(2 * a0), …]` and `[((2 * a0) + 1), …]`;
- one plain contraction, `[a1, …]`.

The first two are again identical.

Measured at 16 tokens on an RTX 5080 Laptop GPU, with the fp4-cell pin above: the gate/up kernel's three matmul pieces
take 139.2, 140.0 and 139.1 µs. That is about 418 µs for work that one pass would do.

## Suspected causes

Not traced. Two candidates, both in `emmy/compiler/pipeline/passes/tile/_cut.py`:

- `_cluster_value_seams` merges seams that materialize the same value only when their captured axes align in count and
  extent. The pair layout (`2*a0`, extent N/2) and the block layout (`16*a0 + a2`, extents N/16 and 16) do not align.
  That explains the pair-versus-block copies.
- Its docstring (around line 531) excludes output-owning and frontier seams from clustering altogether. That would
  explain copies with identical layouts, such as gate/up pieces 0 and 3.

## Compare

A 16-bit model has no re-encode, so this does not arise there. Not checked: whether the FP8 or AWQ Qwen3.8-27B recipes
route their requantize steps the same way.

## Fix criteria

For the two kernels above, under the same cut knobs:

- **Tile IR:** each weight (`p_mlp_gate_proj_weight_bits`, `p_mlp_up_proj_weight_bits`, `p_attn_o_proj_weight_bits`)
  appears in exactly one contraction. If that contraction is split over K, its split parts share one finishing kernel.
  The encode pieces read that one output buffer in their own layouts.
- **CUDA:** one fp4-cell GEMM per projection. The encode kernels load the materialized output and contain no
  contraction over the weight.
- **Measured:** compare projection time and total kernel-set time before and after at 16 and 512 tokens, on the same
  inputs and hardware with clocks and correctness recorded. The IR must remove the duplicate projections. The old
  roughly 140 µs per piece is an indicative baseline, not a guaranteed latency for the shared producer: its layout,
  schedule and intermediate-memory traffic may change. Explain any case where total latency does not improve.
- **Correctness:** the outputs match the current kernel set on the same inputs (`emmy run … --ab`, whose wrong-answer
  check compares both). This check drops out when the reference compile disagrees with itself: the bench then prints
  "wrong-answer reference unusable". A fixer must confirm the check actually ran.
- **Scope:** the q_proj and v_proj copies inside the attention kernel share the mechanism and should disappear with
  the same fix.
