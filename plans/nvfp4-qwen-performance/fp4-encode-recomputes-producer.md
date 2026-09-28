# FP4 re-encode cuts duplicate producer projections

## Summary

A cut splits a fused kernel into smaller kernels, called pieces. Here, the full-projection cut intended to expose
quantized projections to tensor-core schedules instead repeats the same projection for several consumers.
In `Inferact/Qwen3.8-27B-NVFP4` layer 3 at 16 tokens, Tile IR contains three complete gate/up projections and four
complete o_proj projections. Each reads the full weight again. Some copies have identical bodies and layouts.

The consumers need different views of the projection output: pairs of values to pack two 4-bit codes per byte, and
16-value groups to compute and store quantization scales. Those views should be able to read one computed result.
The observed q_proj and v_proj also repeat, four and three times respectively.

## Observed and expected IR

This is a normalized dataflow sketch of the gate/up Tile IR, not literal compiler syntax. `project` means the full
contraction over K = 5120; the piece numbers match the excerpt below. The sketch omits SiLU and indexing details.

```text
Observed after the full-projection cut
piece 0: project(gate_weight, up_weight, x) -> workspace_0   # 16-value groups
piece 2: project(gate_weight, up_weight, x) -> workspace_2   # code pairs
piece 3: project(gate_weight, up_weight, x) -> workspace_3   # 16-value groups again

Expected sharing (illustrative; not emitted today)
producer: project(gate_weight, up_weight, x) -> shared_output
consumer: read shared_output as pairs      -> packed codes
consumer: read shared_output as groups     -> block scales
```

The important difference is one evaluation of each projection, with its result reused across consumers. A legal K
split may still use partial contractions and a finishing reduction; it must not repeat the full projection for each
read layout.

## Reproduce

All commands run from the repository root inside `nix develop`, with a fresh tune DB. The `PLACE@<site>=cut` pins
select kernel boundaries in the schedule tree. The fp4-cell TILE selects the native instruction
`mma_m16n8k64_e2m1_f32`, which multiplies packed codes and applies their block scales in hardware.

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
take 139.2, 140.0 and 139.1 µs. That is about 418 µs across the three pieces. This does not establish the latency of a
shared producer.

## Suspected causes

The causal links are unverified. The full-projection cut comes from `full_projection_seams`; it cuts contraction
occurrences, reductions hoisted ahead of output sweeps, and output-owning branches. Two candidates for why it repeats
work are in `emmy/compiler/pipeline/passes/tile/_cut.py`:

- `_cluster_value_seams` merges seams (potential kernel boundaries) that materialize the same value only when their
  captured axes align in count and
  extent. The pair layout (`2*a0`, extent N/2) and the block layout (`16*a0 + a2`, extents N/16 and 16) do not align.
  That could explain the pair-versus-block copies; the causal link is unverified.
- Its docstring (around line 531) excludes output-owning and frontier seams from clustering altogether. That would
  explain copies with identical layouts, such as gate/up pieces 0 and 3.

## Compare

A 16-bit model has no fp4 re-encode, so it does not have this particular source of repeated reads. Not checked:
whether the FP8 or AWQ Qwen3.8-27B recipes
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
