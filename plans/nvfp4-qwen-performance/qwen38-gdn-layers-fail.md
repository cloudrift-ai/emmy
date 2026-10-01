# Qwen3.8 Gated DeltaNet: tracing, serving, codegen and scheduling failures

2026-09-30 update: [PR #973](https://github.com/cloudrift-ai/emmy/pull/973) repairs the compiler failures and adds
static explicit-state capture. Prefill-to-decode handoff, seeded state and reset pass on both RTX 5080 and RTX 5090;
the compiler/capture fix landed in #973 after full-suite validation and repair of a missing test prerequisite.
Native request dispatch and whole-model serving remain separate integration work. The PR records the implementation
and validation evidence. The observations below are the original report, with their original revisions and dump
provenance.

## Summary

48 of the 64 layers of `Inferact/Qwen3.8-27B-NVFP4` are gated DeltaNet (GDN) layers. GDN is a linear-attention token
mixer. The traced implementation processes 64-token chunks and carries state between them. With emmy's own kernels on
sm_120, these layers fail
at several stages, from tracing and serving capture to CUDA generation and scheduling. These failures block native
serving before meaningful end-to-end tuning can begin. The report groups five observed failures; it does not assume
five independent root causes.

## Observed IR and expected lowering

The saved four-cut Tile IR at `a98fd4f8` declares sibling output sweeps:

```text
    outputs
    ├─ sweep(a0.a1) reshape_9[0, a0, 0, a1] = v40__wsec2ded4f48
    ├─ sweep(a0.a1) to_7[0, a0, a1] = v90__wsec2ded4f48
    ├─ sweep(a0.a1.a6) reshape_5[0, a0, 0, a1, a6] = v134__ws01ef1a7daas0
    ├─ sweep(a0.a1.a6) reshape_7[0, a0, 0, a1, a6] = v135__ws01ef1a7daas0__wsec2ded4f48
    ├─ sweep(a8.a10) type_as[0, a8, a10] = v161__ws825fe6030es0
    └─ sweep(a8.a11) transpose_1[0, a8, a11] = v203__wsbab87715b8
```

The saved default CUDA dump shows the same failure: it nests the independent `a10` and `a11` output domains inside
`a0`, `a1`, `a8` and `a6`. These are actual emitted lines; comments mark omissions, including the closing braces:

```cuda
        for (int a0 = 0; a0 < 48; a0++) {
            for (int a1 = 0; a1 < 64; a1++) {
                // …
                for (int a8 = 0; a8 < 64; a8++) {
                    // …
                    for (int a6 = 0; a6 < 128; a6++) {
                        // …
                        for (int a10 = 0; a10 < 5120; a10++) {
                            // …
                            type_as[a8 * 5120 + a10] = v161__wsdaf9596f95;
                            for (int a11 = 0; a11 < 10240; a11++) {
                                __half v203__wsa98345b786 = type_as__place_a98345b786_0[a8 * 10240 + a11];
                                transpose_1[a8 * 10240 + a11] = v203__wsa98345b786;
```

Expected CUDA structure, **composed, not emitted**. The unchanged value calculations and the other output stores are
omitted; the two shown stores must not be nested inside each other's domains or the unrelated `a0`/`a1` sweeps:

```cuda
for (int a8 = 0; a8 < 64; a8++) {
    for (int a10 = 0; a10 < 5120; a10++) {
        // … unchanged calculation of v161__wsdaf9596f95 …
        type_as[a8 * 5120 + a10] = v161__wsdaf9596f95;
    }
    for (int a11 = 0; a11 < 10240; a11++) {
        __half v203__wsa98345b786 = type_as__place_a98345b786_0[a8 * 10240 + a11];
        transpose_1[a8 * 10240 + a11] = v203__wsa98345b786;
    }
}
```

Tile IR already distinguishes the output domains; generated CUDA is the level where the erroneous nesting is visible.
The independent sweeps may share prefixes or use different tiling. Their extents must not multiply each other's work.

## Observed failures

1. **No trace below 64 tokens.** Compiling a GDN layer at `--seq-len 16` fails with `NotImplementedError: aten.pad
   supports only explicit zero-width padding, got [0, 0, 0, 48]` (`[0, 0, 0, 63]` at `--seq-len 1`). The Hugging Face
   chunked delta rule pads the sequence up to a multiple of 64, and emmy lowers `aten.pad` only for zero-width
   padding. The 16-bit sibling `Qwen/Qwen3.8-27B` fails the same way. `recipes/Qwen3.8-27B-AWQ-INT4/RESULTS.md`
   records the same failure at `--seq-len 1`.
2. **No serving program.** Serving-twin capture refuses the model: `NotImplementedError: serving twins: layer 0
   (Qwen3_5DecoderLayer, linear_attention) has no self_attn; blocks whose token mixer is not attention (e.g. a gated
   delta net) have no serving program yet` (`emmy/serving/twins.py`, line 289). So `emmy serve` cannot run this model
   on emmy-compiled kernels.
3. **CUDA codegen crash at 512 tokens.** Kernel `k_matmul_reduce_81b2ae`, the chunk-to-chunk state recurrence, fails
   to render with `assert len(self.srcs) == 2` in `FragmentRepack.render` (`emmy/compiler/ir/kernel/ir.py`, line
   2038). The failing schedule is the default one here, `TILE=mma_m16n8k16_f16_f16/f1x16/k4 STAGE=d1/reg` with
   `WORK=w4x1`.
4. **Runaway serial loops at 64 tokens.** The input-projection kernel `k_conv1d_linear_mean_reduce_c4b163` covers
   RMSNorm, the `in_proj_qkv`/`in_proj_a`/`in_proj_b` projections and the causal 1-D convolution. It does not finish
   within the 60 s kernel watchdog, with the default schedule or with four cuts pinned. This is not an infinite loop.
   The kernel's final piece, which writes its outputs, nests three output sweeps that Tile IR declares as siblings
   (`sweep(a0.a1)`, `sweep(a8.a10)`, `sweep(a8.a11)`) into one loop nest, `a0<48 > a1<64 > a8<64 > a6<128 > a10<5120 >
   a11<10240`. That is about 1.3×10¹⁷ iterations on one thread. The core kernel `k_linear_matmul_mean_reduce_5c131b`
   (chunked delta rule, gated norm, `z` gate, activation encode) exceeds the 130 s bench budget. Not examined.
5. **The 16-bit input projections stay on the scalar tier.** Threads compute outputs without tensor cores. The
   checkpoint leaves `linear_attn.in_proj_qkv`,
   `in_proj_z`, `in_proj_a`, `in_proj_b` and `conv1d` in bf16, about 168 MB per layer. In Tile IR these weights appear
   as `linear_wt` (`in_proj_qkv`), `linear_1_wt` (`in_proj_z`), `linear_2_wt` and `linear_3_wt`.
   - After the cuts below, the `in_proj_qkv` contraction gets no TILE, only `REDUCE=coop`.
   - Its operand 0 is the weight load `linear_wt[a2, a1]`, stored K×N, and the convolution taps appear as four more
     A-side channels.
   - `in_proj_z` (`linear_1_wt`) appears four times inside the core kernel, all without a TILE.

Also slow: the chunk triangular solve `k_slice_unsqueeze_reduce_b17b4d` takes 16 ms at 64 tokens, with 196,608 thread
blocks of 128 threads. It is a 62-step recurrence the compiler rolled into one loop.

Missing padding and serving capture are support gaps that block this model just as concretely as the codegen and
scheduling failures. They remain part of the finding.

## Reproduce

All commands run from the repository root inside `nix develop`, with a fresh tune DB. `emmy trace` writes a layer
inventory; `--realization <name>` selects one of its kernels. Each compile also applies the inventory's own row for
that kernel, printed as "1 automatic pin".

```sh
M=Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462   # ~26 GB download on first use
rm -f /tmp/q38.db; export EMMY_TUNE_DB=/tmp/q38.db

# 1. No trace at 16 tokens (same with Qwen/Qwen3.8-27B):
./venv/bin/emmy compile $M --layer 0 --seq-len 16 --target sm_120 --ir loop
# NotImplementedError: aten.pad supports only explicit zero-width padding, got [0, 0, 0, 48]

# 2. No serving program (the CLI path needs a serving config; the capture function shows the refusal directly):
./venv/bin/python -c "from emmy.serving.twins import capture_twin_graphs; capture_twin_graphs('$M', decode_bucket=1, prefill_bucket=0, symbolic=False, static_only=True)"
# NotImplementedError: serving twins: layer 0 (Qwen3_5DecoderLayer, linear_attention) has no self_attn; ...

# 3. Codegen crash at 512 tokens:
./venv/bin/emmy trace $M --layer 0 --seq-len 512 --target sm_120 -o /tmp/l0_s512.golden.json
./venv/bin/emmy compile --golden /tmp/l0_s512.golden.json --realization k_matmul_reduce_81b2ae --ir cuda
# AssertionError at emmy/compiler/ir/kernel/ir.py:2038, assert len(self.srcs) == 2

# 4. Runaway loops at 64 tokens:
./venv/bin/emmy trace $M --layer 0 --seq-len 64 --target sm_120 -o /tmp/l0_s64.golden.json
CUTS='PLACE@map.1/map.1/map.1/reduce.1/inner=cut,PLACE@map.1/map.1/map.2/inner=cut,PLACE@map.4/map.1/inner=cut,PLACE@map.1/map.1/map.1/reduce.1/inner.2/map.6/map.1/reduce=cut'
EMMY_KNOBS="$CUTS" ./venv/bin/emmy run --golden /tmp/l0_s64.golden.json --realization k_conv1d_linear_mean_reduce_c4b163 --bench --no-record-evidence
# emmy_runtime.HungKernelError: kernel "k_conv1d_linear_mean_reduce_c4b163" did not complete within 60000 ms
EMMY_KNOBS="$CUTS" ./venv/bin/emmy compile --golden /tmp/l0_s64.golden.json --realization k_conv1d_linear_mean_reduce_c4b163 --ir cuda
#   the last kernel holds the six-deep serial nest described above
./venv/bin/emmy run --golden /tmp/l0_s64.golden.json --realization k_linear_matmul_mean_reduce_5c131b --bench --no-record-evidence
# bench worker exceeded 130.0s wall budget

# 5. The in_proj_qkv contraction without a TILE:
EMMY_KNOBS="$CUTS" ./venv/bin/emmy compile --golden /tmp/l0_s64.golden.json --realization k_conv1d_linear_mean_reduce_c4b163 --ir tile
#   the contraction over linear_wt has REDUCE=coop and no TILE; its operand 0 is `load linear_wt[a2, a1]`
```

Kernel names are the ones this trace gives at commit `a98fd4f8`. `emmy golden kernels <inventory>` prints each
kernel's Loop IR as one JSON line; its `"name"` fields list the current names.

## Known and suspected causes

- **Failure 1:** `transformers`' `torch_chunk_gated_delta_rule` (`modeling_qwen3_5.py`, lines 270–275) pads to the
  chunk size, and emmy's `aten.pad` lowering accepts only zero-width padding.
- **Failure 3, rewrite defect confirmed in isolation:** the name-rewrite of `FragmentRepack` in
  `emmy/compiler/ir/kernel/ir.py` rebuilds the node without its `role`. A CPU-only probe at `a98fd4f8` renders a
  one-source B repack successfully, then renames it through the existing rewrite and gets the two-source assertion.
  The saved failing Kernel IR has the same malformed form: `FragmentRepack _rf[203] <- ('_rf[61]',)` without
  `role=b`. The exact passage of that checkpoint node through the rewrite remains untraced; fixing this defect has
  not yet been shown to make the whole GDN kernel compile.
- **Failure 4, suspected:** the trailing-run rule that places output sweeps (`_sweep_start` in
  `emmy/compiler/ir/tile/ir.py`) nests sibling sweeps. `promoted_sweep` does not promote any of them to the grid,
  because no axis is shared by every store.
- **Failure 5, suspected:** `_node_refusal` in `emmy/compiler/ir/schedule/classic/refusals.py` refuses the tensor-core
  tier because operand 0 is the K×N weight, whose gmem index moves 10,240 elements per contraction column, where the
  fragment loaders need K contiguous. It returns this reason for the contraction: "warp TILE: A fragment loaders read
  16 contraction columns CONTIGUOUSLY, but this operand's gmem index moves 10240 elements per column".

The isolated repack check needs no GPU:

```python
from emmy.compiler.ir.kernel.ir import FragmentRepack, RenderCtx
from emmy.compiler.ir.stmt.passes import rewrite

before = FragmentRepack(frag="b", srcs=("c",), role="b")
after = rewrite(before, lambda name: "renamed_" + name)
print(before.pretty()[0])
print(after.pretty()[0])
before.render(RenderCtx())  # succeeds
assert after.role == "a"  # the rewrite lost role="b"
after.render(RenderCtx())  # AssertionError: the A form requires two sources
```

Actual Kernel IR printed by this probe:

```text
FragmentRepack b <- ('c',) (f16, m16n8k16, part=0, role=b)
FragmentRepack renamed_b <- ('renamed_c',) (f16, m16n8k16, part=0)
```

## Compare: what shipped recipes do

The shipped Qwen3.8-27B recipes (`recipes/Qwen3.8-27B{,-FP8,-AWQ-INT4,-GPTQ-Int4,-EXL3}`, goldens for V100) serve
their GDN layers with vLLM's Triton kernels ("Triton Gated DeltaNet prefill" in their `RESULTS.md`), not emmy's. Their
goldens do cover emmy's GDN kernels at 64 tokens and wider: `recipes/Qwen3.8-27B-EXL3/RESULTS.md` has a section "The
Gated DeltaNet chunk family", and the AWQ inventory traces a GDN decoder layer with 196 configurations.

## Fix criteria

The following observable outcomes correspond to the failures above. Their implementation may share fixes:

1. `emmy compile $M --layer 0 --seq-len 16 --target sm_120 --ir loop` and `--seq-len 1` succeed for both
   `Inferact/Qwen3.8-27B-NVFP4` and `Qwen/Qwen3.8-27B`. Padding has zero-fill semantics, valid input reads stay in
   bounds, and logical outputs have the original unpadded shape. Internal padded buffers are permitted.
2. Serving-twin capture returns programs for the GDN layers, and `emmy serve` boots the model with emmy kernels at
   least on its full-attention layers.
3. `emmy compile … --realization k_matmul_reduce_81b2ae --ir cuda` renders CUDA under the default schedule and under
   every warp TILE its fork offers. The Kernel IR prints `role=b` on a one-source B repack.
4. Under the four cuts above, no piece of `k_conv1d_linear_mean_reduce_c4b163` nests independent sibling sweeps. Each
   output loops over its own axes, with shared prefixes allowed; unrelated output domains must not multiply its work.
   Check this in emitted CUDA and add a focused regression case. The kernel set completes within the existing
   watchdog, with correctness checked. Report its latency and remaining bottlenecks; a weight-bandwidth estimate is
   context, not a justified bound for this combined norm, projection and convolution workload.
5. With some cut set, the `in_proj_qkv` and `in_proj_z` contractions carry a tensor-core TILE (`mma_m16n8k16_…`).
   Their operand 0 is the activation (the normed hidden state, with the convolution taps), their B operand is the
   weight, and the CUDA calls `emmy_mma_m16n8k16_*`.

Correctness for 3–5: compare the checkpoint-based reproducer with a usable eager or independently validated reference,
on identical inputs and with a stated tolerance. Establish whether `emmy run --strict` works for this checkpoint path;
the observed failures of inline `-c … --quantize` programs do not establish a failure here. A scalar comparison via
`emmy run … --ab '<scalar knobs>'` is useful only if that reference runs and its outputs are validated. If the bench
prints "wrong-answer reference unusable", no correctness check passed. A crash or impractically slow scalar reference
must be replaced by a tractable focused reproducer or another validated reference, not counted as success.

## Notes

The card behind these numbers, an RTX 5080 Laptop GPU, was power-capped during the runs (memory clock 9 GHz instead of
14 GHz). Timings are indicative only.
