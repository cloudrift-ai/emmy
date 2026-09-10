# Block-scaled FP8 linear optimization (Qwen3-0.6B-FP8, RTX 5090)

> Written 2026-09-10 from PR #777 / branch `codex/quantization-kernels-rtx5090`. Block-scaled FP8 is a
> support-and-correctness result today, NOT a speed win. This plan is the next round: make it fast, and measure it
> against a matched baseline. Findings doc: `evaluation_results/2026-09-10_fp8-block-linear-rtx5090.md`.

## Goal

Make Emmy's block-scaled FP8 linear projections (`Qwen/Qwen3-0.6B-FP8`: 128x128 weight blocks, per-token dynamic
activations) competitive with a matched block-FP8 kernel, then report the result honestly. Success: at M=16 batched
decode width, Emmy's fused RMSNorm + FP8-quant + block-FP8 matmul is within striking distance of, or beats, vLLM's
block-FP8 CUTLASS / DeepGEMM GEMM on the same shapes, verified on a serving host.

## The one win to chase: a tensor-core block-scale FP8 matmul

This is the whole plan. Block-FP8's weight scale varies every 128 elements ALONG the contraction (K) axis, so it
does not commute out of the reduction and cannot ride the matmul epilogue (the W8A16 mul-hoist — see
`emmy/compiler/ARCHITECTURE.md`, the "does not commute out of the fold" note). Consequences observed:

- Emmy offers NO tensor-core atom for this matmul and falls back to a cooperative scalar fold.
- `FP8_MMA=1`, even with a K-chunked tile (`mma_m16n8k32_f16_f32/.../k4`), does not realize a tensor-core row for
  the fused norm+matvec projection.

What is missing: a K-chunked block-scale FP8 tensor-core matmul that applies each 128-wide K block's scale on the
f32 accumulator per K-chunk (the DeepGEMM shape). Emmy's packed byte-slab stage already does the analogous
per-16-block decode for NVFP4 4-bit weights (`loader/quant.py` `_packed`, and the compiler-arch "packed byte-slab
stage" section) — the fp8 128-along-K case is the same idea one step further. Start there.

## Measured baseline reality (decode M=1, RTX 5090, -O3)

Fused RMSNorm + fp8 quant + matmul, per projection:

| projection | eager | torch.compile | Emmy tuned |
| --- | ---: | ---: | ---: |
| q_proj 1x1024x2048 | 69 us | 14 us | 63 us |
| k_proj 1x1024x1024 | 69 us | 16 us | 61 us |

Emmy beats naive eager 2.6x but is ~4x slower than torch.compile. DO NOT report vs eager — it overstates by ~4x.
Caveat: torch.compile's fast path quantizes ROWWISE (scale on the epilogue), an easier problem than 128-block, so
it is a LOWER BOUND on stock, not the matched kernel.

## What you need to install locally to measure the matched baseline

Nothing already-present is enough: `torch._scaled_mm` (torch 2.14, present) does per-tensor and rowwise fp8 but
NOT 128-block scales, and vLLM is not in the compiler venv. To get the matched block-FP8 baseline you need EITHER:

- **vLLM** (the NVFP4 showcase used 0.29.0) built against this CUDA (13.0). Its block-FP8 path is
  `cutlass_scaled_mm` / the w8a8 block GEMM (`vllm.model_executor.layers.quantization.utils.fp8_utils`). This is
  what `scripts/bench_nvfp4_linear.py` binds for NVFP4; the block-FP8 analog would call the fp8 block path. Heavy
  install, version-sensitive; the NVFP4 showcase ran it on a serving host, not the compiler box. Recommended path:
  run the matched comparison on a serving host, or install it into the local venv (`./venv/bin/pip install vllm`)
  since the box has CUDA 13 + nvcc.
- **DeepGEMM** (`deep_gemm`) for a standalone block-FP8 GEMM reference, if a lighter local baseline than full vLLM
  is wanted. Also needs a matching CUDA/torch build.

nvcc must be on PATH before any vLLM baseline so vLLM's fast kernels actually compile (a misconfigured nvcc silently
drops vLLM to a slow fallback and inflates the win — see [[quant-baseline-flashinfer]] in agent memory). Measure at
M=16, the width the NVFP4 showcase uses.

## Tuning that works today (correct deploy, not a win)

The greedy misprices the fused norm+matvec as a scalar tile (q_proj 2983 us; k_proj greedy 416451 us).
`EMMY_KNOBS="WORK=t256,REDUCE=coop"` recorded via `--record-greedy` takes the cut + cooperative fold: q_proj 63 us,
k_proj 61 us. `coop-t` and `coop/r2` are worse. The committed decode golden records this:
`experiments/golden-bench-2026/quantized_kernels_rtx5090/golden/qwen3-06b-fp8-block-s1_rtx5090.golden.yaml`.

## Open gaps (do not treat as this plan's blockers, but know them)

- **Fused decode-tail hang.** `k_sdpa_linear_mean_reduce_ed9ff3` (attention + projections + MLP) over-fuses the
  whole decode layer into one kernel that HANGS at runtime; `PLACE=cut` offers no seam and a pinned coop hangs the
  compiler >5 min. Still broken after rebasing onto #737. Needs a fusion-policy change (don't fuse the whole decode
  tail into one kernel). Separate from the matmul-atom work above.
- **Prefill selection drift.** At seq>1 (incl. M=16) the origin-goldens hit "the persisted target selects no kernel
  after lowering". Use `emmy trace --loop-targets`, but loop targets have no eager twin (emmy-only bench). The
  prefill matmuls are untuned; they want M-tiled tensor-core tiles, not the M=1 coop fold — which the block-scale
  atom work above also unblocks.

## Compiler fixes already landed on this branch (keep, do not redo)

- `EMMY_PRICE_BUDGET_S` (wall-clock budget on greedy kernel-set pricing, default 30 s) + a shared grid-placed
  schedule view per kernel: together they make the fused decode-tail COMPILE in 45 s (was >30 min).
- Strict-bench accepts an embedded Loop target whose reference is same-input-greedy, not eager.
