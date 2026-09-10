# Block-scaled FP8 linear kernels at decode width on RTX 5090

Date: 2026-09-10. GPU: NVIDIA GeForce RTX 5090 (`sm_120`), driver 580.178.04. Software: CUDA 13.0,
PyTorch 2.14.0+cu130. All timings use deployable nvcc defaults (`-O3`). Checkpoint:
`Qwen/Qwen3-0.6B-FP8` (block-scaled FP8, 128x128 weight blocks, per-token dynamic activations).

This is the block-scaled FP8 companion to the NVFP4 linear showcase. It reports the same shapes and the same
strongest-baseline discipline, and it reaches a different conclusion: block-scaled FP8 is a support-and-correctness
result on Emmy today, not a performance win against a real FP8 kernel.

## The projections beat naive eager but lose to a real FP8 pipeline

The decode (M=1) fused projection kernel computes RMSNorm, the per-token FP8 activation quantization, and the
block-scaled FP8 matmul in one launch. Emmy's greedy scheduled it as a scalar tile (the k projection greedy ran
416 ms); a hand-pinned cooperative fold recovers most of that. But the strongest baseline is not eager.

| projection | eager pipeline | torch.compile pipeline | Emmy tuned | Emmy / torch.compile |
| --- | ---: | ---: | ---: | ---: |
| q_proj (1 x 1024 x 2048) | 69 us | 14 us | 63 us | 0.22x |
| k_proj (1 x 1024 x 1024) | 69 us | 16 us | 61 us | 0.26x |

The eager column is PyTorch run op by op. The torch.compile column is the same RMSNorm + FP8 quantize + FP8 matmul
fused by Inductor into Triton kernels. Emmy beats eager 2.6x but is about 4x slower than torch.compile.

## Why: no tensor-core block-FP8 matmul

torch.compile's fast pipeline quantizes the activation ROWWISE, so its scale is one value per row and rides the
matmul epilogue on the accumulator, leaving a clean FP8 tensor-core GEMM. Block-scaled FP8 is a harder problem: the
weight scale varies every 128 elements ALONG the contraction axis, so it does not commute out of the reduction and
cannot ride the epilogue (`emmy/compiler/ARCHITECTURE.md`, the W8A16 mul-hoist note). Emmy therefore does not offer
a tensor-core atom for this matmul and falls back to a cooperative scalar fold. `FP8_MMA=1` with a K-chunked mma
tile does not realize a tensor-core row for the fused kernel for the same reason.

A competitive block-FP8 kernel applies each 128-wide K block's scale on the f32 accumulator inside a tensor-core
mma, the DeepGEMM shape. Emmy's packed byte-slab stage already does the analogous per-16-block decode for NVFP4
4-bit weights; the FP8 128-block-along-K case is the missing schedule. That is the compiler work a block-FP8
performance claim needs, and it is out of scope for this change.

## The fair baseline needs a serving host

torch.compile rowwise is an easier problem than block-scaled FP8, so it is a lower bound on stock, not the matched
kernel. The matched baseline is vLLM's block-FP8 CUTLASS or DeepGEMM GEMM, which the NVFP4 showcase measured on a
serving host with vLLM installed. vLLM is not installable in the local compiler venv, so this report uses
torch.compile as the strongest LOCAL stock reference and defers the matched vLLM block-FP8 comparison to the
serving host, at the M=16 batched-decode width the NVFP4 showcase uses.

## What is committed

The RTX 5090 decode golden under `experiments/golden-bench-2026/quantized_kernels_rtx5090/golden/` records the
tuned cooperative-fold schedule so the support check deploys a correct, non-pathological kernel set rather than the
scalar greedy. It is correctness-and-support evidence. The performance conclusion above is the reason the recipe's
block-FP8 rows are not presented as a speed win.
