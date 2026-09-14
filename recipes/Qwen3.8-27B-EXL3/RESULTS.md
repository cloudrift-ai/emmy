# Qwen3.8-27B at EXL3 5.0 bpw on one V100 SXM3 32GB

Compiler-qualified 2026-09-14 against `604ce74ea` on `riftuser@66.172.10.131`. This replaces the
2026-09-13 qualification against `9095cb297`: the compiler's region formation changed under it, so
every kernel that file recorded stopped existing and the whole inventory was re-traced and
re-measured.

This recipe has a golden and **no serving lane**. Nothing serves this checkpoint on this card: stock
vLLM has neither sm_70 support nor an EXL3 quantization method, and the 1Cat-vLLM Volta fork does not
read EXL3 either. The golden below is compiler evidence — measured CUDA kernels for the model's
decoder path — and it does not imply that the model can be served here. It cannot.

## What was measured

| Item | Value |
| --- | --- |
| Model | `turboderp/Qwen3.8-27B-exl3@a35e75a73baee51da709329d19294245cbeeb5d8` (5.0 bpw body, 6-bit head, 4-bit MTP) |
| GPU | 1 x NVIDIA Tesla V100-SXM3-32GB, compute capability 7.0, CUDA 12.9 |
| Architecture | 64 decoder layers: 48 `linear_attention` (Gated DeltaNet), 16 `full_attention`, plus 1 MTP layer |
| Inventory | 117 distinct kernels traced from two archetypes (layer 0 and layer 3), covering all 64 layers |
| Golden | 92 targets, 154 measured rows, recorded by `emmy run --golden … --bench --strict --record-greedy` |
| Reference | eager PyTorch (torch 2.13.0+cu126), same inputs, 10 warm-up / 100 measured iterations |

Every row was benched under `--strict`, so a kernel whose answer disagreed with eager was reported
and refused recording. Exactly one did — see "What is wrong" below.

## The result

Of the 97 targets the recording walk benched, **44 beat eager, 15 did not, 35 have no eager
reference** (their kernel holds part of a frontend operation, so no PyTorch slice computes exactly
them), and 3 could not be benched at all.

Where Emmy wins it wins by a lot:

| kernel | emmy us | eager us | vs eager |
| --- | ---: | ---: | ---: |
| `k_unsqueeze_pointwise_dde31a` | 3.4 | 156.9 | **45.7x** |
| `k_reshape_slice_transpose_reduce_8032e7` | 35.8 | 1373.7 | 38.4x |
| `k_reshape_slice_pointwise_b75dea` | 11.5 | 401.6 | 35.0x |
| `k_slice_unsqueeze_reduce_c30d5f` | 3.2 | 90.8 | 28.5x |
| `k_neg_bc_pointwise` | 1.4 | 34.7 | 25.5x |
| `k_tril_1_pointwise` | 15.8 | 546.9 | 34.6x |
| `k_mean_759f6d` | 47.8 | 743.2 | 15.5x |

## The Gated DeltaNet chunk family

`torch.export` unrolls the delta rule's chunk loop, so chunk *k* carries O(*k*) work while eager's
batched form amortizes it. Fourteen of the thirty chunk kernels lose; the other sixteen win
unpinned, the fastest at 3.2 us against 90.8 us eager.

For the fourteen the greedy elects one kernel at **grid 2 on an 80-SM card** with every reduce
serial. Three composed cuts —
`PLACE@map.1/map.1/inner=cut,PLACE@map.2/map.1/reduce=cut,PLACE@map.2/map.2/reduce=cut` — give a
four-kernel set instead, and that is what the golden records:

| kernel | greedy us | recorded us | eager us | recorded vs greedy | vs eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k_slice_unsqueeze_reduce_334002` | 501,703 | **15,848** | 3,320 | **31.7x** | 0.21x |
| `k_slice_unsqueeze_reduce_c47f85` | 405,410 | 14,706 | 3,241 | 27.6x | 0.22x |
| `k_slice_unsqueeze_reduce_492362` | 336,623 | 13,313 | 3,185 | 25.3x | 0.24x |
| `k_slice_unsqueeze_reduce_90cd88` | 262,050 | 12,154 | 3,086 | 21.6x | 0.25x |
| `k_slice_unsqueeze_reduce_09b2ec` | 217,824 | 11,223 | 3,016 | 19.4x | 0.27x |
| `k_slice_unsqueeze_reduce_7caa57` | 165,048 | 10,249 | 2,969 | 16.1x | 0.29x |
| `k_slice_unsqueeze_reduce_37da99` | 128,253 | 9,650 | 2,940 | 13.3x | 0.30x |
| `k_slice_unsqueeze_reduce_988cc6` | 97,736 | 8,756 | 2,848 | 11.2x | 0.33x |
| `k_slice_unsqueeze_reduce_8fe466` | 72,881 | 6,831 | 2,779 | 10.7x | 0.41x |
| `k_slice_unsqueeze_reduce_385df7` | 52,604 | 6,760 | 2,727 | 7.8x | 0.40x |
| `k_slice_unsqueeze_reduce_6fefd0` | 36,548 | 6,432 | 2,701 | 5.7x | 0.42x |
| `k_slice_unsqueeze_reduce_c1c725` | 24,427 | 6,275 | 2,657 | 3.9x | 0.42x |
| `k_slice_unsqueeze_reduce_cd0f4d` | 14,760 | 6,225 | 2,648 | 2.4x | 0.43x |
| `k_slice_unsqueeze_reduce_55cc08` | 3,352 | 3,352 | 2,637 | — | 0.79x |

The family goes from 2.32 s to 130 ms, **17.8x**, and still gives away 93 ms against eager. The last
row is the one case where the cut is a loss — 6,024 us against its own 3,352 us greedy — so its
greedy row is what the golden keeps.

## What is wrong

**`k_linear_left_scale32_bc_pointwise` returns a wrong answer.** 2,617,856 of 2,621,440 elements past
`rtol=1e-3` against eager, same index and same values every run, while timing 11.8 us against 160 us
eager. It is **not** in the golden: `--strict` caught it and the recording refused it. Its three
siblings pass at the same 13.5x. The kernel carries no `STAGE` pin, so this is not the Volta
`d2/smem` hazard; nothing in this inventory reaches the Volta tensor-core tier at all.

**Three EXL3 trellis broadcasts are materialized rather than fused**, at 1.4 ms, 1.7 ms and 2.7 ms,
and a fourth target (`k_conv1d_linear_matmul_reduce_93f65c`) does not lower at all. All four are the
tile lift declining a merged region, which the fusion rule turns into abandoning the whole region.
The chain in isolation fuses to one kernel; only inside the layer does it shatter.

**Twenty-five of the 117 targets are not in the golden**: three bench_fail, one does not lower, one
returns a wrong answer, and twenty were not reached — one target's compile did not terminate in
40 minutes and the walk was stopped there.
