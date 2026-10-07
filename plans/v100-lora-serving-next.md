# V100 LoRA serving: next steps

Llama 3.1 8B Instruct, rank-8 `limo` adapter, one Tesla V100-SXM3-32GB, FP16, 511 input / 256 output tokens.
Golden: `experiments/Meta-Llama-3.1-8B-Instruct/lora_v100_sxm3_32gb/golden/v100_sxm3_sm70_lora8.json`.
State after PR #1080 and the 2026-10-07 prefill pass (#1093): adapter and mixed requests run 1.6–2.1 times faster
than stock vLLM; base time per output token matches stock; base throughput is about 4% below stock at concurrency 8
and 16 because first-token time is 19–26% higher. The serving matrix was not re-run after #1093.

## 1. 512-row post-attention program (first-token time)

Measured on the card (whole program, strict replay, µs): 2970 Emmy before #1093, 2936 after, against
2614 for `torch.compile`. ncu of the program beside eager PyTorch's cuBLAS kernels (per launch):

- Gate/up (one fused kernel, two outputs): 1562 against 667 + 672 for two cuBLAS GEMMs. The kernel already prefetches
  through registers across the barrier (the Volta split copy); its DRAM traffic was 64% of peak because the flat CTA
  order streams the 235 MB of weights once per M block. `RASTER=gm8` (all four M blocks adjacent) reads them once:
  1531 → 1408 µs alone, but only 1497 → 1475 inside the program (1392 → 1347 in the emmy-only re-bench). In place
  under ncu the DRAM share fell from 64% to 21% of peak and the kernel got 1% faster, so it was never DRAM-bound:
  the tensor pipe is active 73% of the time against cuBLAS's 81% with the same occupancy and fewer shared-memory
  bank conflicts. The limiter is on the SM side (issue order, fragment-load latency, the per-chunk barrier); read
  the warp stall reasons (`exp/stalls` on the card) before changing the lowering. Why the same kernel runs 10% slower
  inside the program than alone is also open.
- Down: 831 against 768 (cuBLAS 128x256 tile, also 64 CTAs on 80 SMs). No schedule in a 20-row tune beat the recorded
  `w8x2 f2x4` row in the program; `w1x4 f4x4` won alone (816) and lost in place (853).
- Output projection: 250 against 230. No better row in a 20-row tune or the raster A/B.
- Down LoRA shrink (512×14336×8): 75–81 against 31 for cuBLAS's split-K wmma kernel. The schedule space for this
  shape is scalar cooperative reductions only; a tensor-core split-K form is a kernel-set decision (`g<n>k`), not a
  `--tune` row. Try recording one.
- LoRA expands (512×8×14336): 47 → 40 each with `w1x2 f2x1 k2`; cuBLAS 34.
- The non-GEMM kernels (silu·mul with the adapter mask, norms, residual adds, the adapter shrinks) are about 430 µs
  of the program against roughly 280 for `torch.compile`'s fused elementwise kernels; fusing the expand into the
  silu·mul kernel would save about 60 MB of traffic per layer. That is a placement decision for evidence.
- Attribute the other ~12 ms of first-token time: profile one 511-token prefill in the serving container with the
  PyTorch profiler.

## 2. Measurement contexts

Three numbers exist for one kernel: alone as a one-kernel program (`--kernel`), inside the program beside the torch
backends (the kernel table), and inside the program emmy-only (the "greedy (isolated)" row every golden row stores).
For the gate/up kernel they read 1554 / 1497 / 1392 µs. The recorded 1394 was the isolated number, not a minimum
artifact: the minimum and the median agree within 0.5%. `--record-greedy` now records the median anyway, the
statistic the tune DB ranks by.

`--strict` moves the whole-program numbers: on the same card and realization, `torch.compile` reads 2580 µs without
it and 3150 with it, Emmy 3040 and 2957. A strict bench times every backend on the inputs the correctness gate bound
(`_bind_inputs`, the bench worker's in-child gate); a plain bench draws its own. The GEMMs run at different speeds on
different data, most likely through power and clocks. The seed rows' `latency` entries follow the plain convention;
the kernel rows (emmy-only, on the gate's inputs) are consistent across sessions. Pick one input set for both paths.
`--record` and `--record-greedy` also store different numbers in a seed row's `latency`: the first the pinned row's
own emmy-only whole-program minimum (2681 µs here), the second the greedy table's number beside torch (3003). The
file's entries are the second kind; make the two flags write the same statistic.

## 3. Remaining base gap at concurrency 8 and 16

Unchanged from #1080: an 8-row decode program, and the 16-row gate/up and down projections reaching 850–875 GB/s
where the one-row forms reach 940.

## 4. Smaller items

- `--tune` refuses a program with more than one scheduled kernel, so a split kernel cannot be tuned with it.
- The source weight layout (`LAYOUT=source`) was not evaluated.
