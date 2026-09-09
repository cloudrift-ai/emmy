# V100 (sm_70): the matmul gap is the fragment gather, not the schedule

Written 2026-09-09, after the V100 matmul rows were re-recorded on the two-deep pipeline. That change took the
family from as much as 2.4x cuBLAS to 1.4-1.7x and exhausted the schedule search: split-K, rasterization, lighter
tiles and the fragment ping-pong were each measured on the card and none of them pays. This note records where the
rest of the distance lives, read off the emitted CUDA rather than guessed.

## The number to beat

4096-square f16, Tesla V100-SXM3-32GB, tensor-core peak 131 TFLOPS:

| | TFLOPS | of peak |
| --- | ---: | ---: |
| cuBLAS | 105 | 80% |
| emmy, after the re-record | 72 | 55% |

Same card, same instruction — V100 has only `mma.sync.m8n8k4`, so this is not an instruction gap. It is how
operands reach the tensor core.

## The finding: every fragment is built from four 2-byte shared loads

`emmy_mma884_load_a_impl` / `_load_b_impl` in the emitted CUDA assemble one 4-half fragment element by element:

```c
for (int p = 0; p < 2; ++p) {
    int k = p << 1;
    unsigned packed = 0;
    if (k < k_left)     ((F*)&packed)[0] = F(g[row * ldm + k]);
    if (k + 1 < k_left) ((F*)&packed)[1] = F(g[row * ldm + k + 1]);
    r[p] = packed;
}
```

Four scalar `__half` loads, packed a half at a time into two registers. In the o_proj winner (`w2x2`,
`mma_m8n8k4_f16_f32/f4x4/k8`, `d2/smem`, CTA tile 256x64, K slab 8) one inner `_ki` step issues **8 A fragments +
2 B fragments = 10 fragments, so 40 two-byte shared loads, to feed 16 `mma` instructions**. Roughly 2.5 shared-load
instructions per tensor-core instruction.

Volta has no `ldmatrix`, so the gather must be hand-built — but it does not have to be scalar. CUTLASS's Volta path
stores the slab in a crosswise / swizzled layout so a thread's four halves are contiguous and arrive in **one 64-bit
`LDS`**. That is a 4x cut in shared-load instructions and a much wider access per instruction.

This is the prime suspect and everything else below is smaller.

## Supporting observations, all from the emitted kernel

- **The ping-pong emits real code and still changes nothing.** `STAGE=d2/smem/p2` produces 9 fragment-load sites
  against 2 without it, so the double buffer is genuinely in the source. It measured 294 vs 294 us on o_proj and
  worse on qkv (998 vs 877). Consistent with the gather being issue-bound rather than latency-bound: prefetching
  earlier on a saturated path buys nothing. Expect `/p2` to start paying only after the gather is widened.
- **Occupancy is 12%** — one 128-thread CTA per SM at ~250 registers per thread, so four warps to cover a ~30 cycle
  shared-load latency. CUTLASS's Volta HGEMM runs 256 threads per CTA.
- **The epilogue is scalar as well**: eight separate 2-byte global stores per accumulator fragment, on a strided
  pattern, about 128 scalar stores per thread on the o_proj tile.
- **The staged copy round-trips a `uint4` through eight named scalars** before re-packing it for the `uint4` store.
  Register counts were flat when this was checked, so nvcc folds it, but it is eight extracts and eight inserts of
  source noise sitting on the hot path.
- **The K slab is 8 elements deep** and `bk`'s domain is capped at `(1, 2, 4, 8)` in `ir/schedule/catalog.py`.
  A deeper slab is only worth offering once a gather from it is cheap.

## Levers, ranked

1. **Widen the fragment gather to one wide load.** Store the operand slab in the fragment's own lane order — the
   crosswise layout CUTLASS uses — so `emmy_mma884_load_a/b` becomes a single 64-bit shared load instead of four
   16-bit ones. Biggest expected win by a wide margin, and it is a lowering change, not a new schedule knob.
2. **An eight-warp CTA.** Only after (1): doubling the warps to cover latency is pointless while the shared path is
   the thing that is saturated. `w2x4` / `w4x2` at a smaller per-warp tile already exist in the domain and lost, which
   is itself evidence that latency is not what is missing today.
3. **Vectorize the epilogue stores.** `VECTORIZE_STORES` is on by default, so find out why the fragment writeback is
   not folding into wide stores — the strided lane pattern probably defeats the run detector.
4. **Drop the scalar round-trip in the staged copy**, so a staged chunk moves as a `uint4` end to end.
5. **Raise the `bk` cap above 8**, last, and only if (1) lands.

## Not yet done

- The `ncu` profile was launched on the card but is not reported here. Confirm the diagnosis against the counters —
  shared-load throughput and bank conflicts on the emmy kernel versus the cuBLAS reference — before spending the
  work on lever 1. `emmy run --profile` needs root on that box (`RmProfilingAdminOnly: 1`).
- Read CUTLASS's Volta shared layout directly (`Volta884` / the `mma_sm70` tile iterators) for the exact swizzle,
  rather than the second-hand description above. Useful background:
  [efficient_gemm.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md),
  [Automatic Kernel Generation for Volta Tensor Cores](https://arxiv.org/pdf/2006.12645),
  [demystifying Volta884_h884gemm](https://blog.csdn.net/yiran103/article/details/134537689).

## Caveat carried over

A tuning round on this card searches with a prior these same goldens trained, so read a round's results directly
rather than trusting them to re-rank the recorded rows. Measurement on this box also needs a long warmup and one GPU
at a time — the SM clock ramps 1380 to 1567 MHz, which is enough to reverse a close pair.
