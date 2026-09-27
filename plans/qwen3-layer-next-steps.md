# Qwen3-0.6B layer: findings and next optimization steps (2026-09-26)

The golden-bench corpus target is Qwen3-0.6B layer 0, one fused kernel since #897, deployed through a cut route.
PR #914 made the flash-shaped route reachable and re-recorded the corpus goldens; PR #918 fixed the cross-CTA split of
a one-row matmul. This plan records where the layer stands after both and what to try next, in order.

## Where things stand

End to end, µs, unpinned replay of the #914 goldens (`EMMY_FAST_MATH=0`, `-O3`), recorded with #918's fix:

| card | s512 | eager | torch.compile | s1 | eager | torch.compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 40GB | 323 | 352 | 193 | 205 | 150 | 52 |
| H100 80GB | 231 | 200 | 80 | 88 | 110 | 32 |
| V100 SXM2 | 1153 | 1116 | 636 | — | — | — |

Five GEMV pieces split in each s1 file. Earlier A100 baselines (eager 785, `torch.compile` 429 at s512) were measured
while an orphaned bench worker held the GPU; check `nvidia-smi` for foreign processes before any A100 measurement.

## What the two PRs found

- A cut V projection stored its workspace at the flat output channel, so P·V ran scalar (839 → 95 µs).
- Seam clustering missed copies read at shifted columns, so q, k and o_proj were computed several times.
- A delegated zero-init ran on one block (130 µs on a 4 MB accumulator).
- Re-formed pieces with no contraction kept an uncoalesced grid (RoPE, 39 µs on the H100).
- Fusion dropped the f16 rounding of V and of the attention output, which kept them f32 and out of the chunk tier.
- The chunk tier read a `[seq, head, dim]` query at the wrong row stride (wrong answers).
- A split matvec's partial tiled its partition coordinate as the row, so every partition past the first read the first
  partition's weights (wrong answers at s1, on main since the split existed; #918).

## Where the time goes

Per-piece kernel time from the recorded golden rows (µs):

| card, shape | kernel sum | projections (six GEMMs) | attention piece | largest other |
| --- | ---: | ---: | ---: | --- |
| H100 s512 | 199 | 117 (26.6, 20.8, 20.6, 17.5, 16.8, 15.0) | 21.8 | 6.4 coop reduce |
| A100 s512 | 312 | 186 (42.3, 37.6, 29.5, 28.4, 24.3, 23.6) | 42.1 | 8.3 |
| V100 s512 | 1187 | 573 | 421 | — |
| H100 s1, with splits | 73 | — | — | 15.0 |
| A100 s1, with splits | 183 | — | — | 81.4 six-channel q/k GEMV |

The projections are about 16 GFLOP at s512. At a realistic 650 TFLOPS on the H100 that is about 25 µs, so the GEMM
pieces run at about a fifth of the card; on the A100 (about 300 TFLOPS) they run at about a quarter. The s1 layer reads
about 30 MB of weights, about 9 µs at the H100's bandwidth, against 73 µs of kernels.

## Next steps, in order

1. **Let the evidence pick take a measured split.** Recorded from a sweep's tune DB, the s1 route chose no split (A100
   271 µs, H100 133) although the sweep measured a 52 µs GEMV as a 21 µs partial plus a 1.4 µs finalize: the tune DB
   holds perf rows for the pieces and no routing row, so nothing prices the split arm. The #914 s1 files work around it
   by recording the prior's set from an empty tune DB, which leaves the A100's six-channel GEMV on a poor schedule
   (81 µs under the prior against 47 µs measured). Fixing the pricing gives both at once.
2. **The six-channel q/k GEMV at s1** (the largest A100 s1 piece). The q and k projections and their
   rotate-half copies lower as one GEMV with six channels that reads the q and k weights three times. Find why the
   route leaves the rotate-half copies inside the projection instead of reading the projection's workspace; a cut at
   the projection seam should make it one read of each weight.
3. **wgmma for the H100 projection pieces.** Every recorded GEMM row on the H100 is `mma.sync`. The wgmma tier exists
   (#775, #782), but its earlier result was that a wide N tile does not help the s512 linears on their own. Check
   whether it is offered at all for these cut pieces (several read a computed A through the RMSNorm prologue), then
   bench `--ab` rows against the recorded ones under `--strict`.
4. **The gap between kernel sum and end to end** (199 vs 230 µs on the H100 s512; 81 vs 86 at s1). Twenty pieces per
   layer; check whether graph replay covers the whole route and what the launch gaps are.
5. **V100 attention piece** (421 µs, 35% of the layer): the Volta chunk tier stages without `cp.async`. Profile it
   before tuning; the V100 GEMM pieces (573 µs) are the other half.
6. **Gemma 4 post1024 fast-math routes** came back 4–9% slower after the #914 re-record while the pre-attention routes
   got 13–17% faster; find which piece moved.
7. **DeepSeek V4 atomic-split rows** (`g<n>a`) that the #914 re-record wrote do not replay from the stored row; two
   were dropped. Reproduce on the V100 SXM3 and decide whether the atomic arm into an f16 output should be offered.

## How to measure

- Model level, with eager and `torch.compile` in the same process: `emmy run Qwen/Qwen3-0.6B@<rev> --layer 0
  --seq-len {1,512} --bench --strict`, with the golden in `EMMY_GOLDEN_FILE` and a fresh `EMMY_TUNE_DB`.
- A golden walk (`--golden F --realization R --pin-route --bench`) times a route's recorded kernel set but checks no
  correctness.
- After any Loop IR change every piece row goes stale; expect a per-piece re-record on each card.
