# B-packed-cut-staging: quick hand-pinned measurements

Hand pins only; no greedy pick, prior or autotune row is used as a baseline. Desktop RTX 5090 (sm_120, driver 580.95,
CUDA 13.0, power limit 600 W, idle 29 °C before the runs). Base is `944114c7`; the PR is the fix branch at the commit
that adds this file. Both revisions ran from one venv, each with its own source tree on `PYTHONPATH`.

## Program and method

The report's program: `a = x + 1`, then `b(a)` (4096 → 1024) and `c(a)` (4096 → 512), W4A16 NVFP4, with
`M = 16` (the report's decode batch) and `M = 512` (prefill). `M = 1` is not affected: there the size-one token axis
is dropped, and the pieces already oriented the activation as A on the base.

```sh
EMMY_KNOBS="<baseline row>" emmy run --quantize nvfp4-w4a16 --bench --warmup 50 --iters 500 \
  --no-record-evidence --seed 0 --json OUT.json --ab "<row 1>" --ab "<row 2>" ... -c "$PROG"
```

`--quantize` writes one checkpoint per run from `--seed 0`, so every row in a run reads the same packed weights.
Every compared row is an `--ab` row, so all rows share one lane (`fm`) and one process. Each row ran three times,
in separate processes; the tables give the median of the three whole-program times (both pieces, back to back) and
the min–max spread. Rows spell `PLACE@map.1/inner=cut,REDUCE=` plus `TILE=mma_m16n8k16_f16_f32/<tile>` and
`WORK`. Correctness is covered by the corpus cases and the Python oracle tests, not by these runs.

## Results (µs, both pieces together)

`M = 16`:

| schedule | base | PR |
| --- | ---: | ---: |
| `f1x2/k4, w1x2, d1/smem` (compute fill) | 195.7 (194.9–196.8) | 144.2 (144.1–144.2) |
| `f1x2/k4, w1x2, d2/smem-async` | not offered | 92.2 (92.0–92.4) |
| `f1x2/k4, w1x2, d2/smem-async/p2` | not offered | 92.2 (92.1–92.3) |
| `f1x2/k4, w1x2, d3/smem-async` | not offered | 96.3 (95.8–96.4) |
| `f1x2/k4, w1x2, d4/smem-async` | not offered | 92.3 (92.3–93.2) |
| `f1x2/k4, w1x2, d2/smem-tma` | not offered | 98.4 (96.4–98.4) |
| `f1x2/k4, w1x2, d4/smem-tma` | not offered | 98.4 (98.4–98.4) |
| `f1x2/k4, w1x2, d4/smem-tma/p2` | not offered | 100.4 (100.4–100.4) |
| `f1x1/k4, w1x1, d1/smem` | 207.2 (207.1–207.2) | 139.4 (139.2–139.5) |
| `f1x1/k4, w1x1, d2/smem-async` | not offered | 116.9 (116.9–116.9) |
| `f1x1/k4, w1x1, d4/smem-async` | not offered | 116.9 (116.8–117.0) |
| `f1x1/k4, w1x1, d4/smem-async/p2` | not offered | 116.8 (116.8–118.1) |
| `f1x1/k4, w1x1, d2/smem-tma` | not offered | 121.6 (120.9–122.8) |
| `f1x1/k4, w1x1, d4/smem-tma` | not offered | 121.0 (121.0–121.0) |
| split-K `REDUCE=g4k, f1x2/k4, w1x2, d4/smem-tma` (4 kernels) | 28.7 (28.7–28.8) | 28.8 (28.7–29.0) |
| fused, `PLACE@map.1/inner=fuse,REDUCE=` (1 kernel, scalar) | too slow for the bench budget | ≈ 140,000 (one sample) |

`M = 512`:

| schedule | base | PR |
| --- | ---: | ---: |
| `f2x4/k4, w2x2, d1/smem` | 343.2 (342.6–344.5) | 296.7 (295.5–300.9) |
| `f2x4/k4, w2x2, d2/smem-async` | not offered | 188.6 (188.6–188.6) |
| `f2x4/k4, w2x2, d2/smem-async/p2` | not offered | 188.6 (188.6–188.6) |
| `f2x4/k4, w2x2, d3/smem-async` | not offered | 184.2 (183.0–184.4) |
| `f2x4/k4, w2x2, d4/smem-async` | not offered | 186.1 (185.9–186.5) |
| `f2x4/k4, w2x2, d2/smem-tma` | not offered | 191.1 (190.7–191.1) |
| `f2x4/k4, w2x2, d4/smem-tma` | not offered | 181.9 (180.5–182.1) |
| `f2x4/k4, w2x2, d4/smem-tma/p2` | not offered | 182.4 (182.4–182.5) |
| `f2x2/k4, w1x1, d1/smem` | 509.8 (509.4–512.0) | 243.6 (243.5–244.7) |
| `f2x2/k4, w1x1, d3/smem-async` | not offered | 166.5 (166.2–166.6) |
| `f2x2/k4, w1x1, d4/smem-tma` | not offered | 174.6 (174.3–175.0) |
| split-K `REDUCE=g4k, f2x4/k4, w2x2, d3/smem-async` (4 kernels) | 67.8 (67.8–67.8) | 67.7 (67.7–67.8) |
| fused (1 kernel, scalar) | not measured | ≈ 575,000 (one sample) |

The split-K rows include the two small finishing kernels (about 0.8 µs each at `M = 16`). The fused kernel folds
both projections per token with a one-block grid at these shapes; the bench worker's 60 s limit stopped its
500-iteration run, so its number is one short run with 1 warmup and 3 iterations.

## Reading

**Existing schedules.** The split-K rows are the same kernels on both revisions and time the same. The no-split
compute-fill rows are not the same schedule on both revisions: the base oriented the weight decode as A, so a TILE
spelling put its fragment rows on weight rows and its columns on tokens, and the PR puts them on tokens and weight
rows. The PR's `d1/smem` rows are 14–52% faster than the same spelling on the base, but that compares two different
tilings of the same arithmetic, not a speedup of one schedule.

**Available options.** Within the no-split cut, the new packed-byte staging beats the compute fill it replaces:
`M = 16` 139.4 → 92.2 µs (−34%), `M = 512` 243.6 → 166.5 µs (−32%), comparing the best PR compute fill with the best
PR copy transport. Against the base's best no-split row (195.7 and 343.2 µs), the drop is 53% and 51%. cp.async edges
out TMA at `M = 16` by about 7%; at `M = 512` the two are within 5% of each other in either direction depending on the
tile. `/p2` changes nothing measurable here.

**Overall.** The best option tried on either revision is the split-K cut, which the PR leaves unchanged: 28.7 µs at
`M = 16` and 67.7 µs at `M = 512`, well ahead of every no-split row. So the measured available performance of this
program under the tried pins does not change. The fix makes the no-split pieces' packed staging reachable and cuts
their cost by about a third against the compute fill, but the four-way split is still 2.5–3.2× faster with these
TILE/WORK choices. Why the no-split pieces run this slowly was not investigated. These are hot-cache timings of one
isolated program; they say nothing about serving throughput.

The 5090 tables above were measured before the branch merged main's fp4 and computed-f16 TMA changes (#971, #968).
The same pins render the same kernels after the merge, except that the TMA kernels declare their barrier array at a
different line.

## V100 goldens

A re-formed piece whose single-product contraction reads a computed operand is lowered a second time inside its grid
loops, so its grid follows the output layout. On V100 that reorders the grid of several cut pieces whose A does not
move. Their CUDA changes only in block order, but a stored scalar tiling then runs in another block order, and the
`k_matmul_reduce_50f206` consumer piece gets a new identity, so its rows stop decoding. A first attempt lowered every
piece this way and broke the gate/up twin (two sibling reductions no longer merged); lowering the already formed terms
keeps the twin. A later attempt that kept the first form unless the A moved was dropped by decision: stored schedules
are regenerable, and one rule for all such pieces is simpler.

All V100 work used CloudRift VMs, CUDA 12.9, torch 2.13.0+cu126, `EMMY_NVCC_FLAGS=` (nvcc `-O3`), and each row's own
`FAST_MATH` pin (on for FP8 and GPTQ, off for AWQ, EXL3 and DeepSeek). Times are µs; `(a–b)` is min–max of the runs.

### Stored V100 timings are stale

Before recording, unchanged kernels were benched on the base tree with their stored pins. Several cards agree with
each other within about 1% and disagree with the stored numbers in both directions, so the stored numbers come from
an older compiler or setup, not from a faster card.

V100 SXM2 16GB, FP8 golden, 3 runs, median (ratio to stored):

| kernel | stored | node fdc059a8 | node f89a0546 | node f5def1ea |
| --- | ---: | ---: | ---: | ---: |
| `k_matmul_reduce_50f206__place_3fb5a39ae9` | 34394 | 43721 (1.27) | 44013 (1.28) | 43600 (1.27) |
| `k_matmul_reduce_50f206__place_74220ccf29` | 13558 | 13548 (1.00) | 13557 (1.00) | 13566 (1.00) |
| `k_matmul_reduce_50f206` | 139.6 | 144.6 (1.04) | 142.8 (1.02) | 143.2 (1.03) |
| `k_matmul_c42469` | 2596 | 2251 (0.87) | 2244 (0.86) | 2243 (0.86) |
| `k_matmul_90aa44` | 328.0 | 282.3 (0.86) | 281.3 (0.86) | 281.6 (0.86) |
| `k_transpose_unsqueeze_reduce_1e10fd` | 166.4 | 161.1 (0.97) | 157.7 (0.95) | 160.4 (0.96) |

V100 SXM3 32GB, both GPUs on node f34feb82, AWQ and DeepSeek goldens:

| kernel | stored | GPU 0323318098458 | GPU 0323318098720 |
| --- | ---: | ---: | ---: |
| AWQ `k_matmul_reduce_50f206__place_3fb5a39ae9` | 16474 | 16527 (1.00) | 16527 (1.00) |
| AWQ `k_matmul_reduce_50f206__place_74220ccf29` | 19201 | 19267 (1.00) | 19226 (1.00) |
| AWQ `k_matmul_reduce_50f206` | 178.4 | 185.8 (1.04) | 186.0 (1.04) |
| AWQ `k_matmul_c42469` | 2238.5 | 2185 (0.98) | 2169 (0.97) |
| AWQ `k_matmul_reduce_ea7a5e` | 6262.8 | 6852 (1.09) | 6853 (1.09) |
| DeepSeek `post16.k_div_1_steps0_reduce` | 51.3 | 33.9 (0.66) | 33.4 (0.65) |
| DeepSeek `post4096.k_div_1_steps0_reduce` | 532.5 | 478.2 (0.90) | 478.2 (0.90) |

Every card ran at P0 with no power cap, thermal or hardware slowdown active; SM clocks 1312 MHz (SXM2) and 1380 MHz
(SXM3) under load.

### What changed in the goldens

- **FP8 and GPTQ (V100 SXM2):** every measured row was re-recorded on one card (node f89a0546, GPU serial
  0321418100627) with `emmy run --golden WORKING --realization ROW --bench --record-greedy`, cut sets under
  `--pin-route`, with the rows promoted into the repository files. One GPTQ routing row,
  `k_linear_matmul_mean_reduce_c9255b.9b55c4dff9a9.264634d3a9c6` at 10771 µs, keeps its old number: the second
  routing row of that name routes a kernel set whose pieces have no measured rows, so a strict record refuses it, on
  main as well.
- **AWQ and EXL3 (V100 SXM3):** only the rows this change breaks or slows are replaced, measured on one card (GPU
  0323318098458): the `50f206` consumer row in both files, and the AWQ `3d9a86` piece `07d3e3b957`. Every other row
  keeps main's measurement.
- **After merging main's #969** (recurrence state carried in the term), which restamped all five V100 recipe goldens:
  main's re-keyed recurrence rows are kept, and in FP8 `emmy golden restamp` re-keyed the re-recorded
  `k_slice_unsqueeze_reduce_99e5cf` and `k_slice_unsqueeze_reduce_b17b4d` rows onto main's new identities with their
  measurements kept. Every repository golden then passes `emmy golden check` and its row decode tests.
- **DeepSeek (V100 SXM3):** unchanged. Every row still decodes. The PR changes the CUDA of the kernels
  `k_linear_matmul_softmax_mean_reduce_bcf52a__place_3409a23aa3`, `k_linear_reduce_45bd47` and `k_linear_reduce_b4cf4a`
  (grid order only), so the rows measuring those keep numbers taken on the old block order.

### Regressed and tuned rows

Same card, same pins, whole cut set under `--pin-route --warmup 2 --iters 5`, per-kernel time, two or three runs.

GPTQ `k_linear_matmul_mean_reduce_de59ac__place_b37170b988` (row `…c9255b.9b55c4dff9a9.4f7d9a419000`, SXM2, node
f89a0546). Base with the stored `TILE@map.2/inner=f2x8,WORK=t32x8`: 259 (244–263). PR with that spelling: 373
(372–374). Tried on the PR: `RASTER=gm8` 364–373, `WORK=t32x16` 338–349, `t64x8` 301–324, `t16x16` 240–257, `t16x8`
230–249, `f2x8,t16x16,gm8` 261–282, `f2x4` 224–226, `f2x2` 219–257, `f1x2` 207–209, `f1x6` 216–218, `f1x8` 182–195,
`f1x8,t16x8` 183, `f1x8,t64x8` 210–221, `f1x8,t16x16` 235–249, `f1x10` 185–203, `f1x12` 210–229, `f1x4` 154–169.
Recorded: `f1x4`, 168.8 µs isolated.

AWQ `k_linear_matmul_mean_reduce_3d9a86__place_07d3e3b957` (row `…3d9a86.4f7d9a419000`, SXM3). Base with the stored
`f2x8,t32x8`: 248 (234–253). PR with that spelling: 278.5 (278–302). Tried on the PR: `f1x6` 290–293, `t16x8` 232–234,
`f1x2` 223–225, `f2x4` 211–225, `f1x8` 192–193, `f1x4` 171–178, `f1x4,t16x8` 151–161. Recorded: `f1x4,t16x8`,
156.2 µs isolated.

`k_matmul_reduce_50f206` consumer (new identity `705642f398bb`, rows `…50f206.705642f398bb`). The old spelling still
decodes on the new identity and renders the same per-thread code with the two outer grid axes swapped.

- FAST_MATH on (FP8, GPTQ, SXM2): base `t64x16/f2x6` 142.8 (on node f89a0546) and the PR with that spelling 145.0.
  Tried on the PR (node fdc059a8, two runs each): `t32x16/f2x4` 103–105, `t64x16/f2x2` 106–107, `t32x16/f2x2`
  109–110, `t16x16/f2x4` 111–113, `t32x16/f1x4` 116–118, `t64x8/f2x4` 117–119, `t64x16/f1x4` 117–118, `t32x8/f2x4`
  117–119, `t64x16/f2x4` 125–127, `t32x16/f2x6` 144, `t64x8/f2x6` 144–145, `t32x16/f4x6` 158–160, `t64x16/f1x6` 174,
  `t32x8/f4x10` 185–187, `t64x16/f2x8` 191–192, `t64x16/f4x6` 256. Recorded: `t32x16/f2x4`, 102.9 (FP8) and
  104.0 (GPTQ).
- FAST_MATH off (AWQ, EXL3, SXM3): base `t32x8/f4x10` 185.8 (182.5–186.8), PR with that spelling 187.0
  (184.9–188.0). Tried: `t32x16/f2x4` 102–103, `t64x16/f2x2` 106, `t16x16/f2x4` 108–111, `t32x16/f2x2` 112–113,
  `t32x8/f2x4` 98. Recorded: `t32x8/f2x4`, 97.7 (AWQ) and 98.8 (EXL3).

The tuned consumer rows are about 28% (FAST_MATH on) and 47% (off) faster than the stored schedule on the same card;
the kernel is the same computation as before, so this is a side finding of the re-record, not an effect of the fix.

Other kernels whose CUDA the PR changes, same card, base against PR: GPTQ `de59ac__place_509079086b` 1024 against
1023, `__place_2d9a0db1be` 773 against 774, `k_linear_matmul_mean_reduce_de59ac` 2138 against 2140; AWQ
`3d9a86__place_ab478ce96e` 1271 against 1269, `__place_c316e5a5a3` 988 against 985, `k_linear_matmul_mean_reduce_3d9a86`
2786 against 2789; DeepSeek `bcf52a__place_3409a23aa3` 3.8 against 2.7. V100 has no cp.async or TMA, so the byte-slab
staging this PR exposes does not apply there.

## Bugs found on the way (not fixed here)

- **nvcc rejects DeepSeek V100 kernels on main.** On main (`81af0892`), `k_linear_reduce_45bd47` in
  `expert-sym@mxfp4.k_linear_reduce_ee218c.d7a2bc50f91c.m4096.65ee74075763` and `k_linear_reduce_b4cf4a` in
  `expert4096@mxfp4.k_linear_reduce_9a6fd8.998892e7cf7a.m4096.07c0c577b481` fail to compile with 12 errors such as
  `error: "in9__u0" has already been declared in the current scope` (also `in1__u0`, `in8__u0`, `in10__u0`, `in11__u0`,
  `in12__u0`). The PR's rendering of those two sets compiles. Repro on a V100 with CUDA 12.9:
  `EMMY_FAST_MATH=0 emmy run --golden recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.json --realization
  expert-sym@mxfp4.k_linear_reduce_ee218c.d7a2bc50f91c.m4096.65ee74075763 --pin-route --bench --bench-backends emmy
  --warmup 2 --iters 5`.
- **The dynamic bd75fb DeepSeek set fails nvcc on main and on the PR.** Under `--pin-route`,
  `k_linear_reduce_45bd47__place_d7b367e635` in
  `expert-sym@mxfp4.k_linear_reduce_bd75fb.c7eb618908f4.dynamic.aeb34315b0ec` fails with 100 errors such as
  `error: "v26__c0__u0" has already been declared in the current scope`; the unpinned greedy set of the same program
  builds and runs.
- **Some V100 routes cannot be recorded strictly, on main as well.** `--record-greedy` refuses the DeepSeek routes
  `expert-sym…bd75fb…m16`, `expert-sym…ee218c…m4096`, `expert-sym…bd75fb…dynamic`, `expert-sym…ee218c…m1`,
  `expert1…348ff8…m1`, `pre-sym…3e0f82…dynamic` (three routing rows) and `pre4096…00af35…m4096` (three routing rows):
  a piece of each has a cut fork no measured row decides (`no measured row spells a kernel-set arm`). Pinning
  `PLACE=fuse` does not satisfy strict evidence. The GPTQ `c9255b` kernel-set row fails the same way through its
  pinned-row comparison, while its greedy pick records. The AWQ and EXL3 `k_copy_65_base_steps0_reduce` row (about
  200 ms) exceeds the bench worker's 100 s wall budget; `EMMY_BENCH_WALL_TIMEOUT_S=900` let the EXL3 run finish.
- **A CloudRift IP answered for two machines.** A V100 SXM3 VM (node f34feb82, instance
  `7b2ca3d4-bc30-11f1-9bf7-53fdd5ad31a4`, IP 66.172.10.91) returned two different ED25519 host keys
  (`SHA256:qF1jsEnVpzm40fni98Y6fpayDqdCKhVp0FUMAOsNw/Y` and `SHA256:d96sNtVugX0RgBDvcDzbxJMZaguSXQd7BYrUyzYiG2s`) to
  repeated `ssh-keyscan`; only one of them held the files copied to the VM. The VM was terminated unused.
