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

## V100 golden rows (CloudRift V100 SXM2 16GB, CUDA 12.9, torch 2.13.0+cu126, `-O3`)

The first version of the fix lowered every re-formed piece a second time. That permuted the grid axes of V100 cut
pieces whose contraction orientation it did not change, and it moved their CUDA: four Qwen3.8 golden rows stopped
decoding, and several DeepSeek, AWQ and GPTQ rows kept decoding while measuring kernels the compiler no longer emits.
The final fix keeps the first form of a piece unless the second pass moves some contraction's A, and with it every
V100 golden row decodes again with its stored kernel. The measurements below come from the intermediate version and
explain why it was narrowed; they are not rows of the final golden files.

Same card, same pins, three runs, per-kernel isolated time, median (min–max), µs:

| kernel (golden row) | stored | base | intermediate PR |
| --- | ---: | ---: | ---: |
| `k_matmul_reduce_50f206` FP8, `t64x16/f2x6` (`…50f206.5f16496e27a8`) | 139.6 | 143.7 (143.5–145.2) | 146.8 (145.7–147.5) |
| same kernel in the GPTQ golden | 140.0 | 144.7 (142.8–145.2) | 146.3 (145.4–147.5) |
| `…de59ac__place_b37170b988` GPTQ, `f2x8`/`t32x8` (`…4f7d9a419000`) | 269.8 | 259.1 (231.8–263.4) | 350.2 (349.9–351.2) |
| `…de59ac__place_509079086b` (`…ce29f503f8fe`) | 891.9 | 1024.0 (1022.0–1025.0) | 1023.0 (1023.0–1026.0) |
| `…de59ac__place_2d9a0db1be` (`…7557f72462b3`) | 816.1 | 773.1 (772.1–774.1) | 774.1 (772.1–774.1) |
| `k_linear_matmul_mean_reduce_de59ac` (`…f08b14bf4708`) | 2256.9 | 2138.1 (2137.1–2138.1) | 2140.2 (2139.1–2140.2) |

The `b37170b988` piece lost 35%: the same per-thread tiling ran with the two outer grid axes in the other block
order. That regression is what the narrowing removes. The card also ran some unchanged kernels far from their stored
times (an unchanged `50f206` cut piece at 43.6 ms against 34.4 ms stored), so only base against PR on one card is a
comparison.

A small hand-tuning pass on the `50f206` consumer (FAST_MATH, two runs each, whole cut set under `--pin-route` with the
consumer's `WORK`/`TILE` varied) found a faster row than the stored one on this card: `t32x16/f2x4` 103–105 µs,
`t64x16/f2x2` 106–107, `t32x16/f2x2` 109–110, `t16x16/f2x4` 111–113, `t32x16/f1x4` 116–118, `t64x8/f2x4` 117–119,
`t64x16/f1x4` 117–118, `t32x8/f2x4` 117–119, `t64x16/f2x4` 125–127, stored `t64x16/f2x6` 145–146, `t32x16/f2x6`
144, `t64x8/f2x6` 144–145, `t32x16/f4x6` 158–160, `t64x16/f1x6` 174, `t32x8/f4x10` 185–187, `t64x16/f2x8` 191–192,
`t64x16/f4x6` 256. `t128x8/f2x6`, `t64x16/f2x5`, `t128x8/f2x4`, `t64x16/f2x3`, `t32x32/f2x4` and `t16x32/f2x4` are
not offered. The kernel does not change under the final fix, so this is a tuning opportunity for the golden, not part
of this PR. V100 has no cp.async or TMA, so the byte-slab staging this PR exposes does not apply there.
