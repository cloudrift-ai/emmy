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
