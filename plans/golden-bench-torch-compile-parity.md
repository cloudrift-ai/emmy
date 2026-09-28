# Golden-bench kernels: beat torch.compile on five cards (2026-09-28)

The corpus is `experiments/golden-bench-2026/kernels`: Qwen3-0.6B layer 0 at sequence lengths 1 and 512, one fused
layer deployed through a cut route and timed end to end, on V100, A100, H100, RTX 4090 and RTX 5090. Parity work landed
in #930; every cell is now at, level with, or (V100 s1) accepted below `torch.compile`. What remains is margin: a
kernel-set result good enough for the paper needs the level cells clearly above 1×.

## Overview

| # | Work | Cells it moves | Expected | GPUs needed | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | #930 close-out: drop the `PIECE_FORMATION` ContextVar (one route re-spell), keep a piece's ordinal through a split, restamp | all | none (hygiene) | local RTX 5090 | done (in #930) |
| 1 | mma.sync GEMM: conflict-free fragment loads, fewer loads per mma, a 96-row masked M tile | A100 s512, 4090 s512, 5090 s512 | A100 0.98× → ~1.05×; 4090/5090 margin | A100 (primary), RTX 4090, RTX 5090 | open |
| 2 | sm_90 GEMM: larger wgmma tiles, TMA staging, a producer warp group | H100 s512 | 0.98× → ~1.15–1.2× | H100 | open |
| 3 | Programmatic dependent launch between graph kernels | H100 s1, 5090 s1 (+ s512 tails) | ~0.5–1 µs per launch, 16 launches | H100, RTX 5090 | open |
| 4 | FP8 rows of the recipe (the paper figure's Q/K/V-FP8 bars) | FP8 study | measure | RTX 4090, RTX 5090, H100 | open, only if the figure keeps them |
| 6 | Re-record goldens the close-out left without rows (below) | Gemma 4 serving, fp8-block corpus | restore evidence | RTX 5090 | open |
| 5 | Final re-record of every golden, unpinned `--strict` replay, lane once per card, RESULTS.md | all | the reported numbers | all five cards + V100 SXM3 (AWQ golden) | after 1–3 |

Order: 1 and 2 in parallel (A100 and H100 are separate hosts), then 3, then 5. Item 6 any time on the 5090.

Out of scope: persistent kernels (a persistent GEMV chain was the one lever left for V100 s1 and A100 s1; excluded for
the paper). V100 s1 stays accepted at 0.85× and A100 s1 level.

## Scoreboard (µs, end to end, #930 head)

Unpinned replay of the committed golden, fresh tune DB, `-O3`, `EMMY_FAST_MATH=0`, eager and `torch.compile` in the
same process, Emmy timed on the reference inputs. Ratio is `torch.compile` / Emmy (above 1: Emmy faster).

| card | s1 Emmy | s1 t.c | s1 ratio | s512 Emmy | s512 t.c | s512 ratio | s512 `--strict` misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 5090 | 24.5 | 33 | 1.35× | 132 | 135 | 1.02× | 6 / 524,288 |
| RTX 4090 | 26.5 | 28.9 | 1.09× | 158.7 | 162.0 | 1.02× | 5 |
| H100 | 36.7 | 40 | 1.09× | 95.8 | 94 | 0.98× | 0 |
| A100 40GB | 55–56 | 54–56 | ~1.00× | 182–184 | 180–181 | 0.98× | 2–7 |
| V100 SXM2 | 72 | 61 | 0.85× (accepted) | 520 | 644 | 1.24× | 8 |

Misses are one f16 step (max abs 0.0039), the same spread as between torch's own attention backends; an approximate
match is enough (user decision). Starting point (2026-09-27): H100 s512 141.5 / s1 88, A100 323 / 198, V100 s512 855;
no 4090, 5090 or V100 s1 goldens.

Also recorded in #930: Qwen3.8-27B V100 layers — AWQ 474 → 11.4 ms (SXM3), GPTQ 604 → 9.7 ms (SXM2).

## What bounds each cell now

- **s512 is GEMM-bound.** Attention is ahead or level on every card (4090 23.4 vs 28.4, 5090 19.4, A100 37.4 vs ~38,
  H100 12.7 vs cuDNN 10.5). Emmy's fused norm/RoPE/residual pieces beat torch.compile's small kernels (H100 ~13 vs
  ~20 µs). The GEMMs trail the vendor kernels: ~4 µs per GEMM against cuBLAS on the A100, and on the H100 the wgmma
  GEMMs run at ~270–330 TFLOP/s against ~600–870 for nvjet/Triton (~20 µs of a 96 µs layer).
- **s1 is launch-bound.** Every route has 16 launches after the prologue/epilogue routes (below). A100: the pieces sum
  to 40 of 55 µs. V100: each launch is ≥1.8 µs inside a CUDA graph, ~29 of 72 µs; torch.compile runs 9 kernels.

## 1. mma.sync GEMM efficiency (A100, RTX 4090, RTX 5090)

A100 q projection (512×1024×2048) under ncu, Emmy `w2x2 f2x4/k4 d3/p2` vs cuBLAS: 27.3 vs 21.2 µs; CTA tile 64×64 vs
96×128; 256 vs 96 CTAs; 106 vs 206 registers; SM busy 33% vs 47.5%; load/store instructions 499k vs 319k;
shared-memory load bank conflicts 6.3k vs 0. Wider warp tiles and split-K lost in a sweep (the largest tiles hit
~238 registers and 12% occupancy). The gap is inside the kernel, not in the schedule space:

1. Make the `f2x4` drain's fragment loads bank-conflict free (the swizzle/`ldmatrix` addressing for that fragment
   shape), and cut load instructions per mma. Prove on the A100 q/o/down against `torch.mm`, then on 4090 and 5090.
2. Offer a 96-row M tile with a masked M edge (512 is not a multiple of 96), as cuBLAS uses.
3. Re-sweep and re-record s512 on all three cards.

Bar: each GEMM within 1.05× of `torch.mm` at its shape. Profiles and sweeps from this round: A100 host
`~/gb-a100-w/qprof`.

## 2. sm_90 GEMM efficiency (H100)

The recorded H100 s512 uses wgmma `m64n64 f1x8/k4 d4 gm8` on most GEMMs, `m64n128` on k, and two warp groups
(`w8x1 m64n128 f1x16/k4 d4`) on gate/up (19.6 µs). `m64n256` does not compile ("wgmma needs a w<4k>x1 warp grid";
`w8x1 m64n256` is refused by its pins). TMA was slower than cp.async on the pieces tried, and a producer warp group
needs TMA. Work:

1. Profile one GEMM (q or down) against nvjet at the same shape: tile, stages, waves, issue.
2. Make `m64n256` and 2-warp-group tiles legal where they fit, and TMA + producer warp group win (the FA-3/CUTLASS
   pattern), then sweep.
3. Re-record H100 s512.

Bar: each GEMM within 1.1× of nvjet. Attention (12.7 vs 10.5) is secondary: the two-warp-group key split and the
Q-prologue overlap were measured and did not help (removed).

## 3. Programmatic dependent launch (H100, RTX 5090)

On sm_90+, let each graph kernel start its prologue while the previous one drains (`griddepcontrol` /
`cudaLaunchAttributeProgrammaticStreamSerialization`). A runtime and codegen change; no schedule changes. Measure the
per-launch gap on an empty route first, then s1 on H100 and 5090. Not available on A100 or V100.

Tried and lost for s1 launch count on other paths (do not repeat): unsplit GEMVs (too few CTAs), fewer splits, the
in-kernel last-CTA finalize (82 vs 71 µs on V100 s512 down), the consumer-summed split (`fix/gb-split-sum-in-consumer`,
parked: 1.5 µs pinned, and it rewrites a piece of another decision, which breaks kernel-set pricing).

## 4. FP8 rows (RTX 4090, RTX 5090, H100) — only if the figure keeps them

The paper figure still shows dynamic-FP8 Q/K/V bars from the retired per-kernel corpus. The recipe's FP8 study rows
are unrecorded on #930. Trace, sweep, record, and measure on the FP8-capable cards.

## 5. Close-out

1. Done (#930): `PIECE_FORMATION` is gone (the fold alone orients a slab-and-computed pair), a split keeps its piece's
   name and ordinal (`<piece>__partial`, `<piece>`), a `place_<token>` pin matches whole name segments only, and every
   golden is restamped: 71 routes re-spelled (ten Qwen3-0.6B golden-bench files, 4 DeepSeek-V4 V100 rows, 55 Gemma 4
   RTX 5090 rows), every key mapped, no stored target changed, no row lost its µs. Pin files written before the
   re-spell stop matching (piece tokens changed); the 5090's re-mapped ones are `/tmp/gb-rtx5090/s1_v8_pins.final.txt`
   and `pin512_v7.final.txt`.
2. Item 6, rows to re-record on the RTX 5090:
   - `recipes/gemma-4-12B-it/golden/rtx5090_sm120.json`: the `pre1-global` m1 route's 7 pieces, plain and fast-math,
     28 rows dropped (they lower to other kernel identities now); the routing rows stay, re-spelled.
   - `qwen3-06b-fp8-block-s1_rtx5090` and `-s512` (golden-bench quantized kernels): stale before the close-out — no
     fresh kernel writes their targets' outputs, so a restamp keeps nothing. Re-trace and re-record.
   - Not demoted, same math: `qwen3-06b-s512_v100` renders 2 of 22 kernels (RoPE, SiLU·up) with statements reordered.
3. After 1–3 land, re-record every golden on the final compiler from sweeps (fresh tune DB per run), replay each cell
   unpinned under `--strict`, run the lane (`emmy bench` on the recipe) once per card, write the RESULTS.md section.
4. #930 finalization per AGENTS.md: full `make test`, lint, docs, PR body.

## Hosts

| card | host | notes |
| --- | --- | --- |
| RTX 5090 | dev box (local) | shared with the desktop; one GPU job at a time |
| RTX 4090 | `ssh -p 60011 riftuser@118.163.199.138` | `~/gb-rtx4090*` |
| V100 SXM2 16GB | `ssh riftuser@185.165.50.75` | `~/gb-v100`, work files `~/gbw` |
| V100 SXM3 32GB | `ssh riftuser@66.172.10.144` | AWQ golden's card; `~/gb-gptq`, `~/gbw` |
| A100 40GB | GCP `bench-keep-a100-0921-1621-6784`, us-central1-f | `~/gb-a100`, `~/gb-a100-w`; kept running |
| H100 80GB | GCP `bench-gb-h100-0924-1252-99aa`, us-east4-a | `~/gb-h2`, `~/gbh`; kept running |

GCP needs a live `gcloud auth login` in the session shell (check `gcloud auth print-access-token`).

## Measurement and recording rules

- Unpinned replay of the committed golden, fresh `EMMY_TUNE_DB` per run, eager and `torch.compile` in the same
  process; `nvidia-smi` for foreign processes first; long warmup on the V100.
- Find schedules by sweeping (`--ab`, kernel-scoped `FAMILY@place_<token>` / `@node_<id>` pins, the most specific pin
  wins); do not rely on the prior. Sweeps below the bench standard (warmup ≥ 5, iters ≥ 20) now fail, and
  `--record-greedy` refuses a kernel the prior would decide.
- Never seed a working golden with another card's timings; record under the card the file names (recording into
  another card's golden now refuses).
- After any Loop IR change, piece rows go stale; a slower row after a compiler change is a finding, never re-recorded
  green.
- s1 routes that won (4090/5090/A100/H100): 8 cuts; q/k RoPE uncut (q RoPE rides the q split's finalize, k RoPE the
  attention prologue); down inside the residual root (`REDUCE@node_add_7=g8k` + `…__partial=coop-t/v2`); gate/up one
  GEMV with SiLU·up in its finalize; other GEMVs `g16k` with `t128`/`t256 coop-t/v2` partials. Exact pins are in the
  s1 goldens.
