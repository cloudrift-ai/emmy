# Gemma 4 12B kernel goldens

One working golden per kernel of a Gemma 4 12B decoder layer at sequence length 512 — the five projections and the
sliding layers' causal attention — per card, named `KERNEL-s512_CARD.golden.yaml`. Each holds the traced FP16
program and its Loop IR target, the seed realization, and the receipts of the fastest row measured on that card by
hand pin, once in the standard lane and once under `EMMY_FAST_MATH=1`. The recipe replays them as they are, does not
tune, and fails when a golden is absent. A schedule is a claim about one card: a row recorded on one is never
replayed on another.

## How the rows were found (2026-09-22/23)

The tuner was not used. For every target and lane the compiler's own fork tree was enumerated (the rows a live
compile chooses among: ~17k for a projection in the standard lane, ~32k under fast-math, ~130k / ~410k for the
two-site attention) and swept in three passes of hand-pinned `emmy run --golden … --realization … --ab KNOBS` rows
against eager at deployable `-O3`: tile fragment x K step x worker split (144 pins, the previously recorded rows
first, wide tiles also at each cross-CTA split `g2k`/`g4k`/`g8k`), then staging ring x reduce partition around the
top three (48), then rasterization x deeper K step (24). Only rows with no integrity flag counted. Each lane's
winner was then written with `--record-greedy` from a fresh per-lane tune DB seeded by one bench of that pin, so the
recorded greedy is the sweep winner re-measured. The three 5090 lanes whose winner is a cross-CTA split (`q_proj`
std, `mlp_down` std and fm) keep their previous rows: a hand-pinned split records no row that prices the kernel-set
arm, so `--record-greedy` cannot re-derive it (`EvidenceError` on the cut fork); the sweep's candidates for them are
in `../RESULTS.md`. `../sweeps_<card>_2026-09-22.tar.gz` holds every pass's pins and A/B records.

## RTX 5090

| Golden | Matmul (M x K @ K x N) | Standard lane | Fast-math lane |
| --- | --- | --- | --- |
| `q_proj` | 512 x 3840 @ 3840 x 4096 | `f16_f32/f2x4/k2`, `w4x2`, `g4k` (kept) | `f16_f16/f2x8/k4`, `w4x2`, unsplit |
| `kv_proj` | 512 x 3840 @ 3840 x 2048 | `f16_f32/f2x2/k4`, `w2x4`, `gm8` | `f16_f16/f2x2/k4`, `w2x8`, `d2/smem-tma/p2`, `gm8` |
| `o_proj` | 512 x 4096 @ 4096 x 3840 | `f16_f32/f2x2/k4`, `w2x4`, `d2/smem-async/p2`, `gm8` | `f16_f16/f2x8/k4`, `w4x2`, unsplit |
| `mlp_gate_up` | 512 x 3840 @ 3840 x 30720 | `f16_f32/f4x4/k4`, `w4x4`, `d2/smem-tma/p2`, `gm8` | `f16_f16/f4x8/k4`, `w4x2`, `d2/smem-tma/p2`, `gm8` |
| `mlp_down` | 512 x 15360 @ 15360 x 3840 | `f16_f32/f2x4/k2`, `w4x2`, `g4k` (kept) | `f16_f16/f4x8/k4`, `w4x2`, `g2k` (kept) |

Unless a row says otherwise the ring is the two-slot TMA ring (`STAGE=d2/smem-tma`); `/p2` adds the register stage;
the tile's atom prefix is `mma_m16n8k16_`. `g<n>k` is the cross-CTA split of the contraction axis with a separate
finalize kernel, which the whole-program latency includes. Against the previous rows the fast-math winners are
2–17% faster on four of the five projections (`o_proj` 11%, `mlp_gate_up` 14%, `mlp_down`'s `g8k` candidate 5% —
see RESULTS.md); the standard lane is within noise of them.

### Attention

`attention` is `scaled_dot_product_attention` over `(1, 16, 512, 256)` FP16 inputs with `is_causal=True`, one fused
kernel with two schedule sites: the value expectation `TILE@map.1/twist` spans the 256-wide head in one warp column
(`f1x32`) over 32-key chunks (`k2`), and the score `TILE@map.1/twist.1/inner` tiles those 32 keys (`f1x4`). Four
warps per CTA (`WORK=w4x1`). The sweep found a mixed transport the hand passes had never tried: the value site on
the two-slot asynchronous-copy ring and the score site on the three-slot TMA ring.

| Lane | Value expectation | Score | Rings (value / score) |
| --- | --- | --- | --- |
| Standard | `f16_f32/f1x32/k2` | `f16_f32/f1x4/k8` | `d2/smem-async` / `d3/smem-tma` |
| Fast-math | `f16_f16/f1x32/k2` | `f16_f16/f1x4/k2` | `d2/smem-async` / `d3/smem-tma` |

The fast-math row now accumulates both the value product and the score in FP16 with the periodic promote into the
FP32 carrier; the previous row kept the score in FP32. It is the first row of this kernel at or above eager on the
5090 (1.06x; the previous rows were 0.92x).

## RTX 4090

The card has no TMA, so every row streams its operands through the asynchronous-copy ring; the register stage
(`/p2`) and a four-slot ring (`d4`) now appear where the sweep found them. The projections settle on one warp column
by four (`w1x4`) with `f4x4` fragments in the standard lane, and lose to every cross-CTA split including
`mlp_down`'s, whose previous `g2k` rows are replaced by unsplit ones.

| Golden | Standard lane | Fast-math lane |
| --- | --- | --- |
| `q_proj` | `f16_f32/f4x4/k4`, `w1x4`, `d2/smem-async/p2`, `gm8` | `f16_f16/f4x8/k4`, `w2x2`, `gm8` |
| `kv_proj` | `f16_f32/f2x4/k2`, `w2x2`, `d4/smem-async/p2`, `gm8` | `f16_f16/f4x4/k4`, `w1x4`, `d4/smem-async/p2`, `gm8` |
| `o_proj` | `f16_f32/f4x4/k4`, `w1x4`, `d2/smem-async/p2` | `f16_f16/f2x8/k4`, `w4x2`, `d2/smem-async/p2` |
| `mlp_gate_up` | `f16_f32/f2x8/k2`, `w2x2`, `gm8` | `f16_f16/f4x8/k2`, `w2x2`, `d2/smem-async/p2`, `gm8` |
| `mlp_down` | `f16_f32/f4x4/k4`, `w1x4`, `d2/smem-async/p2`, `gm8` | `f16_f16/f4x8/k4`, `w2x2`, `d2/smem-async/p2` |

Attention keeps its shape — `f1x32/k2` value expectation over four warps, the score at `f1x4/k2` in both lanes — on
the two-slot asynchronous ring for the value site and the three-slot one for the score under fast-math. Every 4090
row re-measured within 1% of, or up to 13% under, the previous row on the same box; the absolute numbers differ from
the previous recording because the box does (see RESULTS.md).
