# What a good schedule for the fused attention target would look like (RTX 4090, 2026-09-21)

**Question.** Emmy's fused q/k-norm + RoPE + scores target deploys at 19.6 ms against eager's sub-millisecond.
Every schedule the compiler enumerates for it is unusable, and the one structural escape it offers — a placement
cut — has been measured as a straight loss. Is the shape itself hopeless, or is the schedule space simply missing
the transformations this kernel needs?

**Answer.** Not hopeless. Three transformations, none of them expressible as a knob, take a hand-written kernel
from Emmy's structure to parity with Inductor. Each was measured separately, so the gap decomposes into three
independent compiler work items with a price on each.

This study deliberately does not use Emmy. It hand-writes the kernel in CUDA to find out what is reachable,
which is the one question the compiler cannot answer about itself.

## Setup

RTX 4090 (sm_89, driver 595.71.05), nvcc 12.8, `-O3`, CUDA-graph-free event timing, median of 11 runs after a
warm-up. Reference lanes: PyTorch 2.6.0+cu124 eager and `torch.compile`, measured on the same box in the same
session.

Shapes are the real ones, recovered from the committed golden
(`golden/qwen3-06b-s512_a100.golden.yaml`, target `k_sdpa_mean_reduce_29d3df`): Q `[16, 512, 128]` fp16, K
`[8, 512, 128]` fp16, grouped-query attention at 2:1, RoPE over the 64-wide halves, RMSNorm per head and
position, then the 512x512 score matrix per head and its row statistics.

Sources: `2026-09-21_fused-attention-schedule-rtx4090/qk_scores.cu` and `ref.py`.

## Result

| variant | median | what it changes |
| --- | ---: | --- |
| A recompute in loop | 10.744 ms | Emmy's structure: each row's norm and rotation re-derived per score |
| B hoisted, then score | 1.968 ms | normalize and rotate once into scratch |
| C tiled, online statistics | 0.504 ms | block through shared memory; the score matrix is never stored |
| D tiled on tensor cores | 0.193 ms | C, with `wmma` computing the score tile |
| — eager PyTorch | 0.439 ms | reference |
| — `torch.compile` | 0.191 ms | reference |

Every variant agrees with B on cancellation-free checksums to 1e-4 or better.

```
A -> B   hoist loop-invariant work        5.5x
B -> C   never materialize the scores     3.9x
C -> D   tensor cores for the score tile  2.6x
A -> D   combined                        55.8x
```

**D lands on Inductor: 0.193 ms against 0.191 ms.** A hand-written kernel with all three transformations matches
the strongest reference, from a starting point 55x worse. That is the headline: the shape supports a good
schedule, and the compiler cannot currently express one.

## Compiler work items, in order of what they buy

**1. Hoist loop-invariant work out of the score loops — 5.5x.** The generated code recomputes each RMSNorm and
each rotation inside the loops that consume them. The previous cycle's A100 analysis named this already: the
code "recomputes Q RMSNorm within output-key work, K RMSNorm within each dot product". No knob moves work
between loop levels — tile size, thread count, staging and rasterization all decide how a fixed loop nest maps
onto threads. This needs loop-invariant code motion during lowering, or a reuse decision that keeps the
normalized row live.

Necessary but nowhere near sufficient: B is still 4.5x behind eager.

**2. Block the score computation and fold statistics online — 3.9x, the bigger win.** A 512x512 fp32 score
matrix per head is 16 MB of traffic that exists only to be reduced away. C tiles over keys, keeps a tile in
shared memory, and accumulates the row statistics as it goes. This is the FlashAttention shape, and it is the
transformation the schedule space is most conspicuously missing.

Note this is **not** the same as the placement cut the compiler already offers. A cut materializes the
intermediate to a workspace buffer — it makes the traffic explicit rather than removing it. The recorded A100
cut results (0.04x, 0.06x) are what that costs. Tiling removes the traffic instead.

**3. Reach the tensor-core tier for this fused shape — 2.6x.** Emmy already has mma schedule tiers; it cannot
apply them here. Once the kernel is tiled, the score tile is an ordinary small matrix multiply and `wmma`
applies directly. This rung is worth the least of the three but it is the one that closes the remaining distance
to Inductor.

## What this does not show

- **The variants are simplified.** They compute cancellation-free checksums (sum of absolute score, sum of
  squared score) rather than a true softmax max-and-sum. The work shape is the same — a full score pass folded
  into per-row reductions — but the arithmetic is not identical to the deployed target.
- **The reference is an easier problem than Emmy's.** `torch.compile` succeeded here; on the real traced target
  the recorded run has it failing outright. The PyTorch lanes run a hand-written equivalent, not the traced
  graph, so they are a fair yardstick for this study and not a like-for-like substitute for the recipe's
  Inductor lane.
- **One shape point.** Batch 1, sequence 512, this model's head geometry. The decode shape (sequence 1) is
  memory-bound and will decompose differently.
- **Emmy's 19.6 ms is carried from a different run** at a different revision. Variant A, at 10.7 ms, is the
  honest same-session stand-in for Emmy's structure — and Emmy being roughly 2x worse than a naive hand-written
  version of its own shape is itself worth investigating separately.

## Method note

The repository forbids writing benchmark scripts, because `emmy run --bench` is the record every consumer reads.
That rule is about measuring Emmy. This study measures what Emmy cannot express, so it sits outside the CLI by
necessity. It belongs here as evidence, not in the tree as a tool.
