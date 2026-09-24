# The golden-bench-2026 fused attention targets: what was fixed, what is next (2026-09-23)

## Where things stand

Qwen3-0.6B layer 0, prefill, sequence length 512, from the `golden-bench-2026/kernels` corpus. Deployable `-O3`,
`--fmad=false`, warmup 10 / iters 100, strict against eager at `rtol=atol=1e-3`, eager and torch.compile timed in the
same process. "Before" is the committed golden replayed on that card at the start of this work, not the September
report — two of these targets did not run at all then.

| card | target | before | now | torch.compile |
| --- | --- | ---: | ---: | ---: |
| A100 | SDPA + o_proj + residual | 245,487 | **86** | 63 |
| A100 | q/k norm + RoPE + score statistics | 393,518 | **840** | 61 |
| A100 | softmax x V | bench_fail | not recorded | 51 |
| H100 | SDPA + o_proj + residual | 1.52 s/iter, bench_fail | **53** | 21 |
| H100 | q/k norm + RoPE + score statistics | 71,898 | **216** | 29 |
| H100 | softmax x V | 144.6 | not recorded | 18 |
| V100 | fused attention block (its own partition) | hangs | hangs | not measured |

The SDPA target is now 1.37x behind Inductor on the A100 and 2.5x on the H100. Everything else still loses badly.

## The mechanism, and why it is worth knowing

Each of these kernels fuses a reducing cone inside another contraction's sweep, so the inner reduction is
re-evaluated once per point of the outer iteration space it does not depend on. The seam to cut is the one whose cone
MISSES an output axis; that analysis is mechanical and predicted the right seam on every target. Cutting is necessary
and NOT sufficient, and the reason the cut alone did nothing is the part worth remembering:

A projection downstream reshapes attention's output to one flat width, so the piece the cut mints has a grid
coordinate that is really two — every operand reads it only as `i / c` beside `i % c`. While it stays fused the
contraction's A operand is indexed by `i / c`, so it DEPENDS on the axis the B operand contracts into, which is not a
matmul; the warp tier never offers a tile and the piece falls to a per-cell reduce walking the whole 512-step softmax
once per `c`. No pin reaches it either, because a global `WORK`/`TILE` lands on the siblings. Splitting the coordinate
back restores the batched matmul the tier already schedules elsewhere: that piece went 609.5 -> 16.5 us on the H100.

## What to try next, in the order the evidence supports

1. **The score-statistics target is tunable and untuned.** Unlike the SDPA target it was never unschedulable: its
   dominant piece carries a real schedule (195 us of 217 on the H100, 768 of 840 on the A100). It is 7.5x and 13.2x
   behind Inductor purely on tile and staging choice. This is the cheapest remaining win and needs a pin sweep over
   its dominant piece, not a compiler change.
2. **`softmax x V` is a different disease and is undiagnosed.** Its axes are already separate, so the fused-coordinate
   story above does not apply. Every route tried left a ~128 us score-cone recompute standing, and cutting the score
   cone drops the remainder onto a scalar schedule that explodes (160 ms). Start by asking the same question that
   cracked the SDPA target: for each seam, which output axis does the cone miss, and what does the piece get offered.
3. **The elected shape is not the fast shape.** On the H100 the greedy elected `w8x2` (181 us) while `w2x1` measured
   61 us on the same route, because `w2x1` had no measured row. Seeding a handful of warp shapes per route before
   harvesting is worth more than any single schedule discovery here.
4. **The V100 needs its own golden and has TWO nested recompute cones**, not one — attention inside the o_proj
   contraction and gate/up inside the down_proj sweep. Cutting either alone still hangs. The composed cut that would
   remove both currently raises (see below), so the compiler work gates the measurement work on that card.

## Open defects found on the way

- **A cross-CTA split of a causally masked attention was wrong, and is now only mostly right.** The slicer reindexed
  the fold's operands to absolute k but not its own lift, so the mask compared against the partition-local index:
  505418/524288 elements wrong. Fixed here; a residual of 256/1048576 elements at max_abs 4.4e-3 still exceeds the
  strict tolerance, so `REDUCE=g<n>k` on that target remains unrecordable. The residual is two output rows' worth
  across all 128 dims and is NOT diagnosed — start there.
- **Composed multi-seam cuts raise** `Lambda body reads ['_r0'] it does not bind`, and the same for `['_ksplit']`.
  This blocks the V100 entirely and reproduces on main without any of this branch's changes.
- **A seam the enumerator offers can silently decide FUSE when pinned.** Several spellings `cuttable_seams` returns
  raise `MissingSiteError` at pin time, which the placement restriction reads as "addresses another kernel" and
  resolves to fuse — the catastrophic choice on these targets, with no diagnostic.
- **`--record-greedy` can write a golden the repository's own strict decode rejects.** The `softmax x V` kernel set
  references piece-routing rows spelling an atomic cross-CTA split; those rows fail strict decode on the compiler that
  wrote them minutes earlier, and removing them leaves the parent naming realizations that no longer exist.
- **The nest-aware serial-work feature is inert.** `D_serial_cell_work` is computed on every row and has no weight in
  either cold-start artifact; the checked-in freeze carries no column for `S_ext_serial_cell_work`, so it cannot be
  fitted from checked-in data. A 65,536x difference in nested recompute moves the cold-start price by exactly 0.0.
  PR #719 kept the stamp on the grounds that the prior can learn serial work; as checked in, it cannot.
