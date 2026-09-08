# Hardware-golden rows that still decode to nothing

Status: 36 of 155 recorded rows across the four hardware goldens equal no enumerated leaf. Down from 105. This memo
covers what is left, why each class is left, and what closing it needs.

Do not re-record a row to make it green. A row that stops decoding because the schedule it names is gone is a
regression in the enumeration, and recording today's pick in its place writes that regression in as the reference.
Tuning is not the repair either: the prior the search steers with is trained on this evidence, so a tuning round
launders the same hole back into the file it is meant to fix.

## Where it stands

| File | Rows | Dead | Diagnosed |
| --- | --- | --- | --- |
| rtx5090_sm120 | 63 | 8 | yes, all attention |
| rtx4090_sm89 | 73 | 10 | yes, all attention |
| rtx4080_sm89 | 9 | 9 | no |
| rtxpro6000_sm120 | 10 | 9 | no |

Two causes are already closed. Forty-two rows omitted a family the enumeration offers at its off value, mostly
`RASTER`; spelling the key preserves the schedule, so the measurements stand. Twenty-seven more carried a `g<n>`
cross-CTA split in their schedule row, which the piece the split mints can never stamp — a decode bug, fixed by
comparing the piece row.

## Class 1 — the kernel offers no staging family (9 rows)

    rtx5090  attention.hd128.softmax_v#2, attention.hd128.dynM.softmax_v#1, attention.hd128.dynM.softmax_v#2,
             attention.hd64.softmax_v#2, attention.hd64.dynM.softmax_v#1, attention.hd64.dynM.softmax_v#2
    rtx4090  attention.hd128.pv#1, attention.hd128.dynM.pv#1, attention.hd64.dynM.pv#1

Each records `STAGE: d1/smem`. The pool for these kernels carries no `STAGE` key at all, so staging is no longer a
decision they expose, and every row decodes the moment the key is dropped. The spelling is settled; the measurement
is not. Those microseconds were taken with a synchronous shared-memory fill, and today's kernel stages however the
tiling stages it, so the stored number no longer describes what would run.

Closing it: drop the key, then re-measure each row on its own card with a pinned bench —
`emmy run --golden <file> --realization <name> --bench --ab "<knobs>" --record`. That is a measurement, not a
search. The 5090 is local; the 4090 needs its host prepared the way the tune-kernels skill describes.

## Class 2 — the f16-accumulate atom is not a candidate (6 rows)

    rtx5090  attention.hd128.softmax_v#1, attention.hd64.softmax_v#1
    rtx4090  attention.hd128.pv#2, attention.hd128.dynM.pv#2, attention.hd64.pv#2, attention.hd64.dynM.pv#2

Every one records `mma_m16n8k16_f16_f16` under `FAST_MATH: true`. The pool offers thousands of
`mma_m16n8k16_f16_f32` rows and not one `f16_f16`, and no respelling of `STAGE` reaches it.

The precision gate is not the cause. `pinned_knobs({"FAST_MATH": True})` resolves
`precision_pin(F16_MMA_F32_ACC)` to `True`, so the replay allows the atom; pinning `F16_MMA_F32_ACC` explicitly
changes nothing. The atom is simply not among the site's candidates.

The evidence that this is a real gap rather than a stale row: the realization corpus case
`sdpa-hd128-softmax-v-mma` passes today with the SAME pins, the same `WORK: w2x4`, and the same
`TILE: mma_m16n8k16_f16_f16/f1x4/k8`. A minimized softmax@V snippet realizes the schedule; the same schedule inside
the whole-attention target does not. So the refusal depends on the target's shape, not on the schedule or the
precision regime.

Closing it: find where the f16-accumulate atom leaves the candidate set for the fused attention target but not for
the standalone snippet. The contraction tiers were rewritten recently (#742), which is the first place to look.

## Class 3 — undiagnosed (3 rows)

    rtx4090  attention.hd256.dynM.pv#1, attention.hd256.dynM.pv#2, attention.hd64.pv#1

`FAST_MATH: false`, f32-accumulate, and still dead after the staging key is dropped. `hd256.dynM` offers only 234
candidate rows against the 2778 its siblings offer, so the enumeration is much narrower there — worth checking
first whether the target itself lowers differently at that head dimension.

## The other two cards

The 4080 (9 of 9) and the PRO 6000 (9 of 10) were never diagnosed; the work above was scoped to the two cards that
are reachable. Both hold matmul rows only, so they are unlikely to share the attention classes here. Diagnose them
the same way: compare the recorded row with the closest offered one and read which families differ.
