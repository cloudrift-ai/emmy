# Hardware-golden rows that still decode to nothing

Status: 13 of 155 recorded rows across the four hardware goldens equal no enumerated leaf, down from 105. Three are
on the RTX 5090, ten on the RTX 4090; the 4080 and the PRO 6000 are clean. This memo covers what is left, what each
class needs, and the dead ends already walked so nobody walks them twice.

Do not re-record a row to make it green. A row that stops decoding because the schedule it names is gone is a
regression in the enumeration, and recording today's pick in its place writes that regression in as the reference.
Tuning is not the repair either: the prior the search steers with is trained on this evidence, so a tuning round
launders the same hole back into the file it is meant to fix. A pinned bench is fine — it is search that is off
limits, not measurement.

## What closed, and how

| Cause | Rows | Repair |
| --- | --- | --- |
| A family omitted where the enumeration spells it at its off value | 60 | Spell the key; measurements survive |
| A `g<n>` split compared to a leaf the pieces cannot spell | 27 | Fixed the decode to compare the piece row |
| The kernel offers no staging family at all | 5 | Drop the key, re-measure on the card |

The first two are meaning-preserving and needed no GPU. The third does not preserve meaning — the stored
microseconds were taken with a synchronous shared-memory fill — so those rows were re-benched at O3 on the 5090.
All five now realize what the greedy pick takes anyway: the tuning win they recorded, 13.51 us against a 59.03 us
reference in one case, is a win the default has since absorbed.

## Class A — the chunk tier never offers the reduced accumulator (5 rows)

    rtx5090  attention.hd128.softmax_v#1
    rtx4090  attention.hd128.pv#2, attention.hd128.dynM.pv#2, attention.hd64.pv#2, attention.hd64.dynM.pv#2

Every one records `mma_m16n8k16_f16_f16` under `FAST_MATH: true`, and the pool offers thousands of
`mma_m16n8k16_f16_f32` rows and not one `f16_f16`.

**Root cause, located.** `classic_projection._atom_families` has a branch per tier. The general path returns
`base + reduced_acc` — the atoms at the plain f32 accumulator plus the ones whose accumulator is the multiplicand
dtype. The CHUNK branch returns only the first: it calls `atoms_for(dtype)` at the default `acc=F32` and never asks
for the reduced accumulator at all. When attention's value channel became a chunked site, the f16-accumulate atom
stopped being a candidate there.

**The precision gate is not involved.** `pinned_knobs({"FAST_MATH": True})` resolves
`precision_pin(F16_MMA_F32_ACC)` to `True`, and `schedule_pin_fingerprint` folds the resolved gate into the pool key,
so the two lanes do not share a cache entry. Pinning `F16_MMA_F32_ACC` explicitly changes nothing.

**The obvious fix works on the goldens and breaks the corpus, so it is not the fix.** Adding
`atoms_for(chunk_dtype, acc=chunk_dtype)` to that branch offers 2282 f16-accumulate rows and makes all five rows
decode (with the staging key dropped) — and it fails `attention/sdpa-hd128-softmax-v-mma`, a closed case, with
`no enumerated row carries the pin (0 rows offered at sm_120)`. The case pins its TILE bare, so once the chunked
score node also accepts `f16_f16` the pin binds at both contraction sites and the pair realizes nothing. Something
in the chunk tier's lowering refuses the f16-accumulate path; the projection is only where it becomes visible.

**Worth knowing for whoever takes it.** The atom is not an f16 result. Its mma chain runs on packed f16 partials
(`_ch{i}_{j}`) and folds them into f32 SHADOWS every 64 K-elements; the shadows keep the `_c{i}_{j}` names every
sink reads (`_atom._mma_c_base`, `_f16acc_promotes`). So `FragmentRepack` — which reads four f32 values a lane and
packs them with `cvt.rn.f16x2.f32` — is gathering an f32 fragment either way, and `c_to_a_repack` keying on shape
alone is right rather than an over-claim. The refusal is elsewhere. Find which site the bare pin binds to and what
the pair cannot realize.

## Class B — the atomic cross-CTA reduce refuses a multi-component carrier (2 rows)

    rtx5090  attention.hd64.softmax_v#1, attention.hd64.softmax_v#2

Both pin `REDUCE: g4a`. They decode — `piece_row` strips the cross-CTA half — but they do not build:

    atomic REDUCE folds ONE additive state component; this carrier has 3 (acc0, acc1, acc2__sum)
    — use the deferred f32 workspace finalize (REDUCE=g<n>k)

Attention's carrier folds a running maximum, a denominator and an expectation, so an atomic fold over one additive
component cannot express it.

**The compiler's own suggestion does not work.** Respelling `g4a` as `g4k` splits the target into pieces that take
no mma tile at all — both lanes come back `unreproducible pin: TILE=... realized (unset)` — and the greedy pick for
that shape falls to a scalar pair at 139.5 us + 4.2 us, against 26.1 us unsplit. So the choice is to make the atomic
reduce carry a multi-component fold, or to accept that these rows have no cross-CTA plan and re-measure them unsplit.

The 4090 has the same class in `attention.hd128.pv#1`/`#2` (`g2a`) and `attention.hd64.pv#1`/`#2` (`g4k`).

## Class C — the 4090's share of the staging class (4 rows)

    rtx4090  attention.hd128.dynM.pv#1, attention.hd128.dynM.pv#2, attention.hd64.dynM.pv#1, attention.hd64.dynM.pv#2

Same as the five already closed on the 5090: `STAGE: d1/smem` where the kernel offers no staging family, decoding as
soon as the key is dropped. Needs the card at `riftuser@211.21.50.85 -p 57010`, prepared the way the tune-kernels
skill describes, then the same pinned bench:

    emmy run --golden <file> --realization <name> --bench --bench-backends emmy --json <out>

Promote `emmy_us` from the pinned row's isolated timing and `reference_us` from the greedy isolated timing, with
`reference_backend: same-input-greedy`. Two of the four also need Class A closed first.

## Class D — undiagnosed (2 rows)

    rtx4090  attention.hd256.dynM.pv#1, attention.hd256.dynM.pv#2

`FAST_MATH: false`, f32-accumulate, still dead after the staging key is dropped and unaffected by Class A.
`hd256.dynM` offers 234 candidate rows where its siblings offer 2778, so that target enumerates far more narrowly —
check first whether it lowers differently at that head dimension.
