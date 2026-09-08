# Hardware-golden rows that still decode to nothing

Status: 11 of 155 recorded rows across the four hardware goldens equal no enumerated leaf, down from 105. Three are
on the RTX 5090, eight on the RTX 4090; the 4080 and the PRO 6000 are clean. Every one that is left is an attention
row. This memo covers what blocks each, and the dead ends already walked so nobody walks them twice.

Every one of the eleven also pins `STAGE: d1/smem`, the synchronous fill a `cp.async` target never offers, so every
one needs a re-measure on its own card whatever else is fixed. That is now worth doing for its own sake: the chunk
tier stages its streamed value since this memo's blocker 1 closed, and a re-measured row would deploy a kernel
1.4–2.6× faster than the gmem-direct one it names today. The row table below names the OTHER blocker each carries.

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
| The kernel offers no staging family at all | 7 | Drop the key, re-measure on the card |

The first two are meaning-preserving and needed no GPU. The third does not preserve meaning — the stored
microseconds were taken with a synchronous shared-memory fill — so those rows were re-benched at O3 on the card
that recorded them, five on the 5090 and two on the 4090. All seven realize what the greedy pick takes anyway: the
tuning win they recorded, 13.51 us against a 59.03 us reference in one case, is a win the default has since
absorbed.

## The 11 that are left

Beside the staging key above, three things block them, and four rows carry two at once. `f16` means the row spells
`mma_m16n8k16_f16_f16`; `atomic` means its `REDUCE` names an atomic cross-CTA reduce; `no mma` means the pool offers
no tensor-core tile of any kind.

| Row | Card | Blocked by |
| --- | --- | --- |
| `attention.hd128.softmax_v#1` | 5090 | f16 |
| `attention.hd64.softmax_v#1` | 5090 | f16, atomic (`g4a`) |
| `attention.hd64.softmax_v#2` | 5090 | atomic (`g4a`) |
| `attention.hd128.pv#1` | 4090 | atomic (`g2a`) |
| `attention.hd128.pv#2` | 4090 | f16, atomic (`g2a`) |
| `attention.hd128.dynM.pv#2` | 4090 | f16 |
| `attention.hd64.dynM.pv#2` | 4090 | f16 |
| `attention.hd64.pv#1` | 4090 | no mma (split pieces) |
| `attention.hd64.pv#2` | 4090 | f16, no mma (split pieces) |
| `attention.hd256.dynM.pv#1` | 4090 | no mma (chunk tier refuses) |
| `attention.hd256.dynM.pv#2` | 4090 | no mma (chunk tier refuses) |

## Blocker 1 — the chunk tier never offers the reduced accumulator (6 rows)

Each of the six spells `mma_m16n8k16_f16_f16` under `FAST_MATH: true`, and the pool offers thousands of
`mma_m16n8k16_f16_f32` rows and not one `f16_f16`. `classic_projection._atom_families` has a branch per tier; the
general path returns the atoms at the plain f32 accumulator plus the ones accumulating in the multiplicand dtype,
and the CHUNK branch returns only the first. When attention's value channel became a chunked site, the
f16-accumulate cell stopped being a candidate there.

**Closing it is worth almost nothing, and the reason matters more than the blocker.** The obvious argument for
closing it is that `attention.hd128.softmax_v` recorded 11.84 us with that cell and its greedy pick measures 16.1 us
today — 1.36× off its own recorded best. That 1.36× is real. It is not the accumulator.

`attention.hd128.qk` still lowers through the general tier, still offers both accumulators, and still decodes, so it
answers both questions at once. Same card, same geometry, only the two variables moving:

| STAGE | accumulator | us |
| --- | --- | --- |
| `d1/smem-tma` | f32 | **5.2** |
| `d1/smem-tma` | f16 | 5.3 |
| none | f32 | 16.2 |
| none | f16 | 16.4 |

The staged fill is worth **3.1×**. The reduced accumulator is worth nothing — marginally negative at both staging
states. The recorded rows say the same thing once they are read as pairs: every `qk` / `softmax_v` pair whose two
lanes share a staging state differs by 1.006–1.05× between accumulators, and the one pair that shows 1.36× —
`hd128.softmax_v` — has a staged f16 row against an UNSTAGED f32 one. The published measurements
(<https://riftstack.ai/research/optimizing-gemma-4-12b-rtx>) are consistent: 1.34–1.61× on the projection kernels,
where the mma chain is the kernel, and 1.03× on attention overall.

**So the six rows are worth closing only for the staging — which is now DONE.** The chunk tier had no staging
family at all by construction: `TileOp.stage_edges` excluded a chunked carrier because its tier read every operand
gmem-direct, so "a transport spelling there would decide nothing". That was the regression the refactor introduced,
and it is closed: the tier stages the value it streams, and on this 5090 `attention.hd128.softmax_v` goes 16.1 →
11.2 us and `attention.hd64.softmax_v` 26.2 → 10.1 us, recovering the 11.84 us the first of them recorded before the
refactor. The rows still need re-measuring on their own cards to deploy it, because each pins `STAGE: d1/smem` — the
synchronous fill a `cp.async` target never offers — and a measured row is what the greedy reads.

### What the staging re-measure reached, and what it did not

Re-measured on the 5090 as a two-candidate pinned bench per target — the recorded row's own geometry against each
transport it offers, nothing else moved:

| Target | recorded | staged | outcome |
| --- | --- | --- | --- |
| `attention.hd128.softmax_v` (std) | 16.15 us | **11.0 us** `d1/smem-tma` | re-recorded; the greedy deploys it |
| `attention.hd128.dynM.softmax_v` | 22.2 us | not offered | row kept |
| `attention.hd64.dynM.softmax_v` | 11.56 us | not offered | row kept |

The two `dynM` targets refuse every copy transport because their KEY extent is symbolic, and both copy gates —
`_warp_vector_copy` and `_warp_tma` — require a static, chunk-divisible K. The chunk tier's own staged emitter
agrees: it reads `key.extent.as_static()` and asserts the chunk is not ragged.

**The work that would close them.** `staged_kloop` already takes a symbolic `k_extent` as a `Dim`, allocates the
full ring against the tuned hint, and lets the transport absorb the over-primed tail — "the drain masks those keys
to the fold identity, so it stays bit-identical to gmem-direct". So the skeleton is not what refuses. What refuses is
the pair of resolver gates above and the two static reads in `_FlashOps.reduce`. Closing it means giving the chunked
carrier the same ragged-tail discipline the gmem-direct path already has (`bound`, the boundary `FragmentMask`) on
the fill side instead of the drain side. Until then a symbolic-sequence attention kernel stays gmem-direct, which on
the static targets is worth 1.4–2.6×.

**What was tried.** The chunk tier was given the promote scheme: its expectation chain accumulates packed and folds
into an f32 partial once per chunk, its score chain widens to f32 on its own. It builds and computes the right
answer on a 5090, and it recovers none of the gap — 16.0 us against the f32 path's 16.1, and 17–40% SLOWER at the
`k4` cadence where the promote fires most often. Reverted, and there is no reason to try it again: the transport
beside it was the whole 1.36×.

### Dead end 1 — the precision gate

It is not involved. `pinned_knobs({"FAST_MATH": True})` resolves `precision_pin(F16_MMA_F32_ACC)` to `True`, and
`schedule_pin_fingerprint` folds the resolved gate into the pool key, so the two lanes do not share a cache entry.
Pinning `F16_MMA_F32_ACC` explicitly changes nothing.

### Dead end 2 — tightening `c_to_a_repack`

`FragmentRepack` reads four f32 values a lane and packs them with `cvt.rn.f16x2.f32`, and `c_to_a_repack` returns
`True` on shape alone. Requiring `operand_dtype("c") == F32` there was tried and reverted, on the argument that the
atom folds its packed partials into f32 shadows and the repack gathers an f32 fragment either way.

That argument holds for the tiers that IMPLEMENT the shadows. The chunk tier does not: it declares its own score,
partial and carried fragments at the atom's C dtype and reads all three back element-wise as f32 — the scale, the row
maximum, the exp and the merge all do. Offered without a promote scheme the cell builds and returns inf, its
epilogue reading a packed f16 pair as four f32 scalars. Keying on shape is still right; the shadows are what the
tier is missing.

### Dead end 3 — the corpus case the widening broke

Adding `atoms_for(chunk_dtype, acc=chunk_dtype)` to that branch also failed `attention/sdpa-hd128-softmax-v-mma`, a
closed corpus case, with `no enumerated row carries the pin (0 rows offered at sm_120)`. That was read as the chunk
tier's lowering refusing the path. It was not: it was a separate compiler bug, now fixed. The case pins its TILE
bare on a kernel that spells TILE at two sites, and the enumeration bound a bare pin at EVERY site that could spell
the value — a conjunction. As soon as the chunked value channel could also spell `f16_f16` the pin bound at both
contraction sites, and no schedule carries one mma tile at both. The same thing was already true on `main` for any
value both sites can spell: a bare `TILE=mma_m16n8k16_f16_f32/f1x4/k2` offered 0 rows while the same value pinned at
the inner site alone offered 2. A bare pin is a disjunction — one site carries it, the others are OFF, the reading
`unreproducible_pin_flag` and `evidence_row_vouches` already gave it.

## Blocker 2 — the atomic cross-CTA reduce refuses a multi-component carrier (4 rows)

`attention.hd64.softmax_v#1`/`#2` on the 5090 (`g4a`) and `attention.hd128.pv#1`/`#2` on the 4090 (`g2a`). The 5090
pair is the verified one: `#2` decodes with the staging key dropped and then fails to build.

    atomic REDUCE folds ONE additive state component; this carrier has 3 (acc0, acc1, acc2__sum)
    — use the deferred f32 workspace finalize (REDUCE=g<n>k)

Attention's carrier folds a running maximum, a denominator and an expectation, so an atomic fold over one additive
component cannot express it. Both pairs are verified on their own card: the 4090's `attention.hd128.pv` refuses to
compile with the same message on both lanes.

### Dead end 4 — the compiler's own suggestion

Respelling `g4a` as `g4k` splits the target into pieces that take no mma tile at all — both lanes come back
`unreproducible pin: TILE=... realized (unset)` — and the greedy pick for that shape falls to a scalar pair at
139.5 us + 4.2 us against 26.1 us unsplit. So the choice is to make the atomic reduce carry a multi-component fold,
or to accept that these rows have no cross-CTA plan and re-measure them unsplit.

## Blocker 3 — the tensor-core tier is absent from these targets (4 rows)

`attention.hd256.dynM.pv#1`/`#2` and `attention.hd64.pv#1`/`#2` on the 4090. Each records an
`mma_m16n8k16_f16_f32` row, and the pool offers no mma tile of any kind — only the scalar tier (`t32x8`, `f26x4`,
`f1x*`). Two different causes, both now named; neither is blocker 1 or 2.

**hd256 is a chunk-tier refusal.** `_node_refusal` answers outright:

    the chunk tier reads its score operands and its streamed value as slabs

So at that head dimension the tier declines the target and the warp atoms are never projected. Whether a 256-wide
head SHOULD reach the chunk tier is the question to settle; the refusal is deliberate, not incidental.

**hd64.pv is the split's doing.** Its node refusal is `None` — the tier is willing — but the row spells
`REDUCE: g4k`, so the replay follows the split arm and what enumerates is the PIECES. Those offer only scalar tiles.
This is the same effect seen when `g4a` was respelled to `g4k` on the 5090 (dead end 4): a cross-CTA split mints
pieces the tensor-core tier does not serve. `attention.hd64.dynM.pv#1`, which spells no split, keeps its mma rows —
that is the controlled comparison.

Closing hd64.pv therefore means the same question as blocker 2: why a split's pieces lose the warp tier.

## Working on the 4090

The card at `riftuser@211.21.50.85 -p 57010` has the CUDA toolkit at `/usr/local/cuda` but nvcc is NOT on the
default PATH, and emmy dropped its NVRTC fallback — so every bench dies with `nvcc unavailable` until the run
carries `CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH`. `make setup` there takes about twenty minutes,
almost all of it pulling CUDA wheels.

The pinned bench that re-measures a staging row:

    emmy run --golden <file> --realization <name> --bench --bench-backends emmy --json <out>

Promote `emmy_us` from the pinned row's isolated timing and `reference_us` from the greedy isolated timing, with
`reference_backend: same-input-greedy`.
