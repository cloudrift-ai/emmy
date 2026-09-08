# Hardware-golden rows that still decode to nothing

Status: 11 of 155 recorded rows across the four hardware goldens equal no enumerated leaf, down from 105. Three are
on the RTX 5090, eight on the RTX 4090; the 4080 and the PRO 6000 are clean. Every one that is left is an attention
row, and every one is now blocked by a NAMED compiler behaviour — nothing left here is a spelling or a measurement.
This memo covers what blocks each, and the four dead ends already walked so nobody walks them twice.

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

Three things block them, and four rows carry two at once. `f16` means the row spells `mma_m16n8k16_f16_f16`;
`atomic` means its `REDUCE` names an atomic cross-CTA reduce; `no mma` means the pool offers no tensor-core tile of
any kind.

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
`mma_m16n8k16_f16_f32` rows and not one `f16_f16`.

**Root cause, located.** `classic_projection._atom_families` has a branch per tier. The general path returns
`base + reduced_acc` — the atoms at the plain f32 accumulator plus the ones whose accumulator is the multiplicand
dtype. The CHUNK branch returns only the first: it calls `atoms_for(dtype)` at the default `acc=F32` and never asks
for the reduced accumulator at all. When attention's value channel became a chunked site, the f16-accumulate cell
stopped being a candidate there.

### Dead end 1 — the precision gate

It is not involved. `pinned_knobs({"FAST_MATH": True})` resolves `precision_pin(F16_MMA_F32_ACC)` to `True`, and
`schedule_pin_fingerprint` folds the resolved gate into the pool key, so the two lanes do not share a cache entry.
Pinning `F16_MMA_F32_ACC` explicitly changes nothing.

### Dead end 2 — tightening `c_to_a_repack`

`FragmentRepack` reads four f32 values a lane and packs them with `cvt.rn.f16x2.f32`, and `c_to_a_repack` returns
`True` on shape alone — which reads like an over-claim for an atom whose C operand is f16, and like the reason the
chunk branch excludes it. It is not. Requiring `operand_dtype("c") == F32` there was tried and reverted.

The atom does not produce an f16 result. Its mma chain runs on packed f16 partials (`_ch{i}_{j}`) and folds them
into f32 SHADOWS every 64 K-elements; the shadows keep the `_c{i}_{j}` names every sink reads
(`_atom._mma_c_base`, `_f16acc_promotes`). The repack gathers an f32 fragment either way, so keying on shape is
right. The technique — full-rate HMMA with periodic promotion into f32 shadows, bounding the error to 64-element
chunks — is written up at <https://riftstack.ai/research/optimizing-gemma-4-12b-rtx>.

### Dead end 3 — just adding the reduced accumulator

Adding `atoms_for(chunk_dtype, acc=chunk_dtype)` to that branch offers 2282 f16-accumulate rows and makes all six
rows decode once the staging key is dropped. It also fails `attention/sdpa-hd128-softmax-v-mma`, a closed corpus
case, with `no enumerated row carries the pin (0 rows offered at sm_120)`. That case pins its TILE bare, so once the
chunked score node also accepts `f16_f16` the pin binds at both contraction sites and the pair realizes nothing.

So the chunk tier's lowering refuses this path for a reason the projection only exposes. Start by finding which site
the bare pin binds to in that case and what the pair cannot realize — not by widening the projection again.

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
