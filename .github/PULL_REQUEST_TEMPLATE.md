<!--
Title: a functional description, readable with no context. "Fix X", "Optimize Y", "Do X because Y".
Not a component name, not a branch name, not a ticket id.

Write this body, then revise it at least twice before posting. Each pass: read it as a reviewer who has no context,
check it against the rules below and against the design philosophy in AGENTS.md, and cut. A first draft is always too
long. Stop when nothing else can come out without losing the point.

Do not hard-wrap the text you write here. GitHub wraps it for the reader, and manual line breaks only make the
body hard to edit. The ~120-character rule applies to files in the repository, not to a pull-request body.
-->

## Abstract

Twenty-three more recorded rows come back, leaving thirteen of the original hundred and five. The 4080 and the PRO 6000 were entirely dead and cost nothing to repair: every one of their rows had simply left a knob unspelled where the enumeration spells it at its off value, and neither card needs to be reachable for that. Five rows on the 5090 pinned a staging choice their kernels no longer expose, so those were re-benched on the card rather than respelled, and the result is worth stating plainly — the tuning win they recorded is one the default pick has since absorbed. The rest of the work went into finding why five attention rows spell a tensor-core cell the enumeration never offers. That is located, and the one-line fix for it is not shippable; the memo says why.

| Card | Rows | Dead before | Dead now |
| --- | --- | --- | --- |
| RTX 4080 | 9 | 9 | 0 |
| RTX PRO 6000 | 10 | 9 | 0 |
| RTX 5090 | 63 | 8 | 3 |
| RTX 4090 | 73 | 10 | 10 |

---

## The two unreachable cards

All 18 of their rows differ from an offered row by the same single thing: the record omits `REDUCE` where the enumeration spells it at off. That is the class already closed on the other two cards, and it is meaning-preserving, so the measurements stand and no GPU is involved. That is exactly why these went first — neither card is reachable, and an omitted off value never needed one.

## The five re-measured rows

Each recorded `STAGE: d1/smem`. Those kernels expose no staging family at all now, so the key names a decision that does not exist and the row decodes the moment it is dropped. The stored microseconds do not survive that: they were taken with a synchronous shared-memory fill, and the kernel stages however the tiling stages it today. So they were re-benched at O3 on the card as pinned rows — a measurement, not a search.

All five land on the same schedule the greedy pick takes. `attention.hd64.dynM.softmax_v` recorded 13.51 us against a 59.03 us reference and now reads 11.56 against 11.54. The row is still evidence; it is no longer a win over the default.

## The f16-accumulate lockout

`classic_projection._atom_families` returns `base + reduced_acc` on its general path and only `base` on the CHUNK branch — it asks `atoms_for` at the default f32 accumulator and never for the reduced one. When attention's value channel became a chunked site, the f16-accumulate cell stopped being a candidate there. That is the whole cause: the pool offers thousands of `mma_m16n8k16_f16_f32` rows and not one `f16_f16`.

The precision gate is not involved. The resolved gate is folded into the pool key, so the two lanes do not share a cache entry, and pinning it explicitly changes nothing.

**Adding the reduced accumulator to that branch is not the fix.** It offers 2282 f16 rows and makes all five rows decode. It also empties the enumeration of `attention/sdpa-hd128-softmax-v-mma`, a closed corpus case: that case pins its TILE bare, so once the chunked score node accepts `f16_f16` the pin binds at both contraction sites and the pair realizes nothing. The chunk tier's lowering refuses this path for a reason the projection only exposes. Reverted rather than shipped with a corpus regression.

One fact for the next attempt, since it inverts the obvious reading: this atom does not produce an f16 result. Its mma chain runs on packed f16 partials and folds them into f32 shadows every 64 K-elements, and the shadows keep the names every sink reads. The repack gathers an f32 fragment either way, so `c_to_a_repack` keying on shape alone is correct and the refusal is somewhere else.

## What is left

13 rows, all attention, in `plans/golden-row-decode-gaps.md`: five behind the lockout above, two whose atomic cross-CTA reduce cannot fold a three-component carrier (and whose compiler-suggested alternative collapses the target to a scalar pair at 139.5 us against 26.1 us unsplit), four needing the 4090 for the staging re-measure, and two undiagnosed.

No row was re-recorded to make it green.

## Verification

339 passed, 7 skipped, 13 xfailed across the search tests. The realization corpus is 424 passed, 8 skipped, 3 xfailed on its GPU-free stages — the check that caught the projection change and the reason it is not in this diff.

`git diff --stat main -- emmy/` is +33 −20, all of it recorded rows. No compiler code changed.

**Draft.** `make test` has not been run.
