# Fused attention is 1.8x off its published number, and the shared memory says why

Status: measured on an RTX 5090, 2026-09-08. Two separate problems; keeping them apart is the point of this memo.

1. **The deploy is broken for a reason unrelated to the kernel.** No measured row reaches the fused attention
   kernel at all, so the greedy draws from the prior and ships an unscheduled scalar loop — up to 280x slower than
   torch. Fixable by recording, not by compiling.
2. **The kernel itself is 1.8x off the published FA-2 parity result**, at that article's own geometry. The emitted
   kernel is identical to the published one in grid, block, registers and occupancy, and differs in ONE number:
   **32 KB of shared memory then, 8 KB now.** That is exactly the ratio of two staged operands at ring depth two,
   against one operand at depth one.

This is about the FUSED single-pass kernel `F.scaled_dot_product_attention` traces to. The attention rows in the
model-agnostic hardware goldens decorate SEPARATE score / value / softmax-value targets that `torch.sdpa` no longer
traces to; they are healthy, and they are not on this path.

## Problem 1 — the deploy has no evidence

`emmy run --bench`, RTX 5090, fp16, `(1, 8, 512, D)`, against torch eager. `torch.compile` matches eager on every
row — it dispatches to the same flash kernel.

| shape | torch | emmy (greedy) | |
| --- | --- | --- | --- |
| hd64 | 9.3 us | 2611 us | 280x slower |
| hd64 causal | 10.2 us | 99 us | 10x |
| hd128 | 18.2 us | 35 us | 1.9x |
| hd128 causal | 12.0 us | 9628 us | 800x |
| hd256 | 29 us | 36700 us | 1270x |
| hd256 causal | 20 us | 574 us | 29x |

The spread is the cold-deploy draw, not the kernel. `--strict-evidence` names it exactly:

    strict evidence: kernel 'k_sdpa_0b2ceb' (node 'scaled_dot_product_attention') has no measured evidence for its
    030_cut fork (no measured row spells a kernel-set arm)

It fails at the FIRST fork, so every choice below is a prior draw over a ~10^9 row pool, and what deploys carries no
schedule at all — 40 registers, no tile, no transport. Identical on `main` (2647 us at hd64), so this predates the
golden-decode work and is untouched by it.

## Problem 2 — the kernel, compared to the published one

The article's own performance shape, `(1, 8, 4096, 64)` fp16, and its own final command. Both kernels below are the
same geometry: 4 warps, 2x8 register tiles, k-chunk 64, `PLACE=fuse`.

| | article, emmy @ `1674d7951` | today, emmy @ `b7673c813` |
| --- | --- | --- |
| latency | **204.7 us** (1.00x eager) | **367.2 us** (0.55x eager) |
| grid | 256 | 256 |
| block | 128 | 128 |
| registers | 254 | 255 |
| occupancy | 17% | 17% |
| **shared memory** | **32.0 K** | **8.0 K** |

Eager reproduces the article's 205 us to within 1% on every run here, so the reference is sound and the machine is
comparable. Everything about the two kernels matches except the slab footprint, and 32 K / 8 K = 4x is precisely
`2 operands x 2 ring slots` against `1 operand x 1 slot`:

- the article staged K **and** V, at `STAGE=d2/tma/ring`;
- today `_chunk_warp_stage` stages only the streamed VALUE — the score's operands reach fragment loaders directly,
  which no copy transport can serve — and returns `depth=1` however deep the row asked, because the chunk loop's
  body carries the whole softmax between fill and drain, so a prefetch would have to interleave with the merge
  rather than with an atom-K loop, "and that scheduling is not built".

Both halves of that are deliberate, documented refusals in today's tree. Together they are the whole difference in
the emitted kernel.

Corroborating: today all three transports land within 3% of each other (368 / 377 / 381 us for cp.async / gmem /
TMA). The article saw the same flatness and said why — "TMA saves about 1M LSU instructions compared to FA-2 but
measures flat: transport is not the bottleneck for this kernel". Transport choice was never the win. **Slab
residency was.**

## It is not a regression from the blocked carrier

The obvious suspect was `0b781fabd`, "FlashAttention-2 out of a blocked twisted carrier", reverted a day later. It
was re-measured here in a worktree, same shape, same geometry:

| tree | pinned schedule | emmy | eager | |
| --- | --- | --- | --- | --- |
| today | `w4x1` `f2x8/k4` `d1/smem-async` | **368.5 us** | 205.2 us | 0.56x |
| today | `w4x1` `f2x8/k4` `d1/smem-tma` | 380.7 us | 205.3 us | 0.54x |
| today | `w4x1` `f2x8/k4` gmem-direct | 377.4 us | 204.7 us | 0.54x |
| `0b781fabd` | `w4x1` `f2x8/k4` `d2/smem` | 394.0 us | 205.0 us | 0.52x |
| `0b781fabd` | `w4x1` `f2x4/k4` `d2/smem` (its own test row) | 716 us | 207 us | 0.29x |

**Today is 7% FASTER than the blocked-carrier tree at the same geometry.** `d2` resolves there and nothing deeper
than `d1` resolves here, and it bought that tree nothing — because it still staged one operand. Depth alone is not
the missing 1.8x; depth **and** the second operand together are.

`9afdeba9c` reverted #726 on purpose — the blocked carrier took the SDPA schedule space from 10^9 to 10^17 and
`--ir tile` from seconds to minutes — and said what it gave up: "FlashAttention-2, until the psi-framed carrier
replaces it". Today's chunk tier is that successor. It is not slower than what it replaced. Neither reaches the
published number.

## Where the loss happened, by date

The article is `cloudrift-landing` `f0f0701`, **2026-07-08**; emmy's main that day was `1674d7951`. Between then and
now the schedule machinery was rebuilt three times:

| date | commit | |
| --- | --- | --- |
| 2026-07-14 | `bbfced9e2` | Alternating **single-slab** staging for the warp-flash stream (`STAGE=d1/tma/alt`) |
| 2026-07-14 | `f2e0a473d` | Liveness-scheduled operand staging: unify `staged_kloop` / `alternating_kloop` |
| 2026-08-05 | `703a09485` | Tile scheduler rebuild: one-kind Fold IR, recursive row enumerator |
| 2026-08-19 | `c1abcc916` | FA restoration: one pass of the score, staged |
| 2026-09-01 | `b93a86c2e` | Rebuild classic scheduling around generic composition |
| 2026-09-05 | `0b781fabd` | FA-2 out of a blocked twisted carrier (reverted next day) |

`bbfced9e2` is the one to look at first, six days after the article: it names the move from two slabs to one in its
own title. `d1/tma/alt` alternated a SINGLE slab between K and V rather than holding both — the first step from 32 K
to 8 K, and the spelling no longer exists.

## The gemma-4 shape did not reach the tier at all

The same sweep at `(1, 16, 512, 256)` causal — the gemma-4 attention shape — landed nothing. Every staged candidate
refused with `STAGE pin does not resolve for this contraction`, at `d1/smem-async`, `d1/smem-tma` and `d2/smem-tma`
alike; the one gmem-direct candidate fell off the tensor cores entirely, 81147 us against eager's 33.4 us with no
mma stamped.

Whether that is a capability gap at a 256-wide head or the wrong geometry in the candidate set is NOT settled —
seven rows were tried and none landed. It wants its own pass.

## Translating the article's spelling

The article predates the `WORK` / `TILE` split and the site routes. It sets one tile for both sites (`TILE@dd` and
`TILE@pj` both read `a:mma_m16n8k16_f16/w4x1/f2x8/k4`). Today the fused kernel spells its families at two sites:
`@map.1/twist` is the EXPECTATION mma (`P·V`), `@map.1/twist.1/inner` the SCORE mma (`Q·K^T`). The fragment seam
ties them, so most pairs are illegal:

- the carrier's chunk is `bk x 16` — the twist tile's `k<n>` times the atom K;
- the score's N tile must EQUAL that chunk, so the inner tile's `n` is `2 x` the twist tile's `k<n>`;
- both tiles carry the same register rows, so their `f<m>x...` agree.

So `PLACE=fuse,TILE=a:mma_m16n8k16_f16/w4x1/f2x8/k4,STAGE=d2/tma/ring,WSPEC=` becomes:

    PLACE=fuse  WORK=w4x1
    TILE@map.1/twist=mma_m16n8k16_f16_f32/f2x8/k4
    TILE@map.1/twist.1/inner=mma_m16n8k16_f16_f32/f2x8/k4
    REDUCE@map.1/twist=  REDUCE@map.1/twist.1/inner=
    STAGE@map.1/twist=d1/smem-async  STAGE@map.1/twist.1/inner=
    RASTER=

`d2` and the second staged operand have no translation today. That is the finding, not a translation problem.

## The f16-accumulate cell, and a correction to #750

The gemma-4 article's "hybrid" — f16 tensor cores with a periodic f32 shadow promote every 64 K-elements — is what
put emmy AHEAD of torch:

| | RTX 5090 | RTX 4090 |
| --- | --- | --- |
| torch SDPA (FA-2) | 30.7 us | 41.0 us |
| emmy, f32 accumulate | 31.7 us | 37.1 us |
| emmy, hybrid | **29.7 us** | **33.8 us** |

On the 5090 the f32-only kernel LOSES to torch and only the hybrid wins. `_atom_families`' chunk branch offers the
plain accumulator alone.

#750 states the reduced accumulator "is not worth carrying at this tier", from a controlled measurement with
nothing but the accumulator moving:

| card | schedule | f32 | f16 |
| --- | --- | --- | --- |
| 5090 | `d1/smem-tma` | 5.2 us | 5.3 us |
| 5090 | gmem-direct | 16.2 us | 16.4 us |
| 4090 | `d2/smem-async` | 7.80 us | 7.94 us |

Those numbers are right and the conclusion drawn from them is too broad. They are `attention.hd128.qk` — the score
contraction ALONE, memory-bound, where the full-rate cell has nothing to pay for. The published win is on the FUSED
kernel, two mma chains, head dimension 256. The claim generalized from one half of the kernel to the whole tier.
The source comment in `_atom_families` carries the same over-reach and needs the same correction: the refusal may
still be right, but not for that reason and not on that evidence.

## What to do, in the order the evidence supports

1. **Give the fused kernel a measured row.** Nothing else matters while the greedy draws at random — the 280x
   deficit at hd64 is the draw, not the tier, and one recorded row per card per shape family removes it. Recording,
   not tuning: the prior is the broken part, so it must not also choose the candidates. The geometries above are
   the candidate set.
2. **Stage the score's operands, and ring deeper than one slot.** This is the measured 1.8x, and the shared-memory
   number says so directly. Start at `bbfced9e2` (2026-07-14) to see what `d1/tma/alt` did and why one slab
   replaced two.
3. **Settle the gemma-4 shape** — whether a 256-wide head can reach the tier at all.
4. **Re-measure the accumulator on the fused kernel**, not on `qk`, and correct or keep the refusal on that
   evidence.

## Artifacts

- Greedy table and strict-evidence failure: `emmy run --bench --bench-backends eager,tcompile,emmy -c ...` over the
  six shapes.
- Pinned sweeps: one `emmy run --bench` per candidate under `EMMY_KNOBS`, best-of against eager from the same run.
- The old tree was measured in a `git worktree` at `0b781fabd` with `PYTHONPATH` pointed at it, so the installed
  venv runs the old source without disturbing the checkout.
- Article source: `~/Projects/cloudrift-landing`, `packages/blog/content/blog/learning-flashattention-the-hard-way-part-2/index.md`,
  committed `f0f0701` on 2026-07-08. Its final bench block is the source of the 204.7 us kernel line quoted above.
- The FA-2 pin as `0b781fabd`'s own test wrote it (`tests/compiler/passes/test_flash_block.py`) — documented there
  as "the row the cold greedy deploys", NOT a tuned row, which is why it measures 716 us:

      TILE@map.1/twist.2/inner=mma_m16n8k16_f16_f32/f2x4/k4
      STAGE@map.1/twist.2/inner=d2/smem
      TILE@map.1/twist.2/inner.1/map.1/inner=mma_m16n8k16_f16_f32/f2x8/k4
      TILE@map.1/twist.1/reduce.1/inner=mma_m16n8k16_f16_f32/f2x8
      WORK=w4x1

- Published results: <https://riftstack.ai/research/learning-flashattention-the-hard-way-part-2> (the 206 us parity,
  the Move 0-4 progression, the FA-2 comparison table) and
  <https://riftstack.ai/research/optimizing-gemma-4-12b-rtx> (the hybrid accumulator, the 1.03x / 1.21x wins).
