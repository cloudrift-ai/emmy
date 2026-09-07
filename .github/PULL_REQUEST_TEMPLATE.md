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

Attention reached the tensor cores through a rewrite that split its key axis into an outer stride and an inner block. The block bought the expectation channel a semiring to live in, and cost eight orders of magnitude of schedule space, three copies of one score in the tree, and a minute of compile time. This change deletes the rewrite and stores what a twist recipe already declares — the plain per-element contribution, plus the recipe itself — so the expectation is a bare product of two operand edges with no block needed to see it. A tensor-core tier then folds the whole carrier one staged chunk at a time: build the chunk's score, reduce it per row into the chunk's pivot, apply the recipe's own channel maps against that pivot, multiply the weight against the streamed value on tensor cores, and merge once per chunk through the recipe's stable combine. The block it folds is the schedule's staged chunk and nothing else, so no width is ever read off an extent.

### f16 SDPA, `(1, 8, 512, 128)`, RTX 5090

| | main | this branch |
| --- | --- | --- |
| tuned | 176 µs | **37 µs** |
| `--ir tile` | 55 s | 2 s |
| cold, before anything is measured | 177 µs | **73 ms** |
| lines under `emmy/` | — | **−633** |

---

## What the term stores

A twisted fold used to store the conjugated program: `Fold.twist` folded the pair into the stable carrier and kept THAT, after which the bilinear product lived inside a rescale program where no reading looks. It now stores the base monoid's own contribution as its `lift`, that monoid's componentwise ⊕ as its `base`, and the recipe — bound to this term's roles — as its `twist`. The stable ⊕ and the ψ-image of the lift are both derived from those (`Fold.combine`, `Fold.injected`), so there is one stored spelling of the algebra.

That is what makes attention's expectation read as `weight ⊗ value` in the term. `Fold.as_contraction` reads one CHANNEL's cone rather than the whole lift, so the channel is recognized beside a running maximum and a denominator that are no product at all; the weight's own cone became an operand edge, which is what leaves the channel a bare monomial.

Nothing may see through ψ. The base form denotes `Sum exp(score)` and overflows, so a serial step folds the recipe's authored per-channel injections instead of evaluating ψ, and the chunk tier folds the recipe's patterns.

## The chunk tier

`Fold.tiles_whole` decides whether a node is a TILE site, which schedule domain it takes, what transport catalog its edges get, whether it holds a fragment at a seam, and whether a root has a chain. It answers yes for a twisted carrier through `Fold.chunked`: the recipe names a pattern for every state past the pivot, supplies the stable ⊕ at an open channel count (which is what a per-chunk merge needs), and leaves exactly one bilinear channel — so every other state rides as a per-row scalar and the one accumulator is the expectation. Neither half of that reading mentions attention or softmax.

Three things keep the emitter small.

The SCORE is the nested contraction the tree already carries as a site of its own, and the fragment seam already tied the two together: the score's N tile must equal the consumer's chunk, one warp column wide, same register rows. That is FlashAttention's shape, stated where the schedule can see it, so the enumeration needed no rule of its own — the tier realizes the agreement rather than assuming it.

The WEIGHT reaches the expectation's `mma.sync` as a register repack of the score's own C fragments. No shared-memory round trip, and `FragmentRepack` was already in the tree for exactly this.

`_residence` evaluates a recipe pattern, the chunk merge and the projection epilogue at whatever residence each value has — a C fragment, the two per-lane registers an m16n8 row rides in, or cell-uniform. None of the three is written for tensor cores; where a value lives is a fact about the tile, not about the program.

## The key extent may be symbolic

Nothing in the tier sizes itself against the key extent, so a ragged last chunk is the only thing a dynamic stream needs. Its overhanging columns fill with the pivot's identity through the node the causal mask already used: the pivot ignores them, every channel's pattern folds a zero weight through them, and the loads past the extent clamp and zero the way every gmem-direct fragment loader already does. Measured: a symbolic key length of 509 against a static query length of 512 compiles, emits 39 `mma.sync`, and matches eager to 2.4e-4.

## What got worse

**The cold pick.** With nothing measured for the shape, the offline prior takes a placement cut whose consumer folds the carrier serially with the value dimension on the grid — which recomputes the score once per output column. That is 73 ms against main's 177 ms cold pick. One `emmy tune` fixes it (37 µs, and the greedy picks it from then on), and a deployed model answers from its goldens rather than from the cold prior, but an ad-hoc unmeasured compile of this shape is much worse than it was. This is a ranking shortfall, not a realization gap, so it is reported rather than recorded as a corpus case.

**A twisted carrier no longer offers a scalar register tile.** The scalar tier folds the stored lift into one accumulator per bilinear channel, which for a twist is the base contribution — it would have folded `Sum exp(score)`. The three fp32 `qwen3emb/sdpa-*` cases pinned exactly that row; they were already recorded gaps and stay recorded ones, with the reason corrected.

**There is no FlashAttention on fp32 or on a materialized score.** The tier needs a 16-bit atom whose C fragment repacks into an A operand, and it builds the chunk's score from a nested contraction. A score that arrives as a slab — the placement cut's arm, and the two `matmul/f16-symbolic-*` cases — reaches no chunk. Both are open cases carrying that reason.

## What may not come back

The blocked emitter went with the rewrite: a carrier term holding one operand per carried component, every one folding the same explicit block, plus the residence evaluator that interpreted a lambda at fragment / row / uniform residence and the kernel IR nodes only it produced. An emitter for this carrier takes its block from the SCHEDULE and never from a second reduce axis carved into the term. `FragmentRowReduce` did come back — a per-row fold over one warp's C fragments is what a chunk pivot IS — but as a leaf the tier emits, not as an interpreter of a term.

## Two fixes the tier exposed

A fragment seam was derived for every site that reads bilinear. A twisted carrier reads bilinear on one channel, so it claimed a `need` at its score's seam — and while no tier folded it, that need spelled "free", which refuses the score producer every tile it has. Attention's `Q·K` came out untiled. The facts are now scoped to `Fold.tiles_whole`, and a carrier the tier does not fold claims nothing at the seam.

A short-query attention traces with the KEY in the score's canonical A slot, and two readings came out transposed: the score got a `(key, query)` tile, and the emitter read the mma's A off the stored operand order. Under a chunked carrier the orientation is the consumer's — the tier builds a `(row, chunk)` tile whichever operand the canonical form put in A — and the emitter picks its A by which operand carries the carrier's row.

## Evidence

`make test`, `make lint` and `make test-goldens` all pass. The realization corpus is green at parity with main (the same two failures main has: `reduce/rms-norm-cut-sweep-work` and a `qwen3emb/gated-mlp-s128` worker crash under xdist, both reproduced on `main` unchanged).

Three e2e cases cover the tier: an f16 SDPA on a pinned chunk row is one `mma.sync` kernel matching torch, once chunk-aligned, once ragged, once symbolic.

Every kernel identity for attention changed by design; the corpus was restamped with `make test-corpus-regen` and the checked-in model goldens will need a tuning round on their cards.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01NHvyWiofb3RYvVoDzqSeoQ
