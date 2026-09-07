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

A fused softmax carrier stored its per-element contribution in the coordinates the algebra is *defined* in rather than the ones it *computes* in, so every attention tree carried an exponential of an unshifted score — a value that overflows if anything evaluates it and that nothing was permitted to evaluate. Keeping it there also forced an extra node into the tree whose only job was to turn that value into an operand so a pattern match would fire. This change stores the contribution in the carrier's own coordinates, where softmax's is a score, a one, and the streamed value, and derives the other form on demand for the one reader that needs it. The tree loses the exponential and the node, and attention's numerics are unchanged against eager.

```
 ├─ operand[v1, e]: Fold  free                    ├─ operand[acc0]: Fold[a3] contraction
 │    v1 = multiply(acc0, 0.125)                  ├─ operand[in7]: load v[...]
 │    e  = exp(v1)              ← overflows       ├─ init: (-1e+30, 0, 0)
 ├─ operand[in7]: load v[...]                     ├─ lift: λ(a2, acc0, in7) -> (v1, one, in7)
 ├─ lift: λ(a2, v1, e, in7) -> (v1, e, ev)        │    v1  = multiply(acc0, 0.125)
 │    ev = multiply(e, in7)                       │    one = 1
 └─ base: (maximum, add, add)                     ├─ combine: λ(acc1, acc3, acc5__sum, …) -> …
                                                  ├─ helper: psi λ(m, D, O) -> (m, d, o)
                                                  └─ helper: base = (maximum, add, add)
             before                                                  after
```

---

## Why the ids differed

The paper states both representations of a twisted fold as equivalent, and its FlashAttention figure draws the stable one. The implementation had taken the base one. The fusion now splices the recipe's authored injection — the singleton in the carrier's own state space, where the pivot *is* the score and `exp(s)·v` has already simplified to `v`. The stable combine derives from the recipe, and the base monoid and ψ ride beside it as helpers.

`Fold.based` is the reading those helpers exist for: ψ⁻¹ over the stored lift, restoring `(s, exp s, exp(s)·v)`. Bilinearity lives in base coordinates and nowhere else, because ψ divides the product away at the singleton. Only `bilinear_channels` and `as_contraction` ask for it, nothing emits it, and it is restricted to the channels a term actually holds — a half-fused carrier is the ordinary case during the rewrite's fixpoint.

## The node that is no longer minted

`_factor_weights` hoisted the weight into an operand of its own so the expectation channel read as `operands[0] ⊗ operands[k]`. That node added no computation; the emitter never placed it. It existed to satisfy a pattern.

The pattern now asks the right question. **A is what `operands[0]` supplies, not what it exposes** — the left factor may be a component of that edge, or a value the reading derives from those components and from kernel-uniform ones. A scale contributes no variation, so a factor reading it varies exactly as A does. A weight derived from the *streamed* value is still refused; offering an mma there would be a wrong answer, not a slow kernel.

## Three things the dump was hiding

A scalar operand is spelled inside the reader that binds it, so a scale stops being a line of tree art around a constant. A lift prints under the operand's own name for every slot it binds, and only the slots it reads — attention's two consumers of one carrier now visibly take different states off it, where before both listed all three. And normalization drops operand components no reader reads: rewrites had left an epilogue cone exposing three constants nobody bound, carried alongside a second copy of the loads defining them. A reader is a consuming lift *or* a boundary store, which is how a sweep's per-cell projection reaches its `Write`.

The cut pass also stops offering a seam at a scalar. That piece was a kernel writing three scalars to a workspace so its reader could read them back.

## What broke

`test_sdpa_score_contraction_reaches_the_mma_tier` is red and left red. The mma rows are still offered — the carrier and the score contraction both answer `contracts` and `tiles_whole`, and the carrier answers `chunked` — but the greedy's default pick moved because kernel identity changed and no recorded evidence matches the new term. It needs a tuning round. Not re-recorded, not weakened to pass.

Kernel identity moved for every twisted term, and the seam under the carrier lost a hop (`PLACE@map.1/twist.1/map.1/inner.2/map` → `PLACE@map.1/twist.1/inner.2/map`). Recorded goldens, realization cases, and PLACE pins of that shape are stale.

## Verification

Accuracy against eager passes on SDPA, causal SDPA, softmax, softmax@V and RMSNorm, worst `max_diff` 9.8e-4 in fp16. That is the check that catches a wrong twist. Goldens and the realization corpus are the outstanding gates; both need re-recording on a GPU before this is a deployable reference.

`git diff --stat main -- emmy/**/*.py` is +318 −146. The growth is new capability — a coordinate reading that did not exist, a derived base view, and a normalization rule — set against `_factor_weights`, `_already_held`, and the subtree walk in `Fold.roles` that all came out.
