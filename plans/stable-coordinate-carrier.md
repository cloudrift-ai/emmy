# Stable-coordinate twisted carrier — state and what is left

Branch `feature/scalar-operand-inline`, draft PR #738, rebased on `main` at `c59212fa9`.

## What the change is

A twisted fold stored its per-element contribution in BASE coordinates (`λ_B`, softmax's
`(s, exp s, exp(s)·v)`), so every attention tree carried an `exp` of an unshifted score — a value that overflows and
that nothing was permitted to evaluate. The paper states both representations as equivalent (`design.tex:583`) and
Figure 4 draws the stable one; the implementation had taken the base one.

The term now stores `λ_S` — the recipe's authored injection, `(score, 1, value)` — with `e_S` as `init`, `κ_S` derived
as `combine`, and ψ / `⊕_B` as helpers. `Fold.based` is the new derived reading (ψ⁻¹ over the stored lift) that
restores the base form for recognition only. Bilinearity lives in base coordinates and nowhere else, so
`bilinear_channels` and `as_contraction` read `based` and nothing emits it.

`_factor_weights` and `_already_held` are gone with it: nothing needs the weight reified into an operand, because A is
now what `operands[0]` SUPPLIES rather than what it exposes.

## Verified

Accuracy against eager on SDPA, causal SDPA, softmax, softmax@V and RMSNorm — worst `max_diff` 9.8e-4 in fp16. The
numerics of the coordinate change are sound.

Green: `tests/compiler/e2e/test_attention_coverage.py` (23 passed), `tests/compiler/passes/test_twisted_rewrite.py`,
`test_cut_forks.py`, `test_schedule_walk.py`.

## The chunk tier: closed

Two gates in sequence, both now fixed.

The atom projection (`ir/schedule/classic_projection.py`, `_node_refusal`) refused a contraction any of whose operands
reduce. A chunked carrier's A IS the score contraction, so it reduces by design; the fragment agreement in `extend`
holds the two to one atom. With the exemption the twist site's TILE domain goes from 1 choice to 62.

The emitter (`pipeline/passes/lowering/kernel/_atom.py`, the chunk tier) was written against the cone the fusion no
longer mints: it reached into `A.operands` for the row-invariant scale and expected `roles[0]` to be the fragment the
score operand's own result keyed. Now A is the contraction — the operand exposes the RAW score, `roles[0]` is the
SCALED one the carrier's lift computes. It seeds the producer's result and evaluates the carrier's own prefix (the
cone of `roles[0]` in the lift) onto that fragment, taking the scale from the prefix's own leaves. `_chunk_refusal`
reads the same prefix and refuses a leaf that varies over the chunk.

## Strict evidence at a cut fork

Nothing was wrong in the evidence pick. A case's further entries had gone stale in the one way the corpus cannot
see — see the note under Outstanding — so the residual kernel after a cut had no row and its own `030_cut` fork
refused. `make test-corpus-regen COMPLETE=1` gives it one.

## The bilinear product's factor names

`bilinear_channels` walks `based`, which is spelled over `applied` — every operand-bound param replaced by the
operand's own result — while its name maps stayed keyed by the term's params. A param normally carries its operand's
name, so the two agreed by accident and nothing showed it. A cut renames the operand's result to its workspace, the
accident ends, and B resolves to nothing: the channel drops, the score stops reading as a contraction, and the cut
consumer loses the tile catalog at the score and at the twist above it. Keyed by the applied name now.

## The cut workspace dtype

A cut operand seam materializes at the consuming contraction's store dtype — what the fused slab would have held.
With A a reducing edge that rule started applying to the score, and the cut wrote f16 scores. The exception is a
zero-axis cone's; a reducing operand keeps the f32 carrier.

## Outstanding

- Realization corpus: 2 failed / 1007 passed. The 32 stale identities are restamped, every stale pin path
  (`…/twist.1/map.1/inner` → `…/twist.1/inner`, ten cases) is respelled, and nine cases got the entry their cut set
  was missing. No `_xfail_` gap closed at any step. What is left is not this branch's:
  - `reduce/rms-norm-cut-sweep-work.yaml` (correct) — `ConstantOp 'p_weight' has no value and was not supplied`.
    Fails on `main` too.
  - `matmul/nvfp4-w4a4-packed-slab-loopify.yaml` (built) loses its xdist worker to a native crash at `-n 8` over the
    whole corpus; it passes alone and at `-n 8` over the packed cases. Not a verdict.
- A corpus case's non-target entries address their kernel by stored IDENTITY, and `regenerate` restamps only the
  target's. A change that moves a PIECE's identity therefore leaves them addressing nothing, with no detection —
  the staleness test compares against what `regenerate` produces, and `regenerate` reproduces the same stale
  identity. `COMPLETE=1` adds the missing entry but never removes the dead one, so those files still carry entries
  for kernels their case no longer compiles to. Worth a look on its own.
- Kernel identity moved for every twisted term, so recorded goldens are stale. Needs a tuning round on a GPU.
- `make test` has not been run since the two fixes above.
- `tests/durations.json` is missing `test_chunk_tier_folds_the_carrier_on_tensor_cores[128]` (7.3 s) — the case only
  started running with the emitter fix. `make test-durations` at finalization.
- COLD-DEPLOY LOTTERY, not a regression to chase in this branch: greedy's pick for a fused attention pool is decided
  by ONE seeded draw out of ~10^11 rows (`_descent_sample`, `_POOL_DESCENT_WORK`), because the fused pool's descent
  bound is ~31k. On a 32-key f16 SDPA `main` draws a matching mma pair and this branch draws the untiled row; the
  schedule space changed, so the seed did. The chunk tier itself realizes — the pinned rows prove it — and every
  gate accepts the mma rows. `test_sdpa_score_contraction_reaches_the_mma_tier` passes on this branch only because
  a cut piece reaches the tensor cores, not because the carrier folds on them.

## Do not

Re-record goldens or corpus verdicts to make the suite green. The corpus's stale half is mechanical; a changed verdict
is a finding. And do not loosen mma guards to chase the chunk tier — a guard loosened blind produces a wrong kernel
rather than a slow one.
