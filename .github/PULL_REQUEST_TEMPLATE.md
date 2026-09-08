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

The scalar register tier wrote out the one thing it folds — a multiply, an add, one accumulator per cell — so attention's carrier, which folds three states under their own operations, had no register tile at all. It now replicates the term's own step per cell, which already holds those states and the merge between them, and the algebra it folds stops being the tier's business. That closes the three qwen3-embedding SDPA gaps in the realization corpus; the two remaining SDPA gaps turn out to be stale case files rather than compiler gaps, and closing them empties the corpus of open attention cases.

---

## What the tier folds

`_ScalarOps` emitted `acc__c{i}_{j} += b·a` per cell, with its multiply, its add and its single state fixed in the emitter. A twisted carrier folds a running pivot, a denominator and an expectation, each under its own operation, seeded by the merge between them — so the projection refused every scalar plan on one, and a plan forced through reached the materializer reading cell copies of the pivot and the denominator that nothing declared.

`Fold.step` is that program already, for any algebra. The tier replicates it per cell through `copy_cell`, the replication mechanic the file already had, rebinding only the two operand results to their per-row and per-column reads. A plain contraction gets the same statements it got before, spelled in the term's own product order rather than the emitter's — which is the one thing that moved in `test_scalar_cell_varying_operand`, and it now reads the factor pair as a pair.

The one thing the tier has no place for is an operand past the streamed one that varies over the tile: those are read once, ahead of the K-loop, so the projection offers a scalar plan only when every one of them is uniform. Attention's scale and its mask fills are; a second streamed B is not, and still rides the warp compute fill, which is where the old `len(operands) == 2` test had been sending it.

## The stale half

Two of the five cases were not compiler gaps at all.

`sdpa-hd128-softmax-v-mma` pinned a synchronous `d1/smem` fill. Only the Volta atom resolves that transport — on a target with `cp.async` the blocking vector copy is never offered — so the pin matched no row and the whole enumeration emptied, which reads exactly like a lockout. `sdpa-computed-value-cut-mma` cut at `map.1/twist.3/inner`, a site the tree no longer numbers, so the cut never fired and the carrier kept two nested contractions, leaving the chunk tier unable to say which one supplies the pivot. Both are respellings, and both then realize, build and run.

A third kind of staleness cuts deeper. All five cases carried entries a `COMPLETE=1` run had added while their gap was open, and in four of them one of those entries was an all-off row addressed by identity to the very kernel the case is about. `golden._replay` decides a fork by the entry owning that kernel, so that row outranked the lead and pinned the carrier untiled: the case asserted the opposite of what it was written to assert, and nothing could see it. The corpus's staleness check compares against what regeneration produces, and regeneration reproduces a dead entry unchanged.

**That hole is still open.** `complete` adds an entry for an undecided kernel and never removes one that has gone dead, and no test detects either. Fixing it is its own change.

## What is not here

`built` and `correct` for `sdpa-computed-value-cut-mma`: it declares sm_80 and this card is sm_120, so the corpus skips them, by the rule that a pinned schedule is a claim about one capability.

Nothing measures. A scalar tile on a carrier recomputes the pivot and the denominator once per register column of a row, which is correct and wasteful; the chunk tier's per-row residence is the answer if the row ever ranks. No case in the corpus carries a timing for these, and `tests/perf` is where that question belongs.

## Verification

Offered, realized, built and correct on an RTX 5090 for every closed case the card can run. The GPU-free corpus is 616 passed with the ten remaining gaps (all matmul and fused — no attention case is open now); `tests/compiler/passes` is 571 passed; attention coverage and the schedule-IR tests are green; `make lint` passes.

`git diff --stat main -- emmy/` is +43 −28, of which +17 −11 is the architecture note. The code is +32 −22, one line of it an import. The growth is the capability: a scalar tile on a carrier that had none.

**Draft.** `make test`, `make test-goldens` and `make test-durations` have not been run. The durations gate already names three tests over five seconds that are missing from `tests/durations.json`.
