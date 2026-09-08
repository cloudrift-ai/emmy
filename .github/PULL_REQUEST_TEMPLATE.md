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

Two tiers each hardcoded one shape of the thing they fold, and attention paid for both. The scalar register tier wrote out a multiply, an add and one accumulator per cell, so a carrier that folds three states under their own operations had no register tile at all; the chunk tier built its score with an inner `mma`, so a carrier whose score arrives already stored had no tensor-core row. Each now takes that shape from the term instead: the scalar tier replicates the term's own step per cell, and the chunk tier gathers its score fragments when nothing computes them. Every open attention case in the realization corpus closes — three on the first change, two on the second, and two more that turned out to be stale case files rather than compiler gaps.

---

## What the scalar tier folds

`_ScalarOps` emitted `acc__c{i}_{j} += b·a` per cell, with its multiply, its add and its single state fixed in the emitter. A twisted carrier folds a running pivot, a denominator and an expectation, each under its own operation, seeded by the merge between them — so the projection refused every scalar plan on one, and a plan forced through reached the materializer reading cell copies of the pivot and the denominator that nothing declared.

`Fold.step` is that program already, for any algebra. The tier replicates it per cell through `copy_cell`, the replication mechanic the file already had, rebinding only the two operand results to their per-row and per-column reads. A plain contraction gets the same statements it got before, spelled in the term's own product order rather than the emitter's — which is the one thing that moved in `test_scalar_cell_varying_operand`, and it now reads the factor pair as a pair.

The one thing the tier has no place for is an operand past the streamed one that varies over the tile: those are read once, ahead of the K-loop, so the projection offers a scalar plan only when every one of them is uniform. Attention's scale and its mask fills are; a second streamed B is not, and still rides the warp compute fill, which is where the old `len(operands) == 2` test had been sending it.

## A chunk whose score is read

The chunk tier's score is the nested contraction the tree carries as a site of its own. Softmax@V has no such site — the probabilities are an input — so `ContractionFacts.producer` was empty, the tier refused, and no tensor-core row was offered for the value channel at all.

Such a carrier now gathers each `(row, chunk)` C fragment from the stored tile at the fragment's own lane map. **Nothing new was needed to do it.** A role-`c` `RegFragment` declares zero, and `FragmentBiasAdd` already reads gmem at exactly that map and adds — so one of those per fragment IS the fragment. It had been in the tree without a caller; giving it one also gave it the `rewrite` handler it had never needed, which is what the first `emmy run` over the new kernel found. Everything above the score — the row reduce, the channel patterns, the repack into an A operand — reads the same C fragments either way.

Both coordinates read wrapped in-bounds. An overhanging row is discarded by the store guard and an overhanging column refilled with the pivot's identity by the boundary mask the tier already emits, exactly as the contracted form's clamped reads are.

## The stale half

Two of the seven cases were not compiler gaps at all.

`sdpa-hd128-softmax-v-mma` pinned a synchronous `d1/smem` fill. Only the Volta atom resolves that transport — on a target with `cp.async` the blocking vector copy is never offered — so the pin matched no row and the whole enumeration emptied, which reads exactly like a lockout. `sdpa-computed-value-cut-mma` cut at `map.1/twist.3/inner`, a site the tree no longer numbers, so the cut never fired and the carrier kept two nested contractions, leaving the chunk tier unable to say which one supplies the pivot. Both are respellings, and both then realize, build and run.

A third kind of staleness cuts deeper. All seven carried entries a `COMPLETE=1` run had added while their gap was open, and in six of them one of those was an all-off row addressed by identity to the very kernel the case is about. `golden._replay` decides a fork by the entry owning that kernel, so that row outranked the lead and pinned the carrier untiled: the case asserted the opposite of what it was written to assert, and nothing could see it. The corpus's staleness check compares against what regeneration produces, and regeneration reproduces a dead entry unchanged.

**That hole is still open.** `complete` adds an entry for an undecided kernel and never removes one that has gone dead, and no test detects either. Fixing it is its own change.

## What is not here

`built` and `correct` for `sdpa-computed-value-cut-mma`: it declares sm_80 and this card is sm_120, so the corpus skips them, by the rule that a pinned schedule is a claim about one capability.

The two softmax@V rows are slow. Re-measured on the card they are 33.4 us (narrow tile) and 84.3 us (wide) against 14.3 us for torch.compile — the stored numbers, 36.9 and 58.0, belonged to a row the compiler could not realize when they were taken, so they are re-recorded rather than kept. Being 2.3x behind torch on a shape that now reaches the tensor cores at all is a code-generation finding; the corpus deliberately does not record one, and `tests/perf` is where it belongs.

A scalar tile on a carrier recomputes the pivot and the denominator once per register column of a row, which is correct and wasteful. The chunk tier's per-row residence is the answer if such a row ever ranks.

## Verification

Offered, realized, built and correct on an RTX 5090 for all seven closed cases, minus the sm_80 skip above. No `_xfail_` case in the corpus is attention any more — the eight left are matmul and fused. The GPU-free corpus plus attention coverage is 643 passed; `tests/compiler/passes` is 571 passed; the schedule-IR tests are green; `make lint` passes.

`tests/durations.json` needed both halves of a rename: thirteen ghost entries for node ids the closures retired, and entries for the stages that now run — the hd128 case's `correct` is 59 s and its `realized` 8 s, both over the staleness gate. One pre-existing hole, `test_depthwise_conv1d_matches_eager[cuda]@cuda` at 7 s, was already failing that gate and is recorded too. Measured under `-n auto --dist=loadgroup` so the ids carry the group tags `make test` gives them.

`git diff --stat main -- emmy/` is +137 −44, of which +17 −11 is the architecture note. The growth is the two capabilities: a scalar tile on a carrier that had none, and a chunk whose score is read.

**Draft.** `make test` and `make test-goldens` have not been run.
