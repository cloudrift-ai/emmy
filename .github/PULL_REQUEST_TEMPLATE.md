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

Two thirds of the rows recorded in the hardware goldens matched nothing the compiler enumerates, so those cards were resolving from the prior instead of from their own measurements. Neither cause was stale data. Most rows left a knob family unspelled where the enumeration spells it at its off value, and the strict comparison reads a missing family as a mismatch. The rest recorded a split across thread blocks, and the check was asking the pieces that split mints to spell a decision their parent made — which no piece can do, and for which the code already had a rule that nothing called. Sixty-nine rows come back, with the measurements they were recorded with.

| Card | Rows | Dead before | Dead now |
| --- | --- | --- | --- |
| RTX 4090 | 73 | 48 | 10 |
| RTX 5090 | 63 | 38 | 8 |
| RTX PRO 6000 | 10 | 10 | 9 |
| RTX 4080 | 9 | 9 | 9 |

---

## The unspelled off values

42 rows differ from an offered row only by a family they do not spell at all: 38 want `RASTER`, three want `REDUCE`, one wants both `RASTER` and `TILE`. Adding the key is meaning-preserving — an off value is what the row already meant — so the recorded microseconds still describe the schedule they were taken on. The dumper round-trips both files byte for byte, so that commit is 43 inserted lines and nothing else, each key placed where sibling rows already spell it.

## The split rows

A recorded `REDUCE: g2k` names the kernel-set arm at a split fork. The replay resolves it and mints the pieces, and no piece can stamp the `g<n>` it came from — so comparing the recorded row against a leaf asked a piece to spell its parent's decision. It never could: 27 rows across three cards decoded to nothing for that reason alone.

`piece_row` is that rule, already written down — reduce the value to what a piece can still stamp — and the decode simply never called it. It also dropped the key when the whole value was the split, which reads as "free" where an enumerated leaf spells the decided off, so it missed on its own account. Both halves are fixed.

**The second half reaches past the test.** `piece_row` also feeds the evidence index, so a split row that measured a card was joining no kernel at all — invisible to the pick that deploys it.

## What is left

36 rows, with a memo at `plans/golden-row-decode-gaps.md`. On the two cards in scope every one is an attention row, in three classes: nine whose kernels no longer offer a staging family at all, where the spelling is settled but the microseconds need re-taking on the card; six whose f16-accumulate atom is not a candidate for the fused attention target, though the minimized corpus case realizes the same schedule under the same pins; and three not yet diagnosed. The 4080 and the PRO 6000 were outside the scope of this change and are untouched.

No row was re-recorded to make it green.

## Verification

340 passed, 7 skipped, 36 xfailed across the search tests and the lowering guardrail — that covers the golden decode, the evidence index, and the piece-row path both consume. The registry shrank from 105 rows to 36, and every removal is a row the ratchet then demanded pass.

## Line balance

`git diff --stat main -- emmy/` is +55 −5. Forty-three of those lines are recorded rows rather than code. `golden.py` is +12 −5, and its executable half is two lines shorter; the rest is the note explaining why a piece row is what a leaf is compared to.

**Draft.** `make test` has not been run.
