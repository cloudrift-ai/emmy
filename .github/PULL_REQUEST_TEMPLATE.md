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

Two more recorded rows come back, and the four the memo could not explain now have named causes. What matters more than the count is that the remaining eleven are all one kind of thing: every one is blocked by a compiler behaviour someone can go and read, and none of them is a stale spelling or a missing measurement. That was not true when this started, when two thirds of every card's rows simply matched nothing and there was no way to tell a codec drift from a lost capability.

| Card | Rows | Dead before | Dead now |
| --- | --- | --- | --- |
| RTX 4090 | 73 | 10 | 8 |
| RTX 5090 | 63 | 3 | 3 |

---

## The two measured rows

Same repair as the five closed on the 5090: they pin `STAGE: d1/smem` where their kernels expose no staging family, so the key names a decision that does not exist and the stored microseconds do not survive dropping it. Re-benched at O3 on the card as pinned rows — a measurement, not a search.

    attention.hd128.dynM.pv  39.23 us against a 39.12 us greedy reference
    attention.hd64.dynM.pv   15.22 us against 15.01 us

Both land on the schedule the greedy pick takes, as all five did on the 5090.

## An inference that became a measurement

`attention.hd128.pv` was the third row of that class on paper. On the card both its lanes refuse to compile, with exactly the message the memo predicted from its spelling: an atomic cross-CTA reduce folds one additive state component and attention's carrier has three. It moves out of the staging class and into that blocker, where it belongs.

## The rows the memo called undiagnosed

Both now answer. `attention.hd256.dynM.pv` is a chunk-tier refusal that states itself — *the chunk tier reads its score operands and its streamed value as slabs* — so at that head dimension the warp atoms are never projected. `attention.hd64.pv` is the split doing it: its node refusal is `None`, the tier is willing, but the row spells a `g4k` split and what enumerates is the pieces, which offer only scalar tiles. `attention.hd64.dynM.pv#1` spells no split and keeps its tensor-core rows, which is the controlled comparison.

That makes `hd64.pv` the same open question as the `g4a` dead end already in the memo: why a cross-CTA split mints pieces the tensor-core tier does not serve.

## What is left

Eleven rows behind three named behaviours, in `plans/golden-row-decode-gaps.md`: the chunk tier never offering the reduced accumulator, the atomic cross-CTA reduce refusing a multi-component carrier, and the tensor-core tier being absent from two targets. Four rows carry two blockers at once. The memo also records the four dead ends walked so far, and what the 4090 host needs — nvcc is installed there but not on PATH, and emmy has no NVRTC fallback, so a bench dies until `CUDA_HOME` is set.

No row was re-recorded to make it green.

## Verification

144 passed, 7 skipped, 11 xfailed on the golden decode. The registry shrank by two, and each removal is a row the ratchet then demanded pass.

`make test` passes: 4,388 passed, 1,042 skipped, and 23 xfailed. The default lane now records every test taking at least 1 s in its output, and the CI-only 5.9 s cold-start row that failed the original run is in `tests/durations.json`.

`make lint` passes.

`make test-goldens` remains red on seven pre-existing model-golden files: Laguna, OLMoE, Qwen3.5, DeepSeek V4, `gemma-4-12B`, and the two `gemma-4-12B-it` card files. This PR changes no model golden or compiler code.

`git diff --stat main -- emmy/` is +6 −8 — two rows re-measured and their dead staging key dropped. No compiler code changed.
