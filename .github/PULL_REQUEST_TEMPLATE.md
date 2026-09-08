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

The realization corpus had no `sm_89` case, so its GPU build and accuracy stages all skipped on an RTX 4090. Four representative rows from the hardware golden now exercise scalar and tensor-core matmul, vectorized pointwise work, and cooperative softmax on that capability. This is a small live-GPU baseline, not full golden qualification.

| Golden row | Distinct schedule path |
| --- | --- |
| f16 matmul | tensor cores with a two-deep async pipeline |
| f32 matmul | scalar tiles with a cross-CTA reduction child |
| ReLU | vectorized, interleaved pointwise loads |
| softmax | cooperative 256-thread reduction |

---

## Scope

The default golden tests decode every hardware row without a GPU. Replaying every sensible schedule would duplicate golden qualification and grow with the schedule space, so the realization corpus remains selective. Its new rule permits a representative baseline only when a capability would otherwise execute no `built` or `correct` nodes.

Recent work is already reflected in the base branch: #742 closed six realization gaps, and #744 through #746 recovered most hardware-golden rows. None of these four cases duplicates an existing exact program and schedule. The eight RTX 4090 rows still marked as decode gaps are attention PV variants; they remain out of this passing baseline.

The cases contain no latency claim or GPU model name. Only their declared compute capability controls whether the live GPU stages run.

## Verification

On the supplied RTX 4090, the selected corpus run passed all 20 nodes. Eight of those nodes compiled or ran the kernels on the GPU.

`make test` passes: 4,400 passed, 1,054 skipped, and 23 xfailed.

`make lint` passes.

`make test-goldens` was not run because no compiler or model golden changed. `git diff --stat main -- emmy/` is empty; this PR adds test data, duration records, and the corpus policy that bounds the new baseline.
