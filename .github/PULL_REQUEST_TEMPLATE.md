## Abstract

More than half the rows recorded in the goldens matched nothing the compiler enumerates, so the gate that says whether a card's measurements still describe today's compiler was reporting most of them dead. The cause was not stale data. A resolved kernel carries an explicit value for every scheduling decision it declined, and that is the row a recording is taken from; a fork offers its alternatives carrying only the decisions that kernel actually has, so the same schedule is written two ways depending on which side names it. The comparison demanded they match exactly. Reading them as the same schedule when only the declined decisions differ, and re-keying twenty-one rows that name a reduction site by a spelling the compiler retired, brings back 719 rows with the measurements they were recorded with.

### Rows that decode

| golden | rows | before | after |
| --- | ---: | ---: | ---: |
| DeepSeek-V4-Flash (V100) | 279 | 20 | 279 |
| gemma-4-12B-it (5090) | 352 | 126 | 346 |
| gemma-4-12B (5090) | 194 | 24 | 194 |
| gemma-4-12B-it (4090) | 240 | 129 | 197 |
| OLMoE (5090) | 13 | 7 | 9 |
| the five hardware goldens | 224 | 224 | 224 |
| Laguna, Qwen3.5 | 52 | 45 | 45 |
| **total** | **1354** | **575** | **1294** |

---

## Why the two sides disagree

The pipeline stamps a pass's declared OFF values onto the variant at the pass boundary, so a resolved kernel spells every schedule family whether or not it uses one. A fork's offered leaves are read as the codec built them, and the codec keys a family off the sites the kernel has — a kernel with no contraction encodes as work and rasterization alone. Recording reads the resolved kernel. Comparison reads both. So a recorded row could only ever equal the one leaf that happened to be the resolved one, and every other leaf differed from it by anchors that carry no schedule content.

Recording keeps the anchors. A forkless kernel's row **is** its OFF anchors, and dropping them there writes an empty row that spells no decision at all, which strict evidence then rejects. Only the comparison changes.

## Why not require the two spellings to agree instead

That is the other repair, and it has now been made three times: #744 hand-inserted 43 anchor keys into two cards, #745 did the 4080 and the PRO 6000, and #748 stamped anchors onto the leaves of the kernels the hardware goldens exercise. Each worked for the file in front of it. None reached the model goldens, which is where 98% of the corpus lives and which nobody hand-edits — they were still at 575 of 1354 before this change.

The cost of this direction is worth stating: a comparison blind to anchors will no longer notice if the enumeration stops stamping one. That check was never what the golden gate was for, and it belongs with the pass that stamps.

## The re-key

Twenty-one rows address their cross-CTA reduction as `REDUCE@a1` or `REDUCE@a0`. Neither parses today — the site codec moved to a route over the stored tree whose last segment names the arrived-at node by kind, and an axis name is not a kind. Four more spell the family bare where the kernel has several sites.

The value is untouched in every case; each row was matched to the one enumerated row carrying the same families at the same values under a different site path, and a row with more than one such match would have been left alone. None had one. This is a re-key, not a re-measurement.

## What still does not decode

Sixty rows, none of them reachable from a file:

| | rows |
| --- | ---: |
| gemma-4-12B-it (4090), schedule no longer offered | 43 |
| Laguna, stored program will not lower | 6 |
| gemma-4-12B-it (5090) | 6 |
| OLMoE | 4 |
| Qwen3.5 | 1 |

The Laguna six are not a schedule problem. Replaying a structural decision asks the fork's root for its identity, the identity lowers the op's body, and that lowering rejects the exl3 bitcast matmul: `no extent for coordinates ['in17'] — the closed program takes them as axes`. Enumerating the target with the record's spelling removed entirely fails the same way, so retuning cannot reach them either. It is a live crash on a path that is not golden-specific.

The rest record a schedule the enumeration no longer offers — a tile width that shrank from `f8` to `f4`, a warp shape that is still offered but no longer alongside that tile and TMA staging, a kernel that now requires a tiling choice where the recording declined one. Those need the card, and only after someone decides the narrowing was intended.

## Verification

`make test` 4505 passed, 1075 skipped, 12 xfailed. `make lint` clean. Decode counts above are the strict decode over the whole repository corpus, run on this branch and on `origin/main` in a separate worktree; both need no GPU.

A test pins both halves of the new rule: a row decodes with the anchors and without them, and still fails when a decided value is changed to nonsense.

## Line balance

`git diff --stat origin/main -- emmy/` is +39 −13. The executable half is four lines — one projection and three call sites. The rest is the note explaining why the two spellings differ, which is the part that was rediscovered three times.
