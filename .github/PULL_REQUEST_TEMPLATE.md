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

Nothing was checking that the schedules recorded in the hardware goldens still exist. The check was written, but it sat behind a marker the default suite deselects and CI runs only the default suite, so nothing ran it. Turning it on shows two thirds of the recorded rows match nothing the compiler enumerates any more — those cards have been resolving from the prior rather than from their own measurements. The check now asks one test per row instead of one per file, which is fast enough to run every time, names the row that died instead of a count, and gives the dead rows somewhere to be listed so the list can only shrink.

| File | Rows | Equal no enumerated leaf |
| --- | --- | --- |
| rtx4080_sm89 | 9 | 9 |
| rtx4090_sm89 | 73 | 48 |
| rtx5090_sm120 | 63 | 38 |
| rtxpro6000_sm120 | 10 | 10 |
| **total** | **155** | **105** |

---

## Why per row

Parsing one hardware golden costs 0.05 s. Deciding whether one recorded row still equals an enumerated leaf costs about a second. All of the cost is per row, and one node per file spent it on four workers, so the set took 132 s with two files running. 155 nodes spread over all sixteen: about 25 s cold, 8 s warm.

That is cheap enough for the default lane, which is where they now run.

## What it found

105 of 155 rows decode to nothing, every one of them for the same reason: the recorded schedule equals no row the enumeration offers today. `main` fails identically, so this branch did not cause it — it is what the marker was hiding. Plain f32 square matmul is among the failures on every card, so whatever moved was not narrow.

Dating it is separate work, and re-recording needs the four cards.

## The registry

`golden_xfails.yaml` sits beside the test and lists those rows as strict expected failures, so the list can only shrink. Closing a row turns its node red until the line is deleted. A line naming a row the file no longer records fails on its own, with a message saying so — that node deliberately carries no expected-failure mark, because marked, its own failure would be the expected one and the dead line would sit there forever. Both directions were exercised.

The realization corpus states the same rule as a filename suffix. Rows are not files, so this is that rule in the only spelling available to it, including the part that matters most: never add a line to make a red row green.

Labels are the row's name, numbered when a file records that name more than once — three of the four files do, for alternate input pins or alternate schedules of one realization.

## What `make test-goldens` covers now

Model goldens alone. They are an order of magnitude more rows and the widest is a multi-megabyte parse, so they keep one case per file behind the marker. Nothing runs them automatically except the nightly onboarding job, on the models it touches. That hole stays open and is worth its own change.

## Verification

The search tests are 247 passed, 7 skipped, 105 xfailed. `ruff check` and `ruff format --check` pass on the changed test.

`tests/durations.json` gains 142 entries. Several rows cost more than 5 s on a cold derivation memo, and the staleness gate fails the suite until a test that heavy is in the baseline. They are measured from a cold run and merged, rather than regenerating the file, which would have meant running the whole suite for a change that touches one directory.

`git diff --stat main -- emmy/` is empty. This is test infrastructure; nothing in the core moved.

**Draft.** `make test` has not been run.
