---
name: refresh-golden
description: >-
  Refresh one repository golden file, several, or all of them after a compiler change left them stale: a stale
  golden's stored kernels are no longer what a fresh lowering of its own programs writes. Use when `emmy golden
  check` or the fresh-lowering test in `make test` names a golden, or when asked to restamp, re-measure, re-record or
  delete a golden. Works file by file: restamps with the CLI where
  it can, sends what needs a card to a record run, and leaves deletion to the author.
---

# Refresh goldens

The unit of work is one golden file. The input is a golden path, a list of them, or nothing, which means every
repository golden — the same argument shape `emmy golden check` and `emmy golden restamp` take. A repository golden is
a hardware golden under `emmy/compiler/pipeline/search/golden/records/` or a recipe's
`recipes/<model>/golden/<card>.json`. It is the tune DB's shape — `kernels`, `routing` and `rows`, beside the traced
`programs` the kernels came from — and `make test` asks one question of it, per traced program and without a card:

**Is the file the fresh lowering of its own programs?** A restamp of it must change nothing: every stored kernel is
the kernel the current compiler lowers the program to (the stored body and the fresh one compute the same exact
identity), and every kernel-set decision is one the fresh parent takes the same way. A golden stores no identity and
no stamps — both are computed from the stored body — so a change to how identity is computed makes no file stale.

A file failing that is *stale*: its rows are rows of kernels a deploy no longer builds, so a strict boot finds no
evidence. The check has no list of expected failures: a red node stays red until the file is refreshed. Never
re-record a row to make it green.

## Inputs to confirm

- **Which files.** A path, a recipe (all its cards), or all repository goldens. With no argument, start from
  `emmy golden check` with no argument: it names every stale file, and a file it passes needs nothing.
- **Which cards are reachable.** Step 3 runs on the golden's exact card (`gpu_name` and `compute_cap` in the file
  head). Without it, the file stops at step 2 with proposals, and the report says so.
- **Whether deletion is on the table.** Step 4 proposes it; the author decides.

## Per file

Run the steps below on one file at a time, and finish each file — committed, its test nodes run, its lines in the
two lists removed — before starting the next. Refreshing every golden is this loop over the list `emmy golden check`
prints, cheapest first: hardware goldens, then recipe goldens by file size; the Qwen3.8 and DeepSeek V100 files lower
whole-layer traces and take minutes each. Group step 3 by card so one rental measures every file that needs it.

### 1. Measure the drift

```bash
emmy golden check recipes/<model>/golden/<card>.json
```

Read the reason on each line:

- `re-keyed NAME old -> new` — the kernel still exists (same outputs), but its stored body and the fresh lowering are
  two kernels: their exact identities (`old`, `new`) differ. A restamp gives it the fresh body under the same name in
  the file; each of its rows survives and loses its measurement.
- `dropped kernel NAME: no fresh kernel writes its outputs` — the layer regrouped (fused into a neighbour, split).
  The kernel and its rows describe no kernel; a restamp drops them, and the fresh kernels are unrecorded inventory.
- `dropped decision PARENT ARM: the fresh parent does not take it the same way` — the seam the arm names moved (a
  node re-read as another kind), or the cut mints another number of pieces. A restamp drops the decision with its
  pieces' rows.
- `demoted to a proposal NAME` — a measured row whose kernel was re-keyed: the schedule stays, the microseconds go.

To see what moved in one program, compare the stored body (the kernel's `loop_ir` entry in the file) with the fresh
one:

```bash
emmy compile --golden recipes/<model>/golden/<card>.json --program <N> --ir loop
```

Name the compiler change that moved the lowering (`git log` over `emmy/compiler/pipeline/passes/`) and say whether it
is a gain, a re-spelling or a loss before touching the file. A loss — a kernel the compiler used to fuse and no longer
does — is a regression to report, not a golden to refresh; stop there for every file it explains.

### 2. Restamp what the CLI can

```bash
emmy golden restamp recipes/<model>/golden/<card>.json    # rewrites the file in place; no card needed
```

It lowers every traced program again, gives each kernel the body the fresh lowering gives it, takes every kernel-set
decision again on the fresh parent, and keeps each row on its kernel. A row keeps its measurement only while its
kernel kept its exact identity; otherwise it keeps its schedule and loses its microseconds: a *proposal*, no evidence
until measured again. It refuses to write a file nothing survives in. Read its report:

| Report line | Meaning | Next |
| --- | --- | --- |
| `demoted to a proposal NAME` | the kernel changed under the row | measure it again on the card (step 3) |
| `dropped decision …` | the fresh parent takes no such decision | record the fresh kernel set (step 3) |
| `dropped kernel NAME: no fresh kernel writes its outputs` | the layer regrouped | trace and record the fresh kernels (step 4) |
| `no kernel survives the fresh lowering` | nothing is left | delete or re-record (step 4) |

Commit the rewritten file with the report's counts in the message. Then run the file's nodes:

```bash
./venv/bin/pytest tests/compiler/pipeline/search/test_golden.py -k "<file id>" -n auto --dist=loadgroup
```

The file id is the hardware golden's name (`h100_sm90.json`) or `<recipe>/<name>` for a model golden.

### 3. Measure the file's proposals on its card

The measurement writers refuse a canonical path, so work on a copy; one copy per file, one run per proposal row:

```bash
mkdir -p _tune/<run> && cp recipes/<model>/golden/<card>.json _tune/<run>/working.json
EMMY_NVCC_FLAGS= EMMY_KNOBS="PLACE=fuse,<the row's knobs, every family spelled>" \
  emmy run --golden _tune/<run>/working.json --realization <exact row name> --bench --record-greedy
```

Under `EMMY_KNOBS` the greedy pick is the pin, so `--record-greedy` writes that pick's kernel set as the DB holds it:
a routing row per decision and a measured row per kernel, named `<name>.<identity12>` — or onto the proposal itself
when the proposal already names that kernel, those sizes, that regime and that schedule, which is the usual case.
Pin the placement too: a plain row spells the fused kernel by carrying no `PLACE` key, and a proposal is no evidence,
so without `PLACE=fuse` the prior decides the cut fork and can record a two-kernel set under the row's name (a row on
a piece replays under its routing row's `PLACE@seam=cut` instead; `--pin-route` spells that). `--record` alone writes
a per-card `latency` block, which is not what a model golden's rows are read by. One run per precision lane, each
spelled explicitly: `EMMY_FAST_MATH=1` for the fast-math row and `EMMY_FAST_MATH=0` for the standard row. Fast math
is the default, so a standard row recorded with the variable unset measures the fast-math kernel under the standard
row's name. Then promote: copy the measured rows — and any kernel or routing row the pick added — from the working
file into the canonical one (`GoldenFile.edit(path)` with `add_kernel`, `add_routing`, `upsert_row`), so the diff is
only those entries. A row and a routing row name a kernel by its `ref` in the file, and the same kernel can be known
by another `ref` in the canonical file, so point each copied entry at the `ref` of the kernel `add_kernel` returns.
Prove it: `emmy golden check PATH` stays clean, and a `--strict-evidence` compile of the target
with `--golden PATH` picks the row.

### 4. Re-record or delete

When the restamp leaves nothing, or drops the kernels a deploy needs (a serving golden's twins):

- **Still a maintained recipe** — the `onboard-model` skill re-traces the inventory on the card, records it
  (`emmy run --golden PATH --bench --record` / `--record-greedy`) and promotes the winners. Start from a fresh
  `emmy trace` inventory; the old file is history, not a seed.
- **A kernel regrouped into a bigger one** (a whole layer fused into one) — the unpinned greedy may hang on it.
  Loop fusion stays maximal, so the fix is a cut, never a smaller region. Find one without scheduling:
  `emmy compile --golden PATH --realization NAME --ir tile --passes dolfnstp` under `EMMY_KNOBS="PLACE@<seam>=cut,…"`
  prints the unscheduled kernel set in seconds, with the seam spellings the cut pass accepts. Cut first where a
  piece's value is recomputed under consumer axes it does not read, judged from the realized piece's grid (a seam's
  raw axes omit the strides the cut applies). Add cuts one at a time; more cuts are not faster by themselves. Then
  record the kernel set with `--record-greedy` under its pins, and for multi-millisecond pieces pass
  `--warmup 2 --iters 5` so the isolated re-bench stays under the GPU cap.
- **No longer relevant** (an experiment's card, a retired quantization) — `git rm` the file and its nodes' entries in
  `tests/durations.json`, and say in the PR what evidence went with it. That call is the author's; propose it, do not
  make it.

### 5. Close the file

The file is refreshed when `emmy golden check PATH` passes and every proposal the restamp left was measured again or
dropped on purpose. For a serving golden the release gate is the strict audit on the card,
`emmy eval golden --golden PATH --serving-config <models/slug.env>`; run it before calling the file refreshed.
`make test`'s check does not replace it: that check has passed while a serving golden's twins refused strict
evidence at a cut fork (DeepSeek V4, after #981 and again after #1003). It proves the file is the fresh lowering,
not that every fork a deploy meets has a measured arm.
Nightly refresh owns prior refits after repository goldens change; leave the weights out of the golden-refresh PR.

## Report

One row per file in the PR body, plus the compiler change that moved the lowering:

| File | Kernels re-keyed / dropped | Decisions dropped | Rows kept / demoted / dropped | Measured on | State |
| --- | --- | --- | --- | --- | --- |
| `h100_sm90.json` | 15 / 8 of 45 | 2 | 22 / 20 / 8 | — | proposals await an H100 |

`State` is one of: refreshed, proposals await `<card>`, needs re-record, proposed for deletion, stopped on a loss.
