---
name: refresh-golden
description: >-
  Refresh one repository golden file, several, or all of them after a compiler change left them stale: a stale
  golden's stored targets are no longer what a fresh lowering of its own programs writes. Use when `emmy golden
  check` or the fresh-lowering test in `make test` names a golden, or when asked to restamp, re-measure, re-record or
  delete a golden. Works file by file: restamps with the CLI where
  it can, sends what needs a card to a record run, and leaves deletion to the author.
---

# Refresh goldens

The unit of work is one golden file. The input is a golden path, a list of them, or nothing, which means every
repository golden — the same argument shape `emmy golden check` and `emmy golden restamp` take. A repository golden is
a hardware golden under `emmy/compiler/pipeline/search/golden/` or a recipe's `recipes/<model>/golden/<card>.json`,
and it answers two questions in `make test`, both without a card:

1. **Does every row still decode?** Each recorded schedule row must equal an enumerated leaf of its stored kernel.
2. **Are the stored targets the fresh lowering?** Each stored kernel must be what the current compiler lowers the
   golden's own traced program to, byte for byte.

A file failing the second is *stale*: its rows decode, yet a deploy builds kernels none of them describe, so a strict
boot finds no evidence. Neither check has a list of expected failures: a red node stays red until the file is
refreshed. Never re-record a row to make it green.

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

Read the reason on each stale target:

- `the fresh kernel's Loop IR differs` — the kernel still exists (same outputs) with another body. A restamp carries
  the target; each row's measurement survives only if the kernel renders the same CUDA source.
- `no fresh kernel writes its outputs` — the layer regrouped (fused into a neighbour, split). The target and its rows
  describe no kernel; a restamp drops them, and the fresh kernels are unrecorded inventory.

To see what moved, diff the two pools of one program, the stored one and the fresh one:

```bash
emmy golden kernels recipes/<model>/golden/<card>.json --program <N> > stored.json
emmy compile --golden recipes/<model>/golden/<card>.json --program <N> --ir loop -o fresh.json
diff stored.json fresh.json
```

Name the compiler change that moved the lowering (`git log` over `emmy/compiler/pipeline/passes/`) and say whether it
is a gain, a re-spelling or a loss before touching the file. A loss — a kernel the compiler used to fuse and no longer
does — is a regression to report, not a golden to refresh; stop there for every file it explains.

### 2. Restamp what the CLI can

```bash
emmy golden restamp recipes/<model>/golden/<card>.json    # rewrites the file in place; no card needed
```

It replaces each stored target with the fresh Loop IR, re-keys a row naming the target to the fresh target (a row naming
a cut or split piece keeps its identity and survives while the fresh set still mints that piece), drops rows that no
longer decode, and keeps a measurement only when the row's kernel renders the same CUDA source from the fresh Loop IR.
Otherwise the row keeps its schedule and loses its microseconds: a *proposal*, no evidence until measured again. It
refuses to write a file nothing survives in. Read its report:

| Report line | Meaning | Next |
| --- | --- | --- |
| `demoted to a proposal NAME` | the kernel changed under the row | measure it again on the card (step 3) |
| `dropped row NAME: …` | the row equals no leaf of the fresh kernel | record a schedule of the fresh kernel (step 3) |
| `dropped target NAME: no fresh kernel writes its outputs` | the layer regrouped | trace and record the fresh kernels (step 4) |
| `no target survives the fresh lowering` | nothing is left | delete or re-record (step 4) |

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

Under `EMMY_KNOBS` the greedy pick is the pin, so `--record-greedy` appends a measured receipt of that row's kernel
set. Pin the placement too: a plain row spells the fused kernel by carrying no `PLACE` key, and a proposal is no
evidence, so without `PLACE=fuse` the prior decides the cut fork and can record a two-kernel set under the row's name
(a route row carries its own `PLACE@seam=cut` keys instead). The receipt is named `<name>.<identity12>`, or lands on
the row itself when the row already carries the identity, with the `measurements` a strict compile reads. `--record`
alone writes a per-card `latency:` block, which is not what a model golden's rows are read by. One run per precision
lane, each spelled explicitly: `EMMY_FAST_MATH=1` for the fast-math row and `EMMY_FAST_MATH=0` for the standard row.
Fast math is the default, so a standard row recorded with the variable unset measures the fast-math kernel under the
standard row's name. Then promote: move the receipt's `measurements`, `knobs` and
`identity` onto the proposal row in the canonical file, or keep the receipt and drop the proposal, and write through
`dump_golden_file(..., validation=REPOSITORY)` so the diff is only the rows. Prove it: `emmy golden check PATH` stays
clean, the row's test node decodes, and a `--strict-evidence` compile of the target with `--golden PATH` picks the
row.

### 4. Re-record or delete

When the restamp leaves nothing, or drops the targets a deploy needs (a serving golden's twins):

- **Still a maintained recipe** — the `onboard-model` skill re-traces and records the inventory on the card, and the
  `tune-kernels` skill tunes and promotes the winners. Start from a fresh `emmy trace` inventory; the old file is
  history, not a seed.
- **A target regrouped into a bigger kernel** (a whole layer fused into one) — the unpinned greedy may hang on it.
  Loop fusion stays maximal, so the fix is a cut route, never a smaller region. Find one without scheduling:
  `emmy compile --golden PATH --realization NAME --ir tile --passes dolfnstp` under `EMMY_KNOBS="PLACE@<seam>=cut,…"`
  prints the unscheduled kernel set in seconds, with the seam spellings the cut pass accepts. Cut first where a
  piece's value is recomputed under consumer axes it does not read, judged from the realized piece's grid (a seam's
  raw axes omit the strides the cut applies). Add cuts one at a time; more cuts are not faster by themselves. Then
  record the route with `--record-greedy` under its pins, and for multi-millisecond pieces pass `--warmup 2 --iters 5`
  so the isolated re-bench stays under the GPU cap.
- **No longer relevant** (an experiment's card, a retired quantization) — `git rm` the file and its nodes' entries in
  `tests/durations.json`, and say in the PR what evidence went with it. That call is the author's; propose it, do not
  make it.

### 5. Close the file

The file is refreshed when `emmy golden check PATH` passes and every one of its rows decodes. For a serving golden the
release gate is the strict audit on the card,
`emmy eval golden --golden PATH --serving-config <models/slug.env>`; run it before calling the file refreshed.

## Report

One row per file in the PR body, plus the compiler change that moved the lowering:

| File | Targets restamped / dropped | Rows kept / demoted / dropped | Measured on | State |
| --- | --- | --- | --- | --- |
| `h100_sm90.json` | 15 / 8 of 45 | 22 / 20 / 8 | — | proposals await an H100 |

`State` is one of: refreshed, proposals await `<card>`, needs re-record, proposed for deletion, stopped on a loss.
