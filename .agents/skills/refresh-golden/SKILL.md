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
| `re-spelled seam PARENT OLD -> NEW` | the seam moved, and the new seam cuts the same pieces, or pieces that read and write alike | where a piece shows as demoted too, compare the fresh cut's operands |
| `dropped decision …` | the fresh parent takes no such decision | record the fresh kernel set (step 3) |
| `dropped kernel NAME: no fresh kernel writes its outputs` | the layer regrouped | trace and record the fresh kernels (step 4) |
| `no kernel survives the fresh lowering` | nothing is left | delete or re-record (step 4) |

When sibling pieces all change identity, the restamp pairs unmatched children by their relative order. A changed
emission order can therefore attach an authored schedule to the wrong child even when counts and freshness pass.
Compare the original and fresh cut operands and declared outputs before recording. Boundary names, dtypes and shapes
help find swaps, but do not prove equal computations. The restamp re-spells a moved seam when the new seam mints the
stored pieces, or, where a piece was re-keyed too, pieces that read the same program inputs and write outputs of the
same dtypes and sizes, one to one, taking the seam that mints the most stored pieces again. That match does not prove
the same cut either, so compare the fresh cut's operands the same way. A route it drops had pieces whose reads or
output sizes changed too, so repair child correspondence explicitly from the original program, then check freshness
again. Never use that correspondence to transfer old timings: evidence
still requires exact kernel identity, bindings, hardware, cache mode, effective precision and schedule.

Commit the rewritten file with the report's counts in the message. Then run the file's nodes:

```bash
./venv/bin/pytest tests/compiler/pipeline/search/test_golden.py -k "<file id>" -n auto --dist=loadgroup
```

The file id is the hardware golden's name (`h100_sm90.json`) or `<recipe>/<name>` for a model golden.

### 3. Measure the file's proposals on its card

The measurement writers refuse a canonical path, so work on a copy. Select both the exact kernel reference and row
name when refreshing an existing schedule, including a cut piece:

```bash
mkdir -p _tune/<run> && cp recipes/<model>/golden/<card>.json _tune/<run>/working.json
EMMY_NVCC_FLAGS= EMMY_KNOBS="PLACE=fuse,<the row's knobs, every family spelled>" \
  emmy run --golden _tune/<run>/working.json --kernel <exact kernel ref> \
    --realization <exact row name> --bench --record-greedy --strict
```

Spell unused schedule families OFF, but do not add a global OFF pin beside a scoped pin of the same family. Spell
each precision lane explicitly, including independent overrides; omitted precision gates inherit `FAST_MATH`.
The named isolated run reconstructs the recorded cut and checks its exact identity. Its correctness reference is
the same-input greedy execution, not eager PyTorch, and its timing never becomes a whole-target latency.

Equivalent aliases may be filled from one successful guarded record only when their complete evidence keys match.
Keep each authored row, preserve unchanged measurements, and verify the recorded schedule and captured DB measurement
against the raw result. The writer may update the first equivalent row rather than the selected alias. Preserve the
writer's isolated median beside its same-run reference. The DB may retain a faster median from an earlier run; verify
its exact key and capture status independently, but do not substitute that median beside a later reference.

Then qualify the containing targets separately with eager PyTorch and `torch.compile`, restoring invalidated
whole-target latencies on working copies. Use `--record` for a measured root or `--record-greedy` for an unmeasured
root seed. An isolated time cannot replace that comparison. Keep capture failures, numerical failures and timeouts
visible, and promote no failed latency. A serving file also needs the strict serving audit in step 5.

For the whole-target check, request `--bench-backends eager,tcompile,emmy --strict --strict-evidence`. Once the
isolated rows are recorded, `--no-record-evidence` keeps that check from replacing their measurements. The
`--record-greedy` writer copies the picked kernel set from the DB and records a root seed's whole-pick latency beside
`torch.compile`; each measured row's `tried` is the number of schedules the DB held for that kernel. Missing
`torch.compile` timing remains a reported gap. A `--tune` run on the card also writes its rows as a file beside the
card's under `search/freezes/`; copy those files back with the working golden (`scp`, as the golden itself comes
back) and commit them from the local checkout, where `git lfs install` has run once (README, "Keep what a tune
measured").

Promote the validated entries through `GoldenFile.edit(path)` with `add_kernel`, `add_routing` and `upsert_row`,
including qualified whole-target latency on its root seed. A canonical file can know the same exact kernel under a
different reference, so copied rows and routing entries must use the reference returned by `add_kernel`. Verify that
`emmy golden check PATH` stays clean, a target compiled with `--strict-evidence --golden PATH` uses the recorded rows,
and `emmy golden list PATH` shows the restored whole-target comparisons. Distinguish proposals already present before
the compiler change from measurements this refresh invalidated.

### 4. Re-record or delete

When the restamp leaves nothing, or drops the kernels a deploy needs (a serving golden's twins):

- **Still a maintained recipe** — the `onboard-model` skill re-traces the inventory on the card, records it
  (`emmy run --golden PATH --bench --record` / `--record-greedy`) and promotes the winners. Start from a fresh
  `emmy trace` inventory; the old file is history, not a seed.
- **A kernel regrouped into a bigger one** (a whole layer fused into one) — the unpinned greedy may hang on it.
  Loop fusion stays maximal, so the fix is a cut, never a smaller region. Find one without scheduling:
  `emmy compile --golden PATH --realization NAME --ir tile --passes dolfstp` under `EMMY_KNOBS="PLACE@<seam>=cut,…"`
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
Keep the prior reproduction gate green (AGENTS.md finalization step 22): a refreshed hardware golden refits both priors
in the same PR; a refreshed recipe golden the shipped priors do not reproduce is refit for or tagged `prior-pending`.

## Report

One row per file in the PR body, plus the compiler change that moved the lowering:

| File | Kernels re-keyed / dropped | Decisions dropped | Rows kept / demoted / dropped | Measured on | Timed vs torch.compile | State |
| --- | --- | --- | --- | --- | --- | --- |
| `h100_sm90.json` | 15 / 8 of 45 | 2 | 22 / 20 / 8 | — | — | proposals await an H100 |

`State` is one of: refreshed, proposals await `<card>`, needs re-record, proposed for deletion, stopped on a loss.
