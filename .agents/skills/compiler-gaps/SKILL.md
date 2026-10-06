---
name: compiler-gaps
description: >-
  Find and explain where the compiler underperforms: kernels slower than torch.compile however they are scheduled,
  kernels slower only because nobody searched them, schedules the prior picks badly, and realizations that do not
  build or give a wrong answer. Use when asked for an overview of compiler performance, for underperformers, gaps,
  bad prior picks or broken realizations, or to read the nightly "Compiler gaps" summary.
---

# Compiler gaps

Three listings hold the data. None of them judges anything; this skill does, and writes the report.

| Question | Listing | Reads |
| --- | --- | --- |
| Which rows are slower than `torch.compile`? | `emmy golden list` | goldens and realization corpus cases |
| Which schedules does the prior pick badly? | `emmy eval prior DATASET --json PATH` | a dataset exported from the goldens |
| Which realizations are broken? | the corpus cases named `*_xfail_<stage>.json` | `tests/compiler/realization/cases/` |

## 1. Rows behind torch.compile

```bash
emmy golden list --behind                                      # every repository golden
emmy golden list tests/compiler/realization/cases --behind     # the realization corpus
emmy golden list recipes/<model>/golden/<card>.json --json -   # one file, as JSON
```

Each line is one measured row on one card: the kernel's time (`emmy_us`) and the reference its bench took beside it,
`tried` (how many schedules of that kernel the card had measured when the row was recorded), and — where a record run
timed the whole row — `whole_us` beside `tcompile_us`, with `x_tc` their ratio. Rows are sorted by `x_tc`, slowest
first. `--gpu` and `--kernel` narrow by substring. A row without `tcompile_us` was recorded before record runs timed
`torch.compile`; it is no evidence either way. `emmy golden list PATH --missing` names what such a file still needs.
The nightly golden fill measures it for the hardware goldens, one rentable card a night; a model golden needs a record
run (the `refresh-golden` skill, step 3).

Read each row behind `torch.compile` as one of two things:

- **A compiler gap** — `tried` is large: the schedule space was searched and nothing reached `torch.compile`. The fix
  is a cut, a schedule the space does not offer yet, or lowering coverage. Loop fusion is maximal and is never the
  fix (AGENTS.md).
- **A search gap** — `tried` is small or absent: the row is the best of a few. Sweep it first (`emmy run --golden
  PATH --realization NAME --bench --ab "<knobs>"`, one `--ab` per candidate), then record the winner. Call it a
  compiler gap only after the sweep.

Before calling either, check the number:

- **The replay floor.** On the dev-box RTX 5090 a graph replay steps in about 2.05 µs, so a `tcompile_us` near 2 µs is
  the floor, not the kernel. A ratio over it says the Emmy row is slow; it does not say by how much.
- **Corpus rows pin a schedule.** A corpus case measures the schedule it names, which need not be the best one. A
  corpus row behind `torch.compile` is a lead; the golden row of the same kernel, or a sweep, settles it.
- **`whole_us` is the whole row.** For a cut it covers every piece; the slow piece is the one to look at
  (`emmy run --golden PATH --realization NAME --bench` prints per-kernel times).

When a row is understood, write one plain sentence about it into its `note` (`GoldenFile.edit(path)` and
`dataclasses.replace(row, note=...)`) and promote it with the file: what was found, such as "load-bound: the K loop
stalls on scalar global loads". Never write a label the numbers decide ("gap", "slow"): the listing recomputes those
every time it runs.

## 2. Bad prior picks

```bash
emmy db import --db _data/gaps.db --fresh --repository
emmy db export --db _data/gaps.db _data/gaps --space schedule      # or --space placement
emmy eval prior _data/gaps --json _data/eval.json
jq '.pools[] | select(.regret != null and .regret > 1.05)' _data/eval.json
```

`eval prior` re-decides every golden pool with no measurement in scope. Each entry of `pools` in the JSON is one pool:
the prior's `pick`, the closest `golden` row, and — when a golden row of the pool is exactly the pick — `pick_us`
over the pool's `best_us` as `regret`. The table prints `unmeasured` where no row is the pick.

- **`regret` above 1** — the prior picks a measured schedule that loses. That is a bad pick. The nightly refit owns
  the fix; do not refit in a PR, name the pools instead.
- **`unmeasured`** — the prior picks a schedule nobody measured, so its cost is unknown. Measure it the way any row is
  recorded: `EMMY_KNOBS="PLACE=fuse,<the pick's knobs>" emmy run --golden <working copy> --realization <row> --bench
  --record-greedy`, then promote the row (the `refresh-golden` skill, step 3). The golden then holds the losing
  schedule beside the winner. That is what makes the next run's regret a number, and it is training data for the
  refit. A slower row never wins the evidence pick, so recording it is safe.

## 3. Broken realizations

```bash
find tests/compiler/realization/cases -name '*_xfail_*.json'
```

A case named `_xfail_<stage>` is a known gap: it fails at `realized`, `built` or `correct`. A schedule found to build
wrong or not at all becomes such a case. Read `tests/compiler/realization/ARCHITECTURE.md` before adding one, and
never add the suffix to make a red case green.

## Report

Lead with the few rows that matter most, then one table per question:

| Row | File | Card | x torch.compile | tried | Reading | Next |
| --- | --- | --- | --- | --- | --- | --- |
| `k_softmax_b2bf51` | `combine-softmax-ilp` | RTX 5090 | 31.2x | — | corpus pin, no golden row | sweep, then record |

Say which rows had no `torch.compile` time, so the reader knows what the listing could not see.
