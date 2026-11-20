---
name: refresh-candidates
description: >-
  Create, repair or fill the candidate files of the hardware goldens: the few measured schedules per group that are
  not the best known one. Use when a hardware golden has no candidate file, when a hardware golden changed and its
  candidate file is stale (the candidate tests in `make test` name it, or the `refresh-golden` skill sends you here),
  when `emmy candidates build` lists groups under their target, or when a stored format or knob was rewritten on main.
  Works file by file: repairs with the CLI where no card is needed, fills groups by tuning their kernels on the
  golden's own card, and reports what cannot be filled and why. Never changes a golden.
---

# Refresh candidate files

A candidate file (GLOSSARY, "Candidate file") sits beside each hardware golden, under
`emmy/compiler/pipeline/search/candidates/records/<same name>`. It copies the golden's programs, kernels and
decisions and holds, for each group of the golden (one kernel, at one set of sizes, in one input regime), a few other
measured schedules: 3 where the kernel's schedule space is small, up to 10 where it is large. No compile reads it.

Two things make a file need work. **It is stale:** the golden it copies changed, or the compiler no longer lowers its
programs to the kernels it stores. **It is short:** groups hold fewer candidates than their target. A stale file is
repaired with no card. A short file is filled by tuning on the golden's card; a tune's measured rows are where every
candidate comes from.

## Inputs to confirm

1. **Which hardware goldens.** Default: every one that has a candidate file, plus any the user names.
2. **A card, only if groups are to be filled.** A box the user hands over, or a rental. Ask which; do not rent
   unasked.
3. **The card is the golden's card.** The name `nvidia-smi` reports must equal the golden's `gpu_name` exactly. The
   RTX PRO 6000 Max-Q and the RTX PRO 6000 Workstation Edition are two cards with two goldens; rows measured on one
   are never filed under the other.

## Repair a stale file (no card)

```bash
emmy candidates check                      # what a restamp would change, per file
emmy candidates restamp                    # rewrite onto the fresh lowering
emmy candidates build GOLDEN…              # copy each golden's tables again; keep the rows whose group it still holds
emmy candidates realize --drop             # drop the rows the compiler no longer realizes
```

- `build` is the step that follows any change to a hardware golden. With no `--db` it adds no row. With `--db PATH`,
  a tune DB copied back from the golden's card, it also tops each group up from the rows that DB measured.
- A restamp leaves a row whose kernel changed with its schedule and no time. The next build drops it: a candidate is
  a measurement. Its group is then short, and a tune refills it.
- When main rewrites a stored spelling by a text edit (a knob retired, a format changed), apply the same edit to the
  candidate files, then build. Read the PR that made the change for the exact rule; do not guess one. A rewrite that
  drops a knob where main respelled it files a measurement under the wrong schedule.

## Fill short groups (on the card)

`emmy candidates build GOLDEN --json short.json` lists the groups under their target: the kernel, the regime and
sizes a tune of it runs at, and what it holds beside its target. An empty list means nothing to fill.

The fill recipe runs the tunes. On a box the user handed over:

```bash
emmy bench .github/candidates-fill --ssh USER@HOST[:PORT] --filter golden=emmy/compiler/pipeline/search/golden/records/<file>.json
```

On a rental, `--provider cloudrift` in place of `--ssh`: the recipe's row names the card, and the run rents it and
tears it down. The recipe stages the checkout, so the card runs the code you are on: be on current main, or the
file comes back in a format main no longer reads.

On the card the recipe tunes each listed kernel into one tune DB, then builds the candidate file again from that DB.
The run returns into `.github/candidates-fill/<timestamp>/`: the filled candidate file, `short.json` (the groups
still under their target), the tune DB and `fill.log`. Copy the candidate file over the repository's:

```bash
run=$(ls -d .github/candidates-fill/20* | tail -1)
cp "$run"/*_candidates.json emmy/compiler/pipeline/search/candidates/records/<file>.json
```

The tune DB is the only copy of every row the tunes measured; nothing in the repository holds it. Keep it outside
the checkout if a later build should draw on it (`emmy candidates build GOLDEN --db PATH`). Delete the run
directory; it is not ignored by git.

## What stays short, and when to stop

Read `short.json` and `fill.log` after the fill. A group still under its target is one of these; say which in
the report, and do not tune it again:

- **A parent a cross-CTA split divides.** Run alone it lowers to two scheduled kernels and the tune refuses it
  ("tunes a program with one scheduled kernel"). Its pieces are groups of their own and are filled.
- **A kernel recorded at a second size.** A tune of a kernel runs at the sizes of its first row in the golden; a
  group at other sizes is not reached.
- **A schedule space that mostly fails.** `fill.log` shows the tune flagging most rows as a wrong answer, a hung
  kernel or a pin the compiler does not realize, and keeping two or three. More tuning adds nothing. This is a
  compiler finding: report the kernel and the counts.
- **A slow kernel that lost its rows to the run limit** ("exceeded … of GPU time"). This one is worth one retry:
  rerun the recipe's `emmy run … --tune` line for that kernel on the box with `EMMY_BENCH_RUN_TIMEOUT_S=300`. If it
  is still short, it is the case above.

## Never

- **Never promote a candidate into a golden, and never edit a golden here.** The build warns when a candidate
  measured faster than its golden's row. Report those lines; whether the golden is beaten is settled by a
  side-by-side bench on the card and lands as a golden change with a prior refit, through the `refresh-golden` skill.
- **Never file rows under another card's name.**
- **Never delete or stop a box you did not rent** without asking.

## Traps

- A tune exits 1 when it flags some rows, and still records the clean ones in the tune DB. The build after the
  tunes is the test, not a tune's exit status.
- `emmy bench` can report the run as failed when the SSH session to the box drops after the tunes are done (exit
  255). The returned files are the result: the candidate file and `short.json`, not the exit status.
- If every bench on a Volta card fails with "no kernel image is available", the installed torch has no sm_70
  kernels: reinstall torch from the cu126 index in the box's venv and run again.
- A remote shell started with `nohup` has no `nvcc` on its path; the recipe exports it. A hand-run command must too
  (`PATH=/usr/local/cuda/bin:$PATH`).
- The candidate tests hold a file to its golden's tables. If a golden changed on main while you worked, merge main
  and build again before the tests.

## Close

```bash
./venv/bin/pytest tests/compiler/pipeline/search/test_candidates.py -n auto --dist=loadgroup
emmy candidates realize                    # every row compiles under its own schedule
```

## Report

Per card: groups, groups under their target with the reason for each kind, candidate rows, and the build's
faster-than-golden lines. Say what was verified and what was not, and the state of any box left running.
