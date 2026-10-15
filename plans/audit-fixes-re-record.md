# Re-record after the audit fixes (PR #1177)

PR #1177 carries four changes that touch what a golden file holds or how its rows are read, and each of them can
leave a row with its schedule but without its microseconds (a proposal, no evidence until a record run on the card
measures it again). This plan says which goldens that touches, on which card, and how to bring them back. It is
executed by the `refresh-golden` skill one file at a time and deleted when the last file is re-recorded.

## What moved

- **A latency names the kernel set it timed.** Every golden with a whole-target latency block was restamped once to
  fill the kernel set from the family the old sweep used. Nothing lost a measurement; nothing to re-record. From now
  on a latency is cleared exactly when one of the kernels it names is re-keyed or dropped.
- **A layout arm is priced over every reader it settles.** A weight's layout decision is priced as the sum of its
  readers' rows, all-or-nothing. A program whose readers were measured in one layout only still prices that layout;
  a program whose first reader has rows in both layouts but whose other readers have rows in one prices only that
  one. A strict compile that used to decide the fork from the first reader's rows alone can now fall through to the
  prior. On 2026-10-15 the Qwen3.8-27B-NVFP4 strict serving audit on an RTX 5090 passed under this pricing (nine
  twins from the rows alone), so that golden needs nothing. The hardware goldens of the A100, H100, RTX 4090, RTX
  5090 and V100 SXM2 also carry `LAYOUT@` routing rows: on each card, compile their layout targets strictly
  (`emmy run --golden PATH --realization NAME --strict-evidence`) before anything else, and re-record the rows of
  any reader the fork now lacks a price for.
- **The cache regime is declared, not inferred.** No repository golden carries a cold-cache row, so no row moved.
  A recipe that serves cold must now say so in its serving env (`SERVE_ENV="... EMMY_COLD_CACHE=1"`) and record its
  rows under `--cold-cache`.
- **Placement constants are stored at import.** A dataset DB imported before this PR lacks the table; re-import
  (`emmy db import --db PATH --fresh --repository`). No golden moved.

## Rows to re-record

None from this PR. The regeneration of 2026-10-15 re-keyed no kernel, demoted no row and cleared no latency: eight
repository goldens and 125 corpus cases only gained the kernel sets their latencies name (382 latencies filled).
What still needs a card is the strict check of the five hardware goldens' layout targets above, and three groups of
files the regeneration left as they were because they are stale beyond the fill:

- `recipes/Voxtral-Mini-3B-2507/golden/v100_sxm3_sm70.json` (13 kernels re-keyed by the current compiler, 9
  latencies) and `recipes/Ministral-3-14B-Instruct-2512/golden/rtx5090_sm120.json` (22 re-keyed, 2 latencies): both
  recipes are outside the maintained set. A restamp demotes their re-keyed rows; a record run on a V100 SXM3 and an
  RTX 5090 brings them back, or the author deletes them. They load today; a write-back is refused until then.
- The 40 `experiments/golden-bench-2026/**/*.golden.json` records: experiment evidence, read by the golden-bench
  study, never loaded through the golden file wire. Restamp or re-record them with the study, not here.

## How to re-record one golden

1. Rent the golden's card (`emmy vm create gpu --gpu NAME --gpu-count 1`, or an idle rented VM the user names).
2. Ship the branch as a `git archive` tarball, extract it to `~/emmy-<name>`, and run `make setup` inside it with
   `CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH`.
3. Work on a copy of the golden, never the repository path (both writers refuse a canonical path):
   `cp recipes/<model>/golden/<card>.json /tmp/<card>.json`.
4. For each row the list above names, `emmy run --golden /tmp/<card>.json --realization NAME --bench
   --record-greedy` under the row's pins (the `EMMY_*` vars the row's `pins` spell: `EMMY_FAST_MATH=1` for a
   fast-math row). A whole-target latency row needs `--record` as well.
5. `emmy golden check /tmp/<card>.json` must say the file is current; `emmy golden list /tmp/<card>.json` shows the
   rows beside `torch.compile`.
6. Copy the file back, `emmy eval golden --golden PATH --serving-config recipes/<model>/serving-<card>.env` for a
   recipe golden, and commit. A hardware golden change refits both priors (README, "Fit the priors") in the same PR.

Never re-record a row to make a red node green: a row whose measured time regressed is a finding for the PR body.
