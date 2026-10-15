# Re-record after the audit fixes (PR #1177) and the Affine normal form (PR #1175)

Two PRs leave golden rows with their schedule but without their microseconds (a proposal, no evidence until a record
run on the card measures it again), and one of them drops cut routes whose seams the fresh Fold tree no longer
offers. This plan says which goldens that touches, on which card, which rows to re-record, which schedules to search
again, and how. It is executed by the `refresh-golden` skill one file at a time and deleted when the last file is
re-recorded. Nothing here was re-recorded in either PR.

## What the audit fixes moved (PR #1177)

- **A latency names the kernel set it timed.** Every golden with a whole-target latency block was restamped once to
  fill the kernel set from the family the old sweep used. Nothing lost a measurement; nothing to re-record. From now
  on a latency is cleared exactly when one of the kernels it names is re-keyed or dropped.
- **A layout arm is priced over every reader it settles.** A weight's layout decision is priced as the sum of its
  readers' rows, all-or-nothing. A program whose readers were measured in one layout only still prices that layout;
  a program whose first reader has rows in both layouts but whose other readers have rows in one prices only that
  one. A strict compile that used to decide the fork from the first reader's rows alone can now fall through to the
  prior. On 2026-10-15 the Qwen3.8-27B-NVFP4 strict serving audit on an RTX 5090 passed under this pricing (nine
  twins from the rows alone), so that golden needed nothing from this PR. The hardware goldens of the A100, H100,
  RTX 4090, RTX 5090 and V100 SXM2 also carry `LAYOUT@` routing rows: on each card, compile their layout targets
  strictly (`emmy run --golden PATH --realization NAME --strict-evidence`) before anything else, and re-record the
  rows of any reader the fork now lacks a price for.
- **The cache regime is declared, not inferred.** No repository golden carries a cold-cache row, so no row moved.
  A recipe that serves cold must now say so in its serving env (`SERVE_ENV="... EMMY_COLD_CACHE=1"`) and record its
  rows under `--cold-cache`.
- **Placement constants are stored at import.** A dataset DB imported before this PR lacks the table; re-import
  (`emmy db import --db PATH --fresh --repository`). No golden moved.

The regeneration of 2026-10-15 re-keyed no kernel, demoted no row and cleared no latency: eight repository goldens
and 125 corpus cases only gained the kernel sets their latencies name (382 latencies filled). Three groups of files
were left as they were because they are stale beyond the fill:

- `recipes/Voxtral-Mini-3B-2507/golden/v100_sxm3_sm70.json` (13 kernels re-keyed by the current compiler, 9
  latencies) and `recipes/Ministral-3-14B-Instruct-2512/golden/rtx5090_sm120.json` (22 re-keyed, 2 latencies): both
  recipes are outside the maintained set. A restamp demotes their re-keyed rows; a record run on a V100 SXM3 and an
  RTX 5090 brings them back, or the author deletes them. They load today; a write-back is refused until then.
- The 40 `experiments/golden-bench-2026/**/*.golden.json` records: experiment evidence, read by the golden-bench
  study, never loaded through the golden file wire. Restamp or re-record them with the study, not here.

## What the Affine normal form moved (PR #1175)

Every sum, difference, negation and product with a literal is one `Affine` statement, `x - x` is the literal `0`, a
literal-only operation is the literal it computes, and a value no store reaches is dropped. That changes the exact
identity of every kernel that sums, so the tune DB is re-created (version 8) and both priors were refit (featurizer
version 11). A golden stores no identity, and a root kernel's stored body normalizes to the same statement as its
fresh lowering, so **no root kernel re-keyed and no root row lost its microseconds**. What moved is the Fold tree
under a cut:

- The one-key softmax weight `exp(s - s) / exp(s - s)` is `1` now, so the score, the dot product that fed it and
  the projection node that carried the weight are gone from decode attention. The composed cut that routed it
  named that projection (`PLACE@map.1/inner.1/map`) as one of its eight seams; the other seven map one to one onto
  the fresh tree (`map.1/inner.1/map.K/…` is `map.1/inner.K/…` now). The route **drops with its pieces and their
  rows** on every card that recorded it. It was not re-spelled by hand: seven seams of eight is a different kernel
  set, and a kernel set is recorded, never assumed. The root kernel's own rows survive, but they timed a kernel
  that still computed the scores: they are upper bounds until the card measures the smaller body.
- A reduce's `/ N` and `+ eps` are one statement, so the constant operand node beside such a reduce is gone and
  the canonical statement order can swap two operands of one node. The GDN conv1d/linear composed cuts of both
  Qwen3.8 recipes and one DeepSeek route name seams by operand position, so they **drop** the same way. Their
  nodes still exist: under `k_conv1d_linear_mean_reduce_970a92` the subtree `reduce.2/inner.1/map.10/…` is
  `reduce.2/inner.2/map.10/…` now and one `map.N` constant child is gone. A hand re-spell through `GoldenFile`
  that maps every seam by site content (axis, operands, loads, ops — compare `sites()` under main and the branch)
  and then restamps is valid exactly when the restamp keeps every piece's identity; a piece it re-keys was a
  different kernel set, and that route needs the record run below.
- Three gemma post-RMSNorm pieces (`k_linear_mean_reduce_5125d3`) have a smaller body and **lost their
  microseconds**; their schedule stays as a proposal.

### What was recorded on 2026-10-15 and what is still owed

The decode attention composed cut was recorded fresh on the A100, H100, RTX 4090, RTX 5090 and V100 SXM2 (a
seven-cut route; the A100 target runs in 36.7 us where main's recorded 48.3 us, the H100 in 23.3 us against 31.0, the
RTX 5090 in 17.1 us against 18.5), the three gemma pieces were re-measured on the RTX 5090, and the Ministral RTX
5090 golden was refreshed. The Qwen3.8 NVFP4 golden got its two linear-matmul GDN routes back by re-spelling their
seams on the fresh tree (every piece kept its identity, 84 measured rows), its gdn1 conv1d route recorded fresh in
both precision lanes and its gdn64 route in the fast-math lane: 433 of 460 rows measured against main's 474 of
495, eleven whole-target latencies against eight. Every repository golden is the fresh lowering.

Later on 2026-10-15 the remaining NVFP4 routes landed: gdn64's plain lane replayed through `--pin-route` with its
pieces tuned (223 us against torch.compile's 138 us; the fast-math lane 308 against 100), and gdn16 recorded fresh in
both lanes from its thirteen root seams with every other placement site fused, under the weight layouts serving
settles (36.4 ms against torch.compile's 88 us; the strict serving audit deploys nine twins of nine). gdn16 is slower
than the route main held (23.5 ms by best-row sums): the nested seam that split its recurrence moved with the normal
form, the fused piece hangs, and the pick keeps one 36 ms nested piece. The FP8 gdn16 route recorded in both lanes on
the V100 SXM2 (22.7 ms against 449 us) with the staged fill pinned off — the staged Volta mma piece the prior picks
fails the accuracy check against eager; a record on that card runs without `--strict`, whose eager comparison trips on
FP8's own rounding steps (e4m3 at 3.5 steps by 0.25). Every repository golden stays current.

PR #1182 recorded the rest. NVFP4 gdn16 took back main's two nested cuts, found on the fresh pieces by site
content (the windowed projection's normalization, the sum-of-squares split): 5.0 ms in both lanes under the folded
layouts serving settles, against main's 23.5 ms; nine twins of nine. The FP8 gdn64 route needed two nested cuts the
same way and records at 10.1 ms in both lanes on the V100 SXM2 (eager 2.44 ms, torch.compile 584 us); six twins of
nine, as on main. The DeepSeek piece got its rows on a V100 SXM3.

Still owed: `recipes/Voxtral-Mini-3B-2507/golden/v100_sxm3_sm70.json` (best-effort recipe) is stale, 31 kernels
re-keyed; a record on a V100 SXM3 brings it back, or the author deletes it.

The seam re-spell that recovered the linear-matmul routes maps each old seam onto the fresh tree by site content; it
is valid exactly when the restamp keeps every piece's identity. Where it re-keyed pieces, the restamp paired children
by order and hung one piece's row on another kernel, so those rows were dropped and the route recorded instead.
The restamp now does this re-spell itself (PR #1180): a route whose seams moved is taken again at the seams that mint
exactly its stored pieces, so a later canonical-order change keeps such routes without a hand mapping.

### Schedules searched again

The kernels whose body shrank were tuned where they were recorded: the decode attention pieces on every card (the
whole-target latencies above), the re-keyed gdn1 pieces on the RTX 5090 (`--kernel PIECE --tune N --bench` before the
strict record), the gemma pieces. 

## How to re-record one golden

1. Rent the golden's card (`emmy vm create gpu --gpu NAME --gpu-count 1`, or an idle rented VM the user names).
2. Ship the branch as a `git archive` tarball, extract it to `~/emmy-<name>`, and run `make setup` inside it with
   `CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH`.
3. Work on a copy of the golden, never the repository path (both writers refuse a canonical path):
   `cp recipes/<model>/golden/<card>.json /tmp/<card>.json`.
4. For each row the tables above name, `emmy run --golden /tmp/<card>.json --realization NAME --bench
   --record-greedy` under the row's pins (the `EMMY_*` vars the row's `pins` spell: `EMMY_FAST_MATH=1` for a
   fast-math row), after the search the row's group asks for. A whole-target latency row needs `--record` as well.
5. `emmy golden check /tmp/<card>.json` must say the file is current; `emmy golden list /tmp/<card>.json` shows the
   rows beside `torch.compile`.
6. Copy the file back, `emmy eval golden --golden PATH --serving-config recipes/<model>/serving-<card>.env` for a
   recipe golden, and commit. A hardware golden change refits both priors (README, "Fit the priors") in the same PR.

Never re-record a row to make a red node green: a row whose measured time regressed is a finding for the PR body.
