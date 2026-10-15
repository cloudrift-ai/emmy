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

### Rows to re-record

| Golden | Card | Realization (`--realization`) | Lost | Kernels |
| --- | --- | --- | --- | --- |
| `golden/records/a100_sm80.json` | A100 | `k_sdpa_linear_mean_reduce_c9e839` (its latency is cleared), `qkv-child-seed` | route + 14 rows | `k_sdpa_linear_mean_reduce_0d9564` and its 7 nested routes |
| `golden/records/h100_sm90.json` | H100 | `k_sdpa_linear_mean_reduce_c9e839` (latency cleared), `k_sdpa_linear_mean_reduce_0d9564` | route + 14 rows | the same decode attention cut |
| `golden/records/rtx4090_sm89.json` | RTX 4090 | `k_sdpa_linear_mean_reduce_c9e839`, `qkv-child-seed`, `qwen3-06b-s1` | route + 17 rows | the decode attention cut |
| `golden/records/rtx5090_sm120.json` | RTX 5090 | `k_sdpa_linear_mean_reduce_c9e839` (latency cleared) | route + 14 rows | the decode attention cut |
| `golden/records/v100_sxm2_sm70.json` | V100 SXM2 | `k_sdpa_linear_mean_reduce_c9e839`, `layer-input-variance`, `v100` | route + 15 rows | the decode attention cut |
| `recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.json` | V100 SXM3 | `k_linear_softmax_mean_reduce_f461f6` | route + 5 rows | `k_linear_softmax_mean_reduce_f461f6`, nested `k_linear_matmul_softmax_mean_reduce_bcf52a__place_7f72f64f9f#2` |
| `recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json` | V100 SXM2 | `gdn16-dense-linear@fp8`, `gdn64-dense-linear@fp8` | 2 routes + 19 rows | `k_conv1d_linear_mean_reduce_13e5d6`, `_353a47` and their nested routes |
| `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json` | RTX 5090 | `gdn1-dense-linear@nvfp4`, `gdn16-dense-linear@nvfp4`, `gdn64-dense-linear@nvfp4` | 6 routes + 159 rows (474 → 284 measured) | `k_conv1d_linear_mean_reduce_0e5142`, `_5f1430`, `_970a92`, `k_linear_matmul_mean_reduce_02d4dd`, `_db4f79` and their nested routes |
| `recipes/gemma-4-12B-it/golden/rtx5090_sm120.json` | RTX 5090 | `k_linear_mean_reduce_5125d3` | 3 rows demoted | pieces of `k_linear_mean_reduce_5125d3` |

The Qwen3.8-27B-NVFP4 serving matrix is the urgent one: its GDN twins were priced by the dropped routes, so a strict
serving boot of that recipe refuses them until the routes are back (`emmy eval golden --golden PATH
--serving-config recipes/Qwen3.8-27B-NVFP4/serving-rtx5090.env` says which). Try the seam re-spell above first; it
costs no card time. The V100 SXM3 recipes wait for that exact card, as they did after PR #1170.

### Schedules to search again

A re-record alone is not enough where the body changed. Two groups need a search, on the card, before the record:

1. **The kernels whose body shrank.** Decode attention (`k_sdpa_linear_mean_reduce_c9e839` on the five cards
   above) no longer computes its scores, and the three gemma pieces lost a constant operand. Their stored schedules
   were picked for the larger body. Tune each from the working copy, then record what wins:

   ```bash
   emmy run --golden /tmp/<card>.json --kernel k_linear_mean_reduce_5125d3#2 --tune 20 --bench
   emmy run --golden /tmp/<card>.json --realization k_sdpa_linear_mean_reduce_c9e839 --bench --record-greedy --record
   ```

   A `.fm` realization is the same kernel under `EMMY_FAST_MATH=1`; tune and record it under that pin.

2. **The dropped routes.** The kernel set has to be found again where the tree lost a node, and confirmed where it
   only renumbered. Record the greedy pick first (it is the prior's proposal, strict), then sweep the cut arms the
   tree offers as `--ab` rows and record the fastest set:

   ```bash
   emmy run --golden /tmp/<card>.json --realization k_sdpa_linear_mean_reduce_c9e839 --bench --record-greedy
   emmy run --golden /tmp/<card>.json --realization k_sdpa_linear_mean_reduce_c9e839 --bench \
     --ab 'PLACE@map.1/inner=cut,PLACE@map.1/inner.1/inner=cut,PLACE@map.1/inner.1/inner.1/map=cut,PLACE@map.1/inner.2/reduce=cut,PLACE@map.1/inner.2/reduce.1/map.1/inner=cut,PLACE@map.2/inner.1/map=cut,PLACE@map.2/inner.1/map.1/inner.1/map=cut'
   ```

   The `--ab` row above is the old decode attention route's seven surviving seams spelled on the fresh tree — the
   first candidate to measure, not the answer. For the GDN conv1d routes, start from the seams the old routes
   named (`git show main:<golden>` lists them under `routing`) mapped onto the fresh tree, and let the sweep
   decide; a nested cut is recorded as one parent route (see the `refresh-golden` skill), never as child pins.

Nothing else needs a search: a root kernel's stored schedule still names the body it was measured on.

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
