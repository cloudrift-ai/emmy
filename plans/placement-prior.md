# Placement prior

Status: proposed 2026-09-30 against `refactor/golden-records`. A second offline prior that ranks the arms of a
placement fork (`030_cut`: keep fused, cut one seam, the full-projection cut), fit on its own dataset, so a
structural fork no measurement decides is answered by a ranker instead of a nested tile resolution per arm.

## Usage

```bash
# 1. Fill the dataset DB as today — the routing rows the golden import writes are the placement evidence.
emmy db import --db _data/dataset.db emmy/compiler/pipeline/search/golden/records/*.json recipes/*/golden/*.json
# 2. Export ONE space per dataset directory.
emmy db export --db _data/dataset.db _data/schedule --space schedule
emmy db export --db _data/dataset.db _data/placement --space placement
# 3. Fit each prior from its own dataset; the manifest names the space, the weights file records it.
emmy fit _data/schedule  emmy/compiler/pipeline/search/prior/weights/schedule.json
emmy fit _data/placement emmy/compiler/pipeline/search/prior/weights/placement.json
# 4. Rank each golden decision among the arms its fork offers, under the shipped weights.
emmy eval prior _data/placement
# 5. A greedy compile loads both weights files; nothing on the command line changes.
emmy compile recipes/gemma-4-12B-it --layer 0
```

`emmy fit` and `emmy eval prior` take no new flag: a dataset holds one space, the reader takes it from the manifest,
and a weights file loaded at the wrong kind of fork is refused the way a stale featurizer version is today.

## Why

A cut arm has no knob row of its own, so the greedy prices it by a nested deterministic resolution of every piece
(`_priced_pick` → `_price_graph` → `_price_kernel` → `_resolved_price`) and sums the schedule prior's scores across
the pieces. That has three costs the deploy path pays today:

- Minutes of nested tile pricing per attention target before any GPU work (golden replay time is kernel-set pricing).
- The Σ compares absolute scores across kernel families. The schedule prior is fit to rank WITHIN one pool; its
  absolute value carries no calibration, and the docstrings name this exposure as "the prior's problem to fix".
- The schedule prior's features describe one kernel's tile geometry. Nothing in them says what a seam costs: the
  workspace it materializes, the recompute it removes, the launch it adds.

A placement prior ranks the arms of one fork against each other, which is the only comparison the deploy needs, with
features that describe the arms. Measured routing rows keep outranking it (deploy evidence hierarchy, unchanged).

## The fork it ranks

`_placement_forks` offers, per kernel with cuttable seams: `PLACE=fuse`, one arm per seam, the full-projection arm
when it exists, and composed arms for routes the DB measured. A pinned compile consumes several seams as ONE composed
decision, stored as one routing row on the parent (arm = its `PLACE@site` keys, children = the pieces). Cutting
continues on the pieces to fixpoint, each piece a new kernel with its own fork. The prior ranks the unpinned offer —
fuse, single seams, full projection — at every one of those forks; composed arms exist only where a measurement
already decides, and are never ranked.

Cross-CTA splits (`REDUCE` with a `g<n>` half) are schedule-space rows and stay with the schedule prior.

## The placement dataset

**Source.** The dataset DB as `emmy db import` fills it today: kernel rows (parent Loop IR, `formed`), routing rows
(parent, arm, pieces), and the golden rows of every kernel. The golden format does not change. Provenance is already
sufficient (Gemma 4 on the 5090: 118 routing rows; DeepSeek-V4 on the V100: 31; the hardware goldens: none).

**Pools.** One pool per fork along a golden's cut walk, labelled by marks like the schedule pools (`GoldenGroup`
reused, `golden_ids` mark the positives; pool key `<gpu>/<parent kernel>#<fork>`):

- On a parent with a routing row: arms are the unpinned offer. Positives are every seam the composed decision names
  (the order the compile took them is not recorded and any order is consistent with the golden), plus the
  full-projection arm when it equals that set. Fuse is negative. Take one positive seam (site order, canonical),
  realize the cut, and continue on each piece: a piece that still holds golden seams gets a pool with those as
  positives; a piece with none gets a pool with fuse positive and every seam negative.
- On a golden kernel with cuttable seams and NO routing row: one pool, fuse positive, every seam negative. This is
  where most of the "do not cut" signal comes from, and it is the only reason the hardware goldens contribute.

Negatives are assumed exactly as the schedule pools assume them: the golden marks the recorded winner and every
other arm counts as worse. A placement pool has a handful of arms, so one arm nobody measured weighs far more than
in a pool of thousands of schedule rows. The eval reports every pool's rank, not only a top-k rate.

**Arm features.** From the pieces' `S_*` stamps as the tile lift writes them, before any schedule enumeration, plus
the card context (`ctx.features()`) — a fixed-length row aggregated over the arm's pieces, under its own prefix
(`P_*`) and its own featurizer version: piece count; bytes read and written per piece, summed and max; cell work per
piece, summed and max; workspace bytes the seam materializes; work the cut removes relative to the fused kernel;
symbolic axes per piece. The fuse arm is the parent alone. This feature set is the real work of the plan; the first
eval runs with a hand-set weight vector (for example, total bytes moved) as the baseline a fit must beat.

**Export.** The space selects the pass list the enumeration resolves under (`enumerate_graph` already takes one):
the schedule space runs the tile lowering as today; the placement space runs `tile/lift` and `tile/cut` only and
captures each fork's `PLACE` leaves. Under `PLACE` pins the cut pass offers one composed arm and no fork, so the walk
takes the golden's arm through the decide callback while the offer stays unpinned, and it expands each captured
leaf to its graph, since an arm's features come from its pieces rather than from its knob row.

`emmy db export --space {schedule,placement} OUT` is explicit, like every other path in the refit; the
shipped schedule weights move from `weights/offline.json` to `weights/schedule.json` in the same change, so the two
artifacts are named by what they rank. The placement export walks only the golden kernels with cuttable seams and their
routing rows, realizes each arm once (tile lift only, CPU), and writes the pools as packed matrices; the manifest
records the space. No reservoir sampling is needed. The schedule export never sees `PLACE`.

## Fit and eval

The manifest's space selects the featurizer and the column view; `LinearTrainer` first, `CatBoostTrainer` only if the
linear ranks say the pools warrant it. The weights artifact records `space` and the placement featurizer version;
`load_prior` refuses a mismatch. `emmy eval prior` on a placement dataset prints, per pool, where the golden's arm
ranks among the offered arms and which arm the prior would take.

## Deploy

`greedy_decide` at a structural fork, in this order: measured routing rows (`_route_candidates`, unchanged), then
the placement prior's argmin over the offered arms, then the no-prior fall-through (first leaf) as today.
`--strict-evidence` is unchanged: a placement fork no routing row decides is still an error.

Once the placement prior ships, `_priced_pick` and the nested pricing behind it are deleted, not kept as a fallback.
The pieces an arm mints go to their own schedule forks exactly as today.

## Steps

1. Arm walker + `P_*` featurizer over a parent's Loop IR → verify: every routing row of the Gemma 5090 and DeepSeek
   V100 goldens realizes at its fork and lands in a pool as a positive (the placement twin of the schedule pools'
   realizes check), as a test under `tests/compiler/pipeline/search/`.
2. `emmy db export --space placement`, manifest space, `Dataset.load` reading it → verify: export of the repository
   goldens, `emmy eval prior` under the hand-set baseline weights, ranks reported per pool.
3. `emmy fit` on the placement dataset, `weights/placement.json` checked in → verify: cross-validated ranks beat the
   baseline; a schedule weights file at a placement fork is refused.
4. `greedy_decide` asks the placement prior; nested pricing deleted → verify: `make test`; pinned golden replay time
   on an attention target before and after; a greedy compile of Gemma 4 layer 0 and DeepSeek-V4 picks the recorded
   cuts under `--strict-evidence` off with the tune DB disabled.
5. Docs: pipeline `ARCHITECTURE.md` Parts 3 and 4 (the hierarchy at structural forks, the four stores), `search/`
   `ARCHITECTURE.md` (dataset spaces), README's "Fit the offline prior" (the second export and fit), `GLOSSARY.md`
   (the dataset's space; no new label beyond that).

## Risks

- Thin data: about 150 routing rows on two cards, none on the hardware goldens. The fuse pools from golden kernels
  without a routing row carry the count, but their mark is weak: a fused golden was recorded fused because the greedy
  kept it fused, not because a cut was measured and lost. A `--record-greedy` run that measures the single-seam arms
  on the Gemma and DeepSeek targets would turn assumed negatives into measured ones; that is a record run, not code.
- A feature set that separates fuse from cut across two model families may not exist at the `S_*` level; if the
  baseline and the fit tie, the answer is a better stamp (a lift-time fact), not a schedule enumeration per arm.
- The order of a composed decision is unrecorded; a canonical order makes the intermediate pools an assumption.
  Recording the order on the routing row is a small format change if the eval shows it matters.
