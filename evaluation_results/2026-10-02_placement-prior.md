# Placement prior refit — Qwen3 hardware goldens

The refit reproduces 472 of 475 exported placement groups (99.4%), up from 459 (96.6%) with the shipped prior.
Thirteen groups improve and none regress. All 73 added groups are reproduced. This is better coverage of known
routes, but not evidence of better generalization: with identical shape folds, held-out accuracy falls from
450/475 (94.7%) to 448/475 (94.3%). No GPU latency was measured in this evaluation.

## Corpus and fit

The compiler is `1be66018`, including the recorded-route export from PR #1037. The baseline weights are those at
`4edf7d90`. The baseline corpus contains the six hardware and nine recipe golden files; the expanded corpus adds
ten Qwen3-0.6B layer-zero FP16 goldens from the golden-bench experiment. Both exports use the new exporter, so the
comparison of the two refits isolates the corpus expansion from the export change.

The promoted files cover sequence lengths 1 and 512 on A100 40GB, H100 80GB, RTX 4090, RTX 5090 and V100 SXM2 16GB.
Each keeps its traced program, selected routing graph and measured descendants. The shared K/V routes replace
superseded alternatives; the V100 decode keeps the accepted raw-variance route. Parent decisions precede child
decisions. Kernel definitions, schedules and retained measurements are unchanged. The experiment now replays the
canonical hardware files; its [results](../experiments/golden-bench-2026/kernels/RESULTS.md) retain the qualification
history. The compressed evidence below records the per-file pruning counts.

| Exported training data | Before promotion | After promotion |
| --- | ---: | ---: |
| Placement groups | 402 | 475 |
| Shape groups used for cross-validation | 145 | 161 |
| Groups labeled with cuts | 150 | 172 |
| Groups labeled keep-fused | 252 | 303 |
| Positive arms | 793 | 978 |

The 18.2% increase in groups adds 22 cut groups and 51 keep-fused groups. Groups are not independent model examples:
several cards and nested forks come from the same program. Export skips 715 contexts with no placement fork and
147 kernels not formed from a loop op. Neither count is an incorrect prediction. RTX 4080 and RTX PRO 6000 still
have no ranked placement groups. All exported placement arms are scored; no pool was sampled down.

Both refits use the existing CatBoost defaults: 200 trees, depth 6, learning rate 0.3, one round, seed 0, the
placement feature view and five shape folds. The negative draw is 500, larger than every placement pool. Shapes
are held out on all cards together. The schedule prior is unchanged.

## Same-corpus ranking and held-out quality

Top-1 means the first arm selected by the prior is a recorded acceptable arm, including emission-order ties.
The two rightmost columns use exactly the same expanded-corpus fold assignment and held-out groups. For the
old-corpus control, the fitter additionally excludes every promoted group from each training fold.

| Card | Groups | Shipped top-1 | Old-corpus refit top-1 | Expanded refit top-1 | Held out, old corpus | Held out, expanded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 40GB | 13 | 12 | 12 | 13 | 12 | 12 |
| RTX 4090 | 28 | 27 | 27 | 28 | 27 | 27 |
| RTX 5090 | 172 | 170 | 170 | 172 | 170 | 169 |
| H100 80GB | 33 | 32 | 32 | 33 | 32 | 32 |
| V100 SXM2 16GB | 55 | 50 | 53 | 55 | 46 | 46 |
| V100 SXM3 32GB | 174 | 168 | 171 | 171 | 163 | 162 |
| Total | 475 | 459 | 465 | 472 | 450 | 448 |

The old-corpus refit fixes five original-corpus choices and one added choice. Training on the promoted data fixes
the remaining seven added choices. On the 402 original groups, both refits get 399 right, versus 394 for the shipped
prior. Held-out accuracy on the 73 added groups is 65/73 (89.0%) for both training corpora; some individual decisions
change even though their total does not. The expansion therefore teaches the known routes but shows no top-1
transfer improvement in this split. The two additional held-out misses are on RTX 5090 and V100 SXM3.

The ordinary five-fold baseline on the original corpus scored 382/402. Its fold assignment changes when the corpus
grows, so it is not the controlled comparison above. Summary counts include every group. The fit's per-golden maps
overwrite repeated display keys across contexts; use summary counts rather than counting those maps as the corpus.

## Remaining limits and decision

The three full-training misses are V100 SXM3 nested forks under parents `3212d0741650`, `3d6165d23d5c` and
`f24f30cf6f08`. In each, emitted arm 1 and the
golden arm 5 have identical selected feature vectors. They tie, and emission order selects arm 1. Increasing fit
iterations cannot distinguish identical inputs. The next feature change should preserve the structural difference
between these cuts, with tests built from these concrete collisions.

Median rank is zero for every card under both priors. The nightly comparison rejects the candidate because no
median can improve by its required 5%; this does not contradict the top-1 improvement. This explicitly requested
refit is retained for its improved known-route coverage, with no claim of a held-out or latency improvement. The
next data expansion should add different program structures and shapes, rather than more copies of these routes.

## Reproduction and evidence

Use the repository at `3fda23b5` for the original corpus and `1be66018` for the promoted corpus, with the same
exporter. Keep DBs and datasets separate. For the expanded corpus:

```bash
emmy db import --db _data/placement.db --repository
emmy db export --db _data/placement.db _data/placement --space placement
emmy fit _data/placement _data/placement-candidate.json --folds 5 --out _data/placement-fit
emmy eval prior _data/placement --offline-file _data/placement-previous.json \
  --compare-to _data/placement-candidate.json --json _data/placement-comparison.json
```

Save the previous weights before fitting. The controlled cross-validation uses the existing `assign_folds` /
`run_folds` machinery over the expanded groups and filters promoted group keys from each old-corpus training fold.
[Compressed raw evidence](2026-10-02_placement-prior.json.gz) includes source digests, fit settings, both fits,
controlled cross-validation, per-group comparisons, promotion counts and weight hashes. It contains no latency
claim inferred from ranks.
