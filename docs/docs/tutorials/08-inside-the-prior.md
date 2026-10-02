---
sidebar_position: 8
title: "8. Inside the Prior"
description: The model that ranks schedules before they are measured — its features, how it is fitted on the golden configurations, and the limits that follow from that.
keywords: [Emmy, prior, features, offline prior, fit, golden configurations]
---

# 8. Inside the Prior

When no measurement answers a fork, something still has to choose. That something is the **prior**: a model that
predicts how fast a configuration will be, from the configuration alone. This page is about how it works, and it goes
slower than the pages before it, because this is where the interesting failures live.

Two properties frame everything else.

**There is one ranking path.** Whatever is choosing asks the same object the same way. Forks carry no score of their
own, and nothing builds a kernel in order to rank it. The knob values go straight into numbers the model consumes.

**The prior is fitted ahead of time and ships with the repository.** It is what answers on a machine that has never
measured anything — and only there: a measured row of the same kernel outranks it at every fork, as [the hierarchy
page](./06-deploy-evidence-hierarchy.md) explained. It is called the *offline* prior for that reason. Nothing on the
machine trains it; `emmy fit` does, from the golden configurations, and writes the model into the repository.

## Features: what the model actually sees

A **feature** is one number describing a candidate, computable without building it. Every prediction is a function of
a row of features, and the row has three groups.

| Group | What it describes | Where it comes from |
| --- | --- | --- |
| Hardware and regime | which GPU, its memory, the optimization level this compile is running at | probed from the machine, or named with `--target` |
| Structure of the operation | counts of the statements and operations in the kernel's body, the loop extents, the data types of the inputs | stamped onto the operation by the stamping pass, before any fork is reached |
| The candidate itself | its knob values, encoded by type; a named tensor core instruction expands into the properties of that instruction | the fork option |

The model additionally computes hand-designed descriptions of a tile's geometry from those knob values — its area, its
shape, how much shared memory it needs, how many groups of threads could be resident at once.

### The subtlety that shapes the feature set

The hardware group has the same value for every candidate competing at one fork. Every option is being compiled for
the same GPU. **So a hardware feature on its own cannot change the ranking within that set** — it moves every
candidate's score by the same amount.

What it can do is change how *another* feature counts. The model is a set of decision trees, and a tree can split on
the card first and on the candidate second: prefer accumulating in 16-bit floats on one generation and in 32-bit
floats on another. A per-candidate feature still has to carry the difference — candidates with the same tile but a
different arrangement of workers have their own features for that reason, because two genuinely different candidates
with identical feature rows can never be told apart.

If you take one thing from this page for future feature work: a feature that is constant across a fork's candidates
cannot influence that fork by itself.

## How it scores

The prior scores a candidate with a CatBoost ranker — a sum of small decision trees over the geometry features —
fitted ahead of time and stored in the repository as one JSON file carrying its own version and a record of where it
came from. The trees sit in that file in CatBoost's own JSON format. It never falls back on the order options were
emitted in.

**Loading it is strict.** A missing artifact, or one whose feature version does not match the running code, is a hard
error — refit it, never silently continue with something else. (A compile treats the load as best-effort so a bad
artifact does not abort a deployment; what it gets instead is the no-prior behaviour from [the hierarchy
page](./06-deploy-evidence-hierarchy.md), where every fork the prior would have ranked falls to the rule's first
option.)

**It is fitted on the golden configurations**, by `emmy fit`, read from the dataset `emmy db export` writes out of
the previous page's store (which `emmy db import` fills from the golden files named on its command line). For each
kernel a golden row was measured on, the fitter enumerates the candidates that kernel offers — from the kernel's own
definition, as the database holds it — and trains the trees so the recorded configuration comes first inside that set.
The other candidates were never measured, so they are not known to be slow; the fit draws a sample of them as the
configurations to rank below the recorded one.

**A feature a candidate does not have is "missing", not zero.** A knob a candidate has not decided yet and a knob
legitimately set to zero are different facts, and a tree can branch on the difference.

**The score is turned into a stand-in for latency by an exponential curve**, so lower is better and the scores of
several kernels can be summed when a structural choice compares kernel sets.

## What it cannot know

A fitted model is only as good as the pools it was fitted on, and three limits follow directly.

**A kernel no golden covers is scored by extrapolation.** The model was fitted to rank recorded rows well inside
their own pools; on a shape nothing recorded, the same model is applied and nothing checks it. That is why a
compile that must not guess runs under `--strict-evidence`, and why the fix for a slow cold pick is a measurement,
never a rule written into a pass.

**Its absolute numbers are not calibrated.** The score is a ranking quantity turned into a stand-in for latency, and
within one pool only the order matters. But a structural choice — keep two operations fused, or cut them apart —
compares sums of per-kernel prices, and where a measured row prices one piece and the prior prices another, the
prior's absolute error does not cancel. A recorded decision, priced from its pieces' measured rows, outranks any such
sum, which is why the goldens page records kernel sets and not only schedules.

**It can rank an invalid tile first.** The prior knows geometry, not the card's limits, so its top pick can need more
shared memory or more threads than the card has. An ordinary compile notices that at the end, blocks the tile, and
resolves again — [the hierarchy page](./06-deploy-evidence-hierarchy.md) describes the retry.

## See it yourself

Evaluate the prior against the golden configurations — where each recorded configuration ranks among the candidates
it competed against. The report reads the dataset `emmy db export` writes, so fill the database from the hardware
golden files and export it first; nothing fills it on its own:

```bash
emmy db import --db _data/dataset.db --fresh emmy/compiler/pipeline/search/golden/records/*.json
emmy db export --db _data/dataset.db _data/dataset
emmy eval prior _data/dataset
```

A rank is only a screen. It says where a good configuration landed in the ordering, never what missing it costs — two
neighbouring ranks in a large pool can be a fraction of a percent apart or three times apart. The question a rank
cannot answer is asked over configurations that were actually measured, which is the next page's dataset:

```bash
emmy db import --db _data/tune.db ~/.cache/emmy/autotune.db
emmy db export --db _data/tune.db _data/tune
emmy eval prior _data/tune --pools measured
```

That one reports, per card and compile setting, how closely the model's ordering follows the hardware's and what its
best guess costs against the fastest configuration measured.

And refit the prior from the golden configurations, with cross-validation, no GPU required:

```bash
emmy fit _data/dataset _tune/fits/offline.json
```

That writes the model to the path you named and a metrics file to a fresh directory under `_tune/fits/`, so two
fits can be compared by diffing their metrics rather than by argument.

Next: [9. Storage, checks and limits](./09-storage-checks-and-limits.md).
