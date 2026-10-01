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
machine trains it; `emmy fit` does, from the golden configurations, and writes the weights into the repository.

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
the same GPU. **So no weight on a hardware feature can change the ranking within that set** — it shifts every
candidate's score by the same amount and cancels out.

That has a counter-intuitive consequence: what tells GPU generations apart has to be a *per-candidate* feature, one
that only takes a value where the hardware offers the thing it describes. Two families of feature exist for exactly
this reason:

- Features that mirror the tile's geometry onto candidates that stage their data through the newer hardware
  transport. On a card that has it, those candidates get a distinguishable description; on a card that does not, they
  never appear. One weight set can then score newer tiles differently from older ones.
- Features that separate candidates with the same tile but a different arrangement of workers. Without them, two
  genuinely different candidates produced identical feature rows, and no model could have told them apart.

If you take one thing from this page for future feature work: a feature that is constant across a fork's candidates
cannot influence that fork.

## How it scores

The prior scores a candidate with a linear formula over the geometry features, fitted ahead of time and stored in the
repository as a small artifact carrying its own version and a record of where it came from. It never falls back on
the order options were emitted in.

**Loading it is strict.** A missing artifact, or one whose feature version does not match the running code, is a hard
error — refit it, never silently continue with something else. (A compile treats the load as best-effort so a bad
artifact does not abort a deployment; what it gets instead is the no-prior behaviour from [the hierarchy
page](./06-deploy-evidence-hierarchy.md), where every fork the prior would have ranked falls to the rule's first
option.)

**It is fitted on the golden configurations**, by `emmy fit`, read from the dataset `emmy db export` writes out of
the previous page's store (which `emmy db import` fills from the golden files named on its command line). For each
kernel a golden row was measured on, the fitter enumerates the candidates that kernel offers — from the kernel's own
definition, as the database holds it — and trains the weights to rank the recorded configuration well inside that set.
The loss has two parts:

- an objective pushing each golden's rank up within its own candidate set, with the kinds of case weighted so that no
  one kind dominates the fit;
- a penalty on the weights, expressed in raw feature units.

**The penalty is there to make the fit well-determined, not to shrink the weights**, and the difference matters. The
ranking objective barely moves when you scale a weight on a feature that hardly varies across the golden candidate
sets. An unpenalized fit is therefore free to pick an arbitrarily large weight there, and nothing in the golden-rank
metrics will show it. It becomes catastrophic at a fork, where a not-yet-decided knob makes that feature zero: the
enormous weight now dominates the score of every candidate that has not yet decided it. The penalty has to be in raw
units, because after rescaling the inflated weight looks like an ordinary one.

**The score is turned into a stand-in for latency by an exponential curve**, and there is a rule about that curve
which is easy to violate: it must never flatten out over the range of scores that actually occur. A curve that
saturates inside the live range collapses good candidates onto one identical value; the choice among them then falls
back to the order the options were emitted in, which is arbitrary. That is not hypothetical — it is how cold
deployments once shipped kernels 12 to 29 times slower than the recorded configuration for the same shape, while the
golden-rank metrics reported the model was choosing correctly. (Why the metrics were fooled is on [the next
page](./09-storage-checks-and-limits.md).)

Two more details are worth knowing. A separate weight set ranks kernels whose tiles are masked because an axis is
symbolic, selected on the stamped structure. And one feature interaction sits outside the linear weights: a term for
the split that combines partial results in a second kernel, rewarded above a split-count threshold and penalized below
it, with both the weight and the threshold fitted.

## What it cannot know

A fitted model is only as good as the pools it was fitted on, and three limits follow directly.

**A kernel no golden covers is scored by extrapolation.** The weights were fitted to rank recorded rows well inside
their own pools; on a shape nothing recorded, the same weights are applied and nothing checks them. That is why a
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

That writes the weights to the path you named and a metrics file to a fresh directory under `_tune/fits/`, so two
fits can be compared by diffing their metrics rather than by argument.

Next: [9. Storage, checks and limits](./09-storage-checks-and-limits.md).
