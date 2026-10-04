---
sidebar_position: 9
title: "9. Storage, Checks and Limits"
description: How measurements are keyed and stored, how the prior is evaluated when it chooses badly, and where the whole design falls short.
keywords: [Emmy, tuning database, structural key, eval, diagnostics, limitations, prior]
---

# 9. Storage, Checks and Limits

The last page. Three subjects, in increasing order of usefulness to somebody deciding whether to trust any of this:
how measurements are stored, how the prior is examined when it chooses badly, and what the design does not do.

## Two identities, and only two

Everything Emmy stores or replays is keyed by one of two identities. When adding a cache or a table, pick one of them
rather than inventing a third.

**Variant identity — the compile context, the kernel and the knob values.** Used by anything that *predicts* or
*replays*. The knob values are decisions and nothing else. The structural facts about the kernel — the counts in its
body, its loop extents, its data types — are computed from the kernel whenever a row is turned into features, never
stored on it, and the prior is a pure function of the three together.

**Measurement identity — the kernel, the sizes it was benched at and its knob values, under the card and the compile
setting.** Ground truth about kernels that were actually built: their measured times, and the deduplication that
collapses 24 identical RMSNorm kernels into one unit of work. The kernel is named by its exact identity, a digest of
the loop program it executes and the buffers it reads and writes, which ignores the size a dynamic axis is expected to
take — so the sizes a measurement ran at are part of the key, and one kernel benched at two sizes is two rows.

## The measurement tables

The database holds kernels, the decisions that minted them, and measurements of them, and every instance of it — the
tuning database a compile reads, the dataset database the evaluations read — holds the same tables.

**Kernels.** One row per kernel: its exact identity, the loop program that defines it and its C name. The identity is
the one computed value the table keeps — a cache key, computed from the loop program by whoever writes the row. The
kernel's structural features are not stored; they are computed from the loop program when a row is turned into
features. A kernel that a cut or a split minted is a row like any other, so the same
kernel reached from two parents has one definition, and that definition is what its candidate pool is enumerated
from.

**Routing.** One row per kernel a structural decision minted: the parent, the decision (which seam was cut, how a
reduction was split across blocks), and the piece. A minted kernel with decisions of its own is the parent of further
rows. The decision has no measurement of its own; its price on a card is the sum of its pieces' fastest measurements
there, which is how a compile reads it.

**Measurements.** One row per measured kernel variant per card and compile setting, the schedule it ran with stored
once and shared between rows. The GPU's name is part of the key: compute capability alone cannot separate two cards
built on the same die — an H100 and an H200 share it, and their SM counts — so without the name their rows would
merge and one card's data would silently overwrite the other's. A better measurement of the same variant replaces a
worse one. **Failures are kept**, with the watchdog's placeholder time, because a search model needs negative
examples; a working row is never downgraded by a later failure. A row that spells a cut or a split is refused: that
is a decision, not a measurement of one kernel.

**Nothing migrates.** A database file written by an older version of the compiler is re-created empty on the next
write, since every row in it can be measured again, and refused by a reader. The file also carries a version for the
way a kernel's identity is computed, and a file written under another version is re-created the same way.

**The tables are checked, not the code.** `emmy db check` verifies that an instance's tables agree with
themselves — every knob row's digest, every reference, every card, the two knob vocabularies — and counts the rows
that fail. It does not re-derive what the compiler wrote: the tuning database is a cache, and a row the current code
disagrees with is re-benched or re-imported. The freeze is what travels between machines.

**A frozen snapshot makes a fit reproducible.** The tuning database is a live store — benches keep writing into it
— so a model fitted straight from it cannot be reproduced later. A freeze is a snapshot written as a golden file per
card: every measured kernel — the loop body the compiler formed it from, with no identity or feature stored beside
it — the kernel-set decisions that reach it, and its measured schedule rows, each with the setting it was measured
under and its median. Importing a freeze puts those rows back, row for row, each under the identity computed from
its kernel's body. A freeze keeps no traced program, so a compiler change that re-keys a kernel leaves its frozen rows
behind; the repository goldens, which do, are restamped instead. Freezing the same database twice produces
byte-identical files. A freeze is named on the `emmy db import` command line like any
other source — a golden configuration file, or a tuning database from this machine or a rented card — and that
database is what every evaluation and the offline fit read; nothing is loaded into it by default, and no freeze is
checked in at the moment.

**Hand-run measurements are recorded too.** A `run --bench` that measured configurations with knob values forced by
hand records each clean result through the same writer as the compiler's own pick, so that manually found optima
are not lost when the
session ends. Rows that were flagged by any of the integrity checks are never recorded at all.

## The version stamp, and what raising it costs

Every stored training artifact carries the version of the feature encoding it was written under. Raising that version
is the correct response to any incompatible change in how knobs are named or features are encoded — old artifacts
are refused instead of poisoning the model with rows whose names no longer mean anything.

It is worth being explicit about how far the consequence reaches, because it is not obvious:

- **The prior's weights from another version are refused**, and so is a dataset exported under another version:
  refit and re-export, never silently continue.
- A compile whose weights fail to load has no prior, and every fork the prior would have ranked falls to the rule's
  first option — **with no warning at compile time** unless `--strict-evidence` is on.
- The measurements table survives, because it is keyed by content rather than by feature names.

A feature change cannot reach the evidence at all. A measured row is matched to a candidate by the kernel's exact
identity and the compile context, and by nothing else: no feature takes part in the join, so adding or changing one
cannot switch off the evidence tier.

## Finding out where the prior is wrong

`emmy eval prior` exists for the question "the prior chose badly — where?". Two views matter.

**Where a golden ranks, with ties counted against it.** For each recorded golden configuration, the view reports how
many candidates the prior scored better. A tie counts as a loss, because when scores are equal the compile takes
whichever came first, which is not the golden. Counting only strictly-better candidates would report a perfect rank
for every configuration sitting inside a plateau of equal scores — and that is precisely how a saturated model scored
top-of-the-list on the goldens while real cold deployments missed by 12 to 29 times. Both counts are reported
side by side, and the gap between them is the width of the tie plateau: an early warning that the scores are
saturating.

These evaluations rebuild each golden's compile context for **the GPU the golden was recorded on**, never the machine
running the evaluation. Building them for the host makes the ranks machine-dependent, since the geometry features
would then be describing tiles for the wrong card.

**What a wrong choice cost.** Over configurations that were all actually measured, this reports two things per card
and compiler setting: how closely the model's ordering follows the hardware's, and how much slower its best guess is
than the fastest configuration in the set. A ratio of 1.00 means the model's pick *is* the best available. This is
the view that tracks deployed speed, and it is why the ranking view above is only a screen — a rank says where a good
configuration landed in the ordering, never what missing it costs.

Every figure carries the count of comparison sets behind it. The sets have different minimum sizes — a correlation
needs more members than a ratio does — and the ones too small for a given figure are excluded rather than averaged
in, so the count is what tells you how much of the data a number actually covers.

## Limitations

Gathered in one place, honestly.

1. **Ranking-setting measurements are known to invert against the deployable setting.** The evidence index is
   keyed by the compile's regime, so a bench at the ranking setting is never read by a deploy — but such a machine
   deploys on the prior for those kernels, and the prior can be wrong about the ordering it is used for.
2. **Raising the feature version silently changes what a machine deploys.** As described above: a refused weights
   file leaves the compile with no prior, no warning at compile time, and the only symptom is that deployments get
   worse.
3. **A fresh machine deploys mostly on prediction.** Only golden configurations travel with a clone. Where no golden
   covers a shape, a rented box is choosing from a model that has never seen that card's measurements.
4. **A cold compile changes which kernels exist on a prediction it cannot check.** Structural choices are costed as
   sums of per-kernel prices; where nothing measured prices a piece, the prior does, and its absolute error does not
   cancel across kernel families. The 1.8-times-faster split on [the goldens page](./07-golden-configurations.md)
   deploys with confidence only because somebody recorded it: its recorded decision, priced from the pieces'
   measured rows, outranks any priced sum.
5. **A recording that no longer realizes is simply not evidence**, and the kernel falls to the prior, which can be
   far slower than the number the recording advertises. `--strict-evidence` makes that fall-through an error, and
   the release gate compiles the serving matrix under it; a plain deploy without the flag falls through silently.
6. **There is no per-fork report of which row decided.** Answering "which evidence answered this fork, and did I
   expect that one?" means correlating warnings, the resolution record and the release gate.
7. **The measured pools are diagnostic-only.** The dataset database is never consulted when deploying, and the
   offline prior trains only on the golden rows in it — fitting it on the measured pools too is a planned path, not
   a current one.
8. **Nothing evaluates a fork no bench ever took.** Both views score configurations that were built and offered
   as candidates. A measured row exists only for a configuration somebody pinned or the compiler picked, and a fork
   nothing took leaves no row anywhere — a good configuration sitting past one is silence that reads as health.

## See it yourself

The measured view reads the dataset exported from the dataset database, filled from a freeze or from a tuning database:

```bash
emmy db import --db _data/tune.db ~/.cache/emmy/autotune.db
emmy db export --db _data/tune.db _data/tune
emmy eval prior _data/tune --pools measured
```

And a candidate weights file can be scored without touching the installed one, which is how two fits are judged
against each other, with `--json` writing the report in the same shape a fit records:

```bash
emmy eval prior _data/tune --offline-file /tmp/candidate-weights.json --json /tmp/candidate.json
```

## Where to go next

You have now seen the whole path: a model becomes a graph, the graph is rewritten pass by pass, a handful of those
rewrites offer several correct answers, and each of those is settled by the best evidence available — a reviewed
measurement if one exists, a local measurement if one was taken, a prediction otherwise, and a safe default when there
is nothing at all.

For the level of detail below this series, the reference documents live beside the code: `ARCHITECTURE.md` in
`emmy/compiler/pipeline/` for the pipeline itself, and its `passes/` sibling for the rewrite rules. The vocabulary is
defined in `GLOSSARY.md` at the root of the repository.
