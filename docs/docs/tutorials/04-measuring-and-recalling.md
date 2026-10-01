---
sidebar_position: 4
title: "4. Measuring and Recalling"
description: The two ways a fork is answered — measure it now, or recall what was measured before — and the three places knowledge is kept.
keywords: [Emmy, autotuning, greedy selection, evidence, tuning database, regime]
---

# 4. Measuring and Recalling

A fork offers several correct options. Something has to pick one. There are exactly two situations in which that
happens, and they could hardly be more different.

## Two situations

**`emmy run --bench` has a GPU and time to spend.** It can build a kernel for a pinned option, run it, time it, and
keep the number. It compares the compiler's own pick with the rows it was asked to measure, and it writes down every
clean measurement. A bench takes minutes.

**`emmy compile` and `emmy run` measure nothing.** An ordinary compile has to produce a program now. It cannot build
four kernels to find out which is faster; it picks one, in a fraction of a second, and moves on to the next fork.
Choosing the option that currently looks best without exploring alternatives is called **greedy selection**, and it is
what every deployment does.

The interesting problem is the second one. An ordinary compile can only *use* knowledge; it can never create any. So
everything depends on what was recorded earlier, and on where.

## The three stores

Three stores hold everything a compile can know. Telling them apart is the single most useful thing to learn early,
because they have different writers, different readers and different lifetimes.

| Store | Where it lives | Written by | Read by |
| --- | --- | --- | --- |
| **Golden configurations** | model files under `recipes/<model>/golden/`, one per exact GPU; model-agnostic files under compiler search | promoted from measured comparisons | an ordinary compile, first of all; also, through the dataset database, the training data for the offline prior |
| **Measurements table** | the tuning database, `~/.cache/emmy/autotune.db` | `emmy run --bench`, one row per benchmarked kernel — the compiler's own pick and every hand-forced row; also the golden configurations in scope, imported before a compile picks | an ordinary compile, together with the golden rows; and as a cache, so a configuration already measured is never re-run |
| **Dataset database** | a file of its own that every `emmy db` command names (`_data/dataset.db` in the examples), the same tables | `emmy db import`, from measurement freezes, the golden configuration files and tuning databases | `emmy db export`, which writes the dataset (`_data/dataset`, a manifest beside one matrix file per pool) the `emmy eval` views and the offline fit read — **never** consulted when compiling |

The last row surprises people. The dataset database holds the same kind of rows as the tuning database — every
benchmarked configuration, failures included — but it is filled from a pinned snapshot rather than from this
machine's benches, and it is deliberately not consulted when deciding what to deploy. It exists to answer questions
about the prior and the measurements themselves, which is what the [last page](./09-storage-checks-and-limits.md) is
about.

```
WRITERS                                     STORES                                READERS

emmy run --bench, the pick and hand-forced rows ──▶ measurements table ──────────▶ ordinary compile

emmy db import, snapshots/goldens/tune DBs ──▶ dataset database ──▶ emmy db export ──▶ dataset ──▶ emmy eval
                                                                                               └── emmy fit ──▶ offline prior weights ──▶ ordinary compile

recorded by hand from those rows ─────────▶ golden configuration files ────────▶ ordinary compile
```

## Only one of them travels

Of the three, **only the golden configurations are in the repository**. The tuning database is a cache under
`~/.cache/emmy` on whichever machine ran the bench.

That has a consequence worth pausing on. A freshly rented GPU box has: the golden configuration files, and the
weights of the offline prior that also ship with the repository. It has no measurements of its own, and nothing local
to fall back on. Every fork on that machine is answered either by a recorded golden configuration or by a model's
prediction. This is the normal case, not an edge case — it is what happens every time somebody rents a machine to
serve a model. It is also why the golden files matter as much as they do, and why they get [a page of their
own](./07-golden-configurations.md).

## Measurements are not interchangeable

One more thing has to be introduced here, because everything after this page depends on it: a measurement is only
true of the settings it was taken under. Those settings are called the **regime**, and the part of it that matters
most is the optimization level the CUDA compiler ran at.

`-O3` is the **deployable** setting. It is what `emmy compile` and `emmy run` use, so it is what a served model
actually runs — and it is what `emmy run --bench` measures at too. **Emmy measures in the regime it deploys
into**, so a recorded latency is the latency you get.

That sounds too obvious to state, so it is worth saying why it needs stating. An earlier search benched thousands
of configurations, and a cheaper compiler setting (`-Xcicc -O1`) was once used to make that affordable, on the
assumption that it would still *rank* correctly even if the absolute numbers were off. It did not. The cheap
setting's error was not random noise but a systematic bias along tile size: it made big register tiles look slow,
which is exactly the family it was most important to get right. Ranking in a regime you do not deploy in means the
winner of the search need not be the winner in production.

The general lesson outlives the specific setting: **a measurement is only evidence about the conditions it was taken
under.** Emmy therefore keeps the regime on every stored measurement and gates on it, so a number taken under some
other setting is never silently read as if it applied here. If you deliberately pin a different optimization level
with `--nvcc-flags`, the bench still runs and still records — but under that regime's own identity, where no ordinary
compile will read it. You will get a warning saying so.

## Where this is going

The next page is the one that answers the question the series opened with — given all of this, in what order does
an ordinary compile consult it?

## See it yourself

Look at what a machine that has benched actually has:

```bash
ls -la ~/.cache/emmy/
```

Model golden configuration files live beside their recipes, one file per exact GPU model and compute capability:

```bash
find recipes -path '*/golden/*.json' -print
```

The central `emmy/compiler/pipeline/search/golden/` directory contains only model-agnostic hardware goldens.

Measure one shape; each clean kernel row lands in the tuning database (this one needs a GPU):

```bash
emmy run --bench -c "torch.nn.Softmax(dim=-1)(torch.randn(1, 28, 2048, 2048))"
```

Run it a second time and the compile picks from the row it just recorded.

Next: [6. The deploy evidence hierarchy](./06-deploy-evidence-hierarchy.md).
