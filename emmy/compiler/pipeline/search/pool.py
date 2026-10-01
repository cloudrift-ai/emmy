"""Sampling a candidate pool DURING enumeration — the size, the draw, and what must survive it.

Building the offline-prior dataset enumerates every golden's candidate pool and featurizes every
row. The corpus is millions of rows and tens of gigabytes, paid again on every experiment, and one
golden's pool alone exceeds the enumerator's materialization budget — so an unsampled fit does not
finish, and a walk that visits every leaf to draw a few (the reservoir this replaced) still pays
O(pool) time per golden: an hour over the hardware goldens, most of it in schedule-context extensions
nothing retained. The draw is therefore taken at the FORK level: ``size`` seeded random descents through
the lazy schedule tree (:func:`~emmy.compiler.pipeline.fork.descent_sample`), a child at random at every
branch, so the cost is ``size`` paths and never the pool. What is bounded is TIME as well as memory, and
what is given up is uniformity: a narrow subtree is over-represented, so a rank within the draw is a rank
within the draw, not an estimate of the rank in the pool.

**The draw is a pure function of the tree and** ``(size, seed, keep)``. Every leaf expansion is
deterministic and the descents are seeded on the sample's own identity, so two byte-identical pools draw
byte-identical samples — which is what keeps the fit reproducible and keeps two goldens over one pool
mergeable into one training group.

**Membership survives the draw exactly.** ``keep`` holds the rows the draw may not lose — the golden rows
recorded on the pool's card and regime — and each is reached by its own directed descent
(:func:`~emmy.compiler.pipeline.fork.leaf_for`), the row-to-leaf walk the decode and the evidence pick share,
so a golden that is genuinely absent from its pool still reads as absent: a real defect class (a pin or
dtype mismatch) the fit and ``eval golden`` both detect by exactly that miss.

**Reported rank is the rank within the draw, and the total beside it is the draw's size**
(:class:`Candidates`): the descents never learn how big the pool is, and a rank is only interpretable next
to what it was ranked among.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.structural import digest

#: Candidates drawn per pool when ``emmy fit`` samples. Measured against the alternative: at this
#: size the linear trainer's z-scored moments and its rank objective both track the full-pool fit,
#: while the corpus fits in memory and builds in minutes rather than an hour.
DEFAULT_SAMPLE = 2000


@dataclass(frozen=True)
class Candidates:
    """One graph's enumerated candidate rows beside what they were ranked among.

    ``total`` equals ``len(rows)`` when nothing was sampled, and the draw's size otherwise — the two travel
    together because a rank is only interpretable next to what it was ranked among.
    """

    rows: list[dict]
    total: int


@dataclass(frozen=True)
class PoolSample:
    """How many candidates a pool contributes, which rows must survive, and where each pool reports its
    size.

    Carried on :class:`~emmy.compiler.context.Context` and folded into the schedule-space stamp.
    That stamp distinguishes a sampled draw from the live space while keeping equal sampled
    problems reproducible. ``None`` on the Context means live, and live never samples.
    """

    #: Complete rows to draw. ``0`` (or a pool whose declared bound is no larger) means the whole pool.
    rows: int
    seed: int = 0
    #: The rows the draw may not drop, each as its sorted ``(knob, value)`` items — the golden rows.
    keep: tuple = ()
    #: Where each drawn pool reports its size, keyed by that pool's schedule-space stamp. The sampled
    #: rows cannot carry it and the fork tree has no channel for it, so the enumerator writes here
    #: and the caller that asked for the sample reads it back. Keyed rather than appended so a
    #: equal pool overwrites instead of double-counting. EXCLUDED from the value
    #: (``compare=False``): a sink is not part of a sample's identity.
    totals: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    @property
    def key(self) -> str:
        """This sample's stable identity: the size, the seed and the kept rows, sorted — the one spelling
        of one sample on every machine."""
        return digest(self.rows, self.seed, sorted(str(row) for row in self.keep))

    def draw(self, options) -> list:
        """The leaves this sample takes from the lazy tree ``options``: the whole pool when nothing is to be
        sampled or its declared bound fits the draw, else ``rows`` descents seeded on the size and seed alone,
        deduplicated by row, with every kept row's own leaf beside them — the keep-set adds to the draw and never
        moves it. Leaves, in a stable order: the kept rows first, then the draw."""
        from emmy.compiler.pipeline.fork import descent_sample, iter_leaves, leaf_for, leaf_knobs  # noqa: PLC0415

        bounds = [getattr(option, "pool_bound", None) for option in options]
        if self.rows <= 0 or (bounds and all(bound is not None and bound <= self.rows for bound in bounds)):
            return list(iter_leaves(options))
        seen: set = set()
        out: list = []
        kept = [found[0] for row in self.keep if (found := leaf_for(options, dict(row))) is not None]
        for leaf in [*kept, *descent_sample(options, draw=self.rows, seed=digest(self.rows, self.seed))]:
            identity = tuple(sorted((str(k), str(v)) for k, v in leaf_knobs(leaf).items()))
            if identity not in seen:
                seen.add(identity)
                out.append(leaf)
        return out


__all__ = ["DEFAULT_SAMPLE", "Candidates", "PoolSample"]
