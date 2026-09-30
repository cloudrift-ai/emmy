"""``Samples`` — a queryable read-view over a bag of :class:`Sample`s, built from a DB instance's
``perf`` rows, with the two grouping axes the consumers need.

The two groupings are deliberately distinct and do **not** collapse:

- :meth:`group_by_op` keys on the full ``S_*`` structural signature — two different
  shapes are different groups. It is deliberately NOT a comparison key: it carries no card and no ``H_opt``, so rows
  measured on different hardware or under different nvcc settings land in one group.
  Anything ranking measured latencies wants ``db/export.measured_groups`` instead.
- :meth:`group_by_kernel_name` keys on the kernel C identifier (the ``kernel`` row's
  name) — which *merges* shapes of the same kernel, by design, so the per-knob regret
  analysis measures relative knob impact across shapes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from emmy.compiler.pipeline.search.dataset.sample import Sample


class Samples:
    """A bag of :class:`Sample`s plus source adapters + grouping."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    # --- adapters: one per source -----------------------------------------

    @classmethod
    def from_rows(cls, rows, names: dict[str, str], *, kernel: str | None = None, min_latency: float = 0.0, status: str = "ok") -> Samples:
        """Every ``perf`` row of ``status`` among ``rows`` as samples, ``names`` the kernel rows' C identifiers by
        exact identity (``SearchDB.kernel_names``); ``kernel`` filters on that identifier. ``bench_fail`` rows carry
        the watchdog-timeout sentinel latency, so the default ``min_latency`` admits them. The caller opens the DB
        (``commands/db.read_samples``): this package describes rows and never reads a store."""
        samples = [
            Sample.from_perf_row(row, names.get(row.kernel)) for row in rows if row.status == status and row.stats.median > min_latency
        ]
        if kernel:
            samples = [s for s in samples if s.name and kernel in s.name]
        return cls(samples)

    # --- grouping ----------------------------------------------------------

    def group_by_op(self) -> dict[tuple, list[Sample]]:
        """Group by the full ``S_*`` structural signature (sorted items), so structurally-distinct
        same-extent ops stay separate."""
        g: dict[tuple, list[Sample]] = defaultdict(list)
        for s in self.samples:
            g[tuple(sorted(s.s_features().items()))].append(s)
        return dict(g)

    def group_by_kernel_name(self, *, min_variants: int = 1, kernel: str | None = None) -> dict[str, list[Sample]]:
        """Group by kernel C identifier (the ``kernel`` row's name), dropping samples with
        no name (golden rows) and groups below ``min_variants``."""
        g: dict[str, list[Sample]] = defaultdict(list)
        for s in self.samples:
            if s.name is None or (kernel and kernel not in s.name):
                continue
            g[s.name].append(s)
        return {k: v for k, v in g.items() if len(v) >= min_variants}
