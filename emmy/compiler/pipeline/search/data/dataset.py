"""``Dataset`` — a queryable read-view over a bag of :class:`Sample`s, with one
adapter per measurement-data source (a DB instance's ``perf`` rows / the online-prior
reservoir) and the two grouping axes the consumers need.

The two groupings are deliberately distinct and do **not** collapse:

- :meth:`group_by_op` keys on the full ``S_*`` structural signature — two different
  shapes are different groups. The golden joins in ``prior/diagnostics.py`` index on it.
  It is deliberately NOT a comparison key: it carries no card and no ``H_opt``, so rows
  measured on different hardware or under different nvcc settings land in one group.
  Anything ranking measured latencies wants ``data/group.group_measured`` instead.
- :meth:`group_by_kernel_name` keys on the kernel C identifier (parsed from
  ``cuda_op.pretty``) — which *merges* shapes of the same kernel, by design, so the
  per-knob regret analysis measures relative knob impact across shapes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from emmy.compiler.pipeline.search.data.sample import Sample


class Dataset:
    """A bag of :class:`Sample`s plus source adapters + grouping."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    # --- adapters: one per source -----------------------------------------

    @classmethod
    def from_db(
        cls, path: Path | str, *, kernel: str | None = None, min_latency: float = 0.0, backend: str | None = None, status: str = "ok"
    ) -> Dataset:
        """Every measured ``ok`` variant in a DB instance's ``perf`` rows, opened
        read-only so a concurrent ``tune`` writer isn't blocked. ``backend=None``
        spans every backend (matching the legacy ``eval knobs`` query); ``kernel``
        filters on the parsed C identifier. ``status`` selects the row status
        (``bench_fail`` rows carry the watchdog-timeout sentinel latency, so the
        default ``min_latency`` admits them)."""
        from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

        db = SearchDB.open_readonly(path)
        try:
            samples = [
                Sample.from_perf_sample(ps) for ps in db.iter_perf_samples(backend=backend, status=status, min_latency_us=min_latency)
            ]
        finally:
            db.close()
        if kernel:
            samples = [s for s in samples if s.name and kernel in s.name]
        return cls(samples)

    @classmethod
    def from_prior(cls, prior) -> Dataset:
        """The online prior's bounded reservoir as samples. Works through
        ``FallbackPrior`` (it delegates ``_dataset`` to the online half)."""
        return cls([Sample.from_prior_row(k, v) for k, v in prior._dataset])

    # --- grouping ----------------------------------------------------------

    def group_by_op(self) -> dict[tuple, list[Sample]]:
        """Group by the full ``S_*`` structural signature (sorted items) — the key
        the prior diagnostics group on, so structurally-distinct same-extent ops
        stay separate."""
        g: dict[tuple, list[Sample]] = defaultdict(list)
        for s in self.samples:
            g[tuple(sorted(s.s_features().items()))].append(s)
        return dict(g)

    def group_by_kernel_name(self, *, min_variants: int = 1, kernel: str | None = None) -> dict[str, list[Sample]]:
        """Group by kernel C identifier (``cuda_op.pretty``), dropping samples with
        no identity (golden / prior rows) and groups below ``min_variants``."""
        g: dict[str, list[Sample]] = defaultdict(list)
        for s in self.samples:
            if s.name is None or (kernel and kernel not in s.name):
                continue
            g[s.name].append(s)
        return {k: v for k, v in g.items() if len(v) >= min_variants}
