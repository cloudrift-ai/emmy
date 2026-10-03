"""``Sample`` — one measured-or-recorded ``(config, latency, identity)`` row,
the common currency over both measurement-data sources.

A golden config and a tune-DB ``perf`` row are the same thing once normalized: a tunable-knob dict, a measured latency, a
structural identity, and (for golden) a reference latency. ``Sample`` is that
normal form. The split into ``knobs`` (tunable) / ``context`` (``H_*``) /
``s_features`` (``S_*``) is by key prefix and therefore lossless — :meth:`all_knobs`
re-merges them to the exact original dict.

Featurization fidelity (the load-bearing invariant): the prior scores the full ``S_*``
histogram stamped by the ``IdentityStrategy``, which every row carries inline.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

from emmy.compiler.pipeline.knob import CTX_PREFIX, IDENTITY_PREFIX, METADATA_PREFIXES, STRUCT_PREFIX
from emmy.compiler.pipeline.search.dataset.shape import ShapeKey
from emmy.compiler.pipeline.search.features import Featurizer


@functools.cache
def _card_features(gpu_name: str, cc: int) -> dict[str, float]:
    """The ``H_*`` features of ``gpu_name`` at compute capability ``cc`` (``H_cc`` encoding), from the
    registry's memorized specs — a row's own card's, never the live device's."""
    from emmy.compiler.context import Context  # noqa: PLC0415

    return Context.from_target(divmod(cc, 10), gpu_name=gpu_name).features()


def measured_features(row, kernel) -> dict[str, float]:
    """The feature row of a measured ``perf`` row of ``kernel`` (its op): its stored tunables, under its card's
    ``H_*`` and the opt level it was measured under. A row stores no ``H_*`` of its own — they are a function of the
    card, derived here by live code so a stored row outlives a change to the device features. Only for rows
    :func:`~.freeze.freeze_reason` admits (a registry card)."""
    return Featurizer({**_card_features(row.gpu, row.cc), "H_opt": float(row.opt)}).features(kernel, row.knobs)


def _split_by_prefix(knobs: dict) -> tuple[dict, dict, dict]:
    """Split a stamped knob dict into ``(tunable, context H_*, structural S_*)`` by
    key prefix. Disjoint prefixes → re-merging is lossless."""
    ctx = {k: v for k, v in knobs.items() if k.startswith((CTX_PREFIX, IDENTITY_PREFIX))}
    s = {k: v for k, v in knobs.items() if k.startswith(STRUCT_PREFIX)}
    tunable = {k: v for k, v in knobs.items() if not k.startswith(METADATA_PREFIXES)}
    return tunable, ctx, s


@dataclass(frozen=True)
class Sample:
    """One ``(config, latency, identity)`` row, normalized across sources.

    ``knobs`` holds *only* tunable knobs (``S_*`` / ``H_*`` live in ``s_full`` /
    ``context``); ``pins`` holds the input knob regime for a golden replay and is
    empty for measurement rows from other sources. ``shape`` is the arithmetic
    identity; ``ref_us`` is the cuBLAS / torch reference (golden only, ``None``
    elsewhere); ``name`` carries the kernel C identifier for DB rows.
    ``source`` ∈ ``{"golden", "db"}`` marks provenance for the
    orthogonality fail-fast (``dataset_args.require_source``)."""

    knobs: dict
    latency_us: float
    shape: ShapeKey | None = None
    name: str | None = None
    dtype: str | None = None
    ref_us: float | None = None
    pins: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    source: str = "db"
    s_full: dict | None = None  # full compiled/derived S_* histogram when known
    error: str | None = None  # bench_fail failure text (db rows only; None on ok rows)
    # Optional exact work count. The intensity-floor gate reads THIS, not a ShapeKey reconstruction: the join key
    # excludes symbolic axes on the matmul side but includes them on the reduce-tier side, so no
    # one hint-multiplier formula over it can be right for both.
    flops: float | None = None

    def s_features(self) -> dict[str, float]:
        """The ``S_*`` features this sample featurizes on: the full stamped histogram
        when known, else the cheap arithmetic extents, else nothing."""
        if self.s_full is not None:
            return self.s_full
        return self.shape.s_features_arith() if self.shape is not None else {}

    def all_knobs(self) -> dict:
        """The original stamped dict — ``context ∪ s_features ∪ knobs``. For a DB
        row this re-merges to exactly the recorded ``perf.knobs``; the per-knob
        regret analysis iterates this so its output is unchanged."""
        return {**self.context, **self.s_features(), **self.knobs}
