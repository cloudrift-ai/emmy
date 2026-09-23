"""``Sample`` — one measured-or-recorded ``(config, latency, identity)`` row,
the common currency over all three measurement-data sources.

A golden config, a tune-DB ``perf`` row, and a online-prior reservoir row are all
the same thing once normalized: a tunable-knob dict, a measured latency, a
structural identity, and (for golden) a reference latency. ``Sample`` is that
normal form. The split into ``knobs`` (tunable) / ``context`` (``H_*``) /
``s_features`` (``S_*``) is by key prefix and therefore lossless — :meth:`all_knobs`
re-merges them to the exact original dict, and :meth:`features` runs the single
featurizer (:func:`features.knob_features`) on that merge, so a ``Sample`` reproduces
the feature vector each source built inline today.

Featurization fidelity (the load-bearing invariant): a trained ``OnlinePrior``
regresses on the full ``S_*`` histogram stamped by
the ``IdentityStrategy``. DB / prior rows carry that histogram inline;
golden rows derive it by lowering their embedded frontend program and selecting
the target through provenance. Neither the histogram nor ``ShapeKey`` is part of
the stable golden format.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

from emmy.compiler.pipeline.knob import CTX_PREFIX, IDENTITY_PREFIX, METADATA_PREFIXES, STRUCT_PREFIX
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.pipeline.search.features import knob_features


@functools.cache
def _card_features(gpu_name: str, cc: int) -> dict[str, float]:
    """The ``H_*`` features of ``gpu_name`` at compute capability ``cc`` (``H_cc`` encoding), from the
    registry's memorized specs — the recipe :meth:`Sample.from_golden` uses for a golden's own card."""
    from emmy.compiler.context import Context  # noqa: PLC0415

    return Context.from_target(divmod(cc, 10), gpu_name=gpu_name).features()


def measured_features(row) -> dict:
    """The full feature dict a measured ``perf`` row featurizes as: its card's ``H_*``, the opt level it
    was measured under, and its stored ``S_*`` stamps + tunables. A row stores no ``H_*`` of its own —
    they are a function of the card, derived here by live code so a stored row outlives a change to
    the device features. Only for rows :func:`~.freeze.freeze_reason` admits (a registry card)."""
    return {**_card_features(row.gpu, row.cc), "H_opt": float(row.opt), **row.knobs}


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
    ``source`` ∈ ``{"golden", "db", "prior"}`` marks provenance for the
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

    def features(self) -> dict[str, float]:
        """The flat numeric feature vector the priors regress on — the single
        featurizer over the merged dict. Merge order ``context, s_*, knobs`` matches
        the inline construction the eval / prior code used (knobs win on collision,
        though the prefixes are disjoint)."""
        return knob_features(self.all_knobs())

    @classmethod
    def from_golden(cls, cfg, *, compile_s_feats: bool = False) -> Sample:
        """A program-backed golden record as a normalized measurement sample.

        ``compile_s_feats`` remains an accepted no-op for callers that used to
        request snippet compilation. Structural features are lazily derived from the
        embedded program and provenance target.
        """
        from emmy.compiler.context import Context  # noqa: PLC0415

        tunable, _ctx, _s = _split_by_prefix(cfg.knobs)
        return cls(
            knobs=tunable,
            latency_us=cfg.emmy_us,
            shape=cfg.shape_key,
            name=cfg.name,
            dtype=cfg.dtype,
            ref_us=cfg.reference_us,
            pins=cfg.pin_map,
            # gpu_name pins the device-physical features (H_sm_count / smem / …) to
            # the golden's OWN card's memorized specs, not the live device's — so a
            # PRO 6000 golden ranked on a 5090 (both cc 12.0) gets 188 SMs, not 170.
            context=Context.from_target(cfg.compute_cap, gpu_name=cfg.gpu_name).features(),
            source="golden",
            s_full=dict(cfg.structural_features),
        )

    @classmethod
    def from_perf_row(cls, row, name: str | None) -> Sample:
        """A DB ``perf`` row (:class:`db.PerfRow`) as a ``Sample``: the recorded knob dict split by
        prefix, and ``name`` the C identifier of its kernel row (for per-knob regret grouping;
        ``None`` when the caller has none)."""
        tunable, ctx, s = _split_by_prefix(row.knobs)
        return cls(knobs=tunable, latency_us=row.stats.median, name=name, context=ctx, source="db", s_full=s, error=row.error)

    @classmethod
    def from_prior_row(cls, knobs: dict, latency_us: float) -> Sample:
        """A online-prior reservoir row ``(stamped_knobs, latency)`` as a ``Sample``.
        The reservoir dicts already carry ``S_*`` / ``H_*`` inline (stamped by the
        live pipeline), so the split + re-merge is lossless for grouping / scoring."""
        tunable, ctx, s = _split_by_prefix(knobs)
        return cls(knobs=tunable, latency_us=latency_us, context=ctx, source="prior", s_full=s)
