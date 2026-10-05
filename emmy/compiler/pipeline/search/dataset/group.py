"""``Group`` — one candidate pool plus its labels: the candidate pools every ranking question is asked over.

A group is one shape's featurized candidate pool on one card, with whatever supervision exists for it. The
base carries the pool and its identity and nothing about supervision, because the two kinds of it are not
one thing: :class:`GoldenGroup` holds the row INDICES goldens verified, so it alone can answer which rows are
the answer, and :class:`MeasuredGroup` holds a benched latency PER ROW, so it alone can answer what a wrong
pick cost. Nothing generic reads either — the metrics take plain sequences and never see a group — so a
single label column on the base would be a field with no consumer and two mutually exclusive meanings.

Groups are built on the DB side (``db/export.py``: ``measured_groups`` for benched rows, and
``ranking.build_golden_groups`` over the golden pools the export reads — a golden pool is one kernel's schedule
space, which the builder has to enumerate before it is a group) and travel as a :class:`~.document.Dataset`; the
trainers and the fold harness consume them through this one shape. There is no iterator/batching layer, the whole
dataset is a small in-memory list.

It lives with the other data types rather than under ``prior/fit/`` because a candidate pool is data, not a
fitter detail — the fit is only its first consumer. The planned evaluation reports rank over the same pools, and
a second pool representation for them would be a second chance to disagree about what a candidate set IS.
Nothing here imports a model: a group carries the columns and the labels, and each model class decides for
itself which columns it wants (see :meth:`Group.matrix`).

Rows are ndarray-backed, not dict-backed: ``feats`` is one float64 matrix (rows × ``feat_names``), packed
once by :func:`pack_features` from the builder's transient per-row feature dicts. A per-row dict of ~63
floats costs ~4 KB; the full golden dataset is ~2.5 M rows, and the dict representation (~10 GB) OOM-killed whole
fit runs — the matrix representation is ~20× smaller.

An absent feature is stored as ``NaN``, the model's own missing bucket: a tree can branch on not-decided as a
state of its own.

``key`` is ``"<gpu>/<name>"`` of the first golden the builder found in this pool, disambiguated when one
name opens several distinct pools (``#2``, ``#3``, … in dataset order); the other goldens over that pool are
further entries in its label set rather than groups of their own — the builder resolves that before it
constructs anything, so a group is built once, already knowing every verified row it holds. ``tier`` is the
fit's case tier (``thread`` / ``warp`` / ``dyn`` / ``reduce`` / ``pointwise``) and is a REPORT LABEL only — it
decides nothing. ``dynamic`` is the regime, read off the routing stamp exactly as the deployed prior reads it.
``shape`` is the cross-validation fold group, and is the one identity here that decides something structural: two
goldens sharing it enumerate the same candidates, so a fold that separated them would train on the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from emmy.compiler.pipeline.search.dataset.pool import GoldenPool
from emmy.compiler.pipeline.search.features import ROUTING_FEATURES, is_dynamic_row

# The default feature view: the ``D_*`` geometry/occupancy features, the two ``MMA_*`` atom features that vary
# between a pool's candidates — ``MMA_tier`` (the warp/scalar tier discriminator) and ``MMA_acc_bits`` (32 for the
# f32-accumulate atom, 16 for the f16-accumulate one) — and ``H_cc``, the card's compute capability. ``H_cc`` is
# constant within a pool, so on its own it moves no ranking; a tree splits on it to rank the same candidates
# differently per architecture (f16 accumulation pays on Ada and Blackwell, not on Volta). Cross-validated over the
# repository goldens it lifted held-out top-1 from 250 to 263 of 733 pools.
DEFAULT_FEATURES = "D_*,MMA_tier,MMA_acc_bits,H_cc"
# Placement arms can rank differently across cards and regimes. Although hardware and regime facts are constant
# within a fork, the tree uses them to condition its ranking on capability, the physical SKU and fast math.
PLACEMENT_FEATURES = "P_*,H_cc,H_total_mem,H_fast_math"


def pack_features(feats: list[dict[str, float]]) -> tuple[tuple[str, ...], np.ndarray, bool]:
    """Per-row feature dicts as ``(feat_names, matrix, dynamic)`` — the stored representation, without the
    labels. Separate from :class:`Group` because a builder needs it BEFORE it can say what a pool's labels
    are: two goldens share a pool when their packed rows are identical, so the packing is what decides
    which groups exist, and only then is it known how many verified rows each one has.

    Callers drop the dicts after this. That is the whole point of packing — a per-row dict of ~63 floats
    costs ~4 KB, and holding the corpus in that form OOM-killed whole fit runs."""
    names = tuple(sorted({k for f in feats for k in f}))
    return names, feature_matrix(feats, list(names), fill=np.nan), bool(feats) and is_dynamic_row(feats[0])


def _matcher(pats: list[str]):
    """``name -> bool`` over a pattern list: exact names, or a trailing ``*`` making a prefix glob."""
    prefixes = tuple(p[:-1] for p in pats if p.endswith("*"))
    exact = frozenset(p for p in pats if not p.endswith("*"))
    return lambda name: name in exact or (bool(prefixes) and name.startswith(prefixes))


def feature_view(spec: str):
    """A feature-view spec — comma-separated feature names, a trailing ``*`` making a prefix glob, and a
    leading ``-`` excluding what a later pattern would otherwise have kept (``"D_*,-D_near_*"``) — parsed
    into a ``keep(name) -> bool`` predicate. The view a fit trained under is recorded in its metrics header
    and artifact provenance, so two fits are only comparable when the recorded specs match.

    Exclusions exist so a view can be written as "everything, minus what this model has no use for". Written as
    an include list instead, such a view would silently go stale the
    moment the featurizer gained a feature — the new column would be dropped without anyone deciding to
    drop it. Excluding is the safe direction: an unforeseen feature arrives in the view, where at worst the
    model ignores it.

    :data:`~..features.ROUTING_FEATURES` are kept by EVERY view, named or not, and cannot be excluded.
    The packed column is what the model splits the two regimes on, so a view that dropped it would price every
    symbolic-axis pool as a static one, silently."""
    pats = [p.strip() for p in spec.split(",") if p.strip()]
    keep = _matcher([p for p in pats if not p.startswith("-")])
    drop = _matcher([p[1:] for p in pats if p.startswith("-")])
    return lambda name: name in ROUTING_FEATURES or (keep(name) and not drop(name))


def feature_matrix(feats: list[dict[str, float]], names: list[str], *, fill: float = 0.0) -> np.ndarray:
    """Feature-dict rows as a dense float64 matrix over ``names`` — absent key = ``fill``."""
    return np.array([[f.get(n, fill) for n in names] for f in feats], dtype=float)


@dataclass(frozen=True)
class Group:
    """One featurized candidate pool, plus the identity the fold axes and metrics keys need."""

    key: str
    name: str
    tier: str
    gpu: str
    # The cross-validation fold group: this pool's extent identity — spelled by the golden builder from the
    # source record's ``ShapeKey``, and by :func:`measured_groups` as the op's own signature. Goldens sharing it
    # compete over the same candidates, so they must be held out TOGETHER by the fold harness. Unlike ``gpu``
    # (a report axis) this decides folds; unlike ``tier`` (a label) it is load-bearing.
    shape: str
    dynamic: bool
    feat_names: tuple[str, ...]
    feats: np.ndarray = field(repr=False)
    # The size of the candidate pool ``feats`` was drawn from, equal to ``len(feats)`` unless the
    # builder sampled. A report needs it and cannot recover it from the matrix: it must print the raw
    # sample rank BESIDE the true total rather than scaling it.
    total: int
    # The last ``matrix()`` projection, ``(names, array)`` — a cache, not part of the group's value, so
    # it stays out of ``__eq__`` / ``repr``. See :meth:`matrix`.
    _cache: tuple | None = field(default=None, repr=False, compare=False)

    def matrix(self, names: list[str]) -> np.ndarray:
        """The pool projected onto ``names`` — column ``j`` is the stored ``names[j]`` column, or ``NaN`` where
        the value is absent, which happens two ways: the pool never stamped that feature at all, or a row inside
        the pool lacks a key its siblings carry. ``NaN`` is the model's missing bucket: "not decided / not
        stamped" is a state a split can branch on, distinct from a knob that is present and legitimately zero.

        **The result is memoized and READ-ONLY.** Building it is a strided column copy of the whole pool, and a
        cross-validated run asks for the same projection over and over — once per fold's fit and again per
        fold's scoring pass. Measured on the golden dataset, those copies were 85% of a fit's wall time. One
        entry is enough: a fit uses one column list for its whole run, so a different ``names`` simply evicts the
        previous one instead of growing without bound. Mutating the shared array raises."""
        want = tuple(names)
        if self._cache is None or self._cache[0] != want:
            idx = {n: j for j, n in enumerate(self.feat_names)}
            out = np.full((len(self.feats), len(names)), np.nan, dtype=float)
            for j, n in enumerate(names):
                if n in idx:
                    out[:, j] = self.feats[:, idx[n]]
            out.flags.writeable = False
            object.__setattr__(self, "_cache", (want, out))  # frozen dataclass; the cache is not part of its value
        return self._cache[1]


@dataclass(frozen=True)
class MeasuredGroup(Group):
    """A pool whose labels ARE the measurement — one benched latency in microseconds per row, lower = faster.

    The counterpart of :class:`GoldenGroup`: there the labels mark which rows are the answer, here they say how
    fast each row ran. What the two kinds have in common is all the base needs, which is why a ranking metric
    takes the base and this class adds only the one fact the base has nowhere to put.

    :attr:`h_opt` is the compile regime every row in the pool was measured under. It is part of the grouping key
    (:func:`measured_groups`) rather than a description of it: ``-O1`` and ``-O3`` reorder the same candidates
    often enough that a pool spanning both would measure neither, so the regime is what makes these rows
    comparable at all. A report reads it as an axis, and it is a FIELD rather than a lookup into the packed
    ``H_opt`` column for a mechanical reason: :meth:`Group.matrix` memoizes exactly one projection, so asking
    for the single-column one right before a model asks for its own would evict and rebuild the model's
    projection for every group — two full strided copies each, on the path whose measurements made that cache
    exist in the first place."""

    # One benched latency in MICROSECONDS per row, lower = faster. Lives here rather than on the base because
    # nothing generic reads it: the metrics take plain sequences and never see a group, so a label on the base
    # would be a column with no consumer and two mutually exclusive meanings.
    latency_us: np.ndarray = field(kw_only=True, repr=False)
    h_opt: float = field(kw_only=True)

    @classmethod
    def from_measured(
        cls, key: str, gpu: str, op_sig: str, h_opt: float, latency_us: Sequence[float], feats: list[dict[str, float]]
    ) -> MeasuredGroup:
        """A pool where every row was benched, labelled with its own measured µs.

        ``op_sig`` serves as both ``name`` and ``shape`` — a measured pool IS one op, so there is no second
        identity to spell — and ``tier`` is empty, because the golden tiers are a closed report vocabulary
        about how a CASE was built and a sixth value would mean nothing to the two places that read it.

        No ``total`` either: a measured pool is not a sample of a larger enumeration, it is exactly the
        configs someone ran, and a rank against anything else would be against candidates never measured."""
        names, matrix, dynamic = pack_features(feats)
        return cls(
            key,
            op_sig,
            "",
            gpu,
            op_sig,
            dynamic,
            names,
            matrix,
            len(matrix),
            latency_us=np.asarray(latency_us, dtype=float),
            h_opt=h_opt,
        )


@dataclass(frozen=True)
class GoldenGroup(Group):
    """A pool whose labels mark the rows a GOLDEN verified — the fit's supervision.

    A subclass rather than a flag on :class:`Group` because the two kinds differ in what can be ASKED of
    them, not merely in what their numbers mean. Only here is "which rows are the answer" a question with an
    answer, so only here is :attr:`golden_ids` defined, and the rank metrics that take it say what they need
    by taking this type. A discriminator field would have moved that check to runtime and to every caller.
    """

    # The rows a golden verified, ascending and deduplicated — the index set the rank metrics take. Several
    # goldens can land on one pool (the same shape recorded twice, or under two names), so this is a SET.
    # Deploy ships one config, so any of them ranked first is a win, which is why the fit's per-group term is
    # the best rank over it and the reported one the matching dual rank. At one golden both collapse to the
    # single-golden functions exactly. Stored as the indices themselves: the metrics take indices, so a
    # per-row marker column would only be an encoding to decode back on every read.
    golden_ids: tuple[int, ...] = field(kw_only=True)
    # The golden pools this group was built from — one, or several that packed identically and folded — each with
    # its kernel's definition and its verified rows, so a dataset carries what the deploy check re-lowers.
    pools: tuple[GoldenPool, ...] = field(default=(), kw_only=True)

    @classmethod
    def from_dicts(
        cls,
        key: str,
        name: str,
        tier: str,
        gpu: str,
        shape: str,
        goldens: int | Sequence[int],
        feats: list[dict[str, float]],
        total: int | None = None,
    ) -> GoldenGroup:
        """Pack per-row feature dicts and mark the rows the goldens verified. ``goldens`` is one row index or
        several; a bare int is accepted because most callers have exactly one and wrapping it would say
        nothing. ``total`` is the size of the pool ``feats`` was drawn from, defaulting to "nothing was
        sampled" — see :attr:`Group.total`."""
        return cls.over(key, name, tier, gpu, shape, pack_features(feats), goldens, total)

    @classmethod
    def over(
        cls,
        key: str,
        name: str,
        tier: str,
        gpu: str,
        shape: str,
        packed: tuple[tuple[str, ...], np.ndarray, bool],
        goldens: int | Sequence[int],
        total: int | None = None,
        pools: Sequence[GoldenPool] = (),
    ) -> GoldenGroup:
        """The same over an already-packed pool — the entry point for a builder that had to pack before it
        could know which goldens landed in it, which is every builder that deduplicates pools. ``goldens`` is
        one row index or several, in any order, with duplicates; what is stored is the sorted set.

        ``tier`` must agree with the routing stamp the rows carry, and disagreeing is a hard error. The two
        reach here by different routes — the stamp through the featurizer, the tier from the source record's
        own flag — and they are the same fact, so a mismatch means one of them is wrong and this pool would
        otherwise train and be scored as the wrong regime with nothing reporting it. That is what
        keeps ``tier``, a label that decides nothing, honest. A measured pool has no second route to check
        against and no tier to check, which is why the check lives here and not on the base."""
        _, matrix, dynamic = packed
        if dynamic != (tier == "dyn"):
            raise ValueError(
                f"{key}: the routing stamp says dynamic={dynamic} but the case tier says {tier!r} — "
                f"the source record's flag and its featurized rows disagree about the regime"
            )
        rows = (goldens,) if isinstance(goldens, int) else goldens
        ids = tuple(sorted({int(i) for i in rows}))
        return cls(
            key,
            name,
            tier,
            gpu,
            shape,
            dynamic,
            packed[0],
            matrix,
            len(matrix) if total is None else total,
            golden_ids=ids,
            pools=tuple(pools),
        )
