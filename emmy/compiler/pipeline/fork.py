"""Fork interface + implementations: the deferred fork options the search engine ranks and resolves.

:class:`Fork` is the interface — ``knobs``, ``is_leaf``, ``expand()``, and the descent hooks ``narrow`` /
``admits``. Two implementations hold their producer's state as data: :class:`DeferredFork`, a leaf whose selected
``Op`` / ``Graph`` is built on expansion (what the cut and split passes offer, and what the search lifts a concrete
option into), and the lazy schedule tree — :class:`_ScheduleTree` with its :class:`_ScheduleFork` prefixes — that
``schedule.py`` builds over a semantic ``ScheduleContext``, whose leaves are its ``ScheduleLeaf``. Siblings are
emitted in grouping order — RANKING IS SEARCH POLICY: the policies rank the frontier with the online prior (Forks
carry no score).

The engine in ``pipeline.py`` consumes ``fork.knobs`` flat (it doesn't walk ancestors): branch Forks pin their
decided slice of the row, leaves carry the whole row. :func:`iter_leaves` and :func:`leaf_for` walk any option
sequence; ``ForkPoint`` wraps them for an offer.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.base import Op

from emmy.compiler.ir.schedule import Schedule, ScheduleContext, ScheduleRefused, schedule
from emmy.compiler.pipeline.knob import EVIDENCE_PREFIXES, METADATA_PREFIXES, evidence_row_vouches, values_equal


class Fork(ABC):
    """Interface for a deferred fork option.

    Two flavors share the interface:

    - **Branch Fork** (``is_leaf=False``) — produced explicitly by a rule's
      ``rewrite()`` to spawn a hierarchical fork point. ``expand()`` returns
      the next level of options (more Forks, concrete leaves, or a mix); the
      decide callback walks them (:func:`iter_leaves`).
    - **Leaf Fork** (``is_leaf=True``) — wraps one concrete ``Op`` /
      ``Graph`` rewrite. ``expand()`` returns ``[option]`` (one element);
      ``Run.resolve`` invokes it once at resolve time to retrieve the leaf
      and apply it.

    ``knobs`` is the knob-delta this Fork pins (the variant identity the
    perf DB and the prior key on, read without expanding). Ranking is the
    decide callback's job: it ranks the leaves with measured evidence and the
    :class:`~emmy.compiler.pipeline.search.prior.Prior` (greedy
    ``mean_score`` argmin). Forks carry no score of their own; siblings are
    emitted in grouping order and the no-prior fallback is that emission
    order."""

    knobs: dict
    is_leaf: bool = False
    structural: bool = False
    #: The enumeration's minted pool identity, carried by every node of a schedule tree
    #: (the schedule adapter mints it from the variant key + hints + pins +
    #: split receipt + spelled key vocabulary). ``None`` for forks outside a schedule enumeration.
    #: Consumers key memoized decisions on THIS, never on a re-derived identity.
    pool_id: str | None = None
    #: Upper bound on the enumeration's leaf count (Π of the per-node option tuples × the RASTER
    #: fan-out — legality only shrinks it), carried the same way. ``None`` outside a schedule
    #: enumeration. The greedy cold-pool budget triggers on this without walking anything.
    pool_bound: int | None = None

    @abstractmethod
    def expand(self) -> list[Op | Graph | Fork]: ...

    def sample_child(self, rng: random.Random) -> Op | Graph | Fork | None:
        """One option below this branch, drawn uniformly, or ``None`` when it has none — the step of a random
        descent (:func:`descent_sample`). The default expands the branch and draws; a branch that can draw
        without expanding (a schedule prefix, whose context draws one extension) overrides it."""
        kids = self.expand()
        return rng.choice(kids) if kids else None

    def narrow(self, row: Mapping) -> Fork:
        """This branch with its enumeration re-sourced to ``row`` where it can be: a schedule
        root offers the row's values at the sites the row names. The default is the branch itself —
        a tree with no enumeration behind it descends as it is."""
        return self

    def admits(self, row: Mapping) -> bool:
        """Whether a knob ``row`` — complete, or partial with the undecided knobs absent — can lie
        below this branch: every knob the branch has decided agrees with the row. The one descent
        rule for a row that names a leaf (the decision memo's replay, the evidence pick's direct
        descent to a measured row). The base reading is value equality; a tree whose branches
        carry a prefix of the leaf's spelling or a level's projection of it refines it."""
        return all(
            name not in row or values_equal(name, row[name], value)
            for name, value in self.knobs.items()
            if not name.startswith(METADATA_PREFIXES)
        )


@dataclass(frozen=True)
class DeferredFork(Fork):
    """A lazy concrete leaf whose selected ``Op`` or ``Graph`` is built on expansion."""

    materialize: Callable[[], Op | Graph]
    knobs: dict = field(default_factory=dict)
    structural: bool = False
    #: Other spellings of a key of ``knobs``, each mapped to the key it names — the occurrences of
    #: one clustered value, whose cut is one decision under any of them.
    aliases: dict = field(default_factory=dict)
    is_leaf = True

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.materialize()]


#: How many extensions a sampled child draws before giving the branch up as dead: a pick the composition
#: refuses is rare (the frontier was filtered for it), so a run of them means the prefix has nothing to offer.
_REFUSED_PICK_DRAWS = 4


@dataclass(frozen=True)
class _ScheduleTree:
    """Shared callbacks and identity for one generic lazy schedule tree."""

    branch_knobs: Mapping
    row_delta: Callable[[ScheduleContext, ScheduleContext], Mapping]
    leaf: Callable[[Schedule], Fork]
    pool_id: str
    exact: Callable[[Mapping[str, str]], Fork | None] | None = None
    exact_keys: frozenset[str] | None = None

    def step(self, context: ScheduleContext, row: Mapping) -> list[Fork]:
        """Every extension of ``context``, composed by the generic driver, as a child of the prefix spelling ``row``."""
        return [self.child(context, row, composed) for composed in schedule(context, recursive=False)]

    def child(self, context: ScheduleContext, row: Mapping, composed: ScheduleContext | Schedule) -> Fork:
        """One composed extension of ``context`` as a child: a complete schedule becomes a leaf, a context the next
        prefix carrying ``row`` plus what the step decided. The one place a child is built, for the walk
        (:meth:`step`) and the draw (``_ScheduleFork.sample_child``)."""
        if isinstance(composed, Schedule):
            return self.leaf(composed)
        return _ScheduleFork(self, composed, {**row, **self.row_delta(context, composed)})


@dataclass(frozen=True)
class _ScheduleFork(Fork):
    """One immutable prefix in a generic lazy schedule tree."""

    tree: _ScheduleTree
    context: ScheduleContext
    row: Mapping

    @property
    def knobs(self) -> dict:
        return {**self.tree.branch_knobs, **self.row}

    @property
    def pool_id(self) -> str:
        return self.tree.pool_id

    @property
    def pool_bound(self) -> int:
        return self.context.problem.bound

    def expand(self) -> list[Fork]:
        return self.tree.step(self.context, self.row)

    def sample_child(self, rng: random.Random) -> Fork | None:
        """One child drawn without expanding: the context draws one compatible extension
        (:meth:`ScheduleContext.random_extension`), composed by :meth:`_ScheduleTree.child` as the walk composes
        every extension. A pick the composition refuses is drawn again, as the walk skips such a pick; ``None``
        is a dead end the descent restarts from."""
        for _ in range(_REFUSED_PICK_DRAWS):
            pick = self.context.random_extension(rng)
            if pick is None:
                return None
            try:
                composed = self.context.extend(pick)
            except ScheduleRefused:
                continue
            return self.tree.child(self.context, self.row, composed.schedule if composed.schedule.kernel is not None else composed)
        return None

    def narrow(self, row: Mapping) -> Fork:
        """The root of a schedule tree re-sourced to ``row``: its problem offers the row's values at
        the sites the row names, so the descent below it instantiates one path. A branch below the
        root has decided sites already and descends as it is."""
        if self.row or self.context.schedule.nodes or self.context.schedule.kernel is not None:
            return self
        return replace(self, context=self.context.narrowed(row))

    def admits(self, row: Mapping) -> bool:
        """A schedule branch spells each decided knob as the PREFIX of what its leaves will spell
        (``w2x2`` before ``w2x2+p1``, ``…/f2x2`` before ``…/f2x2/k2``; an OFF default ``''`` before
        anything), so the row's value must extend the branch's value at a segment boundary. A
        site the row names only by its bare family key reads as a bare pin does
        (``evidence_row_vouches``): the site may be OFF or carry the value, never another — pruned
        here so a row that names no leaf costs O(path), not the pool.

        The prefix reading has one exception, and leaving it out cost the pool bound it advertises:
        every string extends the empty one, so an OFF already in ``row`` admitted every request and
        the descent only failed at leaf matching. One such site doubles the work and these kernels
        carry dozens. The claim is only about the EMPTY spelling — a field in ``row`` is one the
        codec emitted, not necessarily one whose spelling is complete (``WORK`` still grows its
        producer band there) — and an emitted empty never fills in later: the classic codec writes a
        site's node and edge values when that site advances, and an unclaimed inventory stays
        ``None`` rather than spelling ``Work()``. An OFF merely INHERITED through ``branch_knobs``
        is a pin, not a decision, and still admits, as does a bare family key — a bare pin permits
        OFF."""
        for name, value in self.knobs.items():
            if name.startswith(METADATA_PREFIXES):
                continue
            family = name.split("@", 1)[0]
            if name in row:
                want = str(row[name])
                if name in self.row and not str(value) and want:
                    return False
            elif name != family and family in row:
                want = str(row[family])
            else:
                continue
            have = str(value)
            # A cooperative reduce's ``t<coop>`` grows to the packed ``t<coop>x<cells>`` at the kernel site.
            grows = (have + "/", have + "+", have + "x") if family == "WORK" and "x" not in have else (have + "/", have + "+")
            if have and want != have and not want.startswith(grows):
                return False
        return True


def schedule_forks(
    context: ScheduleContext,
    *,
    branch_knobs: Mapping,
    row_delta: Callable[[ScheduleContext, ScheduleContext], Mapping],
    leaf: Callable[[Schedule], Fork],
    pool_id: str,
    exact: Callable[[Mapping[str, str]], Fork | None] | None = None,
    exact_keys: frozenset[str] | None = None,
) -> list[Fork]:
    """Represent any schedule context as a lazy pipeline Fork tree: one unexpanded root, so
    nothing is enumerated until a consumer expands it — or narrows it to a row first."""
    tree = _ScheduleTree(dict(branch_knobs), row_delta, leaf, pool_id, exact, exact_keys)
    return [_ScheduleFork(tree, context, {})]


def exact_schedule_leaf(
    options: Sequence[Op | Graph | Fork], row: Mapping[str, str], required_keys: frozenset[str]
) -> tuple[frozenset[str], Fork | None] | None:
    """Decode one complete row through an unsampled semantic schedule without enumerating it.

    ``None`` means the options are not the single root made by :func:`schedule_forks`. The declared
    key set accompanies an exact hit or miss so a strict replay can distinguish stale site names.
    This does not change :meth:`Fork.admits`: partial rows elsewhere may carry keys for later forks.
    """
    roots = [option for option in options if isinstance(option, _ScheduleFork)]
    if len(roots) != 1 or len(options) != 1 or roots[0].tree.exact is None or roots[0].tree.exact_keys is None:
        return None
    root = roots[0]
    keys = root.tree.exact_keys
    if not required_keys <= keys:
        return keys, None
    return keys, root.tree.exact({key: str(value) for key, value in row.items() if key in keys})


def descent_sample(options: Sequence[Op | Graph | Fork], *, draw: int, seed: object, skip: Callable | None = None) -> list:
    """Up to ``draw`` complete leaves drawn by seeded uniform descents through the lazy tree — a child at
    random at every branch (:meth:`Fork.sample_child`, which a schedule prefix answers without expanding), so
    the draw reaches every level's values the way an emission-order prefix never does and costs the options
    it tries rather than the frontiers it passes. Dead ends (a branch with no child — legality killed the
    subtree) and leaves ``skip`` refuses retry, up to four attempts per row. Duplicates are kept; the caller
    decides whether a repeat matters. The draw is a pure function of the tree, ``draw`` and ``seed``."""
    rng = random.Random(str(seed))
    sample: list = []
    attempts = 4 * draw
    while len(sample) < draw and attempts > 0:
        attempts -= 1
        option = rng.choice(options)
        dead = False
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.sample_child(rng)
            if option is None:
                dead = True
                break
        if dead or (skip is not None and skip(option)):
            continue
        sample.append(option)
    return sample


def iter_leaves(options: Iterable[Op | Graph | Fork]) -> Iterator[Op | Graph | Fork]:
    """Yield complete leaves depth-first without retaining the expanded tree or Python stack."""
    stack = [iter(options)]
    while stack:
        try:
            option = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(option, Fork) and not option.is_leaf:
            stack.append(iter(option.expand()))
        else:
            yield option


#: The ``S_*`` stamps the SCHEDULE fork mints on its own rows — properties of the offered schedule
#: SPACE, not of the kernel. A kernel-set fork is decided before any schedule exists, so its
#: candidates cannot carry them however warp-eligible the kernel turns out to be; a recorded row's
#: copy therefore must not join against them (``policy/greedy._route_candidates``).
SCHEDULE_FORK_STAMPS = frozenset({"S_warp_eligible"})


def fork_signature(root_op: Op, options: Sequence[Op | Graph | Fork], ctx) -> frozenset:
    """The ``S_*`` signature every candidate at one fork shares — the key a measured row of this
    kernel is filed under. The offer op's structural stamp under the run's context features, plus
    the stamps the enumeration itself minted on its options (``S_warp_eligible``: a property of
    the offered space, carried on the pool's top level and inherited by every leaf). Read here by
    the deploy's evidence pick and by the golden replay that keys a record's rows, so the two
    agree by construction."""
    base = {**ctx.features(), **dict(getattr(root_op, "knobs", None) or {})}
    for option in options:
        base.update((key, value) for key, value in (getattr(option, "knobs", None) or {}).items() if key.startswith(EVIDENCE_PREFIXES))
    return stamp_signature(base)


def stamp_signature(knobs: Mapping) -> frozenset:
    """The evidence signature of a knob dict, values as strings: its ``S_*`` stamps and its exact ``I_kernel``
    identity — the one spelling a measured row, a stored kernel and a fork's offer are joined on."""
    return frozenset((key, str(value)) for key, value in knobs.items() if key.startswith(EVIDENCE_PREFIXES))


def leaf_for(options: Sequence[Op | Graph | Fork], row: Mapping, *, skip: Callable[[dict], bool] | None = None):
    """The first leaf a (possibly partial) knob ``row`` vouches for, as ``(leaf, its knobs)``, or
    ``None``. A schedule root is first narrowed to the row (:meth:`Fork.narrow`), so its
    enumeration offers the row's values at the sites the row names and the descent below it is
    one path; every branch is descended only when it admits the row (:meth:`Fork.admits`).
    ``skip`` drops a leaf by its knobs (a blocklisted tile). The one descent the evidence pick,
    the decision memo's replay and the golden replay share."""
    for option in options:
        if isinstance(option, Fork) and not option.is_leaf:
            if option.admits(row):
                found = leaf_for(option.narrow(row).expand(), row, skip=skip)
                if found is not None:
                    return found
            continue
        knobs = leaf_knobs(option)
        if skip is not None and skip(knobs):
            continue
        tunable = {key: str(value) for key, value in knobs.items() if not key.startswith(METADATA_PREFIXES)}
        if evidence_row_vouches(tunable, row):
            return option, knobs
    return None


def leaf_knobs(leaf: Op | Graph | Fork) -> dict:
    """A leaf's complete knob row: a leaf ``Fork`` carries it as ``knobs``; a concrete ``Op``
    carries its own; a ``Graph`` splice has no single row (scored structurally, never by knobs) —
    empty."""
    from emmy.compiler.graph import Graph  # noqa: PLC0415

    if isinstance(leaf, Fork):
        return dict(leaf.knobs)
    return dict(getattr(leaf, "knobs", None) or {}) if not isinstance(leaf, Graph) else {}
