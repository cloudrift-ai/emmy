"""``Body`` — an immutable sequence of body Stmts with built-in def-use / iteration / transform queries.

The methods on Body are the recommended way to phrase common analyses: the walks, the def-use readings
(``definitions``, ``axis_dependencies``), the orders, the type-filtered lookups (``loads``,
``writes``, ``accums``, …) and dependence cones (:class:`Cone`, :meth:`Body.backward_cone`) — the shared substrate
behind the rules that slice computed-operand cones.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from heapq import heappop, heappush

from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.structural import form
from emmy.compiler.wire import Wire
from emmy.utils import cached_method


@dataclass(frozen=True)
class Cone:
    """A dependence cone over ONE scope level: the subset of a Body's
    immediate stmts closed under SSA dependence (in body order), plus every
    name the cone reads from outside itself — sibling/enclosing scopes and
    axis vars alike. Built by :meth:`Body.backward_cone`.

    Construction never fails and applies no eligibility judgment: an
    unresolved name is data (``external_reads``), not an error. Which
    external reads are acceptable, which member kinds are cuttable, and
    whether the cone's values escape are the
    calling rule's conditions — bail decisions stay in rules, the dataflow
    walk lives here (``passes/ARCHITECTURE.md``: phrase conditions over cone
    properties instead of re-walking shapes).

    A member is a whole top-level stmt: a wrapper (Loop / Cond / Tile) joins
    as a unit, exposing names per :func:`exposed_defines` and reading per
    :func:`free_names` (subtree rolled up, internally-bound axes
    excluded). Axis vars from enclosing scopes survive into
    ``external_reads`` — intersect with an axis-name set to get the cone's
    axis usage, subtract it to get the SSA names that must resolve
    elsewhere."""

    members: tuple[Stmt, ...]
    external_reads: frozenset[str]

    @property
    def loads(self) -> tuple[Stmt, ...]:
        """Every ``Load`` in the members, nested included, body order —
        the cone's leaf operands (dtype checks, graph resolution)."""
        return Body(self.members).loads


def exposed_defines(s: Stmt) -> set[str]:
    """SSA names ``s`` makes visible at its own scope level — own defines
    plus every nested define. A deliberate over-approximation: nested non-Accum
    names don't truly cross a Loop boundary, but well-formed SSA never reads them
    from outside, so resolving through them is harmless and cheap."""
    out = set(s.defines())
    for body in s.nested():
        for c in body.iter():
            out.update(c.defines())
    return out


def _seeded_accumulators(s: Stmt) -> frozenset[str]:
    """The accumulators a reduce loop declares in its enclosing scope: the ``Accum`` targets of its
    immediate body, each seeded just ahead of the loop. Empty for any other statement, for a loop
    whose accumulators are declared elsewhere (``seed=False``), and for a loop carrying a state."""
    from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop  # noqa: PLC0415
    from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

    if not isinstance(s, (Loop, StridedLoop)) or not s.seed or s.body.carries:
        return frozenset()
    return frozenset(c.name for c in s.body if isinstance(c, Accum))


def free_names(s: Stmt) -> frozenset[str]:
    """Every name ``s`` (whole subtree) reads from its enclosing scope — SSA reads AND index
    coordinates, less what the subtree defines or binds.

    The WIDE reading, for callers that must resolve a statement against everything around it: a
    dependence cone needs the axis vars as much as the value names, since both have to be
    available where the cone lands. Callers asking a narrower question — is this VALUE read? —
    want :attr:`Body.ssa_uses`, which follows ``deps()`` without adding expression reads or
    subtracting local bindings. Load indices can occur in both: a coordinate and a gathered
    value use the same ``Var`` representation.
    """
    reads: set[str] = set()
    defs: set[str] = set()

    def walk(st: Stmt, bound: frozenset[str]) -> None:
        reads.update(set(st.deps()) - bound)
        for e in st.exprs():
            reads.update(e.free_vars() - bound)
        defs.update(st.defines())
        if st.deps_deep:
            return  # ``deps()`` already rolls up the subtree scope-correctly; the flat re-walk
            # cannot see the lift's params, so a factored operand cone's result read inside the
            # lift would leak out as a phantom capture
        inner_bound = bound | st.binds_axes()
        for body in st.nested():
            for c in body:
                walk(c, inner_bound)

    walk(s, frozenset())
    return frozenset(reads - defs)


class Body(Wire):
    """Immutable Stmt sequence, built from any iterable. A slice or a sum of bodies is a ``Body``, two bodies with
    equal statements are equal and hash alike, and a statement read from one twice is the same object. It is no
    ``tuple``: it equals no tuple, and what it offers of a sequence is what is written here.

    Instances keep a ``__dict__``, so ``functools.cached_property`` holds the analyses on the body they derive from.
    """

    wire_tag = "body"

    def to_wire(self) -> str:
        """The body's text."""
        return str(self)

    @classmethod
    def from_wire(cls, value: object, where: str = "body") -> Body:
        from emmy.compiler.ir.stmt.text import parse_body  # noqa: PLC0415 — the text module imports every kind

        if not isinstance(value, str):
            raise ValueError(f"{where} must be a body's text")
        try:
            return parse_body(value)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc

    def structural_form(self) -> tuple:
        """Elementwise, as a sequence renders: :meth:`structural_key` is the normalize-and-collapse key, not the form."""
        return tuple(form(stmt) for stmt in self._stmts)

    def __init__(self, stmts: Iterable[Stmt] = ()) -> None:
        members = tuple(stmts)
        # A body holds STATEMENTS. A pure term is not one — ``Fold`` duck-types the statement
        # protocol so the shared walks can reach it, but it is not a ``Stmt`` and does not belong
        # in a statement sequence: a Fold tree composes through ``operands``, and a term sitting in
        # a body is a second, competing composition mechanism. Checked by TYPE rather than by
        # naming the kinds that are excluded, so the rule holds for anything else that duck-types
        # its way in later.
        stray = [type(member).__name__ for member in members if not isinstance(member, Stmt)]
        if stray:
            raise TypeError(f"Body holds non-statement member(s) {stray}; a term composes through operand edges")
        self._stmts = members

    def __len__(self) -> int:
        return len(self._stmts)

    def __iter__(self) -> Iterator[Stmt]:
        return iter(self._stmts)

    def __str__(self) -> str:
        """The body's text, which ``parse_body`` reads back to an equal body."""
        from emmy.compiler.ir.stmt.text import body_text  # noqa: PLC0415 — the text module imports every kind

        return body_text(self)

    def __getitem__(self, key):
        r = self._stmts[key]
        return Body(r) if isinstance(key, slice) else r

    def __eq__(self, other: object) -> bool:
        return self._stmts == other._stmts if type(other) is Body else NotImplemented

    def __hash__(self) -> int:
        return hash(self._stmts)

    def __repr__(self) -> str:
        return repr(self._stmts)

    def __add__(self, other: Iterable[Stmt]) -> Body:
        return Body(self._stmts + tuple(other))

    def __radd__(self, other: Iterable[Stmt]) -> Body:
        return Body(tuple(other) + self._stmts)

    @staticmethod
    def coerce(value: Body | Iterable[Stmt]) -> Body:
        """Wrap if not already a Body. Used by ``LoopOp`` /
        ``TileOp`` ``__post_init__`` so the legacy
        ``Op(body=tuple_value)`` construction shape keeps working."""
        return value if isinstance(value, Body) else Body(value)

    # -- iteration -------------------------------------------------------

    def iter(self) -> Iterator[Stmt]:
        """Pre-order iteration over this body and every nested body
        (``Loop`` / ``Tile`` / ``Cond`` / ``StridedLoop`` recurse via
        ``Stmt.nested()``)."""
        for s in self:
            yield s
            for child_body in s.nested():
                yield from child_body.iter()

    # -- transformation --------------------------------------------------

    def rename_buffers(self, rename) -> Body:  # noqa: ANN001 — any str->str mapping
        """This body with every external-buffer reference renamed through ``rename`` — the
        body-level face of :meth:`Stmt.rename_buffers` (the recursive :meth:`map` reaches every
        nested leaf, so wrapper stmts need no handling). An injective spelling-only rename retains
        this body's declared normal-form fixed points, while an aliasing rename must recompute
        ordering because it changes memory dependencies."""
        result = self.map(lambda s: s.rename_buffers(rename))
        if result == self:
            return self
        resources = tuple(dict.fromkeys(name for stmt in self.iter() for name in (*stmt.external_reads(), *stmt.external_writes())))
        if len({rename.get(name, name) for name in resources}) == len(resources) and self.__dict__.get("_normalized") is self:
            result.__dict__["_normalized"] = result
        return result

    def map(self, fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]]) -> Body:
        """Recursive 1:N body transformer. Post-order: each block stmt's
        nested body is mapped first, then ``fn`` is applied to the
        children-rewritten wrapper. Returns a new Body with each stmt
        replaced by ``fn(stmt)``:

        - a single ``Stmt`` (kept in place of the input),
        - ``None`` (drop the input), or
        - an iterable of ``Stmt`` (inline all of them).

        ``fn`` is called on *every* stmt including ``Loop`` / ``Tile`` /
        ``Cond`` / ``StridedLoop`` wrappers — but with their bodies already
        recursively mapped, so ``fn`` only needs to handle the leaf cases
        it cares about (callers no longer need a self-recursive
        ``Loop(..., body=s.body.map(fn))`` branch). Mirrors :meth:`iter`'s
        full-tree traversal.

        Iterable returns *replace* the wrapper: the returned stmts are
        spliced into the body as-is (their interiors have already been
        recursed when the wrapper was visited), so a caller that
        ``return tuple(c for c in s.body)`` to inline a Loop's body sees
        already-rewritten children.
        """

        def descend(s: Stmt) -> Stmt:
            nested = s.nested()
            if not nested:
                return s
            return s.with_bodies(tuple(b.map(fn) for b in nested))

        out: list[Stmt] = []
        for s in self:
            r = fn(descend(s))
            if r is None:
                continue
            if isinstance(r, Stmt):
                out.append(r)
            else:
                out.extend(r)
        return Body(out)

    def topological_order(
        self,
        incoming: Sequence[set[int] | frozenset[int]],
        tie_break: Callable[[int, Stmt], object] | None = None,
    ) -> Body:
        """This body in :meth:`_topological_permutation` order."""
        return Body(self[index] for index in self._topological_permutation(incoming, tie_break))

    def _topological_permutation(
        self,
        incoming: Sequence[set[int] | frozenset[int]],
        tie_break: Callable[[int, Stmt], object] | None = None,
    ) -> tuple[int, ...]:
        """Kahn topological order over indexed predecessor sets, as source indices.

        Source order breaks ties by default. ``tie_break`` may supply a canonical priority while
        the source index remains the final deterministic tie-break. A cycle leaves the order
        unchanged so callers can preserve the validator's error path.
        """
        successors: list[list[int]] = [[] for _ in self]
        degree = [len(sources) for sources in incoming]
        for target, sources in enumerate(incoming):
            for source in sources:
                successors[source].append(target)

        def priority(index: int) -> tuple[object, int]:
            return (index if tie_break is None else tie_break(index, self[index])), index

        ready = [priority(index) for index, count in enumerate(degree) if not count]
        ready.sort()
        ordered: list[int] = []
        while ready:
            _, selected = heappop(ready)
            ordered.append(selected)
            for target in successors[selected]:
                degree[target] -= 1
                if not degree[target]:
                    heappush(ready, priority(target))
        return tuple(ordered) if len(ordered) == len(self) else tuple(range(len(self)))

    @cached_property
    def _normalized(self) -> Body:
        """Executable normal form, cached on this immutable body and its fixed point, and memoized by the body's text
        across bodies: a cut's arms lower the same pieces again and again."""
        result = _normal_form.get(repr(self), lambda: _normalize_body_of(self))
        result.__dict__["_normalized"] = result
        return result

    # -- generic backward dataflow --------------------------------------

    def coalesce(self) -> Body:
        """Assemble shared SSA definitions once and merge adjacent independent reduction loops.

        Used when lowering shared value cones: repeated definitions represent the SAME value,
        not repeated accumulator updates. Equal definitions keep their first occurrence; effects
        are retained. Loops with identical iteration headers can share a pass unless the later
        loop reads a finalized result from the earlier one.

        The first occurrence also wins per accumulator. Two lowerings of one shared fold need not
        be equal statements: one cone edge lowers the fold alone, another fuses it into one pass
        with a sibling fold over the same axis. A later reduce loop that folds an accumulator this
        scope already declared keeps only the folds of its other accumulators. Without this, both
        loops declare the shared accumulator in one scope, which does not compile.
        """
        from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop  # noqa: PLC0415

        out: list[Stmt] = []
        seen: set[Stmt] = set()
        folded: set[str] = set()  # accumulators the reduce loops of ``out`` declare at this scope
        for stmt in self:
            if stmt.has_side_effects:
                seen.clear()
                folded.clear()
            elif exposed_defines(stmt):
                if stmt in seen:
                    continue
                seen.add(stmt)
                accumulators = _seeded_accumulators(stmt)
                if accumulators & folded:
                    if accumulators <= folded:
                        continue
                    stmt = stmt.replace(body=Body(stmt.body.backward_cone(accumulators - folded).members))
            prior = out[-1] if out else None
            if (
                isinstance(stmt, (Loop, StridedLoop))
                and isinstance(prior, (Loop, StridedLoop))
                and prior.is_reduce
                and stmt.is_reduce
                and not (prior.has_side_effects or stmt.has_side_effects)
                and prior.replace(body=stmt.body) == stmt
                and not (free_names(stmt) & prior.body.definitions.keys())
            ):
                stmt = prior.replace(body=(prior.body + stmt.body).coalesce())
                out.pop()
            out.append(stmt)
            folded.update(_seeded_accumulators(stmt))
        return Body(out)

    def fold[T](
        self,
        fn: Callable[[Stmt, tuple[T | None, ...], frozenset[str]], T],
    ) -> dict[int, T]:
        """Generic backward dataflow over this body's def-use DAG.

        Walks every stmt in source order (= SSA topo order). At each stmt,
        calls ``fn(stmt, child_T, bound)``:

        - ``child_T`` — one entry per name in ``stmt.deps()``, pulled from
          the running memo via ``self.definitions``. ``None`` when the dep
          is read but not defined locally (Tile-input buffer reference,
          constant, or an SSA from an enclosing scope — i.e. an external
          read). Position-preserving: same order as ``stmt.deps()``.
        - ``bound`` — set of axis names introduced by enclosing
          ``Loop`` / ``StridedLoop`` / ``Tile`` wrappers via
          ``Stmt.binds_axes()``. ``Cond`` doesn't bind axes. Callbacks
          that don't care about scope can ignore this.

        Returns the per-stmt memo keyed by ``id(stmt)`` without recursively hashing nested bodies.
        Callers that want a name-keyed view do
        ``{n: memo[id(s)] for s in body.iter() for n in s.defines()}``.

        Recursion order: nested bodies are processed *before* the wrapper
        stmt. So when ``fn`` is called on a wrapper that doesn't define a
        name itself (Loop / Tile / StridedLoop), the memo entries for any
        Accums inside its body already exist — downstream consumers at the
        wrapper's scope can read them through ``deps()``.

        Caveat: when multiple stmts define the same SSA name (matmul-shape
        bodies with several ``Accum`` stmts sharing one accumulator),
        ``self.definitions`` resolves to the last definer. ``child_T`` will
        carry that last definer's ``T`` only. Callers needing a multi-defs
        union (e.g. unioning axes across all Accums for ``acc``) iterate
        ``body.accums`` themselves at the call site.
        """
        memo: dict[int, T] = {}
        defs = self.definitions

        def walk(body: Body, bound: frozenset[str]) -> None:
            for s in body:
                child_bound = bound | s.binds_axes()
                for child_body in s.nested():
                    walk(child_body, child_bound)
                child_T: tuple[T | None, ...] = tuple(memo.get(id(defs[d])) if d in defs else None for d in s.deps())
                memo[id(s)] = fn(s, child_T, bound)

        walk(self, frozenset())
        return memo

    # -- def-use analysis ------------------------------------------------

    @cached_property
    def _carried_names(self) -> tuple[str, ...]:
        """Accumulator and carried-state names, deduplicated in structural order."""
        from emmy.compiler.ir.stmt.leaves import Accum, Carry  # noqa: PLC0415

        return tuple(dict.fromkeys(name for stmt in self.iter_of_type(Accum, Carry) for name in stmt.defines()))

    @cached_property
    def _local_defs(self) -> frozenset[str]:
        """Bindings at this scope, including states exported by nested statements."""
        from emmy.compiler.ir.stmt.order import _ordered_sibling_defs  # noqa: PLC0415

        return frozenset(name for stmt in self for name in _ordered_sibling_defs(stmt))

    @cached_property
    def _free_ssa(self) -> frozenset[str]:
        """SSA reads from the enclosing scope, excluding this scope's definitions and exported states."""
        from emmy.compiler.ir.stmt.order import _free_ssa  # noqa: PLC0415

        return frozenset().union(*(_free_ssa(stmt) for stmt in self)) - self._local_defs

    @cached_property
    def definitions(self) -> dict[str, Stmt]:
        """Map every SSA name produced anywhere inside this body
        (recursive) to its defining ``Stmt``.

        Built once per Body via :meth:`Stmt.defines` over :meth:`iter`;
        cached on the instance, so repeated queries (``def_of`` from
        many call sites in a single rule) are O(1) after the first
        access. Body is immutable, so the cache stays valid for its
        lifetime.

        Names not present in the dict are either Tile-input buffer
        references, constants, or SSA names defined in an enclosing
        scope outside this body — i.e. external reads.
        """
        return {n: s for s in self.iter() for n in s.defines()}

    @cached_property
    def axis_names(self) -> frozenset[str]:
        """Every axis name bound by any wrapper anywhere in this body
        (``Loop`` / ``StridedLoop`` / ``Tile.axes``). Axes from
        enclosing scopes above this body are not included."""
        return frozenset(ax for s in self.iter() for ax in s.binds_axes())

    @cached_property
    def ssa_uses(self) -> frozenset[str]:
        """Every name a statement of this immutable subtree reads — a ``Load`` index's ``Var`` names
        among them, since a coordinate is the same ``Var`` a gathered value read would be. The
        immediate ``deps()`` reads only.
        Expression-only reads, such as a Select predicate, require :func:`free_names` instead.
        """
        out: set[str] = set()
        for stmt in self:
            out.update(stmt.deps())
            for child in stmt.nested():
                out.update(child.ssa_uses)
        return frozenset(out)

    @cached_property
    def axis_dependencies(self) -> dict[str, frozenset[str]]:
        """Map each SSA definition to the axes that its value depends on.

        The summary holds axes only, never the SSA names a value is reached through. Its total size is
        bounded by definitions × axes, which is the representation normalization needs for invariant motion.

        An Accum exported from a Loop loses that Loop's reduction axis. A
        StridedLoop keeps its axis because the partial value still varies by
        partition.
        """
        from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop  # noqa: PLC0415
        from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

        dependencies: dict[str, frozenset[str]] = {}
        axis_names = self.axis_names

        def _immediate_axes(stmt: Stmt) -> frozenset[str]:
            reads: set[str] = set(stmt.deps())
            for expr in stmt.exprs():
                reads.update(expr.free_vars())
            axes = reads & axis_names
            for name in reads:
                axes.update(dependencies.get(name, frozenset()))
            return frozenset(axes)

        def walk(body: Body) -> None:
            for stmt in body:
                for child in stmt.nested():
                    walk(child)
                if isinstance(stmt, Loop):
                    for child in stmt.body:
                        if isinstance(child, Accum):
                            dependencies[child.name] = dependencies.get(child.value, frozenset()) - {stmt.axis.name}
                    continue
                if isinstance(stmt, StridedLoop):
                    for child in stmt.body:
                        if isinstance(child, Accum):
                            dependencies[child.name] = dependencies.get(child.value, frozenset())
                    continue
                axes = _immediate_axes(stmt)
                for name in stmt.defines():
                    dependencies[name] = axes

        walk(self)
        return dependencies

    # -- dependence cones --------------------------------------------------

    def backward_cone(self, roots: Iterable[str]) -> Cone:
        """The backward dependence :class:`Cone` of ``roots`` over THIS
        body's immediate stmts: every top-level member whose exposed names
        transitively feed a root, in body order. Names resolving to no
        member here (axis vars, enclosing/sibling scopes, buffer constants)
        surface in ``external_reads`` — chain another scope level by seeding
        its ``backward_cone`` with them. A root not defined at this level is
        itself an external read (members come out empty)."""
        by_name: dict[str, Stmt] = {}
        for s in self:
            for n in exposed_defines(s):
                by_name[n] = s
        member_ids: set[int] = set()
        external: set[str] = set()
        pending = list(roots)
        seen: set[str] = set()
        while pending:
            n = pending.pop()
            if n in seen:
                continue
            seen.add(n)
            s = by_name.get(n)
            if s is None:
                external.add(n)
                continue
            if id(s) in member_ids:
                continue
            member_ids.add(id(s))
            pending.extend(free_names(s))
        return Cone(members=tuple(s for s in self if id(s) in member_ids), external_reads=frozenset(external))

    # -- type-filtered lookups -------------------------------------------

    @cached_method
    def iter_of_type(self, *types: type) -> tuple[Stmt, ...]:
        """Matching statements in preorder, reusing each child's cached query.

        The named helpers (:meth:`loads`, :meth:`writes`, ...) share this lookup."""
        found = []
        for stmt in self:
            if isinstance(stmt, types):
                found.append(stmt)
            for child in stmt.nested():
                found.extend(child.iter_of_type(*types))
        return tuple(found)

    @cached_property
    def loads(self) -> tuple[Stmt, ...]:
        """All ``Load`` stmts in the body (recursive). Replaces the
        per-Op ``loads`` properties on ``LoopOp`` / ``TileOp`` /
        ``KernelOp``. Cached on the instance — Body is immutable."""
        from emmy.compiler.ir.stmt.leaves import Load  # noqa: PLC0415

        return self.iter_of_type(Load)

    @cached_property
    def writes(self) -> tuple[Stmt, ...]:
        """All ``Write`` stmts in the body (recursive)."""
        from emmy.compiler.ir.stmt.leaves import Write  # noqa: PLC0415

        return self.iter_of_type(Write)

    @cached_property
    def carries(self) -> tuple[Stmt, ...]:
        """All ``Carry`` stmts in the body (recursive): the carried states it defines."""
        from emmy.compiler.ir.stmt.leaves import Carry  # noqa: PLC0415

        return self.iter_of_type(Carry)

    @cached_property
    def accums(self) -> tuple[Stmt, ...]:
        """All ``Accum`` stmts in the body (recursive). May contain
        multiple Accums sharing a single accumulator name (matmul-shape
        K-inner reduces, 008's per-cell replicated accumulator chains).
        Validation enforces op-consistency across same-name Accums in
        ``LoopOp.__post_init__``; callers that want a one-per-name view
        can dedup at the call site (``{a.name: a for a in body.accums}``)."""
        from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

        return self.iter_of_type(Accum)

    @cached_property
    def loops(self) -> tuple[Stmt, ...]:
        """All ``Loop`` stmts in the body (recursive)."""
        from emmy.compiler.ir.stmt.blocks import Loop  # noqa: PLC0415

        return self.iter_of_type(Loop)

    # -- structural identity --------------------------------------------

    def identity(self, *, structural: bool = True, types: Mapping[str, object] | None = None):
        """This body's identity material (:func:`~emmy.compiler.ir.stmt.identity.canonicalize_identity`):
        the canonical body over ``b0, b1, …``, which external buffer fills each role, and the
        role's type. Every flavor is cached on the instance — Body is immutable, and an op rebuilt
        over this body (a lift, a cut, a schedule row) reads the identity its predecessor computed
        instead of canonicalizing the same statements again."""
        return self._identity(structural, None if types is None else tuple(sorted(types.items())))

    @cached_method
    def _identity(self, cluster: bool, types: tuple[tuple[str, object], ...] | None):
        from emmy.compiler.ir.stmt.identity import canonicalize_identity  # noqa: PLC0415 — identity imports this module

        return canonicalize_identity(self, cluster=cluster, types=None if types is None else dict(types))

    def structural_key(self, *, structural: bool = True, types: Mapping[str, object] | None = None) -> str:
        """Implements :class:`emmy.compiler.structural.Structural`.

        Canonical digest used for structural-equivalence queries. Two
        bodies that differ only by SSA / axis names, commutative-arg
        order, or external-buffer names produce the same key; with the
        default ``structural=True`` the specific op within a
        compute-unit cluster (``add`` vs ``sub``, ``div`` vs ``mod``)
        also normalizes away — the SCHEDULE-EQUIVALENT reading, since
        cluster siblings offer the same schedule space and want the same
        schedule. ``structural=False`` keeps the exact ops: the reading
        for consumers to whom ``relu`` and ``gelu`` are different
        kernels (their latency differs even under one schedule).

        ``types`` maps external buffer names to what a deployed kernel is bound to (dtype and
        shape); they color the buffers in the identity graph, so two bodies that read the same
        structure through differently typed roles key apart, whatever order the buffers were
        declared in. Structural, not the pretty text it used to join: ``pretty()`` is the human
        rendering, and a cosmetic change to how a statement prints must not re-key every kernel
        that contains it. Clustered identity collapses semantically distinct ops to one cluster
        representative, so this path is only for structural identity, never executable IR."""
        return self.identity(structural=structural, types=types).key


class Memo:
    """A bounded process-wide memo keyed by the digest of a body's text: the text itself (megabytes for a fused
    layer) and the body it spells are no part of the key, so the memo costs its values alone."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[object, object] = OrderedDict()

    def get(self, spelled: str, compute, *extra: object):
        key = (hashlib.sha1(spelled.encode()).hexdigest(), *extra)
        hit = self._store.get(key)
        if hit is None:
            hit = self._store[key] = compute()
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)
        else:
            self._store.move_to_end(key)
        return hit

    def cache_clear(self) -> None:
        self._store.clear()


def _normalize_body_of(body: Body) -> Body:
    from emmy.compiler.ir.stmt.normalize import _normalize_body  # noqa: PLC0415

    return _normalize_body(body)


#: One normal form per raw body text in this process, as many as the identity memo holds. Restamping one traced
#: program normalizes ~700 distinct bodies three to four times each, and one prior-gate pool meets more than
#: 4,096; a resident body costs ~20 KB, so the bound is ~330 MB a process never reaches in practice.
_normal_form = Memo(maxsize=16384)
