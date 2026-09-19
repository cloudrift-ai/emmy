"""Canonical sibling order for statement bodies.

The ordering problem is one colored relation graph for the complete body tree.  Statements,
lexical scopes, bound names, axes, source axes, and external resources are vertices.  Structural
name occurrences and ordering constraints are directed, colored relations.  Exact canonical graph
labeling therefore chooses every sibling order together, without rendering renamed bodies for
each topological prefix or revisiting nested scopes after an enclosing rename.

A body's graph is built once (:func:`relation_graph`) and labeled under any resource coloring
(:meth:`Ordering.label`): the executable normal form labels resources bare and breaks the remaining
ties by spelling, structural identity colors them by type and never reads a spelling.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Carry, Init
from emmy.compiler.structural import form

__all__ = ["Labeling", "Ordering", "bound_axes", "ordering_constraints", "relation_graph", "topological_sort"]


def _ordered_exported_accs(body: Body) -> tuple[str, ...]:
    """Names ``body`` carries out — accumulators and carried states — deduplicated in structural order."""
    carriers = (stmt for stmt in Body.coerce(body).iter() if isinstance(stmt, (Accum, Carry)))
    return tuple(dict.fromkeys(name for stmt in carriers for name in stmt.carried_names()))


def _ordered_sibling_defs(stmt: Stmt) -> tuple[str, ...]:
    """Names visible to siblings in structural body order."""
    children = stmt.nested()
    if not children:
        return stmt.defines()
    return tuple(dict.fromkeys(name for child in children for name in _ordered_exported_accs(child)))


def _free_ssa(stmt: Stmt) -> frozenset[str]:
    """SSA names ``stmt`` reads from the scope around it.

    A nested scope binds only what it defines at its own level, in any order; a deeper scope's
    definition of the same spelling is a different binder and hides nothing read above it.
    """
    children = stmt.nested()
    if not children or stmt.deps_deep:
        return frozenset(stmt.deps())
    reads = set(stmt.deps())
    for child in children:
        reads.update(_scope_free_ssa(child))
    return frozenset(reads)


def _scope_free_ssa(body: Body) -> frozenset[str]:
    defined = {name for stmt in body for name in _ordered_sibling_defs(stmt)}
    return frozenset().union(*(_free_ssa(stmt) for stmt in body)) - defined


def bound_axes(stmt: Stmt) -> tuple[Axis, ...]:
    """The axes a statement binds, as :class:`Axis` values."""
    axis = getattr(stmt, "axis", None)
    if isinstance(axis, Axis):
        return (axis,)
    return tuple(axis for axis in getattr(stmt, "axes", ()) if isinstance(axis, Axis))


class _Slot(str):
    """A collision-proof structural-form leaf that remembers the spelling it replaced."""

    kind: str
    original: str

    def __new__(cls, kind: str, original: str, serial: int) -> _Slot:
        value = super().__new__(cls, f"\0{kind}{serial}")
        value.kind = kind
        value.original = original
        return value


class _Slots:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.values: dict[str, _Slot] = {}

    def get(self, name: str, default: str) -> str:
        del default
        slot = self.values.get(name)
        if slot is None:
            slot = _Slot(self.kind, name, len(self.values))
            self.values[name] = slot
        return slot


class _AbstractNames:
    @staticmethod
    def get(name: str, default: str) -> str:
        del name, default
        return "__name__"


_ABSTRACT_NAMES = _AbstractNames()

#: The vertex each visible spelling is bound to.
_Binders = dict[str, int]
#: One name or resource a statement's shallow form mentions: ``(kind, spelling, role path)``.
_Occurrence = tuple[str, str, tuple[int | str, ...]]


def _statement_shape(stmt: Stmt) -> tuple[str, tuple[_Occurrence, ...]]:
    """Return one name-free shallow form and its name/resource occurrence relations."""
    children = stmt.nested()
    shell = stmt.with_bodies(tuple(Body() for _ in children)) if children else stmt
    names = _Slots("name")
    resources = _Slots("resource")
    templated = shell.rename(names).rename_buffers(resources)
    rendered = form(templated)

    unordered: dict[tuple[int, ...], str] = {}
    stmt_fields = fields(templated)
    if isinstance(templated, Assign) and templated.op.commutative:
        unordered[(1 + next(index for index, field in enumerate(stmt_fields) if field.name == "args"),)] = "bag"
    if isinstance(templated, Accum):
        unordered[(1 + next(index for index, field in enumerate(stmt_fields) if field.name == "axes"),)] = "set"

    occurrences: list[_Occurrence] = []

    def strip(value: object, path: tuple[int | str, ...]) -> object:
        integer_path = tuple(part for part in path if isinstance(part, int))
        mode = unordered.get(integer_path)
        if mode is not None:
            assert isinstance(value, tuple)
            members = value
            if mode == "set":
                members = tuple(dict.fromkeys(members))
            for member in members:
                if isinstance(member, _Slot):
                    occurrences.append((member.kind, member.original, (*path, mode)))
                else:
                    strip(member, (*path, mode))
            return (f"__{mode}__",)
        if isinstance(value, _Slot):
            occurrences.append((value.kind, value.original, path))
            return "__slot__"
        if isinstance(value, tuple):
            return tuple(strip(member, (*path, index)) for index, member in enumerate(value))
        return value

    return repr(strip(rendered, ())), tuple(occurrences)


@dataclass(frozen=True, slots=True)
class _Scope:
    """One lexical scope of the relation graph: its statements' vertices, constraints and shapes,
    in the order of ``body``."""

    body: Body
    vertex: int
    statements: tuple[int, ...]
    incoming: tuple[frozenset[int], ...]
    children: tuple[tuple[_Scope, ...], ...]
    categories: tuple[int, ...]
    shapes: tuple[str, ...]

    def permuted(self, order: Sequence[int], body: Body, children: tuple[tuple[_Scope, ...], ...]) -> _Scope:
        position = {old: new for new, old in enumerate(order)}
        return _Scope(
            body=body,
            vertex=self.vertex,
            statements=tuple(self.statements[index] for index in order),
            incoming=tuple(frozenset(position[source] for source in self.incoming[index]) for index in order),
            children=children,
            categories=tuple(self.categories[index] for index in order),
            shapes=tuple(self.shapes[index] for index in order),
        )


def _resources(stmt: Stmt) -> tuple[set[str], set[str], set[str]]:
    members = tuple(member for child in stmt.nested() for member in child.iter()) or (stmt,)
    reads = {name for member in members for name in member.external_reads()}
    writes = {name for member in members for name in member.external_writes()}
    state = {name for member in members for name in getattr(member, "carried_names", lambda: ())()}
    if isinstance(stmt, Init):
        state.update(stmt.defines())
    return reads, writes, state


def _source_names(stmt: Stmt) -> tuple[str, ...]:
    return tuple(dict.fromkeys(source.name for axis in bound_axes(stmt) for source in axis.sources()))


def ordering_constraints(body: Body, *, effects: bool, redefinitions: bool = True) -> list[set[int]]:
    """Dependency and, when requested, effect predecessors for one lexical scope."""
    defs_uses = [(frozenset(_ordered_sibling_defs(stmt)), _free_ssa(stmt)) for stmt in body]
    definitions: dict[str, list[int]] = {}
    for index, (defines, _) in enumerate(defs_uses):
        for name in defines:
            definitions.setdefault(name, []).append(index)

    def defining_stmt(name: str, consumer: int) -> int | None:
        sites = definitions.get(name, ())
        if not redefinitions:
            return sites[0] if sites and sites[0] != consumer else None
        preceding = [site for site in sites if site < consumer]
        if preceding:
            return preceding[-1]
        return next((site for site in sites if site != consumer), None)

    incoming: list[set[int]] = []
    for index, (_, uses) in enumerate(defs_uses):
        incoming.append({source for name in uses if (source := defining_stmt(name, index)) is not None})
    if redefinitions:
        for reader, (_, uses) in enumerate(defs_uses):
            for name in uses:
                for later_definition in definitions.get(name, ()):
                    if later_definition > reader:
                        incoming[later_definition].add(reader)

    if effects:
        accesses = [_resources(stmt) for stmt in body]
        last_write: dict[str, int] = {}
        readers: dict[str, set[int]] = {}
        last_state: dict[str, int] = {}
        for index, (reads, writes, state) in enumerate(accesses):
            for name in reads:
                if (writer := last_write.get(name)) is not None:
                    incoming[index].add(writer)
                if name not in writes:
                    readers.setdefault(name, set()).add(index)
            for name in writes:
                if (writer := last_write.get(name)) is not None:
                    incoming[index].add(writer)
                incoming[index].update(readers.pop(name, ()))
                last_write[name] = index
            for name in state:
                if (previous := last_state.get(name)) is not None:
                    incoming[index].add(previous)
                last_state[name] = index

        # A no-dataflow, non-pure leaf is an ordered execution protocol (barriers, async
        # commit/wait, declarations, and future primitives of the same kind). Pin it relative to
        # every sibling. Resource and carried-state leaves are already ordered above; ordinary
        # computations expose defs/deps and remain freely topological.
        protocol = {
            index for index, stmt in enumerate(body) if not stmt.pure and not stmt.nested() and not stmt.defines() and not stmt.deps()
        }
        preceding: list[int] = []
        previous_protocol: int | None = None
        for index in range(len(body)):
            if index in protocol:
                incoming[index].update(preceding)
                if previous_protocol is not None:
                    incoming[index].add(previous_protocol)
                preceding.clear()
                previous_protocol = index
            else:
                if previous_protocol is not None:
                    incoming[index].add(previous_protocol)
                preceding.append(index)
    return incoming


def topological_sort(stmts: Body) -> Body:
    """Stable recursive dependency sort used before structural normalization.

    A scope's definitions bind its reads whatever order they were emitted in — the splicer lands
    consumers above producers — and shadow an enclosing scope's binding of the same spelling.
    """
    body = Body(
        stmt.with_bodies(tuple(topological_sort(child) for child in stmt.nested())) if stmt.nested() else stmt
        for stmt in Body.coerce(stmts)
    )
    return body.topological_order(ordering_constraints(body, effects=False, redefinitions=False))


class _Builder:
    def __init__(self) -> None:
        self.colors: list[str] = []
        self.edges: list[tuple[int, int, str]] = []
        self.resources: dict[str, int] = {}
        self.fixed_names: dict[str, int] = {}

    def vertex(self, color: object) -> int:
        index = len(self.colors)
        self.colors.append(repr(color))
        return index

    def relation(self, source: int, target: int, color: object) -> None:
        self.edges.append((source, target, repr(color)))

    def _resource(self, name: str) -> int:
        vertex = self.resources.get(name)
        if vertex is None:
            vertex = self.vertex(("resource", None))
            self.resources[name] = vertex
        return vertex

    def _fixed_name(self, name: str) -> int:
        vertex = self.fixed_names.get(name)
        if vertex is None:
            vertex = self.vertex(("fixed-name", name))
            self.fixed_names[name] = vertex
        return vertex

    def scope(
        self, body: Body, outer_ssa: _Binders, outer_axes: _Binders, outer_sources: _Binders, fixed: _Binders, *, root: bool = False
    ) -> _Scope:
        """Add one lexical scope under the enclosing scope's binders, which it reads and never mutates."""
        body = Body.coerce(body)
        scope_vertex = self.vertex(("scope", "root" if root else "child"))
        sibling_defs = tuple(_ordered_sibling_defs(stmt) for stmt in body)
        definition_sites: dict[str, list[tuple[int, int, int]]] = {}
        definitions_by_stmt: list[dict[str, int]] = []
        binders_by_stmt: list[list[int]] = []  # each statement's binder vertices, in slot order
        aliases: dict[str, int] = {}
        for statement_index, names in enumerate(sibling_defs):
            own: dict[str, int] = {}
            binders: list[int] = []
            for slot, name in enumerate(names):
                vertex = self.vertex(("binder", "ssa"))
                alias = aliases.get(name)
                if alias is None:
                    alias = self.vertex(("binder", "same-name"))
                    aliases[name] = alias
                    self.relation(scope_vertex, alias, ("owns", "same-name"))
                self.relation(scope_vertex, vertex, ("owns", "ssa"))
                self.relation(vertex, alias, ("same-name",))
                if name in fixed:
                    self.relation(vertex, fixed[name], ("exports",))
                definition_sites.setdefault(name, []).append((statement_index, slot, vertex))
                own.setdefault(name, vertex)
                binders.append(vertex)
            definitions_by_stmt.append(own)
            binders_by_stmt.append(binders)

        def defining_vertex(name: str, consumer: int) -> int | None:
            sites = definition_sites.get(name, ())
            preceding = [vertex for index, _slot, vertex in sites if index < consumer]
            if preceding:
                return preceding[-1]
            local = next((vertex for index, _slot, vertex in sites if index != consumer), None)
            return local if local is not None else outer_ssa.get(name)

        local_sources = dict(outer_sources)
        for stmt in body:
            for name in _source_names(stmt):
                if name not in local_sources:
                    vertex = self.vertex(("binder", "source"))
                    local_sources[name] = vertex
                    self.relation(scope_vertex, vertex, ("owns", "source"))

        statement_vertices: list[int] = []
        categories: list[int] = []
        shapes: list[str] = []
        bound: list[dict[str, int]] = []
        for statement_index, stmt in enumerate(body):
            shape, occurrences = _statement_shape(stmt)
            children = stmt.nested()
            category = 2 if children and stmt.has_side_effects else int(not children)
            stmt_vertex = self.vertex(("statement", category, shape))
            statement_vertices.append(stmt_vertex)
            categories.append(category)
            shapes.append(shape)
            self.relation(scope_vertex, stmt_vertex, ("member",))

            axes = dict(outer_axes)
            for name in stmt.binds_axes():
                axis_vertex = self.vertex(("binder", "axis"))
                axes[name] = axis_vertex
                self.relation(stmt_vertex, axis_vertex, ("binds", "axis"))
            bound.append(axes)

            own_definitions = definitions_by_stmt[statement_index]
            for slot, vertex in enumerate(binders_by_stmt[statement_index]):
                self.relation(stmt_vertex, vertex, ("defines", "export" if children else slot))
            for kind, name, role in occurrences:
                if kind == "resource":
                    target = self._resource(name)
                else:
                    target = axes.get(name)
                    if target is None:
                        target = local_sources.get(name)
                    if target is None:
                        target = own_definitions.get(name)
                    if target is None:
                        target = defining_vertex(name, statement_index)
                    if target is None:
                        target = self._fixed_name(name)
                self.relation(stmt_vertex, target, (kind, role))

            reads, writes, state = _resources(stmt)
            for mode, names in (("read", reads), ("write", writes), ("state", state)):
                for name in names:
                    if mode != "state":
                        target = self._resource(name)
                    else:
                        target = aliases.get(name)
                        if target is None:
                            target = outer_ssa.get(name)
                        if target is None:
                            target = self._fixed_name(name)
                    self.relation(stmt_vertex, target, ("access", mode))

        incoming = ordering_constraints(body, effects=True)
        for target, sources in enumerate(incoming):
            for source in sources:
                self.relation(statement_vertices[source], statement_vertices[target], ("before",))
        nested_scopes: list[tuple[_Scope, ...]] = []
        for index, (stmt, stmt_vertex, axes) in enumerate(zip(body, statement_vertices, bound, strict=True)):
            children = stmt.nested()
            if not children:
                nested_scopes.append(())
                continue  # what follows is only what a nested scope sees; a leaf has none
            exported = {
                name: definitions_by_stmt[index][name]
                for child in children
                for name in _ordered_exported_accs(child)
                if name in definitions_by_stmt[index]
            }
            visible_ssa = dict(outer_ssa)
            for name in definition_sites:
                target = definitions_by_stmt[index].get(name)
                if target is None:
                    target = defining_vertex(name, index)
                if target is not None:
                    visible_ssa[name] = target
            built_children = []
            for child_index, child in enumerate(children):
                child_scope = self.scope(child, visible_ssa, axes, local_sources, exported)
                built_children.append(child_scope)
                self.relation(stmt_vertex, child_scope.vertex, ("body", child_index))
            nested_scopes.append(tuple(built_children))

        return _Scope(
            body=body,
            vertex=scope_vertex,
            statements=tuple(statement_vertices),
            incoming=tuple(frozenset(sources) for sources in incoming),
            children=tuple(nested_scopes),
            categories=tuple(categories),
            shapes=tuple(shapes),
        )


@dataclass(slots=True)
class _PartitionCell:
    vertices: set[int]
    serial: int
    queued: bool = False
    #: An input cell nothing has split yet, of a partition that was equitable before one of its
    #: cells was individualized: as a splitter it can split nothing, so its turn is skipped.
    settled: bool = False


def _equitable_partition(
    partition: tuple[tuple[int, ...], ...],
    incoming: Sequence[Sequence[tuple[int, int]]],
    outgoing: Sequence[Sequence[tuple[int, int]]],
    individualized: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Refine vertex colors with the standard smaller-half worklist algorithm.

    Each directed relation color is a separate splitter.  Processing only a cell's smaller
    replacement parts bounds relation visits by ``O((vertices + edges) log vertices)``.

    ``individualized`` is the index of the singleton an individualization just split off a partition
    that was EQUITABLE: every other input cell, bar the remainder that follows it, counts uniformly
    into every cell, so until something splits it its turn as a splitter is a no-op. Skipping those
    turns keeps the queue order and the cell serials — hence the result — exactly what visiting them
    produces, while a search node costs what the individualization disturbs rather than the whole
    graph again.
    """
    owner: list[_PartitionCell | None] = [None] * sum(map(len, partition))
    cells: dict[int, _PartitionCell] = {}
    work: deque[_PartitionCell] = deque()
    next_serial = 0

    def enqueue(cell: _PartitionCell) -> None:
        if not cell.queued:
            cell.queued = True
            work.append(cell)

    for position, vertices in enumerate(partition):
        settled = individualized is not None and position not in (individualized, individualized + 1)
        cell = _PartitionCell(set(vertices), next_serial, settled=settled)
        cells[next_serial] = cell
        next_serial += 1
        for vertex in vertices:
            owner[vertex] = cell
        enqueue(cell)

    while work:
        splitter = work.popleft()
        assert splitter.queued
        splitter.queued = False
        if splitter.settled:
            continue

        # Per directed relation color, how many of the splitter's vertices each neighbour touches.
        # Plain dicts: this is the innermost loop of every identity, and ``Counter`` pays a Python
        # call per new key.
        buckets: dict[tuple[int, int], dict[int, int]] = {}
        for vertex in splitter.vertices:
            for color, target in outgoing[vertex]:
                counts = buckets.setdefault((0, color), {})
                counts[target] = counts.get(target, 0) + 1
            for color, source in incoming[vertex]:
                counts = buckets.setdefault((1, color), {})
                counts[source] = counts.get(source, 0) + 1

        for _relation, counts in sorted(buckets.items()):
            touched: dict[int, tuple[_PartitionCell, dict[int, set[int]]]] = {}
            for vertex, multiplicity in counts.items():
                cell = owner[vertex]
                assert cell is not None
                _, parts = touched.setdefault(cell.serial, (cell, {}))
                parts.setdefault(multiplicity, set()).add(vertex)

            for _, (cell, nonzero_parts) in sorted(touched.items()):
                covered = sum(map(len, nonzero_parts.values()))
                if len(nonzero_parts) == 1 and covered == len(cell.vertices):
                    continue

                parts = dict(nonzero_parts)
                if covered < len(cell.vertices):
                    parts[0] = cell.vertices - set().union(*nonzero_parts.values())
                retained = max(parts, key=lambda value: (len(parts[value]), value))

                was_queued = cell.queued
                children: list[_PartitionCell] = []
                for value, vertices in sorted(parts.items()):
                    if value == retained:
                        cell.vertices = vertices
                        cell.settled = False
                        child = cell
                    else:
                        child = _PartitionCell(vertices, next_serial)
                        cells[next_serial] = child
                        next_serial += 1
                        for vertex in vertices:
                            owner[vertex] = child
                    children.append(child)

                # A queued cell's parts all stay queued; otherwise the retained (largest) part is
                # the one the smaller-half rule lets go.
                for child in children:
                    if was_queued or child is not cell:
                        enqueue(child)

    return tuple(tuple(sorted(cell.vertices)) for cell in cells.values())


def _canonical_labeling(
    colors: Sequence[object], edges: Iterable[tuple[int, int, object]], *, _prune: bool = True
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exact canonical ranks and automorphism-orbit ranks for a colored multigraph."""
    color_text = tuple(repr(color) for color in colors)
    edge_text = tuple((source, target, repr(color)) for source, target, color in edges)
    count = len(color_text)
    if not count:
        return (), ()

    color_ids = {color: index for index, color in enumerate(sorted(set(color_text)))}
    vertex_colors = tuple(color_ids[color] for color in color_text)
    relation_ids = {color: index for index, color in enumerate(sorted({color for _, _, color in edge_text}))}
    relations = tuple((source, target, relation_ids[color]) for source, target, color in edge_text)

    outgoing: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    incoming: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    for source, target, color in relations:
        outgoing[source].append((color, target))
        incoming[target].append((color, source))

    groups: dict[int, list[int]] = {}
    for vertex, color in enumerate(vertex_colors):
        groups.setdefault(color, []).append(vertex)
    initial = tuple(tuple(groups[color]) for color in sorted(groups))
    generators: list[tuple[int, ...]] = []
    #: The vertices each generator moves, parallel to ``generators``: an automorphism between two
    #: leaves that differ in one individualized vertex moves a handful, and every use below —
    #: does it fix a prefix, which cell members does it join — reads only those.
    moved: list[frozenset[int]] = []
    known: set[tuple[int, ...]] = set()

    def automorphism(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        """The vertex map between two leaves of one certificate: relabeling the graph by either
        order spells the same colored relations, so the map is an automorphism by construction."""
        permutation = [0] * count
        for source, target in zip(left, right, strict=True):
            permutation[source] = target
        return tuple(permutation)

    def inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
        out = [0] * count
        for index, mapped in enumerate(permutation):
            out[mapped] = index
        return tuple(out)

    def certificate(order: tuple[int, ...]) -> tuple:
        ranks = {vertex: rank for rank, vertex in enumerate(order)}
        ordered_relations = tuple(sorted((ranks[source], ranks[target], color) for source, target, color in relations))
        return tuple(vertex_colors[vertex] for vertex in order), ordered_relations

    def learn(left: tuple[int, ...], right: tuple[int, ...]) -> None:
        generator = automorphism(left, right)
        if generator not in known:
            known.add(generator)
            support = frozenset(vertex for vertex, mapped in enumerate(generator) if mapped != vertex)
            for permutation in (generator, inverse(generator)):
                generators.append(permutation)
                moved.append(support)

    class _Orbits:
        """The orbits of a node's cell under the generators that fix its prefix pointwise —
        union-find over the cell, fed every such generator learned so far and extended as the
        node's own children learn more, so a child in the orbit of an explored one is never
        descended. A node's eligible generators are its parent's that also fix the vertex it
        individualized, plus whatever was learned since."""

        def __init__(self, cell: tuple[int, ...], prefix: tuple[int, ...], parent: _Orbits | None) -> None:
            self.parents = {vertex: vertex for vertex in cell}
            self.fixed = frozenset(prefix)
            if parent is None:
                self.eligible: list[int] = []
                self.seen = 0
            else:
                self.eligible = [index for index in parent.eligible if prefix[-1] not in moved[index]]
                self.seen = parent.seen
            for index in self.eligible:
                self.join(index)
            self.absorb()

        def root(self, vertex: int) -> int:
            parents = self.parents
            while parents[vertex] != vertex:
                parents[vertex] = parents[parents[vertex]]
                vertex = parents[vertex]
            return vertex

        def join(self, index: int) -> None:
            generator = generators[index]
            for vertex in moved[index]:
                if vertex in self.parents:
                    left, right = self.root(vertex), self.root(generator[vertex])
                    if left != right:
                        self.parents[right] = left

        def absorb(self) -> None:
            for index in range(self.seen, len(generators)):
                if moved[index].isdisjoint(self.fixed):
                    self.eligible.append(index)
                    self.join(index)
            self.seen = len(generators)

    # The first leaf and the least leaf so far, by certificate: a later leaf equal to either is
    # its image under an automorphism, and so is the whole subtree that leaf hangs from, up to the
    # node where its path leaves the reference's — every certificate in there was already seen.
    references: dict[tuple, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def search(
        partition: tuple[tuple[int, ...], ...],
        prefix: tuple[int, ...],
        *,
        refined: bool = False,
        parent: _Orbits | None = None,
    ) -> tuple[tuple, tuple[int, ...], tuple[int, ...]]:
        if not refined:
            partition = _equitable_partition(partition, incoming, outgoing)
        choices = [(len(cell), index) for index, cell in enumerate(partition) if len(cell) > 1]
        if not choices:
            order = tuple(cell[0] for cell in partition)
            return certificate(order), order, prefix

        _, cell_index = min(choices)
        cell = partition[cell_index]
        orbits = _Orbits(cell, prefix, parent) if _prune else None
        explored: set[int] = set()
        best: tuple[tuple, tuple[int, ...], tuple[int, ...]] | None = None
        for vertex in cell:
            if orbits is not None:
                orbits.absorb()
                if orbits.root(vertex) in explored:
                    continue
                explored.add(orbits.root(vertex))
            rest = tuple(member for member in cell if member != vertex)
            individualized = (*partition[:cell_index], (vertex,), rest, *partition[cell_index + 1 :])
            child = _equitable_partition(individualized, incoming, outgoing, cell_index)
            result = search(child, (*prefix, vertex), refined=True, parent=orbits)
            if best is None or result[0] < best[0]:
                best = result
            elif result[0] == best[0]:
                learn(best[1], result[1])
            reference = references.get(result[0])
            if reference is None:
                if _prune and (not references or result[0] < min(references)):
                    if len(references) > 1:
                        del references[max(references)]
                    references[result[0]] = (result[1], result[2])
            elif reference[1] != result[2]:
                learn(reference[0], result[1])
                if reference[1][: len(prefix)] != prefix:
                    return best
        assert best is not None
        return best

    refined = _equitable_partition(initial, incoming, outgoing)
    if all(len(cell) == 1 for cell in refined):
        order = tuple(cell[0] for cell in refined)
        labeling = certificate(order), order
    else:
        # Exact graph canonization has no known near-linear worst-case algorithm.  Keep the
        # individualization search off the ordinary path and use it only for unresolved cells.
        labeling = search(refined, (), refined=True)
    ranks = [0] * count
    for rank, vertex in enumerate(labeling[1]):
        ranks[vertex] = rank

    parents = list(range(count))

    def root(vertex: int) -> int:
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex

    for generator in generators:
        for vertex, mapped in enumerate(generator):
            left, right = root(vertex), root(mapped)
            if left != right:
                parents[right] = left
    minima: dict[int, int] = {}
    for vertex, rank in enumerate(ranks):
        representative = root(vertex)
        minima[representative] = min(rank, minima.get(representative, rank))
    orbit_ranks = tuple(minima[root(vertex)] for vertex in range(count))
    return tuple(ranks), orbit_ranks


def _canonical_ranks(colors: Sequence[object], edges: Iterable[tuple[int, int, object]], *, _prune: bool = True) -> tuple[int, ...]:
    """Exact canonical vertex ranks for a colored directed multigraph."""
    return _canonical_labeling(colors, edges, _prune=_prune)[0]


@dataclass(frozen=True)
class Ordering:
    """One body's relation graph, built once and labeled under any resource coloring.

    The graph never spelled a name, so it also describes every alpha-rename of the body it was
    built from; whoever materializes an order from it renames the result sequentially.
    """

    colors: tuple[str, ...]
    edges: tuple[tuple[int, int, str], ...]
    root: _Scope
    resources: tuple[tuple[str, int], ...]
    #: The spellings the graph could not bind: every free name of the body.
    fixed_names: tuple[str, ...]

    def label(self, resource_color: Callable[[str], object] | None = None) -> Labeling:
        """Canonical ranks with every external resource colored by ``resource_color`` (bare when
        ``None``): the body's structure decides the labeling, never a resource's spelling."""
        colors = list(self.colors)
        if resource_color is not None:
            for name, vertex in self.resources:
                colors[vertex] = repr(("resource", resource_color(name)))
        ranks, orbit_ranks = _canonical_labeling(colors, self.edges)
        return Labeling(self, ranks, orbit_ranks)


@dataclass(frozen=True)
class Labeling:
    """Canonical vertex ranks of one :class:`Ordering`."""

    ordering: Ordering
    ranks: tuple[int, ...]
    orbit_ranks: tuple[int, ...]

    def resources(self) -> tuple[str, ...]:
        """External resource names in canonical rank order."""
        return tuple(name for name, _ in sorted(self.ordering.resources, key=lambda item: self.ranks[item[1]]))

    def materialize(self, *, spelled: bool) -> tuple[Body, Ordering]:
        """The body in canonical dependency-valid order, with the graph re-indexed to it.

        Ready nested scopes stay ahead of leaf epilogues. ``spelled`` breaks the remaining ties by
        each statement's name-free shape and its spelled form before the ranks, which keeps the
        executable order stable under a spelling-only buffer rename; identity never spells.
        """
        body, root = self._materialize(self.ordering.root, spelled)
        return body, Ordering(self.ordering.colors, self.ordering.edges, root, self.ordering.resources, self.ordering.fixed_names)

    def _materialize(self, scope: _Scope, spelled: bool) -> tuple[Body, _Scope]:
        rebuilt: list[Stmt] = []
        children: list[tuple[_Scope, ...]] = []
        for stmt, scopes in zip(scope.body, scope.children, strict=True):
            if scopes:
                built = [self._materialize(child, spelled) for child in scopes]
                stmt = stmt.with_bodies(tuple(body for body, _ in built))
                children.append(tuple(child for _, child in built))
            else:
                children.append(())
            rebuilt.append(stmt)
        body = Body(rebuilt)
        if spelled:
            spellings = tuple(repr(form(stmt.rename(_ABSTRACT_NAMES))) for stmt in body)

            def priority(index: int, _stmt: Stmt) -> tuple:
                vertex = scope.statements[index]
                return scope.categories[index], (scope.shapes[index], spellings[index]), self.orbit_ranks[vertex], self.ranks[vertex]

        else:

            def priority(index: int, _stmt: Stmt) -> tuple:
                return scope.categories[index], self.ranks[scope.statements[index]]

        order = body.topological_permutation(scope.incoming, priority)
        ordered = Body(body[index] for index in order)
        return ordered, scope.permuted(order, ordered, tuple(children[index] for index in order))


def relation_graph(stmts: Body) -> Ordering:
    """Build the complete body tree's colored relation graph."""
    builder = _Builder()
    root = builder.scope(Body.coerce(stmts), {}, {}, {}, {}, root=True)
    return Ordering(tuple(builder.colors), tuple(builder.edges), root, tuple(builder.resources.items()), tuple(builder.fixed_names))
