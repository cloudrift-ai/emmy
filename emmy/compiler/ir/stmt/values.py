"""Value numbering of a Loop IR body with coordinates abstracted, and the identity it gives a body.

A statement's NUMBER is its kind and payload (op, dtype, resource) over its operands' numbers, with every maximal
coordinate-only index expression a PARAMETER of the statement, numbered by first appearance across the statement and
its operands. A statement is thereby a function of coordinate expressions: ``W[k, (h / 384) * 128 + d]`` under one
loop nest and ``W[k, g]`` under another number alike, as do the same computation inlined under two consumers. A reduce
binds the parameters that mention its axis and keeps the free coordinates those parameters read. Commutative operands
order by number. Two statements with one number compute one function; the same number applied to the same coordinate
expressions in one scope is one INSTANCE.

A body's identity is the hash of its SCOPE TREE: every block a node described by its kind and its extent or predicate,
every leaf an instance with its coordinates spelled by binding depth, every scope's members sorted. The external
buffers and carried states it reads through are keyed by USE, never by spelling: the sorted numbers of the statements
touching one, refined where two tie by individualizing each in turn and keeping the one that gives the smaller tree.
The order that ranks them is the order of the roles a deployed kernel binds its buffers to.

Pure functions over a ``Body``; nothing here is stored, every answer is computed where it is read.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import blake2b
from itertools import groupby, permutations, product

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import blocks, leaves
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.order import bound_axes
from emmy.compiler.structural import form

__all__ = ["Numbering", "digest", "distinct", "effect_order", "roles", "scope_tree", "spelled_by_depth", "value_numbers"]

#: A parameter: ``("expr", form)`` of a coordinate-only expression, every axis spelled by its binding depth.
Param = tuple[str, object]


def _var(value: object) -> str | None:
    """The name a rendered ``Var`` spells, else ``None``."""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "Var" and isinstance(value[1], str):
        return value[1]
    return None


def _bound(value: object, out: set[int] | None = None) -> set[int]:
    """Every binding depth a rendered expression reads an axis at."""
    out = set() if out is None else out
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "Var" and isinstance(value[1], int):
        out.add(value[1])
    elif isinstance(value, tuple):
        for part in value:
            _bound(part, out)
    return out


def _hash(*parts: object) -> str:
    return blake2b(repr(parts).encode(), digest_size=12).hexdigest()


@dataclass
class Numbering:
    """One body's numbering, keyed by statement ``id``."""

    #: statement -> (number, parameters)
    numbers: dict[int, tuple[str, tuple[Param, ...]]] = field(default_factory=dict)
    #: statement -> load / assign / init / reduce / select / store / carry / pre / other / block
    kind: dict[int, str] = field(default_factory=dict)
    #: statement -> the enclosing blocks, outermost first, by statement id
    scope: dict[int, tuple[int, ...]] = field(default_factory=dict)
    #: block statement -> (its descriptor, its parameters): a loop's extent, a branch's predicate number
    blocks: dict[int, tuple[object, tuple[Param, ...]]] = field(default_factory=dict)
    #: number -> the numbers of its operands, for cone walks
    operands: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: the stores' (number, parameters)
    stores: list[tuple[str, tuple[Param, ...]]] = field(default_factory=list)
    #: resource -> the numbers of the statements touching it
    touch: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    #: resource -> the statements touching it, by id
    touching: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    #: statement -> the statements defining the names it reads, by id: the dataflow at statement grain
    sources: dict[int, tuple[int, ...]] = field(default_factory=dict)
    #: the external buffers, in first-touch order
    buffers: list[str] = field(default_factory=list)
    #: the carried states, in first-touch order
    states: list[str] = field(default_factory=list)

    def instance(self, statement: int) -> tuple[str, tuple[Param, ...]]:
        """The number applied to its coordinate expressions as spelled: what one scope computes once."""
        return self.numbers[statement]


def distinct(body: Body) -> Body:
    """``body`` with every statement its own object: a numbering is keyed by statement identity, and a body may hold
    one immutable statement twice."""
    body = Body.coerce(body)
    members = list(body.iter())
    return body if len({id(stmt) for stmt in members}) == len(members) else Body(stmt.rename({}) for stmt in body)


def value_numbers(body: Body, resource_key: Callable[[str], object] = lambda name: name) -> Numbering:
    """Number every statement of ``body``, which holds every statement once (:func:`distinct`); ``resource_key``
    spells a buffer or carried state in a payload."""
    from emmy.compiler.ir.stmt.passes import map_exprs  # noqa: PLC0415

    out = Numbering()
    env: dict[str, tuple[str, tuple[Param, ...]]] = {}
    axes: dict[str, str] = {}
    loops: list[str] = []
    path: list[int] = []
    carried: set[str] = set()
    touched: list[str] = []
    binding: list[str] = []
    defined_by: dict[str, int] = {}

    def by_depth(value: object) -> object:
        """A rendered expression with every bound axis spelled by its binding depth."""
        name = _var(value)
        if name is not None and name in binding:
            return ("Var", len(binding) - 1 - binding[::-1].index(name))
        return tuple(by_depth(part) for part in value) if isinstance(value, tuple) else value

    def reads_axis(value: object) -> bool:
        name = _var(value)
        if name is not None:
            return name in axes
        return isinstance(value, tuple) and any(reads_axis(part) for part in value)

    def coordinate_only(value: object) -> bool:
        name = _var(value)
        if name is not None:
            return name in axes
        return not isinstance(value, tuple) or all(coordinate_only(part) for part in value)

    def abstract(expr, params: list, ops: list) -> object:
        """``form(expr)`` with every maximal coordinate-only subexpression a parameter and every SSA read an operand."""

        def walk(value: object) -> object:
            if isinstance(value, tuple) and reads_axis(value) and coordinate_only(value):
                key = ("expr", by_depth(value))
                if key not in params:
                    params.append(key)
                return ("c", params.index(key))
            name = _var(value)
            if name is not None:
                return operand(name, ops)
            if isinstance(value, tuple):
                return tuple(walk(part) for part in value)
            return value

        return walk(form(expr))

    def operand(name: str, ops: list) -> object:
        if name in env:
            ops.append(env[name])
            return ("v", len(ops) - 1)
        if name in carried:
            # A read of a carried state ahead of its update in the loop: the previous step's value.
            return resource(name, "state", touched)
        return ("free", name)

    def resource(name: str, kind: str, number_of: list[str]) -> object:
        """A buffer or state in a payload, keyed by use rather than spelling; ``number_of`` collects what touched it."""
        held = out.states if kind == "state" else out.buffers
        if name not in held:
            held.append(name)
        number_of.append(name)
        return (kind, resource_key(name))

    depths_of: dict[Param, frozenset[int]] = {}

    def depths(param: Param) -> frozenset[int]:
        """The binding depths a parameter reads, memoized: the same parameter recurs in every consumer."""
        found = depths_of.get(param)
        if found is None:
            found = depths_of[param] = frozenset(_bound(param[1]))
        return found

    def laid_out(ops: list, params: list) -> tuple[list, list]:
        """The parent's parameters (``params`` first, then each operand's in order) and each operand's map onto them."""
        parent = list(params)
        mapped = []
        for number, its in ops:
            for param in its:
                if param not in parent:
                    parent.append(param)
            mapped.append((number, tuple(parent.index(param) for param in its)))
        return parent, mapped

    def compose(
        kind: str, payload: object, ops: list, params: list, bound: tuple[str, ...] = (), commutative: bool = False
    ) -> tuple[str, tuple[Param, ...]]:
        """Number ``kind``/``payload`` over ``ops``, unifying parameters by their expression; ``bound`` axes reduce.
        Commutative operands are ordered by number, and operands tied on number by whichever order gives the
        smallest layout and then the smallest parameter order by binding depth, so the source order never reaches
        the number nor the order of the parameters a consumer reduces by position."""
        if commutative:
            ops = sorted(ops, key=lambda value: value[0])
            groups: list[list] = []
            for value in ops:
                if groups and groups[-1][0][0] == value[0]:
                    groups[-1].append(value)
                else:
                    groups.append([value])
            if any(len(group) > 1 for group in groups):
                candidates = [
                    [value for group in arrangement for value in group]
                    for arrangement in product(*(list(permutations(group)) for group in groups))
                ]
                ops = min(candidates, key=lambda order: tuple(repr(by_depth(part)) for part in reversed(laid_out(order, params))))
        parent, mapped = laid_out(ops, params)
        reduced: tuple[int, ...] = ()
        kept = parent
        if bound:
            bound_depths = {len(binding) - 1 - binding[::-1].index(name) for name in bound if name in binding}
            reduced = tuple(index for index, param in enumerate(parent) if depths(param) & bound_depths)
            kept = [param for param in parent if not depths(param) & bound_depths]
            for param in parent:
                if depths(param) & bound_depths:
                    # A bound composite still reads its free coordinates: they stay as bare parameters of the reduce.
                    for index in sorted(depths(param) - bound_depths):
                        if (bare := ("expr", ("Var", index))) not in kept:
                            kept.append(bare)
        # A value is a function of its coordinates whatever their range; a store's sweep is its domain.
        domain = ()
        if kind == "store":
            domain = tuple(axes.get(binding[index]) if index < len(binding) else None for param in kept for index in sorted(depths(param)))
        number = _hash(kind, payload, tuple(mapped), reduced, domain)
        out.operands.setdefault(number, tuple(number for number, _ in ops))
        return number, tuple(kept)

    def define(stmt, names: tuple[str, ...], value: tuple[str, tuple[Param, ...]], kind: str) -> None:
        out.sources[id(stmt)] = tuple(defined_by[name] for name in stmt.deps() if name in defined_by)
        for name in names:
            env[name] = value
            defined_by[name] = id(stmt)
        out.numbers[id(stmt)] = value
        out.kind[id(stmt)] = kind
        out.scope[id(stmt)] = tuple(path)

    def descriptor(axis, stmt) -> str:
        window = axis.window
        parent = None if window is None or window.parent is None else str(window.parent.extent)
        flags = (stmt.unroll, stmt.seed) if isinstance(stmt, (blocks.Loop, blocks.StridedLoop)) else type(stmt).__name__
        return repr((str(axis.extent), None if window is None else (window.partition, parent), flags))

    def generic(stmt, ops: list, params: list, touched: list[str]) -> object:
        """Any statement kind: its shape with names abstracted, its expressions as parameters and operands."""
        exprs = tuple(abstract(expr, params, ops) for expr in stmt.exprs())
        defined = (*stmt.defines(), *stmt.local_decls())
        names: dict[str, str] = {name: f"$d{index}" for index, name in enumerate(defined)}
        for name in stmt.deps():
            if name not in names:
                names[name] = repr(operand(name, ops))
        for name in axes:
            names.setdefault(name, "$axis")
        for index, axis in enumerate(bound_axes(stmt)):
            names[axis.name] = f"$b{index}"
        for name in (*stmt.external_reads(), *stmt.external_writes()):
            if name not in names:
                names[name] = repr(operand(name, ops) if name in env else resource(name, "buffer", touched))
        shell = stmt.with_bodies(tuple(Body() for _ in stmt.nested())) if stmt.nested() else stmt
        shell = map_exprs(shell, lambda _expr: Var("$expr"))
        return form(shell.rename(names).rename_buffers(names)), exprs

    def visit(stmts: Body) -> None:
        nonlocal touched
        carried.update(stmts.carried_names)
        for stmt in stmts:
            touched = []
            if isinstance(stmt, leaves.Load):
                params: list = []
                ops: list = []
                index = tuple(abstract(expr, params, ops) for expr in stmt.index)
                buffer = operand(stmt.input, ops) if stmt.input in env else resource(stmt.input, "buffer", touched)
                value = compose("load", (buffer, index, str(stmt.dtype), stmt.carried, len(stmt.names)), ops, params)
                define(stmt, stmt.names, value, "load")
            elif isinstance(stmt, leaves.Assign):
                ops = [env[arg] for arg in stmt.args if arg in env]
                frees = tuple(arg for arg in stmt.args if arg not in env)
                payload = (str(stmt.op), str(stmt.dtype), frees)
                commutative = bool(getattr(stmt.op, "commutative", False))
                define(stmt, (stmt.name,), compose("assign", payload, ops, [], commutative=commutative), "assign")
            elif isinstance(stmt, leaves.Init):
                define(stmt, (stmt.name,), compose("init", (repr(stmt.identity), str(stmt.dtype)), [], []), "init")
            elif isinstance(stmt, leaves.Accum):
                ops, params = [], []
                redirected = stmt.base is not None and stmt.base != stmt.name
                value = (operand(stmt.value, ops), operand(stmt.base, ops) if redirected else None)
                bound = tuple(stmt.axes) if stmt.axes else tuple(loops[-1:])
                payload = (str(stmt.op), str(stmt.dtype), tuple(axes.get(axis) for axis in bound), value)
                define(stmt, (stmt.name,), compose("reduce", payload, ops, params, bound=bound), "reduce")
            elif isinstance(stmt, leaves.Select):
                params, ops, payload = [], [], []
                for branch in stmt.branches:
                    payload.append((operand(branch.value, ops), abstract(branch.select, params, ops)))
                define(stmt, (stmt.name,), compose("select", tuple(payload), ops, params), "select")
            elif isinstance(stmt, leaves.Write):
                params, ops = [], []
                index = tuple(abstract(expr, params, ops) for expr in stmt.index)
                values = tuple(operand(name, ops) for name in stmt.values)
                buffer = operand(stmt.output, ops) if stmt.output in env else resource(stmt.output, "buffer", touched)
                payload = (buffer, index, values, str(stmt.value_dtype), stmt.atomic, repr(stmt.swizzle))
                value = compose("store", payload, ops, params)
                define(stmt, (), value, "store")
                out.stores.append(value)
            elif isinstance(stmt, leaves.Carry):
                params, ops = [], []
                value = operand(stmt.value, ops)
                index = tuple(abstract(expr, params, ops) for expr in stmt.index)
                seed = resource(stmt.seed, "buffer", touched) if isinstance(stmt.seed, str) else repr(stmt.seed)
                state = resource(stmt.name, "state", touched)
                define(stmt, (stmt.name,), compose("carry", (state, index, value, seed, str(stmt.dtype)), ops, params), "carry")
            elif isinstance(stmt, leaves.Pre):
                params, ops = [], []
                index = tuple(abstract(expr, params, ops) for expr in stmt.index)
                state = resource(stmt.carrier, "state", touched)
                define(stmt, (stmt.name,), compose("pre", (state, index), ops, params), "pre")
            elif stmt.nested():
                params, ops = [], []
                if isinstance(stmt, blocks.Loop):
                    header: tuple = ("loop", descriptor(stmt.axis, stmt))
                elif isinstance(stmt, blocks.Cond):
                    number, params = compose("cond", (abstract(stmt.cond, params, ops),), ops, params)
                    header = ("cond", number)
                else:
                    shape, exprs = generic(stmt, ops, params, touched)
                    number, params = compose("block", (shape, exprs), ops, params)
                    header = ("block", number)
                out.blocks[id(stmt)] = (header, tuple(params))
                out.kind[id(stmt)] = "block"
                out.scope[id(stmt)] = tuple(path)
                out.sources[id(stmt)] = tuple(defined_by[name] for name in stmt.deps() if name in defined_by)
                bound = bound_axes(stmt)
                saved = dict(env), {axis.name: axes.get(axis.name) for axis in bound}
                for axis in bound:
                    axes[axis.name] = descriptor(axis, stmt)
                    binding.append(axis.name)
                if isinstance(stmt, (blocks.Loop, blocks.StridedLoop)):
                    loops.append(stmt.axis.name)
                path.append(id(stmt))
                mine = touched
                for child in stmt.nested():
                    visit(child)
                touched = mine
                path.pop()
                del binding[len(binding) - len(bound) :]
                if isinstance(stmt, (blocks.Loop, blocks.StridedLoop)):
                    loops.pop()
                exported = {name: env[name] for child in stmt.nested() for name in child.carried_names if name in env}
                env.clear()
                env.update(saved[0])
                env.update(exported)
                for name, previous in saved[1].items():
                    if previous is None:
                        del axes[name]
                    else:
                        axes[name] = previous
            else:
                params, ops = [], []
                shape, exprs = generic(stmt, ops, params, touched)
                defined = (*stmt.defines(), *stmt.local_decls())
                define(stmt, defined, compose("other", (shape, exprs), ops, params), "other")
            for name in touched:
                out.touch[name].append(out.numbers[id(stmt)][0] if id(stmt) in out.numbers else out.blocks[id(stmt)][0][1])
                out.touching[name].append(id(stmt))

    visit(Body.coerce(body))
    return out


def spelled_by_depth(value: object, depth: tuple[str, ...]) -> object:
    """A rendered expression or parameter list with every axis of ``depth`` (the bound axes, outermost first)
    spelled by its binding depth, the innermost binding of a name winning."""
    name = _var(value)
    if name is not None and name in depth:
        return ("Var", len(depth) - 1 - depth[::-1].index(name))
    return tuple(spelled_by_depth(part, depth) for part in value) if isinstance(value, tuple) else value


def effect_order(body: Body) -> list[set[int]]:
    """Per member of ``body``, the members it must follow for an effect and not for a value: a write of one buffer
    ahead of its reads and writes, a protocol statement ahead of everything after it."""
    from emmy.compiler.ir.stmt.order import ordering_constraints  # noqa: PLC0415

    return [
        ordered - dataflow
        for ordered, dataflow in zip(ordering_constraints(body, effects=True), ordering_constraints(body, effects=False), strict=True)
    ]


def scope_tree(numbering: Numbering, body: Body, depth: tuple[str, ...] = (), effects: dict[int, list[set[int]]] | None = None) -> str:
    """The hash of ``body``'s scope tree: each block by its header and its children, each leaf by its number and its
    coordinates spelled by binding depth, each scope's members sorted — so neither spelling nor order reaches it. An
    effect order two members must keep (two writes of one buffer, a protocol statement) rides the later member;
    ``effects`` memoizes it by scope across the numberings of one body."""

    def by_depth(value: object) -> object:
        return spelled_by_depth(value, depth)

    body = Body.coerce(body)
    effects = {} if effects is None else effects
    order = effects.get(id(body))
    if order is None:
        order = effects[id(body)] = effect_order(body)
    members: list[str] = []
    for index, stmt in enumerate(body):
        preceding = tuple(sorted(members[source] for source in order[index]))
        if id(stmt) in numbering.blocks:
            header, params = numbering.blocks[id(stmt)]
            bound = tuple(axis.name for axis in bound_axes(stmt))
            children = tuple(scope_tree(numbering, child, (*depth, *bound), effects) for child in stmt.nested())
            members.append(_hash("block", header, by_depth(params), children, preceding))
        else:
            number, params = numbering.numbers[id(stmt)]
            members.append(_hash("stmt", number, by_depth(params), preceding))
    return _hash(sorted(members))


def _downstream(numbering: Numbering) -> dict[str, tuple[str, ...]]:
    """Every resource's forward cone at statement grain: the sorted numbers of every statement that reads, directly or
    through others, a statement touching it. Statement grain, not value grain: two buffers loaded the same way share a
    load's number but not its readers."""
    consumers: dict[int, set[int]] = defaultdict(set)
    for stmt, sources in numbering.sources.items():
        for source in sources:
            consumers[source].add(stmt)

    def number(stmt: int) -> str:
        return numbering.numbers[stmt][0] if stmt in numbering.numbers else repr(numbering.blocks[stmt][0])

    out = {}
    for name, statements in numbering.touching.items():
        seen, stack = set(statements), list(statements)
        while stack:
            for consumer in consumers.get(stack.pop(), ()):
                if consumer not in seen:
                    seen.add(consumer)
                    stack.append(consumer)
        out[name] = tuple(sorted(number(stmt) for stmt in seen))
    return out


def _cells(
    body: Body, paint: Callable[[str], object], keyed: Callable[[], Callable[[str], object]], rank: dict[str, int], cell: dict
) -> Numbering:
    """Number until the open resources' cells stop splitting: each is colored by its type and by everything
    downstream of it. ``cell`` is updated in place."""
    numbering = value_numbers(body, keyed())
    while True:
        downstream = _downstream(numbering)
        fresh = {name: (repr(paint(name)), downstream[name]) for name in (*numbering.buffers, *numbering.states) if name not in rank}
        partition = frozenset(frozenset(name for name in fresh if fresh[name] == value) for value in fresh.values())
        if partition == frozenset(frozenset(name for name in cell if cell[name] == value) for value in cell.values()):
            return numbering
        cell.clear()
        cell.update(fresh)
        numbering = value_numbers(body, keyed())


def roles(body: Body, color: Callable[[str], object] | None = None) -> dict[str, int]:
    """Every buffer's and carried state's role at a glance: the rank of its cell of use among the cells — what an
    executable order spells a buffer by, so a renaming moves nothing unless it renames two the body cannot tell
    apart, which share a rank. Cheaper than :func:`digest`, which individualizes the ties."""
    body = distinct(body)
    paint = (lambda name: None) if color is None else color
    cell: dict[str, object] = {}
    _cells(body, paint, lambda: lambda name: ("open", paint(name), cell.get(name)), {}, cell)
    ranked = {value: index for index, value in enumerate(sorted(set(cell.values())))}
    return {name: ranked[value] for name, value in cell.items()}


def digest(body: Body, color: Callable[[str], object] | None = None) -> tuple[str, tuple[str, ...]]:
    """The body's identity and its external buffers in role order.

    Every buffer and carried state is colored by ``color`` (its type, when given) and by its use: everything
    downstream of it, refined until the cells stop splitting. The members of one cell are told apart by
    individualizing each in turn, the one whose tree hashes smaller ranking first; two whose trees hash alike are
    interchangeable, and the refinement runs again once one of them is fixed."""
    body = distinct(body)
    paint = (lambda name: None) if color is None else color
    rank: dict[str, int] = {}
    cell: dict[str, object] = {}
    effects: dict[int, list[set[int]]] = {}

    def keyed(pick: str | None = None) -> Callable[[str], object]:
        def key(name: str) -> object:
            if name in rank:
                return ("fixed", rank[name], paint(name))
            return ("pick" if name == pick else "open", paint(name), cell.get(name))

        return key

    numbering = _cells(body, paint, keyed, rank, cell)
    names = (*numbering.buffers, *numbering.states)
    while len(rank) < len(names):
        # One round ranks every cell in cell order: a singleton outright, a tie by the certificates of its members
        # under this round's coloring — each group's choice is canonical on its own, so every group fixes its
        # distinct members at once, and only a member interchangeable with another sends the round back to refine.
        for _, members in groupby(sorted((name for name in names if name not in rank), key=cell.__getitem__), key=cell.__getitem__):
            group = list(members)
            if len(group) == 1:
                rank[group[0]] = len(rank)
                continue
            certificate = {name: scope_tree(value_numbers(body, keyed(name)), body, (), effects) for name in group}
            group.sort(key=lambda name: (certificate[name], name))
            for name in group:
                rank[name] = len(rank)
                if sum(certificate[other] == certificate[name] for other in group) > 1:
                    break  # interchangeable with another: fix this one and refine before looking at the rest
        if len(rank) < len(names):
            numbering = _cells(body, paint, keyed, rank, cell)
    numbering = value_numbers(body, keyed())
    return scope_tree(numbering, body, (), effects), tuple(sorted(numbering.buffers, key=rank.__getitem__))
