"""Value numbering of a Loop IR body with coordinates abstracted.

A statement's NUMBER is its kind and payload (op, dtype, buffer) over its operands' numbers, with every maximal
coordinate-only index expression a PARAMETER of the statement, numbered by first appearance across the statement and
its operands. A statement is thereby a function of coordinate expressions: ``W[k, (h / 384) * 128 + d]`` under one
loop nest and ``W[k, g]`` under another number alike, as do the same computation inlined under two consumers. A reduce
binds the parameters that mention its axis and keeps the free coordinates those parameters read. Commutative operands
order by number. Two statements with one number compute one function; the same number applied to the same coordinate
expressions in one scope is one INSTANCE.

The digest of a body is the sorted numbers of its stores, with every buffer keyed by how it is used rather than
spelled (two passes), so a renaming or a dependency-valid reordering of the body never reaches it. Over the
realization corpus it partitions kernels exactly as ``canonicalize_identity`` does.

Pure functions over a ``Body``; nothing here is stored, every answer is computed where it is read.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import blake2b
from itertools import permutations, product

from emmy.compiler.ir.stmt import blocks, leaves
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.structural import form

__all__ = ["Numbering", "digest", "value_numbers"]

_VAR = ("Var",)
_NAMES = re.compile(r"'([^']+)'")


@dataclass
class Numbering:
    """One body's numbering, keyed by statement ``id``."""

    #: statement -> (number, parameters); a parameter is ``("expr", repr)`` of its coordinate expression
    numbers: dict[int, tuple[str, tuple]] = field(default_factory=dict)
    #: statement -> the enclosing loops, outermost first, as (axis name, loop statement id)
    scope: dict[int, tuple[tuple[str, int], ...]] = field(default_factory=dict)
    #: statement -> the axis names its parameters read
    free: dict[int, frozenset[str]] = field(default_factory=dict)
    #: statement -> load / assign / init / reduce / select / store / other
    kind: dict[int, str] = field(default_factory=dict)
    #: number -> the numbers of its operands, for cone walks
    operands: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: the stores' (number, parameters)
    stores: list[tuple[str, tuple]] = field(default_factory=list)
    #: buffer -> the numbers of the loads and stores touching it
    touch: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def instance(self, statement: int) -> tuple[str, tuple[str, ...]]:
        """The number applied to its coordinate expressions as spelled: what one scope computes once."""
        number, params = self.numbers[statement]
        return number, tuple(text for _, text in params)


def value_numbers(body: Body, buffer_key: Callable[[str], object] = lambda name: name) -> Numbering:
    """Number every statement of ``body``; ``buffer_key`` spells a buffer in a load's or store's payload."""
    out = Numbering()
    env: dict[str, tuple[str, tuple]] = {}
    axes: dict[str, str] = {}
    enclosing: list[tuple[str, int]] = []

    def is_axis_var(value: object) -> bool:
        return isinstance(value, tuple) and len(value) == 2 and value[0] == _VAR[0] and isinstance(value[1], str) and value[1] in axes

    def reads_axis(value: object) -> bool:
        return is_axis_var(value) or (isinstance(value, tuple) and any(reads_axis(part) for part in value))

    def coordinate_only(value: object) -> bool:
        if isinstance(value, tuple) and len(value) == 2 and value[0] == _VAR[0] and isinstance(value[1], str):
            return value[1] in axes
        return not isinstance(value, tuple) or all(coordinate_only(part) for part in value)

    def abstract(expr, params: list, ops: list) -> object:
        """``form(expr)`` with every maximal coordinate-only subexpression a parameter and every SSA read an operand."""

        def walk(value: object) -> object:
            if isinstance(value, tuple) and reads_axis(value) and coordinate_only(value):
                key = ("expr", repr(value))
                if key not in params:
                    params.append(key)
                return ("c", params.index(key))
            if isinstance(value, tuple) and len(value) == 2 and value[0] == _VAR[0] and isinstance(value[1], str):
                if value[1] in env:
                    ops.append(env[value[1]])
                    return ("v", len(ops) - 1)
                return ("free", value[1])
            if isinstance(value, tuple):
                return tuple(walk(part) for part in value)
            return value

        return walk(form(expr))

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
    ) -> tuple[str, tuple]:
        """Number ``kind``/``payload`` over ``ops``, unifying parameters by their expression; ``bound`` axes reduce.
        Commutative operands are ordered by number, and operands tied on number by whichever order gives the
        smallest layout, so the source order never reaches the number."""
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
                ops = min(candidates, key=lambda order: repr(laid_out(order, params)[1]))
        parent, mapped = laid_out(ops, params)

        def binds(param: tuple) -> bool:
            return any(f"'{axis}'" in param[1] for axis in bound)

        def extents(param: tuple) -> tuple[str, ...]:
            return tuple(axes[name] for name in _NAMES.findall(param[1]) if name in axes)

        reduced = tuple(index for index, param in enumerate(parent) if binds(param))
        kept = [param for param in parent if not binds(param)]
        for param in parent:
            if binds(param):
                # A bound composite still reads its free coordinates: they stay as bare parameters of the reduce.
                for name in _NAMES.findall(param[1]):
                    if name in axes and name not in bound:
                        bare = ("expr", repr(("Var", name)))
                        if bare not in kept:
                            kept.append(bare)
        # A value is a function of its coordinates whatever their range: extents enter only where a kernel's domain
        # is fixed — a reduce's bound axes (in its payload) and a store's sweep — so a piece that re-parameterizes
        # a composite index as a grid axis numbers its values as the parent did, while two shapes digest apart.
        domain = tuple(extents(param) for param in kept) if kind == "store" else ()
        number = blake2b(repr((kind, payload, tuple(mapped), reduced, domain)).encode(), digest_size=12).hexdigest()
        out.operands.setdefault(number, tuple(number for number, _ in ops))
        return number, tuple(kept)

    def define(stmt, name: str | None, value: tuple[str, tuple], kind: str) -> None:
        if name is not None:
            env[name] = value
        out.numbers[id(stmt)] = value
        out.kind[id(stmt)] = kind
        out.scope[id(stmt)] = tuple(enclosing)
        out.free[id(stmt)] = frozenset(name for _, text in value[1] for name in _NAMES.findall(text) if name in axes)

    def visit(stmts) -> None:
        for stmt in stmts:
            if isinstance(stmt, leaves.Load):
                params: list = []
                ops: list = []
                index = tuple(abstract(expr, params, ops) for expr in stmt.index)
                value = compose("load", (buffer_key(stmt.input), index, str(stmt.dtype), stmt.carried), ops, params)
                out.touch[stmt.input].append(value[0])
                for name in stmt.names:
                    define(stmt, name, value, "load")
            elif isinstance(stmt, leaves.Assign):
                ops = [env[arg] for arg in stmt.args if arg in env]
                frees = tuple(arg for arg in stmt.args if arg not in env)
                payload = (str(stmt.op), str(stmt.dtype), frees)
                define(
                    stmt, stmt.name, compose("assign", payload, ops, [], commutative=bool(getattr(stmt.op, "commutative", False))), "assign"
                )
            elif isinstance(stmt, leaves.Init):
                define(stmt, stmt.name, compose("init", (repr(stmt.identity), str(stmt.dtype)), [], []), "init")
            elif isinstance(stmt, leaves.Accum):
                ops = [env[stmt.value]] if stmt.value in env else []
                bound = tuple(getattr(axis, "name", str(axis)) for axis in stmt.axes)
                payload = (str(stmt.op), str(stmt.dtype), tuple(axes.get(axis) for axis in bound), repr(stmt.base))
                define(stmt, stmt.name, compose("reduce", payload, ops, [], bound=bound), "reduce")
            elif isinstance(stmt, leaves.Select):
                params, ops, payload = [], [], []
                for branch in stmt.branches:
                    if branch.value in env:
                        ops.append(env[branch.value])
                    payload.append((branch.value in env, abstract(branch.select, params, ops)))
                define(stmt, stmt.name, compose("select", tuple(payload), ops, params), "select")
            elif isinstance(stmt, leaves.Write):
                params, ops = [], []
                index = tuple(abstract(expr, params, ops) for expr in stmt.index)
                ops += [env[name] for name in stmt.values if name in env]
                value = compose("store", (buffer_key(stmt.output), index, stmt.atomic, repr(stmt.swizzle)), ops, params)
                out.touch[stmt.output].append(value[0])
                define(stmt, None, value, "store")
                out.stores.append(value)
            elif isinstance(stmt, (blocks.Loop, blocks.StridedLoop)):
                axes[stmt.axis.name] = str(stmt.axis.extent)
                enclosing.append((stmt.axis.name, id(stmt)))
                visit(stmt.body)
                enclosing.pop()
            elif isinstance(stmt, blocks.Cond):
                visit(stmt.body)
                visit(stmt.else_body or ())
            else:
                define(stmt, None, (blake2b(repr(form(stmt)).encode(), digest_size=12).hexdigest(), ()), "other")

    visit(Body.coerce(body))
    return out


def digest(body: Body) -> str:
    """The body's identity material: the sorted numbers of its stores with buffers keyed by use, not spelling."""
    first = value_numbers(body, buffer_key=lambda name: "buffer")
    color = {name: tuple(sorted(numbers)) for name, numbers in first.touch.items()}
    rank = {use: index for index, use in enumerate(sorted(set(color.values())))}
    second = value_numbers(body, buffer_key=lambda name: ("buffer", rank[color[name]]))
    return blake2b(repr(sorted(number for number, _ in second.stores)).encode(), digest_size=16).hexdigest()
