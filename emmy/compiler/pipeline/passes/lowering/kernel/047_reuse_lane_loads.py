"""Keep a cooperative row in registers: unroll short lane-strided loops, then load each cell once.

A cooperative reduce with a full-row projection (RMSNorm, RoPE over a normed row, softmax) walks
its row twice per lane: the reduce loop reads ``x[lane + coop·k]``, and after the combine the
projection loop reads the same cells again — and a rotate-half partner reads a cell another trip
of the same lane already holds. Both loops are short (a 128-wide row at 32 lanes is four trips),
but nvcc cannot reuse the first loop's values in the second: the second loop stores to the
output between trips, and it cannot prove the output does not alias the row.

This pass unrolls every lane-strided loop of at most ``_MAX_TRIPS`` trips whose extent the step
divides, substituting each trip's coordinate, then drops every load of a read-only buffer (one the
kernel never writes) at an index an earlier load of the same body already read — the index
compared after folding against the lane ranges, so a partner read ``(i < 64) ? i + 64 : i - 64``
at ``i = lane + 32k`` resolves to another trip's own read. A reduce loop's carriers are seeded
ahead of the trips, as the loop would have seeded them.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Interval, Literal, SimplifyCtx
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Stmt, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Init, Let, Load, Select, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

#: The longest lane loop unrolled: past it, the registers a row holds outgrow what reuse saves.
_MAX_TRIPS = 8

_FLAT = (Assign, Accum, Load, Select, Let, Write)


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    written = frozenset(op.outputs)
    body = _walk(op.body, written, SimplifyCtx.empty())
    if body == op.body:
        raise RuleSkipped("no short lane loop to unroll")
    return replace(op, body=body)


def _walk(body: Body, written: frozenset[str], ctx: SimplifyCtx) -> Body:
    stmts: list[Stmt] = []
    for stmt in body:
        inner = ctx
        if isinstance(stmt, Tile):
            for axis in stmt.axes:
                if axis.extent.is_static:
                    inner = inner.extend(axis.name, Interval(0, axis.extent.as_static() - 1))
        nested = stmt.nested()
        if nested:
            stmt = stmt.with_bodies(tuple(_walk(b, written, inner) for b in nested))
        stmts.append(stmt)
    unrolled: list[Stmt] = []
    loops = 0
    for stmt in stmts:
        trips = _trips(stmt, ctx)
        if trips is None:
            unrolled.append(stmt)
            continue
        # Sibling loops over independently spliced cones bind the same names, each in its own C scope.
        # Flattened into one, a repeat would be a second declaration: give such a loop's trips their own.
        trip_stmts = _unroll(stmt, trips, "")
        bound = {name for s in unrolled for name in s.defines()}
        if any(name in bound for s in trip_stmts if not isinstance(s, (Init, Accum)) for name in s.defines()):
            trip_stmts = _unroll(stmt, trips, f"_{loops}")
        unrolled.extend(trip_stmts)
        loops += 1
    if not loops:
        return Body(tuple(stmts))
    return Body(tuple(_reuse(unrolled, written, ctx)))


def _trips(stmt: Stmt, ctx: SimplifyCtx) -> int | None:
    """How many trips a short lane loop takes, or ``None`` when it is not one this pass unrolls."""
    if not isinstance(stmt, StridedLoop) or stmt.end is not None or not stmt.axis.extent.is_static:
        return None
    if not isinstance(stmt.step, Literal) or not isinstance(stmt.step.value, int) or stmt.step.value < 1:
        return None
    extent, step = stmt.axis.extent.as_static(), stmt.step.value
    if extent % step or not 1 < extent // step <= _MAX_TRIPS:
        return None
    if not all(isinstance(s, _FLAT) and not s.nested() for s in stmt.body):
        return None
    # Every unrolled trip stays inside the loop only when ``start`` is proven in ``[0, step)``: a lane index or zero.
    start = stmt.start.range(ctx)
    if start is None or start.lo < 0 or start.hi >= step:
        return None
    return extent // step


def _unroll(loop: StridedLoop, trips: int, tag: str) -> list[Stmt]:
    carried = {s.name for s in loop.body if isinstance(s, Accum)}
    out: list[Stmt] = []
    if loop.seed:
        seen: set[str] = set()
        for s in loop.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                out.append(Init(s.name, s.op.identity, dtype=s.dtype))
    defined = {name for s in loop.body for name in s.defines()} - carried
    for k in range(trips):
        coord = BinaryExpr("+", loop.start, Literal(k * loop.step.value, "int"))
        sigma = Sigma({loop.axis.name: coord})
        suffix = f"__u{k}{tag}"
        for s in loop.body:
            out.append(s.rewrite(lambda name, suffix=suffix: f"{name}{suffix}" if name in defined else name, sigma))
    return out


def _reuse(stmts: list[Stmt], written: frozenset[str], ctx: SimplifyCtx) -> list[Stmt]:
    """Drop every scalar load of a read-only buffer an earlier load of this body already read,
    renaming its uses to the earlier value."""
    seen: dict[tuple, str] = {}
    rename: dict[str, str] = {}
    out: list[Stmt] = []
    for stmt in stmts:
        if rename:
            # A name the statement defines, itself or anywhere in a nested body (a staging loop unrolled
            # on its own, with trip names of its own), is a new value: only uses of this body's dropped
            # loads are renamed. Renaming a nested trip's definition onto an outer one declared the
            # outer name twice in one C scope.
            local = {name for s in Body((stmt,)).iter() for name in s.defines()}
            stmt = stmt.rewrite(lambda name, local=local: name if name in local else rename.get(name, name))
        if isinstance(stmt, Load) and stmt.is_scalar and stmt.input not in written and not stmt.carried:
            stmt = replace(stmt, index=tuple(e.simplify(ctx) for e in stmt.index))
            key = (stmt.input, stmt.dtype, tuple(_canonical(e) for e in stmt.index))
            if key in seen:
                rename[stmt.name] = seen[key]
                continue
            seen[key] = stmt.name
        out.append(stmt)
    return out


def _canonical(expr):
    """A comparison key for an index: its affine form over its free variables when it has one."""
    from emmy.compiler.ir.expr import affine_form  # noqa: PLC0415

    names = expr.free_vars()
    form = affine_form(expr, names) if names else None
    if form is None:
        return expr.pretty()
    anchor, coeffs = form
    anchor = anchor.simplify(SimplifyCtx.empty())
    return (anchor.pretty(), tuple(sorted((name, c) for name, c in coeffs.items() if c)))
