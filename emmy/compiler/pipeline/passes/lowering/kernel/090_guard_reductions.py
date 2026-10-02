"""Skip pure scalar reductions outside the coordinates that consume their results."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from importlib import import_module

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, TernaryExpr
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Let, Load, Loop, Select, StridedLoop, Write
from emmy.compiler.ir.stmt.body import free_names
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import GUARD_REDUCTIONS

PATTERN = [Pattern("root", KernelOp)]
_TRUE, _FALSE = Literal(1, "int"), Literal(0, "int")
_resolve = import_module("emmy.compiler.pipeline.passes.lowering.kernel.045_merge_select_loads")._resolve
_stamp = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types")


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if GUARD_REDUCTIONS.name in op.knobs:
        raise RuleSkipped("GUARD_REDUCTIONS already decided")
    enabled = GUARD_REDUCTIONS.narrow((True,))[0]
    body = _guard_stores(_guard(op.body)) if enabled else op.body
    return replace(op, body=body, knobs={**op.knobs, GUARD_REDUCTIONS.name: enabled})


def _guard_stores(body: Body, types=None) -> Body:
    """Sink private scalar cones and their stores into the branch that consumes them.

    Only a select immediately followed by its scalar stores can move. Other readers, nested
    effects and intervening writes to a cone's input keep the original stream unchanged.
    """
    if types is None:
        ctx = _stamp._StampCtx({})
        _stamp._seed_explicit_dtypes(body, ctx)
        _stamp._stamp_body(body, ctx)
        types = ctx.ssa_dtypes
    stmts = [stmt.with_bodies(tuple(_guard_stores(b, types) for b in stmt.nested())) if stmt.nested() else stmt for stmt in body]
    uses = Counter(name for stmt in Body(stmts).iter() for name in free_names(stmt))
    position = 0
    while position < len(stmts):
        select = stmts[position]
        if not isinstance(select, Select) or len(select.branches) != 2:
            position += 1
            continue
        end = position + 1
        names = {select.name}
        continuation = []
        while end < len(stmts) and isinstance(stmts[end], Assign) and names & set(stmts[end].deps()):
            continuation.append(stmts[end])
            names.add(stmts[end].name)
            end += 1
        start = end
        while end < len(stmts) and isinstance(stmts[end], Write) and stmts[end].is_scalar and stmts[end].value in names:
            end += 1
        stores = stmts[start:end]
        readers = Counter(name for stmt in (*continuation, *stores) for name in free_names(stmt))
        if not stores or any(uses[name] != readers[name] for name in names) or any(
            s.atomic or any(names & e.free_vars() for e in s.index) for s in stores
        ):
            position += 1
            continue
        defs = {name: i for i, stmt in enumerate(stmts[:position]) for name in stmt.defines()}

        def cone(name, found):
            i = defs.get(name)
            if i is None or uses[name] != 1 or not isinstance(stmts[i], (Assign, Let, Load, Select)):
                return
            if isinstance(stmts[i], Load) and (not stmts[i].is_scalar or stmts[i].carried):
                return
            found.add(i)
            for arg in free_names(stmts[i]):
                cone(arg, found)

        branches = []
        for branch in select.branches:
            found = set()
            cone(branch.value, found)
            branches.append(found)
        moved = branches[0] | branches[1]
        reads = {stmts[i].input for i in moved if isinstance(stmts[i], Load)}
        if not moved or branches[0] & branches[1] or any(
            not isinstance(stmt, (Assign, Let, Load, Select, Write)) or getattr(stmt, "output", None) in reads
            for i, stmt in enumerate(stmts[:position])
            if min(moved) <= i and i not in moved
        ):
            position += 1
            continue
        cond = select.branches[0].select.simplify(SimplifyCtx.empty())
        guarded = []
        for truth, (branch, found) in zip((True, False), zip(select.branches, branches, strict=True), strict=True):
            chain = []
            for i in sorted(found):
                stmt = stmts[i]
                if isinstance(stmt, Load):
                    stmt = replace(stmt, index=tuple(_resolve(e.simplify(SimplifyCtx.empty()), cond, truth) for e in stmt.index))
                chain.append(stmt)
            chain.append(Assign(select.name, "copy", (branch.value,), dtype=types.get(select.name, F32)))
            guarded.append(_guard_stores(Body((*chain, *continuation, *stores)), types))
        replacement = Cond(cond, guarded[0], guarded[1])
        stmts = [stmt for i, stmt in enumerate(stmts) if i not in moved and not position <= i < end]
        position -= len(moved)
        stmts.insert(position, replacement)
        position += 1
    return Body(stmts)


def _boolean(op: str, left: Expr, right: Expr) -> Expr:
    return left if left == right and op in ("&&", "||") else BinaryExpr(op, left, right).simplify(SimplifyCtx(ranges={}))


def _guard(body: Body, axes: frozenset[str] = frozenset()) -> Body:
    """Propagate each value's coordinate demand backward through flat pure computations.

    Unknown consumers demand the value everywhere. Only enclosing coordinates may bound a
    loop: a predicate computed after it, or by its own iterations, cannot guard its execution.
    Scalar-only bodies exclude stores, synchronization and warp operations. The reduction's
    identity seed stays outside the zero-trip loop, so every SSA name remains defined.
    """
    demand: dict[str, Expr] = {}

    def need(name: str, predicate: Expr) -> None:
        demand[name] = _boolean("||", demand.get(name, _FALSE), predicate)

    rewritten = []
    for stmt in reversed(body):
        if stmt.nested():
            stmt = stmt.with_bodies(tuple(_guard(nested, axes | stmt.binds_axes()) for nested in stmt.nested()))
        if isinstance(stmt, Select):
            remaining = demand.get(stmt.name, _TRUE)
            for index, branch in enumerate(stmt.branches):
                # Rendering treats the final branch as the unconditional fallback.
                if index == len(stmt.branches) - 1:
                    need(branch.value, remaining)
                else:
                    need(branch.value, _boolean("&&", remaining, branch.select))
                    remaining = _boolean("&&", remaining, _boolean("==", branch.select, _FALSE))
                    for name in branch.select.free_vars():
                        need(name, _TRUE)
        elif isinstance(stmt, (Assign, Load)):
            wanted = _FALSE
            for name in stmt.defines():
                wanted = _boolean("||", wanted, demand.get(name, _TRUE))
            for name in free_names(stmt):
                need(name, wanted)
        else:
            if isinstance(stmt, (Loop, StridedLoop)) and stmt.seed:
                carriers = [s.name for s in stmt.body if isinstance(s, Accum)]
                wanted = _FALSE
                for name in carriers:
                    wanted = _boolean("||", wanted, demand.get(name, _TRUE))
                scalar = all(
                    isinstance(s, (Loop, StridedLoop, Load, Assign, Accum, Select)) and (not isinstance(s, (Loop, StridedLoop)) or s.seed)
                    for s in stmt.body.iter()
                )
                if carriers and scalar and wanted != _TRUE and wanted.free_vars() <= axes:
                    end = stmt.end if isinstance(stmt, StridedLoop) and stmt.end is not None else stmt.axis.extent_expr()
                    if isinstance(stmt, Loop):
                        stmt = StridedLoop(stmt.axis, _FALSE, _TRUE, stmt.body, unroll=stmt.unroll, seed=stmt.seed)
                    stmt = replace(stmt, end=TernaryExpr(wanted, end, _FALSE))
            for name in free_names(stmt):
                need(name, _TRUE)
        rewritten.append(stmt)
    return Body(reversed(rewritten))
