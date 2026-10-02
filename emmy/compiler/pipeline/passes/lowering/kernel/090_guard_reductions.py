"""Skip pure scalar reductions outside the coordinates that consume their results."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, TernaryExpr
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Select, StridedLoop
from emmy.compiler.ir.stmt.body import free_names
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import GUARD_REDUCTIONS

PATTERN = [Pattern("root", KernelOp)]
_TRUE, _FALSE = Literal(1, "int"), Literal(0, "int")


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if GUARD_REDUCTIONS.name in op.knobs:
        raise RuleSkipped("GUARD_REDUCTIONS already decided")
    enabled = GUARD_REDUCTIONS.narrow((True,))[0]
    return replace(op, body=_guard(op.body) if enabled else op.body, knobs={**op.knobs, GUARD_REDUCTIONS.name: enabled})


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
