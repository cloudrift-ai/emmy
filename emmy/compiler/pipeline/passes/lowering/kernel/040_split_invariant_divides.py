"""Hoist reciprocals of invariant floating divisors when FAST_MATH permits the rounding change.

Run after dtype stamping so integer division stays exact. Keep this policy out of body
normalization: structural identity must not depend on the live arithmetic flags.
"""

from dataclasses import replace
from itertools import count

from emmy.compiler.dtype import BF16, F16, F32, F64
from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Loop, Stmt, StridedLoop
from emmy.compiler.ir.stmt.order import ordering_constraints
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import FAST_MATH, precision_pin

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    if not precision_pin(FAST_MATH):
        raise RuleSkipped("FAST_MATH is disabled")
    op = root.op
    body = split_invariant_divides(op.body)
    if body == op.body:
        raise RuleSkipped("no invariant floating divisor")
    return replace(op, body=_hoist_loop_invariants(body))


def split_invariant_divides(body: Body) -> Body:
    axes = body.axis_dependencies
    names = set(axes)
    fresh = (f"recip_{i}" for i in count() if f"recip_{i}" not in names)

    def split(stmt: Stmt):
        if isinstance(stmt, Assign) and stmt.op.name == "divide" and stmt.dtype in (F16, BF16, F32, F64):
            numerator, divisor = stmt.args
            if axes.get(divisor, frozenset()) < axes.get(numerator, frozenset()):
                reciprocal = next(fresh)
                return (
                    Assign(reciprocal, "reciprocal", (divisor,), dtype=stmt.dtype),
                    replace(stmt, op="multiply", args=(numerator, reciprocal)),
                )
        return stmt

    return body.map(split)


def _hoist_loop_invariants(stmts: Body) -> Body:
    """Move stmts out of ``Loop``s whose axis they don't depend on.

    Hoists ``Load`` / ``Assign`` / ``Select`` (SSA values) and entire
    ``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond`` blocks whose contents
    transitively avoid the outer axis — provided the block contains no
    ``Write`` (a Write hoist would change observable side effects).
    Block-level hoisting is what lets a Loop and its downstream consumer
    move together: hoisting just the consumer would leave it referencing
    an Accum still defined inside the outer Loop body.

    ``Accum`` / ``Init`` / ``Write`` always stay (iteration-tied
    semantics). Axis-invariance alone does not earn a hoist: the hoisted set is closed under the
    scope's ordering constraints (:func:`~emmy.compiler.ir.stmt.order.ordering_constraints`), so
    a statement that must follow one that stays — the consumer of an accumulator a pinned
    reduction exports, a read of a buffer the loop writes, anything behind an ordered execution
    protocol such as a barrier or a declaration — stays with it.
    """
    stmts = Body.coerce(stmts)
    name_axes = stmts.axis_dependencies
    axis_names = stmts.axis_names
    axis_deps: dict[int, tuple[Stmt, frozenset[str]]] = {}

    def _axis_deps(s: Stmt) -> frozenset[str]:
        """Axes read by one immutable subtree, computed bottom-up once."""
        key = id(s)
        cached = axis_deps.get(key)
        if cached is not None and cached[0] is s:
            return cached[1]
        reads = set(s.deps())
        for expr in s.exprs():
            reads.update(expr.free_vars())
        deps = reads & axis_names
        for name in reads:
            deps.update(name_axes.get(name, frozenset()))
        for child in (child for body in s.nested() for child in body):
            deps.update(_axis_deps(child))
        result = frozenset(deps - s.binds_axes())
        axis_deps[key] = (s, result)
        return result

    def _hoistable(s: Stmt, axis: str) -> bool:
        # Accum / Init are scope-bound to their enclosing Loop's reduction (an Init seeds an
        # Accum or a Carrier's state per output cell) — they can't move alone, but the
        # whole enclosing block can. Side-effecting stmts (Write, or any block containing a
        # Write) pin their iteration count and stay put.
        if isinstance(s, (Accum, Init)) or s.has_side_effects:
            return False
        return axis not in _axis_deps(s)

    def _closed_under_constraints(inner: list[Stmt], candidates: set[int]) -> set[int]:
        """``candidates`` less every statement that must follow one that stays.

        A nested reduction can export an accumulator that varies with none of the outer axes
        while its own loop stays pinned (attention's denominator is produced inside the value
        sweep, which is pinned by the head-dim axis the value slab reads); its consumer then
        reads as invariant and would move above the definition. Iterated: un-hoisting one
        candidate can pin the next."""
        if not candidates:
            return candidates
        incoming = ordering_constraints(Body(inner), effects=True)
        while pinned := {index for index in candidates if incoming[index] - candidates}:
            candidates -= pinned
        return candidates

    def walk(body: Body) -> list[Stmt]:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, (Loop, StridedLoop)):
                inner = walk(s.body)
                axis = s.axis.name
                hoisted = _closed_under_constraints(inner, {index for index, c in enumerate(inner) if _hoistable(c, axis)})
                new_body.extend(c for index, c in enumerate(inner) if index in hoisted)
                new_body.append(replace(s, body=tuple(c for index, c in enumerate(inner) if index not in hoisted)))
            elif isinstance(s, Cond):
                new_body.append(Cond(cond=s.cond, body=tuple(walk(s.body)), else_body=tuple(walk(s.else_body))))
            else:
                new_body.append(s)
        return new_body

    return tuple(walk(stmts))
