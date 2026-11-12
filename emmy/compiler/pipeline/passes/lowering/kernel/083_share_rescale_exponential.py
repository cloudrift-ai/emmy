"""Compute an online softmax's two rescale factors with one exponential.

``SOFTMAX.advance`` (``ir/pure/twist.py``) moves a pivot pair ``(g, g_o)`` to ``gn = maximum(g, g_o)`` and gives each
side the factor ``exp`` of its distance below it: ``alpha = exp(g - gn)`` and ``beta = exp(g_o - gn)``. The serial,
cooperative and register folds compute the pair per element, the butterfly and tree merges per round, a split's
finalize per partial. ``gn`` is one of the two pivots, so one distance is ``0`` and its factor ``1``, and one
exponential of the other distance is the other factor. Where neither distance is ``0`` no pivot is finite and the
first one's distance ``s`` is the exponential's argument; the second distance is then ``-inf`` (factor ``0``) or NaN
(factor NaN), a select:

    s     = dg == 0 ? dg_o : dg
    e     = exp(s)
    alpha = dg == 0 ? 1 : e
    beta  = dg == 0 ? e : (dg_o == 0 ? 1 : (dg_o < 0 ? 0 : dg_o))

Each factor is the one the two exponentials compute — the same bits for a number, a NaN for a NaN — whichever
exponential the kernel takes and with or without flush to zero, so the rewrite is exact. It removes work only where
the exponential is the accurate one: there it saves an ``expf`` for two compares and four selects. The fast
exponential (``FAST_EXP``) is one ``MUFU`` and one multiply, fewer instructions than the selects (a four-row
cooperative softmax ran 1.32x slower on the RTX 5090 with them), so under it the pair keeps both exponentials. The
rewrite decides nothing either way: it reads the kernel's exponential, as ``085_fast_exp``, which runs after it,
does. The recipe stays as it is: it is Loop IR,
every attention kernel's identity reads it, and the chunk tier's matcher reads the factor's shape there.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import Var, apply
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Statement, free_names
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import FAST_EXP, precision_pin

PATTERN = [Pattern("root", KernelOp)]
#: The names the block adds beside the first factor's: the exponential's argument and its result.
_FRESH = ("__s", "__e")


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if precision_pin(FAST_EXP):
        raise RuleSkipped("the fast exponential costs fewer instructions than the selects that would share it")
    body = _walk(op.body)
    if body == op.body:
        raise RuleSkipped("no rescale factor pair")
    return replace(op, body=body)


def _walk(body: Body) -> Body:
    stmts = []
    for stmt in body:
        if stmt.nested():
            stmt = stmt.with_bodies(tuple(_walk(inner) for inner in stmt.nested()))
        # A cooperative combine holds its merge as a statement field, not a body; it is rendered at its ``dtype``.
        merge = getattr(stmt, "combine_states", None)
        if merge and stmt.dtype == F32:
            shared = _share(list(merge), typed=False)
            if shared != list(merge):
                stmt = stmt.replace(combine_states=tuple(shared))
        stmts.append(stmt)
    shared = _share(stmts, typed=True)
    return body if shared == list(body) else Body(tuple(shared))


def _operands(stmt: Statement, name: str, arity: int) -> tuple[str, ...] | None:
    """The names ``stmt`` applies ``name`` to, when it is an ``Assign`` of exactly that over ``arity`` names."""
    if not isinstance(stmt, Assign) or stmt.value.operation is None or stmt.value.operation.name != name:
        return None
    operands = stmt.value.operands
    if len(operands) != arity or not all(isinstance(operand, Var) for operand in operands):
        return None
    return tuple(operand.name for operand in operands)


def _share(stmts: list[Statement], *, typed: bool) -> list[Statement]:
    """``stmts`` with each rescale factor pair of this one scope computed by one exponential."""
    writes: dict[str, list[int]] = {}
    for index, stmt in enumerate(stmts):
        for name in stmt.defines():
            writes.setdefault(name, []).append(index)

    def defined(name: str, before: int) -> int | None:
        """Where the one definition of ``name`` in this scope sits, when it is ahead of ``before``."""
        found = writes.get(name, [])
        return found[0] if len(found) == 1 and found[0] < before else None

    def untouched(name: str, start: int, stop: int) -> bool:
        """Whether nothing between ``start`` and ``stop`` redefines ``name``."""
        return not any(start < index < stop for index in writes.get(name, []))

    sides: dict[str, dict[str, tuple[int, str]]] = {}  # pivot → side → (the factor's position, its distance)
    for index, stmt in enumerate(stmts):
        (distance,) = _operands(stmt, "exp", 1) or (None,)
        if distance is None or stmt.dtype not in ((F32,) if typed else (None, F32)) or len(writes[stmt.name]) != 1:
            continue
        at = defined(distance, index)
        difference = _operands(stmts[at], "subtract", 2) if at is not None else None
        if difference is None or (pivot_at := defined(difference[1], at)) is None:
            continue
        side, pivot = difference
        pair = _operands(stmts[pivot_at], "maximum", 2)
        # The side the distance reads is the one the maximum read.
        if pair is not None and side in pair and pair[0] != pair[1] and untouched(side, pivot_at, at):
            sides.setdefault(pivot, {})[side] = (index, distance)
    replaced: dict[int, list[Statement]] = {}
    for found in sides.values():
        if len(found) != 2:
            continue
        (first, dg), (late, dg_o) = sorted(found.values())
        alpha, beta = stmts[first], stmts[late]
        if alpha.dtype != beta.dtype or any(f"{alpha.name}{suffix}" in writes for suffix in _FRESH):
            continue
        # The block takes the earlier factor's place where both distances are ahead of it, else the later one's; the
        # factor it moves is read by nothing in between. Its two names are fresh.
        between = stmts[first + 1 : late]
        if writes[dg_o][0] < first and not any(beta.name in free_names(stmt) for stmt in between):
            at = first
        elif not any(alpha.name in free_names(stmt) for stmt in between):
            at = late
        else:
            continue
        replaced[first] = replaced[late] = []
        replaced[at] = _one_exponential(alpha, beta, dg, dg_o)
    return [new for index, stmt in enumerate(stmts) for new in replaced.get(index, [stmt])]


def _one_exponential(alpha: Assign, beta: Assign, dg: str, dg_o: str) -> list[Statement]:
    """The two factors ``alpha = exp(dg)`` and ``beta = exp(dg_o)`` from one exponential (the module docstring)."""
    at_pivot = apply("equal", dg, 0.0)
    argument, power = (f"{alpha.name}{suffix}" for suffix in _FRESH)
    other = apply("where", apply("equal", dg_o, 0.0), 1.0, apply("where", apply("less", dg_o, 0.0), 0.0, dg_o))
    return [
        Assign(argument, apply("where", at_pivot, dg_o, dg), dtype=alpha.dtype),
        Assign(power, apply("exp", argument), dtype=alpha.dtype),
        Assign(alpha.name, apply("where", at_pivot, 1.0, power), dtype=alpha.dtype),
        Assign(beta.name, apply("where", at_pivot, power, other), dtype=alpha.dtype),
    ]
