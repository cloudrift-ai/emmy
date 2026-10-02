"""Compute matching coordinate-select branches once, at selected load indices.

A concatenation along an axis lowers to a :class:`Select` whose branches read the same buffers at
different offsets: RoPE's rotate-half is ``cat(-x[..., 64:], x[..., :64])``, so each cell computes
``-(w[i + 64] * norm(x[i + 64]))`` and ``w[i - 64] * norm(x[i - 64])`` and keeps one. Each branch
clamps its own index in range (``(i < 64) ? i : 0``), so every cell issues both branches' loads,
half of them for a value it throws away.

When the two branches' private statement chains are the same computation up to their load indices
— and possibly one unary op on top of one branch, the ``negative`` above — the chain is emitted
once, each load reading at ``cond ? index_a : index_b``, and the select picks between the chain's
value and that unary op of it. The inner clamps fold against the condition, so RoPE's partner read
becomes ``(i < 64) ? i + 64 : i - 64``: one load of the row and one of the weight per cell instead
of two each.

For more than two branches, every private chain must match exactly. Nested index selects retain
the branches' priority and use the last branch as the fallback, regardless of its predicate.

Structural and idempotent: once merged, the select's branches share their chain and no longer match.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import Expr, Literal, TernaryExpr
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Let, Stmt
from emmy.compiler.ir.stmt.leaves import Load, Select, SelectBranch
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    uses = Counter(name for stmt in op.body.iter() for name in _reads(stmt))
    body = _walk(op.body, uses)
    if body == op.body:
        raise RuleSkipped("no coordinate select with isomorphic branches")
    return replace(op, body=body)


def _reads(stmt: Stmt) -> tuple[str, ...]:
    """Every SSA name ``stmt`` reads, its own block bodies' reads left to their statements."""
    return tuple(stmt.deps())


def _walk(body: Body, uses: Counter) -> Body:
    stmts = []
    for stmt in body:
        nested = stmt.nested()
        if nested:
            stmt = stmt.with_bodies(tuple(_walk(inner, uses) for inner in nested))
        stmts.append(stmt)
    position = 0
    while position < len(stmts):
        if isinstance(stmts[position], Select) and (merged := _merge(stmts, position, uses)) is not None:
            # The select moved to where branch B's statements no longer precede it.
            position -= len(stmts) - len(merged)
            stmts = merged
        position += 1
    return Body(tuple(stmts))


def _cone(defs: dict[str, int], stmts: list, name: str, uses: Counter, cone: set[int]) -> bool:
    """Collect into ``cone`` the positions of the private chain defining ``name`` — statements whose
    results nothing outside the chain reads. ``False`` when ``name`` is read elsewhere too."""
    position = defs.get(name)
    if position is None:
        return True  # defined outside this body: a shared input of the chain, not part of it
    stmt = stmts[position]
    if not isinstance(stmt, (Assign, Load)) or (isinstance(stmt, Load) and (not stmt.is_scalar or stmt.carried)):
        return False
    if uses[name] != 1:
        return False
    cone.add(position)
    return all(_cone(defs, stmts, arg, uses, cone) for arg in (stmt.args if isinstance(stmt, Assign) else ()))


def _merge(stmts: list, position: int, uses: Counter) -> list | None:
    select: Select = stmts[position]
    if len(select.branches) != 2:
        return _merge_many(stmts, position, uses)
    first, other = select.branches
    cond = first.select
    defs = {name: i for i, stmt in enumerate(stmts[:position]) for name in stmt.defines()}
    wrapped, inner = _peel(stmts, defs, first.value, other.value)
    if inner is None:
        return None
    a_name, b_name = inner
    cone_a: set[int] = set()
    cone_b: set[int] = set()
    if not (_cone(defs, stmts, a_name, uses, cone_a) and _cone(defs, stmts, b_name, uses, cone_b)):
        return None
    if not cone_a or cone_a & cone_b:
        return None
    if not _safe(stmts, defs, cone_a | cone_b, (cond,)):
        return None
    pairs: dict[str, str] = {}
    if not _match(stmts, defs, a_name, b_name, cone_a, cone_b, pairs):
        return None
    # The merged chain replaces branch A's statements in A's order; branch B's are dropped.
    merged: dict[int, Stmt] = {}
    for i in sorted(cone_a):
        stmt = stmts[i]
        if isinstance(stmt, Load):
            twin = stmts[defs[pairs[stmt.name]]]
            index = tuple(_select_index(cond, a, b) for a, b in zip(stmt.index, twin.index, strict=True))
            if any(defs.get(name, -1) >= i for e in index for name in e.free_vars()):
                return None
            merged[i] = replace(stmt, index=index)
        else:
            merged[i] = stmt
    out: list[Stmt] = []
    for i, stmt in enumerate(stmts):
        if i in cone_b:
            continue
        if i == position:
            if wrapped is None:
                out.append(Assign(select.name, "copy", (a_name,)))
            elif wrapped == "a":
                # Branch A's unary op stays where it was, now over the merged chain.
                out.append(replace(select, branches=(first, SelectBranch(a_name, other.select))))
            else:
                # Branch B's unary op is re-read off the merged chain, beside the select.
                out.append(replace(stmts[defs[other.value]], args=(a_name,)))
                out.append(replace(select, branches=(SelectBranch(a_name, cond), other)))
            continue
        if wrapped == "b" and i == defs.get(other.value):
            continue  # re-emitted beside the select, after the merged chain
        if wrapped == "a" and i == defs.get(first.value):
            out.append(replace(stmt, args=(a_name,)))
            continue
        out.append(merged.get(i, stmt))
    return out


def _safe(stmts, defs, cone, predicates):
    first, last = min(cone), max(cone)
    return not any(not isinstance(s, (Assign, Let, Load, Select)) for s in stmts[first : last + 1]) and not any(
        defs.get(name, -1) >= first for predicate in predicates for name in predicate.free_vars()
    )


def _merge_many(stmts, position, uses):
    """Merge any number of private, isomorphic branches with the select's original priority."""
    select = stmts[position]
    if len(select.branches) < 3:
        return None
    defs = {name: i for i, stmt in enumerate(stmts[:position]) for name in stmt.defines()}
    cones = []
    pairs = []
    first = select.branches[0].value
    for branch in select.branches:
        cone = set()
        if not _cone(defs, stmts, branch.value, uses, cone) or not cone or any(cone & old for old in cones):
            return None
        mapping = {}
        if cones and not _match(stmts, defs, first, branch.value, cones[0], cone, mapping):
            return None
        cones.append(cone)
        pairs.append(mapping)
    combined = set.union(*cones)
    if not _safe(stmts, defs, combined, tuple(b.select for b in select.branches[:-1])):
        return None
    merged = {}
    for i in sorted(cones[0]):
        stmt = stmts[i]
        if not isinstance(stmt, Load):
            continue
        twins = [stmt, *(stmts[defs[mapping[stmt.name]]] for mapping in pairs[1:])]
        index = []
        for coordinates in zip(*(s.index for s in twins), strict=True):
            chosen = coordinates[-1]
            for branch, coordinate in reversed(tuple(zip(select.branches[:-1], coordinates[:-1], strict=True))):
                chosen = _select_index(branch.select, coordinate, chosen)
            if any(defs.get(name, -1) >= i for name in chosen.free_vars()):
                return None
            index.append(chosen)
        merged[i] = replace(stmt, index=tuple(index))
    dropped = combined - cones[0]
    return [
        Assign(select.name, "copy", (first,)) if i == position else merged.get(i, stmt) for i, stmt in enumerate(stmts) if i not in dropped
    ]


def _peel(stmts: list, defs: dict, a: str, b: str) -> tuple[str | None, tuple[str, str] | None]:
    """The two chains to match: the branch values themselves, or — when one branch applies one unary
    op to a value the other branch's shape matches — that op's operand. Returns which side carried the
    op (``"a"`` / ``"b"`` / ``None``) and the pair of names to match."""
    if _same_shape(stmts, defs, a, b):
        return None, (a, b)
    for side, (outer, twin) in (("a", (a, b)), ("b", (b, a))):
        stmt = stmts[defs[outer]] if outer in defs else None
        if isinstance(stmt, Assign) and len(stmt.args) == 1 and _same_shape(stmts, defs, stmt.args[0], twin):
            return side, ((stmt.args[0], twin) if side == "a" else (twin, stmt.args[0]))
    return None, None


def _same_shape(stmts: list, defs: dict, a: str, b: str) -> bool:
    if a == b or a not in defs or b not in defs:
        return False
    sa, sb = stmts[defs[a]], stmts[defs[b]]
    if isinstance(sa, Assign) and isinstance(sb, Assign):
        return sa.op == sb.op and sa.dtype == sb.dtype and len(sa.args) == len(sb.args)
    return isinstance(sa, Load) and isinstance(sb, Load) and sa.input == sb.input


def _match(stmts: list, defs: dict, a: str, b: str, cone_a: set, cone_b: set, pairs: dict) -> bool:
    """Whether the chains defining ``a`` and ``b`` are one computation up to load indices."""
    if a == b:
        return defs.get(a) not in cone_a
    ia, ib = defs.get(a), defs.get(b)
    if ia not in cone_a or ib not in cone_b:
        return False
    sa, sb = stmts[ia], stmts[ib]
    if isinstance(sa, Load) and isinstance(sb, Load):
        ok = sa.input == sb.input and sa.dtype == sb.dtype and len(sa.index) == len(sb.index) and sa.is_scalar and sb.is_scalar
    elif isinstance(sa, Assign) and isinstance(sb, Assign):
        ok = sa.op == sb.op and sa.dtype == sb.dtype and len(sa.args) == len(sb.args)
        ok = ok and all(_match(stmts, defs, x, y, cone_a, cone_b, pairs) for x, y in zip(sa.args, sb.args, strict=True))
    else:
        ok = False
    if ok:
        pairs[a] = b
    return ok


def _select_index(cond: Expr, a: Expr, b: Expr) -> Expr:
    """``cond ? a : b`` with every select on ``cond`` inside each arm resolved to its own side."""
    if a == b:
        return a
    return TernaryExpr(cond, _resolve(a, cond, True), _resolve(b, cond, False))


def _resolve(expr: Expr, cond: Expr, value: bool) -> Expr:
    if isinstance(expr, TernaryExpr) and expr.cond == cond:
        return _resolve(expr.if_true if value else expr.if_false, cond, value)
    fields = getattr(expr, "__dataclass_fields__", {})
    changed = {
        name: _resolve(getattr(expr, name), cond, value)
        for name in fields
        if isinstance(getattr(expr, name), Expr) and not isinstance(getattr(expr, name), Literal)
    }
    return replace(expr, **changed) if changed else expr
