"""The additive coordinate mask the chunk tier lands, read off a score's prefix however the program spelled it.

The SDPA decomposition spells a causal or banded mask as a coordinate ``Select`` whose predicate compares the key and
the row; a program that masks with ``where(key - row <= position, 0, fill)`` lowers to a comparison and a ``where``
over values that alias the coordinates (``Let v = a1``), and the uniform values it reads. Both are the same mask:
``select_masks`` spells the second as the first, so the tier's gate and its emission read one form.
"""

from __future__ import annotations

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.stmt import Affine, Assign, Let, Select, SelectBranch, Stmt

_COMPARE = {"less": "<", "less_equal": "<=", "greater": ">", "greater_equal": ">="}
_NEGATE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}


def coordinate_aliases(stmts, axes: frozenset[str] | set[str]) -> dict[str, str]:
    """``{name: axis}`` for every ``Let`` in ``stmts`` that binds a bare coordinate of ``axes``."""
    return {s.name: s.value.name for s in stmts if isinstance(s, Let) and isinstance(s.value, Var) and s.value.name in axes}


def select_masks(stmts, aliases: dict[str, str], uniform: set[str] | frozenset[str], axes: frozenset[str] | set[str]) -> list[Stmt] | None:
    """``stmts`` with every ``where`` over a comparison of coordinates and uniform values spelled as a coordinate
    ``Select``, the coordinate aliases it reads folded into the predicate — or ``None`` when a coordinate reaches a
    statement no ``Select`` can carry, which no per-fragment evaluation can place."""
    coords: dict[str, Expr] = {name: Var(axis) for name, axis in aliases.items()}
    scalars: dict[str, Expr] = {}
    predicates: dict[str, BinaryExpr] = {}

    def term(name: str) -> Expr | None:
        if name in coords:
            return coords[name]
        if name in scalars:
            return scalars[name]
        if name in uniform or name in axes:
            return Var(name)
        return None

    out: list[Stmt] = []
    for stmt in stmts:
        if isinstance(stmt, Let) and stmt.name in coords:
            continue  # the alias folds into every predicate that reads it
        if isinstance(stmt, Let) and isinstance(stmt.value, Literal):
            scalars[stmt.name] = stmt.value
            out.append(stmt)
            continue
        reads = set(stmt.deps())
        if isinstance(stmt, Affine) and reads & coords.keys():
            terms = [term(arg) for arg in stmt.args]
            if any(t is None for t in terms) or any(not float(c).is_integer() for c in stmt.coefficients) or stmt.offset:
                return None
            expr: Expr | None = None
            for t, coefficient in zip(terms, stmt.coefficients, strict=True):
                scaled = t if coefficient == 1 else BinaryExpr("*", Literal(int(coefficient), "int"), t)
                expr = scaled if expr is None else BinaryExpr("+", expr, scaled)
            coords[stmt.name] = expr
            continue
        if isinstance(stmt, Assign) and stmt.op.name in _COMPARE and reads & coords.keys():
            left, right = (term(arg) for arg in stmt.args)
            if left is None or right is None:
                return None
            predicates[stmt.name] = BinaryExpr(_COMPARE[stmt.op.name], left, right)
            continue
        if isinstance(stmt, Assign) and stmt.op.name == "where" and stmt.args[0] in predicates:
            keep, fill = stmt.args[1:]
            if not {keep, fill} <= uniform | scalars.keys():
                return None
            predicate = predicates[stmt.args[0]]
            negated = BinaryExpr(_NEGATE[predicate.op], predicate.left, predicate.right)
            branches = (SelectBranch(value=keep, select=predicate), SelectBranch(value=fill, select=negated))
            out.append(Select(name=stmt.name, branches=branches))
            continue
        if reads & (coords.keys() | predicates.keys() | set(aliases.values())):
            return None  # a coordinate read outside a mask
        out.append(stmt)
    return out
