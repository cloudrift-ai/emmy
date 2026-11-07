"""The additive coordinate mask the chunk tier lands, read off a score's prefix however the program spelled it.

The SDPA decomposition spells a causal or banded mask as a coordinate ``Select`` whose predicate compares the key and
the row; a program that masks with ``where(key - row <= position, 0, fill)`` lowers to a comparison and a ``where``
over values that alias the coordinates (``Let v = a1``), and the uniform values it reads. Both are the same mask:
``select_masks`` spells the second as the first, so the tier's gate and its emission read one form. A key split's
piece reads the absolute key through its partition (``Let v = ks * b + a2``): an alias is any ``Let`` over the
coordinates and the uniform values, folded into the predicate as the expression it binds.
"""

from __future__ import annotations

from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, Var
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Affine, Assign, Let, Select, SelectBranch, Statement

_COMPARE = {"less": "<", "less_equal": "<=", "greater": ">", "greater_equal": ">="}
_NEGATE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}


def cell_uniform(axes, node: Fold, coord_axes: frozenset[str] | set[str]) -> frozenset[str]:
    """The names one value serves every cell of a CTA: what the carrier's uniform leaves expose, the coordinates
    the launch fixes per CTA — the contraction's batch axes and a key split's partition — and the symbolic extents
    of the kernel's axis table ``axes``."""
    contraction = node.as_contraction()
    per_cta = node.free_axes - contraction.left_axes - contraction.right_axes
    extents = {name for axis in axes for name in axis.extent.expression.free_vars()}
    leaves = {name for edge in node.operands if edge.uniform_leaf(frozenset(coord_axes)) for name in edge.exposes}
    return frozenset(leaves | per_cta | extents)


def coordinate_aliases(stmts, axes: frozenset[str] | set[str], uniform: frozenset[str] | set[str]) -> dict[str, Expression]:
    """``{name: expression}`` for every ``Let`` in ``stmts`` that binds an expression over a coordinate of ``axes``
    and the uniform values — a bare coordinate, or a split piece's absolute key."""
    return {
        s.name: s.value
        for s in stmts
        if isinstance(s, Let) and s.value.free_vars() & set(axes) and s.value.free_vars() <= set(axes) | set(uniform)
    }


def select_masks(
    stmts, aliases: dict[str, Expression], uniform: set[str] | frozenset[str], axes: frozenset[str] | set[str]
) -> list[Statement] | None:
    """``stmts`` with every ``where`` over a comparison of coordinates and uniform values spelled as a coordinate
    ``Select``, the coordinate aliases it reads folded into the predicate — or ``None`` when a coordinate reaches a
    statement no ``Select`` can carry, which no per-fragment evaluation can place."""
    coords: dict[str, Expression] = dict(aliases)
    aliased = {name for expression in aliases.values() for name in expression.free_vars() if name in axes}
    scalars: dict[str, Expression] = {}
    predicates: dict[str, BinaryExpression] = {}

    def term(name: str) -> Expression | None:
        if name in coords:
            return coords[name]
        if name in scalars:
            return scalars[name]
        if name in uniform or name in axes:
            return Var(name)
        return None

    out: list[Statement] = []
    for stmt in stmts:
        if isinstance(stmt, Let) and stmt.name in coords:
            continue  # the alias folds into every predicate that reads it
        if isinstance(stmt, Let) and stmt.value.free_vars() <= set(uniform):
            scalars[stmt.name] = stmt.value  # a literal, or a value over uniform names alone (a symbolic extent)
            out.append(stmt)
            continue
        reads = set(stmt.deps())
        if isinstance(stmt, Affine) and reads & coords.keys():
            terms = [term(arg) for arg in stmt.args]
            if any(t is None for t in terms) or any(not float(c).is_integer() for c in stmt.coefficients) or stmt.offset:
                return None
            expression: Expression | None = None
            for t, coefficient in zip(terms, stmt.coefficients, strict=True):
                scaled = t if coefficient == 1 else BinaryExpression("*", Literal(int(coefficient), "int"), t)
                expression = scaled if expression is None else BinaryExpression("+", expression, scaled)
            coords[stmt.name] = expression
            continue
        if isinstance(stmt, Assign) and stmt.op.name in _COMPARE and reads & coords.keys():
            left, right = (term(arg) for arg in stmt.args)
            if left is None or right is None:
                return None
            predicates[stmt.name] = BinaryExpression(_COMPARE[stmt.op.name], left, right)
            continue
        if isinstance(stmt, Assign) and stmt.op.name == "where" and stmt.args[0] in predicates:
            keep, fill = stmt.args[1:]
            if not {keep, fill} <= uniform | scalars.keys():
                return None
            predicate = predicates[stmt.args[0]]
            negated = BinaryExpression(_NEGATE[predicate.op], predicate.left, predicate.right)
            branches = (SelectBranch(value=keep, select=predicate), SelectBranch(value=fill, select=negated))
            out.append(Select(name=stmt.name, branches=branches))
            continue
        if reads & (coords.keys() | predicates.keys() | aliased):
            return None  # a coordinate read outside a mask
        out.append(stmt)
    return out
