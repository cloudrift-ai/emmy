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
from emmy.compiler.ir.stmt import Assign, Let, Select, SelectBranch, Statement

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

    def resolved(expression: Expression) -> Expression | None:
        """``expression`` with every name it reads spelled by ``term``; ``None`` when one resolves to nothing."""
        terms = {name: term(name) for name in expression.free_vars()}
        if any(t is None for t in terms.values()):
            return None
        return expression.substitute(terms)

    def comparison(expression: Expression) -> BinaryExpression | None:
        """An ordered comparison over resolved sides, as the predicate it is; ``None`` for any other value."""
        op = expression.operation
        if op is None or op.name not in _COMPARE:
            return None
        left, right = (resolved(operand) for operand in expression.operands)
        if left is None or right is None:
            return None
        return BinaryExpression(_COMPARE[op.name], left, right)

    def branch(operand: Expression, name: str) -> str | None:
        """The name a select branch carries: a uniform value, a bound scalar, or a literal bound here."""
        if isinstance(operand, Var):
            return operand.name if operand.name in uniform | scalars.keys() else None
        if isinstance(operand, Literal):
            scalars[name] = operand
            out.append(Let(name, operand))
            return name
        return None

    def selected(stmt: Assign) -> Expression | None:
        """``stmt``'s value with every ``where`` over a coordinate predicate spelled as a ``Select`` statement — the
        value itself when it is one, else a name the sum reads — or ``None`` when a branch is no uniform value."""
        masks = 0

        def rewrite(node: Expression) -> Expression:
            nonlocal masks
            op = node.operation
            if op is None or op.name != "where":
                return node
            cond, keep, fill = node.operands
            predicate = predicates.get(cond.name) if isinstance(cond, Var) else comparison(cond)
            if predicate is None:
                return node
            name = stmt.name if node == stmt.value else f"{stmt.name}__m{masks}"
            masks += 1
            keep_name, fill_name = branch(keep, f"{name}__keep"), branch(fill, f"{name}__fill")
            if keep_name is None or fill_name is None:
                raise ValueError("a mask branch that is no uniform value")
            negated = BinaryExpression(_NEGATE[predicate.op], predicate.left, predicate.right)
            branches = (SelectBranch(value=keep_name, select=predicate), SelectBranch(value=fill_name, select=negated))
            out.append(Select(name=name, branches=branches))
            return Var(name)

        try:
            return stmt.value.rebuild(rewrite)
        except ValueError:
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
        op = stmt.value.operation if isinstance(stmt, Assign) else None
        if isinstance(stmt, Assign) and reads & coords.keys():
            if op is not None and op.name in _COMPARE:
                predicate = comparison(stmt.value)
                if predicate is None:
                    return None
                predicates[stmt.name] = predicate
                continue
            # A coordinate computed from coordinates: the mask's ``key - row``, a split piece's absolute key.
            coordinate = resolved(stmt.value)
            coordinate = _integral(coordinate) if coordinate is not None else None
            if coordinate is None:
                return None
            coords[stmt.name] = coordinate
            continue
        if isinstance(stmt, Assign):
            # A ``where`` over a coordinate predicate — the value itself, or a term of the sum the value is after the
            # normal form folded the mask into its reader — is a coordinate ``Select`` the sum reads by name.
            value = selected(stmt)
            if value is None:
                return None
            if value == Var(stmt.name):
                continue  # the value was the mask, now a Select statement of that name
            stmt = stmt.replace(value=value)
            reads = set(stmt.deps())
        if reads & (coords.keys() | predicates.keys() | aliased):
            return None  # a coordinate read outside a mask
        out.append(stmt)
    return out


def _integral(expression: Expression) -> Expression | None:
    """``expression`` as a coordinate: a float literal that is a whole number becomes the integer; any other float
    literal makes it no coordinate (``None``)."""

    def integer(node: Expression) -> Expression:
        if isinstance(node, Literal) and node.dtype == "float":
            if not float(node.value).is_integer():
                raise ValueError("a fractional coefficient")
            return Literal(int(node.value), "int")
        return node

    try:
        return expression.rebuild(integer)
    except ValueError:
        return None
