"""A softmax carrier's additive coordinate mask, read as the range of the keys a row folds.

A program that masks its scores with ``where(key - row <= position, 0, fill)`` lowers to a comparison and a ``where``
added to the score, over values that alias the coordinates (``Let v = a1``) and the values one cell holds (a position
loaded from memory). A fill that low marks a key the row does not fold, which is what a range says, so the form the
schedule and every tier take states it as one: :func:`keys_ranged` reads each such mask off a carrier's lift and
writes the bound it states of the key axis on the carrier's range (``TileOp.scheduled``). A tier then reads one
thing, the range, whichever way the program said which keys a row attends.

``select_masks`` is the reading's first half: it spells a ``where`` over a comparison of coordinates as a coordinate
``Select``, the aliases folded into the predicate. A key split's piece reads the absolute key through its partition
(``Let v = ks * b + a2``): an alias is any ``Let`` over the coordinates and the uniform values, folded into the
predicate as the expression it binds.
"""

from __future__ import annotations

from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, Var, affine_form, affine_sum, affine_terms, apply
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Assign, Body, Let, Select, SelectBranch, Statement

#: Each ordered comparison and the comparison that is its negation.
_NEGATED = {"less": "greater_equal", "less_equal": "greater", "greater": "less_equal", "greater_equal": "less"}

#: The highest fill that marks a key as one a row does not fold. The exponent of any score with it is zero in
#: FP32 unless the scores themselves span ten thousand, which no softmax input does.
_OUTSIDE = -1e4


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
    predicates: dict[str, Expression] = {}

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

    def comparison(expression: Expression) -> Expression | None:
        """An ordered comparison over resolved sides, as the predicate it is; ``None`` for any other value."""
        op = expression.operation
        if op is None or op.name not in _NEGATED:
            return None
        left, right = (resolved(operand) for operand in expression.operands)
        if left is None or right is None:
            return None
        return apply(op.name, left, right)

    def branch(operand: Expression, name: str) -> str | None:
        """The name a select branch carries: a uniform value, a bound scalar, or a literal bound here."""
        if isinstance(operand, Var):
            return operand.name if operand.name in uniform | scalars.keys() else None
        if isinstance(operand, Literal):
            scalars[name] = operand
            out.append(Let(name, operand))
            return name
        return None

    def selected(stmt: Assign) -> bool | None:
        """Whether ``stmt``'s value is a ``where`` over a coordinate predicate, then spelled as a coordinate ``Select``
        statement of ``stmt``'s name; ``None`` when a branch of one is no uniform value."""
        op = stmt.value.operation
        if op is None or op.name != "where":
            return False
        cond, keep, fill = stmt.value.operands
        predicate = predicates.get(cond.name) if isinstance(cond, Var) else None
        if predicate is None:
            return False
        keep_name, fill_name = branch(keep, f"{stmt.name}__keep"), branch(fill, f"{stmt.name}__fill")
        if keep_name is None or fill_name is None:
            return None
        negated = apply(_NEGATED[predicate.operation.name], *predicate.operands)
        branches = (SelectBranch(value=keep_name, select=predicate), SelectBranch(value=fill_name, select=negated))
        out.append(Select(name=stmt.name, branches=branches))
        return True

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
            if op is not None and op.name in _NEGATED:
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
            # A ``where`` over a coordinate predicate is a coordinate ``Select`` of the statement's name.
            masked = selected(stmt)
            if masked is None:
                return None
            if masked:
                continue
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


def keys_ranged(root: Fold, axes) -> Fold:
    """``root`` with every softmax carrier of its tree read by :func:`_keys_range`; ``root`` itself, and every
    shared term as it stood, where no carrier's score adds a mask the reading takes."""
    done: dict[int, Fold] = {}

    def walk(term: Fold) -> Fold:
        if id(term) not in done:
            operands = tuple(walk(edge) for edge in term.operands)
            same = all(new is old for new, old in zip(operands, term.operands, strict=True))
            new = term if same else term.replace(operands=operands)
            done[id(term)] = _keys_range(new, axes) if new.twist is not None else new
        return done[id(term)]

    return walk(root)


def _keys_range(carrier: Fold, axes) -> Fold:
    """``carrier`` with each additive coordinate mask of its score read as a bound of the range its key axis walks:
    the mask's statements leave the lift, the value it adds to a key the row folds stays, and the bound joins the
    range the carrier already walks. A mask is read where its fill is at most :data:`_OUTSIDE` and its predicate
    bounds the key by the coordinates around the carrier and the values one walk holds; any other stays in the lift."""
    key = carrier.axis
    coords = frozenset(axis.name for axis in axes)
    # In the lift's own names: a param bound to an operand that does not ride the key axis is one value for the
    # whole walk, and a leaf's statements are read under the names the lift gives its results.
    still = {param for param, edge, _ in carrier.bindings if key not in edge.free_axes}
    leaves: list[Statement] = []
    for edge in {id(edge): edge for _, edge, _ in carrier.bindings}.values():
        if edge.axis is None and key not in edge.free_axes:
            named = {edge.exposes[component]: param for param, bound, component in carrier.bindings if bound is edge}
            leaves += [stmt.rewrite(lambda name, named=named: named.get(name, name)) for stmt in edge.applied.body]
    uniform = still | {name for axis in axes for name in axis.extent.expression.free_vars()}
    aliases = coordinate_aliases((*carrier.lift.body, *leaves), coords, uniform)
    stmts = select_masks(carrier.lift.body, aliases, uniform, coords)
    if stmts is None:
        return carrier
    literals = {stmt.name: stmt.value for stmt in stmts if isinstance(stmt, Let) and isinstance(stmt.value, Literal)}
    starts: list[Expression] = []
    ends: list[Expression] = []
    kept: list[Statement] = list(stmts)
    for mask in [stmt for stmt in stmts if isinstance(stmt, Select)]:
        inside, outside = mask.branches
        fill = literals.get(outside.value)
        readers = [stmt for stmt in kept if stmt is not mask and mask.name in stmt.deps()]
        bound = _key_bound(inside.select, key)
        if fill is None or fill.value > _OUTSIDE or bound is None or len(readers) != 1 or not isinstance(readers[0], Assign):
            continue
        (reader,) = readers
        terms = reader.value.chain("add")
        if Var(mask.name) not in terms:
            continue
        # Inside the range the mask adds its other branch: a zero adds nothing, any other value stays a term.
        added = literals.get(inside.value)
        zero = added is not None and added.value == 0
        rest = [term for term in terms if term != Var(mask.name)] + ([] if zero else [Var(inside.value)])
        total = rest[0]
        for term in rest[1:]:
            total = apply("add", total, term)
        kept = [reader.replace(value=total) if stmt is reader else stmt for stmt in kept if stmt is not mask]
        (starts if bound[0] else ends).append(bound[1])
    if not starts and not ends:
        return carrier
    lift = carrier.lift
    body = Body(tuple(kept)).backward_cone(tuple(lift.results)).members
    extent = next(axis for axis in axes if axis.name == key).extent_expression()
    start, end = carrier.range if carrier.range is not None else (Literal(0, "int"), extent)
    for bound in starts:
        start = bound if start == Literal(0, "int") else apply("maximum", start, bound)
    for bound in ends:
        end = apply("minimum", end, bound)
    closed = Lambda.closing(lift.params[: 1 + len(carrier.bindings)], Body(tuple(body)), lift.results)
    return carrier.replace(lift=closed, range=(start, end))


def _key_bound(inside: Expression, key: str) -> tuple[bool, Expression] | None:
    """The bound the comparison ``inside`` states of ``key``: ``(True, start)`` for the least value the key takes,
    ``(False, end)`` for one past the most. ``None`` when it does not compare the key itself, one for one, with what
    the walk holds still: a key it scales has no bound an index expression spells without knowing a sign."""
    op = inside.operation
    if op is None or op.name not in _NEGATED:
        return None
    left, right = inside.operands
    form = affine_form(BinaryExpression("-", left, right), frozenset({key}))
    if form is None or abs(form[1].get(key, 0)) != 1:
        return None
    rest, scale = form
    # ``key + rest`` or ``rest - key``, compared with zero: the key against ``-rest`` or ``rest``.
    against = affine_sum(*affine_terms(BinaryExpression("-", Literal(0, "int"), rest) if scale[key] > 0 else rest))
    below = (op.name in ("less", "less_equal")) == (scale[key] > 0)
    strict = op.name in ("less", "greater")
    if below:  # the key is below ``against``, or at most it
        return False, against if strict else BinaryExpression("+", against, Literal(1, "int"))
    return True, BinaryExpression("+", against, Literal(1, "int")) if strict else against
