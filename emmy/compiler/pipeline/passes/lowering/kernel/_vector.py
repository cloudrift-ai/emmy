"""The contiguous, aligned flat-address proof shared by vector loads and stores."""

from collections.abc import Callable, Iterable, Mapping
from math import gcd

from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.stmt import Statement
from emmy.compiler.pipeline.passes.lowering.kernel._siblings import Siblings


def strided_alignment(start: Expression, step: Expression, aligned: Mapping[str, int]) -> int:
    """The largest power of two every value of a strided loop variable is a multiple of: the
    common divisor of its start's terms and its step. A non-affine or symbolic part gives 1."""
    parts = [step]
    form = affine_form(start, start.free_vars())
    if form is None:
        return 1
    anchor, coefficients = form
    parts.append(anchor.simplify(SimplifyCtx.empty()))
    if not all(isinstance(part, Literal) and isinstance(part.value, int) for part in parts):
        return 1
    divisor = gcd(*(part.value for part in parts), *(c * aligned.get(v, 1) for v, c in coefficients.items()))
    return divisor & -divisor if divisor else 0


#: A flat address as its symbolic part (each variable's coefficient) and its constant offset.
Address = tuple[frozenset[tuple[str, int]], int]


def flat_address(index, tensor, atoms: dict[Expression, Var]) -> Address | None:
    """The row-major flat address of ``index`` in ``tensor``, or ``None`` when it is not affine. ``atoms`` is shared
    by every address compared with this one (:func:`_atomized`)."""
    rank = len(index)
    if not rank or (rank > 1 and (tensor is None or len(tensor.shape) != rank)):
        return None
    flat = index[0]
    for coord, size in zip(index[1:], tensor.shape[1:] if rank > 1 else (), strict=True):
        flat = BinaryExpression("+", BinaryExpression("*", flat, size.expression), coord)
    ctx = SimplifyCtx.empty()
    flat = _atomized(flat.simplify(ctx), atoms)
    form = affine_form(flat, flat.free_vars())
    if form is None:
        return None
    anchor = form[0].simplify(ctx)
    if not isinstance(anchor, Literal) or not isinstance(anchor.value, int):
        return None
    return frozenset((v, c) for v, c in form[1].items() if c), anchor.value


def fuse_runs(
    scope: Siblings,
    addresses: Mapping[int, Address],
    widths: Iterable[int],
    aligned: Mapping[str, int],
    fused: Callable[[list[int]], Statement],
) -> bool:
    """Fuse every naturally aligned vector among the scope's statements at ``addresses`` (position → flat address),
    widest first: ``width`` statements at ``anchor, anchor + 1, …`` with ``anchor`` a multiple of ``width`` for every
    value of its variables, which the scope's ordering constraints let stand together. Where the statements stand
    says nothing (an unrolled pointwise body interleaves each element's load with its arithmetic, and the normal form
    orders siblings its own way). ``aligned`` names index variables known to be multiples of a power of two (an
    enclosing strided loop's); ``fused`` builds the vector statement from its members in address order."""
    at: dict[frozenset[tuple[str, int]], dict[int, list[int]]] = {}
    for position, (coefficients, offset) in sorted(addresses.items()):
        at.setdefault(coefficients, {}).setdefault(offset, []).append(position)
    changed = False
    for width in widths:
        for coefficients, offsets in at.items():
            if any(c * aligned.get(v, 1) % width for v, c in coefficients):
                continue
            for anchor in sorted(offset for offset in offsets if not offset % width):
                members = [offsets[anchor + k][0] for k in range(width) if offsets.get(anchor + k)]
                if len(members) == width and scope.fusable(members):
                    scope.fuse(members, fused(members))
                    for k in range(width):
                        del offsets[anchor + k][0]
                    changed = True
    return changed


def _atomized(expression: Expression, atoms: dict[Expression, Var]) -> Expression:
    """``expression`` with each subterm that is not a sum, difference or product (a gathered index's sign wrap, a cast, a
    quotient) replaced by one variable per distinct subterm. Every such subterm is an integer, the same wherever the
    same subterm appears, which is all the run proof asks of a variable."""
    if isinstance(expression, (Var, Literal)):
        return expression
    if isinstance(expression, BinaryExpression) and expression.op in ("+", "-", "*"):
        return BinaryExpression(expression.op, _atomized(expression.left, atoms), _atomized(expression.right, atoms))
    return atoms.setdefault(expression, Var(f"_run_atom{len(atoms)}"))
