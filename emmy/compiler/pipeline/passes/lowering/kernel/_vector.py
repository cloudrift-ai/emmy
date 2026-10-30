"""The contiguous, aligned flat-address proof shared by vector loads and stores."""

from collections.abc import Mapping
from math import gcd

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var, affine_form


def strided_alignment(start: Expr, step: Expr, aligned: Mapping[str, int]) -> int:
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


def vector_run(indices, tensor, width: int, aligned: Mapping[str, int] | None = None) -> bool:
    """Whether scalar indices form one naturally aligned vector in row-major storage. ``aligned``
    names index variables known to be multiples of a power of two (an enclosing strided loop's)."""
    rank = len(indices[0])
    if not rank or any(len(index) != rank for index in indices):
        return False
    if rank > 1 and (tensor is None or len(tensor.shape) != rank):
        return False
    ctx = SimplifyCtx.empty()
    addresses = []
    for index in indices:
        flat = index[0]
        for coord, size in zip(index[1:], tensor.shape[1:] if rank > 1 else (), strict=True):
            flat = BinaryExpr("+", BinaryExpr("*", flat, size.expr), coord)
        addresses.append(flat.simplify(ctx))
    atoms: dict[Expr, Var] = {}
    addresses = [_atomized(address, atoms) for address in addresses]
    free = frozenset(name for address in addresses for name in address.free_vars())
    first = affine_form(addresses[0], free)
    if first is None:
        return False
    anchor, coefficients = first
    anchor = anchor.simplify(ctx)
    aligned = aligned or {}
    if not isinstance(anchor, Literal) or anchor.value % width or any(c * aligned.get(v, 1) % width for v, c in coefficients.items()):
        return False
    for offset, address in enumerate(addresses[1:], 1):
        form = affine_form(address, free)
        if form is None or form[1] != coefficients:
            return False
        difference = BinaryExpr("-", form[0], anchor).simplify(ctx)
        if not isinstance(difference, Literal) or difference.value != offset:
            return False
    return True


def _atomized(expr: Expr, atoms: dict[Expr, Var]) -> Expr:
    """``expr`` with each subterm that is not a sum, difference or product (a gathered index's sign wrap, a cast, a
    quotient) replaced by one variable per distinct subterm. Every such subterm is an integer, the same wherever the
    same subterm appears, which is all the run proof asks of a variable."""
    if isinstance(expr, (Var, Literal)):
        return expr
    if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*"):
        return BinaryExpr(expr.op, _atomized(expr.left, atoms), _atomized(expr.right, atoms))
    return atoms.setdefault(expr, Var(f"_run_atom{len(atoms)}"))
