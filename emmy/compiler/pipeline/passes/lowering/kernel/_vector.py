"""The contiguous, aligned flat-address proof shared by vector loads and stores."""

from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form


def vector_run(indices, tensor, width: int) -> bool:
    """Whether scalar indices form one naturally aligned vector in row-major storage."""
    rank = len(indices[0])
    if not rank or any(len(index) != rank for index in indices):
        return False
    if rank > 1 and (tensor is None or len(tensor.shape) != rank):
        return False
    ctx = SimplifyCtx.empty()
    # A common row offset cancels from address differences. When its stride is
    # a multiple of the vector width, it also cannot change base alignment,
    # even when the selected row is not affine in the kernel's coordinates.
    ignored = set()
    stride = 1
    for dim in reversed(range(rank)):
        if stride is not None and stride % width == 0 and all(index[dim] == indices[0][dim] for index in indices):
            ignored.add(dim)
        size = tensor.shape[dim] if tensor is not None else None
        stride = stride * size.as_static() if stride is not None and size is not None and size.is_static else None
    addresses = []
    for index in indices:
        index = tuple(Literal(0, "int") if dim in ignored else coord for dim, coord in enumerate(index))
        flat = index[0]
        for coord, size in zip(index[1:], tensor.shape[1:] if rank > 1 else (), strict=True):
            flat = BinaryExpr("+", BinaryExpr("*", flat, size.expr), coord)
        addresses.append(flat.simplify(ctx))
    free = frozenset(name for address in addresses for name in address.free_vars())
    first = affine_form(addresses[0], free)
    if first is None:
        return False
    anchor, coefficients = first
    anchor = anchor.simplify(ctx)
    if not isinstance(anchor, Literal) or anchor.value % width or any(c % width for c in coefficients.values()):
        return False
    for offset, address in enumerate(addresses[1:], 1):
        form = affine_form(address, free)
        if form is None or form[1] != coefficients:
            return False
        difference = BinaryExpr("-", form[0], anchor).simplify(ctx)
        if not isinstance(difference, Literal) or difference.value != offset:
            return False
    return True
