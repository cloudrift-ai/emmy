"""The contiguous, aligned flat-address proof shared by vector loads and stores, and the two widenings that use it
(``050_vectorize_loads``, ``080_vectorize_stores``, run again by ``119_vectorize_copies``)."""

from collections.abc import Callable, Iterable, Mapping
from math import gcd

from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Load, Statement, StridedLoop, Write
from emmy.compiler.pipeline.passes.lowering.kernel._siblings import Siblings
from emmy.compiler.target import vector_widths


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
    says nothing (an unrolled pointwise body interleaves each element's load with its arithmetic). ``aligned`` names
    index variables known to be multiples of a power of two (an enclosing strided loop's); ``fused`` builds the vector
    statement from its members in address order."""
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


def vectorize_loads(top: KernelOp, body: Body, aligned: dict[str, int] | None = None) -> Body:
    """``body`` with each run of scalar loads widened into one vector ``Load`` (``050_vectorize_loads``), nested
    bodies first. ``top`` resolves the inputs, ``aligned`` the alignment of each enclosing strided loop's variable."""
    aligned = aligned or {}
    descended: list[Statement] = []
    for s in body:
        nested = s.nested()
        if nested:
            inner = {**aligned, s.axis.name: strided_alignment(s.start, s.step, aligned)} if isinstance(s, StridedLoop) else aligned
            descended.append(s.with_bodies(tuple(vectorize_loads(top, b, inner) for b in nested)))
        else:
            descended.append(s)

    descended = [s.map_expressions(lambda e: _fold_run_offset(e, aligned)) if isinstance(s, Load) else s for s in descended]
    # No literal-constant loads (those render as embedded scalar floats), none without the stamped dtype that picks the
    # vector type (``030_stamp_types``), and no scalar-constant input: it is inlined at CUDA lowering, so the kernel
    # takes no such buffer to reinterpret.
    atoms: dict[Expression, Var] = {}
    runs: dict[tuple, dict[int, Address]] = {}
    for position, stmt in enumerate(descended):
        if not isinstance(stmt, Load) or not stmt.is_scalar or stmt.carried or stmt.input is None or stmt.dtype is None:
            continue
        tensor = top.inputs.get(stmt.input)
        if tensor is not None and tensor.constant and tensor.value is not None:
            continue
        if (address := flat_address(stmt.index, tensor, atoms)) is not None:
            runs.setdefault((stmt.input, stmt.dtype), {})[position] = address
    scope = Siblings(descended) if any(len(addresses) > 1 for addresses in runs.values()) else None
    changed = False
    for (name, dtype), addresses in runs.items():
        if len(addresses) > 1:
            widths = sorted(vector_widths(dtype.name), reverse=True)
            changed |= fuse_runs(
                scope,
                addresses,
                widths,
                aligned,
                lambda members, name=name, dtype=dtype: Load(
                    names=tuple(scope[m].name for m in members), input=name, index=scope[members[0]].index, dtype=dtype
                ),
            )
    vectorized = Body(tuple(scope.ordered() if changed else descended))
    return body if vectorized == body else vectorized


def _fold_run_offset(expression: Expression, aligned: dict[str, int]) -> Expression:
    """``(Y + r) / d`` as ``Y / d`` where ``Y`` is a non-negative multiple of ``g`` (its variables' coefficients
    times their known alignment), ``g`` divides ``d`` and ``0 < r < g``: the elements of a lane's run then read
    a block scale at one index, the run's own."""
    if not isinstance(expression, BinaryExpression):
        return expression
    left, right = _fold_run_offset(expression.left, aligned), _fold_run_offset(expression.right, aligned)
    if expression.op == "/" and isinstance(right, Literal) and isinstance(right.value, int) and right.value > 0:
        form = affine_form(left, left.free_vars())
        anchor = form[0].simplify(SimplifyCtx.empty()) if form is not None else None
        if isinstance(anchor, Literal) and isinstance(anchor.value, int) and anchor.value >= 0 and form[1]:
            terms = [c * aligned.get(v, 1) for v, c in form[1].items()]
            if all(t > 0 for t in terms):
                offset = anchor.value % gcd(right.value, *terms)
                if offset:
                    left = _affine(form[1], anchor.value - offset)
    return expression if (left, right) == (expression.left, expression.right) else BinaryExpression(expression.op, left, right)


def _affine(coefficients: dict[str, int], constant: int) -> Expression:
    """``Σ c·v + constant`` spelled the way the run's first element spells it."""
    terms = [Var(v) if c == 1 else BinaryExpression("*", Var(v), Literal(c, "int")) for v, c in coefficients.items()]
    expression = terms[0]
    for term in terms[1:]:
        expression = BinaryExpression("+", expression, term)
    return BinaryExpression("+", expression, Literal(constant, "int")) if constant else expression


def _buf_dtype(top: KernelOp, name: str) -> str:
    """Resolve a destination buffer's canonical dtype. Writes target either a
    kernel output or, in rare cases, a kernel input (when an optimization
    folds a copy). Falls back to f32 when not found."""
    t = top.outputs.get(name) or top.inputs.get(name)
    if t is not None:
        return t.dtype.name
    return "f32"


def vectorize_stores(top: KernelOp, body: Body) -> Body:
    """``body`` with each run of scalar writes widened into one vector ``Write`` (``080_vectorize_stores``), nested
    bodies first. ``top`` resolves the buffers' dtypes."""
    descended: list[Statement] = []
    for s in body:
        nested = s.nested()
        if nested:
            descended.append(s.with_bodies(tuple(vectorize_stores(top, b) for b in nested)))
        else:
            descended.append(s)

    # Already-widened Writes never re-merge, and atomic reduce-writes never vectorize (each contributing lane needs
    # its own ``atomicAdd``). A swizzled-slab run merges only under one mode (the alignment proof then keeps the
    # n-aligned run inside one 16-byte swizzle chunk, where the XOR passes bits 0..2 through).
    atoms: dict[Expression, Var] = {}
    runs: dict[tuple, dict[int, Address]] = {}
    for position, stmt in enumerate(descended):
        if isinstance(stmt, Write) and not stmt.is_vector and not stmt.atomic:
            tensor = top.outputs.get(stmt.output) or top.inputs.get(stmt.output)
            if (address := flat_address(stmt.index, tensor, atoms)) is not None:
                runs.setdefault((stmt.output, stmt.swizzle, stmt.value_dtype), {})[position] = address
    scope = Siblings(descended) if any(len(addresses) > 1 for addresses in runs.values()) else None
    changed = False
    for (output, swizzle, value_dtype), addresses in runs.items():
        if len(addresses) > 1:
            widths = [n for n in sorted(vector_widths(_buf_dtype(top, output)), reverse=True) if n <= 8]
            changed |= fuse_runs(
                scope,
                addresses,
                widths,
                {},
                lambda members, output=output, swizzle=swizzle, value_dtype=value_dtype: Write(
                    output=output,
                    index=scope[members[0]].index,
                    values=tuple(scope[m].value for m in members),
                    value_dtype=value_dtype,
                    swizzle=swizzle,
                ),
            )
    vectorized = Body(tuple(scope.ordered() if changed else descended))
    return body if vectorized == body else vectorized
