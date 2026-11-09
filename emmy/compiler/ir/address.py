"""Flat-address `Expression` builders — fold-aware `sum` / `product` over int / `Expression` terms, plus
`split_pair` / `split_addressable` — the re-fused
split-store spelling of an axis across two buffer dims, and whether per-atom-origin addressing is
exact for one — and `gmem_axis_step` / `gmem_row_stride` — how a gmem `Load`'s flat address moves
along one axis, the row step (`ldm`) a fragment loader reads off the index and buffer shape and the
contiguity its columns need, plus `BYTE_SLAB_PAD` — the smem row pad a cp.async-staged byte slab
carries. This is generic IR addressing algebra with no schedule, search, or pass-layer dependency.
"""

from __future__ import annotations

import math

from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, SimplifyCtx, TernaryExpression, Var, affine_form
from emmy.compiler.ir.stmt import Load

# Row pad (in ELEMENTS = bytes) for a cp.async-staged 1-byte (fp8) operand slab. The cooperative
# byte-gather drain (no ldmatrix below sm_100a — the fragment loads are per-lane byte gathers)
# reads the slab at the fragment lane map's row strides, and a dense power-of-two byte row lands
# every 4-row group in the same bank quartet (4-way conflicts measured by the lane→bank oracle on
# both the k16 convert drain and the k32 repack drain, either B orientation). 16 extra bytes per
# row breaks the stride while keeping every 16 B cp.async chunk aligned (the byte-staging
# legality requires the data cols to be 16-divisible, so ``cols + 16`` stays 16 B-periodic).
# Derived, not tuned — the bank-quartet aliasing argument below. cp.async only:
# a TMA box deposit is dense (its byte slab stays unpadded and eats the measured conflicts).
# Shared by the kernel staging pass and the tile-layer byte-staging legality check, so it lives
# here rather than in `kernel/_stage` — a tile pass may not import the kernel pass layer.
BYTE_SLAB_PAD = 16


def add(*terms) -> Expression:
    """Sum int / Expression terms into one Expression (dropping literal zeros)."""
    out = None
    for t in terms:
        e = Literal(t, "int") if isinstance(t, int) else t
        if isinstance(e, Literal) and e.value == 0:
            continue
        out = e if out is None else BinaryExpression("+", out, e)
    return out if out is not None else Literal(0, "int")


def mul(a, b: int) -> Expression:
    """``a · b`` as an Expression, folding the ``b in {0, 1}`` degenerate cases."""
    return (
        add() if b == 0 else (a if b == 1 else BinaryExpression("*", a if not isinstance(a, int) else Literal(a, "int"), Literal(b, "int")))
    )


def clamp_to(position: Expression, extent: Expression) -> Expression:
    """``position`` clamped onto a stream's last position — ``position < extent ? position : extent − 1`` — the
    index an overhanging read takes through a stream or an output sweep: the read stays in bounds with a duplicate,
    and the fold masks what it folds there or the guard discards the store. :func:`clamped` reads it back."""
    return TernaryExpression(BinaryExpression("<", position, extent), position, BinaryExpression("-", extent, Literal(1, "int")))


def clamped(expression: Expression) -> tuple[Expression, Expression] | None:
    """``(position, extent)`` when ``expression`` is :func:`clamp_to` of them, else ``None``."""
    if not isinstance(expression, TernaryExpression) or not isinstance(expression.cond, BinaryExpression) or expression.cond.op != "<":
        return None
    position, extent = expression.cond.left, expression.cond.right
    if expression.if_true == position and expression.if_false == BinaryExpression("-", extent, Literal(1, "int")):
        return position, extent
    return None


def unclamped(load: Load) -> Load:
    """``load`` reading every clamped position as the position itself — the address a loader strides, bounded
    by :func:`clamp_bound` instead."""
    return Load(load.names, load.input, tuple(c[0] if (c := clamped(e)) is not None else e for e in load.index))


def clamp_bound(load: Load, axis_name: str) -> Expression | None:
    """How far along ``axis_name`` ``load`` reads in bounds, from the axis's own origin: the clamp's extent less the
    position at ``axis_name = 0``. ``None`` when no component of the index clamps a position over ``axis_name``."""
    for e in load.index:
        c = clamped(e)
        if c is not None and axis_name in c[0].free_vars():
            return BinaryExpression("-", c[1], c[0].substitute({axis_name: Literal(0, "int")}))
    return None


def gmem_axis_step(load: Load, axis_name: str, inputs) -> tuple[int, int] | None:
    """How ``load``'s FLAT gmem address moves along ``axis_name`` — ``(delta, run)``: the address
    changes by ``delta`` elements per unit step of the axis, and that holds over any ``run``-long
    ``run``-aligned window of the axis (``run == 0`` ⇒ the whole extent). The ONE address-algebra
    read the operand loaders and their schedule gates share; ``None`` when the walk cannot say.

    The flat address is ``Σ_j index[j] · stride_j`` over the buffer's declared strides, so the
    walk is per component (:func:`_axis_step`) and the deltas add. Three index forms answer:

    - the canonical layout, where the axis rides ONE component affinely — ``head_dim`` for
      ``(batch…, row, dd)``, ``H·D`` for an un-transposed ``(B, S, H, D)`` trace, where assuming
      the trailing extent read the WRONG rows (the gemma layer-0 NaN);
    - a BLOCKED read, where an index map splits the axis into a ``/`` block coordinate and a ``%``
      offset — the sdpa→o_proj chain's ``(head, seq, dd)`` gather. The block index does not move
      with the offset, so the offset's delta is exact within one block: that is what ``run``
      carries;
    - a DELINEARIZED read, where a reshape re-strides the rows and the components re-read one flat
      coordinate (:func:`_delinearized`), tried when the per-component walk gives up.

    ``None`` for an unknown buffer, an absent axis, a symbolic trailing extent, or any index the
    two readings do not cover — the schedule gate's decline signal."""
    t = inputs.get(load.input) if inputs else None
    if t is None or len(load.index) != len(tuple(t.shape)):
        return None
    dims = tuple(t.shape)
    delta, run, seen, gave_up = 0, 0, False, False
    for pos, e in enumerate(load.index):
        if axis_name not in e.free_vars():
            continue
        seen = True
        step = _axis_step(e, axis_name)
        if step is None:
            gave_up = True
            break
        run = _run_gcd(run, step[1])
        if not step[0]:
            continue  # the component holds still on the axis — its stride never enters the sum
        stride = 1
        for d in dims[pos + 1 :]:
            if not d.is_static:
                return None
            stride *= d.as_static()
        delta += step[0] * stride
    if not seen:
        return None
    if not gave_up:
        return delta, run
    flat = _delinearized(load.index, dims)
    return None if flat is None else _axis_step(flat, axis_name)


def gmem_row_stride(load: Load, axis_name: str, inputs) -> int | None:
    """The flattened gmem stride between successive ``axis_name`` rows of ``load``'s buffer — the
    fragment loaders' row step (``ldm``). The UNBOUNDED reading of :func:`gmem_axis_step`: a row
    step that only holds inside a block is no ``ldm``, because the loader applies it across the
    whole tile. ``None`` when underivable — the schedule gate's decline signal."""
    step = gmem_axis_step(load, axis_name, inputs)
    return step[0] if step is not None and step[1] == 0 and step[0] > 0 else None


def copy_width(slab_cols: int, elem_bytes: int) -> int:
    """Elements per vector copy (``cp.async`` or the blocking vector load) of a slab row ``slab_cols`` elements
    wide: the widest run whose byte size is a legal copy width (16, 8 or 4) and that divides the row, so a copy
    never straddles one. A byte slab's rows are 16-divisible by its staging rules, so it copies 16 elements. The
    staging lowering copies at this width and the stage resolvers prove :func:`copies_aligned` at it."""
    for nbytes in (16, 8, 4):
        v = nbytes // elem_bytes
        if v >= 1 and slab_cols % v == 0:
            return v
    return 1  # elem_bytes > 16 — never (1/2/4-byte element dtypes only)


def copies_aligned(load: Load, inner: str, chunk: int, inputs, *, per: int = 1) -> bool:
    """Whether every ``chunk``-element copy ``load`` makes along ``inner`` starts on a ``chunk``-element boundary of
    its buffer — the source alignment a vector copy of that width needs. A copy starts where ``inner`` is a multiple
    of ``chunk · per`` (``per`` is how many ``inner`` positions one stored element holds: two for a packed byte), so
    the flat address at ``inner = chunk · per · j`` must be a provable multiple of ``chunk`` for every ``j``. The
    proof reads the whole address, simplified: a slice's offset, a row stride off the grid, and ``inner`` split across
    dims at a pitch off it (a slice reshaped flat) each leave a term it cannot prove, while a dense reshape's quotient
    and remainder recombine. A symbolic dim is an unknown factor of the strides before it, and the buffer's own base
    is aligned, as an allocation is. ``False`` for an unknown buffer."""
    t = inputs.get(load.input) if inputs else None
    if t is None or len(load.index) != len(tuple(t.shape)):
        return False
    start = {inner: BinaryExpression("*", Literal(chunk * per, "int"), Var("__copy"))}
    address, stride = Literal(0, "int"), Literal(1, "int")
    for pos in reversed(range(len(load.index))):
        address = BinaryExpression("+", address, BinaryExpression("*", load.index[pos].substitute(start), stride))
        dim = t.shape[pos]
        stride = BinaryExpression("*", stride, Literal(dim.as_static(), "int") if dim.is_static else Var(f"__dim{pos}"))
    return _divisible(address.simplify(SimplifyCtx.empty()), chunk)


def reads_declared_rows(load: Load, axis_name: str) -> bool:
    """Whether the operand's row step IS the buffer's DECLARED trailing extent — the reading a
    loader falls back on when it is handed no stride (``ir._resolve_ldm``), and the only one that
    survives a SYMBOLIC extent, which no integer derivation can name. True exactly when the axis
    rides the penultimate index component alone at coefficient 1: then the flat row step is the
    product of the extents after it, which IS the declared trailing extent.

    The escape hatch for :func:`gmem_row_stride` declining on a symbolic dim — a masked / dynamic-M
    matmul reads canonical rows and always did. Not a substitute for the derivation: it says the
    declaration happens to be right here, nothing about any other index."""
    if len(load.index) < 2:
        return False
    form = affine_form(load.index[-2], {axis_name})
    if form is None or form[1].get(axis_name, 0) != 1:
        return False
    return not any(axis_name in e.free_vars() for i, e in enumerate(load.index) if i != len(load.index) - 2)


def _run_gcd(a: int, b: int) -> int:
    """Combine two run lengths — ``0`` (unbounded) is the identity, else the gcd: a ``g``-aligned
    window of ``g = gcd(a, b)`` sits inside both a ``a``-aligned ``a``-window and a ``b``-aligned
    ``b``-window, so the joint delta holds there."""
    return b if a == 0 else a if b == 0 else math.gcd(a, b)


def _int_lit(e: Expression) -> int | None:
    """``e``'s value when it is an integer :class:`Literal`, else ``None``."""
    return e.value if isinstance(e, Literal) and isinstance(e.value, int) and not isinstance(e.value, bool) else None


def _divisible(expression: Expression, n: int) -> bool:
    """Whether ``expression`` is PROVABLY a multiple of ``n``: every additive term is; a product's literal factor
    ``c`` leaves ``n / gcd(c, n)`` to the other factor, and any other product needs one factor that is; a quotient
    ``a / c`` is when ``a`` is a multiple of ``c · n``, a remainder ``a % c`` when both ``a`` and ``c`` are. Each rule
    holds for any integer operands under C's truncating division. Conservative: an opaque var answers no."""
    if n == 1:
        return True
    if (v := _int_lit(expression)) is not None:
        return v % n == 0
    if not isinstance(expression, BinaryExpression):
        return False
    if expression.op in ("+", "-"):
        return _divisible(expression.left, n) and _divisible(expression.right, n)
    if expression.op == "*":
        for side, other in ((expression.left, expression.right), (expression.right, expression.left)):
            if (c := _int_lit(other)) is not None:
                return _divisible(side, n // math.gcd(c, n))
        return _divisible(expression.left, n) or _divisible(expression.right, n)
    if (c := _int_lit(expression.right)) is not None and c > 0:
        if expression.op == "/":
            return _divisible(expression.left, c * n)
        if expression.op == "%":
            return c % n == 0 and _divisible(expression.left, n)
    return False


def _axis_step(expression: Expression, axis: str) -> tuple[int, int] | None:
    """``(delta, run)`` for ONE index component — see :func:`gmem_axis_step`. ``None`` when the
    component's dependence on ``axis`` is not a step at all (a clamp, a gather, ``axis·axis``).

    ``/`` and ``%`` by a positive literal ``n`` are where a layout op shows up, and they resolve
    two ways. When ``n`` divides the axis's coefficient the operand's axis-motion is a whole number
    of blocks, so ``E / n`` steps by ``coef / n`` and ``E % n`` does not move at all — the BLOCK
    coordinate of a blocked read. When the coefficient is 1 and the rest of the operand is provably
    ``n``-divisible, the axis IS the within-block offset: ``E / n`` holds still and ``E % n`` steps
    by one, each over an ``n``-aligned ``n``-window.

    Both readings take the operand NON-NEGATIVE, which a buffer coordinate is (the same assumption
    ``expression._div_mod_decompose`` states as a range check); C truncation toward zero would break the
    window argument below zero."""
    if axis not in expression.free_vars():
        return 0, 0
    if isinstance(expression, Var):
        return 1, 0
    if not isinstance(expression, BinaryExpression):
        return None
    if expression.op in ("+", "-"):
        left, right = _axis_step(expression.left, axis), _axis_step(expression.right, axis)
        if left is None or right is None:
            return None
        return left[0] + (right[0] if expression.op == "+" else -right[0]), _run_gcd(left[1], right[1])
    if expression.op == "*":
        for side, other in ((expression.left, expression.right), (expression.right, expression.left)):
            if axis in other.free_vars():
                continue
            mult = _int_lit(other)
            step = _axis_step(side, axis) if mult is not None else None
            return None if step is None else (step[0] * mult, step[1])
        return None  # axis · axis — not a step
    if expression.op in ("/", "%") and (n := _int_lit(expression.right)) is not None and n > 0:
        step = _axis_step(expression.left, axis)
        if step is None:
            return None
        delta, run = step
        if delta % n == 0:  # whole blocks: the block index steps, the offset holds still
            return (delta // n, run) if expression.op == "/" else (0, run)
        if delta == 1 and run == 0 and _divisible(expression.left.substitute({axis: Literal(0, "int")}), n):
            return (0, _run_gcd(run, n)) if expression.op == "/" else (1, _run_gcd(run, n))
    return None


def _delinearized(index: tuple, shape: tuple) -> Expression | None:
    """The ONE flat source coordinate a DELINEARIZING index map re-reads, or ``None`` when
    ``index`` is not such a map.

    ``130_reshape`` spells a reshape as ``c_j = (flat / stride_j) % dim_j`` over the SOURCE
    buffer's strides, with no ``% dim_0`` on the leading coordinate. Flattening that back —
    ``Σ_j c_j · stride_j`` — collapses to ``flat`` exactly, for every value, by the mixed-radix
    identity (C division guarantees ``(a/b)·b + a%b == a``). Recovering ``flat`` is what lets a
    caller read the operand's REAL row stride off a reshaped view instead of the buffer's declared
    one — the declared reading walks the wrong rows.

    Size-1 source dims are ``Literal(0)`` (``030_lower_indexmap`` forces them) and contribute
    nothing, so they are skipped. Anything else — a transpose, a cat's select branches, a gather —
    fails the match and declines."""
    if len(index) != len(shape) or len(index) < 2:
        return None
    dims, strides, s = list(shape), [1] * len(shape), 1
    for j in range(len(dims) - 1, 0, -1):
        strides[j - 1] = s = s * (dims[j].as_static() if dims[j].is_static else 0)
        if not s:
            return None
    flat = None
    for j, e in enumerate(index):
        if not dims[j].is_static:
            return None
        if dims[j].as_static() == 1:
            continue  # a size-1 source dim reads Literal(0) and strides nothing
        if j > 0:  # every non-leading coordinate carries its own ``% dim_j``
            if not (isinstance(e, BinaryExpression) and e.op == "%" and _int_lit(e.right) == dims[j].as_static()):
                return None
            e = e.left
        if strides[j] != 1:
            if not (isinstance(e, BinaryExpression) and e.op == "/" and _int_lit(e.right) == strides[j]):
                return None
            e = e.left
        if flat is None:
            flat = e
        elif e != flat:
            return None  # the coordinates delinearize DIFFERENT expressions — not one reshape
    return flat


def split_pair(index: tuple, name: str) -> int | None:
    """``Q`` when the expressions of ``index`` mentioning ``name`` are exactly one integer quotient and one
    ``name % Q`` (the split-store spelling a re-fused axis reaches a buffer with, in either dim
    order), else ``None``."""
    carrying = [e for e in index if name in e.free_vars()]
    if len(carrying) != 2:
        return None
    parts: dict[str, int] = {}
    for e in carrying:
        if not (isinstance(e, BinaryExpression) and e.op in ("/", "//", "%") and e.left == Var(name) and isinstance(e.right, Literal)):
            return None
        parts["/" if e.op == "//" else e.op] = int(e.right.value)
    return parts["/"] if set(parts) == {"/", "%"} and parts["/"] == parts["%"] else None


def split_addressable(index: tuple, shape, name: str, atom_ext: int, trailing: bool) -> bool:
    """Whether per-atom-origin addressing — the cell base evaluated once at the atom origin, then
    ``+ col`` / ``+ row · ldm`` across an ``atom_ext``-wide cell — is exact for an ``index`` whose
    ``name`` axis reaches the buffer as a split dim pair. It is when the row-major flatten
    recomposes the pair to an affine address (the strides match the split), or when the ``%`` dim
    is the innermost carrier (the trailing dim, when ``trailing``) and its modulus is a whole
    number of atom cells, so an aligned atom never straddles a ``Q`` boundary. An axis carried by
    fewer than two dims is trivially addressable."""
    dims = [d for d, e in enumerate(index) if name in e.free_vars()]
    if len(dims) < 2:
        return True
    q = split_pair(index, name)
    inner = index[dims[1]]
    if q is None or not (isinstance(inner, BinaryExpression) and inner.op == "%"):
        return False  # not a bare f/Q, f%Q pair, or the quotient dim is the inner one
    if trailing and dims[1] != len(index) - 1:
        return False  # the remainder dim is not the contiguous one
    if shape is not None and len(shape) == len(index) and all(getattr(d, "is_static", False) for d in shape):
        stride, strides = 1, [1] * len(index)
        for d in reversed(range(len(index))):
            strides[d] = stride
            stride *= shape[d].as_static()
        if strides[dims[0]] == q * strides[dims[1]]:
            return True  # row-major recomposition — the address is affine in the axis
    return q % atom_ext == 0
