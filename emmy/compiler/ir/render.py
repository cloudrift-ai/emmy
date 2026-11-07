"""C-family source for statement bodies: the render context, index flattening, and one render function per kind.

``render_stmt`` dispatches on the statement kind. The kinds every IR shares register here; Kernel IR registers its
own beside their classes. ``render_body`` is the entry point the CUDA kernel renderer and the Loop IR runner call.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import singledispatch
from typing import TYPE_CHECKING, ClassVar, Protocol

from emmy import config
from emmy.compiler.dim import DYNAMIC_DIM_MAX, Dim
from emmy.compiler.dtype import F32, INTEGER_NAMES
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import reduce_spelling
from emmy.compiler.ir.expression import (
    BinaryExpression,
    Builtin,
    CastExpression,
    Expression,
    FlatIndex,
    FunctionCallExpression,
    Literal,
    SimplifyCtx,
    TernaryExpression,
    Var,
)
from emmy.compiler.ir.stmt import (
    Accum,
    Affine,
    Assign,
    Body,
    Carry,
    Cond,
    Init,
    Let,
    Load,
    Loop,
    Pre,
    Select,
    Statement,
    StridedLoop,
    Write,
    ZeroPrologue,
    dtype_promote,
)

if TYPE_CHECKING:
    from emmy.compiler.render_target import RenderTarget


def _default_render_target():
    """Lazy default: a :class:`CudaRenderTarget` instance. Used when
    ``RenderCtx`` is constructed without an explicit target — keeps the
    legacy "everything is CUDA" behavior for tests / golden output.

    Lazy to avoid importing the backend at IR-module load time."""
    from emmy.compiler.backend.cuda.render_target import CudaRenderTarget  # noqa: PLC0415

    return CudaRenderTarget()


class Memory(Protocol):
    """How a buffer's bytes are reached — the only thing a transport asks of one.

    ``base`` is the pointer to address from and ``flat`` the index to use with it, so a
    transport renders ``f"{base}[{flat}]"`` without knowing whether the buffer is one
    allocation or many. ``vectorizable`` says whether consecutive elements are guaranteed
    contiguous, which is what a vector access needs.
    """

    vectorizable: ClassVar[bool]

    def base(self, index: tuple, ctx: RenderCtx) -> str: ...

    def flat(self, index: tuple, ctx: RenderCtx) -> str: ...


@dataclass
class RenderCtx:
    """Per-render state. ``target`` is the :class:`RenderTarget` that
    owns every target-specific C spelling decision (type names,
    conversion intrinsics, per-dtype op spellings, native-op coverage,
    vector load shapes). Everything else here is generic walk state.

    ``intrinsics`` / ``builtins`` keep the legacy abstract→spelling
    indirection used by ``FunctionCallExpression.render`` / ``Builtin.render``
    for symbols that don't vary by dtype (``thread_idx.x`` →
    ``threadIdx.x``, etc.). Dtype-aware lookups go through
    ``ctx.target.intrinsic(...)`` instead.

    ``shapes`` maps every buffer to its declared shape so multi-dim
    ``Load`` / ``Write`` indices can be flattened row-major.
    """

    target: RenderTarget = field(default_factory=_default_render_target)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    indent: int = 1
    intrinsics: dict[str, str] = field(default_factory=dict)
    builtins: dict[str, str] = field(default_factory=dict)
    literal_constants: dict[str, float] = field(default_factory=dict)
    # SSA names whose defining Load came from a ``literal_constants``
    # input — populated by ``render_body`` after scanning the body, and
    # consumed by ``Var.render`` to inline the float at use sites instead
    # of emitting a named ``float in0 = 0.044f;`` decl.
    literal_ssa: dict[str, float] = field(default_factory=dict)
    # Single-use scalar SSA temps folded into their sole consumer's expression — readability only
    # (``EMMY_READABLE``), so ``m_i__t2 = expf(m_i__t1)`` with ``m_i__t1 = m_i - t0`` collapses to
    # ``m_i__t2 = expf(m_i - t0)`` (the scalar-tier analogue of the fragment chain fusion). Populated
    # by ``render_body`` (per flat body, in def order) and consumed by ``Var.render`` — exactly like
    # ``literal_ssa`` but the inlined value is a rendered sub-expression, not a float.
    inline_expressions: dict[str, str] = field(default_factory=dict)
    # Per-buffer byte offsets into a single ``extern __shared__`` pool
    # ``_smem_pool``. When non-empty, ``Smem``'s render emits a pointer
    # alias into the pool instead of a stand-alone ``__shared__`` array
    # — the only way to exceed the 48 KB static-smem cap.
    smem_dynamic_offsets: dict[str, int] = field(default_factory=dict)
    # How each buffer's bytes are reached, for the buffers that are not one allocation — see
    # :class:`Memory`. Absent names are :class:`Flat`, so an empty map renders every buffer
    # exactly as it always was.
    memory: dict[str, Memory] = field(default_factory=dict)
    # The input buffers that are the e2m1 pair table (``ir.base.is_f4_pair_table``): ``Load`` computes a lookup into
    # one from its index instead of reading it.
    pair_tables: frozenset[str] = frozenset()
    # Per-buffer canonical dtype tokens (``"f32"`` / ``"f16"``) for every
    # global-buffer name (kernel inputs + outputs). ``Load`` declares its
    # SSA-name local in the source buffer's C type so values flow at
    # buffer dtype end-to-end where possible; ``Write`` inserts the
    # target's conversion intrinsic only when the value's dtype disagrees
    # with the destination buffer's dtype. Missing entries default to
    # ``"f32"`` so legacy bodies render unchanged.
    buffer_dtypes: dict[str, str] = field(default_factory=dict)
    # Per-SSA-name canonical dtype tokens, populated as ``render_body``
    # walks. ``Load`` writes the source buffer's dtype; ``Assign`` writes
    # ``promote(args)``. Consumed by downstream ``Assign`` / ``Write`` to
    # decide native-vs-promote-fallback and to insert conversions.
    ssa_dtypes: dict[str, str] = field(default_factory=dict)
    # Render-time hint to ``Literal.render``: when set to a non-default
    # dtype, float literals render via ``target.literal(text, dtype)``
    # so they compose with non-default-dtype operands. Set transiently
    # by ``Assign``'s render around the native expression render.
    literal_default_dtype: str | None = None
    # C-scope-local declared locals (the mma lane vars ``_g`` / ``_t``, the
    # ``cp.async`` staging address ``_smem_addr``). A self-scoping stmt declares
    # these once per ``{ }`` scope and reuses / reassigns them afterward instead
    # of wrapping itself in a redundant block — so a loop body holding a single
    # such stmt renders flat (``for (…) { … }``, not ``for (…) { { … } }``).
    # UNLIKE the tables above this is **scope-local**: ``child()`` copies it (a
    # new C scope inherits the enclosing scope's decls but its own additions do
    # not leak back to the parent or across to siblings).
    scope_decls: set[str] = field(default_factory=set)
    # The loop counters in scope whose every value is a multiple of a known stride — a static
    # ``StridedLoop`` start and step — so an index built from one is provably aligned
    # (``kernel.ir._multiple_of``). Scope-local like ``scope_decls``.
    aligned: dict[str, int] = field(default_factory=dict)

    def child(self) -> RenderCtx:
        """Return a new ctx one indent level deeper. The mutable tables (``shapes``,
        ``ssa_dtypes``, …) stay shared by reference (SSA names are globally unique);
        ``scope_decls`` is COPIED — it tracks per-C-scope local declarations, which an
        inner scope inherits but must not leak back out of."""
        return replace(self, indent=self.indent + 1, scope_decls=set(self.scope_decls), aligned=dict(self.aligned))

    # ---- Convenience wrappers over ``self.target``. These exist so the
    # render methods read ``ctx.type_name(dt)`` instead of pulling the
    # target out by hand; they also default ``None`` dtype to F32 so the
    # call sites don't repeat that boilerplate.

    def memory_for(self, buf: str) -> Memory:
        """How ``buf`` is reached. Every buffer is :class:`Flat` unless declared otherwise."""
        return self.memory.get(buf) or Flat(buf)

    def type_name(self, dtype) -> str:
        """C type spelling for a local declaration. Accepts a
        :class:`DataType`, a canonical-name string, or ``None`` (treated
        as F32)."""
        return self.target.type_name(_canonical_dtype_name(dtype))

    def identity_literal(self, identity: float, dtype) -> str:
        """Render an accumulator's identity (0, 1, -inf, ...) as a C
        literal in ``dtype``, wrapping with the target's dtype cast if
        needed (e.g. ``__float2half(0.0f)`` for fp16). Accepts a
        :class:`DataType`, a canonical-name string, or ``None`` (F32)."""
        return self.target.literal(_float_lit(float(identity)), _canonical_dtype_name(dtype))


def _canonical_dtype_name(dtype) -> str:
    """Normalize ``DataType | str | None`` to the canonical dtype token."""
    if dtype is None:
        return "f32"
    if isinstance(dtype, str):
        return dtype
    return dtype.name


def _pad(n: int) -> str:
    return "    " * n


_BINARY_OP: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "mod": "%",
    # Comparisons + legacy bool combines — the whole-model explicit-mask subgraph.
    # Non-integer operands reach ``op_to_expression`` promoted to f32; C's implicit
    # nonzero→bool conversion makes these logical spellings valid on them. Typed
    # integer operands use ``_INTEGER_BINARY_OP`` instead, including packed
    # bitwise decode operations.
    "equal": "==",
    "not_equal": "!=",
    "greater": ">",
    "less": "<",
    "greater_equal": ">=",
    "less_equal": "<=",
    "bitwise_or": "||",
    "bitwise_and": "&&",
}


_INTEGER_BINARY_OP: dict[str, str] = {
    "floor_divide": "//",
    "remainder": "%",
    "mod": "%",
    "left_shift": "<<",
    "right_shift": ">>",
    "bitwise_or": "|",
    "bitwise_and": "&",
    "bitwise_xor": "^",
}


def op_to_expression(fn: str, inputs: list[Expression], *, dtype: str | None = None) -> Expression:
    """Translate an elementwise op name to an ``Expression`` tree.

    Emits abstract intrinsic names (``"exp"``, ``"fmax"``, ``"fabs"``, ...)
    that targets translate to libm / CUDA spellings via
    ``RenderCtx.intrinsics`` at ``FunctionCallExpression.render`` time. ``dtype``
    distinguishes integer bit manipulation from the traced mask convention:
    integer ``bitwise_and``/``bitwise_or`` render as ``&``/``|``, while the
    historical f32 mask path remains logical ``&&``/``||``.
    """
    if dtype in INTEGER_NAMES and fn in _INTEGER_BINARY_OP:
        return BinaryExpression(_INTEGER_BINARY_OP[fn], inputs[0], inputs[1])
    if dtype in INTEGER_NAMES and fn == "bitwise_count":
        return FunctionCallExpression("bitwise_count", tuple(inputs))
    if fn in _BINARY_OP:
        return BinaryExpression(_BINARY_OP[fn], inputs[0], inputs[1])
    if fn == "divide_rn_f32":
        return FunctionCallExpression(fn, tuple(inputs))
    if fn == "maximum":
        return FunctionCallExpression("fmax", tuple(inputs))
    if fn == "minimum":
        return FunctionCallExpression("fmin", tuple(inputs))
    if fn == "pow":
        return FunctionCallExpression("pow", tuple(inputs))
    if fn == "negative":
        return BinaryExpression("-", Literal(0.0, "float"), inputs[0])
    if fn in ("copy", "pad"):
        return inputs[0]
    if fn in ("from_f8e4m3", "from_f8e5m2"):
        # fp8 decode cast: the value semantics live in the dtype conversion the
        # target already applied to the argument (``Assign``'s render promotes the
        # f8-typed arg through ``target.convert``, which spells the decode), so
        # the op itself is an identity here — like ``copy``, but never a raw
        # byte move because the arg reaches this point converted.
        return inputs[0]
    if fn in ("to_f8e4m3", "to_f8e5m2"):
        # fp8 encode cast — the decode twin: ``Assign``'s render computes the arg at f32 and its
        # result-dtype demotion routes through ``target.convert``, which spells the <cuda_fp8.h>
        # constructor (round-to-nearest-even, saturate-to-finite). Identity here, same as above.
        return inputs[0]
    if fn == "to_f4e2m1":
        # e2m1 encode — NOT the fp8 encodes' identity. Those convert to an fp8 type whose
        # constructor rounds; the codes here land on an ordinary integer carrier, where the same
        # identity would truncate the value instead. The render emits the rounding as a helper
        # (``_F4_ENCODE_PRELUDE``).
        return FunctionCallExpression("emmy_to_f4e2m1", (inputs[0],))
    if fn == "square":
        return BinaryExpression("*", inputs[0], inputs[0])
    if fn == "reciprocal":
        return BinaryExpression("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        return FunctionCallExpression("fmax", (Literal(0.0, "float"), inputs[0]))
    if fn == "sigmoid":
        neg_x = BinaryExpression("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FunctionCallExpression("exp", (neg_x,))
        return BinaryExpression("/", Literal(1.0, "float"), BinaryExpression("+", Literal(1.0, "float"), exp_neg))
    if fn in ("exp", "exp_fast", "log", "rsqrt", "sin", "cos", "tanh", "sqrt", "erf", "floor"):
        return FunctionCallExpression(fn, tuple(inputs))
    if fn == "abs":
        return FunctionCallExpression("fabs", tuple(inputs))
    if fn == "where":
        # np.where(cond, a, b) — the mask-apply tail of the explicit-mask
        # subgraph. The f32-promoted cond is C-truthy (nonzero) in ``?:``.
        return TernaryExpression(inputs[0], inputs[1], inputs[2])
    if fn == "bitwise_not":
        # Torch spells ``~mask`` as bitwise_not even when ``mask`` is bool. Loop
        # IR historically promotes boolean mask SSA values to f32, so both the
        # implicit f32 convention and an explicitly bool-stamped value mean
        # logical-not. Never infer integer complement here: that requires a
        # separately typed and tested ``~`` path.
        if dtype not in (None, "f32", "bool"):
            raise NotImplementedError(f"render: bitwise_not supports boolean masks only, got dtype={dtype!r}")
        if len(inputs) != 1:
            raise ValueError(f"render: bitwise_not requires one input, got {len(inputs)}")
        zero = Literal(0.0, "float") if dtype in (None, "f32") else Literal(0, "int")
        return BinaryExpression("==", inputs[0], zero)
    raise NotImplementedError(f"render: elementwise fn={fn!r} not supported")


def render_merge_program(program, state_names, ctx: RenderCtx, pad: str | None = None, *, dtype=None) -> list[str]:
    """Render a monoid ``merge`` / ``combine_states`` program (a tuple of
    ``Assign``) at ``dtype`` (a :class:`DataType`; ``None`` → fp32). An ``Assign``
    whose target is in ``state_names`` is a reassignment of an already-declared
    carried value (``name = …;``); every other ``Assign`` declares a local temp
    (``<ty> t = …;``). Statement order is load-bearing — a state update must follow
    every read of that state's old value (the carrier builder guarantees it).

    The program is the PURE form a stored ``combine`` holds — ``Assign`` only, its
    final writes reassigning the carried state. Used by the kernel-IR cross-thread
    combine primitives ``WarpShuffle`` / ``TreeHalve``, which render it at the
    carrier's dtype (fp32 for monoids, the accumulator dtype for a degenerate scalar
    reduce) inside a loop they emit themselves."""
    if pad is None:
        pad = _pad(ctx.indent)
    dt = "f32" if dtype is None else dtype.name
    ty = ctx.type_name(dt)
    sset = set(state_names)
    out: list[str] = []
    for a in program:
        # The merge runs at ``dt`` — cast any arg with a different dtype (e.g. a raw
        # ``__half`` value loaded into the partial, as in fp16 flash's ``p · v``) so
        # the operator isn't ambiguous. The cast is a no-op for matching args.
        arg_expressions = [Var(x) if ctx.ssa_dtypes.get(x, dt) == dt else CastExpression(ty, Var(x)) for x in a.args]
        rhs = render_expression(op_to_expression(a.op.name, arg_expressions, dtype=dt), ctx)
        if a.name in sset:
            out.append(f"{pad}{a.name} = {rhs};")  # reassign carried state
        else:
            ctx.ssa_dtypes[a.name] = dt
            out.append(f"{pad}{ty} {a.name} = {rhs};")  # local temp
    return out


def select_to_ternary(s: Select, dtype: str = "float") -> Expression:
    """Cast branches to their common result type, avoiding ambiguous half/float C++ conditionals."""
    branches = list(s.branches)
    result: Expression = CastExpression(dtype, Var(branches[-1].value))
    for b in reversed(branches[:-1]):
        result = TernaryExpression(cond=b.select, if_true=CastExpression(dtype, Var(b.value)), if_false=result)
    return result


#: Largest value C ``int`` addressing can represent; past it a flat address wraps negative.
_INT_MAX = 2**31 - 1


def _float_lit(v: float) -> str:
    """C / CUDA float literal — always carries a decimal so ``0.0`` renders as ``0.0f`` not the invalid ``0f``, and
    large/scientific values keep their exponent."""
    s = repr(float(v))
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return f"{s}f"


def render_expression(expression: Expression, ctx, parent_prec: int = 0) -> str:  # noqa: ANN001 — a ``RenderCtx``
    """Emit a target-language (C / CUDA) expression string. ``ctx`` carries the intrinsic and builtin spelling tables,
    the literal bindings and the inlined temps; ``parent_prec`` drives parenthesization in nested binary chains."""
    if isinstance(expression, Var):
        # An SSA name bound by a literal-constant Load renders as the literal; a single-use temp folded into its
        # consumer (``EMMY_READABLE``) as its rendered sub-expression. ``render_body`` populates both tables.
        lit_map = getattr(ctx, "literal_ssa", None)
        if lit_map and expression.name in lit_map:
            return _float_lit(lit_map[expression.name])
        inl = getattr(ctx, "inline_expressions", None)
        if inl and expression.name in inl:
            return inl[expression.name]
        return expression.name
    if isinstance(expression, Literal):
        if isinstance(expression.value, float) or expression.dtype == "float":
            txt = _float_lit(float(expression.value))
            # A float literal embedded in a non-default-dtype expression composes with the surrounding operands: the
            # target wraps it (``__float2half`` for fp16). Int literals only go through the branch below.
            if ctx is not None:
                wrap_dt = getattr(ctx, "literal_default_dtype", None)
                target = getattr(ctx, "target", None)
                if wrap_dt is not None and target is not None:
                    return target.literal(txt, wrap_dt)
            return txt
        v = int(expression.value)
        return f"{v}LL" if abs(v) > 32768 else str(v)
    if isinstance(expression, BinaryExpression):
        prec = expression.precedence
        left = render_expression(expression.left, ctx, prec)
        right = render_expression(expression.right, ctx, prec + 1)
        # ``//`` is Python floor-div; C integer division ``/`` floors for the non-negative operands shape arithmetic
        # produces.
        c_op = "/" if expression.op == "//" else expression.op
        result = f"{left} {c_op} {right}"
        return f"({result})" if prec < parent_prec else result
    if isinstance(expression, Builtin):
        spelling = ctx.builtins.get(expression.name) if ctx is not None else None
        if spelling is None:
            raise ValueError(f"render: Builtin {expression.name!r} has no target spelling in ctx.builtins")
        return spelling
    if isinstance(expression, FunctionCallExpression):
        spelling = ctx.intrinsics.get(expression.name, expression.name) if ctx is not None else expression.name
        args = ", ".join(render_expression(a, ctx) for a in expression.args)
        return f"{spelling}({args})"
    if isinstance(expression, TernaryExpression):
        c = render_expression(expression.cond, ctx)
        t = render_expression(expression.if_true, ctx)
        f = render_expression(expression.if_false, ctx)
        return f"(({c}) ? ({t}) : ({f}))"
    if isinstance(expression, CastExpression):
        return f"(({expression.dtype})({render_expression(expression.expression, ctx)}))"
    if isinstance(expression, FlatIndex):
        text = render_index(expression.buffer, expression.index, ctx)
        return f"({text})" if parent_prec > 0 else text
    raise NotImplementedError(f"{type(expression).__name__} has no render")


def render_index(buf: str, indices: tuple, ctx: RenderCtx, shape: tuple | None = None) -> str:
    """Row-major flatten ``buf[i0][i1]...`` to a single C/CUDA expression.

    Builds the row-major sum as an ``Expression`` and runs ``simplify`` on it so
    constant-zero indices (typical of size-1 outer dims) drop out via the
    standard ``0 * x → 0`` / ``0 + y → y`` folds rather than emitting
    ``0 * stride`` terms in the output.

    ``shape`` overrides the buffer's declared shape — a paged read flattens over ONE page,
    whose paged axis is the page size rather than the buffer's full extent. A single index is
    already a flat address; any other index spells every dim of a known shape, ``0`` at a size-one dim.
    """
    flat = _flat_index(buf, indices, ctx, shape)
    return flat if isinstance(flat, str) else render_expression(flat, ctx)


def render_address(buf: str, indices: tuple, ctx: RenderCtx) -> str:
    """``&buf[...]`` with the flat index's constant terms added to the pointer: ``&k[i] + 2056``.

    CUDA 12's front end widens an ``int`` index sum to 64 bits before it folds the sum's
    constant, so ``&k[i + 2048]`` and ``&k[i + 2056]`` become two live 64-bit pointers, where
    ``&k[i] + 2048`` is one pointer and a load immediate. A register-carried tile's unrolled
    fragment loads hold a hundred such offsets, and the extra pointers spill a kernel at the
    register cap (the GDN chunk step: 255 registers and a spill on CUDA 12.9, 179 with this form).
    """
    flat = _flat_index(buf, indices, ctx)
    if isinstance(flat, str):
        return f"&{buf}[{flat}]"
    index, offset = _split_offset(flat)
    if not offset:
        return f"&{buf}[{render_expression(flat, ctx)}]"
    return f"&{buf}[{render_expression((index or Literal(0, 'int')), ctx)}] + {offset}"


def _split_offset(e: Expression) -> tuple[Expression | None, int]:
    """``e`` as ``index + offset``, the constant taken through ``+`` and ``* literal``."""
    if isinstance(e, Literal) and e.dtype == "int":
        return None, int(e.value)
    if isinstance(e, BinaryExpression) and e.op == "+":
        (left, a), (right, b) = _split_offset(e.left), _split_offset(e.right)
        return (left if right is None else right if left is None else BinaryExpression("+", left, right)), a + b
    if isinstance(e, BinaryExpression) and e.op == "*":
        for term, scale in ((e.left, e.right), (e.right, e.left)):
            if isinstance(scale, Literal) and scale.dtype == "int":
                index, offset = _split_offset(term)
                return (None if index is None else BinaryExpression("*", index, scale)), offset * int(scale.value)
    return e, 0


def _flat_index(buf: str, indices: tuple, ctx: RenderCtx, shape: tuple | None = None) -> Expression | str:
    """The row-major flat index of ``buf[indices]``: a simplified ``int`` Expression, or the rendered
    ``long long`` sum when the buffer's extent exceeds ``int`` addressing."""
    if len(indices) == 0:
        return Literal(0, "int")
    if len(indices) == 1:
        return indices[0].simplify(SimplifyCtx.empty())
    shape = ctx.shapes.get(buf) if shape is None else shape
    if shape is None or len(shape) != len(indices):
        raise ValueError(f"{buf}: {len(indices)} indices for a buffer of shape {shape} — an index spells every dim of its buffer")
    wide = _exceeds_int_range(shape)
    flat = None
    parts: list[str] = []
    for d, idx in enumerate(indices):
        stride: Expression = Literal(1, "int")
        for k in range(d + 1, len(shape)):
            stride = BinaryExpression("*", stride, _to_expression(shape[k]))
        stride = stride.simplify(SimplifyCtx.empty())
        if wide:
            # Widen the TERM, not the finished sum: `(long long)(a*b + c*d)` computes the whole
            # sum in `int` and widens an already-wrapped result. Casting the index makes each
            # multiply 64-bit, and the accumulating `+` chain stays 64-bit by promotion.
            i_src = render_expression(idx.simplify(SimplifyCtx.empty()), ctx)
            parts.append(f"(long long)({i_src})" if _is_one(stride) else f"(long long)({i_src}) * ({render_expression(stride, ctx)})")
            continue
        term: Expression = idx if _is_one(stride) else BinaryExpression("*", idx, stride)
        flat = term if flat is None else BinaryExpression("+", flat, term)
    if wide:
        return "(" + " + ".join(parts) + ")"
    assert flat is not None
    return flat.simplify(SimplifyCtx.empty())


@dataclass(frozen=True)
class Flat:
    """One contiguous allocation — every buffer, unless it says otherwise."""

    name: str
    vectorizable: ClassVar[bool] = True

    def base(self, index: tuple, ctx: RenderCtx) -> str:
        del index, ctx
        return self.name

    def flat(self, index: tuple, ctx: RenderCtx) -> str:
        return render_index(self.name, index, ctx)


@dataclass(frozen=True)
class Paged:
    """A device table of equal-sized pages, each holding the buffer's shape with the paged axis
    cut to ``page`` — the shape a KV cache has once it is allocated per request.

    The paged index splits: its quotient picks the page (:meth:`base`), its remainder addresses
    inside one (:meth:`flat`), and every other axis flattens row-major exactly as it does for a
    flat buffer. Both halves fold through the ordinary index simplifier, so a loop tiled to a
    multiple of the page size resolves the page once per tile rather than once per element.

    The index along the paged axis is whatever the program spelled — an axis, a value it loaded, a
    mix — so a cache written at an absolute position is a scatter into it, and no shift is applied
    here.

    Not vectorizable: a vector access spans consecutive elements, which may cross a page.
    """

    name: str
    axis: int
    page: int
    vectorizable: ClassVar[bool] = False

    def base(self, index: tuple, ctx: RenderCtx) -> str:
        page = BinaryExpression("//", index[self.axis], Literal(self.page, "int"))
        return f"{self.name}__pages[{render_expression(page.simplify(SimplifyCtx.empty()), ctx)}]"

    def flat(self, index: tuple, ctx: RenderCtx) -> str:
        shape = ctx.shapes.get(self.name)
        if shape is None or len(shape) != len(index):
            raise ValueError(f"paged buffer {self.name!r} needs a declared shape matching its {len(index)} indices; got {shape}")
        within = BinaryExpression("%", index[self.axis], Literal(self.page, "int"))
        inside = tuple(within if d == self.axis else idx for d, idx in enumerate(index))
        return render_index(self.name, inside, ctx, shape=(*shape[: self.axis], self.page, *shape[self.axis + 1 :]))


def _exceeds_int_range(shape) -> bool:
    """Whether this buffer's element count can pass ``INT_MAX``, so its addresses need 64-bit
    arithmetic.

    A flat address is a sum of ``index * stride`` terms, and C evaluates it in ``int`` unless an
    operand is wider. Past ``INT_MAX`` the sum wraps NEGATIVE and the access lands outside the
    allocation — an illegal access that poisons the whole CUDA context, not a wrong answer. It is
    not hypothetical: a 4x512-token Qwen3 trunk plans a 2^32-element activation buffer.

    A symbolic dim is taken at ``DYNAMIC_DIM_MAX``, the exported bound on every ``--dynamic`` axis,
    so the answer is an upper bound over every shape the program can legally be resolved at rather
    than a guess about the one in front of us. (``Dim.hint`` is NOT usable here: it is advisory and
    a run may exceed it.) Buffers that stay inside the range keep their existing 32-bit
    addressing byte for byte, which is most of them — 64-bit index math is not free."""
    numel = 1
    for dim in shape:
        expression = getattr(dim, "expression", None)
        if isinstance(dim, int):
            numel *= dim
        elif isinstance(expression, Literal) and isinstance(expression.value, int):
            numel *= expression.value
        else:
            numel *= DYNAMIC_DIM_MAX
        if numel > _INT_MAX:
            return True
    return False


def _to_expression(d) -> Expression:
    """Pull the ``Expression`` out of a shape element: ``Dim`` exposes its expression
    directly; bare ``int`` becomes ``Literal``. ``Tensor.shape`` is always
    ``tuple[Dim, ...]``, so the ``int`` branch only catches the rare
    bare-int shape that slipped past coercion."""
    if isinstance(d, Dim):
        return d.expression
    if isinstance(d, int):
        return Literal(d, "int")
    raise TypeError(f"_to_expression: unexpected shape element {d!r}")


def _is_one(e: Expression) -> bool:
    return isinstance(e, Literal) and e.value == 1


def render_body(body: Body, ctx: RenderCtx) -> list[str]:
    """Flatten ``stmt.render(ctx)`` over a body sequence.

    Detection of vectorizable Load runs (``LDS.128`` / ``__half2``) is
    no longer done here — the dedicated Kernel-IR pass
    ``050_vectorize_loads`` widens those runs into one :class:`Load`
    with ``extra_names`` populated before render. This function's only
    pre-walk responsibility is registering literal-constant Loads so
    their ``Var(name)`` uses inline as float literals instead of
    materializing a named local. An INTEGER constant is left out of that
    inlining: the map holds floats and every use site renders them as
    float literals, which would put a shift amount or a nibble mask into
    f32 arithmetic that has no spelling for the bit operation reading it.
    Those keep their named local, at their own dtype.
    """
    # Pre-pass: register every literal-constant Load's SSA name in the ctx
    # so subsequent ``Var(name)`` references render as the literal value.
    # The Load itself is then skipped — the loop IR pretty printer behaves
    # the same way (``multiply(in4, 0.044)`` instead of an explicit
    # ``in0 = load mul_1_c1[0]; multiply(in4, in0)`` chain).
    if ctx.literal_constants:
        new_map = dict(ctx.literal_ssa)
        changed = False
        for s in body:
            # Literal-const Loads are always scalar (the vectorize pass
            # excludes literal-const buffers from its widening logic).
            if isinstance(s, Load) and s.is_scalar and s.input in ctx.literal_constants and not _integer_load(s):
                new_map[s.name] = ctx.literal_constants[s.input]
                changed = True
        if changed:
            ctx = replace(ctx, literal_ssa=new_map)

    # Readability (EMMY_READABLE): fold each single-use scalar ``Assign`` temp into its sole
    # consumer's expression — ``m_i__t2 = expf(m_i__t1)`` over ``m_i__t1 = m_i - t0`` becomes
    # ``m_i__t2 = expf(m_i - t0)``. Only when the temp's ONE read is in this same flat body (a nested
    # or repeated use keeps it named), so it never changes evaluation; SASS-identical. Correctness
    # rests on two premises: (1) every node's ``deps()`` is a COMPLETE account of the names its
    # render references — ``_read_counts`` sees readers only through it, so an under-reporting node
    # (a stashed name in a field ``deps`` skips) could hide a second reader of a folded temp; and
    # (2) no statement between the def and its sole read redefines an operand the folded expression
    # reads (a carrier commit — ``l_i = …`` — is a same-body redefinition). (2) is checked below
    # (the straddle guard); (1) is each node's ``deps`` contract.
    inlined: set[str] = set()
    if _readable():
        # The readability scan runs before this body's Loads / Assigns render, so their dtypes are
        # not in ``ctx.ssa_dtypes`` yet. Keep their stamped types locally: folding an expression
        # that needs a target conversion would bypass ``Assign``'s render and drop that conversion.
        local_dtypes = {name: s.dtype.name for s in body if isinstance(s, (Load, Assign)) and s.dtype is not None for name in s.defines()}
        total = _read_counts(body, {})
        local: dict[str, int] = {}
        for s in body:
            for nm in s.deps():
                local[nm] = local.get(nm, 0) + 1
        inline = dict(ctx.inline_expressions)
        bases: dict[str, frozenset[str]] = {}  # inlined temp → the base names its folded expression reads
        stmts = list(body)
        for di, s in enumerate(stmts):
            if not (
                isinstance(s, Assign)
                and s.op.name != "bitcast"
                and total.get(s.name, 0) == 1
                and local.get(s.name, 0) == 1
                and s.name not in inline
            ):
                continue
            # The sole reader must render its operands through ``op_to_expression`` / ``Var.render`` (only
            # ``Assign`` does) — folding into an ``Accum`` / ``Write`` would drop the temp
            # without substituting it, leaving an undefined reference.
            ri = next((j for j, r in enumerate(stmts) if s.name in r.deps()), None)
            if ri is None or not isinstance(stmts[ri], Assign):
                continue
            result_dt = s.dtype.name if s.dtype is not None else "f32"
            arg_dtypes = [local_dtypes.get(arg, ctx.ssa_dtypes.get(arg, "f32")) for arg in s.args]
            if any(dtype != result_dt for dtype in arg_dtypes):
                # The named Assign's target-aware renderer inserts conversions (for example
                # ``__half2float``). The readability-only fold renders a bare Expression tree, so keep
                # the Assign named rather than silently changing its type semantics.
                continue
            # Straddle guard: the fold moves this computation from its def site to the read site, so
            # every base operand — transitively, through already-inlined temps — must not be redefined
            # by a statement in between (``defines()``: a state commit / Accum). Each link checks its
            # own def→read window; chained folds compose, so the windows cover the full span from the
            # earliest constituent's def to the final read.
            names = frozenset(nm for a in s.args for nm in bases.get(a, (a,)))
            if any(nm in st.defines() for st in stmts[di + 1 : ri] for nm in names):
                continue
            expression = op_to_expression(
                s.op.name,
                [Var(a) for a in s.args],
                dtype=s.dtype.name if s.dtype is not None else None,
            )
            rendered = render_expression(expression, replace(ctx, inline_expressions=inline))
            inline[s.name] = f"({rendered})" if isinstance(expression, (BinaryExpression, TernaryExpression)) else rendered
            if s.dtype is not None:
                # The defining Assign is omitted below, so its normal render path cannot
                # register the SSA dtype. Preserve it for the consumer: otherwise an inlined
                # integer shift feeding a bitwise mask is mistaken for f32 and rendered with
                # logical ``&&``/``||`` instead of integer ``&``/``|``.
                ctx.ssa_dtypes[s.name] = s.dtype.name
            bases[s.name] = names
            inlined.add(s.name)
        if inlined:
            ctx = replace(ctx, inline_expressions=inline)

    out: list[str] = []
    for s in body:
        if isinstance(s, Load) and s.is_scalar and s.name in ctx.literal_ssa and s.input in ctx.literal_constants:
            continue  # its value inlines at every use site
        if isinstance(s, Assign) and s.name in inlined:
            continue  # folded into its consumer
        out.extend(render_stmt(s, ctx))
    return out


def _integer_load(load) -> bool:
    """Whether a Load reads an integer buffer — asked of the literal-constant fold, which renders
    its values as float literals."""
    return load.dtype is not None and load.dtype.name in INTEGER_NAMES


def _readable() -> bool:
    return config.readable()


def _read_counts(body: Body, counts: dict[str, int]) -> dict[str, int]:
    """Total reads of each SSA name across ``body`` and every nested body (a name read inside a loop
    counts, so a temp used only in a nested scope is never folded at this level)."""
    for s in body:
        for nm in s.deps():
            counts[nm] = counts.get(nm, 0) + 1
        for b in s.nested():
            _read_counts(b, counts)
    return counts


#: The suffix of a carried cell's next-step slot.
NEXT_STEP = "__next"


@singledispatch
def render_stmt(stmt: Statement, ctx: RenderCtx) -> list[str]:
    """Indented C / CUDA source lines for one statement, dispatched on its kind.

    A block statement recurses through :func:`render_body` with ``ctx.child()``; a leaf returns its
    lines. ``ctx`` carries the target's intrinsic and builtin tables, the current indent, and the
    buffer shapes an index flattens over. Each layer registers the kinds it defines.
    """
    raise NotImplementedError(f"render not registered for {type(stmt).__name__}")


def _resolve_value(name: str, ctx: RenderCtx) -> str:
    """If ``name`` is an SSA name bound by a skipped literal-constant Load,
    return its rendered float-literal form; otherwise return ``name``
    unchanged. ``render_body`` drops the defining Load of every
    literal-constant SSA (its value gets inlined at use sites), so
    Stmts that carry raw SSA-name strings (``Write.values`` / ``Pack.low`` /
    ``Accum.value``) must substitute here instead of emitting an
    undefined identifier."""
    lit = ctx.literal_ssa.get(name) if ctx.literal_ssa else None
    return _float_lit(lit) if lit is not None else name


def _args_at_dtype(target, args: tuple[str, ...], arg_dtypes: list[str], dst_dt: str) -> list[Expression]:
    """Convert each SSA arg to ``dst_dt`` via ``target.convert``, returning
    Expressions ready to drop into ``op_to_expression``. Args already at ``dst_dt``
    pass through as bare ``Var``s. Mismatched args are wrapped in the
    target's conversion intrinsic (e.g. ``__half2float(name)``) by
    parsing ``target.convert``'s output back into a ``FunctionCallExpression`` so
    it composes with the Expression renderer."""

    out: list[Expression] = []
    for a, dt in zip(args, arg_dtypes, strict=True):
        if dt == dst_dt:
            out.append(Var(a))
            continue
        converted = target.convert(a, dt, dst_dt)
        paren = converted.index("(") if "(" in converted else -1
        if paren > 0 and converted.endswith(")"):
            out.append(FunctionCallExpression(converted[:paren], [Var(a)]))
        else:
            out.append(Var(a))
    return out


def _dtype_intrinsics(target, result_dt: str, expression: Expression) -> dict[str, str]:
    """Per-dtype intrinsic overrides for the abstract op names that
    appear inside ``expression``. The Expression renderer reads ``ctx.intrinsics``
    when resolving ``FunctionCallExpression.name`` to a target spelling; we patch
    in the dtype-specific spellings while rendering the fp16-native
    path."""

    overrides: dict[str, str] = {}

    def collect(node):
        if isinstance(node, FunctionCallExpression):
            overrides[node.name] = target.intrinsic(node.name, result_dt)
            for a in node.args:
                collect(a)
        else:
            # Generic Expression walker — the public API is limited; rely on
            # the common subclasses' field names.
            for field_name in ("left", "right", "cond", "if_true", "if_false", "expression"):
                child = getattr(node, field_name, None)
                if child is not None:
                    collect(child)
            args = getattr(node, "args", None)
            if isinstance(args, list):
                for a in args:
                    collect(a)

    collect(expression)
    return overrides


@render_stmt.register
def _(stmt: Load, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    # Inlined scalar constants stay as ``float`` locals — the
    # consumer's ``Assign``'s render will demote / convert if needed.
    # (Only valid for scalar Loads; the vectorize pass excludes
    # literal-const buffers.)
    lit = ctx.literal_constants.get(stmt.input) if ctx.literal_constants else None
    if lit is not None and stmt.is_scalar:
        # An INTEGER constant keeps its own element dtype. Folding it to f32 would decide the
        # CONSUMER's arithmetic too: ``Assign``'s render reads its args' dtypes to choose between
        # the native integer spelling and the f32 promotion, and the bit operations (a pack's
        # shift, a nibble mask) have no f32 spelling at all. Float constants keep the f32
        # binding, which is what every consumer of them already computes in.
        if stmt.dtype is not None and stmt.dtype.name in INTEGER_NAMES:
            ctx.ssa_dtypes[stmt.names[0]] = stmt.dtype.name
            return [f"{pad}{ctx.type_name(stmt.dtype.name)} {stmt.names[0]} = {int(lit)};"]
        ctx.ssa_dtypes[stmt.names[0]] = "f32"
        return [f"{pad}{ctx.type_name('f32')} {stmt.names[0]} = {_float_lit(lit)};"]
    # Prefer the stamped ``stmt.dtype`` (set by ``030_stamp_types``);
    # fall back to ``ctx.buffer_dtypes`` so handwritten test fixtures
    # without a stamped dtype still render correctly.
    src_dt = stmt.dtype.name if stmt.dtype is not None else ctx.buffer_dtypes.get(stmt.input, "f32")
    if stmt.input in ctx.pair_tables and not stmt.carried:
        # Each name's value is computed from its flat table index (``emmy_f4_pair``), exact in every float dtype.
        out = []
        for k, nm in enumerate(stmt.names):
            idx_k = (*stmt.index[:-1], BinaryExpression("+", stmt.index[-1], Literal(k, "int"))) if k else stmt.index
            value = f"emmy_f4_pair({render_index(stmt.input, idx_k, ctx)})"
            if src_dt != "f16":
                value = ctx.target.convert(ctx.target.convert(value, "f16", "f32"), "f32", src_dt)
            ctx.ssa_dtypes[nm] = src_dt
            out.append(f"{pad}{ctx.type_name(src_dt)} {nm} = {value};")
        return out
    mem = ctx.memory_for(stmt.input)
    n = stmt.width
    vec_pair = None if stmt.is_scalar else ctx.target.vector_type(src_dt, n)
    if vec_pair is None or not mem.vectorizable:
        # One local per name, declared in the source buffer's element type so downstream ``Assign``s pick native
        # ops without an immediate promote: a scalar Load, or a run the target cannot read as one vector (no
        # vector type for this dtype and width, or a paged buffer, whose run may cross a page). A carried run's
        # names are then its registers, so ``"load"`` re-assigns them and ``"unpack"`` has nothing to spread.
        if stmt.carried == "unpack":
            return []
        decl = "" if stmt.carried else f"{ctx.type_name(src_dt)} "
        out: list[str] = []
        for k, nm in enumerate(stmt.names):
            idx_k = (*stmt.index[:-1], BinaryExpression("+", stmt.index[-1], Literal(k, "int"))) if k else stmt.index
            ctx.ssa_dtypes[nm] = src_dt
            out.append(f"{pad}{decl}{nm} = {mem.base(idx_k, ctx)}[{mem.flat(idx_k, ctx)}];")
        return out
    # Vector path: one ``<vec_type>`` reinterpret-cast read + N ``.x/.y/.z/.w`` (or indexed) unpacks.
    vec_type, elem_type = vec_pair
    flat = render_index(stmt.input, stmt.index, ctx)  # vectorizable ⇒ Flat, so the plain index
    vname = f"_v_{stmt.names[0]}"
    # ``.x/.y/.z/.w`` accessors only work when ``vec_type``'s native
    # components match ``elem_type`` 1:1 (``float2``→``float``,
    # ``__half2``→``__half``). For wider packed vectors that
    # reinterpret a multi-element pack (``uint2``→4 halves,
    # ``uint4``→8 halves), each ``.x``/``.y`` slot holds multiple
    # elements, so we fall back to array-style indexing through a
    # reinterpret-cast; the bf16 pair reads the same way.
    native_n = {"float2": 2, "float4": 4, "__half2": 2}.get(vec_type)
    use_array_index = native_n != n
    if stmt.carried == "load":  # re-assign the declared vector only (see the class doc)
        return [f"{pad}{vname} = __ldg(reinterpret_cast<const {vec_type}*>(&{stmt.input}[{flat}]));"]
    if stmt.carried == "unpack":
        if use_array_index:
            return [f"{pad}{nm} = {vname}_h[{k}];" for k, nm in enumerate(stmt.names)]
        return [f"{pad}{nm} = {vname}.{c};" for nm, c in zip(stmt.names, ("x", "y", "z", "w"), strict=False)]
    out_lines = [f"{pad}{vec_type} {vname} = *reinterpret_cast<const {vec_type}*>(&{stmt.input}[{flat}]);"]
    if use_array_index:
        arr_name = f"{vname}_h"
        out_lines.append(f"{pad}const {elem_type}* {arr_name} = reinterpret_cast<const {elem_type}*>(&{vname});")
        out_lines.extend(f"{pad}{elem_type} {nm} = {arr_name}[{k}];" for k, nm in enumerate(stmt.names))
    else:
        components = ("x", "y", "z", "w")[:n]
        out_lines.extend(f"{pad}{elem_type} {nm} = {vname}.{c};" for nm, c in zip(stmt.names, components, strict=True))
    for nm in stmt.names:
        ctx.ssa_dtypes[nm] = src_dt
    return out_lines


@render_stmt.register
def _(stmt: Assign, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    op_name = stmt.op.name
    arg_dtypes = [ctx.ssa_dtypes.get(a, "f32") for a in stmt.args]
    if op_name == "bitcast":
        if len(stmt.args) != 1 or stmt.dtype is None:
            raise ValueError("bitcast Assign requires one argument and an explicit destination dtype")
        result_dt = stmt.dtype.name
        body = ctx.target.bitcast(stmt.args[0], arg_dtypes[0], result_dt)
        ctx.ssa_dtypes[stmt.name] = result_dt
        return [f"{pad}{ctx.type_name(result_dt)} {stmt.name} = {body};"]
    # Default rule: f16 stays narrow only when all inputs agree; generic
    # integer arithmetic/bit operations likewise retain one common integer
    # dtype. Other cases promote to f32.
    # Explicit ``stmt.dtype`` overrides — the demote pass uses this
    # to force a narrower dtype on the result.
    result_dt = stmt.dtype.name if stmt.dtype is not None else dtype_promote(op_name, arg_dtypes)
    if result_dt == "f16" and stmt.op.decodes is not None and arg_dtypes == [stmt.op.decodes]:
        # Every fp8 value is an f16 value: the decode converts straight to it, not through f32 and back.
        ctx.ssa_dtypes[stmt.name] = result_dt
        return [f"{pad}{ctx.type_name(result_dt)} {stmt.name} = {ctx.target.convert(stmt.args[0], arg_dtypes[0], result_dt)};"]

    def build(operands: list[Expression], dtype: str) -> Expression:
        expression = op_to_expression(op_name, operands, dtype=dtype)
        if dtype != "f32" and isinstance(expression, BinaryExpression) and ctx.target.intrinsic(op_name, dtype) != op_name:
            # A target that spells this operator as a call (CUDA's non-contracting f16 ``_rn`` ops).
            expression = FunctionCallExpression(op_name, tuple(operands))
        return expression

    return _render_promoted(ctx, stmt.name, op_name, stmt.args, arg_dtypes, result_dt, build)


def _render_promoted(
    ctx: RenderCtx,
    name: str,
    op_name: str,
    args: tuple[str, ...],
    arg_dtypes: list[str],
    result_dt: str,
    build: Callable[[list[Expression], str], Expression],
) -> list[str]:
    """``name = build(args)`` at ``result_dt``: natively when the target computes ``op_name`` at that dtype and every
    argument is already there, else promoted to f32 and converted once."""
    pad = _pad(ctx.indent)
    all_args_at_result = bool(arg_dtypes) and all(d == result_dt for d in arg_dtypes)
    integer_promotion = result_dt in INTEGER_NAMES and bool(arg_dtypes) and all(d in INTEGER_NAMES for d in arg_dtypes)
    if result_dt != "f32" and ctx.target.has_native_op(op_name, result_dt) and (all_args_at_result or integer_promotion):
        # Native non-f32 path: all args already at ``result_dt`` so no per-arg conversion is needed. Render with the
        # target's dtype-specific intrinsics; ``Literal.render`` wraps embedded float literals via the target so
        # they compose with the non-default-dtype operands. Mixed dtypes (an explicit demotion) fall through to
        # the promote-then-demote path: computing in fp32 and converting once preserves precision better than
        # converting each arg to fp16 first.
        expression = build(_args_at_dtype(ctx.target, args, arg_dtypes, result_dt), result_dt)
        saved_intr = ctx.intrinsics
        saved_lit = ctx.literal_default_dtype
        ctx.intrinsics = {**saved_intr, **_dtype_intrinsics(ctx.target, result_dt, expression)}
        ctx.literal_default_dtype = result_dt
        try:
            body = render_expression(expression, ctx)
        finally:
            ctx.intrinsics = saved_intr
            ctx.literal_default_dtype = saved_lit
        ctx.ssa_dtypes[name] = result_dt
        return [f"{pad}{ctx.type_name(result_dt)} {name} = {body};"]

    # f32 / no-native path: promote args to f32, render in f32, then convert the result back when narrower.
    body_str = render_expression(build(_args_at_dtype(ctx.target, args, arg_dtypes, "f32"), "f32"), ctx)
    if result_dt != "f32":
        body_str = ctx.target.convert(body_str, "f32", result_dt)
    ctx.ssa_dtypes[name] = result_dt
    return [f"{pad}{ctx.type_name(result_dt)} {name} = {body_str};"]


@render_stmt.register
def _(stmt: Affine, ctx: RenderCtx) -> list[str]:
    arg_dtypes = [ctx.ssa_dtypes.get(a, "f32") for a in stmt.args]
    result_dt = stmt.dtype.name if stmt.dtype is not None else dtype_promote("add", arg_dtypes)

    def build(operands: list[Expression], dtype: str) -> Expression:
        expression: Expression | None = None
        for coefficient, operand in zip(stmt.coefficients, operands, strict=True):
            term = operand if abs(coefficient) == 1 else BinaryExpression("*", Literal(abs(coefficient), "float"), operand)
            if expression is None:
                expression = term if coefficient > 0 else BinaryExpression("-", Literal(0.0, "float"), term)
            else:
                expression = BinaryExpression("+" if coefficient > 0 else "-", expression, term)
        if stmt.offset != 0 or expression is None:
            literal = Literal(abs(stmt.offset), "float")
            expression = literal if expression is None else BinaryExpression("+" if stmt.offset > 0 else "-", expression, literal)
        return expression

    return _render_promoted(ctx, stmt.name, "add", stmt.args, arg_dtypes, result_dt, build)


@render_stmt.register
def _(stmt: Accum, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    # Accumulator dtype — explicit on Accum once the dtype policy
    # has frozen it; otherwise default to fp32 (legacy behavior).
    acc_dt = (stmt.dtype or F32).name
    ctx.ssa_dtypes[stmt.name] = acc_dt
    value_dt = ctx.ssa_dtypes.get(stmt.value, "f32")
    rhs = ctx.target.convert(stmt.value, value_dt, acc_dt)
    # Left operand of the fold: ``base`` (a rescaled state, already at acc_dt)
    # when set, else the carried ``name`` itself.
    base = stmt.base if stmt.base is not None else stmt.name
    # Spelling (``+=`` / ``*=`` / ``fmax`` / ``fmin``) from the shared reduce
    # registry; defaults to additive for non-reduce ops.
    spelling = reduce_spelling(stmt.op)
    if spelling.intrinsic is not None:
        fn = ctx.target.intrinsic(spelling.intrinsic, acc_dt)
        return [f"{pad}{stmt.name} = {fn}({base}, {rhs});"]
    if base == stmt.name:
        return [f"{pad}{stmt.name} {spelling.compound} {rhs};"]
    return [f"{pad}{stmt.name} = {base} {spelling.infix} {rhs};"]


@render_stmt.register
def _(stmt: Carry, ctx: RenderCtx) -> list[str]:
    # The carrying ``Loop`` declared this slot beside the one reads see, and commits it after
    # every step.
    dtype = (stmt.dtype or F32).name
    value = ctx.target.convert(stmt.value, ctx.ssa_dtypes.get(stmt.value, "f32"), dtype)
    return [f"{_pad(ctx.indent)}{stmt.name}{NEXT_STEP}[{render_index(stmt.name, stmt.index, ctx)}] = {value};"]


@render_stmt.register
def _(stmt: Pre, ctx: RenderCtx) -> list[str]:
    dtype = ctx.ssa_dtypes.get(stmt.carrier, "f32")
    ctx.ssa_dtypes[stmt.name] = dtype
    return [f"{_pad(ctx.indent)}{ctx.type_name(dtype)} {stmt.name} = {stmt.carrier}[{render_index(stmt.carrier, stmt.index, ctx)}];"]


@render_stmt.register
def _(stmt: Init, ctx: RenderCtx) -> list[str]:
    ctx.ssa_dtypes[stmt.name] = stmt.dtype.name
    return [f"{_pad(ctx.indent)}{ctx.type_name(stmt.dtype)} {stmt.name} = {ctx.identity_literal(stmt.identity, stmt.dtype)};"]


@render_stmt.register
def _(stmt: Let, ctx: RenderCtx) -> list[str]:
    if stmt.dtype is None:
        return [f"{_pad(ctx.indent)}auto {stmt.name} = {render_expression(stmt.value, ctx)};"]
    ctx.ssa_dtypes[stmt.name] = stmt.dtype.name
    value = ctx.identity_literal(stmt.value.value, stmt.dtype) if isinstance(stmt.value, Literal) else render_expression(stmt.value, ctx)
    return [f"{_pad(ctx.indent)}{ctx.type_name(stmt.dtype)} {stmt.name} = {value};"]


def _swizzled(stmt: Write, flat: str, ctx: RenderCtx, index: tuple[Expression, ...] | None = None) -> str:
    """The flattened store index under this shared slab's derived physical layout."""
    from emmy.compiler.ir.kernel.ir import smem_layout_index  # noqa: PLC0415 — kernel IR imports this module

    return smem_layout_index(stmt.swizzle, flat, stmt.output, ctx, stmt.index if index is None else index)


@render_stmt.register
def _(stmt: Write, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    out_dt = ctx.buffer_dtypes.get(stmt.output, "f32")
    # Prefer the stamped ``stmt.value_dtype`` (set by ``030_stamp_types``);
    # fall back to ``ctx.ssa_dtypes`` for legacy/handwritten paths.
    stamped_value_dt = stmt.value_dtype.name if stmt.value_dtype is not None else None
    mem = ctx.memory_for(stmt.output)
    if stmt.is_scalar:
        # Scalar path. Convert at the store boundary only when the
        # value's SSA dtype disagrees with the destination buffer's
        # dtype — native chains write through with no conversion.
        flat = _swizzled(stmt, mem.flat(stmt.index, ctx), ctx)
        value_dt = stamped_value_dt or ctx.ssa_dtypes.get(stmt.value, "f32")
        rhs = ctx.target.convert(_resolve_value(stmt.value, ctx), value_dt, out_dt)
        if stmt.atomic:
            # Cross-CTA additive finalize: every contributing CTA accumulates into the
            # same cell (the output is zero-init'd per launch — ``CudaOp.zero_outputs``).
            return [f"{pad}atomicAdd(&{mem.base(stmt.index, ctx)}[{flat}], {rhs});"]
        return [f"{pad}{mem.base(stmt.index, ctx)}[{flat}] = {rhs};"]
    # Vectorized path. Per-value dtype conversion: every SSA arg
    # must be at ``out_dt`` before packing.
    n = stmt.width
    converted: list[str] = []
    for nm in stmt.values:
        src_dt = stamped_value_dt or ctx.ssa_dtypes.get(nm, "f32")
        resolved = _resolve_value(nm, ctx)
        converted.append(resolved if src_dt == out_dt else ctx.target.convert(resolved, src_dt, out_dt))
    vec_pair = ctx.target.vector_type(out_dt, n)
    if vec_pair is None or not mem.vectorizable:
        # Target doesn't support this width, or the buffer does not guarantee consecutive
        # elements are contiguous (a paged store may cross a page) — fall back to scalar
        # writes. The vectorize pass should have avoided this, but
        # render's job is to always produce valid code.
        lines: list[str] = []
        for k in range(n):
            idx_k = tuple(stmt.index[:-1]) + (BinaryExpression("+", stmt.index[-1], Literal(k, "int")),)
            flat = _swizzled(stmt, mem.flat(idx_k, ctx), ctx, idx_k)
            lines.append(f"{pad}{mem.base(idx_k, ctx)}[{flat}] = {converted[k]};")
        return lines
    vec_type, _elem_type = vec_pair
    logical_flat = render_index(stmt.output, stmt.index, ctx)
    flat = _swizzled(stmt, logical_flat, ctx)
    # Native-width vectors (``float2`` / ``float4`` / ``__half2``) take
    # a positional constructor — ``make_float2(a, b)`` for fp32 paths,
    # ``__halves2half2(a, b)`` for the fp16 pair. Wider packed vectors
    # (``uint2`` / ``uint4``) and the bf16 pair re-interpret arrays of elements — we
    # stage the elements into a local array, then store the array's
    # vector view in one transaction.
    # A content digest disambiguates the temp name across sibling Writes
    # that read from the same SSA — e.g. broadcasting a shared
    # literal-constant value (where every Write's ``values[0]`` is the
    # same SSA name) used to collide on ``_vs_<name>``. Siblings differ in
    # destination and/or operands, so (output, index, values) separates
    # them — and unlike the old ``id(stmt)`` suffix it is deterministic
    # across processes, which the content-addressed cubin cache requires
    # (an address-derived name re-keys the kernel on every boot).
    import hashlib  # noqa: PLC0415 — render-path-only dependency

    sig = f"{stmt.output}|{flat}|{','.join(converted)}"
    # 8 hex chars (32 bits): at [:4] two DIFFERENT sibling Writes sharing ``values[0]`` collide
    # on the temp name within one C scope at ~n²/2¹⁷ — a duplicate-declaration nvcc error.
    temp = f"_vs_{stmt.values[0]}_{hashlib.sha1(sig.encode()).hexdigest()[:8]}"
    if vec_type == "__half2":
        return [
            f"{pad}{vec_type} {temp} = __halves2half2({converted[0]}, {converted[1]});",
            f"{pad}*reinterpret_cast<{vec_type}*>(&{stmt.output}[{flat}]) = {temp};",
        ]
    if vec_type in ("float2", "float4"):
        args = ", ".join(converted)
        return [
            f"{pad}{vec_type} {temp} = make_{vec_type}({args});",
            f"{pad}*reinterpret_cast<{vec_type}*>(&{stmt.output}[{flat}]) = {temp};",
        ]
    # Volta A's crosswise layout is 64-bit-granular: an eight-half logical fill run becomes
    # two separately permuted uint2 stores. The gmem side remains one 128-bit Load; only the
    # shared deposit splits. Four-half runs already fit one crosswise access.
    from emmy.compiler.ir.kernel.ir import VOLTA_CROSSWISE  # noqa: PLC0415 — kernel IR imports this module

    if stmt.swizzle == VOLTA_CROSSWISE and n == 8:
        elem_type = ctx.type_name(out_dt)
        arr = temp
        init = ", ".join(converted)
        idx_hi = (*stmt.index[:-1], BinaryExpression("+", stmt.index[-1], Literal(4, "int")))
        flat_hi = _swizzled(stmt, render_index(stmt.output, idx_hi, ctx), ctx, idx_hi)
        return [
            f"{pad}{elem_type} {arr}[{n}] = {{ {init} }};",
            f"{pad}*reinterpret_cast<uint2*>(&{stmt.output}[{flat}]) = *reinterpret_cast<const uint2*>({arr});",
            f"{pad}*reinterpret_cast<uint2*>(&{stmt.output}[{flat_hi}]) = *reinterpret_cast<const uint2*>({arr} + 4);",
        ]
    # Packed widths (uint2 / uint4) over a 16-bit float, and the bf16 pair: stage through a local
    # array of elements so we never need to construct a vector literal directly.
    elem_type = ctx.type_name(out_dt)
    arr = temp
    init = ", ".join(converted)
    return [
        f"{pad}{elem_type} {arr}[{n}] = {{ {init} }};",
        f"{pad}*reinterpret_cast<{vec_type}*>(&{stmt.output}[{flat}]) = *reinterpret_cast<const {vec_type}*>({arr});",
    ]


@render_stmt.register
def _(stmt: Select, ctx: RenderCtx) -> list[str]:
    dtype = dtype_promote("add", [ctx.ssa_dtypes.get(b.value, "f32") for b in stmt.branches])
    ctx.ssa_dtypes[stmt.name] = dtype
    ty = ctx.type_name(dtype)
    expression = select_to_ternary(stmt, ty)
    return [f"{_pad(ctx.indent)}{ty} {stmt.name} = {render_expression(expression, ctx)};"]


@render_stmt.register
def _(stmt: ZeroPrologue, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    p1 = _pad(ctx.indent + 1)
    threads = "(int)(blockDim.x * blockDim.y * blockDim.z)"
    grid = f"{threads} * (int)(gridDim.x * gridDim.y * gridDim.z)"
    block = "(int)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))"
    thread = "(int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))"
    return [
        f"{pad}{{",
        f"{p1}int* _zp_{stmt.dst} = (int*){stmt.dst};",
        f"{p1}for (int _zi = {block} * {threads} + {thread}; _zi < {stmt.words}; _zi += {grid}) _zp_{stmt.dst}[_zi] = 0;",
        f"{pad}}}",
    ]


@render_stmt.register
def _(node: Loop, ctx: RenderCtx) -> list[str]:

    pad = _pad(ctx.indent)
    out: list[str] = []
    # Per-Loop ``<dtype> <acc> = identity;`` for each distinct ``Accum`` fold in the
    # immediate body — the seed rides on the fold, derived here from ``op.identity`` at
    # the fold's ``dtype`` (so fp32-over-fp16 declares ``float acc = 0.0f;``, a fp16
    # ``max`` declares ``__half acc = __float2half(0.0f);``), no explicit ``Init``. This
    # is the SINGLE seed-placement path, and ``Accum`` is the ONLY stmt it reads: a reduce
    # ``Loop``'s body already holds the loose fold ``Accum``\\ s (dissolved at recognition),
    # and a cross-partition combine reaches a statement position only by rendering itself to
    # ``Accum``\\ s (``algebra.merge_stmts``), so no neutral element ever travels on a term and
    # there is no second carrier kind to special-case. A nested fold re-declares per enclosing
    # iteration (scope-local shadowing), so a same-named outer carrier is harmless. This
    # is the *serial* schedule's placement; a cooperative / cross-CTA realization reads
    # the same fold ``Accum``\\ s' ``op.identity`` to seed its partials.
    seen: set[str] = set()
    # A carried state is two slots of its cell shape: the one the previous step left, which
    # every read sees, and the one this step defines, committed when the step ends. The serial
    # placement again; a schedule that keeps one slot, or stores it elsewhere, reads the same state.
    for name, shape in node.carries.items():
        carry = next(c for c in node.body.carries if c.name == name)
        extents = tuple(1 if isinstance(d, int) else int(d.extent.as_static()) for d in shape)
        size = 1
        for extent in extents:
            size *= extent
        # The seed is a constant in every cell or a buffer of the state's shape, copied whole.
        if isinstance(carry.seed, str):
            seed = f"{carry.seed}, {carry.seed} + {size}"
        else:
            seed = f"{size}, {ctx.identity_literal(carry.seed, carry.dtype)}"
        for slot in (name, f"{name}{NEXT_STEP}"):
            ctx.shapes[slot] = extents
            out.append(f"{pad}std::vector<{ctx.type_name(carry.dtype)}> {slot}({seed});")
        ctx.ssa_dtypes[name] = (carry.dtype or F32).name
    for s in node.body:
        if isinstance(s, Accum) and s.name not in seen:
            seen.add(s.name)
            identity = s.op.identity
            if identity is None:
                raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
            if node.seed:
                out.append(f"{pad}{ctx.type_name(s.dtype)} {s.name} = {ctx.identity_literal(identity, s.dtype)};")
            ctx.ssa_dtypes[s.name] = (s.dtype or F32).name
    var = node.axis.name
    extent = _extent_c(node.axis, ctx)
    if node.unroll:
        out.append(f"{pad}#pragma unroll")
    out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
    inner = ctx.child()
    out.extend(render_body(node.body, inner))
    out.extend(f"{_pad(inner.indent)}{name} = {name}{NEXT_STEP};" for name in node.carries)
    out.append(f"{pad}}}")
    return out


def _extent_c(ax: Axis, ctx: RenderCtx) -> str:
    """Render one axis extent as a C expression: literal int for static
    ``Dim``, the symbolic name for ``Dim('seq_len')``. ``Dim.__str__``
    already returns the bare ``value`` for those, so a plain ``str`` is
    enough. A composite extent (the ceil-div block axis of a hint-driven
    masked tile, ``(seq_len+31)//32``) must go through the C expression renderer
    so ``//`` becomes ``/`` — ``str`` would leak the Python spelling."""
    ext = ax.extent
    if ext.is_static or isinstance(ext.expression, Var):
        return str(ext)
    # Parenthesize: the extent lands as a ``%`` / ``/`` operand in the decode
    # (``idx % (seq_len+31)/32`` would otherwise mis-associate).
    return f"({render_expression(ext.expression, ctx)})"


@render_stmt.register
def _(stmt: StridedLoop, ctx: RenderCtx) -> list[str]:
    """``for (int axis = start; axis < extent; axis += step)`` with the
    same per-Loop accumulator-init prelude as ``Loop``'s render."""

    pad = _pad(ctx.indent)
    out: list[str] = []
    seen: set[str] = set()
    for s in stmt.body:
        if isinstance(s, Accum) and s.name not in seen:
            seen.add(s.name)
            identity = s.op.identity
            if identity is None:
                raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
            if stmt.seed:
                out.append(f"{pad}{ctx.type_name(s.dtype)} {s.name} = {ctx.identity_literal(identity, s.dtype)};")
            ctx.ssa_dtypes[s.name] = (s.dtype or F32).name
    var = stmt.axis.name
    start_str = render_expression(stmt.start, ctx)
    step_str = render_expression(stmt.step, ctx)
    # ``_extent_c`` renders a symbolic bound (a cooperative reduce over a dynamic
    # ``seq_len`` strides each lane to the runtime extent — the ``< seq_len`` bound is the
    # masked tail, idle lanes folding the carrier identity) as the C name, a static one as
    # its literal int.
    if stmt.unroll:
        out.append(f"{pad}#pragma unroll")
    if stmt.end is not None:
        # Hoist the override bound into the for-init (``<var>_end``, referenceable by body
        # stmts — the staged prefetch clamp) so a non-literal bound is evaluated once, not
        # per iteration / per use (the inline form measurably spills on the flash stream).
        init = f"int {var} = {start_str}, {var}_end = {render_expression(stmt.end, ctx)}"
        out.append(f"{pad}for ({init}; {var} < {var}_end; {var} += {step_str}) {{")
    else:
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {_extent_c(stmt.axis, ctx)}; {var} += {step_str}) {{")
    inner = ctx.child()
    if isinstance(stmt.start, Literal) and isinstance(stmt.step, Literal):
        start, step = stmt.start.value, stmt.step.value
        if isinstance(start, int) and isinstance(step, int) and (stride := math.gcd(start, step)):
            inner.aligned[var] = stride
    out.extend(render_body(stmt.body, inner))
    out.append(f"{pad}}}")
    return out


@render_stmt.register
def _(stmt: Cond, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    cond = render_expression(stmt.cond, ctx)
    inner = ctx.child()
    body = render_body(stmt.body, inner)
    out = [f"{pad}if ({cond}) {{", *body, f"{pad}}}"]
    if stmt.else_body:
        out[-1] = f"{pad}}} else {{"
        out.extend(render_body(stmt.else_body, inner))
        out.append(f"{pad}}}")
    return out
