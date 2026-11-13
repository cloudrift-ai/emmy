"""C-family source for statement bodies: the render context, index flattening, and one render function per statement type.

``render_stmt`` dispatches on the statement type. The statement types every IR shares register here; Kernel IR registers its
own beside their classes. ``render_body`` is the entry point the CUDA kernel renderer and the Loop IR runner call.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import singledispatch
from itertools import count
from typing import TYPE_CHECKING, ClassVar, Protocol

from emmy.compiler.dtype import F32, INTEGER_NAMES
from emmy.compiler.ir.address import flat_index
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import reduce_spelling
from emmy.compiler.ir.expression import (
    BinaryExpression,
    Builtin,
    CastExpression,
    Expression,
    FunctionCallExpression,
    Literal,
    SimplifyCtx,
    TernaryExpression,
    Var,
)
from emmy.compiler.ir.stmt import (
    Accum,
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
)
from emmy.emmy_ir import DTYPES as _DTYPE_NAMES

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
    conversion intrinsics, per-dtype op spellings, vector load shapes).
    Everything else here is generic walk state.

    ``intrinsics`` / ``builtins`` keep the legacy abstract→spelling
    indirection used by ``FunctionCallExpression.render`` / ``Builtin.render``
    for symbols that don't vary by dtype (``thread_idx.x`` →
    ``threadIdx.x``, etc.). Dtype-aware lookups go through
    ``ctx.target.intrinsic(...)`` instead.

    ``shapes`` maps every buffer to its declared shape, for the statements that hold one index per dimension.
    """

    target: RenderTarget = field(default_factory=_default_render_target)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    indent: int = 1
    intrinsics: dict[str, str] = field(default_factory=dict)
    builtins: dict[str, str] = field(default_factory=dict)
    # The SSA names some carried ``Load`` of the kernel re-assigns. They live in the packed vector their first Load
    # declares, and ``lanes`` holds the text of each one's lane once that Load is rendered: a read of one reads the
    # lane where the read stands. One table per kernel, shared by every scope's context.
    carried_names: frozenset[str] = frozenset()
    lanes: dict[str, str] = field(default_factory=dict)
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
    # Per-buffer canonical dtype tokens (``"f32"`` / ``"f16"``) for every global-buffer name (kernel inputs +
    # outputs) and every shared buffer declared so far. The shared statements do not read it: a typed ``Load`` and
    # ``Write`` carry their dtype. The fragment statements still do: a store converts to its buffer's element type.
    buffer_dtypes: dict[str, str] = field(default_factory=dict)
    # The dtype each name was declared at, recorded as ``render_body`` walks: the source dtype a conversion
    # (``name = (dtype)source``) and a fragment statement's operand read. Nothing is promoted from it.
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
    # Comparisons and the boolean combines of a traced mask, whose operands are floats or truth values: C's
    # nonzero→bool conversion makes the logical spellings valid on them. Integer operands use ``_INTEGER_BINARY_OP``
    # instead, including packed bitwise decode operations.
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


#: The operations a target spells as a call of one argument: a math function, or the approximate form of one.
_UNARY_CALLS = frozenset(
    {"exp", "log", "rsqrt", "sin", "cos", "tanh", "sqrt", "erf", "floor"}
    | {"exp_fast", "log_fast", "sin_fast", "cos_fast", "tanh_fast", "sqrt_fast", "reciprocal_fast"}
)


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
    if fn in ("pow", "pow_fast", "divide_fast"):
        return FunctionCallExpression(fn, tuple(inputs))
    if fn == "negative":
        return BinaryExpression("-", Literal(0, "int") if dtype in INTEGER_NAMES else Literal(0.0, "float"), inputs[0])
    if fn in ("copy", "pad"):
        return inputs[0]
    if fn in ("from_f8e4m3", "from_f8e5m2"):
        # fp8 decode cast: the value is the conversion the typed body spells ahead of it (``target.convert`` spells
        # the decode), so the op itself is an identity here — like ``copy``, never a raw byte move.
        return inputs[0]
    if fn in ("to_f8e4m3", "to_f8e5m2"):
        # fp8 encode cast — the decode twin: the typed body converts the f32 value to the fp8 dtype, which
        # ``target.convert`` spells as the <cuda_fp8.h> constructor (round-to-nearest-even, saturate-to-finite).
        return inputs[0]
    if fn == "to_f4e2m1":
        # e2m1 encode — NOT the fp8 encodes' identity. Those convert to an fp8 type whose
        # constructor rounds; the codes here land on an ordinary integer carrier, where the same
        # identity would truncate the value instead. The render emits the rounding as a helper
        # (``_F4_ENCODE_PRELUDE``).
        return FunctionCallExpression("emmy_to_f4e2m1", (inputs[0],))
    if len(inputs) == 1 and (definition := FunctionCallExpression(fn, (Var("x"),)).definition) is not None:
        # A composite function has no spelling here: the operations' table defines it by primitive ones.
        spelled = definition.rebuild(
            lambda n: n if n.operation is None else op_to_expression(n.operation.name, list(n.operands), dtype=dtype)
        )
        return spelled.substitute({"x": inputs[0]})
    if fn in _UNARY_CALLS:
        return FunctionCallExpression(fn, tuple(inputs))
    if fn == "abs":
        return FunctionCallExpression("fabs", tuple(inputs))
    if fn == "where":
        # np.where(cond, a, b) — the mask-apply tail of the explicit-mask
        # subgraph. A float or truth-value cond is C-truthy (nonzero) in ``?:``.
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

    def spelled(node: Expression) -> Expression:
        # The merge runs at ``dt`` — cast any name read at a different dtype (e.g. a raw ``__half`` value loaded into
        # the partial, as in fp16 flash's ``p · v``) so the operator isn't ambiguous; an operation node takes the
        # target's spelling of its operation.
        if isinstance(node, Var):
            return node if ctx.ssa_dtypes.get(node.name, dt) == dt else CastExpression(ty, node)
        op = node.operation
        return op_to_expression(op.name, list(node.operands), dtype=dt) if op is not None else node

    for a in program:
        rhs = render_expression(a.value.rebuild(spelled), ctx)
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


def _float_lit(v: float) -> str:
    """C / CUDA float literal — always carries a decimal so ``0.0`` renders as ``0.0f`` not the invalid ``0f``, and
    large/scientific values keep their exponent."""
    s = repr(float(v))
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return f"{s}f"


_INDEX_CALLS = {"minimum": "min", "maximum": "max"}
#: The precedence a unary operand renders at: tighter than any binary operator, so a nested operation is parenthesized.
ATOM_PRECEDENCE = 100
_BITWISE = frozenset({"<<", ">>", "&", "|", "^"})


def render_expression(expression: Expression, ctx: RenderCtx, parent_prec: int = 0) -> str:
    """Emit a target-language (C / CUDA) expression string. ``ctx`` carries the intrinsic and builtin spelling tables,
    the literal bindings and the inlined temps; ``parent_prec`` drives parenthesization in nested binary chains."""
    if isinstance(expression, Var):
        return ctx.lanes.get(expression.name, expression.name)
    if isinstance(expression, Literal):
        if isinstance(expression.value, float) or expression.dtype == "float":
            txt = _float_lit(float(expression.value))
            # A float literal embedded in a non-default-dtype expression composes with the surrounding operands: the
            # target wraps it (``__float2half`` for fp16). Int literals only go through the branch below.
            if ctx.literal_default_dtype is not None:
                return ctx.target.literal(txt, ctx.literal_default_dtype)
            return txt
        v = int(expression.value)
        return f"{v}LL" if abs(v) > 32768 else str(v)
    if isinstance(expression, BinaryExpression):
        prec = expression.precedence
        # C ranks the shifts and the bitwise operators apart from the comparisons the table puts them beside (``&``
        # below ``==``, ``|`` below ``&``), so one is parenthesized wherever it nests, and so is a comparison under it.
        bitwise = expression.op in _BITWISE
        left = render_expression(expression.left, ctx, prec + 1 if bitwise else prec)
        right = render_expression(expression.right, ctx, prec + 1)
        # ``//`` is Python floor-div; C integer division ``/`` floors for the non-negative operands shape arithmetic
        # produces.
        c_op = "/" if expression.op == "//" else expression.op
        result = f"{left} {c_op} {right}"
        return f"({result})" if prec < parent_prec or (bitwise and parent_prec) else result
    if isinstance(expression, Builtin):
        spelling = ctx.builtins.get(expression.name) if ctx is not None else None
        if spelling is None:
            raise ValueError(f"render: Builtin {expression.name!r} has no target spelling in ctx.builtins")
        return spelling
    if isinstance(expression, FunctionCallExpression):
        # The table's spellings an index or a predicate carries: a negation is the prefix, a minimum or maximum the
        # C builtin (``min``/``max`` take integers and floats alike).
        if expression.name == "negative" and len(expression.args) == 1:
            return f"(-{render_expression(expression.args[0], ctx, ATOM_PRECEDENCE)})"
        name = _INDEX_CALLS.get(expression.name, expression.name)
        spelling = ctx.intrinsics.get(name, name) if ctx is not None else name
        args = ", ".join(render_expression(a, ctx) for a in expression.args)
        return f"{spelling}({args})"
    if isinstance(expression, TernaryExpression):
        c = render_expression(expression.cond, ctx)
        t = render_expression(expression.if_true, ctx)
        f = render_expression(expression.if_false, ctx)
        return f"(({c}) ? ({t}) : ({f}))"
    if isinstance(expression, CastExpression):
        return f"(({expression.dtype})({render_expression(expression.expression, ctx)}))"
    raise NotImplementedError(f"{type(expression).__name__} has no render")


def render_index(buf: str, indices: tuple, ctx: RenderCtx, shape: tuple | None = None) -> str:
    """The flat element index of ``buf[indices]`` as a C/CUDA expression, for a statement that holds one index per
    dimension (a fragment statement, a paged access, a carried state): the row-major sum (``address.flat_index``),
    simplified, so constant-zero indices (typical of size-1 outer dims) drop out. One index is flat already and
    prints as it is given.

    ``shape`` overrides the buffer's declared shape — a paged read flattens over ONE page,
    whose paged axis is the page size rather than the buffer's full extent.
    """
    return render_expression(_flat_index(buf, indices, ctx, shape), ctx)


def render_address(buf: str, indices: tuple, ctx: RenderCtx) -> str:
    """``&buf[...]`` with the flat index's constant terms added to the pointer: ``&k[i] + 2056``.

    CUDA 12's front end widens an ``int`` index sum to 64 bits before it folds the sum's
    constant, so ``&k[i + 2048]`` and ``&k[i + 2056]`` become two live 64-bit pointers, where
    ``&k[i] + 2048`` is one pointer and a load immediate. A register-carried tile's unrolled
    fragment loads hold a hundred such offsets, and the extra pointers spill a kernel at the
    register cap (the GDN chunk step: 255 registers and a spill on CUDA 12.9, 179 with this form).
    """
    flat = _flat_index(buf, indices, ctx)
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


def _flat_index(buf: str, indices: tuple, ctx: RenderCtx, shape: tuple | None = None) -> Expression:
    """The flat index of ``buf[indices]``: one index as it is given, several flattened over the buffer's shape."""
    if len(indices) == 1:
        return indices[0]
    try:
        return flat_index(indices, ctx.shapes.get(buf) if shape is None else shape).simplify(SimplifyCtx.empty())
    except ValueError as exc:
        raise ValueError(f"{buf}: {exc}") from exc


@dataclass(frozen=True)
class Flat:
    """One contiguous allocation — every buffer, unless it says otherwise."""

    name: str
    vectorizable: ClassVar[bool] = True

    def base(self, index: tuple, ctx: RenderCtx) -> str:
        del index, ctx
        return self.name

    def flat(self, index: tuple, ctx: RenderCtx) -> str:
        """The element index, printed as the statement gives it: a body a code generator reads holds one."""
        if len(index) != 1:
            raise ValueError(f"{self.name}: an access holds one flat index, got {len(index)} (address.flat_accesses writes it)")
        return render_expression(index[0], ctx)


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


def render_body(body: Body, ctx: RenderCtx) -> list[str]:
    """Flatten ``render_stmt`` over a body sequence."""
    out: list[str] = []
    for s in body:
        out.extend(render_stmt(s, ctx))
    return out


#: The suffix of a carried cell's next-step slot.
NEXT_STEP = "__next"


@singledispatch
def render_stmt(stmt: Statement, ctx: RenderCtx) -> list[str]:
    """Indented C / CUDA source lines for one statement, dispatched on its statement type.

    A block statement recurses through :func:`render_body` with ``ctx.child()``; a leaf returns its
    lines. ``ctx`` carries the target's intrinsic and builtin tables, the current indent, and the
    buffer shapes an index flattens over. Each layer registers the statement types it defines.
    """
    raise NotImplementedError(f"render not registered for {type(stmt).__name__}")


def _resolve_value(name: str, ctx: RenderCtx) -> str:
    """The text a statement that carries a raw SSA name (``Write.values``, ``Accum.value``) reads it by: the lane of a
    carried load's packed vector, else the name."""
    return ctx.lanes.get(name, name)


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
    src_dt = _declared(stmt, stmt.dtype)
    if stmt.input in ctx.pair_tables and not stmt.carried:
        # Each name's value is computed from its flat table index (``emmy_f4_pair``), exact in every float dtype.
        out = []
        for k, nm in enumerate(stmt.names):
            idx_k = (*stmt.index[:-1], BinaryExpression("+", stmt.index[-1], Literal(k, "int"))) if k else stmt.index
            value = f"emmy_f4_pair({ctx.memory_for(stmt.input).flat(idx_k, ctx)})"
            if src_dt != "f16":
                value = ctx.target.convert(ctx.target.convert(value, "f16", "f32"), "f32", src_dt)
            ctx.ssa_dtypes[nm] = src_dt
            out.append(f"{pad}{ctx.type_name(src_dt)} {nm} = {value};")
        return out
    mem = ctx.memory_for(stmt.input)
    n = stmt.width
    vec_pair = None if stmt.is_scalar else ctx.target.vector_type(src_dt, n)
    if vec_pair is None or not mem.vectorizable:
        # One local per name, declared at the load's dtype: a scalar Load, or a run the target cannot read as one
        # vector (no vector type for this dtype and width, or a paged buffer, whose run may cross a page). A carried
        # run's names are then its registers, which it re-assigns.
        decl = "" if stmt.carried else f"{ctx.type_name(src_dt)} "
        out: list[str] = []
        for k, nm in enumerate(stmt.names):
            idx_k = (*stmt.index[:-1], BinaryExpression("+", stmt.index[-1], Literal(k, "int"))) if k else stmt.index
            ctx.ssa_dtypes[nm] = src_dt
            out.append(f"{pad}{decl}{nm} = {mem.base(idx_k, ctx)}[{mem.flat(idx_k, ctx)}];")
        return out
    # Vector path: one ``<vec_type>`` reinterpret-cast read + N ``.x/.y/.z/.w`` (or indexed) unpacks.
    vec_type, elem_type = vec_pair
    flat = mem.flat(stmt.index, ctx)
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
    if stmt.carried:  # re-assign the declared vector, whose lanes the names are (see the class doc)
        return [f"{pad}{vname} = __ldg(reinterpret_cast<const {vec_type}*>(&{stmt.input}[{flat}]));"]
    out_lines = [f"{pad}{vec_type} {vname} = *reinterpret_cast<const {vec_type}*>(&{stmt.input}[{flat}]);"]
    arr_name = f"{vname}_h"
    if use_array_index:
        out_lines.append(f"{pad}const {elem_type}* {arr_name} = reinterpret_cast<const {elem_type}*>(&{vname});")
    lanes = [f"{arr_name}[{k}]" for k in range(n)] if use_array_index else [f"{vname}.{c}" for c in ("x", "y", "z", "w")[:n]]
    for nm, lane in zip(stmt.names, lanes, strict=True):
        ctx.ssa_dtypes[nm] = src_dt
        if nm in ctx.carried_names:
            ctx.lanes[nm] = lane  # no local: a later carried Load re-assigns the vector, and a read takes the lane
        else:
            out_lines.append(f"{pad}{elem_type} {nm} = {lane};")
    return out_lines


def _declared(stmt: Statement, dtype) -> str:
    """The dtype a typed body's statement declares; a body reaches the renderer through ``ir.types``."""
    if dtype is None:
        raise ValueError(f"render: {type(stmt).__name__} {stmt} carries no dtype; type the body first (ir.types.explicit_conversions)")
    return dtype.name


def _literal_at(node: Literal, dtype: str) -> Literal:
    """A literal operand of an operation at ``dtype``: it is spelled at that dtype."""
    return node if dtype in INTEGER_NAMES else Literal(float(node.value), "float")


@render_stmt.register
def _(stmt: Assign, ctx: RenderCtx) -> list[str]:
    """``name = value`` at the statement's dtype: one conversion, or one operation whose operands are all at that
    dtype. Nothing is decided here: the body says what converts and where each operation computes. An operation
    nested in the value (a node of its affine region) computes at the same dtype, as a temp declared ahead."""
    dtype = _declared(stmt, stmt.dtype)
    lines: list[str] = []
    temps = count()

    def operand(node: Expression) -> Expression:
        if isinstance(node, Literal):
            return _literal_at(node, dtype)
        if isinstance(node, Var):
            return node
        temp = f"{stmt.name}__e{next(temps)}"
        lines.extend(_render_node(temp, node, dtype, ctx, operand))
        return Var(temp)

    lines.extend(_render_node(stmt.name, stmt.value, dtype, ctx, operand))
    return lines


def _render_node(name: str, node: Expression, dtype: str, ctx: RenderCtx, operand: Callable[[Expression], Expression]) -> list[str]:
    """``name = node`` at ``dtype`` as one C statement, its operands through ``operand``."""
    ctx.ssa_dtypes[name] = dtype
    declaration = f"{_pad(ctx.indent)}{ctx.type_name(dtype)} {name} = "
    op = node.operation
    if isinstance(node, CastExpression) and node.dtype in _DTYPE_NAMES and isinstance(node.expression, Var):
        source = node.expression
        return [f"{declaration}{ctx.target.convert(render_expression(source, ctx), ctx.ssa_dtypes.get(source.name, 'f32'), dtype)};"]
    if op is None:
        # A name, a literal, a select, an index or a cast to a C type: one expression.
        return [f"{declaration}{render_expression(node, ctx)};"]
    if op.name == "bitcast":
        (source,) = node.operands
        if not isinstance(source, Var):
            raise ValueError("bitcast requires one named argument")
        return [f"{declaration}{ctx.target.bitcast(source.name, ctx.ssa_dtypes.get(source.name, 'f32'), dtype)};"]
    operands = [operand(child) for child in node.operands]
    expression = op_to_expression(op.name, operands, dtype=dtype)
    if dtype == "f32":
        return [f"{declaration}{render_expression(expression, ctx)};"]
    if isinstance(expression, BinaryExpression) and ctx.target.intrinsic(op.name, dtype) != op.name:
        # A target that spells this operator as a call (CUDA's non-contracting f16 ``_rn`` ops).
        expression = FunctionCallExpression(op.name, tuple(operands))
    # The target's spellings at this dtype: its intrinsics, and its literals (``__float2half(1.0f)``).
    saved = ctx.intrinsics, ctx.literal_default_dtype
    ctx.intrinsics = {**ctx.intrinsics, **_dtype_intrinsics(ctx.target, dtype, expression)}
    ctx.literal_default_dtype = dtype
    try:
        return [f"{declaration}{render_expression(expression, ctx)};"]
    finally:
        ctx.intrinsics, ctx.literal_default_dtype = saved


@render_stmt.register
def _(stmt: Accum, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    acc_dt = _declared(stmt, stmt.dtype)
    ctx.ssa_dtypes[stmt.name] = acc_dt
    # The folded value is a name at the accumulator's dtype, or a literal spelled at it.
    if isinstance(stmt.value, Literal):
        rhs = ctx.target.literal(render_expression(stmt.value, ctx), acc_dt)
    else:
        rhs = render_expression(stmt.value, ctx)
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
    return [f"{_pad(ctx.indent)}{stmt.name}{NEXT_STEP}[{render_index(stmt.name, stmt.index, ctx)}] = {stmt.value};"]


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
    floating = isinstance(stmt.value, Literal) and stmt.dtype.name not in INTEGER_NAMES
    value = ctx.identity_literal(stmt.value.value, stmt.dtype) if floating else render_expression(stmt.value, ctx)
    return [f"{_pad(ctx.indent)}{ctx.type_name(stmt.dtype)} {stmt.name} = {value};"]


def _store_index(stmt: Write, mem, ctx: RenderCtx, index: tuple[Expression, ...] | None = None) -> str:
    """The element index a store writes at, under its shared slab's physical layout. A Volta layout is a function of
    the logical row and column, which such a write keeps; every other write holds its flat index."""
    from emmy.compiler.ir.kernel.ir import TWO_DIMENSIONAL_LAYOUTS, smem_layout_index  # noqa: PLC0415 — kernel IR imports this module

    index = stmt.index if index is None else index
    flat = "" if stmt.swizzle in TWO_DIMENSIONAL_LAYOUTS else mem.flat(index, ctx)
    return smem_layout_index(stmt.swizzle, flat, stmt.output, ctx, index)


@render_stmt.register
def _(stmt: Write, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    # A typed body's write stores values of its buffer's element type: ``value_dtype`` is both.
    out_dt = _declared(stmt, stmt.value_dtype)
    mem = ctx.memory_for(stmt.output)
    if stmt.is_scalar:
        flat = _store_index(stmt, mem, ctx)
        rhs = _resolve_value(stmt.value, ctx)
        if stmt.atomic:
            # Cross-CTA additive finalize: every contributing CTA accumulates into the
            # same cell (the output is zero-init'd per launch — ``CudaOp.zero_outputs``).
            return [f"{pad}atomicAdd(&{mem.base(stmt.index, ctx)}[{flat}], {rhs});"]
        return [f"{pad}{mem.base(stmt.index, ctx)}[{flat}] = {rhs};"]
    n = stmt.width
    converted = [_resolve_value(nm, ctx) for nm in stmt.values]
    vec_pair = ctx.target.vector_type(out_dt, n)
    if vec_pair is None or not mem.vectorizable:
        # Target doesn't support this width, or the buffer does not guarantee consecutive
        # elements are contiguous (a paged store may cross a page) — fall back to scalar
        # writes. The vectorize pass should have avoided this, but
        # render's job is to always produce valid code.
        lines: list[str] = []
        for k in range(n):
            idx_k = tuple(stmt.index[:-1]) + (BinaryExpression("+", stmt.index[-1], Literal(k, "int")),)
            flat = _store_index(stmt, mem, ctx, idx_k)
            lines.append(f"{pad}{mem.base(idx_k, ctx)}[{flat}] = {converted[k]};")
        return lines
    vec_type, _elem_type = vec_pair
    flat = _store_index(stmt, mem, ctx)
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
        flat_hi = _store_index(stmt, mem, ctx, idx_hi)
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
    # Every branch of a typed body's select is at one dtype, which is the select's.
    dtype = ctx.ssa_dtypes.get(stmt.branches[0].value, "f32")
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
    if node.unrolled:
        out.append(f"{pad}{_ROLLED}")
    out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
    inner = ctx.child()
    out.extend(render_body(node.body, inner))
    out.extend(f"{_pad(inner.indent)}{name} = {name}{NEXT_STEP};" for name in node.carries)
    out.append(f"{pad}}}")
    return out


#: Ahead of a loop whose copies the kernel lowering wrote (``unrolled``): cicc keeps it as it is, as ptxas does under the
#: PTX compiler's ``.pragma "nounroll"``.
_ROLLED = "#pragma unroll 1"


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
    if stmt.unrolled:
        out.append(f"{pad}{_ROLLED}")
    # ``_extent_c`` renders a symbolic bound (a cooperative reduce over a dynamic
    # ``seq_len`` strides each lane to the runtime extent — the ``< seq_len`` bound is the
    # masked tail, idle lanes folding the carrier identity) as the C name, a static one as
    # its literal int.
    if isinstance(stmt.end, (Var, Literal)):
        # A bound that is already one value (the staged chunk loop binds its early stop ahead of the loop).
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {render_expression(stmt.end, ctx)}; {var} += {step_str}) {{")
    elif stmt.end is not None:
        # Hoist the override bound into the for-init (``<var>_end``) so a non-literal bound is
        # evaluated once, not per iteration (the inline form measurably spills on the flash
        # stream).
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
