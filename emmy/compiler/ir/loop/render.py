"""Host C++ for a Loop IR body: the render context, index flattening, and one render function per statement type.

The Loop IR runner (``runner``) JIT-compiles what this prints through cppyy as the float32 reference of a fused graph:
every float is an f32 (the runner coerces its buffers to float32), so a narrow float type is ``float`` and its
conversions are the identity. ``render_stmt`` dispatches on the statement type; ``render_body`` is the entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import singledispatch
from itertools import count

from emmy.compiler.dtype import F32, INTEGER_NAMES
from emmy.compiler.ir.address import flat_index
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
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Cond, Init, Let, Load, Loop, Pre, Select, Statement, StridedLoop, Write
from emmy.emmy_ir import DTYPES as _DTYPE_NAMES

_FLOAT_DTYPES = frozenset({"f32", "f16", "f16x2", "f8e4m3", "f8e5m2"})
_TYPE_NAME = {
    **{dtype: "float" for dtype in _FLOAT_DTYPES},
    "i16": "short",
    "i32": "int",
    "i64": "long long",
    "u8": "unsigned char",
    "u16": "unsigned short",
    "u32": "unsigned int",
    "u64": "unsigned long long",
    "bool": "bool",
}
_HOST_NBYTES = {
    "float": 4,
    "short": 2,
    "int": 4,
    "long long": 8,
    "unsigned char": 1,
    "unsigned short": 2,
    "unsigned int": 4,
    "unsigned long long": 8,
    "bool": 1,
}


def type_name(dtype) -> str:
    """The host C type of ``dtype`` (a :class:`DataType`, a canonical name, or ``None`` for f32): every float is a
    ``float``."""
    return _TYPE_NAME.get(_canonical_dtype_name(dtype), "float")


def convert(value: str, src_dt: str, dst_dt: str) -> str:
    """``value`` converted from ``src_dt`` to ``dst_dt``: a cast between two host types, nothing between floats."""
    src_type, dst_type = type_name(src_dt), type_name(dst_dt)
    return value if src_type == dst_type else f"static_cast<{dst_type}>({value})"


def bitcast(value: str, src_dt: str, dst_dt: str) -> str:
    """``value``'s bits read as ``dst_dt``: the two host types are one width."""
    if src_dt == dst_dt:
        return value
    src_type, dst_type = type_name(src_dt), type_name(dst_dt)
    if _HOST_NBYTES[src_type] != _HOST_NBYTES[dst_type]:
        raise ValueError(f"Loop float32 reference cannot bitcast {src_dt} to {dst_dt}")
    return f"emmy_bitcast<{dst_type}>({value})"


def intrinsic(op_name: str, result_dt: str) -> str:
    """The host function an operation at ``result_dt`` calls, where it differs from the operation's name."""
    if op_name == "bitwise_count":
        return "__builtin_popcountll" if result_dt in ("i64", "u64") else "__builtin_popcount"
    return op_name


@dataclass
class RenderCtx:
    """Per-render state. ``intrinsics`` / ``builtins`` map an abstract function or builtin name to its spelling.
    ``shapes`` maps every buffer to its declared shape, for the statements that hold one index per dimension.
    ``ssa_dtypes`` records the dtype each name was declared at as ``render_body`` walks: the source dtype a
    conversion reads."""

    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    indent: int = 1
    intrinsics: dict[str, str] = field(default_factory=dict)
    builtins: dict[str, str] = field(default_factory=dict)
    ssa_dtypes: dict[str, str] = field(default_factory=dict)

    def child(self) -> RenderCtx:
        """A ctx one indent level deeper; the tables stay shared (SSA names are globally unique)."""
        return replace(self, indent=self.indent + 1)


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


#: The operations spelled as a call of one argument: a math function, or the approximate form of one.
_UNARY_CALLS = frozenset(
    {"exp", "log", "rsqrt", "sin", "cos", "tanh", "sqrt", "erf", "floor"}
    | {"exp_fast", "log_fast", "sin_fast", "cos_fast", "tanh_fast", "sqrt_fast", "reciprocal_fast"}
)


def op_to_expression(fn: str, inputs: list[Expression], *, dtype: str | None = None) -> Expression:
    """Translate an elementwise op name to an ``Expression`` tree.

    Emits abstract intrinsic names (``"exp"``, ``"fmax"``, ``"fabs"``, ...) that the runner translates to libm
    spellings via ``RenderCtx.intrinsics``. ``dtype`` distinguishes integer bit manipulation from the traced mask
    convention: integer ``bitwise_and``/``bitwise_or`` render as ``&``/``|``, while the f32 mask path is logical
    ``&&``/``||``.
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
    if fn in ("copy", "pad", "from_f8e4m3", "from_f8e5m2", "to_f8e4m3", "to_f8e5m2"):
        # An fp8 cast is the conversion the typed body spells ahead of it, the identity on the host's floats.
        return inputs[0]
    if fn == "to_f4e2m1":
        # e2m1 encode — NOT the fp8 encodes' identity: the codes land on an ordinary integer carrier, where the
        # identity would truncate the value instead of rounding it.
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
        # np.where(cond, a, b) — the mask-apply tail of the explicit-mask subgraph. A float or truth-value cond is
        # C-truthy (nonzero) in ``?:``.
        return TernaryExpression(inputs[0], inputs[1], inputs[2])
    if fn == "bitwise_not":
        # Torch spells ``~mask`` as bitwise_not even when ``mask`` is bool. Loop IR promotes boolean mask values to
        # f32, so both the f32 convention and an explicitly bool-stamped value mean logical-not. Never infer integer
        # complement here: that requires a separately typed and tested ``~`` path.
        if dtype not in (None, "f32", "bool"):
            raise NotImplementedError(f"render: bitwise_not supports boolean masks only, got dtype={dtype!r}")
        if len(inputs) != 1:
            raise ValueError(f"render: bitwise_not requires one input, got {len(inputs)}")
        zero = Literal(0.0, "float") if dtype in (None, "f32") else Literal(0, "int")
        return BinaryExpression("==", inputs[0], zero)
    raise NotImplementedError(f"render: elementwise fn={fn!r} not supported")


def select_to_ternary(s: Select, dtype: str = "float") -> Expression:
    """Cast branches to their common result type."""
    branches = list(s.branches)
    result: Expression = CastExpression(dtype, Var(branches[-1].value))
    for b in reversed(branches[:-1]):
        result = TernaryExpression(cond=b.select, if_true=CastExpression(dtype, Var(b.value)), if_false=result)
    return result


def _float_lit(v: float) -> str:
    """C float literal — always carries a decimal so ``0.0`` renders as ``0.0f`` not the invalid ``0f``, and
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
    """Emit a C expression string. ``ctx`` carries the intrinsic and builtin spelling tables; ``parent_prec`` drives
    parenthesization in nested binary chains."""
    if isinstance(expression, Var):
        return expression.name
    if isinstance(expression, Literal):
        if isinstance(expression.value, float) or expression.dtype == "float":
            return _float_lit(float(expression.value))
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
        spelling = ctx.builtins.get(expression.name)
        if spelling is None:
            raise ValueError(f"render: Builtin {expression.name!r} has no target spelling in ctx.builtins")
        return spelling
    if isinstance(expression, FunctionCallExpression):
        # The table's spellings an index or a predicate carries: a negation is the prefix, a minimum or maximum the
        # C++ ``std::min`` / ``std::max``.
        if expression.name == "negative" and len(expression.args) == 1:
            return f"(-{render_expression(expression.args[0], ctx, ATOM_PRECEDENCE)})"
        name = _INDEX_CALLS.get(expression.name, expression.name)
        spelling = ctx.intrinsics.get(name, name)
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


def render_index(buf: str, indices: tuple, ctx: RenderCtx) -> str:
    """The flat element index of ``buf[indices]`` as a C expression: the row-major sum (``address.flat_index``) over
    the buffer's declared shape, simplified, so constant-zero indices (typical of size-1 outer dims) drop out. One
    index is flat already and prints as it is given."""
    if len(indices) == 1:
        return render_expression(indices[0], ctx)
    try:
        flat = flat_index(indices, ctx.shapes.get(buf)).simplify(SimplifyCtx.empty())
    except ValueError as exc:
        raise ValueError(f"{buf}: {exc}") from exc
    return render_expression(flat, ctx)


def _flat(buf: str, index: tuple, ctx: RenderCtx) -> str:
    """The element index of an access, as the statement gives it: a body the renderer reads holds one."""
    if len(index) != 1:
        raise ValueError(f"{buf}: an access holds one flat index, got {len(index)} (address.flat_accesses writes it)")
    return render_expression(index[0], ctx)


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
    """Indented C++ source lines for one statement, dispatched on its statement type.

    A block statement recurses through :func:`render_body` with ``ctx.child()``; a leaf returns its lines.
    """
    raise NotImplementedError(f"render not registered for {type(stmt).__name__}")


def _dtype_intrinsics(result_dt: str, expression: Expression) -> dict[str, str]:
    """The spelling at ``result_dt`` of every function ``expression`` calls."""
    overrides: dict[str, str] = {}

    def collect(node):
        if isinstance(node, FunctionCallExpression):
            overrides[node.name] = intrinsic(node.name, result_dt)
            for a in node.args:
                collect(a)
        else:
            for field_name in ("left", "right", "cond", "if_true", "if_false", "expression"):
                child = getattr(node, field_name, None)
                if child is not None:
                    collect(child)

    collect(expression)
    return overrides


def _declared(stmt: Statement, dtype) -> str:
    """The dtype a typed body's statement declares; a body reaches the renderer through ``ir.types``."""
    if dtype is None:
        raise ValueError(f"render: {type(stmt).__name__} {stmt} carries no dtype; type the body first (ir.types.explicit_conversions)")
    return dtype.name


@render_stmt.register
def _(stmt: Load, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    src_dt = _declared(stmt, stmt.dtype)
    out: list[str] = []
    for k, nm in enumerate(stmt.names):
        idx_k = (*stmt.index[:-1], BinaryExpression("+", stmt.index[-1], Literal(k, "int"))) if k else stmt.index
        ctx.ssa_dtypes[nm] = src_dt
        out.append(f"{pad}{type_name(src_dt)} {nm} = {stmt.input}[{_flat(stmt.input, idx_k, ctx)}];")
    return out


def _literal_at(node: Literal, dtype: str) -> Literal:
    """A literal operand of an operation at ``dtype``: it is spelled at that dtype."""
    return node if dtype in INTEGER_NAMES else Literal(float(node.value), "float")


@render_stmt.register
def _(stmt: Assign, ctx: RenderCtx) -> list[str]:
    """``name = value`` at the statement's dtype: one conversion, or one operation whose operands are all at that
    dtype. An operation nested in the value (a node of its affine region) computes at the same dtype, as a temp
    declared ahead."""
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
    declaration = f"{_pad(ctx.indent)}{type_name(dtype)} {name} = "
    op = node.operation
    if isinstance(node, CastExpression) and node.dtype in _DTYPE_NAMES and isinstance(node.expression, Var):
        source = node.expression
        return [f"{declaration}{convert(render_expression(source, ctx), ctx.ssa_dtypes.get(source.name, 'f32'), dtype)};"]
    if op is None:
        # A name, a literal, a select, an index or a cast to a C type: one expression.
        return [f"{declaration}{render_expression(node, ctx)};"]
    if op.name == "bitcast":
        (source,) = node.operands
        if not isinstance(source, Var):
            raise ValueError("bitcast requires one named argument")
        return [f"{declaration}{bitcast(source.name, ctx.ssa_dtypes.get(source.name, 'f32'), dtype)};"]
    operands = [operand(child) for child in node.operands]
    expression = op_to_expression(op.name, operands, dtype=dtype)
    if dtype == "f32":
        return [f"{declaration}{render_expression(expression, ctx)};"]
    saved = ctx.intrinsics
    ctx.intrinsics = {**ctx.intrinsics, **_dtype_intrinsics(dtype, expression)}
    try:
        return [f"{declaration}{render_expression(expression, ctx)};"]
    finally:
        ctx.intrinsics = saved


@render_stmt.register
def _(stmt: Accum, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    acc_dt = _declared(stmt, stmt.dtype)
    ctx.ssa_dtypes[stmt.name] = acc_dt
    rhs = render_expression(stmt.value, ctx)
    # Left operand of the fold: ``base`` (a rescaled state, already at acc_dt) when set, else the carried ``name``.
    base = stmt.base if stmt.base is not None else stmt.name
    # Spelling (``+=`` / ``*=`` / ``fmax`` / ``fmin``) from the shared reduce registry.
    spelling = reduce_spelling(stmt.op)
    if spelling.intrinsic is not None:
        fn = intrinsic(spelling.intrinsic, acc_dt)
        return [f"{pad}{stmt.name} = {fn}({base}, {rhs});"]
    if base == stmt.name:
        return [f"{pad}{stmt.name} {spelling.compound} {rhs};"]
    return [f"{pad}{stmt.name} = {base} {spelling.infix} {rhs};"]


@render_stmt.register
def _(stmt: Carry, ctx: RenderCtx) -> list[str]:
    # The carrying ``Loop`` declared this slot beside the one reads see, and commits it after every step.
    return [f"{_pad(ctx.indent)}{stmt.name}{NEXT_STEP}[{render_index(stmt.name, stmt.index, ctx)}] = {stmt.value};"]


@render_stmt.register
def _(stmt: Pre, ctx: RenderCtx) -> list[str]:
    dtype = ctx.ssa_dtypes.get(stmt.carrier, "f32")
    ctx.ssa_dtypes[stmt.name] = dtype
    return [f"{_pad(ctx.indent)}{type_name(dtype)} {stmt.name} = {stmt.carrier}[{render_index(stmt.carrier, stmt.index, ctx)}];"]


@render_stmt.register
def _(stmt: Init, ctx: RenderCtx) -> list[str]:
    ctx.ssa_dtypes[stmt.name] = stmt.dtype.name
    return [f"{_pad(ctx.indent)}{type_name(stmt.dtype)} {stmt.name} = {_float_lit(float(stmt.identity))};"]


@render_stmt.register
def _(stmt: Let, ctx: RenderCtx) -> list[str]:
    if stmt.dtype is None:
        return [f"{_pad(ctx.indent)}auto {stmt.name} = {render_expression(stmt.value, ctx)};"]
    ctx.ssa_dtypes[stmt.name] = stmt.dtype.name
    floating = isinstance(stmt.value, Literal) and stmt.dtype.name not in INTEGER_NAMES
    value = _float_lit(float(stmt.value.value)) if floating else render_expression(stmt.value, ctx)
    return [f"{_pad(ctx.indent)}{type_name(stmt.dtype)} {stmt.name} = {value};"]


@render_stmt.register
def _(stmt: Write, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    if stmt.atomic or stmt.swizzle != "NONE":
        raise NotImplementedError(f"render: a host write is neither atomic nor swizzled, got {stmt}")
    lines: list[str] = []
    for k, value in enumerate(stmt.values):
        idx_k = (*stmt.index[:-1], BinaryExpression("+", stmt.index[-1], Literal(k, "int"))) if k else stmt.index
        lines.append(f"{pad}{stmt.output}[{_flat(stmt.output, idx_k, ctx)}] = {value};")
    return lines


@render_stmt.register
def _(stmt: Select, ctx: RenderCtx) -> list[str]:
    # Every branch of a typed body's select is at one dtype, which is the select's.
    dtype = ctx.ssa_dtypes.get(stmt.branches[0].value, "f32")
    ctx.ssa_dtypes[stmt.name] = dtype
    ty = type_name(dtype)
    return [f"{_pad(ctx.indent)}{ty} {stmt.name} = {render_expression(select_to_ternary(stmt, ty), ctx)};"]


@render_stmt.register
def _(node: Loop, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    out: list[str] = []
    # A carried state is two slots of its cell shape: the one the previous step left, which every read sees, and the
    # one this step defines, committed when the step ends.
    for name, shape in node.carries.items():
        carry = next(c for c in node.body.carries if c.name == name)
        extents = tuple(1 if isinstance(d, int) else int(d.extent.as_static()) for d in shape)
        size = 1
        for extent in extents:
            size *= extent
        # The seed is a constant in every cell or a buffer of the state's shape, copied whole.
        seed = f"{carry.seed}, {carry.seed} + {size}" if isinstance(carry.seed, str) else f"{size}, {_float_lit(float(carry.seed))}"
        for slot in (name, f"{name}{NEXT_STEP}"):
            ctx.shapes[slot] = extents
            out.append(f"{pad}std::vector<{type_name(carry.dtype)}> {slot}({seed});")
        ctx.ssa_dtypes[name] = (carry.dtype or F32).name
    out.extend(_seeds(node, ctx))
    var = node.axis.name
    out.append(f"{pad}for (int {var} = 0; {var} < {_extent_c(node, ctx)}; {var}++) {{")
    inner = ctx.child()
    out.extend(render_body(node.body, inner))
    out.extend(f"{_pad(inner.indent)}{name} = {name}{NEXT_STEP};" for name in node.carries)
    out.append(f"{pad}}}")
    return out


def _seeds(node: Loop | StridedLoop, ctx: RenderCtx) -> list[str]:
    """``<dtype> <acc> = identity;`` for each distinct ``Accum`` fold in the loop's immediate body: the seed rides on
    the fold, derived from ``op.identity``. A nested fold re-declares per enclosing iteration."""
    out: list[str] = []
    seen: set[str] = set()
    for s in node.body:
        if isinstance(s, Accum) and s.name not in seen:
            seen.add(s.name)
            identity = s.op.identity
            if identity is None:
                raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
            if node.seed:
                out.append(f"{_pad(ctx.indent)}{type_name(s.dtype)} {s.name} = {_float_lit(float(identity))};")
            ctx.ssa_dtypes[s.name] = (s.dtype or F32).name
    return out


@render_stmt.register
def _(stmt: StridedLoop, ctx: RenderCtx) -> list[str]:
    """``for (int axis = start; axis < bound; axis += step)``, the bound the loop's own end where it has one."""
    out = _seeds(stmt, ctx)
    var = stmt.axis.name
    bound = _extent_c(stmt, ctx) if stmt.end is None else f"({render_expression(stmt.end, ctx)})"
    start, step = render_expression(stmt.start, ctx), render_expression(stmt.step, ctx)
    out.append(f"{_pad(ctx.indent)}for (int {var} = {start}; {var} < {bound}; {var} += {step}) {{")
    out.extend(render_body(stmt.body, ctx.child()))
    out.append(f"{_pad(ctx.indent)}}}")
    return out


def _extent_c(node: Loop | StridedLoop, ctx: RenderCtx) -> str:
    """A loop's extent as a C expression: the int of a static one, the name of a symbolic one, parenthesized where
    it is a composite (``(seq_len+31)//32``), whose ``//`` the expression renderer spells."""
    ext = node.axis.extent
    if ext.is_static or isinstance(ext.expression, Var):
        return str(ext)
    return f"({render_expression(ext.expression, ctx)})"


@render_stmt.register
def _(stmt: Cond, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    inner = ctx.child()
    out = [f"{pad}if ({render_expression(stmt.cond, ctx)}) {{", *render_body(stmt.body, inner), f"{pad}}}"]
    if stmt.else_body:
        out[-1] = f"{pad}}} else {{"
        out.extend(render_body(stmt.else_body, inner))
        out.append(f"{pad}}}")
    return out
