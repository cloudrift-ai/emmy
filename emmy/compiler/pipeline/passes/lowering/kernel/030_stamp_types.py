"""Stamp per-statement dtypes on every Load / Assign / Write.

Runs over a materialized ``KernelOp`` body, after ``010_materialize`` and
before the analytical codegen passes (demote, vectorize) see the IR. After
this pass those passes read dtypes directly off the IR instead of
re-deriving them from the op's ``inputs`` / ``outputs`` side channels.

Stamping rules:

- ``Load(input=B)`` — ``dtype = B``'s buffer dtype, read off the op's
  ``inputs`` / ``outputs`` (matcher-populated graph Tensors). A Load against
  a locally-declared buffer (smem) keeps the render-time fallback.
- ``Assign(value)`` — the promotion of the value's named operands node by node (``promoted_dtype``; a literal
  contributes no dtype, so ``2 * x`` computes at ``x``'s), the same rule ``Assign``'s render applies. A square
  ``x * x`` and ``pow`` are promoted to f32 (overflow guard, below).
- ``Write(output, value)`` — ``value_dtype = ssa[value]``. The destination
  buffer dtype stays a render-time concern; only the value side is stamped.
- ``Accum`` / ``Init`` are already typed — register
  them in the running ``ssa_dtypes`` so downstream Assigns / Writes pick up
  the right arg dtype.

Idempotent: ``None``-stamped fields are filled and existing stamps are kept, except integer algebra whose result dtype
is recovered from typed operands. This repairs stale float stamps retained by structurally cloned, previously untyped
bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from emmy.compiler.dtype import F16, F32, DataType
from emmy.compiler.dtype import get as dtype_get
from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import Expression, Literal, promoted_dtype
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Let, Load, Select, Statement, Write, dtype_promote
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


@dataclass
class _StampCtx:
    """Per-walk state. ``buf_dtypes`` maps an op I/O buffer name to its
    dtype; ``ssa_dtypes`` accumulates as Stmts are stamped."""

    buf_dtypes: dict[str, DataType]
    ssa_dtypes: dict[str, DataType] = field(default_factory=dict)


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    buf_dtypes = {n: t.dtype for n, t in {**op.inputs, **op.outputs}.items()}
    new_body = op.body
    while True:
        ctx = _StampCtx(buf_dtypes=buf_dtypes)
        _seed_explicit_dtypes(new_body, ctx)
        stamped = _stamp_body(new_body, ctx)
        if stamped == new_body:
            break
        new_body = stamped
    if new_body == op.body:
        raise RuleSkipped("every Load/Assign/Write already stamped")
    return replace(op, body=new_body, knobs=dict(op.knobs))


def _seed_explicit_dtypes(body: Body, ctx: _StampCtx) -> None:
    """Register explicit SSA types before inference.

    Materialization can place a typed definition in a sibling body visited after one of its uses. Each inference walk
    still performs promotion in execution order, but no longer mistakes that known integer input for f32. The caller
    repeats the walk until inferred upstream types have propagated through every consumer.
    """
    for s in body:
        if isinstance(s, Load) and s.dtype is not None:
            ctx.ssa_dtypes.update((name, s.dtype) for name in s.names)
        elif isinstance(s, (Assign, Accum, Init, Let)) and s.dtype is not None:
            ctx.ssa_dtypes[s.name] = s.dtype
        for nested in s.nested():
            _seed_explicit_dtypes(nested, ctx)


def _stamp_body(body: Body, ctx: _StampCtx) -> Body:
    return Body(_stamp_stmt(s, ctx) for s in body)


def _stamp_stmt(s: Statement, ctx: _StampCtx) -> Statement:
    if isinstance(s, Load):
        return _stamp_load(s, ctx)
    if isinstance(s, Assign):
        return _stamp_assign(s, ctx)
    if isinstance(s, Select):
        ctx.ssa_dtypes[s.name] = dtype_get(dtype_promote("add", [(ctx.ssa_dtypes.get(b.value) or F32).name for b in s.branches]))
        return s
    if isinstance(s, Write):
        return _stamp_write(s, ctx)
    if isinstance(s, Accum):
        return _stamp_accum(s, ctx)
    if isinstance(s, Init):
        ctx.ssa_dtypes[s.name] = s.dtype or F32
        return s
    if isinstance(s, Let):
        # A float literal takes f32 when nothing narrows it; an index keeps its expression's type.
        if s.dtype is None and isinstance(s.value, Literal) and s.value.dtype == "float":
            s = s.replace(dtype=F32)
        if s.dtype is not None:
            ctx.ssa_dtypes[s.name] = s.dtype
        return s
    # Block-structured stmts (Tile / Loop / StridedLoop / Cond, …):
    # recurse through children via the generic ``nested()`` / ``with_bodies()``
    # protocol — no isinstance ladder.
    nested = s.nested()
    if not nested:
        return s
    return s.with_bodies(tuple(_stamp_body(b, ctx) for b in nested))


def _stamp_accum(s: Accum, ctx: _StampCtx) -> Accum:
    """Freeze the accumulator dtype — the old ``020_place_inits`` policy, applied to the
    ``Accum`` itself (the carrier is the source of truth; ``Loop``'s render derives the seed
    from it). A **selecting** combine (``max`` / ``min``) picks an existing value, so it
    stays in the folded value's dtype — fp16 ``max`` accumulates in fp16. An
    **accumulating** combine (``sum`` / ``prod`` — incl. the matmul fold) builds magnitude,
    so it promotes fp16 to f32 to avoid precision loss / overflow. A fold into a carrier already
    declared (a register copy's partial merged into copy 0) keeps the carrier's dtype, whatever
    the partial's: a masked copy's partial is f32, and the carrier's later readers read it at its
    declaration's dtype. Already-stamped Accums keep their dtype."""
    if s.dtype is not None:
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    value_dt = dtype_get(promoted_dtype(s.value, lambda name: _stamped(ctx, name)))
    dt = ctx.ssa_dtypes.get(s.name) or (value_dt if s.op.selecting else F32)
    ctx.ssa_dtypes[s.name] = dt
    return s.replace(dtype=dt)


def _stamp_load(s: Load, ctx: _StampCtx) -> Load:
    dt = s.dtype if s.dtype is not None else ctx.buf_dtypes.get(s.input)
    if dt is not None:
        for n in s.names:
            ctx.ssa_dtypes[n] = dt
        if s.dtype is None:
            return s.replace(dtype=dt)
    return s


def _stamped(ctx: _StampCtx, name: str) -> str | None:
    dt = ctx.ssa_dtypes.get(name)
    return None if dt is None else dt.name


def _stamp_assign(s: Assign, ctx: _StampCtx) -> Assign:
    result_dt = dtype_get(promoted_dtype(s.value, lambda name: _stamped(ctx, name)))
    if s.dtype is not None and result_dt.np.kind not in "iu":
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    # Integer algebra is determined by its operands. Structural lowering can carry an old f32
    # stamp from a once-untyped body; keeping it would turn shifts into unsupported float ops.
    # Explicit casts use ``copy`` / ``bitcast``, whose inferred result is not integer and therefore
    # retain their declared dtype above.
    # Overflow guard: a square (``x * x``) or a ``pow`` of an fp16 value can
    # blow past fp16's 65504 ceiling, giving inf → a garbage reduction
    # (RMSNorm's mean-of-squares). torch computes that reduction in fp32; do
    # the same here. Matmul products already carry an explicit compute dtype;
    # this fallback promotes only an untyped same-arg square / pow.
    if result_dt == F16 and _is_overflow_prone(s.value):
        result_dt = F32
    ctx.ssa_dtypes[s.name] = result_dt
    return s.replace(dtype=result_dt) if s.dtype != result_dt else s


def _is_overflow_prone(value: Expression) -> bool:
    """True when the value holds an operation that can overflow fp16 from in-range inputs — a square ``x * x`` or any
    ``pow``. Distinct-operand products are excluded."""
    for node in value.subterms():
        op = node.operation
        if op is None:
            continue
        if op.name == "pow" or (op.semiring_product and len(node.operands) == 2 and node.operands[0] == node.operands[1]):
            return True
    return False


def _stamp_write(s: Write, ctx: _StampCtx) -> Write:
    if s.value_dtype is not None:
        return s
    # In a multi-value (vector) Write all values share the same SSA dtype by
    # construction (vectorize_stores widens runs of same-dtype values).
    dt = ctx.ssa_dtypes.get(s.values[0])
    if dt is None:
        return s
    return Write(output=s.output, index=s.index, values=s.values, value_dtype=dt, atomic=s.atomic, swizzle=s.swizzle)
