"""The types of a statement body: the dtype every definition computes at, and every conversion a definition of its own.

Two steps, both pure ``Body -> Body`` and deciding nothing:

- :func:`stamp_types` gives each ``Load`` / ``Assign`` / ``Accum`` / ``Write`` its dtype by the promotion rule. Kernel
  lowering runs it early (``030_stamp_types``), so the passes after it read dtypes off the statements.
- :func:`explicit_conversions` leaves a code generator nothing to work out. Every operation takes its operands at the
  dtype it computes at and yields that dtype; a value read at another dtype is converted by a definition of its own,
  ``name = (dtype)source``; a write stores a value of its buffer's element type; a comparison defines a truth value
  (``bool``), converted once where it is declared another dtype. Which dtype an operation computes at is
  decided here and nowhere else: natively where the target has the operation at the result's dtype and every operand
  is already there, over integers where every operand is one, else in f32 with the result converted once.

The CUDA renderer and the Loop IR runner print what the body then says. ``check_types`` (the crate) is the property
as a check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from emmy.compiler.dtype import BOOL, F16, F32, I32, INTEGER_NAMES, DataType
from emmy.compiler.dtype import get as dtype_get
from emmy.compiler.ir.expression import CastExpression, Expression, Literal, TernaryExpression, Var, promoted_dtype
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Init, Let, Load, Pre, Select, SelectBranch, Statement, Write, dtype_promote

#: The operations an integer dtype computes natively. Any other operation over integers computes in f32.
INTEGER_OPS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "floor_divide",
        "remainder",
        "mod",
        "left_shift",
        "right_shift",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_count",
        "negative",
    }
)

#: The comparisons: whatever their operands compute at, each yields a truth value.
_COMPARISONS = frozenset({"equal", "not_equal", "greater", "less", "greater_equal", "less_equal"})

#: The operations with no integer form that C spells as an operator, so over integer operands alone the generated code
#: computes them in integers (a comparison of two indices, an index quotient) and converts the result.
_INTEGER_OPERATORS = _COMPARISONS | {"divide", "where"}


#: The operations that yield their operand: a copy, and the fp8 casts, whose value is the conversion of the operand.
_IDENTITIES = frozenset({"copy", "pad", "from_f8e4m3", "from_f8e5m2", "to_f8e4m3", "to_f8e5m2"})


@dataclass
class _StampCtx:
    """Per-walk state. ``buf_dtypes`` maps an op I/O buffer name to its
    dtype; ``ssa_dtypes`` accumulates as Stmts are stamped."""

    buf_dtypes: dict[str, DataType]
    ssa_dtypes: dict[str, DataType] = field(default_factory=dict)


def stamp_types(body: Body, buf_dtypes: dict[str, DataType], known: dict[str, DataType] | None = None) -> Body:
    """``body`` with the dtype of every ``Load`` (of a buffer in ``buf_dtypes``), ``Assign``, ``Accum`` and ``Write``
    stamped. A stamp already there is kept: a declared dtype is the result type.

    - ``Load`` takes its buffer's dtype.
    - ``Assign`` takes the promotion of its value's named operands node by node (``promoted_dtype``; a literal
      contributes no dtype, so ``2 * x`` computes at ``x``'s). A square ``x * x`` and ``pow`` of f16 take f32.
    - ``Accum`` takes its value's dtype when its combine selects (``max`` / ``min``), else f32.
    - ``Write`` takes its value's dtype.

    ``known`` types the names the body reads and something else defines.
    """
    while True:
        ctx = _StampCtx(buf_dtypes=buf_dtypes, ssa_dtypes=dict(known or {}))
        _seed_explicit_dtypes(body, ctx)
        stamped = _stamp_body(body, ctx)
        if stamped == body:
            return body
        body = stamped


def value_dtypes(body: Body) -> dict[str, DataType]:
    """The dtype of every name ``body`` types: the declared ones, and those the promotion rule gives the rest."""
    ctx = _StampCtx({})
    _seed_explicit_dtypes(body, ctx)
    _stamp_body(body, ctx)
    return ctx.ssa_dtypes


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
    if s.dtype is not None:
        # A declared dtype is the result type: a width declared on integer algebra and a float declared over integer
        # operands both stay, and ``explicit_conversions`` converts the operation's result once.
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    result_dt = dtype_get(promoted_dtype(s.value, lambda name: _stamped(ctx, name)))
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


def explicit_conversions(
    body: Body, buf_dtypes: dict[str, DataType], native_f16: frozenset[str], known: dict[str, DataType] | None = None
) -> Body:
    """``body`` typed in full (the module doc). ``buf_dtypes`` holds every buffer's element dtype, f32 where absent;
    ``native_f16`` the operations the target computes at f16; ``known`` types the names the body reads and something
    else defines."""
    body = stamp_types(body, buf_dtypes, known)
    seed = _StampCtx(buf_dtypes=buf_dtypes, ssa_dtypes=dict(known or {}))
    _seed_explicit_dtypes(body, seed)
    states: set[str] = set()
    taken: set[str] = set()
    defined: set[str] = set(known or {})
    for s in body.iter():
        defined.update(s.defines())
        taken.update(s.defines())
        taken.update(s.deps())
        if isinstance(s, Let) and s.dtype is None:
            seed.ssa_dtypes[s.name] = I32  # an index
        if isinstance(s, (Accum, Carry)) or (isinstance(s, Load) and s.carried):
            states.update(s.defines())
        if isinstance(s, Carry):
            seed.ssa_dtypes[s.name] = s.dtype or F32
    return _Conversions(buf_dtypes, native_f16, seed.ssa_dtypes, states, taken, defined).body(body, {})


def epilogue_conversions(epilogue: Body, params: tuple[str, ...], buf_dtypes: dict[str, DataType], native_f16: frozenset[str]) -> Body:
    """A ``RegStore`` epilogue typed in full. It computes on the f32 elements of the accumulators ``params`` name: a
    load that declares no dtype reads its buffer's element type, widened to f32 by a conversion of its own when that is
    a narrower float, and a select is f32, every branch at another dtype converted ahead of it. Then the epilogue is
    typed as any body (:func:`explicit_conversions`). Typing it again changes nothing."""
    known = dict.fromkeys(params, F32)
    taken = {name for s in epilogue.iter() for name in (*s.defines(), *s.deps())} | set(params)

    def narrow(name: str, dtype: DataType) -> str:
        """A fresh name for the value ``name`` holds at ``dtype`` before it is widened."""
        fresh, index = f"{name}__{dtype.name}", 0
        while fresh in taken:
            index += 1
            fresh = f"{name}__{dtype.name}_{index}"
        taken.add(fresh)
        return fresh

    out: list[Statement] = []
    for s in epilogue:
        if isinstance(s, Load) and s.dtype is None:
            stored = buf_dtypes.get(s.input, F32)
            if stored == F32 or stored.name in INTEGER_NAMES:
                out.append(s.replace(dtype=stored))
                continue
            loaded = [narrow(name, stored) for name in s.names]
            out.append(s.replace(names=tuple(loaded), dtype=stored))
            out.extend(Assign(name, CastExpression(F32.name, Var(held)), F32) for name, held in zip(s.names, loaded, strict=True))
        elif isinstance(s, Select):
            types = {**value_dtypes(stamp_types(Body(out), buf_dtypes, known)), **known}
            branches = []
            for branch in s.branches:
                dtype = types.get(branch.value, I32)
                if dtype != F32:
                    widened = narrow(branch.value, F32)
                    out.append(Assign(widened, CastExpression(F32.name, Var(branch.value)), F32))
                    branch = SelectBranch(widened, branch.select)
                branches.append(branch)
            out.append(s.replace(branches=tuple(branches)))
        else:
            out.append(s)
    return explicit_conversions(Body(out), buf_dtypes, native_f16, known)


@dataclass
class _Conversions:
    buf_dtypes: dict[str, DataType]
    native_f16: frozenset[str]
    #: The dtype of every name defined so far, seeded with the declared ones.
    types: dict[str, DataType]
    #: The names a later statement defines again (an accumulator, a carried state): a conversion of one is never reused.
    states: set[str]
    taken: set[str]
    #: The names some statement of the body defines. One with no declared dtype is f32 (a register statement's
    #: scalar); a name the body only reads is an integer: an axis, a symbolic size, a thread coordinate.
    defined: set[str]

    def body(self, body: Body, conversions: dict[tuple[str, str], str]) -> Body:
        """``body`` converted. ``conversions`` are the conversions in scope, ``(source, dtype) -> name``: a scope reads
        the ones of the scopes around it, and its own end with it."""
        conversions = dict(conversions)
        out: list[Statement] = []
        for s in body:
            self.statement(s, conversions, out)
        return Body(out)

    def dtype(self, name: str) -> DataType:
        return self.types.get(name) or (F32 if name in self.defined else I32)

    def fresh(self, base: str, first: str = "") -> str:
        """A name no statement of the body holds: ``base`` and ``first``, then ``base`` and a number."""
        name, index = f"{base}{first}", 0
        while name in self.taken:
            index += 1
            name = f"{base}{'_' if first == '' else ''}{index}"
        self.taken.add(name)
        return name

    def at(self, name: str, dtype: DataType, conversions: dict[tuple[str, str], str], out: list[Statement]) -> str:
        """The name that holds ``name`` at ``dtype``: itself, or the conversion defined ahead of its reader."""
        if self.dtype(name) == dtype:
            return name
        key = (name, dtype.name)
        if key in conversions:
            return conversions[key]
        # A register element is read by its subscript (``_c0[1]``); the conversion's name spells it as an identifier.
        converted = self.fresh(f"{re.sub(r'\W+', '_', name).strip('_')}__{dtype.name}")
        out.append(Assign(converted, CastExpression(dtype.name, Var(name)), dtype))
        self.types[converted] = dtype
        if name not in self.states:
            conversions[key] = converted
        return converted

    def statement(self, s: Statement, conversions: dict[tuple[str, str], str], out: list[Statement]) -> None:
        if isinstance(s, Load):
            dtype = s.dtype or self.buf_dtypes.get(s.input) or F32
            self.types.update((name, dtype) for name in s.names)
            out.append(s if s.dtype is not None else s.replace(dtype=dtype))
        elif isinstance(s, Assign):
            dtype = s.dtype or F32
            out.append(Assign(s.name, self.value(s.name, s.value, dtype, conversions, out), dtype))
            self.types[s.name] = dtype
        elif isinstance(s, Accum):
            dtype = s.dtype or F32
            self.types[s.name] = dtype
            out.append(s.replace(value=self.operand(f"{s.name}__v", s.value, dtype, conversions, out), dtype=dtype))
        elif isinstance(s, Carry):
            dtype = s.dtype or F32
            self.types[s.name] = dtype
            out.append(s.replace(value=self.at(s.value, dtype, conversions, out), dtype=dtype))
        elif isinstance(s, Pre):
            self.types[s.name] = self.dtype(s.carrier)
            out.append(s)
        elif isinstance(s, Select):
            dtype = dtype_get(dtype_promote("add", [self.dtype(b.value).name for b in s.branches]))
            out.append(s.replace(branches=tuple(SelectBranch(self.at(b.value, dtype, conversions, out), b.select) for b in s.branches)))
            self.types[s.name] = dtype
        elif isinstance(s, Write):
            dtype = self.buf_dtypes.get(s.output) or F32
            if s.value_dtype is not None:
                for v in s.values:
                    self.types.setdefault(v, s.value_dtype)  # a value no shared statement defines: the stamp is its type
            out.append(s.replace(values=tuple(self.at(v, dtype, conversions, out) for v in s.values), value_dtype=dtype))
        else:
            if isinstance(s, (Init, Let)) and s.dtype is not None:
                self.types[s.name] = s.dtype
            nested = s.nested()
            out.append(s.with_bodies(tuple(self.body(b, conversions) for b in nested)) if nested else s)
            for inner in nested:
                for stmt in inner.iter():
                    self.forget(stmt.defines(), conversions)
        self.forget(s.defines(), conversions)

    @staticmethod
    def forget(names, conversions: dict[tuple[str, str], str]) -> None:
        """Drop the conversions of ``names``, which a statement just defined: a later read sees the new value."""
        for key in [key for key in conversions if key[0] in names]:
            del conversions[key]

    def operand(self, base: str, node: Expression, dtype: DataType, conversions, out: list[Statement]) -> Expression:
        """``node`` as an operand at ``dtype``: a literal as it stands (it is spelled at its reader's dtype), a name
        converted where it must be, an operation a definition of its own ahead of its reader."""
        if isinstance(node, Literal):
            return node
        if not isinstance(node, Var):
            name = self.fresh(base, "0") if base.endswith("__e") else self.fresh(base)
            own = dtype_get(promoted_dtype(node, lambda n: self.dtype(n).name))
            out.append(Assign(name, self.value(name, node, own, conversions, out), own))
            self.types[name] = own
            node = Var(name)
        return Var(self.at(node.name, dtype, conversions, out))

    def value(self, name: str, value: Expression, result: DataType, conversions, out: list[Statement]) -> Expression:
        """The value of the definition ``name`` at ``result``: one operation over operands at its dtype, or one
        conversion. What it needs ahead of it is appended to ``out``."""
        if isinstance(value, Var):
            return value if self.dtype(value.name) == result else CastExpression(result.name, value)
        op = value.operation
        if op is None:
            if isinstance(value, TernaryExpression):
                arms = [self.operand(f"{name}__e", arm, result, conversions, out) for arm in (value.if_true, value.if_false)]
                return TernaryExpression(value.cond, *arms)
            return value  # a literal, an index, or a cast the body already spells
        if op.name == "bitcast":
            return value  # reads its operand's bits at whatever dtype holds them
        typed, compute = self.operation(name, value, result, conversions, out)
        if op.name in _COMPARISONS:
            # A truth value, declared another dtype: the comparison is a definition of its own, converted once.
            return typed if result == BOOL else CastExpression(result.name, Var(self.define(f"{name}__bool", "", typed, BOOL, out)))
        if compute == result or isinstance(typed, CastExpression):
            return typed
        return CastExpression(result.name, Var(self.define(f"{name}__{compute.name}", "", typed, compute, out)))

    def define(self, base: str, first: str, value: Expression, dtype: DataType, out: list[Statement]) -> str:
        name = self.fresh(base, first)
        out.append(Assign(name, value, dtype))
        self.types[name] = dtype
        return name

    def operation(self, name: str, node: Expression, result: DataType, conversions, out: list[Statement]) -> tuple[Expression, DataType]:
        """``node`` with every operand at the dtype it computes at, and that dtype; its result is declared ``result``.
        A nested operation stays in place where it computes at its reader's dtype (a node of one affine region), and
        is a definition of its own ahead of the reader otherwise."""
        op = node.operation
        operands: list[tuple[Expression, DataType | None]] = []  # a literal has no dtype of its own
        for o in node.operands:
            if isinstance(o, Literal):
                operands.append((o, None))
            elif isinstance(o, Var):
                operands.append((o, self.dtype(o.name)))
            else:
                own = dtype_get(promoted_dtype(o, lambda n: self.dtype(n).name))
                if o.operation is None or o.operation.name == "bitcast":
                    operands.append((self.operand(f"{name}__e", o, own, conversions, out), own))
                    continue
                typed, compute = self.operation(name, o, own, conversions, out)
                if compute != own and not isinstance(typed, CastExpression):
                    typed = CastExpression(own.name, Var(self.define(f"{name}__e", "0", typed, compute, out)))
                operands.append((typed, own))
        stand_ins = {f"{name}__operand{i}": dtype.name for i, (_, dtype) in enumerate(operands) if dtype is not None}
        shape = node.with_children([o if dtype is None else Var(f"{name}__operand{i}") for i, (o, dtype) in enumerate(operands)])
        promoted = promoted_dtype(shape, stand_ins.get)
        named = list(stand_ins.values())
        if result == F16 and op.decodes is not None and named == [op.decodes] and isinstance(operands[0][0], Var):
            # Every fp8 value is an f16 value: the decode converts straight to it, not through f32 and back.
            return CastExpression(result.name, operands[0][0]), result
        compute = dtype_get(self._compute(op.name, shape, named, promoted, result.name))
        if op.name in _IDENTITIES and isinstance(operands[0][0], Var):
            # The operation is its operand's conversion: to the dtype it computes at, then to the result's.
            source = operands[0][0]
            if compute != result:
                source = Var(self.at(source.name, compute, conversions, out))
            return self.value(name, source, result, conversions, out), result
        final: list[Expression] = []
        for o, dtype in operands:
            if dtype is None or (dtype == compute and not isinstance(o, CastExpression)):
                final.append(o)
                continue
            if not isinstance(o, Var):
                o = Var(self.define(f"{name}__e", "0", o, dtype, out))
            final.append(Var(self.at(o.name, compute, conversions, out)))
        return node.with_children(final), compute

    def _compute(self, op: str, node: Expression, named: list[str], promoted: str, result: str) -> str:
        """The dtype ``node`` computes at, its result declared ``result``."""
        literals = [o for o in node.operands if isinstance(o, Literal)]
        if named and all(d in INTEGER_NAMES for d in named) and all(o.dtype != "float" for o in literals):
            integers = dtype_promote(op, named)
            if integers in INTEGER_NAMES and op in INTEGER_OPS:
                # Integer algebra computes at its declared width; declared a float, at its operands' promotion, as
                # torch does, and the result converts once: ``(float)(a + b)``, never ``(float)a + (float)b``.
                return result if result in INTEGER_NAMES else integers
            if op in _INTEGER_OPERATORS:
                return dtype_promote("add", named)
            return "f32"
        # A literal is spelled at the dtype the named operands promote to, so it never keeps an operation from its
        # native form. Mixed dtypes (an explicit demotion) compute in f32 and convert once, which keeps more
        # precision than converting each operand to f16 first.
        native = result in ("f16", "f16x2") and op in self.native_f16
        if native and all(d == result for d in named) and (not literals or promoted == result):
            return result
        return "f32"
