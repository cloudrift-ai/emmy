"""``Dim`` — one tensor / axis extent, backed by an ``Expression``.

Every Dim carries a shape expression:

- ``Dim(32)`` wraps ``Literal(32, "int")`` — a static extent.
- ``Dim("seq_len")`` wraps ``Var("seq_len")`` — an atomic symbolic extent.
- ``Dim(seq) * Dim(2)`` wraps ``BinaryExpression("*", Var("seq"), Literal(2))`` —
  a composite symbolic extent. Operators on Dim dispatch to ``Expression`` and
  eagerly fold via ``Expression.simplify``: ``Dim(32) * Dim(64) → Dim(Literal(2048))``,
  ``Dim("s") * Dim(1) → Dim(Var("s"))``.

Reads:

- ``d.expression`` always works — returns the underlying ``Expression``.
- ``d.as_static()`` returns the int for ``Literal``-backed dims, raises otherwise.
- ``d.as_atom_name()`` returns the str name for ``Var``-backed dims, raises otherwise.
  Use at sites that route by symbolic name (kernel param signature, sym_env keys).
- ``d.value`` returns the int (``Literal``) or the name (``Var``); raises on
  composite. Back-compat shim — prefer ``.as_static()`` / ``.as_atom_name()`` / ``.expression``
  at new call sites.
- ``d == 32`` / ``d == "seq_len"`` / ``d == Dim(...)`` all work — the first
  two unwrap to ``Literal.value`` / ``Var.name``; composite dims only compare
  structurally to other Dims.

Runtime resolution:

- ``d.expression.eval(sym_env)`` gives the int extent at launch time, where
  ``sym_env: dict[str, int]`` maps each symbolic name to its concrete value
  (read from input array shapes). Composite Dims (e.g. ``S * 2`` from a cat
  output) resolve uniformly with atomic ones — one path, no branch.

No ``__int__`` / ``__index__``: ``int(d)`` and ``range(d)`` deliberately fail.
Sites that need a static int must say so via ``.as_static()`` — that way
introducing a symbolic dim later breaks loudly at the sites that can't yet
handle it, rather than silently casting through.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.expression import BinaryExpression, Expression, Interval, Literal, SimplifyCtx, Var
from emmy.compiler.wire import Wire

# Default "expected size" for a symbolic dim when none is supplied explicitly.
# Atomic symbolic Dims (input axes like ``Dim("seq_len")``) carry this so the
# planner / tuner always have a size to tile for — even when a pass reconstructs
# a bare ``Dim(name)`` and the original hint would otherwise be lost. 512 is a
# representative sequence length for tuning (vs the small trace example size).
DEFAULT_SEQ_HINT = 512

# Upper BOUND on a symbolic axis — every ``--dynamic`` axis is exported with this as its
# ``torch.export.Dim`` max, so no legal resolution of a symbolic dim exceeds it. Unlike
# ``DEFAULT_SEQ_HINT`` above, which is advisory and may be wrong in either direction, this one is
# load-bearing: address rendering sizes its integer arithmetic against it, so a symbolic buffer
# whose worst-case element count still fits ``int`` keeps 32-bit indexing. It lives here rather
# than in ``trace.dynamic`` (which re-exports it) because the IR layer must be able to read it
# without importing the frontend.
DYNAMIC_DIM_MAX = 4096


def _coerce_expression(value: int | str | Expression | Dim) -> Expression:
    """Coerce a Dim constructor argument to an ``Expression``.

    - bare ``int`` → ``Literal(value, "int")``
    - bare ``str`` → ``Var(value)``
    - ``Expression``     → passed through
    - ``Dim``      → unwrapped to its ``expression``
    """
    if isinstance(value, Dim):
        return value.expression
    if isinstance(value, int):
        return Literal(value, "int")
    if isinstance(value, str):
        return Var(value)
    if isinstance(value, Expression):
        return value
    raise TypeError(f"Dim: cannot wrap {type(value).__name__}: {value!r}")


def _simplify(expression: Expression) -> Expression:
    # Shape vars (every free ``Var`` in a Dim expression) are positive by
    # definition: a tensor extent can't be zero or negative. Expose that
    # via ``SimplifyCtx.ranges`` so the simplifier's ``//`` cancellation
    # path can safely cancel matching factors (otherwise ``(s*128) // (s*4)``
    # has no way to reduce to ``32``).
    ranges = {name: Interval(1, 1 << 30) for name in expression.free_vars()}
    return expression.simplify(SimplifyCtx(ranges))


@dataclass(frozen=True, init=False, eq=False)
class Dim(Wire):
    wire_tag = "dim"

    def to_wire(self):
        """``int`` for a static dim, ``{sym, hint}`` for a bare symbol, ``{expression, hint}`` for a composite."""
        if isinstance(self.expression, Literal) and self.expression.dtype == "int":
            return int(self.expression.value)
        out = {"sym": self.expression.name} if isinstance(self.expression, Var) else {"expr": self.expression.to_wire()}
        if self.hint is not None:
            out["hint"] = self.hint
        return out

    @classmethod
    def from_wire(cls, value: object, where: str = "dim") -> Dim:
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value)
        keys = set(value) if isinstance(value, dict) else set()
        if not (("sym" in keys and keys <= {"sym", "hint"}) or ("expr" in keys and keys <= {"expr", "hint"})):
            raise ValueError(f"{where} must be an integer, {{sym, hint}} or {{expression, hint}}")
        hint = int(value["hint"]) if value.get("hint") is not None else None
        return cls(str(value["sym"]), hint=hint) if "sym" in keys else cls(Expression.from_wire(value["expr"], f"{where}.expr"), hint=hint)

    expression: Expression
    # Advisory "expected size" for a symbolic dim — the value the tuner /
    # partition planner pretends the axis has when picking tile sizes (set
    # at trace time from ``--seq-len``). Pure metadata: excluded from
    # equality / hashing / pretty-rendering so two Dims with the same
    # ``expression`` stay structurally identical (and cache keys hint-independent)
    # regardless of hint. Atomic symbolic Dims carry it; arithmetic results
    # do not (only the input/axis Dim needs it). ``None`` on static dims.
    hint: int | None

    def __init__(self, value: int | str | Expression | Dim, *, hint: int | None = None) -> None:
        # ``frozen=True`` blocks normal assignment; route through object.__setattr__.
        expression = _coerce_expression(value)
        object.__setattr__(self, "expression", expression)
        # Explicit hint wins; otherwise inherit when wrapping an existing Dim,
        # and finally fall back to ``DEFAULT_SEQ_HINT`` for an atomic symbolic
        # dim (a single ``Var``) so the planner always sees an expected size.
        # Composite (``BinaryExpression``) and static (``Literal``) dims keep ``None``.
        if hint is None and isinstance(value, Dim):
            hint = value.hint
        if hint is None and isinstance(expression, Var):
            hint = DEFAULT_SEQ_HINT
        object.__setattr__(self, "hint", hint)

    def structural_key(self) -> str:
        """Identity of the extent expression; the expected-size hint is advisory."""
        from emmy.compiler.structural import digest, form  # noqa: PLC0415

        return digest(form(self.expression))

    # ---- inspection ------------------------------------------------------

    @property
    def is_static(self) -> bool:
        return isinstance(self.expression, Literal) and isinstance(self.expression.value, int)

    def as_static(self) -> int:
        if not (isinstance(self.expression, Literal) and isinstance(self.expression.value, int)):
            raise TypeError(f"Dim({self.expression!r}) is not a static int — cannot resolve to int statically")
        return self.expression.value

    @property
    def value(self) -> int | str:
        """Back-compat shim: ``Literal.value`` for static, ``Var.name`` for atomic
        symbolic. Raises on composite. Prefer ``.expression`` at new call sites."""
        if isinstance(self.expression, Literal) and isinstance(self.expression.value, int):
            return self.expression.value
        if isinstance(self.expression, Var):
            return self.expression.name
        raise TypeError(f"Dim({self.expression!r}).value: composite dim has no scalar value; use .expression")

    def as_atom_name(self) -> str:
        """Return the symbolic name when this Dim is a single ``Var``; raise
        otherwise. Use at sites that route by symbolic name — kernel signature
        params, structural-key entries, runtime ``sym_env`` lookups — and
        want to fail loud if a static or composite Dim slips in."""
        if isinstance(self.expression, Var):
            return self.expression.name
        raise TypeError(f"Dim({self.expression!r}).as_atom_name: dim is not an atomic Var")

    # ---- arithmetic — eager-fold via Expression.simplify -----------------------

    def __add__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("+", self.expression, _coerce_expression(other))))

    def __radd__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("+", _coerce_expression(other), self.expression)))

    def __sub__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("-", self.expression, _coerce_expression(other))))

    def __rsub__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("-", _coerce_expression(other), self.expression)))

    def __mul__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("*", self.expression, _coerce_expression(other))))

    def __rmul__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("*", _coerce_expression(other), self.expression)))

    def __floordiv__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("//", self.expression, _coerce_expression(other))))

    def __rfloordiv__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("//", _coerce_expression(other), self.expression)))

    def __mod__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("%", self.expression, _coerce_expression(other))))

    def __rmod__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpression("%", _coerce_expression(other), self.expression)))

    def ceil_div(self, other: int | Dim) -> Dim:
        """Ceiling division ``ceil(self / other)``, as ``(self + (other - 1)) // other``.

        One masked-tile grid-extent formula for static and symbolic dims alike:
        eager ``Expression.simplify`` folds a static (``Literal``) dim to the integer
        ceil (``Dim(65).ceil_div(16) → Dim(5)``, matching the ``-(-E // b)``
        idiom for positive extents), while a symbolic dim builds the composite
        ceil-div ``Expression`` the launch resolver evals from ``sym_values``. Replaces
        the hand-written static-int / symbolic-``Expression`` ceil-div pair at masked
        block-axis and masked-K sites (``tile/010_partition_loops.py``)."""
        return (self + (other - 1)) // other

    # ---- equality + hashing ---------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dim):
            return self.expression == other.expression
        # Back-compat ergonomics: ``Dim(32) == 32`` and ``Dim("s") == "s"``
        # keep working so test assertions and existing pass code don't churn.
        # Composite dims (``BinaryExpression``-backed) never compare equal to bare
        # int/str — they only compare structurally to other Dims.
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return isinstance(self.expression, Literal) and self.expression.value == other
        if isinstance(other, str):
            return isinstance(self.expression, Var) and self.expression.name == other
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __hash__(self) -> int:
        # Hash on the canonical Expression so Dim(32) and Dim(Literal(32, "int"))
        # hash identically. Atomic Vars and Literals are themselves frozen
        # and hashable; composite BinaryExpression is hashable post-M0.
        return hash(self.expression)

    def __repr__(self) -> str:
        if isinstance(self.expression, Literal) and isinstance(self.expression.value, int):
            return f"Dim({self.expression.value!r})"
        if isinstance(self.expression, Var):
            return f"Dim({self.expression.name!r})"
        return f"Dim({self.expression!r})"

    def __str__(self) -> str:
        # Used by f-strings / ``str()`` — render transparently so pretty IR
        # output (``for i in 0..32``) and ``Body.structural_key`` digests
        # don't change shape just because the type wrapper exists.
        if isinstance(self.expression, Literal) and isinstance(self.expression.value, int):
            return str(self.expression.value)
        if isinstance(self.expression, Var):
            return self.expression.name
        return str(self.expression)


def to_dim(value: int | str | Dim | Expression) -> Dim:
    """Coerce ``int`` / ``str`` / ``Expression`` to ``Dim``; pass ``Dim`` through unchanged."""
    return value if isinstance(value, Dim) else Dim(value)


def as_static(d: object) -> int:
    """Coerce a shape element (``Dim``, ``int``, or ``str``) to a static int.

    Use at boundaries where the upstream type isn't yet uniformly ``Dim`` —
    e.g. ``infer_output_shape`` returns that mix ints/strings from constructor
    args with ``Dim`` from another Tensor's shape.

    Raises if ``d`` is symbolic (``Dim('seq_len')`` or ``str``)."""
    if isinstance(d, Dim):
        return d.as_static()
    if isinstance(d, int):
        return d
    raise TypeError(f"as_static: cannot convert {d!r} to int (symbolic shape element?)")


def paged_declarations(hint, graph=None) -> dict[str, tuple[int, int, int | None]]:
    """The ``cuda.paged_buffers`` hint as ``{name: (axis, page, slots)}``. An entry is ``(name, axis, page)`` or
    ``(name, axis, page, slots)``: ``slots`` is how many pages the device holds at once, ``None`` for every page —
    a buffer the pager never moves, which the compiler then leaves alone past the table lowering — and ``0`` for
    none, which the compiler leaves alone too: every page is read where it lives.

    With ``graph``, a declared weight the compiler binds under other names — its folded transpose, its source layout:
    every constant of ``graph`` bound from it (``ConstantOp.name``) — pages each of them as one page, the only paging
    every layout of a weight shares; the declaration must be one page too, and its own name goes once no buffer has
    it."""
    declared = {entry[0]: (int(entry[1]), int(entry[2]), None if len(entry) < 4 else entry[3]) for entry in hint}
    if graph is None or not declared:
        return declared
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415 — the IR imports this module

    bound = {}
    for nid, node in graph.nodes.items():
        op = node.op
        if isinstance(op, ConstantOp) and op.name != nid and op.name in declared and nid not in declared:
            axis, page, slots = declared[op.name]
            if op.source_shape is None or page < op.source_shape[axis]:
                raise ValueError(f"paged weight {op.name!r} is bound as {nid!r} in another layout: it can be paged as one page only")
            bound[nid] = (0, as_static(node.output.shape[0]), slots)
    gone = {graph.nodes[nid].op.name for nid in bound} - {name for name in declared if graph.buffer(name) is not None}
    return {**{name: paging for name, paging in declared.items() if name not in gone}, **bound}


def spans_one_page(shape, axis: int, page: int) -> bool:
    """Whether a paged buffer's declared shape fits in one page along its paged axis — then one base
    address stands behind it, which the renderer resolves once; a symbolic extent never does."""
    try:
        return as_static(shape[axis]) <= page
    except (TypeError, ValueError):
        return False


__all__ = ["Dim", "to_dim", "as_static", "spans_one_page"]
