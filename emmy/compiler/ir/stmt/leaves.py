"""Leaf ``Stmt`` subclasses — pure compute primitives (no nested bodies).

``Load``, ``Assign``, ``Accum``, ``Init``, ``Write``, ``Select`` — each
produces / writes a single SSA value. Block-structured stmts (Loop /
Tile / Cond) live in ``blocks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import count

from emmy.compiler.dtype import DataType
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, FlatIndex, Literal, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.base import Stmt


@dataclass(frozen=True, init=False)
class Load(Stmt):
    """Read a value (or N consecutive values) from an external input buffer.

    Scalar (``width == 1``): one SSA binding ``names[0]``. Vector
    (``width > 1``): N SSA bindings; lane k reads
    ``index[:-1] + (index[-1] + k,)`` into ``names[k]``. The
    ``050_vectorize_loads`` pass widens a run of consecutive scalar
    Loads into one Load with ``len(names) > 1``. Every pass before that
    produces scalar Loads and can keep using ``s.name`` / ``s.index``.

    The constructor accepts either ``name="x"`` (scalar shorthand,
    normalized to ``names=("x",)``) or ``names=(...)`` directly. Use
    ``.is_vector`` / ``.is_scalar`` / ``.width`` / ``.name`` (asserts
    scalar) / ``.names`` to test shape.

    A scalar Load is rendered as a literal binding (``float name =
    <value>;``) when ``ctx.literal_constants`` carries a value for
    ``input`` — the scalar-constant-inlining path populates that map at
    the cuda lowering boundary so kernels can embed ``ConstantOp``
    values directly instead of taking them as ``float*`` parameters.

    ``dtype`` is optional and, when set, names the source-buffer element
    dtype that this Load produces. The ``030_stamp_types`` Kernel-IR pass
    populates it once before any analytical pass runs; downstream passes
    (vectorize_loads, demote, etc.) read it instead of reaching for the
    matcher-populated ``KernelOp.inputs``/``outputs`` side channels.
    ``None`` keeps the legacy render-time inference path working for
    tests that construct Loads by hand without dtype.

    ``carried`` marks the two halves of a staged copy that crosses the chunk loop's back edge. The
    names and their packed vector are the ones an earlier Load of the same names DECLARED ahead of
    the loop. ``"load"`` re-assigns only the packed vector, through the read-only (non-coherent)
    path — the operand is one the kernel never writes — so the loaded chunk survives into the next
    iteration as the few registers the vector holds. ``"unpack"`` re-assigns the names from that
    vector and touches no memory; it sits right before the names' consumer. Measured on the V100:
    unpacking at the load instead keeps every element live across the mma stream and ptxas then
    serializes the loads behind it (a 128x128 partial 84 us against 67).
    """

    names: tuple[str, ...]
    input: str
    index: tuple[Expr, ...]
    dtype: DataType | None
    carried: str = ""  # a default, so a Load that does not set it renders (and keys) as before

    pure = True  # a pure value binding — legal inside a stored ``Lambda`` body

    def __init__(
        self,
        name: str | None = None,
        input: str | None = None,
        index: tuple[Expr, ...] | None = None,
        *,
        names: tuple[str, ...] | None = None,
        dtype: DataType | None = None,
        carried: str = "",
    ) -> None:
        if names is None:
            if name is None:
                raise TypeError("Load requires either `name=` (scalar) or `names=` (vector)")
            names = (name,)
        elif name is not None:
            raise TypeError("Load: pass `name=` xor `names=`, not both")
        if input is None:
            raise TypeError("Load requires `input=`")
        if index is None:
            raise TypeError("Load requires `index=`")
        if not names:
            raise ValueError("Load.names must be non-empty")
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "index", tuple(index))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "carried", carried)

    @property
    def name(self) -> str:
        """Scalar-only shorthand for ``names[0]``. Asserts ``is_scalar`` —
        use ``.names`` (or guard with ``.is_scalar``) for vector Loads."""
        assert self.is_scalar, f"Load has {self.width} names — use .names for vector Loads"
        return self.names[0]

    @property
    def width(self) -> int:
        return len(self.names)

    @property
    def is_vector(self) -> bool:
        return len(self.names) > 1

    @property
    def is_scalar(self) -> bool:
        return len(self.names) == 1

    def deps(self) -> tuple[str, ...]:
        # SSA names this Load reads through its *index* — a data-dependent
        # (gather) index like ``weight[(int)in0, a]`` reads ``in0``. Returns
        # the index Exprs' free Vars (first-use order, de-duped). Axis-name
        # Vars (``a0``, ``a1``) appear too, but every consumer resolves deps
        # against the body's ``definitions``, where axes are absent — so they
        # pass through as no-ops, exactly as ``Expr.rename`` and
        # the dataflow helpers already assume.
        return tuple(dict.fromkeys(v for e in self.index for v in e.free_vars()))

    def defines(self) -> tuple[str, ...]:
        return self.names

    def external_reads(self) -> tuple[str, ...]:
        return (self.input,)

    def rename_buffers(self, rename):  # noqa: ANN001 — see ``Stmt.rename_buffers``
        new = rename.get(self.input, self.input)
        return self if new == self.input else self.replace(input=new)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(index=exprs)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return self.replace(
            names=tuple(rename(n) for n in self.names),
            index=tuple(sigma.apply(e).rename(rename) for e in self.index),
        )


@dataclass(frozen=True)
class Assign(Stmt):
    """Pure SSA body statement: ``name = op(args)``.

    ``op`` is an ``ElementwiseImpl`` — the elementwise combine (add /
    mul / exp / ...). Reductions live in ``Accum``. ``args`` reference
    ``$N`` ports, an ``Accum.name`` (reads current / finalized acc
    value), or prior SSA names.

    ``dtype`` is optional and overrides the default ``promote(args)``
    rule when set. The ``demote_to_write_dtype`` Kernel-IR pass stamps
    it on Assigns whose results only feed dtype-narrower consumers, so
    a softmax / RMSNorm epilogue feeding an ``__half*`` Write computes
    in ``__half`` end to end without spurious promotions.
    """

    name: str
    op: ElementwiseImpl
    args: tuple[str, ...]
    dtype: DataType | None = None

    pure = True  # a pure value binding — legal inside a stored ``Lambda`` body

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    def deps(self) -> tuple[str, ...]:
        return self.args

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return Assign(name=rename(self.name), op=self.op, args=tuple(rename(a) for a in self.args), dtype=self.dtype)


@dataclass(frozen=True)
class Affine(Stmt):
    """Pure SSA body statement: ``name = offset + Σ coefficient·arg``.

    The one statement for a sum, a difference, a negation or a product with a literal: normalization folds every
    such cone into one ``Affine`` (``fold_affine``), so ``x + x``, ``2 * x`` and ``x + 3 * x - x`` are one body.
    It is n-ary, so identity lists its terms by value and there is no chain of binary operations whose shape
    would need a canonical order. ``args`` are SSA names, one per term after normalization, no coefficient zero; a
    literal operand is a coefficient or the ``offset``. A product of two values stays an ``Assign``.
    """

    name: str
    args: tuple[str, ...]
    coefficients: tuple[float, ...]
    offset: float = 0.0
    dtype: DataType | None = None

    pure = True  # a pure value binding — legal inside a stored ``Lambda`` body

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(float(c) for c in self.coefficients))
        object.__setattr__(self, "offset", float(self.offset))
        if len(self.args) != len(self.coefficients):
            raise ValueError(f"Affine {self.name!r}: {len(self.args)} args and {len(self.coefficients)} coefficients")
        if not self.args:
            raise ValueError(f"Affine {self.name!r} has no term: a constant is a Let")
        if any(c == 0 for c in self.coefficients):
            raise ValueError(f"Affine {self.name!r} carries a zero coefficient: {self.coefficients}")

    @property
    def terms(self) -> tuple[tuple[float, str], ...]:
        return tuple(zip(self.coefficients, self.args, strict=True))

    def deps(self) -> tuple[str, ...]:
        return self.args

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def as_assigns(self) -> list[Stmt]:
        """The same value as binary ``Assign``s over ``Let``-bound literals, the last one named ``name``: for a
        lowering that evaluates one operation at a time (Kernel IR, the fragment programs)."""
        out: list[Stmt] = []
        fresh = (f"{self.name}__t{index}" for index in count())

        def bound(value: float) -> str:
            out.append(Let(next(fresh), Literal(value), self.dtype))
            return out[-1].name

        def step(op: str, args: tuple[str, ...]) -> str:
            out.append(Assign(next(fresh), op, args, self.dtype))
            return out[-1].name

        total: str | None = None
        # A positive term leads when there is one, so the chain opens with a value rather than a negation.
        terms = sorted(self.terms, key=lambda term: term[0] < 0)
        for coefficient, arg in terms:
            term = arg if abs(coefficient) == 1 else step("multiply", (arg, bound(abs(coefficient))))
            if total is None:
                total = term if coefficient > 0 else step("negative", (term,))
            else:
                total = step("add" if coefficient > 0 else "subtract", (total, term))
        if self.offset != 0:
            total = step("add" if self.offset > 0 else "subtract", (total, bound(abs(self.offset))))
        if out and out[-1].name == total:
            out[-1] = out[-1].replace(name=self.name)
        else:
            out.append(Assign(self.name, "copy", (total,), self.dtype))
        return out

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return self.replace(name=rename(self.name), args=tuple(rename(a) for a in self.args))


def binary(stmts) -> tuple[Stmt, ...]:  # noqa: ANN001 — any iterable of Stmt
    """``stmts`` with every ``Affine`` spelled as its binary ``Assign``s, for a lowering that evaluates one
    operation at a time; the other statements pass through."""
    return tuple(part for stmt in stmts for part in (stmt.as_assigns() if isinstance(stmt, Affine) else (stmt,)))


def _rewrite_axis_name(name: str, sigma: Sigma) -> tuple[str, ...]:
    """Apply ``sigma`` to an axis name and return the resulting axis
    name(s). Handles three cases:

    - ``sigma`` doesn't touch ``name``: returns ``(name,)``.
    - Pure rename (``Var(old) → Var(new)``): returns ``(new,)``.
    - σ-split (``Var(K) → Var(K_o)*N + Var(K_i)``, etc.): returns the
      free-var names of the substitution expression. An Accum that
      reduced over the original axis now reduces over the split sub-
      axes.
    """
    replacement = sigma.mapping.get(name)
    if replacement is None:
        return (name,)
    return tuple(sorted(replacement.free_vars()))


@dataclass(frozen=True)
class Accum(Stmt):
    """Reduce accumulator — declares-and-folds in one statement.

    Semantics: ``name = op(name, value)`` inside the enclosing reduce
    ``Loop``. Before the first iteration ``name`` is initialized to
    ``op.identity`` (the combine's neutral element). After the Loop
    completes, ``name`` is an SSA binding visible in the enclosing scope,
    carrying the finalized reduced value.

    ``op`` is an ``ElementwiseImpl`` — typically one of ``ADD`` / ``MAX`` /
    ``MIN`` / ``MUL``. It defines both the combine operation and the
    accumulator's identity value. Multiple ``Accum`` stmts targeting the
    same ``name`` in one reduce Loop must agree on ``op``.

    Default op is ``add`` — fixtures that sum values can omit ``op=``;
    ``max`` / ``min`` / ``mul`` must be passed explicitly.

    ``axes`` is the tuple of axis names this Accum reduces over —
    populated by the lifting pass (one axis name from the wrapping reduce
    ``Loop``) and propagated by every rewrite that renames axes via
    Sigma. Used by the escape-analysis helper to derive cross-thread
    cooperativity (``Accum.axes ∩ enclosing ThreadTile.axes``); empty
    tuple = legacy/handwritten path with no reduction-axis info.
    """

    name: str
    value: str
    op: ElementwiseImpl = field(default_factory=lambda: ElementwiseImpl("add"))
    # Optional accumulator dtype. ``None`` in Loop IR — derived at lowering
    # time. The Init-placement pass freezes this to a concrete
    # :class:`DataType` (typically ``F32`` for fp16 reductions). When set,
    # ``Accum``'s render declares the local in that dtype and inserts a
    # ``__half2float`` / ``__float2half`` on ``value`` only when ``value``'s
    # dtype disagrees with the accumulator's. ``None`` preserves the legacy
    # f32 rendering.
    dtype: DataType | None = None
    axes: tuple[str, ...] = ()
    # Optional rescaled base — the value the fold reads as its left operand instead
    # of ``name`` (``name = op(base, value)``). ``None`` = the ordinary self-fold
    # ``name = op(name, value)``. A twisted carrier's streaming merge lowers each
    # state component to a ``base``-``Accum``: the ψ rescale is a preceding ``Assign``
    # binding ``base`` (e.g. ``lm = l·alpha``), so the fold itself stays an ``Accum``
    # whose seed is still ``op.identity``. The seed (and ``Loop``'s render's per-Accum
    # init) is unchanged — ``base`` only redirects the in-loop left operand.
    base: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))
        if self.base == self.name:  # the ordinary self-fold has one spelling
            object.__setattr__(self, "base", None)

    @property
    def init(self) -> Expr:
        """Identity value for the accumulator (from the op's metadata)."""
        identity = self.op.identity
        return Literal(identity if identity is not None else 0.0)

    def deps(self) -> tuple[str, ...]:
        # ``base`` (when it redirects the left operand) is a same-scope read; the
        # carried ``name`` read is implicit (loop-carried), like the default fold.
        if self.base is not None and self.base != self.name:
            return (self.value, self.base)
        return (self.value,)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    # Algebraic traits forward to the scalar combine op — a ``max`` Accum and a
    # ``sum`` Accum differ, and ``self.op`` is the source of truth.
    @property
    def associative(self) -> bool:
        return self.op.associative

    @property
    def commutative(self) -> bool:
        return self.op.commutative

    @property
    def has_identity(self) -> bool:
        return self.op.has_identity

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        new_axes = tuple(sorted({rename(n) for old in self.axes for n in _rewrite_axis_name(old, sigma)}))
        return Accum(
            name=rename(self.name),
            value=rename(self.value),
            op=self.op,
            dtype=self.dtype,
            axes=new_axes,
            base=rename(self.base) if self.base is not None else None,
        )


#: The suffix of the slot a :class:`Carry` defines during a step.
@dataclass(frozen=True)
class Carry(Stmt):
    """The next value of one cell of a CARRIED STATE: ``name[index] <- value``.

    A recurrence's state, which a fold is not: its steps run in order, its step is any computation,
    and it holds ``seed`` before the first. So it has no op — an op would declare an algebra (a
    reorderable sum) the state does not have — and nothing else to declare either: the loop that
    carries it is the nearest enclosing ``Loop`` whose axis ``index`` does not read
    (:func:`~emmy.compiler.ir.stmt.blocks.carried_cells`), and its shape is those of the loops over
    its cells, each entry of ``index`` a bare axis. The loop over the steps therefore sits OUTSIDE
    the loops over the cells, which is what lets a step read OTHER cells of its own state (a delta
    rule's ``k @ S``) through :class:`Pre`; a scalar ``Accum`` lives inside the cell loops and
    cannot. One statement defines a state; a cell it skips keeps its value.
    """

    name: str
    value: str
    index: tuple[Expr, ...]
    #: What the state holds before the first step: a constant, or the name of a buffer of the
    #: state's shape read cell by cell — the tensor an unrolled loop started from.
    seed: float | str
    dtype: DataType | None = None

    def external_reads(self) -> tuple[str, ...]:
        return (self.seed,) if isinstance(self.seed, str) else ()

    def rename_buffers(self, rename):  # noqa: ANN001 — see ``Stmt.rename_buffers``
        return self.replace(seed=rename.get(self.seed, self.seed)) if isinstance(self.seed, str) else self

    @property
    def cells(self) -> tuple[str, ...]:
        """The axes of the cell this statement defines. An entry that is no axis is the ``0``
        normalization leaves where a size-one axis stood."""
        return tuple(e.name for e in self.index if isinstance(e, Var))

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(index=exprs)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        index = tuple(sigma.apply(e).rename(rename) for e in self.index)
        return Carry(name=rename(self.name), value=rename(self.value), index=index, seed=self.seed, dtype=self.dtype)


@dataclass(frozen=True)
class Pre(Stmt):
    """A read of one cell of a carried state: ``name = carrier[index]``.

    Inside the ``Loop`` that carries it, the value the PREVIOUS step left (the seed on the first
    step) — no read of a step sees what that step defines; after that loop closes, the last
    step's. ``index`` is any coordinate expression, so a step may read a cell other than the one
    it defines — under a reduce over it, a contraction over the state. See :class:`Carry`.

    A read like a ``Load``, so a term may hold one: a carrier read is a slab over one ``Pre``
    (``Fold.carrier_read``), the way a gmem read is a slab over one ``Load``.
    """

    pure = True

    name: str
    carrier: str
    index: tuple[Expr, ...]

    def deps(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.carrier, *(v for e in self.index for v in e.free_vars()))))

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(index=exprs)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return Pre(name=rename(self.name), carrier=rename(self.carrier), index=tuple(sigma.apply(e).rename(rename) for e in self.index))


@dataclass(frozen=True)
class Init(Stmt):
    """Explicit accumulator / carried-state seed at this scope:
    ``<dtype> <name> = <identity>;`` — a scope-local declaration.

    A fold's OWN seed does not ride here — it lives on its ``Accum`` and is derived by
    ``Loop``'s render from ``op.identity``. What produces an ``Init`` is the CLAMP-TO-IDENTITY
    masking: a masked tail's ``Select`` needs the identity as an SSA name to select, so the
    masked ILP copy (``lowering/kernel/_factor``) and the compute fill's K mask
    (``lowering/kernel/_atom._k_masked``) each bind one. ``identity`` is the neutral element
    (one scalar — 0 / 1 / -inf), held directly.
    """

    name: str
    identity: float
    # Accumulator / state dtype — required. Placing an ``Init`` is the freeze
    # point; the pass that emits it must commit to a concrete dtype. The
    # same pass stamps the matching ``Accum``'s ``dtype`` to this value
    # so the IR stays self-consistent.
    dtype: DataType = field(kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            from emmy.compiler.dtype import get as _get  # noqa: PLC0415

            object.__setattr__(self, "dtype", _get(self.dtype))

    def deps(self) -> tuple[str, ...]:
        return ()

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        # ``identity`` is a constant scalar — only the name moves. Renamed in lockstep with
        # the fold's ``Accum`` (registered above) so the seed stays paired.
        return Init(name=rename(self.name), identity=self.identity, dtype=self.dtype)


@dataclass(frozen=True)
class Let(Stmt):
    """A pure binding of one expression to an SSA name: ``<dtype> <name> = <value>;``.

    Three things reach a body this way — a scalar literal (the twisted carrier's injection binds
    the ``1`` online softmax folds each element with, so the lift can return it by name), an
    integer index precomputed once outside a nested hot loop, and a buffer coordinate's flattened
    offset (:class:`~emmy.compiler.ir.expr.FlatIndex`) reused across a copy's trips. Pure, so it
    is legal inside a stored ``Lambda`` body. A float literal's ``dtype`` is stamped at kernel
    lowering like an ``Assign``'s (f32 when nothing narrows it); an untyped index renders as ``auto``
    and takes its expression's type. A bare number coerces to a float ``Literal``.
    """

    pure = True

    name: str
    value: Expr
    dtype: DataType | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, (int, float)):
            object.__setattr__(self, "value", Literal(float(self.value)))

    def deps(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.value.free_vars()))

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def exprs(self) -> tuple[Expr, ...]:
        return (self.value,)

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(value=exprs[0])

    def external_reads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.buffer for e in self.value.subterms() if isinstance(e, FlatIndex)))

    def rename_buffers(self, rename):  # noqa: ANN001 — see ``Stmt.rename_buffers``
        value = self.value.rebuild(lambda e: replace(e, buffer=rename.get(e.buffer, e.buffer)) if isinstance(e, FlatIndex) else e)
        return self if value == self.value else self.replace(value=value)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return Let(name=rename(self.name), value=sigma.apply(self.value).rename(rename), dtype=self.dtype)


@dataclass(frozen=True, init=False)
class Write(Stmt):
    """Write an SSA value (or N consecutive values) to ``output``.

    Scalar (``width == 1``): stores one SSA value at ``index``. Vector
    (``width > 1``): stores N values; ``values[0]`` goes to ``index``,
    ``values[k]`` goes to ``index[:-1] + (index[-1] + k,)``. The
    ``080_vectorize_stores`` pass widens a run of consecutive scalar
    Writes into one Write with ``len(values) > 1``. Every pass before
    that produces scalar Writes and can keep using ``s.value`` /
    ``s.index``.

    The constructor accepts either ``value="v"`` (scalar shorthand,
    normalized to ``values=("v",)``) or ``values=(...)`` directly. Use
    ``.is_vector`` / ``.is_scalar`` / ``.width`` / ``.value`` (asserts
    scalar) / ``.values`` to test shape.

    ``output`` is the destination buffer's name (matches the owning graph
    node's id, or — for multi-output kernels — one of its output buffer
    names). ``index`` uses axis Vars to compute the per-dim offset.

    ``value_dtype`` is optional; when set, names the SSA-value dtype being
    stored. Stamped by ``030_stamp_types``; downstream passes read it
    instead of querying ``ctx.ssa_dtypes`` at render time.
    """

    output: str
    index: tuple[Expr, ...]
    values: tuple[str, ...]
    value_dtype: DataType | None
    # An ``atomicAdd`` reduce-write (cross-CTA split-reduce, ``030_cut``'s atomic finalize):
    # every contributing CTA adds its partial into the SAME output cell, so the store is an
    # atomic accumulate (the output is zero-init'd per launch — ``CudaOp.zero_outputs``).
    # Scalar only; never vectorized (each lane needs its own ``atomicAdd``).
    atomic: bool
    # Software smem slab swizzle mode ("NONE"/"B32"/"B64"/"B128") — the sync compute-fill's
    # producer-side XOR: the flattened destination element index passes through
    # ``emmy_swizzle_<mode>`` exactly like a :class:`~emmy.compiler.ir.kernel.ir.CpAsyncCopy`
    # fill, so the ``LdmatrixLoad`` drain reads the slab back through the identical
    # permutation. Intra-chunk bits (0..2) pass through the XOR, so a ≤8-element vectorized
    # Write stays inside its relocated 16-byte chunk. "NONE" (every non-slab Write) is inert.
    swizzle: str

    def __init__(
        self,
        output: str | None = None,
        index: tuple[Expr, ...] | None = None,
        value: str | None = None,
        *,
        values: tuple[str, ...] | None = None,
        value_dtype: DataType | None = None,
        atomic: bool = False,
        swizzle: str = "NONE",
    ) -> None:
        if values is None:
            if value is None:
                raise TypeError("Write requires either `value=` (scalar) or `values=` (vector)")
            values = (value,)
        elif value is not None:
            raise TypeError("Write: pass `value=` xor `values=`, not both")
        if output is None:
            raise TypeError("Write requires `output=`")
        if index is None:
            raise TypeError("Write requires `index=`")
        if not values:
            raise ValueError("Write.values must be non-empty")
        if atomic and len(values) != 1:
            raise ValueError("an atomic Write is scalar (one value per atomicAdd)")
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "index", tuple(index))
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "value_dtype", value_dtype)
        object.__setattr__(self, "atomic", atomic)
        object.__setattr__(self, "swizzle", swizzle)

    @property
    def value(self) -> str:
        """Scalar-only shorthand for ``values[0]``. Asserts ``is_scalar`` —
        use ``.values`` (or guard with ``.is_scalar``) for vector Writes."""
        assert self.is_scalar, f"Write has {self.width} values — use .values for vector Writes"
        return self.values[0]

    @property
    def width(self) -> int:
        return len(self.values)

    @property
    def is_vector(self) -> bool:
        return len(self.values) > 1

    @property
    def is_scalar(self) -> bool:
        return len(self.values) == 1

    def deps(self) -> tuple[str, ...]:
        # The stored values, then the SSA names the index reads — a scatter ``out[(int)idx] = v``
        # reads ``idx``. Axis Vars pass through as no-ops, exactly as in ``Load.deps``.
        return tuple(dict.fromkeys((*self.values, *(v for e in self.index for v in e.free_vars()))))

    def external_writes(self) -> tuple[str, ...]:
        return (self.output,)

    def rename_buffers(self, rename):  # noqa: ANN001 — see ``Stmt.rename_buffers``
        new = rename.get(self.output, self.output)
        return self if new == self.output else self.replace(output=new)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(index=exprs)

    @cached_property
    def has_side_effects(self) -> bool:
        return True

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return Write(
            output=self.output,
            index=tuple(sigma.apply(e).rename(rename) for e in self.index),
            values=tuple(rename(n) for n in self.values),
            value_dtype=self.value_dtype,
            atomic=self.atomic,
            swizzle=self.swizzle,
        )


@dataclass(frozen=True)
class SelectBranch:
    """One branch of a ``Select`` body statement."""

    value: str  # SSA name when predicate holds
    select: Expr  # predicate over axis Vars

    def replace(self, **changes) -> SelectBranch:  # noqa: ANN003 — field values
        return replace(self, **changes)


@dataclass(frozen=True)
class Select(Stmt):
    """Coord-predicated value binding — replaces Mux.

    At each iteration coord, exactly one branch's ``select`` predicate
    should be True; its ``value`` is bound to ``name`` in the SSA scope.
    Branches are expected to be disjoint; later branches act as
    catch-alls when no earlier predicate matches.
    """

    pure = True  # a coord-predicated value binding — no effect; legal inside a stored ``Lambda``

    name: str
    branches: tuple[SelectBranch, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("Select.branches must be non-empty")

    def deps(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for b in self.branches for name in (b.value, *sorted(b.select.free_vars()))))

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def exprs(self) -> tuple[Expr, ...]:
        return tuple(b.select for b in self.branches)

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(branches=tuple(SelectBranch(b.value, select) for b, select in zip(self.branches, exprs, strict=True)))

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        # A branch condition reads coordinates exactly as a Load index does: σ first, then the rename
        # — an axis renumbering that reached the indices and not the conditions left the causal mask
        # comparing loop variables its kernel no longer bound.
        return Select(
            name=rename(self.name),
            branches=tuple(SelectBranch(value=rename(b.value), select=sigma.apply(b.select).rename(rename)) for b in self.branches),
        )


def mask_select_predicate(select: Select) -> Expr | None:
    """The mask branch of an ordered two-way comparison, or ``None`` for another Select form."""
    if len(select.branches) != 2:
        return None
    keep, mask = select.branches
    if not isinstance(keep.select, BinaryExpr) or not isinstance(mask.select, BinaryExpr):
        return None
    comparison = mask.select
    if keep.select.left == comparison.right and keep.select.right == comparison.left:
        swapped = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
        if comparison.op not in swapped:
            return None
        comparison = BinaryExpr(swapped[comparison.op], comparison.right, comparison.left)
    if keep.select.left != comparison.left or keep.select.right != comparison.right:
        return None
    if (keep.select.op, comparison.op) not in (("<", ">="), ("<=", ">"), (">", "<="), (">=", "<")):
        return None
    return mask.select


@dataclass(frozen=True)
class ZeroPrologue(Stmt):
    """Zero another launch's atomic accumulator from THIS kernel — the delegated zero-init
    (``lowering/cuda/005_delegate_zero_init``).

    An atomic accumulator (atomic ``Write`` / ``RegStore`` output)
    must be zero before its kernel launches; the runtime's per-launch ``zero_outputs`` memset
    costs a CUDA-graph MEMSET node per site. This stmt rides a kernel that launches strictly
    BEFORE the accumulator's kernel in the same stream (a dataflow predecessor — topological
    launch order guarantees it), so stream serialization makes the zero happen-before the
    atomics and the MEMSET node disappears. Every thread of the grid writes a stride of it, ahead
    of the kernel's own work: one CTA alone left the grid waiting on a serial tail (130 us for a
    4 MB accumulator on the H100). The zero is over raw 32-bit words (``words = nbytes / 4`` —
    all-zero bytes are 0.0 in every buffer dtype, the same argument the runtime memset makes), so
    the render never needs the buffer's dtype.

    ``dst`` names the target buffer — it joins the carrying kernel's ``outputs`` (and thus its
    signature / ``arg_names``) WITHOUT a graph edge: an edge would be a cycle (the target is
    the downstream kernel's own output). The slab planner learns the earlier first-write
    through ``CudaOp.zero_prologues`` instead."""

    dst: str  # the delegated buffer (the downstream kernel's atomic accumulator)
    words: int  # 32-bit word count (nbytes / 4; the delegating rule guarantees divisibility)

    def external_writes(self) -> tuple[str, ...]:
        return (self.dst,)

    def rename_buffers(self, rename):  # noqa: ANN001 — see ``Stmt.rename_buffers``
        new = rename.get(self.dst, self.dst)
        return self if new == self.dst else self.replace(dst=new)

    @cached_property
    def has_side_effects(self) -> bool:
        return True

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        # Buffer name + word count only — like ``Write.output``, ``dst`` is a buffer, and buffers
        # are not SSA names this rewrite renames. Identity, so a body carrying a delegated
        # zero-init can normalize (``structural_key`` / ``cache_key`` on the carrying kernel).
        return self


@dataclass(frozen=True)
class OutputSpec:
    """One output write at the kernel boundary — the effect the stored term does not carry.
    ``write`` is the store verbatim (target buffer, index template, stored value names, the atomic
    flag — holding the ``Write`` whole keeps every field lossless); ``TileOp.output_specs`` owns
    the tuple. The term places its stores itself (``Fold.lower``), each after the term defining
    its value; a stream spelled without the term reconstitutes them (``apply_output_specs``).
    ``sweep`` names the output loop NEST the store rode in the source, outermost first (``()`` is
    a plain kernel-tail store): the axes a store alone is evaluated over when no term declares
    them (a broadcast, ``o[j] = acc``), and the loops a stream wraps around the trailing stmts
    reading them (``apply_output_specs`` — a prefix two paths share is one source loop, opened
    once). The path is stored because it is the one fact extraction destroys: a write's index
    names its coordinates, not the order its loops nested in the source (DeepSeek-V4 post4096's
    gate stream nests two write-only sweeps beside the outer store). The term route takes only
    the path's extents: ``Fold.lower`` places a store from its write's coordinates and orders the
    free loops it opens itself."""

    write: Write
    sweep: tuple[Axis, ...] = ()
    # The store's guard: it runs only where ``when`` holds. A scatter pointed past the end of its
    # buffer stores nothing (``045_lower_scatter``); the guard reads the same values the index does.
    when: Expr | None = None

    def stmt(self) -> Stmt:
        """The store as it stands in a statement stream: the write, under its guard when it has one."""
        from emmy.compiler.ir.stmt.blocks import Cond  # noqa: PLC0415 — blocks imports this module

        return self.write if self.when is None else Cond(cond=self.when, body=(self.write,))

    @property
    def reads(self) -> tuple[str, ...]:
        """Every name the store reads: the values it writes, then what its coordinates and its guard name — a
        scatter's row is a value computed like any other, and the term it comes from has to keep it."""
        guard = () if self.when is None else (self.when,)
        return tuple(dict.fromkeys((*self.write.values, *(name for expr in (*self.write.index, *guard) for name in expr.free_vars()))))
