"""``Stmt``, the abstract base of every IR body statement, and the dtype rule its kinds share.

Concrete kinds live in ``leaves`` (pure compute) and ``blocks`` (control flow); ``Body`` and the body-level
normalization live in ``body`` and ``normalize``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING

from emmy.compiler.dtype import INTEGER_NAMES
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.structural import Structural
from emmy.compiler.wire import Wire

if TYPE_CHECKING:
    from emmy.compiler.ir.stmt.body import Body


INDENT = "    "


_INTEGER_ELEMENTWISE_OPS = frozenset(
    {
        "add",
        "subtract",
        "negative",
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
    }
)


def dtype_promote(op_name: str, arg_dtypes: list[str]) -> str:
    """Promote an elementwise op's arg dtypes to a single result dtype.

    The rule the render of an ``Assign`` applies: the result is the
    common dtype iff every arg agrees; any disagreement promotes to f32.
    Floating point keeps the historical "all f16 → f16, otherwise f32"
    rule. Integer arithmetic/bit manipulation must instead retain a common
    integer dtype: promoting packed words to f32 loses bits before the CUDA
    renderer sees the operation. Comparisons and logical masks deliberately
    stay on the floating-point rule.

    The ``030_stamp_types`` pass uses the same rule to stamp ``Assign.dtype`` on the IR.
    """
    if arg_dtypes and op_name in _INTEGER_ELEMENTWISE_OPS and all(d in INTEGER_NAMES for d in arg_dtypes):
        # Keep packed arithmetic in the integer family.  NumPy promotes u64+i32 to f64 because
        # neither integer type contains the other's full range; that is disastrous for bit
        # extraction.  Use the usual C-style rank/sign rule instead: the widest unsigned type
        # wins a tie, while a strictly wider signed type can represent the narrower unsigned
        # range.  This gives u16+u32 -> u32, u64+i32 -> u64, and i64+u32 -> i64.
        widths = {"u8": 8, "i16": 16, "u16": 16, "i32": 32, "u32": 32, "i64": 64, "u64": 64}
        signed_width = max((widths[d] for d in arg_dtypes if d.startswith("i")), default=0)
        unsigned_width = max((widths[d] for d in arg_dtypes if d.startswith("u")), default=0)
        if unsigned_width >= signed_width:
            return f"u{unsigned_width}"
        return f"i{signed_width}"
    if arg_dtypes and all(d == "f16" for d in arg_dtypes):
        return "f16"
    return "f32"


def spell_literal(value: float) -> str:
    """The one spelling of a literal: an integer without a point, anything else as Python repr."""
    value = float(value)
    return str(int(value)) if value.is_integer() and abs(value) < 2**53 else repr(value)


def _identity_rename(name: str) -> str:
    """Default SSA rename — :meth:`Stmt.substitute` renames nothing."""
    return name


def _axis_identity(a: Axis) -> Axis:
    """Default ``axis_fn`` for ``Loop.rewrite`` / ``StridedLoop.rewrite``."""
    return a


# ---------------------------------------------------------------------------
# Stmt — abstract base
# ---------------------------------------------------------------------------


class Stmt(Structural, Wire):
    """Base class for IR body statements.

    Every concrete Stmt implements:

    - ``deps()`` — SSA names this stmt reads.
    - ``rewrite(rename_ssa, sigma)`` — return a copy with SSA names mapped
      through ``rename_ssa`` and Expr subterms σ-substituted.
    - ``nested()`` — child statement bodies for tree traversal (default:
      no children; block-structured stmts override).
    """

    # Purity trait — True iff this stmt is a pure value binding, a legal member of a stored
    # ``Lambda`` body (the λ-foldMap term vocabulary). Declared on the interface with a
    # CONSERVATIVE ``False`` default and per-class opt-in (no isinstance whitelist): ``Load`` /
    # ``Assign`` declare pure; the effectful / scope-bound kinds (``Accum``, ``Write``, ``Init``,
    # ``Loop``…) never do; the structural nodes (``Fold`` / zero-axis ``Fold``) declare pure — a term is a
    # value, its internals are its own. A NEW stmt kind is excluded from lambdas until it
    # declares itself. ``Lambda.__post_init__`` is the enforcing formation gate.
    pure: bool = False

    # Rolled-deps trait — True iff :meth:`deps` already reports the ENTIRE subtree's reads,
    # scope-correctly (every nested lambda's params subtracted). A read walk then takes ``deps()``
    # and must not re-walk ``nested()``: the flat re-walk cannot see an inner lambda's params, so
    # a factored operand cone's result read inside the lift leaks out as a phantom capture.
    # Conservative default; ``Fold`` (the memoized scoped rollup) opts in.
    deps_deep: bool = False

    def deps(self) -> tuple[str, ...]:
        """SSA names this stmt reads — its 'requirements'.

        Default: ``()`` (no SSA deps). Mirrors :meth:`defines` so stmts
        with no SSA inputs (kernel-IR primitives like ``Sync`` /
        ``CpAsyncCommit`` / ``MbarrierInit``) don't need to override.
        """
        return ()

    def defines(self) -> tuple[str, ...]:
        """SSA names this stmt produces — its 'bindings'.

        Default: ``()`` (no SSA def). Name-bearing leaves (``Load``,
        ``Assign``, ``Accum``, ``Init``, ``Select``) override to return
        ``(self.name,)``. Block stmts (``Loop`` / ``StridedLoop`` /
        ``Tile`` / ``Cond``) inherit the default — their bodies define
        names, but the wrapper itself doesn't bind one. ``Write``
        also inherits the default since it writes to a buffer, not
        an SSA value.

        Together with :meth:`deps` this is the def-use surface that
        body-level dependency analyses query (without resorting to
        ``getattr(s, "name", None)`` patterns).
        """
        return ()

    def external_reads(self) -> tuple[str, ...]:
        """External-buffer names this stmt reads from. Default: ``()``.

        Overrides: ``Load`` returns ``(self.input,)``; ``CpAsyncCopy``
        returns ``(self.src,)``; ``TmaDescriptor`` returns
        ``(self.src_buf,)``; ``Stage`` returns the source-Load inputs
        from its body. The reads of any name covered by some stmt's
        :meth:`local_decls` (smem / staged buffers) get filtered out at
        :class:`BodyOp` aggregation time, so leaves don't need to know
        about the surrounding declarations."""
        return ()

    def rename_buffers(self, rename) -> Stmt:  # noqa: ANN001 — any str->str mapping
        """This stmt with every external-buffer field renamed through ``rename`` (a
        ``str -> str`` mapping; absent names keep). Default: identity — the counterpart of
        :meth:`external_reads` / :meth:`external_writes`, overridden by exactly the leaves that
        override those (``Load.input``, ``Write.output``, ``ZeroPrologue.dst``,
        ``CpAsyncCopy.src``, ``TmaDescriptor.src_buf``). Wrapper stmts
        need no override: callers rename whole bodies through ``Body.rename_buffers``, whose
        recursive ``map`` reaches every nested leaf."""
        del rename
        return self

    def external_writes(self) -> tuple[str, ...]:
        """External-buffer names this stmt writes to. Default: ``()``.

        Overrides: ``Write`` returns ``(self.output,)``. Smem writes
        (``CpAsyncCopy`` / ``TmaLoad``) are local — they target a name
        covered by some :class:`Smem` decl's :meth:`local_decls` — and
        get filtered out at aggregation rather than fighting to suppress
        them at the leaf."""
        return ()

    def local_decls(self) -> tuple[str, ...]:
        """Buffer names this stmt declares as kernel-local — the body's
        external-buffer aggregator filters reads / writes naming these
        names out of its inputs / outputs (they live inside the kernel,
        not on the signature). Default: ``()``.

        Overrides: ``Smem`` returns ``(self.name,)`` (kernel-IR shared
        buffers); ``Stage`` returns ``(self.name,)`` (tile-IR staged
        buffers — materialized into an ``Smem`` later)."""
        return ()

    def rename(self, names) -> Stmt:
        """α-RENAME this subtree — binders and uses, SSA names and axis names, through one map.

        Capture-free by construction: a binder renamed is renamed everywhere it scopes, so the
        rename passes THROUGH a scope that binds a mapped name — renaming that binder is the
        point. Renaming an ``Axis`` means keeping its extent and window and changing its name,
        which is this method's business rather than every caller's.

        The counterpart of :meth:`substitute`, and the reason neither needs a hygiene flag: what
        a caller is doing is stated by which one it calls.
        """
        lookup = names.get if hasattr(names, "get") else None

        def rename_name(name: str) -> str:
            return lookup(name, name) if lookup is not None else names(name)

        def rename_expr(expr: Expr | None) -> Expr | None:
            return None if expr is None else expr.rename(rename_name)

        def rename_axis_tree(axis: Axis, seen: frozenset[int]) -> Axis:
            window = axis.window
            if window is not None:
                parent = window.parent
                if parent is not None:
                    parent = (
                        replace(parent, name=rename_name(parent.name))
                        if id(parent) in seen
                        else rename_axis_tree(parent, seen | {id(axis)})
                    )
                window = replace(
                    window,
                    parent=parent,
                    base=rename_expr(window.base),
                    bound=rename_expr(window.bound),
                )
            return replace(axis, name=rename_name(axis.name), window=window)

        return self.rewrite(rename_name, Sigma.IDENTITY, lambda axis: rename_axis_tree(axis, frozenset()))

    def substitute(self, coords: Sigma) -> Stmt:
        """β-SUBSTITUTE free coordinates by expressions — the split's reindex, the per-cell fill.

        Stops at a scope that re-binds a substituted name: that binder introduces a DIFFERENT
        variable, and substituting through it is capture. A nested reduce over ``k`` under an outer
        ``k`` is the case — a blind σ made its 128-element reduce read one slab element 128 times.
        """
        return self.rewrite(_identity_rename, coords)

    def rewrite(
        self,
        rename_ssa: Callable[[str], str],
        sigma: Sigma = Sigma.IDENTITY,
        axis_fn: Callable[[Axis], Axis] = _axis_identity,
    ) -> Stmt:
        """Rename, σ-substitute and axis-map this subtree: every SSA name (binding and reads) through
        ``rename_ssa``, every Expr subterm through ``sigma``, every loop axis through ``axis_fn``. Prefer
        :meth:`rename` / :meth:`substitute`, which say which of the two operations a caller means.

        σ is applied HYGIENICALLY: the mapping for a name this stmt binds is dropped before its
        subtree is rewritten, because that binder introduces a different variable. A RENAME is
        unaffected — it travels through ``rename_ssa``/``axis_fn`` and renames the binder itself, which
        is exactly why it may pass where a substitution may not.

        Each kind states how its own fields rewrite in :meth:`_rewrite`; children go back through this entry.
        """
        shadowed = self.binds_axes() & sigma.mapping.keys() if sigma.mapping else ()
        if shadowed:
            sigma = Sigma({name: value for name, value in sigma.mapping.items() if name not in shadowed})
        return self._rewrite(rename_ssa, sigma, axis_fn)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see :meth:`rewrite`
        raise NotImplementedError(f"rewrite not implemented for {type(self).__name__}")

    def nested(self) -> tuple[Body, ...]:
        """Child statement bodies for tree traversal.

        Default: no children (leaf stmt). Block-structured stmts override
        to return their body tuple(s) — ``Loop`` returns ``(self.body,)``;
        ``Cond`` returns ``(self.body, self.else_body)``; ``Tile`` returns
        ``(self.body,)``.

        ``iter_body`` walks all IR layers via this single method — every
        node knows its own children, so the walker doesn't need to
        switch on type.
        """
        return ()

    @cached_property
    def has_side_effects(self) -> bool:
        """True iff executing this stmt produces an externally observable
        effect (a buffer write). For compound stmts (Loop / StridedLoop /
        Tile / Cond), True iff any nested stmt does.

        Use this to gate transforms that change execution count
        (hoisting, loop interchange, predication): a side-effecting
        stmt run N times instead of M is observable, so it pins the
        enclosing iteration to its current scope.

        Note: ``Accum`` / ``Init`` are *not* side-effecting in this
        sense — they're scope-bound (their semantics depend on which
        Loop encloses them) but moving the *whole enclosing block* is
        safe. Hoisting passes that want to move a Loop containing an
        carried state need a separate scope-bound check on the leaf, not
        ``has_side_effects`` on the wrapper."""
        return any(c.has_side_effects for sub in self.nested() for c in sub)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        """Write-side counterpart to :meth:`nested`. Return a copy of this
        stmt with its child bodies replaced by ``bodies`` (positionally
        matching :meth:`nested`'s order).

        Default: leaves have no children, so ``bodies`` must be empty and
        ``self`` is returned unchanged. Block-structured stmts override
        to rebuild themselves from the new bodies. Used by ``Body.map``
        to recurse without an isinstance ladder over the block-stmt set.
        """
        assert not bodies, f"{type(self).__name__}.with_bodies: leaf stmt got {len(bodies)} bodies"
        return self

    def binds_axes(self) -> frozenset[str]:
        """Axes this stmt introduces into scope for its nested bodies.

        Default: ``frozenset()`` (no axis binding). ``Loop`` / ``StridedLoop``
        return ``{self.axis.name}``; ``Tile`` returns the axis names of every
        ``BoundAxis``; ``Cond`` keeps the default. Used by ``Body.fold``
        to thread the bound-axis set through the def-use walk without an
        isinstance ladder over the block-stmt set.
        """
        return frozenset()

    def exprs(self) -> Iterable[Expr]:
        """Direct Expr fields of this stmt (non-recursive into nested bodies).

        Default: ``()``. Concrete stmts override to surface their carried
        Expr trees: ``Load`` / ``Write`` yield ``self.index``; ``Select``
        yields each branch's predicate; ``Cond`` yields ``self.cond``;
        ``StridedLoop`` yields ``self.start`` and ``self.step`` (when an
        ``Expr``); Tile-IR ``Stage`` yields its source-index template.
        ``Loop`` / ``Tile`` keep the default — their Exprs live inside
        their bodies, which the fold's tree traversal reaches separately.

        The third slice of the per-stmt analysis surface alongside
        :meth:`deps` (SSA reads) and :meth:`nested` (child bodies). The
        fold callback uses ``exprs()`` to pull free Vars at each stmt
        for direct axis contributions, with ``free_vars()`` filtered by
        the ``bound`` set threaded through :meth:`binds_axes`.
        """
        return ()

    def replace(self, **changes) -> Stmt:  # noqa: ANN003 — field values
        """A copy of this stmt with the named fields changed — the one way a statement is edited."""
        return replace(self, **changes)

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        """Write-side counterpart to :meth:`exprs`: this stmt with its expressions replaced by ``exprs``,
        positionally. A kind that reports expressions and is met by an expression map overrides it."""
        if exprs:
            raise NotImplementedError(f"{type(self).__name__} cannot be rebuilt from mapped expressions")
        return self

    def pretty(self, indent: str = "") -> list[str]:
        """Render this stmt as a list of indented lines.

        Block-structured stmts recurse into their bodies via
        ``child.pretty(indent + INDENT)``; leaves return a single line.
        Subclasses override to control formatting; default surfaces the
        class name as a placeholder for any stmt that forgot to override.
        """
        return [f"{indent}<unrecognized {type(self).__name__}>"]


def pretty_body(body: Body, indent: str = "") -> list[str]:
    """Flatten ``stmt.pretty(indent)`` over a body sequence."""
    out: list[str] = []
    for s in body:
        out.extend(s.pretty(indent))
    return out
