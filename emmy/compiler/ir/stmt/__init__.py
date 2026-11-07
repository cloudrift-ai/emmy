"""Shared statement primitives — the leaves and control flow used across every IR layer.

The statement kinds, ``Body``, the text form, normalization and identity are implemented in Rust (the ``emmy-ir`` crate,
reached through the ``emmy.emmy_ir`` extension); this package is their one face to the rest of the compiler.
``__all__`` is the whole interface: the rest of the compiler imports these names and no submodule. The kinds every
layer shares are here; the Kernel IR's thirty are exported by ``emmy.compiler.ir.kernel.ir``. Rendering a statement
as C / CUDA source is not statement business and lives in :mod:`emmy.compiler.ir.render`.

The boundary: Rust owns the data and Python reads it through accessors; a field read hands back a held object, so a
statement read twice from a body is one object; only strings, numbers, booleans, tuples of those and the module's own
classes cross it. The few members that take a Python function and keep its results (``rename`` by a function,
``rewrite``, ``Body.fold``) are written here over the Rust reads; ``Body.map`` and ``map_expressions`` call the function
from Rust through the fallible walks and hold nothing of it afterwards.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import emmy.emmy_ir as _ir
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Expression
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.wire import register as _register

Statement = _ir.Statement
Body = _ir.Body
Accum = _ir.Accum
Affine = _ir.Affine
Assign = _ir.Assign
Call = _ir.Call
Carry = _ir.Carry
Cond = _ir.Cond
Init = _ir.Init
Let = _ir.Let
Load = _ir.Load
Loop = _ir.Loop
Pre = _ir.Pre
Select = _ir.Select
SelectBranch = _ir.SelectBranch
StridedLoop = _ir.StridedLoop
Subroutine = _ir.Subroutine
Write = _ir.Write
ZeroPrologue = _ir.ZeroPrologue

parse_body = _ir.parse_body
pretty_body = _ir.pretty_body
normalize_body = _ir.normalize_body
fold_affine = _ir.fold_affine
simplify_body = _ir.simplify_body
prepare_body = _ir.prepare_body
free_names = _ir.free_names
exposed_defines = _ir.exposed_defines
rename_free = _ir.rename_free
binary = _ir.binary
mask_select_predicate = _ir.mask_select_predicate
ordering_constraints = _ir.ordering_constraints
dtype_promote = _ir.dtype_promote

_register(Body)


def fields_of(cls: type) -> tuple[str, ...]:
    """The names of the fields a kind holds, in declaration order: what ``stmt.replace(**changes)`` takes."""
    return cls._fields


@dataclass(frozen=True)
class Identity:
    """A body's identity material: its key, the external buffer that fills each role ``b<i>`` — spelling, so outside
    the key — and each role's type."""

    key: str
    #: External buffer names in canonical rank order: ``arguments[i]`` fills role ``i``.
    arguments: tuple[str, ...]
    #: Each role's type in the same order; ``None`` when none was given.
    roles: tuple[object, ...]


def canonicalize_identity(
    stmts: Body | Iterable[Statement], *, cluster: bool = False, types: Mapping[str, object] | None = None
) -> Identity:
    """Key the normal form of ``stmts``, optionally with operations collapsed to compute-unit clusters.

    ``types`` colors each external buffer, so differently typed roles never share a rank and the order the buffers
    were declared in never reaches the key. A color enters the key as its ``repr``.
    """
    body = Body.coerce(stmts)
    colors = None if types is None else {name: repr(value) for name, value in types.items()}
    key, arguments = body._identity(cluster, colors)
    return Identity(key, arguments, tuple(None if types is None else types.get(name) for name in arguments))


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
    names its coordinates, not the order its loops nested in the source. The term route takes only
    the path's extents: ``Fold.lower`` places a store from its write's coordinates and orders the
    free loops it opens itself."""

    write: Write
    sweep: tuple[Axis, ...] = ()
    # The store's guard: it runs only where ``when`` holds. A scatter pointed past the end of its
    # buffer stores nothing (``045_lower_scatter``); the guard reads the same values the index does.
    when: Expression | None = None

    def stmt(self) -> Statement:
        """The store as it stands in a statement stream: the write, under its guard when it has one."""
        return self.write if self.when is None else Cond(cond=self.when, body=(self.write,))

    @property
    def reads(self) -> tuple[str, ...]:
        """Every name the store reads: the values it writes, then what its coordinates and its guard name — a
        scatter's row is a value computed like any other, and the term it comes from has to keep it."""
        guard = () if self.when is None else (self.when,)
        return tuple(
            dict.fromkeys((*self.write.values, *(name for expression in (*self.write.index, *guard) for name in expression.free_vars())))
        )


# ---------------------------------------------------------------------------
# The members that take a Python function, written over the Rust reads.
# ---------------------------------------------------------------------------


def _mapping(names, spelled: Iterable[str]) -> dict[str, str]:  # noqa: ANN001 — a mapping or a ``str -> str`` function
    """``names`` as the dict a rename takes: a mapping as it is, a function asked once per name in ``spelled``."""
    if hasattr(names, "get"):
        return dict(names)
    return {name: names(name) for name in spelled}


def _rename(self: Statement, names) -> Statement:  # noqa: ANN001
    """α-RENAME this subtree — binders and uses, SSA names and axis names, through one map (a mapping or a function).

    Capture-free by construction: a binder renamed is renamed everywhere it scopes, so the rename passes THROUGH a
    scope that binds a mapped name. The counterpart of :meth:`substitute`.
    """
    return self._rename(_mapping(names, self._names()))


def _rewrite(self: Statement, rename_ssa, sigma: Sigma = Sigma.IDENTITY, axis_fn=None) -> Statement:  # noqa: ANN001
    """Rename, σ-substitute and axis-map this subtree in one rewrite: every SSA name through ``rename_ssa``, every
    expression through ``sigma`` (hygienically: a mapping a statement's own binder shadows is dropped for its
    subtree), every loop axis through ``axis_fn``. Prefer :meth:`rename` / :meth:`substitute`."""
    names = _mapping(rename_ssa, self._names())
    mapping = sigma.mapping if isinstance(sigma, Sigma) else sigma
    axes = None
    if axis_fn is not None:
        bound = [axis for stmt in Body((self,)).iter() for axis in stmt.bound_axes()]
        axes = [(axis, axis_fn(axis)) for axis in bound]
    rewritten = self._rewrite(names, mapping, axes)
    if isinstance(sigma, Sigma) and type(sigma) is not Sigma:
        rewritten = _apply_sigma(rewritten, sigma)
    return rewritten


def _apply_sigma(stmt: Statement, sigma: Sigma) -> Statement:
    """``sigma.after`` over every expression of the subtree, under the mapping's hygiene: what a sigma that does more
    than substitute adds to the rewrite, once the substitution itself has landed."""
    shadowed = stmt.binds_axes() & sigma.mapping.keys()
    if shadowed:
        sigma = sigma.restrict(sigma.mapping.keys() - shadowed)
    stmt = stmt._with_expressions([sigma.after(expression) for expression in stmt.expressions()])
    nested = stmt.nested()
    return stmt.with_bodies(tuple(Body(_apply_sigma(child, sigma) for child in body) for body in nested)) if nested else stmt


def _fold[T](self: Body, fn: Callable[[Statement, tuple[T | None, ...], frozenset[str]], T]) -> dict[int, T]:
    """Generic backward dataflow over this body's def-use DAG: ``fn(stmt, child_T, bound)`` per statement in source
    order, nested bodies before their wrapper, ``child_T`` one entry per name in ``stmt.deps()`` (``None`` for a read
    no statement of this body defines), ``bound`` the axis names the enclosing statements bind. The memo is keyed by
    ``id(stmt)``: a statement placed in a body is the same object every time it is read."""
    memo: dict[int, T] = {}
    defs = self.definitions

    def walk(body: Body, bound: frozenset[str]) -> None:
        for stmt in body:
            child_bound = bound | stmt.binds_axes()
            for child in stmt.nested():
                walk(child, child_bound)
            child_T: tuple[T | None, ...] = tuple(memo.get(id(defs[d])) if d in defs else None for d in stmt.deps())
            memo[id(stmt)] = fn(stmt, child_T, bound)

    walk(self, frozenset())
    return memo


def _identity(self: Body, *, structural: bool = True, types: Mapping[str, object] | None = None) -> Identity:
    """This body's identity material: the canonical body over ``b0, b1, …``, which external buffer fills each role,
    and the role's type. Cached with the body in Rust, so an op rebuilt over this body reads the identity its
    predecessor computed."""
    return canonicalize_identity(self, cluster=structural, types=types)


def _structural_key(self: Body, *, structural: bool = True, types: Mapping[str, object] | None = None) -> str:
    """The canonical digest for structural-equivalence queries: two bodies that differ only by SSA / axis names,
    commutative-arg order or external-buffer names produce the same key; with ``structural=True`` the op within a
    compute-unit cluster normalizes away too (the schedule-equivalent reading), ``structural=False`` keeps the exact
    ops. ``types`` colors the buffers, so bodies reading one structure through differently typed roles key apart."""
    return canonicalize_identity(self, cluster=structural, types=types).key


def _repr(self: Statement) -> str:
    """The constructor form, ``Load(name='x', input='X', ...)``: what ``repr`` spells and the graph's dict form reads back
    with ``eval``; ``str`` is the text."""
    return f"{type(self).__name__}({', '.join(f'{name}={getattr(self, name)!r}' for name in fields_of(type(self)))})"


def _repr_body(self: Body) -> str:
    return f"Body({tuple(self)!r})"


Statement.__repr__ = _repr
_ir.SelectBranch.__repr__ = _repr
_ir.FragmentArg.__repr__ = _repr
Body.__repr__ = _repr_body
Statement.rename = _rename
Statement.rewrite = _rewrite
Body.fold = _fold
Body.identity = _identity
Body.structural_key = _structural_key

__all__ = [
    "Accum",
    "Affine",
    "Assign",
    "Body",
    "Call",
    "Carry",
    "Cond",
    "Init",
    "Let",
    "Load",
    "Loop",
    "OutputSpec",
    "Pre",
    "Select",
    "SelectBranch",
    "Statement",
    "StridedLoop",
    "Subroutine",
    "Write",
    "ZeroPrologue",
    "binary",
    "canonicalize_identity",
    "dtype_promote",
    "exposed_defines",
    "fields_of",
    "fold_affine",
    "free_names",
    "mask_select_predicate",
    "normalize_body",
    "ordering_constraints",
    "prepare_body",
    "parse_body",
    "pretty_body",
    "rename_free",
    "simplify_body",
]
