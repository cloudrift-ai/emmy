"""``Fold`` — the ONE reduce term: ``reduce(⊕) ∘ map(f)`` in the λ-foldMap spelling.

The whole stored vocabulary of Tile IR, and a PURE term throughout: an optional iteration
``axis``, a pure ``lift`` :class:`~emmy.compiler.ir.pure.lam.Lambda`, the monoid's ``(init, base)``
pair with the optional ``twist`` recipe that conjugates it, and a tuple of ``operands`` — the
closed inputs, each an edge bound positionally to a lift param. Every reading (the ⊕ itself, the
map at zero axes, the bilinear :class:`ContractionView`, the :class:`SlabView` leaf, the serial
step) is DERIVED from those params; nothing else is stored.

The term and everything it answers live in the ``emmy-pure`` crate (``emmy.emmy_ir.Fold``): the fields and
formation, every reading and view, ``merge`` / ``step``, ``fuse`` (the recipe matching), the tree normalization
(``normalized``) and the lowering (``lower``). Terms are interned there, so two terms with the same stored parts are
one object. Python adds one member, ``rewrite``, which takes Python functions.

The fold's algebra is the twisted monoid of :mod:`~emmy.compiler.ir.pure.twist`, spelled in one
vocabulary with the recipes: ``lift`` is the per-element contribution to the BASE state, ``base``
the base monoid's componentwise ⊕, and ``twist`` the recipe carrying the bijection ``psi`` onto the
stable carrier. The ⊕ the fold actually folds with is then derived —
``combine(x, y) = psi(psi_inv(x) base psi_inv(y))`` — in the stable spelling the recipe certifies,
and ``twist=None`` is the planar case where ``psi`` is the identity and ``combine`` IS ``base``.

Nothing here is a :class:`~emmy.compiler.ir.stmt.Statement`. A composed step — flash's ``Σ Q·K``
ahead of its ``Σ_j P·V``, split-K's sliced contraction — is reached through ``operands``, and its
POSITION in the emitted nest is produced by the derivation (:meth:`Fold.lower` places every term
at the shallowest scope binding its free coordinates, operands ahead of their readers), not by
sitting in a statement list. The term becomes statements in exactly one place, :meth:`Fold.lower`,
which also places the kernel's boundary stores, each after the term defining its value.
See ``ir/ARCHITECTURE.md``, "Pure terms vs statements".

The schedule is deliberately absent: an accepted, site-indexed ``Schedule`` lives on the
``TileOp`` boundary (``ir/tile/ir.py``), so the term is IMMUTABLE across the whole schedule search
and kernel identity — the Loop IR the term lowers to — is the algebra alone.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import emmy.emmy_ir as _ir
from emmy.compiler.ir.pure._fields import MISSING, declare
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.pure.twist import Twist
from emmy.compiler.ir.sigma import Sigma

#: The symbolic extents of the program being compiled. The launch binds them, so a term reads one free — like a
#: coordinate the grid binds — and never as a value some level defines: a scatter's guard compares its index with its
#: destination's extent. The pipeline binds them per graph (:func:`binding_extents`).
EXTENTS: ContextVar[frozenset[str]] = ContextVar("EXTENTS", default=frozenset())


@contextmanager
def binding_extents(names):
    """Read ``names`` as symbolic extents, on top of those already bound, for the enclosed block."""
    token = EXTENTS.set(EXTENTS.get() | frozenset(names))
    try:
        yield
    finally:
        EXTENTS.reset(token)


Fold = _ir.Fold
SlabView = _ir.SlabView
ContractionView = _ir.ContractionView
ReductionView = _ir.ReductionView
AffineView = _ir.AffineView
declare(
    Fold,
    (
        ("operands", tuple[Fold, ...], ()),
        ("init", tuple[float | str, ...], ()),
        ("lift", Lambda, MISSING),
        ("cells", tuple[str, ...], ()),
        ("base", Lambda | None, None),
        ("twist", Twist | None, None),
        ("observe", Lambda | None, None),
    ),
)


def _rewrite(self: Fold, rename, sigma: Sigma = Sigma.IDENTITY, axis_fn=None) -> Fold:  # noqa: ANN001
    """This term with every SSA name through ``rename`` (a mapping or a function), every expression through ``sigma``
    and every axis through ``axis_fn``, the way ``Statement.rewrite`` rewrites a statement: the lift and the monoid
    rename in lockstep (params track the operand names positionally, the combine's results ARE the accumulator names),
    and σ is applied hygienically to the term's own binder — the reduce axis this term binds is a different variable
    from any σ key spelled alike, so its mapping is dropped for the subtree. Operand edges are terms and rewrite
    through this same entry. A term takes a substitution's mapping and its folding; one that does more is refused."""
    if type(sigma).after is not Sigma.after:
        raise TypeError(f"a term takes a substitution's mapping and folding alone, got {type(sigma).__name__}")
    names = dict(rename) if hasattr(rename, "get") else {name: rename(name) for name in self._names()}
    axes = None if axis_fn is None else [(axis, axis_fn(axis)) for axis in self._bound_axes()]
    return self._rewrite(names, sigma.mapping, sigma.folding, axes)


Fold.rewrite = _rewrite

__all__ = [
    "AffineView",
    "ContractionView",
    "Fold",
    "ReductionView",
    "SlabView",
]
