"""Sigma — Expression-substitution helper used across all IR layers.

Wraps the bare ``dict[str, Expression]`` substitution form so call sites can use
``.apply(e)`` / ``.extend(name, expression)`` / ``.restrict(names)`` instead of
open-coding ``substitute`` and key flattening. Used by ``Statement.rewrite``
(every IR layer) and the Loop-IR fusion splicer.

Lives at the top of ``ir/`` rather than under any one IR package because
``Statement.rewrite`` is shared infrastructure — same reasoning as
``ir/axis.py`` and ``ir/stmt.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from emmy.compiler.ir.expression import Expression, SimplifyCtx


@dataclass(frozen=True, eq=False)
class Sigma:
    """Axis substitution: axis name → replacement ``Expression``.

    Carried by the fusion splicer as it walks from the consumer into the
    producer's expression chain — each producer axis gets rewritten into
    the consumer's namespace.

    Equality and hashing are by canonical form — sorted ``(name,
    str(expression))`` pairs — so two Sigmas built from different dicts
    that denote the same substitution compare equal and share a hash
    bucket.

    ``after`` is the extension point: a subclass that does more than substitute (fold under a range context,
    abstract a read to a coordinate) overrides it, and ``Statement.rewrite`` applies it to every expression of the
    subtree once the mapping has landed, under the mapping's hygiene, with the subclass's state carried through
    ``restrict``.
    """

    mapping: dict[str, Expression] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = tuple(sorted((k, str(v)) for k, v in self.mapping.items()))
        object.__setattr__(self, "_key", key)

    def _from_key(self, mapping: dict[str, Expression], key: tuple[tuple[str, str], ...]) -> Sigma:
        """A derived substitution whose canonical entries are already known, with this one's other state."""
        sigma = copy.copy(self)
        object.__setattr__(sigma, "mapping", mapping)
        object.__setattr__(sigma, "_key", key)
        return sigma

    def apply(self, e: Expression) -> Expression:
        return self.after(e.substitute(self.mapping))

    def after(self, e: Expression) -> Expression:
        """What this substitution does to an expression once its mapping has landed: nothing."""
        return e

    def reduce(self, e: Expression, ctx: SimplifyCtx) -> Expression:
        """Substitute then simplify under ``ctx``. Use when the substitution
        is expected to expose constant folding — e.g. anchor / coefficient
        probes that pin axes to literals. Distinct from ``Expression.eval`` which
        evaluates a fully-bound expression to a concrete int/float."""
        return e.substitute(self.mapping).simplify(ctx)

    def extend(self, name: str, expression: Expression) -> Sigma:
        mapping = {**self.mapping, name: expression}
        existing = ((axis, canonical) for axis, canonical in self._key if axis != name)
        key = tuple(sorted((*existing, (name, str(expression)))))
        return self._from_key(mapping, key)

    def restrict(self, names: set[str]) -> Sigma:
        """Return a new Sigma keeping only bindings whose axis name is in ``names``."""
        mapping = {k: v for k, v in self.mapping.items() if k in names}
        key = tuple(entry for entry in self._key if entry[0] in names)
        return self._from_key(mapping, key)

    def get(self, name: str) -> Expression | None:
        return self.mapping.get(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sigma):
            return NotImplemented
        return self._key == other._key  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(self._key)  # type: ignore[attr-defined]


Sigma.IDENTITY = Sigma({})

__all__ = ["Sigma"]
