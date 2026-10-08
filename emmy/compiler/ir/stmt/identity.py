"""A statement body's identity: the digest of its value-numbered scope tree, and the roles its buffers fill.

The output is digest material and must never be executed. All semantics-preserving canonicalization remains in
:mod:`emmy.compiler.ir.stmt.normalize`; identity numbers the normal form's values with every external buffer keyed by
its type and its use instead of its spelling (:func:`emmy.compiler.ir.stmt.values.digest`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.normalize import normalize_body
from emmy.compiler.ir.stmt.values import digest

__all__ = ["Identity", "canonicalize_identity"]


@dataclass(frozen=True)
class Identity:
    """A body's identity material: its key, the external buffer that fills each role ``b<i>`` — spelling, so outside
    the key — and each role's type."""

    key: str
    #: External buffer names in canonical rank order: ``arguments[i]`` fills role ``i``.
    arguments: tuple[str, ...]
    #: Each role's type in the same order; ``None`` when none was given.
    roles: tuple[object, ...]


def canonicalize_identity(stmts: Body, *, cluster: bool = False, types: Mapping[str, object] | None = None) -> Identity:
    """Key the normal form of ``stmts``, optionally with operations collapsed to compute-unit clusters.

    ``types`` colors each external buffer, so differently typed roles never share a rank and the order the buffers
    were declared in never reaches the key.
    """
    stmts = normalize_body(stmts)
    if cluster:
        stmts = normalize_body(_canonicalize_op_clusters(stmts))
    key, arguments = digest(stmts, None if types is None else types.get)
    return Identity(key, arguments, tuple(None if types is None else types.get(name) for name in arguments))


# ---------------------------------------------------------------------------
# Pass: collapse ops to their compute-unit cluster representative.
# ---------------------------------------------------------------------------


def _canonicalize_op_clusters(stmts: Body) -> Body:
    """Replace operation fields with their compute-unit cluster representative.

    Generic dataclass field inspection covers every operation-bearing statement without coupling
    identity to individual IR dialects. The result is digest material and must not be executed.
    """
    from dataclasses import fields, is_dataclass  # noqa: PLC0415

    from emmy.compiler.ir.elementwise import ElementwiseImpl, cluster_representative  # noqa: PLC0415

    def fn(s: Stmt) -> Stmt:
        if not is_dataclass(s):
            return s
        changes: dict[str, ElementwiseImpl] = {}
        for f in fields(s):
            val = getattr(s, f.name)
            if isinstance(val, ElementwiseImpl):
                rep = cluster_representative(val)
                if rep != val:
                    changes[f.name] = rep
        if not changes:
            return s
        return replace(s, **changes)

    return stmts.map(fn)
