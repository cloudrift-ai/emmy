"""Canonical argument names and operation clusters for statement-body identity.

The output is digest material and must never be executed. All semantics-preserving canonicalization
remains in :mod:`emmy.compiler.ir.stmt.normalize`; identity labels the graph that normalization
ordered by once more, with the external buffers colored by type instead of left bare, and
materializes that order without reading a spelling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.normalize import _canonicalize_exprs, normalize_body, rename_ssa_sequential, sort_commutative_args
from emmy.compiler.structural import digest, form

__all__ = ["Identity", "canonicalize_identity"]


@dataclass(frozen=True)
class Identity:
    """A body's identity material: its canonical body over ``b0, b1, …``, the type of each of
    those roles, and which external buffer fills each — spelling, so outside the key."""

    body: Body
    #: External buffer names in canonical rank order: ``arguments[i]`` fills ``b<i>``.
    arguments: tuple[str, ...]
    #: Each role's type in the same order; ``None`` when none was given.
    roles: tuple[object, ...]

    @cached_property
    def key(self) -> str:
        """The digest: the canonical body rendered structurally, beside the typed roles."""
        return digest(form(self.body), self.roles)


def canonicalize_identity(stmts: Body, *, cluster: bool = False, types: Mapping[str, object] | None = None) -> Identity:
    """Rename external arguments by canonical rank and optionally collapse operations to compute-unit clusters.

    ``types`` colors each external buffer in the relation graph, so differently typed roles never
    share a rank and the order the buffers were declared in never reaches the key.
    """
    stmts = Body.coerce(stmts)
    if cluster:
        stmts = _canonicalize_op_clusters(stmts)
    stmts = normalize_body(stmts)
    labeling = stmts._ordering.label(None if types is None else types.get)
    ordered, _ = labeling.materialize(spelled=False)
    resources = labeling.resources()
    rename = {name: f"b{index}" for index, name in enumerate(resources)}
    body = Body.coerce(sort_commutative_args(rename_ssa_sequential(ordered.rename_buffers(rename))))
    # Canonical renaming can reverse lexical order in remaining commutative expressions,
    # so normalize their spelling after the final rename as well.
    body = _canonicalize_exprs(body)
    return Identity(body, resources, tuple(None if types is None else types.get(name) for name in resources))


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
