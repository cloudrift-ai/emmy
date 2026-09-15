"""Canonical argument names and operation clusters for statement-body identity.

The output is digest material and must never be executed. All semantics-preserving canonicalization
remains in :mod:`emmy.compiler.ir.stmt.normalize`.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.normalize import _renormalize_external_order, normalize_body
from emmy.compiler.ir.stmt.order import _canonical_resource_order

__all__ = ["canonicalize_identity"]


# ---------------------------------------------------------------------------
# Identity-only canonicalization: external arguments and operation clusters.
# ---------------------------------------------------------------------------


def canonicalize_identity(stmts: Body, *, cluster: bool = False) -> Body:
    """Rename external arguments and optionally collapse operations to compute-unit clusters."""
    stmts = Body.coerce(stmts)
    if cluster:
        stmts = _canonicalize_op_clusters(stmts)
    stmts = normalize_body(stmts, hoist=False)
    resources = _canonical_resource_order(stmts)
    if not resources:
        return stmts
    rename = {name: f"b{index}" for index, name in enumerate(resources)}
    return _renormalize_external_order(stmts.rename_buffers(rename), frozenset(rename.values()))


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
