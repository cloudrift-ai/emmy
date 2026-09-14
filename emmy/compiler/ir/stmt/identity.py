"""Canonical argument names and operation clusters for statement-body identity.

The output is digest material and must never be executed. All semantics-preserving canonicalization
remains in :mod:`emmy.compiler.ir.stmt.normalize`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from itertools import product

from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.normalize import (
    _orders_modulo_transpositions,
    _pure_tokens,
    normalize_body,
    sort_commutative_args,
)

__all__ = ["canonicalize_identity"]


# ---------------------------------------------------------------------------
# Identity-only canonicalization: external arguments and operation clusters.
# ---------------------------------------------------------------------------


def canonicalize_identity(stmts: Body, *, cluster: bool = False) -> Body:
    """Rename external arguments and optionally collapse operations to compute-unit clusters."""
    stmts = Body.coerce(stmts)
    if cluster:
        stmts = _canonicalize_op_clusters(stmts)
    buffers: dict[str, None] = {}
    for stmt in stmts.iter():
        for name in (*stmt.external_reads(), *stmt.external_writes()):
            buffers.setdefault(name, None)

    if not buffers:
        return normalize_body(stmts, hoist=False)

    from emmy.compiler.structural import form  # noqa: PLC0415

    # A buffer's occurrence contexts give it an isomorphism-invariant initial role without merging
    # the other buffers. Merging them to one placeholder would invent memory dependencies while the
    # sibling-order pass runs (an input and an unrelated output would suddenly alias).
    names = {name for stmt in stmts.iter() for name in (*stmt.defines(), *stmt.deps(), *stmt.binds_axes())}
    abstract_names = {name: "__name__" for name in names}
    roles: dict[str, str] = {}
    for focus in buffers:
        focused = stmts.rename_buffers({name: "__self__" if name == focus else "__other__" for name in buffers})
        pure_tokens = _pure_tokens(focused)
        definitions = {name: stmt for stmt in focused.iter() if stmt.pure for name in stmt.defines()}
        contexts = []
        for stmt in focused.iter():
            if "__self__" not in (*stmt.external_reads(), *stmt.external_writes()):
                continue
            if stmt.pure:
                contexts.append(pure_tokens[id(stmt)])
                continue
            dependency_roles = {
                name: f"__dep_{pure_tokens[id(owner)]}" for name in stmt.deps() if (owner := definitions.get(name)) is not None
            }
            renamed = stmt.rename({**abstract_names, **dependency_roles})
            contexts.append(repr(form(sort_commutative_args(Body((renamed,)))[0])))
        roles[focus] = repr(tuple(sorted(contexts)))

    groups: dict[str, list[str]] = {}
    for name, role in roles.items():
        groups.setdefault(role, []).append(name)

    # A role tie is not assumed to be a symmetry. Try every ordering inside each tied partition and
    # choose the least complete form. Proven transposition symmetries are quotiented first: assigning
    # labels to interchangeable buffers cannot change the result and must not cause factorial work.
    ordered_groups = [groups[role] for role in sorted(groups)]
    base = repr(form(normalize_body(stmts, hoist=False)))

    def buffer_orders(group: list[str]) -> Iterator[tuple[str, ...]]:
        def interchangeable(left: str, right: str) -> bool:
            swapped = stmts.rename_buffers({left: right, right: left})
            return repr(form(normalize_body(swapped, hoist=False))) == base

        yield from _orders_modulo_transpositions(group, interchangeable)

    best: tuple[str, Body] | None = None
    for choices in product(*(buffer_orders(group) for group in ordered_groups)):
        names = tuple(name for group in choices for name in group)
        renamed = stmts.rename_buffers({name: f"b{index}" for index, name in enumerate(names)})
        candidate = normalize_body(renamed, hoist=False)
        rendered = repr(form(candidate))
        if best is None or rendered < best[0]:
            best = (rendered, candidate)
    assert best is not None
    return best[1]


# ---------------------------------------------------------------------------
# Pass: collapse ops to their compute-unit cluster representative.
# ---------------------------------------------------------------------------


def _canonicalize_op_clusters(stmts: Body) -> Body:
    """Replace every ``ElementwiseImpl`` field on every stmt with its
    cluster representative from :func:`cluster_representative`.

    The pass walks ``stmts`` with :meth:`Body.map` and uses
    ``dataclasses.fields`` to locate any field currently holding an
    ``ElementwiseImpl`` (covers ``Init.op`` / ``Assign.op`` /
    ``Accum.op`` without coupling this module to those IR dialects). A
    fold algebra and the kernel-IR cross-thread combine
    stmts (``WarpShuffle`` / ``TreeHalve``) carry their op inside an
    ``Assign`` program (``merge`` / ``combine_states``), already
    canonicalized at the carrier before lowering. The replacement is
    destructive — the resulting body is only safe to consume from
    :attr:`Body.structural_key()`.
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
