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
    _renormalize_external_order,
    normalize_body,
    sort_commutative_args,
)

__all__ = ["canonicalize_identity"]


def _buffer_roles(stmts: Body) -> dict[str, str]:
    """Name-free access and downstream-use role for every external buffer."""
    from emmy.compiler.structural import form  # noqa: PLC0415

    members = tuple(stmts.iter())
    buffers = tuple(
        dict.fromkeys(name for stmt in members for name in (*stmt.external_reads(), *stmt.external_writes()))
    )
    if not buffers:
        return {}

    masked = stmts.rename_buffers(dict.fromkeys(buffers, "__buffer__"))
    masked_members = tuple(masked.iter())
    pure_tokens = _pure_tokens(masked_members)
    definitions = {name: stmt for stmt in masked_members if stmt.pure for name in stmt.defines()}
    abstract_names = {name: "__name__" for name in masked.ssa_defs | masked.ssa_uses | masked.axis_names}
    contexts: dict[str, list[tuple[tuple[int, ...], tuple[int, ...], str]]] = {name: [] for name in buffers}
    for original, statement in zip(members, masked_members, strict=True):
        reads, writes = original.external_reads(), original.external_writes()
        resources = tuple(dict.fromkeys((*reads, *writes)))
        if not resources:
            continue
        if statement.pure:
            token = pure_tokens[id(statement)]
        else:
            dependency_roles = {
                name: f"__dep_{pure_tokens[id(owner)]}"
                for name in statement.deps()
                if (owner := definitions.get(name)) is not None
            }
            renamed = statement.rename({**abstract_names, **dependency_roles})
            token = repr(form(sort_commutative_args(Body((renamed,)))[0]))
        for resource in resources:
            contexts[resource].append(
                (
                    tuple(index for index, name in enumerate(reads) if name == resource),
                    tuple(index for index, name in enumerate(writes) if name == resource),
                    token,
                )
            )
    return {name: repr(tuple(sorted(contexts[name]))) for name in buffers}


# ---------------------------------------------------------------------------
# Identity-only canonicalization: external arguments and operation clusters.
# ---------------------------------------------------------------------------


def canonicalize_identity(stmts: Body, *, cluster: bool = False) -> Body:
    """Rename external arguments and optionally collapse operations to compute-unit clusters."""
    stmts = Body.coerce(stmts)
    if cluster:
        stmts = _canonicalize_op_clusters(stmts)
    stmts = normalize_body(stmts, hoist=False)
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
    groups: dict[str, list[str]] = {}
    for name, role in _buffer_roles(stmts).items():
        groups.setdefault(role, []).append(name)

    # A role tie is not assumed to be a symmetry. Try every ordering inside each tied partition and
    # choose the least complete form. Proven transposition symmetries are quotiented first: assigning
    # labels to interchangeable buffers cannot change the result and must not cause factorial work.
    ordered_groups = [groups[role] for role in sorted(groups)]
    base = repr(form(stmts))

    def buffer_orders(group: list[str]) -> Iterator[tuple[str, ...]]:
        def interchangeable(left: str, right: str) -> bool:
            swapped = stmts.rename_buffers({left: right, right: left})
            return repr(form(_renormalize_external_order(swapped, frozenset({left, right})))) == base

        yield from _orders_modulo_transpositions(group, interchangeable)

    def candidate(choices: tuple[tuple[str, ...], ...]) -> tuple[str, Body]:
        names = tuple(name for group in choices for name in group)
        rename = {name: f"b{index}" for index, name in enumerate(names)}
        renamed = stmts.rename_buffers(rename)
        normalized = _renormalize_external_order(renamed, frozenset(rename.values()))
        return repr(form(normalized)), normalized

    return min(candidate(choices) for choices in product(*(buffer_orders(group) for group in ordered_groups)))[1]


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
