"""Canonical structural identity for normalized statement bodies.

This module contains identity-only transforms. Their output is digest material and must never be
executed. Executable body normalization remains in :mod:`emmy.compiler.ir.stmt.normalize`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from itertools import product

from emmy.compiler.ir.expr import (
    BinaryExpr,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    SimplifyCtx,
    TernaryExpr,
    Var,
    affine_form,
)
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Init, SelectBranch
from emmy.compiler.ir.stmt.normalize import (
    _ordered_sibling_defs,
    _orders_modulo_transpositions,
    _sibling_defs_uses,
    rename_ssa_sequential,
    sort_commutative_args,
)

__all__ = ["canonicalize_identity"]


# ---------------------------------------------------------------------------
# Identity-only canonicalization: external arguments + dependency-valid statement order.
# ---------------------------------------------------------------------------


def canonicalize_identity(stmts: Body) -> Body:
    """Canonicalize the name-free choices used only by structural identity.

    The ordinary normalization above preserves independent statement order and external buffer
    names. Both are right for an executable body, but together they let two traces of one kernel
    key apart: changing which argument is loaded first changes both its source spelling and the
    sequential SSA names.

    Give each external buffer a structural role from its access and downstream-use contexts, order
    buffers by that role, then dependency-sort siblings while retaining true memory and accumulator
    conflicts. A final SSA rename removes the construction order exposed by those two changes.
    The result is identity material only; callers must never execute it."""
    stmts = Body.coerce(stmts)
    buffers: dict[str, None] = {}
    for stmt in stmts.iter():
        for name in (*stmt.external_reads(), *stmt.external_writes()):
            buffers.setdefault(name, None)

    if not buffers:
        return _canonicalize_fixed_buffers(stmts)

    from emmy.compiler.structural import form  # noqa: PLC0415

    # A buffer's occurrence contexts give it an isomorphism-invariant initial role without merging
    # the other buffers. Merging them to one placeholder would invent memory dependencies while the
    # sibling-order pass runs (an input and an unrelated output would suddenly alias).
    names = {name for stmt in stmts.iter() for name in (*stmt.defines(), *stmt.deps(), *stmt.binds_axes())}
    abstract_names = {name: "__name__" for name in names}
    roles: dict[str, str] = {}
    for focus in buffers:
        focused = stmts.rename_buffers({name: "__self__" if name == focus else "__other__" for name in buffers})
        pure_tokens = _pure_identity_tokens(focused)
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
    base = repr(form(_canonicalize_fixed_buffers(stmts)))

    def buffer_orders(group: list[str]) -> Iterator[tuple[str, ...]]:
        def interchangeable(left: str, right: str) -> bool:
            swapped = stmts.rename_buffers({left: right, right: left})
            return repr(form(_canonicalize_fixed_buffers(swapped))) == base

        yield from _orders_modulo_transpositions(group, interchangeable)

    best: tuple[str, Body] | None = None
    for choices in product(*(buffer_orders(group) for group in ordered_groups)):
        names = tuple(name for group in choices for name in group)
        renamed = stmts.rename_buffers({name: f"b{index}" for index, name in enumerate(names)})
        candidate = _canonicalize_fixed_buffers(renamed)
        rendered = repr(form(candidate))
        if best is None or rendered < best[0]:
            best = (rendered, candidate)
    assert best is not None
    return best[1]


def _canonicalize_fixed_buffers(stmts: Body) -> Body:
    """Canonicalize pure dataflow after external buffers have fixed labels."""
    from emmy.compiler.structural import form  # noqa: PLC0415

    stmts = _canonicalize_identity_exprs(stmts)
    best: tuple[str, Body] | None = None
    for ordered in _canonicalize_order_variants(stmts):
        candidate = Body.coerce(sort_commutative_args(rename_ssa_sequential(ordered)))
        rendered = repr(form(candidate))
        if best is None or rendered < best[0]:
            best = (rendered, candidate)
    assert best is not None
    return best[1]


def _canonicalize_identity_exprs(stmts: Body) -> Body:
    """Canonicalize equivalent expression spellings without changing an executable body."""
    from dataclasses import fields  # noqa: PLC0415

    from emmy.compiler.structural import form  # noqa: PLC0415

    commutative = frozenset({"+", "*", "==", "!=", "&&", "||", "&", "|", "^"})
    dual = {">": "<", ">=": "<="}
    axis_names = stmts.axis_names

    def affine(expr: Expr) -> Expr:
        variables = expr.free_vars()
        # Reassociation and coefficient folding are exact for integer coordinates. An SSA value
        # may be floating point, where changing the operation tree changes rounding and kernel work.
        if not variables or not variables <= axis_names or (decomposed := affine_form(expr, variables)) is None:
            return expr
        anchor, coefficients = decomposed
        anchor = anchor.simplify(SimplifyCtx.empty())
        terms: list[Expr] = []
        for name, coefficient in sorted(coefficients.items()):
            variable = Var(name)
            terms.append(variable if coefficient == 1 else BinaryExpr("*", Literal(coefficient, "int"), variable))
        if not (isinstance(anchor, Literal) and anchor.value == 0):
            terms.append(anchor)
        if not terms:
            return Literal(0, "int")
        result = terms[0]
        for term in terms[1:]:
            result = BinaryExpr("+", result, term)
        return result

    def expression(expr: Expr) -> Expr:
        if isinstance(expr, BinaryExpr):
            left, right = expression(expr.left), expression(expr.right)
            op = expr.op
            if op in dual:
                op, left, right = dual[op], right, left
            if op in commutative and repr(form(right)) < repr(form(left)):
                left, right = right, left
            result = BinaryExpr(op, left, right)
            return affine(result) if op in {"+", "-", "*"} else result
        if isinstance(expr, FuncCallExpr):
            return FuncCallExpr(expr.name, tuple(expression(arg) for arg in expr.args))
        if isinstance(expr, TernaryExpr):
            return TernaryExpr(expression(expr.cond), expression(expr.if_true), expression(expr.if_false))
        if isinstance(expr, CastExpr):
            return CastExpr(expr.dtype, expression(expr.expr))
        return expr

    def value(item):
        if isinstance(item, Expr):
            return expression(item)
        if isinstance(item, tuple):
            return tuple(value(member) for member in item)
        if isinstance(item, SelectBranch):
            return SelectBranch(value=item.value, select=expression(item.select))
        return item

    def statement(stmt: Stmt) -> Stmt:
        changes = {field.name: value(getattr(stmt, field.name)) for field in fields(stmt)}
        return replace(stmt, **changes)

    return Body.coerce(stmts).map(statement)


def _pure_identity_tokens(stmts: Body) -> dict[int, str]:
    """Name- and order-free structural token for every pure definition.

    The forward half describes what a statement computes. The reverse half describes every use of
    its results. Equal producers that feed different operations or operand positions therefore get
    different tokens before the exact ordering search has to branch.
    """
    from emmy.compiler.structural import digest, form  # noqa: PLC0415

    definitions: dict[str, tuple[Stmt, int]] = {}
    for stmt in stmts.iter():
        if stmt.pure:
            for slot, name in enumerate(stmt.defines()):
                definitions.setdefault(name, (stmt, slot))

    forward_tokens: dict[int, str] = {}
    visiting_forward: set[int] = set()

    def forward_token(stmt: Stmt) -> str:
        identity = id(stmt)
        if identity in forward_tokens:
            return forward_tokens[identity]
        if identity in visiting_forward:
            return "__cycle__"
        visiting_forward.add(identity)
        mapping = {name: f"__own{slot}" for slot, name in enumerate(stmt.defines())}
        for name in stmt.deps():
            owner = definitions.get(name)
            if owner is not None:
                producer, slot = owner
                mapping[name] = f"__dep_{digest(forward_token(producer))}_{slot}"
            else:
                mapping[name] = "__free__"
        renamed = stmt.rename(mapping)
        renamed = sort_commutative_args(Body((renamed,)))[0]
        result = repr(form(renamed))
        visiting_forward.remove(identity)
        forward_tokens[identity] = result
        return result

    consumers: dict[str, list[Stmt]] = {name: [] for name in definitions}
    for stmt in stmts.iter():
        if stmt.pure:
            forward_token(stmt)
        for name in dict.fromkeys(stmt.deps()):
            if name in consumers:
                consumers[name].append(stmt)

    reverse_tokens: dict[str, str] = {}
    visiting_reverse: set[str] = set()

    def reverse_token(name: str) -> str:
        if name in reverse_tokens:
            return reverse_tokens[name]
        if name in visiting_reverse:
            return "__cycle__"
        visiting_reverse.add(name)
        contexts = []
        for consumer in consumers[name]:
            mapping = {defined: f"__own{slot}" for slot, defined in enumerate(consumer.defines())}
            for dependency in consumer.deps():
                if dependency == name:
                    mapping[dependency] = "__self__"
                    continue
                owner = definitions.get(dependency)
                if owner is not None:
                    producer, slot = owner
                    mapping[dependency] = f"__other_{digest(forward_token(producer))}_{slot}"
                else:
                    mapping[dependency] = "__free__"
            renamed = consumer.rename(mapping)
            renamed = sort_commutative_args(Body((renamed,)))[0]
            downstream = tuple(reverse_token(defined) for defined in consumer.defines() if defined in definitions)
            contexts.append((repr(form(renamed)), downstream))
        result = repr(tuple(sorted(contexts)))
        visiting_reverse.remove(name)
        reverse_tokens[name] = result
        return result

    tokens: dict[int, str] = {}
    for stmt in stmts.iter():
        if stmt.pure:
            tokens[id(stmt)] = repr((forward_token(stmt), tuple(reverse_token(name) for name in stmt.defines())))
    return tokens


def _canonicalize_order_variants(stmts: Body) -> Iterator[Body]:
    """Yield canonical candidates for dependency- and effect-valid sibling orderings."""
    stmts = Body.coerce(stmts)
    statement_choices = []
    for stmt in stmts:
        children = stmt.nested()
        if not children:
            statement_choices.append((stmt,))
            continue
        child_choices = [tuple(_canonicalize_order_variants(child)) for child in children]
        statement_choices.append(tuple(stmt.with_bodies(choice) for choice in product(*child_choices)))

    for statements in product(*statement_choices):
        yield from (Body(choice) for choice in _canonicalize_sibling_order_variants(list(statements)))


def _canonicalize_sibling_order_variants(stmts: list[Stmt]) -> Iterator[tuple[Stmt, ...]]:
    """Yield every unresolved canonical Kahn order for one sibling scope."""
    if len(stmts) <= 1:
        yield tuple(stmts)
        return

    from emmy.compiler.structural import form  # noqa: PLC0415

    pure_tokens = _pure_identity_tokens(Body(stmts))
    members = tuple((stmt, *(member for child in stmt.nested() for member in child.iter())) for stmt in stmts)
    all_names = {name for subtree in members for member in subtree for name in (*member.defines(), *member.deps(), *member.binds_axes())}
    abstract = {name: "__name__" for name in all_names}
    tokens = {
        id(stmt): pure_tokens.get(
            id(stmt),
            repr(form(sort_commutative_args(Body((stmt.rename(abstract),)))[0])),
        )
        for stmt in stmts
    }

    definitions: dict[str, list[int]] = {}
    for index, stmt in enumerate(stmts):
        for name in _sibling_defs_uses(stmt)[0]:
            definitions.setdefault(name, []).append(index)

    def defining_stmt(name: str, consumer: int) -> int | None:
        sites = definitions.get(name, ())
        preceding = [site for site in sites if site < consumer]
        if preceding:
            return preceding[-1]
        return next((site for site in sites if site != consumer), None)

    incoming = []
    for index, stmt in enumerate(stmts):
        sources = {source for name in _sibling_defs_uses(stmt)[1] if (source := defining_stmt(name, index)) is not None}
        incoming.append(sources)
    for reader, stmt in enumerate(stmts):
        for name in _sibling_defs_uses(stmt)[1]:
            for later_definition in definitions.get(name, ()):
                if later_definition > reader:
                    incoming[later_definition].add(reader)

    def resources(stmt: Stmt) -> tuple[set[str], set[str], set[str]]:
        members = tuple(stmt for child in stmt.nested() for stmt in child.iter()) or (stmt,)
        reads = {name for member in members for name in member.external_reads()}
        writes = {name for member in members for name in member.external_writes()}
        state = {name for member in members for name in getattr(member, "carried_names", lambda: ())()}
        if isinstance(stmt, Init):
            state.update(stmt.defines())
        return reads, writes, state

    effects = [resources(stmt) for stmt in stmts]
    for later in range(len(stmts)):
        later_reads, later_writes, later_state = effects[later]
        for earlier in range(later):
            reads, writes, state = effects[earlier]
            if writes & (later_reads | later_writes) or reads & later_writes or state & later_state:
                incoming[later].add(earlier)

    edges = {(source, target) for target, sources in enumerate(incoming) for source in sources}

    def interchangeable(left: int, right: int) -> bool:
        left_defs = _ordered_sibling_defs(stmts[left])
        right_defs = _ordered_sibling_defs(stmts[right])
        if len(left_defs) != len(right_defs):
            return False
        rename = {**dict(zip(left_defs, right_defs, strict=True)), **dict(zip(right_defs, left_defs, strict=True))}
        swap = {left: right, right: left}
        if {(swap.get(a, a), swap.get(b, b)) for a, b in edges} != edges:
            return False
        for index, stmt in enumerate(stmts):
            rewritten = sort_commutative_args(Body((stmt.rename(rename),)))[0]
            if form(rewritten) != form(stmts[swap.get(index, index)]):
                return False
        return True

    def walk(remaining: frozenset[int], ordered: tuple[int, ...]) -> Iterator[tuple[Stmt, ...]]:
        if not remaining:
            yield tuple(stmts[index] for index in ordered)
            return
        ready = [index for index in remaining if not incoming[index] & remaining]
        if not ready:
            yield tuple(stmts)
            return
        least = min(tokens[id(stmts[index])] for index in ready)
        tied = [index for index in ready if tokens[id(stmts[index])] == least]
        representatives: list[int] = []
        for selected in tied:
            if not any(interchangeable(selected, earlier) for earlier in representatives):
                representatives.append(selected)
        for selected in representatives:
            yield from walk(remaining - {selected}, (*ordered, selected))

    yield from walk(frozenset(range(len(stmts))), ())
