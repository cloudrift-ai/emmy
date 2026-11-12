"""Move a Load only one guarded block reads into that block.

A Load that only a guarded block (a ``Cond`` with no else) reads is issued by every cell, the ones the guard skips
included: a scatter pointed at its spare row loads a value it throws away. Inside the block, a cell the guard skips
never reads it. The body's normal form keeps a Load on the side of a guard it stands on, so the move is this pass's.

Where a Load stands among its siblings is not this pass's: a kernel body is in normal form, which orders them.

It decides nothing: the kernel computes the same values and issues fewer loads, so it answers to no knob.
Idempotent: a Load already inside its guard has no guarded block to move into.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Cond, Load, Statement
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    body, changed = _walk(op.body)
    if not changed:
        raise RuleSkipped("no Load stands outside the one guard that reads it")
    return replace(op, body=body)


def _walk(body: Body) -> tuple[Body, bool]:
    """``body`` with each Load only one guarded block reads moved into that block, then the same inside every nested
    body: a Load moved into a guard can be read by one guard inside it."""
    stmts = tuple(body)
    # A carried staged copy (``Load.carried``) is ordered against the barrier and the unpack that reads its vector,
    # neither of which its ``defines`` names: it keeps its place, and so does the rest of that body.
    guarded: dict[int, list[Load]] = {}
    if not any(isinstance(s, Load) and s.carried for s in stmts):
        reads = [_nested_deps(s) for s in stmts]
        for i, load in enumerate(stmts):
            if not isinstance(load, Load):
                continue
            names = frozenset(load.defines())
            readers = [j for j in range(len(stmts)) if j != i and names & reads[j]]
            if len(readers) == 1:
                guard = stmts[readers[0]]
                if isinstance(guard, Cond) and not guard.else_body and not names.intersection(guard.deps()):
                    guarded.setdefault(readers[0], []).append(load)
    moved = {id(load) for loads in guarded.values() for load in loads}
    changed = bool(moved)
    out: list[Statement] = []
    for j, s in enumerate(stmts):
        if id(s) in moved:
            continue
        if j in guarded:
            s = s.with_bodies((Body((*guarded[j], *s.body)), s.else_body))
        nested = s.nested()
        if nested:
            walked = [_walk(b) for b in nested]
            if any(c for _, c in walked):
                s = s.with_bodies(tuple(b for b, _ in walked))
                changed = True
        out.append(s)
    return (Body(tuple(out)) if changed else body), changed


def _nested_deps(stmt: Statement) -> frozenset[str]:
    """Return SSA dependencies used anywhere in ``stmt``'s lexical subtree."""
    deps = set(stmt.deps())
    for nested in stmt.nested():
        for child in nested:
            deps.update(_nested_deps(child))
    return frozenset(deps)
