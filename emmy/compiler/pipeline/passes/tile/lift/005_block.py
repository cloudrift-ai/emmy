"""A carried walk in blocks of steps: the kernel in two levels of steps, before the lift.

A kernel that carries a state and sums over the rows of it each step reads, at step ``t``, rows that steps before
``t`` wrote. Cut the steps into blocks of ``BLOCK`` and that sum splits in two at the block's first step:

- the rows before the block are finished when it starts. Their part of every step's sum is computed once a block,
  for all of the block's steps at once: a second state, carried by the blocks, written beside the steps of the
  block before;
- the rows from the block's first on are few, and the steps keep summing over them as they go.

The result is still one kernel: an outer level of steps over the blocks, the walk's own steps under it as a
further level (``Loop.seed`` false), and the second state. Nothing here decides the kernel set: the cut pass takes
a kernel of two levels apart (``_split.realize_levels``).

Whether the split is right is one dependence question, asked of the kernel's own comparisons (``ir/guards.py``):
no step of a block writes a cell that a later step of the block reads among the rows before the block.

``BLOCK`` is pin-only: nothing enumerates it, so no compile takes this form unless told to.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.constraints import always, can_hold
from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, TernaryExpression, Var, apply
from emmy.compiler.ir.guards import carried
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Let, Loop, Pre, Select, SelectBranch, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import BLOCK

PATTERN = [Pattern("root", LoopOp)]

_TRUE = Literal(True, "bool")


def rewrite(match: Match, root: Node, ctx=None) -> LoopOp:
    del match, ctx
    width = BLOCK.read_int(0)
    if width < 1 or not root.op.body.carries:
        raise RuleSkipped("no block width is pinned, or the kernel carries no state")
    try:
        return replace(root.op, body=in_blocks(root.op.body, width))
    except ValueError as why:
        raise RuleSkipped(str(why)) from None


def _int(value: int) -> Literal:
    return Literal(value, "int")


def _op(op: str, left: Expression, right: Expression | int) -> BinaryExpression:
    return BinaryExpression(op, left, _int(right) if isinstance(right, int) else right)


def in_blocks(body: Body, width: int) -> Body:
    """``body``, a kernel that carries one state, with its steps in blocks of ``width``.

    Raises ``ValueError`` for a walk the form does not hold: its steps are not counted, they do not write one row
    each, it sums over its state in more than one place or under no test that ends the sum, a step reads among the
    rows before its block what the block writes, or one block holds every step."""
    facts = carried(body)
    loop, carry, step = facts.loop, facts.carry, facts.step
    moved = facts.moving()
    sums = [
        stmt
        for stmt in loop.body.iter()
        if isinstance(stmt, Loop) and stmt.is_reduce and any(isinstance(read, Pre) and read.carrier == carry.name for read in stmt.body)
    ]
    if not loop.axis.extent.is_static or moved is None or moved[1] < 0 or len(sums) != 1:
        raise ValueError("the block form takes a counted walk that writes one row a step and sums over its state in one place")
    (summed,) = sums
    steps, over, rows = loop.axis.extent.as_static(), summed.axis.name, summed.axis.extent.as_static()
    lets = {stmt.name: stmt.value for stmt in body if isinstance(stmt, Let)}
    accums = [stmt for stmt in summed.body if isinstance(stmt, Accum)]
    tests = [stmt for stmt in summed.body if isinstance(stmt, Select) and len(stmt.branches) == 2 and stmt.branches[1].select == _TRUE]
    if (
        len(accums) != 1
        or len(tests) != 1
        or accums[0].value != Var(tests[0].name)
        or lets.get(tests[0].branches[1].value) != accums[0].init
    ):
        raise ValueError("the block form takes a sum of one term under one test, which adds nothing where the test fails")
    (total,), (term,) = accums, tests
    test = term.branches[0].select
    # What the sum is computed from, and nothing else of the step.
    only = loop.body.map(lambda stmt: None if isinstance(stmt, Carry | Write) else stmt).pruned((total.name,))

    # The split is right when the rows before a block are finished: no step of the block, ``then``, writes a cell
    # that a later one reads for the part of its sum over them.
    start, then, here = Var("__start"), Var("__then"), Var(step)
    block = (_op("<=", _int(0), start), _op("<=", start, then), _op("<", then, here), _op("<", Var(over), start))
    feeding = {read.name for read in only.iter() if isinstance(read, Pre)}
    for read in facts.reads:
        cell = {step: then} | dict(zip(facts.cells, read.index, strict=True))
        if read.name in feeding and any(
            can_hold(*read.guard, *facts.bounds, *block, *(condition.substitute(cell) for condition in guard)) for guard in facts.written
        ):
            raise ValueError("a step reads, among the rows before its block, a cell the block writes")
    # How far past the block's first step its steps sum: the rows the steps themselves keep summing over.
    within = (_op("<=", start, here), _op("<", here, _op("+", start, width)))
    reach = next((e for e in range(1, rows) if always(_op("<", Var(over), _op("+", start, e)), test, *facts.bounds, *within)), None)
    blocks = -(-steps // width)
    if reach is None or blocks < 2:
        raise ValueError("the block form gains nothing: the sum reaches every row from a block's first step, or one block holds the walk")

    outer, local, ahead, work = f"{step}__block", f"{step}__local", f"{step}__ahead", f"{carry.name}__before"
    first = _op("*", _int(width), Var(outer))

    def held(at: Expression) -> Expression:
        # A step past the walk's last is that last one again: it reads in range, and writes nothing. A width the
        # walk's length is no multiple of leaves such steps in the last block; the normal form folds the test away
        # wherever the block is known.
        return TernaryExpression(_op("<", at, steps), at, _int(steps - 1))

    position, offset = moved
    moving = facts.cells[position]
    others = tuple(Var(name) for name in facts.cells if name != moving)

    # The walk's own steps, a further level under the blocks: the sum over the rows from the block's first on,
    # and the part over the rows before it read from where the block before left it.
    row = _op("+", first, Var(over))
    in_range = Sigma({over: TernaryExpression(_op("<", row, rows), row, _int(0))})
    kept, live = f"{carry.value}__kept", f"{carry.value}__live"

    def walked(stmt):
        if isinstance(stmt, Loop) and stmt.axis.name == over:
            parts = []
            for part in stmt.body:
                if isinstance(part, Pre):
                    parts.append(part.substitute(in_range))
                elif part.name == term.name:
                    ends = BinaryExpression("&&", test, _op("<", Var(over), rows)).substitute(Sigma({over: row}))
                    parts.append(part.replace(branches=(SelectBranch(term.branches[0].value, ends), term.branches[1])))
                else:
                    parts.append(
                        part.replace(name=f"{total.name}__own") if isinstance(part, Accum) else part.substitute(Sigma({over: row}))
                    )
            return (
                stmt.replace(axis=replace(stmt.axis, extent=Dim(reach)), body=Body(parts)),
                Pre(name=f"{total.name}__before", carrier=work, index=(Var(local), *others)),
                Assign(total.name, apply("add", f"{total.name}__own", f"{total.name}__before")),
            )
        if isinstance(stmt, Carry):
            runs = _op("<", _op("+", first, Var(local)), steps)
            return (
                Pre(name=kept, carrier=stmt.name, index=stmt.index),
                Select(name=live, branches=(SelectBranch(stmt.value, runs), SelectBranch(kept, _TRUE))),
                stmt.replace(value=live),
            )
        if isinstance(stmt, Write):
            return stmt.replace(values=tuple(live if value == carry.value else value for value in stmt.values))
        return stmt

    # The step a block's local one stands for; the read of the second state is at the local step itself.
    at_step = Sigma({step: held(_op("+", first, here))})
    stepped = tuple(stmt.substitute(at_step).substitute(Sigma({local: here})) for stmt in loop.body.map(walked))
    steps_level = loop.replace(axis=replace(loop.axis, extent=Dim(width)), seed=False, body=Body(stepped))

    # Beside them, once a block: the same sum over the rows before the NEXT block, for each of its steps, at the
    # row that step writes. The state read here is what this block's steps left.
    next_first = _op("+", first, width)

    def beside(stmt):
        if isinstance(stmt, Loop) and stmt.axis.name == over:
            early = BinaryExpression("&&", test, _op("<", Var(over), next_first))
            parts = tuple(
                part.replace(branches=(SelectBranch(term.branches[0].value, early), term.branches[1])) if part.name == term.name else part
                for part in stmt.body
            )
            return (stmt.replace(body=Body(parts)), Carry(name=work, index=(Var(ahead), *others), value=total.name, seed=0.0))
        if isinstance(stmt, Loop) and stmt.axis.name == moving:
            return tuple(stmt.body)
        return stmt

    next_step = held(_op("+", next_first, Var(ahead)))
    at_next = Sigma({step: next_step, moving: _op("+", next_step, offset)})
    before_next = Loop(axis=Axis(ahead, Dim(width)), body=Body(tuple(stmt.substitute(at_next) for stmt in only.map(beside))))

    blocks_level = Loop(axis=Axis(outer, Dim(blocks)), body=Body((steps_level, before_next)))
    return Body(tuple(blocks_level if stmt is loop else stmt for stmt in body))
