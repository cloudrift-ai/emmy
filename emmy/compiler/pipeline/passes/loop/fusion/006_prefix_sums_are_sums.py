"""PROTOTYPE. A prefix sum is a plain value: each cell is the sum over the prefix up to it.

A loop that stores its running accumulator at every trip is ordered, and the splicer refuses to inline it: a reader
would need the accumulator as it stood at one trip. Spelled as a reduction over the range ``0..trip + 1`` under a
plain loop over the trips, the same value is a function of its cell, and a reader inlines it like any other. It
costs recomputation, which a cut takes back.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var
from emmy.compiler.ir.loop import LoopOp, observes_running_accumulator
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import wrap_multi_output_fragment

PATTERN = [Pattern("scan", LoopOp)]


#: The value a running accumulation starts from, by its operation.
_START = {"add": 0.0, "multiply": 1.0}
_SIGN = {"add": "+", "multiply": "*"}


def as_state(body: Body) -> Body | None:
    """PROTOTYPE. ``body`` with its running accumulation written as a carried state: the loop it runs over is the
    step, outermost, and each trip reads the state one step back, folds its value in, stores the result and carries
    it. ``None`` where the body is not loops around that one loop and nothing else."""
    outer: list[Loop] = []
    current = tuple(stmt for stmt in body)
    while len(current) == 1 and isinstance(current[0], Loop) and not current[0].is_reduce:
        outer.append(current[0])
        current = tuple(current[0].body)
    if len(current) != 1 or not isinstance(current[0], Loop) or not current[0].is_reduce or current[0].range is not None:
        return None
    steps = current[0]
    parts = tuple(steps.body)
    folds = [part for part in parts if isinstance(part, Accum)]
    stores = [part for part in parts if isinstance(part, Write)]
    names = {fold.name for fold in folds}
    if not stores or any(isinstance(part, Loop) for part in parts) or any(not set(store.values) <= names for store in stores):
        return None
    if any(fold.op.name not in _START or fold.base is not None or fold.axes != (steps.axis.name,) for fold in folds):
        return None
    cell = tuple(Var(loop.axis.name) for loop in outer)
    step = Var(steps.axis.name)
    moved: list[Statement] = [part for part in parts if not isinstance(part, (Accum, Write))]
    for fold in folds:
        state, held = f"{fold.name}__state", f"{fold.name}__held"
        moved += [
            Load(names=(held,), input=state, index=(step - Literal(1, "int"), *cell)),
            Assign(fold.name, BinaryExpression(_SIGN[fold.op.name], Var(held), fold.value)),
            Carry(name=state, value=fold.name, index=(step, *cell), seed=_START[fold.op.name]),
        ]
    inside = Body((*moved, *stores))
    for loop in reversed(outer):
        inside = Body((loop.replace(body=inside),))
    return Body((steps.replace(body=inside),))


def rewrite(match: Match, scan: Node) -> Graph:
    if not observes_running_accumulator(scan.op):
        raise RuleSkipped("no store observes a running accumulator")
    stated = as_state(scan.op.body)
    if stated is not None:
        op = replace(scan.op, body=stated, inputs={})
        fragment, match.output = wrap_multi_output_fragment(match.graph, op, tuple(scan.buffer_names()))
        return fragment
    taken = set(scan.op.body.axis_names)

    def plain(stmt: Statement) -> Statement:
        if not isinstance(stmt, Loop) or not stmt.is_reduce or stmt.carries:
            return stmt
        writes = [part for part in stmt.body if isinstance(part, Write)]
        folded = {part.name for part in stmt.body if isinstance(part, Accum)}
        if not writes or any(not set(write.values) <= folded for write in writes) or stmt.range is not None:
            return stmt
        trip = f"{stmt.axis.name}p"
        while trip in taken:
            trip += "_"
        taken.add(trip)
        upto = (Literal(0, "int"), Var(trip) + Literal(1, "int"))
        prefix = stmt.replace(body=Body(part for part in stmt.body if not isinstance(part, Write)), range=upto)
        stores = tuple(write.substitute({stmt.axis.name: Var(trip)}) for write in writes)
        return Loop(axis=Axis(trip, stmt.axis.extent), body=Body((prefix, *stores)))

    body = scan.op.body.map(plain)
    if body == scan.op.body or observes_running_accumulator(op := replace(scan.op, body=body, inputs={})):
        raise RuleSkipped("the ordered loop is not a reduction whose accumulators are all it stores")
    fragment, match.output = wrap_multi_output_fragment(match.graph, op, tuple(scan.buffer_names()))
    return fragment
