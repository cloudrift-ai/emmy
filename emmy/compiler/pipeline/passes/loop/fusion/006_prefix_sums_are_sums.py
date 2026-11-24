"""PROTOTYPE. A prefix sum is a plain value: each cell is the sum over the prefix up to it.

A loop that stores its running accumulator at every trip is ordered, and the splicer refuses to inline it: a reader
would need the accumulator as it stood at one trip. Spelled as a reduction over the range ``0..trip + 1`` under a
plain loop over the trips, the same value is a function of its cell, and a reader inlines it like any other, at
whatever positions it reads.

That form says what the value is and nothing about memory. Summing again at every read, keeping the running sum as
the kernel walks the positions, and storing the prefix in a kernel of its own are three ways to realize it, which
trade recomputation for memory: a choice for the schedule and the cut, on evidence, never for fusion.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.loop import LoopOp, observes_running_accumulator
from emmy.compiler.ir.stmt import Accum, Body, Loop, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import wrap_multi_output_fragment

PATTERN = [Pattern("scan", LoopOp)]


def rewrite(match: Match, scan: Node) -> Graph:
    if not observes_running_accumulator(scan.op):
        raise RuleSkipped("no store observes a running accumulator")
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
