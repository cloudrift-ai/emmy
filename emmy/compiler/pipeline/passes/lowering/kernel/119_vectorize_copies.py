"""Widen the runs of loads and writes that the unrolled copies line up, as ``050`` and ``080`` widen the body's own.

``117_unroll_loops`` and ``118_unroll_by_size`` write a loop's trips out as copies of its body, and the copies of a
read along the axis are consecutive elements: four copies of an f32 load from a shared slab are one 16-byte run.
The two widenings run again on the unrolled body
(:func:`_vector.vectorize_loads`, :func:`_vector.vectorize_stores`), with the same alignment proof. A loop stepping
several trips at a time states its step, so an axis it steps four at a time is a multiple of four; a copy of the
remainder, at an offset the launch decides, proves nothing and stays scalar.

It widens what the vector passes widened before it, under the same decisions: nothing when ``VECTORIZE_LOADS`` /
``VECTORIZE_STORES`` was pinned off. What is already a vector stays one, so a second run changes nothing.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._vector import vectorize_loads, vectorize_stores
from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS, VECTORIZE_STORES

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    body = op.body
    if op.knobs.get(VECTORIZE_LOADS.name):
        body = vectorize_loads(op, body)
    if op.knobs.get(VECTORIZE_STORES.name):
        body = vectorize_stores(op, body)
    if body == op.body:
        raise RuleSkipped("no run of copies to widen")
    return replace(op, body=body, knobs=dict(op.knobs))
