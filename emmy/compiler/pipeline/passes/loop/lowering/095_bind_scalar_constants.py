"""Represent a static FP32 scalar's stored value in the body before fusion and kernel identity.

The literal is the value after storage rounding, not the Python float the trace captured. Runtime
and externally bound constants remain loads. Other storage types and nonfinite values retain their
buffer representation until their exact typed literal contract is covered.
"""

from dataclasses import replace

import numpy as np

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Let
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import wrap_multi_output_fragment

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> Graph:
    graph = match.graph
    replacements = {}
    for load in root.op.loads:
        producer = graph.producer(load.input)
        tensor = graph.buffer(load.input)
        if producer is None or not isinstance(producer.op, ConstantOp) or producer.op.value is None:
            continue
        constant = producer.op
        if tensor.dtype != F32 or not all(dim.is_static and dim.as_static() == 1 for dim in tensor.shape):
            continue
        if constant.load_ops or constant.source_path or constant.source_parts or constant.source_graph is not None:
            continue
        if not load.is_scalar or load.carried or load.dtype not in (None, F32):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.float32(float(constant.value)))
        if np.isfinite(value):
            replacements[id(load)] = Let(load.name, Literal(value), dtype=F32)
    if not replacements:
        raise RuleSkipped("no supported static scalar load")
    body = root.op.body.map(lambda stmt: replacements.get(id(stmt), stmt))
    bound = replace(root.op, body=body, inputs={})
    fragment, match.output = wrap_multi_output_fragment(graph, bound, tuple(root.buffer_names()))
    return fragment
