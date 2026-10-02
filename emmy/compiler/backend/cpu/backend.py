"""CPU backend: cut the fused graph into kernels, compile each to native code through LLVM, run them on threads.

``compile`` runs the Loop passes and the tile cut with every cuttable seam pinned to ``cut``, turns each cut
piece back into a ``LoopOp`` (its ``TileOp.loop_body``), and compiles every piece with
:func:`~emmy.compiler.backend.cpu.codegen.generate`. A piece the generator cannot express keeps its ``LoopOp``
and runs through ``LoopOp.forward``. ``run`` walks the graph in topological order like
:meth:`Backend.run <emmy.compiler.backend.base.Backend.run>`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from emmy import config
from emmy.compiler.backend import Backend, RunResult
from emmy.compiler.backend.base import _coerce
from emmy.compiler.backend.cpu.codegen import Unsupported, generate
from emmy.compiler.backend.cpu.kernel import CpuKernel, default_threads
from emmy.compiler.dtype import decode_bf16, encode_bf16
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

CUT_PASSES = [*LOOP_PASSES, "tile/lift", "tile/cut"]
# A cut piece can expose seams of its own; each round pins the new ones.
MAX_CUT_ROUNDS = 4


@dataclass
class CpuProgram:
    graph: Graph
    kernels: dict[str, CpuKernel] = field(default_factory=dict)
    # Node id → why the piece runs through ``LoopOp.forward`` instead.
    fallbacks: dict[str, str] = field(default_factory=dict)


class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads or default_threads()

    def compile(self, graph: Graph) -> CpuProgram:
        cut = _cut_everywhere(graph)
        program = CpuProgram(cut)
        for nid, node in cut.nodes.items():
            if type(node.op).__name__ == "TileOp":
                if node.op.loop_body is None:
                    raise ValueError(f"{nid}: TileOp without a loop body")
                node.op = LoopOp(body=node.op.loop_body, name=node.op.name)
            if not isinstance(node.op, LoopOp):
                continue
            try:
                program.kernels[nid] = _build(cut, node)
            except Unsupported as exc:
                program.fallbacks[nid] = str(exc)
        return program

    def run(
        self,
        compiled: CpuProgram,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        pre_run: Callable[[], Any] | None = None,
    ) -> tuple[RunResult, Any]:
        pre_result = pre_run() if pre_run is not None else None
        graph, input_data = compiled.graph, input_data or {}
        sizes = graph.symbolic_env(input_data)
        values: dict[str, np.ndarray] = {}
        inputs = set(graph.inputs)

        def shape_of(t) -> tuple[int, ...]:  # noqa: ANN001
            return tuple(d.as_static() if d.is_static else int(d.expr.eval(sizes)) for d in t.shape)

        t0 = time.perf_counter()
        for nid in graph.topological_order():
            node = graph.nodes[nid]
            if nid in inputs:
                for buf, t in zip(node.buffer_names(), node.outputs, strict=True):
                    if buf not in input_data:
                        raise KeyError(f"Missing input for node {nid!r}")
                    values[buf] = _coerce(input_data[buf], shape_of(t), t.dtype.np)
                continue
            if isinstance(node.op, ConstantOp):
                shape, dtype = shape_of(node.output), node.output.dtype.np
                if nid in input_data:
                    values[nid] = _coerce(input_data[nid], shape, dtype)
                elif node.op.value is not None:
                    values[nid] = np.array([node.op.value], dtype=dtype)
                else:
                    raise KeyError(f"ConstantOp {nid!r} has no value and was not supplied in input_data")
                continue
            if isinstance(node.op, InputOp):
                continue
            if nid in compiled.kernels:
                _launch(compiled.kernels[nid], graph, node, values, sizes, shape_of, self.threads)
            elif isinstance(node.op, LoopOp):
                _interpret(graph, node, values, shape_of)
            else:
                _forward(node, values, shape_of)

        elapsed = (time.perf_counter() - t0) * 1000
        return RunResult(outputs={name: values[name] for name in graph.outputs}, time_ms=elapsed), pre_result


@contextmanager
def _pins(sites: set[str]) -> Iterator[None]:
    keys = set()
    for site in sites:
        family, at, scope = site.partition("@")
        keys.add(config.knob_var(family) + at + scope)
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(dict.fromkeys(keys, "cut"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _seams(graph: Graph) -> set[str]:
    from emmy.compiler.pipeline.passes.tile._cut import cuttable_seams  # noqa: PLC0415

    found: set[str] = set()
    for node in graph.nodes.values():
        if type(node.op).__name__ == "TileOp" and node.op.op is not None:
            for site in cuttable_seams(node.op):
                found |= {site.spelling, *site.aliases}
    return found


def _cut_everywhere(graph: Graph) -> Graph:
    sites = _seams(Pipeline.build([*LOOP_PASSES, "tile/lift"]).run(graph))
    for _ in range(MAX_CUT_ROUNDS):
        with _pins(sites):
            cut = Pipeline.build(CUT_PASSES).run(graph)
        new = _seams(cut) - sites
        if not new:
            return cut
        sites |= new
    return cut


def _static_or_name(dim) -> int | str:  # noqa: ANN001
    try:
        return dim.value
    except (TypeError, ValueError) as exc:
        raise Unsupported(f"composite dim {dim.expr.pretty()}") from exc


def _build(graph: Graph, node) -> CpuKernel:  # noqa: ANN001
    op: LoopOp = node.op
    buffers = (*op.inputs, *op.outputs)
    if not op.outputs or not set(op.outputs) <= set(node.buffer_names()):
        raise Unsupported(f"outputs {list(op.outputs)} are not the node's buffers {list(node.buffer_names())}")
    tensors = {b: graph.buffer(b) for b in buffers}
    if any(t is None for t in tensors.values()):
        raise Unsupported("a buffer missing from the graph")
    shapes = {b: tuple(_static_or_name(d) for d in t.shape) for b, t in tensors.items()}
    dtypes = {b: str(t.dtype) for b, t in tensors.items()}
    try:
        ir_text, plan = generate(op, shapes, dtypes)
    except (TypeError, ValueError, KeyError) as exc:
        raise Unsupported(f"{type(exc).__name__}: {exc}") from exc
    return CpuKernel(ir_text, plan, buffers)


def _launch(kernel: CpuKernel, graph: Graph, node, values, sizes, shape_of, threads: int) -> None:  # noqa: ANN001
    arrays = {}
    for b in node.op.inputs:
        t = graph.buffer(b)
        shape, arr = shape_of(t), values[b]
        if arr.size == 1 and int(np.prod(shape)) != 1:
            arr = np.broadcast_to(arr.reshape(()), shape)
        arrays[b] = np.ascontiguousarray(arr, dtype=t.dtype.np)
    for b in node.op.outputs:
        t = graph.buffer(b)
        arrays[b] = np.zeros(shape_of(t), t.dtype.np)
    kernel(arrays, sizes, threads)
    for b in node.op.outputs:
        values[b] = arrays[b]


def _as_f32(arr: np.ndarray, dtype) -> np.ndarray:  # noqa: ANN001
    return decode_bf16(arr) if str(dtype) == "bf16" else np.asarray(arr, np.float32)


def _interpret(graph: Graph, node, values, shape_of) -> None:  # noqa: ANN001
    op: LoopOp = node.op
    args = [_as_f32(values[b], graph.buffer(b).dtype) for b in op.inputs]
    results = op.forward(*args)
    results = results if isinstance(results, tuple) else (results,)
    for b, r in zip(op.outputs, results, strict=True):
        t = graph.buffer(b)
        r = encode_bf16(np.asarray(r, np.float32)) if str(t.dtype) == "bf16" else r
        values[b] = _coerce(r, shape_of(t), t.dtype.np)


def _forward(node, values, shape_of) -> None:  # noqa: ANN001
    dtype = node.output.dtype.np
    args = [values[b] for b in node.inputs]
    if isinstance(node.op, ElementwiseOp) and dtype.kind == "f":
        args = [a.astype(np.promote_types(a.dtype, dtype), copy=False) if a.dtype.kind == "f" else a for a in args]
    results = node.op.forward(*args)
    results = results if isinstance(results, tuple) else (results,)
    for b, t, r in zip(node.buffer_names(), node.outputs, results, strict=True):
        values[b] = _coerce(r, shape_of(t), t.dtype.np)
