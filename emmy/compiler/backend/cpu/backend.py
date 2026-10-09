"""CPU backend: cut the fused graph into kernels, compile each to native code through LLVM, run them on the Rust runtime.

``compile`` runs the Loop passes and the tile cut with every cuttable seam pinned to ``cut``, turns each cut
piece back into a ``LoopOp`` (its ``TileOp.loop_body``), translates every piece with
:func:`~emmy.compiler.backend.cpu.codegen.generate`, and builds the program for the Rust runtime
(:mod:`~emmy.compiler.backend.cpu.runtime`). A piece the generator cannot express fails the compile.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from emmy import config
from emmy.compiler.backend import Backend, RunResult
from emmy.compiler.backend.cpu.codegen import Unsupported, generate
from emmy.compiler.backend.cpu.runtime import CpuKernel, NativeProgram, default_threads
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

CUT_PASSES = [*LOOP_PASSES, "tile/lift", "tile/cut"]
# A cut piece can expose seams of its own; each round pins the new ones.
MAX_CUT_ROUNDS = 4


@dataclass
class CpuProgram:
    graph: Graph
    kernels: dict[str, CpuKernel]
    native: NativeProgram


class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads or default_threads()

    def compile(self, graph: Graph) -> CpuProgram:
        cut = _cut_everywhere(graph)
        kernels, unsupported = {}, {}
        for nid, node in cut.nodes.items():
            if isinstance(node.op, TileOp):
                if node.op.loop_body is None:
                    raise ValueError(f"{nid}: TileOp without a loop body")
                node.op = LoopOp(body=node.op.loop_body, name=node.op.name)
            if not isinstance(node.op, LoopOp):
                continue
            try:
                kernels[nid] = _build(cut, node)
            except Unsupported as exc:
                unsupported[nid] = str(exc)
        if unsupported:
            raise Unsupported("; ".join(f"{nid}: {reason}" for nid, reason in unsupported.items()))
        return CpuProgram(cut, kernels, NativeProgram(cut, kernels, self.threads))

    def run(
        self,
        compiled: CpuProgram,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        pre_run: Callable[[], Any] | None = None,
    ) -> tuple[RunResult, Any]:
        pre_result = pre_run() if pre_run is not None else None
        t0 = time.perf_counter()
        compiled.native.bind(input_data or {})
        compiled.native.run_once()
        outputs = compiled.native.outputs()
        return RunResult(outputs=outputs, time_ms=(time.perf_counter() - t0) * 1000), pre_result


@contextmanager
def _pins(knobs: dict[str, str]) -> Iterator[None]:
    """Set knob pins (``"PLACE@<site>"`` → value) as environment variables for the duration of the block."""
    values = {}
    for name, value in knobs.items():
        family, at, scope = name.partition("@")
        values[config.knob_var(family) + at + scope] = value
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
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
        if isinstance(node.op, TileOp) and node.op.op is not None:
            for site in cuttable_seams(node.op):
                found |= {site.spelling, *site.aliases}
    return found


def _cut_everywhere(graph: Graph) -> Graph:
    sites = _seams(Pipeline.build([*LOOP_PASSES, "tile/lift"]).run(graph))
    for _ in range(MAX_CUT_ROUNDS):
        # Cut every seam and decide the other kernel-set forks the plain way, since no CPU evidence ranks them:
        # keep every reduction whole (a cross-CTA split leaves one kernel adding atomically into a single cell,
        # which the CPU runs on one thread) and every constant in its folded layout.
        with _pins({**dict.fromkeys(sites, "cut"), "REDUCE": "", "LAYOUT": "folded"}):
            cut = Pipeline.build(CUT_PASSES).run(graph)
        new = _seams(cut) - sites
        if not new:
            return cut
        sites |= new
    logger.warning("cpu: %d seam(s) still uncut after %d cut rounds; those pieces stay fused", len(new), MAX_CUT_ROUNDS)
    return cut


def _static_or_name(dim) -> int | str:  # noqa: ANN001
    try:
        return dim.value
    except (TypeError, ValueError) as exc:
        raise Unsupported(f"composite dim {dim.expr.pretty()}") from exc


def _build(graph: Graph, node) -> CpuKernel:  # noqa: ANN001
    op: LoopOp = node.op
    buffers = (*op.inputs, *op.outputs)
    if set(op.outputs) != set(node.buffer_names()):
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
