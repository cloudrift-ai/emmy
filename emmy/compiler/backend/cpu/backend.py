"""CPU backend: cut the fused graph into kernels, compile each to native code through LLVM, run them on the Rust runtime.

``compile`` runs the Loop passes and the tile cut under the host CPU's context with every cuttable seam pinned to
``cut``, then the schedule pass, which gives each piece its CPU schedule. Every piece is translated from its
``TileOp.loop_body`` and that schedule by :func:`~emmy.compiler.backend.cpu.codegen.generate`, and the program is
built for the Rust runtime (:mod:`~emmy.compiler.backend.cpu.runtime`). A piece the generator cannot express fails the
compile.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from emmy.compiler.backend import Backend, RunResult
from emmy.compiler.backend.base import resolve_tune_db
from emmy.compiler.backend.cpu.codegen import PAIR_TABLE, Unsupported, generate
from emmy.compiler.backend.cpu.runtime import CpuKernel, NativeProgram, default_threads
from emmy.compiler.context import Context
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs

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
    # Each kernel's scheduled tile by node id: the kernel a measurement of it is filed under.
    tiles: dict[str, TileOp]
    ctx: Context


class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, threads: int | None = None, *, tune_db: Path | str | None = None) -> None:
        self.threads = threads or default_threads()
        # The measured rows the schedule picks from, as ``CudaBackend`` resolves them (``resolve_tune_db``); the CPU's
        # are filed under its own context, so they never meet a card's.
        self.tune_db = resolve_tune_db(tune_db)

    def compile(self, graph: Graph) -> CpuProgram:
        from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

        ctx = Context.for_cpu(self.threads)
        db = SearchDB.for_compile(self.tune_db) if self.tune_db is not None else None
        cut = Pipeline.build(["tile/schedule"]).run(_cut_everywhere(graph, ctx), ctx=ctx, db=db)
        kernels, unsupported, tiles = {}, {}, {}
        for nid, node in cut.nodes.items():
            schedule = None
            if isinstance(node.op, TileOp):
                if node.op.loop_body is None:
                    raise ValueError(f"{nid}: TileOp without a loop body")
                schedule = node.op.schedule.kernel if node.op.schedule is not None else None
                tiles[nid] = node.op
                node.op = LoopOp(body=node.op.loop_body, name=node.op.name)
            if not isinstance(node.op, LoopOp):
                continue
            try:
                kernels[nid] = _build(cut, node, schedule)
            except Unsupported as exc:
                unsupported[nid] = str(exc)
        if unsupported:
            raise Unsupported("; ".join(f"{nid}: {reason}" for nid, reason in unsupported.items()))
        return CpuProgram(cut, kernels, NativeProgram(cut, kernels, self.threads), tiles, ctx)

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


def _reads(term) -> set[str]:
    """The buffers ``term`` loads anywhere inside it."""
    found: set[str] = set()
    stack, seen = [term], set()
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if type(value).__name__ == "Load":
            found.add(value.input)
        if isinstance(value, (tuple, list)):
            stack.extend(value)
        elif hasattr(value, "__dataclass_fields__"):
            stack.extend(getattr(value, name) for name in value.__dataclass_fields__)
    return found


def _seams(graph: Graph) -> dict[str, str]:
    """Every seam of ``graph``'s tiles and its placement: a seam whose cone reads constants alone — a weight's
    decode — stays inside its consumer, so the weight streams once at its stored width; every other seam is cut."""
    from emmy.compiler.pipeline.passes.tile._cut import cuttable_seams  # noqa: PLC0415

    constants = {nid for nid, node in graph.nodes.items() if isinstance(node.op, ConstantOp)}
    found: dict[str, str] = {}
    for node in graph.nodes.values():
        if isinstance(node.op, TileOp) and node.op.op is not None:
            for site in cuttable_seams(node.op):
                reads = _reads(site.node)
                for spelling in (site.spelling, *site.aliases):
                    found[spelling] = "fuse" if reads and reads <= constants else "cut"
    return found


def _cut_everywhere(graph: Graph, ctx: Context) -> Graph:
    sites = _seams(Pipeline.build([*LOOP_PASSES, "tile/lift"]).run(graph, ctx=ctx))
    for _ in range(MAX_CUT_ROUNDS):
        # Place every seam (``_seams``) and decide the other kernel-set forks the plain way, since no CPU evidence
        # ranks them: keep every reduction whole (a cross-CTA split leaves one kernel adding atomically into a single
        # cell, which the CPU runs on one thread) and every constant in its source layout — a checkpoint's [out, in]
        # weight, its reduce axis contiguous, as a tiled projection streams it and a serving path lends it.
        with pinned_knobs({**sites, "REDUCE": "", "LAYOUT": "source"}):
            cut = Pipeline.build(CUT_PASSES).run(graph, ctx=ctx)
        new = {spelling: place for spelling, place in _seams(cut).items() if spelling not in sites}
        if not new:
            return cut
        sites |= new
    logger.warning("cpu: %d seam(s) still uncut after %d cut rounds; those pieces stay fused", len(new), MAX_CUT_ROUNDS)
    return cut


def _is_pair_table(node) -> bool:  # noqa: ANN001
    """Whether ``node`` is the e2m1 pair table an NVFP4 weight decodes through: the 256 x 2 constant the loader builds
    from ``from_f4e2m1`` of each byte's two nibbles (``loader.quant``). The generator reads a lookup into it as that
    decode, which vectorizes, where the lookup itself would be a gather of one 16-bit value per lane."""
    op = getattr(node, "op", None)
    source = getattr(op, "source_graph", None)
    return (
        isinstance(op, ConstantOp)
        and source is not None
        and tuple(op.source_shape or ()) == (256, 2)
        and any(getattr(getattr(n.op, "op", None), "name", None) == "from_f4e2m1" for n in source.nodes.values())
    )


def _static_or_name(dim) -> int | str:  # noqa: ANN001
    try:
        return dim.value
    except (TypeError, ValueError) as exc:
        raise Unsupported(f"composite dim {dim.expr.pretty()}") from exc


def _build(graph: Graph, node, schedule) -> CpuKernel:  # noqa: ANN001
    op: LoopOp = node.op
    buffers = (*op.inputs, *op.outputs)
    if set(op.outputs) != set(node.buffer_names()):
        raise Unsupported(f"outputs {list(op.outputs)} are not the node's buffers {list(node.buffer_names())}")
    tensors = {b: graph.buffer(b) for b in buffers}
    if any(t is None for t in tensors.values()):
        raise Unsupported("a buffer missing from the graph")
    shapes = {b: tuple(_static_or_name(d) for d in t.shape) for b, t in tensors.items()}
    dtypes = {b: PAIR_TABLE if _is_pair_table(graph.nodes.get(b)) else str(t.dtype) for b, t in tensors.items()}
    try:
        ir_text, plan = generate(op, shapes, dtypes, schedule)
    except (TypeError, ValueError, KeyError) as exc:
        raise Unsupported(f"{type(exc).__name__}: {exc}") from exc
    return CpuKernel(ir_text, plan, buffers)
