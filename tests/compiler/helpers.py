"""Reusable support for compiler tests.

Pytest fixtures stay in ``conftest.py``. Ordinary callables and shared skip markers live here so their dependencies
are explicit at each call site and test modules never import implementation from another test module.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Mapping
from itertools import product
from pathlib import Path

import numpy as np
import pytest


def classic_cartesian_schedules(context):
    """Enumerate a classic context's literal kernel × node × edge test oracle."""
    from emmy.compiler.ir.schedule import Schedule, ScheduleRefused

    problem = context.problem
    nodes = tuple(problem.node_site(site).nodes for site in context.tile_op.node_sites)
    edges = tuple(problem.node_site(edge[0]).edges for edge in context.tile_op.edge_sites)
    for kernel, node_values, edge_values in product(problem.kernel_site.kernels, product(*nodes), product(*edges)):
        schedule = Schedule(
            kernel,
            dict(zip(context.tile_op.node_sites, node_values, strict=True)),
            dict(zip(context.tile_op.edge_sites, edge_values, strict=True)),
        )
        try:
            context.extend(schedule)
        except (ScheduleRefused, TypeError):
            accepted = False
        else:
            accepted = True
        yield schedule, accepted


def enumerate_classic_reference(context):
    """Yield the accepted subset of the literal classic schedule product."""
    for schedule, accepted in classic_cartesian_schedules(context):
        if accepted:
            yield schedule


@functools.cache
def _literal_classic_problem_type():
    """The classic site classes with their catalogs replaced by hand-written factors.

    The compiler knows nothing of this: a site is the source of its own candidates, and the only way to hand it
    others is to be a different site. Built once, since each class definition binds its own descriptors.
    """
    from emmy.compiler.ir.schedule.classic.sites import ClassicKernelSite, ClassicNodeSite, ClassicProblem

    class _LiteralNodeSite(ClassicNodeSite):
        @functools.cached_property
        def nodes(self):
            return self.problem.literal_nodes[self.id]

        @functools.cached_property
        def edges(self):
            incident = self.problem.tile.incident_edges[self.id]
            return self.problem.literal_edges[incident[0]] if incident else ()

        @functools.cached_property
        def warp_eligible(self) -> bool:
            return any(choice.tile.is_warp for choice in self.nodes)

    class _LiteralKernelSite(ClassicKernelSite):
        @functools.cached_property
        def kernels(self):
            return self.problem.literal_kernels

    @dataclasses.dataclass(frozen=True, eq=False)
    class _LiteralProblem(ClassicProblem):
        literal_kernels: tuple = ()
        literal_nodes: Mapping = dataclasses.field(default_factory=dict)
        literal_edges: Mapping = dataclasses.field(default_factory=dict)

        @functools.cached_property
        def node_sites(self):
            return tuple(_LiteralNodeSite(self, site) for site in self.tile.node_sites)

        @functools.cached_property
        def kernel_site(self):
            return _LiteralKernelSite(self)

    return _LiteralProblem


def literal_classic_context(tile, target, *, kernel, nodes, edges, row=None, order=None):
    """A classic context whose sites offer exactly the factors named here — the bounded space a test can answer
    by hand, against which the lazy traversal is checked.

    ``kernel`` is the kernel factor, ``nodes`` maps a node site to its choices and ``edges`` an edge site to its
    transports. ``row`` narrows the kernel factor by its bare ``WORK`` / ``RASTER`` spellings, which is all a
    bounded test asks of a row; the sites of a real problem narrow their own catalogs.
    """
    from emmy.compiler.ir.schedule.classic import ClassicScheduleContext

    row = dict(row or {})
    for key, spell in (("WORK", lambda choice: choice.work.spell()), ("RASTER", lambda choice: choice.raster.spell())):
        if key in row:
            kernel = tuple(choice for choice in kernel if spell(choice) == row[key])
    problem = _literal_classic_problem_type()(
        tile,
        target,
        row=row,
        literal_kernels=tuple(kernel),
        literal_nodes={site: tuple(choices) for site, choices in nodes.items()},
        literal_edges={edge: tuple(choices) for edge, choices in edges.items()},
    )
    return ClassicScheduleContext(tile, target, problem, order=order)


def case_target_tile(case: str):
    """Lift the target of a realization corpus case, named by its path under ``realization/cases``: the one kernel
    its traced program lowers to, as the cut pass is offered it."""
    from dataclasses import replace

    from emmy.compiler.context import Context
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.pipeline import Pipeline
    from emmy.compiler.pipeline.search.golden import GoldenFile

    document = GoldenFile.load(Path(__file__).parent / "realization/cases" / case)
    [target] = document.targets()
    lifted = Pipeline.build(["tile/lift"]).run(target.program({}), ctx=Context.from_target(tuple(document.compute_cap)), db=None)
    [node] = [node for node in lifted.nodes.values() if isinstance(node.op, TileOp)]
    tile = node.op.with_io(lifted, node)
    return replace(tile, name=next(iter(tile.outputs)))  # addressed by its primary output, as a graph node is


def has_cuda_gpu() -> bool:
    """Check that the runtime extension loads and sees a CUDA device."""
    from emmy.compiler.backend.cuda.device import compute_capability

    return compute_capability() is not None


def has_cuda_toolchain() -> bool:
    """Check that Emmy can both see a GPU and assemble kernels."""
    if not has_cuda_gpu():
        return False
    from emmy.compiler.backend.cuda.ptxas import ptxas_path

    return ptxas_path() is not None


requires_cuda = pytest.mark.skipif(
    not has_cuda_toolchain(),
    reason="CUDA not available (need the emmy.emmy_runtime extension + GPU + ptxas)",
)


@functools.cache
def device_compute_capability() -> tuple[int, int] | None:
    """Return the live CUDA device compute capability, or ``None`` when no GPU is visible."""
    from emmy.compiler.backend.cuda.device import compute_capability

    return compute_capability()


def kernel_ptx(name: str, body: str, params, *, threads: int, arch_specific: bool = False) -> str:
    """The PTX of a hand-written kernel for the live card: ``body`` is a Kernel IR body in the text form, ``params``
    its parameters in call order, each ``(name, kind, dtype)`` with ``kind`` an ``input``, an ``output`` or a
    ``descriptor``, and ``threads`` the threads of a block. Without a card the kernel is for ``sm_80``."""
    from emmy.emmy_ir import compile_ptx, parse_body

    from emmy.compiler.dtype import get
    from emmy.compiler.ir.kernel.interface import Interface, Param

    major, minor = device_compute_capability() or (8, 0)
    interface = Interface(
        tuple(Param(n, kind, None if dtype is None else get(dtype), "" if kind == "descriptor" else n) for n, kind, dtype in params),
        threads,
        {},
        0,
        arch_specific,
    )
    return compile_ptx(name, parse_body(body), interface, sm=major * 10 + minor)


def requires_sm(major: int, minor: int = 0):
    """Skip an instruction-specific CUDA test below its minimum compute capability."""
    return pytest.mark.skipif(
        (device_compute_capability() or (0, 0)) < (major, minor),
        reason=f"requires sm_{major}{minor} or newer",
    )


# Skip the mma.sync warp-tier tests below sm_90. On sm_80-89 the pin-only path is currently non-functional because
# nvcc rejects the ``sm_NNa`` target and ``ldmatrix`` faults at runtime on at least Ada (sm_89).
requires_sm90 = pytest.mark.skipif(
    (device_compute_capability() or (0, 0)) < (9, 0),
    reason="mma.sync warp tier needs sm_90+ (pin-only / non-functional on sm_80-89: ldmatrix fault + sm_NNa TMA compile)",
)


def direct_classic_leaf(fork_point):
    """Select the direct classic schedule without flattening the schedule tree."""
    from emmy.compiler.pipeline.fork import Fork
    from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES, family_of

    def allowed(option) -> bool:
        knobs = option.knobs if isinstance(option, Fork) else getattr(option, "knobs", {})
        return all(family_of(key) not in SCHEDULE_FAMILIES or str(value) == "" for key, value in knobs.items())

    def descend(options):
        for option in options:
            if not allowed(option):
                continue
            if not isinstance(option, Fork) or option.is_leaf:
                return option
            try:
                return descend(option.expand())
            except ValueError:
                continue
        raise ValueError("fork has no direct classic schedule")

    return descend(fork_point.options)


def dtype_tol(dtype) -> dict[str, float]:
    """Return default ``np.testing.assert_allclose`` tolerances for a compiler dtype."""
    return {"rtol": 5e-3, "atol": 5e-3} if dtype.name == "f16" else {"rtol": 1e-4, "atol": 1e-5}


def dtype_input_scale(dtype) -> float:
    """Scale random inputs so large fp16 reductions do not overflow."""
    return 0.05 if dtype.name == "f16" else 1.0


def grouped_raster_decode(kernel, group: int) -> list[str]:
    """The block-axis decodes of ``kernel``'s grid that take a grouped raster of ``group`` block-tiles, as text: a
    block axis read from the CTA index alone as its group's base plus its offset in the group (``group * (...) +
    ...``, the factor on either side)."""
    from emmy.compiler.ir.kernel.ir import Grid

    (grid,) = [stmt for stmt in kernel.body if isinstance(stmt, Grid)]
    decodes = [str(decode) for decode in grid.decode]
    return [
        decode
        for decode in decodes
        if "block_idx.x" in decode and "thread_idx" not in decode and (f" * {group} + " in decode or decode.startswith(f"{group} * ("))
    ]


def skip_if_no_cuda() -> None:
    """Skip the current test when Emmy cannot compile CUDA kernels."""
    if not has_cuda_toolchain():
        pytest.skip("CUDA not available (need the emmy.emmy_runtime extension + GPU + nvcc)")


def matmul_graph(m: int, k: int, n: int):
    """Build a plain ``(m, k) @ (k, n) -> (m, n)`` graph for compiler tests."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp

    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    graph.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    graph.inputs = ["a", "b"]
    graph.outputs = ["o"]
    return graph


def inject_constants(input_data: dict[str, np.ndarray], graph) -> dict[str, np.ndarray]:
    """Supply scalar and rebound tensor constants omitted from backend input data."""
    from emmy.compiler.ir.base import ConstantOp

    for node_id, node in graph.nodes.items():
        if not isinstance(node.op, ConstantOp) or node_id in input_data:
            continue
        if node.op.value is not None:
            input_data[node_id] = np.array([node.op.value], dtype=np.float32)
            continue
        source = input_data.get(node.op.name)
        if source is not None:
            from emmy.compiler.loader.binder import apply_load_ops

            array = np.asarray(source, dtype=np.float32)
            if node.op.load_ops and node.op.source_shape is not None:
                array = array.reshape(node.op.source_shape)
            input_data[node_id] = apply_load_ops(array, node.op.load_ops)
    return input_data


def dyn_M(mode: str, M: int):
    """Return ``Dim('seq_len')`` for dynamic mode and the integer ``M`` otherwise."""
    if mode == "dynamic":
        from emmy.compiler.dim import Dim

        return Dim("seq_len")
    return M


def from_pretrained_or_skip(loader, *args, **kwargs):
    """Run a Hugging Face download, skipping the test when the Hub is unavailable."""
    try:
        return loader(*args, **kwargs)
    except OSError as exc:
        model = args[0] if args else kwargs.get("pretrained_model_name_or_path", "?")
        pytest.skip(f"HuggingFace Hub unavailable for {model} (likely rate-limited): {exc}")


def inventory_document(graph, compute_cap=(12, 0), gpu_name: str | None = None, **kwargs):
    """The golden a trace inventory of ``graph`` writes for a card — every kernel it lowers to and an unmeasured row
    per kernel — as a document, for tests that author rows onto it."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import inventory  # noqa: PLC0415

    return inventory(graph.copy(), Context.from_target(tuple(compute_cap), gpu_name=gpu_name), **kwargs)


QWEN3_EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"


def qwen3_embedding_config():
    """The Qwen3-Embedding-0.6B config, trimmed to one layer.

    Config only — no checkpoint — so the head geometry (16 query heads over 8 KV heads, head_dim 128) is
    readable without downloading weights."""
    from transformers import AutoConfig  # noqa: PLC0415

    config = from_pretrained_or_skip(AutoConfig.from_pretrained, QWEN3_EMBEDDING)
    config.num_hidden_layers = 1
    return config


def qwen3_embedding_model(config=None, auto_class=None):
    """A random-weight fp32 model built from :func:`qwen3_embedding_config` — the shared trunk of the
    dynamic-shape tests. Seeded, so two calls build the same weights. ``auto_class`` picks the head:
    the bare trunk by default, or a causal LM for the whole-model trace."""
    import torch  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    torch.manual_seed(0)
    return (auto_class or AutoModel).from_config(config if config is not None else qwen3_embedding_config()).float().eval()


def render_typed(statements, ctx) -> list[str]:
    """The host C++ lines of hand-built ``statements`` once their types are complete: the renderer prints a typed body
    (``ir.types.explicit_conversions``), and the names ``ctx`` already types are taken as declared."""
    from emmy.compiler.dtype import get
    from emmy.compiler.ir.loop.render import render_body
    from emmy.compiler.ir.stmt import Body
    from emmy.compiler.ir.types import explicit_conversions
    from emmy.compiler.target import native_f16_ops

    known = {name: get(dtype) for name, dtype in ctx.ssa_dtypes.items()}
    return render_body(explicit_conversions(Body(statements), {}, native_f16_ops(), known), ctx)


def _kernel_recorder():
    """A pipeline strategy that records the ``KernelOp`` each ``CudaOp`` was lowered from; its ``kernels(compiled)``
    gives the Kernel IR of each kernel node of a compiled graph, by node id."""
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.ir.kernel import KernelOp
    from emmy.compiler.pipeline.strategy import PipelineStrategy

    class LoweredKernels(PipelineStrategy):
        def __init__(self) -> None:
            self.lowered: list[tuple[KernelOp, CudaOp]] = []

        def on_rebind(self, e) -> None:
            if isinstance(e.replaced, KernelOp) and isinstance(e.node.op, CudaOp):
                self.lowered.append((e.replaced, e.node.op))

        def kernels(self, compiled) -> dict:
            found = {}
            for node_id, node in compiled.nodes.items():
                if isinstance(node.op, CudaOp):
                    # A retried resolution lowers a kernel again: the one the result holds is the one it was generated from.
                    found[node_id] = next(kernel for kernel, op in reversed(self.lowered) if op is node.op)
            return found

    return LoweredKernels()


def compile_with_kernels(graph, *, ctx=None, passes=None, decide=None, **run):
    """``graph`` compiled to CUDA ops, and the Kernel IR each kernel's code was generated from, by node id.

    A ``CudaOp`` holds its kernel's code, not the statements it was generated from. This runs the one compile a test
    would run (``passes``, the whole pipeline when omitted; ``ctx`` and ``run`` as ``Pipeline.run`` takes them; the
    pins in scope apply as they do there) and keeps each ``KernelOp`` the last lowering pass read: the grid decoded,
    every index flat, every type said. A check of a kernel's structure reads that body; only a check of how an
    instruction is spelled reads the code. With ``decide`` the forks are resolved by it (``Run.resolve``) on the
    live device's context, where ``Pipeline.run`` takes the greedy pick."""
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

    recorder = _kernel_recorder()
    pipeline = Pipeline.build(passes or CUDA_PASSES).with_strategies(recorder)
    if decide is None:
        compiled = pipeline.run(graph, ctx=ctx, **run)
    else:
        from emmy.compiler.context import Context
        from emmy.compiler.pipeline.pipeline import Run

        compiled, _ = Run(pipeline=pipeline, ctx=ctx or Context.probe()).resolve(graph, decide)
    return compiled, recorder.kernels(compiled)


@contextlib.contextmanager
def recording_kernels():
    """Record the Kernel IR of every kernel lowered inside the block, for a compile the test does not run itself (a
    backend's, a serving split's). Yields a function from a compiled graph to its kernels by node id, as
    :func:`compile_with_kernels` returns them."""
    from unittest import mock

    from emmy.compiler.pipeline import Pipeline

    recorder = _kernel_recorder()
    build = Pipeline.build.__func__

    def recording(cls, passes, **kwargs):
        return build(cls, passes, **kwargs).with_strategies(recorder)

    with mock.patch.object(Pipeline, "build", classmethod(recording)):
        yield recorder.kernels


def statements(kernels, *types) -> list:
    """Every statement of ``types`` in ``kernels`` (one ``KernelOp``, or the ``{node id: KernelOp}`` of a compile),
    nested ones included, in body order; every statement when no type is given."""
    ops = kernels.values() if isinstance(kernels, dict) else (kernels,)
    return [stmt for op in ops for stmt in op.body.iter() if not types or isinstance(stmt, types)]


def operations(kernels) -> set[str]:
    """The elementwise operations ``kernels`` compute, by name: in a value, as a fold's or a fragment statement's own
    operation, inside a cross-thread combine or a store's epilogue."""
    found: set[str] = set()
    pending = statements(kernels)
    while pending:
        stmt = pending.pop()
        found |= {node.fn for e in stmt.expressions() for node in e.subterms() if hasattr(node, "fn")}
        found |= {node.operation.name for e in stmt.expressions() for node in e.subterms() if getattr(node, "operation", None)}
        if getattr(stmt, "op", None) is not None:
            found.add(stmt.op.name)
        for carried in ("combine_states", "epilogue", "run"):
            pending.extend(getattr(stmt, carried, None) or ())
    return found


def text(kernels) -> str:
    """The Kernel IR of ``kernels`` (the kernels of a compile, one of them, or a body) in its text form: what a check
    of a name the lowering gave, or of an index's form, reads."""
    from emmy.compiler.ir.stmt import Body, pretty_body

    if isinstance(kernels, Body):
        return "\n".join(pretty_body(kernels))
    ops = kernels.values() if isinstance(kernels, dict) else (kernels,)
    return "\n".join(line for op in ops for line in pretty_body(op.body))


def case_kernels(case, ctx=None):
    """A realization corpus case lowered with itself as its only evidence, as ``realization.helpers.lowered`` lowers
    it: the compiled graph and the Kernel IR of its kernels by node id."""
    from tests.compiler.realization.helpers import evidence_scope

    with evidence_scope(case):
        return compile_with_kernels(case.program(), ctx=ctx or case.context(), db=None)
