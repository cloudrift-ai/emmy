"""A measured kernel's Loop IR wire — the definition a ``kernel`` row stores — decodes to the kernel it came
from.

The tune DB keys a kernel on its EXACT identity and stores :func:`kernel_wire`'s wire beside it, so the same
kernel reached from two parents (the fused kernel of a slice, a piece a cut or a split minted) has one
definition and one candidate set. That only holds if the decoded wire's loop op carries the same exact and
clustered identities as the tile kernel — its body is what the identities digest, its buffers the io half —
for every kind of kernel the compiler mints, which is what the realization corpus lets this assert without a
GPU: a fused kernel, the pieces of a placement cut, a cross-CTA split's pieces, a nested cut, an attention
kernel whose stored body the tile lift does not accept back."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.loop_wire import kernel_bindings, kernel_tile, kernel_wire, loop_graph_from_wire, symbolic_vars
from emmy.compiler.pipeline.search.golden import _replay, kernel_identity, lead_of, siblings_of
from tests.compiler.realization import helpers as corpus

CASES = (
    "fused/norm-linear-f16-scalar-reduce.yaml",
    "fused/linear-add-place-cut-sm70.yaml",
    "reduce/cross-cta-matmul-kernel.yaml",
    "reduce/sinkhorn-nested-cut-derived-read-sm70.yaml",
    "attention/sdpa-hd128-softmax-v-mma.yaml",
)


def _decoded(wire: dict) -> LoopOp:
    """The wire's one loop op, bound to the wire's own buffers — what a ``kernel`` row defines."""
    graph = loop_graph_from_wire(wire)
    [node] = [node for node in graph.nodes.values() if isinstance(node.op, LoopOp)]
    return node.op.with_io(graph, node)


@pytest.mark.parametrize("case_path", CASES)
def test_every_kernel_of_a_set_has_wires_that_decode_to_its_identities(case_path):
    case = corpus.load_case(corpus.CASES_DIR / case_path)
    ctx = case.context()
    graph, taken = corpus.lowered(case, ctx)
    kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    assert kernels
    deploy = set()
    for cuda in kernels:
        tile = kernel_tile(cuda)
        assert tile is not None, cuda.kernel_name
        exact, clustered = tile.identity_key(structural=False, with_io=True), tile.identity_key(with_io=True)
        assert exact is not None
        wire = kernel_wire(tile)
        op = _decoded(wire)
        assert (op.identity_key(structural=False, with_io=True), op.identity_key(with_io=True)) == (exact, clustered), cuda.kernel_name
        assert symbolic_vars(wire) == set(kernel_bindings(tile)), cuda.kernel_name
        deploy.add(clustered)
    # The clustered flavour read off the same tile is the deploy identity the golden side mints for
    # the same kernels when it replays the case (what a receipt names, what an import computes). The
    # replay may know more kernels — the arms it looked into and did not take.
    primary = case.record
    replay = _replay(primary, siblings=siblings_of(primary, case.records), lead=lead_of(primary, case.records))
    assert deploy <= set(replay.kernels)
    if not any(taken):
        assert kernel_identity(primary) in deploy


def test_bindings_are_the_hints_a_bench_sizes_a_symbolic_kernel_by():
    case = corpus.load_case(corpus.CASES_DIR / "reduce/combine-amax-ilp-symbolic.yaml")
    graph, _taken = corpus.lowered(case, case.context())
    [cuda] = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    tile = kernel_tile(cuda)
    bindings = kernel_bindings(tile)
    assert bindings, "a symbolic case binds at least one dim"
    assert all(isinstance(size, int) and size > 0 for size in bindings.values())
    assert symbolic_vars(kernel_wire(tile)) == set(bindings)
    # A static kernel binds nothing: two rows of it never differ by size.
    static = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.yaml")
    static_graph, _taken = corpus.lowered(static, static.context())
    assert all(kernel_bindings(kernel_tile(node.op)) == {} for node in static_graph.nodes.values() if isinstance(node.op, CudaOp))
