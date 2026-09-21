"""A measured kernel's Loop IR wire — the definition a ``kernel`` row stores — re-lifts to the kernel
it came from.

The tune DB keys a kernel on its EXACT identity and stores :func:`kernel_wire` beside it, so the same
kernel reached from two parents (the fused kernel of a slice, a piece a cut or a split minted) has
one definition and one candidate set. That only holds if lifting the wire again yields the same
exact identity, for every kind of kernel the compiler mints — which is what the realization corpus
lets this assert without a GPU: a fused kernel, the pieces of a placement cut, a cross-CTA split's
pieces, a nested cut."""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.loop_wire import kernel_bindings, kernel_tile, kernel_wire, loop_graph_from_wire
from emmy.compiler.pipeline.search.golden import _replay, kernel_identity, lead_of, siblings_of
from tests.compiler.realization import helpers as corpus

CASES = (
    "fused/norm-linear-f16-scalar-reduce.yaml",
    "fused/linear-add-place-cut-sm70.yaml",
    "reduce/cross-cta-matmul-kernel.yaml",
    "reduce/sinkhorn-nested-cut-derived-read-sm70.yaml",
)


def _lift(wire: dict):
    """The decoded kernel lifted the way ``golden._lifted_target`` lifts a target's kernel node: the
    matcher's io refresh, the lift, the twist rewrite. No Loop passes: a kernel wire is post-fusion,
    and the passes would normalize a size-one axis away and mint another kernel."""
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op
    from emmy.compiler.pipeline.passes.lowering.tile._twist import rewrite_twisted

    graph = loop_graph_from_wire(wire)
    [node] = [node for node in graph.nodes.values() if isinstance(node.op, LoopOp)]
    node.op = node.op.with_io(graph, node)
    tile = lift_loop_op(node.op, name=node.id)
    tile = replace(tile, op=rewrite_twisted(tile.op, tile.axes))
    return tile.with_io(graph, node)


@pytest.mark.parametrize("case_path", CASES)
def test_every_kernel_of_a_set_has_a_wire_that_re_lifts_to_its_exact_identity(case_path):
    case = corpus.load_case(corpus.CASES_DIR / case_path)
    ctx = case.context()
    graph, taken = corpus.lowered(case, ctx)
    kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    assert kernels
    deploy = set()
    for cuda in kernels:
        tile = kernel_tile(cuda)
        assert tile is not None, cuda.kernel_name
        exact = tile.identity_key(structural=False, with_io=True)
        assert exact is not None
        lifted = _lift(kernel_wire(tile))
        assert lifted.identity_key(structural=False, with_io=True) == exact, cuda.kernel_name
        deploy.add(tile.identity_key(with_io=True))
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
    # A static kernel binds nothing: two rows of it never differ by size.
    static = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.yaml")
    static_graph, _taken = corpus.lowered(static, static.context())
    assert all(kernel_bindings(kernel_tile(node.op)) == {} for node in static_graph.nodes.values() if isinstance(node.op, CudaOp))
