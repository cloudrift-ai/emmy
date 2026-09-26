"""A measured kernel's Loop IR wire — the definition a ``kernel`` row stores — re-lowers to the kernel it came
from.

The tune DB keys a kernel on its EXACT identity and stores :func:`kernel_wire`'s wire beside it: the body the
kernel was formed from, bound to its own buffers. A freeze is re-lowered from those wires, so each has to come
back as the same kernel under the current compiler — the lift, the twist and the identity strategy giving it the
same exact identity, clustered identity and ``S_*`` stamps — for every kind of kernel the compiler mints, which is
what the realization corpus lets this assert without a GPU: a fused kernel, the pieces of a placement cut, a
cross-CTA split's pieces, a nested cut, a twisted attention kernel. The one kind formed from no loop op — a piece
carved from a twisted tree, whose derived body the lift does not take back — keeps its derived body, whose buffers
still say which symbolic dims a measurement of it binds (what ``priced_arms`` projects a parent's sizes onto), and is
reached through its parent's program instead (``formed`` is false on its row)."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.loop_wire import formed_from, kernel_bindings, kernel_tile, kernel_wire, loop_graph_from_wire, symbolic_vars
from emmy.compiler.pipeline.search.golden import _replay, kernel_identity, lead_of, siblings_of
from tests.compiler.realization import helpers as corpus

CASES = (
    "fused/norm-linear-f16-scalar-reduce.yaml",
    "fused/linear-add-place-cut-sm70.yaml",
    "reduce/cross-cta-matmul-kernel.yaml",
    "reduce/sinkhorn-nested-cut-derived-read-sm70.yaml",
    "matmul/f16-cut-splitk-unit-row.yaml",
    "attention/sdpa-hd128-softmax-v-mma.yaml",
)
#: Pieces carved from a twisted attention tree: the decode split-KV pair and an attention cut piece.
UNFORMED_CASES = (
    "attention/sdpa-gqa-decode-split-kv.yaml",
    "attention/rmsnorm-qk-sdpa-stat-cut.yaml",
)


def _relowered(wire: dict, ctx):
    """The wire alone through the lowering passes, every fork at its first leaf — the tile kernels it mints."""
    from emmy.compiler.pipeline import LOWERING_PASSES, Pipeline
    from emmy.compiler.pipeline.fork import iter_leaves
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.pins import unpinned_decisions

    run = Run(pipeline=Pipeline.build(LOWERING_PASSES), ctx=ctx)
    with unpinned_decisions():
        graph, _trace = run.resolve(loop_graph_from_wire(wire), lambda fp: next(iter_leaves(fp.options)))
    return [kernel_tile(node.op) for node in graph.nodes.values() if isinstance(node.op, CudaOp)]


def _stamps(tile) -> dict[str, float]:
    return {k: float(v) for k, v in (tile.knobs or {}).items() if k.startswith("S_")}


def _identities(op) -> tuple[str, str]:
    return op.identity_key(structural=False, with_io=True), op.identity_key(with_io=True)


@pytest.mark.parametrize("case_path", CASES)
def test_every_kernel_of_a_set_re_lowers_from_its_wire_to_itself(case_path):
    case = corpus.load_case(corpus.CASES_DIR / case_path)
    ctx = case.context()
    graph, taken = corpus.lowered(case, ctx)
    kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    assert kernels
    deploy = set()
    for cuda in kernels:
        tile = kernel_tile(cuda)
        assert tile is not None and formed_from(tile) is not None, cuda.kernel_name
        wire = kernel_wire(tile)
        assert symbolic_vars(wire) == set(kernel_bindings(tile)), cuda.kernel_name
        [again] = _relowered(wire, ctx)
        assert _identities(again) == _identities(tile), cuda.kernel_name
        assert _stamps(again) == _stamps(tile), cuda.kernel_name
        deploy.add(_identities(tile)[1])
    # The clustered flavour read off the same tile is the deploy identity the golden side mints for
    # the same kernels when it replays the case (what a receipt names, what an import computes). The
    # replay may know more kernels — the arms it looked into and did not take.
    primary = case.record
    replay = _replay(primary, siblings=siblings_of(primary, case.records), lead=lead_of(primary, case.records))
    assert deploy <= set(replay.kernels)
    if not any(taken):
        assert kernel_identity(primary) in deploy


@pytest.mark.parametrize("case_path", UNFORMED_CASES)
def test_a_piece_of_a_twisted_kernel_keeps_a_wire_that_names_its_symbolic_dims(case_path):
    case = corpus.load_case(corpus.CASES_DIR / case_path)
    graph, _taken = corpus.lowered(case, case.context())
    tiles = [kernel_tile(node.op) for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    unformed = [tile for tile in tiles if formed_from(tile) is None]
    assert unformed, "the case mints a piece the lift cannot form"
    for tile in unformed:
        assert symbolic_vars(kernel_wire(tile)) == set(kernel_bindings(tile)), tile.name


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
