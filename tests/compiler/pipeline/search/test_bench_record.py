"""The kernel row the perf writers store — what every reader joins evidence on."""

from __future__ import annotations

from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.wire import kernel_tile


def test_a_kernel_row_carries_the_stamps_the_deploy_joins_on() -> None:
    """The row's ``S_*`` stamps are the identity strategy's — written at the fusion boundary onto the fused
    loop body — because that is what the deploy's fork signature and the golden replay key evidence by. For
    a twisted kernel (online softmax) the body the twist derives spells another reduction, so a row stamped
    from that body would never price its own fork: the RTX 5090 hardware golden's softmax and attention rows
    fell to the prior that way, at eighty times the compile time. The wire holds the fused body the kernel
    was formed from, whose features are the stamps."""
    from emmy.compiler.pipeline.fork import SCHEDULE_FORK_STAMPS
    from emmy.compiler.pipeline.search.bench_record import kernel_row
    from emmy.compiler.pipeline.search.features import kernel_stamps
    from tests.compiler.realization import helpers as corpus

    case = corpus.load_case(corpus.CASES_DIR / "attention/sdpa-hd128-softmax-v-mma.json")
    graph, _taken = corpus.lowered(case, case.context())
    [cuda] = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    tile = kernel_tile(cuda)
    row = kernel_row(tile, cuda.kernel_name)

    assert row.stamps == {k: float(v) for k, v in cuda.knobs.items() if k.startswith("S_")}, (
        "the strategy's stamps, as the kernel carries them"
    )
    # The enumeration's own stamps (``S_warp_eligible``) ride beside the body's features.
    structural = {k: v for k, v in row.stamps.items() if k not in SCHEDULE_FORK_STAMPS}
    assert row.formed and structural == kernel_stamps(row.loop_ir), "the wire is the body the stamps were taken from"
