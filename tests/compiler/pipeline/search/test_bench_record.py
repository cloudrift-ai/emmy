"""The kernel row the perf writers store — what every reader joins evidence on."""

from __future__ import annotations

from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.wire import kernel_tile


def test_a_kernel_row_is_keyed_by_the_identity_its_wire_computes() -> None:
    """The row a perf writer stores is the kernel's definition, keyed by the live tile's exact identity, and
    lifting the stored wire again computes that same identity and the same ``S_*`` stamps — so a stored row, a
    golden's stored kernel and a fork's offer name one kernel alike without any of them storing what it computed.
    A twisted kernel (online softmax) is the case that matters: the body the twist derives spells another
    reduction than the loop body the kernel was formed from, and a row keyed or featurized off one while its fork
    read the other never priced its own fork — the RTX 5090 hardware golden's softmax and attention rows fell to
    the prior that way, at eighty times the compile time."""
    from emmy.compiler.pipeline.search.bench_record import kernel_row
    from emmy.compiler.pipeline.search.dataset import KernelDef
    from emmy.compiler.pipeline.search.features import stamps
    from tests.compiler.realization import helpers as corpus

    case = corpus.load_case(corpus.CASES_DIR / "attention/sdpa-hd128-softmax-v-mma.json")
    graph, _taken = corpus.lowered(case, case.context())
    [cuda] = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    tile = kernel_tile(cuda)
    row = kernel_row(tile, cuda.kernel_name)

    assert tile.schedule is None, "the kernel is the tile its schedule fork was offered"
    stored = KernelDef(loop_ir=row.loop_ir, name=row.name, formed=row.formed)  # as a file holds it: nothing known
    assert row.formed and stored.exact_identity == row.exact_identity == tile.identity_key(structural=False, with_io=True)
    assert stamps(stored.op()) == stamps(tile) == stamps(cuda), "one kernel, one S_* row, wherever it is read"
    assert stamps(tile)["S_n_load"] > 0 and not any(key.endswith("_?") for key in stamps(tile)), "the dtypes are the kernel's io"
    assert not any(str(key).startswith(("S_", "I_", "H_")) for key in cuda.knobs), "an op's knobs hold decisions only"


def test_a_hang_blames_the_kernel_the_runtime_names(monkeypatch) -> None:
    """The runtime's watchdog quotes the hung kernel with Rust's ``{:?}`` — double quotes — and the bench
    worker hands the message over as a ``repr``. Each spelling must blame that one kernel: a hang that blames
    nobody records nothing, and the next compile elects the same hanging kernel again."""
    from types import SimpleNamespace

    from emmy.compiler.pipeline.search import bench_record

    blamed: list[str] = []
    monkeypatch.setattr(bench_record, "persist_kernel_perf", lambda db, ctx, backend, op, **kw: blamed.append(op.kernel_name))
    nodes = [SimpleNamespace(op=SimpleNamespace(kernel_name=name)) for name in ("k_a", "k_b__place_0765e8")]
    hang = RuntimeError('kernel "k_b__place_0765e8" did not complete within 2000 ms — hung kernel')
    for exc in (hang, RuntimeError(repr(hang)), RuntimeError("kernel 'k_b__place_0765e8 (iter 0)' did not complete")):
        blamed.clear()
        bench_record.persist_bench_failure(None, None, "cuda", nodes, exc, 1.0)
        assert blamed == ["k_b__place_0765e8"], str(exc)
