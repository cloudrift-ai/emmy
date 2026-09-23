"""The drift checks over a DB instance — what the stored definitions let a later emmy verify.

A kernel row stores its Loop IR before and after normalization and both identities, so a change to the
normalization or the identity digest shows up as rows that no longer re-derive. Each check counts the rows
that fail it; ``emmy dataset check`` prints the counts. Nothing is fixed here: a failing row is re-tuned or
re-imported, never patched. The ``S_*`` stamps are not checked: they are the identity strategy's features of
the fused loop body the kernel was lifted from, which the row does not store, so a featurizer change is
caught where a freeze is loaded (its manifest names the version), not here.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.db import SearchDB


def drift(db: SearchDB) -> dict[str, int]:
    """Every drift check by name, with the count of rows failing it: the three the compiler decides (the raw
    wire normalizes to the stored one; the stored one decodes to a kernel with the stored identities; a perf
    row's bindings name the kernel's symbolic dims and nothing else) and the four the tables decide
    (:meth:`SearchDB.drift`). A wire the current code cannot decode fails every check that reads it."""
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.loop_wire import loop_graph_from_wire, loop_graph_to_wire, symbolic_vars  # noqa: PLC0415

    normalizes = identities = 0
    dims: dict[str, set[str]] = {}
    for k in db.iter_kernels():
        try:
            normalizes += loop_graph_to_wire(loop_graph_from_wire(k.loop_ir)) != k.normalized_loop_ir
        except Exception:  # noqa: BLE001 — undecodable is drift
            normalizes += 1
        try:
            # The wire's loop op IS the kernel: its body is what the identities digest, its buffers the io half.
            graph = loop_graph_from_wire(k.normalized_loop_ir)
            [node] = [node for node in graph.nodes.values() if isinstance(node.op, LoopOp)]
            op = node.op.with_io(graph, node)
            identities += (op.identity_key(structural=False, with_io=True), op.identity_key(with_io=True)) != (
                k.exact_identity,
                k.structural_identity,
            )
        except Exception:  # noqa: BLE001
            identities += 1
        try:
            dims[k.exact_identity] = symbolic_vars(k.normalized_loop_ir)
        except Exception:  # noqa: BLE001 — counted above, where the decode first failed
            pass
    # A row whose kernel row is gone is the foreign-key check's finding, not this one's.
    bindings = sum(row.kernel in dims and set(row.bindings) != dims[row.kernel] for row in db.iter_perf_rows(backend=None))
    return {
        "loop_ir normalizes to normalized_loop_ir": normalizes,
        "normalized_loop_ir decodes to the stored identities": identities,
        "perf bindings name the kernel's symbolic dims": bindings,
        **db.drift(),
    }
