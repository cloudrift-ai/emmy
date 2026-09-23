"""The drift checks over a DB instance — what the stored definitions let a later emmy verify.

A kernel row stores its Loop IR before and after normalization, both identities and its ``S_*`` stamps,
so a change to the normalization, the lift, the identity digest or the featurizer shows up as rows that
no longer re-derive. Each check counts the rows that fail it; ``emmy dataset check`` prints the counts.
Nothing is fixed here: a failing row is re-tuned or re-imported, never patched.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.db import SearchDB


def drift(db: SearchDB) -> dict[str, int]:
    """Every drift check by name, with the count of rows failing it: the four the compiler decides (the raw
    wire normalizes to the stored one; the stored one lifts to the stored identities; it stamps to the
    stored features; a perf row's bindings name the kernel's symbolic dims and nothing else) and the four
    the tables decide (:meth:`SearchDB.drift`). A wire the current code cannot decode fails every check
    that reads it."""
    from emmy.compiler.loop_wire import (  # noqa: PLC0415
        kernel_from_wire,
        kernel_stamps,
        loop_graph_from_wire,
        loop_graph_to_wire,
        symbolic_vars,
    )

    normalizes = identities = stamps = 0
    dims: dict[str, set[str]] = {}
    for k in db.iter_kernels():
        try:
            normalizes += loop_graph_to_wire(loop_graph_from_wire(k.loop_ir)) != k.normalized_loop_ir
        except Exception:  # noqa: BLE001 — undecodable is drift
            normalizes += 1
        try:
            tile = kernel_from_wire(k.normalized_loop_ir)
            identities += (tile.identity_key(structural=False, with_io=True), tile.identity_key(with_io=True)) != (
                k.exact_identity,
                k.structural_identity,
            )
        except Exception:  # noqa: BLE001
            identities += 1
        try:
            stamps += kernel_stamps(k.normalized_loop_ir) != k.stamps
            dims[k.exact_identity] = symbolic_vars(k.normalized_loop_ir)
        except Exception:  # noqa: BLE001
            stamps += 1
    # A row whose kernel row is gone is the foreign-key check's finding, not this one's.
    bindings = sum(row.kernel in dims and set(row.bindings) != dims[row.kernel] for row in db.iter_perf_rows(backend=None))
    return {
        "loop_ir normalizes to normalized_loop_ir": normalizes,
        "normalized_loop_ir lifts to the stored identities": identities,
        "normalized_loop_ir stamps to kernel_feature": stamps,
        "perf bindings name the kernel's symbolic dims": bindings,
        **db.drift(),
    }
