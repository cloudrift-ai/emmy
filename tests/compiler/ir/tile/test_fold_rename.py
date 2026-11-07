"""The ``Fold`` term's stored combine and lift track an SSA rename in lockstep."""

from __future__ import annotations

from emmy.compiler.ir.stmt import Load, parse_body
from emmy.compiler.pipeline.passes.tile._fromloop import fold_from_loop


def test_fold_combine_tracks_accum_rename() -> None:
    # The algebra lives on the ``Fold`` TERM — its stored combine/lift must track an SSA rename in
    # lockstep (the Fold rewrite handler renames the combine through ``Lambda.rename``), or the
    # cooperative combine reads a state name the renamed lift no longer defines (the M=1
    # cut-consumer's ``acc1``-undefined miscompile). A term is never a ``Body`` member, so the
    # rename reaches it through the rewrite registry directly.
    (loop,) = parse_body(
        """
        for k0 in 0..8:
            v9 = load a[k0]
            acc1 <- add(acc1, v9) over k0
        """
    )
    fold = fold_from_loop(loop)
    assert fold is not None

    renamed = fold.rewrite(lambda name: f"{name}_r")

    assert renamed.combine.results == ("acc1_r",), renamed.combine.results
    assert renamed.exposes == ("acc1_r",)
    (load,) = (stmt for edge in renamed.operands for stmt in edge.lift.body if isinstance(stmt, Load))
    assert load.names == ("v9_r",) and renamed.applied.results == ("v9_r",), (load.names, renamed.applied.results)
