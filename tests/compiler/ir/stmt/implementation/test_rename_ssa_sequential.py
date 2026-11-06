"""Regression tests for ``rename_ssa_sequential`` SSA renumbering.

The renumber assigns Load names ``in0, in1, ...`` in definition order. A
gather (a Load whose *index* references another Load's SSA name) must keep
pointing at the producing Load after the renumber — even when the producer's
new name collides with a *surviving* old name elsewhere in the body.

This guards the ``in24``-undefined embedding-lookup bug: the renumber used to
publish each Load rename into both the SSA channel (``rename``) and the axis
channel (``sigma``). The Load/Write rewriter applies both to index expressions, so an
indirect index Var was substituted twice — and a rename chain (``in2_3 → in5``
while a pre-existing ``in5`` → ``in26``) collapsed transitively, wiring the
gather to the wrong row.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.stmt import Accum, Body, Init, Load, Write, parse_body
from emmy.compiler.ir.stmt.normalize import rename_ssa_sequential
from emmy.compiler.pipeline.passes.tile._fromloop import fold_from_loop


def _find_load(body, input_buf: str) -> Load:
    for s in Body.coerce(body).iter():
        if isinstance(s, Load) and s.input == input_buf:
            return s
    raise AssertionError(f"no Load from {input_buf!r} in body")


def test_gather_index_survives_rename_chain_collision() -> None:
    # Definition order fixes the renumber: ids loads → in0/in1, gather → in2,
    # then a Load *literally named* ``in1`` (a surviving old name) → in3. The
    # gather's index references the second ids load (old name ``idx1``), whose
    # new name ``in1`` collides with that surviving ``in1`` — the exact shape
    # of the embedding + RMSNorm-weight kernel that triggered the bug.
    body = parse_body(
        """
        idx0 = load ids[a0]
        idx1 = load ids[a0]
        gather = load w[idx1, a0]
        in1 = load weight[a0]
        """
    )

    out = rename_ssa_sequential(body)

    idx1_new = _find_load(out, "ids").names  # first ids load → in0
    # The second ids load renumbers to in1; the gather's indirect index must
    # resolve to *that* in1, not to the renamed weight load (in3).
    gather = _find_load(out, "w")
    weight = _find_load(out, "weight")

    assert weight.names == ("in3",)
    assert str(gather.index[0]) == "in1", gather.index[0]
    # And the gather index must NOT have been double-substituted onto the weight.
    assert str(gather.index[0]) != weight.names[0]
    # Sanity: idx0 is the very first Load → in0.
    assert idx1_new == ("in0",)


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


def test_canonical_names_do_not_capture_free_arguments():
    body = parse_body(
        """
        for k in 0..4:
            value = load x[a0, k, in0]
            total <- add(total, value)
        """
    )
    renamed = rename_ssa_sequential(body)
    (loop,) = renamed
    load = _find_load(renamed, "x")
    assert loop.axis.name != "a0" and load.name != "in0"
    assert tuple(str(index) for index in load.index) == ("a0", loop.axis.name, "in0")
    assert rename_ssa_sequential(renamed) == renamed


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("seed", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
def test_nested_loop_updates_keep_their_enclosing_seed(strided, seed, conditional):
    inner = f"for k in 0..4{' step 1' if strided else ''}{'' if seed else ' seed=false'}:"
    wrapper = "if enabled:" if conditional else "for i in 0..2:"
    original = parse_body(
        f"""
        init f32 total = 0
        {wrapper}
            {inner}
                value = load x[k]
                total <- add(total, value) over k
        out[] = total
        """
    )

    renamed = rename_ssa_sequential(original)

    (initial,) = renamed.iter_of_type(Init)
    (update,) = renamed.iter_of_type(Accum)
    (output,) = renamed.iter_of_type(Write)
    assert output.values == (initial.name,)
    assert (update.name == initial.name) is not seed
    assert rename_ssa_sequential(renamed) == renamed
