"""The structural dump (``ops.pretty`` / ``TileOp.pretty_body``) — the STORED tree, as a tree, and
NOTHING derived.

The dump is the one place a reader meets the tile term directly, so what it shows has to be what
the term IS: each node's kind and stored params, an operand edge recursed into (a computed edge is
visibly a subtree, a materialized one visibly a leaf ``Load``), and the caller facts that live
BESIDE the term — placement, workers, schedule, output specifications — in their own regions. A derived
evaluation (the per-cell step, the nodes synthesized inside it, the lowered nest) is a CONSEQUENCE
of the stored params and is never printed: showing it beside storage is the inversion the layer
exists to prevent, and it was the bulk of the output.

These pin: (a) every stored param of each node kind reaches the dump; (b) edges say which lambda
params they bind, nest, are labelled by inhabitant, and appear exactly once; (c) nothing derived
appears; (d) accepted schedule choices annotate a node only when the owning ``TileOp`` supplies
them — never from the term; (e) a λ that is not closed says what it captures.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.schedule import Placement, Raster, Reduce, Schedule, Stage, Tile, Work
from emmy.compiler.ir.schedule.classic import (
    ClassicMaterialization,
    ClassicScheduleContext,
    EdgeSchedule,
    KernelSchedule,
    ProjectionSchedule,
    ReductionSchedule,
)
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, TileOp
from emmy.compiler.ir.tile._dump import pretty
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.terms import contraction, projection

K_STAT, K_PRODUCT = Axis("k", 512), Axis("k", 256)


def _stat_fold() -> Fold:
    """RMSNorm's statistic — ``acc0 += x[m, k]²``; the lift forms its load as a slab operand."""
    body = Body(
        (
            Load(name="in0", input="x", index=(Var("m"), Var("k"))),
            Assign(name="v1", op="multiply", args=("in0", "in0")),
            Accum(name="acc0", value="v1", op="add", axes=("k",)),
        )
    )
    return fold_from_loop(Loop(axis=K_STAT, body=body))


def _cone() -> Fold:
    """A computed A edge — ``xhat = x[m, k] * w[k]``."""
    return projection(
        body=(
            Load(name="xhat_e", input="x", index=(Var("m"), Var("k"))),
            Load(name="xhat_s", input="w", index=(Var("k"),)),
            Assign(name="xhat", op="multiply", args=("xhat_e", "xhat_s")),
        )
    )


def _product(a=None) -> Fold:
    """The gate⊗up shape — two channels over ONE shared ``a`` edge."""
    return contraction(
        K_PRODUCT,
        a if a is not None else Load(name="a_e", input="x", index=(Var("m"), Var("k"))),
        *((Load(name=f"{acc}_b", input=w, index=(Var("k"), Var("n"))), acc) for acc, w in (("acc_g", "Wg"), ("acc_u", "Wu"))),
    )


# --- the stored params all reach the dump ------------------------------------------------------- #


def test_fold_dump_shows_every_stored_param() -> None:
    """A fold's storage IS ``lift`` + ``(init, combine)`` + ``operands``; each is a labelled
    branch, and the kind — DERIVED, never stored — rides the header beside the axis the lift
    binds (a bare term names it; only the owning ``TileOp`` knows its extent)."""
    text = "\n".join(pretty(_stat_fold()))
    assert text.splitlines()[0] == "Fold[k] reduce"
    assert "├─ operand[in0]: load x[m, k]   ‹materialized›" in text
    assert "├─ init: (0)" in text
    # Every param: the iteration var, then the operand result the lift binds positionally.
    assert "├─ lift: λ(k, in0) -> (v1)" in text
    assert "└─ combine: λ(acc0, acc0__o) -> (acc0)" in text
    # The lift's own body nests two under its signature — the stored program, read as the
    # binder's body rather than as a sibling of the branch labels.
    assert "│    v1 = multiply(in0, in0)" in text


def test_contraction_dump_shows_the_k_axis_and_every_channel() -> None:
    text = "\n".join(pretty(_product()))
    assert "Fold[k] contraction" in text
    assert "├─ operand[a_e]: load x[m, k]" in text
    # Sharing is arity: one ``a`` first, one branch per channel; the accumulators are the combine's.
    assert "├─ operand[acc_g_b]: load Wg[k, n]" in text
    assert "├─ operand[acc_u_b]: load Wu[k, n]" in text
    # The binding labels connect the edges above to their positional lift params, A first.
    assert "lift: λ(k, a_e, acc_g_b, acc_u_b) -> (acc_g__v, acc_u__v)" in text
    assert "└─ combine: λ(acc_g, acc_u, acc_g__o, acc_u__o) -> (acc_g, acc_u)" in text


def test_projection_dump_shows_the_binder_and_its_operands() -> None:
    """A zero-axis Fold stores its projection lambda and operands. The binder rides its own branch,
    next to the body it binds rather than in the header. Every lambda-valued field reads the same
    way: signature, then body."""
    m = projection((_stat_fold(),), (Assign(name="o", op="rsqrt", args=("acc0",)),))
    text = "\n".join(pretty(m))
    assert text.splitlines()[0] == "Fold  free"
    assert "├─ operand[acc0]: Fold[k] reduce   ‹computed›" in text
    assert "└─ lift: λ(acc0) -> (o)" in text
    assert "     o = rsqrt(acc0)" in text  # the body, indented two under the signature


def test_a_product_edge_shows_every_lambda_param_it_binds() -> None:
    """One operand edge can supply several result components, so the dump names every scalar
    substituted for that edge instead of leaving the positional binding implicit."""
    node = projection((_product(),), (Assign(name="o", op="add", args=("acc_g", "acc_u")),))
    text = "\n".join(pretty(node))
    assert "operand[acc_g, acc_u]: Fold[k] contraction" in text
    assert "lift: λ(acc_g, acc_u) -> (o)" in text


def test_the_fn_branch_survives_an_empty_body() -> None:
    """The branch carries the SIGNATURE, so it is emitted even with nothing to compute — an
    identity projection still binds, and dropping the branch would lose the binder entirely."""
    text = "\n".join(pretty(projection((_stat_fold(),))))
    assert "└─ lift: λ(acc0) -> (acc0)" in text


def test_a_sourceless_map_is_marked_pointwise() -> None:
    assert "‹pointwise›" in "\n".join(pretty(_cone()))


# --- edges nest, and say which of the two inhabitants they are ---------------------------------- #


def test_a_computed_edge_nests_as_a_subtree_a_materialized_one_is_a_leaf() -> None:
    """The two inhabitants of an operand edge, told apart in the dump: the cone recurses into its
    own node, the gmem loads do not."""
    lines = pretty(_product(a=_cone()))
    (a_line,) = [ln for ln in lines if "operand[xhat]:" in ln]
    assert "‹computed›" in a_line and "Fold  free" in a_line
    assert any("‹materialized›" in ln and "load Wg" in ln for ln in lines)
    # The cone's own body is reached BELOW the a edge — the subtree is really rendered.
    assert any("xhat = multiply(xhat_e, xhat_s)" in ln for ln in lines)


def _two_operand_reader() -> Fold:
    """A reader with its own names for both slots, one of which it never reads — the shape the
    twisted fusion leaves when it re-seats a carrier's channels."""
    return Fold(
        operands=(_stat_fold(), _cone()),
        lift=Lambda(params=("_unread0", "w"), body=Body((Assign(name="o", op="exp", args=("w",)),)), results=("o",)),
    )


def test_a_lift_prints_the_operand_name_for_a_slot_it_reads() -> None:
    """The signature spells a slot the way the operand does, so it echoes the bracket above it. What
    a reader happens to call a slot is spelling, not computation. The body is re-spelled with it, so
    the two stay consistent."""
    text = "\n".join(pretty(_two_operand_reader()))
    assert "├─ operand[xhat]: Fold  free" in text
    assert "└─ lift: λ(xhat) -> (o)" in text
    assert "     o = exp(xhat)" in text  # the body follows the signature
    assert "(w)" not in text


def test_a_lift_omits_a_slot_it_never_reads() -> None:
    """Once a param is spelled by the operand result it binds, the bracket one line up resolves which
    component it is — so a slot this reader does not use is length and nothing else. Its edge keeps
    its branch: only the signature entry goes."""
    text = "\n".join(pretty(_two_operand_reader()))
    assert "├─ operand[acc0]: Fold[k] reduce" in text  # the edge is still there in full
    assert "acc0" not in text.rsplit("lift: ", 1)[1].splitlines()[0]  # but not on the reader's line
    assert "_unread0" not in text


# --- a scalar operand is spelled inside its reader ------------------------------------------------ #


def _scaled_scores() -> Fold:
    """The sdpa score shape — a reduce scaled by a constant the frontend broadcast into a one-element
    buffer. The lift closes over that buffer's read, so the scale arrives as its own operand edge."""
    scale = projection(
        body=(
            Load(name="s0", input="sdpa_scale", index=(Literal(0, "int"),)),
            Load(name="s1", input="sdpa_mask_fill", index=(Literal(0, "int"),)),
        ),
        results=("s0", "s1"),
    )
    return projection(
        (_stat_fold(), scale),
        (
            Assign(name="v2", op="multiply", args=("acc0", "s0")),
            Assign(name="v3", op="add", args=("v2", "s1")),
        ),
    )


def test_a_scalar_operand_is_inlined_into_the_lift_that_reads_it() -> None:
    """One value for the whole kernel decides nothing — no residence, no partition, no seam — so its
    branch would be tree art around a constant. It is spelled where it is read instead, and the
    params it bound leave the signature with it."""
    text = "\n".join(pretty(_scaled_scores()))
    assert "operand[s0, s1]" not in text  # no branch of its own
    assert "lift: λ(acc0) -> (v3)" in text  # nor the params it bound
    # Its statements open the reader's body, under the reader's own names for them.
    assert "     s0 = load sdpa_scale[0]" in text
    assert "     v2 = multiply(acc0, s0)" in text
    # The reduce beside it is untouched: an edge that DOES decide keeps its branch.
    assert "operand[acc0]: Fold[k] reduce   ‹computed›" in text


def test_a_twisted_node_prints_its_stable_combine_then_psi_and_the_base_as_helpers() -> None:
    """The carrier IS its stable lift and κ_S; ψ and the componentwise base monoid are what a
    matcher reads back through, so they follow as helpers. κ_S prints as a signature — the stable
    program is a dozen statements and ``--ir loop`` has them."""
    from emmy.compiler.ir.pure.twist import SOFTMAX, Twist

    fold = Fold(
        operands=(_stat_fold(),),
        lift=Lambda(params=("k", "acc0"), body=Body((Const(name="one", value=1.0),)), results=("acc0", "one")),
        init=(-1e30, 0.0),
        base=Lambda.componentwise(SOFTMAX.base[:2], ("m", "l")),
        twist=Twist(recipe=SOFTMAX, channels=(0,)),
    )
    text = "\n".join(pretty(fold))
    assert "├─ combine: λ(m, l, m__o, l__o) -> (m, l)" in text, "the stable ⊕, as a signature"
    assert "├─ helper: psi λ(m, D, O) -> (m, d, o)" in text
    assert "│    d = multiply(D, f)" in text, "the coordinate map's own program, in the recipe's names"
    assert "└─ helper: base = (maximum, add)" in text
    assert "acc0 <- " not in text  # still nothing derived from the step


# --- nothing DERIVED reaches the dump ------------------------------------------------------------ #


def _split_k() -> Fold:
    """Split-K's outer reduce — the identity-lift composition over one bilinear ``Fold`` edge. Its
    derived step embeds that very object, so a dump that printed the step would show it twice."""
    inner = contraction(
        "kslice",
        Load(name="a_e", input="x", index=(Var("m"), Var("kslice"))),
        (Load(name="b_e", input="w", index=(Var("kslice"), Var("n"))), "acc0"),
    )
    return Fold(
        operands=(inner,),
        lift=Lambda(params=("ksplit", "acc0"), body=Body(()), results=("acc0",)),
        init=(0.0,),
        base=Lambda.componentwise(("add",), ("acc0",)),
    )


def test_the_derived_step_is_not_printed() -> None:
    """The step is a CONSEQUENCE of ``lift`` + ``combine``, as re-derivable as ``Fold.lower()``'s
    output — printing it beside storage is the inversion the dump exists to prevent, and it was
    the bulk of the output. ``--ir loop`` is where a body lives."""
    fold = _stat_fold()
    text = "\n".join(pretty(fold))
    assert "derived" not in text
    # The step's serial form — the combine specialized at the singleton, an ``Accum`` — is exactly
    # what must not appear: it is impure, so it is not even spellable as one of the stored λs.
    assert "acc0 <- add" not in text
    assert [type(s).__name__ for s in fold.step()][-1] == "Accum"  # the premise


def test_an_edge_is_rendered_once_not_once_per_derived_position() -> None:
    """The lowered nest places the operand edge again, inside the outer reduce loop, so printing
    anything derived duplicated every edge. With storage only, each edge appears exactly once —
    under its own branch."""
    fold = _split_k()
    text = "\n".join(pretty(fold))
    assert text.count("Fold[kslice] contraction") == 1
    assert text.count("operand[a_e]: load x[m, kslice]") == 1


# --- a λ that is not closed says so -------------------------------------------------------------- #


def _capturing_cone() -> Fold:
    """The flash ``P = exp(s − m)`` shape — a cone that READS a value the enclosing body defines
    (``m_run``, the carrier's running max) instead of producing it."""
    return projection(
        body=(
            Load(name="p_e", input="x", index=(Var("m"), Var("k"))),
            Assign(name="p", op="subtract", args=("p_e", "m_run")),
        )
    )


def test_a_lambda_binds_an_enclosing_value_as_a_trailing_param() -> None:
    """Being closed is exactly what decides whether a subtree can become an operand edge, so a λ
    that reads an enclosing value BINDS it — the dump shows it in the signature rather than in a
    separate capture annexe. ``m`` / ``k`` are iteration space (placement + the contraction's own
    axis) and ``m_run`` is a value, and binding everything is what removes that distinction
    instead of deciding it: all three arrive as params, in the order ``Lambda.closing`` appends."""
    node = _product(a=_capturing_cone())
    m = Axis("m", 128)
    tile = TileOp(op=node, name="k_flashish", place=Placement(free=(m,), grid=(m,), mapped=True), axes=(m, K_PRODUCT))
    text = tile.pretty_body()
    assert "│  └─ lift: λ(k, m, m_run) -> (p)" in text
    assert "captures" not in text


def test_iteration_vars_are_not_captures() -> None:
    """A λ reading an axis is not capturing — the nest binds it. This covers the three places an
    axis can come from: the term's own axes, the placement, and an output specification's sweep."""
    m, n = Axis("m", 128), Axis("n", 64)
    body = Body((Load(name="w_e", input="w", index=(Var("m"), Var("n"))), Assign(name="o", op="multiply", args=("acc0", "w_e"))))
    tile = TileOp(
        op=projection((_stat_fold(),), body),
        name="k_stat",
        place=Placement(free=(m, n), grid=(m,), mapped=True),
        axes=(m, n, K_STAT),
        output_specs=(OutputSpec(write=Write(output="y", index=(Var("m"), Var("n")), value="o"), sweep=(n,)),),
    )
    # ``m`` comes from the placement, ``n`` from the output sweep — the sweep axis left the term
    # at 1q, so a dump reading only the term would wrongly call it captured.
    assert "captures" not in tile.pretty_body()


def test_the_capture_set_is_omitted_when_the_iteration_space_is_unknown() -> None:
    """A bare term has no placement, so its grid coordinates are indistinguishable from captured
    values. The dump declines to answer rather than claim the λ is closed."""
    assert "captures" not in "\n".join(pretty(_capturing_cone()))


# --- accepted choices annotate from the TileOp, never from the term ----------------------------- #


def test_slices_annotate_a_node_only_when_the_owning_tileop_supplies_them() -> None:
    fold = _stat_fold()
    bare = TileOp(op=fold, name="k_stat", axes=(K_STAT,))
    assert "REDUCE=" not in bare.pretty_body()

    context = ClassicScheduleContext(bare)
    nodes = {
        site: ProjectionSchedule(Tile()) if view.axis is None else ReductionSchedule(Tile(), Reduce.of(reg=4))
        for site, view in enumerate(context.tile_op.views)
    }
    classic = Schedule(
        KernelSchedule(Work(), Raster()),
        nodes,
        {edge: EdgeSchedule(Stage.direct()) for edge in context.tile_op.edge_sites},
    )
    scheduled = TileOp(
        op=fold,
        name="k_stat",
        axes=(K_STAT,),
        schedule=classic,
        materialization=ClassicMaterialization({}, {}),
    )
    assert "⟨REDUCE=r4⟩" in scheduled.pretty_body()
    # The annotation is the TileOp's; the term is untouched by it.
    assert "REDUCE=" not in "\n".join(pretty(fold))


# --- the caller facts beside the term get their own regions ------------------------------------- #


def test_pretty_body_separates_placement_and_outputs_from_the_term() -> None:
    m, n = Axis("m", 128), Axis("n", 64)
    tile = TileOp(
        op=projection((_stat_fold(),), (Assign(name="o", op="rsqrt", args=("acc0",)),)),
        name="k_stat",
        place=Placement(free=(m, n), grid=(m,), mapped=True),
        axes=(m, n, K_STAT),
        output_specs=(OutputSpec(write=Write(output="y", index=(Var("m"), Var("n")), value="o"), sweep=(n,)),),
    )
    text = tile.pretty_body()
    assert "place  free=(m, n)  grid=(m)" in text
    assert "outputs" in text and "└─ sweep(n) y[m, n] = o" in text


def test_an_unmapped_placement_says_so() -> None:
    m = Axis("m", 128)
    tile = TileOp(op=_stat_fold(), name="k_stat", place=Placement(free=(m,)), axes=(m, K_STAT))
    assert "unmapped" in tile.pretty_body()
