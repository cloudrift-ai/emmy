"""Canonical Loop IR lifts completely into a Fold tree without recognition.

The prologue-placement tests at the end were deleted with the recognition-era suite and are
RESTORED: one of them guards a miscompile that actually shipped (the decode-attention ``v1``/``v3``
defect, where a pure prologue value sank past the statement that reads it and the lowered kernel
referenced an undefined name). Deleting a regression test for a fixed miscompile removes the only
thing that would notice it coming back — and it comes back silently, as a compile or a wrong
number, never as a slow kernel.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Body, Load, Loop, parse_body
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.passes.tile._cut import cuttable_seams
from emmy.compiler.pipeline.passes.tile._fromloop import lift_loop_op
from tests.compiler.terms import contraction, slab


def deep_defines(node) -> tuple[str, ...]:
    """Every name a term tree or statement defines — lift bodies and exposed names, operands included."""
    if isinstance(node, Fold):
        inner = (name for stmt in node.lift.body for name in deep_defines(stmt))
        return (*inner, *node.exposes, *(name for edge in node.operands for name in deep_defines(edge)))
    return (*node.defines(), *(name for body in node.nested() for stmt in body for name in deep_defines(stmt)))


def _grid(tile) -> frozenset[str]:
    """The kernel-scope binding: the grid binds the free axes, the term opens its output sweeps."""
    return frozenset(axis.name for axis in tile.place.free)


def _tile(body: Body):
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["tile/lift"], select=["lift"]).run(graph).nodes["out"].op


def _matmul_body() -> Body:
    return parse_body(
        """
        for m in 0..32:
            for n in 0..64:
                for k in 0..128:
                    xv = load x[m, k]
                    wv = load w[n, k]
                    prod = multiply(xv, wv)
                    acc <- add(acc, prod) over k
                out[m, n] = acc
        """
    )


def test_matmul_cell_lifts_and_canonicalizes_as_contraction() -> None:
    tile = _tile(_matmul_body())
    assert [axis.extent for axis in tile.place.free] == [Dim(32), Dim(64)]
    node = tile.op
    assert isinstance(node, Fold) and node.axis is not None
    assert node.as_contraction() is not None
    assert [edge.as_slab().load.input for edge in node.operands] == ["x", "w"]
    assert len(tile.output_specs) == 1 and tile.output_specs[0].sweep == ()


def test_epilogue_stays_in_the_projection_body() -> None:
    tile = _tile(
        parse_body(
            """
            for m in 0..32:
                for n in 0..64:
                    for k in 0..128:
                        xv = load x[m, k]
                        wv = load w[n, k]
                        prod = multiply(xv, wv)
                        acc <- add(acc, prod) over k
                    bias = load b[n]
                    outv = add(acc, bias)
                    out[m, n] = outv
            """
        )
    )
    node = tile.op
    assert isinstance(node, Fold) and node.axis is None and len(node.operands) == 1
    assert node.operands[0].axis is not None
    assert {stmt.input for stmt in node.lift.body if isinstance(stmt, Load)} == {"b"}


def test_a_product_argument_computed_in_the_step_factors_into_a_cone_operand() -> None:
    """The norm→linear step ``acc += (x[m,k] * r[m]) * w[n,k]`` is a semiring step whose A is
    COMPUTED: formation hoists the ``x * r`` chain into a zero-axis operand so the lift is the
    product alone and the term reads as a contraction, A first, over the cone and the weight slab."""
    tile = _tile(
        parse_body(
            """
            for m in 0..32:
                for n in 0..64:
                    for k in 0..128:
                        xv = load x[m, k]
                        rv = load r[m]
                        xn = multiply(xv, rv)
                        wv = load w[n, k]
                        prod = multiply(xn, wv)
                        acc <- add(acc, prod) over k
                    out[m, n] = acc
            """
        )
    )
    node = tile.op
    view = node.as_contraction()
    assert view is not None and (tile.axis_of(view.left).extent, tile.axis_of(view.right).extent) == (Dim(32), Dim(64))
    cone, weight = node.operands
    assert cone.axis is None and len(cone.exposes) == 1 and weight.as_slab().load.input == "w"
    assert {edge.as_slab().load.input for edge in cone.operands} == {"x", "r"}
    assert [stmt.value.operation.name for stmt in node.lift.body] == ["multiply"]


def test_non_distributing_lift_stays_planar() -> None:
    tile = _tile(
        parse_body(
            """
            for m in 0..32:
                for n in 0..64:
                    for k in 0..128:
                        xv = load x[m, k]
                        wv = load w[n, k]
                        prod = maximum(xv, wv)
                        acc <- add(acc, prod) over k
                    out[m, n] = acc
            """
        )
    )
    node = tile.op
    assert isinstance(node, Fold) and node.axis is not None and node.axis is not None
    assert node.as_contraction() is None


def test_the_contraction_reading_gates_on_the_semiring() -> None:
    """Formation stores any product; the BILINEAR reading is what a ``(⊗, ⊕)`` pair that is not a
    registered semiring never gets."""
    a, b = slab("av", "a", "m", "k"), slab("bv", "b", "n", "k")
    assert contraction("k", a, (b, "acc"), product="maximum").as_contraction() is None
    view = contraction("k", a, (b, "acc")).as_contraction()
    assert (view.product.name, view.plus.name) == ("multiply", "add")


def test_total_lift_fires_through_the_pipeline() -> None:
    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    graph.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    graph.inputs, graph.outputs = ["x"], ["o"]
    lowered = Pipeline.build(TILE_PASSES).run(graph)

    from emmy.compiler.ir.tile import TileOp

    assert any(isinstance(node.op, TileOp) for node in lowered.nodes.values())


def test_singleton_reduce_over_an_enclosing_value_lifts_as_a_projection() -> None:
    """Decode softmax can hoist its sole value ahead of the extent-one max/sum loop.  The
    canonical Loop IR must collapse that identity reduction before total lift, rather than ask a
    fold lambda to return a name defined only by its enclosing projection."""
    body = parse_body(
        """
        for m in 0..4:
            xv = load x[m]
            for k in 0..1:
                acc <- maximum(acc, xv) over k
            out[m] = acc
        """
    )

    tile = _tile(body)

    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    graph.add_node(op=LoopOp(body=body), inputs=["x"], output=Tensor("out", (4,)), node_id="out")
    graph.inputs, graph.outputs = ["x"], ["out"]
    values = np.array([3.0, -2.0, 7.5, 0.25], dtype=np.float32)
    result = NumpyBackend().run(NumpyBackend().compile(graph), input_data={"x": values})[0].outputs["out"]

    assert tuple(axis.extent for axis in tile.place.free) == (Dim(4),)
    assert isinstance(tile.op, Fold) and tile.op.axis is None
    assert not any(edge.axis is not None for edge in tile.op.operands)
    assert tile.op.lower(_grid(tile), tile.output_specs, tile.axes)[-1].value in deep_defines(tile.op)
    np.testing.assert_array_equal(result, values)


def test_sibling_q_and_kv_regions_total_lift_with_separate_outputs() -> None:
    """A shared row feeding different Q and K/V extents remains one pure TileOp."""

    def contraction(axis: str, weight: str, acc: str) -> str:
        return f"""
                for k in 0..5:
                    x_{acc} = load x[m, k]
                    w_{acc} = load {weight}[{axis}, k]
                    p_{acc} = multiply(x_{acc}, w_{acc})
                    {acc} <- add({acc}, p_{acc}) over k"""

    body = parse_body(
        f"""
        for m in 0..3:
            for q in 0..4:{contraction("q", "wq", "q_acc")}
                q_out[m, q] = q_acc
            for kv in 0..2:{contraction("kv", "wk", "k_acc")}{contraction("kv", "wv", "v_acc")}
                k_out[m, kv] = k_acc
                v_out[m, kv] = v_acc
        """
    )

    tile = _tile(body)

    # The root projection has nothing of its own: its operands are the q contraction and the k/v
    # pair normalization merged over their shared row; the writes are boundary specs. Each output
    # loop's axis is a contraction output, but neither is the kernel's SHARED output sweep — the q
    # store does not ride kv, nor the k/v stores q — so post-init leaves both as sweeps: promoting
    # one would evaluate the other region once per cell of it. The row alone is on the grid.
    assert tile.op.axis is None and not tile.op.lift.body
    assert sorted((edge.as_contraction() is not None, len(edge.exposes)) for edge in tile.op.operands) == [(True, 1), (True, 2)]
    assert [axis.extent for axis in tile.place.free] == [Dim(3)]
    assert [tuple(axis.extent for axis in spec.sweep) for spec in tile.output_specs] == [(Dim(2),), (Dim(2),), (Dim(4),)]
    # The closed program opens the two sweeps as SIBLING loops under the row: no term is evaluated
    # over both, so neither nests in the other.
    (m_loop,) = tile.loop_body
    assert {loop.axis.extent for loop in m_loop.body} == {Dim(2), Dim(4)} and len(m_loop.body) == 2, "sibling sweeps, not a chain"
    assert {write.output for loop in m_loop.body for write in loop.body.writes} == {"q_out", "k_out", "v_out"}
    seam_scopes = {frozenset(axis.extent for axis in seam.axes) for seam in cuttable_seams(tile)}
    assert {frozenset((Dim(3), Dim(4))), frozenset((Dim(3), Dim(2)))} <= seam_scopes


def test_an_output_sweeps_epilogue_lifts_to_a_term_declaring_the_sweep_axis() -> None:
    """The per-cell projection under an output loop is a zero-axis term of the level, evaluated
    over the sweep coordinate it declares and reading the reduce as an operand; the root wrapper
    over it dissolves, the sweep axis (a contraction output) promotes onto the grid, and the
    store follows the term's exposed value."""
    tile = _tile(
        parse_body(
            """
            for m in 0..3:
                for n in 0..4:
                    for k in 0..5:
                        xv = load x[m, k]
                        wv = load w[n, k]
                        p = multiply(xv, wv)
                        acc <- add(acc, p) over k
                    b = load bias[n]
                    y = add(acc, b)
                    out[m, n] = y
            """
        )
    )

    epilogue = tile.op
    assert epilogue.axis is None and len(epilogue.exposes) == 1
    assert [edge.as_contraction() is not None for edge in epilogue.operands] == [True]
    assert Dim(4) in {axis.extent for axis in tile.axes if axis.name in epilogue.free_axes}
    assert [axis.extent for axis in tile.place.free] == [Dim(3), Dim(4)]
    (spec,) = tile.output_specs
    assert spec.write.values == epilogue.exposes and spec.sweep == ()
    assert [type(stmt).__name__ for stmt in tile.op.lower(_grid(tile), tile.output_specs, tile.axes)] == ["Loop", "Load", "Assign", "Write"]


def test_nested_output_sweep_preserves_an_intermediate_value() -> None:
    """A cell stores both its transformed result and an intermediate broadcast by a nested sweep."""
    body = parse_body(
        """
        for m in 0..2:
            scale = load s[m]
            for k in 0..3:
                xv = load x[m, k]
                v = multiply(scale, xv)
                q = exp(v)
                coded[m, k] = q
                for n in 0..4:
                    broadcast[m, k, n] = v
        for r in 0..5:
            zv = load z[r]
            tail[r] = zv
        """
    )
    graph = Graph()
    for name, shape in {"s": (2,), "x": (2, 3), "z": (5,)}.items():
        graph.add_node(InputOp(), [], Tensor(name, shape), node_id=name)
    outputs = [Tensor("coded", (2, 3)), Tensor("broadcast", (2, 3, 4)), Tensor("tail", (5,))]
    loop = LoopOp(body=body, inputs={name: graph.buffer(name) for name in ["s", "x", "z"]}, outputs={t.name: t for t in outputs})
    tile = lift_loop_op(loop)
    graph.add_node(LoopOp(body=tile.loop_body, inputs=loop.inputs, outputs=loop.outputs), ["s", "x", "z"], outputs=outputs, node_id="coded")
    graph.inputs, graph.outputs = ["s", "x", "z"], ["coded", "broadcast", "tail"]
    values = {
        "s": np.array([0.5, -0.25], dtype=np.float32),
        "x": np.arange(6, dtype=np.float32).reshape(2, 3),
        "z": np.arange(5, dtype=np.float32),
    }
    backend = NumpyBackend()
    got = backend.run(backend.compile(graph), input_data=values)[0].outputs
    intermediate = values["s"][:, None] * values["x"]
    np.testing.assert_allclose(got["coded"], np.exp(intermediate), rtol=1e-6)
    np.testing.assert_array_equal(got["broadcast"], np.broadcast_to(intermediate[..., None], (2, 3, 4)))
    np.testing.assert_array_equal(got["tail"], values["z"])


# ===================================================================
# Prologue placement (restored)
# ===================================================================


def test_reduce_feeding_prologue_reaches_the_fold_step() -> None:
    """A pure prologue value the reduce body reads must reach the fold's step; an epilogue-only
    value must not.

    The lift closes the fold over the prologue value it reads: the load arrives as the fold's own
    slab operand, hoisted once ahead of the loop by ``Fold.lower``, rather than sunk into the step.
    What matters, and what the miscompile below is about, is only that the step can see it and
    that the epilogue-only value stays out of the fold."""
    node = _tile(
        parse_body(
            """
            for m in 0..8:
                scale = load s[0]
                bias = load b[0]
                for k in 0..16:
                    xv = load x[m, k]
                    sx = multiply(xv, scale)
                    acc <- add(acc, sx) over k
                outv = add(acc, bias)
                out[m] = outv
            """
        )
    ).op
    assert isinstance(node, Fold) and node.axis is None
    folds = [edge for edge in node.operands if edge.axis is not None]
    assert len(folds) == 1, "the reduce did not lift to exactly one fold"
    step = folds[0]
    buffers = {edge.as_slab().load.input for edge in step.operands if edge.as_slab() is not None}
    assert buffers == {"x", "s"}, "the fold's step cannot see the prologue value it multiplies by"
    assert {stmt.input for stmt in node.lift.body if isinstance(stmt, Load)} == {"b"}, "an epilogue-only value must not reach the fold"


def test_multi_pass_cell_defines_every_name_before_it_is_read() -> None:
    """The decode-attention ``v1``/``v3`` miscompile: a pure prologue value read by an EARLIER
    reduce pass must stay ahead of it. Sinking it into a later pass's lambda leaves the earlier
    pass reading an undefined name — a wrong kernel, not a slow one, and no numerics assert
    downstream attributes the wrong answer to placement.

    The subject is SDPA, the shape the defect came from: three passes over one cell, where the
    score feeds the maximum, the denominator AND the value product. The assertion is the general
    well-formedness invariant, so it survives however those passes are represented — every name a
    statement reads is defined by an earlier statement at the same or an enclosing level."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES  # noqa: PLC0415

    graph, _, _ = graph_from_code(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 8, 4, dtype=torch.float16), "
        "torch.randn(1, 1, 8, 4, dtype=torch.float16), "
        "torch.randn(1, 1, 8, 4, dtype=torch.float16))"
    )
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    graph = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    tiles = [node.op for node in graph.nodes.values() if isinstance(node.op, TileOp)]
    assert tiles, "SDPA produced no Tile IR"

    def check(body, defined):
        for stmt in body:
            if isinstance(stmt, Loop):
                check(list(stmt.body), set(defined) | {stmt.axis.name})
                defined |= set(deep_defines(stmt))
                continue
            reads = set(stmt.deps())
            assert reads <= defined, f"{stmt} reads {sorted(reads - defined)} before definition"
            defined |= set(stmt.defines())

    for tile in tiles:
        check(list(tile.op.lower(_grid(tile), tile.output_specs, tile.axes)), {axis.name for axis in tile.place.free})


def _tile_with_shapes(body: Body, out_shape: tuple, inputs: dict) -> object:
    """Lift ``body`` with REAL operand and output shapes, which the size-one row reading needs."""
    graph = Graph()
    for name, shape in inputs.items():
        graph.add_node(InputOp(), [], Tensor(name, shape), node_id=name)
    graph.add_node(LoopOp(body=body), list(inputs), Tensor("out", out_shape), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["tile/lift"], select=["lift"]).run(graph).nodes["out"].op


def _decode_body() -> Body:
    """``out[0, h, 0, n] = Σ_k q[0, h, 0, k] · v[0, h, k, n]`` — decode's expectation, whose row
    Loop IR elides at extent one and whose only other free axis, the head, BOTH operands read."""
    return parse_body(
        """
        for h in 0..8:
            for n in 0..16:
                for k in 0..32:
                    qv = load q[0, h, 0, k]
                    vv = load v[0, h, k, n]
                    prod = multiply(qv, vv)
                    acc <- add(acc, prod) over k
                out[0, h, 0, n] = acc
        """
    )


def test_a_contraction_with_no_row_binds_the_size_one_output_coordinate() -> None:
    """A term whose A owns no free axis has no row for any tier to tile: everything it reads it
    shares with the operand it multiplies, and a B that moves with its row is no slab per tile. The
    coordinate that would have been the row is the size-one output dimension normalization inlined,
    and the lift binds it back — into the OPERAND's index too, since a row an operand never reads
    leaves ``left_axes`` empty and is refused just the same.

    ``_implicit_unit_row`` cannot serve this shape: A reads the shared head coordinate, so the
    missing row must be bound into its index rather than merely announced in the placement.
    """
    tile = _tile_with_shapes(_decode_body(), (1, 8, 1, 16), {"q": (1, 8, 1, 32), "v": (1, 8, 32, 16)})
    bound = [axis for axis in tile.place.free if axis.extent == Dim(1)]
    assert len(bound) == 1, f"expected one bound row, got {[axis.name for axis in tile.place.free]}"
    row = bound[0].name
    view = tile.op.as_contraction()
    assert view is not None and view.left_axes == frozenset({row})
    reads = [edge.as_slab().load for edge in tile.op.operands if edge.as_slab() is not None]
    assert any(load.input == "q" and Var(row) in load.index for load in reads), "the row must be READ by A"
    assert all(load.input != "v" or Var(row) not in load.index for load in reads), "the value must not read the row"


def test_a_contraction_that_owns_a_row_gains_no_bound_axis() -> None:
    """The binding fires on a missing row only: a matmul already has one, and adding an extent-one
    axis there would change every such kernel's identity for nothing."""
    tile = _tile_with_shapes(_matmul_body(), (32, 64), {"x": (32, 128), "w": (64, 128)})
    assert [axis.extent for axis in tile.place.free] == [Dim(32), Dim(64)]


def test_a_sum_ahead_of_an_inner_parallel_loop_peels_onto_the_placement() -> None:
    """A row statistic the outer level computes as a sum (one node of the lift, as the normal form's affine form is one
    sum) is straight-line code like any ``Assign``: the inner parallel loop still peels onto the placement, so the
    kernel keeps both free axes and its vector tiles instead of an outer sweep over a hoisted operand."""
    tile = _tile(
        parse_body(
            """
            for r in 0..3:
                s = load stat[r]
                twice = add(s, s)
                scale = rsqrt(twice)
                for c in 0..4:
                    x = load x[r, c]
                    y = multiply(scale, x)
                    out[r, c] = y
            """
        )
    )
    assert [axis.extent for axis in tile.place.free] == [Dim(3), Dim(4)]
    assert tile.op.axis is None and tile.op.operands == ()
    assert [type(stmt).__name__ for stmt in tile.op.lift.body] == ["Load", "Assign", "Assign", "Load", "Assign"]


def test_a_body_lifts_to_one_fold_tree_before_and_after_the_fold() -> None:
    """The lift forms one node per operation, whatever the Loop IR spelling: the normal form folds a value read once
    into its reader, and the chain spelling of the same program names every value. Both lift to the same Fold tree,
    which is what keeps the cones, the edges and the seams independent of the fold."""
    # The chain spelling of the normal form below, under the normal form's own names: every value a statement.
    chain = parse_body(
        """
        for a0 in 0..32:
            in0 = load r[a0]
            for a1 in 0..64:
                for a2 in 0..128:
                    in1 = load w[a1, a2]
                    in2 = load x[a0, a2]
                    e = exp(in2)
                    xn = multiply(e, in0)
                    prod = multiply(xn, in1)
                    acc0 <- add(acc0, prod) over a2
                in3 = load s[a1]
                scaled = multiply(acc0, in3)
                v0 = add(scaled, in3)
                out[a0, a1] = v0
        """
    )
    op = LoopOp(body=chain)  # normalized at construction: the single-use values fold into their readers
    assert "acc0 <- add(acc0, exp(in2) * in0 * in1) over a2" in str(op.body)
    folded, spelled = lift_loop_op(op, name="k"), lift_loop_op(op, name="k", body=chain)
    assert folded.op.canonical() == spelled.op.canonical()
    assert {seam.spelling for seam in cuttable_seams(folded)} == {seam.spelling for seam in cuttable_seams(spelled)}
