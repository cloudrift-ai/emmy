"""The free-axis canonicalization every kernel forms through: re-fuse adjacent free axes that a fused reshape split.

A view fused into a contraction iterates the post-view axes while the operand loads address the
producer's single axis through a composite index — which locks the kernel out of contraction
binding (the trailing free pair reads the wrong row and the weight load carries a third grid
axis). The canonicalization restores the single-axis spelling; these tests pin the fuse cases
(N split, M split, the pair across an intervening free loop — the transposed projection, the
permuted split store), the decline case (an axis addressed alone), the downstream classification
of the canonical nest, and the warp tier's split-store addressability.

The operand ROLE-PURITY section at the end was deleted with ``_classify.bind_bilinear`` and is
RESTORED against the canonical Fold tree. Its contracts are about correctness, not coverage: a
grouped B address that varies with the output row is not one slab per tile, and trying
the opposite operand orientation is licensed only for a COMMUTATIVE product — reordering a
noncommutative one computes a different value."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Body, Load, Loop, Write, parse_body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.tile import _free_axes as _SPLIT

M, H, D, K = 8, 3, 4, 16  # N = H*D = 12


def _graph(body: Body, out_shape: tuple) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K)), node_id="w")
    g.add_node(LoopOp(body=body), ["x", "w"], Tensor("out", out_shape), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]
    return g


def _split_n_body(write_index: str) -> Body:
    """The fused-reshape spelling: free (m, h, d), the weight addressed ``w[h*D + d, k]``."""
    return parse_body(
        f"""
        for a0 in 0..{M}:
            for a1 in 0..{H}:
                for a2 in 0..{D}:
                    for k in 0..{K}:
                        wv = load w[a1 * {D} + a2, k]
                        xv = load x[0, a0, k]
                        prod = multiply(wv, xv)
                        acc <- add(acc, prod) over k
                    out[{write_index}] = acc
        """
    )


def _free_chain(op: LoopOp) -> list[Loop]:
    out, cur = [], op.body
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        out.append(cur[0])
        cur = cur[0].body
    return out


def _run(g: Graph) -> LoopOp:
    """The ``out`` kernel's loop op with its free coordinates canonical, as the tile lift leaves them."""
    node = g.nodes["out"]
    op = node.op.with_io(g, node)
    body = _SPLIT.canonical_free_axes(op.body, {name: t.shape for name, t in {**op.inputs, **op.outputs}.items()})
    return op if body is None else replace(op, body=body)


def _mixed_row_head_graph(extent: int) -> Graph:
    body = parse_body(
        f"""
        for i in 0..{extent}:
            for d in 0..{D}:
                for k in 0..{K}:
                    xv = load x[0, i // {H}, k]
                    wv = load w[i % {H} * {D} + d, k]
                    prod = multiply(xv, wv)
                    acc <- add(acc, prod) over k
                out[i, d] = acc
        """
    )
    return _graph(body, (extent, D))


def test_flattened_row_head_restores_separate_operand_axes():
    graph = _mixed_row_head_graph(M * H)
    op = _run(graph)
    loads = {load.input: load for load in op.loads}
    assert not any(
        isinstance(expression, BinaryExpr) and expression.op in ("/", "//", "%")
        for load in op.loads
        for index in load.index
        for expression in index.subterms()
    )
    assert len(loads["x"].index[1].free_vars()) == 1
    assert not loads["x"].index[1].free_vars() & loads["w"].index[0].free_vars()
    assert _run(graph).body == op.body


def test_flattened_row_head_with_partial_last_head_stays_flat():
    graph = _mixed_row_head_graph(M * H - 1)
    before = graph.nodes["out"].op
    assert _run(graph).body == before.body


def _lift(op: LoopOp, shape=(1,)) -> TileOp:
    graph = Graph()
    graph.add_node(op, [], Tensor("out", shape), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["tile/lift"], select=["lift"]).run(graph).nodes["out"].op


def _reduce_name(op: LoopOp) -> str:
    return next(s.axis.name for s in op.body.iter() if isinstance(s, Loop) and s.is_reduce)


def test_split_n_pair_fuses():
    op = _run(_graph(_split_n_body("0, a0, a1, a2"), (1, M, H, D)))
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)], "the (h, d) pair must fuse into one N axis"
    n = chain[1].axis.name
    wv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "w")
    assert wv.index == (Var(n), Var(_reduce_name(op))), "the composite operand index must collapse to the bare fused axis"
    wr = next(s for s in op.body.iter() if isinstance(s, Write))
    assert wr.index[2] == BinaryExpr("/", Var(n), Literal(D, "int"))
    assert wr.index[3] == BinaryExpr("%", Var(n), Literal(D, "int"))


def test_split_m_pair_fuses():
    """An M-side split (``view`` carving the row axis) fuses the same way: composite x load,
    split write rows."""
    body = parse_body(
        f"""
        for a0 in 0..2:
            for a1 in 0..4:
                for n in 0..{H * D}:
                    for k in 0..{K}:
                        wv = load w[n, k]
                        xv = load x[0, a0 * 4 + a1, k]
                        prod = multiply(wv, xv)
                        acc <- add(acc, prod) over k
                    out[0, a0, a1, n] = acc
        """
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K)), node_id="w")
    g.add_node(LoopOp(body=body), ["x", "w"], Tensor("out", (1, 2, 4, H * D)), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]
    op = _run(g)
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)]
    xv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "x")
    assert xv.index == (Literal(0, "int"), Var(chain[0].axis.name), Var(_reduce_name(op)))


def _packed_split_graph() -> Graph:
    """The deployed shape: a pair-packed weight under an output whose N axis a fused fp4-block
    reshape split into ``(H, D)``. The NVFP4 weight read: two 4-bit codes per byte, so the operand
    address is the FLAT element offset divided by 2 and wrapped back into the packed row —
    ``w[n, ((n*K + k) / 2) % (K/2)]``, the address a pair-packed constant carries."""
    body = parse_body(
        f"""
        for a0 in 0..{M}:
            for a1 in 0..{H}:
                for a2 in 0..{D}:
                    for k in 0..{K}:
                        wv = load w[a1 * {D} + a2, ((a1 * {D} + a2) * {K} + k) / 2 % {K // 2}]
                        xv = load x[0, a0, k]
                        prod = multiply(wv, xv)
                        acc <- add(acc, prod) over k
                    out[0, a0, a1, a2] = acc
        """
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K // 2)), node_id="w")
    g.add_node(LoopOp(body=body), ["x", "w"], Tensor("out", (1, M, H, D)), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]
    return g


def test_packed_operand_split_pair_fuses():
    """A pair-packed weight address holds the row axis inside a division. The pair still fuses: the
    composite collapses to the bare axis, and the division that remains reads ``k`` alone."""
    op = _run(_packed_split_graph())
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)], "the packed pair must fuse"
    n = chain[1].axis.name
    wv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "w")
    assert wv.index[0] == Var(n), "the packed row dim must collapse to the bare fused axis"
    assert n not in wv.index[1].free_vars(), f"the row axis stayed inside the packed offset: {wv.index[1]}"


def test_packed_split_nest_classifies_as_contraction():
    """Downstream: the fused packed nest binds the contraction with the weight as B — the tensor-core
    tier becomes reachable, which is what the split had locked out."""
    tile = _lift(_run(_packed_split_graph()))
    assert [a.extent for a in tile.place.free] == [Dim(M), Dim(H * D)]
    assert tile.op.axis is not None and tile.op.as_contraction() is not None


def test_axis_addressed_alone_declines():
    """A per-column bias read ``b[d]`` pins ``d``'s identity apart from ``h`` — the pair must
    not fuse (the residue ``b[f%D]`` would survive in a load)."""
    body = parse_body(
        f"""
        for a0 in 0..{M}:
            for a1 in 0..{H}:
                for a2 in 0..{D}:
                    for k in 0..{K}:
                        wv = load w[a1 * {D} + a2, k]
                        xv = load x[0, a0, k]
                        prod = multiply(wv, xv)
                        acc <- add(acc, prod) over k
                    bv = load b[a2]
                    outv = add(acc, bv)
                    out[0, a0, a1, a2] = outv
        """
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K)), node_id="w")
    g.add_node(InputOp(), [], Tensor("b", (D,)), node_id="b")
    g.add_node(LoopOp(body=body), ["x", "w", "b"], Tensor("out", (1, M, H, D)), node_id="out")
    g.inputs, g.outputs = ["x", "w", "b"], ["out"]
    op = _run(g)
    assert [ln.axis.extent for ln in _free_chain(op)] == [Dim(M), Dim(H), Dim(D)], "the pair must decline"


def test_permuted_split_store_keeps_the_axes_separate():
    """Output-storage order is canonical, so a transposed ``[d, h]`` store keeps the pair split."""
    op = _run(_graph(_split_n_body("0, a0, a2, a1"), (1, M, D, H)))
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(D), Dim(H)]
    wr = next(s for s in op.body.iter() if isinstance(s, Write))
    assert wr.index == (Literal(0, "int"), *(Var(loop.axis.name) for loop in chain))


def _transposed_body() -> Body:
    """The attention projection's ``view(b, s, h, d).transpose(1, 2)``: free (h, s, d) — the
    split pair (h, d) is NOT adjacent, ``s`` sits between — with the weight addressed
    ``w[h*D + d, k]`` and the output stored ``[0, h, s, d]``."""
    return parse_body(
        f"""
        for a0 in 0..{H}:
            for a1 in 0..{M}:
                for a2 in 0..{D}:
                    for k in 0..{K}:
                        wv = load w[a0 * {D} + a2, k]
                        xv = load x[0, a1, k]
                        prod = multiply(wv, xv)
                        acc <- add(acc, prod) over k
                    out[0, a0, a1, a2] = acc
        """
    )


def test_split_pair_fuses_across_an_intervening_free_loop():
    """The transposed projection: the pair interchanges past ``s`` (free loops are parallel) and
    fuses under it — the weight load collapses to the bare axis, the store keeps the permuted
    ``[0, f//D, s, f%D]``."""
    op = _run(_graph(_transposed_body(), (1, H, M, D)))
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)], "s stays outer; the (h, d) pair fuses under it"
    n = chain[1].axis.name
    wv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "w")
    assert wv.index == (Var(n), Var(_reduce_name(op)))
    wr = next(s for s in op.body.iter() if isinstance(s, Write))
    s_name = chain[0].axis.name
    assert wr.index == (
        Literal(0, "int"),
        BinaryExpr("/", Var(n), Literal(D, "int")),
        Var(s_name),
        BinaryExpr("%", Var(n), Literal(D, "int")),
    )


def test_transposed_canonical_nest_orders_free_by_the_remainder_dim():
    """Downstream: the lift positions an axis by the INNERMOST store dim carrying it, so under
    the permuted store the fused axis stays the trailing ``n`` (its ``%`` dim is the contiguous
    one) and the contraction binds with the weight as B. Positioning by the quotient dim would
    make the fused axis ``m`` and the stride-``D`` ``s`` the column — the mma store's ``+ col``
    would then address the wrong element (and did: a hang on the GPU)."""
    op = _run(_graph(_transposed_body(), (1, H, M, D)))
    tile = _lift(op)
    assert [a.extent for a in tile.place.free] == [Dim(M), Dim(H * D)]
    node = tile.op
    assert node.axis is not None and node.as_contraction() is not None
    assert node.operands[1].as_slab().load.input == "w"


def test_canonical_nest_classifies_as_contraction():
    """Downstream of the canonicalization: the fused nest binds the contraction with the split
    store, and the free order keeps the fused N trailing (position read through the div/mod
    expressions, not bare Vars)."""
    body = parse_body(
        f"""
        for a0 in 0..{M}:
            for n in 0..{H * D}:
                for k in 0..{K}:
                    wv = load w[n, k]
                    xv = load x[0, a0, k]
                    prod = multiply(wv, xv)
                    acc <- add(acc, prod) over k
                out[0, a0, n // {D}, n % {D}] = acc
        """
    )
    tile = _lift(LoopOp(body=body))
    assert [a.extent for a in tile.place.free] == [Dim(M), Dim(H * D)]
    node = tile.op
    assert node.axis is not None and node.as_contraction() is not None
    assert node.operands[1].as_slab().load.input == "w"


# --- the warp tier's split-store addressability --------------------------------------------------- #


def _split_store_ok(index: str, shape: tuple, free_names=("m", "n"), atom=(16, 8, 16)) -> bool:
    """Whether an mma fragment store with output ``atom`` cells can address ``index`` — the
    scheduler's own gate (``_split_store_refusal``), so the roles mapping under test is the
    production one."""
    from emmy.compiler.ir.schedule.classic.refusals import _split_store_refusal

    free = tuple(Axis(nm, Dim(4)) for nm in free_names)
    shapes = {"out": Tensor("out", shape)}
    return _split_store_refusal(list(parse_body(f"out[{index}] = acc")), free, atom, shapes) is None


def test_warp_split_store_legality():
    """The fragment store evaluates the cell base once per atom and adds ``col`` / ``row·ldm``:
    a split dim pair is addressable when the row-major flatten recomposes it (strides match the
    split, any ``Q``), or when the ``%`` dim is the innermost carrier, contiguous for ``n``, with
    ``Q`` a multiple of the atom extent — an aligned atom never straddles a ``Q`` boundary."""
    # #561's row-major N split: affine recomposition, Q=4 is fine.
    assert _split_store_ok("0, m, n // 4, n % 4", (1, 8, 3, 4))
    # The transposed projection: permuted strides, Q % 8 == 0.
    assert _split_store_ok("0, n // 32, m, n % 32", (1, 2, 8, 32))
    # Permuted with Q=12: an 8-wide atom straddles a head boundary.
    assert not _split_store_ok("0, n // 12, m, n % 12", (1, 4, 8, 12))
    # The within-pair transpose (quotient dim inner) never addresses — in either stride regime.
    assert not _split_store_ok("0, m, n % 32, n // 32", (1, 8, 32, 2))
    # An M-side permuted split (a batch dim between the row's pair) needs 16-row atoms inside
    # one ``P`` block.
    assert _split_store_ok("0, m // 32, b, m % 32, n", (1, 2, 3, 32, 16))
    assert not _split_store_ok("0, m // 8, b, m % 8, n", (1, 2, 3, 8, 16))


def test_warp_epilogue_preserves_split_and_block_scale_coordinates():
    """The lane moves the source coordinate before division, including across a split boundary."""
    from emmy.compiler.ir.kernel.ir import ELEM_COL, ELEM_ROW
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue

    tail = list(
        parse_body(
            """
            scale = load scales[b, m, n // 32, n % 32, n // 8]
            result = multiply(acc, scale)
            out[m, n] = result
            """
        )
    )
    sigma = Sigma({"b": Literal(2, "int"), "m": Literal(16, "int"), "n": Literal(30, "int")})
    epilogue = _warp_epilogue(tail, "acc", "m", "n", sigma)["epilogue"]
    load = epilogue[0]
    for col in range(8):
        got = tuple(expression.eval({ELEM_ROW: 3, ELEM_COL: col}) for expression in load.index)
        assert got == (2, 19, (30 + col) // 32, (30 + col) % 32, (30 + col) // 8)


# --- operand role purity and product orientation (restored) --------------------------------------- #


def _bilinear_fold(w_index: str, x_index: str, product: str = "multiply", swapped: bool = False):
    """The lifted ``Σ_k w ⊗ x`` cell; ``swapped`` spells the loads and the product's arguments the other way round."""
    from emmy.compiler.pipeline.passes.tile._fromloop import fold_from_loop

    loads = (f"wv = load w[{w_index}]", f"xv = load x[{x_index}]")
    args = ("wv", "xv")
    if swapped:
        loads, args = loads[::-1], args[::-1]
    (loop,) = parse_body(
        f"""
        for k in 0..{K}:
            {loads[0]}
            {loads[1]}
            prod = {product}({args[0]}, {args[1]})
            acc <- add(acc, prod) over k
        """
    )
    return fold_from_loop(loop)


def _bind(fold, free_names):
    """The lifted fold as the term a kernel over ``free_names`` schedules; the contraction, or ``None``
    when the term keeps its PLANAR reading (the decline every negative case below asserts)."""
    from emmy.compiler.ir.tile import Placement

    free = tuple(Axis(n, Dim(4)) for n in free_names)
    tile = TileOp(op=fold, place=Placement(free=free), axes=(*free, Axis("k", Dim(K))))
    return tile.op if tile.contracts(tile.node_id(tile.op)) else None


def test_bilinear_batched_operand_still_binds():
    """Role purity must not over-reach: a batch offset riding a SEPARATE dim of the A load
    (batched GEMM) binds exactly as an unbatched one does.

    The batch dim is a grid offset, not a scheduling axis — free-axis canonicalization folds a leading
    batch into the row axis before lowering, so the canonical term sees the ordinary ``(m, n)``
    pair with the offset still spelled in A's index."""
    con = _bind(_bilinear_fold("n, k", "b, a0, k"), ("a0", "n"))
    assert con is not None, "the batched GEMM demoted to PLANAR"
    assert con.operands[0].as_slab().load.input == "x"


def test_bilinear_binds_a_composite_role_expression():
    """A third free axis composed into the SAME index expression as the role axis (the split-axis
    composite ``4*a1 + n``) binds as the direct B load: each side's own axes stay its role, the
    tile orients the trailing one and the rest ride the grid, which defines them. The Qwen3.8 AWQ
    down projection has this shape; refusing it left the piece off the tensor cores."""
    con = _bind(_bilinear_fold(f"a1 * {D} + n, k", "b, a0, k"), ("b", "a1", "a0", "n"))
    assert con is not None, "the composite demoted to PLANAR"
    assert con.as_contraction().right_axes == {"a1", "n"}
    assert con.operands[1].as_slab().load.input == "w"


def test_bilinear_binding_is_independent_of_the_product_argument_order():
    """A flattened GQA value row binds the same whichever way the commutative product is spelled.

    Fusion emits the value load first in the deployed attention cell, so an order-sensitive binder
    would bind one spelling and decline its twin. What the B edge becomes is NOT asserted here: the
    old binder answered slab addressability itself by forcing a computed cone, and that question
    now belongs to the scheduler's warp-atom gate. Order independence is the part that is still this rule's."""
    w_index = f"0, k, 0, h // 3 * {D} + n"
    x_index = "h, m, k"

    forward = _bind(_bilinear_fold(w_index, x_index), ("h", "m", "n"))
    assert forward is not None, "the grouped value row demoted to PLANAR"
    assert forward.operands[0].as_slab().load.input == "x"

    reversed_products = _bind(_bilinear_fold(w_index, x_index, swapped=True), ("h", "m", "n"))
    assert reversed_products is not None
    assert reversed_products.canonical() == forward.canonical()


def test_bilinear_binds_grouped_b_per_batch():
    """The grouped B address varies between h,m batches while each fragment has one row."""
    fold = _bilinear_fold(f"0, k, 0, m * {H * D} + h // 3 * {D} + n", "h, m, k")

    bound = _bind(fold, ("h", "m", "n"))
    assert bound is not None
    assert bound.as_contraction().shared_axes == {"h", "m"}
    assert bound.operands[1].as_slab().load.input == "w"


def test_bilinear_does_not_reorder_a_noncommutative_product():
    """Trying the opposite direct/computed role is licensed only for a COMMUTATIVE product.
    Reordering ``subtract`` computes a different value — a miscompile, not a missed schedule."""
    fold = _bilinear_fold(f"0, k, 0, h // 3 * {D} + n", "h, m, k", product="subtract")

    assert _bind(fold, ("h", "m", "n")) is None


# The quotient split: a free coordinate read through ``/ Q`` and ``% Q`` splits into its two factors,
# unless every reduction that reads those factors also reads the coordinate whole — a packed int4
# weight reads its channel whole beside the zero-point's ``n / 8`` and the shift's ``n % 8``.


def _split(text: str) -> bool:
    return _SPLIT._split_once(parse_body(text), frozenset({"n", "k", "j"})) is not None


def test_a_coordinate_read_only_through_its_factors_splits():
    assert _split(
        """
        for n in 0..64:
            for k in 0..16:
                z = load zeros[n / 8]
                s = load shift[n % 8]
                acc_k <- add(acc_k, z) over k
        """
    )
    # A whole read after the reduction (an epilogue gate) owns nothing the contraction reads.
    assert _split(
        """
        for n in 0..64:
            for k in 0..16:
                z = load zeros[n / 8]
                s = load shift[n % 8]
                acc_k <- add(acc_k, z) over k
            g = load gate[n]
        """
    )


def test_a_packed_channel_read_whole_by_its_weight_stays_one_axis():
    assert not _split(
        """
        for n in 0..64:
            for k in 0..16:
                w = load weights[n]
                z = load zeros[n / 8]
                s = load shift[n % 8]
                acc_k <- add(acc_k, w) over k
        """
    )


def test_a_head_pair_splits_when_another_reduction_reads_only_its_factors():
    """One contraction reads the flattened head coordinate whole beside its head; another reads the
    head and dim apart. Only the split lets the second bind its operands, so it still happens."""
    assert _split(
        """
        for n in 0..64:
            for k in 0..16:
                q = load q[n]
                h = load h[n / 8]
                acc_k <- add(acc_k, q) over k
            for j in 0..16:
                a = load a[n / 8]
                b = load b[n % 8]
                acc_j <- add(acc_j, a) over j
        """
    )
