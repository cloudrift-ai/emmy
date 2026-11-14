"""The canonical Tile IR tree — what formation states about a term and what only the whole tree
can: ``TileOp.__post_init__`` (an elided unit row recovered from the stores, a shared store sweep
promoted to a free axis, the identity projection dissolved), the bilinear form the lift builds
(cone operands, orientation, one shared A), and ``_share_common_cones`` restoring one object per
value."""

from __future__ import annotations

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Write, parse_body
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.normalize import _share_common_cones
from emmy.compiler.ir.tile.path import family_sites, sites
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.tile._cut import cuttable_seams
from emmy.compiler.pipeline.passes.tile._fromloop import fold_from_loop
from tests.compiler.helpers import case_target_tile
from tests.compiler.terms import contraction, projection, reduction, slab

M8, N16, K32 = Axis("m", 8), Axis("n", 16), Axis("k", Dim(32))


def _lift(body: Body) -> TileOp:
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["tile/lift"], select=["lift"]).run(graph).nodes["out"].op


def _reduce_loop(text: str) -> Fold:
    """A reduce loop's text lifted to its term: the ONE former, so a fixture reads exactly like a
    lifted kernel — every load a slab, every computed product argument a cone operand."""
    (loop,) = parse_body(text)
    return fold_from_loop(loop)


def _spec(text: str, *sweep: Axis) -> OutputSpec:
    """The output specification of the one store ``text`` spells."""
    (write,) = parse_body(text)
    return OutputSpec(write=write, sweep=sweep)


def _matmul(a_index: str = "m, k", b_index: str = "n, k", *, extent: int = 32) -> Fold:
    return _reduce_loop(
        f"""
        for k in 0..{extent}:
            left = load x[{a_index}]
            right = load w[{b_index}]
            product = multiply(left, right)
            acc <- add(acc, product) over k
        """
    )


def _input(edge: Fold) -> str | None:
    """The buffer a slab operand reads, ``None`` for a computed edge."""
    view = edge.as_slab()
    return None if view is None else view.load.input


def _tile(op, *axes: Axis, **fields) -> TileOp:
    free = fields.pop("free", ())
    return TileOp(op=op, place=Placement(free=free), axes=(*free, *axes), **fields)


# ---- construction canonical forms -------------------------------------------------------------- #


def test_tile_post_init_canonicalizes_contraction() -> None:
    tile = _tile(projection((_matmul(),)), K32, free=(M8, N16))

    assert isinstance(tile.op, Fold) and tile.op.as_contraction() is not None  # the identity projection dissolves
    assert _input(tile.op.operands[0]) == "x"
    assert _input(tile.op.operands[1]) == "w"
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes).op == tile.op


def test_tile_post_init_canonicalizes_broadcast_batched_contraction() -> None:
    planar = _matmul(a_index="k, batch * 8 + m")

    tile = _tile(planar, K32, free=(Axis("batch", 4), M8, N16))

    assert tile.op.as_contraction() is not None
    assert _input(tile.op.operands[0]) == "w"
    assert _input(tile.op.operands[1]) == "x"


def test_tile_post_init_recovers_an_elided_unit_contraction_row() -> None:
    planar = _matmul(a_index="0, k", extent=16)

    tile = _tile(planar, Axis("k", Dim(16)), free=(N16,), output_specs=(_spec("out[0, n] = acc"),))

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.op.as_contraction() is not None


@pytest.mark.parametrize("index", ("n, 0", "0, n * 2"), ids=("varying-coordinate-before-zero", "strided-column"))
def test_tile_post_init_does_not_infer_a_unit_row_from_a_non_dense_boundary(index) -> None:
    planar = _matmul(a_index="0, k", extent=16)

    tile = _tile(planar, Axis("k", Dim(16)), free=(N16,), output_specs=(_spec(f"out[{index}] = acc"),))

    assert tuple(axis.name for axis in tile.place.free) == ("n",)


def test_contraction_promotes_a_shared_store_sweep_once() -> None:
    stores = tuple(_spec("out[m, n] = acc", N16) for _ in range(2))

    tile = _tile(_matmul(), N16, K32, free=(M8,), output_specs=stores)

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert all(store.sweep == () for store in tile.output_specs)


def test_contraction_promotes_a_shared_store_sweep_after_grid_mapping() -> None:
    """A mapped tile keeps promotion as a construction invariant."""
    normalized = _tile(_matmul(), K32, free=(M8, N16)).op
    tile = TileOp(
        op=normalized,
        place=Placement(free=(M8,), grid=(M8,), mapped=True),
        axes=(M8, N16, K32),
        output_specs=(_spec("out[m, n] = acc", N16),),
    )

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tuple(axis.name for axis in tile.place.grid) == ("m", "n")
    assert tile.place.is_mapped
    assert tile.output_specs[0].sweep == ()


def test_nested_contraction_promotes_a_shared_store_sweep() -> None:
    """A sibling reduction can be the root-most node while a later contraction reads the sweep axis."""
    r = Axis("r", 4)
    stat = reduction(r, (slab("sample", "s", "m", "r"),), parse_body("stat__v = copy(sample)"), ("stat",))
    root = projection((stat, _matmul()), parse_body("result = add(stat, acc)"), ("result",))
    tile = _tile(root, N16, r, K32, free=(M8,), output_specs=(_spec("out[m, n] = result", N16),))

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tile.output_specs[0].sweep == ()


def test_sibling_output_sweeps_stay_sweeps() -> None:
    """Two sibling output nests, each holding the contraction whose output coordinate it sweeps. Neither
    axis is the kernel's shared output sweep — every store rides its own — so neither is promoted:
    promoting one would evaluate the other nest per cell. DeepSeek-V4 post4096's residual root holds
    four such nests, and promoting all seven of their axes made its grid their 2^56-cell product."""
    p = Axis("p", 4)
    second = _reduce_loop(
        """
        for k in 0..32:
            left2 = load x[m, k]
            right2 = load u[p, k]
            product2 = multiply(left2, right2)
            acc2 <- add(acc2, product2) over k
        """
    )
    body = parse_body(
        """
        a = relu(acc)
        b = relu(acc2)
        """
    )
    root = projection((_matmul(), second), body, ("a", "b"))
    stores = (_spec("out[m, n] = a", N16), _spec("out2[m, p] = b", p))

    tile = _tile(root, N16, p, K32, free=(M8,), output_specs=stores)

    assert tuple(axis.name for axis in tile.place.free) == ("m",)
    assert tuple(tuple(axis.name for axis in store.sweep) for store in tile.output_specs) == (("n",), ("p",))


def _independent_output_tile(reverse=False):
    p, q, r, s = (Axis(name, size) for name, size in zip(("p", "q", "r", "s"), (3, 5, 7, 2), strict=True))
    first = slab("xv", "x", "p", "q")
    second = projection((slab("yv", "y", "p", "r"),), parse_body("zv = relu(yv)"), ("zv",))
    third = slab("wv", "w", "s")
    root = projection((first, second, third), parse_body("let unused = 0.0"), ("unused",))
    specs = (_spec("xo[p, q] = xv", p, q), _spec("yo[p, r] = zv", p, r), _spec("wo[s] = wv", s))
    return _tile(root, p, q, r, s, output_specs=specs[::-1] if reverse else specs)


@pytest.mark.parametrize("reverse", [False, True])
def test_dead_root_value_does_not_multiply_independent_output_sweeps(reverse):
    tile = _independent_output_tile(reverse)
    domains = {}

    def visit(body, path=()):
        for stmt in body:
            if isinstance(stmt, Loop):
                visit(stmt.body, (*path, stmt.axis.name))
            elif isinstance(stmt, Write):
                domains[stmt.output] = set(path)

    visit(tile.loop_body)
    assert domains == {"xo": {"p", "q"}, "yo": {"p", "r"}, "wo": {"s"}}


@pytest.mark.xdist_group("cuda")
def test_independent_output_computations_match_on_cuda():
    import numpy as np

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.base import InputOp
    from tests.compiler.helpers import skip_if_no_cuda

    skip_if_no_cuda()
    tile = _independent_output_tile()
    graph = Graph()
    arrays = {
        name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 4 for name, shape in (("x", (3, 5)), ("y", (3, 7)), ("w", (2,)))
    }
    for name, values in arrays.items():
        graph.add_node(InputOp(), [], Tensor(name, values.shape), node_id=name)
    graph.add_node(tile, list(arrays), outputs=[Tensor(name + "o", values.shape) for name, values in arrays.items()])
    graph.inputs = list(arrays)
    graph.outputs = [name + "o" for name in arrays]
    backend = CudaBackend()
    actual = backend.run(backend.compile(graph), input_data=arrays)[0].outputs
    for name, values in arrays.items():
        np.testing.assert_array_equal(actual[name + "o"], np.maximum(values, 0) if name == "y" else values)


def _swept_reduce(*, per_cell: bool) -> TileOp:
    """One store swept over ``n``, computed from a reduce over ``r``.

    ``per_cell`` decides whether that reduce reads the sweep coordinate. Reading it, the reduce is
    a fact about one output element — the block statistic an NVFP4 encode piece keeps once its
    contractions are cut away. Not reading it, the reduce is the ROW's statistic, evaluated once
    ahead of the sweep and shared by all of it: rms-norm's sum of squares, softmax's maximum.
    """
    r = Axis("r", 4)
    index = (Var("m"), Var("n"), Var("r")) if per_cell else (Var("m"), Var("r"))
    read = slab("sample", "s", *(str(part.name) for part in index))
    stat = reduction(r, (read,), parse_body("stat__v = copy(sample)"), ("stat",))
    root = projection((stat,), parse_body("result = negative(stat)"), ("result",))
    return _tile(root, N16, r, free=(M8,), output_specs=(_spec("out[m, n] = result", N16),))


def test_a_pointwise_sweep_stays_a_loop_for_the_worker_split() -> None:
    """A kernel whose only work IS the sweep keeps it. Nothing folds, so nothing would be
    replicated by binding it — but the kernel materializer already distributes a bare output sweep
    across a worker inventory, and `cases/reduce/rms-norm-cut-sweep-work.json` pins the row that
    does it (885.9 us walking the sweep in one thread, 4.2 us split across 512). Taking the axis
    onto the grid here would decide that for the schedule instead of offering it."""
    body = parse_body(
        """
        v = load x[m, n]
        out_v = negative(v)
        """
    )
    store = _spec("out[m, n] = out_v", N16)
    tile = _tile(projection((), body, ("out_v",)), N16, free=(M8,), output_specs=(store,))

    assert tuple(axis.name for axis in tile.place.free) == ("m",)
    assert tuple(axis.name for axis in tile.output_specs[0].sweep) == ("n",)


def test_a_pointwise_sweep_promotes_when_the_launch_has_no_other_axis() -> None:
    """The same bare sweep with nothing else to launch over. One block runs the whole kernel however
    the workers are split, so the sweep is the only axis the grid could spread across and it takes
    it — the residual-add piece a cut leaves behind in the serving post-attention kernel."""
    body = parse_body(
        """
        v = load x[n]
        out_v = negative(v)
        """
    )
    store = _spec("out[n] = out_v", N16)
    tile = _tile(projection((), body, ("out_v",)), N16, output_specs=(store,))

    assert tuple(axis.name for axis in tile.place.free) == ("n",)
    assert tile.output_specs[0].sweep == ()


def test_a_per_cell_reduce_promotes_the_sweep_it_reads() -> None:
    """A reduce evaluated at the sweep coordinate is per-cell work already, so binding the sweep
    replicates nothing — each cell folds its own."""
    tile = _swept_reduce(per_cell=True)

    assert tuple(axis.name for axis in tile.place.free) == ("m", "n")
    assert tile.output_specs[0].sweep == ()


def test_a_row_statistic_keeps_its_sweep_a_loop() -> None:
    """The protection the contraction clause was written for. A reduce that does NOT read the sweep
    is evaluated once ahead of it and shared by every position, so binding the sweep would recompute
    the whole statistic per output element — softmax's maximum and rms-norm's sum of squares, both
    of which the corpus holds. The sweep stays a loop."""
    tile = _swept_reduce(per_cell=False)

    assert tuple(axis.name for axis in tile.place.free) == ("m",)
    assert tuple(axis.name for axis in tile.output_specs[0].sweep) == ("n",)


def test_nested_contraction_promotes_a_swept_column_beside_an_implicit_unit_row() -> None:
    """A nested linear site turns the swept column into grid placement before scheduling."""
    r = Axis("r", 4)
    stat = reduction(r, (slab("sample", "s", "r"),), parse_body("stat__v = copy(sample)"), ("stat",))
    linear = _matmul(a_index="0, k", b_index="k, n")
    root = projection((stat, linear), parse_body("result = add(stat, acc)"), ("result",))
    tile = _tile(root, N16, r, K32, output_specs=(_spec("out[0, 0, n] = result", N16),))

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert tile.output_specs[0].sweep == ()
    assert any(site.node.as_contraction() is not None for site in sites(tile.op))
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs).op is tile.op


def test_matvec_recovers_an_implicit_unit_row_through_an_output_reshape() -> None:
    """A split head/value boundary is still one varying matrix column coordinate."""
    n, k = Axis("n", 2048), Axis("k", Dim(1024))
    product = _matmul(a_index="k", b_index="k, n", extent=1024)
    tile = _tile(product, k, free=(n,), output_specs=(_spec("out[0, 0, n / 128, n % 128] = acc"),))

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "n")
    assert isinstance(tile.op, Fold) and tile.op.as_contraction() is not None


def test_matvec_keeps_its_unit_row_beside_grouped_columns() -> None:
    group, n, k = Axis("group", 8), Axis("n", 16), Axis("k", Dim(32))
    product = _matmul(a_index="k", b_index="k, group * 16 + n")
    tile = _tile(product, k, free=(group, n), output_specs=(_spec("out[0, group, n] = acc"),))

    assert tuple(axis.name for axis in tile.place.free) == ("_um", "group", "n")
    assert tuple(axis.name for axis in tile.grid_sched._mn_for(tile.op)) == ("_um", "n")
    assert tile.family_sites["TILE"] == (0,)
    rebuilt = TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs)
    assert rebuilt.place == tile.place


@pytest.mark.parametrize("channels", (1, 2))
@pytest.mark.parametrize("projected", (False, True))
@pytest.mark.parametrize("scattered", (False, True))
@pytest.mark.parametrize("mapped", (False, True))
def test_rowless_grouped_columns_gain_a_unit_row(channels, projected, scattered, mapped) -> None:
    group, n = Axis("group", 4), Axis("n", 16)
    product = contraction(
        K32,
        slab("a", "a", "k"),
        *((slab(f"b{i}", f"b{i}", "k", Var("group") * 16 + Var("n")), f"acc{i}") for i in range(channels)),
    )
    values = tuple(f"out{i}" if projected else f"acc{i}" for i in range(channels))
    root = (
        projection((product,), parse_body("\n".join(f"{name} = relu(acc{i})" for i, name in enumerate(values))), values)
        if projected
        else product
    )
    index = "group / 2, group % 2, n * 2" if scattered else "group, n"
    stores = tuple(_spec(f"out{i}[{index}] = {value}") for i, value in enumerate(values))
    tile = TileOp(
        op=root,
        axes=(K32,),
        place=Placement(free=(group, n), grid=(group, n) if mapped else (), mapped=mapped),
        output_specs=stores,
    )

    assert tuple(axis.name for axis in tile.place.free) == ("group", "_um", "n")
    assert tuple(axis.name for axis in tile.place.grid) == (("group", "_um", "n") if mapped else ())
    assert tile.axis_of("_um").extent == Dim(1)
    assert tuple(spec.write.index for spec in tile.output_specs) == tuple(store.write.index for store in stores)
    rebuilt = TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs)
    assert rebuilt.place == tile.place
    assert rebuilt.identity_key(with_io=True, with_knobs=True) == tile.identity_key(with_io=True, with_knobs=True)


@pytest.mark.parametrize("second_has_row", (False, True))
@pytest.mark.parametrize(("mirror_first", "mirror_second"), ((False, False), (True, False), (False, True), (True, True)))
def test_unit_row_requires_every_output_root_to_be_rowless(second_has_row, mirror_first, mirror_second) -> None:
    group, n = Axis("group", 4), Axis("n", 16)
    first = contraction(K32, slab("a", "a", "k"), (slab("b", "b", "k", "group", "n"), "acc"))
    second = contraction(
        K32,
        slab("c", "c", *(("group", "k") if second_has_row else ("k",))),
        (slab("d", "d", "k", "n") if second_has_row else slab("d", "d", "k", "group", "n"), "acc2"),
    )
    if mirror_first or mirror_second:

        def mirror(root):
            left, right = (
                projection((edge,), parse_body(f"v{i} = relu({', '.join(edge.exposes)})"), (f"v{i}",))
                for i, edge in enumerate(root.operands)
            )
            return contraction(K32, right, (left, root.exposes[0]))

        first = mirror(first) if mirror_first else first
        second = mirror(second) if mirror_second else second
    root = projection(
        (first, second),
        parse_body(
            """
            x = relu(acc)
            y = relu(acc2)
            """
        ),
        ("x", "y"),
    )
    tile = _tile(root, K32, free=(group, n), output_specs=tuple(_spec(f"{name}[group, n] = {name}") for name in ("x", "y")))

    expected = ("group", "n") if second_has_row else ("group", "_um", "n")
    assert tuple(axis.name for axis in tile.place.free) == expected
    if not second_has_row:
        for site in tile.sites:
            if (view := site.node.as_contraction()) is not None:
                m, n = tile.grid_sched._mn_for(site.node)
                assert (m.name, n.name) == (("n", "_um") if view.left_axes else ("_um", "n"))


def test_promoted_attention_output_sweep_closes_the_a100_b_seam_idempotently() -> None:
    """The reduced Qwen3 target needs its promoted value-width axis to close computed B."""
    tile = case_target_tile("attention/rmsnorm-gqa-b-cut.json")
    reconstructed = TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs)

    assert tuple(axis.extent for axis in tile.place.free) == (Dim(4), Dim(4), Dim(16))
    assert all(spec.sweep == () for spec in tile.output_specs)
    assert reconstructed.op is tile.op
    # The authored seam — the score's K cone — is offered on the promoted tree.
    assert "PLACE@map.1/twist.1/inner.2/map" in {seam.spelling for seam in cuttable_seams(tile)}


# ---- closure at formation ---------------------------------------------------------------------- #


def _key_swept_score(shared_reader: bool = False) -> TileOp:
    """A reduced attention shape: a key sweep whose body computes the per-key statistic and rsqrt
    feeding the score contraction's computed B operand cone. With ``shared_reader`` the sweep's
    own result also reads the rsqrt, so the chain does not die into the edge."""
    mixed = "\n                    mixed = multiply(acc, inv)" if shared_reader else ""
    return _lift(
        parse_body(
            f"""
            for m in 0..4:
                for t in 0..8:
                    for d in 0..16:
                        ksq = load k[t, d]
                        sq = multiply(ksq, ksq)
                        ss <- add(ss, sq) over d
                    inv = rsqrt(ss)
                    for d in 0..16:
                        qv = load q[m, d]
                        kv = load k[t, d]
                        kn = multiply(kv, inv)
                        prod = multiply(qv, kn)
                        acc <- add(acc, prod) over d{mixed}
                    mx <- maximum(mx, {"mixed" if shared_reader else "acc"}) over t
                out[m] = mx
            """
        )
    )


def _score_and_b(tile: TileOp) -> tuple[Fold, Fold, Fold]:
    sweep = tile.op if tile.op.axis is not None else tile.op.operands[0]
    (score,) = (edge for edge in sweep.operands if edge.as_contraction() is not None)
    return sweep, score, score.operands[1]


def test_key_swept_statistic_closes_the_computed_b_operand() -> None:
    """The chain feeding only the score's B cone rides the edge, closed at its axes: the rsqrt is an
    operand cone of the B cone, which reads the statistic through it."""
    tile = _key_swept_score()

    _, score, b = _score_and_b(tile)
    assert isinstance(b, Fold) and b.axis is None
    assert {tile.axis_of(name).extent for name in b.free_axes} <= {Dim(4), Dim(8), Dim(16)}, b.free_axes  # m, t, d
    assert any(site.node.axis is not None for site in sites(b))
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs).op is tile.op
    # A synthetic TileOp carries no graph output tensors, so the workspace dtype rule cannot
    # resolve a cuttable seam here; the reduced-target test below asserts the offered seam.
    assert id(b) in {id(site.node) for site in family_sites("PLACE", sites(tile.op))}


def test_key_swept_statistic_stays_when_a_sibling_reads_it() -> None:
    """A chain the sweep's own result also reads keeps its place in the sweep's step, and the B
    cone reads it through an operand cone of its own over the SAME statistic fold: a term is
    closed, so the cheap rsqrt is spelled where each reader stands, while the reduce it derives
    from is one shared object and never repeats."""
    tile = _key_swept_score(shared_reader=True)

    sweep, score, b = _score_and_b(tile)
    assert any(isinstance(stmt, Assign) and "rsqrt(" in str(stmt.value) for stmt in sweep.lift.body)
    (inv,) = [edge for edge in b.operands if edge.as_slab() is None]
    assert any(isinstance(stmt, Assign) and "rsqrt(" in str(stmt.value) for stmt in inv.lift.body)
    (stat,) = inv.operands
    assert stat.axis is not None and any(stat is edge for edge in sweep.operands), "the statistic fold is shared, not copied"


def test_normalization_prunes_an_operand_component_no_reader_reads() -> None:
    """A rewrite can leave a cone exposing a value that went dead when the folds around it fused.
    Normalization restricts the edge to what its reader binds and cuts the body to match, so the
    dead half stops riding the interface and the loads defining it stop being carried twice."""
    epilogue = projection(
        (slab("acc", "workspace", "m"),),
        parse_body(
            """
            scale = load sdpa_scale[0]
            v10 = reciprocal(acc)
            """
        ),
        results=("scale", "v10"),
    )
    root = projection((epilogue,), parse_body("out = multiply(v10, v10)"))
    tile = TileOp(op=root, place=Placement(free=(M8,)), axes=(M8,), output_specs=(_spec("o[m] = out"),))
    (edge,) = tile.op.operands
    assert edge.exposes == ("v10",)  # ``scale`` went with the param nothing bound
    assert tile.op.lift.params == ("v10",)
    assert not [stmt for stmt in edge.lift.body if isinstance(stmt, Load) and stmt.input == "sdpa_scale"]


@pytest.mark.parametrize("stored_second", [False, True])
def test_unread_planar_state_drops_its_independent_coordinate(stored_second) -> None:
    """A shared reduction must not give a reader the coordinates of an unused state."""
    r = Axis("r", 4)
    stats = reduction(
        r,
        (slab("xv", "x", "m", "r"), slab("yv", "y", "n", "r")),
        parse_body(
            """
            sx__v = copy(xv)
            sy__v = copy(yv)
            """
        ),
        ("sx", "sy"),
        "maximum",
    )
    root = projection((stats,), parse_body("out = negative(sx)"), ("out",))
    specs = [_spec("o[m] = out")]
    if stored_second:
        specs.append(_spec("tap[n] = sy"))
    tile = _tile(root, N16, r, free=(M8,), output_specs=tuple(specs))
    states = {site.node.exposes: site.node for site in sites(tile.op) if site.node.axis is not None}
    assert set(states) == ({("sx",), ("sy",)} if stored_second else {("sx",)})
    assert states[("sx",)].free_axes == frozenset({"m"})
    if stored_second:
        assert states[("sy",)].free_axes == frozenset({"n"})
    assert all(len(stat.init) == len(stat.base.results) == len(stat.lift.results) == 1 for stat in states.values())
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs).op is tile.op

    import numpy as np

    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp

    x = np.arange(32, dtype=np.float32).reshape(8, 4) - 17
    y = np.arange(64, dtype=np.float32).reshape(16, 4) - 33
    shapes = {"o": (8,), **({"tap": (16,)} if stored_second else {})}
    loop = LoopOp(body=tile.loop_body)
    actual = execute_loop_op_cpp(loop, {"x": x, "y": y}, shapes)
    outputs = dict(zip(loop.outputs, actual if isinstance(actual, tuple) else (actual,), strict=True))
    if stored_second:
        np.testing.assert_array_equal(outputs["tap"], y.max(axis=1))
    np.testing.assert_array_equal(outputs["o"], -x.max(axis=1))


def test_planar_state_pruning_does_not_depend_on_operand_object_sharing() -> None:

    stat = reduction(
        K32,
        (slab("xv", "x", "m", "k"),),
        parse_body(
            """
            sx__v = copy(xv)
            ss__v = multiply(xv, xv)
            """
        ),
        ("sx", "ss"),
    )

    def normalized(second):
        left = projection((stat,), parse_body("left = negative(sx)"), ("left",))
        right = projection((second,), parse_body("right = negative(ss)"), ("right",))
        root = projection((left, right), parse_body("out = add(left, right)"), ("out",))
        return _tile(root, K32, free=(M8,)).op

    assert normalized(stat) == normalized(stat.replace())


def test_readers_of_independent_shared_statistics_keep_only_their_own_coordinates() -> None:
    stat = reduction(
        K32,
        (slab("xv", "x", "m", "k"), slab("yv", "y", "n", "k")),
        parse_body(
            """
            sx__v = copy(xv)
            sy__v = copy(yv)
            """
        ),
        ("sx", "sy"),
        "maximum",
    )
    left = projection((stat,), parse_body("left = negative(sx)"), ("left",))
    right = projection((stat,), parse_body("right = negative(sy)"), ("right",))
    root = projection((left, right), parse_body("out = add(left, right)"), ("out",))
    tile = _tile(root, K32, free=(M8, N16))
    by_result = {site.node.exposes: site.node for site in sites(tile.op)}
    assert by_result[("left",)].free_axes == frozenset({"m"})
    assert by_result[("right",)].free_axes == frozenset({"n"})
    assert tile.op.free_axes == frozenset({"m", "n"})
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes).op is tile.op


def test_normalization_keeps_a_component_only_a_boundary_store_reads() -> None:
    """A store is a reader too: a sweep's per-cell projection reaches its ``Write`` with no lift
    binding it, so the prune keeps what the output specifications name."""
    cell = projection((slab("acc", "workspace", "m"),), parse_body("v = exp(acc)"), results=("acc", "v"))
    root = projection((cell,), parse_body("out = rsqrt(acc)"))
    specs = (_spec("o[m] = out"), _spec("tap[m] = v"))
    tile = TileOp(op=root, place=Placement(free=(M8,)), axes=(M8,), output_specs=specs)
    (edge,) = tile.op.operands
    assert edge.exposes == ("acc", "v")  # ``v`` survives on the store's word alone


def test_normalization_shares_structurally_identical_cones() -> None:
    """The tree-wide invariant: after normalization, no two DISTINCT Fold objects in the tree are
    the same value with the same interface names — copies fusion inlined into several consumption
    sites (attention's softmax statistics, once in the weight cone and once in the epilogue) are
    one object, so placement sees one value and a composed cut materializes it once. Severed
    sharing is the recompute class PR #679 measured at three orders of magnitude."""
    tile = case_target_tile("attention/rmsnorm-qk-sdpa-composed-cut.json")

    by_identity = {id(site.node): site.node for site in sites(tile.op)}
    by_value: dict[tuple, list[Fold]] = {}
    for node in by_identity.values():
        by_value.setdefault((node.canonical(), node.exposes), []).append(node)
    twins = [nodes for nodes in by_value.values() if len(nodes) > 1]
    assert not twins, "normalization left same-value cones as distinct objects"


def _statistic_under_two_binders() -> TileOp:
    """A row statistic feeding a contraction nested under a separate fold binder."""
    r, k, t = Axis("r", Dim(16)), Axis("k", Dim(16)), Axis("t", Dim(4))
    stat = reduction(r, (slab("xr", "x", "m", "r"),), parse_body("ss__v = multiply(xr, xr)"), ("ss",))
    # The statistic and its rsqrt are the B cone's SOURCE — an operand a projection evaluates
    # before its body, so the chain rides its own edge rather than a sibling body position.
    source = projection((stat,), parse_body("inv = rsqrt(ss)"), ("inv",))
    b_cone = projection((slab("wv", "w", "t", "k"), source), parse_body("wn = multiply(wv, inv)"), ("wn",))
    score = contraction(k, slab("qv", "q", "m", "k"), (b_cone, "acc"))
    sweep = reduction(t, (score,), parse_body("mx__v = copy(acc)"), ("mx",), "maximum")
    return _tile(projection((sweep,), results=("mx",)), r, k, t, free=(Axis("m", 4),))


def _loop_scopes(stmts, enclosing: tuple[str, ...] = ()) -> list[tuple[str, tuple[str, ...]]]:
    """Each synthesized loop axis paired with its enclosing loop axes."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for stmt in stmts:
        axis = getattr(stmt, "axis", None)
        inner = (*enclosing, axis.name) if axis is not None else enclosing
        if axis is not None:
            out.append((axis.name, enclosing))
        for body in stmt.nested():
            out.extend(_loop_scopes(list(body), inner))
    return out


def test_a_statistic_is_not_sunk_beneath_new_binders() -> None:
    """Placement must preserve the statistic's evaluation multiplicity: a row statistic is evaluated
    once per row, ahead of every binder that reads it."""
    tile = _statistic_under_two_binders()

    scopes = [enclosing for name, enclosing in _loop_scopes(tile.op.lower(axes=tile.axes)) if name == "r"]
    assert scopes == [()], scopes


def test_a_fold_reading_a_sweep_axis_declares_it_on_the_operand_edge() -> None:
    """A per-column reduce whose streamed load reads the boundary store's sweep axis is an operand
    EDGE like any other term, and it DECLARES that axis.

    Body position used to carry this: a body member is re-wrapped inside the sweep ``Loop`` by
    reconstitution while an edge lowers at kernel scope, so a sweep-reading fold had to stay in the
    body or its column rendered as an undefined identifier (found live: DeepSeek-V4 post16's
    ``k_div_36``). A term cannot be a body member — a Fold tree composes through ``operands`` — so
    the fact is carried explicitly instead: the sweep axis is a free coordinate of the edge (its
    slab declares it), which is what lets the evaluation domain be recovered from the term rather
    than inferred from where it sits."""
    k, j = Axis("k", Dim(4)), Axis("j", Dim(4))
    fold = _reduce_loop(
        """
        for k in 0..4:
            xin = load x[m, k, j]
            acc <- add(acc, xin) over k
        """
    )
    root = projection((fold,), parse_body("v = relu(acc)"), ("v",))

    tile = _tile(root, j, k, free=(M8,), output_specs=(_spec("out[m, j] = v", j),))

    assert not any(isinstance(stmt, Fold) for stmt in tile.op.lift.body)
    (edge,) = tile.op.operands
    assert "j" in edge.free_axes, edge.free_axes


# ---- the bilinear form the lift builds ---------------------------------------------------------- #


def _channels(fold: Fold) -> tuple[Fold, ...]:
    """A contraction's streamed operands, one per channel in order — the edges after A."""
    return fold.operands[1:]


def test_contraction_clusters_alpha_equivalent_shared_operands() -> None:
    planar = _reduce_loop(
        """
        for k in 0..32:
            left0 = load x[m, k]
            right0 = load w0[n, k]
            product0 = multiply(left0, right0)
            left1 = load x[m, k]
            right1 = load w1[n, k]
            product1 = multiply(left1, right1)
            acc0 <- add(acc0, product0) over k
            acc1 <- add(acc1, product1) over k
        """
    )

    tile = _tile(planar, K32, free=(M8, N16))

    contraction = tile.op
    assert contraction.as_contraction() is not None
    assert len(contraction.combine.results) == 2
    assert _input(contraction.operands[0]) == "x"


def test_contraction_coalesces_overlapping_equivalent_shared_operands() -> None:
    planar = _reduce_loop(
        """
        for k in 0..32:
            left = load x[m, k]
            scale = load s[k]
            scaled0 = multiply(left, scale)
            scaled1 = multiply(left, scale)
            right0 = load w0[n, k]
            right1 = load w1[n, k]
            product0 = multiply(scaled0, right0)
            product1 = multiply(scaled1, right1)
            acc0 <- add(acc0, product0) over k
            acc1 <- add(acc1, product1) over k
        """
    )

    tile = _tile(planar, K32, free=(M8, N16))

    assert tile.op.as_contraction() is not None
    assert len(tile.op.combine.results) == 2
    a = tile.op.operands[0]
    assert a.axis is None and a.exposes == ("scaled0",)
    assert sum(isinstance(stmt, Assign) and stmt.name.startswith("scaled") for stmt in a.lift.body) == 1


def test_contraction_orients_a_shared_commutative_argument_first() -> None:
    planar = _reduce_loop(
        """
        for k in 0..32:
            left0 = load x0[m, k]
            left1 = load x1[m, k]
            right = load w[k, n]
            product0 = multiply(left0, right)
            product1 = multiply(left1, right)
            acc0 <- add(acc0, product0) over k
            acc1 <- add(acc1, product1) over k
        """
    )

    tile = _tile(planar, K32, free=(M8, N16))

    view = tile.op.as_contraction()
    assert view is not None and (view.product.name, view.plus.name) == ("multiply", "add")
    assert _input(tile.op.operands[0]) == "w"
    assert [_input(edge) for edge in _channels(tile.op)] == ["x0", "x1"]
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes).op is tile.op


def test_contraction_computes_an_equivalent_channel_once() -> None:
    planar = _reduce_loop(
        """
        for k in 0..32:
            left = load x[m, k]
            right = load w[n, k]
            product0 = multiply(left, right)
            product1 = multiply(left, right)
            acc0 <- add(acc0, product0) over k
            acc1 <- add(acc1, product1) over k
        """
    )

    tile = _tile(planar, K32, free=(M8, N16))

    assert tile.op.as_contraction() is not None and len(tile.op.combine.results) == 2
    assert len(tile.op.operands) == 2, "two channels over one B occupy one operand slot"
    assert sum(isinstance(stmt, Load) and stmt.input == "w" for stmt in tile.op.lower(axes=tile.axes).iter()) == 1


def test_projection_keeps_only_the_maximal_shared_operand() -> None:
    small = projection(body=parse_body("a = load x[m]"), results=("a",))
    large = projection((small,), parse_body("b = copy(a)"), ("a", "b"))

    root = projection((small, large), parse_body("o = copy(b)"))

    assert root.operands == (large,)
    assert root.lift.params == ("a", "b")
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in root.lower(axes=())) == 1


def test_semiring_merges_overlapping_operand_cones_into_one_multi_result_edge() -> None:
    planar = _reduce_loop(
        """
        for k in 0..32:
            left = load x[m, k]
            scale = load s[k]
            scaled = multiply(left, scale)
            right0 = load w0[k, n]
            right1 = load w1[k, n]
            product0 = multiply(left, right0)
            product1 = multiply(scaled, right1)
            acc0 <- add(acc0, product0) over k
            acc1 <- add(acc1, product1) over k
        """
    )

    tile = _tile(planar, K32, free=(M8, N16))

    assert tile.op.as_contraction() is not None
    shared = tile.op.operands[0]
    assert tuple(shared.exposes) == ("left", "scaled"), "the overlapping arguments are ONE multi-result A edge"
    assert sum(isinstance(stmt, Load) and stmt.input == "x" for stmt in tile.op.lower(axes=tile.axes).iter()) == 1


def _computed_matmul(*, computed_a: bool, computed_b: bool) -> TileOp:
    body = ["for k in 0..32:", "    left = load x[m, k]"]
    left = "left"
    if computed_a:
        body += ["    left_scale = load xs[k]", f"    computed_left = multiply({left}, left_scale)"]
        left = "computed_left"
    body.append("    right = load w[k, n]")
    right = "right"
    if computed_b:
        body += ["    right_scale = load ws[k]", f"    computed_right = multiply({right}, right_scale)"]
        right = "computed_right"
    body += [f"    product = multiply({left}, {right})", "    acc <- add(acc, product) over k"]
    return _tile(projection((_reduce_loop("\n".join(body)),)), K32, free=(M8, N16))


def test_contraction_factors_a_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=False)

    assert tile.op.as_contraction() is not None
    a, b = tile.op.operands
    assert a.axis is None and a.as_slab() is None and a.exposes == ("computed_left",)
    assert _input(b) == "w"


def test_contraction_factors_b_computed_operand_cone() -> None:
    tile = _computed_matmul(computed_a=False, computed_b=True)

    assert tile.op.as_contraction() is not None
    a, b = tile.op.operands
    assert _input(a) == "x"
    assert b.axis is None and b.as_slab() is None and b.exposes == ("computed_right",)


def test_contraction_factors_both_computed_operand_cones_idempotently() -> None:
    tile = _computed_matmul(computed_a=True, computed_b=True)

    assert tile.op.as_contraction() is not None
    assert all(edge.as_slab() is None and edge.axis is None for edge in tile.op.operands)
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes).op is tile.op


def test_contraction_preserves_computed_operand_statement_order() -> None:
    """A computed A that reads a statistic through a chain lowers in dependency order: the row-max
    loop, the reciprocal, the value loop that reads it."""
    tile = _lift(
        parse_body(
            """
            for m in 0..8:
                for n in 0..16:
                    for k in 0..64:
                        for j in 0..32:
                            score = load s[m, j]
                            row_max <- maximum(row_max, score) over j
                        inv = reciprocal(row_max)
                        for j in 0..32:
                            v = load v[j, k]
                            weighted = multiply(inv, v)
                            value <- add(value, weighted) over j
                        weight = load w[k, n]
                        product = multiply(value, weight)
                        out <- add(out, product) over k
                    out[m, n] = out
            """
        )
    )

    assert tile.op.as_contraction() is not None
    computed = tile.op.operands[0]
    lowered = computed.lower(axes=tile.axes)
    assert isinstance(lowered[0], Loop) and isinstance(lowered[-1], Loop), [type(stmt).__name__ for stmt in lowered]
    assert any(isinstance(stmt, Assign) and "reciprocal(" in str(stmt.value) for stmt in lowered[1:-1])
    assert TileOp(op=tile.op, place=tile.place, axes=tile.axes, output_specs=tile.output_specs).op is tile.op


def test_share_common_cones_unifies_internally_renamed_copies() -> None:
    """Alpha-equal cones with identical captures and interface names collapse to one object even
    when their internal binder spelling differs — plain structural equality would keep them
    distinct and silently sever the sharing."""

    def cone(load: str, product: str, row: str = "m") -> Fold:
        body = parse_body(
            f"""
            {load} = load x[{row}, k]
            {product} = multiply({load}, {load})
            """
        )
        return Fold(lift=Lambda.closing(("k",), body, (product,)), init=(0.0,), base=Lambda.componentwise(("add",), ("acc",)))

    def consumer(inner: Fold, out: str) -> Fold:
        return projection((inner,), parse_body(f"{out} = exp(acc)"), (out,))

    def rooted(second: Fold) -> Fold:
        root = projection((consumer(cone("l1", "p1"), "u"), consumer(second, "v")), parse_body("w = add(u, v)"), ("w",))
        return _share_common_cones(root)

    unified = rooted(cone("l2", "p2"))
    assert unified.operands[0].operands[0] is unified.operands[1].operands[0]

    distinct = rooted(cone("l2", "p2", row="q"))  # same form, different capture — a different value
    assert distinct.operands[0].operands[0] is not distinct.operands[1].operands[0]


def test_a_carrier_of_two_independent_products_becomes_one_term_per_state() -> None:
    """Two matmuls the fusion put in one nest — each its own A and its own B — fold whole one at a
    time and not together: a tier holds ONE A fragment against a B slab per channel. Normalization
    hands back one term per state, each an ordinary contraction the mma tier tiles, under a
    projection exposing what the carrier exposed."""
    a28 = Axis("a28", Dim(64))
    carrier = reduction(
        a28,
        (slab("wg", "Wg", "n", "a28"), slab("xg", "Xg", "m", "a28"), slab("wu", "Wu", "n", "a28"), slab("xu", "Xu", "m", "a28")),
        parse_body(
            """
            acc_g__v = multiply(wg, xg)
            acc_u__v = multiply(wu, xu)
            """
        ),
        ("acc_g", "acc_u"),
    )
    assert len(carrier.bilinear_channels()) == 1 and not carrier.tiles_whole()

    apart = carrier.per_state()
    assert apart is not None and apart.axis is None and apart.exposes == carrier.exposes
    assert [child.exposes for child in apart.operands] == [("acc_g",), ("acc_u",)]
    assert all(child.tiles_whole() and len(child.operands) == 2 for child in apart.operands)
    assert apart.per_state() is None, "idempotent: what it hands back folds whole"

    tile = TileOp(op=carrier, place=Placement(free=(M8, N16)), axes=(M8, N16, a28))
    assert [site for site in tile.node_sites if tile.contracts(site)] == [1, 2]


def test_a_contraction_beside_a_plain_reduction_keeps_its_own_term() -> None:
    """A non-product channel must not hide a sibling's tensor-core schedule, even when both
    results vary over the same output coordinates."""
    k = Axis("k", Dim(64))
    carrier = reduction(
        k,
        (slab("x", "X", "m", "k"), slab("w", "W", "n", "k"), slab("y", "Y", "m", "k"), slab("v", "V", "n", "k")),
        parse_body(
            """
            sum__v = add(x, w)
            product__v = multiply(y, v)
            """
        ),
        ("sum", "product"),
    )
    apart = carrier.per_state()
    assert apart is not None and apart.exposes == carrier.exposes
    plain, product = apart.operands
    assert plain.as_contraction() is None and product.tiles_whole()
    assert plain.free_axes == product.free_axes
    tile = TileOp(op=carrier, place=Placement(free=(M8, N16)), axes=(M8, N16, k))
    assert [site for site in tile.node_sites if tile.contracts(site)] == [2]


def test_a_carrier_that_folds_whole_or_carries_no_product_stays_one_term() -> None:
    """Two shapes normalization must leave alone: the FUSED gate/up carrier, whose channels share
    their A — the form the atom wants, one ldmatrix'd A fragment and an mma chain per channel — and
    a carrier whose states are no product at all (a sum beside a sum of squares over one loaded
    value), which splitting would make read that value twice for nothing."""
    a28 = Axis("a28", Dim(64))
    shared = contraction(a28, slab("x", "X", "m", "a28"), (slab("wg", "Wg", "n", "a28"), "acc_g"), (slab("wu", "Wu", "n", "a28"), "acc_u"))
    assert shared.tiles_whole() and shared.per_state() is None

    statistics = reduction(
        a28,
        (slab("x", "X", "m", "a28"),),
        parse_body(
            """
            sum__v = multiply(x, x)
            count__v = add(x, x)
            """
        ),
        ("sum", "count"),
    )
    assert len(statistics.bilinear_channels()) < 2 and statistics.per_state() is None


def test_total_lift_produces_canonical_contraction() -> None:
    tile = _lift(
        parse_body(
            """
            for m in 0..8:
                for n in 0..16:
                    for k in 0..32:
                        left = load x[m, k]
                        right = load w[n, k]
                        product = multiply(left, right)
                        acc <- add(acc, product) over k
                    out[m, n] = acc
            """
        )
    )

    assert tile.op.as_contraction() is not None
    assert tile.op.axis is not None


def _fed_chain_root(feed: bool) -> Fold:
    """A scalar chain beside a reduce whose step reads the chain's result when ``feed`` (the composed
    placement cut's consumer shape), lifted: the chain becomes the reduce's PROVIDER operand."""
    scale_arg = "v25" if feed else "left"
    return _lift(
        parse_body(
            f"""
            ws = load cutbuf[0]
            v25 = rsqrt(ws)
            for k in 0..8:
                left = load x[k]
                scaled = multiply(left, {scale_arg})
                acc <- add(acc, scaled) over k
            out[0] = acc
            """
        )
    )


def test_a_body_fed_fold_takes_its_provider_as_a_sibling_operand() -> None:
    """A fold whose subtree reads a name the scalar chain defines becomes an operand edge, and the
    chain becomes a PROVIDER operand ordered ahead of it.

    Body membership used to carry this: a projection evaluates its operands before its scalar body,
    so hoisting a fed fold emitted its capture as an undefined identifier (DeepSeek-V4 post4096's
    two-cut consumer piece). A term cannot be a body member, so the ordering is carried structurally
    instead — the lift closes the reduce over the chain as an operand cone, and ``Fold.lower``
    places every operand ahead of the term that reads it."""
    tile = _fed_chain_root(feed=True)
    out = tile.op

    assert not any(isinstance(stmt, Fold) for stmt in out.lift.body), "a term is never a body member"
    lowered = out.lower(axes=tile.axes)
    assert any(isinstance(stmt, Loop) for stmt in lowered), [type(stmt).__name__ for stmt in lowered]
    kinds = ["provider" if isinstance(stmt, Assign) and "rsqrt(" in str(stmt.value) else type(stmt).__name__ for stmt in lowered]
    assert kinds.index("provider") < kinds.index("Loop"), "the provider is emitted before its consumer"


def test_an_unfed_fold_is_an_operand_edge_too() -> None:
    """The same shape with the reduce reading only its own operand: also an edge. Position no longer
    varies with the capture, because position no longer encodes anything."""
    out = _fed_chain_root(feed=False).op

    assert out.axis is not None or any(edge.axis is not None for edge in out.operands)
    assert not any(isinstance(stmt, Fold) for stmt in out.lift.body)


def test_a_name_two_stores_ride_at_different_extents_is_not_one_axis() -> None:
    """Loop IR scopes its axes lexically, so two sibling nests may spell one name at two extents,
    and ``LoopOp.axes`` deduplicates them on the name. Promoting such a name sizes the grid for
    whichever extent the dedup kept, and every store riding the other is written over the wrong
    cells: a Gated DeltaNet chunk kernel this compiler minted returned half of one output as zeros
    that way, faster than eager and with nothing to say it was wrong. Declining leaves the kernel
    sweeping in one block — slow, and correct, which is what this refusal is for."""
    from emmy.compiler.ir.tile.ir import promoted_sweep

    op = projection((_matmul(),))
    wide = _spec("wide[m, n] = acc", N16)
    narrow = _spec("narrow[m, n] = acc", Axis("n", 8))

    assert promoted_sweep(op, (wide, wide)) == {"n"}, "one axis two stores ride at one extent still promotes"
    assert promoted_sweep(op, (wide, narrow)) == set()
    assert promoted_sweep(op, (narrow, wide)) == set(), "and the refusal does not depend on which store is first"


def test_a_pointwise_sweep_under_a_unit_row_promotes() -> None:
    """A decode row is a static unit free axis: it launches one block like no free axis at all, so a
    pointwise term's shared sweep is the only axis the launch can spread over. Kept a sweep, the
    Gemma 4 width-1 q-norm piece walked 16 heads in 16 threads and 512 dims serially (20 us for 1)."""
    from emmy.compiler.ir.tile.ir import promoted_sweep

    op = projection((slab("x", "x", "h", "d"),))
    store = _spec("out[h, d] = x", Axis("h", 16), Axis("d", 512))

    assert promoted_sweep(op, (store,), free=(Axis("_row", 1),)) == {"h", "d"}
    assert promoted_sweep(op, (store,), free=(Axis("m", 4),)) == set(), "a real free axis still leaves the sweep to the schedule"


def test_root_collapses_onto_the_operand_the_stores_read() -> None:
    """The Gated DeltaNet chunk shape: the boundary reads its value off an operand and the root's own
    result is a statistic nothing keeps. Left standing, the root holds an invariant reduce alive, and
    ``promoted_sweep`` will not promote a store sweep past a reduce that does not read it — so the
    kernel keeps a one-axis placement and walks every cell of the sweep in one block."""
    carried = reduction(K32, (slab("near", "x", "m", "n", "k"),), parse_body("carried__v = multiply(near, near)"), ("carried",))
    invariant = reduction(K32, (slab("far", "y", Literal(0, "int"), Var("k")),), parse_body("stat__v = multiply(far, far)"), ("stat",))
    root = projection((invariant, carried), parse_body("dead = multiply(stat, stat)"), ("dead",))
    tile = _tile(root, N16, K32, free=(M8,), output_specs=(_spec("out[m, n] = carried", N16),))

    assert tile.op == carried  # the root and the operand only its dead lift bound are gone
    assert all(_input(edge) != "y" for edge in tile.op.operands)
    assert [axis.name for axis in tile.place.free] == ["m", "n"]  # the store sweep promotes
