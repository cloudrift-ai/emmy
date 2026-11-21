"""A compute-filled B slab stages the value ITS channel multiplies, not the producer's last result.

Two contraction channels over one A can share one producer edge that exposes two values — the gate and
up halves of a packed weight, dequantized by one lift (the DeepSeek V4 expert cut piece after #829
clustered the two dequant cones into one operand). The staged fill took ``edge.exposes[-1]`` for every
channel, so both B slabs held the up half and the piece computed the up projection twice; the width-16
expert twin then failed its random-input check by 70% of the output's peak and single-token decode
served noise.

The shared operand has no such answer: its one slab holds one value. Two products of one computed matrix under two
scales expose two values of it, and such a node is offered no stage and no tensor-core tile."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.schedule import Tile, Work
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _sync_operands
from emmy.compiler.pipeline.passes.lowering.kernel._stage import CtaTile
from tests.compiler.helpers import compile_with_kernels, requires_cuda
from tests.compiler.terms import projection, reduction


def _two_channel_shared_producer():
    """``acc0 = Σ_k x·g``, ``acc1 = Σ_k x·u`` with ``g`` and ``u`` the two results of ONE producer edge
    that reads two row bands of the same weight buffer."""
    ka = Axis("k", Dim(2048))
    [a] = parse_body("f16 in_x = load x[m, k]")
    b = projection(
        body=parse_body(
            """
            f16 in_g = load w[n + 2048, k]
            f16 in_u = load w[n, k]
            g = negative(in_g)
            u = negative(in_u)
            """
        ),
        results=("g", "u"),
    )
    products = parse_body("acc0__v = multiply(in_x, g)\nacc1__v = multiply(in_x, u)")
    return reduction(ka, (a, b), products, ("acc0", "acc1"), "add"), b, ka


def test_each_channel_stages_the_result_it_multiplies():
    c, b, ka = _two_channel_shared_producer()
    assert [index for index, _ in c.bilinear_channels()] == [0, 1], "two product channels over one A"
    tile = Tile.parse("mma_m8n8k4_f16_f32/f1x1/k8", Work.parse("w1x1"))
    mn = tile.at(Axis("m", Dim(16)), Axis("n", Dim(2048))).mn
    cta = CtaTile(linear_tid=Var("_t"), n_threads=32)
    channels = ((b, "acc0"), (b, "acc1"))
    _, sync_ops, *_ = _sync_operands(c, 32, mn, cta, k_axis=ka, channels=channels)
    fills = {op.tag: op for op in sync_ops if op.tag.startswith("b")}
    assert set(fills) == {"b", "b_x1"}
    staged = {}
    for tag, op in fills.items():
        _stmts, name = op.value(Literal(0, "int"), Var("_row"), Var("_col"))
        staged[tag] = name
    assert staged["b"] != staged["b_x1"], f"both B slabs stage {staged['b']!r}; each channel must stage its own result"
    assert {staged["b"], staged["b_x1"]} == {"g", "u"}


def _two_scaled_products(second: str, m: int = 32, n: int = 32, k: int = 64):
    """``out = (exp(a) * s) @ b`` and ``other = (exp(a) * second) @ c``: one computed matrix, scaled along the
    summed axis by ``s`` for one product and by ``second`` (``t``, or ``s`` again) for the other."""
    from emmy.compiler.dtype import F32
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.loop import LoopOp

    body = parse_body(
        f"""
for a0 in 0..{m}:
  for a1 in 0..{n}:
    for a2 in 0..{k}:
      in0 = load a[a0, a2]
      in1 = load s[a2]
      in2 = load {second}[a2]
      in3 = load b[a2, a1]
      in4 = load c[a2, a1]
      w = exp(in0)
      v0 = w * in1
      v1 = w * in2
      v2 = v0 * in3
      v3 = v1 * in4
      acc0 <- add(acc0, v2) over a2
      acc1 <- add(acc1, v3) over a2
    out[a0, a1] = acc0
    other[a0, a1] = acc1
"""
    )
    graph = Graph()
    shapes = {"a": (m, k), "s": (k,), "b": (k, n), "c": (k, n), **({"t": (k,)} if second == "t" else {})}
    for name, shape in shapes.items():
        graph.add_node(InputOp(), [], Tensor(name, shape, F32), node_id=name)
    graph.add_node(LoopOp(body=body), list(shapes), outputs=(Tensor("out", (m, n), F32), Tensor("other", (m, n), F32)), node_id="out")
    graph.inputs, graph.outputs = list(shapes), ["out", "other"]
    return graph


def test_channels_that_multiply_two_values_of_the_shared_operand_take_no_stage():
    from emmy.compiler.context import Context
    from emmy.compiler.ir.schedule.classic import refusals
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.ir.tile.path import sites
    from emmy.compiler.pipeline import Pipeline

    offered = {}
    for second in ("t", "s"):
        lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_two_scaled_products(second))
        (tile,) = (node.op for node in lifted.nodes.values() if isinstance(node.op, TileOp))
        (product,) = (site.node for site in sites(tile.op) if site.node.axis is not None)
        assert len(product.bilinear_channels()) == 2, "two products in one fold"
        offered[second] = (len(product.operands[0].exposes), bool(refusals._warp_atoms(tile, Context.from_target((12, 0)), product)))
    # Under one scale the shared operand is one value and one slab, and the pair is staged as any pair is.
    assert offered == {"t": (2, False), "s": (1, True)}


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_products_of_one_matrix_under_two_scales_match_numpy_on_the_gpu():
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 32, 32, 64
    with pinned_knobs({"FAST_MATH": False}):
        compiled, _ = compile_with_kernels(_two_scaled_products("t", m, n, k))
    rng = np.random.default_rng(0)
    feed = {
        name: rng.standard_normal(shape).astype(np.float32) * 0.5
        for name, shape in {"a": (m, k), "s": k, "b": (k, n), "c": (k, n), "t": k}.items()
    }
    result, _ = CudaBackend().run(compiled, input_data=dict(feed))
    wide = {name: value.astype(np.float64) for name, value in feed.items()}
    for name, scale, right in (("out", "s", "b"), ("other", "t", "c")):
        exact = (np.exp(wide["a"]) * wide[scale]) @ wide[right]
        np.testing.assert_allclose(result.outputs[name].reshape(m, n), exact, rtol=1e-4, atol=1e-4)
