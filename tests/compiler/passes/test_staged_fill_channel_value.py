"""A compute-filled B slab stages the value ITS channel multiplies, not the producer's last result.

Two contraction channels over one A can share one producer edge that exposes two values — the gate and
up halves of a packed weight, dequantized by one lift (the DeepSeek V4 expert cut piece after #829
clustered the two dequant cones into one operand). The staged fill took ``edge.exposes[-1]`` for every
channel, so both B slabs held the up half and the piece computed the up projection twice; the width-16
expert twin then failed its random-input check by 70% of the output's peak and single-token decode
served noise."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.schedule import Tile, Work
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _sync_operands
from emmy.compiler.pipeline.passes.lowering.kernel._stage import CtaTile
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
