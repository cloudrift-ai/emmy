"""``083_share_rescale_exponential``: an online softmax's two rescale factors from one exponential, bit for bit."""

from importlib import import_module
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Literal
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import WarpShuffle
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.pure.twist import SOFTMAX, Twist
from emmy.compiler.ir.stmt import Body, parse_body
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.pipeline import CUDA_PASSES, FINAL_LOWERING_PASS, Pipeline, RuleSkipped
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import recording_kernels, requires_cuda, statements
from tests.compiler.realization.helpers import CASES_DIR, load_case, pin_of, seeded_inputs
from tests.compiler.terms import projection, slab

rule = import_module("emmy.compiler.pipeline.passes.lowering.kernel.083_share_rescale_exponential")


def _rewrite(op: KernelOp) -> KernelOp:
    """The rewrite under the accurate exponential, where it runs."""
    with pinned_knobs({"FAST_EXP": False}):
        return rule.rewrite(SimpleNamespace(op=op))


#: Every kind of pivot: finite values, signed zeros, a subnormal, the ``maximum`` seed ``∓1e30``, the largest finite
#: values (whose distance overflows), infinities and NaN.
_PIVOTS = np.float32([0.0, -0.0, 1.5, -2.25, 1e-40, -1e30, 1e30, 3.4e38, -3.4e38, np.inf, -np.inf, np.nan])

_ADVANCE = """
    f32 gn = maximum(g, g_o)
    f32 dg = g - gn
    f32 alpha = exp(dg)
    f32 dg_o = g_o - gn
    f32 beta = exp(dg_o)
"""


def _exps(stmts) -> int:
    return sum(node.operation.name == "exp" for stmt in stmts for e in stmt.expressions() for node in e.subterms() if node.operation)


def _same_bits(got: np.ndarray, want: np.ndarray) -> bool:
    """Equal as floats, a NaN matching a NaN and a zero matching a zero of the same sign."""
    nan = np.isnan(want)
    return bool(np.array_equal(np.isnan(got), nan) and np.array_equal(got[~nan].view(np.uint32), want[~nan].view(np.uint32)))


# --------------------------------------------------------------------------- #
# The rewrite
# --------------------------------------------------------------------------- #


def _kernel(advance: str = _ADVANCE) -> KernelOp:
    """A kernel that reads the two pivots and stores the two factors: the body is in normal form, which holds only
    what something reads."""
    return KernelOp(body=parse_body(f"    f32 g = load G[0]\n    f32 g_o = load H[0]{advance}    O[0] = alpha\n    O[1] = beta"))


def _shared(advance: str) -> list:
    """The statements of ``advance`` through the pass's scope rewrite, in the order they are written."""
    return rule._share(list(parse_body(advance)), typed=True)


def test_the_pair_takes_one_exponential_and_the_rewrite_is_idempotent() -> None:
    op = _kernel()
    shared = _rewrite(op)
    assert _exps(op.body) == 2 and _exps(shared.body) == 1
    with pytest.raises(RuleSkipped, match="no rescale factor pair"):
        _rewrite(shared)


def test_a_difference_is_read_in_either_spelling() -> None:
    """The normal form writes ``g - gn`` as ``-gn + g`` where the negated term sorts first."""
    spelled = _ADVANCE.replace("f32 dg = g - gn", "f32 dg = add(negative(gn), g)")
    assert _exps(_shared(spelled)) == 1


def test_the_fast_exponential_keeps_both() -> None:
    """One ``__expf`` is fewer instructions than the selects that would share it."""
    body = parse_body(f"{_ADVANCE}\n    f32 out = alpha * beta")
    with pinned_knobs({"FAST_EXP": True}), pytest.raises(RuleSkipped, match="fast exponential"):
        rule.rewrite(SimpleNamespace(op=KernelOp(body=body)))


@pytest.mark.parametrize(("dtype", "shared"), [(F32, True), (F16, False)])
def test_a_cooperative_merge_held_as_a_field_takes_one_exponential_at_f32(dtype, shared) -> None:
    """The butterfly's merge is a statement field of the combine, rendered at the combine's dtype."""
    merge = parse_body(_ADVANCE.replace("f32 ", "").replace("g_o", "g__o"))
    combine = WarpShuffle(state=("g",), state_b=("g__o",), combine_states=tuple(merge), length=32, dtype=dtype)
    op = KernelOp(body=Body((combine,)))
    if not shared:
        with pytest.raises(RuleSkipped):
            _rewrite(op)
        return
    (combine,) = _rewrite(op).body
    assert _exps(combine.combine_states) == 1


def test_a_pair_whose_earlier_factor_is_read_before_the_later_one_is_left_alone() -> None:
    advance = _ADVANCE.replace("    f32 dg_o = g_o - gn", "    f32 early = alpha * 2.0\n    f32 dg_o = g_o - gn")
    assert _shared(advance) == list(parse_body(advance))


def test_two_distances_below_another_pivot_are_no_pair() -> None:
    """Only distances below the maximum of the two sides they read are a pair: one of them is then ``0``."""
    advance = _ADVANCE.replace("gn = maximum(g, g_o)", "gn = maximum(g, h)")
    assert _shared(advance) == list(parse_body(advance))


#: The order the normal form can give the pair: the later factor read before the earlier one's exponential, both
#: distances ahead of it, so the block takes the earlier factor's place.
_READ_FIRST = """
    f32 gn = maximum(g, g_o)
    f32 dg = g - gn
    f32 dg_o = g_o - gn
    f32 beta = exp(dg_o)
    f32 early = beta * 2.0
    f32 alpha = exp(dg)
"""


@pytest.mark.parametrize("advance", [_ADVANCE, _READ_FIRST], ids=["recipe-order", "read-first"])
def test_the_factors_are_the_two_exponentials_bits_on_the_host(advance) -> None:
    """Every pair of pivots, the host's exponential and its NaN-propagating maximum: the same bits as two
    exponentials."""
    g, g_o = (np.float32(column) for column in zip(*product(_PIVOTS, repeat=2), strict=True))
    before = parse_body(advance)
    after = _shared(advance)
    assert _exps(after) == 1
    results = []
    for body in (before, after):
        env = {"g": g, "g_o": g_o}
        with np.errstate(all="ignore"):
            for stmt in body:
                env[stmt.name] = np.float32(stmt.value.eval(env))
        results.append((env["alpha"], env["beta"]))
    for got, want in zip(results[1], results[0], strict=True):
        assert _same_bits(got, want)


# --------------------------------------------------------------------------- #
# On the device, against the two-exponential kernel
# --------------------------------------------------------------------------- #


def _compile(graph: Graph, pins: dict, *, cap: tuple[int, int], shared: bool, passes: list[str]) -> Graph:
    """``graph`` through ``passes`` under ``pins``, with or without the rewrite."""
    pipeline = Pipeline.build(passes)
    if not shared:
        for pass_ in pipeline.passes:
            pass_.rules = [r for r in pass_.rules if r.name != "083_share_rescale_exponential"]
    with pinned_knobs(pins):
        return pipeline.run(graph.copy(), ctx=Context.from_target(cap), db=None)


def _exponentials(kernels) -> int:
    """How many exponentials ``kernels`` compute: each call in a value, and each fragment statement that is one."""

    def name(node) -> str:
        return getattr(node, "fn", None) or getattr(getattr(node, "operation", None), "name", "")

    count = 0
    for stmt in statements(kernels):
        count += sum(name(node).startswith("exp") for e in stmt.expressions() for node in e.subterms())
        count += getattr(getattr(stmt, "op", None), "name", "").startswith("exp")
    return count


def _both(graph: Graph, pins: dict, feeds: list[dict], *, cap: tuple[int, int], passes: list[str] = CUDA_PASSES) -> None:
    """Run ``graph`` with and without the rewrite on every feed: the same bits out."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    with recording_kernels() as kernels_of:
        compiled = {shared: _compile(graph, pins, cap=cap, shared=shared, passes=passes) for shared in (True, False)}
    exps = {shared: _exponentials(kernels_of(g)) for shared, g in compiled.items()}
    assert exps[True] < exps[False], f"the rewrite took no factor pair: {exps}"
    backend = CudaBackend()
    for feed in feeds:
        out = {shared: backend.run(g, input_data=dict(feed))[0].outputs for shared, g in compiled.items()}
        for name, want in out[False].items():
            assert _same_bits(np.asarray(out[True][name], np.float32), np.asarray(want, np.float32)), name


@requires_cuda
@pytest.mark.parametrize(
    "pins",
    [
        {"FAST_MATH": False},
        {"FAST_MATH": True, "FAST_EXP": False},
        {"FAST_MATH": True, "FAST_EXP": False, "FLUSH_TO_ZERO": False},
    ],
    ids=["standard", "fast-math-flush", "fast-math"],
)
def test_the_factors_are_the_two_exponentials_bits_on_the_device(pins) -> None:
    """The factor pair itself, for every pair of pivots, under the accurate exponential (where the rewrite runs) in
    each flush mode."""
    g, g_o = (np.float32(column) for column in zip(*product(_PIVOTS, repeat=2), strict=True))
    n = len(g)
    body = parse_body(
        f"Tile(axes=(a0 in 0..{n})):\n    f32 g = load G[a0]\n    f32 g_o = load GO[a0]\n"
        + "".join(f"    {line.strip()}\n" for line in _ADVANCE.strip().splitlines())
        + "    f32 F[0, a0] = alpha\n    f32 F[1, a0] = beta"
    )
    io = {name: Tensor(name, (n,), F32) for name in ("G", "GO")}
    graph = Graph()
    for name, tensor in io.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    factors = Tensor("F", (2, n), F32)
    graph.add_node(KernelOp(body=body, name="k_advance", inputs=io, outputs={"F": factors}), list(io), factors, node_id="F")
    graph.inputs, graph.outputs = list(io), ["F"]
    _both(graph, pins, [{"G": g, "GO": g_o}], cap=(12, 0), passes=["lowering/kernel", FINAL_LOWERING_PASS])


def _partition_merge(count: int) -> Graph:
    """Finished softmax states ``(m, d, o)`` of ``count`` partitions merged per column, as a split's finalize does."""
    n = Axis("n", 8)
    k = Axis("k", count, window=Window(parent=Axis("split", count), partition=True))
    values = ("pm", "pd", "po")
    fold = Fold(
        operands=tuple(slab(value, "x", Literal(i, "int"), "k", "n") for i, value in enumerate(values)),
        lift=Lambda(("k", *values), parse_body(""), values),
        init=(ElementwiseImpl("maximum").identity, 0.0, 0.0),
        base=Lambda.componentwise(SOFTMAX.base, ("m", "d", "o")),
        twist=Twist(SOFTMAX, ()),
    )
    tile = TileOp(
        op=projection((fold,), parse_body("out = divide(o, d)"), ("out",)),
        place=Placement(free=(n,)),
        axes=(n, k),
        inputs={"x": Tensor("x", (3, count, 8), F32)},
        outputs={"y": Tensor("y", (8,), F32)},
        output_specs=(OutputSpec(parse_body("y[n] = out")[0]),),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], tile.inputs["x"], node_id="x")
    graph.add_node(tile, ["x"], tile.outputs["y"], node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    return graph


@requires_cuda
@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize(("reduce", "work"), [("", ""), ("r4", ""), ("coop-t", "t128")], ids=["serial", "register", "coop"])
def test_a_partition_merge_keeps_every_bit(fast_math, reduce, work) -> None:
    """Each partition's pivot drawn from every kind of value, so the merge meets each pair of them."""
    count = 16
    rng = np.random.default_rng(1)
    feeds = []
    for _ in range(8):
        states = np.abs(rng.normal(size=(3, count, 8))).astype(np.float32) + 0.5
        states[0] = rng.choice(_PIVOTS, size=(count, 8))
        feeds.append({"x": states})
    pins = {"WORK": work, "REDUCE": reduce, "RASTER": "", "PLACE": "fuse", "LOOPIFY": "0", "FAST_MATH": fast_math, "FAST_EXP": False}
    _both(_partition_merge(count), pins, feeds, cap=(8, 0))


@requires_cuda
@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize(
    "case",
    ["combine-softmax-serial", "combine-softmax-coop-warp", "combine-softmax-coop-hier", "combine-softmax-ilp", "attention-coop-warp"],
)
def test_a_streamed_softmax_keeps_every_bit(fast_math, case) -> None:
    """The per-element fold and the cooperative merges, over rows that hold every kind of value."""
    case = load_case(CASES_DIR / f"reduce/{case}.json")
    program = case.program()
    rng = np.random.default_rng(2)
    feeds = []
    for _ in range(3):
        feed = seeded_inputs(program)
        for name in program.inputs:
            values = rng.normal(size=feed[name].shape).astype(np.float32) * 4
            mask = rng.random(values.shape) < 0.05
            values[mask] = rng.choice(_PIVOTS, size=int(mask.sum()))
            feed[name] = values.astype(program.buffer(name).dtype.np)
        feeds.append(feed)
    _both(program, {**pin_of(case, case.row), "FAST_MATH": fast_math, "FAST_EXP": False}, feeds, cap=case.compute_cap)
