"""The strategy system: discovery, engine events, and the discovered strategy.

The engine is IR-agnostic — it emits events (``RunStartEvent`` / ``SpliceEvent`` /
``SplicedEvent`` / ``PassEndEvent``) and every cross-cutting concern is a strategy class
discovered from the ``passes/`` top level (``pipeline.strategy.discovered_strategies``). These
tests pin the discovery contract, the event dispatch, and op provenance's observable behavior
(mint at decomposition, aggregate after, absent without the strategy). A kernel's facts — its
exact identity and its ``S_*`` stamps — are no strategy's: nothing writes them onto an op, and
the last section pins that they are computed from the kernel wherever they are read.
"""

from __future__ import annotations

from emmy.compiler import provenance
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.provenance import ProvenanceStrategy
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.features import stamps, structure_features
from emmy.compiler.pipeline.strategy import PipelineStrategy, discovered_strategies
from emmy.compiler.wire import formed_from, kernel_identity, kernel_tile

_CTX = Context.from_target((12, 0))


def _matmul(m: int = 64, k: int = 64, n: int = 64) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=F16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _norm_linear(m: int = 2, h: int = 16) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (Dim(1), Dim(m), Dim(h)), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("wn", (Dim(h),), dtype=F16), node_id="wn")
    g.add_node(InputOp(), [], Tensor("w", (Dim(h), Dim(h)), dtype=F16), node_id="w")
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (Dim(1), Dim(m), Dim(h)), dtype=F16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("y", (Dim(1), Dim(m), Dim(h)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def _resolve(passes, graph):
    """Option-0 resolution over a freshly built pipeline (strategies discovered)."""
    return Run(pipeline=Pipeline.build(passes), ctx=_CTX).resolve(graph, lambda fp: next(fp.leaves()))


# --- discovery --------------------------------------------------------------------------------


def test_discovery_finds_the_strategy_once() -> None:
    """Every ``PipelineStrategy`` subclass defined in a ``passes/`` top-level module is discovered,
    instantiated once (shared instances), in deterministic class-name order."""
    found = discovered_strategies()
    assert [type(s).__name__ for s in found] == ["ProvenanceStrategy"]
    assert found is discovered_strategies(), "instances are cached and shared"
    assert all(isinstance(s, PipelineStrategy) for s in found)


def test_built_pipelines_carry_the_discovered_strategies() -> None:
    assert Pipeline.build(LOOP_PASSES).strategies == discovered_strategies()
    assert Pipeline.from_pattern([]).strategies == (), "test shims carry no strategies"


# --- provenance -------------------------------------------------------------------------------


def test_decomposition_mints_fresh_pieces_and_fusion_aggregates() -> None:
    """The matmul's decomposition pieces each become a distinct piece of the 'o' origin
    (mint); fusion then aggregates them back so the fused kernel covers the origin."""
    out, _ = _resolve(LOOP_PASSES, _matmul())
    kernels = [(nid, n) for nid, n in out.nodes.items() if isinstance(n.op, LoopOp)]
    assert kernels, "the matmul must fuse into at least one loop kernel"
    totals = provenance.totals(out)
    assert "o" in totals, "the original op is an origin"
    covered = set()
    for _nid, node in kernels:
        prov = provenance.get(node)
        assert prov, "every kernel carries provenance"
        covered |= set(prov.get("o", {}).get("pieces", []))
    assert covered == totals["o"], "the kernels together cover every piece of the origin"


def test_one_origin_rewrites_keep_the_frontend_source_object() -> None:
    """Decomposition and lifting fragments retain one ultimate object for private-edge checks."""
    graph = _matmul()
    origin = graph.nodes["o"].op
    out, _ = _resolve(["frontend/decomposition", "frontend/optimization", "loop/lifting"], graph)
    loops = [node.op for node in out.nodes.values() if isinstance(node.op, LoopOp)]
    assert loops
    assert all(list(op.source_chain())[-1] is origin for op in loops)


def test_mixed_origin_rewrite_keeps_the_result_frontend_source_object() -> None:
    """A fused result retains its consumer origin while its input keeps a distinct origin."""
    graph = _norm_linear()
    origin = graph.nodes["y"].op
    input_origins = {name: graph.nodes[name].op for name in graph.inputs}
    out, _ = _resolve(LOOP_PASSES, graph)
    result = out.nodes["y"].op
    assert list(result.source_chain())[-1] is origin
    assert all(list(out.nodes[name].op.source_chain())[-1] is input_origin for name, input_origin in input_origins.items())


def test_a_pipeline_without_the_provenance_strategy_has_no_provenance() -> None:
    """PipelineStrategy-scoped concern: strip ProvenanceStrategy from the pipeline and NO node carries
    provenance — the graph and engine hold none of it."""
    pipeline = Pipeline.build(LOOP_PASSES)
    stripped = Pipeline(
        passes=pipeline.passes,
        strategies=tuple(s for s in pipeline.strategies if not isinstance(s, ProvenanceStrategy)),
    )
    out, _ = Run(pipeline=stripped, ctx=_CTX).resolve(_matmul(), lambda fp: next(fp.leaves()))
    assert all(not provenance.get(n) for n in out.nodes.values()), "no strategy → no provenance anywhere"


# --- kernel facts: computed, never stored -------------------------------------------------------


def test_an_ops_knobs_hold_decisions_only() -> None:
    """THE invariant's guard on the op side: a source of truth holds inputs only. An op's knobs are the decisions
    taken on it, every one a registered knob — never a fact computed from its body (an ``S_*`` stamp, its
    identity) or from the card (``H_*``), which would be a second copy to keep in agreement with the computation.
    Checked over every op on every rewrite chain of fully lowered programs: fused, twisted, cut and split."""
    from emmy.compiler.pipeline.knob import family_of, get
    from tests.compiler.realization import helpers as corpus

    cases = (
        "fused/norm-linear-f16-scalar-reduce.json",
        "attention/sdpa-hd128-softmax-v-mma.json",
        "fused/linear-add-place-cut-sm70.json",
        "reduce/cross-cta-sum-kernel.json",
    )
    for name in cases:
        case = corpus.load_case(corpus.CASES_DIR / name)
        graph, _taken = corpus.lowered(case, case.context())
        keys = {key for node in graph.nodes.values() for op in node.op.source_chain() for key in op.knobs}
        assert keys, name
        assert not [key for key in keys if get(family_of(key)) is None], f"{name}: an op carries a knob no fork decides"


def test_a_kernels_stamps_are_computed_from_its_own_body() -> None:
    """A loop kernel's ``S_*`` row is the features of its body, with its dtypes read off its own io — computed
    where it is read, the same on every read."""
    out, _ = _resolve(LOOP_PASSES, _matmul())
    loops = [node.op.with_io(out, node) for node in out.nodes.values() if isinstance(node.op, LoopOp)]
    assert loops
    for op in loops:
        assert stamps(op) == structure_features(op.body, {**op.outputs, **op.inputs}) and stamps(op)["S_dtype_f16"] == 2.0
        assert kernel_tile(op) is None and kernel_identity(op) is None, "a kernel is named by its tile, and none stands behind it yet"


def test_the_kernel_is_the_tile_its_schedule_fork_was_offered() -> None:
    """One rule names the kernel of any op on a rewrite chain: the nearest unscheduled tile. An online softmax is a
    twisted kernel, so its tile is not its loop op's body — the identity is the tile's — and its facts read the same
    off the scheduled tile, off the tile itself and off every op lowered from it."""
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.ir.tile import TileOp
    from tests.compiler.realization import helpers as corpus

    case = corpus.load_case(corpus.CASES_DIR / "reduce/online-softmax-4x128.json")
    graph, _taken = corpus.lowered(case, case.context())
    [cuda] = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    tile = kernel_tile(cuda)
    scheduled = next(op for op in cuda.source_chain() if isinstance(op, TileOp))
    assert tile.schedule is None and scheduled.schedule is not None and kernel_tile(scheduled) is tile
    loop = formed_from(tile)
    assert kernel_identity(cuda) == tile.identity_key(structural=False, with_io=True) != loop.identity_key(structural=False, with_io=True)
    assert stamps(cuda) == stamps(scheduled) == stamps(tile) == structure_features(tile.loop_body, {**tile.outputs, **tile.inputs})
    assert stamps(tile) != structure_features(loop.body, {**tile.outputs, **tile.inputs}), "the twist derives another reduction"


# --- events -----------------------------------------------------------------------------------


def test_events_fire_in_loop_order() -> None:
    """A run-scoped observer sees run start, then splices (with receipts), then pass ends —
    the engine's own moments, one protocol."""

    class Recorder(PipelineStrategy):
        def __init__(self) -> None:
            self.events: list[str] = []

        def on_run_start(self, e) -> None:
            self.events.append("run_start")

        def on_splice(self, e) -> None:
            self.events.append(f"splice:{e.pass_name}")

        def on_spliced(self, e) -> None:
            assert e.receipt.redirected, "the receipt names what was redirected"
            self.events.append(f"spliced:{e.pass_name}")

        def on_pass_end(self, e) -> None:
            self.events.append(f"pass_end:{e.pass_name}")

    rec = Recorder()
    run = Run(pipeline=Pipeline.build(LOOP_PASSES).with_strategies(rec), ctx=_CTX)
    run.resolve(_matmul(), lambda fp: next(fp.leaves()))
    assert rec.events[0] == "run_start"
    assert any(ev.startswith("splice:frontend/decomposition") for ev in rec.events)
    # Every pre-splice event has its post-splice receipt event.
    assert sum(ev.startswith("splice:") for ev in rec.events) == sum(ev.startswith("spliced:") for ev in rec.events)
    pass_ends = [ev.removeprefix("pass_end:") for ev in rec.events if ev.startswith("pass_end:")]
    assert pass_ends == LOOP_PASSES, "one pass-end per pass, in pipeline order"
