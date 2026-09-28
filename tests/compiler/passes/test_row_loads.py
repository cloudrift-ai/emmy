"""A cooperative norm + RoPE row reads each cell of its row and its weight once per lane.

Qwen3's q/k RMSNorm feeds RoPE's rotate-half, a concatenation whose two branches read the row at
``i + 64`` and ``i - 64``. Before ``045_merge_select_loads`` every cell loaded both branches (each
clamped in range), and before ``047_reuse_lane_loads`` the projection re-read the cells the reduce
had just read, and a partner read re-read a cell another trip of the same lane holds.
"""

from __future__ import annotations

import numpy as np
import torch

from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt.leaves import Load
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda

_ROWS, _HEADS, _WIDTH = 64, 4, 128
_LANES = 32

_CODE = (
    "(lambda x, w, c, s: (lambda t: t * c + torch.cat((-t[..., 64:], t[..., :64]), -1) * s)"
    "(torch.nn.functional.rms_norm(x, (128,), w, 1e-6)))"
    f"(torch.randn({_ROWS}, {_HEADS}, {_WIDTH}, dtype=torch.float16), torch.randn({_WIDTH}, dtype=torch.float16),"
    f" torch.randn({_ROWS}, 1, {_WIDTH}, dtype=torch.float16), torch.randn({_ROWS}, 1, {_WIDTH}, dtype=torch.float16))"
)
_PINS = {"PLACE": "fuse", "WORK": f"t{_LANES}x4", "REDUCE": "coop", "FAST_MATH": "False"}


def _kernel_loads() -> dict[str, int]:
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline

    graph = graph_from_code(_CODE)[0]
    with pinned_knobs(_PINS):
        lowered = Pipeline.build(KERNEL_PASSES).run(graph, ctx=Context.from_target((8, 9)))
    (op,) = [node.op for node in lowered.nodes.values() if isinstance(node.op, KernelOp)]
    counts: dict[str, int] = {}
    for stmt in op.body.iter():
        if isinstance(stmt, Load):
            counts[stmt.input] = counts.get(stmt.input, 0) + stmt.width
    return counts


def test_a_norm_rope_lane_loads_each_row_cell_and_weight_once() -> None:
    """A lane owns ``width / lanes`` cells of the row: it loads each of them once (the reduce's
    read serves the projection and the rotate-half partner), and each weight cell once."""
    counts = _kernel_loads()
    per_lane = _WIDTH // _LANES
    row, weight, cos, sin = (name for name in counts if name.startswith("x"))
    assert counts[row] == per_lane, counts
    assert counts[weight] == per_lane, counts
    assert counts[cos] == counts[sin] == per_lane, counts


@requires_cuda
def test_a_norm_rope_row_computes_the_right_answer() -> None:
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.backend import CudaBackend

    graph = graph_from_code(_CODE)[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((_ROWS, _HEADS, _WIDTH)).astype(np.float16)
    w = rng.standard_normal(_WIDTH).astype(np.float16)
    c, s = (rng.standard_normal((_ROWS, 1, _WIDTH)).astype(np.float16) for _ in range(2))
    with pinned_knobs(_PINS):
        compiled = CudaBackend().compile(graph)
    inputs = dict(zip(compiled.inputs, (x, w, c, s), strict=True))
    (out,) = compiled.outputs
    got = CudaBackend().run(compiled, input_data=inputs)[0].outputs[out].astype(np.float32)
    t = torch.nn.functional.rms_norm(torch.from_numpy(x).float(), (_WIDTH,), torch.from_numpy(w).float(), 1e-6)
    rotated = torch.cat((-t[..., 64:], t[..., :64]), -1)
    expected = (t * torch.from_numpy(c).float() + rotated * torch.from_numpy(s).float()).numpy()
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_a_lane_loop_unrolls_only_when_its_start_is_below_its_step() -> None:
    """Unrolling writes coordinates ``start + k·step`` for every ``k < extent / step``; they stay inside the
    loop only when ``start`` is proven in ``[0, step)``. A start at or past the step (or unknown) keeps the loop."""
    import importlib

    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import Interval, Literal, SimplifyCtx, Var
    from emmy.compiler.ir.stmt import Body, StridedLoop
    from emmy.compiler.ir.stmt.leaves import Load

    trips = importlib.import_module("emmy.compiler.pipeline.passes.lowering.kernel.047_reuse_lane_loads")._trips
    body = Body((Load(name="v", input="x", index=(Var("i"),), dtype="float16"),))
    lanes = SimplifyCtx.empty().extend("lane", Interval(0, 15))

    def loop(start):
        return StridedLoop(axis=Axis("i", 64), start=start, step=Literal(16, "int"), body=body)

    assert trips(loop(Var("lane")), lanes) == 4
    assert trips(loop(Literal(0, "int")), lanes) == 4
    assert trips(loop(Literal(32, "int")), lanes) is None
    assert trips(loop(Var("unknown")), lanes) is None


def test_sibling_lane_loops_over_one_cone_unroll_to_distinct_names() -> None:
    """Independently spliced operand cones bind the same names, each inside its own lane loop. Unrolled into
    one scope, a repeat of those names is a second C declaration nvcc rejects; the later loop takes its own."""
    import importlib

    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import Interval, Literal, SimplifyCtx, Var
    from emmy.compiler.ir.stmt import Assign, Body, StridedLoop
    from emmy.compiler.ir.stmt.leaves import Load, Write

    walk = importlib.import_module("emmy.compiler.pipeline.passes.lowering.kernel.047_reuse_lane_loads")._walk
    lanes = SimplifyCtx.empty().extend("lane", Interval(0, 15))

    def loop(out):
        cone = (Load(name="v", input="x", index=(Var("i"),), dtype="float16"), Assign("w", "exp", ("v",)))
        body = Body((*cone, Write(out, (Var("i"),), "w")))
        return StridedLoop(axis=Axis("i", 64), start=Var("lane"), step=Literal(16, "int"), body=body)

    one = walk(Body((loop("a"),)), frozenset({"a"}), lanes)
    assert [s.name for s in one if isinstance(s, Assign)] == [f"w__u{k}" for k in range(4)]  # a lone loop keeps its names
    two = walk(Body((loop("a"), loop("b"))), frozenset({"a", "b"}), lanes)
    names = [name for s in two for name in s.defines()]
    assert len(names) == len(set(names))
    assert sum(isinstance(s, Load) for s in two) == 4  # the second cone still reads each cell from the first's loads
