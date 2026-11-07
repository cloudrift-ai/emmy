"""A body's text over the realization corpus: ``str`` and ``parse_body`` are exact inverses on every stored kernel and
on lowered Kernel IR. Every statement kind's own text is the crate's test."""

import json

import pytest

from emmy.compiler.ir.stmt import Body, Loop, StridedLoop, parse_body
from tests.compiler.realization import helpers


def _exact(body: Body) -> list:
    """Everything the statements hold, including what equality leaves out: an axis's window, a dim's hint."""
    axes = [stmt.axis for stmt in body.iter() if isinstance(stmt, (Loop, StridedLoop))]
    return [list(body.iter()), [(axis.window, axis.extent.hint) for axis in axes]]


def _stored_bodies(value) -> list:
    if isinstance(value, dict):
        own = [value["attrs"]["body"]] if value.get("op") == "loop" else []
        return own + [body for item in value.values() for body in _stored_bodies(item)]
    return [body for item in value for body in _stored_bodies(item)] if isinstance(value, list) else []


def test_every_corpus_kernel_reads_back_from_its_text():
    stored = [Body.from_wire(wire) for path in helpers.case_files() for wire in _stored_bodies(json.loads(path.read_text()))]
    assert len(stored) > 200
    for body in stored:
        back = parse_body(str(body))
        assert _exact(back) == _exact(body)
        assert str(back) == str(body)


#: Six cases whose lowered Kernel IR spans the kernel kinds the corpus lowers to.
KERNEL_CASES = [
    "attention/sdpa-hd64-wgmma-value-sm90.json",
    "matmul/f16-mma-f16acc-tma.json",
    "matmul/f16-warp-band-p1-tma-d1-m128.json",
    "matmul/fp4-tma-w4a4-linear-ring.json",
    "reduce/computed-provider-coop-t-sm70.json",
    "scan/f32-cumsum-warp-prefix.json",
]


def _kernel_bodies(case_path: str) -> list[Body]:
    from emmy.compiler.ir.kernel.ir import KernelOp

    case = helpers.load_case(helpers.CASES_DIR / case_path)
    graph, _ = helpers.lowered(case, case.context())
    return [op.body for node in graph.nodes.values() for op in node.op.source_chain() if isinstance(op, KernelOp)]


@pytest.mark.parametrize("case_path", KERNEL_CASES)
def test_a_kernel_ir_body_reads_back_from_its_text(case_path):
    """A kind of the Kernel IR prints and reads through its declaration, so a lowered kernel's text is exact."""
    bodies = _kernel_bodies(case_path)
    assert bodies
    for body in bodies:
        back = parse_body(str(body))
        assert _exact(back) == _exact(body)
        assert str(back) == str(body)
