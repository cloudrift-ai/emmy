"""``HOIST``: a value every trip of a loop computes alike leaves the loop where the knob asks, and nowhere else.

An online softmax's fold loop computes the constant ``one`` every trip. The knob offers one arm until a tune measures
the motion, so a compile that pins nothing leaves the loop as the lowering spelled it; ``HOIST=1`` computes the
constant once ahead of the loop and the kernel still computes the reference answer.
"""

from __future__ import annotations

from emmy.compiler.ir.cuda.ir import CudaOp
from tests.compiler.helpers import requires_cuda
from tests.compiler.realization import helpers

_CASE = "reduce/online-softmax-4x128.json"


def _source(monkeypatch, hoist: str | None) -> str:
    if hoist is not None:
        monkeypatch.setenv("EMMY_HOIST", hoist)
    case = helpers.load_case(helpers.CASES_DIR / _CASE)
    graph, _ = helpers.lowered(case, case.context())
    (op,) = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    return op.kernel_source


def _depth(source: str, line: str) -> int:
    """How deep ``line`` is indented: one level per scope around it."""
    text = next(text for text in source.splitlines() if text.strip() == line)
    return len(text) - len(text.lstrip())


def test_only_the_knob_moves_the_trip_invariant_out_of_the_loop(monkeypatch) -> None:
    placed = _source(monkeypatch, None)
    hoisted = _source(monkeypatch, "1")
    assert _depth(hoisted, "float one = 1.0f;") < _depth(placed, "float one = 1.0f;")


@requires_cuda
def test_a_hoisted_kernel_computes_the_reference(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_HOIST", "1")
    case = helpers.load_case(helpers.CASES_DIR / _CASE)
    helpers.correct(case, helpers.built(case))
