"""A range and a value defined once, from the Loop IR body to the kernel a tier compiles.

A reduction masked to a triangle reaches the term as a range on the fold's axis, and the classic schedule and the
kernel lowering read it as the mask it stands for (``TileOp.masked``). A value the body defines once and reads by
index lifts as the kernel that computes it at each read. Both are held to numpy on the CPU backend, which compiles
the lowered kernel.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Select, parse_body
from emmy.compiler.pipeline import Pipeline
from tests.compiler.helpers import inject_constants

#: ``O[i, j] = sum(X[i, k] * Y[k, j] for k <= i)``, the bound written as a mask to the sum's neutral element.
MASKED = """
for i in 0..8:
  for j in 0..8:
    for k in 0..8:
      x = load X[i, k]
      y = load Y[k, j]
      p = multiply(x, y)
      let zero = 0.0
      m = select:
        p when k < i + 1
        zero when true
      s <- add(s, m) over k
    O[i, j] = s
"""

#: The same sum over the keys at or past the row: a range that does not start at zero.
FROM_THE_ROW = """
for i in 0..8:
  for j in 0..8:
    for k in i..8 of 8:
      x = load X[i, k]
      y = load Y[k, j]
      p = multiply(x, y)
      s <- add(s, p) over k
    O[i, j] = s
"""

#: A normalized key read at the cell and inside the sum, defined once.
DEFINED = """
for h in 0..4:
  for a in 0..16:
    k = load K[h, a]
    w = load W[h]
    v = multiply(k, w)
    kn[h, a] <- v
  for n in 0..16:
    for m in 0..16:
      b = pre kn[h, m]
      c = pre kn[h, n]
      p = multiply(b, c)
      s <- add(s, p) over m
    O[h, n] = s
"""

#: The same kernel with the key computed at each read: what fusion writes today.
COPIED = """
for h in 0..4:
  for n in 0..16:
    for m in 0..16:
      w = load W[h]
      k = load K[h, m]
      b = multiply(k, w)
      j = load K[h, n]
      c = multiply(j, w)
      p = multiply(b, c)
      s <- add(s, p) over m
    O[h, n] = s
"""


def _graph(text: str, inputs: dict[str, tuple[int, ...]], shape: tuple[int, ...]) -> Graph:
    graph = Graph()
    for name, dims in inputs.items():
        graph.add_node(InputOp(), [], Tensor(name, dims), node_id=name)
    graph.add_node(LoopOp(body=parse_body(text)), list(inputs), Tensor("O", shape), node_id="O")
    graph.inputs, graph.outputs = list(inputs), ["O"]
    return graph


def _tile(text: str, inputs: dict[str, tuple[int, ...]], shape: tuple[int, ...]):
    return Pipeline.build(["tile/lift"], select=["lift"]).run(_graph(text, inputs, shape)).nodes["O"].op


def _data(inputs: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in inputs.items()}


def test_a_mask_to_the_neutral_element_is_a_range_in_the_body_and_on_the_fold() -> None:
    square = {"X": (8, 8), "Y": (8, 8)}
    tile = _tile(MASKED, square, (8, 8))
    assert "for a2 in 0..a0 + 1 of 8:" in str(tile.loop_body) and "select" not in str(tile.loop_body)
    start, end = tile.op.range
    assert (str(start), str(end)) == ("0", "a0 + 1")
    assert "a0" in tile.op.free_axes, "the range is evaluated where the term is"


def test_a_tier_reads_a_range_as_the_mask_it_stands_for() -> None:
    """The masked form is the tree the lift gave before the range existed: the fold over the whole axis, its value
    selected against the seed, the seed a literal operand. It is another spelling of one kernel."""
    tile = _tile(MASKED, {"X": (8, 8), "Y": (8, 8)}, (8, 8))
    masked = tile.masked()
    assert masked.op.range is None and masked is not tile
    (mask,) = [stmt for stmt in masked.op.lift.body if isinstance(stmt, Select)]
    assert [str(branch.select) for branch in mask.branches] == ["a2 < a0 + 1", "true"]
    assert len(masked.op.operands) == len(tile.op.operands) + 1, "the seed is a literal operand"
    assert masked.identity_key(structural=False, with_io=True) == tile.identity_key(structural=False, with_io=True)
    assert masked.masked() is masked, "a tree without a range is itself"


def test_a_range_that_starts_past_zero_walks_from_zero_and_clamps_what_it_reads() -> None:
    tile = _tile(FROM_THE_ROW, {"X": (8, 8), "Y": (8, 8)}, (8, 8))
    assert [str(bound) for bound in tile.op.range] == ["a0", "8"]
    masked = tile.masked()
    (mask,) = [stmt for stmt in masked.op.lift.body if isinstance(stmt, Select)]
    assert str(mask.branches[0].select) == "a0 + a2 < 8"
    loads = [str(edge.as_slab().load) for edge in masked.op.operands if edge.as_slab() is not None]
    assert all("a0 + a2 < 8 ? a0 + a2 : 7" in load for load in loads), loads


@pytest.mark.parametrize(("text", "expected"), [(MASKED, np.tril), (FROM_THE_ROW, np.triu)], ids=["masked", "from_the_row"])
def test_the_masked_form_computes_what_the_range_means(text: str, expected) -> None:
    """The tree a tier reads, lowered and run as it stands on the reference model: a walk of the whole extent from
    zero whose trips past the range fold the seed and read inside their buffers."""
    from tests.compiler.ir import reference_model  # noqa: PLC0415

    shapes = {"X": (8, 8), "Y": (8, 8)}
    data = _data(shapes)
    body = _tile(text, shapes, (8, 8)).masked().loop_body
    assert " of " not in str(body), "no range is left"
    got = reference_model.run(body, data, {"O": (8, 8)})["O"]
    np.testing.assert_allclose(got, expected(data["X"]) @ data["Y"], rtol=1e-4, atol=1e-4)


def test_a_value_defined_once_is_the_kernel_that_computes_it_at_each_read() -> None:
    shapes = {"K": (4, 16), "W": (4,)}
    defined, copied = _tile(DEFINED, shapes, (4, 16)), _tile(COPIED, shapes, (4, 16))
    assert defined.identity_key(structural=False, with_io=True) == copied.identity_key(structural=False, with_io=True)
    op = LoopOp(body=parse_body(DEFINED))
    assert "<-" in str(op.body) and sorted(op.inputs) == ["K", "W"] and list(op.outputs) == ["O"], "the value is the kernel's own"


def _native_available() -> bool:
    try:
        import llvmlite  # noqa: F401, PLC0415

        from emmy import emmy_runtime  # noqa: PLC0415
    except ImportError:
        return False
    return hasattr(emmy_runtime, "CpuExecutor") and bool(shutil.which("cc") or shutil.which("clang"))


@pytest.mark.skipif(not _native_available(), reason="needs llvmlite, the runtime extension and a C linker")
@pytest.mark.parametrize(
    ("text", "inputs", "shape"),
    [
        (MASKED, {"X": (8, 8), "Y": (8, 8)}, (8, 8)),
        (FROM_THE_ROW, {"X": (8, 8), "Y": (8, 8)}, (8, 8)),
        (DEFINED, {"K": (4, 16), "W": (4,)}, (4, 16)),
    ],
    ids=["masked", "from_the_row", "defined"],
)
def test_the_lowered_kernel_computes_what_the_body_means(text: str, inputs: dict, shape: tuple) -> None:
    from emmy.compiler.backend.cpu import CpuBackend  # noqa: PLC0415

    data = _data(inputs)
    reference = NumpyBackend()
    compiled = reference.compile(_graph(text, inputs, shape))
    want = reference.run(compiled, input_data=inject_constants(dict(data), compiled))[0].outputs["O"]
    cpu = CpuBackend(threads=2)
    program = cpu.compile(_graph(text, inputs, shape))
    got = cpu.run(program, input_data=inject_constants(dict(data), program.graph))[0].outputs["O"]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    if text is MASKED:
        np.testing.assert_allclose(want, np.tril(data["X"]) @ data["Y"], rtol=1e-4, atol=1e-4)
    if text is FROM_THE_ROW:
        np.testing.assert_allclose(want, np.triu(data["X"]) @ data["Y"], rtol=1e-4, atol=1e-4)
