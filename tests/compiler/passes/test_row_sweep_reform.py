"""Independent output sweeps formed into one shared-coordinate contraction."""

from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.pipeline.passes.tile._row import reformed
from tests.compiler.terms import contraction, projection, slab


def _siblings(
    *,
    second_extent: int = 8,
    second_window: Window | None = None,
    cross_coordinate: bool = False,
    cross_write: bool = False,
    captured_name: bool = False,
    rectangular: bool = False,
    rectangular_window: Window | None = None,
    value_coordinate: bool = False,
) -> TileOp:
    m, n, p, k = Axis("m", 1), Axis("n", 8), Axis("p", second_extent, window=second_window), Axis("k", 16)
    q = Axis("q", 4, window=rectangular_window)
    if rectangular:
        p = Axis("p", 2, window=second_window)
    first = contraction(k, slab("x0", "x", "m", "k"), (slab("w0v", "w0", "k", "n"), "acc0"))
    second_index = ("k", "n", "p") if cross_coordinate else ("k", "p", "q") if rectangular else ("k", "p")
    second = contraction(k, slab("x1", "x", "m", "k"), (slab("n" if captured_name else "w1v", "w1", *second_index), "acc1"))
    if value_coordinate:
        second = projection((second,), body=(Assign(name="acc1_value", op="add", args=("acc1", "p")),), results=("acc1_value",))
    root = projection((first, second), results=("acc0", "acc1_value" if value_coordinate else "acc1"))
    return TileOp(
        op=root,
        place=Placement(free=(m,)),
        axes=(m, n, p, q, k) if rectangular else (m, n, p, k),
        inputs={
            "x": Tensor("x", (1, 16), "f16"),
            "w0": Tensor("w0", (16, 8), "f16"),
            "w1": Tensor("w1", (16, 8, second_extent) if cross_coordinate else (16, 2, 4) if rectangular else (16, second_extent), "f16"),
        },
        outputs={"out0": Tensor("out0", (1, 8), "f16"), "out1": Tensor("out1", (1, 2, 4) if rectangular else (1, second_extent), "f16")},
        output_specs=(
            OutputSpec(write=Write(output="out0", index=(Var("m"), Var("n")), value="acc0"), sweep=(n,)),
            OutputSpec(
                write=Write(
                    output="out1",
                    index=(Var("m"), Var("p"), Var("q")) if rectangular else (Var("m"), Var("p") + Var("n") if cross_write else Var("p")),
                    value="acc1_value" if value_coordinate else "acc1",
                ),
                sweep=(p, q) if rectangular else (p,),
            ),
        ),
    )


def test_equal_independent_sweeps_form_one_contraction() -> None:
    formed = reformed(_siblings())
    contractions = [site.node for site in formed.sites if site.node.as_contraction() is not None]
    assert len(contractions) == 1 and len(contractions[0].bilinear_channels()) == 2
    assert tuple(axis.extent.as_static() for axis in formed.place.free) == (1, 8)
    assert all(not spec.sweep for spec in formed.output_specs)
    assert formed.output_specs[0].write.index == formed.output_specs[1].write.index


def test_rectangular_and_flat_sweeps_form_one_contraction() -> None:
    formed = reformed(_siblings(rectangular=True))
    contractions = [site.node for site in formed.sites if site.node.as_contraction() is not None]
    assert len(contractions) == 1 and len(contractions[0].bilinear_channels()) == 2
    assert tuple(axis.extent.as_static() for axis in formed.place.free) == (1, 8)
    assert all(not spec.sweep for spec in formed.output_specs)
    assert len(next(spec.write.index for spec in formed.output_specs if spec.write.output == "out1")) == 3


@pytest.mark.parametrize("rectangular", [False, True], ids=["flat", "rectangular"])
def test_reform_preserves_both_output_values(rectangular) -> None:
    tile = _siblings(rectangular=rectangular)
    rng = np.random.default_rng(1)
    inputs = {
        name: rng.standard_normal(tuple(dim.as_static() for dim in tensor.shape)).astype(np.float32)
        for name, tensor in tile.inputs.items()
    }
    expected = {
        "out0": inputs["x"] @ inputs["w0"],
        "out1": np.tensordot(inputs["x"], inputs["w1"], axes=([1], [0])),
    }
    for piece in (tile, reformed(tile)):
        loop = LoopOp(body=piece.loop_body, inputs=piece.inputs, outputs=piece.outputs)
        actual = loop.forward(*(inputs[name] for name in piece.inputs))
        for name, value in zip(piece.outputs, actual, strict=True):
            np.testing.assert_allclose(value, expected[name], rtol=1e-6, atol=1e-6)


def test_unequal_sweeps_and_windows_remain_independent() -> None:
    nested = Window(parent=Axis("parent", 8, window=Window(partition=True)))
    for tile in (_siblings(second_extent=7), _siblings(second_window=Window(parent=Axis("parent", 8))), _siblings(second_window=nested)):
        assert reformed(tile) is tile
    windowed_rectangle = _siblings(rectangular=True, rectangular_window=Window(parent=Axis("parent", 4)))
    assert reformed(windowed_rectangle) is windowed_rectangle


def test_cross_coordinate_branch_keeps_separate_sweeps() -> None:
    tile = _siblings(cross_coordinate=True)
    assert reformed(tile) is tile


def test_cross_coordinate_store_keeps_separate_sweeps() -> None:
    tile = _siblings(cross_write=True)
    assert reformed(tile) is tile


def test_capture_of_first_sweep_name_keeps_separate_sweeps() -> None:
    tile = _siblings(captured_name=True)
    assert reformed(tile) is tile


def test_sweep_used_as_value_keeps_separate_sweeps() -> None:
    tile = _siblings(rectangular=True, value_coordinate=True)
    assert reformed(tile) is tile


def test_repeated_output_buffer_keeps_separate_sweeps() -> None:
    tile = _siblings()
    first, second = tile.output_specs
    duplicate = replace(tile, output_specs=(first, replace(second, write=replace(second.write, output="out0"))))
    assert reformed(duplicate) is duplicate
