"""Independent output sweeps formed into one shared-coordinate contraction."""

from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.pipeline.passes.tile._row import _align_owned_sweeps, reformed
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
        name: rng.standard_normal(tuple(dim.as_static() for dim in tensor.shape)).astype(np.float32) for name, tensor in tile.inputs.items()
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


def _transposed_siblings(*, incompatible=False, repeated=False, batched=False) -> TileOp:
    n, m, h, row, d = Axis("n", 8), Axis("m", 3), Axis("h", 2), Axis("row", 3), Axis("d", 4)
    k = Axis("k", 3 if repeated else 5)
    k1 = Axis("k1", k.extent)
    b, batch = Axis("b", 2), Axis("batch", 2)
    first_index = ("b", "m", "k") if batched else ("m", "k")
    second_index = ("batch", "row", "k1") if batched else ("row", "row") if repeated else ("k1", "row") if incompatible else ("row", "k1")
    first = contraction(k, slab("x0", "x", *first_index), (slab("w0v", "w0", "k", "n"), "acc0"))
    second = contraction(k1, slab("x1", "x", *second_index), (slab("w1v", "w1", "k1", "h", "d"), "acc1"))
    first_sweep, second_sweep = ((n, b, m), (h, row, batch, d)) if batched else ((n, m), (h, row, d))
    return TileOp(
        op=projection((first, second), results=("acc0", "acc1")),
        place=Placement(free=()),
        axes=(n, m, h, row, d, k, k1, b, batch),
        inputs={
            "x": Tensor("x", (2, 3, k.extent) if batched else (3, k.extent), "f32"),
            "w0": Tensor("w0", (k.extent, 8), "f32"),
            "w1": Tensor("w1", (k.extent, 2, 4), "f32"),
        },
        outputs={
            "out0": Tensor("out0", (8, 2, 3) if batched else (8, 3), "f32"),
            "out1": Tensor("out1", (2, 3, 2, 4) if batched else (2, 3, 4), "f32"),
        },
        output_specs=(
            OutputSpec(write=Write(output="out0", index=tuple(Var(axis.name) for axis in first_sweep), value="acc0"), sweep=first_sweep),
            OutputSpec(write=Write(output="out1", index=tuple(Var(axis.name) for axis in second_sweep), value="acc1"), sweep=second_sweep),
        ),
    )


@pytest.mark.parametrize("batched", [False, True], ids=["one_shared_axis", "two_shared_axes"])
def test_transposed_sweeps_preserve_shared_rows_and_output_order(batched) -> None:
    tile = _transposed_siblings(batched=batched)
    formed = reformed(tile)
    contractions = [site.node for site in formed.sites if site.node.as_contraction() is not None]
    assert len(contractions) == 1 and len(contractions[0].bilinear_channels()) == 2
    inputs = {
        name: np.arange(np.prod([dim.as_static() for dim in tensor.shape]), dtype=np.float32).reshape(
            tuple(dim.as_static() for dim in tensor.shape)
        ) / 10
        for name, tensor in tile.inputs.items()
    }
    expected = {
        "out0": np.tensordot(inputs["x"], inputs["w0"], axes=([-1], [0])).transpose((2, 0, 1) if batched else (1, 0)),
        "out1": np.tensordot(inputs["x"], inputs["w1"], axes=([-1], [0])).transpose((2, 1, 0, 3) if batched else (1, 0, 2)),
    }
    for piece in (tile, formed):
        loop = LoopOp(body=piece.loop_body, inputs=piece.inputs, outputs=piece.outputs)
        actual = loop.forward(*(inputs[name] for name in piece.inputs))
        for name, value in zip(piece.outputs, actual, strict=True):
            np.testing.assert_allclose(value, expected[name], rtol=1e-6, atol=1e-6)
    assert {spec.write.output: len(spec.write.index) for spec in formed.output_specs} == {"out0": 3 if batched else 2, "out1": 4 if batched else 3}


def test_incompatible_row_mapping_keeps_separate_sweeps() -> None:
    tile = _transposed_siblings(incompatible=True)
    assert reformed(tile) is tile


def test_repeated_input_coordinate_cannot_bind_the_reduction_as_a_row() -> None:
    tile = _transposed_siblings(repeated=True)
    assert reformed(tile) is tile


@pytest.mark.parametrize("extent,window", [(Dim("width"), None), (Dim(8), Window(parent=Axis("whole", 8)))])
def test_equal_single_axis_symbolic_and_windowed_sweeps_keep_their_domain(extent, window) -> None:
    tile = _siblings()
    specs = tuple(replace(spec, sweep=(replace(spec.sweep[0], extent=extent, window=window),)) for spec in tile.output_specs)
    axes = tuple(replace(axis, extent=extent, window=window) if axis.name in {"n", "p"} else axis for axis in tile.axes)
    tile = replace(tile, axes=axes, output_specs=specs)
    formed = _align_owned_sweeps(tile)
    assert formed is not tile
    assert all(not spec.sweep for spec in formed.output_specs)
    assert formed.place.free[-1].extent == extent and formed.place.free[-1].window == window
