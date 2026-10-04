"""Offer a constant's folded transpose and its original storage layout as measured alternatives."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import TransposeOp
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import family_pins
from emmy.compiler.pipeline.passes.tile._row import reformed
from emmy.compiler.pipeline.passes.tile._split import add_output_piece


def _source_shape(node: Node) -> tuple | None:
    op = node.op
    if not isinstance(op, ConstantOp) or op.value is not None or not op.load_ops or len(node.output.shape) != 2:
        return None
    last = op.load_ops[-1]
    if not isinstance(last, TransposeOp) or len(last.axes) != 2:
        return None
    axes = tuple(axis % 2 for axis in last.axes)
    return tuple(reversed(node.output.shape)) if axes in {(0, 1), (1, 0)} else None


def _edit(term: Fold, names: dict[str, str]) -> Fold:
    def load(stmt):
        if isinstance(stmt, Load) and stmt.input in names:
            return replace(stmt, input=names[stmt.input], index=tuple(reversed(stmt.index)))
        return stmt

    return replace(
        term,
        operands=tuple(_edit(operand, names) for operand in term.operands),
        lift=replace(term.lift, body=term.lift.body.map(load)),
    )


def _source_fragment(match: Match, root: Node, names: tuple[str, ...]) -> Graph:
    graph = match.graph
    source = {name: f"{name}__source" for name in names}
    fragment = Graph()
    for name in root.inputs:
        if name not in source:
            fragment.add_node(InputOp(), [], graph.buffer(name), node_id=name)
            continue
        folded = graph.producer(name)
        assert folded is not None and isinstance(folded.op, ConstantOp)
        raw = source[name]
        shape = _source_shape(folded)
        assert shape is not None
        op = replace(folded.op, name=raw, load_ops=folded.op.load_ops[:-1])
        if (existing := graph.buffer(raw)) is not None:
            assert existing.shape == shape and graph.producer(raw).op == op
            fragment.add_node(InputOp(), [], existing, node_id=raw)
        else:
            fragment.add_node(op, [], Tensor(raw, shape, folded.output.dtype), node_id=raw)
    tile: TileOp = root.op
    piece = reformed(replace(tile, op=_edit(tile.op, source), layout_decided=(*tile.layout_decided, *names)))
    inputs = [source.get(name, name) for name in root.inputs]
    return add_output_piece(match, fragment, root, piece, inputs, suffix="__layout")


def layout_forks(match: Match, root: Node) -> list[DeferredFork] | None:
    """The first undecided transposed constant read by this kernel, with a joint arm for equal reads.

    Equal read expressions can be two channels of one contraction. Offering their joint source arm
    lets evidence compare that kernel with the folded one without requiring either mixed layout to win first.
    """
    tile: TileOp = root.op
    if tile.op is None or tile.loop_body is None:
        return None
    loads: dict[str, list[Load]] = {}
    for stmt in tile.loop_body.iter():
        if isinstance(stmt, Load):
            loads.setdefault(stmt.input, []).append(stmt)
    eligible = []
    for name in root.inputs:
        node = match.graph.producer(name)
        if (
            name not in tile.layout_decided
            and node is not None
            and _source_shape(node) is not None
            and name in loads
            and all(len(load.index) == 2 for load in loads[name])
        ):
            eligible.append(name)
    if not eligible:
        return None
    first = eligible[0]
    signature = tuple(load.index for load in loads[first])
    group = tuple(name for name in eligible if tuple(load.index for load in loads[name]) == signature)
    folded = DeferredFork(lambda: replace(tile, layout_decided=(*tile.layout_decided, first)), {f"LAYOUT@{first}": "folded"})
    single = DeferredFork(lambda: _source_fragment(match, root, (first,)), {f"LAYOUT@{first}": "source"}, structural=True)
    options = [folded, single]
    if len(group) > 1:
        options.append(
            DeferredFork(
                lambda: _source_fragment(match, root, group),
                {f"LAYOUT@{name}": "source" for name in group},
                structural=True,
            )
        )
    pins = dict(family_pins("LAYOUT"))
    if any(value not in {"folded", "source"} for value in pins.values()):
        raise ValueError("LAYOUT must be 'folded' or 'source'")
    if len(group) > 1 and all(pins.get(f"LAYOUT@{name}", pins.get("LAYOUT")) == "source" for name in group):
        return [options[-1]]
    return [
        option
        for option in options
        if all(pins.get(key, pins.get("LAYOUT", value)) == value for key, value in option.knobs.items())
    ]
