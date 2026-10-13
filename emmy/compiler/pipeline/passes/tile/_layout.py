"""Offer a constant's folded transpose and its original storage layout as measured alternatives."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from contextvars import ContextVar
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
from emmy.compiler.pipeline.passes.tile._row import io_shapes, reformed
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


#: A kernel reads constant ``name`` in its source layout from buffer ``name + SOURCE_SUFFIX``.
SOURCE_SUFFIX = "__source"

_SETTLED: ContextVar[Mapping[str, str] | None] = ContextVar("settled_layouts", default=None)


@contextlib.contextmanager
def settled_layouts(layouts: Mapping[str, str]):
    """Compile with the layout of each constant in ``layouts`` (buffer name -> ``folded`` / ``source``) already
    settled, by another program that shares its device copy: the fork offers a settled constant in that layout only.
    Unlike a ``LAYOUT`` pin, which asks for a choice the compile must realize, this constrains a fork only where one
    arises."""
    token = _SETTLED.set(dict(layouts))
    try:
        yield
    finally:
        _SETTLED.reset(token)


def _source_fragment(match: Match, root: Node, names: tuple[str, ...]) -> Graph:
    graph = match.graph
    source = {name: f"{name}{SOURCE_SUFFIX}" for name in names}
    fragment = Graph()
    tile: TileOp = root.op
    shapes = io_shapes(tile)  # every buffer the piece reads, each raw source among them
    for name in root.inputs:
        if name not in source:
            fragment.add_node(InputOp(), [], graph.buffer(name), node_id=name)
            continue
        folded = graph.producer(name)
        assert folded is not None and isinstance(folded.op, ConstantOp)
        raw = source[name]
        shape = shapes[raw] = _source_shape(folded)
        assert shape is not None
        op = replace(folded.op, load_ops=folded.op.load_ops[:-1])
        if (existing := graph.buffer(raw)) is not None:
            assert existing.shape == shape and graph.producer(raw).op == op
            fragment.add_node(InputOp(), [], existing, node_id=raw)
        else:
            fragment.add_node(op, [], Tensor(raw, shape, folded.output.dtype), node_id=raw)
    piece = reformed(replace(tile, op=_edit(tile.op, source), source=None, layout_decided=(*tile.layout_decided, *names)), shapes)
    inputs = [source.get(name, name) for name in root.inputs]
    return add_output_piece(match, fragment, root, piece, inputs, suffix="__layout")


def _loads(tile: TileOp) -> dict[str, list[Load]]:
    loads: dict[str, list[Load]] = {}
    if tile.loop_body is not None:
        for stmt in tile.loop_body.iter():
            if isinstance(stmt, Load):
                loads.setdefault(stmt.input, []).append(stmt)
    return loads


def _undecided(op, name: str) -> bool:
    """Whether this kernel still offers a layout fork for constant ``name``. It does not when it is no tile kernel or
    has no loop body yet, when it carries a state (its one kernel-set decision is the split), when it is already
    placed or scheduled, when its layout of ``name`` is already decided, or when it reads ``name`` other than through
    2-D indices (or not at all)."""
    if not isinstance(op, TileOp) or op.op is None or op.loop_body is None or op.carries:
        return False
    if op.place.is_mapped or op.schedule is not None or name in op.layout_decided:
        return False
    reads = _loads(op).get(name)
    return bool(reads) and all(len(load.index) == 2 for load in reads)


def fork_less_reads(graph: Graph) -> set[str]:
    """The transposed constants of ``graph`` that some kernel reads with no layout fork (:func:`_undecided`): those
    stay folded whatever layout another reader decides, so a program sharing them settles them folded."""
    return {
        name
        for node in graph.nodes.values()
        for name in node.inputs
        if (producer := graph.producer(name)) is not None and _source_shape(producer) is not None and not _undecided(node.op, name)
    }


def _taken_layout(graph: Graph, root: Node, name: str) -> str | None:
    """The layout another kernel of this program, or a program settling it (:func:`settled_layouts`), already reads
    constant ``name`` in, or ``None``.

    A reader the cut pass no longer offers a layout fork for ``name`` (see :func:`_undecided`) reads it folded for good.
    Serving binds one device copy per layout, so a second layout would hold the weight twice."""
    if (settled := (_SETTLED.get() or {}).get(name)) is not None:
        return settled
    if graph.buffer_users(f"{name}{SOURCE_SUFFIX}"):
        return "source"
    if any(user != root.id and not _undecided(graph.nodes[user].op, name) for user in graph.buffer_users(name)):
        return "folded"
    return None


def layout_forks(match: Match, root: Node) -> list[DeferredFork] | None:
    """The first undecided transposed constant read by this kernel, with a joint arm for equal reads.

    Equal read expressions can be two channels of one contraction. Offering their joint source arm
    lets evidence compare that kernel with the folded one without requiring either mixed layout to win first.
    A constant another kernel of the program already reads in one layout is offered in that layout only.
    """
    tile: TileOp = root.op
    if tile.op is None or tile.loop_body is None:
        return None
    loads = _loads(tile)
    eligible = [
        name
        for name in root.inputs
        if (node := match.graph.producer(name)) is not None and _source_shape(node) is not None and _undecided(tile, name)
    ]
    if not eligible:
        return None
    # The parent owns redirects even when a worker materializes the source-layout arm.
    match.output = {name: f"{name}__layout" for name in root.buffer_names()}
    first = eligible[0]
    taken = _taken_layout(match.graph, root, first)
    signature = tuple(load.index for load in loads[first])
    group = tuple(
        name
        for name in eligible
        if tuple(load.index for load in loads[name]) == signature and _taken_layout(match.graph, root, name) is None
    )
    folded = DeferredFork(lambda: replace(tile, layout_decided=(*tile.layout_decided, first)), {f"LAYOUT@{first}": "folded"})
    single = DeferredFork(lambda: _source_fragment(match, root, (first,)), {f"LAYOUT@{first}": "source"}, structural=True)
    if taken is not None:
        return [folded if taken == "folded" else single]
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
    return [option for option in options if all(pins.get(key, pins.get("LAYOUT", value)) == value for key, value in option.knobs.items())]
