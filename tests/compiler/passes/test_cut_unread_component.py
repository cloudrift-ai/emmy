"""A cut materializes the components its consumer READS, and no others.

A reduce carries its accumulators together — six channels folded in one term — while a reader may
take a single one. ``Fold.lower`` narrows an operand edge to what its reader takes and drops the
rest, so a cut that minted one workspace per exposed component left the extra buffers written by
the piece and loaded by nobody. The CUDA backend's liveness plan refuses exactly that: "scratch
buffer … has no consuming launch (dead scratch)", which killed the candidate before it was ever
measured — a hundred consecutive placements of one Gated DeltaNet reduction died this way, so the
tuner had no valid schedule to rank against a greedy pick three orders of magnitude off.

The oracle is the backend's own precondition, asked of the fragment the cut builds: every buffer a
piece writes has a reader.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.stmt import Assign, Load, Write
from emmy.compiler.ir.stmt.leaves import OutputSpec
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, realize
from tests.compiler.terms import projection, reduction, slab

_ROW, _COL = Axis("m", Dim(8)), Axis("k", Dim(32))


def _kernel() -> tuple[Graph, Node]:
    """A kernel whose one reducing branch carries TWO accumulators and whose consumer reads one.

    ``hi`` is folded beside ``lo`` and nothing downstream takes it — neither the consumer's cell
    nor the kernel's boundary store.
    """
    both = reduction(
        _COL,
        (slab("cell", "x", "m", "k"),),
        (Assign(name="lo__v", op="copy", args=("cell",)), Assign(name="hi__v", op="multiply", args=("cell", "cell"))),
        ("lo", "hi"),
    )
    root = projection((both,), (Assign(name="squared", op="multiply", args=("lo", "lo")),), results=("squared",))
    tile = TileOp(
        op=root,
        place=Placement(free=(_ROW,)),
        axes=(_ROW, _COL),
        output_specs=(OutputSpec(Write(output="square", index=(Var("m"),), value="squared")),),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8, 32), dtype=F32), node_id="x")
    graph.add_node(tile, ["x"], outputs=[Tensor("square", (8,), dtype=F32)], node_id="square")
    graph.inputs, graph.outputs = ["x"], ["square"]
    return graph, graph.nodes["square"]


class _Match:
    """What ``realize`` asks a match for: the graph it looks input buffers up in, and the splice
    identities each piece registers its renamed output ports under."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.output: dict[str, str] = {}


def _loaded(tile: TileOp) -> set[str]:
    """The buffers this piece's LAUNCH reads — its lowered body's loads.

    Not the graph node's input list: the consumer is handed every workspace of the cut as an input
    whether or not its term reads one, and a slab no reader takes is dropped at lowering. What
    reaches the backend's liveness plan is the launch's arguments, which is this.
    """
    names: set[str] = set()
    pending = list(tile.op.lower(frozenset(axis.name for axis in tile.place.free), tile.output_specs, tile.axes))
    while pending:
        stmt = pending.pop()
        if isinstance(stmt, Load):
            names.add(stmt.input)
        pending.extend(inner for body in stmt.nested() for inner in body)
    return names


def test_a_cut_writes_no_workspace_its_consumer_never_reads() -> None:
    graph, node = _kernel()
    seams = [seam for seam in cuttable_seams(node.op) if seam.node.axis is not None]
    assert seams, "the reducing branch must be offered as a cuttable seam for this shape to arise"

    fragment = realize(_Match(graph), node, (seams[0],))
    pieces = [piece for piece in fragment.nodes.values() if isinstance(piece.op, TileOp)]
    read = set().union(*(_loaded(piece.op) for piece in pieces))
    written = {tensor.name for piece in pieces for tensor in piece.outputs}
    # The kernel's own outputs leave the fragment; everything else a piece writes is scratch, which
    # the backend plans a live interval for and refuses without a reader.
    dead = sorted(written - read - set(node.buffer_names()))
    assert not dead, f"the cut wrote workspaces no launch reads: {dead}"
