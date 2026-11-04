"""A cut materializes the components its consumer READS, and no others, and builds no piece nothing reads.

A reduce carries its accumulators together — six channels folded in one term — while a reader may
take a single one. ``Fold.lower`` narrows an operand edge to what its reader takes and drops the
rest, so a cut that minted one workspace per exposed component left the extra buffers written by
the piece and loaded by nobody. The CUDA backend's liveness plan refuses exactly that: "scratch
buffer … has no consuming launch (dead scratch)", which killed the candidate before it was ever
measured — a hundred consecutive placements of one Gated DeltaNet reduction died this way, so the
tuner had no valid schedule to rank against a greedy pick three orders of magnitude off.

A whole piece can go unread the same way. A chosen seam inside another chosen seam's piece, which computes it in
place, has no reader left; sibling fusion then spliced its piece into one something reads, and that piece wrote the
dead buffer.

The oracle is the backend's own precondition, asked of the fragment the cut builds: every buffer a
piece writes has a reader.
"""

from __future__ import annotations

import pytest

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
from emmy.compiler.pipeline.passes.tile._cut import cuttable_seams, join_pairs, realize
from tests.compiler.helpers import case_target_tile
from tests.compiler.terms import projection, reduction, slab

_ROW, _COL = Axis("m", Dim(8)), Axis("k", Dim(32))


def _kernel(store_input: bool) -> tuple[Graph, Node]:
    """A kernel whose one reducing branch carries TWO accumulators and whose consumer reads one.

    ``hi`` is folded beside ``lo`` and nothing downstream takes it — neither the consumer's cell
    nor the kernel's boundary store.
    """
    cell = slab("cell", "x", "m", "k")
    both = reduction(
        _COL,
        (cell,),
        (Assign(name="lo__v", op="copy", args=("cell",)), Assign(name="hi__v", op="multiply", args=("cell", "cell"))),
        ("lo", "hi"),
    )
    root = projection(
        (both, cell) if store_input else (both,),
        (Assign(name="squared", op="multiply", args=("lo", "lo")),),
        results=("squared",),
    )
    stores = (OutputSpec(Write(output="square", index=(Var("m"),), value="squared")),)
    outputs = [Tensor("square", (8,), dtype=F32)]
    if store_input:
        stores += (OutputSpec(Write(output="copied", index=(Var("m"), Var("k")), value="cell"), sweep=(_ROW, _COL)),)
        outputs.append(Tensor("copied", (8, 32), dtype=F32))
    tile = TileOp(
        op=root,
        place=Placement(free=(_ROW,)),
        axes=(_ROW, _COL),
        output_specs=stores,
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8, 32), dtype=F32), node_id="x")
    graph.add_node(tile, ["x"], outputs=outputs, node_id="square")
    graph.inputs, graph.outputs = ["x"], [tensor.name for tensor in outputs]
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


@pytest.mark.parametrize("store_input", [False, True])
def test_a_cut_writes_no_workspace_its_consumer_never_reads(store_input) -> None:
    graph, node = _kernel(store_input)
    seams = [seam for seam in cuttable_seams(node.op) if seam.node.axis is not None]
    assert seams, "the reducing branch must be offered as a cuttable seam for this shape to arise"

    fragment = realize(_Match(graph), node, (seams[0],))
    dead = _unread(fragment, node)
    assert not dead, f"the cut wrote workspaces no launch reads: {dead}"


def _unread(fragment: Graph, node: Node) -> list[str]:
    """The buffers a piece of ``fragment`` writes and no piece loads. The kernel's own outputs leave the fragment,
    under their own names or the temporary ones the splice redirects; everything else a piece writes is scratch,
    which the backend plans a live interval for and refuses without a reader."""
    pieces = [piece for piece in fragment.nodes.values() if isinstance(piece.op, TileOp)]
    read = set().union(*(_loaded(piece.op) for piece in pieces))
    written = {tensor.name for piece in pieces for tensor in piece.outputs}
    return sorted(written - read - set(node.buffer_names()) - set(fragment.outputs))


#: The width-4 Gated DeltaNet prologue of the float16 hybrid the vLLM GDN test serves. Its qkv projection is read at
#: three places, one contraction seam over all three; the RMS norm cones under the third place are seams of their own.
_GDN_CONV = "fused/gdn-conv-cut-inside-contraction.json"
_PROJECTION = "PLACE@map.3/map.2/inner"
_NORMED = "PLACE@map.3/map.2/inner.2/map"


@pytest.mark.parametrize(
    "spellings",
    [(_PROJECTION, _NORMED, "PLACE@map.3/map.2/inner.2/map.6/map"), (_PROJECTION, _NORMED, "PLACE@map.4/map")],
    ids=["nested", "joined"],
)
def test_a_seam_inside_a_cut_contraction_leaves_no_unread_piece(spellings) -> None:
    """The contraction's piece is built from the one occurrence that spans its whole space, and the norm cones chosen
    under another occurrence are then read by nothing: their pieces are dropped, not fused into the contraction's.
    Joined with an output region, the norm's workspace piece is dropped and the region's piece stands alone."""
    tile = case_target_tile(_GDN_CONV)
    seams = cuttable_seams(tile)
    chosen = tuple(next(seam for seam in seams if spelling in (seam.spelling, *seam.aliases)) for spelling in spellings)
    joins = tuple(pair for pair in join_pairs(seams) if pair[0] is chosen[1] and pair[1] is chosen[2])
    assert bool(joins) == (spellings[-1] == "PLACE@map.4/map"), "the norm and the region must be a joined pair"
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    outputs = list(tile.outputs.values())
    graph.add_node(tile, list(tile.inputs), outputs=outputs, node_id=outputs[0].name)
    graph.inputs, graph.outputs = list(tile.inputs), [tensor.name for tensor in outputs]
    node = graph.nodes[outputs[0].name]

    fragment = realize(_Match(graph), node, chosen, joins=joins)
    dead = _unread(fragment, node)
    assert not dead, f"the cut wrote workspaces no launch reads: {dead}"
