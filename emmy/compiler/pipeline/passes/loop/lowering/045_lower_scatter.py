"""Wrap a standalone ScatterOp as a LoopOp that stores into its destination in place.

The mirror of ``040_lower_gather``: the kernel loops over the values, loads each one's index and
stores it at that row of the output. It never reads or copies the destination — the output buffer
IS the destination's memory (``Tensor.alias``), so every cell the kernel does not store keeps the
destination's value. Two index forms, as the gather has:

- **same rank** (``torch.scatter``): ``idx`` and ``values`` have the destination's rank; cell
  ``(o_0, ..., o_{n-1})`` of ``values`` lands at ``(o_0, ..., idx[o], ..., o_{n-1})``.
- **rows** (``index_copy`` / ``index_put``): ``idx`` is 1-D and names rows along ``axis``; ``values``
  covers ``dest.shape[:axis] + idx.shape + dest.shape[axis+1:]`` and may broadcast into it.

A negative index counts from the end. An index equal to the extent names no row: the store is
skipped, so a program points a store past the end to drop it.

Nothing copies the destination first, so its other readers run before the scatter — the graph's
topological order puts a node stored over a buffer after that buffer's readers. A reader that also
needs the scatter's result, or a destination the program returns, wants both the old and the new
cells: such a destination gets its own copy here, and the copy is what gets written.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal, TernaryExpr, Var, placeholder
from emmy.compiler.ir.loop import Axis, Cond, Load, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource, ScatterOp
from emmy.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", ScatterOp)]


def _downstream(graph: Graph, node_id: str) -> set[str]:
    """Every node that reads ``node_id``'s outputs, directly or through others."""
    seen: set[str] = set()
    pending = [node_id]
    while pending:
        for user in graph.users(pending.pop()):
            if user not in seen:
                seen.add(user)
                pending.append(user)
    return seen


def rewrite(match: Match, root: Node, inp_dest: Node, inp_idx: Node, inp_values: Node, out: Tensor) -> Graph | None:
    dest_shape = tuple(inp_dest.output.shape)
    rank = len(dest_shape)
    axis = int(root.op.axis) % rank
    idx_shape = tuple(inp_idx.output.shape)
    value_shape = tuple(inp_values.output.shape)
    extent = dest_shape[axis].expr
    index = CastExpr("int", Var("idx"))
    index = TernaryExpr(index.lt(0), index + extent, index)

    if len(idx_shape) == rank:
        # torch.scatter: idx, values and the destination share a rank; the loop walks the values, and
        # the index replaces the one coordinate along ``axis``.
        axes = tuple(Axis(name=f"a{i}", extent=d) for i, d in enumerate(idx_shape))
        idx_index = value_index = tuple(Var(a.name) for a in axes)
        indexed = 1
    else:
        # Rows: the loop walks the cells the index fills, idx read at its own axes and the values
        # broadcast against that domain, trailing dims aligned.
        filled = (*dest_shape[:axis], *idx_shape, *dest_shape[axis + 1 :])
        axes = tuple(Axis(name=f"a{i}", extent=d) for i, d in enumerate(filled))
        idx_index = tuple(Var(axes[i].name) for i in range(axis, axis + len(idx_shape)))
        lead = len(filled) - len(value_shape)
        value_index = tuple(
            Literal(0, "int")
            if value_shape[i - lead].is_static and value_shape[i - lead].as_static() == 1 and filled[i] != value_shape[i - lead]
            else Var(axes[i].name)
            for i in range(lead, len(filled))
        )
        indexed = len(idx_shape)
    out_index = (*(Var(a.name) for a in axes[:axis]), index, *(Var(a.name) for a in axes[axis + indexed :]))

    inner: Body = (
        Load(name="idx", input=inp_idx.id, index=idx_index),
        Load(name="val", input=inp_values.id, index=value_index),
        Cond(cond=BinaryExpr("<", index, extent), body=(Write(output=f"loop_{root.id}", index=out_index, value="val"),)),
    )
    body: Body = inner
    for a in reversed(axes):
        body = (Loop(axis=a, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    for buf in (inp_idx, inp_values, inp_dest):
        if buf.id not in frag.nodes:
            frag.add_node(InputOp(), [], Tensor(buf.id, buf.output.shape, buf.output.dtype), node_id=buf.id)
    # The destination's memory is the output's. A reader of the destination runs before the scatter
    # (``Graph.topological_order``), unless it also reads the scatter's result or the destination is a
    # program output: then it wants both, so the destination is copied once and the copy is written.
    dest_id = inp_dest.id
    if dest_id in match.graph.outputs or match.graph.buffer_users(dest_id) & _downstream(match.graph, root.id):
        dest_id = frag.add_node(
            IndexMapOp(out_shape=dest_shape, sources=(IndexSource(input_idx=0, coord_map=tuple(placeholder(i) for i in range(rank))),)),
            [inp_dest.id],
            Tensor(f"{root.id}__dest", dest_shape, inp_dest.output.dtype),
            node_id=f"{root.id}__dest",
        )
    out_id = frag.add_node(
        kernel,
        [*kernel.inputs, dest_id],
        Tensor(out.name, dest_shape, out.dtype, alias=dest_id),
        node_id=f"loop_{root.id}",
    )
    frag.outputs = [out_id]
    return frag
