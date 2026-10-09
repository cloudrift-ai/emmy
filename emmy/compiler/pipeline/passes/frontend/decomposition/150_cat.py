"""Lower CatOp([t_0, …, t_{n-1}], dim) → IndexMapOp.

Tracer convention: CatOp.inputs = [t_0, …, t_{n-1}, dim_const], one tensor or more: Qwen rotary's
``cat(neg, slice_1, dim=-1)`` has two, a windowed encoder's attention one per window.

After decomposition: IndexMapOp.inputs = [t_0, …, t_{n-1}]; the dim is baked into the source selects and
each source's coord_map offset.

**In-bounds clamping**: The cat-source Selects gate which value is *used*
at each output coordinate, but downstream lifting / fusion turns each
source into an unconditional ``Load``. With the naive coord_map A/B
would read out-of-range indices on the half where the other source is
selected (e.g. rotary's ``cat([-x[..., half:], x[..., :half]], -1)``
issues ``Load(x, dim - half)`` for ``dim < half`` → negative offset).
Most allocator layouts mask the OOB Load behind a same-page read, but
swap-bencher allocations land tight enough to segfault. Each source's
cat-dim coord is wrapped in a ``TernaryExpr`` that clamps to its valid
range when out of domain — the loaded value is wrong but the Select
chain never picks it.
"""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.expr import Literal, TernaryExpr, placeholder
from emmy.compiler.ir.frontend.ir import CatOp
from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", CatOp)]


def rewrite(match: Match, root: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    *parts, inp_dim = (graph.producer(ref) for ref in root.inputs)
    out_shape = tuple(out.shape)
    ndim = len(out_shape)

    if not parts or not (isinstance(inp_dim.op, ConstantOp) and inp_dim.op.value is not None):
        raise RuleSkipped("cat dim must be a ConstantOp with a value")
    dim = int(inp_dim.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim

    # Every source but the last ends at a split point; the last one takes the rest.
    ends = []
    for part in parts[:-1]:
        extent = part.output.shape[norm_dim]
        if not extent.is_static:
            raise RuleSkipped(f"cat split point {extent!r} must be a static int")
        ends.append((ends[-1] if ends else 0) + extent.as_static())

    frag = open_fragment(graph, parts)

    # Source i is valid for start_i <= dim < end_i and is selected where dim < end_i (the
    # first matching select wins). When the post-fusion Load fires on an off-domain side,
    # the ternaries collapse the cat-dim coord into the source's valid range so the read
    # stays in-bounds (the Select downstream discards the value).
    cat_var = placeholder(norm_dim)
    zero = Literal(0, "int")
    sources = []
    for i in range(len(parts)):
        start = ends[i - 1] if i else 0
        coord = cat_var - Literal(start, "int") if start else cat_var
        select = cat_var.lt(Literal(ends[i], "int")) if i < len(ends) else None
        if select is not None:
            coord = TernaryExpr(cond=select, if_true=coord, if_false=zero)
        if start:
            coord = TernaryExpr(cond=cat_var.lt(Literal(start, "int")), if_true=zero, if_false=coord)
        coord_map = tuple(coord if d == norm_dim else placeholder(d) for d in range(ndim))
        sources.append(IndexSource(input_idx=i, coord_map=coord_map, select=select))

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=tuple(sources)),
        inputs=parts,
        output=Tensor(out.name, out_shape, out.dtype),
    )

    frag.outputs = [new_id]
    return frag
