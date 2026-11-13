"""Give every element access one flat index: the address is in the kernel, not in whatever prints it.

A ``Load``, a ``Write`` and a ``CpAsyncCopy`` are lowered with one index per dimension of their buffer, which the
passes before this one read (the vector passes prove alignment over them, the staging passes tile them). Here each
becomes one row-major element index into its buffer, symbolic sizes included (``address.flat_index``), with each
coordinate widened to 64 bits where the buffer can pass the 32-bit range. So does a ``FlatIndex`` a staging pass left
in a value. The body's optimization then simplifies the index under the ranges in scope, which the launch's ranges of
the decoded grid are part of, and a code generator reads an address it only has to scale.

What its dimensions define keeps them:

- an access of a paged buffer, whose index along the paged axis picks the page before the rest addresses inside it;
- a ``Write`` under a Volta slab layout, a function of the logical row and column;
- the shared-memory side of a ``CpAsyncCopy`` that carries a lane offset, whose swizzle is split over the tile's row
  and column;
- the fragment statements (``LdmatrixLoad``, ``BlockScaleLoad``, ``FragmentApply``'s gathered operand, ``RegStore``
  and the loads of its epilogue) and ``TmaLoad``, until they say what they are.

It runs here and not with the kernel passes because which buffers are paged is a fact of the graph, which the cached
kernel does not know, and after ``007_decode_grid`` so that the simplification sees the launch. It follows the
conversion passes by standing after every kernel pass, and needs nothing of them: an index is integer arithmetic
that no conversion touches. It decides nothing: every access reads and writes the element it did.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import spans_one_page
from emmy.compiler.graph import Node
from emmy.compiler.ir.address import flat_accesses, flat_index
from emmy.compiler.ir.expression import Expression, FlatIndex
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import TWO_DIMENSIONAL_LAYOUTS, CpAsyncCopy, Smem
from emmy.compiler.ir.stmt import Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.cuda._helpers import paged_buffers

PATTERN = [Pattern("root", KernelOp)]


def rewrite(match: Match, root: Node) -> KernelOp:
    kernel: KernelOp = root.op
    shapes = {name: tensor.shape for name, tensor in (*kernel.inputs.items(), *kernel.outputs.items())}
    shapes.update((stmt.name, stmt.extents) for stmt in kernel.body.iter_of_type(Smem))
    paged = {name for name, axis, page in paged_buffers(kernel, match.graph) if not spans_one_page(shapes[name], axis, page)}

    def keeps(stmt: Statement) -> bool:
        if isinstance(stmt, Write):
            return stmt.output in paged or stmt.swizzle in TWO_DIMENSIONAL_LAYOUTS
        return stmt.input in paged

    def resolved(expression: Expression) -> Expression:
        return flat_index(expression.index, shapes.get(expression.buffer)) if isinstance(expression, FlatIndex) else expression

    def copy(stmt: Statement) -> Statement:
        if isinstance(stmt, CpAsyncCopy):
            stmt = stmt.replace(src_index=(flat_index(stmt.src_index, shapes.get(stmt.src)),))
            if stmt.lane_index is None:
                stmt = stmt.replace(smem_index=(flat_index(stmt.smem_index, shapes.get(stmt.smem)),))
        return stmt.map_expressions(lambda expression: expression.rebuild(resolved))

    body = flat_accesses(kernel.body, shapes, keeps).map(copy)
    if body == kernel.body:
        raise RuleSkipped("every access already holds one index")
    return replace(kernel, body=body)
