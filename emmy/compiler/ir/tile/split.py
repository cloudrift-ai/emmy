"""The split of a Fold's axis into blocks: block ``b`` walks the positions ``length * b .. length * (b + 1)``.

One operation on the term. :func:`split_axis` returns the fold over one block, with the block coordinate free in
it, and :func:`merged` the fold over the blocks that combines what each block folded. Where the two levels run is
the caller's: the split across CTAs runs the inner one as a kernel of its own and the outer one over a workspace.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var, apply
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.pure.twist import Twist
from emmy.compiler.ir.stmt import Body, Load


def split_axis(fold: Fold, axis: Axis, blocks: int) -> tuple[Axis, Axis, Fold]:
    """``fold`` over one of ``blocks`` equal blocks of ``axis``: ``(block axis, in-block axis, the fold)``.

    The fold keeps its body and its reads: it walks the range of one block, ``length * block .. length * (block +
    1)``, inside the range it walked before. A symbolic axis has blocks of ``ceil(K / blocks)`` positions and its
    last block ends where the stream does. The in-block axis is the axis the fold binds, its parent recorded as a
    partition window, so nothing splits it again."""
    if axis.extent.is_static:
        length = Literal(axis.extent.as_static() // blocks, "int")
    else:
        length = BinaryExpression("/", BinaryExpression("+", axis.extent.expression, Literal(blocks - 1, "int")), Literal(blocks, "int"))
    block = Axis(name=f"_{axis.name}_ks", extent=Dim(blocks))
    inner = replace(axis, window=Window(parent=axis.source_axis or axis, partition=True))
    first = BinaryExpression("*", Var(block.name), length)
    last = BinaryExpression("+", first, length)
    zero = Literal(0, "int")
    start, end = fold.range if fold.range is not None else (zero, None if axis.extent.is_static else axis.extent.expression)
    walked = (first if start == zero else apply("maximum", start, first), last if end is None else apply("minimum", last, end))
    return block, inner, fold.replace(range=walked)


def merged(block: Axis, fold: Fold, loads: tuple[Load, ...]) -> Fold:
    """The fold over ``block`` that combines what each block of a split ``fold`` folded, read back through
    ``loads``: the same monoid over finished states, so no contribution is left for a twist to be applied to."""
    values = tuple(name for load in loads for name in load.defines())
    return Fold(
        operands=tuple(Fold.slab(load) for load in loads),
        lift=Lambda(params=(block.name, *values), body=Body(), results=values),
        init=fold.init,
        base=fold.base,
        twist=None if fold.twist is None else Twist(recipe=fold.twist.recipe, channels=()),
    )
