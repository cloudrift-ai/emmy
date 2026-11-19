"""The split of a Fold's axis into blocks: ``k = length * block + k``.

One operation on the term. :func:`split_axis` returns the fold over one block, with the block coordinate free in
it, and :func:`merged` the fold over the blocks that combines what each block folded. Where the two levels run is
the caller's: the split across CTAs runs the inner one as a kernel of its own and the outer one over a workspace.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.address import clamp_to
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var, apply
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.pure.twist import Twist
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Let, Load, Select, SelectBranch, Statement


def split_axis(fold: Fold, axis: Axis, blocks: int) -> tuple[Axis, Axis, Fold]:
    """``fold`` over one of ``blocks`` equal blocks of ``axis``: ``(block axis, in-block axis, the fold)``.

    The in-block axis keeps the name the fold binds, its parent recorded as a partition window, so nothing splits
    it again. Every read below the fold takes the position ``length * block + k``; a lift is closed again, so the
    block coordinate joins as a trailing param. A symbolic axis has blocks of ``ceil(K / blocks)`` positions: its
    loads clamp to the last position and the fold folds nothing past the end (:func:`_masked`). A lift reads the
    position unclamped: a mask over it is what a staged tier reads its bounds from."""
    if axis.extent.is_static:
        length = Literal(axis.extent.as_static() // blocks, "int")
    else:
        length = BinaryExpression("/", BinaryExpression("+", axis.extent.expression, Literal(blocks - 1, "int")), Literal(blocks, "int"))
    block = Axis(name=f"_{axis.name}_ks", extent=Dim(blocks))
    inner = replace(axis, extent=Dim(length), window=Window(parent=axis.source_axis or axis, partition=True))
    position = BinaryExpression("+", BinaryExpression("*", Var(block.name), length), Var(axis.name))
    loads = Sigma({axis.name: position if axis.extent.is_static else clamp_to(position, axis.extent.expression)})

    def below(edge):
        if isinstance(edge, Load):
            return edge.replace(index=tuple(loads.apply(e) for e in edge.index))
        operands = tuple(below(e) if axis.name in e.free_axes else e for e in edge.operands)
        body = Body(tuple(s.substitute(loads) for s in edge.lift.body))
        return edge.replace(operands=operands, lift=Lambda.closing(edge.lift.params, body, edge.lift.results))

    body = Body(tuple(stmt.substitute(Sigma({axis.name: position})) for stmt in fold.lift.body))
    lift = Lambda.closing(fold.lift.params, body, fold.lift.results)
    return (
        block,
        inner,
        _masked(
            fold.replace(operands=tuple(below(edge) for edge in fold.operands), lift=lift, range=_in_block(fold, block, length)),
            axis,
            position,
        ),
    )


def _in_block(fold: Fold, block: Axis, length) -> tuple | None:
    """The range ``fold`` walks, stated of the position inside one block: a range bounds the position on the whole
    axis, which a block reads as ``length * block + k``, so both bounds move back by the block's start and stay
    inside the block."""
    if fold.range is None:
        return None
    start, end = fold.range
    origin = BinaryExpression("*", Var(block.name), length)
    first = start if start == Literal(0, "int") else apply("maximum", Literal(0, "int"), BinaryExpression("-", start, origin))
    return first, apply("minimum", length, BinaryExpression("-", end, origin))


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


def _masked(fold: Fold, k_axis: Axis, position) -> Fold:
    """``fold`` folding nothing at any ``position`` past a symbolic stream's end, which the last slice overhangs; a
    static stream's slices tile it exactly. A monoid folds its seed there. A twisted carrier's score takes the
    pivot's identity, added through the `where` selects a program's own mask spells — the coordinate mask the chunk tier
    lands — so the carrier past the end weighs its channels by nothing, and an all-overhang slice folds a finite
    state the combine then scales away."""
    if k_axis.extent.is_static:
        return fold
    end = k_axis.extent.expression
    stmts: list[Statement] = []
    if fold.twist is None:
        inside = BinaryExpression("<", position, end)
        results = []
        for name, seed in zip(fold.lift.results, fold.init, strict=True):
            stmts += [
                Let(name=f"{name}__seed", value=seed),
                Select(name=f"{name}__in", branches=(SelectBranch(name, inside), SelectBranch(f"{name}__seed", Literal(True, "bool")))),
            ]
            results.append(f"{name}__in")
    else:
        score, *channels = fold.lift.results
        stmts += [
            Let(name=f"{score}__key", value=position),
            Let(name=f"{score}__end", value=end),
            Let(name=f"{score}__keep", value=Literal(0.0, "float")),
            Let(name=f"{score}__fill", value=Literal(float(fold.init[0]), "float")),
            Assign(f"{score}__inside", apply("less", f"{score}__key", f"{score}__end")),
            Assign(f"{score}__mask", apply("where", f"{score}__inside", f"{score}__keep", f"{score}__fill")),
            Assign(f"{score}__in", apply("add", score, f"{score}__mask")),
        ]
        results = [f"{score}__in", *channels]
    body = Body((*fold.lift.body, *stmts))
    return fold.replace(lift=Lambda.closing(fold.lift.params, body, tuple(results)))
