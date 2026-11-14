"""Pick the approximate form of each instruction class a precision knob turns on.

An approximate instruction is an operation of the kernel body, named after the exact one: ``exp_fast``,
``divide_fast``, ``reciprocal_fast``, ``sqrt_fast``, ``log_fast``, ``sin_fast``, ``cos_fast``, ``tanh_fast``,
``pow_fast``. No compiler flag picks one. This pass is the one place that does, each class by its own decision:

- ``FAST_EXP``: the exponential.
- ``FAST_DIV``: division, and the reciprocal, which is a division of one.
- ``FAST_SQRT``: the square root. The reciprocal square root has one instruction, the approximate one, in every regime.
- ``FAST_MATH`` with ``FAST_EXP``: the logarithm, sine, cosine, hyperbolic tangent and power, which no knob names.
  The hyperbolic tangent has its instruction from sm_75 on.

Each approximate instruction is an f32 instruction, so an operation is picked where it computes in f32 and stays as
it is where it computes natively in f16 or over integers. That is why this runs last: ``120_explicit_conversions`` has
said where every operation computes and ``125_spell_composites`` has written every composite out, so the exponential
and the division of a ``sigmoid`` are picked like any other. Two operations are picked at a half too, and the type
rules then compute them in f32 and round back: the exponential, by its knob's own rule, and the hyperbolic tangent,
because the native one (CUDA's ``htanh``) is the f32 function on the converted value, which nvcc's fast library
reached.

An operation sits in four places:

- An ``Assign`` of the body, typed in full: it computes at the statement's dtype.
- The merge program of a cooperative combine (``TreeHalve``, ``WarpShuffle``), which runs at the statement's dtype.
- A ``FragmentApply``'s operation and its unary tail. The fragment algebra is f32.
- An ``Assign`` of a ``RegStore`` epilogue, typed in full like one of the body.

A combine, a ``FragmentApply`` and an epilogue hold one operation where the body holds a definition, so a composite
stays whole there: a ``reciprocal`` becomes ``reciprocal_fast``, and a ``sigmoid`` becomes ``sigmoid_fast`` when the
exponential and the division are both approximate (it keeps both exact otherwise).

``FAST_EXP`` is recorded on the kernel, as it always was: the record makes the pass idempotent.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Node
from emmy.compiler.ir.elementwise import ElementwiseImpl, is_elementwise
from emmy.compiler.ir.expression import Expression, FunctionCallExpression, Literal, apply
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.interface import buffer_dtypes
from emmy.compiler.ir.kernel.ir import FragmentApply, RegStore, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Assign, Statement
from emmy.compiler.ir.types import epilogue_conversions, explicit_conversions
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._approximate import approximate_forms
from emmy.compiler.pipeline.search.space import FAST_EXP, precision_pin
from emmy.compiler.target import native_f16_ops

PATTERN = [Pattern("root", KernelOp)]

#: The operations picked at a half too (the module doc).
_AT_A_HALF = ("exp", "tanh")


def rewrite(match: Match, root: Node, ctx: Context) -> KernelOp:
    op: KernelOp = root.op
    if FAST_EXP.name in op.knobs:
        raise RuleSkipped("the approximate forms are already picked (FAST_EXP is recorded)")
    picks = approximate_forms(ctx)
    buffers = buffer_dtypes(op)
    halves: list[Assign] = []  # the operations picked at a half: the type rules place them
    body = op.body.map(lambda stmt: _picked(stmt, picks, buffers, halves)) if picks else op.body
    if halves:
        body = explicit_conversions(body, buffers, native_f16_ops())
    return replace(op, body=body, knobs={**op.knobs, FAST_EXP.name: precision_pin(FAST_EXP)})


def _picked(stmt: Statement, picks: dict[str, str], buffers: dict, halves: list[Assign]) -> Statement:
    if isinstance(stmt, Assign):
        if (stmt.dtype or F32) == F32:
            return stmt.replace(value=_pick(stmt.value, picks))
        picked = stmt.replace(value=_pick(stmt.value, {name: picks[name] for name in _AT_A_HALF if name in picks}))
        if picked != stmt:
            halves.append(picked)
        return picked
    if isinstance(stmt, (TreeHalve, WarpShuffle)) and stmt.dtype == F32:
        return stmt.replace(combine_states=tuple(merge.replace(value=_pick(merge.value, picks)) for merge in stmt.combine_states))
    if isinstance(stmt, FragmentApply):
        return stmt.replace(op=ElementwiseImpl(picks.get(stmt.op.name, stmt.op.name)))
    if isinstance(stmt, RegStore):
        # A store holds the stores grouped with it (``run``), each with an epilogue of its own.
        run = tuple(_picked(cell, picks, buffers, halves) for cell in stmt.run)
        if stmt.epilogue is None:
            return stmt.replace(run=run)
        placed: list[Assign] = []
        epilogue = stmt.epilogue.map(lambda inner: _picked(inner, picks, buffers, placed))
        if placed:
            epilogue = epilogue_conversions(epilogue, stmt.epilogue_params, buffers, native_f16_ops())
        return stmt.replace(epilogue=epilogue, run=run)
    return stmt


def _pick(node: Expression, picks: dict[str, str]) -> Expression:
    """``node`` with each operation ``picks`` names as its approximate form. Only operations are entered: a
    condition, an index and a conversion stay as they are, and so does a call of an intrinsic the target spells
    itself (a Volta swizzle), which is no elementwise operation."""
    if isinstance(node, FunctionCallExpression) and not is_elementwise(node.name):
        return node
    op = node.operation
    if op is None:
        return node
    operands = [_pick(operand, picks) for operand in node.operands]
    if op.name not in picks:
        return node.with_children(operands)
    if op.name == "divide" and isinstance(operands[0], Literal) and operands[0].value == 1:
        return apply("reciprocal_fast", operands[1])  # the instruction nvcc picks for ``1.0f / x``
    return apply(picks[op.name], *operands)
