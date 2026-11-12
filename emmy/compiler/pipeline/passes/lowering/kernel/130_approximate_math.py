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
- A ``RegStore`` epilogue, typed only when it is printed. Each of its definitions is typed here by the same rules,
  to read where its operation will compute.

The last three hold one operation where the body holds a definition, so a composite stays whole there: a
``reciprocal`` becomes ``reciprocal_fast``, and a ``sigmoid`` becomes ``sigmoid_fast`` when the exponential and the
division are both approximate (it keeps both exact otherwise).

``FAST_EXP`` is recorded on the kernel, as it always was: the record makes the pass idempotent.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.dtype import F32, I32, INTEGER_NAMES
from emmy.compiler.graph import Node
from emmy.compiler.ir.elementwise import ElementwiseImpl, is_elementwise
from emmy.compiler.ir.expression import Expression, FunctionCallExpression, Literal, apply
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.interface import buffer_dtypes
from emmy.compiler.ir.kernel.ir import FragmentApply, RegStore, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Assign, Body, Load, Select, Statement
from emmy.compiler.ir.types import explicit_conversions
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import FAST_DIV, FAST_EXP, FAST_MATH, FAST_SQRT, precision_pin
from emmy.compiler.target import native_f16_ops

PATTERN = [Pattern("root", KernelOp)]

#: The operations picked at a half too (the module doc).
_AT_A_HALF = ("exp", "tanh")


def rewrite(match: Match, root: Node, ctx: Context) -> KernelOp:
    op: KernelOp = root.op
    if FAST_EXP.name in op.knobs:
        raise RuleSkipped("the approximate forms are already picked (FAST_EXP is recorded)")
    picks = _picks(ctx)
    buffers = buffer_dtypes(op)
    halves: list[Assign] = []  # the operations picked at a half: the type rules place them
    body = op.body.map(lambda stmt: _picked(stmt, picks, buffers, halves)) if picks else op.body
    if halves:
        body = explicit_conversions(body, buffers, native_f16_ops())
    return replace(op, body=body, knobs={**op.knobs, FAST_EXP.name: precision_pin(FAST_EXP)})


def _picks(ctx: Context) -> dict[str, str]:
    """The exact operation to its approximate form, for each class the live precision decisions turn on and the
    target has an instruction for."""
    picks: dict[str, str] = {}
    if precision_pin(FAST_EXP):
        picks["exp"] = "exp_fast"
    if precision_pin(FAST_DIV):
        picks |= {"divide": "divide_fast", "reciprocal": "reciprocal_fast"}
    if precision_pin(FAST_SQRT):
        picks["sqrt"] = "sqrt_fast"
    if precision_pin(FAST_MATH) and precision_pin(FAST_EXP):
        picks |= {name: f"{name}_fast" for name in ("log", "sin", "cos", "pow", *(("tanh",) if ctx.has_tanh_approx else ()))}
    if "exp" in picks and "divide" in picks:
        picks["sigmoid"] = "sigmoid_fast"
    return picks


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
        op, *post = (ElementwiseImpl(picks.get(held.name, held.name)) for held in (stmt.op, *stmt.post))
        return stmt.replace(op=op, post=tuple(post))
    if isinstance(stmt, RegStore):
        # A store holds the stores grouped with it (``run``), each with an epilogue of its own.
        run = tuple(_picked(cell, picks, buffers, halves) for cell in stmt.run)
        return stmt.replace(epilogue=None if stmt.epilogue is None else _epilogue(stmt, picks, buffers), run=run)
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


def _epilogue(store: RegStore, picks: dict[str, str], buffers: dict) -> Body:
    """``store``'s epilogue with its approximate forms picked. The program is typed when it is printed, one definition
    at a time, over the types its printer reads each name at: a register element and a floating load are f32, an
    integer load its own dtype, a binding its declared dtype, a definition the dtype the rules give it. Each
    definition is typed the same way here, and the operations that come out computing in f32 are the ones picked."""
    types = dict.fromkeys(store.epilogue_params, F32)
    out: list[Statement] = []
    for stmt in store.epilogue:
        if isinstance(stmt, Load):
            dtype = stmt.dtype or buffers.get(stmt.input) or F32
            types.update((name, dtype if dtype.name in INTEGER_NAMES else F32) for name in stmt.names)
        elif isinstance(stmt, Assign):
            typed = explicit_conversions(Body([stmt]), {}, native_f16_ops(), {name: types[name] for name in stmt.deps() if name in types})
            in_f32 = {name for s in typed if isinstance(s, Assign) and s.dtype == F32 for name in _operations(s.value)}
            placed = {name: fast for name, fast in picks.items() if name in in_f32 or name in _AT_A_HALF}
            stmt = stmt.replace(value=_pick(stmt.value, placed))
            types[stmt.name] = next(s.dtype for s in typed if isinstance(s, Assign) and s.name == stmt.name)
        else:  # a select is f32; a binding its declared dtype, an index when it declares none
            types.update(dict.fromkeys(stmt.defines(), F32 if isinstance(stmt, Select) else getattr(stmt, "dtype", None) or I32))
        out.append(stmt)
    return Body(out)


def _operations(node: Expression) -> set[str]:
    """The elementwise operations ``node`` applies, by name."""
    if (isinstance(node, FunctionCallExpression) and not is_elementwise(node.name)) or node.operation is None:
        return set()
    return {node.operation.name}.union(*(_operations(operand) for operand in node.operands))
