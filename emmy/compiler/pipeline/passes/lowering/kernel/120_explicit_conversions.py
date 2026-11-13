"""Type the lowered body in full: every operation at one dtype, every conversion a definition of its own.

The kernel pass that types the body (``125_spell_composites`` follows it and keeps the types; ``130_approximate_math``
calls the same rules for a half exponential it moves to f32). Until here a
conversion is implied: an ``Assign`` over operands of another dtype, a ``Write`` of an f32 value into an f16 buffer.
The rules that say where each operation computes (natively in f16 where the target has the operation, else in f32
with the result converted once) are :func:`emmy.compiler.ir.types.explicit_conversions`; after it a code generator
reads the body and decides nothing, and ``check_types`` holds. A ``FragmentLoad`` takes its two element types here
too: its source buffer's, and the one its fragment was declared at. So does a ``RegStore`` epilogue, by the same rules
over the f32 elements of its accumulators (:func:`emmy.compiler.ir.types.epilogue_conversions`).

It runs after the passes that match patterns because they match one-operation values over the names the lowering
wrote: a conversion between a load and its reader would hide the pattern from the vector passes, the shared carry
and the load reuse. It decides nothing and answers to no knob: the kernel computes the same values.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.interface import buffer_dtypes
from emmy.compiler.ir.kernel.ir import FragmentLoad, RegFragment, RegStore
from emmy.compiler.ir.stmt import Statement
from emmy.compiler.ir.types import epilogue_conversions, explicit_conversions
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.target import native_f16_ops

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    buffers = buffer_dtypes(op)
    fragments = {s.name: s.dtype for s in op.body.iter_of_type(RegFragment)}

    def typed(stmt: Statement) -> Statement:
        # A fragment gathered element by element converts each one: it says both types, as a ``Load`` says its own.
        if isinstance(stmt, FragmentLoad):
            return stmt.replace(src_dtype=buffers[stmt.src_buffer], dtype=fragments[stmt.frag])
        if isinstance(stmt, RegStore):
            # A store holds the stores grouped with it (``run``), each with an epilogue of its own.
            run = tuple(typed(cell) for cell in stmt.run)
            if stmt.epilogue is None:
                return stmt.replace(run=run)
            return stmt.replace(epilogue=epilogue_conversions(stmt.epilogue, stmt.epilogue_params, buffers, native_f16_ops()), run=run)
        return stmt

    body = explicit_conversions(op.body, buffers, native_f16_ops()).map(typed)
    if body == op.body:
        raise RuleSkipped("every conversion already explicit")
    return replace(op, body=body, knobs=dict(op.knobs))
