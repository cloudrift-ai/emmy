"""Stamp per-statement dtypes on every Load / Assign / Accum / Write.

Runs over a materialized ``KernelOp`` body, after ``010_materialize`` and before the analytical codegen passes
(shared carry, vectorize) see the IR: those passes read dtypes off the statements, and match one-operation values
over names, so the stamps come early and the conversions (``120_explicit_conversions``) last. The rules are
:func:`emmy.compiler.ir.types.stamp_types`. A ``Load`` against a locally-declared buffer (smem) is stamped with the
conversions.

A cross-thread combine is built before its state has a dtype, so it runs at f32 unless its combine program names
one. Once the stamps give its states an integer dtype, the combine and the shared slabs it folds through take that
dtype: through f32 an integer rounds past 2**24.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import F32, INTEGER_NAMES
from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import Smem, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.types import stamp_types, value_dtypes
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def _integer_combines(body: Body) -> Body:
    """``body`` with each f32 cross-thread combine over states of one integer dtype at that dtype, its slabs too."""
    dtypes = value_dtypes(body)
    slabs: dict[str, object] = {}

    def combines(stmts: Body) -> Body:
        out = []
        for s in stmts:
            if isinstance(s, (WarpShuffle, TreeHalve)) and s.dtype == F32:
                held = {dtypes.get(name) for name in s.state}
                (one,) = held if len(held) == 1 else (None,)
                if one is not None and one.name in INTEGER_NAMES:
                    if isinstance(s, TreeHalve):
                        slabs.update(dict.fromkeys(s.bufs, one))
                    s = s.replace(dtype=one)
            if s.nested():
                s = s.with_bodies(tuple(combines(b) for b in s.nested()))
            out.append(s)
        return Body(tuple(out))

    def retyped(stmts: Body) -> Body:
        out = []
        for s in stmts:
            if isinstance(s, Smem) and s.name in slabs:
                s = s.replace(dtype=slabs[s.name])
            if s.nested():
                s = s.with_bodies(tuple(retyped(b) for b in s.nested()))
            out.append(s)
        return Body(tuple(out))

    body = combines(body)
    return retyped(body) if slabs else body


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    body = _integer_combines(stamp_types(op.body, {n: t.dtype for n, t in {**op.inputs, **op.outputs}.items()}))
    if body == op.body:
        raise RuleSkipped("every Load/Assign/Write already stamped")
    return replace(op, body=body, knobs=dict(op.knobs))
