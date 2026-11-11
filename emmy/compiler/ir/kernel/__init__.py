"""Kernel IR — fully-scheduled kernel form, lowered directly to CUDA source.

- :mod:`.ir` — dataclass definitions: the ``KernelOp`` wrapper plus the hardware
  primitives (launch geometry, shared memory, barriers, the transports into shared
  memory, the cross-thread combines, the tensor-core fragment nodes). Shared leaves and
  structural types (``Load``, ``Assign``, ``Loop``, ``StridedLoop``, …) come from
  ``ir.stmt``.
- :mod:`.render` — ``render_kernelop`` emitting CUDA source.
"""

import emmy.emmy_ir as _ir
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import (
    Accum,
    Assign,
    BinaryExpression,
    Builtin,
    CastExpression,
    Cond,
    ElementwiseImpl,
    Expression,
    FunctionCallExpression,
    KernelOp,
    Literal,
    Load,
    Loop,
    Select,
    SelectBranch,
    Smem,
    Statement,
    StridedLoop,
    Sync,
    TernaryExpression,
    Tile,
    TreeHalve,
    Var,
    WarpShuffle,
    Write,
)

#: The Kernel IR optimizer's exact rewrites over a kernel body (the crate's ``optimize`` module).
optimize_kernel_body = _ir.optimize_kernel_body

__all__ = [
    "optimize_kernel_body",
    "Var",
    "Literal",
    "BinaryExpression",
    "Builtin",
    "FunctionCallExpression",
    "TernaryExpression",
    "CastExpression",
    "Expression",
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "WarpShuffle",
    "StridedLoop",
    "Statement",
    "KernelOp",
    "Axis",
    "ElementwiseImpl",
]
