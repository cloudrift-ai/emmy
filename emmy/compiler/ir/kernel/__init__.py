"""Kernel IR — fully-scheduled kernel form, lowered directly to CUDA source.

- :mod:`.ir` — dataclass definitions: the ``KernelOp`` wrapper plus the hardware
  primitives (launch geometry, shared memory, barriers, the transports into shared
  memory, the cross-thread combines, the tensor-core fragment nodes). Shared leaves and
  structural types (``Load``, ``Assign``, ``Loop``, ``StridedLoop``, …) come from
  ``ir.stmt``.
- :mod:`.render` — ``render_kernelop`` emitting CUDA source.
"""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import (
    Accum,
    Assign,
    BinaryExpr,
    Builtin,
    CastExpr,
    Cond,
    ElementwiseImpl,
    Expression,
    FuncCallExpr,
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
    TernaryExpr,
    Tile,
    TreeHalve,
    Var,
    WarpShuffle,
    Write,
)

__all__ = [
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
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
