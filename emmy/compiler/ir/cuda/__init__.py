"""CUDA IR — device-level graph-op carrying a compiled kernel.

Submodules:
- :mod:`.ir` — ``CudaOp`` graph-op, produced by
  ``passes/lowering/cuda`` after a ``KernelOp`` is compiled to PTX text.

The public surface below re-exports the types so callers use
``from emmy.compiler.ir.cuda import CudaOp``.
"""

from emmy.compiler.ir.cuda.ir import CudaOp, TmaDescMeta

__all__ = ["CudaOp", "TmaDescMeta"]
