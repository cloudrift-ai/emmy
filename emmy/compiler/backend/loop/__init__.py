"""Loop backend: pure-Python interpreter for a fused ``Graph[LoopOp]``.

Evaluates the post-fusion graph via numpy whole-tensor ops. Used as a
reference/debugging backend — sits between the ``NumpyBackend``
(evaluates the pre-fusion Graph) and the ``CudaBackend`` (lowers to
``Graph[CudaOp]`` and runs its PTX) so bugs can be attributed to
fusion vs codegen.
"""

from emmy.compiler.backend.loop.backend import LoopBackend

__all__ = ["LoopBackend"]
