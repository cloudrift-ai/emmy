"""The Python entry of the Kernel IR to PTX compiler (the ``emmy-ptx`` crate, through the ``emmy.emmy_ir`` extension).

:func:`compile_kernel` returns a kernel's PTX text, or raises :class:`NoRule` when the kernel holds something the
compiler has no rule for, and ``ValueError`` when the kernel is not what the compiler takes.
"""

from __future__ import annotations

from emmy.compiler.ir.kernel.interface import Interface
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.emmy_ir import NoRule, compile_ptx

#: The first compute capability whose kernels wait on the grid launched ahead of them (``griddepcontrol.wait``): the
#: compiler writes the wait from here on, so only such a kernel may be launched as a programmatic dependent launch.
GRID_WAIT_CAPABILITY = (9, 0)


def compile_kernel(
    kernel: KernelOp,
    interface: Interface,
    *,
    compute_capability: tuple[int, int],
    fuse: bool,
    flush_to_zero: bool,
    single_wave: bool = False,
) -> str:
    """The PTX text of ``kernel`` under ``interface`` for the card of ``compute_capability``, with the kernel's two
    whole-kernel precision decisions: multiply-add fusion and flush to zero. ``single_wave`` says the launch places at
    most one block on each multiprocessor of the card."""
    major, minor = compute_capability
    sm = major * 10 + minor
    return compile_ptx(kernel.name, kernel.body, interface, sm=sm, fuse=fuse, ftz=flush_to_zero, single_wave=single_wave)


__all__ = ["NoRule", "compile_kernel"]
