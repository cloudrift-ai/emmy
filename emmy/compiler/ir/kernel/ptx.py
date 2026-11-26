"""The Python entry of the Kernel IR to PTX compiler (the ``emmy-ptx`` crate, through the ``emmy.emmy_ir`` extension).

:func:`compile_kernel` returns a kernel's PTX text and whether the kernel waits on the grid launched ahead of it, or
raises ``NotImplementedError`` when the kernel holds something the compiler has no rule for, and ``ValueError`` when
the kernel is not what the compiler takes or no card is named (:data:`NO_TARGET`).
"""

from __future__ import annotations

from emmy.compiler.ir.kernel.interface import Interface
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.emmy_ir import compile_ptx

#: The compute capability of a context no device answered for and no target named: there is no card to write for.
NO_TARGET = (0, 0)


def compile_kernel(
    kernel: KernelOp,
    interface: Interface,
    *,
    compute_capability: tuple[int, int],
    fuse: bool,
    flush_to_zero: bool,
    single_wave: bool = False,
) -> tuple[str, bool]:
    """The PTX text of ``kernel`` under ``interface`` for the card of ``compute_capability``, with the kernel's two
    whole-kernel precision decisions: multiply-add fusion and flush to zero; and whether the kernel waits on the grid
    launched ahead of it (``griddepcontrol.wait``), so it may be launched as a programmatic dependent launch. The
    text's ``.target`` names the instruction set it needs. ``single_wave`` says the launch places at most one block on
    each multiprocessor of the card."""
    if compute_capability == NO_TARGET:
        raise ValueError(
            f"no card to compile kernel {kernel.name!r} for: no CUDA device answered and no target was named; "
            "name one (--target sm_NN, or a context from Context.from_target)"
        )
    major, minor = compute_capability
    sm = major * 10 + minor
    return compile_ptx(kernel.name, kernel.body, interface, sm=sm, fuse=fuse, ftz=flush_to_zero, single_wave=single_wave)


__all__ = ["compile_kernel"]
