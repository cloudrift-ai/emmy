"""The Python entry of the Kernel IR to PTX compiler (the ``emmy-ptx`` crate, through the ``emmy.emmy_ir`` extension).

:func:`compile_kernel` returns a kernel's PTX text, or raises :class:`NoRule` when the kernel holds something the
compiler has no rule for yet: the kernel is well formed and its code is generated the other way, as CUDA. Every
other error says the kernel is not what the compiler takes, or the compiler is wrong, and is never a reason to
generate the code another way.

Temporary seam: while the CUDA renderer exists, what it works out as it prints and the compiler does not read is
refused here with the same error. That is the e2m1 pair table, whose loads the renderer computes instead of reading;
the crate refuses a paged buffer, whose page the renderer resolves, itself.
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
    pair_tables: frozenset[str] = frozenset(),
) -> str:
    """The PTX text of ``kernel`` under ``interface`` for the card of ``compute_capability``, with the kernel's two
    whole-kernel precision decisions: multiply-add fusion and flush to zero. ``pair_tables`` names the inputs that
    are the e2m1 pair table."""
    computed = sorted(pair_tables & set(kernel.inputs))
    if computed:
        raise NoRule(f"no PTX rule for a load of the e2m1 pair table, which is computed and not read: `{', '.join(computed)}`")
    major, minor = compute_capability
    return compile_ptx(kernel.name, kernel.body, interface, sm=major * 10 + minor, fuse=fuse, ftz=flush_to_zero)


__all__ = ["NoRule", "compile_kernel"]
