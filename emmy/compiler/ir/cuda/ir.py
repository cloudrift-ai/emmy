"""CudaOp — graph-level wrapper around a compiled kernel.

Produced by ``passes/lowering/cuda`` by compiling each ``KernelOp`` body to PTX text. The final graph before codegen
is ``Graph[CudaOp + InputOp + ConstantOp]``; the CUDA backend walks it in topological order, emits one launch per
node, and wires buffer pointers by node id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from emmy.compiler.ir.base import Op
from emmy.compiler.ir.expression import Expression
from emmy.compiler.structural import digest


@dataclass(frozen=True)
class TmaDescMeta:
    """Metadata the CUDA backend needs to encode a TMA descriptor at launch.

    ``name`` matches the kernel signature parameter (added to
    ``arg_order`` after the buffer args). ``src_buf`` names the graph
    buffer whose device pointer + shape feed
    ``cuTensorMapEncodeTiled``. ``box_extents`` and ``swizzle`` are the
    descriptor's per-dim box and swizzle mode."""

    name: str
    src_buf: str
    box_extents: tuple[int, ...]
    swizzle: str = "NONE"


# int = static factor; str = symbolic axis name (resolved at launch from
# sym_values); Expression = a composite extent (e.g. ceil-div ``(seq_len+15)//16`` for
# a hint-driven masked block axis) evaluated against sym_values at launch.
_GridFactor = int | str | Expression
GridDimSpec = tuple[_GridFactor, ...]  # product of factors → one grid dim's extent


@dataclass(frozen=True)
class CudaOp(Op):
    """One CUDA kernel invocation as a graph-op.

    ``grid`` and ``block`` each carry three per-dim ``GridDimSpec`` tuples;
    every entry in a spec is multiplied together at launch time. Pure-int
    specs (e.g. ``((128,), (1,), (1,))``) describe static launch geometry;
    specs containing strings (e.g. ``(("seq_len",), (1,), (1,))``)
    reference symbolic dims that the launch resolver looks up in the
    runtime ``sym_values`` env. ``runtime_args`` lists those symbolic
    names in the order they appear in the kernel signature (one ``int``
    parameter per name, slotted after the buffer args and before any
    TMA descriptor params).
    """

    kernel_source: str = ""  # the kernel's PTX text
    kernel_name: str = ""
    arg_order: tuple[str, ...] = ()  # kernel-param names in positional order
    grid: tuple[GridDimSpec, GridDimSpec, GridDimSpec] = ((1,), (1,), (1,))
    block: tuple[GridDimSpec, GridDimSpec, GridDimSpec] = ((1,), (1,), (1,))
    smem_bytes: int = 0
    zero_outputs: tuple[str, ...] = ()
    # Buffers this kernel zero-writes as a DELEGATED prologue (``ZeroPrologue`` stmts injected
    # by ``005_delegate_zero_init``): downstream kernels' atomic accumulators. Not read at
    # launch (the kernel body does the zeroing) — the slab planner uses it to start those
    # buffers' live intervals at THIS launch instead of their own producer's.
    zero_prologues: tuple[str, ...] = ()
    comment: str = ""
    tma_descriptors: tuple[TmaDescMeta, ...] = field(default_factory=tuple)
    runtime_args: tuple[str, ...] = ()
    #: Serial launch axes ``(name, extent)``: the launcher runs the kernel once per coordinate in
    #: order, passing the coordinate through the ``runtime_args`` slot of the same name.
    serial: tuple[tuple[str, int], ...] = ()
    #: Whether the kernel must be assembled for the arch-suffixed target (``sm_<cc>a``): it holds TMA, the
    #: block-scaled fp4 ``mma`` or ``wgmma``.
    arch_specific: bool = False
    #: Whether the kernel waits on the grid ahead of it before its first memory access, so the runtime may launch
    #: it as a programmatic dependent launch. Every lowered kernel does; a hand-written source says so itself.
    dependent_launch: bool = False

    def pretty_body(self) -> str:
        return self.kernel_source

    def identity_key(self, *, structural: bool = True, with_io: bool = False, with_knobs: bool = False) -> str | None:
        """Override :meth:`Op.identity_key` whole: digest of the kernel's code + launch params — past code
        generation, body / io / knobs are all realized into the artifact, so the lattice flags have nothing left to
        fold and are accepted-and-ignored. Name-invariant: the kernel's name stands in its code (its entry and the
        symbols derived from it, :func:`renamed`) but doesn't change runtime behavior, so it normalizes out —
        renaming a kernel (e.g. via op provenance) neither busts the perf cache nor blocks an isolated-kernel tune
        from transferring to a whole-model compile."""
        src = renamed(self.kernel_source, self.kernel_name, "_K_") if self.kernel_name else self.kernel_source
        return digest(type(self).__name__, src, self.arg_order, self.grid, self.block, self.smem_bytes)


def renamed(source: str, old: str, new: str) -> str:
    """The PTX text ``source`` of kernel ``old`` with the kernel named ``new``: its entry and the parameter and shared
    symbols the PTX printer derives from the entry's name (``<name>_param_<i>``, ``<name>_shared_<i>``), and nothing
    else."""
    return re.sub(rf"(?<![\w$]){re.escape(old)}(?=(?:_param_\d+|_shared_\d+)?(?![\w$]))", new, source)


def resolve_dim(spec, sym_values: dict[str, int]) -> int:
    """Multiply a ``GridDimSpec``'s factors, resolving ``str`` factors
    from ``sym_values`` and ``Expression`` factors via ``Expression.eval`` (e.g. a
    ceil-div block extent ``(seq_len+15)//16``). Accepts a bare ``int``
    (legacy static grid) as shorthand for a single-int spec — keeps
    pre-symbolic CudaOps working until every producer has been migrated.
    Raises ``KeyError`` on an unknown symbolic name."""
    if isinstance(spec, int):
        return spec
    total = 1
    for factor in spec:
        if isinstance(factor, int):
            total *= factor
        elif isinstance(factor, str):
            total *= sym_values[factor]
        else:
            total *= factor.eval(sym_values)
    return total
