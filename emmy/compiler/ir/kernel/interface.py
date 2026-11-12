"""A kernel's interface: what the launch and the code generator must agree on, computed from the kernel.

Its parameters in call order, the per-block thread bound, the shared-memory pool, and whether it needs the
arch-specific ISA. One function computes it; the CUDA lowering takes the launch's argument order from it, the renderer
its signature and its shared-memory declarations. A computed fact, never stored on the op.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.dtype import F32, DataType
from emmy.compiler.ir.kernel.ir import (
    BLOCK_SIZE,
    Grid,
    KernelOp,
    MmaSyncPtx,
    Tile,
    TmaDescriptor,
    WgmmaDescriptor,
    WgmmaMma,
    pack_smem,
    tile_cells,
)
from emmy.compiler.tensor import Tensor
from emmy.gpu import STATIC_SMEM_CAP


@dataclass(frozen=True)
class Param:
    """One kernel parameter. ``name`` is its name in the kernel and in the launch's argument order; ``buffer`` the
    graph buffer behind an ``input`` or ``output`` (a paged buffer binds its page table, so the two differ)."""

    name: str
    kind: str  # "input" | "output" | "descriptor" | "int"
    dtype: DataType | None = None
    buffer: str = ""
    paged: bool = False


@dataclass(frozen=True)
class Interface:
    params: tuple[Param, ...]
    #: The most threads a block of this kernel launches with.
    threads: int
    #: Byte offset of each shared buffer in the launch-sized pool; empty when the buffers are static arrays.
    shared_offsets: dict[str, int]
    #: The pool's size in bytes, 0 when the buffers are static arrays.
    shared_pool_bytes: int
    #: Whether the kernel must be assembled for the arch-suffixed target (``sm_<cc>a``).
    arch_specific: bool

    @property
    def arg_order(self) -> tuple[str, ...]:
        """The parameters the launcher binds by name: every one but the runtime ``int`` args, which it appends."""
        return tuple(p.name for p in self.params if p.kind != "int")


def kernel_interface(
    kernel: KernelOp,
    tensors: dict[str, Tensor] | None = None,
    runtime_args: tuple[str, ...] = (),
    paged: frozenset[str] = frozenset(),
) -> Interface:
    """The interface of ``kernel``. ``tensors`` gives each buffer's dtype (f32 where absent), ``runtime_args`` the
    symbolic sizes it takes as ``int`` parameters, ``paged`` the buffers bound as a page table."""
    tensors = tensors or {}

    def buffer(name: str, kind: str) -> Param:
        dtype = tensors[name].dtype if name in tensors else F32
        return Param(f"{name}__pages" if name in paged else name, kind, dtype, name, name in paged)

    descriptors = tuple(dict.fromkeys(s.name for s in kernel.body.iter_of_type(TmaDescriptor)))
    params = (
        *(buffer(n, "input") for n in kernel.inputs),
        *(buffer(n, "output") for n in kernel.outputs),
        *(Param(n, "descriptor") for n in descriptors),
        *(Param(n, "int") for n in runtime_args),
    )
    offsets, total = shared_pool(kernel)
    return Interface(params, thread_bound(kernel), offsets, total, needs_arch_isa(kernel))


def thread_bound(kernel: KernelOp) -> int:
    """The per-block thread count of the kernel's grid, its producer band included."""
    for s in kernel.body:
        if isinstance(s, Grid):
            return s.block_threads + s.aux_threads
        if isinstance(s, Tile):
            return tile_cells(s) + s.aux_threads
    return BLOCK_SIZE


def shared_pool(kernel: KernelOp) -> tuple[dict[str, int], int]:
    """``({name: byte offset}, total bytes)`` when the kernel's shared buffers exceed the static cap and live in one
    launch-sized pool, else ``({}, 0)``. The packer is the one ``KernelOp.smem_bytes`` uses, so the declaration and
    the launch-time size agree."""
    smems = list(kernel.smem_buffers.values())
    if not smems:
        return {}, 0
    offsets, total = pack_smem(smems)
    return (offsets, total) if total > STATIC_SMEM_CAP else ({}, 0)


def needs_arch_isa(kernel: KernelOp) -> bool:
    """Whether the kernel holds an instruction ptxas assembles only for the arch-suffixed target: TMA, the
    block-scaled fp4 ``mma``, or ``wgmma``."""
    return any(
        isinstance(s, (TmaDescriptor, WgmmaMma, WgmmaDescriptor)) or (isinstance(s, MmaSyncPtx) and s.sfa_frag is not None)
        for s in kernel.body.iter()
    )


def buffer_dtypes(kernel: KernelOp, tensors: dict[str, Tensor] | None = None) -> dict[str, DataType]:
    """The element dtype of every buffer the kernel's statements read or write by name: its inputs and outputs
    (``tensors`` where given, else the kernel's own), and the shared buffers it declares."""
    from emmy.compiler.backend.cuda.dtype import canonical_from_cuda_name  # noqa: PLC0415 — the backend imports the IR
    from emmy.compiler.dtype import get  # noqa: PLC0415

    dtypes = {n: t.dtype for n, t in (tensors or {**kernel.inputs, **kernel.outputs}).items()}
    for name, smem in kernel.smem_buffers.items():
        canonical = canonical_from_cuda_name(smem.dtype)
        if canonical is not None:
            dtypes[name] = get(canonical)
    return dtypes
