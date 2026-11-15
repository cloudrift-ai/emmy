"""A kernel's interface: what the launch and the code generator must agree on, computed from the kernel.

Its parameters in call order and the shared-memory pool. One function computes it; the CUDA lowering takes the
launch's argument order from it, the PTX compiler its parameters and its shared-memory places. The thread bound is the
grid's, and the instruction set the kernel needs follows from its instructions: the compiler reads both itself. A
computed fact, never stored on the op.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.dim import to_dim
from emmy.compiler.dtype import F32, DataType
from emmy.compiler.ir.kernel.ir import (
    KernelOp,
    TmaDescriptor,
    pack_smem,
)
from emmy.compiler.tensor import Tensor
from emmy.gpu import STATIC_SMEM_CAP

#: The bytes every buffer's first element is aligned to: what the widest access a kernel makes needs, a 16-byte vector
#: load or async copy. The runtime gives its own memory more (a driver allocation per region, and scratch buffers at
#: ``SLAB_ALIGN`` offsets of theirs). Memory a host lends to a region is taken at this alignment and not checked.
BUFFER_ALIGN = 16


@dataclass(frozen=True)
class Param:
    """One kernel parameter. ``name`` is its name in the kernel and in the launch's argument order; ``buffer`` the
    graph buffer behind an ``input`` or ``output`` (a paged buffer binds its page table, so the two differ, and
    ``page`` says how the pages cut it: the dimension and a page's extent along it);
    ``align`` the bytes the launch aligns a buffer's first element to; ``shape`` a buffer's extents, an ``int`` or,
    for a symbolic one, its expression over the runtime sizes; ``pair_table`` says the buffer is the e2m1 pair table,
    whose loads are computed from the index."""

    name: str
    kind: str  # "input" | "output" | "descriptor" | "int"
    dtype: DataType | None = None
    buffer: str = ""
    page: tuple[int, int] | None = None
    align: int = BUFFER_ALIGN
    shape: tuple = ()
    pair_table: bool = False

    @property
    def body_name(self) -> str:
        """The name the kernel body reads the parameter by: the graph buffer behind an input or output (a paged
        buffer's parameter is its page table, under a name of its own), the parameter's own name otherwise."""
        return self.buffer or self.name


@dataclass(frozen=True)
class Interface:
    params: tuple[Param, ...]
    #: Byte offset of each shared buffer in the launch-sized pool; empty when the buffers are static arrays.
    shared_offsets: dict[str, int]
    #: The pool's size in bytes, 0 when the buffers are static arrays.
    shared_pool_bytes: int

    @property
    def arg_order(self) -> tuple[str, ...]:
        """The parameters the launcher binds by name: every one but the runtime ``int`` args, which it appends."""
        return tuple(p.name for p in self.params if p.kind != "int")


def kernel_interface(
    kernel: KernelOp,
    tensors: dict[str, Tensor] | None = None,
    runtime_args: tuple[str, ...] = (),
    pages: dict[str, tuple[int, int]] | None = None,
    pair_tables: frozenset[str] = frozenset(),
) -> Interface:
    """The interface of ``kernel``. ``tensors`` gives each buffer's dtype (f32 where absent) and shape,
    ``runtime_args`` the symbolic sizes it takes as ``int`` parameters, ``pages`` the buffers bound as a page table
    with the dimension the pages cut and a page's extent, ``pair_tables`` the inputs that are the e2m1 pair table."""
    tensors = tensors or {}

    def buffer(name: str, kind: str) -> Param:
        dtype = tensors[name].dtype if name in tensors else F32
        dims = [to_dim(d) for d in tensors[name].shape] if name in tensors else []
        shape = tuple(d.as_static() if d.is_static else d.expression for d in dims)
        page = (pages or {}).get(name)
        return Param(f"{name}__pages" if page else name, kind, dtype, name, page, shape=shape, pair_table=name in pair_tables)

    descriptors = tuple(dict.fromkeys(s.name for s in kernel.body.iter_of_type(TmaDescriptor)))
    params = (
        *(buffer(n, "input") for n in kernel.inputs),
        *(buffer(n, "output") for n in kernel.outputs),
        *(Param(n, "descriptor") for n in descriptors),
        *(Param(n, "int") for n in runtime_args),
    )
    offsets, total = shared_pool(kernel)
    return Interface(params, offsets, total)


def shared_pool(kernel: KernelOp) -> tuple[dict[str, int], int]:
    """``({name: byte offset}, total bytes)`` when the kernel's shared buffers exceed the static cap and live in one
    launch-sized pool, else ``({}, 0)``. The packer is the one ``KernelOp.smem_bytes`` uses, so the declaration and
    the launch-time size agree."""
    smems = list(kernel.smem_buffers.values())
    if not smems:
        return {}, 0
    offsets, total = pack_smem(smems)
    return (offsets, total) if total > STATIC_SMEM_CAP else ({}, 0)


def buffer_dtypes(kernel: KernelOp, tensors: dict[str, Tensor] | None = None) -> dict[str, DataType]:
    """The element dtype of every buffer the kernel's statements read or write by name: its inputs and outputs
    (``tensors`` where given, else the kernel's own), and the shared buffers it declares."""
    dtypes = {n: t.dtype for n, t in (tensors or {**kernel.inputs, **kernel.outputs}).items()}
    return {**dtypes, **{name: smem.dtype for name, smem in kernel.smem_buffers.items()}}
