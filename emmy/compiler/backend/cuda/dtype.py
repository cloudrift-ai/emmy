"""CUDA-backend dtype traits.

Augments :class:`emmy.compiler.dtype.DataType` with the C type name a
shared declaration of Kernel IR spells its element type with
(``Smem.dtype``).

Per-buffer / per-decl dtype lookup helpers (``nbytes_of``) accept the
legacy CUDA C-name spellings (``"float"``, ``"half"``, ...) that older
Kernel IR fields still carry, so all four duplicated ``_DTYPE_BYTES``
tables in the compiler can route through this module.
"""

from __future__ import annotations

from emmy.compiler import dtype as _dtype
from emmy.compiler.dtype import DataType

_CUDA_NAME: dict[DataType, str] = {
    _dtype.F32: "float",
    _dtype.F16: "__half",
    _dtype.F64: "double",
    _dtype.BF16: "__nv_bfloat16",
    _dtype.F8E4M3: "__nv_fp8_e4m3",
    _dtype.F8E5M2: "__nv_fp8_e5m2",
    # Packed e2m1 pairs move as raw bytes.
    _dtype.F4E2M1x2: "unsigned char",
    _dtype.F16x2: "__half2",
    # Raw int16 checkpoint carriers are represented directly in generic tensor algebra.
    _dtype.I16: "short",
    _dtype.I32: "int",
    _dtype.I64: "long long",
    _dtype.U8: "unsigned char",
    _dtype.U16: "unsigned short",
    _dtype.U32: "unsigned int",
    _dtype.U64: "unsigned long long",
    _dtype.BOOL: "bool",
}

# Inverse of _CUDA_NAME for the kernel-internal C-name -> canonical
# lookup: the element dtype of a shared buffer (``interface.buffer_dtypes``).
# Unknown C names (e.g. ``"unsigned long long"`` for mbarriers) map to
# None and the caller treats the smem buffer as "not a tensor".
#
# NOT a bijection: ``f4e2m1x2`` and ``u8`` are both one raw byte and both spell ``unsigned char``,
# so the inverse has to choose. It chooses DELIBERATELY rather than by dict order, because the
# order is invisible and the next byte-wide dtype would flip it silently. The consumer is a
# kernel-level one — an ``unsigned char`` smem slab in this compiler is a packed byte slab, which
# is what the byte-gather drains read back — while ``u8`` is a graph-level storage carrier that
# reaches reconstruction algebra and never declares a slab. Both give the same C type and the same
# width either way; what the choice decides is which canonical NAME downstream predicates see.
_SHARED_CUDA_NAMES: dict[str, str] = {"unsigned char": _dtype.F4E2M1x2.name}
_CANONICAL_FROM_CUDA_NAME: dict[str, str | None] = {
    **{v: k.name for k, v in _CUDA_NAME.items()},
    **_SHARED_CUDA_NAMES,
}


def canonical_from_cuda_name(name: str) -> str | None:
    """Inverse of :func:`cuda_name` over the dtypes we model on the graph.
    Returns ``None`` for CUDA C type names that don't correspond to a
    :class:`DataType` (mbarrier slots, alignment helpers, etc.)."""
    return _CANONICAL_FROM_CUDA_NAME.get(name)


# Bytes for raw CUDA C type names that appear in Kernel IR today (``Smem.dtype``,
# ``Local.dtype``, ``Literal.dtype`` are ``str`` for now — step 2 of the dtype
# migration converts them to :class:`DataType`). Keep this table in sync with
# the canonical sizes on :class:`DataType`.
_C_NAME_BYTES: dict[str, int] = {
    "float": _dtype.F32.nbytes,
    "double": 8,
    "short": 2,
    "int": 4,
    "half": _dtype.F16.nbytes,
    "__half": _dtype.F16.nbytes,
    "__half2": _dtype.F16x2.nbytes,
    "__nv_bfloat16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "__nv_fp8_e4m3": _dtype.F8E4M3.nbytes,
    "__nv_fp8_e5m2": _dtype.F8E5M2.nbytes,
    # One byte either way — ``f4e2m1x2`` (two packed 4-bit values) and ``u8`` share this spelling.
    "unsigned char": _dtype.F4E2M1x2.nbytes,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "f64": 8,
    "long long": 8,
    "unsigned short": 2,
    "unsigned int": 4,
    "unsigned long long": 8,
}


def cuda_name(dtype: str | DataType) -> str:
    """C type name for a kernel signature or local decl (``float``, ``__half``)."""
    return _CUDA_NAME[_dtype.get(dtype)]


def nbytes_of(dtype: str | DataType) -> int:
    """Bytes per element for any dtype spelling that appears in the compiler:

    canonical names (``"f32"``, ``"f16"``), aliases (``"float32"``,
    ``"float16"``, ``"half"``, ``"float"``), CUDA C type names
    (``"double"``, ``"int"``, ``"unsigned long long"``, ``"bfloat16"``),
    and :class:`DataType` instances.
    """
    if isinstance(dtype, DataType):
        return dtype.nbytes
    if dtype in _C_NAME_BYTES:
        return _C_NAME_BYTES[dtype]
    return _dtype.get(dtype).nbytes
