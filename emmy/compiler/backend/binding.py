"""Host bytes for a plan's buffers: the single fill policy every backend's bound buffers start from.

Inputs and constants become bytes in their buffer's storage dtype (bf16 as its encoded ``uint16`` bits), the
plan's generated constants are evaluated once, and every symbolic axis resolves from the supplied input shapes.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, apply_weight_loads
from emmy.compiler.dtype import encode_bf16


def is_device_tensor(value) -> bool:
    """A torch CUDA tensor — bound by address, never copied through the host."""
    return hasattr(value, "data_ptr") and hasattr(value, "is_cuda") and bool(value.is_cuda)


def numpy_storage(src, dtype) -> np.ndarray:
    """Return a contiguous host array in one buffer's physical storage dtype."""
    arr = np.asarray(src)
    if getattr(dtype, "name", dtype) == "bf16":
        return np.ascontiguousarray(arr) if arr.dtype == np.uint16 else encode_bf16(arr)
    return np.ascontiguousarray(arr, dtype=dtype.np)


def host_bytes(buf: BufferSpec, shape: tuple[int, ...], src, constants: dict[str, float]) -> bytes:
    """The bytes one input or constant buffer starts from: the supplied array, the plan's scalar
    constant, seeded normal values for an unsupplied input, zeros otherwise."""
    np_dtype = buf.dtype.np
    is_bf16 = getattr(buf.dtype, "name", buf.dtype) == "bf16"
    n = math.prod(shape)
    if src is not None:
        arr = numpy_storage(src, buf.dtype)
        if arr.size != n:
            raise ValueError(f"buffer {buf.name!r}: {arr.size} element(s) supplied for shape {shape}")
        return arr.reshape(shape).tobytes()
    if buf.role == "constant" and buf.name in constants:
        v = float(constants[buf.name])
        if is_bf16:
            # bf16 buffers ride as uint16 BITS (``BF16.np``) — casting the float would zero it;
            # encode the value to bf16 bits (round-to-nearest-even on the dropped mantissa half).
            bits = int(np.float32(v).view(np.uint32))
            return np.full(shape, np.uint16((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16), dtype=np.uint16).tobytes()
        return np.full(shape, v, dtype=np_dtype).tobytes()
    if buf.role == "input":
        # Seeded normal values for an un-supplied input, one stream per buffer name — what the
        # reference path draws (``standard_normal``). A kernel's cost can depend on its values
        # (an IEEE division's slow path, an exp near overflow): the index ramp this replaced
        # repeats every 101 elements, and through a whole layer it drove softmax rows into
        # that slow path, so a timing on it was not the timing on real data. Integer carriers
        # keep a small ramp: their codes are data, not magnitudes.
        if not is_bf16 and np.issubdtype(np_dtype, np.integer):
            return (np.arange(n, dtype=np.int64) % 101).astype(np_dtype).tobytes()
        vals = np.random.default_rng(zlib.crc32(buf.name.encode())).standard_normal(n, dtype=np.float32)
        vals = encode_bf16(vals) if is_bf16 else vals.astype(np_dtype)
        return vals.tobytes()
    return np.zeros(shape, dtype=np_dtype).tobytes()


def host_bindings(plan: ExecutionPlan, input_data: dict, sym_values: dict[str, int], *, only=None) -> dict[str, bytes]:
    """Starting bytes for input and constant buffers (``only`` narrows the set). A buffer bound
    to a device tensor is skipped: its memory is lent to the runtime instead; so is a paged one,
    which has no slab to fill and takes its page table instead. Output and scratch buffers start
    zeroed inside the runtime. Saturating casts here are intended, not bugs: an
    SDPA mask-fill constant (``-1e9``) is meant to become ``-inf`` in fp16 (masked → 0 after
    softmax)."""
    out: dict[str, bytes] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for buf in plan.buffers:
            if buf.role not in ("input", "constant") or (only is not None and buf.name not in only) or buf.name in plan.paged:
                continue
            src = input_data.get(buf.name)
            if is_device_tensor(src):
                continue
            out[buf.name] = host_bytes(buf, buf.resolve_shape(sym_values) or (1,), src, plan.constants)
    return out


def with_generated_constants(plan: ExecutionPlan, input_data: dict) -> dict:
    """Add the plan's SELF-CONTAINED constants that ``input_data`` does not already carry.

    A deterministic source-free bind record is evaluated once while the graph is projected and
    its bytes ride the plan (``PLAN_FORMAT_GENERATED``) — a coded linear's Hadamard factor and
    its coordinate tables are exactly that. Nothing outside the plan can supply them, and an
    unsupplied constant buffer starts as ZEROS, so the runtime must read them here.
    Caller-supplied arrays win: serving binds the same specs through ``assemble_source`` and
    shares one device copy across its twins.
    """
    from emmy.compiler.loader.binder import assemble_source  # noqa: PLC0415

    feed = dict(input_data)
    for nid, w in plan.weights.items():
        # ``load_ops is None`` marks a weight the plan cannot rebind at all (see ``WeightSpec``).
        if w.generated is None or w.load_ops is None or nid in feed:
            continue
        feed[nid] = apply_weight_loads(assemble_source(w, {}), w.load_ops)
    return feed


def resolve_symbolic(plan: ExecutionPlan, input_data: dict) -> dict[str, int]:
    """Bind every symbolic axis name to a concrete ``int``. Reads the runtime value from the
    supplied input array shape (``plan.symbolic_bindings`` says which input + dim each name
    reads from). When no array is supplied for that input — the autotuner benches without real
    inputs — falls back to the ``Dim`` hint so the graph runs at its expected (tuned) size. A
    capacity-capped kernel bakes its smem slab at the hint and is only correct up to that cap,
    so a larger supplied extent is an error rather than an out-of-bounds read."""
    env: dict[str, int] = {}
    for name, (buf, dim_idx) in plan.symbolic_bindings.items():
        arr = input_data.get(buf)
        if arr is not None:
            env[name] = int(arr.shape[dim_idx])
            cap = plan.symbolic_caps.get(name)
            if cap is not None and env[name] > cap:
                raise ValueError(
                    f"symbolic dim {name!r} = {env[name]} exceeds the capacity-capped kernel's hint ({cap}); "
                    f"this build bakes its smem slab at {cap} and cannot run a larger extent — "
                    f"re-trace with a larger --seq-len hint or use the ceil-div (uncapped) lowering"
                )
        elif name in plan.symbolic_hints:
            env[name] = plan.symbolic_hints[name]
        else:
            raise ValueError(
                f"symbolic dim {name!r} reads from input {buf!r}.shape[{dim_idx}] but no array was supplied and the dim carries no hint"
            )
    return env
