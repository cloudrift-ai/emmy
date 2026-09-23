"""CUDA program execution for ``Graph[CudaOp]``: the Python facade over the runtime.

Python compiles — nvcc into the content-addressed cubin cache, the execution plan, the host
bytes every input and constant starts from — and hands the runtime (``crates/emmy-runtime``,
hosted in-process through the ``emmy_runtime`` extension) the plan's JSON form, one cubin path
per kernel, those bytes, and the memory every region of the program's layout lives in. The
runtime owns the launches: it resolves symbolic geometry, encodes TMA descriptors, captures
graphs, times events and polls a hung launch against a deadline. Nothing in this module holds
a device pointer for longer than it takes to hand one over.

Memory is allocated here, through torch, so the vLLM plugin's profiler sees every byte a program
holds and the serving runners hand tensors in and out with no copy. The runtime derives the
layout — one region per input / constant / output buffer, every scratch buffer packed by
liveness into one slab — and this side allocates a tensor per region (pooled across programs by
a :class:`BufferArena`). Without torch the runtime allocates for itself.

Buffer roles come from the graph: ``graph.inputs`` → input, ``ConstantOp`` → constant,
``graph.outputs`` → output, everything else → scratch. Launch order is
``graph.topological_order()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os as _os
import pickle
import sys as _sys
import time as _time_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import emmy_runtime
import numpy as np

from emmy import config
from emmy.compiler.backend import BenchmarkResult, LaunchTime, RunResult
from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.backend.cuda.device import device
from emmy.compiler.backend.plan import BufferSpec as _Buffer
from emmy.compiler.backend.plan import ExecutionPlan, KernelSpec, apply_weight_loads, plan_from_graph, plan_to_dict
from emmy.compiler.backend.plan import LaunchSpec as _Launch
from emmy.compiler.dtype import encode_bf16
from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

#: The runtime's hung-launch error, raised by a timed launch whose completion event misses its
#: deadline. The kernel stays **resident on the device** after the raise — nothing in-process can
#: evict it, only the SIGKILL-isolated bench worker resets the device — so a caller that catches
#: it must treat the device as poisoned and stop, or its next blocking synchronize (the torch
#: peer bench) waits behind the still-running kernel. A ``RuntimeError`` subclass, so existing
#: ``except RuntimeError`` handlers (the autotune sweep) keep marking the variant ``bench_fail``.
HungKernelError = emmy_runtime.HungKernelError


# ---------------------------------------------------------------------------
# Kernels: compile through the cubin cache, hand the runtime a path
# ---------------------------------------------------------------------------


def _cubin_path(name: str, spec: KernelSpec, *, cubin_dir: Path | None = None) -> Path:
    """The cubin for one :class:`KernelSpec`. A ``binary_key`` (the pack path) names the
    content-addressed cubin straight from the cache; otherwise the ``source`` compiles through
    the same cache. A key whose cubin has been evicted falls back to the source when present,
    and errors otherwise — the pack loader pre-checks cubin existence, so hitting this means the
    cache was cleared mid-boot."""
    if spec.binary_key is not None:
        path = (cubin_dir or nvcc.cubin_cache_dir()) / f"{spec.binary_key}.cubin"
        if path.exists():
            return path
        if spec.source is None:
            raise RuntimeError(
                f"kernel {name!r}: cubin {spec.binary_key} is gone from the cache and the plan carries no source — "
                "regenerate the pack or boot without it (full compile)"
            )
    if spec.source is None:
        raise RuntimeError(f"kernel {name!r}: plan carries neither a source nor a cached cubin")
    return nvcc.compile_kernel(spec.source, name, arch_specific=spec.arch_specific)


def _compile_kernels(plan: ExecutionPlan, *, deadline: float | None = None, cubin_dir: Path | None = None) -> dict[str, str]:
    """One cubin path per kernel of ``plan``, compiled through the cache where needed.

    ``deadline`` is the compile budget's monotonic expiry, checked BETWEEN kernels — the only
    boundary a Python-level check has, since one compile is a single call into nvcc. Checking
    here rather than after the whole load is what keeps a cold multi-kernel compile from
    outliving the wall cap that SIGKILLs the bench worker: past the cap the operator is told a
    worker died, which reads as a slow kernel, and the fact that nothing about the kernel was
    measured is lost."""
    binaries: dict[str, str] = {}
    for index, (name, spec) in enumerate(plan.kernels.items(), start=1):
        binaries[name] = str(_cubin_path(name, spec, cubin_dir=cubin_dir))
        if deadline is not None and _time_module.monotonic() > deadline:
            raise CompileBudgetExceeded(
                f"compile stage exceeded its budget after {index} of {len(plan.kernels)} kernel(s) "
                f"({name}) — nothing measured; raise {config.BENCH_COMPILE_TIMEOUT_S} to compile it"
            )
    return binaries


def kernel_attributes(name: str, spec: KernelSpec) -> dict[str, int]:
    """Register count and local / static shared bytes of one kernel, read off its cubin."""
    return device().kernel_attributes(str(_cubin_path(name, spec)), name)


# ---------------------------------------------------------------------------
# Host bytes: the single fill policy every bound buffer starts from
# ---------------------------------------------------------------------------


def _is_device_tensor(value) -> bool:
    """A torch CUDA tensor — bound by address, never copied through the host."""
    return hasattr(value, "data_ptr") and hasattr(value, "is_cuda") and bool(value.is_cuda)


def _numpy_storage(src, dtype) -> np.ndarray:
    """Return a contiguous host array in one buffer's physical storage dtype."""
    arr = np.asarray(src)
    if getattr(dtype, "name", dtype) == "bf16":
        return np.ascontiguousarray(arr) if arr.dtype == np.uint16 else encode_bf16(arr)
    return np.ascontiguousarray(arr, dtype=dtype.np)


def _host_bytes(buf: _Buffer, shape: tuple[int, ...], src, constants: dict[str, float]) -> bytes:
    """The bytes one input or constant buffer starts from: the supplied array, the plan's scalar
    constant, a deterministic pseudo-random ramp for an unsupplied input, zeros otherwise."""
    np_dtype = buf.dtype.np
    is_bf16 = getattr(buf.dtype, "name", buf.dtype) == "bf16"
    n = math.prod(shape)
    if src is not None:
        arr = _numpy_storage(src, buf.dtype)
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
        # Pseudo-random fill for un-supplied inputs. The index ramp is built in int64, not
        # ``np_dtype``: a float16 buffer past 65504 elements would overflow to ``inf`` (then
        # ``inf % 101`` → ``nan``). Compute in fp32 and cast the final values — always in
        # ``[-0.5, 0.5]``, so fp16-safe.
        idx = np.arange(n, dtype=np.int64)
        vals = 0.01 * ((idx.astype(np.float32) * 7 + 13) % 101 - 50)
        vals = encode_bf16(vals) if is_bf16 else vals.astype(np_dtype)
        return vals.tobytes()
    return np.zeros(shape, dtype=np_dtype).tobytes()


def _host_bindings(plan: ExecutionPlan, input_data: dict, sym_values: dict[str, int], *, only=None) -> dict[str, bytes]:
    """Starting bytes for input and constant buffers (``only`` narrows the set). A buffer bound
    to a device tensor is skipped: its memory is lent to the runtime instead. Output and scratch
    buffers start zeroed inside the runtime. Saturating casts here are intended, not bugs: an
    SDPA mask-fill constant (``-1e9``) is meant to become ``-inf`` in fp16 (masked → 0 after
    softmax)."""
    out: dict[str, bytes] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for buf in plan.buffers:
            if buf.role not in ("input", "constant") or (only is not None and buf.name not in only):
                continue
            src = input_data.get(buf.name)
            if _is_device_tensor(src):
                continue
            out[buf.name] = _host_bytes(buf, buf.resolve_shape(sym_values) or (1,), src, plan.constants)
    return out


def _with_generated_constants(plan: ExecutionPlan, input_data: dict) -> dict:
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


def _resolve_symbolic(plan: ExecutionPlan, input_data: dict) -> dict[str, int]:
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


# ---------------------------------------------------------------------------
# Memory: torch tensors lent to the runtime, pooled across programs
# ---------------------------------------------------------------------------


def _torch():
    """torch, or ``None`` when it is missing or sees no device — the runtime then allocates."""
    try:
        import torch  # noqa: PLC0415

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def _torch_dtype(np_dtype):
    import torch  # noqa: PLC0415

    return torch.from_numpy(np.zeros(0, dtype=np_dtype)).dtype


def _new_backing(nbytes: int):
    """One zeroed device tensor for a region."""
    import torch  # noqa: PLC0415

    return torch.zeros(max(1, int(nbytes)), dtype=torch.uint8, device="cuda")


def _flat_bytes(tensor):
    """The same memory as a flat byte tensor — how every lent region is kept, whatever the
    caller's shape and dtype."""
    import torch  # noqa: PLC0415

    return tensor.contiguous().view(-1).view(torch.uint8)


_DLPACK_CODES = {"f": 2, "i": 0, "u": 1}


def device_view(pinned):
    """Address a pinned (page-locked) host tensor as a CUDA tensor of the same shape and dtype:
    under unified addressing a mapped host allocation is reachable from the device at its own
    address, so a kernel gathers from it over PCIe with no copy. ``pinned`` must stay alive as
    long as the view; the view owns nothing."""
    import torch  # noqa: PLC0415

    if not pinned.is_pinned() or not pinned.is_contiguous():
        raise ValueError("device_view needs a contiguous pinned host tensor")
    host, device_ptr = device().pointer_attributes(pinned.data_ptr())
    if not host or device_ptr != pinned.data_ptr():
        raise RuntimeError("this platform does not map host allocations into the device address space")
    np_dtype = np.dtype(str(pinned.dtype).removeprefix("torch."))
    capsule = emmy_runtime.device_tensor_capsule(
        pinned.data_ptr(), list(pinned.shape), _DLPACK_CODES[np_dtype.kind], np_dtype.itemsize * 8, 0
    )
    return torch.from_dlpack(capsule)


class BufferArena:
    """Cross-program pooling of a program's regions. Programs that run sequentially share one
    backing per region name (``role:name`` for activations, ``scratch`` for the slab), sized to
    the largest request so far; constants are never pooled. Growth allocates a fresh backing and
    keeps the older generations alive under the programs that still view them, so captured
    graphs never dangle. Safety is the caller's contract: programs sharing an arena never run
    concurrently, and each program's outputs are consumed before the next program runs."""

    def __init__(self) -> None:
        self._backings: dict[str, list] = {}

    def backing(self, key: str, nbytes: int):
        generations = self._backings.setdefault(key, [])
        if generations and generations[-1].numel() >= max(1, nbytes):
            return generations[-1]
        generations.append(_new_backing(nbytes))
        return generations[-1]


# ---------------------------------------------------------------------------
# Budgets and errors
# ---------------------------------------------------------------------------


def _launch_deadline_ms(iters_done: int, batch: int) -> float:
    """The watchdog deadline for one event window of ``batch`` launches: the first iteration's
    own budget on iter 0, the steady per-launch deadline after."""
    per_launch = config.first_iter_timeout_ms() if iters_done == 0 else config.kernel_timeout_ms()
    return per_launch * batch


class CompileBudgetExceeded(RuntimeError):
    """The compile stage ran past ``compile_timeout_s``, before any launch happened.

    Distinct from a plain ``RuntimeError`` because **nothing about the kernel's speed was
    measured**: cicc was slow, which is a fact about the compiler and the tile's unroll size, not
    about the kernel. Callers must record no latency for it — inventing one mislabels the config,
    and a persisted row is worse than mislabelled, because it is then served as a cache hit and
    the config is never re-benched. Subclasses ``RuntimeError`` so existing handlers still catch
    it. The budget is enforced BETWEEN kernels (:func:`_compile_kernels`) and once
    more when the whole setup returns, so it fires on a compile that is still running rather than
    only on one that finished. That ordering is what the distinction rests on: a bench worker's
    wall cap SIGKILLs the child, and a killed child reports a dead worker — which reads as a slow
    kernel and is the opposite of what a compile overrun means."""


def compile_budget_overrun(exc: BaseException) -> bool:
    """``True`` iff ``exc`` is :class:`CompileBudgetExceeded`, from either bench path — the class
    itself in-process, or the flag :class:`BenchWorkerJobError` carries when the exception was
    raised in the worker subprocess (the protocol pickles ``error`` as a string, losing the
    class). Lives here, beside the two classes it bridges."""
    return isinstance(exc, CompileBudgetExceeded) or bool(getattr(exc, "compile_budget", False))


class GraphCaptureError(RuntimeError):
    """CUDA graph capture of the bench launch loop failed.

    Raised by :meth:`CompiledProgram.capture_launch_graphs` and
    :meth:`CompiledProgram.capture_program_graph`; the runtime ends the capture before
    reporting, so the stream is clean and the caller can simply retry the bench uncaptured.
    Only the per-kernel reproducer bench enables capture (the autotune sweep never does), so
    this can't be misclassified as a ``bench_fail`` there."""


_AUTO_BUDGET_MS = 100.0
# Iter-count cap on ``num_iters="auto"``. Combined with the GPU-time
# target above: whichever fires first wins. The cap is the binding
# constraint for fast kernels (sub-ms / launch, where 100 ms target
# would otherwise mean 100s of iters and the corresponding atomic /
# clock-state pressure on heavy-fanout K-split kernels); the GPU-time
# target is the binding constraint for slow kernels (>= 1 ms / launch,
# where 100 iters would over-measure relative to confidence needs).
_AUTO_MAX_ITERS = 100
# Target per-kernel-position timing window. Sub-millisecond kernels are
# dominated by per-iter host framing overhead; we amortize it by repeating
# each launch ``batch_size`` times inside one CUDA event window, where
# ``batch_size = ceil(_BATCH_TARGET_MS / per_launch_ms)``. Calibrated after
# warmup from the last-warmup iter's per-launch timings, then held fixed
# during measurement.
_BATCH_TARGET_MS = 1.0
# Minimum total GPU time the warmup window should cover. sm_120 (and
# other consumer GPUs with auto-boost) take several ms to ramp clocks
# from idle. For tiny kernels the requested ``warmup`` iters may sum
# to << 1 ms — the first measured iters then see mid-ramp clocks and
# the median jitters across runs. After the post-warmup batch-size
# calibration we extend ``warmup`` so total warmup GPU time clears
# this threshold.
_WARMUP_TARGET_MS = 10.0


# ---------------------------------------------------------------------------
# CompiledProgram: one loaded program in the runtime + uniform iter loop
# ---------------------------------------------------------------------------


@dataclass
class CompiledProgram:
    """One program loaded into the runtime: its plan on this side, its buffers, kernels and
    captured graphs on the other.

    Constructed inside ``gpu_lock()`` by the public entry points (:func:`run_program`,
    :func:`run_program_debug`, :func:`benchmark_program`) so every CUDA-touching phase — nvcc,
    allocation, the kernel-launch loop, the output copy — runs with the lock held. Peer xdist
    workers never interleave with us on the device, which previously surfaced as small numerical
    divergence in multi-kernel attention tests when the suite ran in parallel.

    All three entry points walk launches through the same :meth:`iter_once`. What differs
    between them — single pass vs warmup+measure vs snapshot-every-launch — collapses to which
    optional callbacks they pass."""

    plan: ExecutionPlan
    program: Any
    executor: Any
    load_times_ms: dict[str, float] = field(default_factory=dict)
    # The symbolic environment the program is currently bound at (``{}`` for a static graph).
    sym_values: dict[str, int] = field(default_factory=dict)
    # Cross-program pooling for the regions this program's memory came from (``None`` → the
    # program's own tensors, or the runtime's allocations when torch is absent).
    arena: BufferArena | None = None
    # Region name → the tensor lent to the runtime (empty when the runtime allocates).
    _tensors: dict[str, Any] = field(default_factory=dict, repr=False)
    # Number of completed ``iter_once`` calls — iter 0 runs under the first-iteration watchdog
    # deadline (first-launch lazy-load / carveout stalls are not hangs; see ``_launch_deadline_ms``).
    _iters_done: int = field(default=0, repr=False)

    @property
    def launches(self) -> list[_Launch]:
        return self.plan.launches

    def _buffer(self, name: str) -> _Buffer:
        return next(b for b in self.plan.buffers if b.name == name)

    @classmethod
    def build(
        cls,
        graph: Graph,
        input_data: dict | None = None,
        *,
        compile_timeout_s: float | None = None,
        arena: BufferArena | None = None,
    ) -> CompiledProgram:
        """Compile ``graph`` and build — ``plan_from_graph`` + :meth:`build_from_plan`; the
        graph is never consulted after the projection (one runtime path whether the plan came
        from a fresh compile or from a stored pack)."""
        return cls.build_from_plan(plan_from_graph(graph), input_data, compile_timeout_s=compile_timeout_s, arena=arena)

    @classmethod
    def build_from_plan(
        cls,
        plan: ExecutionPlan,
        input_data: dict | None = None,
        *,
        compile_timeout_s: float | None = None,
        arena: BufferArena | None = None,
        cubin_dir: Path | None = None,
    ) -> CompiledProgram:
        """Compile every kernel (cubin-by-key or source-via-cache), resolve the symbolic
        environment from the supplied input shapes, allocate every region of the runtime's
        layout, then load the program: the runtime uploads the input and constant bytes (the
        plan's generated constants fill themselves — see :func:`_with_generated_constants`) and
        fills the runtime constants. A constant supplied as a device tensor is lent to the
        runtime as it is — the serving path's weights, uploaded once and shared across twins.

        ``compile_timeout_s`` bounds the setup phase at a C-call boundary: the kernel compile
        checks it between kernels and the load is checked when it returns, so an overrun raises
        :class:`CompileBudgetExceeded` before the caller proceeds to launches, leaving no
        in-flight kernels queued. ``arena`` pools this program's regions with every other
        program built on it (see :class:`BufferArena`).

        Caller is expected to hold ``gpu_lock()`` around this call and every subsequent method
        on the returned program."""
        t0 = _time_module.monotonic()
        binaries = _compile_kernels(plan, deadline=None if compile_timeout_s is None else t0 + compile_timeout_s, cubin_dir=cubin_dir)
        compiled = _time_module.monotonic()
        input_data = _with_generated_constants(plan, input_data or {})
        sym_values = _resolve_symbolic(plan, input_data)
        program = emmy_runtime.Program(json.dumps(plan_to_dict(plan)))
        self = cls(plan=plan, program=program, executor=None, sym_values=sym_values, arena=arena)
        regions = self._provision(sym_values, input_data)
        bindings = _host_bindings(plan, input_data, sym_values)
        self.executor = emmy_runtime.Executor(device(), program, binaries, bindings, sym_values, regions)
        self._track()
        elapsed = _time_module.monotonic() - t0
        if compile_timeout_s is not None and elapsed > compile_timeout_s:
            raise CompileBudgetExceeded(f"compile stage exceeded {compile_timeout_s:.1f}s budget ({elapsed:.2f}s) — nothing measured")
        logger.info(
            "[cuda] CompiledProgram.build: %d launch(es) compile+alloc=%.2fs kernels=[%s]",
            len(plan.launches),
            elapsed,
            ", ".join(f"{li}:{lc.kernel_name}" for li, lc in enumerate(plan.launches)),
        )
        self.load_times_ms = {"compile_ms": (compiled - t0) * 1000, **self.executor.load_times_ms()}
        return self

    def _provision(self, sym_values: dict[str, int], input_data: dict) -> dict[str, tuple[int, int]] | None:
        """The memory every region of the layout at ``sym_values`` lives in, as ``{region:
        (address, bytes)}`` — ``None`` when torch is absent and the runtime allocates. A region
        already backed by a large enough tensor is kept (a rebind that grows nothing keeps every
        address); the arena pools everything but constants; a constant bound to a device tensor
        is that tensor."""
        if _torch() is None:
            return None
        layout = self.program.layout(sym_values)
        regions: dict[str, tuple[int, int]] = {}
        for name, nbytes in layout["regions"].items():
            role, _, buffer = name.partition(":")
            src = input_data.get(buffer) if buffer else None
            if _is_device_tensor(src):
                if not src.is_contiguous() or src.numel() * src.element_size() < nbytes:
                    raise ValueError(f"buffer {buffer!r}: device tensor must be contiguous and hold {nbytes} bytes")
                tensor = _flat_bytes(src)
            elif self.arena is not None and role != "constant":
                tensor = self.arena.backing(name, nbytes)
            else:
                tensor = self._tensors.get(name)
                if tensor is None or tensor.numel() < max(1, nbytes):
                    tensor = _new_backing(nbytes)
            self._tensors[name] = tensor
            regions[name] = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        return regions

    def _track(self) -> None:
        """Record every lent tensor on the runtime's stream, so torch's caching allocator waits
        for the launches that read a block before handing it to another tensor."""
        if not self._tensors:
            return
        import torch  # noqa: PLC0415

        stream = torch.cuda.ExternalStream(int(self.executor.stream()))
        for tensor in self._tensors.values():
            tensor.record_stream(stream)

    def rebind(self, input_data: dict) -> None:
        """Re-bind ``input_data`` on an already-built program, re-sizing symbolic-shaped buffers
        to the new runtime dims — the serving path, where one compiled dynamic-seq_len program
        runs request after request.

        Supplied buffers are re-uploaded; un-supplied buffers whose shape carries a symbolic dim
        (scratch/outputs sized by seq_len) re-materialize at the new shape under the same fill
        policy as ``build``; static-shaped un-supplied buffers — the weights — keep their device
        memory untouched. Regions grow when the new layout needs more bytes (the arena keeps the
        older generation alive); captured graphs and descriptors are dropped by the runtime,
        since they bake addresses. Caller must hold ``gpu_lock()``."""
        new_sym = _resolve_symbolic(self.plan, input_data)
        touched = {name for name, value in input_data.items() if not _is_device_tensor(value)}
        touched |= {b.name for b in self.plan.buffers if b.role in ("input", "constant") and b.is_symbolic}
        bindings = _host_bindings(self.plan, input_data, new_sym, only=touched)
        regions = self._provision(new_sym, input_data)
        self.executor.rebind(new_sym, bindings, regions)
        self._track()
        self.sym_values = new_sym

    def set_sym_values(self, values: dict[str, int]) -> None:
        """Set the host symbolic values that resolve launch grids + by-value kernel args,
        WITHOUT re-allocating buffers — they stay at the build (capacity) shape. The serving
        capture path: buffers sized once at capacity, grids + frozen seq_len baked per request
        via :meth:`capture_program_graph`, results sliced to the real shape by
        ``outputs(sym_values=…)``. Errors if any value exceeds the allocated capacity (the caller
        falls back to ``rebind`` above capacity)."""
        self.executor.set_env(dict(values))
        self.sym_values = self.executor.env()

    @contextlib.contextmanager
    def on_torch_stream(self):
        """:meth:`on_stream` for torch's current stream when torch sees the device, else a
        no-op — the bench and run paths, where peer torch work (the eager reference, the
        interleaved torch benches) must stay ordered with the program's launches."""
        torch = _torch()
        if torch is None:
            yield
            return
        with self.on_stream(torch.cuda.current_stream()):
            yield

    @contextlib.contextmanager
    def on_stream(self, stream):
        """Issue every launch and copy inside the block on ``stream`` (a torch CUDA stream) —
        the serving path, where inputs arrive and outputs leave on torch's current stream and
        the work must stay ordered with it. Under torch's own graph capture, :meth:`run_once`
        is the recordable form; the program's own graphs are not."""
        self.executor.set_stream(int(stream.cuda_stream))
        try:
            yield
        finally:
            self.executor.set_stream(None)

    def run_once(self) -> None:
        """Launch every kernel once in program order with no per-launch event record / sync /
        watchdog — the serving hot path (timing semantics live in :meth:`iter_once`). The
        caller's subsequent ``outputs()`` synchronizes."""
        self.executor.run_once()

    def capture_launch_graphs(self, batch_sizes: list[int]) -> None:
        """Capture each launch position's batch into one CUDA graph, so :meth:`iter_once` replays
        it with one call and the event window measures dense GPU work. Unchanged batch sizes are
        a no-op; changed ones re-capture. A failure raises :class:`GraphCaptureError`."""
        try:
            self.executor.capture_launch_graphs([int(b) for b in batch_sizes])
        except HungKernelError:
            raise
        except RuntimeError as exc:
            raise GraphCaptureError(f"per-launch CUDA graph capture failed: {exc}") from exc

    def capture_program_graph(self) -> None:
        """Capture EVERY launch in program order into one CUDA graph at the current symbolic
        environment — the emmy analogue of timing a captured torch forward, and the serving
        replay unit. The runtime keeps one graph per environment (a graph baked at seq_len S
        only replays at S: every kernel's grid and by-value seq_len are frozen by capture),
        least recently used out; a repeated environment is a no-op. A failure raises
        :class:`GraphCaptureError`."""
        try:
            self.executor.capture_program_graph()
        except HungKernelError:
            raise
        except RuntimeError as exc:
            raise GraphCaptureError(f"whole-program CUDA graph capture failed: {exc}") from exc

    def replay_program_graph(self) -> None:
        self.executor.replay_program_graph()

    def upload_prefix(self, input_data: dict) -> None:
        """H2D each supplied input into the contiguous prefix of its capacity buffer: a
        logically ``(1, S, …)`` tensor occupies the first ``S·…`` elements."""
        bindings = _host_bindings(self.plan, input_data, self.sym_values, only=set(input_data))
        for name, data in bindings.items():
            self.executor.bind(name, data)

    def upload_prefix_device(self, input_data: dict) -> None:
        """Device twin of :meth:`upload_prefix`: copy each supplied CUDA tensor into its buffer's
        prefix device-to-device on the current stream — no host hop. A tensor that already IS
        the buffer (a producer's output chained onto this input) is skipped."""
        for name, tensor in input_data.items():
            if not _is_device_tensor(tensor):
                raise TypeError(f"buffer {name!r}: expected a CUDA tensor, got {type(tensor).__name__}")
            tensor = tensor.contiguous()
            self.executor.bind_device(name, tensor.data_ptr(), tensor.numel() * tensor.element_size())

    def time_program_window(self, replays: int) -> float:
        """Per-replay ms of ``replays`` back-to-back whole-program graph replays in one event
        window (:meth:`capture_program_graph` must have run)."""
        return float(self.executor.time_program_window(int(replays), _launch_deadline_ms(self._iters_done, replays)))

    def iter_once(
        self,
        *,
        batch_sizes: list[int] | None = None,
        pre_iter=None,
        per_launch_hook=None,
    ) -> list[float]:
        """Run every launch once. Returns per-launch time in ms, already event-synced before
        return.

        ``batch_sizes[i]`` repeats launch ``i`` ``N`` times inside one CUDA event window so
        per-iter host framing overhead amortizes across launches when the kernel is faster than
        the framing. Returned dt is divided by the batch size so callers always see per-call ms.
        Once :meth:`capture_launch_graphs` ran, each window replays that launch's graph instead.

        ``pre_iter(max_batch_size)`` runs once before the launch loop and inside the GPU lock the
        caller is holding — that's where ``_bench_interleaved`` issues its peer torch backends so
        they share the same warm GPU state emmy measures from.

        ``per_launch_hook(i, launch)`` runs after each launch's stop event has synced.
        :func:`run_program_debug` uses it to snapshot every non-input buffer after each launch.

        The runtime syncs per launch, which makes per-launch attribution accurate — without it,
        one kernel's stop event can slide into a downstream kernel's scheduling window — and
        polls each stop event against the watchdog deadline, so a hung kernel raises
        :class:`HungKernelError` instead of blocking. A 0.0 reading raises too: a real launch
        consumes at least one device cycle, so zero means a no-op launch that must never win a
        benchmark."""
        n = len(self.plan.launches)
        if batch_sizes is None:
            batch_sizes = [1] * n
        if pre_iter is not None:
            pre_iter(max(batch_sizes))
        dts = [0.0] * n
        for i, launch in enumerate(self.plan.launches):
            b = int(batch_sizes[i])
            dts[i] = float(self.executor.time_launch(i, b, _launch_deadline_ms(self._iters_done, b)))
            if per_launch_hook is not None:
                per_launch_hook(i, launch)
        self._iters_done += 1
        return dts

    def _shape(self, name: str, sym_values: dict[str, int] | None = None) -> tuple[int, ...]:
        return self._buffer(name).resolve_shape({**self.sym_values, **(sym_values or {})})

    def _read(self, name: str, sym_values: dict[str, int] | None = None) -> np.ndarray:
        buf = self._buffer(name)
        shape = self._shape(name, sym_values)
        n = math.prod(shape) if shape else 1
        return np.frombuffer(self.executor.read(name), dtype=buf.dtype.np)[:n].reshape(shape).copy()

    def outputs(self, sym_values: dict[str, int] | None = None) -> dict[str, np.ndarray]:
        """Copy every output buffer back to host after every queued launch has completed.
        Caller must hold the GPU lock so peer workers' kernels never interleave with the copy.

        ``sym_values`` (serving's capture path) slices each output to its real-S shape — the
        buffer is allocated at capacity but only the ``resolve_shape(sym_values)`` prefix holds
        the request's result; the rest is unmasked garbage from the oversized allocation."""
        return {name: self._read(name, sym_values) for name in self.plan.outputs}

    def buffer_view(self, name: str, sym_values: dict[str, int] | None = None):
        """A torch view of one buffer's real-shape prefix in its lent memory — no copy. The
        shared buffer is overwritten by the next request's replay, so a caller that keeps the
        result clones it. Requires the program's memory to be torch's (it is whenever torch sees
        the device)."""
        buf = self._buffer(name)
        placement = self.program.layout(self.sym_values)["buffers"][name]
        backing = self._tensors.get(placement["region"])
        if backing is None:
            raise RuntimeError(f"buffer {name!r} lives in runtime-owned memory; device views need torch")
        shape = self._shape(name, sym_values)
        n = math.prod(shape) if shape else 1
        start = placement["offset"]
        flat = backing[start : start + placement["bytes"]].view(_torch_dtype(buf.dtype.np))
        return flat[:n].reshape(shape)

    def output_prefix_device(self, sym_values: dict[str, int] | None = None) -> dict:
        """Device twin of :meth:`outputs`: each output buffer's real-S prefix as a torch view
        with NO host copy — the serving zero-copy path. ``sym_values`` slices to the real shape
        exactly like :meth:`outputs`; without it the whole buffer view is returned."""
        return {name: self.buffer_view(name, sym_values) for name in self.plan.outputs}

    def alias_buffer(self, name: str, tensor) -> None:
        """Point one operand at ``tensor``'s memory: a buffer chained onto another program's (a
        producer's output onto a consumer's input, so the consumer's device upload becomes a
        self-copy skip), or an operand the plan never declares as a buffer — an indirect
        operand's pointer table or selector, which only the caller can supply. A buffer's
        tensor must be contiguous and at least as large as its region; the runtime drops any
        captured graph, since it baked the old address."""
        if not _is_device_tensor(tensor) or not tensor.is_contiguous():
            raise TypeError(f"operand {name!r}: expected a contiguous CUDA tensor")
        flat = _flat_bytes(tensor)
        placement = self.program.layout(self.sym_values)["buffers"].get(name)
        if placement is None:
            self.executor.set_external(name, flat.data_ptr(), flat.numel())
            self._tensors[f"external:{name}"] = flat
        else:
            self.executor.set_region(placement["region"], flat.data_ptr(), flat.numel())
            self._tensors[placement["region"]] = flat
        self._track()

    def release_buffer(self, name: str) -> None:
        """Give a buffer's memory back: an operand the kernels resolve through an indirect table
        and never read directly. The buffer keeps a valid one-byte address."""
        region = self.program.layout(self.sym_values)["buffers"][name]["region"]
        self.executor.release_region(region)
        self._tensors.pop(region, None)

    def regions(self) -> dict[str, tuple[int, int]]:
        """Every region's ``(address, bytes)`` as lent to the runtime (empty when the runtime
        allocates for itself)."""
        return {name: (t.data_ptr(), t.numel() * t.element_size()) for name, t in self._tensors.items()}

    def buffer_nbytes(self, name: str) -> int:
        """The bytes a buffer holds at its allocated capacity."""
        return int(self.executor.buffer(name)[1])

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copy every non-input buffer (scratch + constants + outputs) to host. Used by
        :func:`run_program_debug` to capture every intermediate state for per-launch comparison
        against a reference backend. A scratch buffer read after its last use reflects the
        slab slot's new tenant; each kernel's own output is valid at its launch."""
        return {b.name: self._read(b.name) for b in self.plan.buffers if b.role != "input"}


# ---------------------------------------------------------------------------
# Public entry points: thin shells around CompiledProgram
# ---------------------------------------------------------------------------


def run_program(
    graph: Graph,
    input_data: dict | None = None,
    *,
    pre_run=None,
) -> tuple[RunResult, Any]:
    """Run the lowered graph once, return ``(RunResult, pre_run_result)``.

    ``pre_run`` runs once inside the GPU lock, before emmy's
    kernel launches. Its return value flows through as the tuple's
    second element. Tests use this to compute a torch eager reference
    on the same GPU window the emmy launches will see, so peer-
    worker CUDA activity can't interleave the eager forward with the
    emmy comparison."""
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    with gpu_lock():
        pre_result = pre_run() if pre_run is not None else None
        prog = CompiledProgram.build(graph, input_data)
        with prog.on_torch_stream():
            dts = prog.iter_once()
            outputs = prog.outputs()
    return RunResult(outputs=outputs, time_ms=sum(dts)), pre_result


@dataclass
class DebugResult:
    outputs: dict[str, np.ndarray]
    per_launch: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def run_program_debug(
    graph: Graph,
    input_data: dict | None = None,
    *,
    pre_run=None,
) -> tuple[DebugResult, Any]:
    """Run the graph once, snapshotting every non-input buffer after
    each launch. Returns ``(DebugResult, pre_run_result)`` — same
    ``pre_run`` semantics as :func:`run_program`."""
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    per_launch: dict[int, dict[str, np.ndarray]] = {}
    with gpu_lock():
        pre_result = pre_run() if pre_run is not None else None
        prog = CompiledProgram.build(graph, input_data)
        with prog.on_torch_stream():
            prog.iter_once(per_launch_hook=lambda li, _lc: per_launch.__setitem__(li, prog.snapshot()))
            outputs = prog.outputs()
    return DebugResult(outputs=outputs, per_launch=per_launch), pre_result


def benchmark_program(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    warmup: int = 5,
    num_iters: int | str = 20,
    on_iter=None,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    capture_graphs: bool = True,
) -> BenchmarkResult:
    """Time the graph's launches with per-kernel CUDA events.

    Single loop covers warmup + measurement: the first ``warmup`` iters
    are discarded, the rest are counted toward the result. The
    per-launch ``config.kernel_timeout_ms()`` watchdog (inside
    :meth:`CompiledProgram.iter_once`) runs every iter — warmup or
    measured — so a single hung kernel raises cleanly instead of
    stalling the whole sweep.

    ``num_iters`` accepts an explicit count or the string ``"auto"``.
    In auto mode the loop accumulates measured GPU time until it
    reaches ``_AUTO_BUDGET_MS`` (capped at ``_AUTO_MAX_ITERS`` measured
    iters). For a 7-µs RMSNorm that's ~14k iters; a 1-ms matmul gets
    ~100. The result's per-launch ``time_ms`` is the *median* of
    measured iters (mean was sensitive to single-iter outliers from
    thermal blips and GPU-lock-contention spikes — the autotune
    ``_pick_best_candidate`` selects on the lowest summed latency, so
    noise-driven dips made it pick variants whose post-tune bench was
    slower than the heuristic). Total ``time_ms`` is the sum of
    per-launch medians.

    ``on_iter(batch_size)`` is invoked once at the top of every iter
    inside the GPU lock — that's where ``_bench_interleaved`` runs
    peer torch backends so they time the same number of back-to-back
    calls emmy does per CUDA event window, no warm-vs-cold
    asymmetry.

    ``compile_timeout_s`` bounds NVRTC + alloc + descriptor setup;
    raised inside :meth:`CompiledProgram.build`.

    ``run_timeout_s`` bounds the iter loop on **accumulated GPU time**
    (sum of per-launch CUDA-event measurements), not wall-clock — so
    Python/cupy framing overhead doesn't shrink the budget for tiny
    ops. Catches the gap left by the per-launch ``config.kernel_timeout_ms()``
    watchdog: a variant where every launch fits under the watchdog but
    summed across iters exceeds the budget (e.g. 999 ms × N iters).
    Checked between iters so no in-flight launch is mid-kernel when
    the function raises.

    ``capture_graphs`` (default on) captures each launch's batch into a CUDA
    graph (:meth:`CompiledProgram.capture_launch_graphs`) once batch sizes are
    calibrated, so the event windows measure dense GPU work instead of
    per-launch dispatch gaps. Warmup iters always run uncaptured, which keeps
    the zero-elapsed degenerate-launch guard and the hung-kernel watchdog
    probing real launches before any graph is built. A capture failure is
    non-fatal: the bench logs a warning and continues uncaptured, reporting it
    via ``BenchmarkResult.captured`` — callers pairing this result with peer
    torch timings (``bench_lowered_vs_torch`` / the e2e comparison) use that
    flag to re-run all-or-nothing so one table never mixes semantics, and the
    tune sweep persists it on each ``perf`` row (captured measurements
    supersede wall-semantics ones on write — see ``SearchDB.record_perf``).

    Multi-launch programs additionally get a WHOLE-program time per measured
    iter — one event window around back-to-back replays of a single CUDA
    graph holding every launch in program order
    (:meth:`CompiledProgram.time_program_window`) — reported as
    ``BenchmarkResult.e2e_ms`` / ``e2e_min_ms``. The per-launch windows each
    replay one kernel solo, so their sum misses cross-kernel cache effects;
    only the whole-program window is comparable against a captured torch
    forward. Automatic when capture holds and the program has more than one
    launch (for a single launch the solo window IS the program time, so the
    fields stay ``None`` and nothing is measured twice — the common case for
    the autotune sweep's single-node slices); also ``None`` when capture is
    off or fell back."""
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    target_total_ms, max_measured, auto = _resolve_iter_budget(num_iters)

    with gpu_lock(), contextlib.ExitStack() as stack:
        prog = CompiledProgram.build(graph, input_data, compile_timeout_s=compile_timeout_s)
        stack.enter_context(prog.on_torch_stream())
        n = len(prog.plan.launches)
        batch_sizes = [1] * n
        # Per-launch sample list — kept around to compute the median
        # across measured iters (more robust than the arithmetic mean
        # against thermal blips, GPU-lock-contention spikes, and other
        # one-off outliers the autotune's variant ranking previously
        # got confused by; see ``project_..._noise`` write-ups).
        samples: list[list[float]] = [[] for _ in range(n)]
        measure_e2e = n > 1  # single launch: the solo window IS the program time
        e2e_samples: list[float] = []
        e2e_replays = 0  # calibrated lazily on the first measured iter
        iters_run = 0
        measured = 0
        cumulative_gpu_ms = 0.0  # measured-iter GPU time, for the "auto" stop target
        total_gpu_ms = 0.0  # all-iter GPU time (incl. warmup), for the run-stage budget

        def _try_capture(sizes: list[int]) -> bool:
            """Best-effort capture; a failure logs + continues uncaptured."""
            try:
                prog.capture_launch_graphs(sizes)
            except GraphCaptureError as exc:
                logger.warning("[cuda] %s — continuing with uncaptured (dispatch-inclusive) timing", exc)
                return False
            return True

        if capture_graphs and warmup == 0:
            # No warmup → the calibration below never fires; capture the
            # uncalibrated all-1 batches so measurement is still dense.
            capture_graphs = _try_capture(batch_sizes)
        while True:
            iter_dts = prog.iter_once(batch_sizes=batch_sizes, pre_iter=on_iter)
            iters_run += 1
            total_gpu_ms += sum(iter_dts[i] * batch_sizes[i] for i in range(n))
            # GPU-time run budget: bail if the cumulative GPU time
            # across all iters (warmup + measured) exceeds
            # ``run_timeout_s``. Catches the "every launch is just
            # under the per-launch watchdog" pathology. Counts warmup
            # iters too so a slow kernel can't hide behind warmup
            # discards.
            if run_timeout_s is not None and total_gpu_ms > run_timeout_s * 1000.0:
                raise RuntimeError(f"benchmark run stage exceeded {run_timeout_s:.1f}s of GPU time — variant marked bench_fail")
            if iters_run == warmup:
                batch_sizes = _calibrate_batch_sizes(iter_dts)
                if capture_graphs:
                    # Capture (or re-capture) at the calibrated batch sizes.
                    # The warmup extension below can re-fire this calibration
                    # branch with new batch sizes — ``capture_launch_graphs``
                    # no-ops when they're unchanged and re-captures when not,
                    # so graphs and batches never go out of sync.
                    capture_graphs = _try_capture(batch_sizes)
                # Extend warmup until total warmup GPU time clears the
                # clock-ramp floor. Post-batching, each subsequent
                # warmup iter spends roughly
                # ``sum(iter_dts[i] * batch_sizes[i])`` of GPU time —
                # use the just-measured per-launch dts to estimate how
                # many extra iters are needed.
                if total_gpu_ms < _WARMUP_TARGET_MS:
                    per_iter_ms = sum(iter_dts[i] * batch_sizes[i] for i in range(n))
                    if per_iter_ms > 0:
                        warmup += int(math.ceil((_WARMUP_TARGET_MS - total_gpu_ms) / per_iter_ms))
            if iters_run <= warmup:
                continue
            # Measured iter: store per-launch sample (already
            # normalized to per-launch ms inside ``iter_once``).
            # Reduced via median at the end so a single outlier iter
            # can't shift the result.
            for i in range(n):
                samples[i].append(iter_dts[i])
            cumulative_gpu_ms += sum(iter_dts[i] * batch_sizes[i] for i in range(n))
            measured += 1
            # Whole-program window, one per measured iter — shares the same
            # warm GPU state as the per-launch windows and the ``on_iter``
            # torch closures. Counted toward the run-stage GPU budget but NOT
            # the auto-stop target, so it never starves per-launch sampling.
            if measure_e2e and capture_graphs:
                if e2e_replays == 0:
                    try:
                        prog.capture_program_graph()
                    except GraphCaptureError as exc:
                        logger.warning("[cuda] %s — skipping whole-program e2e timing", exc)
                        measure_e2e = False
                    else:
                        iter_ms = sum(iter_dts)
                        e2e_replays = max(1, int(round(_BATCH_TARGET_MS / iter_ms))) if 0 < iter_ms < _BATCH_TARGET_MS else 1
                if measure_e2e:
                    e2e_dt = prog.time_program_window(e2e_replays)
                    e2e_samples.append(e2e_dt)
                    total_gpu_ms += e2e_dt * e2e_replays
            if measured >= max_measured:
                break
            if auto and cumulative_gpu_ms >= target_total_ms:
                break

    return _samples_to_result(samples, prog.plan.launches, captured=capture_graphs, e2e_samples=e2e_samples)


def _resolve_iter_budget(num_iters: int | str) -> tuple[float, int, bool]:
    """Resolve ``num_iters`` to ``(target_total_ms, max_measured, auto)``."""
    if isinstance(num_iters, str):
        if num_iters != "auto":
            raise ValueError(f"num_iters must be int or 'auto', got {num_iters!r}")
        return (_AUTO_BUDGET_MS, _AUTO_MAX_ITERS, True)
    return (float("inf"), int(num_iters), False)


def _calibrate_batch_sizes(iter_dts: list[float]) -> list[int]:
    """Pick per-position batch sizes so each CUDA event window covers
    ~``_BATCH_TARGET_MS`` of GPU time. Per-position 1 when the kernel
    already exceeds the target — no benefit to batching there."""
    return [max(1, int(round(_BATCH_TARGET_MS / dt))) if 0 < dt < _BATCH_TARGET_MS else 1 for dt in iter_dts]


def _samples_to_result(
    samples: list[list[float]], launches: list[_Launch], *, captured: bool = False, e2e_samples: list[float] | None = None
) -> BenchmarkResult:
    """Collapse per-launch sample lists to a ``BenchmarkResult`` keyed
    on the median of each launch's measured iters."""
    import statistics as _stats  # noqa: PLC0415

    n = len(launches)
    medians = [(_stats.median(samples[i]) if samples[i] else 0.0) for i in range(n)]
    mins = [(min(samples[i]) if samples[i] else 0.0) for i in range(n)]
    per_launch = [
        LaunchTime(
            idx=i,
            kernel_name=launches[i].kernel_name,
            time_ms=medians[i],
            samples=tuple(samples[i]) if samples[i] else None,
        )
        for i in range(n)
    ]
    # ``time_ms`` is the per-launch median (stable for tune's ranking); ``min_ms``
    # is the per-launch best-case (least OS/thermal noise — what ``run --bench``
    # reports, matching tune's min-over-variants reporting).
    return BenchmarkResult(
        time_ms=sum(medians),
        min_ms=sum(mins),
        num_launches=n,
        per_launch=per_launch if per_launch else None,
        captured=captured,
        e2e_ms=_stats.median(e2e_samples) if e2e_samples else None,
        e2e_min_ms=min(e2e_samples) if e2e_samples else None,
    )


# ---------------------------------------------------------------------------
# Subprocess-isolated benchmark worker (single async transport)
# ---------------------------------------------------------------------------


class BenchWorkerJobError(RuntimeError):
    """A worker job that ran and failed (``ok: False`` response — the child is alive).
    ``cache_miss`` marks the one retryable kind: a job referenced a ``run_inputs_key``
    a freshly-respawned child no longer holds. ``compile_budget`` marks a
    :class:`CompileBudgetExceeded` raised in the child — the exception CLASS cannot cross the
    process boundary (the protocol carries ``error`` as a string), so the child flags the kind
    and the parent rebuilds the distinction here."""

    def __init__(self, message: str, *, cache_miss: bool = False, compile_budget: bool = False) -> None:
        super().__init__(message)
        self.cache_miss = cache_miss
        self.compile_budget = compile_budget


class _AsyncBenchWorker:
    """The parent-side transport for the SIGKILL-able ``_bench_worker`` subprocess —
    the **single** isolated-bench transport (the old sync ``_BenchWorker`` is gone).

    Lets the parent enforce a hard wall-clock cap on a bench: if the worker doesn't
    respond within ``wall_timeout_s``, the parent SIGKILLs it. The dirty CUDA stream
    (and any kernels still queued behind a hung launch) dies with the process, so the
    *next* bench starts on a clean device — fixing the "autotune hangs on the variant
    AFTER a bench_fail" pathology. The worker imports cupy lazily on its first
    request, so spawn cost is just Python startup (~0.2 s).

    Drives the ``_bench_worker`` protocol (``<8-byte LE length><pickle>``, both
    directions) over asyncio streams, so one event loop can keep N device-pinned
    workers benching concurrently — the per-kernel multi-GPU autotune path
    (``two_level.TwoLevelStrategy``). The deployable ``--bench`` comparison awaits
    ``benchmark_compare_isolated_async`` over a one-shot instance (via
    ``_run_job_oneshot``); the autotune sweep awaits a persistent instance per GPU
    directly via ``benchmark_program_isolated_async``.

    Pin a worker to a physical GPU with ``device_id``: the spawn env gets
    ``CUDA_VISIBLE_DEVICES=<id>`` (so the child's logical device 0 *is* that
    GPU — every argumentless ``cp.cuda.Device()`` in the child resolves
    correctly with no other call-site change) and, when a base
    ``EMMY_GPU_LOCK`` is set, a per-device lock path so workers on
    different GPUs take distinct ``FileLock``s instead of serialising. The
    env overlay rides the child only — the parent's ``os.environ`` is never
    mutated (it's shared by every slot on the one event-loop thread).

    The wall-clock cap is :func:`asyncio.wait_for`; on overrun the child is
    SIGKILLed and respawned on the next bench."""

    _WORKER_MODULE = "emmy.compiler.backend.cuda._bench_worker"
    _STDERR_TAIL_CHARS = 4000
    _ATTEMPTS = 2

    def _command(self) -> list[str]:
        return [_sys.executable, "-m", self._WORKER_MODULE]

    @staticmethod
    def _encode(request: dict) -> bytes:
        from emmy.compiler.pipeline.search.space import FAST_MATH, precision_pin  # noqa: PLC0415

        return pickle.dumps({**request, "fast_math": precision_pin(FAST_MATH)}, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _decode(body: bytes) -> dict:
        return pickle.loads(body)

    def __init__(self, *, device_id: int | None = None) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._device_id = device_id
        # Bounded tail of the CURRENT child's stderr, fed by a background drain task. A
        # chatty child (HF shard-download progress, nvcc warnings) would otherwise fill
        # the ~64 KB stderr pipe and block mid-job — which the parent misreads as a
        # wall-timeout hang — and the tail is the diagnostic every failure path wants.
        self._stderr_tail = ""
        self._stderr_task: asyncio.Task | None = None
        # ``run_inputs_key``s this child has cached (see ``benchmark_pinned_isolated_async``).
        # Cleared on every (re)spawn — a fresh child holds no cache.
        self.cached_input_keys: set[str] = set()

    def _child_env(self) -> dict:
        env = dict(_os.environ)
        if self._device_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self._device_id)
            from emmy import config  # noqa: PLC0415

            base = config.gpu_lock_path()
            if base:
                # Per-device lock so concurrent device-pinned workers don't
                # serialise on one FileLock (the lock is taken inside the child).
                env["EMMY_GPU_LOCK"] = f"{base}-{self._device_id}"
        return env

    async def _spawn(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._child_env(),
        )
        self._stderr_tail = ""
        self._stderr_task = asyncio.ensure_future(self._drain_stderr(self._proc))
        self.cached_input_keys.clear()
        logger.info("[bench-worker] spawned (async) pid=%s device=%s", self._proc.pid, self._device_id)

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """Continuously drain the child's stderr into the bounded tail. Runs for the
        child's whole life (exits on EOF, i.e. child exit / SIGKILL) so the pipe can never
        fill and block the child mid-job."""
        try:
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    return
                self._stderr_tail = (self._stderr_tail + chunk.decode(errors="replace"))[-self._STDERR_TAIL_CHARS :]
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the drain is best-effort diagnostics
            return

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def close(self) -> None:
        """Terminate the worker (driver teardown). The subprocess transport is
        reaped when the event loop closes."""
        self._kill()

    async def aclose(self) -> None:
        """Terminate the worker and await its reap — for one-shot bridges that
        spawn + run + tear down within a single ``asyncio.run`` (so no orphaned
        subprocess transport survives the loop)."""
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        if self._stderr_task is not None:
            # The drain ends on the killed child's stderr EOF; reap it so no pending
            # task survives the caller's event loop.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._stderr_task, timeout=2.0)
            self._stderr_task = None

    @staticmethod
    async def _death_reason(proc) -> str:
        """How the child died, for the EOF diagnostics.

        A worker that raises returns the exception to the parent (the request loop catches
        ``BaseException`` and answers with a traceback), so an EOF means the process went down
        WITHOUT answering — a signal, or a silent ``return`` out of the request loop. Those write
        nothing to stderr, which is why the tail is routinely empty and the exit status is the only
        evidence there is. Reap briefly rather than reading ``returncode`` directly: at the moment
        the pipe breaks the child is usually dead but not yet awaited, so the attribute is still
        ``None``."""
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=2.0)
        except (TimeoutError, ProcessLookupError):
            return "child still unreaped"
        if rc is None:
            return "no exit status"
        if rc < 0:
            import signal as _signal  # noqa: PLC0415 — only needed on this failure path

            try:
                return f"killed by {_signal.Signals(-rc).name}"
            except ValueError:
                return f"killed by signal {-rc}"
        return f"child exited rc={rc}"

    async def _stderr_snapshot(self) -> str:
        """The drained stderr tail, letting the drain task flush briefly first (after a
        kill it ends at EOF almost immediately)."""
        if self._stderr_task is not None and not self._stderr_task.done():
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=0.5)
        return self._stderr_tail

    async def run_job(self, request_obj: dict, *, wall_timeout_s: float) -> dict:
        try:
            return await self._run_job(request_obj, wall_timeout_s=wall_timeout_s)
        except BenchWorkerJobError:
            raise  # The response's retirement flag decides whether the context is healthy.
        except BaseException:
            await self.aclose()
            raise

    async def _run_job(self, request_obj: dict, *, wall_timeout_s: float) -> dict:
        """Send one request, read the response within ``wall_timeout_s`` (else SIGKILL
        + raise ``RuntimeError``), and return the unpickled response. A stale-worker
        race on send respawns and retries once; a response-side timeout is a hard
        error. A response-side EOF (the child went down mid-job without answering) respawns
        and retries ONCE after a short drain grace — see the handler for why. A response
        flagged ``_retire_worker`` (a hung kernel or a poisoned context in the child) retires
        the child first — SIGKILL + reap — so the next request respawns clean."""
        request = self._encode(request_obj)
        frame = len(request).to_bytes(8, "little") + request
        deadline = _time_module.perf_counter() + wall_timeout_s
        for attempt in range(self._ATTEMPTS):
            if self._proc is None or self._proc.returncode is not None:
                await self._spawn()
            assert self._proc is not None  # for type narrowing
            proc = self._proc
            try:
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                proc.stdin.write(frame)
                await asyncio.wait_for(proc.stdin.drain(), timeout=remaining)
            except TimeoutError as exc:
                await self.aclose()
                raise RuntimeError(
                    f"bench worker did not accept the request within {wall_timeout_s:.1f}s wall budget — SIGKILL'd, stream cleaned"
                    f"{self._tail_suffix()}"
                ) from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                await self.aclose()
                if attempt + 1 == self._ATTEMPTS:
                    raise RuntimeError(f"bench worker died during request send: {exc}{self._tail_suffix()}") from exc
                logger.info("[bench-worker] stale async worker on send (%s) — respawning", exc)
                continue

            try:
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                header = await asyncio.wait_for(proc.stdout.readexactly(8), timeout=remaining)
                n = int.from_bytes(header, "little")
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                body = await asyncio.wait_for(proc.stdout.readexactly(n), timeout=remaining)
            except TimeoutError as exc:
                await self.aclose()
                raise RuntimeError(
                    f"bench worker exceeded {wall_timeout_s:.1f}s wall budget — SIGKILL'd, stream cleaned{self._tail_suffix()}"
                ) from exc
            except asyncio.IncompleteReadError as exc:
                stderr_tail = await self._stderr_snapshot()
                death = await self._death_reason(proc)
                await self.aclose()
                if attempt + 1 == self._ATTEMPTS:
                    raise RuntimeError(f"bench worker EOF before response ({death}); stderr tail: {stderr_tail}") from exc
                # A mid-job EOF means the child went down without answering (a crash, a signal).
                # Right after a SIGKILL'd predecessor (a wall kill, or a retired hung child), the
                # dead child's zombie context can still hold the GPU while the driver tears it
                # down, taking an INNOCENT first launch on the fresh child down with it — a
                # transient the golden refresh sweeps kept hitting on the row right after a
                # hang. One respawn + retry after a short drain grace tells that apart from the
                # config's own crash (the same row replays clean once the zombie context is
                # gone); a second EOF fails loudly.
                logger.info("[bench-worker] child EOF'd mid-job (%s) — draining the device and retrying once%s", death, self._tail_suffix())
                await asyncio.sleep(min(2.0, max(0.0, deadline - _time_module.perf_counter() - 1.0)))
                continue

            resp = self._decode(body)
            if resp.pop("_retire_worker", False):
                # The child's verdict that its context is done for: a hung kernel (a watchdog
                # failure, or a greedy timing that hung after its same-input reference completed)
                # or a sticky error. Retire it HERE, SIGKILL + reap — the same teardown a wall
                # overrun takes — rather than trust its own exit: a hung kernel stays resident until
                # its context dies and the interpreter's CUDA teardown blocks behind it, so the
                # child left alone is a zombie holding the GPU, and the next request wedges on it
                # before its first launch and is priced as a wall failure. A queued /
                # non-terminating kernel must never share a context with the next candidate or a
                # pinned row.
                await self.aclose()
            if not resp.get("ok"):
                # The in-child traceback (and the stderr tail, where CLI-style helpers log
                # their cause before exiting) would otherwise be silently discarded.
                if resp.get("traceback"):
                    logger.error("[bench-worker] job failed in the child; traceback:\n%s%s", resp["traceback"], self._tail_suffix())
                raise BenchWorkerJobError(
                    f"bench worker error: {resp.get('error', '?')}",
                    cache_miss=bool(resp.get("cache_miss")),
                    compile_budget=bool(resp.get("compile_budget")),
                )
            return resp
        raise RuntimeError("bench worker unreachable")  # both attempts exhausted (defensive)

    async def warmup(self, *, wall_timeout_s: float = 60.0) -> None:
        """Initialize the child CUDA context outside a candidate's wall budget."""
        response = await self.run_job({"worker_warmup": True}, wall_timeout_s=wall_timeout_s)
        if not response.get("warmed"):
            raise RuntimeError("bench worker did not acknowledge CUDA warmup")

    def _tail_suffix(self) -> str:
        """The drained stderr tail as an error-message suffix ('' when the child was quiet)."""
        return f"; child stderr tail:\n{self._stderr_tail}" if self._stderr_tail.strip() else ""


async def benchmark_program_isolated_async(
    graph: Graph,
    *,
    worker: _AsyncBenchWorker,
    wall_timeout_s: float,
    warmup: int = 5,
    num_iters: int | str = 20,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    nvcc_flags: str | None = None,
    capture_graphs: bool = True,
) -> BenchmarkResult:
    """Wall-time-bounded ``benchmark_program`` in a subprocess, benching through a
    caller-supplied device-pinned ``worker`` so one event loop can drive N GPUs
    concurrently — the autotune sweep's transport. The in-worker
    ``compile_timeout_s`` / ``run_timeout_s`` budgets apply, and ``wall_timeout_s`` is
    the SIGKILL backstop for a kernel that keeps the GPU busy past them. No ``on_iter``
    (interleaved ``run --bench`` benches in-process via ``benchmark_program``)."""
    resp = await worker.run_job(
        {
            "graph": graph,
            "nvcc_flags": nvcc_flags,
            "torch_spec": None,  # no torch comparison — pure emmy bench
            "kwargs": {
                "warmup": warmup,
                "num_iters": num_iters,
                "compile_timeout_s": compile_timeout_s,
                "run_timeout_s": run_timeout_s,
                "capture_graphs": capture_graphs,
            },
        },
        wall_timeout_s=wall_timeout_s,
    )
    return resp["result"]


async def benchmark_pinned_isolated_async(
    graph: Graph,
    *,
    worker: _AsyncBenchWorker,
    wall_timeout_s: float,
    run_inputs: dict | None = None,
    run_inputs_key: str | None = None,
    warmup: int = 5,
    num_iters: int | str = 20,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
) -> tuple[BenchmarkResult, dict | None]:
    """One ``run --bench`` pinned-row job through the persistent ``worker``: an optional
    single execution on ``run_inputs`` (the greedy run's inputs — the wrong-answer gate's
    measurement side, outputs returned for the parent to compare) followed by the emmy-only
    bench. One job per row over one persistent worker per run session; a hung kernel dies
    with the SIGKILL'd child and the next row's job respawns a clean context.

    ``run_inputs_key`` (a session-unique token) lets the reference inputs cross the pipe
    ONCE per child instead of per row — hundreds of MB on the big ``--code`` shapes. The
    child caches them under the key; later rows send the key alone. The worker tracks which
    keys the CURRENT child holds (``cached_input_keys``, cleared on respawn), and a
    cache-miss response — a respawn raced the tracking — retries once with the inputs
    included."""

    def _request(with_inputs: bool) -> dict:
        return {
            "graph": graph,
            "torch_spec": None,
            "run_inputs": run_inputs if with_inputs else None,
            "run_inputs_key": run_inputs_key,
            "kwargs": {
                "warmup": warmup,
                "num_iters": num_iters,
                "compile_timeout_s": compile_timeout_s,
                "run_timeout_s": run_timeout_s,
            },
        }

    send_inputs = run_inputs is not None and (run_inputs_key is None or run_inputs_key not in worker.cached_input_keys)
    try:
        resp = await worker.run_job(_request(send_inputs), wall_timeout_s=wall_timeout_s)
    except BenchWorkerJobError as exc:
        if not (exc.cache_miss and run_inputs is not None):
            raise
        resp = await worker.run_job(_request(True), wall_timeout_s=wall_timeout_s)
    if run_inputs_key is not None and run_inputs is not None:
        worker.cached_input_keys.add(run_inputs_key)
    return resp["result"], resp.get("run_outputs")


async def benchmark_compare_worker_async(
    *,
    worker: _AsyncBenchWorker,
    lowered: Graph,
    torch_spec: tuple,
    bench_backends: str,
    wall_timeout_s: float,
    warmup: int,
    iters: int,
    seed: int,
    accuracy: bool = False,
    want_ref: bool = False,
    strict_accuracy: bool = False,
) -> dict:
    """``run --bench``'s greedy-row transport: the same comparison job as
    :func:`benchmark_compare_isolated_async` but over a caller-supplied persistent
    ``worker`` (shared with the pinned-row jobs — one worker per run session) and with the
    run path's extras: ``accuracy`` (the in-child real-input emmy-vs-eager verdict) and
    ``want_ref`` (that run's ``(inputs, outputs)`` for the pinned rows' wrong-answer gate).
    Returns the normalized response dict — keys ``results`` / ``result`` /
    ``torch_available`` / ``captured`` / ``accuracy_error`` / ``run_io`` plus the optional
    ``greedy_error`` / ``reference_run_us`` when an embedded Loop reference completed before
    its repeated greedy timing failed."""
    resp = await worker.run_job(
        {
            "graph": lowered,
            "torch_spec": torch_spec,
            "bench_backends": bench_backends,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "accuracy": accuracy,
            "want_ref": want_ref,
            "strict_accuracy": strict_accuracy,
        },
        wall_timeout_s=wall_timeout_s,
    )
    return {
        "results": resp["results"],
        "result": resp["result"],
        "torch_available": resp["torch_available"],
        "captured": resp.get("captured", False),
        "accuracy_error": resp.get("accuracy_error"),
        "run_io": resp.get("run_io"),
        "greedy_error": resp.get("greedy_error"),
        "reference_run_us": resp.get("reference_run_us"),
        "sym_env": resp.get("sym_env"),
        "correctness": resp.get("correctness"),
    }


async def _run_job_oneshot(request_obj: dict, *, wall_timeout_s: float, device_id: int | None = None) -> dict:
    """Spawn a fresh ``_AsyncBenchWorker``, run one job, tear it down.
    The transport for the synchronous one-shot bridges below — they each wrap this
    in ``asyncio.run`` (the worker's streams bind to the loop, so it can't persist
    across ``asyncio.run`` calls; the per-call ~0.2 s spawn is negligible against a
    deployable ``--bench``). ``device_id`` keeps the comparison on the selected
    tune GPU instead of silently falling back to ordinal 0."""
    worker = _AsyncBenchWorker(device_id=device_id)
    try:
        return await worker.run_job(request_obj, wall_timeout_s=wall_timeout_s)
    finally:
        await worker.aclose()


async def benchmark_compare_isolated_async(
    *,
    lowered: Graph,
    torch_spec: tuple,
    bench_backends: str,
    wall_timeout_s: float,
    warmup: int,
    iters: int,
    seed: int,
    nvcc_flags: str | None = None,
    device_id: int | None = None,
) -> tuple:
    """Run the deployable eager / torch.compile / emmy comparison in the
    SIGKILL-able worker, awaiting a fresh one-shot :class:`_AsyncBenchWorker`
    (the one transport).

    Unlike the emmy-only autotune bench, the deployable comparison interleaves emmy with
    real torch in one process and so couldn't be isolated before — a hung generated kernel wedged
    the whole run. This ships the *entire* comparison to the worker: a hung kernel hangs the child,
    which the parent SIGKILLs on ``wall_timeout_s``, freeing the device and leaving the parent clean.

    ``torch_spec`` rebuilds the torch side **in the child** (no live module crosses the pipe), reusing
    the same core functions the in-process path uses:

    - ``("trace_args", {code, input, adapter, layer, seq_len, dynamic})`` → ``load_or_trace`` rebuilds the real
      module (an HF model id or a ``--code`` expression), benched via ``bench_full_model_real``.
    - ``("frontend_graph", Graph | None)`` → ``bench_lowered_vs_torch`` (per-kernel reproducer; ``None``
      benches emmy-only when the graph isn't torch-runnable).

    Returns ``(results, bench, torch_available, captured, accuracy_error)`` — the shape
    ``bench_lowered_vs_torch`` returns (``captured``: all backends were timed under CUDA graph
    capture; False means the all-or-nothing fallback ran and the timings include host dispatch).
    ``accuracy_error`` is the non-fatal eager-reference verdict for a frontend reproducer."""
    resp = await _run_job_oneshot(
        {
            "graph": lowered,
            "nvcc_flags": nvcc_flags,
            "torch_spec": torch_spec,
            "bench_backends": bench_backends,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
        },
        wall_timeout_s=wall_timeout_s,
        device_id=device_id,
    )
    return resp["results"], resp["result"], resp["torch_available"], resp.get("captured", False), resp.get("accuracy_error")
