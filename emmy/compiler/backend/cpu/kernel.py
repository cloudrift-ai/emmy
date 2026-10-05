"""JIT-compile one generated kernel with llvmlite and launch it on a thread pool."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cache, cached_property

import llvmlite.binding as llvm
import numpy as np

from emmy.compiler.backend.cpu.codegen import Plan

_PART = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p)
_FINISH = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p)


@cache
def initialize_llvm() -> None:
    try:
        llvm.initialize()
    except RuntimeError:
        pass  # llvmlite >= 0.44 initializes itself and raises here
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


# The Rust runtime's chunk count for a split reduction, so both paths add the same partials in the same order.
REDUCE_CHUNKS = 32


@cache
def pool(threads: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=threads, thread_name_prefix="emmy-cpu")


@cache
def default_threads() -> int:
    """The performance cores: an efficiency core in a lockstep launch holds every other thread up."""
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=False)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    return os.cpu_count() or 1


class CpuKernel:
    """A generated kernel, JIT-compiled on its first call. ctypes releases the GIL around each call, so the chunks
    run in parallel."""

    def __init__(self, ir_text: str, plan: Plan, buffers: tuple[str, ...]):
        self.ir, self.plan, self.buffers = ir_text, plan, buffers

    @cached_property
    def _compiled(self):
        """The JIT engine and the kernel's ``part`` and ``finish`` (``None`` unless it reduces); the engine must
        outlive the function pointers."""
        initialize_llvm()
        # The engine takes ownership of its target machine, so every kernel gets its own.
        tm = llvm.Target.from_default_triple().create_target_machine(
            cpu=llvm.get_host_cpu_name(), features=llvm.get_host_cpu_features().flatten(), opt=3
        )
        module = llvm.parse_assembly(self.ir)
        module.verify()
        tuning = llvm.create_pipeline_tuning_options(speed_level=3)
        tuning.loop_vectorization = tuning.slp_vectorization = tuning.loop_interleaving = tuning.loop_unrolling = True
        builder = llvm.create_pass_builder(tm, tuning)
        builder.getModulePassManager().run(module, builder)
        engine = llvm.create_mcjit_compiler(module, tm)
        engine.finalize_object()
        part = _PART(engine.get_function_address("part"))
        finish = _FINISH(engine.get_function_address("finish")) if self.plan.mode == "reduce" else None
        return engine, part, finish

    def __call__(self, arrays: dict[str, np.ndarray], sizes: dict[str, int], threads: int) -> None:
        """Run on ``arrays`` (every buffer by name, C-contiguous, storage dtype); outputs are written in place."""
        plan = self.plan
        _, part, finish = self._compiled
        bufs = (ctypes.c_void_p * len(self.buffers))(*(arrays[b].ctypes.data for b in self.buffers))
        size_vals = (ctypes.c_int64 * max(1, len(plan.sizes)))(*(int(sizes[n]) for n in plan.sizes))
        extent = plan.extent if isinstance(plan.extent, int) else int(sizes[plan.extent])
        n = max(1, min(REDUCE_CHUNKS if plan.mode == "reduce" else threads, extent)) if plan.parallel else 1
        bounds = [(extent * t // n, extent * (t + 1) // n) for t in range(n)]
        partials = np.zeros(n * plan.partial_floats + 32, np.float32) if plan.mode == "reduce" else None
        base = (partials.ctypes.data + 127) // 128 * 128 if partials is not None else 0
        step = plan.partial_floats * 4

        def run(t: int) -> None:
            lo, hi = bounds[t]
            part(bufs, lo, hi, base + t * step if base else None, size_vals)

        if n == 1:
            run(0)
        else:
            list(pool(threads).map(run, range(n)))
        if finish is not None:
            finish(bufs, base, n, size_vals)
