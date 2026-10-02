"""JIT-compile one generated kernel with llvmlite and launch it on a thread pool."""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import llvmlite.binding as llvm
import numpy as np

from emmy.compiler.backend.cpu.codegen import Plan

_PART = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p)
_FINISH = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p)


@cache
def _initialize() -> None:
    try:
        llvm.initialize()
    except RuntimeError:
        pass  # llvmlite >= 0.44 initializes itself and raises here
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


@cache
def pool(threads: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=threads, thread_name_prefix="emmy-cpu")


def default_threads() -> int:
    return os.cpu_count() or 1


class CpuKernel:
    """A compiled kernel. ctypes releases the GIL around each call, so the chunks run in parallel."""

    def __init__(self, ir_text: str, plan: Plan, buffers: tuple[str, ...]):
        _initialize()
        # The engine takes ownership of its target machine, so every kernel gets its own.
        tm = llvm.Target.from_default_triple().create_target_machine(
            cpu=llvm.get_host_cpu_name(), features=llvm.get_host_cpu_features().flatten(), opt=3
        )
        module = llvm.parse_assembly(ir_text)
        module.verify()
        tuning = llvm.create_pipeline_tuning_options(speed_level=3)
        tuning.loop_vectorization = tuning.slp_vectorization = tuning.loop_interleaving = tuning.loop_unrolling = True
        builder = llvm.create_pass_builder(tm, tuning)
        builder.getModulePassManager().run(module, builder)
        self.engine = llvm.create_mcjit_compiler(module, tm)
        self.engine.finalize_object()
        self.plan, self.buffers = plan, buffers
        self.part = _PART(self.engine.get_function_address("part"))
        self.finish = _FINISH(self.engine.get_function_address("finish")) if plan.mode == "reduce" else None

    def __call__(self, arrays: dict[str, np.ndarray], sizes: dict[str, int], threads: int) -> None:
        """Run on ``arrays`` (every buffer by name, C-contiguous, storage dtype); outputs are written in place."""
        plan = self.plan
        bufs = (ctypes.c_void_p * len(self.buffers))(*(arrays[b].ctypes.data for b in self.buffers))
        size_vals = (ctypes.c_int64 * max(1, len(plan.sizes)))(*(int(sizes[n]) for n in plan.sizes))
        extent = plan.extent if isinstance(plan.extent, int) else int(sizes[plan.extent])
        n = max(1, min(threads, extent)) if plan.parallel else 1
        bounds = [(extent * t // n, extent * (t + 1) // n) for t in range(n)]
        partials = np.zeros(n * plan.partial_floats, np.float32) if plan.mode == "reduce" else None
        base = partials.ctypes.data if partials is not None else 0
        step = plan.partial_floats * 4

        def run(t: int) -> None:
            lo, hi = bounds[t]
            self.part(bufs, lo, hi, base + t * step if base else None, size_vals)

        if n == 1:
            run(0)
        else:
            list(pool(threads).map(run, range(n)))
        if self.finish is not None:
            self.finish(bufs, base, n, size_vals)
