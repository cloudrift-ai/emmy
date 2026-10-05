"""Run a compiled CPU program through the Rust runtime instead of the Python graph walk.

Every kernel's LLVM IR is linked into one module, its ``part``/``finish`` renamed
``<kernel>_part``/``<kernel>_finish``, optimized, emitted as one object file and linked into a
shared library cached by content. The program becomes an :class:`ExecutionPlan` with
``backend="cpu"``: the same buffers, constants and symbolic bindings a CUDA plan carries, and one
launch per kernel whose ``cpu`` section says how it splits across threads.
``emmy_runtime.CpuExecutor`` lays the plan out and runs it on its own thread pool.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import llvmlite.binding as llvm
import numpy as np

from emmy import config
from emmy.compiler.backend.binding import host_bindings, resolve_symbolic, with_generated_constants
from emmy.compiler.backend.cpu.codegen import Unsupported
from emmy.compiler.backend.cpu.kernel import initialize_llvm
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec, plan_to_dict, weight_specs
from emmy.compiler.ir.base import ConstantOp, InputOp

if TYPE_CHECKING:
    from emmy.compiler.backend.cpu.backend import CpuProgram


def build_library(kernels: dict[str, str]) -> Path:
    """One shared library exporting every kernel in ``kernels`` (name → LLVM IR text)."""
    initialize_llvm()
    tm = llvm.Target.from_default_triple().create_target_machine(
        cpu=llvm.get_host_cpu_name(), features=llvm.get_host_cpu_features().flatten(), opt=3, reloc="pic"
    )
    module = None
    for name, ir_text in kernels.items():
        piece = llvm.parse_assembly(ir_text)
        for fn in piece.functions:
            if not fn.is_declaration:
                fn.name = f"{name}_{fn.name}"
        if module is None:
            module = piece
        else:
            module.link_in(piece)
    if module is None:
        raise Unsupported("no kernels")
    module.verify()
    key = hashlib.sha256(f"{tm.triple}|{llvm.get_host_cpu_name()}|{module}".encode()).hexdigest()[:32]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    path = config.cpu_kernel_cache_dir() / f"{key}{suffix}"
    if path.is_file():
        return path
    tuning = llvm.create_pipeline_tuning_options(speed_level=3)
    tuning.loop_vectorization = tuning.slp_vectorization = tuning.loop_interleaving = tuning.loop_unrolling = True
    builder = llvm.create_pass_builder(tm, tuning)
    builder.getModulePassManager().run(module, builder)
    linker = shutil.which("cc") or shutil.which("clang")
    if linker is None:
        raise Unsupported("no C compiler driver to link the kernel library")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Link beside the cache entry, then rename: on one filesystem the rename is atomic, so a concurrent build
    # never sees a partial library.
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        obj = Path(tmp) / "kernels.o"
        obj.write_bytes(tm.emit_object(module))
        staged = Path(tmp) / path.name
        linked = subprocess.run([linker, "-shared", "-o", str(staged), str(obj), "-lm"], capture_output=True, text=True)
        if linked.returncode != 0:
            raise Unsupported(f"linking the kernel library failed: {linked.stderr.strip()}")
        staged.replace(path)
    return path


def plan_for(program: CpuProgram) -> tuple[ExecutionPlan, dict[str, str]]:
    """The CPU execution plan of ``program`` and the LLVM IR of each of its kernels, by kernel name."""
    graph = program.graph
    buffers = [
        BufferSpec(name=buf, shape=tuple(t.shape), dtype=t.dtype, role=graph.buffer_role(buf))
        for node in graph.nodes.values()
        for buf, t in zip(node.buffer_names(), node.outputs, strict=True)
    ]
    launches, kernels, sources = [], {}, {}
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if isinstance(node.op, (InputOp, ConstantOp)):
            continue
        kernel = program.kernels.get(nid)
        if kernel is None:
            raise Unsupported(f"node {nid} has no native kernel ({program.fallbacks.get(nid, type(node.op).__name__)})")
        name = f"k{len(launches)}"
        plan = kernel.plan
        launches.append(
            LaunchSpec(
                node_id=nid,
                kernel_name=name,
                arg_names=tuple(kernel.buffers),
                grid=(),
                block=(),
                smem_bytes=0,
                zero_outputs=tuple(node.op.outputs),
                runtime_args=plan.sizes,
                writes=node.buffer_names(),
                cpu={"mode": plan.mode, "extent": plan.extent, "partial_floats": plan.partial_floats, "parallel": plan.parallel},
            )
        )
        kernels[name] = KernelSpec()
        sources[name] = kernel.ir
    execution = ExecutionPlan(
        backend="cpu",
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        buffers=buffers,
        constants={nid: float(op.value) for nid, op in graph.constant_ops() if op.value is not None},
        runtime_constants={nid: op.context_value for nid, op in graph.constant_ops() if op.context_value is not None},
        launches=launches,
        kernels=kernels,
        weights=weight_specs(graph),
        symbolic_bindings=graph.symbolic_bindings(),
        symbolic_hints=graph.symbolic_hints(),
    )
    return execution, sources


class NativeProgram:
    """A CPU plan and its kernel library, run by one ``emmy_runtime.CpuExecutor``."""

    def __init__(self, program: CpuProgram, threads: int):
        try:
            from emmy import emmy_runtime  # noqa: PLC0415
        except ImportError as exc:
            raise Unsupported("the runtime extension is not built") from exc

        self.plan, sources = plan_for(program)
        self.library = build_library(sources)
        self.runtime = emmy_runtime.Program(json.dumps(plan_to_dict(self.plan)))
        self.threads = threads
        self.executor = None

    def bind(self, input_data: dict) -> None:
        """Lay the plan out for ``input_data``'s sizes and upload its inputs and constants."""
        from emmy import emmy_runtime  # noqa: PLC0415

        feed = with_generated_constants(self.plan, input_data)
        filled = {*self.plan.constants, *self.plan.runtime_constants}
        needed = [b.name for b in self.plan.buffers if b.role == "input" or (b.role == "constant" and b.name not in filled)]
        missing = [name for name in needed if name not in feed]
        if missing:
            raise KeyError(f"no data supplied for {', '.join(missing)}")
        sizes = resolve_symbolic(self.plan, feed)
        for buf in self.plan.buffers:
            value = feed.get(buf.name)
            n = math.prod(buf.resolve_shape(sizes))
            if value is not None and np.size(value) == 1 and n != 1:
                feed[buf.name] = np.broadcast_to(np.asarray(value).reshape(()), buf.resolve_shape(sizes))
        bindings = host_bindings(self.plan, feed, sizes)
        if self.executor is None:
            self.executor = emmy_runtime.CpuExecutor(self.runtime, str(self.library), bindings, sizes, self.threads)
        else:
            self.executor.rebind(sizes, bindings)

    def run_once(self) -> None:
        self.executor.run_once()

    def outputs(self) -> dict[str, np.ndarray]:
        out = {}
        specs = {b.name: b for b in self.plan.buffers}
        for name in self.plan.outputs:
            _, _, shape = self.executor.buffer(name)
            n = math.prod(shape)
            out[name] = np.frombuffer(self.executor.output(name), dtype=specs[name].dtype.np)[:n].reshape(shape).copy()
        return out
