"""Launch one kernel written as Kernel IR text: ``compile_ptx``, ptxas and the runtime, on the card of the process."""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from emmy.emmy_ir import compile_ptx, parse_body

from emmy.compiler import dtype as dtypes
from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.backend.cuda.program import CompiledProgram
from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import DataType
from emmy.compiler.ir.cuda import TmaDescMeta
from emmy.compiler.ir.kernel.interface import Interface, Param
from tests.compiler.helpers import device_compute_capability

#: The ptxas levels every kernel is launched at: its default, and no optimization, where a register read before
#: it is defined or an instruction form ptxas rewrites away would show.
LEVELS = (None, "-O0")

DTYPES = {np.dtype(t.np): t for t in (dtypes.F32, dtypes.F16, dtypes.I32, dtypes.U32, dtypes.U8)}


def target() -> tuple[int, str]:
    """The compute capability of the card as ``compile_ptx`` and ptxas take it."""
    major, minor = device_compute_capability()
    return major * 10 + minor, f"sm_{major}{minor}"


def assemble(ptx: str, level: str, directory: Path, arch_specific: bool = False) -> str:
    """Assemble ``ptx`` at ``level`` into ``directory`` and return the key the cubin is filed under."""
    key = hashlib.sha1(f"{level}\0{ptx}".encode()).hexdigest()
    source = directory / f"{key}.ptx"
    source.write_text(ptx)
    ptxas = str(Path(nvcc.nvcc_path()).with_name("ptxas"))
    arch = target()[1] + ("a" if arch_specific else "")
    command = [ptxas, level, f"-arch={arch}", "-m64", "-o", str(directory / f"{key}.cubin"), str(source)]
    subprocess.run(command, check=True, capture_output=True)
    return key


@contextlib.contextmanager
def built(
    body: str | list[str],
    inputs: dict[str, np.ndarray],
    outputs: dict[str, tuple[type, int]],
    *,
    blocks: int,
    threads: int = 256,
    level: str | None = None,
    fuse: bool = False,
    ftz: bool = False,
    shared: tuple[dict[str, int], int] = ({}, 0),
    arch_specific: bool = False,
    types: dict[str, DataType] | None = None,
    descriptors: tuple[TmaDescMeta, ...] = (),
) -> Iterator[CompiledProgram]:
    """Compile ``body`` to PTX and build the program that launches it as ``blocks`` blocks of ``threads`` threads,
    under the GPU lock; several bodies are launched in order, over the same buffers. ``outputs`` gives each
    output's element type and count; every output starts at zero. ``level`` is the ptxas level, its default when
    ``None``. ``shared`` is the launch-sized pool: each shared buffer's byte offset and the pool's size.
    ``arch_specific`` assembles for the arch-suffixed target. ``types`` names the element type of an input NumPy
    has no type of its own for (``bf16``, held as its 16 bits). ``descriptors`` are the tensor maps the launch
    encodes and passes after the buffers; an input a descriptor copies from has two dimensions, its array's."""
    in_dtypes = {name: (types or {}).get(name) or DTYPES[data.dtype] for name, data in inputs.items()}
    out_dtypes = {name: (DTYPES[np.dtype(t)], size) for name, (t, size) in outputs.items()}
    params = (
        *(Param(name, "input", dtype, name) for name, dtype in in_dtypes.items()),
        *(Param(name, "output", dtype, name) for name, (dtype, _) in out_dtypes.items()),
        *(Param(descriptor.name, "descriptor") for descriptor in descriptors),
    )
    offsets, pool = shared
    interface = Interface(params, threads, offsets, pool, arch_specific)
    names = tuple(param.name for param in params)
    copied = {descriptor.src_buf for descriptor in descriptors}
    grid = ((blocks,), (1,), (1,)), ((threads,), (1,), (1,))
    kernels, launches = {}, []
    with tempfile.TemporaryDirectory() as scratch:
        for index, text in enumerate([body] if isinstance(body, str) else body):
            name = f"k{index}"
            ptx = compile_ptx(name, parse_body(text), interface, sm=target()[0], fuse=fuse, ftz=ftz)
            if level is None:
                kernels[name] = KernelSpec(source=ptx, arch_specific=arch_specific)
            else:
                key = assemble(ptx, level, Path(scratch), arch_specific)
                kernels[name] = KernelSpec(binary_key=key, arch_specific=arch_specific)
            zeroed = tuple(outputs) if index == 0 else ()
            launches.append(LaunchSpec(name, name, names, *grid, pool, zeroed, writes=tuple(outputs), tma_descriptors=descriptors))
        plan = ExecutionPlan(
            "cuda",
            list(inputs),
            list(outputs),
            [
                *(
                    BufferSpec(name, tuple(Dim(d) for d in (data.shape if name in copied else (data.size,))), in_dtypes[name], "input")
                    for name, data in inputs.items()
                ),
                *(BufferSpec(name, (Dim(size),), dtype, "output") for name, (dtype, size) in out_dtypes.items()),
            ],
            {},
            {},
            launches,
            kernels,
        )
        with gpu_lock():
            yield CompiledProgram.build_from_plan(plan, dict(inputs), cubin_dir=Path(scratch))


def run(body: str | list[str], inputs: dict[str, np.ndarray], outputs: dict[str, tuple[type, int]], **launch) -> dict[str, np.ndarray]:
    """Launch ``body`` once, as ``built`` builds it, and return its outputs."""
    with built(body, inputs, outputs, **launch) as program:
        program.run_once()
        return program.outputs()


def under(grid: str, statements: str, depth: int = 1) -> str:
    """``statements`` as the body of ``grid``, ``depth`` bodies deep."""
    return grid.strip("\n") + "\n" + textwrap.indent(textwrap.dedent(statements).strip("\n"), "    " * depth)


def floats(n: int, dtype, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(dtype)


def same_bits(got: np.ndarray, expected: np.ndarray) -> None:
    assert got.dtype == expected.dtype
    unsigned = np.dtype(f"u{got.dtype.itemsize}")
    np.testing.assert_array_equal(got.view(unsigned), expected.view(unsigned))
