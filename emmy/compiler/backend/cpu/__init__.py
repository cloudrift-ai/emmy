"""CPU backend: Loop IR kernels compiled to native code through LLVM."""

from emmy.compiler.backend.cpu.backend import CpuBackend, CpuProgram

__all__ = ["CpuBackend", "CpuProgram"]
