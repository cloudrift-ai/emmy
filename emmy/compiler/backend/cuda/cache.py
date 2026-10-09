"""L2 eviction outside a benchmark's CUDA-event window."""

from __future__ import annotations

import ctypes
from pathlib import Path


def cuda_runtime():
    """The CUDA runtime already loaded by Torch, or the platform's runtime library."""
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and "libcudart.so" in parts[-1] and parts[-1].startswith("/"):
                return ctypes.CDLL(parts[-1])
    except OSError:
        pass
    return ctypes.CDLL("libcudart.so")


class L2Eviction:
    """Queue a memset over twice the card's L2 before each single timed launch.

    This measures a kernel whose weights come from memory, not serving's partial cache reuse.
    The caller records its start event after this same-stream operation, so eviction is untimed.
    The scratch allocation stays alive for the entire benchmark.
    """

    def __init__(self):
        import torch  # noqa: PLC0415

        size = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
        if size <= 0:
            raise RuntimeError("cold-cache benchmarking requires a positive device L2 cache size")
        self.scratch = torch.empty(2 * size, dtype=torch.uint8, device="cuda")
        self.runtime = cuda_runtime()
        self.memset = self.runtime.cudaMemsetAsync
        self.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        self.memset.restype = ctypes.c_int

    def __call__(self):
        import torch  # noqa: PLC0415

        error = self.memset(self.scratch.data_ptr(), 0, self.scratch.numel(), torch.cuda.current_stream().cuda_stream)
        if error:
            raise RuntimeError(f"cold-cache cudaMemsetAsync failed with CUDA error {error}")
