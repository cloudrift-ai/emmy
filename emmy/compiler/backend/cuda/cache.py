"""Cache eviction outside a benchmark's CUDA-event window."""

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


#: Straight-line code the eviction runs: twice a V100's 128 KB per-SM instruction cache, the card it was sized on.
EVICTED_CODE_BYTES = 256 * 1024


def straight_line_source(name: str, code_bytes: int, prologue: str = "", epilogue: str = "") -> str:
    """A CUDA kernel ``name(const float4* scratch, float* sink)`` whose first warp on each of an SM's four schedulers
    runs about ``code_bytes`` of straight-line integer code, eight independent chains so the run is bound by fetching
    the code. ``prologue`` and ``epilogue`` are statements every thread runs before and after it."""
    lines = []
    for i in range(code_bytes // 48):  # three 16-byte instructions a line
        r = i % 8
        lines.append(f"        x{r} = (x{r} ^ (x{r} >> {3 + i * 7 % 27})) * 0x{(0x9E3779B1 * (i + 1)) & 0xFFFFFFFF | 1:08x}u + y;")
    chains = " ".join(f"unsigned x{r} = threadIdx.x + {r}u * y;" for r in range(8))
    body = "\n".join(lines)
    return f"""extern "C" __global__ void {name}(const float4* __restrict__ scratch, float* __restrict__ sink) {{
    extern __shared__ float claim[];
{prologue}
    unsigned y = blockIdx.x + 1u;
    {chains}
    if (threadIdx.x < 128) {{
{body}
    }}
{epilogue}
    if ((x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7) == 0x9e3779b9u) {{ claim[0] = 1.0f; sink[blockIdx.x] = claim[threadIdx.x]; }}
}}
"""


class CacheEviction:
    """Before each single timed launch, evict what serving finds cold: L2's data, by reading a scratch allocation
    twice the card's L2, and every SM's instruction caches, by running straight-line code larger than them.

    Serving runs dozens of other kernels between two launches of one, so a kernel's weights and its code both come
    from memory. A read leaves L2 holding clean lines, as serving's weight reads do; a write would leave dirty lines
    the timed kernel pays to write back. A kernel whose code outgrows the instruction cache stalls on every fetch, and
    a bench that repeats one kernel never sees it. One launch does both: one block per SM, each claiming more than
    half the SM's shared memory so no SM holds two, every thread reading its share of the scratch. It runs on the
    benched program's stream before the start event, so eviction is untimed.
    """

    def __init__(self):
        import torch  # noqa: PLC0415

        from emmy.compiler.backend.cuda.program import CompiledProgram  # noqa: PLC0415
        from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec  # noqa: PLC0415
        from emmy.compiler.dim import Dim  # noqa: PLC0415
        from emmy.compiler.dtype import F32  # noqa: PLC0415

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.L2_cache_size <= 0:
            raise RuntimeError("cold-cache benchmarking requires a positive device L2 cache size")
        sms, vectors = props.multi_processor_count, 2 * props.L2_cache_size // 16
        smem = min(props.shared_memory_per_multiprocessor // 2 + 1024, props.shared_memory_per_block_optin)
        read = f"""    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < {vectors}u; i += gridDim.x * blockDim.x) {{
        float4 v = scratch[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }}"""
        keep = "    if (acc.x + acc.y + acc.z + acc.w == 1.5f) sink[blockIdx.x] = acc.x;"
        source = straight_line_source("emmy_cache_eviction", EVICTED_CODE_BYTES, read, keep)
        plan = ExecutionPlan(
            "cuda",
            ["scratch"],
            ["sink"],
            [BufferSpec("scratch", (Dim(4 * vectors),), F32, "input"), BufferSpec("sink", (Dim(sms),), F32, "output")],
            {},
            {},
            [
                LaunchSpec(
                    "evict",
                    "emmy_cache_eviction",
                    ("scratch", "sink"),
                    ((sms,), (1,), (1,)),
                    ((1024,), (1,), (1,)),
                    smem,
                    (),
                    writes=("sink",),
                )
            ],
            {"emmy_cache_eviction": KernelSpec(source=source)},
        )
        # The scratch is lent as it is: what it holds does not matter, only that reading it fills L2.
        self.program = CompiledProgram.build_from_plan(plan, {"scratch": torch.empty(4 * vectors, device="cuda")})

    def __call__(self):
        with self.program.on_torch_stream():
            self.program.run_once()
