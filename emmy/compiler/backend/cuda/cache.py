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


def straight_line_source(name: str, code_bytes: int, compute_capability: tuple[int, int], vectors: int = 0) -> str:
    """The PTX of a kernel ``name(const float4* scratch, float* sink)`` whose first warp on each of an SM's four
    schedulers runs about ``code_bytes`` of straight-line integer code, eight independent chains so the run is bound
    by fetching the code, for a card of ``compute_capability``. With ``vectors``, every thread first reads its share
    of the ``vectors`` float4 of the scratch, which leaves L2 holding it. The kernel claims the launch's dynamic
    shared memory, so a launch that sizes it past half an SM's places one block on each."""
    major, minor = compute_capability
    lines = [
        f".version {'8.7' if major >= 10 else '7.8'}",
        f".target sm_{major}{minor}",
        ".address_size 64",
        ".extern .shared .align 16 .b8 claim[];",
        f".visible .entry {name}(.param .u64 scratch_param, .param .u64 sink_param)",
        "{",
        "\t.reg .pred %p<4>;",
        "\t.reg .b32 %x<8>, %t, %y, %tx, %block, %i, %stride, %claim;",
        "\t.reg .f32 %a<4>, %v<4>, %s;",
        "\t.reg .b64 %scratch, %sink, %at, %off;",
        "\tld.param.u64 %scratch, [scratch_param];",
        "\tcvta.to.global.u64 %scratch, %scratch;",
        "\tld.param.u64 %sink, [sink_param];",
        "\tcvta.to.global.u64 %sink, %sink;",
        "\tmov.u32 %tx, %tid.x;",
        "\tmov.u32 %block, %ctaid.x;",
        "\tadd.u32 %y, %block, 1;",
        *(f"\tmov.f32 %a{k}, 0f00000000;" for k in range(4)),
    ]
    if vectors:
        lines += [
            "\tmov.u32 %stride, %ntid.x;",
            "\tmad.lo.u32 %i, %block, %stride, %tx;",
            "\tmov.u32 %t, %nctaid.x;",
            "\tmul.lo.u32 %stride, %stride, %t;",
            "READ:",
            f"\tsetp.ge.u32 %p0, %i, {vectors};",
            "\t@%p0 bra READ_DONE;",
            "\tmul.wide.u32 %off, %i, 16;",
            "\tadd.u64 %at, %scratch, %off;",
            "\tld.global.v4.f32 {%v0, %v1, %v2, %v3}, [%at];",
            *(f"\tadd.f32 %a{k}, %a{k}, %v{k};" for k in range(4)),
            "\tadd.u32 %i, %i, %stride;",
            "\tbra READ;",
            "READ_DONE:",
        ]
    lines += [f"\tmad.lo.u32 %x{r}, %y, {r}, %tx;" for r in range(8)]
    lines += ["\tsetp.ge.u32 %p1, %tx, 128;", "\t@%p1 bra CODE_DONE;"]
    for i in range(code_bytes // 48):  # three 16-byte instructions a step
        r = i % 8
        lines += [
            f"\tshr.u32 %t, %x{r}, {3 + i * 7 % 27};",
            f"\txor.b32 %t, %x{r}, %t;",
            f"\tmad.lo.u32 %x{r}, %t, {(0x9E3779B1 * (i + 1)) & 0xFFFFFFFF | 1}, %y;",
        ]
    lines += [
        "CODE_DONE:",
        # What the read summed is stored where it cannot be predicted, so the read is never dropped.
        "\tadd.f32 %s, %a0, %a1;",
        "\tadd.f32 %s, %s, %a2;",
        "\tadd.f32 %s, %s, %a3;",
        "\tmul.wide.u32 %off, %block, 4;",
        "\tadd.u64 %at, %sink, %off;",
        "\tsetp.neu.f32 %p2, %s, 0f3FC00000;",
        "\t@%p2 bra KEEP_DONE;",
        "\tst.global.f32 [%at], %a0;",
        "KEEP_DONE:",
        "\txor.b32 %t, %x0, %x1;",
        *(f"\txor.b32 %t, %t, %x{r};" for r in range(2, 8)),
        "\tsetp.ne.u32 %p3, %t, 2654435769;",
        "\t@%p3 bra END;",
        "\tmov.u32 %claim, claim;",
        "\tst.shared.f32 [%claim], 0f3F800000;",
        "\tmad.lo.u32 %claim, %tx, 4, %claim;",
        "\tld.shared.f32 %s, [%claim];",
        "\tst.global.f32 [%at], %s;",
        "END:",
        "\tret;",
        "}",
    ]
    return "\n".join(lines) + "\n"


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
        source = straight_line_source("emmy_cache_eviction", EVICTED_CODE_BYTES, (props.major, props.minor), vectors)
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
