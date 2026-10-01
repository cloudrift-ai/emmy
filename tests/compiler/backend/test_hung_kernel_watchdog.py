"""A timed launch must raise ``HungKernelError`` *promptly* on a real hung kernel.

This is the device-side half of the "bench must not get stuck" fix (the control-flow half —
``_run_bench`` skipping the per-kernel sweep once the device is poisoned — is covered CUDA-free
in ``tests/compiler/cli/test_tune_bench_hung_kernel.py``). Here we launch a genuinely
non-terminating kernel and assert the runtime's event wait gives up at its deadline and raises,
rather than blocking forever the way a plain ``cudaDeviceSynchronize`` would.

Run in a subprocess: an infinite kernel poisons the device for the rest of the process, so we
isolate it (like ``test_bench_worker_recovery``) and let process exit tear the context down.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from tests.compiler.helpers import requires_cuda

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@requires_cuda
def test_timed_launch_raises_hungkernelerror_promptly() -> None:
    # Child: launch a kernel that spins on a volatile global flag that is never set, through the
    # runtime's timed launch with a 500 ms deadline. The volatile read keeps the compiler from
    # eliding the (otherwise UB, side-effect-free) infinite loop, so the kernel never finishes —
    # a blocking sync would hang forever; the deadline must instead raise HungKernelError well
    # before our 30 s parent timeout. HUNG_OK + exit 0 is the success signal.
    # ``os._exit`` (not ``sys.exit``) on the way out: destroying a CUDA context that still has a
    # running kernel blocks on it, so a normal exit would hang here even though the wait
    # already returned. The hard exit skips that teardown; the OS reclaims the context and the
    # driver kills the kernel. ``flush=True`` because os._exit doesn't flush Python buffers.
    child = """
        import os
        from emmy.compiler.backend.cuda.program import CompiledProgram, HungKernelError
        from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec
        from emmy.compiler.dim import Dim
        from emmy.compiler.dtype import I32

        source = r'extern "C" __global__ void spin(volatile int* f) { while (f[0] == 0) {} }'
        plan = ExecutionPlan(
            "cuda", ["f"], [], [BufferSpec("f", (Dim(1),), I32, "input")], {}, {},
            [LaunchSpec("spin", "spin", ("f",), ((1,), (1,), (1,)), ((1,), (1,), (1,)), 0, ())],
            {"spin": KernelSpec(source=source)},
        )
        prog = CompiledProgram.build_from_plan(plan, {"f": [0]})   # never set → the loop never exits
        try:
            prog.executor.time_launch(0, 1, 500.0)
        except HungKernelError:
            print('HUNG_OK', flush=True); os._exit(0)
        print('NO_RAISE', flush=True); os._exit(1)
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(child)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT,
        text=True,
    )
    try:
        # 30 s is generous headroom over the 500 ms deadline — if the wait were broken (blocking
        # sync behind the hung kernel) this would time out instead.
        out, err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise AssertionError("the deadline did not return — the runtime blocked on a hung kernel") from None
    assert proc.returncode == 0 and "HUNG_OK" in out, f"rc={proc.returncode} out={out!r} err={err[-500:]!r}"
