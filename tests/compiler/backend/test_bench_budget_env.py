"""``EMMY_BENCH_*`` env overrides of the bench budgets — read live through ``emmy.config``.

The budgets are constructor policy on the backend (``tune`` shrinks them, the bench worker
raises them); the paired env vars override every caller uniformly, read on every access
(never cached), so a raised budget reaches the in-child backend and the pinned-row wall
cap alike.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from emmy import config
from emmy.compiler.backend import Backend
from emmy.compiler.backend.cuda.backend import CudaBackend


class _StubBackend(Backend):
    def compile(self, graph):
        return graph


def test_defaults_unchanged_without_env(monkeypatch):
    monkeypatch.delenv(config.BENCH_COMPILE_TIMEOUT_S, raising=False)
    monkeypatch.delenv(config.BENCH_RUN_TIMEOUT_S, raising=False)
    backend = _StubBackend()
    assert backend.bench_compile_timeout_s == 30.0
    assert backend.bench_run_timeout_s == 10.0


def test_env_overrides_constructor_policy(monkeypatch):
    backend = CudaBackend(bench_compile_timeout_s=12.0, bench_run_timeout_s=2.0)
    monkeypatch.setenv(config.BENCH_COMPILE_TIMEOUT_S, "120")
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "300")
    assert backend.bench_compile_timeout_s == 120.0
    assert backend.bench_run_timeout_s == 300.0


def test_reads_are_live_not_cached(monkeypatch):
    backend = CudaBackend()
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "90")
    assert backend.bench_run_timeout_s == 90.0
    monkeypatch.delenv(config.BENCH_RUN_TIMEOUT_S)
    assert backend.bench_run_timeout_s == 10.0


def test_unparseable_env_falls_back_to_policy_value(monkeypatch):
    backend = CudaBackend(bench_run_timeout_s=2.0)
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "not-a-number")
    assert backend.bench_run_timeout_s == 2.0


# ---------------------------------------------------------------------------
# The first-iteration watchdog deadline: its own knob, defaulting to the steady one plus a fixed grace
# ---------------------------------------------------------------------------


def test_first_iter_deadline_defaults_to_the_steady_one_plus_a_fixed_grace(monkeypatch):
    """Unset, the first iteration gets the steady deadline plus the one-off setup grace — a term, not a
    multiple: raising the steady watchdog to 20 s for a slow program's kernels prices a hang at 80 s, not
    the 600 s the 30x coupling charged."""
    monkeypatch.delenv(config.FIRST_ITER_TIMEOUT_MS, raising=False)
    monkeypatch.delenv(config.KERNEL_TIMEOUT_MS, raising=False)
    assert config.first_iter_timeout_ms() == 2_000.0 + config.FIRST_LAUNCH_GRACE_MS == 62_000.0
    monkeypatch.setenv(config.KERNEL_TIMEOUT_MS, "20000")
    assert config.first_iter_timeout_ms() == 80_000.0
    monkeypatch.setenv(config.FIRST_ITER_TIMEOUT_MS, "not-a-number")
    assert config.first_iter_timeout_ms() == 80_000.0


def test_first_iter_deadline_has_its_own_knob(monkeypatch):
    """``EMMY_FIRST_ITER_TIMEOUT_MS`` sets the first-iteration deadline alone, whatever the steady one."""
    monkeypatch.setenv(config.KERNEL_TIMEOUT_MS, "30000")
    monkeypatch.setenv(config.FIRST_ITER_TIMEOUT_MS, "60000")
    assert config.first_iter_timeout_ms() == 60_000.0


def test_launch_deadline_uses_the_first_iter_budget_on_iter_zero_only(monkeypatch):
    """The per-launch watchdog reads the first-iteration budget on iter 0 and the steady deadline
    after, each scaled by the event window's launch count."""
    from emmy.compiler.backend.cuda.program import _launch_deadline_ms

    monkeypatch.setenv(config.KERNEL_TIMEOUT_MS, "30000")
    monkeypatch.setenv(config.FIRST_ITER_TIMEOUT_MS, "60000")
    assert _launch_deadline_ms(0, 1) == 60_000.0
    assert _launch_deadline_ms(1, 1) == 30_000.0
    assert _launch_deadline_ms(2, 3) == 90_000.0


# ---------------------------------------------------------------------------
# The bench worker's wall caps: the watchdog must fire first, and the Torch twin's JIT is paid for
# ---------------------------------------------------------------------------


def _graph(n_kernels: int, n_other: int = 0):
    """A graph of ``n_kernels`` CUDA kernels and ``n_other`` non-kernel nodes, as the wall cap counts them."""
    from emmy.compiler.ir.cuda.ir import CudaOp

    kernels = {f"k{i}": SimpleNamespace(op=CudaOp.__new__(CudaOp)) for i in range(n_kernels)}
    others = {f"n{i}": SimpleNamespace(op=object()) for i in range(n_other)}
    return SimpleNamespace(nodes={**kernels, **others})


def test_compare_wall_cap_outlasts_the_first_iteration_watchdog(monkeypatch):
    """A hung kernel must be named by the watchdog (a ``bench_fail`` row the next pick reads), never by the
    parent's wall kill, which names none and leaves the next greedy to elect the same arm. The cap therefore
    holds the first-iteration deadline on top of every per-kernel budget, at any steady watchdog."""
    from emmy.commands.run import _compare_wall_s

    backend = CudaBackend(bench_compile_timeout_s=30.0, bench_run_timeout_s=10.0)
    monkeypatch.delenv(config.FIRST_ITER_TIMEOUT_MS, raising=False)
    monkeypatch.setenv(config.KERNEL_TIMEOUT_MS, "20000")
    cap = _compare_wall_s(_graph(3), backend, base_s=60.0)
    assert cap == 60.0 + 3 * 30.0 + 80.0 + 10.0
    assert cap > config.first_iter_timeout_ms() / 1000.0 + 3 * 30.0 + 60.0
    assert _compare_wall_s(_graph(0, 2), backend, base_s=60.0) == 60.0 + 30.0 + 80.0 + 10.0  # a kernel-less graph pays one compile


def test_compare_wall_cap_pays_for_the_torch_twins_jit_per_node(monkeypatch):
    """With ``torch.compile`` among the benched backends, the Torch twin's Inductor JIT runs inside the same
    job and takes minutes on a thousand-node program; the cap grants it a term per node of the twin's
    program. Without that backend nothing is added."""
    from emmy.commands.run import _TWIN_JIT_S_PER_NODE, _compare_wall_s

    backend = CudaBackend(bench_compile_timeout_s=30.0, bench_run_timeout_s=10.0)
    monkeypatch.setenv(config.FIRST_ITER_TIMEOUT_MS, "60000")
    plain = _compare_wall_s(_graph(2), backend, base_s=60.0)
    with_twin = _compare_wall_s(_graph(2), backend, base_s=60.0, twin=_graph(0, 3000))
    assert with_twin - plain == 3000 * _TWIN_JIT_S_PER_NODE
    assert with_twin > 60.0 * 60.0  # over an hour of JIT before the kill, where the old cap gave 130 s


async def test_pinned_wall_cap_outlasts_the_first_iteration_watchdog(monkeypatch):
    """The pinned-row cap carries the first-iteration deadline for the same reason as the comparison cap."""
    import emmy.compiler.backend.cuda.program as program_mod

    seen = {}

    async def _fake_pinned(compiled, **kwargs):
        seen.update(kwargs)
        return "bench", None

    monkeypatch.setattr(program_mod, "benchmark_pinned_isolated_async", _fake_pinned)
    monkeypatch.setenv(config.FIRST_ITER_TIMEOUT_MS, "600000")
    backend = CudaBackend(bench_compile_timeout_s=30.0, bench_run_timeout_s=10.0)
    monkeypatch.setattr(backend, "_async_worker", lambda: object())
    await backend.bench_pinned_async("G")
    assert seen["wall_timeout_s"] == 30.0 + 10.0 + 600.0 + 60.0


# ---------------------------------------------------------------------------
# The compile budget is enforced DURING the compile, not when it returns
# ---------------------------------------------------------------------------


def _stub_plan(*names):
    """The fields ``_compile_kernels`` and the build read — kernels to compile, and the pure data."""
    from types import SimpleNamespace

    return SimpleNamespace(
        kernels=dict.fromkeys(names, object()),
        buffers=(),
        constants={},
        launches=(),
        outputs=(),
        symbolic_bindings={},
        symbolic_hints={},
        symbolic_caps={},
        runtime_constants={},
    )


def test_compile_budget_stops_between_kernels_instead_of_after_the_whole_load(monkeypatch):
    """A cold multi-kernel compile must not outlive the wall cap that SIGKILLs the bench worker.

    Checked only when the load RETURNED, a compile that runs past the cap never got to check: the
    parent killed the child first and reported a dead worker, which reads as a slow kernel and is
    the opposite of what a compile overrun means — nothing about the kernel's speed was measured.
    That false negative cost a V100 Gated DeltaNet chunk its schedule: the target was recorded as
    unmeasurable under every schedule tried, and the one schedule nobody reached runs 9.9x eager.

    So the budget is checked between kernels. Here the second kernel's compile crosses it and the
    third is never handed to nvcc; under a budget the same compile fits, all three are."""
    from emmy.compiler.backend.cuda import program

    compiled = []
    monkeypatch.setattr(program, "_cubin_path", lambda name, spec, **kwargs: compiled.append(name))

    def _at(*ticks):
        clock = iter(ticks)
        monkeypatch.setattr(program, "_time_module", SimpleNamespace(monotonic=lambda: next(clock)))

    _at(1.0, 9.0, 9.0)
    with pytest.raises(program.CompileBudgetExceeded) as exc:
        program._compile_kernels(_stub_plan("k_one", "k_two", "k_three"), deadline=5.0)
    assert compiled == ["k_one", "k_two"], "the third kernel is never handed to nvcc"
    assert "2 of 3 kernel(s)" in str(exc.value) and config.BENCH_COMPILE_TIMEOUT_S in str(exc.value)

    compiled.clear()
    _at(1.0, 2.0, 3.0)
    program._compile_kernels(_stub_plan("k_one", "k_two", "k_three"), deadline=5.0)
    assert compiled == ["k_one", "k_two", "k_three"], "a compile inside the budget is not cut short"


def test_the_compile_budget_reaches_the_kernel_load_as_a_deadline(monkeypatch):
    """``compile_timeout_s`` has to arrive at the load as an expiry, or the check between kernels
    is never armed and the budget is once more only read when the whole setup returns."""
    from emmy.compiler.backend.cuda import program

    seen = {}
    monkeypatch.setattr(program, "_time_module", SimpleNamespace(monotonic=lambda: 100.0))

    def _capture(plan, *, deadline=None, cubin_dir=None):
        seen["deadline"] = deadline
        raise program.CompileBudgetExceeded("stop here")

    monkeypatch.setattr(program, "_compile_kernels", _capture)
    with pytest.raises(program.CompileBudgetExceeded):
        program.CompiledProgram.build_from_plan(_stub_plan("k_one"), None, compile_timeout_s=7.5)
    assert seen["deadline"] == 107.5

    seen.clear()
    with pytest.raises(program.CompileBudgetExceeded):
        program.CompiledProgram.build_from_plan(_stub_plan("k_one"), None)
    assert seen["deadline"] is None, "no budget, no deadline — an unbounded compile stays unbounded"
