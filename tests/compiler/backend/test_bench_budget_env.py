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
    monkeypatch.delenv(config.BENCH_WALL_TIMEOUT_S, raising=False)
    backend = _StubBackend()
    assert backend.bench_compile_timeout_s == 30.0
    assert backend.bench_run_timeout_s == 10.0
    assert backend.bench_wall_timeout_s is None


def test_env_overrides_constructor_policy(monkeypatch):
    backend = CudaBackend(bench_compile_timeout_s=12.0, bench_run_timeout_s=2.0, bench_wall_timeout_s=16.0)
    monkeypatch.setenv(config.BENCH_COMPILE_TIMEOUT_S, "120")
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "300")
    monkeypatch.setenv(config.BENCH_WALL_TIMEOUT_S, "500")
    assert backend.bench_compile_timeout_s == 120.0
    assert backend.bench_run_timeout_s == 300.0
    assert backend.bench_wall_timeout_s == 500.0


def test_reads_are_live_not_cached(monkeypatch):
    backend = CudaBackend()
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "90")
    assert backend.bench_run_timeout_s == 90.0
    monkeypatch.delenv(config.BENCH_RUN_TIMEOUT_S)
    assert backend.bench_run_timeout_s == 10.0


def test_wall_env_selects_isolated_budget_over_none(monkeypatch):
    backend = CudaBackend()  # constructor default: None (in-process bench)
    monkeypatch.setenv(config.BENCH_WALL_TIMEOUT_S, "42.5")
    assert backend.bench_wall_timeout_s == 42.5


def test_unparseable_env_falls_back_to_policy_value(monkeypatch):
    backend = CudaBackend(bench_run_timeout_s=2.0)
    monkeypatch.setenv(config.BENCH_RUN_TIMEOUT_S, "not-a-number")
    assert backend.bench_run_timeout_s == 2.0


# ---------------------------------------------------------------------------
# The first-iteration watchdog deadline: its own knob, defaulting to 30x the steady one
# ---------------------------------------------------------------------------


def test_first_iter_deadline_defaults_to_thirty_times_the_steady_one(monkeypatch):
    """Unset, the first iteration keeps today's grace: 30x ``EMMY_KERNEL_TIMEOUT_MS``."""
    monkeypatch.delenv(config.FIRST_ITER_TIMEOUT_MS, raising=False)
    monkeypatch.delenv(config.KERNEL_TIMEOUT_MS, raising=False)
    assert config.first_iter_timeout_ms() == 60_000.0
    monkeypatch.setenv(config.KERNEL_TIMEOUT_MS, "30000")
    assert config.first_iter_timeout_ms() == 900_000.0
    monkeypatch.setenv(config.FIRST_ITER_TIMEOUT_MS, "not-a-number")
    assert config.first_iter_timeout_ms() == 900_000.0


def test_first_iter_deadline_has_its_own_knob(monkeypatch):
    """The target that needs a 30 s steady watchdog paid 900 s to price every hang, because the
    first-iteration grace was hard-coupled at 30x. ``EMMY_FIRST_ITER_TIMEOUT_MS`` sets it alone."""
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
# The compile budget is enforced DURING the compile, not when it returns
# ---------------------------------------------------------------------------


def _stub_plan(*names):
    """The fields ``_load_plan`` reads — kernels to compile, and the pure data it adopts."""
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
    monkeypatch.setattr(program, "_load_kernel", lambda name, spec: compiled.append(name))

    def _at(*ticks):
        clock = iter(ticks)
        monkeypatch.setattr(program, "_time_module", SimpleNamespace(monotonic=lambda: next(clock)))

    _at(1.0, 9.0, 9.0)
    with pytest.raises(program.CompileBudgetExceeded) as exc:
        program._load_plan(_stub_plan("k_one", "k_two", "k_three"), deadline=5.0)
    assert compiled == ["k_one", "k_two"], "the third kernel is never handed to nvcc"
    assert "2 of 3 kernel(s)" in str(exc.value) and config.BENCH_COMPILE_TIMEOUT_S in str(exc.value)

    compiled.clear()
    _at(1.0, 2.0, 3.0)
    program._load_plan(_stub_plan("k_one", "k_two", "k_three"), deadline=5.0)
    assert compiled == ["k_one", "k_two", "k_three"], "a compile inside the budget is not cut short"


def test_the_compile_budget_reaches_the_kernel_load_as_a_deadline(monkeypatch):
    """``compile_timeout_s`` has to arrive at the load as an expiry, or the check between kernels
    is never armed and the budget is once more only read when the whole setup returns."""
    from emmy.compiler.backend.cuda import program

    seen = {}
    monkeypatch.setattr(program, "_time_module", SimpleNamespace(monotonic=lambda: 100.0))

    def _capture(plan, *, deadline=None):
        seen["deadline"] = deadline
        raise program.CompileBudgetExceeded("stop here")

    monkeypatch.setattr(program, "_load_plan", _capture)
    with pytest.raises(program.CompileBudgetExceeded):
        program.CompiledProgram.build_from_plan(_stub_plan("k_one"), None, compile_timeout_s=7.5)
    assert seen["deadline"] == 107.5

    seen.clear()
    with pytest.raises(program.CompileBudgetExceeded):
        program.CompiledProgram.build_from_plan(_stub_plan("k_one"), None)
    assert seen["deadline"] is None, "no budget, no deadline — an unbounded compile stays unbounded"
