"""Boot roofline audit (serving/roofline.py) — decision logic and advisory-only contract.
Pure CPU: the CUDA-touching measurement helpers are stubbed."""

import logging
import sys
import types

import pytest

from emmy.compiler.backend import gpu_lock as gpu_lock_mod
from emmy.serving import roofline
from emmy.serving.roofline import audit_boot_programs, flag_ratio

GB = 1e9


@pytest.fixture(autouse=True)
def _no_ambient_gpu_lock(monkeypatch):
    """These tests audit decision logic, not locking. ``tests/conftest.py`` exports
    ``EMMY_GPU_LOCK`` suite-wide; an unusable path there (a stale lock file owned by another user on
    a shared runner) would otherwise skip every audit and read as a decision-logic regression."""
    monkeypatch.delenv("EMMY_GPU_LOCK", raising=False)


class _Prog:
    def __init__(self, const_bytes, input_weight_bytes=0):
        self.program = object()
        self.const_bytes = const_bytes
        self.input_weight_bytes = input_weight_bytes

    @property
    def weight_bytes(self):
        return self.const_bytes + self.input_weight_bytes


def test_flag_ratio_thresholds():
    # 100 MB at 1 TB/s → floor 100 µs. 10x is the boundary: at it, no flag; above it, flag.
    assert flag_ratio(1000.0, 100_000_000, 1000 * GB) is None  # exactly 10x → silent
    floor_us, ratio = flag_ratio(1001.0, 100_000_000, 1000 * GB)
    assert abs(floor_us - 100.0) < 1e-6
    assert ratio > 10.0
    # The incident shape: ~46 MB of weights, ~700 GB/s → floor ~66 µs; measured 10,085 µs → ~153x.
    verdict = flag_ratio(10_085.0, 46_000_000, 700 * GB)
    assert verdict is not None
    assert verdict[1] > 100.0


def test_flag_ratio_skips_tiny_and_degenerate():
    """A program with no usable floor is silent while it stays cheap. Each measured value here is
    µs-class, which is what MIN_FLOOR_US exists to ignore — see the companion test for what happens
    when such a program is not cheap."""
    assert flag_ratio(150.0, 1_000, 1000 * GB) is None  # floor below MIN_FLOOR_US → no ratio
    assert flag_ratio(150.0, 0, 1000 * GB) is None  # no weights at all
    assert flag_ratio(150.0, 1_000_000, 0.0) is None  # broken bandwidth measurement


def test_flag_ratio_reports_a_mispick_whose_floor_is_too_small_to_form():
    """The 2026-09-12 DeepSeek-V4 V100 incident. A decode program elected at 29.7 s per forward had
    a sub-MIN_FLOOR_US weight floor, so the audit exempted it entirely: sixteen workers booted clean
    and every request died on the engine's RPC deadline. A small floor bounds what a HEALTHY program
    costs, never what a mispicked one does, so absolute cost decides when no ratio can be formed."""
    assert flag_ratio(roofline.MAX_ABS_US, 1_000, 1000 * GB) is None  # at the bar → still silent
    verdict = flag_ratio(29_693_246.0, 1_000, 1000 * GB)
    assert verdict is not None
    floor_us, ratio = verdict
    assert floor_us == roofline.MIN_FLOOR_US  # reported against the noise threshold
    assert ratio > 1_000_000.0
    # Degenerate inputs are judged the same way once the cost is real.
    assert flag_ratio(29_693_246.0, 0, 1000 * GB) is not None


def test_flag_ratio_compute_floor_bounds_compute_bound_shapes():
    """The 2026-08-12 m4096 chunk-prefill correction: a compute-bound twin sits 24x over the weight
    floor but ~1.2x over its compute floor — max() keeps it silent. At 210 TFLOP/s, 68 MB of f16
    weights at m4096 need ~1.33 ms; the weight floor alone (68 µs at 1 TB/s) misreads 1.62 ms as 24x."""
    wb = 68_000_000
    flops = 2.0 * (wb / 2) * 4096
    assert flag_ratio(1_620.0, wb, 1000 * GB) is not None  # weight floor alone: the old 24x false flag
    assert flag_ratio(1_620.0, wb, 1000 * GB, flops, 210e12) is None  # compute floor binds → healthy
    # A genuinely slow pick still clears the higher floor: 10x the compute floor warns.
    verdict = flag_ratio(14_000.0, wb, 1000 * GB, flops, 210e12)
    assert verdict is not None
    assert abs(verdict[0] - flops / 210e12 * 1e6) < 1.0  # reported floor is the compute floor


def test_flag_ratio_compute_floor_negligible_at_m1():
    """At m1 decode the compute floor is sub-µs — the weight floor binds and the post-twin incident
    class (68x over the weight floor) still warns with the compute args passed."""
    wb = 46_000_000
    flops = 2.0 * (wb / 2) * 1
    verdict = flag_ratio(4_500.0, wb, 700 * GB, flops, 210e12)
    assert verdict is not None
    assert verdict[1] > 60.0


def test_audit_counts_weight_inputs(monkeypatch, caplog):
    """An expert program holds no constants — its weights arrive as per-launch INPUTS. Counting
    only the constant side would give it a zero floor and audit nothing."""
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    monkeypatch.setattr(roofline, "measure_matmul_flops", lambda: 210e12)
    monkeypatch.setattr(roofline, "time_program_us", lambda program, **kw: 50_000.0)  # 500x its 100 µs floor
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("moe.expert.one", _Prog(0, input_weight_bytes=100_000_000), 1)])
    assert len(caplog.records) == 1
    assert "moe.expert.one" in caplog.text


def test_audit_warns_on_outlier_and_stays_quiet_on_healthy(monkeypatch, caplog):
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    monkeypatch.setattr(roofline, "measure_matmul_flops", lambda: 210e12)
    times = iter([50_000.0, 150.0])  # slow program then healthy program, both floor 100 µs
    monkeypatch.setattr(roofline, "time_program_us", lambda program, **kw: next(times))
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("L0.post.decode.m8", _Prog(100_000_000), 8), ("L0.pre.decode.m8", _Prog(100_000_000), 8)])
    assert len(caplog.records) == 1
    assert "L0.post.decode.m8" in caplog.text
    assert "emmy tune" in caplog.text


def test_audit_compute_bound_chunk_twin_is_silent(monkeypatch, caplog):
    """The m4096 correction end to end: a chunk twin 24x over its weight floor but ~1.2x over its
    compute floor stays silent, while an m1 post twin 68x over the weight floor still warns."""
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    monkeypatch.setattr(roofline, "measure_matmul_flops", lambda: 210e12)
    times = iter([1_620.0, 4_500.0])  # chunk.m4096 then the mispicked decode.m1
    monkeypatch.setattr(roofline, "time_program_us", lambda program, **kw: next(times))
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("L0.post.chunk.m4096", _Prog(68_000_000), 4096), ("L0.post.decode.m1", _Prog(46_000_000), 1)])
    assert len(caplog.records) == 1
    assert "L0.post.decode.m1" in caplog.text


def test_audit_degrades_to_weight_floor_without_matmul_calibration(monkeypatch, caplog):
    """A failed compute-throughput calibration must not kill the audit — the weight floor still warns."""
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    monkeypatch.setattr(roofline, "measure_matmul_flops", lambda: (_ for _ in ()).throw(RuntimeError("no cublas")))
    monkeypatch.setattr(roofline, "time_program_us", lambda program, **kw: 50_000.0)
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("L0.post.decode.m8", _Prog(100_000_000), 8)])
    assert len(caplog.records) == 1
    assert "L0.post.decode.m8" in caplog.text


def test_audit_warns_when_gpu_lock_unusable(monkeypatch, caplog):
    """An unusable lock path is an environment fault, not a clean audit — it must be visible at
    warning level (the CI incident: a stale ``/tmp/emmy-gpu.lock`` owned by another user)."""
    monkeypatch.setattr(
        gpu_lock_mod, "gpu_lock", lambda: (_ for _ in ()).throw(PermissionError(13, "Permission denied", "/tmp/emmy-gpu.lock"))
    )
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("L0.post.decode.m8", _Prog(100_000_000), 8)])
    assert len(caplog.records) == 1
    assert "GPU lock" in caplog.text


def test_audit_never_raises(monkeypatch, caplog):
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: (_ for _ in ()).throw(RuntimeError("no gpu")))
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("L0.pre.decode.m8", _Prog(100_000_000), 8)])
    assert not caplog.records  # swallowed to debug level — a boot warning is never a boot blocker


class _FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def _fake_cupy(elapsed_ms):
    """A cupy stand-in whose event timer yields these millisecond readings, one per call."""
    readings = iter(elapsed_ms)
    cuda = types.SimpleNamespace(Event=_FakeEvent, get_elapsed_time=lambda a, b: next(readings))
    return types.SimpleNamespace(cuda=cuda)


def test_time_program_us_stops_after_a_warmup_that_blows_the_budget(monkeypatch):
    """The mispick this audit exists to report is also the most expensive thing to measure. Once
    the warmup alone is past the budget the verdict is settled, so the timed runs are skipped —
    the V100 incident ran one such program four times and held the boot for six hours."""
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy([2.0]))
    runs = []
    program = types.SimpleNamespace(run_once=lambda: runs.append(1))
    assert roofline.time_program_us(program, budget_us=1000.0) == pytest.approx(2000.0)
    assert len(runs) == 1, "a program past its budget is run once, not four times"


def test_time_program_us_medians_the_timed_runs_inside_the_budget(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy([0.1, 0.3, 0.2, 0.4]))
    runs = []
    program = types.SimpleNamespace(run_once=lambda: runs.append(1))
    assert roofline.time_program_us(program, budget_us=1000.0) == pytest.approx(300.0)
    assert len(runs) == 4, "warmup plus three timed runs"


def test_audit_bounds_each_measurement_by_the_warn_threshold(monkeypatch):
    """The audit derives the floor BEFORE measuring and hands it down: nothing is worth timing
    past the ratio that would warn anyway."""
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    monkeypatch.setattr(roofline, "measure_matmul_flops", lambda: 210e12)
    seen = {}

    def _timer(program, *, budget_us=None):
        seen["budget_us"] = budget_us
        return 150.0

    monkeypatch.setattr(roofline, "time_program_us", _timer)
    audit_boot_programs([("L0.post.decode.m8", _Prog(100_000_000), 8)])
    assert seen["budget_us"] == pytest.approx(roofline.WARN_RATIO * 100.0)
