"""Bench-to-DB recording — ``run --bench`` measurements become ``perf`` rows in the tune DB.

A ``run --bench`` invocation that benched pinned rows (a golden, an ``--ab`` row) or the greedy
pick records each clean measurement as per-kernel ``perf`` rows through the tuner's own writer, so a
replayed golden or a hand-pinned row becomes what the next ``compile`` / ``run`` / ``serve`` deploys,
and — once the tune DB is imported into a dataset instance — a training row like any tune
measurement. A greedy pick whose bench failed records the kernel the failure blames as a
``bench_fail`` row, exactly as the tuner files a hung terminal.

Recording is **default-on behind a quality bar** (:func:`meets_quality_bar` — the tuner's own
pinned-bench standard; ``run --no-record-evidence`` opts out). The caller
(``emmy/commands/run.py``) owns which rows are honest enough to record — never a ``pin_unmatched``
row (the claimed config never ran) or one carrying an integrity flag (wrong answer, intensity
floor); this module records what it is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.context import Context

# The tuner's pinned-bench measurement standard (``CudaBackend.bench_pinned_async``
# defaults). A run benched below it is a quick look, not a measurement — recording it
# would let a noisy drive-by median displace a tune-grade row (the upsert keeps the lowest).
MIN_RECORD_WARMUP = 5
MIN_RECORD_ITERS = 20


def meets_quality_bar(warmup: int, iters: int) -> bool:
    """Whether a ``run --bench`` invocation measures well enough to record."""
    return warmup >= MIN_RECORD_WARMUP and iters >= MIN_RECORD_ITERS


def record_bench_perf(db_path: Path | str, ctx: Context, compiled, bench) -> int:
    """Persist a benched compiled graph's per-kernel measurements as ``perf`` rows under the live
    context — the deploy evidence the greedy pick reads — through the tuner's own writer
    (:func:`~emmy.compiler.pipeline.search.policy.terminal_bench.persist_kernel_perf`). Kernels
    pair with ``bench.per_launch`` by launch order; a bench without per-launch windows records
    nothing (a whole-graph time is not a kernel's). Returns the rows written."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.policy.terminal_bench import persist_kernel_perf, stats_from_launch  # noqa: PLC0415

    nodes = [compiled.nodes[nid] for nid in compiled.topological_order() if isinstance(compiled.nodes[nid].op, CudaOp)]
    per_launch = list(getattr(bench, "per_launch", None) or [])
    if not nodes or len(per_launch) != len(nodes):
        return 0
    db = SearchDB(Path(db_path))
    written = 0
    try:
        for node, launch in zip(nodes, per_launch, strict=True):
            written += persist_kernel_perf(
                db,
                ctx,
                "cuda",
                node.op,
                stats=stats_from_launch(launch),
                status="ok",
                captured=bool(getattr(bench, "captured", False)),
            )
    finally:
        db.close()
    return written


def record_bench_failure(db_path: Path | str, ctx: Context, compiled, exc, fail_us: float) -> list[str]:
    """Persist a compiled graph's failed bench as ``bench_fail`` perf rows for the kernels the
    failure blames — the kernel a watchdog named, or a one-kernel graph's only kernel — through the
    tuner's own writer (:func:`~emmy.compiler.pipeline.search.policy.terminal_bench.persist_bench_failure`),
    so the next compile's evidence pick disqualifies the arm that hung instead of electing it again.
    A compile-budget overrun measured nothing and records nothing. Returns the blamed kernel names."""
    from emmy.compiler.backend.cuda.program import compile_budget_overrun  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.policy.terminal_bench import persist_bench_failure  # noqa: PLC0415

    if compile_budget_overrun(exc):
        return []
    nodes = [compiled.nodes[nid] for nid in compiled.topological_order() if isinstance(compiled.nodes[nid].op, CudaOp)]
    db = SearchDB(Path(db_path))
    try:
        blamed = persist_bench_failure(db, ctx, "cuda", nodes, exc, fail_us)
    finally:
        db.close()
    return [node.op.kernel_name for node in blamed]
