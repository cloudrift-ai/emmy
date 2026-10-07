"""Bench-to-DB recording — ``run --bench`` measurements become ``perf`` rows in the tune DB.

A ``run --bench`` invocation that benched pinned rows (a golden, an ``--ab`` row) or the greedy
pick records each clean measurement as per-kernel ``perf`` rows through the tuner's own writer, so a
replayed golden or a hand-pinned row becomes what the next ``compile`` / ``run`` / ``serve`` deploys,
and — once the tune DB is imported into a dataset instance — a training row like any tune
measurement. A greedy pick whose bench failed records the kernel the failure blames as a
``bench_fail`` row, exactly as the tuner files a hung terminal.
Recording is **default-on behind a quality bar** (:func:`meets_quality_bar`; ``run --no-record-evidence``
opts out). The caller (``emmy/commands/run.py``) owns which rows are honest enough to record — never a
``pin_unmatched`` row (the claimed config never ran) or one carrying an integrity flag (wrong answer,
intensity floor); this module records what it is given.

The row writers live here too: :func:`persist_kernel_perf` is the ONE writer for a kernel measurement and
:func:`persist_bench_failure` the one for a failed bench, so ``run --bench``'s pinned rows and the golden
import are indistinguishable to the evidence pick.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

from emmy.compiler.pipeline.search.db import KernelDef, PerfStats, knobs_json
from emmy.compiler.wire import formed_from, kernel_bindings, kernel_tile, kernel_wire

if TYPE_CHECKING:
    from emmy.compiler.context import Context

# The pinned-bench measurement standard (``CudaBackend.bench_pinned_async`` defaults). A run benched
# below it is a quick look, not a measurement — recording it would let a noisy drive-by median
# displace a recorded row (the upsert keeps the lowest).
MIN_RECORD_WARMUP = 5
MIN_RECORD_ITERS = 20


def meets_quality_bar(warmup: int, iters: int) -> bool:
    """Whether a ``run --bench`` invocation measures well enough to record."""
    return warmup >= MIN_RECORD_WARMUP and iters >= MIN_RECORD_ITERS


def record_bench_perf(db_path: Path | str, ctx: Context, compiled, bench) -> int:
    """Persist a benched compiled graph's per-kernel measurements as ``perf`` rows under the live
    context — the deploy evidence the greedy pick reads. Kernels pair with ``bench.per_launch`` by
    launch order; a bench without per-launch windows records nothing (a whole-graph time is not a
    kernel's). Returns the rows written."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

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
    failure blames — the kernel a watchdog named, or a one-kernel graph's only kernel — so the next
    compile's evidence pick disqualifies the arm that hung instead of electing it again. A
    compile-budget overrun measured nothing and records nothing. Returns the blamed kernel names."""
    from emmy.compiler.backend.cuda.program import compile_budget_overrun  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    if compile_budget_overrun(exc):
        return []
    nodes = [compiled.nodes[nid] for nid in compiled.topological_order() if isinstance(compiled.nodes[nid].op, CudaOp)]
    db = SearchDB(Path(db_path))
    try:
        blamed = persist_bench_failure(db, ctx, "cuda", nodes, exc, fail_us)
    finally:
        db.close()
    return [node.op.kernel_name for node in blamed]


def measured_schedules(db_path: Path | str, ctx: Context, cuda_ops) -> list[int | None]:
    """How many schedules the tune DB at ``db_path`` holds an ``ok`` measurement of for each kernel ``cuda_ops``
    realizes, at its sizes, under ``ctx``'s card and regime — the search behind a recorded row, which counts at least
    itself; ``None`` for a kernel the DB cannot name."""
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    keys = [kernel_key(op) for op in cuda_ops]
    db = SearchDB(Path(db_path))
    try:
        schedules: dict[tuple, set[str]] = {}
        for row in db.iter_perf(ctx, backend="cuda"):
            if row.status == "ok":
                schedules.setdefault((row.kernel, knobs_json(row.bindings)), set()).add(knobs_json(row.knobs))
    finally:
        db.close()
    return [(len(schedules.get((key[1], knobs_json(key[2])), ())) or 1) if key else None for key in keys]


def point_stats(us: float) -> PerfStats:
    """A single-sample ``PerfStats`` — the shape a whole-graph time takes when no per-launch samples exist."""
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)


def stats_from_launch(lt) -> PerfStats:
    """``PerfStats`` for one benched launch: over its samples when it carries them, else the point time."""
    if lt.samples and len(lt.samples) >= 1:
        us = [s * 1000.0 for s in lt.samples]
        return PerfStats(
            median=statistics.median(us),
            min=min(us),
            max=max(us),
            mean=statistics.fmean(us),
            variance=statistics.pvariance(us) if len(us) > 1 else 0.0,
            n_samples=len(us),
        )
    return point_stats(lt.time_ms * 1000.0)


def kernel_key(cuda_op) -> tuple | None:
    """The kernel half of a measured ``cuda_op``'s ``perf`` key: ``(tile, exact identity, bindings)``
    of the kernel it realizes (``wire.kernel_tile``) — the kernel row's identity and the sizes the bench
    bound its symbolic dims to (its knobs are the other half). ``None`` for a kernel no tile stands
    behind, which is no kernel the tune DB can name."""
    tile = kernel_tile(cuda_op)
    identity = tile.identity_key(structural=False, with_io=True) if tile is not None else None
    return None if identity is None else (tile, identity, kernel_bindings(tile))


def kernel_row(tile, name: str) -> KernelDef:
    """The ``kernel`` row of a tile kernel: its wire, the C name it was rendered under and whether the wire is
    the body it was formed from, keyed by the tile's own exact identity — which lifting the wire again computes
    too (``KernelDef.exact_identity``)."""
    row = KernelDef(loop_ir=kernel_wire(tile), name=name, formed=formed_from(tile) is not None)
    return row.keyed(tile.identity_key(structural=False, with_io=True))


def persist_kernel_perf(
    db,
    ctx,
    backend_name: str,
    cuda_op,
    *,
    stats,
    status: str,
    captured: bool = False,
    error: str | None = None,
    knobs: dict | None = None,
    source: str = "measured",
) -> bool:
    """Persist one measured kernel as deploy evidence: its ``kernel`` row (the definition the
    measurement is of) and its ``perf`` row under ``ctx``'s card and regime (keep-best policy, see
    :meth:`SearchDB.record_perf`). The ONE writer for a kernel measurement — ``run --bench``'s
    pinned rows and the golden import both come here, so a replayed golden and a recorded pick are
    indistinguishable to the evidence pick. The row is the op's knobs — the decisions taken on it —
    unless ``knobs`` says otherwise — a golden's recorded schedule row, stored as written rather
    than as the import's lowering realized it; ``source`` names where the measurement came from.
    Returns whether a row was written (a kernel no tile stands behind persists nothing)."""
    key = kernel_key(cuda_op)
    if key is None:
        return False
    tile, identity, bindings = key
    db.record_kernel(kernel_row(tile, cuda_op.kernel_name))
    if knobs is None:
        knobs = getattr(cuda_op, "knobs", None) or {}
    db.record_perf(
        ctx,
        identity,
        bindings=bindings,
        knobs=knobs,
        backend=backend_name,
        status=status,
        stats=stats,
        captured=captured,
        error=error,
        source=source,
    )
    return True


#: The kernel a failure message NAMES — ``kernel 'k_foo (iter 0)' did not complete …`` from the
#: watchdog, ``nvcc compile failed for kernel 'k_foo': …`` from the compiler. The exception class
#: does not survive the bench worker's pipe (it arrives wrapped in a ``BenchWorkerJobError``
#: carrying the child exception's ``repr``), so the label is recovered from the text — and
#: ``repr`` escapes the name's quote as ``\'`` when the message also holds a ``"`` (nvcc quotes
#: identifiers that way), so the quote is matched with or without its backslash.
_NAMED_KERNEL = re.compile(r"kernel \\?'([A-Za-z_][A-Za-z0-9_]*)")


def persist_bench_failure(db, ctx, backend_name: str, cuda_nodes, exc, fail_us: float) -> list:
    """Persist a failed bench as the per-kernel evidence it is: a ``bench_fail`` perf row at the
    ``fail_us`` sentinel for every node the failure is EVIDENCE ABOUT — usually not every kernel
    benched — and return those nodes. The ONE writer for a bench failure, as
    :func:`persist_kernel_perf` is for a measurement, so a hang blames the same kernel whichever
    command measured it.

    A bench runs many kernels together and one of them hanging fails the whole run, so blaming all
    of them records a failure for kernels that were never shown to fail. That is not a cosmetic
    mislabel: those rows are read as deploy evidence, and on DeepSeek-V4's post block 70 recorded
    failures carried only 7 distinct errors — 20 kernels condemned by one hang, and 21 by a
    bench-worker startup timeout that is not a property of any kernel. So blame is recorded only
    where it is unambiguous: the kernel the watchdog named, or the single kernel of a one-kernel
    graph. Otherwise no kernel earns a row — the run failed, but which kernel failed is unknown,
    and unknown is not the same as failed. The DB holds measurements of kernels and nothing else,
    so such a slice is spent for this session and benched again, at the run budget, by the next."""
    named = _NAMED_KERNEL.search(str(exc))
    if named is not None:
        blamed = [n for n in cuda_nodes if getattr(n.op, "kernel_name", "") == named.group(1)]
    else:
        blamed = list(cuda_nodes) if len(cuda_nodes) == 1 else []
    stats = point_stats(fail_us)
    error = f"{type(exc).__name__}: {exc}"
    for node in blamed:
        persist_kernel_perf(db, ctx, backend_name, node.op, stats=stats, status="bench_fail", error=error)
    return blamed
