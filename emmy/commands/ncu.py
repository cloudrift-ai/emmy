"""``run --profile``: the same ``emmy run`` invocation again under ``ncu``, and its hardware counters as a table."""

from __future__ import annotations

import logging

from emmy import config

logger = logging.getLogger(__name__)

_NCU_RECURSE_GUARD = config.NCU_CHILD


# Curated ncu metric set — verified to populate on RTX 5090 (sm_120) +
# TMA kernels. ``--set detailed`` is broken there (``SpeedOfLight_Roofline``
# divides by zero, ``SourceCounters`` / ``PCSamplingData`` need missing
# ``smsp__pcsamp_*`` metrics) so we enumerate explicit metrics instead.
# Add to this list to surface new columns in the perf summary; the
# downstream parser keys on metric names directly.
_NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_lsu.sum",
    "launch__registers_per_thread",
)


def run_ncu_profile(args, *, dump_dir=None):
    """Re-launch the same ``emmy run`` invocation under ``ncu`` to
    collect a curated set of hardware counters (occupancy, bank
    conflicts, SM/DRAM/FMA throughput, register pressure). Output is
    captured in CSV form; when ``EMMY_DUMP_DIR`` (or ``--dump-dir``
    propagated as ``dump_dir``) is set, the raw CSV and a parsed
    per-kernel JSON are written there. Otherwise the counters print to
    stdout in the same CSV form for one-shot inspection.

    Spawns one extra subprocess at minimal iter count — ncu's per-launch
    overhead is huge (10-100×). The ``EMMY_NCU_CHILD`` env var
    prevents the profiled child from re-spawning ncu recursively.

    Skipped silently when ``ncu`` is not on PATH. ncu's own stderr is
    relayed when it fails (typical failure: NVIDIA's perf-counter
    permission gate)."""
    import json as _json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path as _Path

    if config.ncu_child():
        return

    ncu = shutil.which("ncu")
    if ncu is None:
        logger.info("ncu not found on PATH; skipping --profile output")
        return

    env = dict(os.environ)
    env[_NCU_RECURSE_GUARD] = "1"
    # The ncu child process re-runs trace + compile, which would
    # ``shutil.rmtree`` the parent's dump dir from ``CompilerDump``'s
    # ``__post_init__``. Drop the env var for the child so the parent's
    # ``60_*.json`` (bench results) survive — ncu output is captured via
    # stdout and saved by the parent below.
    env.pop(config.DUMP_DIR, None)

    cmd: list[str] = [
        ncu,
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(_NCU_METRICS),
        sys.executable,
        "-m",
        "emmy.emmy",
        "run",
    ]
    # The child compiles the program the parent benched: every flag that selects the input or shapes its compile
    # goes with it. ``--cold-cache`` stays behind, since it times kernels and the child takes no timing.
    traced = args.input is not None and args.ir is None  # a model id, traced again by the child
    for flag, value in (
        ("--golden", args.golden),
        ("--realization", args.realization),
        ("--kernel", args.kernel),
        ("--program", args.program),
        ("--code", args.code),
        ("--ir", args.ir),
        ("--adapter", args.adapter if traced else None),
        ("--layer", args.layer),
        ("--seq-len", args.seq_len if traced else None),
        ("--quantize", args.quantize),
        ("--gpu-arch", args.gpu_arch),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    for flag, values in (("--dynamic", args.dynamic), ("--paged", args.paged)):
        for value in values or []:
            cmd.extend([flag, value])
    if args.pin_route:
        cmd.append("--pin-route")
    if traced:
        cmd.append(args.input)
    # ncu's per-launch overhead means we want one or two launches per
    # kernel — enough for the counters to populate, not so many that
    # the run drags out. Match ``emmy run --bench``'s minimal
    # warmup so the profiled launches see a realistic-ish steady state.
    cmd.extend(["--warmup", "2", "--iters", "3"])

    print()
    print("=" * 80)
    print("ncu --csv (curated hardware metrics)")
    print("=" * 80)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("ncu profiling timed out")
        return

    # ncu writes ``==PROF==`` status lines first, then a CSV table
    # starting at the ``"ID"`` header. Split the two so we can save just
    # the CSV part as a file and surface the status lines for
    # diagnostics.
    stdout = result.stdout
    lines = stdout.splitlines()
    csv_start = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            csv_start = i
            break
    if csv_start is None:
        if stdout.strip():
            print(stdout)
        if result.returncode != 0:
            logger.warning("ncu exit=%d", result.returncode)
            if result.stderr.strip():
                print(result.stderr, file=sys.stderr)
        return

    csv_text = "\n".join(lines[csv_start:]) + "\n"
    status_text = "\n".join(lines[:csv_start])

    parsed = _parse_ncu_csv(csv_text)
    _print_ncu_compare(parsed, _ncu_units(csv_text))

    if dump_dir is not None:
        out_dir = _Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "61_ncu_metrics.csv").write_text(csv_text)
        (out_dir / "61_ncu_metrics.json").write_text(_json.dumps(parsed, indent=2, default=str))
        print(f"ncu metrics → {out_dir / '61_ncu_metrics.csv'}")
        print(f"ncu metrics → {out_dir / '61_ncu_metrics.json'}")
    elif status_text.strip():
        # No dump dir: the comparison table above is the product; relay the
        # ==PROF== status lines for diagnostics (the raw CSV lands in the dump
        # dir when one is set).
        print(status_text)

    if result.returncode != 0:
        logger.warning("ncu exit=%d", result.returncode)
        if result.stderr.strip():
            print(result.stderr, file=sys.stderr)


# Display order + short labels for the per-kernel comparison table — keys are
# entries of the curated ``_NCU_METRICS`` set.
_NCU_COMPARE_COLS = (
    ("gpu__time_duration.sum", "dur"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "occ%"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm%"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram%"),
    ("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", "fma%"),
    ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "tensor%"),
    ("smsp__inst_executed_pipe_lsu.sum", "lsu.inst"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "ld.cnflct"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "st.cnflct"),
    ("launch__registers_per_thread", "regs"),
)


_NCU_REF_ROWS_MAX = 12  # eager forwards can launch dozens of tiny aten kernels


def _print_ncu_compare(parsed: dict, units: dict[str, str] | None = None) -> None:
    """One aligned table over the parsed ncu metrics: the emmy ``k_*``
    kernels first, then the reference backend's kernels (cuBLAS / cutlass /
    aten) — so the counter deltas (occupancy, SM/DRAM/FMA utilization, smem
    bank conflicts, register pressure) read straight down a column instead of
    across two CSV dumps. Each side sorts by duration; reference rows truncate
    to the :data:`_NCU_REF_ROWS_MAX` slowest with a count of the rest."""
    from emmy.commands.table import Col, render_table  # noqa: PLC0415

    if not parsed:
        return
    dur_key = _NCU_COMPARE_COLS[0][0]
    dep = sorted((k for k in parsed if k.startswith("k_")), key=lambda k: -parsed[k].get(dur_key, 0.0))
    ref = sorted((k for k in parsed if not k.startswith("k_")), key=lambda k: -parsed[k].get(dur_key, 0.0))
    hidden_ref = max(0, len(ref) - _NCU_REF_ROWS_MAX)
    ref = ref[:_NCU_REF_ROWS_MAX]

    def fmt(v) -> str:
        if v is None:
            return "-"
        return f"{v:,.0f}" if abs(v) >= 100 or v == int(v) else f"{v:.1f}"

    def row(side: str, kname: str) -> list[str]:
        shown = kname if len(kname) <= 60 else kname[:57] + "..."
        return [side, shown, *(fmt(parsed[kname].get(metric)) for metric, _ in _NCU_COMPARE_COLS)]

    unit = (units or {}).get(dur_key, "")
    cols = [Col("side"), Col("kernel")] + [
        Col(f"dur ({unit})" if label == "dur" and unit else label, "r") for _, label in _NCU_COMPARE_COLS
    ]
    data = [row("dep", k) for k in dep] + [row("ref", k) for k in ref]
    print()
    print("ncu compare — emmy kernels vs the torch/cuBLAS reference (same counters, one table):")
    for line in render_table(cols, data, rule=True):
        print(line)
    if hidden_ref:
        print(f"  … {hidden_ref} more reference kernel(s) below the duration cut")


def _ncu_units(csv_text: str) -> dict[str, str]:
    """``{metric: unit}`` from the CSV's ``Metric Unit`` column (first seen wins)."""
    import csv as _csv  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    units: dict[str, str] = {}
    for row in _csv.DictReader(_io.StringIO(csv_text)):
        metric, unit = row.get("Metric Name", ""), row.get("Metric Unit", "")
        if metric and unit and metric not in units:
            units[metric] = unit
    return units


def _parse_ncu_csv(csv_text: str) -> dict:
    """Reduce ncu's launch-by-launch CSV into per-kernel metric dicts.

    Each row in the CSV is one (kernel, metric) datum for one launch;
    multi-launch profiling produces many rows per (kernel, metric) which
    we aggregate: ``.sum`` and ``smsp__*`` counters get summed, every
    other metric (percentages, per-thread regs) gets averaged. Returns
    ``{kernel_name: {metric_name: numeric_value}}`` ready to be merged
    with the bench-comparison JSON downstream.

    Keeps every kernel — the emmy ``k_*`` rows AND the reference
    backend's (cuBLAS / cutlass / aten) rows the child's eager forward
    contributes — so :func:`_print_ncu_compare` can put the two sides in
    one table. Consumers split the sides on the ``k_*`` naming convention.
    """
    import csv as _csv
    import io as _io

    reader = _csv.DictReader(_io.StringIO(csv_text))
    per_kernel: dict[str, dict[str, list[float]]] = {}
    for row in reader:
        kname = row.get("Kernel Name", "")
        metric = row.get("Metric Name", "")
        raw = row.get("Metric Value", "").replace(",", "")
        if not (kname and metric and raw):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        per_kernel.setdefault(kname, {}).setdefault(metric, []).append(val)

    out: dict[str, dict[str, float]] = {}
    for kname, metrics in per_kernel.items():
        reduced: dict[str, float] = {}
        for metric, vals in metrics.items():
            if metric.endswith(".sum") or metric.startswith("smsp__"):
                reduced[metric] = sum(vals)
            else:
                reduced[metric] = sum(vals) / len(vals)
        out[kname] = reduced
    return out
