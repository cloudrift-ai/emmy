"""Run an inline torch expression through the emmy CUDA pipeline.

Compiles ``--code`` to CUDA, executes it on real input data, and verifies
correctness against eager PyTorch. With ``--bench``, also benchmarks all
backends (eager, torch.compile, emmy) and prints a comparison table —
the same shape as ``scripts/bench_block.py`` but for arbitrary inline ops.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import sys
import tempfile
import uuid
from collections import namedtuple
from pathlib import Path

from emmy import config
from emmy.commands.compile import add_golden_arg
from emmy.compiler.pipeline.search.pins import (
    PLACEMENT_APPLIED_PINS_HINT,
    PLACEMENT_DECISIONS_HINT,
    pinned_knobs,
    unreproducible_pin_flag,
)

logger = logging.getLogger(__name__)


def register_run_command(subparsers):
    from emmy.commands.compile import add_quantize_arg  # noqa: PLC0415
    from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415

    parser = subparsers.add_parser("run", help="Compile + run a model / inline torch expression on the CUDA backend")
    add_quantize_arg(parser)
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "HuggingFace model ID or .json IR file. A model ID is traced + compiled + executed and "
            "(with --bench) timed end-to-end against the real torch module; a .json IR file behaves "
            "like --ir. Mutually exclusive with --code / --ir."
        ),
    )
    parser.add_argument(
        "--code",
        "-c",
        help=(
            "Inline Python expression whose last statement is a call (same grammar as "
            "``compile --code``). Example: 'torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))'. "
            "Mutually exclusive with the positional input / --ir."
        ),
    )
    parser.add_argument(
        "--adapter",
        choices=["causal-lm", "dit"],
        default="causal-lm",
        help=(
            "Model trace adapter. ``causal-lm`` preserves the existing HuggingFace text-model path; "
            "``dit`` traces one fixed-shape FP16 Diffusers DiT transformer block (requires --layer)."
        ),
    )
    parser.add_argument(
        "--ir",
        help=(
            "Path to a JSON IR dump (any stage: torch / tensor / loop / tile / kernel / cuda). "
            "The remaining lowering passes are run, then the kernel(s) are executed with random "
            "inputs and benchmarked. Skips eager accuracy check (no reference model available). "
            "Equivalent to passing the same .json path as the positional input."
        ),
    )
    add_golden_arg(parser)  # --golden PATH / --realization NAME / --program N; alone, --golden PATH walks every realization
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "With --golden PATH --bench, write each benched realization's emmy and torch.compile timings back into "
            "the file as a `latency:` entry keyed by this card's hardware id. Recording is always deliberate: nothing "
            "else writes that block, so a suite run can never quietly rewrite checked-in data."
        ),
    )
    parser.add_argument(
        "--record-greedy",
        action="store_true",
        help=(
            "With --golden PATH --realization NAME --bench, write the greedy pick's kernel set back into the file as "
            "measured rows: one routing row per kernel-set decision it took and one schedule row per kernel, timed by "
            "the isolated re-bench with the greedy comparison row as the reference. With --pin-route the greedy row is "
            "the kernel set the named row describes; the rows recorded are what price "
            "that set for a compile nothing pins, and what a strict-evidence compile picks it from. Implies "
            "--strict-evidence: every schedule must come from a measured row or a pin, and a piece the prior would "
            "decide stops the record with its name."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index (when the positional input is a model ID). Omit to run the whole model.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_HINT,
        help="Sequence length for full-model tracing when the input is a model ID (default: 512, matching ``DEFAULT_SEQ_HINT``).",
    )
    parser.add_argument("--bench", "-b", action="store_true", help="Benchmark eager / torch.compile / emmy and print a comparison table.")
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="With --bench, time Emmy kernels one launch at a time after L2 eviction; keep evidence separate from hot rows.",
    )
    parser.add_argument(
        "--paged",
        action="append",
        metavar="NAME",
        help="Declare the program input NAME paged, one page per row of its leading axis, as a serving boot below "
        "its device budget does (EMMY_GEN_EXPERTS_GPU): the schedules a paged operand can take are what the compile "
        "offers and the bench times. Repeatable.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Re-launch each kernel under ``ncu`` to collect hardware counters "
        "(SM-active %%, FMA pipe util, L1/DRAM bandwidth, smem bank-conflict %%) and print a "
        "side-by-side table of the emmy kernels vs the torch/cuBLAS reference kernels. "
        "Implies --bench. Skipped if ncu is not on PATH or the user lacks performance-counter permissions.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for --bench (default: 10).")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for --bench (default: 100).")
    parser.add_argument(
        "--bench-backends",
        default=None,
        help=(
            "Comma-separated subset of backends to time under --bench: any of "
            "``eager``, ``tcompile`` (a.k.a. ``torch.compile`` / ``compile``), "
            "or ``emmy``. Falls back to ``EMMY_BENCH_BACKENDS`` env var, "
            "then to the default ``eager,emmy`` (drops the ~0.8 s "
            "torch.compile JIT from the per-case cost). ``emmy`` is "
            "implicit even if omitted."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --ir random inputs (default: 0).")
    parser.add_argument(
        "--ab",
        action="append",
        default=None,
        metavar="KNOBS",
        help=(
            "Compile + bench an extra variant with these knobs pinned (``K1=V1,K2=V2`` — the "
            "``EMMY_KNOBS`` grammar — a hand pin, published to the environment for that one compile) and show it "
            "as a live A/B row beneath the matching greedy kernel in the --bench kernel table, knob diffs red. "
            "Repeatable. Requires --bench and a re-lowerable input (--code / --realization / --ir)."
        ),
    )
    parser.add_argument(
        "--kernel",
        metavar="NAME",
        help=(
            "Run one kernel of --golden PATH as the whole program — its stored body, a cut piece included — by exact "
            "name or an unambiguous name substring. With --tune, tunes that kernel alone: a schedule pin reaches only "
            "it and each compile takes seconds where its layer takes minutes."
        ),
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Autotune the program's one scheduled kernel: measure N of its schedule rows, the prior's ten best first, "
            "then the rows Bayesian optimization expects to improve most, each benched as an --ab row would be. "
            "Clean rows land in the tune DB, so the next compile picks the fastest. Requires --bench and a "
            "re-lowerable input (--code / --realization / --ir)."
        ),
    )
    parser.add_argument(
        "--dynamic",
        action="append",
        default=None,
        metavar="NAME@INPUT:AXIS",
        help=(
            "Make a tensor dim symbolic. Form: ``NAME@INPUT:AXIS`` — axis ``AXIS`` of the "
            "traced input named ``INPUT`` becomes ``Dim(NAME)``. Repeatable. Forwards to "
            "``torch.export(..., dynamic_shapes={...})``; the compiled CUDA kernel signature "
            "gains an ``int <NAME>`` runtime arg per dim. Example: ``--dynamic seq_len@x:1``."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help=(
            "With --bench: also write the whole comparison as machine-readable JSON to PATH. When --golden PATH "
            "walks multiple realizations, PATH is an output directory containing one JSON file per realization. Records include the "
            "backend table (eager / torch.compile / emmy), the per-kernel greedy rows (plus, when "
            "pinned rows benched, the ``greedy.isolated`` emmy-only re-bench — the pinned-comparable "
            "greedy baseline), and every golden / --ab pinned row with its integrity flags "
            "(realized-vs-pinned knob check, arithmetic-intensity floor, wrong-answer check). "
            "Each pinned row and the greedy block carry a ``lane`` (fm/std) so the sweep filters to "
            "the greedy's lane — comparing a pinned [fm] latency against a std greedy is a phantom "
            "regression. Retires ad-hoc table parsing in the golden-sweep workflow."
        ),
    )
    parser.add_argument(
        "--strict",
        dest="strict_correctness",
        action="store_true",
        help=(
            "With --bench on a traced model, runnable frontend IR, or an embedded golden, fail unless every requested backend, "
            "captured timing, exact pin, and direct Emmy-vs-eager comparison is valid."
        ),
    )
    parser.add_argument(
        "--no-record-evidence",
        action="store_true",
        help=(
            "Skip the default bench-to-DB recording. When pinned rows bench (golden / --ab) at tune-standard "
            "quality (warmup >= 5, iters >= 20), each clean row AND the greedy pick's isolated re-bench are recorded "
            "as per-kernel perf rows in the tune DB — the evidence the next compile deploys, and training data once "
            "the tune DB is imported into a dataset. A greedy pick that fails to bench records the kernel it blames "
            "as a bench_fail row. Flagged rows (pin mismatch, wrong answer, intensity floor) and the --ir path never record."
        ),
    )
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts.")
    parser.add_argument("--debug", action="store_true", help="Per-launch tensor dumps in the emmy backend.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase verbosity. Default (no flag): only the accuracy verdict and (with --bench) "
            "the timing tables are printed. -v: also tracer messages and pass / per-rule timings. "
            "-vv: also per-rule application snapshots."
        ),
    )
    from emmy.commands.compile import add_nvcc_args
    from emmy.compiler.target import add_target_arg

    add_nvcc_args(parser)
    add_target_arg(parser, dest="gpu_arch", option="--gpu-arch")
    parser.set_defaults(func=handle_run)


def handle_run(args):
    from emmy.commands.compile import apply_nvcc_flags
    from emmy.compiler.target import CPU_TARGET, apply_target_arg

    apply_nvcc_flags(args, default="")  # run uses nvcc default -O3 (representative codegen)
    cpu = args.gpu_arch == CPU_TARGET
    if not cpu:
        apply_target_arg(args, dest="gpu_arch")
    if args.profile:
        args.bench = True  # --profile re-launches under ncu via the bench path; profiling implies benching
    if getattr(args, "cold_cache", False) and (not args.bench or getattr(args, "pack", None)):
        logger.error("--cold-cache requires --bench on a compiler input, not a pack")
        sys.exit(2)
    if getattr(args, "cold_cache", False) and args.record:
        logger.error("--record stores whole-row comparisons; cold-cache mode measures individual Emmy kernels. Use --record-greedy.")
        sys.exit(2)
    verbose = getattr(args, "verbose", 0)
    if verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.record and not (args.golden and args.bench):
        logger.error("--record requires --golden PATH and --bench")
        sys.exit(2)
    if args.record_greedy and not (args.golden and args.bench):
        logger.error("--record-greedy requires --golden PATH and --bench")
        sys.exit(2)
    if getattr(args, "kernel", None) and (args.record or (args.record_greedy and not args.realization)):
        logger.error("isolated kernel recording requires --realization NAME --record-greedy; --record is a whole-target latency")
        sys.exit(2)
    if getattr(args, "program", None) is not None and (args.record or args.record_greedy):
        logger.error("--program benches a whole traced program; --record / --record-greedy write the rows of one --realization NAME")
        sys.exit(2)
    if (args.ab or getattr(args, "tune", None)) and args.bench and not getattr(args, "no_record_evidence", False):
        # A sweep exists to leave its rows in the tune DB for a later record; below the bench standard
        # it records nothing, and a record run that copies that DB would silently fall to the prior.
        from emmy.compiler.pipeline.search.bench_record import MIN_RECORD_ITERS, MIN_RECORD_WARMUP, meets_quality_bar  # noqa: PLC0415

        if not meets_quality_bar(args.warmup, args.iters):
            logger.error(
                "--ab / --tune sweep at --warmup %d / --iters %d is below the tune bench standard (--warmup >= %d, --iters >= %d), "
                "so none of its rows would reach the tune DB; raise them, or pass --no-record-evidence to measure only",
                args.warmup,
                args.iters,
                MIN_RECORD_WARMUP,
                MIN_RECORD_ITERS,
            )
            sys.exit(2)
    if args.record_greedy:
        # A recorded row must price a schedule a measurement or a pin chose: a piece the prior would decide
        # raises EvidenceError naming it, instead of being written into the golden as measured.
        args.strict_evidence = True
    if args.record or args.record_greedy:
        # A row is evidence only on the card its file names (``golden.documents_for_card``): measurements
        # written under another card's header are what no replay on this card ever reads.
        from emmy.compiler.context import Context  # noqa: PLC0415
        from emmy.compiler.pipeline.search.golden import (
            GoldenFile,  # noqa: PLC0415
            validate_working_gpu,  # noqa: PLC0415
        )

        try:
            validate_working_gpu(GoldenFile.load(args.golden), Context.probe())
        except ValueError as exc:
            logger.error("--record / --record-greedy: %s — record into a file seeded for this card", exc)
            sys.exit(2)
    with (
        config.strict_evidence_override(True if getattr(args, "strict_evidence", False) else None),
        pinned_knobs({"COLD_CACHE": True} if getattr(args, "cold_cache", False) else {}),
    ):
        if cpu:
            _handle_run_cpu(args)
            return
        if args.golden and not args.realization and not getattr(args, "kernel", None) and getattr(args, "program", None) is None:
            _run_golden_targets(args)
            return
        _handle_run_once(args)


def _handle_run_cpu(args) -> None:
    """``run --gpu-arch cpu``: compile the traced input on the CPU backend, check it against eager torch on the CPU and,
    with ``--bench``, time each kernel alone — after cache eviction under ``--cold-cache`` — beside eager torch's whole
    forward. A bench at the tune standard files each kernel's row in the tune DB under the host CPU's context."""
    import statistics  # noqa: PLC0415
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from emmy.commands.compile import load_or_trace  # noqa: PLC0415
    from emmy.compiler.backend.cpu import CpuBackend  # noqa: PLC0415
    from emmy.compiler.backend.cpu.codegen import Unsupported  # noqa: PLC0415
    from emmy.compiler.ir.schedule.cpu import CpuCodec, CpuContext, CpuProblem  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import meets_quality_bar, persist_kernel_perf  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import PerfStats, SearchDB  # noqa: PLC0415
    from emmy.compiler.target import CPU_TARGET  # noqa: PLC0415

    other = [
        flag
        for flag, on in (
            ("--golden", args.golden),
            ("--ir", args.ir),
            ("--ab", args.ab),
            ("--tune", args.tune),
            ("--profile", args.profile),
        )
        if on
    ]
    if other or (args.input is None and args.code is None):
        logger.error("--gpu-arch %s runs a model id or --code%s", CPU_TARGET, f", without {', '.join(other)}" if other else "")
        sys.exit(2)
    graph, _name, bundle = load_or_trace(args)
    module, example_args, example_kwargs = bundle
    checkpoint = None
    if getattr(args, "quantize", None):
        from emmy.commands.compile import _quantize_traced  # noqa: PLC0415

        checkpoint = _quantize_traced(graph, bundle, args)
    backend = CpuBackend(tune_db="auto")
    try:
        program = backend.compile(graph)
    except Unsupported as exc:
        logger.error("the CPU backend cannot compile this program: %s", exc)
        sys.exit(1)
    result, _ = backend.run(program, input_data=_bind_inputs(program.graph, module, example_args, example_kwargs, checkpoint=checkpoint))
    if checkpoint is not None:
        # The graph declares the quantization, so eager torch is no longer its reference: the numpy backend on the
        # same graph is, as on CUDA.
        from emmy.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415

        numpy = NumpyBackend()
        data = _bind_inputs(graph, module, example_args, example_kwargs, checkpoint=checkpoint)
        reference = numpy.run(numpy.compile(graph), input_data=data)[0].outputs
        reference = {name: reference[name] for name in result.outputs}
    else:
        reference = _eager_output(module, example_args, example_kwargs, device="cpu")
    with correctness_oracle():
        err = _check_accuracy(result.outputs, reference)
    if err is not None:
        logger.error("%s", err)
        sys.exit(1)
    if not args.bench:
        return
    samples = program.native.time_launches(warmup=args.warmup, iters=args.iters, cold_cache=args.cold_cache)
    with torch.no_grad():
        for _ in range(args.warmup):
            module(*example_args, **example_kwargs)
        eager = []
        for _ in range(args.iters):
            started = time.perf_counter()
            module(*example_args, **example_kwargs)
            eager.append((time.perf_counter() - started) * 1e6)
    from emmy.commands.table import Col, render_table  # noqa: PLC0415

    decisions = {nid: CpuCodec(CpuContext(CpuProblem(tile, program.ctx))).encode(tile.schedule) for nid, tile in program.tiles.items()}
    rows = [
        [
            program.tiles[nid].name,
            program.kernels[nid].plan.strategy,
            ",".join(f"{key}={value}" for key, value in decisions[nid].items()),
            f"{statistics.median(us):.1f}",
            f"{min(us):.1f}",
        ]
        for nid, us in samples.items()
    ]
    print(f"{program.ctx.hardware_id()}, {program.ctx.host_cpu}, {'cold' if args.cold_cache else 'hot'} cache, each kernel timed alone")
    columns = [Col("Kernel"), Col("Strategy"), Col("Schedule"), Col("Median (us)", "r"), Col("Min (us)", "r")]
    for line in render_table(columns, rows, rule=True):
        print(line)
    total = sum(statistics.median(us) for us in samples.values())
    _print_table({"Eager PyTorch": statistics.median(eager), "Emmy": total}, note="Emmy is the sum of its kernels' medians.")
    if args.json:
        import json  # noqa: PLC0415

        record = {
            "cpu": program.ctx.hardware_id(),
            "arch": program.ctx.arch(),
            "cold_cache": bool(args.cold_cache),
            "sizes": program.native.executor.env(),
            "eager_us": eager,
            "kernels": [
                {"name": program.tiles[nid].name, "strategy": program.kernels[nid].plan.strategy, "knobs": decisions[nid], "us": us}
                for nid, us in samples.items()
            ],
        }
        Path(args.json).write_text(json.dumps(record, indent=1) + "\n")
    if getattr(args, "no_record_evidence", False) or not meets_quality_bar(args.warmup, args.iters):
        return
    db = SearchDB(backend.tune_db)
    try:
        for nid, us in samples.items():
            tile = program.tiles[nid]
            stats = PerfStats(
                median=statistics.median(us),
                min=min(us),
                max=max(us),
                mean=statistics.fmean(us),
                variance=statistics.pvariance(us),
                n_samples=len(us),
            )
            persist_kernel_perf(
                db, program.ctx, tile, stats=stats, status="ok", knobs=decisions[nid], name=tile.name, sizes=program.native.executor.env()
            )
    finally:
        db.close()


def _handle_run_once(args):
    try:
        import torch
    except ImportError:
        logger.error("torch is required: pip install torch")
        sys.exit(1)

    from emmy.commands.compile import load_or_trace, resolve_golden_arg, validate_trace_adapter_args
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.dump import CompilerDump

    if args.golden or args.realization or args.kernel or getattr(args, "program", None) is not None:
        resolve_golden_arg(args)
        if getattr(args, "cold_cache", False):
            if any(row.pins.get("COLD_CACHE") is False for row in getattr(args, "_golden_rows", ())):
                raise ValueError("--cold-cache contradicts the selected golden row's explicit COLD_CACHE=false pin")
    else:
        args.golden_configs = []
    from emmy.commands.compile import selected_regime  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import cold_cache  # noqa: PLC0415

    regime = selected_regime(args)  # the selected rows' input regime, published around the compile and bench below
    cold = cold_cache() or bool(regime.get("COLD_CACHE"))
    if cold and getattr(args, "record", False):
        raise ValueError("--record stores whole-row comparisons; cold-cache mode measures individual Emmy kernels. Use --record-greedy.")
    if (getattr(args, "record", False) or getattr(args, "record_greedy", False)) and not cold:
        args.bench_backends = f"{config.bench_backends_raw(args.bench_backends)},tcompile"
    if sum(x is not None for x in (args.input, args.code, args.ir)) > 1:
        logger.error("input / --code / --ir are mutually exclusive")
        sys.exit(1)
    validate_trace_adapter_args(args)

    # A .json input (via --ir or as the positional) takes the IR path: finish
    # lowering an arbitrary-stage dump, no traced module to bench against torch.
    ir_path = args.ir
    if ir_path is None and args.input is not None and Path(args.input).suffix == ".json" and Path(args.input).exists():
        ir_path = args.input
    # ONE target writes ONE file, and the write is the last thing this command does. A directory
    # there used to reach ``Path(args.json).write_text`` and raise ``IsADirectoryError`` after the
    # bench had run and the recording had been skipped — a whole 14-target ``--record-greedy`` batch
    # reported success per target and recorded nothing. The multi-target walk keeps taking a
    # directory; it hands each target a file inside it, so this never fires there.
    if args.json and (args.json.endswith(("/", os.sep)) or Path(args.json).is_dir()):
        logger.error(
            "--json names ONE file for a single target, but %r is a directory. A directory is for a "
            "multi-target --golden walk, which writes one file per realization inside it.",
            args.json,
        )
        sys.exit(2)
    if args.ab or getattr(args, "tune", None):
        flag = "--ab" if args.ab else "--tune"
        if not args.bench:
            logger.error("%s requires --bench (the A/B rows render in the kernel table)", flag)
            sys.exit(2)
        if args.code is None and ir_path is None and not hasattr(args, "_golden_graph"):
            logger.error("%s requires a re-lowerable input: --code, --realization, or --ir (each config re-lowers a fresh graph)", flag)
            sys.exit(2)
    if args.ab:
        try:
            _ab_samples(args.ab)  # fail fast on a malformed KNOBS spec
        except ValueError as exc:
            logger.error("--ab: %s", exc)
            sys.exit(2)
    if args.strict_correctness and not args.bench:
        logger.error("--strict requires --bench")
        sys.exit(2)
    if not torch.cuda.is_available():
        logger.error("CUDA GPU required")
        sys.exit(1)

    if ir_path is not None or hasattr(args, "_golden_graph"):
        from emmy.commands.compile import selected_decisions  # noqa: PLC0415

        if ir_path is not None:
            args.ir = ir_path
        if getattr(args, "cold_cache", False):
            regime["COLD_CACHE"] = True
        if args.kernel:
            regime.update(selected_decisions(args))  # Keep the selected row's pins live for the existing record guards.
        with pinned_knobs(regime), _family_layouts(args):
            _handle_run_ir(args, CudaBackend, CompilerDump)
        return

    if args.input is None and args.code is None:
        logger.error("Either a model ID / .json input, --code, --golden PATH / --realization NAME, or --ir is required")
        sys.exit(1)

    # Model ID or --code: trace to a frontend graph + keep the runnable module
    # (+ example inputs) so accuracy / --bench compare against real torch.
    graph, _base_name, bundle = load_or_trace(args)

    # Resolve the dump before placing a synthesized checkpoint inside it.
    # ``CompilerDump.__post_init__`` deliberately clears a stale dump directory;
    # constructing it after ``_quantize_traced`` therefore deleted the checkpoint
    # that the isolated bench worker needs to bind packed weights and scales.
    dump = CompilerDump.resolve(args.dump_dir)
    quantized_checkpoint = None
    if getattr(args, "quantize", None):
        from emmy.commands.compile import _quantize_traced  # noqa: PLC0415

        quantized_checkpoint = _quantize_traced(graph, bundle, args)
    module, example_args, example_kwargs = bundle

    if dump:
        dump.dump_input_graph(graph)

    # Backend auto-resolves ``EMMY_TUNE_DB`` env → ``~/.cache/emmy/autotune.db``, created on
    # first use: the card's golden rows are imported into it before the compile picks.
    backend = CudaBackend(debug=args.debug or None, dump=dump, tune_db="auto")
    if backend.tune_db is not None:
        logger.info("Using tuning DB: %s", backend.tune_db)
    compiled = backend.compile(graph)

    # The ncu child of a ``--profile`` invocation skips the accuracy *check*
    # (the parent already verified it) but still launches both sides once —
    # the emmy program AND the eager reference — because the comparison
    # table needs the cuBLAS / aten kernel rows in the captured CSV beside
    # the ``k_*`` rows.
    skip_accuracy = config.ncu_child()
    # ``--quantize`` compiles a DIFFERENT program from the module it traced, on purpose: the
    # graph declares the quantization, so eager torch is no longer its reference and the eager
    # tolerance is not a statement about this program. The comparison still runs and still
    # prints — the delta it reports IS the quantization error, which is worth seeing — but it
    # stops gating. The oracle for a quantized program is the numpy backend on the same graph.
    quantized = bool(getattr(args, "quantize", None))
    if quantized:
        logger.info("--quantize: eager torch is the UNQUANTIZED module, so its delta is reported, not enforced")

    if not args.bench:
        # No bench: the accuracy probe is the command's whole GPU action. It stays
        # in-process — the ``--debug`` per-launch dumps and the ncu child's profiled
        # launches live here — so a hung kernel still poisons this process's stream.
        try:
            input_data = _bind_inputs(compiled, module, example_args, example_kwargs, checkpoint=quantized_checkpoint or args.input)
        except RuntimeError as exc:
            logger.error(exc)
            sys.exit(1)
        try:
            if not skip_accuracy:
                run_result, _ = backend.run(compiled, input_data=input_data)
                if dump and backend.last_debug_result is not None:
                    dump.dump_per_launch_values(backend.last_debug_result.per_launch)
                with correctness_oracle():
                    err = _check_accuracy(run_result.outputs, _eager_output(module, example_args, example_kwargs))
                if err is not None:
                    logger.log(logging.INFO if quantized else logging.ERROR, "%s", err)
                    if not quantized:
                        sys.exit(1)
            else:
                # ncu child: one emmy launch (our metrics) + one eager forward
                # (the reference rows for the comparison table); no accuracy diff.
                backend.run(compiled, input_data=input_data)
                _eager_output(module, example_args, example_kwargs)
        except RuntimeError as exc:
            # Per-launch watchdog fired in ``run_program`` (kernel >1 s).
            # The CUDA context is dirty — bypass Python cleanup so the driver's
            # teardown doesn't block on the still-running kernel.
            sys.stderr.write(f"accuracy check failed: {exc}\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return

    # --bench: the parent does NO GPU work. The greedy comparison (eager / torch.compile /
    # emmy, torch rebuilt in-child from ``trace_args``) and every pinned golden / ``--ab``
    # row run as jobs on ONE persistent SIGKILL-able bench worker — the same isolation and
    # measurement path for every row. A hung kernel dies with the child (the parent stays
    # clean, the row records bench_fail, the next job respawns a fresh context), so no
    # ``os._exit`` and no per-row special-casing. The greedy job also carries the accuracy
    # check (in-child, on the rebuilt module's real inputs) and — when pinned rows exist —
    # returns that run's (inputs, outputs) as their wrong-answer reference. When pinned
    # rows exist the greedy graph is ALSO re-benched emmy-only through the pinned-row path
    # (``_bench_greedy_isolated``): the interleaved greedy number is torch-comparable but
    # NOT pinned-comparable (the documented ~7% environment skew), so pinned speedups read
    # against the isolated row.
    backends = _resolve_backends(args.bench_backends)  # validate the spelling before spending a worker job
    pinned = list(getattr(args, "golden_configs", None) or []) + (_ab_samples(args.ab, dynamic=args.dynamic) if args.ab else [])
    trace_payload = {
        "code": args.code,
        "input": quantized_checkpoint or args.input,
        "adapter": getattr(args, "adapter", "causal-lm"),
        "layer": args.layer,
        "seq_len": args.seq_len,
        "dynamic": list(args.dynamic) if getattr(args, "dynamic", None) else None,
    }

    strict_correctness = bool(getattr(args, "strict_correctness", False))

    async def _bench_session():
        greedy_fail = results = bench = accuracy_error = correctness = ab_ref = golden_benches = greedy_iso = None
        greedy_reference_us = None
        captured = True
        try:
            try:
                resp = await backend.benchmark_compare_async(
                    compiled,
                    torch_spec=("trace_args", trace_payload),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_compare_wall_s(
                        compiled, backend, base_s=_GREEDY_COMPARE_BASE_S, twin=graph if "tcompile" in backends else None
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    accuracy=not (skip_accuracy or quantized),
                    want_ref=bool(pinned or getattr(args, "tune", None)),
                    strict_accuracy=strict_correctness,
                )
            except RuntimeError as exc:
                # Worker SIGKILL (hung kernel), in-child bench budget, EOF — the greedy
                # pick failing to bench is a *finding*: record it, keep going.
                greedy_fail = f"greedy run/bench failed: {exc}"
                _record_greedy_failure(args, backend, compiled, exc)
            else:
                results, bench, captured = resp["results"], resp["result"], resp["captured"]
                accuracy_error, ab_ref = resp["accuracy_error"], resp["run_io"]
                correctness = resp.get("correctness")
                greedy_reference_us = resp.get("reference_run_us")
            if (pinned or getattr(args, "tune", None)) and accuracy_error is None:
                if greedy_fail:
                    logger.error("%s — greedy row marked bench_fail; pinned rows still bench in the worker", greedy_fail)
                ref_key = uuid.uuid4().hex if ab_ref is not None else None
                greedy_iso = await _bench_greedy_isolated(
                    backend, compiled, warmup=args.warmup, iters=args.iters, ref=ab_ref, ref_key=ref_key
                )
                golden_benches = await _bench_golden_variants(
                    backend,
                    args.code,
                    pinned,
                    warmup=args.warmup,
                    iters=args.iters,
                    ref=ab_ref,
                    ref_key=ref_key,
                    quantize=args.quantize,
                    ref_knobs=_cuda_knob_dicts(compiled),
                )
                if getattr(args, "tune", None):
                    golden_benches += await _tune(backend, args, args.code, compiled, ab_ref, ref_key)
        finally:
            await backend.aclose_async_worker()
        return (
            greedy_fail,
            results,
            bench,
            captured,
            accuracy_error,
            golden_benches,
            greedy_iso,
            greedy_reference_us,
            correctness,
        )

    (
        greedy_fail,
        results,
        bench,
        captured,
        accuracy_error,
        golden_benches,
        greedy_iso,
        greedy_reference_us,
        correctness,
    ) = asyncio.run(_bench_session())

    if accuracy_error is not None:
        # Correctness gate: the deployed program computes the wrong answer, so no latency
        # comparison is meaningful (the child skipped the bench; pinned rows were skipped).
        logger.error(accuracy_error)
        sys.exit(1)

    if dump and bench is not None:
        dump.dump_benchmark(bench)
        _dump_bench_compare(dump.dir, results, args.warmup, args.iters)

    if results is not None:
        capture_note = None if captured else "(graph-capture fallback: timings include host launch overhead)"
        notes = [n for n in (_symbolic_bench_note(_collect_sym_env([compiled])), capture_note) if n]
        _print_table(results, note="\n".join(notes) if notes else None)
    _print_kernel_stats(compiled, bench, golden_benches=golden_benches, greedy_fail=greedy_fail, greedy_iso=greedy_iso)
    strict_errors = _strict_benchmark_errors(args, results, bench, captured, correctness, golden_benches) if strict_correctness else None
    if getattr(args, "json", None):
        _write_ab_json(
            args,
            results or {},
            compiled,
            bench,
            golden_benches,
            greedy_fail=greedy_fail,
            greedy_iso=greedy_iso,
            greedy_reference_us=greedy_reference_us,
            correctness=correctness,
            strict_errors=strict_errors,
            accuracy_error=accuracy_error,
            checked=True,
        )
    for error in strict_errors or []:
        logger.error("strict: %s", error)
    _record_bench_evidence(args, golden_benches, greedy_iso)
    if args.profile and greedy_fail is None:
        _run_ncu_profile(args, dump_dir=dump.dir if dump else None)
    if (
        bool(strict_errors)
        or greedy_fail is not None
        or (greedy_iso is not None and greedy_iso.status != "ok")
        or any(gb.status != "ok" for gb in golden_benches or [])
    ):
        sys.exit(1)  # every row is reported above; any failed row (greedy or pinned) exits non-zero


def _bench_total_us(bench) -> tuple[float | None, str | None]:
    """One benchmark's headline microseconds and the semantics they carry.

    A whole-program end-to-end minimum when the run captured one; otherwise the kernel minimum,
    summed across launches. Shared by the ``--json`` record and ``--record`` so a stored latency
    and a reported one are the same number.
    """
    if bench is None:
        return None, None
    e2e_min_ms = getattr(bench, "e2e_min_ms", None)
    if e2e_min_ms is not None:
        return e2e_min_ms * 1000, "whole_program_e2e"
    num_launches = getattr(bench, "num_launches", 0) or len(bench.per_launch or [])
    total_ms = bench.min_ms if bench.min_ms is not None else bench.time_ms
    return total_ms * 1000, "single_launch" if num_launches <= 1 else "per_launch_sum"


def _record_golden_latency(args, results: dict, golden_benches) -> None:
    """Write this card's emmy / torch.compile timings back into the benched working golden.

    The emmy number is the PINNED row's when one was measured, not the greedy pick's: a corpus
    case names a schedule, and recording whatever the prior happened to choose would store a
    timing for a different kernel than the one the file describes.
    """
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import record_latency  # noqa: PLC0415

    # Only a row of the NAMED realization can carry that realization's latency. A piece's row
    # of the same target is benched beside it and holds ONE kernel of the program, so neither its
    # timing nor its schedule knobs describe the row this writes — narrowing by them selects
    # nothing and the write is refused.
    measured = [gb for gb in golden_benches or [] if gb.status == "ok" and gb.bench is not None and gb.sample.name == args.realization]
    # Several measured schedules of one target in one regime: the evidence pick deploys the fastest, so its row
    # carries the target's latency.
    measured = sorted(measured, key=lambda gb: _bench_total_us(gb.bench)[0] or float("inf"))[:1]
    emmy_us = _bench_total_us(measured[0].bench)[0] if measured else results.get("Emmy")
    tcompile_us, eager_us = _torch_timings(results)
    if not emmy_us:
        logger.error("--record measured no Emmy timing for %s", args.realization)
        sys.exit(2)
    if not tcompile_us:
        # Not fatal: the ratchet is `emmy_us`, and some targets have no torch twin to compile.
        logger.warning("--record: no torch.compile timing for %s; storing the timings it has", args.realization)
    record_latency(
        args.golden,
        args.realization,
        hardware_id=Context.probe().hardware_id(),
        emmy_us=emmy_us,
        tcompile_us=tcompile_us,
        eager_us=eager_us,
        knobs=measured[0].sample.knobs if measured else None,
        pins=measured[0].sample.pins if measured else None,
    )
    logger.info(
        "recorded %s: emmy %.2f us (%s)%s%s",
        args.realization,
        emmy_us,
        "pinned row" if measured else "greedy pick",
        f", torch.compile {tcompile_us:.2f} us" if tcompile_us else "",
        f", eager {eager_us:.2f} us" if eager_us else "",
    )


def _torch_timings(results: dict) -> tuple[float | None, float | None]:
    """The torch.compile and eager timings a bench took, in microseconds; a backend that failed reports its error
    text, which is no timing."""
    return tuple(us if isinstance(us, float | int) else None for us in (results.get("torch.compile"), results.get("Eager PyTorch")))


def _seed_latency(results: dict):
    """The whole pick's timings for a record's seed row: Emmy's time beside whatever torch timed, so a target no Torch
    twin times still replaces the latency an earlier pick left on the row. ``None`` when Emmy itself has no time."""
    from emmy.compiler.pipeline.search.golden import Latency  # noqa: PLC0415

    emmy_us = results.get("Emmy")
    if not isinstance(emmy_us, float | int):
        return None
    tcompile_us, eager_us = _torch_timings(results)
    return Latency(emmy_us=emmy_us, tcompile_us=tcompile_us, eager_us=eager_us)


def _record_greedy_pick(args, graph, bench, greedy_iso, taken, results: dict) -> None:
    """Write the greedy pick's kernel set back into the benched working golden as the DB holds it: each kernel-set
    decision the compile took (``taken``, as the splice watcher reports them) a routing row with the kernels it
    names, and each kernel the compile produced a measured row at its isolated launch timing — the pinned-comparable
    number every golden row carries — with the schedules the tune DB had measured for it (``tried``). The greedy
    comparison row, the same graph timed once more beside torch, is every row's reference: the pair checks
    measurement parity, not framework correctness. The whole pick's time beside torch.compile and eager goes onto
    the seed row (``latency``), the number a gap against torch.compile is read from."""
    from emmy.commands.compile import resolve_tune_db  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import measured_schedules  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import record_greedy_pick  # noqa: PLC0415
    from emmy.compiler.pipeline.search.inventory import distinct_decisions  # noqa: PLC0415

    isolated = greedy_iso.bench if greedy_iso is not None and greedy_iso.status == "ok" else None
    nodes = _launch_order_cuda_nodes(graph)
    launches = [list(getattr(side, "per_launch", None) or []) for side in (isolated, bench)]
    if not nodes or any(len(side) != len(nodes) for side in launches):
        logger.error("--record-greedy needs the greedy row and its isolated re-bench timed per kernel for %s", args.realization)
        sys.exit(2)

    def us(launch) -> float:
        # The per-launch median, the statistic the tune DB ranks by and a freeze exports: rows are ranked against
        # each other, and the minimum sample ran up to 10% below every later median on a V100 gate/up projection.
        return launch.time_ms * 1000

    ctx = Context.probe()
    tried = measured_schedules(resolve_tune_db(), ctx, [node.op for node in nodes])
    decisions = distinct_decisions(taken)
    latency = None
    if not getattr(args, "kernel", None):  # An isolated piece's timing is not its containing target's latency.
        timings = _seed_latency(results)
        if timings is None:
            logger.warning("--record-greedy: no Emmy timing for %s; the seed row keeps no latency", args.realization)
        else:
            if timings.tcompile_us is None:
                logger.warning("--record-greedy: no torch.compile timing for %s; storing the timings it has", args.realization)
            latency = (ctx.hardware_id(), timings)
    written = record_greedy_pick(
        args.golden,
        args.realization,
        decisions=decisions,
        kernels=[(node.op, us(mine), us(theirs), n) for node, mine, theirs, n in zip(nodes, *launches, tried, strict=True)],
        reference_backend="same-input-greedy",
        latency=latency,
    )
    logger.info(
        "recorded the greedy pick of %s: %d decision(s) stored as routing rows, %d row(s)", args.realization, len(decisions), len(written)
    )


def _run_golden_targets(args) -> None:
    """Run every target of a golden file sequentially in this process.

    Reached only by a bare ``--golden PATH``; naming one realization with ``--realization NAME`` goes straight down
    the single-run path. Each target kernel — the kernels lowered from a traced program; a piece a decision minted
    runs with its target — runs once per input regime its rows record, named by the shortest row name on it — the seed a
    record run wrote the set under, which ``--record`` writes the whole target's latency onto — or, with no row on
    the target itself, on its pieces. The walk benches each target's measured rows
    (``_explicit_realization`` false), so proposals are not benched as if they were recorded truths. Targets run in
    serving's build order (:func:`_targets_in_build_order`), each compiled with the weight layouts its family settled
    so far.
    """
    from copy import copy  # noqa: PLC0415

    from emmy.compiler.pipeline.passes.tile._layout import settled_layouts  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import GoldenFile  # noqa: PLC0415
    from emmy.serving.twins import LayoutConflict, graph_layouts  # noqa: PLC0415

    if args.input or args.code or args.ir:
        logger.error("--golden is mutually exclusive with positional input / --code / --ir")
        sys.exit(2)
    try:
        document = GoldenFile.load(args.golden)
    except (OSError, ValueError) as exc:
        logger.error("cannot load --golden %s: %s", args.golden, exc)
        sys.exit(2)
    if not document.rows:
        logger.error("--golden contains no realizations: %s", args.golden)
        sys.exit(2)
    from emmy.compiler.pipeline.search.pins import hand_pin_conflicts  # noqa: PLC0415

    # A target recorded under a regime a hand pin contradicts is skipped, not failed: the walk runs the lane the pin names.
    every_target = document.target_rows()
    target_rows = {key: rows for key, rows in every_target.items() if not hand_pin_conflicts(document.shared_regime(rows))}
    if skipped := len(every_target) - len(target_rows):
        logger.warning("%d target(s) recorded under a regime the hand pins contradict are skipped", skipped)
    ordered = _targets_in_build_order(document, target_rows)
    names = [name for name, _, _, _ in ordered]

    output_dir = None
    if len(names) > 1 and args.json:
        output_dir = Path(args.json)
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            logger.error("multi-target --json directory must be empty: %s", output_dir)
            sys.exit(2)
        output_dir.mkdir(parents=True, exist_ok=True)

    # One target's failure (a compile error, a wrong answer, a hung bench) must not hide the targets after it: every
    # target runs and reports, and the walk exits non-zero at the end.
    failed: list[str] = []
    for index, (name, pins, _target, scope) in enumerate(ordered):
        target_args = copy(args)
        target_args._golden_document = document
        target_args._golden_pins = pins
        target_args.realization = name
        target_args._explicit_realization = False
        if output_dir is not None:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "target"
            target_args.json = str(output_dir / f"{index:03d}-{safe_name}.json")
        if scope.layouts:
            logger.info("%s compiles with %s", name, scope.describe())
        try:
            with settled_layouts(scope.settled()):
                _handle_run_once(target_args)
        except SystemExit as exc:
            if exc.code:
                failed.append(name)
        except Exception as exc:  # noqa: BLE001 — one target's lowering error must not hide the rest
            logger.error("%s: %s", name, exc)
            failed.append(name)
        # What the target compiled binds its layouts for the family, whatever its bench did after.
        if (compiled := getattr(target_args, "_compiled", None)) is not None:
            try:
                scope.record(name.split(".", 1)[0], graph_layouts(compiled))
            except LayoutConflict as exc:
                logger.error("%s: %s", name, exc)
                failed.append(name)
    if failed:
        logger.error("%d of %d realizations failed: %s", len(failed), len(names), ", ".join(failed))
        sys.exit(1)


def _targets_in_build_order(document, target_rows) -> list[tuple[str, dict, str, object]]:
    """The targets of ``target_rows`` (:meth:`GoldenFile.target_rows`) as ``(name, pins, target, scope)``, in the order
    serving builds twins (``serving.twins.twins_in_build_order``, which reads the checkpoint tensors each target's
    traced program binds and its input and output sizes): per input regime, a family's programs narrowest first, each
    with its family's layout scope. A target is named by the shortest row name on it or, with none, on its pieces."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.serving.twins import fork_less_weights, twins_in_build_order  # noqa: PLC0415

    targets = [
        (
            min([row.name for row in rows if row.kernel == target] or [row.name for row in rows], key=lambda name: (len(name), name)),
            dict(pins),
            target,
        )
        for (target, pins), rows in target_rows.items()
    ]
    traced = {kernel.ref: kernel.traced for kernel in document.targets()}
    ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name or None)
    programs = {index: document.program(index) for index in set(traced.values())}
    fork_less_of: dict[int, set[str]] = {}

    def fork_less(graph):  # one lowering per traced program, whatever pin sets reach it
        if id(graph) not in fork_less_of:
            fork_less_of[id(graph)] = fork_less_weights(graph, ctx)
        return fork_less_of[id(graph)]

    ordered = []
    for pin_set in dict.fromkeys(tuple(sorted(pins.items())) for _, pins, _ in targets):
        same_pins = [(name, pins, target) for name, pins, target in targets if tuple(sorted(pins.items())) == pin_set]
        reached = {index: programs[index] for index in dict.fromkeys(traced[target] for _, _, target in same_pins)}
        for index, scope in twins_in_build_order(reached, fork_less):
            ordered += [(name, pins, target, scope) for name, pins, target in same_pins if traced[target] == index]
    return ordered


def _family_layouts(args):
    """Compile a named realization with the weight layouts its family has settled when serving builds it: a context,
    empty when the run is no explicit realization of a traced target (the whole-file walk settles its own). The
    family's narrower twins in the same input regime compile first, in serving's build order
    (:func:`_targets_in_build_order`), from the file's evidence through the golden replay path and with no
    hand-pinned decision, as serving compiles them, and each records the layouts it binds — the scope the walk gives
    the target when it reaches it. Recorded alone, a wide twin would measure kernels serving never builds."""
    from copy import copy  # noqa: PLC0415

    from emmy.commands.compile import resolve_golden_arg, resolve_tune_db  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.tile._layout import settled_layouts  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import evidence_scope  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import unpinned_decisions  # noqa: PLC0415
    from emmy.serving.twins import LayoutConflict, graph_layouts  # noqa: PLC0415

    document, rows = getattr(args, "_golden_document", None), getattr(args, "_golden_rows", None)
    if document is None or not rows or getattr(args, "kernel", None) or not getattr(args, "_explicit_realization", True):
        return contextlib.nullcontext()
    path = document.path_to(rows[0].kernel)
    target = path[0].parent if path else rows[0].kernel
    regime = {key: group for key, group in document.target_rows().items() if dict(key[1]) == rows[0].pins}
    ordered = _targets_in_build_order(document, regime)
    scope = next((scope for _, _, ref, scope in ordered if ref == target), None)
    if scope is None:
        return contextlib.nullcontext()
    db = SearchDB.for_compile(resolve_tune_db())
    for name, pins, ref, family in ordered:
        if ref == target:
            break
        if family is not scope:
            continue
        twin = copy(args)
        twin.realization, twin._golden_pins, twin._explicit_realization = name, pins, False
        resolve_golden_arg(twin)
        graph = twin._golden_graph.copy()
        _stage, tail = _replay_stage_and_passes(graph, embedded_golden=True)
        with settled_layouts(scope.settled()), evidence_scope(twin._golden_scope), unpinned_decisions():
            compiled = Pipeline.build(tail).run(graph, db=db)
        try:
            scope.record(name.split(".", 1)[0], graph_layouts(compiled))
        except LayoutConflict as exc:
            logger.error("%s: %s", name, exc)
            sys.exit(2)
    if scope.layouts:
        logger.info("%s compiles with %s", args.realization, scope.describe())
    return settled_layouts(scope.settled())


def _records_evidence(args, *, timed: bool = True) -> bool:
    """Whether this run's bench writes tune-DB evidence: not opted out (``--no-record-evidence``), compiled in the
    deployable regime, and, for a timing (``timed``), benched at the tune bench standard. A run at another
    optimization level (the test suite's ``-Xcicc -O1`` compile) measures a kernel no deploy compiles; a run below
    the standard is a quick look, whose median could displace a recorded row. A ``bench_fail`` row is no timing."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import meets_quality_bar  # noqa: PLC0415

    return (
        not getattr(args, "no_record_evidence", False)
        and float(Context.probe().features().get("H_opt", 3.0)) == 3.0
        and (not timed or meets_quality_bar(args.warmup, args.iters))
    )


def _note_below_standard(args) -> None:
    """Say why a bench that would otherwise record evidence wrote none, when the tune bench standard is the one
    thing it lacks. ``print``, not ``logger.info``: ``emmy run`` sets the root logger to WARNING at default
    verbosity, and a default-on write to the user's tune DB, or its absence, must announce itself."""
    if _records_evidence(args, timed=False) and not _records_evidence(args):
        print(
            f"[record-evidence] --warmup {args.warmup} / --iters {args.iters} below the tune bench standard — "
            f"measurements NOT recorded into the tune DB (raise them or pass --no-record-evidence to silence)"
        )


def _record_greedy_failure(args, backend, graph, exc) -> None:
    """A greedy pick that failed to bench is evidence too (``--no-record-evidence`` opts out): the
    kernel the watchdog named earns its ``bench_fail`` perf row at the run budget's fail sentinel,
    exactly as the tuner files a hung terminal, so the next ``compile`` / ``run`` / ``serve``
    disqualifies that arm instead of electing the same route and hanging again. Recorded only in
    the deployable regime (:func:`_records_evidence`)."""
    if not _records_evidence(args, timed=False):
        return
    from emmy.commands.compile import resolve_tune_db  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import record_bench_failure  # noqa: PLC0415

    db_path = resolve_tune_db()
    blamed = record_bench_failure(db_path, Context.probe(), graph, exc, float(backend.bench_run_timeout_s) * 1_000_000.0)
    if blamed:
        print(f"[record-evidence] bench_fail perf row(s) for {', '.join(blamed)} recorded into {db_path} (opt out: --no-record-evidence)")


def _record_bench_evidence(args, golden_benches, greedy_iso, decisions=()) -> None:
    """Default-on bench-to-DB recording (``--no-record-evidence`` opts out): every clean pinned row
    and the greedy isolated re-bench become per-kernel ``perf`` rows in the tune DB — the measured
    evidence the next compile's greedy pick deploys, which is how a replayed golden or a hand-pinned
    ``--ab`` row becomes what ``compile`` / ``run`` / ``serve`` choose. Records only at tune-standard
    measurement quality, and only in the deployable regime; ``record_perf`` keeps the best
    measurement per kernel. A clean row is an ``ok`` row with NO integrity flag — a flagged ok row
    measured something untrue (wrong answer, intensity floor). ``decisions`` are the kernel-set
    decisions the greedy compile took; once the greedy re-bench is recorded they become ``routing``
    rows, each only when this bench measured all its pieces — a piece that is itself cut again counts
    as measured when the decision that cuts it is stored too. So a fresh nested cut (a cut one of whose
    pieces is cut again), whose piece cut again never runs as a kernel, is priced at the next compile,
    and a strict ``--record-greedy`` can take it with no pinned decisions or schedules. A tune
    (``--tune N``) writes the same rows once more, into a scratch DB frozen as one file beside the card's
    in the freeze directory (``freeze.run_freeze_dir``), so its pool reaches the repository with the
    change it was measured for; the same gate, so ``--no-record-evidence`` turns both off."""
    if greedy_iso is None and not golden_benches:
        return
    if not _records_evidence(args):
        _note_below_standard(args)
        return
    from emmy.commands.compile import resolve_tune_db  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import record_bench_perf, record_bench_routing  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db.freeze import run_freeze_dir, write_run_freeze  # noqa: PLC0415

    benched = ([greedy_iso] if greedy_iso is not None else []) + list(golden_benches or [])
    clean = [gb for gb in benched if gb.status == "ok" and not gb.flags]
    if not clean:
        return
    db_path = resolve_tune_db()
    freeze_dir = run_freeze_dir() if getattr(args, "tune", None) else None
    with tempfile.TemporaryDirectory(prefix="emmy-run-freeze-") as scratch:
        paths = (db_path, Path(scratch) / "run.db") if freeze_dir is not None else (db_path,)
        n_perf = 0
        routing = None
        for gb in clean:
            if gb.graph is not None and gb.bench is not None:
                for path in paths:
                    with pinned_knobs(_sample_replay_knobs(gb.sample) if gb is not greedy_iso else {}):
                        n_gb = record_bench_perf(path, Context.probe(), gb.graph, gb.bench)
                    if gb is greedy_iso and n_gb and decisions:
                        routing = record_bench_routing(path, gb.graph, decisions)
                n_perf += n_gb
        print(f"[record-evidence] {n_perf} kernel perf row(s) recorded into {db_path} — opt out with --no-record-evidence")
        if routing is not None:
            stored, refused = routing
            print(f"[record-evidence] {stored} of {stored + len(refused)} kernel-set decision(s) stored as routing rows")
            for reason in refused:
                print(f"[record-evidence] routing row not recorded — {reason}")
        if freeze_dir is not None and n_perf:
            run_files = write_run_freeze(paths[1], freeze_dir)
            if not run_files:
                logger.warning(
                    "[record-evidence] no row of this tune passes freeze admission (a card the GPU registry does not know, "
                    "or a kernel that no longer lowers); the tune DB keeps them"
                )
            for run_file in run_files:
                print(f"[record-evidence] the tune's rows frozen into {run_file} — commit it with your change")


def _reset_persisting_l2_cache() -> None:
    """Drop every persisting-L2 carveout on the current CUDA context.

    Wraps ``cudaCtxResetPersistingL2Cache`` (CUDA 11+). cuBLAS, cuDNN,
    and other libraries can install ``cudaAccessPolicyWindow`` hints
    that pin specific gmem regions in L2 across kernel launches; once
    set, the carveout persists until the context tears down (it is NOT
    released by ``torch.cuda.empty_cache`` or stream sync). That eats
    L2 bandwidth from later cp.async-staged kernels — the article
    matmul drops from 1660 µs to 330 µs after the reset. Best-effort:
    silently skipped if the runtime symbol isn't resolvable (CUDA < 11,
    non-Linux, etc.); a non-zero return is logged at DEBUG so the
    bench path doesn't fail loud on driver quirks.

    Loads ``libcudart`` from the already-mapped image so the call
    targets the SAME runtime torch is using (the system
    ``libcudart.so`` may belong to a different CUDA install and would
    operate on a different driver context — its reset returns success
    but doesn't touch our context's L2 carveout). Walks
    ``/proc/self/maps`` for the first mapped ``libcudart.so.*`` path.
    """
    import ctypes

    from emmy.compiler.backend.cuda.cache import cuda_runtime

    try:
        cudart = cuda_runtime()
        cudart.cudaCtxResetPersistingL2Cache.restype = ctypes.c_int
        cudart.cudaCtxResetPersistingL2Cache.argtypes = []
        err = cudart.cudaCtxResetPersistingL2Cache()
        if err != 0:
            logger.debug("cudaCtxResetPersistingL2Cache returned %d (continuing)", err)
    except (OSError, AttributeError) as e:
        logger.debug("cudaCtxResetPersistingL2Cache unavailable: %s", e)


def _dump_bench_compare(dump_dir, results: dict, warmup: int, iters: int) -> None:
    """Persist the eager / torch.compile / emmy comparison table
    so downstream tooling (``make bench-kernels``) can parse one file
    per case instead of grepping kernel stdout."""
    import json as _json
    from pathlib import Path as _Path

    eager_us = results.get("Eager PyTorch")
    payload = {
        "warmup": warmup,
        "iters": iters,
        "backends": {name: {"error": us} if isinstance(us, str) else {"latency_us": us} for name, us in results.items()},
    }
    if eager_us:
        for name, us in results.items():
            if not isinstance(us, str):
                payload["backends"][name]["speedup_vs_eager"] = (eager_us / us) if us else 0.0
    out = _Path(dump_dir) / "60_bench_compare.json"
    out.write_text(_json.dumps(payload, indent=2, default=str))


# Base setup budgets (seconds) for the ``run --bench`` comparison jobs' SIGKILL wall caps —
# what the child pays BEFORE any per-kernel work: the ``trace_args`` greedy job reloads the
# model / re-traces the --code expression; the ``--ir`` job only rebuilds a torch reference
# from the shipped frontend snapshot.
_GREEDY_COMPARE_BASE_S = 240.0
_IR_COMPARE_BASE_S = 60.0
# Seconds the cap grants the Torch twin's ``torch.compile`` JIT per node of its program. Inductor at
# ``max-autotune`` prices its matmuls and traces the whole program, which takes minutes on the
# 300-3000-node program of a GDN layer; the cap only decides when the parent kills the child, so
# generous costs nothing while nothing hangs.
_TWIN_JIT_S_PER_NODE = 2.0


def _compare_wall_s(graph, backend, *, base_s: float, twin=None) -> float:
    """Workload-scaled SIGKILL cap for a worker comparison job. A fixed cap false-fails
    legitimate big runs: the child pays a first nvcc compile of EVERY kernel (the in-child
    compile budget bounds each) plus the bench itself, on top of ``base_s`` for its own
    setup — a dozen-kernel model layer legitimately needs several hundred seconds where a
    single --code kernel needs a fraction. ``twin`` is the frontend graph the Torch twin is
    built from when ``torch.compile`` is among the benched backends; its JIT runs inside the
    same job and is granted :data:`_TWIN_JIT_S_PER_NODE` per node. The first-iteration
    watchdog deadline is in the cap too, so a hung kernel is named by the watchdog (a
    ``bench_fail`` row the next pick reads) before the parent's kill, which names none."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    n_kernels = sum(1 for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    jit_s = _TWIN_JIT_S_PER_NODE * len(twin.nodes) if twin is not None else 0.0
    return (
        base_s
        + max(1, n_kernels) * backend.bench_compile_timeout_s
        + config.first_iter_timeout_ms() / 1000.0
        + jit_s
        + backend.bench_run_timeout_s
    )


# One recorded golden config compiled + benched with its knobs pinned this run.
# ``flags`` are the integrity-gate verdicts (empty = clean): the arithmetic-intensity
# floor and the output-correctness check against the greedy or eager reference. ``bench`` is ``None``
# for a row that never benched — ``status`` says why: ``"pin_unmatched"`` (the pinned config
# didn't realize, so benching would measure the planner's own pick under the pin's name) or
# ``"bench_fail"`` (compile/bench of the pinned config failed); ``"ok"`` rows carry a bench.
_GoldenBench = namedtuple("_GoldenBench", "sample graph bench flags status correctness", defaults=("ok", None))


def _lane(knobs: dict) -> str:
    """The precision REGIME a knob dict realizes or explicitly requests: ``"fm"`` for a
    precision-trading input pin, f16-accumulate mma atom, or ``FAST_EXP``; else ``"std"``.
    Successful rows derive it from realized knobs; a failed row falls back to its parsed pins.
    A working-golden target
    pins BOTH the std and the ``[fm]`` config recorded under one name; comparing a pinned ``[fm]``
    latency against a ``"std"`` greedy manufactures a phantom regression, so every A/B row (and the
    greedy it's compared to) carries its lane and the parser filters to matching lanes."""
    from emmy.compiler.pipeline.search.pins import fast_math_knobs, precision_trading_pins  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import PRECISION_KNOBS  # noqa: PLC0415

    input_pins = {}
    for knob in PRECISION_KNOBS:
        if knob.name not in knobs:
            continue
        raw = knobs[knob.name]
        input_pins[knob.name] = raw if isinstance(raw, bool) else knob.parse(str(raw))
    return "fm" if fast_math_knobs(knobs) or precision_trading_pins(input_pins) else "std"


def _graph_lane(graph) -> str:
    """The lane the greedy graph deployed in — ``"fm"`` when ANY realized CUDA kernel's knobs are
    fast-math (any kernel on an f16acc atom ⇒ the run was under the fast-math gate). Judged
    per-kernel, never on a dict union: two kernels both realizing ``TILE`` collide on the key,
    the union keeps only the last kernel's value, and the reported lane becomes
    launch-order-dependent — the phantom-regression trap this feature exists to prevent."""
    return "fm" if any(_lane(d) == "fm" for d in _cuda_knob_dicts(graph)) else "std"


def _pinned_lane(golden_bench) -> str:
    """The realized lane of a pinned row, with its requested lane only as a failure fallback."""
    if golden_bench.graph is not None:
        return _graph_lane(golden_bench.graph)
    return _lane(_sample_replay_knobs(golden_bench.sample))


def _intensity_floor_flag(sample, total_us: float) -> str | None:
    """The arithmetic-intensity floor gate: the FLOP/s a benched row implies from its
    shape must stay below the device's peak dense throughput — a row above it is a wrong
    bench (skipped finalize, silent failure), not a fast kernel (the 8.2 µs "2 PFLOP/s"
    2048³ golden row of the sixth sweep). Returns a flag string, or ``None`` when clean /
    ungateable (no shape, no FLOPs, unknown device peak)."""
    # Use a FLOP count only when the sample source can derive one without guessing. Never reconstruct it from
    # the ShapeKey: the join key excludes symbolic axes on the matmul side but includes them on
    # the reduce-tier/attention side, so the old hint-multiplier formula overcounted reduce-tier
    # ``.dynM`` replays 512× and flagged every one "impossible" at its correct recorded value.
    flops = getattr(sample, "flops", None)
    if flops is None or flops <= 0 or total_us <= 0:
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(gpu.live_name() or "")
    peak = spec.peak_tflops(getattr(sample, "dtype", None) or "fp32") if spec else None
    if peak is None:
        return None
    implied = flops / total_us / 1e6  # FLOP / µs → TFLOP/s
    if implied > peak:
        return f"impossible: implies {implied:.0f} TFLOP/s > {peak:.0f} device peak"
    return None


def _wrong_answer_flag(outputs: dict, ref_outputs: dict, quantized: frozenset[str]) -> str | None:
    """Output-correctness gate for a pinned A/B row: compare the pinned kernel's outputs
    against the greedy run's on the SAME inputs. Both sides are emmy kernels over one
    graph, so they agree to reduction-reorder noise — a large deviation means the pinned
    config computed the wrong answer (the ``g2a`` atomic-split re-bench class: a skipped
    zero-init / finalize benches fast and silently wrong). Returns a flag string or
    ``None``; loose 5% relative tolerance so split-K / atomic reorders never trip it.

    ``quantized`` names the outputs whose buffers are packed 4-bit codes (``f4e2m1x2``, compared decoded). Such an
    output flips a code wherever a value sits on a rounding boundary, which moves that element by a whole
    quantization step, so its worst element alone proves nothing: it flags only when its mean error also exceeds 0.5%
    of its peak. That bound comes from the Qwen3.8-27B-NVFP4 MLP on an RTX 5090, whose encoded output differs from
    both the NumPy and the Torch reference on 1.2% of its codes with mean error 0.17% of peak, under every schedule
    tried, the scalar one included. Every other output flags on its worst element, so a sparse fault (one bad row,
    a wrong tail element) still fails."""
    import numpy as np  # noqa: PLC0415

    assert isinstance(quantized, frozenset), f"quantized must be a frozenset of output names, got {type(quantized).__name__}"
    worst = 0.0
    for nid, ref in ref_outputs.items():
        got = outputs.get(nid)
        if got is None:
            return f"wrong-answer: output {nid!r} missing"
        a = np.asarray(got, dtype=np.float64)
        b = np.asarray(ref, dtype=np.float64)
        if a.shape != b.shape:
            return f"wrong-answer: output {nid!r} shape {a.shape} != greedy {b.shape}"
        if _nonfinite_mismatch(a, b):
            return f"wrong-answer: output {nid!r} has non-finite values the greedy output does not"
        finite = np.isfinite(b)
        if finite.any():
            denom = float(np.abs(b[finite]).max()) or 1.0
            diff = np.abs(a[finite] - b[finite]) / denom
            if nid not in quantized or float(diff.mean()) > 0.005:
                worst = max(worst, float(diff.max()))
    if worst > 0.05:
        return f"wrong-answer: rel err {worst:.3f} vs greedy output"
    return None


def _quantized_outputs(outputs: dict, graph) -> frozenset[str]:
    """The names in ``outputs`` whose buffers in ``graph`` are packed 4-bit codes — the ones
    :func:`_comparison_outputs` decodes, under the same rule."""
    if not hasattr(graph, "buffer"):
        return frozenset()
    return frozenset(name for name in outputs if graph.buffer(name).dtype.name == "f4e2m1x2")


def _nonfinite_mismatch(actual, expected) -> bool:
    """Whether a NaN or an infinity sits where the other side does not hold the same value.

    A mask legitimately holds ``-inf``, so an infinity of one sign at the same position on both sides passes;
    so does a NaN on both sides. Both arguments are float64 arrays of one shape."""
    import numpy as np  # noqa: PLC0415

    finite = np.isfinite(expected)
    return not np.array_equal(np.isfinite(actual), finite) or not np.array_equal(actual[~finite], expected[~finite], equal_nan=True)


def env_pin_refusal(
    kernel_knobs: list[dict],
    placement_knobs: list[dict] | None = None,
    kernel_names: list[tuple[str, ...]] | None = None,
    applied_place_pins: dict[str, str] | None = None,
) -> str | None:
    """The live ``EMMY_<KNOB>`` pins a compiled graph did not realize, or ``None``.

    An ``--ab`` row has always been gated this way (:func:`unreproducible_pin_flag`), because benching a
    row whose pin did not take would measure the planner's own pick under the pin's name. A pin published
    through ``EMMY_KNOBS`` / ``EMMY_<KNOB>`` gates the GREEDY compile instead, and nothing checked it --
    so the same mistake was invisible exactly where a hand-run sweep puts its pins.

    That does not produce a wrong number, it produces an unfalsifiable one: the sweep reports the
    planner's pick under the experiment's name, and a pin that did nothing reads as a flat result rather
    than a broken experiment. Measured while comparing Emmy's int4 decode against its dense kernel, a
    ``WORK``/``TILE``/``STAGE`` pin silently failed to realize at one shape and the run reported the
    greedy's own unscheduled tier with no indication the pin had been ignored.

    Same comparison and same message as the ``--ab`` gate; only the source of the pins differs.
    """
    from emmy.compiler.pipeline.knob import KERNEL_DECISION_FAMILIES, family_pins  # noqa: PLC0415

    pins = {name: value for family in KERNEL_DECISION_FAMILIES for name, value in family_pins(family, kernels=True)}
    return (
        unreproducible_pin_flag(
            pins,
            kernel_knobs,
            placement_knobs=placement_knobs,
            kernel_names=kernel_names,
            applied_place_pins=applied_place_pins,
        )
        if pins
        else None
    )


def record_refusal(
    kernel_knobs: list[dict], accuracy_error: str | None, kernel_names: list[tuple[str, ...]] | None = None, *, benches=()
) -> str | None:
    """A flagged comparison, rejected answer, or unrealized env pin cannot become recorded evidence."""
    flags = [flag for row in benches if row is not None for flag in row.flags or []]
    if flags:
        return "; ".join(flags)
    if accuracy_error is not None:
        return f"it failed the strict accuracy check: {accuracy_error}"
    return env_pin_refusal(kernel_knobs, kernel_names=kernel_names)


REFERENCE_SELF_DISAGREES = (
    "wrong-answer reference unusable: a row realizing the greedy's own config disagrees with the "
    "greedy output, so no row's comparison against it carries information"
)


def resolve_reference_disagreement(verdicts, ref_knobs) -> list[str | None]:
    """The wrong-answer flags a session can actually justify, one per benched row.

    ``verdicts`` is one ``(realized_knob_dicts, wrong_answer_flag_or_None)`` per row and
    ``ref_knobs`` is the greedy's own realized knobs — the configuration whose outputs every row is
    compared against by :func:`_wrong_answer_flag`.

    A row that realized the reference configuration computes the reference. If such a row is flagged
    as disagreeing with it, the reference is not reproducible, and then NO row's comparison against
    it means anything: a genuinely wrong row and a correct one are indistinguishable. Emitting a
    wrong-answer flag per row there is worse than emitting none, because a flag that fires on
    everything is one people learn to skip past — which is how a real deviation gets waved through.

    So in that case every wrong-answer verdict is dropped and the session says once, on the row that
    demonstrates it, that the reference itself is unusable. Every other kind of flag is untouched,
    and where the reference does reproduce the per-row verdicts stand exactly as before.
    """
    verdicts = list(verdicts)
    if not ref_knobs:
        return [flag for _, flag in verdicts]
    witness = next((i for i, (knobs, flag) in enumerate(verdicts) if flag and knobs == ref_knobs), None)
    if witness is None:
        return [flag for _, flag in verdicts]
    return [REFERENCE_SELF_DISAGREES if i == witness else None for i in range(len(verdicts))]


def _comparison_outputs(outputs: dict, graph) -> dict:
    """Decode physical carriers before command-layer correctness checks."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import decode_bf16  # noqa: PLC0415

    if not hasattr(graph, "buffer"):
        return outputs
    decoded = outputs
    for name, value in outputs.items():
        arr = np.asarray(value)
        dtype = graph.buffer(name).dtype
        if dtype.name == "bf16" and arr.dtype == np.uint16:
            if decoded is outputs:
                decoded = dict(outputs)
            decoded[name] = decode_bf16(arr)
        elif dtype.name in _QUANTIZED_FORMATS and arr.dtype == np.uint8:
            if decoded is outputs:
                decoded = dict(outputs)
            decoded[name] = _quantized_values(arr, dtype.name)
    return decoded


#: Mantissa bits of the IEEE float formats, by dtype name: the fewer, the coarser the rounding step.
_FLOAT_MANTISSA_BITS = {"bf16": 7, "f16": 10, "f32": 23, "f64": 52}


def _computed_from_formats(frontend) -> dict:
    """The coarsest IEEE float format each output of ``frontend`` is computed in: the coarsest among the output and
    every value its producing chain computes. Graph inputs and constants do not count (the FP64 evaluation reads them
    exactly), nor do quantized values (the math decodes them to a float first). ``None`` for an output with no float
    value in its chain."""
    from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415

    def coarsest(output: str) -> str | None:
        found, stack, seen = [], [output], set()
        while stack:
            buf = stack.pop()
            node = frontend.producer(buf)
            if buf in seen or node is None or isinstance(node.op, (ConstantOp, InputOp)):
                continue
            seen.add(buf)
            found.append(frontend.buffer(buf).dtype.name)
            stack.extend(node.inputs)
        return min((f for f in found if f in _FLOAT_MANTISSA_BITS), key=_FLOAT_MANTISSA_BITS.get, default=None)

    return {output: coarsest(output) for output in frontend.outputs}


def _output_formats(outputs: dict, graph, computed_from: dict | None = None) -> dict:
    """The format whose rounding step each of ``outputs`` that ``graph`` keeps as a buffer may differ by (see
    ``_strict_correctness_proof``): a quantized output's storage format; for a float output, the coarser of its
    storage format and the format it is computed in (``computed_from``, from ``_computed_from_formats``)."""
    if not hasattr(graph, "buffer"):
        return {}
    stored = {name: graph.buffer(name).dtype.name for name in outputs if graph.buffer(name) is not None}
    computed_from = computed_from or {}
    return {
        name: min(fmt, computed_from.get(name) or fmt, key=_FLOAT_MANTISSA_BITS.__getitem__) if fmt in _FLOAT_MANTISSA_BITS else fmt
        for name, fmt in stored.items()
    }


def _strict_correctness_proof(
    outputs: dict,
    reference_out,
    *,
    reference="eager",
    rtol: float = 1e-3,
    atol: float = 1e-3,
    exact_out=None,
    formats: dict | None = None,
) -> dict:
    """Return a tolerance verdict against one named reference with reproducible error statistics.

    The pass rule is the same elementwise rule used by ``torch.testing.assert_close`` for
    compiler baselines: ``abs(actual - expected) <= atol + rtol * abs(expected)``, and a non-finite
    value must match the reference exactly. Reference outputs may be tensors, a positional tensor
    sequence, or an output-name mapping.

    ``exact_out`` is a high-precision (FP64) evaluation of the same graph on the same inputs. The
    reference is itself a low-precision evaluation, so a different summation order can round a few
    elements one step apart from it — visible past the elementwise rule after a later cancellation —
    without being any less accurate. When the elementwise rule fails and ``exact_out`` is given, the
    proof passes only if every output passes one of two rules against the exact values. Both require
    the candidate's mean absolute error to sit at most ``_EXACT_MEAN_SLACK`` above the reference's.

    - Largest error: the candidate's largest absolute error is at most the reference's plus one
      rounding step of the output's format (``formats``) at the exact value of the candidate's worst
      element (one ULP of a float format, one code of a quantized one; ``_rounding_step``). The two
      sides' worst elements sit at different magnitudes, so the same half-step rounding error is a
      larger absolute error where the values are larger; the step absorbs that and nothing more.
    - One code (quantized outputs only): every candidate value is the reference's or one of the two
      codes beside it (``_adjacent_codes``), and the candidate misses the exact code on no more
      elements than the reference. A value near the midpoint between two codes rounds either way,
      and one code step is a large jump that no float slack covers.

    Without ``exact_out`` (two compiled programs compared on the same inputs, with no FP64
    evaluation to weigh them against), a quantized output whose every value is the reference's or one
    code beside it passes, and ``one_code_outputs`` counts its differing values. This is the rule a
    strict check applies at the quantize nodes inside a program (``_strict_eager_proof``).

    ``formats`` names each output's format (a dtype name, ``_output_formats``): a quantized output's
    storage format; for a float output, the coarser of its storage format and the coarsest float format
    it is computed in. An f32 output cast from a bf16 value carries bf16 rounding errors, and two summation
    orders of that bf16 math land up to a bf16 step apart, thousands of f32 steps. An output without a
    format gets no step, only the largest-error rule, and no one-code pass. A real fault — a wrong element beyond the reference's own
    error and one step, a code two steps off, or a bias across the output — still fails.
    """
    import numpy as np  # noqa: PLC0415

    def _array(value):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    names = list(outputs)
    if isinstance(reference_out, dict):
        missing = [name for name in names if name not in reference_out]
        extra = [name for name in reference_out if name not in outputs]
        if missing or extra:
            return {
                "status": "fail",
                "reference": reference,
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_rel_error": None,
                "error": f"output names differ: missing={missing}, extra={extra}",
            }
        refs = [reference_out[name] for name in names]
    else:
        refs = list(reference_out) if isinstance(reference_out, (tuple, list)) else [reference_out]
        if len(refs) != len(names):
            return {
                "status": "fail",
                "reference": reference,
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_rel_error": None,
                "error": f"output count differs: Emmy={len(names)}, {reference}={len(refs)}",
            }

    max_abs = 0.0
    max_rel = 0.0
    abs_sum = 0.0
    count = 0
    failure = None
    structural = False
    one_code = {}
    for name, ref in zip(names, refs, strict=True):
        actual = _array(outputs[name])
        expected = _array(ref)
        if actual.shape != expected.shape:
            failure = f"output {name!r} shape {actual.shape} != {reference} {expected.shape}"
            structural = True
            break
        if _nonfinite_mismatch(actual, expected):
            failure = f"output {name!r} has non-finite values the {reference} output does not"
            structural = True
            break
        finite = np.isfinite(expected)
        actual, expected = actual[finite], expected[finite]
        absolute = np.abs(actual - expected)
        tolerance = atol + rtol * np.abs(expected)
        if absolute.size:
            max_abs = max(max_abs, float(absolute.max()))
            max_rel = max(max_rel, float((absolute / np.maximum(np.abs(expected), atol)).max()))
            abs_sum += float(absolute.sum())
            count += int(absolute.size)
            fmt = (formats or {}).get(name)
            if (
                exact_out is None
                and fmt in _QUANTIZED_FORMATS
                and not np.all(absolute <= tolerance)
                and np.all(_adjacent_codes(actual, expected, fmt))
            ):
                one_code[name] = int((actual != expected).sum())
            elif failure is None and not np.all(absolute <= tolerance):
                worst = int(np.argmax(absolute - tolerance))
                failure = (
                    f"output {name!r} exceeds rtol={rtol:g}, atol={atol:g}: "
                    f"{int((absolute > tolerance).sum())}/{absolute.size} elements, "
                    f"worst at flat index {worst}: emmy={actual.flat[worst]:.6g} {reference}={expected.flat[worst]:.6g}"
                )

    exact = None
    if failure is not None and not structural and exact_out is not None and count:
        values = exact_out() if callable(exact_out) else exact_out
        exact = _exact_error_comparison(outputs, refs, values, names, _array, formats or {}) if values is not None else None
        if exact is not None and exact["candidate_no_less_accurate"]:
            failure = None
    proof = {
        "status": "fail" if failure else "pass",
        "reference": reference,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs,
        "mean_abs_error": abs_sum / count if count else 0.0,
        "max_rel_error": max_rel,
    }
    if exact is not None:
        proof["exact_comparison"] = exact
    if one_code:
        proof["one_code_outputs"] = one_code
    if failure:
        proof["error"] = f"{failure} (max_abs={max_abs:.3g}, mean_abs={proof['mean_abs_error']:.3g}, max_rel={max_rel:.3g})"
    return proof


def _exact_outputs(frontend, input_tensors: dict, result_outputs: dict) -> dict:
    """The frontend graph evaluated in FP64 on the same inputs, by output name: the exact values a strict
    proof weighs the candidate and the low-precision reference against."""
    import torch  # noqa: PLC0415

    from emmy.compiler.backend import torch_ref  # noqa: PLC0415

    try:
        fn, tensors = torch_ref.build_callable(frontend, input_tensors, compute_dtype=torch.float64)
        with torch.no_grad():
            out = fn(*tensors)
    except Exception as exc:  # noqa: BLE001 — no FP64 evaluation leaves the elementwise verdict standing
        logger.warning("FP64 reference unavailable (%s) — the strict check stays elementwise", exc)
        return None
    return _eager_outputs_by_name(result_outputs, out[: len(result_outputs)] if isinstance(out, tuple) else out, frontend)


#: How far above the reference's mean absolute error, relative to it, a candidate's may sit against the exact values
#: and still count as no less accurate: rounding-level reorderings move it by hundredths of a percent.
_EXACT_MEAN_SLACK = 0.01


def _rounding_step(value: float, fmt: str | None) -> float:
    """One rounding step of storage format ``fmt`` at ``value``: the ULP of a float format, the wider of the two gaps
    beside it on a quantized format's grid of codes. Zero for an unknown format or a non-finite value."""
    import numpy as np  # noqa: PLC0415

    if fmt is None or not np.isfinite(value):
        return 0.0
    magnitude = abs(float(value))
    if fmt in _QUANTIZED_FORMATS:
        grid = _quantized_values(np.arange(256, dtype=np.uint8), fmt)
        grid = np.unique(np.abs(grid[np.isfinite(grid)]))
        below = grid[max(int(np.searchsorted(grid, magnitude, "left")) - 1, 0)]
        above = grid[min(int(np.searchsorted(grid, magnitude, "right")), len(grid) - 1)]
        return float(above - below) if magnitude not in grid else float(max(above - magnitude, magnitude - below))
    if fmt == "bf16":
        # 7 stored mantissa bits; below the smallest normal (2**-126) the spacing stays 2**-133.
        return float(np.ldexp(1.0, max(int(np.frexp(magnitude)[1]), -125) - 8)) if magnitude else 2.0**-133
    floats = {"f16": np.float16, "f32": np.float32, "f64": np.float64}
    return float(np.spacing(floats[fmt](magnitude))) if fmt in floats else 0.0


def _exact_error_comparison(outputs: dict, refs: list, exact_out, names: list, to_array, formats: dict) -> dict:
    """The candidate's and the reference's largest and mean absolute errors against a high-precision evaluation,
    per output, and whether the candidate is no less accurate on every one (see ``_strict_correctness_proof``)."""
    import numpy as np  # noqa: PLC0415

    exacts = (
        [exact_out[name] for name in names]
        if isinstance(exact_out, dict)
        else (list(exact_out) if isinstance(exact_out, (tuple, list)) else [exact_out])
    )
    per_output, ok = {}, len(exacts) == len(names)
    for name, ref, ex in zip(names, refs, exacts, strict=False):
        exact = to_array(ex)
        actual, expected = to_array(outputs[name]).reshape(exact.shape), to_array(ref).reshape(exact.shape)
        finite = np.isfinite(exact)
        cand, base = np.abs(actual - exact)[finite], np.abs(expected - exact)[finite]
        fmt = formats.get(name)
        stats = {
            "candidate_max": float(cand.max()) if cand.size else 0.0,
            "candidate_mean": float(cand.mean()) if cand.size else 0.0,
            "reference_max": float(base.max()) if base.size else 0.0,
            "reference_mean": float(base.mean()) if base.size else 0.0,
            "rounding_step": _rounding_step(exact[finite].flat[int(cand.argmax())], fmt) if cand.size else 0.0,
        }
        largest = stats["candidate_max"] <= stats["reference_max"] + stats["rounding_step"]
        if fmt in _QUANTIZED_FORMATS:
            stats["candidate_misses"] = int((cand > 0).sum())
            stats["reference_misses"] = int((base > 0).sum())
            stats["beyond_one_code"] = int((~_adjacent_codes(actual[finite], expected[finite], fmt)).sum())
            one_code = stats["beyond_one_code"] == 0 and stats["candidate_misses"] <= stats["reference_misses"]
        else:
            one_code = False
        per_output[name] = stats
        ok = ok and (largest or one_code) and stats["candidate_mean"] <= stats["reference_mean"] * (1 + _EXACT_MEAN_SLACK)
    return {"per_output": per_output, "candidate_no_less_accurate": bool(ok)}


#: The storage formats a frontend node can round a computed value into. Two correct evaluations of such a node land one
#: representable value apart wherever the value sits near the midpoint between two codes, and the layers after it
#: amplify that step past any elementwise tolerance.
_QUANTIZED_FORMATS = frozenset({"f4e2m1x2", "f8e4m3", "f8e5m2"})


def _quantized_twin(frontend) -> bool:
    """Whether the twin rounds a computed value into a quantized storage format (a quantized input is data, not a
    rounding). Two correct evaluations of such a twin land codes apart, so no elementwise verdict holds torch.compile's
    output to eager's: the column is timed unchecked, and the record says so."""
    from emmy.compiler.provenance import is_boundary  # noqa: PLC0415

    return frontend is not None and any(
        node.output.dtype.name in _QUANTIZED_FORMATS and not is_boundary(node.op) for node in frontend.nodes.values()
    )


def _quantize_nodes(frontend, lowered) -> tuple[str, ...]:
    """The quantize nodes at which a strict check adopts the candidate's values (see ``_strict_eager_proof``).

    A quantize node is a frontend node that rounds a computed value into a quantized storage format and is not a graph
    output. The lowered program must keep it as a buffer of the same name, dtype and shape. The check compares a
    quantize node the lowered program does not keep like the rest of the graph, without adoption."""
    from emmy.compiler.provenance import is_boundary  # noqa: PLC0415

    def kept(name: str) -> bool:
        buffer, tensor = lowered.buffer(name), frontend.nodes[name].output
        return buffer is not None and buffer.dtype == tensor.dtype and tuple(buffer.shape) == tuple(tensor.shape)

    return tuple(
        name
        for name in frontend.topological_order()
        if frontend.nodes[name].output.dtype.name in _QUANTIZED_FORMATS
        and not is_boundary(frontend.nodes[name].op)
        and name not in frontend.outputs
        and kept(name)
    )


def _graft_candidate(frontend, nodes: tuple[str, ...]):
    """``frontend`` with every consumer of each of ``nodes`` reading a new input ``<node>_candidate`` instead. The graft
    appends the nodes to the graph outputs, so the eager graph's own quantized value at each one comes out after the
    original outputs."""
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    graph = frontend.copy()
    for name in nodes:
        tensor = graph.nodes[name].output
        fed = graph.add_node(op=InputOp(), inputs=[], output=Tensor(f"{name}_candidate", tensor.shape, tensor.dtype.name))
        for user in graph.consumers(name):
            graph.replace_input(user, name, fed)
    graph.outputs = [*frontend.outputs, *nodes]
    return graph


def _quantized_values(raw, fmt: str):
    """The numbers the raw bytes of a quantized storage format encode (two values per byte for ``f4e2m1x2``)."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import decode_f4x2, decode_f8  # noqa: PLC0415

    raw = np.asarray(raw, dtype=np.uint8)
    return decode_f4x2(raw) if fmt == "f4e2m1x2" else decode_f8(raw, fmt)


def _adjacent_codes(ours, theirs, fmt: str):
    """Where ``ours`` is ``theirs`` or one of the two values beside it on the format's grid of representable values:
    the one step a value near the midpoint between two codes rounds either way across. A non-finite reference value
    accepts only itself."""
    import numpy as np  # noqa: PLC0415

    grid = _quantized_values(np.arange(256, dtype=np.uint8), fmt)
    grid = np.unique(grid[np.isfinite(grid)])
    at = np.searchsorted(grid, theirs)
    beside = (ours == grid[np.maximum(at - 1, 0)]) | (ours == grid[np.minimum(at + 1, len(grid) - 1)])
    return (ours == theirs) | (np.isfinite(theirs) & beside)


def _strict_eager_proof(frontend, input_tensors: dict, outputs: dict, lowered, taps: dict) -> tuple[dict, dict, object]:
    """The ``_strict_correctness_proof`` verdict on the compiled program's ``outputs`` (the candidate) against the
    eager evaluation of ``frontend`` on the same inputs.

    ``taps`` holds the candidate's values at the quantize nodes (``_quantize_nodes``). The eager reference adopts them
    and computes everything after each quantize node from them, so the tolerance check covers every computation
    except the rounding at those nodes, which the proof judges code by code instead: the candidate's value must be the
    reference's or one of the two values beside it on the format's grid (``_adjacent_codes``). Comparing bytes there
    would fail correct programs — a value near the midpoint between two codes rounds either way, and the layers after
    it amplify that one step past the tolerance — while a value further away fails the proof like any other mismatch,
    naming the node and the count. The verdict's ``quantize_nodes`` reports, per node, how many values the reference
    rounds differently from the candidate and how many of those lie beyond one code. Returns the verdict, the
    reference outputs by name, and a zero-argument callable for their FP64 evaluation."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from emmy.compiler.backend import torch_ref  # noqa: PLC0415

    graph = _graft_candidate(frontend, tuple(taps)) if taps else frontend
    device = next((t.device for t in input_tensors.values() if torch.is_tensor(t)), torch.device("cpu"))
    tensors = {**input_tensors, **{f"{name}_candidate": torch.from_numpy(np.ascontiguousarray(v)).to(device) for name, v in taps.items()}}
    fn, args = torch_ref.build_callable(graph, tensors)
    with torch.no_grad(), correctness_oracle():
        out = fn(*args)
    program_out = out[: len(outputs)] if taps else out
    eager_values = _eager_outputs_by_name(outputs, program_out, lowered)
    memo: list = []

    def exact_values():
        if not memo:
            memo.append(_exact_outputs(graph, tensors, outputs))
        return memo[0]

    formats = _output_formats(outputs, lowered, _computed_from_formats(graph))
    proof = _strict_correctness_proof(outputs, eager_values, exact_out=exact_values, formats=formats)
    if taps:
        rounding, faults = {}, []
        for name, reference in zip(taps, out[len(outputs) :], strict=True):
            fmt = frontend.nodes[name].output.dtype.name
            ours, theirs = _quantized_values(taps[name], fmt), _quantized_values(reference.cpu().numpy(), fmt)
            differ, beyond = ours != theirs, ~_adjacent_codes(ours, theirs, fmt)
            rounding[name] = {
                "values": int(differ.size),
                "rounded_differently": int(differ.sum()),
                "beyond_one_code": int(beyond.sum()),
                "max_abs_difference": float(np.abs(ours - theirs)[differ].max()) if differ.any() else 0.0,
            }
            logger.info("quantize node %s: %d/%d values rounded differently", name, rounding[name]["rounded_differently"], differ.size)
            if beyond.any():
                faults.append(f"quantize node {name!r}: {int(beyond.sum())}/{differ.size} values more than one code from the eager value")
        proof["quantize_nodes"] = rounding
        if faults:
            proof["status"] = "fail"
            proof["error"] = "; ".join([proof["error"], *faults] if "error" in proof else faults)
    return proof, eager_values, exact_values


def _eager_outputs_by_name(outputs: dict, eager_out, graph=None) -> dict:
    """Map eager outputs to lowered names, preserving packed carriers for decoding."""
    refs = list(eager_out) if isinstance(eager_out, (tuple, list)) else [eager_out]
    if len(refs) != len(outputs):
        return {}
    mapped = {}
    for name, ref in zip(outputs, refs, strict=True):
        if hasattr(ref, "detach"):
            tensor = ref.detach()
            if graph is not None and graph.buffer(name).dtype.name in _QUANTIZED_FORMATS:
                import torch  # noqa: PLC0415

                tensor = tensor.contiguous().view(torch.uint8)
            try:
                ref = tensor.cpu().numpy()
            except TypeError:
                ref = tensor.float().cpu().numpy()
        mapped[name] = ref
    return _comparison_outputs(mapped, graph) if graph is not None else mapped


def _cuda_knob_dicts(graph) -> list[dict]:
    """Raw ``op.knobs`` per ``CudaOp`` of a compiled pinned graph, in launch order —
    the realized side of the pin gate."""
    return [dict(n.op.knobs or {}) for n in _launch_order_cuda_nodes(graph)]


def _cuda_kernel_names(graph) -> list[tuple[str, str]]:
    """The kernel name and node id of each ``CudaOp`` beside :func:`_cuda_knob_dicts` — what a kernel pin names."""
    return [(n.op.kernel_name, n.id) for n in _launch_order_cuda_nodes(graph)]


def _placement_knob_dicts(graph) -> list[dict]:
    """Placement receipts from the final greedy resolution of ``graph``."""
    return list(graph.hints.get(PLACEMENT_DECISIONS_HINT, []))


def _applied_place_pins(graph) -> dict[str, str]:
    """Piece-qualified PLACE source receipts from the final cut resolution."""
    return dict(graph.hints.get(PLACEMENT_APPLIED_PINS_HINT, {}))


def _ab_samples(specs, dynamic=None, route=None):
    """One shapeless pseudo-sample per ``--ab "K1=V1,K2=V2"`` spec: ``.knobs`` holds
    schedule pins, ``.pins`` Boolean input pins, ``.name`` the table label, and ``.shape None`` —
    the marker :func:`_print_kernel_stats` uses to nest the row by the benched
    kernel's own ``S_*`` signature instead of a golden's matmul shape. ``dynamic``
    stamps the run's own ``--dynamic`` specs on each pseudo-sample so the A/B
    re-trace builds the same symbolic graph as the greedy run. ``route`` is the kernel-set decisions
    ``--pin-route`` pins on the greedy compile: each row compiles under them too, so a row differs
    from the greedy only by its own knobs."""
    from types import SimpleNamespace  # noqa: PLC0415

    from emmy.compiler.pipeline.knob import KnobType, family_of, get, parse_knob_spec  # noqa: PLC0415

    dyn = tuple(dynamic) if dynamic else None
    samples = []
    for raw in specs:
        parsed = parse_knob_spec(raw)
        pins = {
            name: value
            for name, value in parsed.items()
            if (descriptor := get(family_of(name))) is not None and descriptor.type is KnobType.BOOL
        }
        knobs = {name: value for name, value in parsed.items() if name not in pins}
        samples.append(SimpleNamespace(name=f"ab {raw}", pins=pins, knobs=knobs, route=dict(route or {}), shape=None, dynamic=dyn))
    return samples


def _sample_replay_knobs(sample) -> dict:
    """All exact knob pins needed to reproduce a golden winner or explicit A/B row."""
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import precision_pins  # noqa: PLC0415

    pins = dict(getattr(sample, "pins", {}))
    if getattr(sample, "record", None) is not None:
        pins.update(precision_pins(pins))
    return {**pins, **getattr(sample, "route", {}), **dict(tuning_knob_items(sample.knobs))}


def _failed_bench_status(exc: BaseException) -> str:
    """The status a raised bench earns. A compile-budget overrun measured nothing about the
    kernel, so it gets its own status, which no recorder files as a failure — recording it would put
    a false negative into the very rows the prior trains on. Anything else compiled and then failed,
    which IS evidence about the config, and stays the honest ``bench_fail`` it has always been."""
    from emmy.compiler.backend.cuda.program import compile_budget_overrun  # noqa: PLC0415

    return "compile_timeout" if compile_budget_overrun(exc) else "bench_fail"


async def _bench_golden_variants(
    backend,
    source,
    golden_configs,
    *,
    warmup,
    iters,
    ref=None,
    strict_correctness=False,
    strict_reference="eager",
    quantize=None,
    unverified=None,
    ref_knobs=None,
    ref_key=None,
    computed_from=None,
):
    """Compile + bench each recorded golden config with its knobs pinned — one
    ``_GoldenBench`` per config so :func:`_print_kernel_stats` can show each as a measured
    row beside the greedy pick. ``golden_configs`` are the duck-typed rows
    :func:`~emmy.commands.compile.golden_row` builds from golden records (``pins`` hold the
    input regime, ``knobs`` the tunable-only row — ``S_*`` / ``H_*`` features are not knobs)
    or the shapeless ``--ab`` pseudo-samples from :func:`_ab_samples` (same duck type). Both are
    MEASURED under a hand pin published to the environment for that one compile: the pin forces
    the row so its latency can be taken; deploying it is the evidence pick's business once the
    measurement is a row in the tune DB (:func:`_record_bench_evidence`). Each config re-traces a
    **fresh** graph from ``code`` — a frontend graph can't be re-compiled (the first lowering
    mutates it in place, so a reused graph would yield the first config's kernel every time). A sample carrying ``dynamic``
    specs (a dynamic golden, or an ``--ab`` row of a ``--dynamic`` run) re-traces
    symbolically, so the pinned kernel is the same masked-tile artifact the greedy
    run deployed and benches at the same hint. A config that fails to compile / bench
    is kept as a ``bench_fail`` row (never dropped — the table / ``--json`` show why).

    Compiling (trace + lowering) is parent CPU work; every GPU touch is one job on the
    backend's persistent SIGKILL-able worker (``bench_pinned_async``) — the same path the
    greedy row measured through, so all rows share one timing semantics, and a pinned
    config that hangs dies with the child: the row records ``bench_fail``, the parent's
    CUDA context stays clean, and the next row's job respawns a fresh child.

    **A pin matching no offered row fails the row loudly, before any GPU time**: the
    realized-vs-pinned check (:func:`unreproducible_pin_flag`) runs right after the pinned
    compile, and a miss marks the row ``pin_unmatched`` / NOT benched. Benching the fallback
    realization would measure the planner's own pick under the pin's name — the silent-degrade
    class that misled the hd256 flash sweep (a misspelled pin read as a form refusal).

    Two more integrity gates flag (never drop) each benched row: the arithmetic-intensity
    floor (:func:`_intensity_floor_flag`) and — when ``ref`` carries reference inputs and
    outputs — an output-correctness check. The default check compares against the greedy
    Emmy output with :func:`_wrong_answer_flag`; ``strict_correctness`` compares every pinned
    row directly against ``strict_reference`` under rtol=atol=1e-3 and records error statistics;
    ``computed_from`` (``_computed_from_formats`` of the frontend program) sets each float output's
    rounding step there, since the compiled ``source`` no longer holds the values it is computed from.
    Flags render as a ``!`` marker in the table and ride the ``--json`` record."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs  # noqa: PLC0415

    out = []
    ref_inputs, ref_outputs, *ref_exact = ref if ref is not None else (None, None)
    if strict_correctness and (ref_inputs is None or ref_outputs is None):
        raise ValueError("strict pinned correctness requires same-input reference outputs")
    # Session-unique cache key: the (potentially hundreds-of-MB) reference inputs cross
    # the worker pipe once per child, not once per row (see benchmark_pinned_isolated_async).
    ref_key = ref_key or (uuid.uuid4().hex if ref_inputs is not None else None)
    # Row index -> its deferred wrong-answer verdict and the knobs it actually realized.
    wrong_answer: dict[int, str | None] = {}
    realized: dict[int, list] = {}
    for sample in golden_configs or []:
        replay_knobs = _sample_replay_knobs(sample)
        # String sources must be retraced at the recorded symbolic shape. An
        # embedded stable program already carries its symbolic dimensions.
        dyn = getattr(sample, "dynamic", None) if isinstance(source, str) else None
        # ``unverified`` rides every row of a session with no reference outputs, so a measurement
        # taken without an output check is never mistaken for a checked one.
        flags = [unverified] if unverified else []
        try:
            dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs(list(dyn))) if dyn else None
            with pinned_knobs(replay_knobs):
                # Fresh graph; lowering mutates it and bakes the pins into the kernel.
                if isinstance(source, str):
                    graph, _, bundle = graph_from_code(source, dynamic_shapes=dynamic_shapes)
                    if quantize:
                        # ``--ab`` must re-lower the same quantized program as the primary
                        # compile. The old path re-traced only the unquantized module, so its
                        # rows silently benchmarked ordinary float linear kernels under NVFP4
                        # knob names. Keep the temporary checkpoint private to this variant;
                        # pinned timing has no eager correctness reference for quantized code.
                        import tempfile

                        from emmy.compiler.loader.synthesize import quantize_and_spell

                        quantize_and_spell(graph, bundle, tempfile.mkdtemp(prefix="emmy-ab-"), scheme=quantize)
                else:
                    graph = source.copy()
                g_compiled = backend.compile(graph)
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own bench table
            logger.warning("[golden] %s: compile of the pinned config failed (%s) — row kept as bench_fail", sample.name, exc)
            out.append(_GoldenBench(sample, None, None, [f"compile failed: {exc}"], "bench_fail"))
            continue
        # The row's own pins AND the live env pins: a sweep publishes its route through
        # ``EMMY_KNOBS`` and varies schedules per row, so a row whose compile dropped that route
        # would otherwise bench the planner's own kernel set under the row's name.
        with pinned_knobs(replay_knobs):
            names = _cuda_kernel_names(g_compiled)
            flag = unreproducible_pin_flag(
                replay_knobs,
                _cuda_knob_dicts(g_compiled),
                placement_knobs=_placement_knob_dicts(g_compiled),
                kernel_names=names,
                applied_place_pins=_applied_place_pins(g_compiled),
            ) or env_pin_refusal(_cuda_knob_dicts(g_compiled), _placement_knob_dicts(g_compiled), names, _applied_place_pins(g_compiled))
        if flag:
            flags.append(f"{flag} — row NOT benched")
            logger.error(
                "[golden] %s: %s — the pinned config did not realize, so benching it would measure the planner's "
                "own pick under the pin's name; fix the pin spelling (row kept unbenched in the table / --json)",
                sample.name,
                flag,
            )
            out.append(_GoldenBench(sample, g_compiled, None, flags, "pin_unmatched"))
            continue
        try:
            with pinned_knobs(replay_knobs):
                g_bench, run_outputs = await backend.bench_pinned_async(
                    g_compiled, run_inputs=ref_inputs, run_inputs_key=ref_key, warmup=warmup, num_iters=iters
                )
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own bench table
            st = _failed_bench_status(exc)
            logger.warning("[golden] %s: bench of the pinned config failed (%s) — row kept as %s", sample.name, exc, st)
            out.append(_GoldenBench(sample, g_compiled, None, [f"{st}: {exc}"], st))
            continue
        run_outputs = _comparison_outputs(run_outputs, g_compiled) if run_outputs is not None else None
        correctness = None
        if run_outputs is not None and ref_outputs is not None:
            if strict_correctness:
                correctness = _strict_correctness_proof(
                    run_outputs,
                    ref_outputs,
                    reference=strict_reference,
                    exact_out=ref_exact[0] if ref_exact else None,
                    formats=_output_formats(run_outputs, g_compiled, computed_from),
                )
                if correctness["status"] != "pass":
                    flags.append(f"strict {strict_reference} correctness failed: {correctness.get('error', 'tolerance exceeded')}")
            elif not strict_correctness:
                # Held back until every row is in: whether this verdict means anything depends on
                # whether the reference reproduces, which only the whole set can answer.
                wrong_answer[len(out)] = _wrong_answer_flag(run_outputs, ref_outputs, _quantized_outputs(run_outputs, g_compiled))
                realized[len(out)] = _cuda_knob_dicts(g_compiled)
        total_us = (g_bench.min_ms if g_bench.min_ms is not None else g_bench.time_ms) * 1000
        flag = _intensity_floor_flag(sample, total_us)
        if flag:
            flags.append(flag)
        out.append(_GoldenBench(sample, g_compiled, g_bench, flags, "ok", correctness))

    resolved = resolve_reference_disagreement(((realized.get(i, []), wrong_answer.get(i)) for i in range(len(out))), ref_knobs)
    for i, flag in enumerate(resolved):
        if flag:
            out[i].flags.append(flag)
    if REFERENCE_SELF_DISAGREES in resolved:
        logger.error("[golden] %s — every per-row wrong-answer verdict dropped as uninformative", REFERENCE_SELF_DISAGREES)
    for row in out:
        for f in row.flags:
            logger.warning("[golden] %s: %s — row flagged (marked ! in the table, flagged in --json)", row.sample.name, f)
    return out


async def _tune(backend, args, source, compiled, ref, ref_key, route=None) -> list:
    """``--tune N``: measure N schedule rows of the program's one scheduled kernel, in batches the autotuner
    proposes from the rows measured so far, each benched exactly as an ``--ab`` row. A row that failed, did not
    realize, or was flagged (a wrong answer among them) counts as a failure, never as a time."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.autotune import Autotuner, prior_scores, schedule_space  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    graph = source.copy() if not isinstance(source, str) else graph_from_code(source)[0]
    ctx = Context.probe()
    db = SearchDB.for_compile(backend.tune_db) if backend.tune_db is not None else None
    op, rows = schedule_space(graph, ctx, db)
    tuner = Autotuner(rows, prior_scores(op, rows, ctx))
    logger.warning("[tune] %s: %d schedule rows; measuring %d", op.name, len(rows), args.tune)
    ref_knobs = _cuda_knob_dicts(compiled)
    out: list = []
    while len(tuner.us) < args.tune and (batch := tuner.propose(min(8, args.tune - len(tuner.us)))):
        specs = [",".join(f"{k}={v}" for k, v in rows[i].items()) for i in batch]
        samples = _ab_samples(specs, dynamic=getattr(args, "dynamic", None), route=route)
        for sample in samples:
            sample.name = "tune " + sample.name.removeprefix("ab ")
        benches = await _bench_golden_variants(
            backend, source, samples, warmup=args.warmup, iters=args.iters, ref=ref, ref_key=ref_key, ref_knobs=ref_knobs
        )
        tuner.observe(
            (i, us if gb.status == "ok" and not gb.flags and us else math.inf)
            for i, gb, (us, _) in ((i, gb, _bench_total_us(gb.bench)) for i, gb in zip(batch, benches, strict=True))
        )
        out += benches
        best = tuner.best()
        logger.warning("[tune] %d/%d measured, best %s", len(tuner.us), args.tune, "none" if best is None else f"{tuner.us[best]:.2f} us")
    return out


async def _bench_greedy_isolated(backend, compiled, *, warmup, iters, ref=None, ref_key=None):
    """Re-bench the greedy deploy's compiled graph emmy-only through the pinned-row worker
    path (``bench_pinned_async``) — the pinned-comparable greedy baseline. The greedy
    comparison row times emmy interleaved with the live torch closures (same warm clocks /
    caches for the eager / torch.compile table — deliberate), which leaves torch allocator
    state and cuBLAS L2 carveouts active while emmy is measured; a pinned row never runs
    torch, so the same config reads ~7% apart between the two positions (worst on split-K
    pairs, whose finalize re-reads the partials workspace). One number can't be both
    torch-comparable and pinned-comparable, so the greedy config benches twice: this row is
    the one pinned golden / ``--ab`` speedups read against. Reuses the already-compiled
    greedy graph — one extra worker job, no recompile. ``ref`` (the greedy run's
    ``(inputs, outputs)``) times it on the inputs the pinned rows are timed on, under the
    pinned rows' ``ref_key``. Returns a ``_GoldenBench`` (status ``ok`` / ``bench_fail``); a
    failure never blocks the pinned rows."""
    from types import SimpleNamespace  # noqa: PLC0415

    sample = SimpleNamespace(name="greedy (isolated)", knobs={}, shape=None, dynamic=None)
    try:
        inputs = ref[0] if ref is not None else None
        g_bench, _ = await backend.bench_pinned_async(compiled, run_inputs=inputs, run_inputs_key=ref_key, warmup=warmup, num_iters=iters)
    except Exception as exc:  # noqa: BLE001 — an iso-bench failure must not abort the pinned rows
        st = _failed_bench_status(exc)
        logger.warning("greedy isolated re-bench failed (%s) — row kept as %s; pinned rows still bench", exc, st)
        return _GoldenBench(sample, compiled, None, [f"{st}: {exc}"], st)
    return _GoldenBench(sample, compiled, g_bench, [], "ok")


def _launch_order_cuda_nodes(graph):
    """CudaOp nodes in the backend's launch order (``graph.topological_order()`` — the order
    ``bench.per_launch`` indexes). Graph dict order diverges from launch order after
    node-splitting passes (a split partial lands after its consumer in dict order), which
    cross-labels the per-kernel timings if used for pairing."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    return [graph.nodes[nid] for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, CudaOp)]


def _print_kernel_stats(graph, bench, golden_benches=None, greedy_fail=None, greedy_iso=None, sym_env=None):
    """Per-kernel breakdown. Pulls structural stats off each ``CudaOp``
    (block / grid / smem), per-launch timings from ``bench.per_launch``,
    and per-kernel hardware attributes read off the compiled cubins
    (register count, achieved theoretical occupancy). One row per kernel
    — quick at-a-glance for spotting which kernel dominates, whether
    register pressure is killing occupancy, etc.

    ``golden_benches`` (from a selected working-golden target, see :func:`_bench_golden_variants`)
    are each their own live compile+bench of a recorded golden config's pinned
    knobs; their kernels print beneath the greedy kernel of matching shape, labeled
    ``golden NAME [fm|std]`` in the Kernel column (the lane tag so an ``[fm]``-vs-``std``
    row is never misread against a ``std`` greedy) — a real A/B, not the recorded number. Their
    ``%`` column is ``--`` (they're not part of the emmy TOTAL), and their knob
    cells are colored red where they differ from the greedy pick (like ``eval``).
    A pinned row that never benched (unmatched pin / bench_fail — ``gb.bench is None``)
    still prints its kernels with ``--`` timings so the failure is visible, never dropped.

    ``greedy_iso`` (a ``_GoldenBench`` from :func:`_bench_greedy_isolated`, present when
    pinned rows benched) is the greedy graph re-benched emmy-only through the pinned-row
    path: each greedy kernel row gets a ``greedy (isolated)`` twin right beneath it (same
    kernel, pinned-row timing semantics) — the number to compare golden / ``--ab`` rows
    against; the greedy row's own µs is torch-comparable, not pinned-comparable (the
    documented ~7% skew). A failed iso re-bench prints as a ``!`` note, its rows skipped
    (they would duplicate the greedy geometry with ``--`` timings).

    ``bench=None`` + ``greedy_fail`` is the degraded mode: the greedy deploy itself failed
    (hung / blew the bench budget) in its worker child, so its rows print with ``--``
    timings and the TOTAL row reads ``bench_fail``. Kernel attributes (regs / occupancy)
    still collect — every bench runs in a child process, so THIS process's CUDA context is
    clean regardless of row failures, and regs/occ are exactly the diagnostic a hung deploy
    needs."""
    from emmy.commands.table import Col, knob_columns, render_table  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp, resolve_dim
    from emmy.compiler.ir.expr import Var  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415
    from emmy.compiler.pipeline.search.dataset import ShapeKey  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import stamps  # noqa: PLC0415

    cuda_nodes = _launch_order_cuda_nodes(graph)
    if not cuda_nodes:
        return

    # Per-kernel best-case (min) to match the min-based comparison table.
    times_by_idx = (
        {} if bench is None else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (bench.per_launch or [])}
    )
    total_us = None if bench is None else (bench.min_ms if bench.min_ms is not None else bench.time_ms) * 1000
    attrs_by_kname = _collect_kernel_attrs(graph)
    occ_limits = _occupancy_limits()

    # Symbolic grids (ceil-div over a dynamic axis) need a concrete env to
    # resolve for display — use each symbolic input dim's hint, so the printed
    # geometry reflects the hint-sized tile the kernel was tuned for.
    if sym_env is None:
        sym_env = {}
        for nid in graph.inputs:
            for dim in graph.buffer(nid).shape:
                if isinstance(dim.expr, Var) and dim.hint is not None:
                    sym_env.setdefault(dim.expr.name, dim.hint)

    def _geom(op, attrs: dict):
        block_dims = [resolve_dim(d, sym_env) for d in op.block]
        grid_dims = [resolve_dim(d, sym_env) for d in op.grid]
        block_threads = block_dims[0] * block_dims[1] * block_dims[2]
        grid_total = grid_dims[0] * grid_dims[1] * grid_dims[2]
        regs = (attrs.get(op.kernel_name) or {}).get("num_regs", 0)
        local = (attrs.get(op.kernel_name) or {}).get("local_size_bytes", "--")
        occ_pct = _theoretical_occupancy(regs, op.smem_bytes, block_threads, occ_limits)
        occ_str = f"{occ_pct:>3.0f}%" if occ_pct is not None else "  --"
        return grid_total, block_threads, op.smem_bytes / 1024, regs, local, occ_str

    def _op_sig(op):
        return ShapeKey.from_s_features(stamps(op))

    used_ab: set[int] = set()
    matched_golden: set[int] = set()

    def _matching(op):
        """Benched pinned variants whose shape matches this kernel — keyed via
        ``ShapeKey.from_s_features`` over the kernel's ``S_*`` stamps, the
        same join key the prior diagnostics match goldens on (so the dtype flag
        splits fp32/fp16 twins here too). A golden carries its ``ShapeKey``
        on ``sample.shape``; a shapeless ``--ab`` entry matches through its own
        benched kernels' signatures and nests under the first greedy kernel it
        matches (``used_ab`` / ``matched_golden`` track placement; every unmatched
        entry — including a golden whose shape matches NO greedy kernel because
        greedy deployed a split partial+finalize pair — is appended after the
        greedy rows, so a row is never silently dropped)."""
        sig = _op_sig(op)
        out = []
        for gb in golden_benches or []:
            if gb.sample.shape is not None:
                if gb.sample.shape == sig:
                    matched_golden.add(id(gb))
                    out.append(gb)
            elif (
                gb.graph is not None
                and id(gb) not in used_ab
                and any(_op_sig(n.op) == sig for n in gb.graph.nodes.values() if isinstance(n.op, CudaOp))
            ):
                used_ab.add(id(gb))
                out.append(gb)
        return out

    # Build row records first (the knob columns are aligned across all rows, so we
    # can't stream): (name, t_us, pct_cell, geom, knobs, ref_knobs_or_None). A
    # golden row's ``ref`` is its matching greedy kernel's knobs — cells that differ
    # are colored red.
    records: list[tuple] = []

    def _gb_rows(gb, ref):
        """Append the rows of one benched pinned variant (golden / --ab), each
        kernel's knobs diffed red against ``ref`` (the greedy pick's knobs). A
        flagged row (unmatched pin / bench_fail / intensity floor / wrong-answer)
        gets a ``!`` marker; an unbenched row (``gb.bench is None``) prints its
        kernels with ``--`` timings (a compile failure has no kernels — one bare
        label row keeps the failure visible)."""
        label = f"golden {gb.sample.name}" if gb.sample.shape is not None else gb.sample.name
        label = f"{label} [{_pinned_lane(gb)}]"  # so an [fm]-vs-std A/B can't be misread
        if gb.flags:
            label = f"! {label}"
        if gb.graph is None:
            records.append((label, None, "--", (0, 0, 0.0, 0, "--", "  --"), {}, ref))
            return
        g_times = (
            {}
            if gb.bench is None
            else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (gb.bench.per_launch or [])}
        )
        g_attrs = {} if gb.bench is None else _collect_kernel_attrs(gb.graph)
        g_nodes = _launch_order_cuda_nodes(gb.graph)
        for gidx, gnode in enumerate(g_nodes):
            gk = dict(tuning_knob_items(gnode.op.knobs or {}))
            records.append(
                (label, g_times.get(gidx) if gb.bench is None else g_times.get(gidx, 0.0), "--", _geom(gnode.op, g_attrs), gk, ref)
            )

    # Per-launch times of the isolated greedy re-bench — same graph, so indexes align
    # 1:1 with ``cuda_nodes`` and each greedy kernel row gets its iso twin beneath it.
    iso_times = (
        None
        if greedy_iso is None or greedy_iso.bench is None
        else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (greedy_iso.bench.per_launch or [])}
    )
    for idx, node in enumerate(cuda_nodes):
        op = node.op
        t_us = None if bench is None else times_by_idx.get(idx, 0.0)
        pct_cell = "--" if total_us is None else f"{((t_us / total_us * 100) if total_us > 0 else 0.0):.1f}%"
        gknobs = dict(tuning_knob_items(op.knobs or {}))
        records.append((op.kernel_name, t_us, pct_cell, _geom(op, attrs_by_kname), gknobs, None))
        if iso_times is not None:
            records.append(("greedy (isolated)", iso_times.get(idx, 0.0), "--", _geom(op, attrs_by_kname), gknobs, gknobs))
        for gb in _matching(op):
            _gb_rows(gb, gknobs)
    # Catch-all: pinned rows that matched no greedy kernel still print — the golden
    # rows attach to the SHAPE (the run), not to a kernel node, so a greedy split
    # partial+finalize deploy can no longer silently drop the shape's golden A/B rows.
    for gb in golden_benches or []:
        if id(gb) not in used_ab and id(gb) not in matched_golden:
            _gb_rows(gb, None)

    kcols, kcells = knob_columns(
        [{k: (str(v), ref is not None and str(ref.get(k)) != str(v)) for k, v in knobs.items()} for *_, knobs, ref in records]
    )
    columns = [
        Col("Kernel"),
        Col("us", "r"),
        Col("%", "r"),
        Col("grid", "r"),
        Col("block", "r"),
        Col("smem", "r"),
        Col("regs", "r"),
        Col("local B", "r"),
        Col("occ", "r"),
        *kcols,
    ]
    data = []
    for rec, kc in zip(records, kcells, strict=True):
        name, t_us, pct_cell, (grid_total, block_threads, smem_kb, regs, local, occ_str) = rec[:4]
        us_cell = "--" if t_us is None else f"{t_us:.1f}"
        data.append(
            [name, us_cell, pct_cell, str(grid_total), str(block_threads), f"{smem_kb:.1f}K", str(regs), str(local), occ_str.strip(), *kc]
        )
    data.append(["TOTAL", "bench_fail" if total_us is None else f"{total_us:.1f}", *[""] * (len(columns) - 2)])
    # TOTAL sums per-launch solo windows (each kernel replayed back-to-back in
    # its own event window); the whole-program row is one window around the
    # full launch sequence — the number the backend comparison table reports.
    if bench is not None and bench.e2e_min_ms is not None:
        data.append(["whole-program (e2e)", f"{bench.e2e_min_ms * 1000:.1f}", *[""] * (len(columns) - 2)])

    print()
    print("knobs (greedy pick); golden / ab rows are red where they differ from the greedy pick:")
    for line in render_table(columns, data):
        print(line)
    if greedy_fail is not None:
        print(f"! greedy: bench_fail — {greedy_fail}")
    for gb in ([greedy_iso] if greedy_iso is not None else []) + list(golden_benches or []):
        for flag in gb.flags:
            print(f"! {gb.sample.name}: {flag}")


def _emmy_correctness(correctness: dict | None, accuracy_error: str | None, *, reference: bool) -> dict:
    """The Emmy row's correctness verdict — a proof, a failure, a loose pass, or nobody checked.

    ``--strict`` builds a proof and it stands as written. Without it the bench still compares
    against eager, on dtype-scaled tolerances wide enough to pass fp16 accumulation drift, and
    that verdict used to reach the log and nothing else. So the record carried a latency and no
    way to tell a number backed by a passing reference from one backed by a failing reference or
    by none at all — and a survey pass ranks on this file. The absence read as a pass.

    ``tolerance`` is what separates the two passing states: ``scaled`` is the wide check, and a
    row that has only that has not been asked the strict question. One kernel here passed the
    scaled check and failed the strict one on 198 of 3.1M elements."""
    if correctness is not None:
        return correctness
    if accuracy_error is not None:
        return {"status": "fail", "reference": "eager", "error": accuracy_error}
    return {"status": "pass", "reference": "eager", "tolerance": "scaled"} if reference else {"status": "unchecked"}


def _write_ab_json(
    args,
    results: dict,
    graph,
    bench,
    golden_benches,
    greedy_fail=None,
    greedy_iso=None,
    greedy_reference_us=None,
    correctness=None,
    strict_errors=None,
    accuracy_error=None,
    checked=False,
    tc_checked=True,
) -> None:
    """``--json PATH``: the whole ``--bench`` comparison as one machine-readable record —
    the backend table (eager / torch.compile / emmy), the per-kernel greedy rows, and every
    ``--golden`` / ``--ab`` pinned row with its recorded reference latencies and integrity
    flags. ``pinned_knobs`` is the exact input-regime-plus-winner map used for replay. This is
    the golden-sweep workflow's parse target (it retires the ad-hoc stdout
    table parsers) and where the intensity-floor / wrong-answer verdicts become fields —
    the confirm-twice rule diffs two of these files instead of two terminal scrollbacks. The
    Emmy row always carries a ``correctness`` verdict, including the two that used to be an
    absence: a non-strict check that FAILED, and a target with no eager reference to check
    against (:func:`_emmy_correctness`; ``checked`` says whether that check ran, whichever
    columns were timed). The torch.compile row carries its own: ``tc_checked`` is false on a
    quantized twin, where the column is timed unchecked (:func:`_quantized_twin`).

    Each kernel row carries ``record_knobs`` — the realized tuning knobs with the exact complete
    classic row validated by :func:`~emmy.compiler.pipeline.knob.complete_kernel_row` — the map
    to copy verbatim into a golden file ``knobs:`` entry. Failure states are fields, not absences: the greedy
    block carries ``status`` (``"bench_fail"`` + ``error`` when the deploy failed) and each
    pinned row its ``status`` (``ok`` / ``pin_unmatched`` / ``bench_fail``) with ``us`` /
    ``total_us`` null where nothing was measured.

    ``greedy_iso`` (present when pinned rows benched) lands as ``greedy.isolated`` —
    the same graph re-benched emmy-only through the pinned-row path, shaped like a pinned
    row (``status`` / ``total_us`` / ``e2e_us`` / ``kernels`` / ``flags``). Sweep tooling
    compares pinned ``e2e_us`` when a multi-kernel program exposes it, otherwise ``total_us``,
    against THIS block; the greedy block's own ``total_us`` is the
    interleaved (torch-comparable) number, ~7% apart from pinned-row semantics.

    Every pinned row and the greedy block carry a ``lane`` (``"fm"`` / ``"std"``, :func:`_lane`):
    a working-golden target pins BOTH lanes recorded under one name, and comparing a pinned
    ``[fm]`` latency against a ``"std"`` greedy is the phantom-regression trap the sweep must not
    fall into — a parser filters ``pinned`` to the rows whose ``lane`` matches ``greedy.lane``
    (which ``greedy.isolated`` shares, being the same graph)."""
    import json as _json  # noqa: PLC0415

    from emmy import gpu  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import complete_kernel_row, schedule_row_key, tuning_knob_items  # noqa: PLC0415

    def _record_row(knobs: dict) -> dict:
        # The canonical row a golden entry for this kernel spells — including a forkless kernel's,
        # which is its OFF anchors alone. A row ``complete_kernel_row`` still refuses is malformed
        # (a missing kernel family, a non-canonical site key), and one such kernel must not sink
        # the whole record, so it lands as its schedule key.
        try:
            return complete_kernel_row(knobs)
        except ValueError:
            return dict(schedule_row_key(knobs))

    def _kernel_rows(g, b) -> list[dict]:
        import hashlib  # noqa: PLC0415

        if g is None:
            return []
        times = {} if b is None else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (b.per_launch or [])}
        rows = []
        for idx, node in enumerate(_launch_order_cuda_nodes(g)):
            op = node.op
            source = getattr(op, "kernel_source", None) or ""
            rows.append(
                {
                    "kernel": op.kernel_name,
                    "us": None if b is None else times.get(idx, 0.0),
                    "smem_bytes": op.smem_bytes,
                    "knobs": {k: str(v) for k, v in tuning_knob_items(op.knobs or {})},
                    "record_knobs": _record_row(op.knobs or {}),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                }
            )
        return rows

    def _timing(b) -> dict:
        if b is None:
            return {"total_us": None, "timing_semantics": None, "captured": None, "num_launches": 0}
        per_launch = b.per_launch or []
        num_launches = getattr(b, "num_launches", 0) or len(per_launch)
        total_us, semantics = _bench_total_us(b)
        return {
            "total_us": total_us,
            "timing_semantics": semantics,
            "captured": bool(getattr(b, "captured", False)),
            "num_launches": num_launches,
        }

    def _e2e_us(b) -> float | None:
        return None if b is None or b.e2e_min_ms is None else b.e2e_min_ms * 1000

    pinned = []
    for gb in golden_benches or []:
        sample = gb.sample
        pinned.append(
            {
                "name": sample.name,
                "kind": "golden" if sample.shape is not None else "ab",
                "lane": _pinned_lane(gb),
                "status": gb.status,
                "pinned_knobs": {k: str(v) for k, v in _sample_replay_knobs(sample).items()},
                **_timing(gb.bench),
                "e2e_us": _e2e_us(gb.bench),
                "kernels": _kernel_rows(gb.graph, gb.bench),
                "flags": list(gb.flags),
                "correctness": gb.correctness,
                "recorded_emmy_us": getattr(sample, "latency_us", None),
                "recorded_ref_us": getattr(sample, "ref_us", None),
            }
        )
    greedy = {
        "status": "ok" if greedy_fail is None else "bench_fail",
        "lane": _graph_lane(graph),
        **_timing(bench),
        "kernels": _kernel_rows(graph, bench),
    }
    # An env pin gates THIS graph, so an unrealized one misrepresents the greedy row itself.
    env_miss = env_pin_refusal(_cuda_knob_dicts(graph), _placement_knob_dicts(graph), _cuda_kernel_names(graph), _applied_place_pins(graph))
    if env_miss:
        greedy["flags"] = [f"{env_miss} — the env pin did not realize, so this row is the planner's own pick"]
        logger.error(
            "%s — EMMY_KNOBS / EMMY_<KNOB> pin did not realize; the greedy row below is the planner's own pick, not the pinned config",
            env_miss,
        )
    if greedy_fail is not None:
        greedy["error"] = greedy_fail
    if greedy_reference_us is not None:
        greedy["reference_run_us"] = greedy_reference_us
    if greedy_iso is not None:
        greedy["isolated"] = {
            "status": greedy_iso.status,
            **_timing(greedy_iso.bench),
            "e2e_us": _e2e_us(greedy_iso.bench),
            "kernels": _kernel_rows(greedy_iso.graph, greedy_iso.bench),
            "flags": list(greedy_iso.flags),
        }
    captured = bool(getattr(bench, "captured", False))
    backend_semantics = "captured_whole_forward" if captured else "uncaptured_forward"
    from emmy.compiler.pipeline.search.space import cold_cache  # noqa: PLC0415

    if cold_cache():
        backend_semantics = "cold_per_kernel_sum"
    backend_rows = {
        name: {"status": "failed", "error": str(us)}
        if isinstance(us, str)
        else {
            "latency_us": us,
            "captured": captured,
            "timing_semantics": backend_semantics,
            **(
                {"correctness": {"status": "pass", "reference": "eager", "tolerance": "scaled", "fullgraph": True}}
                if name == "torch.compile" and tc_checked
                else {"correctness": {"status": "unchecked", "fullgraph": True}}
                if name == "torch.compile"
                else {}
            ),
        }
        for name, us in (results or {}).items()
    }
    if "Emmy" in backend_rows:
        backend_rows["Emmy"]["correctness"] = _emmy_correctness(correctness, accuracy_error, reference=checked)
    eager_us = (results or {}).get("Eager PyTorch")
    if eager_us:
        for name, us in (results or {}).items():
            if not isinstance(us, str):
                backend_rows[name]["speedup_vs_eager"] = eager_us / us if us else 0.0

    payload = {
        "cold_cache": cold_cache(),
        "input": args.code or args.input or getattr(args, "ir", None),
        "golden": getattr(args, "realization", None),
        "dynamic": list(args.dynamic) if getattr(args, "dynamic", None) else [],
        "gpu": gpu.live_name(),
        "warmup": args.warmup,
        "iters": args.iters,
        "backends": backend_rows,
        "greedy": greedy,
        "pinned": pinned,
    }
    if strict_errors is not None:
        payload["strict"] = {"status": "fail" if strict_errors else "pass", "errors": list(strict_errors)}
    out = Path(args.json)
    out.write_text(_json.dumps(payload, indent=2, default=str))
    print(f"A/B record → {out}")


def _collect_kernel_attrs(graph) -> dict[str, dict]:
    """Read attributes from the runtime's cached cubins, using its compile flags and target."""
    from emmy.compiler.backend.cuda.program import kernel_attributes
    from emmy.compiler.backend.plan import KernelSpec

    out = {}
    for node in _launch_order_cuda_nodes(graph):
        op = node.op
        if op.kernel_name in out:
            continue
        try:
            out[op.kernel_name] = kernel_attributes(op.kernel_name, KernelSpec.from_op(op))
        except Exception:  # pragma: no cover — unavailable GPU/compiler or a failed kernel
            continue
    return out


def _occupancy_limits() -> dict | None:
    """Per-device limits used to estimate theoretical occupancy. ``None`` when no device answers."""
    from emmy.compiler.backend.cuda.device import properties

    props = properties()
    if props is None:
        return None
    return {
        "max_threads_per_sm": int(props["max_threads_per_sm"]),
        "max_blocks_per_sm": int(props["max_blocks_per_sm"]),
        "max_regs_per_sm": int(props["regs_per_sm"]),
        "max_smem_per_sm": int(props["smem_per_sm"]),
        "warp_size": int(props["warp_size"]),
    }


def _theoretical_occupancy(regs_per_thread: int, smem_per_block: int, threads_per_block: int, limits: dict | None) -> float | None:
    """Active-warps-per-SM ÷ peak-warps-per-SM × 100. Computed from the
    static-occupancy limits: register file, shared memory, and per-SM
    block / thread caps. Doesn't account for the dynamic-only spill +
    stack overhead but is enough to flag occupancy cliffs (regs > 64
    drops most consumer GPUs from 100% → 50%, smem > 49KB likewise)."""
    if not limits or threads_per_block <= 0 or regs_per_thread <= 0:
        return None
    warp_size = limits["warp_size"]
    max_warps = limits["max_threads_per_sm"] // warp_size
    if max_warps <= 0:
        return None

    blocks_by_threads = limits["max_threads_per_sm"] // threads_per_block
    blocks_by_blocks = limits["max_blocks_per_sm"]
    blocks_by_regs = limits["max_regs_per_sm"] // max(regs_per_thread * threads_per_block, 1)
    blocks_by_smem = limits["max_smem_per_sm"] // max(smem_per_block, 1) if smem_per_block > 0 else blocks_by_threads
    active_blocks = max(0, min(blocks_by_threads, blocks_by_blocks, blocks_by_regs, blocks_by_smem))
    active_warps = active_blocks * (threads_per_block // warp_size)
    return min(100.0, 100.0 * active_warps / max_warps)


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


def _run_ncu_profile(args, *, dump_dir=None):
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
    if args.golden or args.realization:
        if args.golden:
            cmd.extend(["--golden", args.golden])
        if args.realization:
            cmd.extend(["--realization", args.realization])
        if getattr(args, "kernel", None):
            cmd.extend(["--kernel", args.kernel])
    elif args.code is not None:
        cmd.extend(["--code", args.code])
    elif args.ir is not None:
        cmd.extend(["--ir", args.ir])
    else:
        # Positional model ID / .json path. Forward the trace shape flags so the
        # child profiles the same graph the parent benched.
        cmd.append(args.input)
        if args.layer is not None:
            cmd.extend(["--layer", str(args.layer)])
        cmd.extend(["--seq-len", str(args.seq_len)])
    if getattr(args, "gpu_arch", None) is not None:
        cmd.extend(["--gpu-arch", args.gpu_arch])
    # Forward the symbolic-dim specs so the child re-traces the SAME (masked-tile)
    # graph the parent benched. ``args.dynamic`` is the ``NAME@INPUT:AXIS`` CLI form
    # (a dynamic ``--golden`` sets it too, via ``resolve_golden_arg``); without this
    # the child re-compiles the static twin and ncu profiles the wrong kernel.
    for spec in getattr(args, "dynamic", None) or []:
        cmd.extend(["--dynamic", spec])
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


def _detect_stage(graph) -> str:
    """Identify the IR stage by scanning op type names. Returns one of
    ``torch | tensor | loop | tile | kernel | cuda`` — the highest-stage
    op present in the graph wins, since lowering produces a graph mixed
    only briefly during a pass and stable in the post-pass form."""
    stage_by_op: dict[str, str] = {
        "CudaOp": "cuda",
        "KernelOp": "kernel",
        "TileOp": "tile",
        "LoopOp": "loop",
    }
    order = ["torch", "tensor", "loop", "tile", "kernel", "cuda"]
    best = "torch"
    for node in graph.nodes.values():
        s = stage_by_op.get(type(node.op).__name__)
        if s and order.index(s) > order.index(best):
            best = s
    # Anything that's not Loop/Tile/Kernel/Cuda but is a frontend/tensor
    # op stays at "torch" — they get rewritten by the frontend passes.
    return best


def _passes_after_stage(stage: str) -> list[str]:
    """Pipeline tail to run after a graph has reached ``stage``."""
    from emmy.compiler.pipeline import (
        CUDA_PASSES,
        KERNEL_PASSES,
        LOOP_PASSES,
        TENSOR_PASSES,
        TILE_PASSES,
    )

    completed = {
        "torch": [],
        "tensor": TENSOR_PASSES,
        "loop": LOOP_PASSES,
        "tile": TILE_PASSES,
        "kernel": KERNEL_PASSES,
        "cuda": CUDA_PASSES,
    }[stage]
    return [p for p in CUDA_PASSES if p not in completed]


def _replay_stage_and_passes(graph, *, embedded_golden: bool) -> tuple[str, list[str]]:
    """The input label and pass list for an IR replay.

    A persisted golden Loop target stores stable algebra and is replayed through the full pipeline,
    as the compile that recorded it ran. A direct ``--ir`` input keeps its declared-stage tail semantics.
    """
    if embedded_golden:
        from emmy.compiler.pipeline import CUDA_PASSES  # noqa: PLC0415

        return "golden Loop", CUDA_PASSES
    stage = _detect_stage(graph)
    return stage, _passes_after_stage(stage)


#: The standard deviation (std) of each synthesized tensor: :func:`_random_source_values` draws unquantized weights
#: at 0.02, :func:`_random_input_values` draws activations at 1. The key is the suffix of the checkpoint key of the
#: NVFP4 per-tensor scale that calibrates that tensor.
_SYNTH_WEIGHT_STD = 0.02
_SYNTH_STD = {"_scale_2": _SYNTH_WEIGHT_STD, "input_scale": 1.0}
#: How many std from zero a synthesized tensor's calibrated amax (largest absolute value) sits. Real activations
#: carry outliers far past their std, and inside a program an activation outgrows the unit std of the inputs: a
#: gated MLP multiplies two projections. The headroom keeps the e4m3 block scales of such an activation well under
#: the e4m3 maximum of 448.
_SYNTH_AMAX_STDS = 64.0


def _random_source_values(rng, shape, dtype, *, name: str | None = None):
    """Return nontrivial deterministic values in a constant's declared storage dtype.

    Packed 4-bit pairs are uniform random bytes, so every code appears in both halves of a byte, as in a real
    checkpoint. A source whose ``name`` (the checkpoint key) ends in a scale leaf is positive, like every calibrated
    scale. NVFP4 scales follow how calibration sets them for a tensor of the synthesized std: a per-tensor scale
    (``*_scale_2``, ``input_scale``) is ``amax / (6 * 448)``, 6 and 448 being the largest e2m1 and e4m3 values.
    An 8-bit float scale (in checkpoints, only NVFP4 weight block scales are one) is ``block amax / (6 * per-tensor
    scale)``, so that each 16-value weight block has an amax of one to four std. A code times its block scale times
    the per-tensor scale then reproduces a weight of the 0.02 std the unquantized sources have, and an activation
    quantized with the drawn ``input_scale`` gets block scales well under 448. A block-FP8 weight's scale
    (``*_scale_inv``) is its block's amax over 448, an amax of one to four std, so that the weight it scales has
    the same std; larger scales overflow an FP16 trunk's projections. Any other scale gets a
    log-uniform draw from 1e-4 to 1e-1. A scale drawn from a normal distribution can be negative or near zero, and its
    reciprocal then overflows the 8-bit block scales computed from it."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import F4_VALUES, decode_f8  # noqa: PLC0415
    from emmy.compiler.dtype import get as get_dtype  # noqa: PLC0415

    canonical = get_dtype(dtype or "f32").name
    leaf = name.rsplit(".", 1)[-1] if name is not None else ""
    e4m3_max, e2m1_max = float(np.nanmax(decode_f8(np.arange(256, dtype=np.uint8), "f8e4m3"))), max(F4_VALUES)
    std = next((v for suffix, v in _SYNTH_STD.items() if leaf.endswith(suffix)), None)
    if canonical == "f4e2m1x2":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if canonical in {"f8e4m3", "f8e5m2"}:
        if "scale" in leaf:
            codes = np.arange(256, dtype=np.uint8)
            values = decode_f8(codes, canonical)
            lo, hi = e4m3_max / _SYNTH_AMAX_STDS, 4 * e4m3_max / _SYNTH_AMAX_STDS
            return rng.choice(codes[(values >= lo) & (values <= hi)], size=shape)
        bits = rng.integers(0, 256, shape, dtype=np.uint8)
        bits[~np.isfinite(decode_f8(bits, canonical))] = np.uint8(0)
        return bits
    if std is not None:
        return np.full(shape, _SYNTH_AMAX_STDS * std / (e2m1_max * e4m3_max), dtype=np.float32)
    if leaf.endswith("_scale_inv"):
        return (rng.uniform(1, 4, shape) * _SYNTH_WEIGHT_STD / e4m3_max).astype(np.float32)
    if "scale" in leaf:
        return np.exp(rng.uniform(np.log(1e-4), np.log(1e-1), shape)).astype(np.float32)
    return rng.standard_normal(shape, dtype=np.float32) * 0.02


def _random_input_values(rng, shape, dtype, *, name: str | None = None):
    """Return random runtime values, preserving bit carriers and finite E8M0 scales."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import get as get_dtype  # noqa: PLC0415
    from emmy.compiler.trace.huggingface import EXPERT_WEIGHT_INPUTS  # noqa: PLC0415
    from emmy.serving.lora import WEIGHT_INPUTS  # noqa: PLC0415

    canonical = get_dtype(dtype).name
    if name == "lora_mask":
        return rng.integers(0, 2, shape, dtype=np.uint8).astype(np.float32)
    if name in WEIGHT_INPUTS:
        return rng.standard_normal(shape, dtype=np.float32) * 0.02
    if canonical in _QUANTIZED_FORMATS:
        return _random_source_values(rng, shape, dtype, name=name)
    if canonical == "u8" and name is not None and "scale" in name:
        # MXFP4 boundary scales are E8M0 bytes with bias 127. Arbitrary u8 codes span
        # exponents that overflow a two-linear random reproducer before correctness can run.
        return rng.integers(120, 123, shape, dtype=np.uint8)
    if name in EXPERT_WEIGHT_INPUTS and canonical in {"f16", "bf16", "f32"}:
        # A routed expert's weights are program inputs. At unit scale its two FP16 projections overflow, so the
        # reference holds infinities and no correctness check or isolated re-bench can run.
        return rng.standard_normal(shape, dtype=np.float32) * 0.02
    return rng.standard_normal(shape, dtype=np.float32)


async def bench_lowered_vs_torch(
    frontend,
    lowered,
    backend,
    *,
    seed,
    do_bench,
    warmup,
    iters,
    bench_backends,
    capture_graphs=True,
    ref_out=None,
    ref_us_out=None,
    sym_env_out=None,
    strict_accuracy=False,
    return_reference=False,
):
    """Run + (optionally) benchmark a lowered graph against its torch reference on
    shared random inputs. The common bench primitive behind ``run --ir`` and
    ``tune --bench``.

    ``frontend`` is the pristine frontend-dialect snapshot (must be
    ``torch_ref``-runnable); pass ``None`` for a non-frontend / unsupported graph to
    bench emmy-only. One random source is drawn per distinct constant
    ``source_path`` and each side replays its own ``load_ops`` (so a weight lowering
    transposed + renamed stays the same underlying tensor on both sides). The lowered
    graph runs once for a non-fatal accuracy check vs the torch eager reference; then,
    when ``do_bench``, the selected backends are timed — interleaved when a torch ref
    exists, else emmy-only, both with the requested ``warmup``/``iters``.

    ``capture_graphs`` (default on — this function's callers are the per-kernel
    reproducer paths, where the torch side replays the frontend graph op-by-op and
    would otherwise be dispatch-bound) wraps every timed backend in a CUDA graph so
    the event windows measure pure GPU time. All-or-nothing: if any torch backend
    or the emmy launch loop fails to capture, the whole bench retries
    uncaptured (warning logged) so one table never mixes timing semantics. The
    accuracy check always runs uncaptured, before any capture.

    Returns ``(results, bench, torch_available, captured, accuracy_error)``: ``results``
    is the ``{backend: latency_us}`` dict (``None`` when ``do_bench`` is False), ``bench``
    the emmy ``BenchmarkResult`` (``None`` when ``do_bench`` is False),
    ``torch_available`` whether an eager/torch.compile reference was built, ``captured``
    whether the timings came from graph-captured (pure-GPU) windows, and
    ``accuracy_error`` the accuracy verdict or reference exception (``None`` = passed or no frontend;
    also logged here — returned so a worker-side run can ship it back to the parent, whose
    child logs are invisible). With ``return_reference``, appends the strict correctness
    proof and ``(input_data, eager_outputs_by_name)`` for same-input pinned replay. When
    ``ref_out`` is a list, that reference is also stored before repeated timing so a later
    watchdog cannot discard completed same-input evidence; reference-free Loop IR instead
    stores its greedy Emmy output. ``ref_us_out`` receives the initial Emmy execution timing.
    Does no printing / dumping — callers own that."""
    import numpy as np
    import torch

    from emmy.compiler.backend import torch_ref
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.loader.binder import bind_constants

    rng = np.random.default_rng(seed)

    # Symbolic reproducers bench at each dim's hint (``DEFAULT_SEQ_HINT`` for a
    # bare seq axis) — the same size the backend resolves a symbolic graph to
    # when no inputs are supplied — so the random inputs built here get concrete
    # hint-sized shapes. Atomic dims carry their own hint; composite exprs
    # (e.g. ``S * 2``) eval over the collected env.
    from emmy.compiler.dim import Dim

    sym_env = _collect_sym_env(([frontend] if frontend is not None else []) + [lowered])
    if sym_env_out is not None:
        sym_env_out.append(dict(sym_env))

    def _static(shape):
        return tuple((d.as_static() if d.is_static else int(d.expr.eval(sym_env))) if isinstance(d, Dim) else int(d) for d in shape)

    # One random source array per distinct constant ``source_path`` — shared by
    # emmy (lowered graph) and the torch ref (frontend graph). ``bind_constants``
    # replays each constant's ``load_ops`` on it, so a weight that lowering transposed +
    # renamed (linear: out×in → in×out) stays the same underlying tensor on both sides.
    sources: dict[str, np.ndarray] = {}
    for gph in ([frontend] if frontend is not None else []) + [lowered]:
        for node in gph.nodes.values():
            op = node.op
            if not (isinstance(op, ConstantOp) and op.value is None):
                continue
            if op.source_path and op.source_path not in sources:
                shp = _static(op.source_shape or node.output.shape)
                sources[op.source_path] = _random_source_values(rng, shp, op.source_dtype, name=op.source_path)
            # A merged (source_parts) constant draws one random source PER PART, keyed by the
            # part path — the same tensors the pre-merge frontend reference binds its separate
            # weights from, so emmy's concat and the torch ref stay numerically aligned.
            for path, shp in op.source_parts:
                if path not in sources:
                    sources[path] = _random_source_values(rng, _static(shp), op.source_dtype, name=path)

    input_data: dict[str, object] = {}
    input_tensors: dict[str, object] = {}
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, InputOp):
            arr = _random_input_values(rng, _static(node.output.shape), node.output.dtype, name=nid)
            # Keep the ndarray shape (no flatten) — a symbolic graph's launch
            # reads the runtime seq_len off the input array's shape.
            input_data[nid] = arr
            # Reference-free embedded Loop targets may intentionally expose a
            # price-probe boundary tensor whose logical broadcast shape is far
            # larger than device memory. The Emmy backend already owns the one
            # device materialization needed to execute that target; a second
            # Torch copy is useful only when a frontend reference exists.
            if frontend is not None:
                input_tensors[nid] = _to_cuda_tensor(arr, node.output.dtype)
        elif isinstance(node.op, ConstantOp) and node.op.value is not None:
            input_data[nid] = [float(node.op.value)]

    input_data.update(bind_constants(lowered, sources))
    if frontend is not None:
        for fid, arr in bind_constants(frontend, sources).items():
            input_tensors[fid] = _to_cuda_tensor(arr, frontend.nodes[fid].output.dtype)

    # Fallback for any value-None constant with no source_path (synthetic).
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, ConstantOp) and node.op.value is None and nid not in input_data:
            arr = rng.standard_normal(_static(node.output.shape), dtype=np.float32) * 0.02
            input_data[nid] = arr.flatten().tolist()

    quantized = _quantize_nodes(frontend, lowered) if strict_accuracy and frontend is not None else ()
    result, _ = backend.run(lowered, input_data=input_data, taps=quantized)
    result_outputs = _comparison_outputs(result.outputs, lowered)
    if frontend is None and ref_out is not None:
        ref_out.append((input_data, result_outputs))
    from emmy.compiler.pipeline.search.space import cold_cache  # noqa: PLC0415

    if ref_us_out is not None and not cold_cache():
        ref_us_out.append(result.time_ms * 1000)
    for nid, arr in result_outputs.items():
        finite = np.isfinite(arr).all()
        logger.info("Output %s: shape=%s finite=%s mean=%.4f", nid, arr.shape, bool(finite), float(arr.mean()))

    # Build the torch reference from the frontend snapshot, fed the same inputs.
    torch_fn = torch_inputs = accuracy_error = correctness = reference = None
    if frontend is not None:
        try:
            torch_fn, torch_inputs = torch_ref.build_callable(frontend, input_tensors)
            exact = None
            if strict_accuracy:
                correctness, eager_values, exact_values = _strict_eager_proof(
                    frontend, input_tensors, result_outputs, lowered, result.taps if quantized else {}
                )
                if correctness["status"] != "pass":
                    accuracy_error = f"strict eager correctness failed: {correctness.get('error', 'tolerance exceeded')}"
                if ref_out is not None or return_reference:
                    exact = exact_values()  # the pinned rows weigh their own strict failures against it
            else:
                with torch.no_grad(), correctness_oracle():
                    eager_values = _eager_outputs_by_name(result_outputs, torch_fn(*torch_inputs), lowered)
                accuracy_error = _check_accuracy(result_outputs, eager_values)
            reference = (input_data, eager_values) if exact is None else (input_data, eager_values, exact)
            if ref_out is not None:
                ref_out.append(reference)
            if accuracy_error is not None:
                qualifier = "fatal when strict correctness is requested" if strict_accuracy else "non-fatal (random-input reproducer)"
                logger.warning("%s — %s; benching anyway", accuracy_error, qualifier)
        except Exception as exc:  # noqa: BLE001 — torch ref is best-effort
            reference_kind = "strict eager correctness" if strict_accuracy else "torch reference"
            accuracy_error = f"{reference_kind} unavailable: {type(exc).__name__}: {exc}"
            logger.warning("%s — skipping vs-torch comparison", accuracy_error)
            torch_fn = None

    if strict_accuracy and frontend is None:
        accuracy_error = "strict eager correctness unavailable: frontend IR is not runnable"
    torch_available = torch_fn is not None
    if not do_bench:
        base = (None, None, torch_available, False, accuracy_error)
        return (*base, correctness, reference) if return_reference else base

    if torch_available:
        backends = _resolve_backends(bench_backends)

        async def timed(torch_fns):
            if capture_graphs:
                return await _bench_interleaved_captured(
                    torch_fn, torch_inputs, {}, backend, lowered, warmup, iters, torch_fns=torch_fns, input_data=input_data
                )
            results, bench = await _bench_interleaved(
                torch_fn,
                torch_inputs,
                {},
                backend,
                lowered,
                warmup,
                iters,
                torch_fns=torch_fns,
                capture_graphs=False,
                input_data=input_data,
            )
            return results, bench, False

        # A quantized twin's codes are what two correct evaluations land one apart on, so torch.compile is timed
        # unchecked, like Emmy's row; and its quantize can trip a device-side assert that poisons the context, so it
        # is built and timed in a window of its own after Emmy's, whose number then survives it.
        deferred = "tcompile" in backends and _quantized_twin(frontend)
        torch_fns = _build_torch_fns(torch_fn, torch_inputs, {}, warmup, backends=backends - ({"tcompile"} if deferred else set()))
        results, bench, captured = await timed(torch_fns)
        if deferred:
            try:
                torch_fns = _build_torch_fns(torch_fn, torch_inputs, {}, warmup, backends={"tcompile"}, check=False)
                torch.cuda.synchronize()  # a device-side assert in the twin surfaces here, as torch.compile's failure
                compiled = torch_fns["torch.compile"]
                results["torch.compile"] = (
                    compiled if isinstance(compiled, BackendFailure) else (await timed(torch_fns))[0]["torch.compile"]
                )
            except RuntimeError as exc:
                results["torch.compile"] = BackendFailure(f"{type(exc).__name__}: {exc}")
        base = (results, bench, True, captured, accuracy_error)
        return (*base, correctness, reference) if return_reference else base
    # Emmy-only: a capture failure falls back inside ``benchmark_program``
    # (warned + reported via ``bench.captured``) — nothing to de-mix.
    bench = await backend.benchmark_async(lowered, warmup=warmup, num_iters=iters, capture_graphs=capture_graphs, input_data=input_data)
    base = ({"Emmy": bench.time_ms * 1000}, bench, False, bench.captured, accuracy_error)
    return (*base, correctness, reference) if return_reference else base


async def bench_full_model_real(module, args_t, kwargs, lowered, backend, *, warmup, iters, bench_backends, input_data=None):
    """End-to-end full-model bench against the **real torch module** — eager /
    ``torch.compile`` / Emmy — using the all-or-nothing CUDA-graph-captured
    interleaved bench (real modules occasionally resist capture; those fall back
    to uncaptured wall timing, flagged in ``captured``). The module + its
    trace-time inputs come from ``load_or_trace``'s bundle; for a symbolic
    graph the torch closures run on hint-tiled inputs (``_hint_sized_inputs``)
    so both sides bench the hint shape. ``input_data`` (the module's own inputs bound to the
    emmy program, :func:`_bind_inputs`) times emmy on the values torch reads; a symbolic graph
    keeps the synthetic fill, since its torch side runs tiled hint-sized copies instead.
    Returns ``(results, bench, captured)``."""
    import torch

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args_t)
    cuda_kwargs = _to_cuda_kwargs(kwargs)
    cuda_args, cuda_kwargs, sym_env = _hint_sized_inputs(lowered, cuda_args, cuda_kwargs)
    input_data = None if sym_env else input_data
    backends = _resolve_backends(bench_backends)
    torch_fns = _build_torch_fns(cuda_module, cuda_args, cuda_kwargs, warmup, backends=backends)
    if not torch_fns:
        # Emmy-only bench (``--bench-backends emmy``): any prior torch activity in this
        # process (the accuracy probe's eager forward, the warmups above) left
        # cuBLAS-installed ``cudaAccessPolicyWindow`` carveouts pinned in L2, which steal
        # L2 bandwidth from cp.async-staged kernels (~4x on the article tile — 1660 µs
        # with the window live, 330 µs after the reset). With torch backends in the table
        # the state must stay alive and the comparison absorbs it intrinsically; emmy-only,
        # reset it. (Caller-held module refs keep allocator blocks alive — the persisting-L2
        # window is the dominant term and resets regardless.)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _reset_persisting_l2_cache()
    return await _bench_interleaved_captured(
        cuda_module, cuda_args, cuda_kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns, input_data=input_data
    )


_NO_GREEDY_REF = "pinned embedded-Loop verification requires same-input greedy outputs, but none were returned"
UNVERIFIED_ROW = "benched with no reference outputs"


def pinned_reference_refusal(*, ab_ref, torch_twin: bool, greedy_fail: str | None) -> str | None:
    """Why pinned rows cannot be benched AT ALL, or ``None`` to bench them without an output check.

    A greedy that never returned run outputs leaves nothing for a pinned row's outputs to be
    compared against. Whether that is fatal turns on ONE question: does any other reference exist?
    An exact Loop target has no Torch twin, so the greedy Loop execution is the only reference
    obtainable and a row measured without it is unfalsifiable — refuse. Where a Torch twin does
    exist, refusing costs more than it protects: a greedy too slow to time is exactly the case a
    pinned alternative exists to escape, so refusing the alternative leaves the target with no
    measurement and no way to find one.

    The predicate is the twin's existence, NOT ``same_input_greedy``. The two differ by a
    ``strict_correctness`` conjunct, and keying on the latter would relax the case this is meant to
    keep: a non-strict embedded-Loop target with no twin would bench unverified when in fact no
    reference exists for it at all.

    A row benched this way carries the :data:`UNVERIFIED_ROW` flag, and that flag is a hard barrier
    downstream, not a note. It keeps the row out of the golden (``--record`` refuses it), out of the
    tune DB's ``perf`` rows, because unverified data must never become
    deployable evidence — the whole value of the reference comparison is that it is the only thing
    standing between a tile that miscompiles silently and a recorded row that deploys it.
    """
    if ab_ref is not None:
        return None
    if not torch_twin:
        return f"{_NO_GREEDY_REF}: {greedy_fail or 'the greedy worker returned no run outputs'}"
    return None


def _pinned_samples_for_ir(args, embedded):
    """Automatic verified pins plus explicit ``--ab`` pins for an embedded golden target."""
    pinned = list(getattr(args, "golden_configs", None) or [])
    if embedded is not None and (specs := getattr(args, "ab", None)):
        from emmy.commands.compile import selected_decisions  # noqa: PLC0415

        pinned.extend(_ab_samples(specs, dynamic=getattr(args, "dynamic", None), route=selected_decisions(args)))
    return pinned


def _strict_benchmark_errors(
    args,
    results,
    bench,
    captured,
    correctness,
    pinned_rows,
    *,
    frontend_runnable=True,
    same_input_reference=False,
) -> list[str]:
    """Return every strict verification failure in one ordinary run result."""

    reference = "same-input-greedy" if not frontend_runnable and same_input_reference else "eager"

    def valid_proof(proof) -> bool:
        if not isinstance(proof, dict):
            return False
        required = {"status": "pass", "reference": reference, "rtol": 1e-3, "atol": 1e-3}
        if any(proof.get(key) != value for key, value in required.items()):
            return False
        return all(
            not isinstance(proof.get(metric), bool) and isinstance(proof.get(metric), int | float) and proof[metric] >= 0
            for metric in ("max_abs_error", "mean_abs_error", "max_rel_error")
        )

    errors = []
    display_names = {"eager": "Eager PyTorch", "tcompile": "torch.compile", "emmy": "Emmy"}
    for backend in _resolve_backends(args.bench_backends):
        if backend != "emmy" and not frontend_runnable:
            continue  # an embedded Loop target has no Torch twin; its reference is the same-input greedy replay
        name = display_names[backend]
        latency = (results or {}).get(name)
        if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
            errors.append(f"requested backend {name} has no positive latency")
    if captured is not True:
        errors.append("backend comparison did not use CUDA graph capture")
    if bench is None or not bool(getattr(bench, "captured", False)):
        errors.append("deployed Emmy timing was not captured")
    if not frontend_runnable and not same_input_reference:
        errors.append("same-input greedy reference is unavailable")
    if not valid_proof(correctness):
        errors.append(f"deployed Emmy row lacks strict {reference} correctness")

    pinned_rows = list(pinned_rows or [])
    ab_rows = [row for row in pinned_rows if getattr(row.sample, "shape", None) is None]
    automatic_rows = [row for row in pinned_rows if getattr(row.sample, "shape", None) is not None]
    expected_automatic_rows = len(getattr(args, "golden_configs", []) or [])
    if len(automatic_rows) != expected_automatic_rows:
        errors.append(f"expected {expected_automatic_rows} exact working-golden row(s), got {len(automatic_rows)}")
    expected_ab_rows = len(args.ab or [])
    if len(ab_rows) != expected_ab_rows:
        errors.append(f"expected {expected_ab_rows} exact --ab row(s), got {len(ab_rows)}")
    for row in pinned_rows:
        label = row.sample.name
        if row.status != "ok" or row.flags:
            errors.append(f"{label} failed exact-pin integrity: status={row.status}, flags={list(row.flags)}")
            continue
        kernels = list(_launch_order_cuda_nodes(row.graph)) if row.graph is not None else []
        if not kernels:
            errors.append(f"{label} has no generated kernel inventory")
        if row.bench is None or not bool(getattr(row.bench, "captured", False)):
            errors.append(f"{label} timing was not captured")
        else:
            num_launches = getattr(row.bench, "num_launches", 0) or len(row.bench.per_launch or [])
            if num_launches <= 0:
                errors.append(f"{label} has no measured launches")
            e2e_ms = getattr(row.bench, "e2e_min_ms", None)
            if num_launches > 1 and e2e_ms is None:
                errors.append(f"{label} multi-launch timing is not whole-program end-to-end")
            total_ms = e2e_ms if e2e_ms is not None else getattr(row.bench, "min_ms", None)
            if total_ms is None:
                total_ms = getattr(row.bench, "time_ms", None)
            if isinstance(total_ms, bool) or not isinstance(total_ms, int | float) or total_ms <= 0:
                errors.append(f"{label} has no positive whole-program timing")
        if not valid_proof(row.correctness):
            errors.append(f"{label} lacks strict {reference} correctness")

    return errors


def _handle_run_ir(args, CudaBackend, CompilerDump):
    """Run path: load JSON IR (any stage), finish lowering, execute, bench."""
    import json

    from emmy.compiler.backend import torch_ref
    from emmy.compiler.graph import Graph
    from emmy.compiler.pipeline import Pipeline

    strict_correctness = bool(getattr(args, "strict_correctness", False))
    embedded = getattr(args, "_golden_graph", None)
    if (paged := getattr(args, "paged", None)) and embedded is not None:
        # A paged input, one page per row of its leading axis, as a serving boot below its device budget declares it
        # (``gen_runner._compile_split``): every compile of this program — the greedy one and each tune or --ab row —
        # offers only the schedules a paged operand can take. The runtime allocates the pages and fills them from the
        # input, so the bench times the paged schedules on resident pages.
        embedded.hints.set("cuda.paged_buffers", tuple((name, 0, 1) for name in paged))
    path = Path(args.ir) if embedded is None else None
    if embedded is None:
        with open(path) as f:
            data = json.load(f)
        graph = Graph.from_dict(data)
    else:
        graph = embedded.copy()
    if getattr(args, "dynamic", None):
        logger.error("--dynamic is incompatible with --ir (the trace is already complete)")
        sys.exit(2)

    stage, tail = _replay_stage_and_passes(graph, embedded_golden=embedded is not None)
    logger.info("Loaded %s IR; running tail passes: %s", stage, tail or "(none)")

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Snapshot the pre-lowering frontend graph so we can build a torch
    # reference (eager + torch.compile) and compare accuracy/latency vs torch —
    # the same table the --code path produces for a debug Graph IR input.
    # Non-frontend IR (loop/tile/…) has no torch twin → emmy-only bench, unless it is a
    # stored golden kernel whose embedded program holds the PyTorch slice it computes.
    reference = graph if torch_ref.is_runnable(graph) else getattr(args, "_golden_reference", None)
    frontend = reference.copy() if reference is not None and torch_ref.is_runnable(reference) else None
    same_input_greedy = strict_correctness and embedded is not None and frontend is None

    backend = CudaBackend(debug=args.debug or None, dump=dump, tune_db="auto")
    db = None
    if backend.tune_db is not None:
        from emmy.compiler.pipeline.search.db import SearchDB

        db = SearchDB.for_compile(backend.tune_db)
        logger.info("Using tuning DB: %s", backend.tune_db)
    from emmy.compiler.pipeline.search.inventory import KernelInventory  # noqa: PLC0415

    # Every kernel-set decision the greedy compile takes, as the splice watcher reports it: the tile
    # kernel the fork was offered on, the arm, and the pieces it minted.
    taken: list[tuple[object, dict, list]] = []
    watcher = KernelInventory(on_routing=lambda parent, arm, pieces, _ids: taken.append((parent, arm, pieces)))
    if tail:
        # Finish the tail lowering — the greedy compile — with the selected golden's records as its
        # golden evidence, their shared input regime published (by the caller) and, under
        # ``--pin-route``, the kernel-set decisions the named realization records pinned, as
        # ``compile`` does. The same compile captures its kernel-set decisions: the bench-evidence
        # recording (``_record_bench_evidence``) stores them beside the perf rows, and
        # ``--record-greedy`` writes them into the golden.
        from emmy.commands.compile import selected_decisions  # noqa: PLC0415
        from emmy.compiler.pipeline.search.golden import evidence_scope  # noqa: PLC0415

        pipeline = Pipeline.build(tail).with_strategies(watcher)
        with evidence_scope(getattr(args, "_golden_scope", None)), pinned_knobs(selected_decisions(args)):
            graph = pipeline.run(graph, db=db, dump=dump)
    # The golden target walk records the weight layouts this compile bound (``_run_golden_targets``).
    args._compiled = graph

    if not args.bench:
        # No bench: one in-process run + non-fatal accuracy vs the torch reference
        # (random boundary inputs — informational, logged by ``bench_lowered_vs_torch``).
        try:
            asyncio.run(
                bench_lowered_vs_torch(
                    frontend,
                    graph,
                    backend,
                    seed=args.seed,
                    do_bench=False,
                    warmup=args.warmup,
                    iters=args.iters,
                    bench_backends=args.bench_backends,
                )
            )
        except RuntimeError as exc:
            # Per-launch watchdog fired — the CUDA context is dirty; bypass the driver's teardown.
            sys.stderr.write(f"run failed: {exc}\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return

    # --bench: same uniform worker path as ``handle_run`` — the greedy comparison
    # (``bench_lowered_vs_torch`` rebuilt in-child from the frontend snapshot) and every
    # ``--ab`` row are jobs on one persistent SIGKILL-able worker; a hung kernel dies with
    # the child and the row records bench_fail.
    if args.ab and not tail:
        logger.warning("--ab ignored: %s IR is fully lowered (no forks left to pin)", stage)

    async def _bench_session():
        greedy_fail = results = bench = accuracy_error = correctness = ab_ref = reference_error = ab_benches = greedy_iso = None
        greedy_reference_us = None
        same_input_reference = False
        stats_sym_env = _collect_sym_env(([frontend] if frontend is not None else []) + [graph])
        torch_available = captured = False
        pinned = _pinned_samples_for_ir(args, embedded)
        # Golden write-back and default bench-to-DB evidence both need isolated per-kernel timings,
        # including a fresh file walk with no measured row to compare as a pinned schedule.
        measure_greedy = bool(getattr(args, "record_greedy", False) or (embedded is not None and _records_evidence(args)))
        try:
            try:
                resp = await backend.benchmark_compare_async(
                    graph,
                    torch_spec=("frontend_graph", frontend),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_compare_wall_s(
                        graph,
                        backend,
                        base_s=_IR_COMPARE_BASE_S,
                        twin=frontend if "tcompile" in _resolve_backends(args.bench_backends) else None,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    want_ref=bool(tail and (same_input_greedy or pinned or measure_greedy or args.ab or getattr(args, "tune", None))),
                    strict_accuracy=strict_correctness and not same_input_greedy,
                )
            except RuntimeError as exc:
                greedy_fail = f"greedy run/bench failed: {exc}"
                _record_greedy_failure(args, backend, graph, exc)
            else:
                results, bench, captured = resp["results"], resp["result"], resp["captured"]
                torch_available, accuracy_error = resp["torch_available"], resp["accuracy_error"]
                ab_ref = resp["run_io"]
                greedy_reference_us = resp.get("reference_run_us")
                stats_sym_env = resp.get("sym_env")
                correctness = resp.get("correctness")
                same_input_reference = same_input_greedy and ab_ref is not None
                if same_input_reference:
                    correctness = _strict_correctness_proof(
                        ab_ref[1],
                        ab_ref[1],
                        reference="same-input-greedy",
                    )
                if resp.get("greedy_error"):
                    greedy_fail = f"greedy timing failed after reference execution: {resp['greedy_error']}"
                    _record_greedy_failure(args, backend, graph, resp["greedy_error"])
            ref_key = uuid.uuid4().hex if ab_ref is not None else None
            if not pinned and tail and greedy_fail is None:
                if measure_greedy:
                    if same_input_greedy or accuracy_error is None:
                        greedy_iso = await _bench_greedy_isolated(
                            backend, graph, warmup=args.warmup, iters=args.iters, ref=ab_ref, ref_key=ref_key
                        )
                elif embedded is not None:
                    _note_below_standard(args)
            if pinned and tail:
                reference_error = pinned_reference_refusal(ab_ref=ab_ref, torch_twin=frontend is not None, greedy_fail=greedy_fail)
                unverified = None
                if reference_error is None and ab_ref is None:
                    unverified = f"{UNVERIFIED_ROW}: {greedy_fail or 'the greedy worker returned no run outputs'}"
                if reference_error is None and (same_input_greedy or not strict_correctness or accuracy_error is None):
                    to_bench = pinned
                    if greedy_fail:
                        # An automatic golden-config row with no knobs pins nothing beyond the input
                        # regime the greedy compile already used, so re-compiling it reaches the exact
                        # same election and re-fails the same way (the DeepSeek-V4 double-hang cost).
                        # An --ab row or a row with real schedule knobs is a genuinely different
                        # config and still benches.
                        same_election = [s for s in pinned if getattr(s, "shape", None) is not None and not s.knobs]
                        if same_election:
                            to_bench = [s for s in pinned if s not in same_election]
                            logger.warning(
                                "%s — pinned re-bench of %s skipped: it has no knobs to pin it away from the greedy pick that just failed",
                                greedy_fail,
                                ", ".join(s.name for s in same_election),
                            )
                        if to_bench:
                            logger.error("%s — untimed greedy is ineligible; pinned rows still bench", greedy_fail)
                    else:
                        greedy_iso = await _bench_greedy_isolated(
                            backend, graph, warmup=args.warmup, iters=args.iters, ref=ab_ref, ref_key=ref_key
                        )
                    ab_benches = await _bench_golden_variants(
                        backend,
                        embedded,
                        to_bench,
                        warmup=args.warmup,
                        iters=args.iters,
                        ref=ab_ref,
                        ref_key=ref_key,
                        strict_correctness=strict_correctness and ab_ref is not None,
                        strict_reference="same-input-greedy" if same_input_greedy else "eager",
                        unverified=unverified,
                        ref_knobs=_cuda_knob_dicts(graph),
                        computed_from=_computed_from_formats(frontend) if frontend is not None else None,
                    )
            elif not pinned and args.ab and tail:
                if greedy_fail:
                    logger.error("%s — greedy row marked bench_fail; --ab rows still bench in the worker", greedy_fail)
                if greedy_iso is None:
                    greedy_iso = await _bench_greedy_isolated(
                        backend, graph, warmup=args.warmup, iters=args.iters, ref=ab_ref, ref_key=ref_key
                    )
                ab_benches = await _bench_ab_variants_ir(
                    backend, path, tail, args.ab, warmup=args.warmup, iters=args.iters, db=db, ref=ab_ref, ref_key=ref_key
                )
            if getattr(args, "tune", None) and tail and embedded is not None and greedy_fail is None:
                from emmy.commands.compile import selected_decisions  # noqa: PLC0415

                if greedy_iso is None:
                    greedy_iso = await _bench_greedy_isolated(
                        backend, graph, warmup=args.warmup, iters=args.iters, ref=ab_ref, ref_key=ref_key
                    )
                tuned = await _tune(backend, args, embedded, graph, ab_ref, ref_key, route=selected_decisions(args))
                ab_benches = (ab_benches or []) + tuned
        finally:
            await backend.aclose_async_worker()
        return (
            greedy_fail,
            results,
            bench,
            torch_available,
            captured,
            accuracy_error,
            reference_error,
            ab_benches,
            greedy_iso,
            greedy_reference_us,
            stats_sym_env,
            correctness,
            same_input_reference,
        )

    from emmy.compiler.pipeline.search.golden import evidence_scope  # noqa: PLC0415

    # Pinned rows compile under the golden scope the greedy row did — the file named, not the card's whole corpus.
    with evidence_scope(getattr(args, "_golden_scope", None)):
        session = asyncio.run(_bench_session())
    (
        greedy_fail,
        results,
        bench,
        torch_available,
        captured,
        accuracy_error,
        reference_error,
        ab_benches,
        greedy_iso,
        greedy_reference_us,
        stats_sym_env,
        correctness,
        same_input_reference,
    ) = session

    if reference_error is not None:
        logger.error(reference_error)
        sys.exit(1)

    if accuracy_error is not None:
        # Random boundary-input reproducers keep their historical informational check, while
        # strict verification aborts after preserving JSON.
        if strict_correctness:
            logger.error(accuracy_error)
        else:
            logger.warning("%s — non-fatal (random-input reproducer); benched anyway", accuracy_error)
    if dump and bench is not None:
        dump.dump_benchmark(bench)
    if torch_available:
        note = None if captured else "(graph-capture fallback: timings include host launch overhead)"
        _print_table(results, note=note)
    elif bench is not None:
        # Non-frontend IR (loop/tile/…): no torch twin → emmy-only.
        from emmy.commands.table import Col, render_table  # noqa: PLC0415

        print()
        for line in render_table([Col("Backend"), Col("Latency (us)", "r")], [["Emmy", f"{bench.time_ms * 1000:.0f}"]], rule=True):
            print(line)
    _print_kernel_stats(
        graph,
        bench,
        golden_benches=ab_benches,
        greedy_fail=greedy_fail,
        greedy_iso=greedy_iso,
        sym_env=stats_sym_env,
    )
    strict_errors = (
        _strict_benchmark_errors(
            args,
            results,
            bench,
            captured,
            correctness,
            ab_benches,
            frontend_runnable=frontend is not None,
            same_input_reference=same_input_reference,
        )
        if strict_correctness
        else None
    )
    if getattr(args, "json", None):
        _write_ab_json(
            args,
            results or {},
            graph,
            bench,
            ab_benches,
            greedy_fail=greedy_fail,
            greedy_iso=greedy_iso,
            greedy_reference_us=greedy_reference_us,
            correctness=correctness,
            strict_errors=strict_errors,
            accuracy_error=accuracy_error,
            checked=torch_available,
            tc_checked=not _quantized_twin(frontend),
        )
    if getattr(args, "record", False) or getattr(args, "record_greedy", False):
        # Missing inventory rows are not rejected answers: a fresh recording may create them.
        refusal = "; ".join(strict_errors or []) or record_refusal(
            _cuda_knob_dicts(graph),
            accuracy_error if strict_correctness else None,
            _cuda_kernel_names(graph),
            benches=[greedy_iso, *(ab_benches or [])],
        )
        if refusal is not None:
            logger.error("not recording %s — %s", args.realization, refusal)
            sys.exit(1)
    if getattr(args, "record", False):
        _record_golden_latency(args, results or {}, ab_benches)
    if getattr(args, "record_greedy", False):
        _record_greedy_pick(args, graph, bench, greedy_iso, taken, results or {})
    for error in strict_errors or []:
        logger.error("strict: %s", error)
    if embedded is not None:
        # An embedded golden lowers in-process, knobs and all; only the ``--ir`` JSON path, whose
        # serialization drops them, stays unrecorded.
        _record_bench_evidence(args, ab_benches, greedy_iso, decisions=taken)
    if args.profile and greedy_fail is None:
        _run_ncu_profile(args, dump_dir=dump.dir if dump else None)
    if (
        (strict_correctness and accuracy_error is not None)
        or bool(strict_errors)
        or greedy_fail is not None
        or (greedy_iso is not None and greedy_iso.status != "ok")
        or any(gb.status != "ok" for gb in ab_benches or [])
    ):
        sys.exit(1)  # every row is reported above; any failed row (greedy or --ab) exits non-zero


async def _bench_ab_variants_ir(backend, ir_path, tail, specs, *, warmup, iters, db=None, ref=None, ref_key=None):
    """The ``--ab`` counterpart of :func:`_bench_golden_variants` for the ``--ir``
    path: each config reloads the IR file fresh (the tail lowering mutates the graph
    in place) and re-lowers it with the knobs pinned, so the pin collapses every
    remaining fork. Same row semantics as the golden path: a config that fails to
    compile / bench is kept as a ``bench_fail`` row, a pin the re-lowering didn't
    realize fails its row loudly BEFORE benching (``pin_unmatched`` — benching the
    fallback would measure the planner's own pick under the pin's name), and every
    bench is one job on the backend's persistent SIGKILL-able worker, so a hung row
    dies with the child and the remaining rows continue on a fresh one. Serialized
    ops drop ``knobs``, so only tail-lowered kernels carry realized values — a pinned
    family the tail never re-decides has no stamp to check and is skipped
    (ungateable), while a family the tail did re-decide still verifies."""
    import json as _json  # noqa: PLC0415

    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

    out = []
    for sample in _ab_samples(specs):
        replay_knobs = _sample_replay_knobs(sample)
        try:
            with pinned_knobs(replay_knobs):
                g = Graph.from_dict(_json.loads(Path(ir_path).read_text()))
                if tail:
                    g = Pipeline.build(tail).run(g, db=db)
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own table
            logger.warning("[ab] %s: compile of the pinned config failed (%s) — row kept as bench_fail", sample.name, exc)
            out.append(_GoldenBench(sample, None, None, [f"compile failed: {exc}"], "bench_fail"))
            continue
        flag = unreproducible_pin_flag(
            replay_knobs,
            _cuda_knob_dicts(g),
            placement_knobs=_placement_knob_dicts(g),
            kernel_names=_cuda_kernel_names(g),
            applied_place_pins=_applied_place_pins(g),
        )
        if flag:
            logger.error("[ab] %s: %s — the pinned config did not realize; fix the pin spelling (row kept unbenched)", sample.name, flag)
            out.append(_GoldenBench(sample, g, None, [f"{flag} — row NOT benched"], "pin_unmatched"))
            continue
        try:
            with pinned_knobs(replay_knobs):
                inputs = ref[0] if ref is not None else None
                g_bench, _ = await backend.bench_pinned_async(g, run_inputs=inputs, run_inputs_key=ref_key, warmup=warmup, num_iters=iters)
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own table
            st = _failed_bench_status(exc)
            logger.warning("[ab] %s: bench of the pinned config failed (%s) — row kept as %s", sample.name, exc, st)
            out.append(_GoldenBench(sample, g, None, [f"{st}: {exc}"], st))
            continue
        out.append(_GoldenBench(sample, g, g_bench, []))
    return out


def _bind_inputs(compiled, module, example_args, example_kwargs, checkpoint=None):
    """Match graph inputs and constants to tensors from ``module`` / call args.

    Activations come from the call's positional/keyword tensors. Constants
    come from ``module.named_parameters()`` / ``named_buffers()`` keyed
    by each ``ConstantOp.source_path`` recorded at trace time. Each
    constant's ``load_ops`` chain is replayed via the NumPy backend
    (see ``compiler.loader.binder``), so any compile-time-folded
    transpose / reshape is honored uniformly.

    ``checkpoint`` (the model id / path, when the caller has one) covers the
    constants a live module cannot supply: an fp8 checkpoint's weights bind as
    raw bits + scale sources (or a folded ``source_graph`` record over them)
    that exist only in the safetensors shards, so any still-unbound constant is
    re-tried through the safetensors loader before the hard error below.
    """
    import numpy as np
    import torch

    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.loader.binder import bind_constants

    flat_inputs: list[torch.Tensor] = []
    for v in example_args:
        flat_inputs.extend(_flatten_tensors(v))
    for v in example_kwargs.values():
        flat_inputs.extend(_flatten_tensors(v))

    input_ids = list(compiled.inputs)
    if len(input_ids) != len(flat_inputs):
        # Raise, don't sys.exit: this also runs inside the bench worker child, where an
        # exit(1) surfaces to the parent as an opaque ``SystemExit(1)`` with the real
        # cause stranded in the child's log stream.
        raise RuntimeError(f"Input arity mismatch: graph has {len(input_ids)} inputs, code provided {len(flat_inputs)}")

    input_data: dict[str, np.ndarray] = {}
    torch_f8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    for nid, tensor in zip(input_ids, flat_inputs, strict=True):
        dtype = compiled.nodes[nid].output.dtype
        np_dtype = dtype.np
        tensor = tensor.detach().cpu()
        # An fp8 activation/weight INPUT binds the raw bit pattern on its uint8 carrier —
        # numpy has no fp8 scalar type, and a value cast would decode the bits (the same
        # rule the constant side and the serving expert feed already follow).
        if tensor.dtype in torch_f8:
            tensor = tensor.view(torch.uint8)
        if dtype.name == "bf16":
            from emmy.compiler.dtype import encode_bf16  # noqa: PLC0415

            if tensor.dtype == torch.bfloat16:
                input_data[nid] = tensor.contiguous().view(torch.uint16).numpy()
            else:
                input_data[nid] = encode_bf16(tensor.numpy())
        else:
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            input_data[nid] = tensor.numpy().astype(np_dtype, copy=False)

    # ``remove_duplicate=False`` so tied weights (e.g. a model whose lm_head
    # shares the embedding matrix) are surfaced under *every* name, including
    # the ``source_path`` the tracer recorded — otherwise the alias the trace
    # picked may be the one torch dedups away and the constant won't bind.
    sources: dict[str, np.ndarray] = {}
    for path, tensor in module.named_parameters(remove_duplicate=False):
        tensor = tensor.detach().cpu()
        sources[path] = (
            tensor.contiguous().view(torch.uint16).numpy()
            if tensor.dtype == torch.bfloat16
            else tensor.numpy().astype(np.float32, copy=False)
        )
    for path, tensor in module.named_buffers(remove_duplicate=False):
        tensor = tensor.detach().cpu()
        sources[path] = (
            tensor.contiguous().view(torch.uint16).numpy()
            if tensor.dtype == torch.bfloat16
            else tensor.numpy().astype(np.float32, copy=False)
        )

    input_data.update(bind_constants(compiled, sources))

    if checkpoint is not None and any(nid not in input_data for nid, _op in compiled.loadable_constants()):
        from emmy.compiler.trace.huggingface import quantized_checkpoint_dir  # noqa: PLC0415

        quant_dir = quantized_checkpoint_dir(checkpoint)
        if quant_dir is not None:
            from emmy.compiler.loader.safetensors import load_constants_from_safetensors  # noqa: PLC0415

            for nid, arr in load_constants_from_safetensors(compiled, str(quant_dir)).items():
                input_data.setdefault(nid, arr)

    for nid, node in compiled.nodes.items():
        if not isinstance(node.op, ConstantOp) or nid in input_data:
            continue
        if node.op.value is not None:
            continue  # backend materializes scalars from node.op.value
        raise RuntimeError(f"Could not bind constant {nid} (source_path={node.op.source_path!r})")
    return input_data


def _flatten_tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten_tensors(v))
        return out
    return []


def _collect_sym_env(graphs) -> dict[str, int]:
    """Every symbolic dim var appearing in ``graphs`` bound to the size the backend resolves it to
    when benching without supplied inputs (:func:`symbolic_bindings`, the tune DB's rule too)."""
    from emmy.compiler.wire import symbolic_bindings

    return symbolic_bindings(node.output for gph in graphs for node in gph.nodes.values())


def _tile_to(tensor, axis: int, size: int):
    """Resize ``tensor`` along ``axis`` to ``size`` by repeating its values
    (ceil) and slicing — keeps every element drawn from the original (token
    ids stay in-vocab, masks keep their value range), unlike fresh randoms."""
    old = tensor.shape[axis]
    if old == size:
        return tensor
    reps = [1] * tensor.dim()
    reps[axis] = -(-size // old)
    return tensor.repeat(*reps).narrow(axis, 0, size)


def _map_tensors(value, fn):
    """Rebuild ``value``'s nested list/tuple structure with ``fn`` applied to
    each tensor leaf, in :func:`_flatten_tensors` order."""
    import torch

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_map_tensors(v, fn) for v in value)
    return value


def _hint_sized_inputs(lowered, example_args, example_kwargs):
    """Tile a symbolic graph's example inputs out to its ``Dim`` hints.

    The emmy side of the full-model bench runs a symbolic graph at each
    dim's hint (``backend.benchmark`` builds hint-sized synthetic inputs when
    none are supplied), while the torch closures close over the trace-time
    example tensors — different shapes, so the table would compare e.g. a
    seq-512 emmy program against seq-32 eager. Grow every symbolic input
    axis to its hint (positional ``graph.inputs`` ↔ flattened example tensors
    pairing, same as :func:`_bind_inputs`) so both sides run the hint shape.
    Values are tiled repeats of the trace inputs — valid for latency, which is
    all this table compares (accuracy runs on the original trace inputs).

    Returns ``(args, kwargs, sym_env)`` — the inputs unchanged and ``{}`` for
    a static graph.
    """
    sym_env = _collect_sym_env([lowered])
    if not sym_env:
        return example_args, example_kwargs, {}
    from emmy.compiler.dim import Dim

    flat: list = []
    for v in example_args:
        flat.extend(_flatten_tensors(v))
    for v in example_kwargs.values():
        flat.extend(_flatten_tensors(v))
    input_ids = list(lowered.inputs)
    if len(input_ids) != len(flat):
        logger.warning("Input arity mismatch (graph %d vs example %d) — benching torch on trace-sized inputs", len(input_ids), len(flat))
        return example_args, example_kwargs, sym_env
    resized = []
    for nid, tensor in zip(input_ids, flat, strict=True):
        for axis, d in enumerate(lowered.buffer(nid).shape):
            if isinstance(d, Dim) and not d.is_static:
                tensor = _tile_to(tensor, axis, int(d.expr.eval(sym_env)))
        resized.append(tensor)
    it = iter(resized)
    args = tuple(_map_tensors(v, lambda _: next(it)) for v in example_args)
    kwargs = {k: _map_tensors(v, lambda _: next(it)) for k, v in example_kwargs.items()}
    return args, kwargs, sym_env


def _symbolic_bench_note(sym_env: dict[str, int]) -> str | None:
    """The full-model table's shape label for a symbolic graph (``None`` for
    static): both sides bench at the hint, and the reader should know the
    number is hint-shaped, not trace-shaped."""
    if not sym_env:
        return None
    dims = ", ".join(f"{name}={size}" for name, size in sorted(sym_env.items()))
    return f"benched at {dims} (symbolic hint; torch inputs tiled to match)"


@contextlib.contextmanager
def correctness_oracle():
    """Torch as the CORRECTNESS reference: its GEMMs without the reduced-precision reductions it enables by
    default. Those trade accuracy for speed inside the reference itself — at K=15360 the default FP16 GEMM
    leaves 9.6% of its own elements outside ``rtol=atol=1e-3`` of an FP64 product, 0.1% with them off — so a
    kernel nearer the truth than eager failed ``--strict`` for eager's error. The TIMED eager forward keeps
    torch's defaults: that is the library a user runs."""
    import torch

    matmul = torch.backends.cuda.matmul
    saved = (matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction)
    matmul.allow_fp16_reduced_precision_reduction = matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction = saved


def _eager_output(module, args, kwargs, *, device: str = "cuda"):
    """Eager reference forward on ``device``. Returns the module's output — a Tensor, or a
    TUPLE of Tensors for a multi-output module (``_check_accuracy`` compares
    each graph output buffer against the tuple positionally)."""
    import torch

    if device == "cuda":
        module, args, kwargs = (
            module.to("cuda"),
            tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args),
            _to_cuda_kwargs(kwargs),
        )
    with torch.no_grad():
        out = module(*args, **kwargs)
    if isinstance(out, tuple):
        tensors = tuple(t for t in out if isinstance(t, torch.Tensor))
        return tensors if len(tensors) > 1 else tensors[0] if tensors else out[0]
    return out


def _to_cuda_tensor(arr, dtype):
    """numpy array → CUDA torch tensor in the node's dtype (default fp32).

    A BF16 value bound from its bits (the ``uint16`` carrier a folded table constant evaluates to) is reinterpreted,
    not converted: converting would read each bit pattern as a number."""
    import numpy as np
    import torch

    from emmy.compiler.backend.torch_ref import torch_dtype

    target = torch_dtype(dtype) or torch.float32
    if target is torch.bfloat16 and arr.dtype == np.uint16:
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).to("cuda").view(torch.bfloat16)
    return torch.from_numpy(arr).to("cuda").to(target)


def _to_cuda_kwargs(kwargs):
    import torch

    cuda_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            cuda_kwargs[k] = v.to("cuda")
        elif isinstance(v, tuple):
            cuda_kwargs[k] = tuple(t.to("cuda") if isinstance(t, torch.Tensor) else t for t in v)
        else:
            cuda_kwargs[k] = v
    return cuda_kwargs


def _check_accuracy(outputs, eager_out) -> str | None:
    """Compare backend ``outputs`` against the eager reference — a **pure verdict**:
    ``None`` when every output passes, else a message naming what failed. No printing
    and no exit here: the check runs inside the bench worker child, whose stdout is the
    pickle protocol channel, so the verdict must travel as a return value and fatality
    is the caller's call — the ``--code`` / model path aborts on it, while ``run --ir``
    treats it as informational (a sliced reproducer is fed random *boundary* inputs
    that can be out-of-domain for the op, e.g. a kernel expecting a mean-of-squares
    gets random signed data → NaN from a downstream rsqrt)."""
    import numpy as np  # noqa: PLC0415

    # Multi-output: the eager reference may be a tuple — compare each backend
    # output against its positional eager counterpart (graph-output order).
    # Historic single-output behavior (every output vs THE eager tensor) is the
    # degenerate case of the fallback-to-first pairing below.
    if isinstance(eager_out, dict):
        if set(eager_out) != set(outputs):
            return f"CORRECTNESS FAIL: output names differ: Emmy={list(outputs)}, eager={list(eager_out)}"
        eager_refs = [eager_out[name] for name in outputs]
    else:
        eager_refs = list(eager_out) if isinstance(eager_out, (tuple, list)) else [eager_out]
    # Arrays, never Python lists: a list holds one float object per element. Measured at 2M cells the lists
    # peaked at 13.3 f64 copies of the output against 4.3 here, which at an LM-head output (134M cells) is 14 GB.
    eager_flats = [
        t.detach().cpu().double().flatten().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float64).ravel() for t in eager_refs
    ]
    if any(np.isnan(flat).any() for flat in eager_flats):
        return "eager reference contains NaN (reproducer inputs out of domain)"
    failures: list[str] = []
    for pos, (buf_name, arr) in enumerate(outputs.items()):
        eager_flat = eager_flats[pos] if len(eager_flats) == len(outputs) else eager_flats[0]
        values = np.asarray(arr, dtype=np.float64).ravel()
        if np.isnan(values).any():
            return f"CORRECTNESS FAIL: output {buf_name} contains NaN"
        if len(values) == len(eager_flat):
            diffs = np.abs(values - eager_flat)
            max_diff = float(diffs.max())
            mean_diff = float(diffs.mean())
            # Scale tolerance by max|eager| and by output dtype.
            #
            # fp32: matmul reduction-order drift grows with both K and
            # output magnitude. A fixed threshold flags benign drift on
            # randn×randn at large K as a failure (cp.async + split-K
            # atomic-add ordering vs eager's pairwise sum). Dual check:
            # ``max_diff <= 8% of peak`` (tight ceiling for the typical
            # case) OR ``mean_diff <= 0.5% of peak`` (escape hatch for
            # the long tail — splitK atomic-reduce on randn×randn
            # produces a handful of outliers at K=1024 / 2048 even when
            # the bulk of the output is accurate to 4+ decimals).
            # Codegen bugs that systematically corrupt the output (e.g.
            # the matmul_add fusion adding the residual per-K_s CTA)
            # fail both clauses: mean_diff lifts to ~2-3% of peak.
            #
            # fp16: every step has ~3 fewer decimal digits than fp32. The
            # split-K matmul path is dominated by atomicAdd into an
            # ``__half*`` buffer — each per-CTA partial converts to fp16
            # at the atomic boundary and loses ~11 bits per write. After
            # 1024 K-partials that's RMS error on the order of
            # ``peak * 0.3``. The proper fix (f32 scratch for split-K +
            # separate cast pass) is a future architectural change; for
            # now ``emmy run --bench`` needs to remain usable on
            # legitimate fp16 graphs, so the rtol budget tracks the
            # achievable accuracy of the current path. Bugs that
            # actually corrupt outputs still fail (whole-row mismatch /
            # NaN / order-of-magnitude wrong).
            is_fp16 = arr.dtype == np.float16
            # Three-part verdict — each clause with ONE job:
            #
            #   PASS iff (max_ok AND mean_diff ≤ mean_tol) OR escape_ok
            #   max_ok    = max_diff ≤ tol                                  (fp32)
            #             = max_diff ≤ 4·tol AND count(diff > tol) ≤ budget (fp16; budget
            #               EARNED by a heavy-tailed output — 0 on a gaussian one)
            #   escape_ok = mean_diff ≤ escape_tol [fp16: AND max_diff ≤ 4·tol AND count ≤ budget]
            #
            # - ``tol`` (the MAX clause) is the loose per-cell OUTLIER ceiling: fp16
            #   atomic-reduce accumulation can drift a cell up to ``peak`` in pathological
            #   cancellation cases (random-signed partials), fp32 stays at 8% of peak.
            # - ``mean_tol`` is the SYSTEMATIC-CORRUPTION gate and must stay tight: a
            #   permuted/scrambled output lands at mean_diff ≈ E|a−e| of independent
            #   same-distribution draws (≈ 15% of peak on a gaussian matmul output), while a
            #   correct kernel — including the half-atomic fp16 split-K path — measures
            #   ≤ ~0.1% of peak. The historical fp16 form (``max ≤ peak OR mean ≤ peak``)
            #   PASSED a matmul whose A slab was scrambled by a fill/drain swizzle mismatch
            #   (mean_diff 52 under tol 340, values of scale ~60 — the PR #354 sync-fill
            #   incident); 3% of peak keeps ~300× headroom over measured correct runs and
            #   fails any permutation-class bug REGARDLESS of its max (the old OR let any
            #   output with max_diff ≤ peak pass without ever consulting the mean).
            #   On HEAVY-TAILED outputs 3% of peak over-scales (a gemma-4 layer output:
            #   peak ≈ 24·RMS — a permutation scores mean_diff ≈ 1.04× that gate, a coin
            #   flip), so the gate is additionally floored by the output's own
            #   PERMUTATION SCORE: ``floor = mean|e_i − e_perm(i)|`` over a fixed shuffle
            #   of the eager values — what a scramble of THIS draw would measure. The
            #   ``min(3%·peak, 0.7·floor)`` keeps gaussian outputs on the 3%-of-peak gate
            #   (floor ≈ 1.13·σ ≫ it) and gives heavy-tailed ones ≥ 1.4× margin on BOTH
            #   sides (measured gemma layer-0: correct 0.52×, scramble 1.43× the gate).
            # - ``escape_tol`` (0.5% of peak) preserves the split-K outlier escape hatch: a
            #   handful of atomic-reorder outliers may exceed the max ceiling on randn×randn
            #   at K=1024/2048 while the bulk stays accurate to 4+ decimals — a near-zero
            #   mean vouches for them.
            #
            # fp16 MAX clause = outlier BUDGET + hard garbage ceiling, not a single-cell
            # max. HEAVY-TAILED fp16 outputs (a gemma-4 layer output: peak ≈ 24·RMS, the
            # layer_scalar / outlier-channel dynamic range) put single-cell diffs at
            # O(peak) IN-DISTRIBUTION on the current legitimate fp16 path (residual-
            # cancellation cells + the half-atomic split-K partials): measured correct
            # layer-0 runs show max_diff 0.7–0.9× peak with a tail draw occasionally
            # crossing 1.0× — a binary ``max ≤ peak`` intermittently aborts the bench on
            # a rerun-passes coin flip. The budget separates "a handful of tail cells"
            # (≤ ~0.0015% of cells over ``tol``, each under the 4× garbage ceiling) from
            # the real bug classes: a garbage/saturated cell (uninitialized read, Inf
            # math) blows the HARD ceiling regardless of count; a corrupt tile / row /
            # tail-guard window (the historical incidents: thousands of cells) blows the
            # BUDGET on either pass path (the escape hatch consults it too — a near-zero
            # mean must not vouch for a low-amplitude mass corruption); a permutation
            # blows the mean gate. The budget itself is EARNED by the output's tail shape
            # (``peak > 8·RMS`` — gemma layers sit at ~24×, a gaussian at ~5×): only there
            # do in-distribution cells legitimately land past ``tol``. A gaussian output
            # keeps the strict pre-budget max clause, and its handful of split-K reorder
            # outliers pass only through the escape hatch (near-zero mean AND ≤ budget).
            # Residual blind spot, accepted: ≤ budget cells at ≤ 4·tol with a near-zero
            # mean is statistically indistinguishable from the legit reorder class.
            # NOTE the mean gate's scramble
            # margin THINS on heavy-tailed outputs (a permuted gemma layer output scores
            # ≈ 1.04× mean_tol — 4% clear, vs ~5× on gaussian matmuls): that margin is
            # bounded by the current fp16 path's honest noise (mean_diff ≈ 0.4× mean_tol
            # on correct gemma runs), and widens only when split-K gets its f32 scratch.
            rel_tol = 1.0 if is_fp16 else 0.08
            abs_tol = 1e-1 if is_fp16 else 1e-3
            peak = float(np.abs(eager_flat).max())
            tol = max(abs_tol, rel_tol * peak)
            perm_floor = float(np.abs(np.random.default_rng(0).permutation(eager_flat) - eager_flat).mean())
            mean_tol = max(abs_tol, min(0.03 * peak, 0.7 * perm_floor))
            escape_tol = max(abs_tol, 0.005 * peak)
            outliers = int((diffs > tol).sum())
            budget = max(4, len(diffs) // 65536)
            rms = float(np.sqrt(np.mean(np.square(eager_flat))))
            heavy_tailed = peak > 8.0 * rms
            if is_fp16:
                # The hard 4·tol garbage ceiling bounds BOTH pass paths: the escape hatch
                # vouches for reorder outliers (peak-bounded by construction), never for a
                # garbage/saturated cell a near-zero mean would otherwise hide.
                max_ok = max_diff <= 4 * tol and outliers <= (budget if heavy_tailed else 0)
                escape_ok = mean_diff <= escape_tol and max_diff <= 4 * tol and outliers <= budget
            else:
                max_ok = max_diff <= tol
                escape_ok = mean_diff <= escape_tol
            verdict = "PASS" if (max_ok and mean_diff <= mean_tol) or escape_ok else "FAIL"
            if verdict == "FAIL":
                failures.append(
                    f"output {buf_name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} tol={tol:.6f}"
                    + (f" outliers={outliers}/budget {budget}" if is_fp16 else "")
                )
            else:
                logger.info(
                    "Accuracy vs eager: max_diff=%.6f mean_diff=%.6f tol=%.6f outliers=%d/%d PASS",
                    max_diff,
                    mean_diff,
                    tol,
                    outliers,
                    budget,
                )
        else:
            failures.append(f"output {buf_name}: size {len(values)} does not match eager {len(eager_flat)}")
    return f"accuracy check failed vs eager: {'; '.join(failures)}" if failures else None


_BACKEND_ALIASES = {
    "eager": "eager",
    "emmy": "emmy",
    "tcompile": "tcompile",
    "torch.compile": "tcompile",
    "compile": "tcompile",
}


def _resolve_backends(cli_value: str | None) -> set[str]:
    """Pick which bench backends to time. Precedence:

    1. ``--bench-backends`` CLI arg (comma-separated).
    2. ``EMMY_BENCH_BACKENDS`` env var (same syntax).
    3. Default ``eager,emmy`` — torch.compile is excluded so the
       per-case wall time isn't dominated by a ~0.8 s Inductor JIT
       that most users don't need on every run.

    ``emmy`` is always included even if omitted (the kernel under
    test is the point of the bench). Returns the canonical backend
    keys ``{"eager", "tcompile", "emmy"}``.
    """
    from emmy.compiler.pipeline.search.space import cold_cache  # noqa: PLC0415

    raw = config.bench_backends_raw(cli_value, default="emmy" if cold_cache() else "eager,emmy")
    selected: set[str] = {"emmy"}
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        canonical = _BACKEND_ALIASES.get(tok)
        if canonical is None:
            logger.error("unknown bench backend %r — choose from %s", tok, sorted(set(_BACKEND_ALIASES.values())))
            sys.exit(1)
        selected.add(canonical)
    if cold_cache() and selected != {"emmy"}:
        raise ValueError(
            "cold-cache mode times individual Emmy kernels; Torch forwards retain internal cache reuse. Use --bench-backends emmy."
        )
    return selected


class BackendFailure(str):
    """A requested torch backend that could not be built, in the closure dict and then in the
    results dict under its name, where its message stands where its latency would."""


def _build_torch_fns(module, args, kwargs, warmup, *, backends: set[str], check: bool = True):
    """Pre-build the per-backend ``torch_fns`` dict, including the
    ``torch.compile`` JIT step when requested. The JIT (mostly
    Inductor CPU work plus a few warmup launches) sits *outside* the
    GPU lock in ``handle_run`` so parallel workers can compile
    concurrently — the lock then only wraps the actual measurement
    iters.

    ``check`` holds the compiled output to eager on the dtype-scaled verdict; a quantized twin
    (:func:`_quantized_twin`) is timed without it, as Emmy's row is.

    Returns only the torch-side closures; emmy's bench loop is
    driven separately by ``backend.benchmark`` in ``_bench_interleaved``.
    """
    import torch

    torch_fns: dict[str, callable] = {}
    if "eager" in backends:
        torch_fns["Eager PyTorch"] = lambda: module(*args, **kwargs)
    if "tcompile" in backends:
        try:
            # Persistent/repeated bench workers compile a fresh closure of
            # the SAME ``torch_ref`` ``fn`` code object per row; dynamo's
            # recompile limit is keyed per code object, so after ~8 rows it
            # silently stops compiling and the tcompile column quietly measures
            # eager (observed: 8 µs rows reading 55-80 µs). Reset dynamo so
            # every bench compiles fresh.
            import torch._dynamo  # noqa: PLC0415

            torch._dynamo.reset()
            # The benchmark owns one outer CUDA graph for every backend. Letting
            # Inductor create its own graph here makes that capture nested and
            # invalidates it on recent PyTorch releases.
            compiled_torch_module = torch.compile(module, fullgraph=True, mode="max-autotune-no-cudagraphs")
            for _ in range(warmup + 5):
                with torch.no_grad():
                    compiled_torch_module(*args, **kwargs)
            if check:
                with torch.no_grad():
                    eager_output = module(*args, **kwargs)
                    compiled_output = compiled_torch_module(*args, **kwargs)
                # The same dtype-scaled verdict the Emmy output gets: a flat 1e-3 rejected Inductor's own
                # FP16 GEMM against cuBLAS on three of the six Gemma 4 projections on an RTX 4090.
                outputs = compiled_output if isinstance(compiled_output, (tuple, list)) else (compiled_output,)
                verdict = _check_accuracy({f"out{i}": t.detach().cpu().double().numpy() for i, t in enumerate(outputs)}, eager_output)
                if verdict:
                    raise ValueError(verdict)
            torch_fns["torch.compile"] = lambda: compiled_torch_module(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            # The column is reported failed, never silently dropped: this runs in the bench worker,
            # whose log the parent only shows on a crash.
            logger.warning("torch.compile failed: %s", e)
            torch_fns["torch.compile"] = BackendFailure(f"{type(e).__name__}: {e}")
    return torch_fns


def _capture_torch_fn(fn, batch_size=1):
    """Capture a complete batch of fixed-input forwards; one replay submits the whole window."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        for _ in range(batch_size):
            fn()
    return g.replay


def _capture_torch_fns(torch_fns: dict, batch_size=1) -> dict | None:
    """Capture every backend closure into a CUDA graph; ``None`` if ANY fails.

    All-or-nothing by design: a comparison table must not mix captured
    (pure-GPU) and uncaptured (dispatch-inclusive) timings, so a single
    backend resisting capture (e.g. torch.compile guard machinery) falls the
    whole invocation back to uncaptured timing."""
    captured: dict = {}
    for name, fn in torch_fns.items():
        if isinstance(fn, BackendFailure):
            captured[name] = fn
            continue
        try:
            captured[name] = _capture_torch_fn(fn, batch_size)
        except Exception as exc:  # noqa: BLE001 — capture is best-effort, fallback covers
            logger.warning("CUDA graph capture failed for %s (%s) — benching all backends uncaptured", name, exc)
            return None
    return captured


async def _bench_interleaved_captured(module, args, kwargs, backend, lowered, warmup, iters, *, torch_fns, input_data=None):
    """All-or-nothing CUDA-graph-captured interleaved bench.

    A capture failure at any batch size re-runs every backend uncaptured. Other benchmark
    failures propagate. Returns ``(results, bench, captured)`` on the same live inputs."""
    from emmy.compiler.backend.cuda.program import GraphCaptureError

    try:
        results, bench = await _bench_interleaved(
            module, args, kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns, capture_graphs=True, input_data=input_data
        )
        if not torch_fns:
            return results, bench, bench.captured  # cold Emmy-only timing deliberately has no whole-program window
        if bench.captured and (bench.num_launches <= 1 or bench.e2e_ms is not None):
            return results, bench, True
    except GraphCaptureError:
        pass
    logger.warning("CUDA graph capture failed — re-benching all backends uncaptured")
    results, bench = await _bench_interleaved(
        module, args, kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns, capture_graphs=False, input_data=input_data
    )
    return results, bench, False


async def _bench_interleaved(
    module, args, kwargs, backend, compiled_graph, warmup, iters, *, torch_fns, capture_graphs=False, input_data=None
):
    """Time the selected backends by alternating one iter of each per
    loop step. All backends see the same warm GPU state across the
    measurement window — same clocks, same caches, same thermal drift
    — instead of running in sequential phases that each get a
    different steady state.

    Driven by ``backend.benchmark_async(on_iter=...)``: emmy is the
    backbone, ``on_iter`` runs each torch closure and records its
    cuda events, and the same call returns per-launch emmy
    timings — so the kernel-stats breakdown shares the same warm
    state as the comparison numbers.

    Per-iter ``torch.cuda.Event``s queue on the (legacy) default
    stream, and the emmy program's launches are issued on that same stream
    (``CompiledProgram.on_stream``), so events see all preceding work.

    ``torch_fns`` is the pre-built backend closure dict from
    :func:`_build_torch_fns` (``handle_run`` builds it outside the
    GPU lock so the slow ``torch.compile`` JIT runs concurrently with
    peer workers; the lock then wraps only this measurement loop).

    ``capture_graphs`` captures each calibrated batch before its first event window on both
    sides. :func:`_bench_interleaved_captured` owns the all-or-nothing fallback.

    ``input_data`` binds the emmy program to the same values the torch closures read. Without
    it emmy times a synthetic fill while torch times real inputs, and a kernel whose cost
    depends on its values (a division's slow path, an exp near overflow) reads differently.
    """
    import torch

    from emmy.compiler.backend.cuda.program import GraphCaptureError

    # Each entry: (start_event, stop_event, batch_size_used). The callback supplies the largest
    # per-launch batch; whole-program Emmy windows calibrate their batch from the summed latency.
    # Each captured window submits one graph, with elapsed time normalized by its own batch size.
    failed = {name: fn for name, fn in torch_fns.items() if isinstance(fn, BackendFailure)}
    torch_fns = {name: fn for name, fn in torch_fns.items() if name not in failed}
    torch_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event, int]]] = {name: [] for name in torch_fns}
    captures = {}

    def on_iter(batch_size: int = 1) -> None:
        if capture_graphs and batch_size not in captures:
            captures[batch_size] = _capture_torch_fns(torch_fns, batch_size)
            if captures[batch_size] is None:
                raise GraphCaptureError("torch batch capture failed")
        for name, fn in torch_fns.items():
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                if capture_graphs:
                    captures[batch_size][name]()
                else:
                    for _ in range(batch_size):
                        fn()
                stop.record()
            torch_events[name].append((start, stop, batch_size))

    bench = await backend.benchmark_async(
        compiled_graph, warmup=warmup, num_iters=iters, on_iter=on_iter, capture_graphs=capture_graphs, input_data=input_data
    )
    torch.cuda.synchronize()

    results: dict[str, float] = {}
    measured_count = len(bench.per_launch[0].samples or ()) if bench.per_launch else 0
    for name, evt in torch_events.items():
        measured = evt[-measured_count:] if measured_count else []
        if measured:
            # ``elapsed_time`` is in ms across the whole batch; divide
            # by the batch size and multiply by 1000 to get per-call us.
            # ``min`` (best-case iter) for both torch and emmy — the
            # least-noise latency, matching tune's min-over-variants reporting
            # so tune and run numbers are comparable.
            per_iter_us = [s.elapsed_time(e) * 1000.0 / b for s, e, b in measured]
            results[name] = min(per_iter_us)
    # Whole-program (e2e) time when available — windows around replays of one
    # all-launches CUDA graph, the same semantics the captured torch closures
    # above get. The fallback sums per-launch solo windows, which is not an
    # end-to-end number (no cross-kernel cache effects).
    dep_ms = bench.e2e_min_ms if bench.e2e_min_ms is not None else (bench.min_ms if bench.min_ms is not None else bench.time_ms)
    results["Emmy"] = dep_ms * 1000
    results.update(failed)
    return results, bench


def _print_table(results, note: str | None = None):
    from emmy.commands.table import Col, render_table  # noqa: PLC0415

    eager_us = results.get("Eager PyTorch", 0)
    cols = [Col("Backend"), Col("Latency (us)", "r"), Col("vs Eager", "r")]
    rows = [
        [
            name,
            "failed" if isinstance(us, str) else f"{us:.0f}",
            f"{eager_us / us:.2f}x" if eager_us and not isinstance(us, str) and us > 0 else "-",
        ]
        for name, us in results.items()
    ]
    print()
    for line in render_table(cols, rows, rule=True):
        print(line)
    for name, us in results.items():
        if isinstance(us, str):
            print(f"{name}: {us}")
    if note:
        print(note)
