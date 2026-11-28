"""``emmy run``: compile a model, a ``--code`` expression, an IR file or a golden target for CUDA, execute it and
check it against eager PyTorch. With ``--bench`` it times eager, ``torch.compile`` and Emmy, benches the pinned
rows a golden or ``--ab`` names beside the greedy pick, records the clean measurements in the tune DB, and with
``--record`` / ``--record-greedy`` writes them into a working golden; ``--tune N`` measures N schedule rows of one
kernel.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from emmy import config
from emmy.commands.compile import CommandError, GoldenSelection, PinnedRow, add_golden_arg
from emmy.commands.ncu import run_ncu_profile
from emmy.compiler.backend.correctness import (
    REFERENCE_SELF_DISAGREES,
    check_accuracy,
    comparison_outputs,
    computed_from_formats,
    eager_output,
    output_formats,
    quantized_outputs,
    quantized_twin,
    reference_precision,
    resolve_reference_disagreement,
    strict_correctness_proof,
    wrong_answer_flag,
)
from emmy.compiler.backend.cuda.compare import bench_lowered_vs_torch, collect_sym_env, resolve_backends
from emmy.compiler.loader.binder import bind_inputs
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
        help="Declare NAME paged as a serving boot below its device budget does: a program input one page per row of "
        "its leading axis (EMMY_GEN_EXPERTS_GPU), a weight one page (EMMY_GEN_VISION_GPU). The schedules a paged "
        "operand can take are what the compile offers and the bench times. Repeatable.",
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
            "name or an unambiguous name substring. With --tune, tunes that kernel alone, its own cut, layout and split "
            "arms included: each compile takes seconds where its layer takes minutes."
        ),
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Autotune the program: measure N schedule rows of each kernel alone, the prior's ten best first, then the "
            "rows Bayesian optimization expects to improve most, each benched as an --ab row would be. A program of "
            "several kernels has each kernel of its kernel set tuned, then the kernels of the kernel-set arms the "
            "compile met and did not take, so the next compile picks its kernel set from measured rows. Clean rows land "
            "in the tune DB. Requires --bench and --code, --realization or --kernel: an --ir file holds no program the "
            "schedule space can be enumerated from."
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
            "(realized-vs-pinned knob check, wrong-answer check). "
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
            "as a bench_fail row. Flagged rows (pin mismatch, wrong answer) and the --ir path never record."
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
    from emmy.compiler.target import add_target_arg

    add_target_arg(parser, dest="gpu_arch", option="--gpu-arch")
    parser.set_defaults(func=handle_run)


def handle_run(args):
    try:
        code = _run(args)
    except CommandError as exc:
        logger.error("%s", exc)
        sys.exit(exc.code)
    if code:
        sys.exit(code)  # every row is reported; a failed row, greedy or pinned, exits non-zero


def _run(args) -> int:
    from emmy.compiler.target import CPU_TARGET, apply_target_arg

    cpu = args.gpu_arch == CPU_TARGET
    if not cpu:
        apply_target_arg(args, dest="gpu_arch")
    if args.profile:
        args.bench = True  # --profile re-launches under ncu via the bench path; profiling implies benching
    if getattr(args, "cold_cache", False) and not args.bench:
        raise CommandError("--cold-cache requires --bench")
    if getattr(args, "cold_cache", False) and args.record:
        raise CommandError("--record stores whole-row comparisons; cold-cache mode measures individual Emmy kernels. Use --record-greedy.")
    verbose = getattr(args, "verbose", 0)
    if verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.record and not (args.golden and args.bench):
        raise CommandError("--record requires --golden PATH and --bench")
    if args.record_greedy and not (args.golden and args.bench):
        raise CommandError("--record-greedy requires --golden PATH and --bench")
    if getattr(args, "kernel", None) and (args.record or (args.record_greedy and not args.realization)):
        raise CommandError("isolated kernel recording requires --realization NAME --record-greedy; --record is a whole-target latency")
    if getattr(args, "program", None) is not None and (args.record or args.record_greedy):
        raise CommandError("--program benches a whole traced program; --record / --record-greedy write the rows of one --realization NAME")
    if (args.ab or getattr(args, "tune", None)) and args.bench and not getattr(args, "no_record_evidence", False):
        # A sweep exists to leave its rows in the tune DB for a later record; below the bench standard
        # it records nothing, and a record run that copies that DB would silently fall to the prior.
        from emmy.compiler.pipeline.search.bench_record import MIN_RECORD_ITERS, MIN_RECORD_WARMUP, meets_quality_bar  # noqa: PLC0415

        if not meets_quality_bar(args.warmup, args.iters):
            raise CommandError(
                f"--ab / --tune sweep at --warmup {args.warmup} / --iters {args.iters} is below the tune bench standard (--warmup "
                f">= {MIN_RECORD_WARMUP}, --iters >= {MIN_RECORD_ITERS}), so none of its rows would reach the tune DB; raise "
                "them, or pass --no-record-evidence to measure only"
            )
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
            raise CommandError(f"--record / --record-greedy: {exc} — record into a file seeded for this card") from exc
    with (
        config.strict_evidence_override(True if getattr(args, "strict_evidence", False) else None),
        pinned_knobs({"COLD_CACHE": True} if getattr(args, "cold_cache", False) else {}),
    ):
        if cpu:
            _handle_run_cpu(args)
            return 0
        if args.golden and not args.realization and not getattr(args, "kernel", None) and getattr(args, "program", None) is None:
            _run_golden_targets(args)
            return 0
        return _handle_run_once(args)


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
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
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
        raise CommandError(f"--gpu-arch {CPU_TARGET} runs a model id or --code{f', without {", ".join(other)}' if other else ''}")
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
        raise CommandError(f"the CPU backend cannot compile this program: {exc}", 1) from exc
    result, _ = backend.run(program, input_data=bind_inputs(program.graph, module, example_args, example_kwargs, checkpoint=checkpoint))
    if checkpoint is not None:
        # The graph declares the quantization, so eager torch is no longer its reference: the numpy backend on the
        # same graph is, as on CUDA.
        from emmy.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415

        numpy = NumpyBackend()
        data = bind_inputs(graph, module, example_args, example_kwargs, checkpoint=checkpoint)
        reference = numpy.run(numpy.compile(graph), input_data=data)[0].outputs
        reference = comparison_outputs({name: reference[name] for name in result.outputs}, graph)
    else:
        reference = eager_output(module, example_args, example_kwargs, device="cpu")
    with reference_precision():
        err = check_accuracy(comparison_outputs(result.outputs, program.graph), reference)
    if err is not None:
        raise CommandError(err, 1)
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
            persist_kernel_perf(
                db,
                program.ctx,
                program.tiles[nid],
                us=statistics.median(us),
                status="ok",
                knobs=decisions[nid],
                sizes=program.native.executor.env(),
            )
    finally:
        db.close()


def _handle_run_once(args, *, pins=None, explicit=True, document=None, on_compiled=None) -> int:
    """One program compiled, run and, with ``--bench``, benched: the exit code. ``pins``, ``explicit`` and
    ``document`` are what a whole-file walk knows of its target (``compile.resolve_golden_arg``), and ``on_compiled``
    hears the greedy compile of a golden target."""
    try:
        import torch
    except ImportError as exc:
        raise CommandError("torch is required: pip install torch", 1) from exc

    from emmy.commands.compile import load_or_trace, resolve_golden_arg, selected_regime, validate_trace_adapter_args
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.dump import CompilerDump
    from emmy.compiler.pipeline.search.space import cold_cache  # noqa: PLC0415

    selection = resolve_golden_arg(args, pins=pins, explicit=explicit, document=document)
    if selection is not None and getattr(args, "cold_cache", False):
        if any(row.pins.get("COLD_CACHE") is False for row in selection.rows or ()):
            raise ValueError("--cold-cache contradicts the selected golden row's explicit COLD_CACHE=false pin")
    regime = selected_regime(selection)  # the selected rows' input regime, published around the compile and bench below
    cold = cold_cache() or bool(regime.get("COLD_CACHE"))
    if cold and getattr(args, "record", False):
        raise ValueError("--record stores whole-row comparisons; cold-cache mode measures individual Emmy kernels. Use --record-greedy.")
    if (getattr(args, "record", False) or getattr(args, "record_greedy", False)) and not cold:
        args.bench_backends = f"{config.bench_backends_raw(args.bench_backends)},tcompile"
    try:
        backends = resolve_backends(args.bench_backends) if args.bench else set()
    except ValueError as exc:
        raise CommandError(str(exc), 1) from exc
    if sum(x is not None for x in (args.input, args.code, args.ir)) > 1:
        raise CommandError("input / --code / --ir are mutually exclusive", 1)
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
        raise CommandError(
            f"--json names ONE file for a single target, but {args.json!r} is a directory. A directory is for a multi-target "
            "--golden walk, which writes one file per realization inside it."
        )
    if args.ab or getattr(args, "tune", None):
        flag = "--ab" if args.ab else "--tune"
        if not args.bench:
            raise CommandError(f"{flag} requires --bench (the A/B rows render in the kernel table)")
        if args.code is None and ir_path is None and selection is None:
            raise CommandError(
                f"{flag} requires a re-lowerable input: --code, --realization, or --ir (each config re-lowers a fresh graph)"
            )
        if getattr(args, "tune", None) and ir_path is not None:
            raise CommandError("--tune requires --code, --realization or --kernel: an IR file holds no program to enumerate schedules from")
    if args.ab:
        try:
            _ab_samples(args.ab)  # fail fast on a malformed KNOBS spec
        except ValueError as exc:
            raise CommandError(f"--ab: {exc}") from exc
    if args.strict_correctness and not args.bench:
        raise CommandError("--strict requires --bench")
    if not torch.cuda.is_available():
        raise CommandError("CUDA GPU required", 1)

    if ir_path is not None or selection is not None:
        from emmy.commands.compile import selected_decisions  # noqa: PLC0415

        if ir_path is not None:
            args.ir = ir_path
        if getattr(args, "cold_cache", False):
            regime["COLD_CACHE"] = True
        if args.kernel:
            regime.update(selected_decisions(args, selection))  # Keep the selected row's pins live for the existing record guards.
        with pinned_knobs(regime), _family_layouts(args, selection, explicit):
            return _handle_run_ir(args, CudaBackend, CompilerDump, selection, on_compiled)

    if args.input is None and args.code is None:
        raise CommandError("Either a model ID / .json input, --code, --golden PATH / --realization NAME, or --ir is required", 1)

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
            input_data = bind_inputs(compiled, module, example_args, example_kwargs, checkpoint=quantized_checkpoint or args.input)
        except RuntimeError as exc:
            raise CommandError(str(exc), 1) from exc
        try:
            if not skip_accuracy:
                run_result, _ = backend.run(compiled, input_data=input_data)
                if dump and backend.last_debug_result is not None:
                    dump.dump_per_launch_values(backend.last_debug_result.per_launch)
                with reference_precision():
                    outputs = comparison_outputs(run_result.outputs, compiled)
                    err = check_accuracy(outputs, eager_output(module, example_args, example_kwargs))
                if err is not None:
                    if not quantized:
                        raise CommandError(err, 1)
                    logger.info("%s", err)
            else:
                # ncu child: one emmy launch (our metrics) + one eager forward
                # (the reference rows for the comparison table); no accuracy diff.
                backend.run(compiled, input_data=input_data)
                eager_output(module, example_args, example_kwargs)
        except RuntimeError as exc:
            # Per-launch watchdog fired in ``run_program`` (kernel >1 s).
            # The CUDA context is dirty — bypass Python cleanup so the driver's
            # teardown doesn't block on the still-running kernel.
            sys.stderr.write(f"accuracy check failed: {exc}\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 0

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
    session = asyncio.run(
        _bench_session(
            args,
            backend,
            compiled,
            torch_spec=(
                "trace_args",
                {
                    "code": args.code,
                    "input": quantized_checkpoint or args.input,
                    "adapter": getattr(args, "adapter", "causal-lm"),
                    "layer": args.layer,
                    "seq_len": args.seq_len,
                    "dynamic": list(args.dynamic) if getattr(args, "dynamic", None) else None,
                },
            ),
            base_s=_GREEDY_COMPARE_BASE_S,
            twin=graph if "tcompile" in backends else None,
            source=args.code,
            rows=_ab_samples(args.ab, dynamic=args.dynamic) if args.ab else [],
            accuracy=not quantized,
            accuracy_fatal=True,
            quantize=args.quantize,
        )
    )
    return _report(args, compiled, session, dump=dump, accuracy_fatal=True, note=_symbolic_bench_note(collect_sym_env([compiled])))


@dataclass(frozen=True)
class _Session:
    """What one ``--bench`` session measured: the greedy comparison beside torch (``results``, ``bench``), the greedy
    graph's isolated re-bench and the pinned rows. ``greedy_fail`` says why the greedy row has no timing,
    ``reference_error`` why no pinned row could be benched at all."""

    results: dict | None = None
    bench: object = None
    captured: bool = False
    torch_available: bool = False
    accuracy_error: str | None = None
    correctness: dict | None = None
    greedy_fail: str | None = None
    greedy_reference_us: float | None = None
    sym_env: dict | None = None
    same_input_reference: bool = False
    reference_error: str | None = None
    isolated: _GoldenBench | None = None
    rows: list | None = None


async def _bench_session(
    args,
    backend,
    graph,
    *,
    torch_spec,
    base_s,
    twin,
    source,
    rows,
    passes=None,
    route=None,
    torch_twin=True,
    same_input_greedy=False,
    measure_greedy=False,
    accuracy=False,
    accuracy_fatal=False,
    quantize=None,
    computed_from=None,
    sym_env=None,
    keep_worker=False,
) -> _Session:
    """One ``--bench`` session, every GPU touch a job on the backend's one persistent SIGKILL-able worker: the parent
    launches no kernel. A hung kernel dies with the child, its row is reported ``bench_fail``, and the next job starts
    a fresh one.

    First the greedy comparison of ``graph`` — eager / torch.compile / emmy, the torch side rebuilt in the child from
    ``torch_spec``; ``accuracy`` asks for the in-child check on the module's own inputs, and ``twin`` is the frontend
    graph ``torch.compile`` jits, which the wall cap grants time for (``base_s`` is the child's setup budget). Then,
    when ``source`` holds a program to lower again (a ``--code`` string, a golden's embedded program, or an IR graph
    with the ``passes`` left to run), the pinned ``rows`` and the ``--tune`` rows (``route``: the kernel-set decisions
    a tune row compiles under), beside the greedy graph benched once more emmy-only through the pinned-row path. The
    comparison row is timed interleaved with torch, whose allocator state and cuBLAS L2 carveouts stay resident, so
    it reads about 7% apart from a pinned row of the same config: pinned speedups read against the isolated row.

    A greedy pick that fails to bench is a finding, not the end: it is recorded (:func:`_record_greedy_failure`) and
    the pinned rows still bench, unverified (:data:`UNVERIFIED_ROW`) where a Torch twin exists (``torch_twin``) and
    refused where the greedy outputs were the only reference (:func:`pinned_reference_refusal`). A failed greedy is
    not benched again, and a golden row with no knobs, which would compile to the same pick, is skipped. A wrong
    answer stops the pinned rows where it is fatal (``accuracy_fatal``: a traced module on its own inputs) or strict.
    ``same_input_greedy`` makes the greedy outputs the strict reference of a target with no Torch twin, and
    ``measure_greedy`` benches the isolated row with no pinned row to compare, for a record. ``keep_worker`` leaves the
    worker to the session this one runs inside (a kernel's tune inside its program's)."""
    strict = bool(getattr(args, "strict_correctness", False))
    rows = list(rows) if source is not None else []
    tune = getattr(args, "tune", None) if source is not None else None
    results = bench = accuracy_error = correctness = ab_ref = greedy_fail = reference_us = reference_error = isolated = benched = None
    torch_available = captured = same_input_reference = False
    try:
        try:
            resp = await backend.benchmark_compare_async(
                graph,
                torch_spec=torch_spec,
                bench_backends=args.bench_backends,
                wall_timeout_s=_compare_wall_s(graph, backend, base_s=base_s, twin=twin),
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                accuracy=accuracy,
                want_ref=source is not None and bool(same_input_greedy or rows or measure_greedy or tune),
                strict_accuracy=strict and not same_input_greedy,
            )
        except RuntimeError as exc:
            # Worker SIGKILL (hung kernel), in-child bench budget, EOF.
            greedy_fail = f"greedy run/bench failed: {exc}"
            _record_greedy_failure(args, backend, graph, exc)
        else:
            results, bench, captured = resp["results"], resp["result"], resp["captured"]
            torch_available, accuracy_error, ab_ref = resp["torch_available"], resp["accuracy_error"], resp["run_io"]
            reference_us, correctness, sym_env = resp.get("reference_run_us"), resp.get("correctness"), resp.get("sym_env")
            same_input_reference = same_input_greedy and ab_ref is not None
            if same_input_reference:
                correctness = strict_correctness_proof(ab_ref[1], ab_ref[1], reference="same-input-greedy")
            if resp.get("greedy_error"):
                greedy_fail = f"greedy timing failed after reference execution: {resp['greedy_error']}"
                _record_greedy_failure(args, backend, graph, resp["greedy_error"])
        ref_key = uuid.uuid4().hex if ab_ref is not None else None
        accurate = accuracy_error is None or not (accuracy_fatal or (strict and not same_input_greedy))

        async def isolated_greedy():
            return await _bench_greedy_isolated(backend, graph, warmup=args.warmup, iters=args.iters, ref=ab_ref, ref_key=ref_key)

        if rows:
            reference_error = pinned_reference_refusal(ab_ref=ab_ref, torch_twin=torch_twin, greedy_fail=greedy_fail)
            if reference_error is None and accurate:
                unverified = (
                    None if ab_ref is not None else f"{UNVERIFIED_ROW}: {greedy_fail or 'the greedy worker returned no run outputs'}"
                )
                if greedy_fail:
                    same_election = [row for row in rows if row.record is not None and not row.knobs]
                    rows = [row for row in rows if row not in same_election]
                    if same_election:
                        logger.warning(
                            "%s — pinned re-bench of %s skipped: it has no knobs to pin it away from the greedy pick that just failed",
                            greedy_fail,
                            ", ".join(row.name for row in same_election),
                        )
                    if rows:
                        logger.error("%s — untimed greedy is ineligible; pinned rows still bench", greedy_fail)
                else:
                    isolated = await isolated_greedy()
                benched = await _bench_golden_variants(
                    backend,
                    source,
                    rows,
                    passes=passes,
                    warmup=args.warmup,
                    iters=args.iters,
                    ref=ab_ref,
                    ref_key=ref_key,
                    strict_correctness=strict and ab_ref is not None,
                    strict_reference="same-input-greedy" if same_input_greedy else "eager",
                    quantize=quantize,
                    unverified=unverified,
                    ref_knobs=_cuda_knob_dicts(graph),
                    computed_from=computed_from,
                )
        elif measure_greedy and source is not None and greedy_fail is None and (same_input_greedy or accuracy_error is None):
            isolated = await isolated_greedy()
        if tune and greedy_fail is None and (accuracy_error is None or not accuracy_fatal):
            if isolated is None:
                isolated = await isolated_greedy()
            benched = (benched or []) + await _tune(backend, args, source, graph, ab_ref, ref_key, route=route)
    finally:
        if not keep_worker:
            await backend.aclose_async_worker()
    return _Session(
        results=results,
        bench=bench,
        captured=captured,
        torch_available=torch_available,
        accuracy_error=accuracy_error,
        correctness=correctness,
        greedy_fail=greedy_fail,
        greedy_reference_us=reference_us,
        sym_env=sym_env,
        same_input_reference=same_input_reference,
        reference_error=reference_error,
        isolated=isolated,
        rows=benched,
    )


def _report(
    args, graph, session: _Session, *, dump, accuracy_fatal, note=None, frontend_runnable=True, tc_checked=True, golden_rows=0, decisions=()
) -> int:
    """Everything a bench session is followed by, and the exit code: the tables, the ``--json`` record, the golden
    write-back (``--record`` / ``--record-greedy``), the tune DB evidence and the ncu profile. Every row is reported
    before anything fails: a failed row, greedy or pinned, makes the code non-zero. ``decisions`` are the kernel-set
    decisions the greedy compile of a golden target took, ``None`` where the tune DB gets no evidence (an IR file,
    whose serialization drops the knobs); ``golden_rows`` is how many golden rows a strict run must have benched."""
    strict = bool(getattr(args, "strict_correctness", False))
    if session.reference_error is not None:
        raise CommandError(session.reference_error, 1)
    if session.accuracy_error is not None:
        if accuracy_fatal:
            # The deployed program computes the wrong answer, so no latency comparison is meaningful (the child
            # skipped the bench; the pinned rows were skipped).
            raise CommandError(session.accuracy_error, 1)
        if strict:
            logger.error(session.accuracy_error)
        else:
            logger.warning("%s — non-fatal (random-input reproducer); benched anyway", session.accuracy_error)
    results, bench, rows, isolated = session.results, session.bench, session.rows, session.isolated
    if dump and bench is not None:
        dump.dump_benchmark(bench)
        if session.torch_available:
            _dump_bench_compare(dump.dir, results, args.warmup, args.iters)
    if session.torch_available:
        capture_note = None if session.captured else "(graph-capture fallback: timings include host launch overhead)"
        _print_table(results, note="\n".join(n for n in (note, capture_note) if n) or None)
    elif bench is not None:
        # No Torch twin (a Loop, Tile or later IR): emmy alone.
        from emmy.commands.table import Col, render_table  # noqa: PLC0415

        print()
        for line in render_table([Col("Backend"), Col("Latency (us)", "r")], [["Emmy", f"{bench.time_ms * 1000:.0f}"]], rule=True):
            print(line)
    _print_kernel_stats(graph, bench, golden_benches=rows, greedy_fail=session.greedy_fail, greedy_iso=isolated, sym_env=session.sym_env)
    strict_errors = None
    if strict:
        strict_errors = _strict_benchmark_errors(
            args,
            results,
            bench,
            session.captured,
            session.correctness,
            rows,
            golden_rows=golden_rows,
            frontend_runnable=frontend_runnable,
            same_input_reference=session.same_input_reference,
        )
    if getattr(args, "json", None):
        _write_ab_json(
            args,
            results or {},
            graph,
            bench,
            rows,
            greedy_fail=session.greedy_fail,
            greedy_iso=isolated,
            greedy_reference_us=session.greedy_reference_us,
            correctness=session.correctness,
            strict_errors=strict_errors,
            accuracy_error=session.accuracy_error,
            checked=session.torch_available,
            tc_checked=tc_checked,
        )
    if getattr(args, "record", False) or getattr(args, "record_greedy", False):
        # Missing inventory rows are not rejected answers: a fresh recording may create them.
        refusal = "; ".join(strict_errors or []) or record_refusal(
            _cuda_knob_dicts(graph),
            session.accuracy_error if strict else None,
            _cuda_kernel_names(graph),
            benches=[isolated, *(rows or [])],
        )
        if refusal is not None:
            raise CommandError(f"not recording {args.realization} — {refusal}", 1)
    if getattr(args, "record", False):
        _record_golden_latency(args, results or {}, rows)
    if getattr(args, "record_greedy", False):
        _record_greedy_pick(args, graph, bench, isolated, decisions, results or {})
    for error in strict_errors or []:
        logger.error("strict: %s", error)
    if decisions is not None:
        _record_bench_evidence(args, rows, isolated, decisions=decisions)
    if args.profile and session.greedy_fail is None:
        run_ncu_profile(args, dump_dir=dump.dir if dump else None)
    failed = (
        session.greedy_fail is not None
        or (isolated is not None and isolated.status != "ok")
        or any(row.status != "ok" for row in rows or [])
    )
    return int(bool((strict and session.accuracy_error is not None) or strict_errors or failed))


def _record_golden_latency(args, results: dict, golden_benches) -> None:
    """Write this card's emmy / torch.compile timings back into the benched working golden.

    The emmy number is the PINNED row's when one of the target's root was measured, not the greedy
    pick's: a corpus case names a schedule, and recording whatever the prior happened to choose would
    store a timing for a different kernel than the one the file describes. A piece's row benches
    under the decisions that mint its kernel, with its schedule replayed on every kernel of the
    target: a kernel set no deploy builds, so it times nothing the target deploys, and the greedy
    pick's whole time is recorded instead.
    """
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import record_latency, regime_live  # noqa: PLC0415

    # Only a row of the NAMED realization can carry that realization's latency. A piece's row
    # of the same target is benched beside it and holds ONE kernel of the program, so neither its
    # timing nor its schedule knobs describe the row this writes — narrowing by them selects
    # nothing and the write is refused. A name recorded in several regimes benches a row of each,
    # but the run deploys the live regime, so only a row of it carries the latency.
    measured = [
        gb
        for gb in golden_benches or []
        if gb.status == "ok"
        and gb.bench is not None
        and gb.sample.name == args.realization
        and regime_live(gb.sample.pins)
        and set(gb.sample.route) <= {"PLACE"}  # a row of the root: its route names no decision
    ]
    # Several measured schedules of one target in one regime: the evidence pick deploys the fastest, so its row
    # carries the target's latency.
    measured = sorted(measured, key=lambda gb: gb.bench.total_us or float("inf"))[:1]
    emmy_us = measured[0].bench.total_us if measured else results.get("Emmy")
    tcompile_us, eager_us = _torch_timings(results)
    if not emmy_us:
        raise CommandError(f"--record measured no Emmy timing for {args.realization}")
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
    launches = [list((side.per_launch if side is not None else None) or []) for side in (isolated, bench)]
    if not nodes or any(len(side) != len(nodes) for side in launches):
        raise CommandError(f"--record-greedy needs the greedy row and its isolated re-bench timed per kernel for {args.realization}")

    def us(launch) -> float:
        # The per-launch median, the statistic the tune DB ranks by: rows are ranked against
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
    (``explicit`` false), so proposals are not benched as if they were recorded truths. Targets run in
    serving's build order (:func:`_targets_in_build_order`), each compiled with the weight layouts its family settled
    so far.
    """
    from copy import copy  # noqa: PLC0415

    from emmy.compiler.pipeline.passes.tile._layout import settled_layouts  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import GoldenFile  # noqa: PLC0415
    from emmy.serving.twins import LayoutConflict, graph_layouts  # noqa: PLC0415

    if args.input or args.code or args.ir:
        raise CommandError("--golden is mutually exclusive with positional input / --code / --ir")
    try:
        document = GoldenFile.load(args.golden)
    except (OSError, ValueError) as exc:
        raise CommandError(f"cannot load --golden {args.golden}: {exc}") from exc
    if not document.rows:
        raise CommandError(f"--golden contains no realizations: {args.golden}")
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
            raise CommandError(f"multi-target --json directory must be empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # One target's failure (a compile error, a wrong answer, a hung bench) must not hide the targets after it: every
    # target runs and reports, and the walk exits non-zero at the end.
    failed: list[str] = []
    for index, (name, pins, _target, scope) in enumerate(ordered):
        target_args = copy(args)
        target_args.realization = name
        if output_dir is not None:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "target"
            target_args.json = str(output_dir / f"{index:03d}-{safe_name}.json")
        if scope.layouts:
            logger.info("%s compiles with %s", name, scope.describe())
        compiled: list = []
        try:
            with settled_layouts(scope.settled()):
                if _handle_run_once(target_args, pins=pins, explicit=False, document=document, on_compiled=compiled.append):
                    failed.append(name)
        except Exception as exc:  # noqa: BLE001 — one target's error, a lowering one included, must not hide the rest
            logger.error("%s: %s", name, exc)
            failed.append(name)
        # What the target compiled binds its layouts for the family, whatever its bench did after.
        for graph in compiled:
            try:
                scope.record(name.split(".", 1)[0], graph_layouts(graph))
            except LayoutConflict as exc:
                logger.error("%s: %s", name, exc)
                failed.append(name)
    if failed:
        raise CommandError(f"{len(failed)} of {len(names)} realizations failed: {', '.join(failed)}", 1)


def _targets_in_build_order(document, target_rows, *, of=None) -> list[tuple[str, dict, str, object]]:
    """The targets of ``target_rows`` (:meth:`GoldenFile.target_rows`) as ``(name, pins, target, scope)``, in the order
    serving builds twins (``serving.twins.twins_in_build_order``, which reads the checkpoint tensors each target's
    traced program binds and its input and output sizes): per input regime, a family's programs narrowest first, each
    with its family's layout scope — with ``of``, a traced program's index, its family's alone. A target is named by
    the shortest row name on it or, with none, on its pieces."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.serving.twins import fork_less_weights, twins_in_build_order  # noqa: PLC0415

    targets = [
        (
            min([row.name for row in rows if row.kernel == target] or [row.name for row in rows], key=lambda name: (len(name), name)),
            dict(pins),
            target,
        )
        for (target, pins, _), rows in target_rows.items()
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
    for pin_set in dict.fromkeys(_regime(pins) for _, pins, _ in targets):
        same_pins = [(name, pins, target) for name, pins, target in targets if _regime(pins) == pin_set]
        reached = {index: programs[index] for index in dict.fromkeys(traced[target] for _, _, target in same_pins)}
        for index, scope in twins_in_build_order(reached, fork_less, of=of):
            ordered += [(name, pins, target, scope) for name, pins, target in same_pins if traced[target] == index]
    return ordered


def _regime(pins: dict) -> tuple:
    """The regime ``pins`` spell, as a comparable key: a row that spells every precision gate and one that spells only
    the ``FAST_MATH`` umbrella name the same regime, so twins recorded either way are one family."""
    from emmy.compiler.pipeline.search.space import regime_pins  # noqa: PLC0415

    return tuple(sorted(regime_pins(pins).items()))


def _family_layouts(args, selection: GoldenSelection | None, explicit: bool = True):
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

    if selection is None or not selection.rows or getattr(args, "kernel", None) or not explicit:
        return contextlib.nullcontext()
    document, rows = selection.document, selection.rows
    path = document.path_to(rows[0].kernel)
    target = path[0].parent if path else rows[0].kernel
    regime = {key: group for key, group in document.target_rows().items() if _regime(dict(key[1])) == _regime(rows[0].pins)}
    ordered = _targets_in_build_order(document, regime, of=document.kernel(target).traced)
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
        twin.realization = name
        narrower = resolve_golden_arg(twin, pins=pins, explicit=False, document=document)
        graph = narrower.graph.copy()
        _stage, tail = _replay_stage_and_passes(graph, embedded_golden=True)
        with settled_layouts(scope.settled()), evidence_scope(narrower.scope), unpinned_decisions():
            compiled = Pipeline.build(tail).run(graph, db=db)
        try:
            scope.record(name.split(".", 1)[0], graph_layouts(compiled))
        except LayoutConflict as exc:
            raise CommandError(f"{name}: {exc}") from exc
    if scope.layouts:
        logger.info("%s compiles with %s", args.realization, scope.describe())
    return settled_layouts(scope.settled())


def _records_evidence(args, *, timed: bool = True) -> bool:
    """Whether this run's bench writes tune-DB evidence: not opted out (``--no-record-evidence``) and, for a timing
    (``timed``), benched at the tune bench standard. A run below the standard is a quick look, whose median could
    displace a recorded row. A ``bench_fail`` row is no timing."""
    from emmy.compiler.pipeline.search.bench_record import meets_quality_bar  # noqa: PLC0415

    return not getattr(args, "no_record_evidence", False) and (not timed or meets_quality_bar(args.warmup, args.iters))


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
    disqualifies that arm instead of electing the same route and hanging again (:func:`_records_evidence`)."""
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
    measurement quality; a schedule benched again takes the new numbers
    (``SearchDB.record_perf_row``). A clean row is an ``ok`` row with NO integrity flag — a flagged ok row
    measured something untrue (a wrong answer). ``decisions`` are the kernel-set
    decisions the greedy compile took; once the greedy re-bench is recorded they become ``routing``
    rows, each only when this bench measured all its pieces — a piece that is itself cut again counts
    as measured when the decision that cuts it is stored too. So a fresh nested cut (a cut one of whose
    pieces is cut again), whose piece cut again never runs as a kernel, is priced at the next compile,
    and a strict ``--record-greedy`` can take it with no pinned decisions or schedules."""
    if greedy_iso is None and not golden_benches:
        return
    if not _records_evidence(args):
        _note_below_standard(args)
        return
    from emmy.commands.compile import resolve_tune_db  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import record_bench_perf, record_bench_routing  # noqa: PLC0415

    benched = ([greedy_iso] if greedy_iso is not None else []) + list(golden_benches or [])
    clean = [gb for gb in benched if gb.status == "ok" and not gb.flags]
    if not clean:
        return
    db_path = resolve_tune_db()
    n_perf = 0
    routing = None
    for gb in clean:
        if gb.graph is not None and gb.bench is not None:
            with pinned_knobs(gb.sample.replay_knobs()):
                n_gb = record_bench_perf(db_path, Context.probe(), gb.graph, gb.bench)
            if gb is greedy_iso and n_gb and decisions:
                routing = record_bench_routing(db_path, gb.graph, decisions)
            n_perf += n_gb
    print(f"[record-evidence] {n_perf} kernel perf row(s) recorded into {db_path} — opt out with --no-record-evidence")
    if routing is not None:
        stored, refused = routing
        print(f"[record-evidence] {stored} of {stored + len(refused)} kernel-set decision(s) stored as routing rows")
        for reason in refused:
            print(f"[record-evidence] routing row not recorded — {reason}")


def _dump_bench_compare(dump_dir, results: dict, warmup: int, iters: int) -> None:
    """Persist the eager / torch.compile / emmy comparison table, the file ``emmy compare`` diffs between two dumps."""
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
    legitimate big runs: the child pays a first compile of EVERY kernel (the in-child
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


@dataclass(frozen=True)
class _GoldenBench:
    """One pinned row compiled and benched under its own pins this run. ``flags`` are the integrity-gate verdicts
    (empty = clean): the output-correctness check against the greedy or eager reference. ``bench`` is ``None`` for a
    row that never benched, and ``status`` says why: ``"pin_unmatched"`` (the pinned config did not realize, so
    benching would measure the planner's own pick under the pin's name), ``"bench_fail"`` or ``"compile_timeout"``
    (the compile or the bench of the pinned config failed); an ``"ok"`` row carries a bench."""

    sample: PinnedRow
    graph: object
    bench: object
    flags: list = field(default_factory=list)
    status: str = "ok"
    correctness: dict | None = None


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
    return _lane(golden_bench.sample.replay_knobs())


def env_pin_refusal(
    kernel_knobs: list[dict],
    placement_knobs: list[dict] | None = None,
    kernel_names: list[str] | None = None,
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
    kernel_knobs: list[dict], accuracy_error: str | None, kernel_names: list[str] | None = None, *, benches=()
) -> str | None:
    """A flagged comparison, rejected answer, or unrealized env pin cannot become recorded evidence."""
    flags = [flag for row in benches if row is not None for flag in row.flags or []]
    if flags:
        return "; ".join(flags)
    if accuracy_error is not None:
        return f"it failed the strict accuracy check: {accuracy_error}"
    return env_pin_refusal(kernel_knobs, kernel_names=kernel_names)


def _cuda_knob_dicts(graph) -> list[dict]:
    """Raw ``op.knobs`` per ``CudaOp`` of a compiled pinned graph, in launch order —
    the realized side of the pin gate."""
    return [dict(n.op.knobs or {}) for n in _launch_order_cuda_nodes(graph)]


def _cuda_kernel_names(graph) -> list[str]:
    """Each launched kernel's name, in launch order beside ``_cuda_knob_dicts``."""
    return [n.op.kernel_name for n in _launch_order_cuda_nodes(graph)]


def _placement_knob_dicts(graph) -> list[dict]:
    """Placement receipts from the final greedy resolution of ``graph``."""
    return list(graph.hints.get(PLACEMENT_DECISIONS_HINT, []))


def _applied_place_pins(graph) -> dict[str, str]:
    """Piece-qualified PLACE source receipts from the final cut resolution."""
    return dict(graph.hints.get(PLACEMENT_APPLIED_PINS_HINT, {}))


def _hand_row(name: str, knobs: dict, dynamic=None, route=None) -> PinnedRow:
    """A hand row (``--ab``, ``--tune``): its BOOL knobs are input pins, the rest its schedule. ``dynamic`` is the
    run's own ``--dynamic`` specs, so the row's trace builds the same symbolic graph as the greedy run; ``route`` is
    the kernel-set decisions pinned on the greedy compile, so a row differs from it only by its own knobs."""
    from emmy.compiler.pipeline.knob import KnobType, family_of, get  # noqa: PLC0415

    pins = {key: value for key, value in knobs.items() if (knob := get(family_of(key))) is not None and knob.type is KnobType.BOOL}
    schedule = {key: value for key, value in knobs.items() if key not in pins}
    return PinnedRow(name=name, knobs=schedule, pins=pins, route=dict(route or {}), dynamic=tuple(dynamic) if dynamic else None)


def _ab_samples(specs, dynamic=None, route=None) -> list[PinnedRow]:
    """One hand row per ``--ab "K1=V1,K2=V2"`` spec (:func:`_hand_row`)."""
    from emmy.compiler.pipeline.knob import parse_knob_spec  # noqa: PLC0415

    return [_hand_row(f"ab {raw}", parse_knob_spec(raw), dynamic, route) for raw in specs]


def _failed_bench_status(exc: BaseException) -> str:
    """The status a raised bench earns. A compile-budget overrun measured nothing about the
    kernel, so it gets its own status, which no recorder files as a failure — recording it would put
    a false negative into the very rows the prior trains on. Anything else compiled and then failed,
    which IS evidence about the config, and stays the honest ``bench_fail`` it has always been."""
    from emmy.compiler.backend.cuda.program import compile_budget_overrun  # noqa: PLC0415

    return "compile_timeout" if compile_budget_overrun(exc) else "bench_fail"


def _compile_pinned_row(compiling: tuple, index: int):
    """Row ``index`` of ``compiling`` (the backend, each row's fresh graph beside its pins, and the passes left to run)
    lowered under its own pins: the compiled graph, or the text of the error that stopped its trace or its lowering.
    The pins are published to this process's environment for this one compile."""
    backend, rows, passes = compiling
    graph, knobs = rows[index]
    if isinstance(graph, str):
        return graph
    try:
        with pinned_knobs(knobs):
            if passes is None:
                return backend.compile(graph)
            from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
            from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

            return Pipeline.build(passes).run(graph, db=SearchDB.for_compile(backend.tune_db) if backend.tune_db is not None else None)
    except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own bench table
        return str(exc)


def _compile_pinned_rows(backend, source, samples, quantize=None, passes=None) -> list:
    """Each sample compiled under its own pins, in order: the compiled graph, or the text of the error that stopped
    its trace or its lowering. Each row gets a **fresh** graph, traced here, in this process — a frontend graph
    can't be re-compiled (the first lowering mutates it in place, so a reused graph would yield the first config's
    kernel every time). A sample carrying ``dynamic`` specs (an ``--ab`` row of a ``--dynamic`` run) re-traces
    symbolically, so the pinned kernel is the same masked-tile artifact the greedy run deployed and benches at the
    same hint. ``passes`` is the tail an IR file's graph still has to run; a traced or embedded program takes the
    whole pipeline. The lowerings are the CPU work of a pinned bench, and no row's reads another's, so they run side
    by side on up to ``EMMY_WORKERS`` forked processes: a worker holds one row's pins in its own environment, and
    the rows are the same at any count."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import forked  # noqa: PLC0415
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs  # noqa: PLC0415

    rows = []
    for sample in samples:
        replay_knobs = sample.replay_knobs()
        # String sources must be retraced at the recorded symbolic shape. An
        # embedded stable program already carries its symbolic dimensions.
        dyn = sample.dynamic if isinstance(source, str) else None
        try:
            dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs(list(dyn))) if dyn else None
            with pinned_knobs(replay_knobs):
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
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own bench table
            graph = str(exc)
        rows.append((graph, replay_knobs))
    return forked(_compile_pinned_row, range(len(rows)), config.workers(), shared=(backend, rows, passes))


async def _bench_golden_variants(
    backend,
    source,
    rows,
    *,
    warmup,
    iters,
    passes=None,
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
    row beside the greedy pick. ``rows`` are golden rows (``compile.golden_row``) or the hand rows of
    ``--ab`` / ``--tune`` (:func:`_hand_row`). Both are
    MEASURED under a hand pin published to the environment for that one compile: the pin forces
    the row so its latency can be taken; deploying it is the evidence pick's business once the
    measurement is a row in the tune DB (:func:`_record_bench_evidence`). A config that fails to
    compile / bench is kept as a ``bench_fail`` row (never dropped — the table / ``--json`` show why).

    Compiling (trace + lowering) is CPU work, done for every row before the first bench
    (:func:`_compile_pinned_rows`); every GPU touch is one job on the
    backend's persistent SIGKILL-able worker (``bench_pinned_async``), one row after another in the
    rows' order — the same path the
    greedy row measured through, so all rows share one timing semantics, and a pinned
    config that hangs dies with the child: the row records ``bench_fail``, the parent's
    CUDA context stays clean, and the next row's job respawns a fresh child.

    **A pin matching no offered row fails the row loudly, before any GPU time**: the
    realized-vs-pinned check (:func:`unreproducible_pin_flag`) runs right after the pinned
    compile, and a miss marks the row ``pin_unmatched`` / NOT benched. Benching the fallback
    realization would measure the planner's own pick under the pin's name — the silent-degrade
    class that misled the hd256 flash sweep (a misspelled pin read as a form refusal).

    One more integrity gate flags (never drops) each benched row when ``ref`` carries reference inputs and
    outputs: an output-correctness check. The default check compares against the greedy
    Emmy output with :func:`wrong_answer_flag`; ``strict_correctness`` compares every pinned
    row directly against ``strict_reference`` under rtol=atol=1e-3 and records error statistics;
    ``computed_from`` (``computed_from_formats`` of the frontend program) sets each float output's
    rounding step there, since the compiled ``source`` no longer holds the values it is computed from.
    Flags render as a ``!`` marker in the table and ride the ``--json`` record."""
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
    samples = list(rows or [])
    for sample, g_compiled in zip(samples, _compile_pinned_rows(backend, source, samples, quantize, passes), strict=True):
        replay_knobs = sample.replay_knobs()
        # ``unverified`` rides every row of a session with no reference outputs, so a measurement
        # taken without an output check is never mistaken for a checked one.
        flags = [unverified] if unverified else []
        if isinstance(g_compiled, str):
            logger.warning("[golden] %s: compile of the pinned config failed (%s) — row kept as bench_fail", sample.name, g_compiled)
            out.append(_GoldenBench(sample, None, None, [f"compile failed: {g_compiled}"], "bench_fail"))
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
        run_outputs = comparison_outputs(run_outputs, g_compiled) if run_outputs is not None else None
        correctness = None
        if run_outputs is not None and ref_outputs is not None:
            if strict_correctness:
                correctness = strict_correctness_proof(
                    run_outputs,
                    ref_outputs,
                    reference=strict_reference,
                    exact_out=ref_exact[0] if ref_exact else None,
                    formats=output_formats(run_outputs, g_compiled, computed_from),
                )
                if correctness["status"] != "pass":
                    flags.append(f"strict {strict_reference} correctness failed: {correctness.get('error', 'tolerance exceeded')}")
            elif not strict_correctness:
                # Held back until every row is in: whether this verdict means anything depends on
                # whether the reference reproduces, which only the whole set can answer.
                wrong_answer[len(out)] = wrong_answer_flag(run_outputs, ref_outputs, quantized_outputs(run_outputs, g_compiled))
                realized[len(out)] = _cuda_knob_dicts(g_compiled)
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


async def _tune_kernel_sets(backend, args, program, ctx, db, found) -> None:
    """A tune of a program of several kernels (``search/autotune``): in rounds, tune each kernel of the kernel set the
    greedy compile takes (``found``, of ``program()``) alone, then the kernels of each arm a fork of that compile
    offered and did not take, the placement prior's best ``ARMS`` of them a fork, taken on the fork's kernel alone;
    every decision whose pieces now hold rows is stored as a routing row. A kernel of the compiled set gets
    ``--tune N`` rows; a kernel only an untaken arm leaves gets the prior's ``START`` best, enough to price the arm,
    and the rest once a compile takes it. The next compile then picks each of those
    forks from measured arms, and a round on its kernel set follows while that set changes, ``ROUNDS`` at most. A
    kernel alone is a program of one kernel, tuned as any: its rows are benched, checked and recorded by a bench
    session of its own."""
    from emmy.compiler.pipeline.search.autotune import ARMS, ROUNDS, START, kernel_set  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import knobs_json  # noqa: PLC0415
    from emmy.compiler.pipeline.search.inventory import record_measured_routing  # noqa: PLC0415
    from emmy.compiler.wire import kernel_bindings, kernel_identity, kernel_name, kernel_tile  # noqa: PLC0415

    def key_of(op) -> tuple:
        return kernel_identity(op), kernel_bindings(kernel_tile(op))

    def price(kernels) -> float | None:
        times = [db.best_per_op_time(ctx, kernel, bindings=bindings) for kernel, bindings in map(key_of, kernels)]
        return None if None in times else sum(times)

    tuned: dict = {}
    explored: set = set()

    async def measure(kernels, decisions, rows: int) -> None:
        tune = argparse.Namespace(**{**vars(args), "tune": rows})
        for op, alone in kernels:
            kernel, bindings = key_of(op)
            if tuned.get((kernel, knobs_json(bindings)), 0) >= rows:
                continue
            tuned[kernel, knobs_json(bindings)] = rows
            if len(db.measured(ctx, kernel, bindings=bindings)) >= rows:
                continue  # an earlier tune's rows
            try:
                compiled = backend.compile(alone.copy())
                session = await _bench_session(
                    tune,
                    backend,
                    compiled,
                    torch_spec=("frontend_graph", None),
                    base_s=_IR_COMPARE_BASE_S,
                    twin=None,
                    source=alone,
                    rows=[],
                    keep_worker=True,
                )
            except Exception as exc:  # noqa: BLE001 — a kernel that does not lower leaves its arm without a price
                logger.warning("[tune] %s: not measured (%s)", kernel_name(op), exc)
                continue
            _record_bench_evidence(tune, session.rows, session.isolated)
        measured = {kernel for kernel, bindings in (key_of(op) for op, _ in kernels) if db.measured(ctx, kernel, bindings=bindings)}
        record_measured_routing(db, decisions, measured)

    last = None
    for _ in range(ROUNDS):
        kernels = sorted(str(kernel_identity(op)) for op, _ in found.kernels)
        if kernels == last:
            break
        last = kernels
        await measure(found.kernels, found.decisions, args.tune)
        logger.warning("[tune] kernel set of %d kernel(s): %s us", len(kernels), price(op for op, _ in found.kernels))
        for kernel, alone, arms, taken in found.forks:
            for arm in [arm for arm in arms if arm != taken][:ARMS]:
                if (kernel, knobs_json(arm)) in explored:
                    continue
                explored.add((kernel, knobs_json(arm)))
                try:
                    other = kernel_set(alone.copy(), ctx, db, take=arm)
                except Exception as exc:  # noqa: BLE001 — an arm that does not lower has no price
                    logger.warning("[tune] arm %s: not measured (%s)", arm, exc)
                    continue
                await measure(other.kernels, other.decisions, min(args.tune, START))
                logger.warning("[tune] arm %s: %s us", arm, price(op for op, _ in other.kernels))
        found = kernel_set(program(), ctx, db)
    logger.warning("[tune] the compile now takes %d kernel(s): %s us", len(found.kernels), price(op for op, _ in found.kernels))
    for op, _ in found.kernels:
        logger.warning("[tune]   %s %s us", kernel_name(op), price([op]))


async def _tune(backend, args, source, compiled, ref, ref_key, route=None) -> list:
    """``--tune N``: measure N schedule rows of the program's one scheduled kernel, in batches the autotuner
    proposes from the rows measured so far, each benched exactly as an ``--ab`` row. A row that failed, did not
    realize, or was flagged (a wrong answer among them) counts as a failure, never as a time. A program of several
    kernels, or one whose kernel met a kernel-set fork of several arms, is tuned kernel by kernel
    (:func:`_tune_kernel_sets`) and adds no row to this session's table."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.autotune import Autotuner, kernel_set, prior_scores, schedule_space  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    def program():
        return source.copy() if not isinstance(source, str) else graph_from_code(source)[0]

    ctx = Context.probe()
    db = SearchDB.for_compile(backend.tune_db) if backend.tune_db is not None else None
    found = kernel_set(program(), ctx, db)
    if len(found.kernels) != 1 or any(len(arms) > 1 for _, _, arms, _ in found.forks):
        if db is None or not _records_evidence(args):
            raise CommandError("--tune of a program of several kernels picks from the rows it records: it needs the tune DB")
        await _tune_kernel_sets(backend, args, program, ctx, db, found)
        return []
    op, rows = schedule_space(program(), ctx, db)
    tuner = Autotuner(rows, prior_scores(op, rows, ctx))
    logger.warning("[tune] %s: %d schedule rows; measuring %d", op.name, len(rows), args.tune)
    ref_knobs = _cuda_knob_dicts(compiled)
    out: list = []
    while len(tuner.us) < args.tune and (batch := tuner.propose(min(8, args.tune - len(tuner.us)))):
        samples = [
            _hand_row(
                "tune " + ",".join(f"{k}={v}" for k, v in rows[i].items()),
                {str(k): str(v) for k, v in rows[i].items()},
                getattr(args, "dynamic", None),
                route,
            )
            for i in batch
        ]
        benches = await _bench_golden_variants(
            backend, source, samples, warmup=args.warmup, iters=args.iters, ref=ref, ref_key=ref_key, ref_knobs=ref_knobs
        )
        tuner.observe(
            (i, gb.bench.total_us if gb.status == "ok" and not gb.flags and gb.bench.total_us else math.inf)
            for i, gb in zip(batch, benches, strict=True)
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
    sample = PinnedRow(name="greedy (isolated)")
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
    from emmy.compiler.ir.expression import Var  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415
    from emmy.compiler.pipeline.search.dataset import ShapeKey  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import stamps  # noqa: PLC0415

    cuda_nodes = _launch_order_cuda_nodes(graph)
    if not cuda_nodes:
        return

    # Per-kernel best-case (min) to match the min-based comparison table.
    times_by_idx = {} if bench is None else bench.launch_us()
    total_us = None if bench is None else bench.launch_sum_us
    attrs_by_kname = _collect_kernel_attrs(graph)
    occ_limits = _occupancy_limits()

    # Symbolic grids (ceil-div over a dynamic axis) need a concrete env to
    # resolve for display — use each symbolic input dim's hint, so the printed
    # geometry reflects the hint-sized tile the kernel was tuned for.
    if sym_env is None:
        sym_env = {}
        for nid in graph.inputs:
            for dim in graph.buffer(nid).shape:
                if isinstance(dim.expression, Var) and dim.hint is not None:
                    sym_env.setdefault(dim.expression.name, dim.hint)

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
            if gb.sample.record is not None:
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
        flagged row (unmatched pin / bench_fail / wrong-answer)
        gets a ``!`` marker; an unbenched row (``gb.bench is None``) prints its
        kernels with ``--`` timings (a compile failure has no kernels — one bare
        label row keeps the failure visible)."""
        label = f"golden {gb.sample.name}" if gb.sample.record is not None else gb.sample.name
        label = f"{label} [{_pinned_lane(gb)}]"  # so an [fm]-vs-std A/B can't be misread
        if gb.flags:
            label = f"! {label}"
        if gb.graph is None:
            records.append((label, None, "--", (0, 0, 0.0, 0, "--", "  --"), {}, ref))
            return
        g_times = {} if gb.bench is None else gb.bench.launch_us()
        g_attrs = {} if gb.bench is None else _collect_kernel_attrs(gb.graph)
        g_nodes = _launch_order_cuda_nodes(gb.graph)
        for gidx, gnode in enumerate(g_nodes):
            gk = dict(tuning_knob_items(gnode.op.knobs or {}))
            records.append(
                (label, g_times.get(gidx) if gb.bench is None else g_times.get(gidx, 0.0), "--", _geom(gnode.op, g_attrs), gk, ref)
            )

    # Per-launch times of the isolated greedy re-bench — same graph, so indexes align
    # 1:1 with ``cuda_nodes`` and each greedy kernel row gets its iso twin beneath it.
    iso_times = None if greedy_iso is None or greedy_iso.bench is None else greedy_iso.bench.launch_us()
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
    table parsers) and where the wrong-answer verdicts become fields —
    the confirm-twice rule diffs two of these files instead of two terminal scrollbacks. The
    Emmy row always carries a ``correctness`` verdict, including the two that used to be an
    absence: a non-strict check that FAILED, and a target with no eager reference to check
    against (:func:`_emmy_correctness`; ``checked`` says whether that check ran, whichever
    columns were timed). The torch.compile row carries its own: ``tc_checked`` is false on a
    quantized twin, where the column is timed unchecked (:func:`quantized_twin`).

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
        times = {} if b is None else b.launch_us()
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
        return {
            "total_us": b.total_us,
            "timing_semantics": b.timing_semantics,
            "captured": b.captured,
            "num_launches": b.num_launches or len(b.per_launch or []),
        }

    def _e2e_us(b) -> float | None:
        return None if b is None or b.e2e_min_ms is None else b.e2e_min_ms * 1000

    pinned = []
    for gb in golden_benches or []:
        sample = gb.sample
        measured = sample.record.measurements if sample.record is not None else None
        pinned.append(
            {
                "name": sample.name,
                "kind": "golden" if sample.record is not None else "ab",
                "lane": _pinned_lane(gb),
                "status": gb.status,
                "pinned_knobs": {k: str(v) for k, v in sample.replay_knobs().items()},
                **_timing(gb.bench),
                "e2e_us": _e2e_us(gb.bench),
                "kernels": _kernel_rows(gb.graph, gb.bench),
                "flags": list(gb.flags),
                "correctness": gb.correctness,
                "recorded_emmy_us": measured.emmy_us if measured is not None else None,
                "recorded_ref_us": measured.reference_us if measured is not None else None,
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
    captured = bench is not None and bench.captured
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


def _detect_stage(graph) -> str:
    """Identify the IR stage by scanning op type names. Returns one of
    ``torch | tensor | loop | tile | kernel | ptx`` — the highest-stage
    op present in the graph wins, since lowering produces a graph mixed
    only briefly during a pass and stable in the post-pass form."""
    stage_by_op: dict[str, str] = {
        "CudaOp": "ptx",
        "KernelOp": "kernel",
        "TileOp": "tile",
        "LoopOp": "loop",
    }
    order = ["torch", "tensor", "loop", "tile", "kernel", "ptx"]
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
    from emmy.commands.compile import _IR_STAGES  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES  # noqa: PLC0415

    completed, _ = _IR_STAGES[stage]
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


def _strict_benchmark_errors(
    args,
    results,
    bench,
    captured,
    correctness,
    pinned_rows,
    *,
    golden_rows=0,
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
    for backend in resolve_backends(args.bench_backends):
        if backend != "emmy" and not frontend_runnable:
            continue  # an embedded Loop target has no Torch twin; its reference is the same-input greedy replay
        name = display_names[backend]
        latency = (results or {}).get(name)
        if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
            errors.append(f"requested backend {name} has no positive latency")
    if captured is not True:
        errors.append("backend comparison did not use CUDA graph capture")
    if bench is None or not bench.captured:
        errors.append("deployed Emmy timing was not captured")
    if not frontend_runnable and not same_input_reference:
        errors.append("same-input greedy reference is unavailable")
    if not valid_proof(correctness):
        errors.append(f"deployed Emmy row lacks strict {reference} correctness")

    pinned_rows = list(pinned_rows or [])
    benched_golden = sum(row.sample.record is not None for row in pinned_rows)
    if benched_golden != golden_rows:
        errors.append(f"expected {golden_rows} exact working-golden row(s), got {benched_golden}")
    benched_ab = len(pinned_rows) - benched_golden
    if benched_ab != len(args.ab or []):
        errors.append(f"expected {len(args.ab or [])} exact --ab row(s), got {benched_ab}")
    for row in pinned_rows:
        label = row.sample.name
        if row.status != "ok" or row.flags:
            errors.append(f"{label} failed exact-pin integrity: status={row.status}, flags={list(row.flags)}")
            continue
        kernels = list(_launch_order_cuda_nodes(row.graph)) if row.graph is not None else []
        if not kernels:
            errors.append(f"{label} has no generated kernel inventory")
        if row.bench is None or not row.bench.captured:
            errors.append(f"{label} timing was not captured")
        else:
            num_launches = row.bench.num_launches or len(row.bench.per_launch or [])
            if num_launches <= 0:
                errors.append(f"{label} has no measured launches")
            if num_launches > 1 and row.bench.e2e_min_ms is None:
                errors.append(f"{label} multi-launch timing is not whole-program end-to-end")
            if row.bench.total_us <= 0:
                errors.append(f"{label} has no positive whole-program timing")
        if not valid_proof(row.correctness):
            errors.append(f"{label} lacks strict {reference} correctness")

    return errors


def _handle_run_ir(args, CudaBackend, CompilerDump, selection: GoldenSelection | None = None, on_compiled=None) -> int:
    """Run path of an IR file (any stage) or a golden selection: finish the lowering, execute, bench. The exit code."""
    import json

    from emmy.commands.compile import selected_decisions  # noqa: PLC0415
    from emmy.compiler.backend import torch_ref
    from emmy.compiler.graph import Graph
    from emmy.compiler.pipeline import Pipeline
    from emmy.compiler.pipeline.search.golden import evidence_scope  # noqa: PLC0415
    from emmy.compiler.pipeline.search.inventory import KernelInventory  # noqa: PLC0415

    strict_correctness = bool(getattr(args, "strict_correctness", False))
    embedded = selection.graph if selection is not None else None
    scope = selection.scope if selection is not None else None
    if (paged := getattr(args, "paged", None)) and embedded is not None:
        # A paged input, one page per row of its leading axis, or a paged weight, one page, as a serving boot below
        # its device budget declares it (``gen_runner._compile_split``, ``vision_tower.page_projections``): every
        # compile of this program — the greedy one and each tune or --ab row — offers only the schedules a paged
        # operand can take. The runtime allocates the pages and fills them from the input or the weight, so the bench
        # times the paged schedules on resident pages.
        weights = {op.name: op.source_shape for _, op in embedded.loadable_constants()}  # by the name the trace gave them
        embedded.hints.set("cuda.paged_buffers", tuple((name, 0, weights[name][0] if name in weights else 1) for name in paged))
    if embedded is None:
        with open(args.ir) as f:
            graph = Graph.from_dict(json.load(f))
    else:
        graph = embedded.copy()
    if getattr(args, "dynamic", None):
        raise CommandError("--dynamic is incompatible with an IR file or a golden's stored program (the trace is already complete)")

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
    reference = graph if torch_ref.is_runnable(graph) else (selection.reference if selection is not None else None)
    frontend = reference.copy() if reference is not None and torch_ref.is_runnable(reference) else None
    # What a pinned row lowers again: the golden's embedded program, or the IR file's graph as it stood before its tail.
    source = embedded if embedded is not None else graph.copy() if tail and args.ab else None

    backend = CudaBackend(debug=args.debug or None, dump=dump, tune_db="auto")
    db = None
    if backend.tune_db is not None:
        from emmy.compiler.pipeline.search.db import SearchDB

        db = SearchDB.for_compile(backend.tune_db)
        logger.info("Using tuning DB: %s", backend.tune_db)

    # Every kernel-set decision the greedy compile takes, as the splice watcher reports it: the tile
    # kernel the fork was offered on, the arm, and the pieces it minted.
    taken: list[tuple[object, dict, list]] = []
    watcher = KernelInventory(on_routing=lambda parent, arm, pieces, _ids: taken.append((parent, arm, pieces)))
    route = selected_decisions(args, selection)
    if tail:
        # Finish the tail lowering — the greedy compile — with the selected golden's records as its
        # golden evidence, their shared input regime published (by the caller) and, under
        # ``--pin-route``, the kernel-set decisions the named realization records pinned, as
        # ``compile`` does. The same compile captures its kernel-set decisions: the bench-evidence
        # recording (``_record_bench_evidence``) stores them beside the perf rows, and
        # ``--record-greedy`` writes them into the golden.
        pipeline = Pipeline.build(tail).with_strategies(watcher)
        with evidence_scope(scope), pinned_knobs(route):
            graph = pipeline.run(graph, db=db, dump=dump)
    if on_compiled is not None:
        on_compiled(graph)  # the golden target walk records the weight layouts this compile bound

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
        return 0

    if args.ab and not tail:
        logger.warning("--ab ignored: %s IR is fully lowered (no forks left to pin)", stage)
    rows = list(selection.pinned) if selection is not None else []
    if args.ab:
        rows += _ab_samples(args.ab, dynamic=getattr(args, "dynamic", None), route=route)
    # An exact Loop target has no Torch twin: under --strict the greedy outputs are its reference.
    same_input_greedy = strict_correctness and embedded is not None and frontend is None
    # Golden write-back and default bench-to-DB evidence both need isolated per-kernel timings,
    # including a fresh file walk with no measured row to compare as a pinned schedule.
    measure_greedy = bool(getattr(args, "record_greedy", False) or (embedded is not None and _records_evidence(args)))
    # Pinned rows compile under the golden scope the greedy row did — the file named, not the card's whole corpus.
    with evidence_scope(scope):
        session = asyncio.run(
            _bench_session(
                args,
                backend,
                graph,
                torch_spec=("frontend_graph", frontend),
                base_s=_IR_COMPARE_BASE_S,
                twin=frontend if "tcompile" in resolve_backends(args.bench_backends) else None,
                source=source if tail else None,
                rows=rows,
                passes=None if embedded is not None else tail,
                route=route,
                torch_twin=frontend is not None,
                same_input_greedy=same_input_greedy,
                measure_greedy=measure_greedy,
                computed_from=computed_from_formats(frontend) if frontend is not None else None,
                sym_env=collect_sym_env(([frontend] if frontend is not None else []) + [graph]),
            )
        )
    if embedded is not None and tail and not rows and session.greedy_fail is None and not measure_greedy:
        _note_below_standard(args)
    return _report(
        args,
        graph,
        session,
        dump=dump,
        accuracy_fatal=False,
        frontend_runnable=frontend is not None,
        tc_checked=not quantized_twin(frontend),
        golden_rows=len(selection.pinned) if selection is not None else 0,
        # An embedded golden lowers in-process, knobs and all; only the ``--ir`` JSON path, whose
        # serialization drops them, stays unrecorded.
        decisions=taken if embedded is not None else None,
    )


def _symbolic_bench_note(sym_env: dict[str, int]) -> str | None:
    """The full-model table's shape label for a symbolic graph (``None`` for
    static): both sides bench at the hint, and the reader should know the
    number is hint-shaped, not trace-shaped."""
    if not sym_env:
        return None
    dims = ", ".join(f"{name}={size}" for name, size in sorted(sym_env.items()))
    return f"benched at {dims} (symbolic hint; torch inputs tiled to match)"


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
