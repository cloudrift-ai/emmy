"""``emmy eval <prior|golden>`` — evaluate the prior's ranking and a golden file's serving envelope.

- ``eval prior``     — how well the prior RANKS, over a dataset ``emmy db export`` wrote: its golden
  pools (``--pools golden``: the golden-rank screen, plus the greedy pipeline pick vs golden) or its
  measured pools (``--pools measured``: Spearman + regret, what a wrong pick costs).
  The summaries are assembled by ``search/prior/report.py`` and rendered here; ``emmy fit``
  writes the same summaries into its ``metrics.json``, so a fit and an eval state the golden
  screen with one implementation rather than two that agree by coincidence.
- ``eval golden``    — validate one canonical golden file against the pinned serving
  configuration and live GPU, then reproduce its rows and audit the exact serving matrix.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

from emmy import config, storage
from emmy.commands.eval_args import add_dataset_args, resolve_offline_arg
from emmy.commands.table import GREEN as _GREEN
from emmy.commands.table import RED as _RED
from emmy.commands.table import YELLOW as _YELLOW
from emmy.commands.table import Col, col_widths, knob_columns, render_table
from emmy.compiler.pipeline.search.prior import report as report_mod

logger = logging.getLogger(__name__)


def _format_pins(pins) -> str:
    def spell(value) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    return ", ".join(f"{name}={spell(value)}" for name, value in pins) or "unpinned"


def _realization_label(name: str, pins) -> str:
    return name if dict(pins) == {"FAST_MATH": False} else f"{name} [{_format_pins(pins)}]"


def register_eval_command(subparsers) -> None:
    """``emmy eval <prior|golden>`` — evaluate the prior's ranking or a golden file."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate how well the prior ranks candidate pools, or a golden file's serving envelope",
    )
    sub = parser.add_subparsers(dest="eval_target", required=True)

    pp = sub.add_parser(
        "prior",
        help="Report how well the prior ranks the pools of an exported dataset — golden (default) or measured",
    )
    pp.add_argument(
        "--offline-file",
        "--analytic-file",  # pre-rename spelling
        dest="offline_file",
        help="Offline weights artifact (JSON) to score with, for A/Bing candidate fits. "
        "Default: EMMY_OFFLINE_FILE or the repo-checked prior/weights/offline.json.",
    )
    add_dataset_args(pp)
    pp.add_argument("--json", dest="json_out", metavar="PATH", help="Also write the report as JSON, for diffing two runs.")
    pp.set_defaults(func=handle_eval_prior)

    pg = sub.add_parser(
        "golden",
        help="Validate one golden file against its pinned serving configuration and the live target GPU",
    )
    pg.add_argument("--golden", required=True, metavar="PATH", help="The exact canonical golden file to validate.")
    pg.add_argument(
        "--serving-config",
        required=True,
        metavar="PATH",
        help="Pinned release env that names the model, GPU, golden file, and reachable realization matrix.",
    )
    pg.set_defaults(func=handle_eval_golden)


def _prior_halves(space: str):
    """The priors the report labels — one today, the offline model of the dataset's ``space``: the
    ``--offline-file`` override, else the shipped weights of that space. Fails the command up front on an
    unloadable artifact or one fit for the other space — the per-shape eval harness catches exceptions into
    ERR rows, which would let a broken A/B exit 0."""
    from emmy.compiler.pipeline.search.prior import OfflinePrior  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior.offline import default_file  # noqa: PLC0415

    try:
        prior = OfflinePrior(path=None if config.offline_path() else str(default_file(space)))
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    if prior.space != space:
        logger.error("the weights rank the %s space; the dataset is the %s space", prior.space, space)
        sys.exit(2)
    return [("offline", prior)]


def _measured_report(args, halves, dataset, source: str):
    """``eval prior --pools measured`` — the report over the dataset's measured pools: every benched row of the DB
    the export read, grouped by ``db/export.measured_groups`` (one op, one card, one regime). The header names the
    sources the rows came from: two reports are comparable only when computed over the same rows, and a freeze's
    digest in the source name is what says so.

    ``--kernel`` matches the op LABEL, since a pool's own op identity is a digest with nothing readable in it. The
    label is a function of the ``S_*`` stamps every row of one pool shares — read off the pool's first row — so a
    filter keeps or drops a whole pool atomically."""
    from emmy.compiler.pipeline.search.dataset import op_label  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior.report import EvalReport, measured_summaries  # noqa: PLC0415

    groups = dataset.measured
    if args.kernel:
        groups = [g for g in groups if args.kernel in op_label(_stamps(g))]
    header = {
        "dataset": "measured",
        "source": source,
        **{k: v for k, v in dataset.provenance.items() if k != "source"},
        "kernel": args.kernel,
        "rows": sum(len(g.feats) for g in groups),
        "groups": len(groups),
        "dropped": dataset.dropped["measured"],
    }
    return EvalReport(header, [c for half, prior in halves for c in measured_summaries(half, groups, prior.score_rows)])


def _stamps(group) -> dict[str, float]:
    """The ``S_*`` stamps every row of a measured pool shares, read off its first row; the matrix fills an absent
    feature with NaN, which is no stamp."""
    return {k: float(v) for k, v in zip(group.feat_names, group.feats[0], strict=True) if k.startswith("S_") and not math.isnan(v)}


def _golden_report(args, halves, dataset, source: str):
    """``eval prior --pools golden`` — the report over the dataset's golden pools: the rows the golden files record,
    each ranked among the candidates its kernel offers — the groups ``emmy fit`` trains on, over the FULL
    featurization rather than the fit's ``D_*`` view. The view is a property of the model being fitted, and this
    command scores the model class the artifact names: the linear model reads only its own weight names, so its
    ranks are identical either way, while a tree model reads the ``S_*`` / ``H_*`` columns a narrow view drops and would
    otherwise be asked about a kernel with no shape. ``--kernel`` keeps the pools whose kernel's C name contains it
    — a view; each retained pool's rank is unchanged by it."""
    from emmy.compiler.pipeline.search.prior.report import EvalReport, golden_summaries  # noqa: PLC0415

    groups = [g for g in dataset.golden if not args.kernel or args.kernel in g.name]
    header = {
        "dataset": "golden",
        "source": source,
        **{k: v for k, v in dataset.provenance.items() if k != "source"},
        "kernel": args.kernel,
        "groups": len(groups),
        "positives": sum(len(g.golden_ids) for g in groups),
        "skipped": len(dataset.skipped),
        "dropped": dataset.dropped["golden"],
    }
    return EvalReport(header, [c for half, prior in halves for c in golden_summaries(half, groups, prior.score_rows)])


def handle_eval_prior(args) -> None:
    """``eval prior`` — how well the prior ranks a candidate pool, over a dataset ``emmy db export`` wrote.

    Two kinds of pool, two different questions, one report schema (see ``search/prior/report.py``): benched pools
    say what a wrong pick COST, golden pools only say where the known-good row landed. ``--pools golden``
    additionally runs the deploy-faithful check the ranks are a screen for — the greedy pipeline pick vs the golden
    rows, with the deployable -O3 latency of the prior's pick beside it."""
    from emmy.compiler.pipeline.search.dataset import Dataset  # noqa: PLC0415

    resolve_offline_arg(args)
    try:
        dataset = Dataset.load(args.dataset)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(2)
    space = dataset.provenance.get("space", "schedule")
    halves = _prior_halves(space)
    golden = args.pools == "golden"
    report = (_golden_report if golden else _measured_report)(args, halves, dataset, str(args.dataset))
    _emit_report(report)
    if args.json_out:
        storage.write_json(Path(args.json_out), report.to_json(), indent=2)
        logger.info("wrote %s", args.json_out)
    if golden and space == "placement":
        _emit_placement_deploy_check(args, dataset, halves[0][1])
    elif golden:
        _emit_golden_deploy_check(args, [pool for group in dataset.golden for pool in group.pools])


def _emit_placement_deploy_check(args, dataset, prior) -> None:
    """The deploy-faithful half of ``eval prior`` over a placement dataset: each pool's kernel walked through
    the lift and the cut pass again with the prior deciding every placement fork, and the arm it takes at the
    kernel's first fork printed beside the golden's — ``fuse``, or the seams cut. The rank says where the
    golden's arm sat in the offer; this says which arm the prior would actually hand the cut pass."""
    from emmy.compiler.pipeline.search.ranking import placement_decisions, pool_context, walk_placement  # noqa: PLC0415

    pools = list({(pool.gpu, pool.regime, pool.name): pool for group in dataset.golden for pool in group.pools}.values())
    pools = [pool for pool in pools if not args.kernel or args.kernel in pool.kernel.name]
    decisions = placement_decisions(pools)
    rows: list[tuple[str, str, str, bool]] = []
    for pool in pools:
        try:
            forks, _ = walk_placement(pool, pool_context(pool), decisions, scorer=prior.mean_scores_features)
        except ValueError:
            continue
        if not forks or forks[0].pick is None:
            continue
        root = forks[0]
        golden = " | ".join(root.labels[i] for i in root.positives)
        rows.append((pool.name, root.labels[root.pick], golden, root.pick in root.positives))
    logger.info("Golden reproduction — the prior's pick at each kernel's first placement fork vs the golden's arm:")
    width = max((len(name) for name, *_ in rows), default=6)
    logger.info("  %-*s  %-4s  %s", width, "kernel", "ok", "pick -> golden")
    for name, pick, golden, ok in rows:
        logger.info("  %-*s  %-4s  %s -> %s", width, name, "ok" if ok else "MISS", pick, golden)
    logger.info("  TOTAL %d/%d", sum(ok for *_, ok in rows), len(rows))


def _metric(block: dict, key: str, fmt: str) -> str:
    """One metric value with the pools it was computed over, or ``—`` when nothing in the summary qualified.

    The count is appended only where the block carries one, which is where the metric has a size minimum and so
    can cover fewer pools than the summary holds."""
    value = block.get(key)
    if value is None:
        return "—"
    return f"{fmt.format(value)} ({block['groups']})" if "groups" in block else fmt.format(value)


# Per dataset: the axis columns, then ``(header, render(summary))`` for each metric column. The axes are the ones the
# report keyed its summaries on — the renderer names them rather than discovering them, so a column order is a
# decision made here and not a side effect of dict insertion.
_REPORT_TABLES = {
    "db": (
        ["half", "gpu", "H_opt"],
        [
            ("rho", lambda c: _metric(c.metrics["spearman"], "median", "{:+.2f}")),
            ("regret@1", lambda c: _metric(c.metrics["regret1"], "median", "{:.2f}x")),
            ("worst@1", lambda c: _metric(c.metrics["regret1"], "worst", "{:.2f}x")),
            (f"regret@{report_mod.TOPK}", lambda c: _metric(c.metrics[f"regret{report_mod.TOPK}"], "median", "{:.2f}x")),
        ],
    ),
    "golden": (
        ["half", "gpu", "tier", "pool"],
        [
            ("rank", lambda c: _metric(c.metrics["rank"], "median", "{:g}")),
            ("rank(opt)", lambda c: _metric(c.metrics["rank"], "median_optimistic", "{:g}")),
            *((f"top{k}", lambda c, k=k: f"{c.metrics[f'top{k}']['count']}/{c.groups - c.unscored}") for k in (1, 10, 50)),
        ],
    ),
}

_REPORT_CAPTIONS = {
    "db": [
        "ranking quality over benched pools (rho: +1 = the model orders them as the hardware does;",
        "regret: 1.00x = the pick IS the measured best). Each number's (n) is the pools it covers.",
    ],
    "golden": [
        "golden rank — a SCREEN, not a gate: it says where a verified config landed, never what",
        "missing it costs. Only regret over measured pools (--pools measured) measures that.",
    ],
}


def _emit_report(report) -> None:
    """Print an :class:`EvalReport` — the provenance header, then one table of summaries.

    Which columns appear follows the report's dataset, since that is what decided which metrics the summaries carry.
    The ``pools`` column is the summary's own total, annotated when the model could not score some of them: an
    unscored pool is not a small one, and a report that dropped it silently would show a healthy corpus with no
    sign that part of the deploy surface is unmeasured."""
    head = report.header
    logger.info("")
    logger.info("[prior] %s dataset — %s", head.get("dataset", "?"), head.get("source", ""))
    # Every remaining header key, whatever the dataset put there. Printed generically so a builder that starts
    # recording a new provenance field does not also have to teach this about it — a count nobody prints is a
    # count nobody checks.
    provenance = ", ".join(f"{k}={head[k]}" for k in head if k not in ("dataset", "source", "dropped") and head[k] is not None)
    if provenance:
        logger.info("  %s", provenance)
    if dropped := head.get("dropped"):
        logger.info("  rows dropped before grouping: %s", ", ".join(f"{n} {why}" for why, n in sorted(dropped.items())))
    if not report.summaries:
        logger.info("  no candidate pools to score")
        return

    axes, metrics = _REPORT_TABLES[head["dataset"]]
    columns = [Col(a) for a in (*axes, "pools")] + [Col(name) for name, _ in metrics]
    rows = [
        [summary.axes.get(a, "") for a in axes]
        + [str(summary.groups) + (f" ({summary.unscored} unscored)" if summary.unscored else "")]
        + [render(summary) for _, render in metrics]
        for summary in report.summaries
    ]
    logger.info("")
    for line in _REPORT_CAPTIONS[head["dataset"]]:
        logger.info("  %s", line)
    for line in render_table(columns, rows, rule=True, indent="  "):
        logger.info("%s", line)


def _emit_golden_deploy_check(args, pools: list) -> None:
    """The deploy-faithful half of ``eval prior --pools golden``: the greedy tile-lowering pick vs the golden
    rows, per matmul pool of the **live** card (every card's when none is visible). This is what the golden RANK
    is only a screen for — a rank says where the verified row sat in the enumeration, this says what actually
    gets compiled. Scoping to the live GPU keeps the view about the card in hand: two cards' pools of one shape
    would otherwise mix (RTX 5090 / RTX PRO 6000 even share ``compute_cap``).

    Stops at the tile dialect (every knob fork resolves there: no codegen /
    nvcc). One row per pool — the kernel's definition at the pool's sizes, under the regime's pins alone: the
    pick is scored against the pool's *closest* golden row (most knobs reproduced), so several goldens on one
    pool don't duplicate rows. A trailing ``TOTAL`` row carries per-knob match counts over the rows + the
    exactly-reproduced row count. Rows print with column-aligned ``found/golden`` knobs (canonical order)."""
    import logging as _logging  # noqa: PLC0415

    from emmy.compiler.pipeline import TILE_LOWERING, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import METADATA_PREFIXES  # noqa: PLC0415
    from emmy.compiler.pipeline.search.dataset import is_matmul  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import live_gpu_key  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unpinned_decisions  # noqa: PLC0415

    pools = [p for p in pools if p.kernel.formed and is_matmul(p.kernel.stamps) and (not args.kernel or args.kernel in p.kernel.name)]
    if (live := live_gpu_key()) is not None:
        # The live card's pools, or every card's when none are recorded for it — as ``goldens_for_live_gpu`` scopes.
        pools = [p for p in pools if (p.gpu, p.cap) == live] or pools

    def tunable(knobs: dict) -> dict:
        return {k: v for k, v in knobs.items() if not k.startswith(METADATA_PREFIXES)}

    def picked(pool) -> dict:
        with pinned_knobs(pool.pins), unpinned_decisions():
            compiled = Pipeline.build(TILE_LOWERING).run(pool.kernel.program(pool.bindings))  # tile dialect only
        knobs: dict = {}
        for node in compiled.nodes.values():
            k = getattr(node.op, "knobs", None)
            if k:
                knobs.update(k)
        return _bare_families(tunable(knobs))

    logger.info("")
    logger.info("Golden reproduction — greedy pipeline pick vs the golden rows:")
    # Silence the compile chatter so this function's own ``logger`` can stream one clean result line per pool.
    quiet = _logging.getLogger("emmy.compiler")
    prev = quiet.level
    quiet.setLevel(_logging.WARNING)
    n_match = n_rows = 0
    knob_match: dict[str, int] = {}  # rows where the pick matched this knob
    knob_total: dict[str, int] = {}  # rows whose golden carries this knob
    entries: list[tuple] = []  # ("row", lead_cells, gold, got) | ("err", name, message)
    try:
        for pool in pools:
            label = _realization_label(pool.name, pool.pins.items())
            try:
                got = picked(pool)
            except Exception as e:  # noqa: BLE001 — one pool's error shouldn't abort the report
                entries.append(("err", label, " ".join(f"{type(e).__name__}: {e}".split())[:100]))
                continue
            # Closest golden: most knobs reproduced (registry-canonical values_equal, via _knob_eq — a legacy
            # spelling vs the site-form pick), tie-broken by match fraction.
            scored = [(sum(1 for k in gd if _knob_eq(k, gd[k], got)), gd) for gd in pool.schedule_rows()]
            matched, gold = max(scored, key=lambda t: (t[0], t[0] / len(t[1]) if t[1] else 1.0))
            n_match += matched == len(gold)
            n_rows += 1
            for k in gold:
                knob_total[k] = knob_total.get(k, 0) + 1
                knob_match[k] = knob_match.get(k, 0) + _knob_eq(k, gold[k], got)
            lead = [label, (f"{matched}/{len(gold)}", _ratio_color(matched, len(gold)))]
            entries.append(("row", lead, gold, got))
    finally:
        quiet.setLevel(prev)
    # Totals row (replaces a trailing summary line): per-knob match counts over the rows, plus the
    # exactly-reproduced row count in the m/t column.
    total_cells = {k: (f"{knob_match[k]}/{knob_total[k]}", knob_match[k] != knob_total[k]) for k in knob_total}
    total_lead = ["TOTAL", (f"{n_match}/{n_rows}", _ratio_color(n_match, n_rows))]
    entries.append(("total", total_lead, total_cells))
    _emit_golden_table([Col("kernel"), Col("m/t")], entries, "knobs (found/golden)")


def handle_eval_golden(args) -> None:
    """Validate one file-scoped golden corpus against the pinned serving envelope."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import GoldenFile, sole_evidence
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from emmy.serving.release import load_serving_config, model_matches  # noqa: PLC0415
    from emmy.serving.twins import capture_twin_graphs  # noqa: PLC0415

    try:
        serving = load_serving_config(args.serving_config)
        golden_path = Path(args.golden).resolve()
        if golden_path != serving.golden_file:
            raise ValueError(f"serving config names {serving.golden_file}, not {golden_path}")
        document = GoldenFile.load(golden_path)
        records = document.records()
        ctx = Context.probe()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("golden evaluation setup failed: %s", exc)
        sys.exit(2)

    cap = tuple(document.compute_cap)
    if document.gpu_name != serving.gpu_name:
        logger.error("golden GPU %r does not match serving config GPU %r", document.gpu_name, serving.gpu_name)
        sys.exit(1)
    if ctx.gpu_name != serving.gpu_name or tuple(ctx.compute_capability) != cap:
        logger.error(
            "live GPU %r sm_%d%d does not match golden/config target %r sm_%d%d",
            ctx.gpu_name,
            *ctx.compute_capability,
            serving.gpu_name,
            *cap,
        )
        sys.exit(1)
    if not records or any(not model_matches(record.model, serving) for record in records):
        recorded = sorted({record.model or "(missing)" for record in records})
        logger.error("golden model provenance %s does not cover %s", ", ".join(recorded) or "(none)", serving.model_provenance)
        sys.exit(1)

    from emmy.serving.twins import twin_width  # noqa: PLC0415

    missing = []
    for config_index, entry in enumerate(document.configs):
        # A target's rows are the ones its twin reaches: a static twin is compiled at its own
        # width, a symbolic one for any width. The twin is the realization name's first field.
        twin = entry.realizations[0].name.split(".", 1)[0]
        expected = {(row.bindings, row.pins) for row in serving.realizations_for(twin_width(twin))}
        actual = {
            (tuple(sorted(realization.bindings.items())), tuple(sorted(realization.pins.items()))) for realization in entry.realizations
        }
        for bindings, pins in sorted(expected - actual, key=lambda item: (item[1], item[0])):
            missing.append((config_index, dict(bindings), pins))
    if missing:
        for config_index, bindings, pins in missing[:20]:
            logger.error("configs[%d] missing realization bindings=%s pins=%s", config_index, bindings, dict(pins))
        if len(missing) > 20:
            logger.error("... and %d more missing target realizations", len(missing) - 20)
        sys.exit(1)

    logger.info("OK: %d verified realizations cover %s on %s.", len(records), serving.model_provenance, serving.gpu_name)
    if _emit_offer_audit(records):
        sys.exit(1)

    source = serving.model_provenance
    try:
        if serving.static_only:
            graphs = capture_twin_graphs(source, decode_bucket=1, prefill_bucket=0, symbolic=False, static_only=True)
        else:
            graphs = capture_twin_graphs(
                source,
                decode_bucket=0,
                prefill_bucket=0,
                extra_widths=serving.static_widths,
                symbolic=True,
            )
    except (NotImplementedError, ValueError) as exc:
        logger.error("in-model audit cannot represent %s: %s", source, exc)
        sys.exit(1)

    # The serving-matrix half of the gate: each lane's twins compiled with that lane's rows as
    # the only evidence, strictly, on the live card the golden names — a fork no golden row
    # decides is an EvidenceError naming the kernel, never a prediction the prior makes. A lane
    # reaches the widths the config warms in it (a shape's ``:fm`` suffix names its lane), so a
    # static twin is compiled in the lanes that list its width; a symbolic twin in every lane.
    failed = False
    for pins in sorted({row.pins for row in serving.realizations}, key=repr):
        lane = _format_pins(pins)
        reached = {dict(row.bindings).get("num_tokens") if row.bindings else None for row in serving.realizations if row.pins == pins}
        broken = 0
        with pinned_knobs(dict(pins)), sole_evidence([record for record in records if record.pins == pins]):
            for name, graph in graphs.items():
                if twin_width(name) not in reached:
                    continue
                try:
                    Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
                except Exception as exc:  # noqa: BLE001 — one twin's failure is that twin's verdict
                    broken += 1
                    logger.error("%s: %s: %s", lane, name, " ".join(f"{type(exc).__name__}: {exc}".split()))
        compiled = sum(1 for name in graphs if twin_width(name) in reached)
        logger.info("%s: %d twin(s) deploy from the golden rows alone, %d do not", lane, compiled - broken, broken)
        failed |= bool(broken)
    if failed:
        logger.error("serving audit failed: every fork of every reachable kernel must be decided by a golden row")
        sys.exit(1)


def _ratio_color(matched: int, total: int) -> str:
    """Green (all match) / yellow (>80%) / red (otherwise)."""
    frac = matched / total if total else 1.0
    return _GREEN if matched == total else (_YELLOW if frac > 0.8 else _RED)


def _knob_cells(entry: tuple) -> dict[str, tuple[str, bool]]:
    """``{knob: (value_text, red?)}`` for one renderable entry (no ``NAME=`` prefix —
    :func:`~emmy.commands.table.knob_columns` puts the name in the column header).
    A ``("row", lead, gold, got)`` entry renders ``found/golden`` per knob, red where the
    two differ (``knob.values_equal`` — so a legacy golden spelling compares equal to the
    site-form pick it realizes as); a ``("total", lead, summaries)`` entry carries its summaries
    pre-built."""
    if entry[0] == "total":
        return entry[2]
    _, _, gold, got = entry
    return {k: (f"{got.get(k, '-')}/{gold[k]}", not _knob_eq(k, gold[k], got)) for k in gold}


def _knob_eq(k: str, gv, got: dict) -> bool:
    """Whether the picked knob dict ``got`` reproduces the golden's ``k=gv`` — value equality
    through the registry-canonical :func:`~emmy.compiler.pipeline.knob.values_equal`, so the
    legacy golden corpus keeps matching the site-form picks during the step-7 window."""
    from emmy.compiler.pipeline.knob import values_equal  # noqa: PLC0415

    return k in got and values_equal(k, gv, got[k])


def _emit_golden_table(lead_cols: list[Col], entries: list[tuple], caption: str) -> None:
    """Stream a golden table via ``logger``: ``lead_cols`` (kernel, m/t, …) plus the aligned
    ``found/golden`` knob columns (knob name in the header, value-only summaries). ``entries``
    preserves config order — each is ``("row", lead_cells, gold, got)``,
    ``("total", lead_cells, knob_cells)`` (a pre-built aggregate row), or
    ``("err", kernel_name, message)``; an error row prints its kernel name (aligned to the
    kernel column) then the raw message in place. ``caption`` is printed above the table."""
    body = [e for e in entries if e[0] != "err"]
    kcols, kcells = knob_columns([_knob_cells(e) for e in body])
    columns = lead_cols + kcols
    data = [e[1] + kc for e, kc in zip(body, kcells, strict=True)]
    # Floor the kernel column to the widest error-row name so error rows align with the table.
    floor = [max((len(e[1]) for e in entries if e[0] == "err"), default=0)] + [0] * (len(columns) - 1)
    kernel_w = col_widths(columns, data, floor)[0]
    lines = iter(render_table(columns, data, indent="  ", min_widths=floor))
    logger.info("  " + caption)
    logger.info(next(lines))  # header row (column names, knobs included)
    for e in entries:
        logger.info("  " + e[1].ljust(kernel_w) + "  ERR  " + e[2] if e[0] == "err" else next(lines))


def _bare_families(knobs: dict) -> dict:
    """Aggregate exact-site tuning knobs by family for the single-site summary table.

    The table compares one resolved choice per family rather than schedule identities. First key
    wins on a family collision; a multi-node kernel has no single family value to summarize.
    """
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415

    out: dict = {}
    for k, v in knobs.items():
        out.setdefault(family_of(k), v)
    return out


def _emit_offer_audit(configs: list) -> bool:
    """The offer audit — does each recorded row still equal an enumerated leaf of its own target?

    The strict decode (``golden.decode_record``) asks it per entry: the persisted program replayed
    under the entry's own input pins, with the target's other entries walking the same path, and
    the spelled row compared with that kernel's enumerated leaves by exact schedule-row identity.
    An entry whose row equals no leaf is ``UNREALIZED``: it is no evidence a deploy can use, so
    the gate fails — re-record an offered row in this input regime, or close the enumeration gap.
    This is the OWN-SNIPPET view; the serving-matrix compile in :func:`handle_eval_golden` closes
    the other side, whether the fused serving graphs are decided by these rows. Returns True when any entry
    is unrealized (``eval golden`` exits 1)."""
    from emmy.compiler.pipeline.search.golden import decode_record  # noqa: PLC0415

    def kstr(g) -> str:  # the entry's distinguishing knobs, empty families dropped
        return ",".join(f"{k}={v}" for k, v in g.knobs.items() if v not in ("", None))

    logger.info("")
    logger.info("Offer audit — does each recorded row still equal an enumerated leaf (own snippet, deployable regime)?")
    unrealized = 0
    for g in configs:
        try:
            reason = decode_record(g, configs)
        except Exception as exc:  # noqa: BLE001 — one entry's error is that entry's verdict
            reason = f"{type(exc).__name__}: {exc}"
        if reason is None:
            continue
        unrealized += 1
        why = " ".join(reason.split())[:120]
        logger.info("  %-44s  UNREALIZED  %.1fus  %s  (%s)", _realization_label(g.name, g.pins), g.emmy_us, kstr(g), why)
    if unrealized:
        logger.error("  offer audit: %d of %d entries equal no enumerated leaf in their input regime", unrealized, len(configs))
        return True
    logger.info("  offer audit: all %d entries equal an enumerated leaf", len(configs))
    return False
