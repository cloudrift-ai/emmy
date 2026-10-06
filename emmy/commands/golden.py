"""``emmy golden {list,check,restamp}`` — a golden's measured rows, and a golden against the fresh lowering of its
own programs.

``list`` prints every measured row — its time, the reference the bench took beside it, the whole row's time beside
``torch.compile`` where a record run timed it, the schedules tried and the note — sortable and filterable, and as
JSON; it reads the file and judges nothing.

``check`` says what a restamp would change: a kernel the fresh lowering keys differently, a decision it no longer
takes the same way, a row whose measurement is of a kernel the compiler no longer builds. ``restamp`` writes that
rewrite (``search/golden/restamp.py`` decides what each row keeps). Both are GPU-free and default to every
repository golden; the ``refresh-golden`` skill is the flow around them, including what needs a card.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SHOWN = 12


def register_golden_command(subparsers) -> None:
    parser = subparsers.add_parser("golden", help="Check repository goldens against a fresh lowering, or restamp them onto it")
    sub = parser.add_subparsers(dest="golden_target", required=True)

    pl = sub.add_parser(
        "list",
        help="List measured rows with their time beside torch.compile, slowest relative to it first",
    )
    pl.add_argument("paths", nargs="*", help="Golden files, or directories searched for them. Default: every repository golden.")
    pl.add_argument("--gpu", help="Keep the rows timed on a card whose name contains this.")
    pl.add_argument("--kernel", help="Keep the rows whose name or kernel contains this.")
    pl.add_argument("--behind", action="store_true", help="Keep the rows slower than torch.compile.")
    pl.add_argument(
        "--missing",
        action="store_true",
        help="List what a record run on the file's card must measure instead: each proposal row (no Emmy time) and each "
        "target with no torch.compile time, named by the realization to run.",
    )
    pl.add_argument("--json", dest="json_out", metavar="PATH", help="Write the listed rows as JSON to PATH ('-' for stdout).")
    pl.set_defaults(func=handle_golden_list)

    pc = sub.add_parser("check", help="Say what a restamp onto the fresh lowering of the golden's own programs would change")
    pc.add_argument("paths", nargs="*", help="Golden files to check. Default: every repository golden.")
    pc.set_defaults(func=handle_golden_check)

    pr = sub.add_parser(
        "restamp",
        help="Rewrite a golden onto the fresh lowering: kernels take the fresh body, decisions are taken "
        "again, a row whose kernel's body changed keeps its schedule and loses its measurement (a proposal)",
    )
    pr.add_argument("paths", nargs="*", help="Golden files to restamp. Default: every repository golden.")
    pr.set_defaults(func=handle_golden_restamp)


def _goldens(paths: list[str]) -> list[Path]:
    from emmy.compiler.pipeline.search.golden import repository_golden_paths  # noqa: PLC0415

    if paths:
        found = [Path(path).expanduser() for path in paths]
        return [golden for path in found for golden in (sorted(path.rglob("*.json")) if path.is_dir() else [path])]
    with repository_golden_paths() as repository:
        return list(repository)


def handle_golden_check(args) -> None:
    from emmy.compiler.pipeline.search.golden import GoldenFile, restamp  # noqa: PLC0415

    failed = False
    for path in _goldens(args.paths):
        document = GoldenFile.load(path)
        fresh, report = restamp(document)
        if fresh == document:
            logger.info("%s: current (%d kernels)", path, len(document.kernels))
            continue
        failed = True
        lines = report.lines()
        logger.error("%s: not the fresh lowering", path)
        for line in lines[:_SHOWN]:
            logger.error("  %s", line)
        if len(lines) > _SHOWN:
            logger.error("  ... and %d more", len(lines) - _SHOWN)
    if failed:
        sys.exit(1)


def handle_golden_restamp(args) -> None:
    from emmy.compiler.pipeline.search.golden import GoldenFile, restamp  # noqa: PLC0415

    failed = False
    for path in _goldens(args.paths):
        document = GoldenFile.load(path)
        fresh, report = restamp(document)
        if fresh == document:
            logger.info("%s: already the fresh lowering (%d kernels)", path, len(document.kernels))
            continue
        for line in report.lines():
            logger.info("%s: %s", path, line)
        if not fresh.kernels:
            logger.error("%s: no kernel survives the fresh lowering; delete the file or re-record it on its card", path)
            failed = True
            continue
        fresh.dump(path, overwrite=True)
    if failed:
        sys.exit(1)


def listing(path: Path, document) -> list[dict]:
    """One entry per measured row of ``document`` and card it was timed on: the row's own measurement (its kernel's
    time, the reference beside it, the schedules tried) and, where a record run timed the whole row on that card,
    the row's time beside ``torch.compile`` and eager (``vs_tcompile`` the ratio). A proposal is not listed."""
    out = []
    for row in document.rows:
        measured = row.measurements
        cards = dict(row.latency or {})
        if measured is not None:
            cards.setdefault(document.gpu_name or "", None)
        for card, latency in cards.items():
            whole = latency.emmy_us if latency is not None else None
            tcompile = latency.tcompile_us if latency is not None else None
            out.append(
                {
                    "file": str(path),
                    "card": card,
                    "row": row.name,
                    "kernel": row.kernel,
                    "pins": row.pins,
                    "knobs": row.knobs,
                    "emmy_us": measured.emmy_us if measured is not None else None,
                    "reference_us": measured.reference_us if measured is not None else None,
                    "reference_backend": measured.reference_backend if measured is not None else None,
                    "tried": measured.tried if measured is not None else None,
                    "whole_us": whole,
                    "tcompile_us": tcompile,
                    "eager_us": latency.eager_us if latency is not None else None,
                    "vs_tcompile": whole / tcompile if whole and tcompile else None,
                    "note": row.note,
                }
            )
    return out


def missing(path: Path, document) -> list[dict]:
    """What a record run on the file's card must measure: each proposal (``emmy``: a row with a schedule and no
    measurement — ``piece`` when its kernel is a cut piece, which replays under its route), then each target and input
    regime with no ``torch.compile`` time on any of its rows (``tcompile``: named by its shortest row name, the
    realization ``run --golden PATH`` benches it as). A target with a proposal is measured by the proposal's record,
    which times ``torch.compile`` too, so it is not listed twice."""
    targets = {kernel.ref for kernel in document.targets()}

    def entry(row, what: str) -> dict:
        return {
            "file": str(path),
            "gpu": document.gpu_name,
            "row": row.name,
            "kernel": row.kernel,
            "pins": row.pins,
            "knobs": row.knobs,
            "missing": what,
            "piece": row.kernel not in targets,
        }

    out, timed = [], []
    for rows in document.target_rows().values():
        proposals = [row for row in rows if row.measurements is None and not row.latency]
        out.extend(entry(row, "emmy") for row in proposals)
        if not proposals and not any(latency.tcompile_us for row in rows for latency in (row.latency or {}).values()):
            timed.append(entry(min(rows, key=lambda row: (len(row.name), row.name)), "tcompile"))
    return out + timed


def _us(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def handle_golden_list(args) -> None:
    from emmy.commands.table import Col, render_table  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import GoldenFile  # noqa: PLC0415

    if args.missing:
        entries = [entry for path in _goldens(args.paths) for entry in missing(path, GoldenFile.load(path))]
        entries = [
            entry
            for entry in entries
            if (not args.gpu or args.gpu in (entry["gpu"] or "")) and (not args.kernel or args.kernel in entry["row"])
        ]
        if args.json_out:
            text = json.dumps(entries, indent=2)
            print(text) if args.json_out == "-" else Path(args.json_out).write_text(text + "\n")
            return
        for entry in entries:
            print(f"{Path(entry['file']).stem}  {entry['gpu']}  {entry['missing']:8s}  {entry['row']}")
        print(f"{len(entries)} measurement(s) missing")
        return
    entries = [entry for path in _goldens(args.paths) for entry in listing(path, GoldenFile.load(path))]
    entries = [
        entry
        for entry in entries
        if (not args.gpu or args.gpu in entry["card"])
        and (not args.kernel or args.kernel in entry["row"] or args.kernel in entry["kernel"])
        and (not args.behind or (entry["vs_tcompile"] or 0) > 1)
    ]
    entries.sort(key=lambda entry: (-(entry["vs_tcompile"] or 0), -(entry["emmy_us"] or entry["whole_us"] or 0)))
    if args.json_out == "-":
        print(json.dumps(entries, indent=2))
        return
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(entries, indent=2) + "\n")
    columns = [
        Col("file"),
        Col("card"),
        Col("row"),
        *(Col(name, "r") for name in ("emmy_us", "ref_us", "tried", "whole_us", "tcompile_us", "x_tc")),
    ]
    columns.append(Col("note"))
    rows = [
        [
            Path(entry["file"]).stem,
            entry["card"],
            entry["row"],
            _us(entry["emmy_us"]),
            _us(entry["reference_us"]),
            str(entry["tried"] or "-"),
            _us(entry["whole_us"]),
            _us(entry["tcompile_us"]),
            "-" if entry["vs_tcompile"] is None else f"{entry['vs_tcompile']:.2f}x",
            entry["note"] or "",
        ]
        for entry in entries
    ]
    for line in render_table(columns, rows, rule=True):
        print(line)
    print(f"{len(entries)} row(s)")
