"""``emmy candidates {build,check,restamp,realize}`` — the candidate files: a few measured schedules that are not the
best known one, for each group a hardware golden holds (``search/candidates``).

``build`` fills each hardware golden's candidate file from the measurements a DB instance holds on the same card —
the tune DB of the machine the tunes ran on — keeping the rows the file already holds, and says which groups are still
under their target: what a tune on the card must fill, as JSON with ``--json``. With no ``--db`` it adds nothing: the
file takes its golden's tables again, which is what follows a change to the golden.
``check``, ``restamp`` and ``realize`` are ``emmy golden``'s, over the candidate files: a candidate file is in the
golden file format and carries its golden's programs, so it is checked against a fresh lowering, restamped onto it and
pruned of the rows the compiler no longer realizes exactly as a golden is. All four are GPU-free.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from emmy.commands.golden import handle_golden_check, handle_golden_realize, handle_golden_restamp

logger = logging.getLogger(__name__)

_SHOWN = 12


def register_candidates_command(subparsers) -> None:
    parser = subparsers.add_parser("candidates", help="Build, check, restamp or prune the candidate files of the hardware goldens")
    sub = parser.add_subparsers(dest="candidates_target", required=True)
    files = "Candidate files, or directories searched for them. Default: every repository candidate file."

    pb = sub.add_parser("build", help="Fill each hardware golden's candidate file from the measurements a DB holds on its card")
    pb.add_argument("goldens", nargs="*", help="Hardware goldens to build candidate files for. Default: every hardware golden.")
    pb.add_argument(
        "--db",
        help="DB instance the candidates are taken from: the tune DB the card's tunes wrote. Default: none; each file "
        "takes its golden's tables again and keeps its rows.",
    )
    pb.add_argument("--out", help="Directory the candidate files are written to. Default: the repository's candidate files.")
    pb.add_argument(
        "--json",
        dest="json_out",
        metavar="PATH",
        help="Write the groups still under their target as JSON to PATH: the golden file, its card, the "
        "kernel, the regime and sizes a tune of it runs at, and the candidates it holds beside its target.",
    )
    pb.set_defaults(func=handle_candidates_build)

    pc = sub.add_parser("check", help="Say what a restamp onto the fresh lowering of the file's own programs would change")
    pc.add_argument("paths", nargs="*", help=files)
    pc.set_defaults(func=_over_candidates(handle_golden_check))

    pr = sub.add_parser("restamp", help="Rewrite a candidate file onto the fresh lowering, as `emmy golden restamp` does a golden")
    pr.add_argument("paths", nargs="*", help=files)
    pr.set_defaults(func=_over_candidates(handle_golden_restamp))

    pz = sub.add_parser("realize", help="Compile every candidate row under its schedule and say which the compiler no longer realizes")
    pz.add_argument("paths", nargs="*", help=files)
    pz.add_argument("--jobs", type=int, help="Worker processes compiling rows side by side (default: EMMY_WORKERS, one per core).")
    pz.add_argument("--drop", action="store_true", help="Write each file back without the rows the compiler no longer realizes.")
    pz.add_argument("--json", dest="json_out", metavar="PATH", help="Write each file's report as JSON to PATH ('-' for stdout).")
    pz.set_defaults(func=_over_candidates(handle_golden_realize))


def _over_candidates(handler):
    """``handler``, an ``emmy golden`` subcommand, over the repository's candidate files where no path is named."""

    def handle(args) -> None:
        from emmy.compiler.pipeline.search.candidates import candidate_paths  # noqa: PLC0415

        args.paths = args.paths or [str(path) for path in candidate_paths()]
        if args.paths:
            handler(args)

    return handle


def handle_candidates_build(args) -> None:
    from emmy.commands.db import db_path  # noqa: PLC0415
    from emmy.compiler.pipeline.search.candidates import RECORDS, build  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import GoldenFile  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import hardware_golden_paths  # noqa: PLC0415

    goldens = [Path(path).expanduser() for path in args.goldens] or hardware_golden_paths()
    out = Path(args.out).expanduser() if args.out else RECORDS
    try:
        db = SearchDB.open_readonly(db_path(args.db)) if args.db else None
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    short: list[dict] = []
    for path in goldens:
        golden = GoldenFile.load(path)
        target = out / path.name
        document, report = build(golden, db, GoldenFile.load(target) if target.is_file() else None)
        logger.info(
            "%s: %d candidate rows over %d groups, %d groups under their target", target, report.rows, report.groups, len(report.short)
        )
        if document.rows:
            document.dump(target, overwrite=True)
        else:
            logger.info("  no candidate measured on this card yet; no file written")
        for entry in report.short[:_SHOWN]:
            logger.info("  %s (%s): %d of %d", entry["row"], entry["kernel"], entry["held"], entry["target"])
        short.extend({"file": str(path), "gpu": golden.gpu_name, **entry} for entry in report.short)
        if len(report.short) > _SHOWN:
            logger.info("  ... and %d more", len(report.short) - _SHOWN)
        for name, us, best in report.faster:
            logger.warning("  %s measured %.2f us, faster than the golden's %.2f us", name, us, best)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(short, indent=2) + "\n")
