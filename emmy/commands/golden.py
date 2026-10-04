"""``emmy golden {check,restamp}`` — a golden against the fresh lowering of its own programs.

``check`` says what a restamp would change: a kernel the fresh lowering keys differently, a decision it no longer
takes the same way, a row whose measurement is of a kernel the compiler no longer builds. ``restamp`` writes that
rewrite (``search/golden/restamp.py`` decides what each row keeps). Both are GPU-free and default to every
repository golden; the ``refresh-golden`` skill is the flow around them, including what needs a card.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SHOWN = 12


def register_golden_command(subparsers) -> None:
    parser = subparsers.add_parser("golden", help="Check repository goldens against a fresh lowering, or restamp them onto it")
    sub = parser.add_subparsers(dest="golden_target", required=True)

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
        return [Path(path).expanduser() for path in paths]
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
