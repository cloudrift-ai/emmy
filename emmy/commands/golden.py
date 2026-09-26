"""``emmy golden {check,restamp,kernels}`` — a repository golden against the fresh lowering of its own programs.

``check`` names the stored targets the current compiler no longer lowers the golden's programs to:
per traced program, the diff between ``kernels`` (the Loop IR pool the golden stores) and
``emmy compile --golden PATH --program N --ir loop -o fresh.yaml`` (the same pool lowered fresh), restricted
to the targets the file stores. ``restamp`` rewrites the golden onto that lowering
(``compiler/pipeline/search/restamp.py`` decides what each row keeps). All are GPU-free; ``check``
and ``restamp`` default to every repository golden, and the ``refresh-golden`` skill is the flow
around them, including what needs a card.
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

    pc = sub.add_parser("check", help="Name the stored targets a fresh lowering of the golden's own programs no longer writes")
    pc.add_argument("paths", nargs="*", help="Golden files to check. Default: every repository golden.")
    pc.set_defaults(func=handle_golden_check)

    pr = sub.add_parser(
        "restamp",
        help="Rewrite a golden onto the fresh lowering: targets take the fresh Loop IR, rows are re-keyed, a row whose "
        "kernel renders differently keeps its schedule and loses its measurement (a proposal), a row that no longer "
        "decodes is dropped",
    )
    pr.add_argument("paths", nargs="*", help="Golden files to restamp. Default: every repository golden.")
    pr.set_defaults(func=handle_golden_restamp)

    pk = sub.add_parser(
        "kernels",
        help="Print the Loop IR kernels a golden stores, as its pool sorted by output set — what "
        "`emmy compile --golden PATH --program N --ir loop -o fresh.yaml` must write for the file to be current",
    )
    pk.add_argument("path", help="The golden file.")
    pk.add_argument("--program", type=int, metavar="N", help="Only the targets of traced program N.")
    pk.set_defaults(func=handle_golden_kernels)


def _goldens(paths: list[str]) -> list[Path]:
    from emmy.compiler.pipeline.search.golden.repository import _repository_golden_paths  # noqa: PLC0415

    if paths:
        return [Path(path).expanduser() for path in paths]
    with _repository_golden_paths() as repository:
        return list(repository)


def handle_golden_check(args) -> None:
    from emmy.compiler.pipeline.search.golden import GoldenFile  # noqa: PLC0415
    from emmy.compiler.pipeline.search.restamp import stale_targets  # noqa: PLC0415

    failed = False
    for path in _goldens(args.paths):
        document = GoldenFile.load(path)
        stale = list(stale_targets(document))
        if not stale:
            logger.info("%s: every stored target is the fresh lowering (%d)", path, len(document.configs))
            continue
        failed = True
        logger.error("%s: %d of %d stored targets are not the fresh lowering", path, len(stale), len(document.configs))
        for reason in stale[:_SHOWN]:
            logger.error("  %s", reason)
        if len(stale) > _SHOWN:
            logger.error("  ... and %d more", len(stale) - _SHOWN)
    if failed:
        sys.exit(1)


def handle_golden_restamp(args) -> None:
    from emmy.compiler.pipeline.search.golden import GoldenFile  # noqa: PLC0415
    from emmy.compiler.pipeline.search.restamp import restamp  # noqa: PLC0415

    failed = False
    for path in _goldens(args.paths):
        document, report = restamp(GoldenFile.load(path))
        if not report.changed:
            logger.info("%s: already the fresh lowering (%d targets)", path, report.targets)
            continue
        for line in report.lines():
            logger.info("%s: %s", path, line)
        if document is None:
            logger.error("%s: no target survives the fresh lowering; delete the file or re-record it on its card", path)
            failed = True
            continue
        document.dump(path, overwrite=True)
    if failed:
        sys.exit(1)


def handle_golden_kernels(args) -> None:
    from emmy.compiler.pipeline.search.golden import GoldenFile, kernel_pool_text  # noqa: PLC0415

    path = Path(args.path).expanduser()
    document = GoldenFile.load(path)
    if args.program is not None and not 0 <= args.program < len(document.programs):
        logger.error("--program %d: %s stores %d program(s)", args.program, path, len(document.programs))
        sys.exit(2)
    sys.stdout.write(kernel_pool_text(document.kernels(args.program)))
