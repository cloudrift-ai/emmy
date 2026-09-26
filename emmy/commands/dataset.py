"""``emmy dataset {import,freeze,check}`` — fill a dataset DB instance, snapshot one into a measurement
freeze, and check that one's tables agree with themselves.

A dataset DB is the tune DB's schema in its own file (``EMMY_DATASET_DB``): the measurement-data readers
(``eval prior``) read it, and no compile ever does, so what is imported into it cannot change a deploy.

- ``import`` loads golden-shaped sources into it: a measurement freeze directory (the checked-in one by
  default) or any golden file, and a tune DB file, which is frozen first (the way a card's measurements
  from a rented GPU reach the dataset). Every kernel is re-lowered from its definition by the current
  compiler (``golden_import.import_goldens``), so the instance holds today's identities and stamps whatever
  compiler wrote the source. A file's rows are sourced by its digest, and a file the instance already holds
  is skipped: ``--fresh`` rebuilds from nothing.
- ``freeze`` writes a DB instance's admitted rows as a directory of golden files, one per card — the
  artifact that gets checked in, so a reported number is one anyone can reproduce.
- ``check`` counts the rows of an instance whose tables disagree with themselves (a knob row's digest, a
  reference, a card, the two knob vocabularies). A DB is a cache: a row the current code disagrees with is
  re-tuned or re-imported, so nothing here decodes what the compiler wrote.

:func:`dataset_db` is the readers' way in: it resolves the instance and refuses a missing one, or a
default one that does not hold the checked-in freeze, with the command that fixes it.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from emmy import config

logger = logging.getLogger(__name__)


def register_dataset_command(subparsers) -> None:
    parser = subparsers.add_parser("dataset", help="Fill a dataset DB from freezes, golden files and tune DBs, or freeze one")
    sub = parser.add_subparsers(dest="dataset_target", required=True)

    pi = sub.add_parser("import", help="Import measurement freezes, golden files and tune DBs into a dataset DB")
    pi.add_argument(
        "sources",
        nargs="*",
        help="Freeze directories, golden files and tune DB files to import. Default: the checked-in measurement freeze "
        "(EMMY_FREEZE_DIR, else search/freezes/).",
    )
    pi.add_argument("--db", help="Dataset DB to fill (default: EMMY_DATASET_DB or ~/.cache/emmy/dataset.db).")
    pi.add_argument("--fresh", action="store_true", help="Delete the dataset DB first, so it holds exactly these sources.")
    pi.set_defaults(func=handle_dataset_import)

    pf = sub.add_parser("freeze", help="Write a DB instance's admitted rows as a measurement freeze: a golden file per card")
    pf.add_argument("--db", help="DB instance to freeze (default: EMMY_DATASET_DB or ~/.cache/emmy/dataset.db).")
    pf.add_argument("--out", required=True, help="Freeze directory to write (an existing freeze there is replaced).")
    pf.set_defaults(func=handle_dataset_freeze)

    pc = sub.add_parser("check", help="Count the rows of a DB instance whose tables disagree with themselves")
    pc.add_argument("--db", help="DB instance to check (default: EMMY_DATASET_DB or ~/.cache/emmy/dataset.db).")
    pc.set_defaults(func=handle_dataset_check)


def handle_dataset_import(args) -> None:
    from emmy.compiler.pipeline.search.data.freeze import write_freeze  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden_import import import_file  # noqa: PLC0415

    db_path = Path(args.db).expanduser() if args.db else config.dataset_db_path()
    sources = [Path(s).expanduser() for s in args.sources] or [config.freeze_path()]
    for src in sources:
        if not src.exists():
            logger.error("no freeze directory, golden file or tune DB at %s", src)
            sys.exit(2)
    if args.fresh:
        for suffix in ("", "-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
    db = SearchDB(db_path)
    try:
        for src in sources:
            if src.is_dir() or src.suffix == ".yaml":
                files = sorted(src.glob("*.yaml")) if src.is_dir() else [src]
                if not files:
                    logger.error("no golden files in %s", src)
                    sys.exit(2)
            else:
                # A tune DB is frozen first, so what reaches the dataset is what a freeze of it would hold.
                frozen = Path(tempfile.mkdtemp()) / "freeze"
                write_freeze(src, frozen)
                files = sorted(frozen.glob("*.yaml"))
            for path in files:
                try:
                    import_file(db, path)
                except ValueError as exc:
                    logger.error("%s", exc)
                    sys.exit(2)
    finally:
        db.close()
    logger.info("dataset DB: %s", db_path)


def handle_dataset_freeze(args) -> None:
    from emmy.compiler.pipeline.search.data.freeze import write_freeze  # noqa: PLC0415

    db_path = Path(args.db).expanduser() if args.db else config.dataset_db_path()
    if not db_path.is_file():
        logger.error("no DB at %s", db_path)
        sys.exit(2)
    out = Path(args.out).expanduser()
    digests = write_freeze(db_path, out)
    for name, digest in digests.items():
        logger.info("froze %s (sha256 %s)", name, digest)
    logger.info("%d file(s) from %s -> %s/", len(digests), db_path, out)


def handle_dataset_check(args) -> None:
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    db = SearchDB.open_readonly(dataset_db(args.db))
    try:
        counts = db.drift()
    finally:
        db.close()
    for name, n in counts.items():
        logger.info("%s: %d row(s) fail", name, n)
    if any(counts.values()):
        sys.exit(1)


def dataset_db(db_arg: str | None) -> Path:
    """The DB instance a measurement-data reader reads: ``--db`` when given, else the dataset DB.

    Exits with the fixing command when the file is missing, or when the DEFAULT dataset DB does not hold
    every file of the checked-in measurement freeze — a report computed over another freeze's rows, or an
    old one's, would carry today's label and yesterday's numbers. An explicit ``--db`` is read as it is."""
    from emmy.compiler.pipeline.search.data.freeze import freeze_source  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    if db_arg:
        path = Path(db_arg).expanduser()
        if not path.is_file():
            logger.error("no DB at %s", path)
            sys.exit(2)
        return path
    path = config.dataset_db_path()
    if not path.is_file():
        logger.error("no dataset DB at %s — run `emmy dataset import` to fill it from the measurement freeze", path)
        sys.exit(2)
    freeze = config.freeze_path()
    want = {freeze_source(f) for f in sorted(freeze.glob("*.yaml"))} if freeze.is_dir() else set()
    if want:
        db = SearchDB.open_readonly(path)
        try:
            held = db.perf_sources()
        finally:
            db.close()
        if not want <= set(held):
            missing = ", ".join(sorted(want - set(held)))
            logger.error("dataset DB %s lacks the current measurement freeze (%s) — run `emmy dataset import --fresh`", path, missing)
            sys.exit(2)
    return path
