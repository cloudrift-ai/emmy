"""``emmy dataset {import,freeze}`` — fill a dataset DB instance, and snapshot one into a measurement freeze.

A dataset DB is the tune DB's schema in its own file (``EMMY_DATASET_DB``): the measurement-data readers
(``eval prior``) read it, and no compile ever does, so what is imported into it cannot change a deploy.

- ``import`` loads sources into it: a measurement freeze directory (the checked-in one by default) or
  a tune DB file, whose CUDA ``perf`` rows are copied over (the way a card's measurements from a rented
  GPU reach the dataset). Rows keep the source they arrived from, and the upsert is the tune DB's own,
  so importing the same source twice changes nothing.
- ``freeze`` writes a DB instance's admitted rows as a digest-pinned freeze directory — the artifact
  that gets checked in, so a reported number is one anyone can reproduce.

:func:`dataset_db` is the readers' way in: it resolves the instance and refuses a missing one, or a
default one that does not hold the checked-in freeze, with the command that fixes it.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from emmy import config

logger = logging.getLogger(__name__)


def register_dataset_command(subparsers) -> None:
    parser = subparsers.add_parser("dataset", help="Fill a dataset DB from freezes / tune DBs, or freeze one")
    sub = parser.add_subparsers(dest="dataset_target", required=True)

    pi = sub.add_parser("import", help="Import measurement freezes and tune DBs into a dataset DB")
    pi.add_argument(
        "sources",
        nargs="*",
        help="Freeze directories and tune DB files to import. Default: the checked-in measurement freeze "
        "(EMMY_FREEZE_DIR, else search/freezes/).",
    )
    pi.add_argument("--db", help="Dataset DB to fill (default: EMMY_DATASET_DB or ~/.cache/emmy/dataset.db).")
    pi.add_argument("--fresh", action="store_true", help="Delete the dataset DB first, so it holds exactly these sources.")
    pi.set_defaults(func=handle_dataset_import)

    pf = sub.add_parser("freeze", help="Write a DB instance's admitted rows as a digest-pinned measurement freeze")
    pf.add_argument("--db", help="DB instance to freeze (default: EMMY_DATASET_DB or ~/.cache/emmy/dataset.db).")
    pf.add_argument("--out", required=True, help="Freeze directory to write (an existing freeze there is replaced).")
    pf.add_argument("--note", default="", help="Freeform collection-policy note stamped into the manifest.")
    pf.set_defaults(func=handle_dataset_freeze)


def handle_dataset_import(args) -> None:
    from emmy.compiler.pipeline.search.data.freeze import load_freeze  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    db_path = Path(args.db).expanduser() if args.db else config.dataset_db_path()
    sources = [Path(s).expanduser() for s in args.sources] or [config.freeze_path()]
    for src in sources:
        if not src.exists():
            logger.error("no freeze directory or tune DB at %s", src)
            sys.exit(2)
    if args.fresh:
        for suffix in ("", "-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
    db = SearchDB(db_path)
    try:
        for src in sources:
            if src.is_dir():
                manifest, rows = load_freeze(src)
                n = db.record_perf_rows(rows)
                logger.info("imported %d row(s) from freeze %s (sha256 %s)", n, src, manifest["sha256"])
                continue
            tune_db = SearchDB.open_readonly(src)
            try:
                rows = list(tune_db.iter_perf_rows(backend="cuda"))
            finally:
                tune_db.close()
            keyed = [r for r in rows if r.gpu]
            n = db.record_perf_rows(keyed)
            logger.info("imported %d row(s) from tune DB %s", n, src)
            if len(rows) > len(keyed):
                logger.info("  skipped %d row(s) recorded before the card joined the key — no dataset reads them", len(rows) - len(keyed))
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
    manifest = write_freeze(db_path, out, note=args.note)
    counts = manifest["counts"]
    logger.info("froze %d row(s) (%d ok + %d bench_fail) from %s", counts["rows"], counts["ok"], counts["bench_fail"], db_path)
    logger.info("  per card: %s", ", ".join(f"{gpu}: {n}" for gpu, n in counts["per_gpu"].items()))
    logger.info(
        "  commit %s, sha256 %s over %d per-GPU file(s) -> %s/", manifest["repo_commit"], manifest["sha256"], len(manifest["files"]), out
    )


def dataset_db(db_arg: str | None) -> Path:
    """The DB instance a measurement-data reader reads: ``--db`` when given, else the dataset DB.

    Exits with the fixing command when the file is missing, or when the DEFAULT dataset DB does not hold
    the checked-in measurement freeze — a report computed over another freeze's rows, or an old one's,
    would carry today's label and yesterday's numbers. An explicit ``--db`` is read as it is."""
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
    manifest = config.freeze_path() / "manifest.json"
    if manifest.is_file():
        want = f"freeze:{json.loads(manifest.read_text())['sha256'][:12]}"
        db = SearchDB.open_readonly(path)
        try:
            held = db.perf_sources()
        finally:
            db.close()
        if want not in held:
            logger.error("dataset DB %s does not hold the current measurement freeze (%s) — run `emmy dataset import --fresh`", path, want)
            sys.exit(2)
    return path
