"""``emmy db {import,export,freeze,check}`` — the DB instance the offline prior's data passes through: fill it from
golden-shaped sources, export its rows as a dataset, snapshot it into a measurement freeze, and check that its tables
agree with themselves.

The instance is the tune DB's schema in a file of its own, the one ``--db PATH`` names on every subcommand — never a
default, so a refit can never touch the tune DB a compile reads. The README's examples put it at ``_data/dataset.db``
beside the exported dataset, under ``_data/`` where git ignores it, so both are there to inspect and neither is a
commit away from the repository. No compile ever reads it, so what is imported into it cannot change a deploy.

- ``import SOURCES…`` loads golden-shaped sources: measurement freeze directories, golden files, and tune DB files,
  which are frozen first (the way a card's measurements from a rented GPU reach the dataset). Nothing is loaded by
  default: the sources are named on the command line — for the offline prior, the hardware goldens
  (``search/golden/records/*.json``) under ``--fresh``, the README's "Fit the offline prior" workflow. Every kernel is
  re-lowered from its definition by the current compiler (``golden.evidence.import_goldens``), so the instance holds
  today's identities and stamps whatever compiler wrote the source. A file's rows are sourced by its kind and digest
  (``freeze:`` for a freeze directory's files, ``golden:`` for a golden file), and a file the instance already holds
  is skipped: ``--fresh`` rebuilds from nothing.
- ``export OUT`` writes the instance's rows as a dataset directory (``search/dataset``): every golden pool
  enumerated from its kernel's definition and packed, every measured pool, and the provenance. ``emmy fit`` and
  ``emmy eval prior`` read that directory, never the DB.
- ``freeze`` writes an instance's admitted rows as a directory of golden files, one per card — the artifact that
  gets checked in, so a reported number is one anyone can reproduce.
- ``check`` counts the rows of an instance whose tables disagree with themselves (a knob row's digest, a reference,
  a card, the two knob vocabularies). A DB is a cache: a row the current code disagrees with is re-tuned or
  re-imported, so nothing here decodes what the compiler wrote.

:func:`db_path` is every subcommand's way in: the instance ``--db`` names, refused when missing with the command that
fills it. :func:`read_samples` is the per-kernel ``eval`` views' way into a tune DB.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from emmy.compiler.pipeline.search.pool import DEFAULT_SAMPLE

logger = logging.getLogger(__name__)


def register_db_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "db", help="Fill a DB from golden files, freezes and tune DBs; export it as a dataset; freeze or check it"
    )
    sub = parser.add_subparsers(dest="db_target", required=True)
    db_help = "DB instance file, e.g. _data/dataset.db (git ignores _data/); never the tune DB a compile reads."

    pi = sub.add_parser("import", help="Import golden files, measurement freezes and tune DBs into a DB instance")
    pi.add_argument(
        "sources",
        nargs="+",
        help="Freeze directories, golden files and tune DB files to import — for the offline prior, the hardware goldens "
        "emmy/compiler/pipeline/search/golden/records/*.json (README, 'Fit the offline prior'). Nothing is imported by default.",
    )
    pi.add_argument("--db", required=True, help=db_help)
    pi.add_argument("--fresh", action="store_true", help="Delete the DB first, so it holds exactly these sources.")
    pi.set_defaults(func=handle_db_import)

    pe = sub.add_parser("export", help="Write a DB instance's rows as a dataset directory: golden pools, measured pools, provenance")
    pe.add_argument("out", help="Dataset directory to write, e.g. _data/dataset.")
    pe.add_argument("--db", required=True, help=db_help)
    pe.add_argument(
        "--pool-sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Candidates drawn per golden pool during enumeration (default {DEFAULT_SAMPLE}; 0 enumerates every row). "
        "Recorded in the dataset's provenance — two datasets are comparable only when it matches.",
    )
    pe.add_argument("--seed", type=int, default=0, help="Seed of the per-pool draw (default: 0).")
    pe.set_defaults(func=handle_db_export)

    pf = sub.add_parser("freeze", help="Write a DB instance's admitted rows as a measurement freeze: a golden file per card")
    pf.add_argument("--db", required=True, help=db_help)
    pf.add_argument("--out", required=True, help="Freeze directory to write (an existing freeze there is replaced).")
    pf.set_defaults(func=handle_db_freeze)

    pc = sub.add_parser("check", help="Count the rows of a DB instance whose tables disagree with themselves")
    pc.add_argument("--db", required=True, help=db_help)
    pc.set_defaults(func=handle_db_check)


def handle_db_import(args) -> None:
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db.freeze import write_freeze  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.evidence import file_source, import_file  # noqa: PLC0415

    path = Path(args.db).expanduser()
    sources = [Path(s).expanduser() for s in args.sources]
    for src in sources:
        if not src.exists():
            logger.error("no freeze directory, golden file or tune DB at %s", src)
            sys.exit(2)
    if args.fresh:
        for suffix in ("", "-wal", "-shm"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = SearchDB(path)
    try:
        for src in sources:
            # A file is a golden file; a directory is a freeze; a tune DB is frozen first, so what reaches the
            # dataset is what a freeze of it would hold.
            if src.suffix == ".json":
                files = [(src, file_source("golden", src))]
            else:
                frozen = src
                if not src.is_dir():
                    frozen = Path(tempfile.mkdtemp()) / "freeze"
                    write_freeze(src, frozen)
                files = [(file, file_source("freeze", file)) for file in sorted(frozen.glob("*.json"))]
                if not files:
                    logger.error("no golden files in %s", src)
                    sys.exit(2)
            for file, source in files:
                try:
                    import_file(db, file, source)
                except ValueError as exc:
                    logger.error("%s", exc)
                    sys.exit(2)
    finally:
        db.close()
    logger.info("DB: %s", path)


def handle_db_export(args) -> None:
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db.export import export_dataset  # noqa: PLC0415

    path = db_path(args.db)
    db = SearchDB.open_readonly(path)
    try:
        dataset = export_dataset(db, source=str(path), pool_sample=args.pool_sample, seed=args.seed)
    finally:
        db.close()
    try:
        out = dataset.dump(args.out)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    logger.info(
        "dataset: %s (%d golden groups, %d measured groups, %d golden rows skipped)",
        out,
        len(dataset.golden),
        len(dataset.measured),
        len(dataset.skipped),
    )


def handle_db_freeze(args) -> None:
    from emmy.compiler.pipeline.search.db.freeze import write_freeze  # noqa: PLC0415

    path = db_path(args.db)
    out = Path(args.out).expanduser()
    digests = write_freeze(path, out)
    for name, digest in digests.items():
        logger.info("froze %s (sha256 %s)", name, digest)
    logger.info("%d file(s) from %s -> %s/", len(digests), path, out)


def handle_db_check(args) -> None:
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    db = SearchDB.open_readonly(db_path(args.db))
    try:
        counts = db.drift()
    finally:
        db.close()
    for name, n in counts.items():
        logger.info("%s: %d row(s) fail", name, n)
    if any(counts.values()):
        sys.exit(1)


def db_path(db_arg: str) -> Path:
    """The DB instance a subcommand reads — the file ``--db`` names — exiting with the command that fills it when the
    file is missing."""
    path = Path(db_arg).expanduser()
    if not path.is_file():
        logger.error("no DB at %s — fill it with `emmy db import --fresh SOURCES…` (README, 'Fit the offline prior')", path)
        sys.exit(2)
    return path


def read_samples(path: Path | str, *, kernel: str | None = None, status: str = "ok", backend: str | None = None):
    """The per-kernel ``eval`` views' way into a tune DB: its ``perf`` rows of ``status`` as
    :class:`~emmy.compiler.pipeline.search.dataset.Samples`, opened read-only so a concurrent ``tune`` writer is
    not blocked. ``backend=None`` spans every backend; ``kernel`` filters on the kernel row's C identifier."""
    from emmy.compiler.pipeline.search.dataset import Samples  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    db = SearchDB.open_readonly(path)
    try:
        return Samples.from_rows(db.iter_perf_rows(backend=backend), db.kernel_names(), kernel=kernel, status=status)
    finally:
        db.close()
