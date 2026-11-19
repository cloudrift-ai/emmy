"""SQLite-backed measurement store for the search package.

Pure persistence layer. One schema, several instances: the tune DB (``EMMY_TUNE_DB``) is what a compile reads
and ``run --bench`` writes; a dataset instance — the file
``emmy db import --db PATH`` names — holds the same tables, and ``emmy db export`` turns it into the dataset the
measurement-data readers read.

Tables (the DDL is the reference):

- ``kernel`` — one row per compilable kernel, keyed by its EXACT identity (``identity_key(structural=False,
  with_io=True)``: the digest of the normalized body's form plus each buffer's dtype and hint-free shape),
  holding the kernel's Loop IR wire (``wire.kernel_wire``: the body it was formed from, which the tile lift
  takes back to the kernel — ``formed`` — or, for a piece carved from a twisted tree, its derived body, which
  only its parent's program reaches). A piece a cut or a split minted is a row like any other,
  so the same kernel reached from two parents has one definition. The key is the one computed value the DB
  holds: a cache key, computed from the wire by whoever writes the row (a bench off its live kernel, an import
  off a golden's stored one) and valid under :data:`VERSION`. Nothing else derived from a kernel is stored —
  its ``S_*`` stamps are computed from the wire where a row is featurized.
- ``context`` — one row per backend, card and regime: the card's product name (``Context.hardware_id``),
  the context's target as the backend spells it (``sm_120`` on CUDA — the regime's, never a kernel's
  ``sm_120a``; LLVM's name for the host CPU, ``znver5``, on the CPU backend, whose card is the CPU and its thread
  count), effective precision decisions, and whether each launch was measured after L2 eviction.
- A knob row is a text column (:func:`row_text`: the knobs as strings, keys sorted), which is also its key. A
  ``perf`` row's ``schedule`` is the in-kernel choices a leaf kernel was measured with (``WORK``, ``TILE``,
  ``STAGE``, ``RASTER``, the in-kernel half of ``REDUCE``), never a placement knob. A ``routing`` or ``taken``
  row's ``arm`` is the kernel-set decision a cut arm spells: ``PLACE@seam = cut`` keys and the cross-CTA half of
  ``REDUCE`` (``g2k``), alias-resolved (one key per seam actually cut), never an in-kernel knob; the fuse arm is
  no decision.
- ``routing`` — one row per PIECE of one decision on one parent, in the fragment's order, and the ``source`` that
  stored it (``measured`` for a bench here, else the golden scope or file it was imported from). A decision has
  no measurement of its own: its price on a context is the sum of its children's best rows there,
  all-or-nothing (:meth:`SearchDB.best_per_op_time`).
- ``taken`` — one row per kernel-set decision a golden file took: the context and the sizes it took it under, the
  parent, the ``arm`` and the ``source`` that holds it. A ``routing`` row says what a decision mints on any
  card; this says where a golden chose it, measured or not — the label the placement dataset reads.
- ``constant`` — one row per constant input a golden file's program binds on a target kernel at one set of sizes:
  the source that holds the program, the kernel, the sizes, the input's name and the constant's wire — what
  ``emmy db import`` copies out of the file's traced programs (``golden.evidence.import_constants``), so the
  placement export walks the forks a deploy sees without re-opening the file. A compile never writes or reads it.
- ``perf`` — one measurement per COMPILABLE kernel variant per context: the kernel, the sizes its symbolic
  dims were benched at (``bindings``, ``{}`` static), its schedule row, ``status``, the median of its timed
  launches (``us``), a ``bench_fail`` row's ``error`` and ``source`` (``measured``, or the golden file or freeze
  it was imported from, ``golden:<digest>`` / ``freeze:<digest>``). No route rows, no whole-slice totals, no
  kernel-set verdicts.
- ``source`` — one row per import this instance holds: a golden-shaped file ``emmy db import`` read (under context
  ``0``), or the golden scope a compile imported under one context. What an import asks before it reads a file
  again.

Readers see a FLAT :class:`PerfRow`: the context's columns, the kernel's exact identity, the sizes and the
schedule row (``knobs``). The joins live here and nowhere else.

Nothing migrates. A file whose tables have other columns than this DDL was written by another emmy: a
writer open re-creates every table empty — the rows are regenerable (a tune DB is benched again, a dataset DB
re-imports with ``emmy db import --fresh``) — and a read-only open refuses the file. Foreign keys are
enforced on every connection (``PRAGMA foreign_keys = ON``, off only while the re-create drops tables).

Concurrency: opened in WAL mode so parallel benches can read while one writes. The connection is kept
open for the DB's lifetime; callers can share one ``SearchDB`` instance across threads (sqlite3 handles
locking).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.search.dataset.kernel import KernelDef
from emmy.compiler.structural import digest
from emmy.compiler.wire import Wire

if TYPE_CHECKING:
    from emmy.compiler.context import Context

logger = logging.getLogger(__name__)


def knobs_json(knobs) -> str:
    """The one text spelling of a knob row — sorted keys, compact — that the DB stores and digests.
    Native JSON values only: a value the encoder cannot spell raises instead of being stringified,
    because the spelling identifies a row and a lossy fallback would let two writers mint two rows
    for one."""
    return json.dumps(dict(knobs), sort_keys=True, separators=(",", ":"), allow_nan=False)


def row_text(knobs) -> str:
    """A knob row as the text the DB stores it as and keys it by: the knobs as strings — a knob's value is its
    spelling, so ``4`` and ``"4"`` are one row — in :func:`knobs_json`'s spelling."""
    return knobs_json({str(name): str(value) for name, value in dict(knobs).items()})


def is_placement_knob(name: str, value) -> bool:
    """Whether a knob is a graph decision rather than an in-kernel schedule choice."""
    family = family_of(str(name))
    if family in {"PLACE", "LAYOUT"}:
        return True
    if family != "REDUCE":
        return False
    from emmy.compiler.pipeline.search.pins import parse_reduce  # noqa: PLC0415

    plan = parse_reduce(value)
    return plan is not None and plan.needs_split


@dataclass(frozen=True)
class PerfRow:
    """One measurement, flat: the context's columns, the kernel (its exact identity), the sizes it was benched at,
    ``knobs`` — the schedule row it ran with — and ``us``, the median of its timed launches in microseconds (for a
    ``bench_fail`` row the budget it overran, beside the failure text in ``error``).

    ``arch`` is the context's target (``Context.arch``), whose compute capability :attr:`cc` reads in the ``H_cc``
    encoding (``major * 10 + minor``) on CUDA. ``precision`` is the row's effective precision decisions, the one
    carrier of its arithmetic regime; a row handed in with some omitted is keyed by the regime they spell when it
    enters the DB."""

    gpu: str
    arch: str
    kernel: str
    bindings: dict
    knobs: dict
    backend: str
    status: str
    us: float
    error: str | None = None
    source: str = "measured"
    cold_cache: bool = False
    precision: dict[str, bool] = field(default_factory=dict)

    @property
    def cc(self) -> int:
        """The compute capability of a CUDA row's target, in the ``H_cc`` encoding."""
        return _cc(self.arch)


@dataclass(frozen=True)
class RoutingRow(Wire):
    """One kernel-set decision on one parent: the arm's knobs (``PLACE@seam: cut`` keys, or a cross-CTA
    ``REDUCE`` half) and the pieces it minted, in the fragment's order. The DB names the parent and the pieces by
    exact identity; a golden file stores the same row with each kernel pointed at by its ``ref`` in the file."""

    parent: str | int
    arm: dict
    children: tuple[str | int, ...]


def _ddl(statement: str) -> str:
    return " ".join(statement.split())


# Each table's DDL, as sqlite keeps it: a file is this module's when its tables were created by these statements.
_DDL = {
    "kernel": _ddl("""
        CREATE TABLE kernel (
            exact_identity  TEXT PRIMARY KEY,
            loop_ir         TEXT NOT NULL,
            formed          INTEGER NOT NULL
        )"""),
    "context": _ddl("""
        CREATE TABLE context (
            id          INTEGER PRIMARY KEY,
            backend     TEXT NOT NULL,
            gpu_name    TEXT NOT NULL,
            arch        TEXT NOT NULL,
            cold_cache  INTEGER NOT NULL,
            precision   TEXT NOT NULL,
            UNIQUE (backend, gpu_name, arch, cold_cache, precision)
        )"""),
    "source": _ddl("""
        CREATE TABLE source (
            name     TEXT NOT NULL,
            context  INTEGER NOT NULL,
            PRIMARY KEY (name, context)
        )"""),
    "routing": _ddl("""
        CREATE TABLE routing (
            parent    TEXT NOT NULL REFERENCES kernel (exact_identity),
            arm       TEXT NOT NULL,
            position  INTEGER NOT NULL,
            child     TEXT NOT NULL REFERENCES kernel (exact_identity),
            source    TEXT NOT NULL,
            PRIMARY KEY (parent, arm, position)
        )"""),
    "taken": _ddl("""
        CREATE TABLE taken (
            context   INTEGER NOT NULL REFERENCES context (id),
            kernel    TEXT NOT NULL REFERENCES kernel (exact_identity),
            bindings  TEXT NOT NULL,
            arm       TEXT NOT NULL,
            source    TEXT NOT NULL,
            PRIMARY KEY (context, kernel, bindings, arm)
        )"""),
    "constant": _ddl("""
        CREATE TABLE constant (
            source    TEXT NOT NULL,
            kernel    TEXT NOT NULL REFERENCES kernel (exact_identity),
            bindings  TEXT NOT NULL,
            name      TEXT NOT NULL,
            wire      TEXT NOT NULL,
            PRIMARY KEY (source, kernel, bindings, name)
        )"""),
    "perf": _ddl("""
        CREATE TABLE perf (
            context   INTEGER NOT NULL REFERENCES context (id),
            kernel    TEXT NOT NULL REFERENCES kernel (exact_identity),
            bindings  TEXT NOT NULL,
            schedule  TEXT NOT NULL,
            status    TEXT NOT NULL,
            us        REAL NOT NULL,
            error     TEXT,
            source    TEXT NOT NULL,
            PRIMARY KEY (context, kernel, bindings, schedule)
        )"""),
}
# ``perf_failed`` holds the failed rows alone, which is what :meth:`SearchDB.failed` scans.
_INDEXES = ("CREATE INDEX routing_child ON routing (child)", "CREATE INDEX perf_failed ON perf (context, kernel) WHERE status != 'ok'")
# Tables an older emmy wrote that nothing reads any more, dropped alongside the rest on a re-create.
_OBSOLETE_TABLES = (
    "loop_op",
    "tile_op",
    "kernel_op",
    "cuda_op",
    "lowering",
    "kernel_set",
    "kernel_feature",
    "schedule",
    "schedule_knob",
    "placement",
    "placement_knob",
)
#: The schema and cached identities (``PRAGMA user_version``): kernel wire, exact kernel identity and context key, and
#: what a context's regime measures. A file holding another is re-created; bump when any changes. The DB tests pin
#: kernel and context identities.
VERSION = 19  # A knob row is a text column; a perf row keeps its median alone; routing rows and imports name a source.

# What one perf read selects, in ``_row_to_perf`` order: the context's columns, then the row's.
_PERF_SEL = "c.gpu_name, c.arch, c.backend, p.kernel, p.bindings, p.schedule, p.status, p.us, p.error, p.source, c.cold_cache, c.precision"
_PERF_FROM = "FROM perf p JOIN context c ON c.id = p.context"


def _cc(arch: str) -> int:
    """The ``H_cc`` encoding of a context's target: ``sm_120`` -> 120. Only the CUDA spelling has one."""
    if not arch.startswith("sm_") or not arch[3:].isdigit():
        raise ValueError(f"no compute capability in the context target {arch!r}")
    return int(arch[3:])


def _schedule_row(knobs: dict) -> dict:
    """``knobs`` as the schedule row a ``perf`` row stores. A placement knob here is an error — a row spelling a cut
    or a cross-CTA split is a kernel-set decision, not a measurement of one kernel."""
    for name, value in knobs.items():
        if is_placement_knob(name, value):
            raise ValueError(f"{name}={value!r} is a placement knob: a kernel-set decision is a routing row, not a perf row")
    return {str(name): value for name, value in knobs.items()}


def _wire_json(wire: dict) -> str:
    return json.dumps(wire, sort_keys=True, separators=(",", ":"))


class SearchDB:
    """Persistent store of compiled kernels, the decisions that minted them, and their measurements.

    Pass ``path=None`` for an in-memory database (default — keeps tests hermetic; tuning runs pass an
    explicit path like ``~/.cache/emmy/autotune.db``)."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        if path is None:
            self._conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
            self._create_tables()
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One process at a time opens the file. Several `emmy` commands opening one file within
        # milliseconds of each other (the suite's CLI subprocesses) would otherwise each see the tables
        # missing and collide on CREATE TABLE, and a fresh file cannot switch to WAL while another
        # connection is mid-transaction.
        with self.exclusive():
            self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_tables()

    @property
    def path(self) -> Path | None:
        """The backing file, ``None`` for an in-memory DB."""
        return self._path

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """One process at a time, through the lock beside the file (nothing to hold for an in-memory DB):
        held for the open, and by ``golden.evidence`` for a golden import."""
        if self._path is None:
            yield
            return
        with open(f"{self._path}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """The writes inside as one transaction: they commit together, or none does when an error leaves the block.
        A bulk import writes thousands of rows, and the connection otherwise commits each statement on its own
        (``isolation_level=None``), each commit a sync of the write-ahead log."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _create_tables(self) -> None:
        """The tables, created where missing; a file another emmy wrote is re-created empty."""
        self._conn.execute("PRAGMA foreign_keys = OFF")
        if self._mismatched():
            logger.warning("%s: tables written by another emmy — re-created empty (their rows are regenerable)", self._path)
            for table in (*_DDL, *_OBSOLETE_TABLES):
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        if not self._tables():
            for statement in (*_DDL.values(), *_INDEXES):
                self._conn.execute(statement)
            self._conn.execute(f"PRAGMA user_version = {VERSION}")
        self._conn.execute("PRAGMA foreign_keys = ON")

    def _tables(self) -> dict[str, str]:
        """Which of this module's tables the file holds, each as the statement that created it — none in a fresh or
        foreign file."""
        held = dict(self._conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'"))
        return {table: _ddl(held[table]) for table in (*_DDL, *_OBSOLETE_TABLES) if table in held}

    def _mismatched(self) -> bool:
        """Whether the file's tables are not exactly this DDL's: a table created by another statement or one no
        emmy reads any more (a file another emmy wrote), some of the tables without the rest (a creation that was
        interrupted), or kernel rows another emmy wrote (:data:`VERSION`) — any of them would fail on the first
        read, or answer it with a key this emmy does not compute."""
        tables = self._tables()
        return bool(tables) and (tables != _DDL or self._conn.execute("PRAGMA user_version").fetchone()[0] != VERSION)

    @classmethod
    def for_compile(cls, path: Path | str) -> SearchDB | None:
        """The DB a compile picks from, made when absent (the golden rows in scope are imported into it), or
        ``None`` where the file cannot be made or opened — a read-only cache directory — so the compile picks
        from an in-memory instance instead, as one given no DB does."""
        try:
            return cls(path=Path(path))
        except (OSError, sqlite3.OperationalError) as exc:
            logger.warning("tune DB %s cannot be opened (%s); picking from an in-memory instance", path, exc)
            return None

    @classmethod
    def open_readonly(cls, path: Path | str) -> SearchDB:
        """Open an existing DB **read-only** — no schema creation, no table drop, no WAL pragma — so a
        read-side consumer (``eval``, the dataset import) never contends with a concurrent writer
        or mutates the file. The read methods work; any write raises (the connection is ``?mode=ro``).
        Raises ``sqlite3.OperationalError`` if the file is absent. A non-sqlite file fails HERE with a
        named reason (sqlite itself defers header validation to the first query, which would surface as a
        bare ``DatabaseError`` deep inside a PRAGMA) — the foreseeable case being a measurement freeze
        directory's file handed over by mistake. A file whose tables another emmy wrote is refused too: a
        reader cannot re-create them."""
        p = Path(path)
        if p.is_file():
            with p.open("rb") as fh:
                magic = fh.read(16)
            if magic and magic != b"SQLite format 3\x00":  # empty file = valid empty DB
                raise RuntimeError(f"{p} is not a sqlite database — a measurement freeze is read by `emmy db import`")
        self = cls.__new__(cls)
        self._path = p
        self._conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        if self._mismatched():
            self._conn.close()
            raise RuntimeError(f"{p}: tables written by another emmy — bench it again, or `emmy db import --fresh` a dataset DB")
        self._conn.execute("PRAGMA foreign_keys = ON")
        return self

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    @staticmethod
    def _regime(ctx: Context) -> tuple[str, str, bool, str]:
        """The context columns a measurement under ``ctx`` is keyed by: the card, the target, the cache mode and the
        effective precision decisions."""
        return ctx.hardware_id(), ctx.arch(), ctx.cold_cache, knobs_json(ctx.precision_pins())

    def _context_id(self, backend: str, gpu_name: str, arch: str, cold_cache: bool, precision: str, *, create: bool) -> int | None:
        key = (backend, gpu_name, arch, cold_cache, precision)
        row = self._conn.execute(
            "SELECT id FROM context WHERE backend = ? AND gpu_name = ? AND arch = ? AND cold_cache = ? AND precision = ?",
            key,
        ).fetchone()
        if row is not None:
            return row[0]
        if not create:
            return None
        return self._conn.execute(
            "INSERT INTO context (backend, gpu_name, arch, cold_cache, precision) VALUES (?, ?, ?, ?, ?)", key
        ).lastrowid

    def _context_of(self, ctx: Context, *, create: bool = False) -> int | None:
        """The id of the ``context`` row ``ctx`` names — its backend, card and regime — or ``None`` when there is
        none and it is not to be made."""
        return self._context_id(ctx.backend_name, *self._regime(ctx), create=create)

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def record_kernel(self, row: KernelDef) -> None:
        """Store a kernel's definition under its exact identity. A kernel already stored keeps its row (the
        definition is the same)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO kernel (exact_identity, loop_ir, formed) VALUES (?, ?, ?)",
            (row.exact_identity, _wire_json(row.loop_ir), int(row.formed)),
        )

    def kernel(self, exact_identity: str) -> KernelDef | None:
        """The stored definition of the kernel ``exact_identity`` names, or ``None``."""
        found = self._conn.execute("SELECT loop_ir, formed FROM kernel WHERE exact_identity = ?", (exact_identity,)).fetchone()
        return None if found is None else KernelDef(json.loads(found[0]), formed=bool(found[1])).keyed(exact_identity)

    def iter_kernels(self) -> Iterator[KernelDef]:
        for exact, loop_ir, formed in self._conn.execute(
            "SELECT exact_identity, loop_ir, formed FROM kernel ORDER BY exact_identity"
        ).fetchall():
            yield KernelDef(json.loads(loop_ir), formed=bool(formed)).keyed(exact)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def record_routing(self, row: RoutingRow, *, source: str = "measured") -> None:
        """Store what one decision on one parent minted, one row per piece, under the ``source`` that stores it; a
        later splice of the same decision (a compiler that now mints other pieces) replaces them. A decision a bench
        here recorded stays ``measured`` whatever imports it again, so no import's :meth:`forget` lets it go. The
        parent and the pieces must be kernel rows already."""
        if not row.arm or not all(is_placement_knob(k, v) for k, v in row.arm.items()):
            raise ValueError(f"a routing row's arm holds placement knobs only, got {row.arm!r}")
        arm = row_text(row.arm)
        held = self._conn.execute("SELECT source FROM routing WHERE parent = ? AND arm = ? LIMIT 1", (row.parent, arm)).fetchone()
        if held is not None and held[0] == "measured":
            source = "measured"
        self._conn.execute("DELETE FROM routing WHERE parent = ? AND arm = ?", (row.parent, arm))
        self._conn.executemany(
            "INSERT INTO routing (parent, arm, position, child, source) VALUES (?, ?, ?, ?, ?)",
            [(row.parent, arm, i, child, source) for i, child in enumerate(row.children)],
        )

    def iter_routing(self) -> Iterator[RoutingRow]:
        grouped: dict[tuple[str, str], list[str]] = {}
        for parent, arm, child in self._conn.execute("SELECT parent, arm, child FROM routing ORDER BY parent, arm, position").fetchall():
            grouped.setdefault((parent, arm), []).append(child)
        for (parent, arm), children in grouped.items():
            yield RoutingRow(parent=parent, arm=json.loads(arm), children=tuple(children))

    def record_taken(self, ctx: Context, kernel: str, *, bindings: dict, arm: dict, source: str) -> None:
        """Mark ``arm`` as a kernel-set decision ``source`` took on ``kernel`` under ``ctx`` at ``bindings``. A golden
        file's decision stays the file's: a freeze repeating it never takes it over, as with a schedule row
        (:meth:`record_perf`), since only a golden's decisions label the placement pools."""
        key = (self._context_of(ctx, create=True), kernel, knobs_json(bindings), row_text(arm))
        existing = self._conn.execute(
            "SELECT source FROM taken WHERE context = ? AND kernel = ? AND bindings = ? AND arm = ?", key
        ).fetchone()
        if existing is not None and existing[0].startswith("golden:") and source.startswith("freeze:"):
            return
        self._conn.execute("INSERT OR REPLACE INTO taken (context, kernel, bindings, arm, source) VALUES (?, ?, ?, ?, ?)", (*key, source))

    def iter_taken(self) -> Iterator[tuple[str, int, str, dict, dict, bool, dict, str]]:
        """Every decision a source took (:meth:`record_taken`), as ``(gpu, cc, kernel, bindings, arm, cold_cache,
        precision, source)`` in content order — a kernel's decisions at one set of sizes by the digest of their
        arms — on CUDA contexts: ``cc`` is a compute capability."""
        rows = self._conn.execute(
            "SELECT c.gpu_name, c.arch, t.kernel, t.bindings, t.arm, c.cold_cache, c.precision, t.source FROM taken t "
            "JOIN context c ON c.id = t.context WHERE c.backend = 'cuda'"
        ).fetchall()
        for gpu, arch, kernel, bindings, arm, cold_cache, precision, source in sorted(rows, key=lambda row: (*row[:4], digest(row[4]))):
            yield (gpu, _cc(arch), kernel, json.loads(bindings), json.loads(arm), bool(cold_cache), json.loads(precision), source)

    # ------------------------------------------------------------------
    # Constant
    # ------------------------------------------------------------------

    def record_constants(self, source: str, kernel: str, *, bindings: dict, constants: dict[str, dict]) -> None:
        """Store the constant inputs ``source``'s program binds on ``kernel`` at ``bindings``, each wire by input name
        (``golden.evidence.import_constants``); a later import of the same source replaces them."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO constant (source, kernel, bindings, name, wire) VALUES (?, ?, ?, ?, ?)",
            [(source, kernel, knobs_json(bindings), name, _wire_json(wire)) for name, wire in constants.items()],
        )

    def constants(self) -> dict[tuple[str, str, str], dict[str, dict]]:
        """Every stored constant input, each wire by input name, under ``(source, kernel, bindings)`` — the sizes
        spelled as :func:`knobs_json` spells them, which is how a placement pool asks for its own."""
        out: dict[tuple[str, str, str], dict[str, dict]] = {}
        for source, kernel, bindings, name, wire in self._conn.execute(
            "SELECT source, kernel, bindings, name, wire FROM constant ORDER BY source, kernel, bindings, name"
        ):
            out.setdefault((source, kernel, bindings), {})[name] = json.loads(wire)
        return out

    # ------------------------------------------------------------------
    # Perf — write
    # ------------------------------------------------------------------

    def record_perf(
        self,
        ctx: Context,
        kernel: str,
        *,
        bindings: dict,
        knobs: dict,
        status: str,
        us: float,
        error: str | None = None,
        source: str = "measured",
    ) -> None:
        """Record one measurement taken under ``ctx``: keyed by the context ``ctx`` names, and upserted through
        :meth:`record_perf_row`. ``knobs`` is the schedule row the kernel ran with and ``us`` the median of its
        timed launches. ``error`` is the failure text of a ``bench_fail`` row (whitespace-collapsed, truncated), so
        a reader of the file learns why a kernel is disqualified without the bench's log. ``source`` names where
        the row came from: a live bench, or the golden scope or file it was imported from (``golden:<digest>``)."""
        gpu, arch, cold_cache, precision = self._regime(ctx)
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self.record_perf_row(
            PerfRow(
                gpu=gpu,
                arch=arch,
                cold_cache=cold_cache,
                precision=json.loads(precision),
                kernel=kernel,
                bindings=dict(bindings),
                knobs=dict(knobs),
                backend=ctx.backend_name,
                status=status,
                us=us,
                error=error,
                source=source,
            )
        )

    def record_perf_row(self, row: PerfRow) -> None:
        """Upsert one measurement — a live one (:meth:`record_perf`) or a golden's. The row written last
        wins: a schedule benched again takes the new number, faster or slower, and of two imports holding
        the same key the later one's row stays. Three rows are never replaced: a row measured here
        (``source`` ``measured``) by an imported one — the import is a cache fill, and the local row is the
        one copy of what this machine measured; a golden file's row by a freeze's — the golden readers read
        ``golden:`` rows alone, so a freeze repeating a golden's schedule would otherwise take the pool out
        of the fit; and a clean row by a ``bench_fail`` — a failure is a flag, not a measurement. The kernel
        must be a ``kernel`` row already. The row's ``precision`` enters as the regime it spells, every
        decision written (``space.precision_pins``)."""
        from emmy.compiler.pipeline.search.space import precision_pins  # noqa: PLC0415

        context = self._context_id(row.backend, row.gpu, row.arch, row.cold_cache, knobs_json(precision_pins(row.precision)), create=True)
        key = (context, row.kernel, knobs_json(row.bindings), row_text(_schedule_row(row.knobs)))
        existing = self._conn.execute(
            "SELECT status, source FROM perf WHERE context = ? AND kernel = ? AND bindings = ? AND schedule = ?", key
        ).fetchone()
        if existing is not None:
            prev_status, prev_source = existing
            if prev_source == "measured" and row.source != "measured":
                return  # an import never replaces a measurement taken here
            if prev_source.startswith("golden:") and row.source.startswith("freeze:"):
                return  # a golden file's row is the file's; a freeze repeating its schedule never displaces it
            if prev_status == "ok" and row.status != "ok":
                return  # a failure never replaces a clean measurement
        self._conn.execute(
            "INSERT OR REPLACE INTO perf (context, kernel, bindings, schedule, status, us, error, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (*key, row.status, row.us, row.error, row.source),
        )

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def record_source(self, name: str, ctx: Context | None = None) -> None:
        """Note that this instance holds the import ``name``: a golden-shaped file the import has read, whatever
        became of its rows, or — with ``ctx`` — the golden scope a compile imported under that card and regime.
        What a later import asks (:meth:`sources`) before it reads the files again."""
        self._conn.execute(
            "INSERT OR IGNORE INTO source (name, context) VALUES (?, ?)", (name, 0 if ctx is None else self._context_of(ctx, create=True))
        )

    def sources(self, ctx: Context | None = None) -> set[str]:
        """The imports this instance holds (:meth:`record_source`): the files, or with ``ctx`` the golden scopes
        imported under that card and regime."""
        context = 0 if ctx is None else self._context_of(ctx)
        return {name for [name] in self._conn.execute("SELECT name FROM source WHERE context = ?", (context,))}

    def forget(self, ctx: Context, source_prefix: str) -> None:
        """Let go of what the sources starting with ``source_prefix`` imported under ``ctx``'s card and regime —
        their ``perf`` and ``taken`` rows and the note that they were imported — and of every ``routing`` row such a
        source stored that no context holds an import of any more: how the cache drops a golden scope before the
        current one is imported, since a row or a decision the current files no longer hold would otherwise stay.
        A decision the scopes of two cards hold goes with the scope that stored it last."""
        context, like = self._context_of(ctx), source_prefix + "%"
        self._conn.execute("DELETE FROM perf WHERE context = ? AND source LIKE ?", (context, like))
        self._conn.execute("DELETE FROM taken WHERE context = ? AND source LIKE ?", (context, like))
        self._conn.execute("DELETE FROM source WHERE context = ? AND name LIKE ?", (context, like))
        self._conn.execute("DELETE FROM routing WHERE source LIKE ? AND source NOT IN (SELECT name FROM source)", (like,))

    # ------------------------------------------------------------------
    # Perf — read
    # ------------------------------------------------------------------

    def perf_sources(self) -> dict[str, int]:
        """How many ``perf`` rows each source contributed — what a report over this instance names as its data."""
        return dict(self._conn.execute("SELECT source, COUNT(*) FROM perf GROUP BY 1 ORDER BY 1"))

    def measured(self, ctx: Context, kernel: str, *, bindings: dict | None) -> list[tuple[dict, float]]:
        """The clean measurements of ``kernel`` a compile under ``ctx`` may deploy at ``bindings``, the sizes it binds
        the kernel's symbolic dims to, as ``(schedule row, µs)`` — this instance's own rows and the golden rows
        imported among them; with ``None``, its measurements at every size. A row measured at other sizes timed
        another run of the kernel: native's attention, priced per range of windows, is one kernel measured at two
        windows, and its unsplit fold at 131,072 must not rank at its 4,096 time. A row that names no size (a dynamic
        kernel's, recorded before rows named one) stays evidence at every size. Rows another card, regime or backend
        measured are of another context."""
        sized, sizes = ("", ()) if bindings is None else (" AND bindings IN (?, '{}')", (knobs_json(bindings),))
        rows = self._conn.execute(
            f"SELECT schedule, us FROM perf WHERE context = ? AND kernel = ?{sized} AND status = 'ok' AND us > 0 "  # noqa: S608
            "ORDER BY bindings = '{}', bindings, schedule",
            (self._context_of(ctx), kernel, *sizes),
        )
        return [(json.loads(schedule), us) for schedule, us in rows]

    def failed(self, ctx: Context) -> set[str]:
        """The kernels whose every measured variant under ``ctx`` failed. A failed row is evidence a ranker cannot
        use — its time is the budget the bench overran, not a speed — so it disqualifies instead, and only where no
        variant of the kernel ran clean: one clean row means the kernel is realizable and merely has bad rows."""
        rows = self._conn.execute(
            "SELECT DISTINCT kernel FROM perf p WHERE context = ? AND status != 'ok' AND NOT EXISTS "
            "(SELECT 1 FROM perf o WHERE o.context = p.context AND o.kernel = p.kernel AND o.status = 'ok')",
            (self._context_of(ctx),),
        )
        return {kernel for [kernel] in rows}

    def iter_perf_rows(self, *, backend: str | None = "cuda") -> Iterator[PerfRow]:
        """Every ``perf`` row, of every card and regime — the view the measurement-data readers and
        ``emmy db`` read. ``backend=None`` spans every backend."""
        sql = f"SELECT {_PERF_SEL} {_PERF_FROM}"  # noqa: S608
        params: list = []
        if backend is not None:
            sql += " WHERE c.backend = ?"
            params.append(backend)
        for gpu, arch, backend_name, kernel, bindings, schedule, status, us, error, source, cold_cache, precision in self._conn.execute(
            sql, params
        ).fetchall():
            yield PerfRow(
                gpu=gpu,
                arch=arch,
                cold_cache=bool(cold_cache),
                precision=json.loads(precision),
                kernel=kernel,
                bindings=json.loads(bindings),
                knobs=json.loads(schedule),
                backend=backend_name,
                status=status,
                us=us,
                error=error,
                source=source,
            )

    def _best_leaf(self, context: int | None, kernel: str, bindings: dict) -> float | None:
        """The fastest ``ok`` median of ``kernel`` at ``bindings`` under one context row, or ``None``."""
        [us] = self._conn.execute(
            "SELECT MIN(us) FROM perf WHERE context = ? AND kernel = ? AND bindings = ? AND status = 'ok'",
            (context, kernel, knobs_json(bindings)),
        ).fetchone()
        return us

    def has_perf(self, ctx: Context) -> bool:
        """Whether ``ctx``'s context holds any clean measurement at all — what a pick asks before it prices
        kernel-set arms from their pieces' rows, since pricing an arm builds its pieces."""
        return (
            self._conn.execute("SELECT 1 FROM perf WHERE context = ? AND status = 'ok' LIMIT 1", (self._context_of(ctx),)).fetchone()
            is not None
        )

    def priced_arms(self, ctx: Context, kernel: str, *, bindings: dict) -> list[tuple[dict, float]]:
        """Every kernel-set decision stored on ``kernel`` that ``ctx`` can price, as ``(arm, us)``: the sum
        of its pieces' best times there, each piece at its own projection of ``bindings`` onto the symbolic
        dims it kept, all-or-nothing — a piece with no price leaves that decision out. A piece is priced the
        way its parent is (:meth:`best_per_op_time`): from its own rows when it compiled, from its own pieces
        when it was cut again, so a nested cut prices through and no cut piece needs a row. A decision has no
        measurement of its own; this is its price wherever one is read."""
        from emmy.compiler.wire import symbolic_vars  # noqa: PLC0415

        pieces: dict[str, list[tuple[str, str]]] = {}
        for arm, child, wire in self._conn.execute(
            "SELECT r.arm, r.child, k.loop_ir FROM routing r JOIN kernel k ON k.exact_identity = r.child "
            "WHERE r.parent = ? ORDER BY r.arm, r.position",
            (kernel,),
        ):
            pieces.setdefault(arm, []).append((child, wire))
        out: list[tuple[dict, float]] = []
        for arm, children in pieces.items():
            total: float | None = 0.0
            for child, wire in children:
                projected = {v: bindings[v] for v in symbolic_vars(json.loads(wire)) if v in bindings}
                us = self.best_per_op_time(ctx, child, bindings=projected)
                if us is None:
                    total = None
                    break
                total += us
            if total is not None:
                out.append((json.loads(arm), total))
        return out

    def best_per_op_time(self, ctx: Context, kernel: str, *, bindings: dict) -> float | None:
        """The best measured median (us) of ``kernel`` at ``bindings`` under ``ctx``, or ``None`` when it has
        no clean measurement: its fastest ``ok`` row as a leaf, or the cheapest of its priced kernel-set
        decisions (:meth:`priced_arms`), whichever is smaller."""
        leaf = self._best_leaf(self._context_of(ctx), kernel, bindings)
        candidates = [us for us in (leaf, *(us for _arm, us in self.priced_arms(ctx, kernel, bindings=bindings))) if us is not None]
        return min(candidates) if candidates else None

    def evidence_digest(self, ctx: Context, kernels: Iterable[str]) -> str:
        """A digest of the evidence a compile under ``ctx`` that read ``kernels`` (exact identities:
        ``policy.greedy.evidence_reads``) picks from: their rows there, the decisions stored on them, the same of
        every piece those decisions reach, and whether the context holds any clean measurement at all, since a
        compile under one that holds none prices no kernel-set arm from its pieces and so looks none of them up.
        Nothing else in this instance can move that compile's picks."""
        context = self._context_of(ctx)
        reached, frontier, read = set(kernels), list(kernels), {}
        while frontier:
            kernel = frontier.pop()
            routes = self._conn.execute("SELECT arm, child FROM routing WHERE parent = ? ORDER BY arm, position", (kernel,)).fetchall()
            rows = self._conn.execute(
                "SELECT bindings, schedule, status, us FROM perf WHERE context = ? AND kernel = ? ORDER BY bindings, schedule",
                (context, kernel),
            ).fetchall()
            if routes or rows:
                read[kernel] = [routes, rows]
            frontier.extend(child for _, child in routes if child not in reached)
            reached.update(child for _, child in routes)
        return hashlib.sha256(json.dumps([self.has_perf(ctx), sorted(read.items())]).encode()).hexdigest()[:16]

    def drift(self) -> dict[str, int]:
        """The drift checks — each the count of rows that fail it: a row naming a row that is gone (the foreign
        keys, which a file written with them off can break); a context naming a card the GPU registry lost.
        Nothing decodes a stored wire: a tune DB is a cache, and a row the current code disagrees with is benched
        or imported again, never patched."""
        from emmy import gpu  # noqa: PLC0415

        return {
            "every row names the rows it references": len(self._conn.execute("PRAGMA foreign_key_check").fetchall()),
            "every CUDA context names a registry card": sum(
                gpu.by_name(n) is None for [n] in self._conn.execute("SELECT gpu_name FROM context WHERE backend = 'cuda'")
            ),
        }

    def decisions(self) -> list[tuple[str, dict]]:
        """Every stored kernel-set decision as ``(the parent's exact identity, the arm)`` — what offers a composed
        cut to a later compile of the parent kernel, without decoding any wire."""
        return [
            (parent, json.loads(arm)) for parent, arm in self._conn.execute("SELECT DISTINCT parent, arm FROM routing ORDER BY parent, arm")
        ]

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
