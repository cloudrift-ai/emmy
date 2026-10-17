"""SQLite-backed measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. One schema, several instances: the tune
DB (``EMMY_TUNE_DB``) is what compile reads and tune writes; a dataset instance — the file
``emmy db import --db PATH`` names — holds the same tables, and ``emmy db export`` turns it into the dataset the
measurement-data readers read.

Tables (the DDL is the reference):

- ``kernel`` — one row per compilable kernel, keyed by its EXACT identity (``identity_key(structural=False,
  with_io=True)``: the digest of the normalized body's form plus each buffer's dtype and hint-free shape),
  holding the kernel's Loop IR wire (``wire.kernel_wire``: the body it was formed from, which the tile lift
  takes back to the kernel — ``formed`` — or, for a piece carved from a twisted tree, its derived body, which
  only its parent's program reaches) and its C name. A piece a cut or a split minted is a row like any other,
  so the same kernel reached from two parents has one definition. The key is the one computed value the DB
  holds: a cache key, computed from the wire by whoever writes the row (a bench off its live kernel, an import
  off a golden's stored one) and valid under :data:`_VERSION`. Nothing else derived from a kernel is stored —
  its ``S_*`` stamps are computed from the wire where a row is featurized.
- ``context`` — one row per backend, card and regime: the card's product name (``Context.hardware_id``),
  the context's target as the backend spells it (``sm_120`` on CUDA — the regime's, never a kernel's
  ``sm_120a``; LLVM's name for the host CPU, ``znver5``, on the CPU backend, whose card is the CPU and its thread
  count), the compiler's opt level, its residual flags, effective precision decisions, and whether each
  launch was measured after L2 eviction.
- ``schedule`` / ``schedule_knob`` — one row per distinct schedule row: the in-kernel choices a leaf
  kernel was measured with (``WORK``, ``TILE``, ``STAGE``, ``RASTER``, the in-kernel half of ``REDUCE``),
  keyed by ``digest(knobs_json(row))``. Never a placement knob.
- ``placement`` / ``placement_knob`` — one row per distinct kernel-set decision a cut arm spells:
  ``PLACE@seam = cut`` keys and the cross-CTA half of ``REDUCE`` (``g2k``), alias-resolved (one row per
  seam actually cut). Never an in-kernel knob; the fuse arm is no placement.
- ``routing`` — one row per PIECE of one decision on one parent, in the fragment's order. A decision has
  no measurement of its own: its price on a context is the sum of its children's best rows there,
  all-or-nothing (:meth:`SearchDB.best_per_op_time`).
- ``taken`` — one row per kernel-set decision a golden file took: the context and the sizes it took it under, the
  parent, the ``placement`` and the ``source`` that holds it. A ``routing`` row says what a decision mints on any
  card; this says where a golden chose it, measured or not — the label the placement dataset reads.
- ``constant`` — one row per constant input a golden file's program binds on a target kernel at one set of sizes:
  the source that holds the program, the kernel, the sizes, the input's name and the constant's wire — what
  ``emmy db import`` copies out of the file's traced programs (``golden.evidence.import_constants``), so the
  placement export walks the forks a deploy sees without re-opening the file. A compile never writes or reads it.
- ``perf`` — one measurement per COMPILABLE kernel variant per context: the kernel, the sizes its symbolic
  dims were benched at (``bindings``, ``{}`` static), its schedule row, the statistics, ``captured``, a
  ``bench_fail`` row's ``error`` and ``source`` (``measured``, or the golden file or freeze it was imported from,
  ``golden:<digest>`` / ``freeze:<digest>``). No route rows, no whole-slice totals, no kernel-set verdicts.

Readers see a FLAT :class:`PerfRow`: the context's columns, the kernel's exact identity, the sizes and the
schedule row (``knobs``). The joins live here and nowhere else.

Nothing migrates. A file whose tables have other columns than this DDL was written by another emmy: a
writer open re-creates every table empty — the rows are regenerable (a tune DB re-tunes, a dataset DB
re-imports with ``emmy db import --fresh``) — and a read-only open refuses the file. Foreign keys are
enforced on every connection (``PRAGMA foreign_keys = ON``, off only while the re-create drops tables).

Concurrency: opened in WAL mode so parallel benches can read while one writes. The connection is kept
open for the DB's lifetime; callers can share one ``SearchDB`` instance across threads (sqlite3 handles
locking).
"""

from __future__ import annotations

import fcntl
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
class PerfStats:
    """Summary statistics over per-iter kernel latencies (microseconds)."""

    median: float
    min: float
    max: float
    mean: float
    variance: float
    n_samples: int


@dataclass(frozen=True)
class PerfRow:
    """One measurement, flat: the context's columns, the kernel (its exact identity), the sizes it was benched at,
    and ``knobs`` — the schedule row it ran with.

    ``captured``: the measurement ran under CUDA graph capture (pure GPU time); False = wall semantics
    including per-launch dispatch. On write, a captured measurement supersedes an uncaptured one for
    the same key — see :meth:`SearchDB.record_perf_row`. ``arch`` is the context's target (``Context.arch``), whose
    compute capability :attr:`cc` reads in the ``H_cc`` encoding (``major * 10 + minor``) on CUDA; ``error`` is a ``bench_fail`` row's
    failure text. ``precision`` is the row's effective precision decisions, the one carrier of its arithmetic
    regime (``flags`` is the residual nvcc flag string, never read for it); a row handed in with some omitted
    is keyed by the regime they spell when it enters the DB."""

    gpu: str
    arch: str
    opt: int
    flags: str
    kernel: str
    bindings: dict
    knobs: dict
    backend: str
    status: str
    stats: PerfStats
    measured_at: str
    captured: bool = False
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
    exact identity; a golden file stores the same row with each kernel named by its ``ref`` in the file."""

    parent: str
    arm: dict
    children: tuple[str, ...]


# Each table's DDL and the column set a file must have for this module to read it.
_DDL = {
    "source": """
        CREATE TABLE source (
            name  TEXT PRIMARY KEY
        )""",
    "kernel": """
        CREATE TABLE kernel (
            exact_identity       TEXT PRIMARY KEY,
            loop_ir              TEXT NOT NULL,
            kernel_name          TEXT NOT NULL,
            formed               INTEGER NOT NULL
        )""",
    "context": """
        CREATE TABLE context (
            id        INTEGER PRIMARY KEY,
            backend   TEXT NOT NULL,
            gpu_name  TEXT NOT NULL,
            arch      TEXT NOT NULL,
            opt       INTEGER NOT NULL,
            flags     TEXT NOT NULL,
            cold_cache INTEGER NOT NULL,
            precision TEXT NOT NULL,
            UNIQUE (backend, gpu_name, arch, opt, flags, cold_cache, precision)
        )""",
    "schedule": """
        CREATE TABLE schedule (
            id      INTEGER PRIMARY KEY,
            digest  TEXT NOT NULL UNIQUE
        )""",
    "schedule_knob": """
        CREATE TABLE schedule_knob (
            schedule  INTEGER NOT NULL REFERENCES schedule (id),
            name      TEXT NOT NULL,
            value     TEXT NOT NULL,
            PRIMARY KEY (schedule, name)
        )""",
    "placement": """
        CREATE TABLE placement (
            id      INTEGER PRIMARY KEY,
            digest  TEXT NOT NULL UNIQUE
        )""",
    "placement_knob": """
        CREATE TABLE placement_knob (
            placement  INTEGER NOT NULL REFERENCES placement (id),
            name       TEXT NOT NULL,
            value      TEXT NOT NULL,
            PRIMARY KEY (placement, name)
        )""",
    "routing": """
        CREATE TABLE routing (
            parent     TEXT NOT NULL REFERENCES kernel (exact_identity),
            placement  INTEGER NOT NULL REFERENCES placement (id),
            position   INTEGER NOT NULL,
            child      TEXT NOT NULL REFERENCES kernel (exact_identity),
            PRIMARY KEY (parent, placement, position)
        )""",
    "taken": """
        CREATE TABLE taken (
            context    INTEGER NOT NULL REFERENCES context (id),
            kernel     TEXT NOT NULL REFERENCES kernel (exact_identity),
            bindings   TEXT NOT NULL,
            placement  INTEGER NOT NULL REFERENCES placement (id),
            source     TEXT NOT NULL,
            PRIMARY KEY (context, kernel, bindings, placement)
        )""",
    "constant": """
        CREATE TABLE constant (
            source    TEXT NOT NULL,
            kernel    TEXT NOT NULL REFERENCES kernel (exact_identity),
            bindings  TEXT NOT NULL,
            name      TEXT NOT NULL,
            wire      TEXT NOT NULL,
            PRIMARY KEY (source, kernel, bindings, name)
        )""",
    "perf": """
        CREATE TABLE perf (
            context    INTEGER NOT NULL REFERENCES context (id),
            kernel     TEXT NOT NULL REFERENCES kernel (exact_identity),
            bindings   TEXT NOT NULL,
            schedule   INTEGER NOT NULL REFERENCES schedule (id),
            status     TEXT NOT NULL,
            latency_us_median    REAL NOT NULL,
            latency_us_min       REAL NOT NULL,
            latency_us_max       REAL NOT NULL,
            latency_us_mean      REAL NOT NULL,
            latency_us_variance  REAL NOT NULL,
            n_samples   INTEGER NOT NULL,
            measured_at TEXT NOT NULL,
            captured    INTEGER NOT NULL,
            error       TEXT,
            source      TEXT NOT NULL,
            PRIMARY KEY (context, kernel, bindings, schedule)
        )""",
}
_INDEXES = ("CREATE INDEX routing_child ON routing (child)",)
_COLS = {
    "source": ("name",),
    "kernel": ("exact_identity", "loop_ir", "kernel_name", "formed"),
    "context": ("id", "backend", "gpu_name", "arch", "opt", "flags", "cold_cache", "precision"),
    "schedule": ("id", "digest"),
    "schedule_knob": ("schedule", "name", "value"),
    "placement": ("id", "digest"),
    "placement_knob": ("placement", "name", "value"),
    "routing": ("parent", "placement", "position", "child"),
    "taken": ("context", "kernel", "bindings", "placement", "source"),
    "constant": ("source", "kernel", "bindings", "name", "wire"),
    "perf": (
        "context",
        "kernel",
        "bindings",
        "schedule",
        "status",
        "latency_us_median",
        "latency_us_min",
        "latency_us_max",
        "latency_us_mean",
        "latency_us_variance",
        "n_samples",
        "measured_at",
        "captured",
        "error",
        "source",
    ),
}
# Tables an older emmy wrote that nothing reads any more, dropped alongside the rest on a re-create.
_OBSOLETE_TABLES = ("loop_op", "tile_op", "kernel_op", "cuda_op", "lowering", "kernel_set", "kernel_feature")
#: The schema and cached identities (``PRAGMA user_version``): kernel wire, exact kernel identity and context key.
#: A file holding another is re-created; bump when any changes. The DB tests pin kernel and context identities.
_VERSION = 8  # The Affine normal form re-keys every kernel that sums, negates or scales a value by a constant.
# Drop order respects the foreign keys; create order is the reverse.
_DROP_ORDER = (
    "perf",
    "taken",
    "constant",
    "routing",
    "placement_knob",
    "placement",
    "schedule_knob",
    "schedule",
    "context",
    "kernel",
    "source",
)

# What one perf read selects, in ``_row_to_perf`` order: the context's columns, then the row's.
_PERF_SEL = (
    "c.gpu_name, c.arch, c.opt, c.flags, c.backend, p.kernel, p.bindings, p.schedule, p.status, "
    "p.latency_us_median, p.latency_us_min, p.latency_us_max, p.latency_us_mean, p.latency_us_variance, p.n_samples, "
    "p.measured_at, p.captured, p.error, p.source, c.cold_cache, c.precision"
)
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
        # The backing file (``None`` for an in-memory DB) — read by the deploy-side
        # ``_db_measured_index`` cache to key its process-wide memo on (path, mtime).
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

    def _create_tables(self) -> None:
        """The tables, created where missing; a file another emmy wrote is re-created empty."""
        self._conn.execute("PRAGMA foreign_keys = OFF")
        if self._mismatched():
            logger.warning("%s: tables written by another emmy — re-created empty (their rows are regenerable)", self._path)
            for table in _DROP_ORDER:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        for table in _OBSOLETE_TABLES:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        if not self._columns("perf"):
            for table in reversed(_DROP_ORDER):
                self._conn.execute(_DDL[table])
            for stmt in _INDEXES:
                self._conn.execute(stmt)
            self._conn.execute(f"PRAGMA user_version = {_VERSION}")
        self._conn.execute("PRAGMA foreign_keys = ON")

    def _columns(self, table: str) -> set[str]:
        """A table's columns — empty when there is no such table (a fresh or foreign file)."""
        return {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}

    def _mismatched(self) -> bool:
        """Whether the file's tables are not exactly this DDL's: a table with other columns (a file another
        emmy wrote), or some of the tables without the rest (a creation that was interrupted, or an emmy one
        table older), or kernel rows another emmy wrote (:data:`_VERSION`) — any of them would fail on the first
        read, or answer it with a key this emmy does not compute."""
        present = {table: self._columns(table) for table in _COLS}
        if any(cols and cols != set(_COLS[table]) for table, cols in present.items()):
            return True
        if any(present.values()) and self._conn.execute("PRAGMA user_version").fetchone()[0] != _VERSION:
            return True
        return 0 < sum(bool(cols) for cols in present.values()) < len(present)

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
        read-side consumer (``eval``, the dataset import) never contends with a concurrent ``tune`` writer
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
            raise RuntimeError(f"{p}: tables written by another emmy — re-tune it, or `emmy db import --fresh` a dataset DB")
        self._conn.execute("PRAGMA foreign_keys = ON")
        return self

    # ------------------------------------------------------------------
    # The dimension tables: context, schedule, placement
    # ------------------------------------------------------------------

    @staticmethod
    def _regime(ctx: Context) -> tuple[str, str, int, str, bool, str]:
        """The context columns a measurement under ``ctx`` is keyed by: the card, the target, the opt level
        and the residual compiler flags plus cache mode — one spelling of the regime however its flags were written
        (:func:`~emmy.compiler.context.split_opt_level`)."""
        from emmy.compiler.context import split_opt_level  # noqa: PLC0415

        opt, flags = split_opt_level(ctx.compile_flags)
        return ctx.hardware_id(), ctx.arch(), opt, flags, ctx.cold_cache, knobs_json(ctx.precision_pins())

    def _context_id(
        self, backend: str, gpu_name: str, arch: str, opt: int, flags: str, cold_cache: bool, precision: str, *, create: bool
    ) -> int | None:
        key = (backend, gpu_name, arch, opt, flags, cold_cache, precision)
        row = self._conn.execute(
            "SELECT id FROM context WHERE backend = ? AND gpu_name = ? AND arch = ? AND opt = ? "
            "AND flags = ? AND cold_cache = ? AND precision = ?",
            key,
        ).fetchone()
        if row is not None:
            return row[0]
        if not create:
            return None
        return self._conn.execute(
            "INSERT INTO context (backend, gpu_name, arch, opt, flags, cold_cache, precision) VALUES (?, ?, ?, ?, ?, ?, ?)", key
        ).lastrowid

    def _row_id(self, table: str, knobs: dict, *, create: bool) -> int | None:
        """The id of the ``schedule`` / ``placement`` row spelling ``knobs``, minted when absent: the row
        keyed by ``digest(knobs_json(knobs))`` over the knobs as strings — a knob's value is its spelling,
        so ``4`` and ``"4"`` are one row — its knobs one per row in the ``<table>_knob`` table."""
        knobs = {str(name): str(value) for name, value in knobs.items()}
        key = digest(knobs_json(knobs))
        row = self._conn.execute(f"SELECT id FROM {table} WHERE digest = ?", (key,)).fetchone()  # noqa: S608
        if row is not None:
            return row[0]
        if not create:
            return None
        rid = self._conn.execute(f"INSERT INTO {table} (digest) VALUES (?)", (key,)).lastrowid  # noqa: S608
        self._conn.executemany(
            f"INSERT INTO {table}_knob ({table}, name, value) VALUES (?, ?, ?)",  # noqa: S608
            [(rid, name, value) for name, value in knobs.items()],
        )
        return rid

    def _knobs_of(self, table: str, rid: int) -> dict:
        return dict(self._conn.execute(f"SELECT name, value FROM {table}_knob WHERE {table} = ?", (rid,)))  # noqa: S608

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def record_kernel(self, row: KernelDef) -> None:
        """Store a kernel's definition under its exact identity. A kernel already stored keeps its row (the
        definition is the same; the name only serves the per-kernel views)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO kernel (exact_identity, loop_ir, kernel_name, formed) VALUES (?, ?, ?, ?)",
            (row.exact_identity, _wire_json(row.loop_ir), row.name, int(row.formed)),
        )

    def kernel_names(self) -> dict[str, str]:
        """Every stored kernel's C name by exact identity — the per-kernel views' grouping key."""
        return dict(self._conn.execute("SELECT exact_identity, kernel_name FROM kernel"))

    def iter_kernels(self) -> Iterator[KernelDef]:
        for exact, loop_ir, name, formed in self._conn.execute(
            "SELECT exact_identity, loop_ir, kernel_name, formed FROM kernel ORDER BY exact_identity"
        ).fetchall():
            yield KernelDef(json.loads(loop_ir), name, formed=bool(formed)).keyed(exact)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def record_routing(self, row: RoutingRow) -> None:
        """Store what one decision on one parent minted, one row per piece; a later splice of the same
        decision (a compiler that now mints other pieces) replaces them. The parent and the pieces must
        be kernel rows already."""
        if not row.arm or not all(is_placement_knob(k, v) for k, v in row.arm.items()):
            raise ValueError(f"a routing row's arm holds placement knobs only, got {row.arm!r}")
        pid = self._row_id("placement", row.arm, create=True)
        self._conn.execute("DELETE FROM routing WHERE parent = ? AND placement = ?", (row.parent, pid))
        self._conn.executemany(
            "INSERT INTO routing (parent, placement, position, child) VALUES (?, ?, ?, ?)",
            [(row.parent, pid, i, child) for i, child in enumerate(row.children)],
        )

    def iter_routing(self) -> Iterator[RoutingRow]:
        rows = self._conn.execute("SELECT parent, placement, position, child FROM routing ORDER BY parent, placement, position").fetchall()
        grouped: dict[tuple[str, int], list[str]] = {}
        for parent, pid, _position, child in rows:
            grouped.setdefault((parent, pid), []).append(child)
        for (parent, pid), children in grouped.items():
            yield RoutingRow(parent=parent, arm=self._knobs_of("placement", pid), children=tuple(children))

    def record_taken(self, ctx: Context, kernel: str, *, bindings: dict, arm: dict, source: str) -> None:
        """Mark ``arm`` as a kernel-set decision ``source`` took on ``kernel`` under ``ctx`` at ``bindings``."""
        context = self._context_id(ctx.backend_name, *self._regime(ctx), create=True)
        self._conn.execute(
            "INSERT OR REPLACE INTO taken (context, kernel, bindings, placement, source) VALUES (?, ?, ?, ?, ?)",
            (context, kernel, knobs_json(bindings), self._row_id("placement", arm, create=True), source),
        )

    def iter_taken(self) -> Iterator[tuple[str, int, str, dict, dict, bool, dict, str]]:
        """Every decision a source took (:meth:`record_taken`), as ``(gpu, cc, kernel, bindings, arm, cold_cache,
        precision, source)`` in content order, on CUDA contexts: ``cc`` is a compute capability."""
        rows = self._conn.execute(
            "SELECT c.gpu_name, c.arch, t.kernel, t.bindings, t.placement, c.cold_cache, c.precision, t.source FROM taken t "
            "JOIN context c ON c.id = t.context JOIN placement p ON p.id = t.placement WHERE c.backend = 'cuda' "
            "ORDER BY 1, 2, 3, 4, p.digest"
        ).fetchall()
        for gpu, arch, kernel, bindings, pid, cold_cache, precision, source in rows:
            yield (
                gpu,
                _cc(arch),
                kernel,
                json.loads(bindings),
                self._knobs_of("placement", pid),
                bool(cold_cache),
                json.loads(precision),
                source,
            )

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
        stats: PerfStats,
        captured: bool = False,
        error: str | None = None,
        source: str = "measured",
    ) -> None:
        """Record one measurement taken now under ``ctx``: keyed by the context ``ctx`` names, and upserted
        through :meth:`record_perf_row`. ``knobs`` is the schedule row the kernel ran with.
        ``error`` is the failure text for a ``bench_fail`` row (whitespace-collapsed, truncated) so failure
        forensics (``eval failures``) need no tune-log grepping. ``source`` names where the row came from: a
        live bench, or the golden file it was imported from (``golden:<digest>``)."""
        gpu, arch, opt, flags, cold_cache, precision = self._regime(ctx)
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self.record_perf_row(
            PerfRow(
                gpu=gpu,
                arch=arch,
                opt=opt,
                flags=flags,
                cold_cache=cold_cache,
                precision=json.loads(precision),
                kernel=kernel,
                bindings=dict(bindings),
                knobs=dict(knobs),
                backend=ctx.backend_name,
                status=status,
                stats=stats,
                measured_at=datetime.now(UTC).isoformat(),
                captured=captured,
                error=error,
                source=source,
            )
        )

    def record_perf_row(self, row: PerfRow) -> None:
        """Upsert one measurement — a live one (:meth:`record_perf`) or a golden's. The row written last
        wins: a schedule benched again takes the new numbers, faster or slower, and of two imports holding
        the same key the later one's row stays. Three rows are never replaced: a row measured here
        (``source`` ``measured``) by an imported one — the import is a cache fill, and the local row is the
        one copy of what this machine measured; a golden file's row by a freeze's — the golden readers read
        ``golden:`` rows alone, so a freeze repeating a golden's schedule would otherwise take the pool out
        of the fit; and a clean row by a ``bench_fail`` — a failure is a flag, not a measurement. The kernel
        must be a ``kernel`` row already. The row's ``precision`` enters as the regime it spells, every
        decision written (``space.precision_pins``)."""
        from emmy.compiler.pipeline.search.space import precision_pins  # noqa: PLC0415

        context = self._context_id(
            row.backend, row.gpu, row.arch, row.opt, row.flags, row.cold_cache, knobs_json(precision_pins(row.precision)), create=True
        )
        schedule = self._row_id("schedule", _schedule_row(row.knobs), create=True)
        key = (context, row.kernel, knobs_json(row.bindings), schedule)
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
        s = row.stats
        self._conn.execute(
            "INSERT OR REPLACE INTO perf (context, kernel, bindings, schedule, status, latency_us_median, latency_us_min, latency_us_max, "
            "latency_us_mean, latency_us_variance, n_samples, measured_at, captured, error, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            key
            + (
                row.status,
                s.median,
                s.min,
                s.max,
                s.mean,
                s.variance,
                s.n_samples,
                row.measured_at,
                int(row.captured),
                row.error,
                row.source,
            ),
        )

    # ------------------------------------------------------------------
    # Perf — read
    # ------------------------------------------------------------------

    def record_source(self, name: str) -> None:
        """Mark ``name`` as a source this instance holds — a golden-shaped file the import has read, whatever
        became of its rows — so the readers' freshness check can ask for it by name."""
        self._conn.execute("INSERT OR IGNORE INTO source (name) VALUES (?)", (name,))

    def sources(self) -> set[str]:
        """The sources this instance holds (:meth:`record_source`)."""
        return {name for [name] in self._conn.execute("SELECT name FROM source")}

    def perf_sources(self, ctx: Context | None = None) -> dict[str, int]:
        """How many ``perf`` rows each source contributed — what a report over this instance names as its data;
        the rows under ``ctx``'s card and regime only when one is given."""
        if ctx is None:
            return dict(self._conn.execute("SELECT source, COUNT(*) FROM perf GROUP BY 1 ORDER BY 1"))
        return dict(
            self._conn.execute(
                "SELECT p.source, COUNT(*) FROM perf p JOIN context c ON c.id = p.context "
                "WHERE c.gpu_name = ? AND c.arch = ? AND c.opt = ? AND c.flags = ? AND c.cold_cache = ? "
                "AND c.precision = ? GROUP BY 1 ORDER BY 1",
                self._regime(ctx),
            )
        )

    def forget_perf(self, ctx: Context, source_prefix: str) -> int:
        """Delete the rows measured under ``ctx``'s card and regime whose ``source`` starts with ``source_prefix``
        — how the cache lets a golden file's rows go before the file's current rows are imported, since a row
        the current file no longer holds would otherwise stay. Returns how many were deleted."""
        gpu, arch, opt, flags, cold_cache, precision = self._regime(ctx)
        return self._conn.execute(
            "DELETE FROM perf WHERE source LIKE ? AND context IN "
            "(SELECT id FROM context WHERE gpu_name = ? AND arch = ? AND opt = ? AND flags = ? AND cold_cache = ? AND precision = ?)",
            (source_prefix + "%", gpu, arch, opt, flags, cold_cache, precision),
        ).rowcount

    def lookup_perf(self, ctx: Context, kernel: str, *, bindings: dict, knobs: dict) -> PerfRow | None:
        """The row ``ctx``'s context measured for this kernel variant."""
        gpu, arch, opt, flags, cold_cache, precision = self._regime(ctx)
        context = self._context_id(ctx.backend_name, gpu, arch, opt, flags, cold_cache, precision, create=False)
        schedule = self._row_id("schedule", _schedule_row(knobs), create=False)
        if context is None or schedule is None:
            return None
        row = self._conn.execute(
            f"SELECT {_PERF_SEL} {_PERF_FROM} WHERE p.context = ? AND p.kernel = ? AND p.bindings = ? AND p.schedule = ?",  # noqa: S608
            (context, kernel, knobs_json(bindings), schedule),
        ).fetchone()
        return self._row_to_perf(row) if row else None

    def iter_perf(self, ctx: Context) -> Iterator[PerfRow]:
        """Every row measured under ``ctx``'s regime on ``ctx``'s card by ``ctx``'s backend — the deploy evidence a
        compile under ``ctx`` may read."""
        gpu, arch, opt, flags, cold_cache, precision = self._regime(ctx)
        sql = (  # noqa: S608
            f"SELECT {_PERF_SEL} {_PERF_FROM} WHERE c.gpu_name = ? AND c.arch = ? AND c.opt = ? "
            "AND c.flags = ? AND c.cold_cache = ? AND c.precision = ? AND c.backend = ?"
        )
        params: list = [gpu, arch, opt, flags, cold_cache, precision, ctx.backend_name]
        for row in self._conn.execute(sql, params).fetchall():
            yield self._row_to_perf(row)

    def iter_perf_rows(self, *, backend: str | None = "cuda") -> Iterator[PerfRow]:
        """Every ``perf`` row, of every card and regime — the view the measurement-data readers and
        ``emmy db`` read. ``backend=None`` spans every backend."""
        sql = f"SELECT {_PERF_SEL} {_PERF_FROM}"  # noqa: S608
        params: list = []
        if backend is not None:
            sql += " WHERE c.backend = ?"
            params.append(backend)
        for row in self._conn.execute(sql, params).fetchall():
            yield self._row_to_perf(row)

    def _best_leaf(self, context: int | None, kernel: str, bindings: dict) -> float | None:
        """The fastest ``ok`` median of ``kernel`` at ``bindings`` under one context row, or ``None``."""
        if context is None:
            return None
        [us] = self._conn.execute(
            "SELECT MIN(latency_us_median) FROM perf WHERE context = ? AND kernel = ? AND bindings = ? AND status = 'ok'",
            (context, kernel, knobs_json(bindings)),
        ).fetchone()
        return us

    def has_perf(self, ctx: Context) -> bool:
        """Whether ``ctx``'s context holds any clean measurement at all — what a pick asks before it prices
        kernel-set arms from their pieces' rows, since pricing an arm builds its pieces."""
        gpu, arch, opt, flags, cold_cache, precision = self._regime(ctx)
        context = self._context_id(ctx.backend_name, gpu, arch, opt, flags, cold_cache, precision, create=False)
        if context is None:
            return False
        return self._conn.execute("SELECT 1 FROM perf WHERE context = ? AND status = 'ok' LIMIT 1", (context,)).fetchone() is not None

    def priced_arms(self, ctx: Context, kernel: str, *, bindings: dict) -> list[tuple[dict, float]]:
        """Every kernel-set decision stored on ``kernel`` that ``ctx`` can price, as ``(arm, us)``: the sum
        of its pieces' best times there, each piece at its own projection of ``bindings`` onto the symbolic
        dims it kept, all-or-nothing — a piece with no price leaves that decision out. A piece is priced the
        way its parent is (:meth:`best_per_op_time`): from its own rows when it compiled, from its own pieces
        when it was cut again, so a nested cut prices through and no cut piece needs a row. A decision has no
        measurement of its own; this is its price wherever one is read (the tuner's reward, the deploy pick's
        ballot)."""
        from emmy.compiler.wire import symbolic_vars  # noqa: PLC0415

        pieces: dict[int, list[tuple[str, str]]] = {}
        for pid, child, wire in self._conn.execute(
            "SELECT r.placement, r.child, k.loop_ir FROM routing r JOIN kernel k ON k.exact_identity = r.child "
            "WHERE r.parent = ? ORDER BY r.placement, r.position",
            (kernel,),
        ):
            pieces.setdefault(pid, []).append((child, wire))
        out: list[tuple[dict, float]] = []
        for pid, children in pieces.items():
            total: float | None = 0.0
            for child, wire in children:
                projected = {v: bindings[v] for v in symbolic_vars(json.loads(wire)) if v in bindings}
                us = self.best_per_op_time(ctx, child, bindings=projected)
                if us is None:
                    total = None
                    break
                total += us
            if total is not None:
                out.append((self._knobs_of("placement", pid), total))
        return out

    def best_per_op_time(self, ctx: Context, kernel: str, *, bindings: dict) -> float | None:
        """The best measured median (us) of ``kernel`` at ``bindings`` under ``ctx``, or ``None`` when it has
        no clean measurement: its fastest ``ok`` row as a leaf, or the cheapest of its priced kernel-set
        decisions (:meth:`priced_arms`), whichever is smaller."""
        gpu, arch, opt, flags, cold_cache, precision = self._regime(ctx)
        context = self._context_id(ctx.backend_name, gpu, arch, opt, flags, cold_cache, precision, create=False)
        leaf = self._best_leaf(context, kernel, bindings)
        candidates = [us for us in (leaf, *(us for _arm, us in self.priced_arms(ctx, kernel, bindings=bindings))) if us is not None]
        return min(candidates) if candidates else None

    def drift(self) -> dict[str, int]:
        """The drift checks — each the count of rows that fail it: a schedule or placement row whose digest is
        not its knob rows'; a row naming a row that is gone (the foreign keys, which a file written with them
        off can break); a context naming a card the GPU registry lost; a schedule knob that is a placement
        knob, or a placement knob that is not. Nothing decodes a stored wire: a tune DB is a cache, and a row
        the current code disagrees with is re-tuned or re-imported, never patched."""
        from emmy import gpu  # noqa: PLC0415

        digests = 0
        for table in ("schedule", "placement"):
            for rid, stored in self._conn.execute(f"SELECT id, digest FROM {table}").fetchall():  # noqa: S608
                digests += stored != digest(knobs_json(self._knobs_of(table, rid)))
        apart = sum(is_placement_knob(n, v) for n, v in self._conn.execute("SELECT name, value FROM schedule_knob"))
        apart += sum(not is_placement_knob(n, v) for n, v in self._conn.execute("SELECT name, value FROM placement_knob"))
        return {
            "schedule and placement digests match their knob rows": digests,
            "every row names the rows it references": len(self._conn.execute("PRAGMA foreign_key_check").fetchall()),
            "every CUDA context names a registry card": sum(
                gpu.by_name(n) is None for [n] in self._conn.execute("SELECT gpu_name FROM context WHERE backend = 'cuda'")
            ),
            "schedule knobs and placement knobs stay apart": apart,
        }

    def decisions(self) -> list[tuple[str, dict]]:
        """Every stored kernel-set decision as ``(the parent's exact identity, the arm)`` — what offers a composed
        cut to a later compile of the parent kernel, without decoding any wire."""
        return [
            (parent, self._knobs_of("placement", pid))
            for parent, pid in self._conn.execute("SELECT DISTINCT parent, placement FROM routing ORDER BY parent, placement")
        ]

    def _row_to_perf(self, row) -> PerfRow:
        """A row selected as :data:`_PERF_SEL`, its ``knobs`` the schedule row it names."""
        (gpu, arch, opt, flags, backend, kernel, bindings, schedule, status, med, lo, hi, mean, var, n) = row[:15]
        measured_at, captured, error, source, cold_cache, precision = row[15:]
        return PerfRow(
            gpu=gpu,
            arch=arch,
            opt=opt,
            flags=flags,
            cold_cache=bool(cold_cache),
            precision=json.loads(precision),
            kernel=kernel,
            bindings=json.loads(bindings),
            knobs=self._knobs_of("schedule", schedule),
            backend=backend,
            status=status,
            stats=PerfStats(median=med, min=lo, max=hi, mean=mean, variance=var, n_samples=n),
            measured_at=measured_at,
            captured=bool(captured),
            error=error,
            source=source,
        )

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
