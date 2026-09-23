"""SQLite-backed measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. One schema, several instances: the tune
DB (``EMMY_TUNE_DB``) is what compile reads and tune writes; a dataset instance (``EMMY_DATASET_DB``)
holds the same tables filled by ``emmy dataset import``, and is what the measurement-data readers read.

Tables (the DDL is the reference):

- ``kernel`` — one row per compilable kernel, keyed by its EXACT identity (``identity_key(structural=False,
  with_io=True)``: the digest of the normalized body's form plus each buffer's dtype and hint-free shape).
  The clustered deploy identity (``identity_key(with_io=True)``, pointwise ops merged — the identity
  golden receipts store) is beside it, with the kernel's Loop IR before and after normalization and its
  C name. A piece a cut or a split minted is a row like any other, so the same kernel reached from two
  parents has one definition.
- ``kernel_feature`` — the kernel's ``S_*`` stamps, one per row: what the identity strategy writes onto a
  kernel at birth, a function of its body only. The structural signature deploy evidence joins and
  candidate pools group on is the digest of these rows, derived on read (``data.group.kernel_sig``).
- ``context`` — one row per backend, card and regime: the card's product name (``Context.hardware_id``),
  the context's target as the backend spells it (``sm_120`` on CUDA — the regime's, never a kernel's
  ``sm_120a``), the compiler's opt level and its residual flags (``''`` in the plain regime).
- ``schedule`` / ``schedule_knob`` — one row per distinct schedule row: the in-kernel choices a leaf
  kernel was measured with (``WORK``, ``TILE``, ``STAGE``, ``RASTER``, the in-kernel half of ``REDUCE``),
  keyed by ``digest(knobs_json(row))``. Never a placement knob.
- ``placement`` / ``placement_knob`` — one row per distinct kernel-set decision a cut arm spells:
  ``PLACE@seam = cut`` keys and the cross-CTA half of ``REDUCE`` (``g2k``), alias-resolved (one row per
  seam actually cut). Never an in-kernel knob; the fuse arm is no placement.
- ``routing`` — one row per PIECE of one decision on one parent, in the fragment's order. A decision has
  no measurement of its own: its price on a context is the sum of its children's best rows there,
  all-or-nothing (:meth:`SearchDB.best_per_op_time`).
- ``perf`` — one measurement per COMPILABLE kernel variant per context: the kernel, the sizes its symbolic
  dims were benched at (``bindings``, ``{}`` static), its schedule row, the statistics, ``captured``, a
  ``bench_fail`` row's ``error`` and ``source`` (``measured``, or ``freeze:<digest>`` when imported). No
  route rows, no whole-slice totals, no kernel-set verdicts.

Readers see a FLAT :class:`PerfRow`: the context's columns, and ``knobs`` reassembled as the kernel's
stamps plus the schedule row, so the featurizer, the measured pools and the freeze predicates read what
they always read. The joins live here and nowhere else.

Nothing migrates. A file whose tables have other columns than this DDL was written by another emmy: a
writer open re-creates every table empty — the rows are regenerable (a tune DB re-tunes, a dataset DB
re-imports with ``emmy dataset import --fresh``) — and a read-only open refuses the file. Foreign keys are
enforced on every connection (``PRAGMA foreign_keys = ON``, off only while the re-create drops tables).

Concurrency: opened in WAL mode so parallel benches can read while one writes. The connection is kept
open for the DB's lifetime; callers can share one ``SearchDB`` instance across threads (sqlite3 handles
locking).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from emmy.compiler.pipeline.knob import CTX_PREFIX, STRUCT_PREFIX, family_of
from emmy.compiler.structural import digest

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
    """Whether a knob is a kernel-set decision — a ``PLACE`` key, or a ``REDUCE`` value carrying a
    cross-CTA ``g<n>`` half — as opposed to an in-kernel choice. THE rule that keeps the ``schedule`` and
    ``placement`` vocabularies apart."""
    family = family_of(str(name))
    if family == "PLACE":
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
    """One measurement, flat: the context's columns, the kernel, the sizes it was benched at, and ``knobs``
    — the kernel's ``S_*`` stamps plus the schedule row, as the featurizer reads them.

    ``captured``: the measurement ran under CUDA graph capture (pure GPU time); False = wall semantics
    including per-launch dispatch. On write, a captured measurement supersedes an uncaptured one for
    the same key — see :meth:`SearchDB.record_perf_row`. ``cc`` is in the ``H_cc`` encoding
    (``major * 10 + minor``), read off the context's ``arch``; ``error`` is a ``bench_fail`` row's
    failure text."""

    gpu: str
    cc: int
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


@dataclass(frozen=True)
class KernelRow:
    """One ``kernel`` row with its stamps: the exact identity, the clustered deploy identity, the Loop IR
    wire before normalization and after (what the identities digest), the C name, and the ``S_*`` dict."""

    exact_identity: str
    structural_identity: str
    loop_ir: dict
    normalized_loop_ir: dict
    name: str
    stamps: dict


@dataclass(frozen=True)
class RoutingRow:
    """One kernel-set decision on one parent: the arm's knobs (``PLACE@seam: cut`` keys, or a cross-CTA
    ``REDUCE`` half) and the exact identities of the pieces it minted, in the fragment's order."""

    parent: str
    arm: dict
    children: tuple[str, ...]


# Each table's DDL and the column set a file must have for this module to read it.
_DDL = {
    "kernel": """
        CREATE TABLE kernel (
            exact_identity       TEXT PRIMARY KEY,
            structural_identity  TEXT NOT NULL,
            loop_ir              TEXT NOT NULL,
            normalized_loop_ir   TEXT NOT NULL,
            kernel_name          TEXT NOT NULL
        )""",
    "kernel_feature": """
        CREATE TABLE kernel_feature (
            kernel  TEXT NOT NULL REFERENCES kernel (exact_identity),
            name    TEXT NOT NULL,
            value   REAL NOT NULL,
            PRIMARY KEY (kernel, name)
        )""",
    "context": """
        CREATE TABLE context (
            id        INTEGER PRIMARY KEY,
            backend   TEXT NOT NULL,
            gpu_name  TEXT NOT NULL,
            arch      TEXT NOT NULL,
            opt       INTEGER NOT NULL,
            flags     TEXT NOT NULL,
            UNIQUE (backend, gpu_name, arch, opt, flags)
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
_INDEXES = (
    "CREATE INDEX routing_child ON routing (child)",
    "CREATE INDEX kernel_structural ON kernel (structural_identity)",
)
_COLS = {
    "kernel": ("exact_identity", "structural_identity", "loop_ir", "normalized_loop_ir", "kernel_name"),
    "kernel_feature": ("kernel", "name", "value"),
    "context": ("id", "backend", "gpu_name", "arch", "opt", "flags"),
    "schedule": ("id", "digest"),
    "schedule_knob": ("schedule", "name", "value"),
    "placement": ("id", "digest"),
    "placement_knob": ("placement", "name", "value"),
    "routing": ("parent", "placement", "position", "child"),
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
_OBSOLETE_TABLES = ("loop_op", "tile_op", "kernel_op", "cuda_op", "lowering", "kernel_set")
# Drop order respects the foreign keys; create order is the reverse.
_DROP_ORDER = ("perf", "routing", "placement_knob", "placement", "schedule_knob", "schedule", "kernel_feature", "context", "kernel")

# What one perf read selects, in ``_row_to_perf`` order: the context's columns, then the row's.
_PERF_SEL = (
    "c.gpu_name, c.arch, c.opt, c.flags, c.backend, p.kernel, p.bindings, p.schedule, p.status, "
    "p.latency_us_median, p.latency_us_min, p.latency_us_max, p.latency_us_mean, p.latency_us_variance, p.n_samples, "
    "p.measured_at, p.captured, p.error, p.source"
)
_PERF_FROM = "FROM perf p JOIN context c ON c.id = p.context"


def _arch(cc: tuple[int, int] | int) -> str:
    """The CUDA backend's spelling of a compute capability as a context target: ``sm_120``."""
    major, minor = cc if isinstance(cc, tuple) else divmod(cc, 10)
    return f"sm_{major}{minor}"


def _cc(arch: str) -> int:
    """The ``H_cc`` encoding of a context's target: ``sm_120`` -> 120. Only the CUDA spelling has one."""
    if not arch.startswith("sm_") or not arch[3:].isdigit():
        raise ValueError(f"no compute capability in the context target {arch!r}")
    return int(arch[3:])


def _split_knobs(knobs: dict) -> dict:
    """The schedule row of a knob dict: the in-kernel families, without ``S_*`` / ``H_*`` (they are the
    kernel's and the context's). A placement knob here is an error — a row spelling a cut or a cross-CTA
    split is a kernel-set decision, not a measurement of one kernel."""
    schedule = {}
    for name, value in knobs.items():
        if str(name).startswith((STRUCT_PREFIX, CTX_PREFIX)):
            continue
        if is_placement_knob(name, value):
            raise ValueError(f"{name}={value!r} is a placement knob: a kernel-set decision is a routing row, not a perf row")
        schedule[str(name)] = value
    return schedule


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
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
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
        self._conn.execute("PRAGMA foreign_keys = ON")

    def _columns(self, table: str) -> set[str]:
        """A table's columns — empty when there is no such table (a fresh or foreign file)."""
        return {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}

    def _mismatched(self) -> bool:
        """Whether any table exists with other columns than the DDL — a file another emmy wrote."""
        return any(self._columns(table) not in (set(), set(cols)) for table, cols in _COLS.items())

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
                raise RuntimeError(f"{p} is not a sqlite database — a measurement freeze is read by `emmy dataset import`")
        self = cls.__new__(cls)
        self._path = p
        self._conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        if self._mismatched():
            self._conn.close()
            raise RuntimeError(f"{p}: tables written by another emmy — re-tune it, or `emmy dataset import --fresh` a dataset DB")
        self._conn.execute("PRAGMA foreign_keys = ON")
        return self

    def _transaction(self, rows: Iterable, write) -> int:
        """``write(row)`` for every row in ONE transaction — an import of thousands of rows on this
        autocommit connection would otherwise pay one fsync per row. Returns the rows offered."""
        n = 0
        self._conn.execute("BEGIN")
        try:
            for row in rows:
                write(row)
                n += 1
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")
        return n

    # ------------------------------------------------------------------
    # The dimension tables: context, schedule, placement
    # ------------------------------------------------------------------

    @staticmethod
    def _regime(ctx: Context) -> tuple[str, str, int, str]:
        """The context columns a measurement under ``ctx`` is keyed by: the card, the target, the opt level
        and the residual compiler flags — one spelling of the regime however its flags were written
        (:func:`~emmy.compiler.context.split_opt_level`)."""
        from emmy.compiler.context import split_opt_level  # noqa: PLC0415

        opt, flags = split_opt_level(ctx.compile_flags)
        return ctx.hardware_id(), _arch(ctx.compute_capability), opt, flags

    def _context_id(self, backend: str, gpu_name: str, arch: str, opt: int, flags: str, *, create: bool) -> int | None:
        key = (backend, gpu_name, arch, opt, flags)
        row = self._conn.execute(
            "SELECT id FROM context WHERE backend = ? AND gpu_name = ? AND arch = ? AND opt = ? AND flags = ?", key
        ).fetchone()
        if row is not None:
            return row[0]
        if not create:
            return None
        return self._conn.execute("INSERT INTO context (backend, gpu_name, arch, opt, flags) VALUES (?, ?, ?, ?, ?)", key).lastrowid

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

    def _stamps(self, kernel: str) -> dict:
        return dict(self._conn.execute("SELECT name, value FROM kernel_feature WHERE kernel = ?", (kernel,)))

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def record_kernel(self, row: KernelRow) -> None:
        """Store a kernel's definition and its stamps. A kernel already stored keeps its row (the definition
        is the same; the name only serves the per-kernel views); its stamps are replaced when they differ —
        the one place a re-stamp under a new featurizer lands."""
        fresh = (
            self._conn.execute(
                "INSERT OR IGNORE INTO kernel (exact_identity, structural_identity, loop_ir, normalized_loop_ir, kernel_name) "
                "VALUES (?, ?, ?, ?, ?)",
                (row.exact_identity, row.structural_identity, _wire_json(row.loop_ir), _wire_json(row.normalized_loop_ir), row.name),
            ).rowcount
            == 1
        )
        stamps = {str(k): float(v) for k, v in row.stamps.items()}
        if fresh or self._stamps(row.exact_identity) != stamps:
            self._conn.execute("DELETE FROM kernel_feature WHERE kernel = ?", (row.exact_identity,))
            self._conn.executemany(
                "INSERT INTO kernel_feature (kernel, name, value) VALUES (?, ?, ?)", [(row.exact_identity, k, v) for k, v in stamps.items()]
            )

    def record_kernels(self, rows: Iterable[KernelRow]) -> int:
        return self._transaction(rows, self.record_kernel)

    def kernel_names(self) -> dict[str, str]:
        """Every stored kernel's C name by exact identity — the per-kernel views' grouping key."""
        return dict(self._conn.execute("SELECT exact_identity, kernel_name FROM kernel"))

    def iter_kernels(self) -> Iterator[KernelRow]:
        for exact, structural, loop_ir, normalized, name in self._conn.execute(
            "SELECT exact_identity, structural_identity, loop_ir, normalized_loop_ir, kernel_name FROM kernel ORDER BY exact_identity"
        ).fetchall():
            yield KernelRow(exact, structural, json.loads(loop_ir), json.loads(normalized), name, self._stamps(exact))

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

    def record_routings(self, rows: Iterable[RoutingRow]) -> int:
        return self._transaction(rows, self.record_routing)

    def iter_routing(self) -> Iterator[RoutingRow]:
        rows = self._conn.execute("SELECT parent, placement, position, child FROM routing ORDER BY parent, placement, position").fetchall()
        grouped: dict[tuple[str, int], list[str]] = {}
        for parent, pid, _position, child in rows:
            grouped.setdefault((parent, pid), []).append(child)
        for (parent, pid), children in grouped.items():
            yield RoutingRow(parent=parent, arm=self._knobs_of("placement", pid), children=tuple(children))

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
        backend: str,
        status: str,
        stats: PerfStats,
        captured: bool = False,
        error: str | None = None,
    ) -> None:
        """Record one measurement taken now under ``ctx``: keyed by the context ``ctx`` names, and upserted
        through :meth:`record_perf_row`. ``knobs`` is the kernel's stamped dict as the tuner holds it; the
        ``S_*`` / ``H_*`` entries are the kernel's and the context's and are not stored with the row.
        ``error`` is the failure text for a ``bench_fail`` row (whitespace-collapsed, truncated) so failure
        forensics (``eval failures``) need no tune-log grepping."""
        gpu, arch, opt, flags = self._regime(ctx)
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self.record_perf_row(
            PerfRow(
                gpu=gpu,
                cc=_cc(arch),
                opt=opt,
                flags=flags,
                kernel=kernel,
                bindings=dict(bindings),
                knobs=dict(knobs),
                backend=backend,
                status=status,
                stats=stats,
                measured_at=datetime.now(UTC).isoformat(),
                captured=captured,
                error=error,
            )
        )

    def record_perf_row(self, row: PerfRow) -> None:
        """Upsert one measurement — a live one (:meth:`record_perf`) or an imported one. Keep-best-``ok``
        policy: a ``bench_fail`` never overwrites a prior ``ok`` row, and among same-semantics ``ok`` rows
        the lowest median wins. ``captured`` (CUDA-graph-captured, pure GPU time) adds a precedence axis:
        a captured measurement supersedes an uncaptured (wall-semantics) one regardless of median — the
        numbers aren't comparable, and captured is the better truth — while an uncaptured measurement
        never overwrites a captured one. The kernel must be a ``kernel`` row already."""
        context = self._context_id(row.backend, row.gpu, _arch(row.cc), row.opt, row.flags, create=True)
        schedule = self._row_id("schedule", _split_knobs(row.knobs), create=True)
        key = (context, row.kernel, knobs_json(row.bindings), schedule)
        existing = self._conn.execute(
            "SELECT status, captured, latency_us_median FROM perf WHERE context = ? AND kernel = ? AND bindings = ? AND schedule = ?", key
        ).fetchone()
        if existing is not None:
            prev_status, prev_captured, prev_median = existing
            if prev_status == "ok":
                if row.status != "ok":
                    return  # a failure never replaces a good measurement
                if prev_captured and not row.captured:
                    return  # wall semantics never overwrites a captured row
                if not (row.captured and not prev_captured) and row.stats.median >= prev_median:
                    return  # same semantics: keep the best median
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

    def record_perf_rows(self, rows: Iterable[PerfRow]) -> int:
        return self._transaction(rows, self.record_perf_row)

    # ------------------------------------------------------------------
    # Perf — read
    # ------------------------------------------------------------------

    def perf_sources(self) -> dict[str, int]:
        """How many ``perf`` rows each source contributed — what a report over this instance names as its data."""
        return dict(self._conn.execute("SELECT source, COUNT(*) FROM perf GROUP BY 1 ORDER BY 1"))

    def lookup_perf(self, ctx: Context, kernel: str, *, bindings: dict, knobs: dict, backend: str) -> PerfRow | None:
        """The row ``ctx``'s context measured for this kernel variant."""
        gpu, arch, opt, flags = self._regime(ctx)
        context = self._context_id(backend, gpu, arch, opt, flags, create=False)
        schedule = self._row_id("schedule", _split_knobs(knobs), create=False)
        if context is None or schedule is None:
            return None
        row = self._conn.execute(
            f"SELECT {_PERF_SEL} {_PERF_FROM} WHERE p.context = ? AND p.kernel = ? AND p.bindings = ? AND p.schedule = ?",  # noqa: S608
            (context, kernel, knobs_json(bindings), schedule),
        ).fetchone()
        return self._row_to_perf(row) if row else None

    def iter_perf(self, ctx: Context, *, backend: str | None = None) -> Iterator[PerfRow]:
        """Every row measured under ``ctx``'s regime on ``ctx``'s card — the deploy evidence a compile
        under ``ctx`` may read."""
        gpu, arch, opt, flags = self._regime(ctx)
        sql = f"SELECT {_PERF_SEL} {_PERF_FROM} WHERE c.gpu_name = ? AND c.arch = ? AND c.opt = ? AND c.flags = ?"  # noqa: S608
        params: list = [gpu, arch, opt, flags]
        if backend is not None:
            sql += " AND c.backend = ?"
            params.append(backend)
        for row in self._conn.execute(sql, params).fetchall():
            yield self._row_to_perf(row)

    def iter_perf_rows(self, *, backend: str | None = "cuda") -> Iterator[PerfRow]:
        """Every ``perf`` row, of every card and regime — the view the measurement-data readers and
        ``emmy dataset`` read. ``backend=None`` spans every backend."""
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

    def priced_arms(self, ctx: Context, kernel: str, *, bindings: dict, backend: str = "cuda") -> list[tuple[dict, float]]:
        """Every kernel-set decision stored on ``kernel`` that ``ctx`` can price, as ``(arm, us)``: the sum
        of its pieces' fastest ``ok`` rows there, each piece at its own projection of ``bindings`` onto the
        symbolic dims it kept, all-or-nothing — a piece with no row leaves that decision out. A decision has
        no measurement of its own; this is its price wherever one is read (the tuner's reward, the deploy
        pick's ballot)."""
        from emmy.compiler.loop_wire import symbolic_vars  # noqa: PLC0415

        gpu, arch, opt, flags = self._regime(ctx)
        context = self._context_id(backend, gpu, arch, opt, flags, create=False)
        pieces: dict[int, list[tuple[str, str]]] = {}
        for pid, child, wire in self._conn.execute(
            "SELECT r.placement, r.child, k.normalized_loop_ir FROM routing r JOIN kernel k ON k.exact_identity = r.child "
            "WHERE r.parent = ? ORDER BY r.placement, r.position",
            (kernel,),
        ):
            pieces.setdefault(pid, []).append((child, wire))
        out: list[tuple[dict, float]] = []
        for pid, children in pieces.items():
            total: float | None = 0.0
            for child, wire in children:
                projected = {v: bindings[v] for v in symbolic_vars(json.loads(wire)) if v in bindings}
                us = self._best_leaf(context, child, projected)
                if us is None:
                    total = None
                    break
                total += us
            if total is not None:
                out.append((self._knobs_of("placement", pid), total))
        return out

    def best_per_op_time(self, ctx: Context, kernel: str, *, bindings: dict, backend: str = "cuda") -> float | None:
        """The best measured median (us) of ``kernel`` at ``bindings`` under ``ctx``, or ``None`` when it has
        no clean measurement: its fastest ``ok`` row as a leaf, or the cheapest of its priced kernel-set
        decisions (:meth:`priced_arms`), whichever is smaller."""
        gpu, arch, opt, flags = self._regime(ctx)
        context = self._context_id(backend, gpu, arch, opt, flags, create=False)
        leaf = self._best_leaf(context, kernel, bindings)
        candidates = [
            us for us in (leaf, *(us for _arm, us in self.priced_arms(ctx, kernel, bindings=bindings, backend=backend))) if us is not None
        ]
        return min(candidates) if candidates else None

    def drift(self) -> dict[str, int]:
        """The table-level drift checks, each the count of rows that fail it: a schedule or placement row
        whose digest is not its knob rows'; a row naming a row that is gone (the foreign keys, which a file
        written with them off can break); a context naming a card the GPU registry
        lost; a schedule knob that is a placement knob, or a placement knob that is not. The checks that
        need the compiler are :func:`emmy.compiler.pipeline.search.data.check.drift`."""
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
            "every context names a registry card": sum(
                gpu.by_name(n) is None for [n] in self._conn.execute("SELECT gpu_name FROM context")
            ),
            "schedule knobs and placement knobs stay apart": apart,
        }

    def decisions(self) -> list[tuple[dict, dict]]:
        """Every stored kernel-set decision as ``(the parent's stamps, the arm)`` — what offers a composed cut
        to a later compile of a kernel with the parent's signature, without decoding any wire."""
        return [
            (self._stamps(parent), self._knobs_of("placement", pid))
            for parent, pid in self._conn.execute("SELECT DISTINCT parent, placement FROM routing ORDER BY parent, placement")
        ]

    def _row_to_perf(self, row) -> PerfRow:
        """A row selected as :data:`_PERF_SEL`, its ``knobs`` reassembled from the kernel's stamps and the
        schedule row."""
        (gpu, arch, opt, flags, backend, kernel, bindings, schedule, status, med, lo, hi, mean, var, n) = row[:15]
        measured_at, captured, error, source = row[15:]
        return PerfRow(
            gpu=gpu,
            cc=_cc(arch),
            opt=opt,
            flags=flags,
            kernel=kernel,
            bindings=json.loads(bindings),
            knobs={**self._stamps(kernel), **self._knobs_of("schedule", schedule)},
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
