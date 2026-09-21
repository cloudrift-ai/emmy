"""SQLite-backed measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``kernel`` — one row per kernel, keyed by its exact identity (``identity_key(structural=False,
  with_io=True)``): the Loop IR wire that defines it (:func:`~emmy.compiler.loop_wire.kernel_wire`)
  and its C name. The fused kernel of a slice and a piece a cut or a split minted are rows alike, so
  the same kernel reached from two parents has one definition — what its candidate pool enumerates
  from. The exact identity, not the clustered deploy identity: that one merges kernels that differ
  only in their pointwise op, and a definition cannot stand for several kernels.
- ``perf`` — backend-agnostic measurement store, one row per measured kernel variant per card and
  regime, keyed ``(gpu, cc, opt, flags, kernel, bindings, knobs, backend)``. ``gpu`` is the card
  (``Context.hardware_id``): two cards sharing a capability (RTX 5090 / RTX PRO 6000, H100 / H200)
  must never meet under the keep-best upsert. ``cc``, ``opt`` and ``flags`` spell the regime
  readably — the compute capability, the cicc opt level and the residual compiler flags (``""`` in
  the plain regime). ``kernel`` names the kernel measured: a ``kernel`` row's identity, or, for a
  verdict filed against a kernel set as a whole, the set's own digest, which no kernel row backs.
  ``bindings`` are the sizes a dynamic kernel's symbolic dims were benched at (``{}`` for a static
  kernel), so rows at two sizes stay apart. ``knobs`` is the row's ``S_*`` stamps plus its schedule
  row; both dict columns are the canonical JSON of :func:`knobs_json`. ``backend`` partitions the
  table so the loop interpreter and the CUDA backend can coexist. ``feat_ver`` is the featurizer
  vocabulary the knobs are spelled in, ``source`` how the row arrived — ``measured`` on this
  machine, or imported (``freeze:<digest>``).

One schema, several instances: the tune DB (``EMMY_TUNE_DB``) is what compile reads and tune
writes; a dataset instance (``EMMY_DATASET_DB``) holds the same tables filled by ``emmy dataset
import``, and is what the measurement-data readers (``eval prior``, the fit) read.

A table is never migrated. A file whose table has other columns than this module's DDL was written
by another emmy: a writer open re-creates that table empty — the rows are regenerable (a tune DB
re-tunes, a dataset DB re-imports with ``emmy dataset import --fresh``) — and a read-only open refuses
the file. Tables an older emmy wrote and nothing reads any more are dropped on every writer open.

Concurrency: opened in WAL mode so parallel benches can read while one
writes. The connection is kept open for the DB's lifetime; callers can
share one ``SearchDB`` instance across threads (sqlite3 handles
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

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

if TYPE_CHECKING:
    from emmy.compiler.context import Context

logger = logging.getLogger(__name__)


def knobs_json(knobs) -> str:
    """The one text spelling of a knob row — sorted keys, compact — that ``perf`` stores and keys on.
    Native JSON values only: a value the encoder cannot spell raises instead of being stringified,
    because the spelling identifies a row and a lossy fallback would let two writers mint two rows
    for one."""
    return json.dumps(dict(knobs), sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    """One ``perf`` row — see the module docstring for the columns.

    ``captured``: the measurement ran under CUDA graph capture (pure GPU time);
    False = wall semantics including per-launch dispatch (all pre-capture rows).
    Both kinds stay usable (replay, prior training); on write, a captured
    measurement supersedes an uncaptured one for the same key — see
    :meth:`SearchDB.record_perf_row`. ``cc`` is in the ``H_cc`` encoding (``major * 10 + minor``);
    ``error`` is a ``bench_fail`` row's failure text."""

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
    feat_ver: int = FEATURIZER_VERSION
    source: str = "measured"


@dataclass(frozen=True)
class KernelRow:
    """One ``kernel`` row: the kernel's exact identity, its Loop IR wire and its C name."""

    identity: str
    wire: dict
    name: str


# Each table's columns, in the order its row type reads them — the SELECT list, and the column set a
# file must have for this module to read it.
_PERF_COLS = (
    "gpu",
    "cc",
    "opt",
    "flags",
    "kernel",
    "bindings",
    "knobs",
    "backend",
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
    "feat_ver",
    "source",
)
_KERNEL_COLS = ("identity", "wire", "name")
_PERF_SEL = ", ".join(f"perf.{col}" for col in _PERF_COLS)
_PERF_KEY = "gpu = ? AND cc = ? AND opt = ? AND flags = ? AND kernel = ? AND bindings = ? AND knobs = ? AND backend = ?"

# Tables an older emmy wrote (op inventories along the lowering chain, the best-known-child
# ``lowering`` edges, the ``cuda_op`` source the kernel row replaces) that nothing reads any more.
_OBSOLETE_TABLES = ("loop_op", "tile_op", "kernel_op", "cuda_op", "lowering")


class SearchDB:
    """Persistent store of measured kernels and their perf.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs pass an explicit path like
    ``~/.cache/emmy/autotune.db``).
    """

    _DDL = {
        "kernel": """
        CREATE TABLE IF NOT EXISTS kernel (
            identity TEXT PRIMARY KEY,
            wire     TEXT NOT NULL,
            name     TEXT NOT NULL
        )
        """,
        "perf": """
        CREATE TABLE IF NOT EXISTS perf (
            gpu                  TEXT NOT NULL,
            cc                   INTEGER NOT NULL,
            opt                  INTEGER NOT NULL,
            flags                TEXT NOT NULL,
            kernel               TEXT NOT NULL,
            bindings             TEXT NOT NULL,
            knobs                TEXT NOT NULL,
            backend              TEXT NOT NULL,
            status               TEXT NOT NULL,
            latency_us_median    REAL NOT NULL,
            latency_us_min       REAL NOT NULL,
            latency_us_max       REAL NOT NULL,
            latency_us_mean      REAL NOT NULL,
            latency_us_variance  REAL NOT NULL,
            n_samples            INTEGER NOT NULL,
            measured_at          TEXT NOT NULL,
            captured             INTEGER NOT NULL DEFAULT 0,
            error                TEXT,
            feat_ver             INTEGER NOT NULL,
            source               TEXT NOT NULL,
            PRIMARY KEY (gpu, cc, opt, flags, kernel, bindings, knobs, backend)
        )
        """,
    }
    _COLS = {"kernel": _KERNEL_COLS, "perf": _PERF_COLS}

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
        for table in _OBSOLETE_TABLES:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        for table, cols in self._COLS.items():
            if self._columns(table) not in (set(), set(cols)):
                logger.warning("%s: %s table written by another emmy — re-created empty (its rows are regenerable)", self._path, table)
                self._conn.execute(f"DROP TABLE {table}")
            self._conn.execute(self._DDL[table])

    def _columns(self, table: str) -> set[str]:
        """A table's columns — empty when there is no such table (a fresh or foreign file)."""
        return {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}

    @classmethod
    def open_readonly(cls, path: Path | str) -> SearchDB:
        """Open an existing DB **read-only** — no schema creation, no table drop, no WAL pragma — so a
        read-side consumer (``eval``, the dataset import) never contends with a concurrent ``tune``
        writer or mutates the file. The read methods work; any write raises (the
        connection is ``?mode=ro``). Raises ``sqlite3.OperationalError`` if the
        file is absent. A non-sqlite file fails HERE with a named reason (sqlite
        itself defers header validation to the first query, which would surface
        as a bare ``DatabaseError`` deep inside a PRAGMA) — the foreseeable case
        being a measurement freeze directory's file handed over by mistake. A file whose tables
        another emmy wrote is refused too: a reader cannot re-create them."""
        p = Path(path)
        if p.is_file():
            with p.open("rb") as fh:
                magic = fh.read(16)
            if magic and magic != b"SQLite format 3\x00":  # empty file = valid empty DB
                raise RuntimeError(f"{p} is not a sqlite database — a measurement freeze is read by `emmy dataset import`")
        self = cls.__new__(cls)
        self._path = p
        self._conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        for table, cols in cls._COLS.items():
            if self._columns(table) not in (set(), set(cols)):
                self._conn.close()
                raise RuntimeError(
                    f"{p}: {table} table written by another emmy — re-tune it, or `emmy dataset import --fresh` a dataset DB"
                )
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

    @staticmethod
    def _regime(ctx: Context) -> tuple[str, int, int, str]:
        """The regime columns a measurement under ``ctx`` is keyed by: the card, ``cc``, the opt level
        and the residual compiler flags — one spelling of the regime however its flags were written
        (:func:`~emmy.compiler.context.split_opt_level`)."""
        from emmy.compiler.context import split_opt_level  # noqa: PLC0415

        major, minor = ctx.compute_capability
        opt, flags = split_opt_level(ctx.compile_flags)
        return ctx.hardware_id(), major * 10 + minor, opt, flags

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def record_kernel(self, row: KernelRow) -> None:
        """Store a kernel's definition; a kernel already stored keeps its row (the definition is the
        same, and the name only serves the per-kernel views)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO kernel (identity, wire, name) VALUES (?, ?, ?)",
            (row.identity, json.dumps(row.wire, sort_keys=True, separators=(",", ":")), row.name),
        )

    def record_kernels(self, rows: Iterable[KernelRow]) -> int:
        return self._transaction(rows, self.record_kernel)

    def kernel_names(self) -> dict[str, str]:
        """Every stored kernel's C name by identity — the per-kernel views' grouping key."""
        return dict(self._conn.execute("SELECT identity, name FROM kernel"))

    def iter_kernels(self) -> Iterator[KernelRow]:
        for identity, wire, name in self._conn.execute("SELECT identity, wire, name FROM kernel ORDER BY identity"):
            yield KernelRow(identity=identity, wire=json.loads(wire), name=name)

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
        """Record one measurement taken now under ``ctx``: keyed by the card and the regime ``ctx``
        names, and upserted through :meth:`record_perf_row`. ``error`` is the failure text for a
        ``bench_fail`` row (whitespace-collapsed, truncated) so failure forensics (``eval failures``)
        need no tune-log grepping."""
        gpu, cc, opt, flags = self._regime(ctx)
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self.record_perf_row(
            PerfRow(
                gpu=gpu,
                cc=cc,
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
        """Upsert one ``perf`` row — a live measurement (:meth:`record_perf`) or an imported one. Keep-best-``ok``
        policy: a ``bench_fail`` never overwrites a prior ``ok`` row, and among same-semantics ``ok``
        rows the lowest median wins. ``captured`` (CUDA-graph-captured, pure GPU
        time) adds a precedence axis: a captured measurement supersedes an
        uncaptured (wall-semantics) one regardless of median — the numbers
        aren't comparable, and captured is the better truth — while an
        uncaptured measurement never overwrites a captured one. Rows of different cards never
        meet: the card is part of the key."""
        key = (row.gpu, row.cc, row.opt, row.flags, row.kernel, knobs_json(row.bindings), knobs_json(row.knobs), row.backend)
        existing = self._conn.execute(f"SELECT {_PERF_SEL} FROM perf WHERE {_PERF_KEY}", key).fetchone()  # noqa: S608
        if existing is not None:
            prev = _row_to_perf(existing)
            if prev.status == "ok":
                if row.status != "ok":
                    return  # a failure never replaces a good measurement
                if prev.captured and not row.captured:
                    return  # wall semantics never overwrites a captured row
                if not (row.captured and not prev.captured) and row.stats.median >= prev.stats.median:
                    return  # same semantics: keep the best median
        s = row.stats
        self._conn.execute(
            f"INSERT OR REPLACE INTO perf ({', '.join(_PERF_COLS)}) VALUES ({', '.join('?' * len(_PERF_COLS))})",  # noqa: S608
            key
            + (row.status, s.median, s.min, s.max, s.mean, s.variance, s.n_samples, row.measured_at, int(row.captured), row.error)
            + (row.feat_ver, row.source),
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
        """The row ``ctx``'s card measured for this kernel variant under ``ctx``'s regime."""
        key = (*self._regime(ctx), kernel, knobs_json(bindings), knobs_json(knobs), backend)
        row = self._conn.execute(f"SELECT {_PERF_SEL} FROM perf WHERE {_PERF_KEY}", key).fetchone()  # noqa: S608
        return _row_to_perf(row) if row else None

    def iter_perf(self, ctx: Context, *, backend: str | None = None) -> Iterator[PerfRow]:
        """Every row measured under ``ctx``'s regime on ``ctx``'s card — the deploy evidence a compile
        under ``ctx`` may read."""
        sql = f"SELECT {_PERF_SEL} FROM perf WHERE gpu = ? AND cc = ? AND opt = ? AND flags = ?"  # noqa: S608
        params: list = list(self._regime(ctx))
        if backend is not None:
            sql += " AND backend = ?"
            params.append(backend)
        for row in self._conn.execute(sql, params):
            yield _row_to_perf(row)

    def iter_perf_rows(self, *, backend: str | None = "cuda") -> Iterator[PerfRow]:
        """Every ``perf`` row, of every card and regime — the view the measurement-data readers and
        ``emmy dataset`` read. ``backend=None`` spans every backend."""
        sql = f"SELECT {_PERF_SEL} FROM perf"  # noqa: S608
        params: list = []
        if backend is not None:
            sql += " WHERE backend = ?"
            params.append(backend)
        for row in self._conn.execute(sql, params):
            yield _row_to_perf(row)

    def best_per_op_time(self, ctx: Context, kernel: str, *, bindings: dict, backend: str = "cuda") -> float | None:
        """Best measured median (us) of ``kernel`` at ``bindings`` in ``ctx``'s regime, or ``None``
        when it has no clean ``ok`` measurement: the kernel's whole-slice total when the two-level
        inner search recorded one (the row with no knobs — ``Σ`` over the slice's CudaOps, so a
        split's pieces both count), else its fastest ``ok`` row."""
        total = self.lookup_perf(ctx, kernel, bindings=bindings, knobs={}, backend=backend)
        if total is not None and total.status == "ok":
            return total.stats.median
        [best] = self._conn.execute(
            "SELECT MIN(latency_us_median) FROM perf WHERE gpu = ? AND cc = ? AND opt = ? AND flags = ? AND kernel = ? AND bindings = ? "
            "AND backend = ? AND status = 'ok'",
            (*self._regime(ctx), kernel, knobs_json(bindings), backend),
        ).fetchone()
        return best

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


def _row_to_perf(row) -> PerfRow:
    """A ``perf`` row selected as :data:`_PERF_SEL` (:data:`_PERF_COLS` order)."""
    (gpu, cc, opt, flags, kernel, bindings, knobs, backend, status, med, lo, hi, mean, var, n) = row[:15]
    measured_at, captured, error, feat_ver, source = row[15:]
    return PerfRow(
        gpu=gpu,
        cc=cc,
        opt=opt,
        flags=flags,
        kernel=kernel,
        bindings=json.loads(bindings),
        knobs=json.loads(knobs),
        backend=backend,
        status=status,
        stats=PerfStats(median=med, min=lo, max=hi, mean=mean, variance=var, n_samples=n),
        measured_at=measured_at,
        captured=bool(captured),
        error=error,
        feat_ver=int(feat_ver),
        source=source,
    )
