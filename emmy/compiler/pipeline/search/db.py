"""SQLite-backed measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``cuda_op`` — one row per measured CUDA kernel, keyed by ``identity_key(with_io=True, with_knobs=True)``:
  its source, launch geometry and pretty form, for the per-kernel views.
- ``perf`` — backend-agnostic measurement store, one row per measured kernel per card and
  regime: keyed ``(gpu, context_key, op_key, backend)``. ``op_key`` is whichever terminal op
  the backend measured, or the finalized Loop cache key for a directly measured whole-slice
  structural route. ``backend`` partitions the table so the loop interpreter and the CUDA
  backend can coexist in the same DB. ``gpu`` is the card (``Context.hardware_id``): the
  regime key folds only the compute capability and the compiler flags, so without it two
  cards sharing a capability (RTX 5090 / RTX PRO 6000, H100 / H200) would collide and the
  keep-best upsert would silently drop one card's row. ``cc`` / ``opt`` spell the regime
  readably (the key is a digest), ``feat_ver`` the featurizer vocabulary the stored knobs are
  spelled in, and ``source`` how the row arrived — ``measured`` on this machine, or imported
  (``freeze:<digest>``).

A table is never migrated. A file whose table has other columns than this module's DDL was written
by another emmy: a writer open re-creates that table empty — the rows are regenerable (a tune DB
re-tunes, a dataset DB re-imports with ``emmy dataset import --fresh``) — and a read-only open refuses
the file. Tables an older emmy wrote and nothing reads any more are dropped on every writer open.

One schema, several instances: the tune DB (``EMMY_TUNE_DB``) is what compile reads and tune
writes; a dataset instance (``EMMY_DATASET_DB``) holds the same tables filled by ``emmy dataset
import``, and is what the measurement-data readers (``eval prior``, the fit) read.

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


def _jsonable_geometry(geometry) -> list:
    """Render a launch grid/block to a JSON-safe nested list for the inventory
    tables. Int / str pass through; nested specs recurse; composite ``Expr``
    factors (ceil-div block extents for hint-driven masked tiles) render to
    their pretty string — inventory rows are for human inspection, not
    re-execution."""

    def conv(x):
        if isinstance(x, (int, str)):
            return x
        if isinstance(x, (tuple, list)):
            return [conv(e) for e in x]
        return x.pretty()  # Expr

    return [conv(spec) for spec in geometry]


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
    """One ``perf`` row.

    ``captured``: the measurement ran under CUDA graph capture (pure GPU time);
    False = wall semantics including per-launch dispatch (all pre-capture rows).
    Both kinds stay usable (replay, prior training); on write, a captured
    measurement supersedes an uncaptured one for the same key — see
    :meth:`SearchDB.record_perf_row`.

    ``gpu`` / ``cc`` / ``opt`` are the card and the regime the row was measured under — ``cc`` in
    the ``H_cc`` encoding (``major * 10 + minor``). ``feat_ver`` is the featurizer vocabulary ``knobs``
    is spelled in, ``source`` how the row arrived (see the module docstring), ``error`` a
    ``bench_fail`` row's failure text."""

    context_key: str
    op_key: str
    backend: str
    status: str
    stats: PerfStats
    measured_at: str
    knobs: dict
    captured: bool = False
    gpu: str = ""
    cc: int | None = None
    opt: int | None = None
    feat_ver: int = FEATURIZER_VERSION
    source: str = "measured"
    error: str | None = None


@dataclass(frozen=True)
class PerfSample:
    """One measured terminal kernel — a ``perf`` row with its ``cuda_op`` source when it has one.

    The minimal row the per-kernel analyses read: the kernel's pretty source (for the C
    identifier; ``None`` for an imported row, which has no compiled kernel), the recorded knobs
    (``S_*`` stamps + tunables), and the median latency. Backs :meth:`SearchDB.iter_perf_samples`."""

    pretty: str | None
    knobs: dict
    latency_us: float
    error: str | None = None  # bench_fail failure text (None on ok rows)


# The ``perf`` columns, in ``_row_to_perf`` order — the SELECT list, and the column set a file must
# have for this module to read it.
_PERF_COLS = (
    "context_key",
    "op_key",
    "backend",
    "status",
    "latency_us_median",
    "latency_us_min",
    "latency_us_max",
    "latency_us_mean",
    "latency_us_variance",
    "n_samples",
    "measured_at",
    "knobs",
    "captured",
    "gpu",
    "cc",
    "opt",
    "feat_ver",
    "source",
    "error",
)
_PERF_SEL = ", ".join(f"perf.{col}" for col in _PERF_COLS)

# Tables an older emmy wrote (op inventories along the lowering chain, the best-known-child
# ``lowering`` edges) that nothing reads any more; a writer open drops them.
_OBSOLETE_TABLES = ("loop_op", "tile_op", "kernel_op", "lowering")


class SearchDB:
    """Persistent inventory of compiled ops + their measured perf.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs pass an explicit path like
    ``~/.cache/emmy/autotune.db``).
    """

    _PERF_DDL = """
        CREATE TABLE IF NOT EXISTS perf (
            gpu                  TEXT NOT NULL DEFAULT '',
            context_key          TEXT NOT NULL,
            op_key               TEXT NOT NULL,
            backend              TEXT NOT NULL,
            status               TEXT NOT NULL,
            latency_us_median    REAL NOT NULL,
            latency_us_min       REAL NOT NULL,
            latency_us_max       REAL NOT NULL,
            latency_us_mean      REAL NOT NULL,
            latency_us_variance  REAL NOT NULL,
            n_samples            INTEGER NOT NULL,
            measured_at          TEXT NOT NULL,
            knobs                TEXT NOT NULL DEFAULT '{}',
            captured             INTEGER NOT NULL DEFAULT 0,
            error                TEXT,
            cc                   INTEGER,
            opt                  INTEGER,
            feat_ver             INTEGER NOT NULL DEFAULT 1,
            source               TEXT NOT NULL DEFAULT 'measured',
            PRIMARY KEY (gpu, context_key, op_key, backend)
        )
        """

    _SCHEMA = [
        """
        CREATE TABLE IF NOT EXISTS cuda_op (
            key           TEXT PRIMARY KEY,
            kernel_source TEXT NOT NULL,
            arg_order     TEXT NOT NULL,
            grid          TEXT NOT NULL,
            block         TEXT NOT NULL,
            smem_bytes    INTEGER NOT NULL,
            pretty        TEXT NOT NULL
        )
        """,
        _PERF_DDL,
    ]

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
        if self._perf_columns() not in (set(), set(_PERF_COLS)):
            logger.warning("%s: perf table written by another emmy — re-created empty (its rows are regenerable)", self._path)
            self._conn.execute("DROP TABLE perf")
        for stmt in self._SCHEMA:
            self._conn.execute(stmt)

    def _perf_columns(self) -> set[str]:
        """The ``perf`` table's columns — empty when there is no table (a fresh or foreign file)."""
        return {r[1] for r in self._conn.execute("PRAGMA table_info(perf)")}

    @classmethod
    def open_readonly(cls, path: Path | str) -> SearchDB:
        """Open an existing DB **read-only** — no schema creation, no table drop, no WAL pragma — so a
        read-side consumer (``eval``, the dataset import) never contends with a concurrent ``tune``
        writer or mutates the file. The read methods work; any write raises (the
        connection is ``?mode=ro``). Raises ``sqlite3.OperationalError`` if the
        file is absent. A non-sqlite file fails HERE with a named reason (sqlite
        itself defers header validation to the first query, which would surface
        as a bare ``DatabaseError`` deep inside a PRAGMA) — the foreseeable case
        being a measurement freeze directory's file handed over by mistake. A file whose ``perf``
        table another emmy wrote is refused too: a reader cannot re-create it."""
        p = Path(path)
        if p.is_file():
            with p.open("rb") as fh:
                magic = fh.read(16)
            if magic and magic != b"SQLite format 3\x00":  # empty file = valid empty DB
                raise RuntimeError(f"{p} is not a sqlite database — a measurement freeze is read by `emmy dataset import`")
        self = cls.__new__(cls)
        self._path = p
        self._conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        if self._perf_columns() not in (set(), set(_PERF_COLS)):
            self._conn.close()
            raise RuntimeError(f"{p}: perf table written by another emmy — re-tune it, or `emmy dataset import --fresh` a dataset DB")
        return self

    # ------------------------------------------------------------------
    # Op-inventory write (idempotent INSERT OR IGNORE)
    # ------------------------------------------------------------------

    def record_cuda_op(
        self,
        key: str,
        *,
        kernel_source: str,
        arg_order: list[str],
        grid: list[int],
        block: list[int],
        smem_bytes: int,
        pretty: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO cuda_op (key, kernel_source, arg_order, grid, block, smem_bytes, pretty) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                kernel_source,
                json.dumps(list(arg_order)),
                json.dumps(_jsonable_geometry(grid)),
                json.dumps(_jsonable_geometry(block)),
                int(smem_bytes),
                pretty,
            ),
        )

    # ------------------------------------------------------------------
    # Perf — write
    # ------------------------------------------------------------------

    def record_perf(
        self,
        ctx: Context,
        op_key: str,
        *,
        backend: str,
        status: str,
        stats: PerfStats,
        knobs: dict | None = None,
        captured: bool = False,
        error: str | None = None,
    ) -> None:
        """Record one measurement taken now under ``ctx``: keyed by the card and the regime ``ctx``
        names, and upserted through :meth:`record_perf_row`. ``error`` is the failure text for a
        ``bench_fail`` row (whitespace-collapsed, truncated) so failure forensics (``eval failures``)
        need no tune-log grepping."""
        from emmy.compiler.context import split_opt_level  # noqa: PLC0415

        major, minor = ctx.compute_capability
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self.record_perf_row(
            PerfRow(
                context_key=ctx.structural_key(),
                op_key=op_key,
                backend=backend,
                status=status,
                stats=stats,
                measured_at=datetime.now(UTC).isoformat(),
                knobs=knobs or {},
                captured=captured,
                gpu=ctx.hardware_id(),
                cc=major * 10 + minor,
                opt=split_opt_level(ctx.compile_flags)[0],
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
        existing = self._conn.execute(
            f"SELECT {_PERF_SEL} FROM perf WHERE gpu = ? AND context_key = ? AND op_key = ? AND backend = ?",  # noqa: S608
            (row.gpu, row.context_key, row.op_key, row.backend),
        ).fetchone()
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
            "INSERT OR REPLACE INTO perf "
            "(gpu, context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
            " latency_us_mean, latency_us_variance, n_samples, measured_at, knobs, captured, error, cc, opt, feat_ver, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row.gpu, row.context_key, row.op_key, row.backend, row.status, s.median, s.min, s.max, s.mean, s.variance)
            + (s.n_samples, row.measured_at, json.dumps(row.knobs, sort_keys=True, default=str), int(row.captured), row.error)
            + (row.cc, row.opt, row.feat_ver, row.source),
        )

    def record_perf_rows(self, rows: Iterable[PerfRow]) -> int:
        """Upsert ``rows`` through :meth:`record_perf_row` in ONE transaction — an import of thousands of rows
        on this autocommit connection would otherwise pay one fsync per row. Returns the rows offered."""
        n = 0
        self._conn.execute("BEGIN")
        try:
            for row in rows:
                self.record_perf_row(row)
                n += 1
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")
        return n

    # ------------------------------------------------------------------
    # Perf — read
    # ------------------------------------------------------------------

    def perf_sources(self) -> dict[str, int]:
        """How many ``perf`` rows each source contributed — what a report over this instance names as its data."""
        return dict(self._conn.execute("SELECT source, COUNT(*) FROM perf GROUP BY 1 ORDER BY 1"))

    def lookup_perf(self, ctx: Context, op_key: str, *, backend: str) -> PerfRow | None:
        """The row ``ctx``'s card measured for ``op_key`` under ``ctx``'s regime."""
        row = self._conn.execute(
            f"SELECT {_PERF_SEL} FROM perf WHERE gpu = ? AND context_key = ? AND op_key = ? AND backend = ?",  # noqa: S608
            (ctx.hardware_id(), ctx.structural_key(), op_key, backend),
        ).fetchone()
        return _row_to_perf(row) if row else None

    def iter_perf(self, ctx: Context, *, backend: str | None = None) -> Iterator[PerfRow]:
        """Every row measured under ``ctx``'s regime on ``ctx``'s card — the deploy evidence a compile
        under ``ctx`` may read."""
        sql = f"SELECT {_PERF_SEL} FROM perf WHERE gpu = ? AND context_key = ?"  # noqa: S608
        params: list = [ctx.hardware_id(), ctx.structural_key()]
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

    def iter_perf_samples(self, *, backend: str | None = "cuda", status: str = "ok", min_latency_us: float = 0.0) -> Iterator[PerfSample]:
        """Yield one :class:`PerfSample` per measured terminal kernel — ``perf``
        with its ``cuda_op`` source where one exists (an imported row has none). The single place
        the two tables are joined; backs ``Dataset.from_db``. ``backend=None`` spans every backend.
        Filters to ``status`` (default ``ok``) and ``latency_us_median >
        min_latency_us`` so callers don't re-filter stale / failed rows."""
        sql = (
            f"SELECT cuda_op.pretty, {_PERF_SEL} "  # noqa: S608
            "FROM perf LEFT JOIN cuda_op ON perf.op_key = cuda_op.key "
            "WHERE perf.status = ? AND perf.latency_us_median > ?"
        )
        params: list = [status, min_latency_us]
        if backend is not None:
            sql += " AND perf.backend = ?"
            params.append(backend)
        for pretty, *cols in self._conn.execute(sql, params):
            row = _row_to_perf(cols)
            yield PerfSample(pretty=pretty, knobs=row.knobs, latency_us=row.stats.median, error=row.error)

    # ------------------------------------------------------------------
    # Per-op best time (summed into the outer terminal reward)
    # ------------------------------------------------------------------

    def best_per_op_time(self, ctx: Context, op_key: str, *, backend: str = "cuda") -> float | None:
        """Best measured median (us) recorded under ``op_key`` in ``ctx``'s regime, or ``None`` when it
        has no clean ``ok`` measurement. ``op_key`` is typically a finalized ``LoopOp`` key (the unit the
        outer search hands to the inner per-op tuner), under which the two-level inner search records the
        best *whole-slice* total (``Σ`` over the slice's CudaOps, so split-K main + combine are both
        counted)."""
        direct = self.lookup_perf(ctx, op_key, backend=backend)
        return direct.stats.median if direct is not None and direct.status == "ok" else None

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


def _row_to_perf(row) -> PerfRow:
    """A ``perf`` row selected as :data:`_PERF_SEL` (:data:`_PERF_COLS` order)."""
    (ctx_key, op_key, backend, status, med, lo, hi, mean, var, n, measured_at, knobs_json, captured) = row[:13]
    gpu, cc, opt, feat_ver, source, error = row[13:]
    return PerfRow(
        context_key=ctx_key,
        op_key=op_key,
        backend=backend,
        status=status,
        stats=PerfStats(median=med, min=lo, max=hi, mean=mean, variance=var, n_samples=n),
        measured_at=measured_at,
        knobs=json.loads(knobs_json) if knobs_json else {},
        captured=bool(captured),
        gpu=gpu,
        cc=cc,
        opt=opt,
        feat_ver=int(feat_ver),
        source=source,
        error=error,
    )
