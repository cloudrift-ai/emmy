"""SQLite-backed inventory + measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``loop_op`` / ``tile_op`` / ``kernel_op`` / ``cuda_op`` — one row per
  op encountered along a lowering chain. Keyed by ``identity_key(with_io=True, with_knobs=True)``.
  Each row stores the JSON form (for programmatic inspection) and the
  pretty-printed form (for human inspection).
- ``lowering`` — best-known child for each parent op, one row per
  rewrite hop along the lowering chain (Loop→Tile, every intra-Tile
  autotune step, Tile→Kernel, Kernel→Cuda). Each row carries the knob
  delta the rule stamped at that hop plus a best-median upsert — the
  chain :meth:`SearchDB.best_per_op_time` walks to resolve a pre-final
  op's measured cost (greedy fork picks come from the ``Prior``, never
  DB replay). ``record_lowering`` upserts uniformly across
  dialects: a strictly better measured median replaces the row; a
  None measurement (bench_fail terminal) never overwrites a
  known-good row. Deterministic rewrites (single option) trivially
  win their own slot via the same path.
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
  (``freeze:<digest>``). Rows written before the card was keyed carry ``gpu = ''``: they keep
  serving the machine that measured them and are never read as a dataset.

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
    the ``H_cc`` encoding (``major * 10 + minor``); ``''`` / ``None`` on a row written before the
    card was keyed. ``feat_ver`` is the featurizer vocabulary ``knobs`` is spelled in, ``source``
    how the row arrived (see the module docstring), ``error`` a ``bench_fail`` row's failure
    text."""

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


@dataclass(frozen=True)
class LoweringRow:
    """One ``lowering`` row — best-known child for a parent op.

    ``knobs`` is the delta added at this rewrite step (e.g.
    ``005_blockify_launch`` adds ``{"BN": 64, "BM": 64}``). Greedy
    replay picks the fork whose newly-stamped knobs agree with this
    delta — no need to compare structural keys per fork."""

    parent_key: str
    parent_dialect: str
    child_key: str
    child_dialect: str
    knobs: dict
    best_median_us: float | None


# The ``perf`` columns a read selects, in ``_row_to_perf`` order, with the value each one reads as on
# a DB that predates it. A writer open migrates the table (``SearchDB.__init__``); a read-only open
# never does, so its SELECT substitutes the default for a missing column instead of failing.
_PERF_READ_COLS = (
    ("context_key", None),
    ("op_key", None),
    ("backend", None),
    ("status", None),
    ("latency_us_median", None),
    ("latency_us_min", None),
    ("latency_us_max", None),
    ("latency_us_mean", None),
    ("latency_us_variance", None),
    ("n_samples", None),
    ("measured_at", None),
    ("knobs", None),
    ("captured", None),
    ("gpu", "''"),
    ("cc", "NULL"),
    ("opt", "NULL"),
    ("feat_ver", "1"),
    ("source", "'measured'"),
    ("error", "NULL"),
)

# The columns a pre-card ``perf`` table carries over into the card-keyed one (``error`` only when present).
_PERF_CARRIED_COLS = (
    "context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
    "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs, captured"
)


class SearchDB:
    """Persistent inventory of compiled ops + their measured perf.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs pass an explicit path like
    ``~/.cache/emmy/autotune.db``).
    """

    # Bumped whenever the fork-tree topology shifts in ways that change
    # ``parent_key`` / ``child_key`` for the same physical decision —
    # stale ``lowering`` rows from older versions won't match the new
    # keys and would silently slow the next tune sweep. On version
    # mismatch we drop the ``lowering`` table only; ``perf`` /
    # ``loop_op`` / ``tile_op`` etc. survive (source-hash keyed,
    # parent-tree-independent).
    #
    # Version log:
    #   1: M9.4 — planner-hoisted FM / FN / BN / BM forks. Parent-tree
    #       topology shifted vs. the legacy downstream forks.
    #   2: explicit-knob OFF sentinels — every variant now stamps every planner
    #       knob (tier-foreign ones get an OFF value: WM/WN/MMA on scalar,
    #       BM/BN/BR/FK on warp), so ``identity_key(with_io=True, with_knobs=True)`` (which folds the knob dict)
    #       shifts for every TileOp/KernelOp. Stale ``lowering`` rows won't match.
    #   3: the RASTER launch-order codec — every contraction row now spells a fifth
    #       schedule family (``RASTER: ''``/``gm8``), so ``identity_key(with_io=True, with_knobs=True)`` shifts for every
    #       matmul TileOp/KernelOp; cached pre-RASTER chains would silently replay
    #       old-key kernels and starve the new rows of evidence.
    #   4: the ``S_ext_serial_cell_work`` structural stamp — every op's knob row gains one
    #       ``S_*`` feature, so ``identity_key(with_io=True, with_knobs=True)`` shifts for every
    #       TileOp/KernelOp (the realization corpus's 211 restamped identities are the same
    #       shift); stale ``lowering`` rows would silently never match.
    #   5: typed buffer roles — ``identity_key(with_io=True)`` colors each buffer in the identity
    #       graph by dtype and shape instead of folding an io list in declaration order, so every
    #       deploy identity and variant key shifts; stale rows would silently never match. The same
    #       version folds ``Const`` into the pure ``Let`` binding: the online-softmax fold's body
    #       changes, so every softmax and attention identity shifts with it.
    _SCHEMA_VERSION = 5

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
        CREATE TABLE IF NOT EXISTS loop_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tile_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS kernel_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
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
        """
        CREATE TABLE IF NOT EXISTS lowering (
            parent_key      TEXT PRIMARY KEY,
            parent_dialect  TEXT NOT NULL,
            child_key       TEXT NOT NULL,
            child_dialect   TEXT NOT NULL,
            knobs           TEXT NOT NULL DEFAULT '{}',
            best_median_us  REAL
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
        # Drop the ``lowering`` table when an older schema is detected;
        # everything else (op inventory, perf rows) is keyed off content
        # hashes and remains valid across fork-tree changes.
        cur_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if cur_version != self._SCHEMA_VERSION:
            self._conn.execute("DROP TABLE IF EXISTS lowering")
            self._conn.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
        if self._perf_columns() and "gpu" not in self._perf_columns():
            self._migrate_perf_to_card_key()
        for stmt in self._SCHEMA:
            self._conn.execute(stmt)
        self._perf_sel = self._perf_select()

    def _perf_columns(self) -> set[str]:
        """The ``perf`` table's columns — empty when there is no table (a fresh or foreign file)."""
        return {r[1] for r in self._conn.execute("PRAGMA table_info(perf)")}

    def _migrate_perf_to_card_key(self) -> None:
        """Rebuild a ``perf`` table written before the card joined its key.

        SQLite cannot change a primary key in place, so the table is copied into the card-keyed shape in
        one transaction. The old rows take ``gpu = ''`` (the card that measured them is not recorded
        anywhere) and ``feat_ver = 1`` (unknown vocabulary): the machine that measured them keeps reading
        them as deploy evidence, and no dataset reader admits them."""
        error = "error" if "error" in self._perf_columns() else "NULL"
        self._conn.execute("BEGIN")
        try:
            self._conn.execute("ALTER TABLE perf RENAME TO perf_pre_card")
            self._conn.execute(self._PERF_DDL)
            self._conn.execute(
                f"INSERT INTO perf ({_PERF_CARRIED_COLS}, error) SELECT {_PERF_CARRIED_COLS}, {error} FROM perf_pre_card"  # noqa: S608
            )
            self._conn.execute("DROP TABLE perf_pre_card")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _perf_select(self, table: str = "perf") -> str:
        """The ``perf`` SELECT list in ``_row_to_perf`` order, qualified by ``table``, with the pre-column
        default for any column this file lacks (see :data:`_PERF_READ_COLS`)."""
        have = self._perf_columns()
        return ", ".join(f"{table}.{col}" if col in have or default is None else default for col, default in _PERF_READ_COLS)

    @classmethod
    def open_readonly(cls, path: Path | str) -> SearchDB:
        """Open an existing DB **read-only** — no schema creation, no version
        check, no ``DROP TABLE lowering``, no migration, no WAL pragma — so a read-side consumer
        (``eval``, the dataset import) never contends with a concurrent ``tune``
        writer or mutates the file. The read methods work; any write raises (the
        connection is ``?mode=ro``). Raises ``sqlite3.OperationalError`` if the
        file is absent. A non-sqlite file fails HERE with a named reason (sqlite
        itself defers header validation to the first query, which would surface
        as a bare ``DatabaseError`` deep inside a PRAGMA) — the foreseeable case
        being a measurement freeze directory's file handed over by mistake."""
        p = Path(path)
        if p.is_file():
            with p.open("rb") as fh:
                magic = fh.read(16)
            if magic and magic != b"SQLite format 3\x00":  # empty file = valid empty DB
                raise RuntimeError(f"{p} is not a sqlite database — a measurement freeze is read by `emmy dataset import`")
        self = cls.__new__(cls)
        self._path = p
        self._conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        self._perf_sel = self._perf_select()
        return self

    # ------------------------------------------------------------------
    # Op-inventory writes (idempotent INSERT OR IGNORE)
    # ------------------------------------------------------------------

    def record_loop_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO loop_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_tile_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO tile_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_kernel_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO kernel_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

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
    # Lowering edges
    # ------------------------------------------------------------------

    def record_lowering(
        self,
        parent_key: str,
        parent_dialect: str,
        child_key: str,
        child_dialect: str,
        *,
        knobs: dict | None = None,
        measured_median_us: float | None,
    ) -> None:
        """Upsert one ``parent_key`` → ``child_key`` lowering edge.

        ``knobs`` is the delta this rewrite step stamps onto the child
        (e.g. partition_loops adds ``{"BN": 64, "BM": 64, ...}``;
        launch_geometry adds nothing). Greedy replay picks forks by
        knob-subset match against this delta, so the row is enough to
        reconstruct the chain without re-querying ``perf``.

        Best-of upsert across every dialect — autotune fork rules live
        at Tile→Tile (blockify, split_register_axes) and used to be excluded
        here; recording every hop is how the chain stays replayable.
        Rows where the rewrite is genuinely deterministic (a single
        option) still trivially win their own slot, just via the same
        upsert path.
        """
        knobs_json = json.dumps(knobs or {}, sort_keys=True, default=str)
        existing = self._conn.execute(
            "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
            (parent_key,),
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO lowering (parent_key, parent_dialect, child_key, child_dialect, knobs, best_median_us) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (parent_key, parent_dialect, child_key, child_dialect, knobs_json, measured_median_us),
            )
            return
        # Replace iff the new measurement is strictly better than the
        # stored best (or the stored best is NULL). A None measurement
        # never overwrites a known-good row.
        cur_best = existing[1]
        if measured_median_us is None:
            return
        if cur_best is None or measured_median_us < cur_best:
            self._conn.execute(
                "UPDATE lowering SET child_key = ?, child_dialect = ?, knobs = ?, best_median_us = ? WHERE parent_key = ?",
                (child_key, child_dialect, knobs_json, measured_median_us, parent_key),
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
            f"SELECT {self._perf_sel} FROM perf WHERE gpu = ? AND context_key = ? AND op_key = ? AND backend = ?",  # noqa: S608
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
        source = "source" if "source" in self._perf_columns() else "'measured'"
        return dict(self._conn.execute(f"SELECT {source}, COUNT(*) FROM perf GROUP BY 1 ORDER BY 1"))  # noqa: S608

    def lookup_lowering(self, parent_key: str) -> LoweringRow | None:
        """Return the best-known child for ``parent_key``, or ``None``
        when no row exists. Used by :meth:`best_per_op_time`'s chain
        walk to resolve a pre-final op's measured cost."""
        row = self._conn.execute(
            "SELECT parent_key, parent_dialect, child_key, child_dialect, knobs, best_median_us FROM lowering WHERE parent_key = ?",
            (parent_key,),
        ).fetchone()
        if row is None:
            return None
        return LoweringRow(
            parent_key=row[0],
            parent_dialect=row[1],
            child_key=row[2],
            child_dialect=row[3],
            knobs=json.loads(row[4]) if row[4] else {},
            best_median_us=row[5],
        )

    def lookup_perf(self, ctx: Context, op_key: str, *, backend: str) -> PerfRow | None:
        """The row ``ctx``'s card measured for ``op_key`` under ``ctx``'s regime — or, failing that, one
        written before the card was keyed (``gpu = ''``), which only the machine that measured it holds."""
        row = self._conn.execute(
            f"SELECT {self._perf_sel} FROM perf WHERE gpu IN (?, '') AND context_key = ? AND op_key = ? AND backend = ? "  # noqa: S608
            "ORDER BY gpu = '' LIMIT 1",
            (ctx.hardware_id(), ctx.structural_key(), op_key, backend),
        ).fetchone()
        return _row_to_perf(row) if row else None

    def iter_perf(self, ctx: Context, *, backend: str | None = None) -> Iterator[PerfRow]:
        """Every row measured under ``ctx``'s regime on ``ctx``'s card, plus the rows written before the card
        was keyed — the deploy evidence a compile under ``ctx`` may read."""
        sql = f"SELECT {self._perf_sel} FROM perf WHERE gpu IN (?, '') AND context_key = ?"  # noqa: S608
        params: list = [ctx.hardware_id(), ctx.structural_key()]
        if backend is not None:
            sql += " AND backend = ?"
            params.append(backend)
        for row in self._conn.execute(sql, params):
            yield _row_to_perf(row)

    def iter_perf_rows(self, *, backend: str | None = "cuda") -> Iterator[PerfRow]:
        """Every ``perf`` row, of every card and regime — the view the measurement-data readers and
        ``emmy dataset`` read. ``backend=None`` spans every backend."""
        sql = f"SELECT {self._perf_sel} FROM perf"  # noqa: S608
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
            f"SELECT cuda_op.pretty, {self._perf_sel} "  # noqa: S608
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
        """Best measured median (us) for the kernel that ``op_key`` lowers
        to under ``ctx``, or ``None`` when it has no clean ``ok``
        measurement.

        ``op_key`` is typically a finalized ``LoopOp`` key (the unit the
        outer search hands to the inner per-op tuner). Two ways it carries a
        time:

        1. **Direct row** — the two-level inner search records the best
           *whole-slice* total (``Σ`` over the slice's CudaOps, so split-K
           main + combine are both counted) under the LoopOp key itself.
           Preferred when present.
        2. **Chain walk** — otherwise follow the ``lowering`` best-known child
           links down to the ``cuda`` dialect and read that terminal's
           median under ``ctx``. A ``CudaOp`` key resolves here directly (no
           lowering row as parent).
        """
        direct = self.lookup_perf(ctx, op_key, backend=backend)
        if direct is not None and direct.status == "ok":
            return direct.stats.median
        cur: str | None = op_key
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            row = self.lookup_lowering(cur)
            if row is None:
                break
            cur = row.child_key
            if row.child_dialect == "cuda":
                break
        if cur is None or cur == op_key:
            return None
        perf = self.lookup_perf(ctx, cur, backend=backend)
        return perf.stats.median if perf is not None and perf.status == "ok" else None

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


def _row_to_perf(row) -> PerfRow:
    """A ``perf`` row selected through :meth:`SearchDB._perf_select` (:data:`_PERF_READ_COLS` order)."""
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
