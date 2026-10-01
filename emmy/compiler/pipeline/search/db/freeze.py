"""Measurement freeze — a DB instance's admitted rows as a golden file per card.

A tune DB is a live store (tunes and imports write into it), so a model fit or evaluated straight from it is
not reproducible. A *freeze* is the snapshot a reported number is computed over — identical wherever it is
read — and the training data the priors are fit on (``eval prior --pools measured``). It is a golden file with no
traced program: one document per card holding the kernels the admitted rows measured (and the decisions that reach
them), those decisions, and a row per measurement — its schedule row, the regime it was measured under
(``FAST_MATH`` on or off, the two a golden records) and its median. ``emmy db import`` reads it back as it reads
any golden.

What freezes (:func:`freeze_reason`): every ``ok`` CUDA row measured on a card the GPU registry knows, at the
deployable opt level, that passes the physical-plausibility predicates; the fast-math flag decides which of
the two precision regimes a row is in, and no other compiler flag is stored or gated on. A failed bench is
not a measurement; the tune DB keeps it. A row a compile imported from a golden file is the file's, and is
not frozen again.

Freezing the same rows twice yields the same bytes: rows sort by content and the golden dump is
deterministic. A file's identity is its bytes — ``emmy db import`` sources its rows as
``freeze:<sha256[:12]>`` of the file (``golden.evidence.file_source``), so a report over a dataset DB names the
exact snapshot it was computed over. A freeze checked into the repository lives under ``search/freezes/`` (payload
in git LFS) and is named on the import command line like any other source.

Produced by ``emmy db freeze``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path

from emmy.compiler.pipeline.knob import METADATA_PREFIXES
from emmy.compiler.pipeline.search.dataset import KernelDef, regime_of
from emmy.compiler.pipeline.search.dataset.pool import REGIME_PINS
from emmy.compiler.pipeline.search.db import PerfRow, SearchDB, knobs_json
from emmy.compiler.pipeline.search.features import DEPLOYABLE_OPT

logger = logging.getLogger(__name__)

_LFS_POINTER = "version https://git-lfs.github.com/spec/v1"


def freeze_reason(row: PerfRow) -> str | None:
    """Why ``row`` is excluded from a measurement freeze and from every measured-pool reader, or
    ``None`` to keep it.

    THE admission filter, and nothing else — keep every ``ok`` row measured at the DEPLOYABLE opt level, on a
    card the GPU registry knows, that passes the shared plausibility predicates.

    The opt-level gate is what keeps a freeze a fair yardstick. A freeze is the corpus a reported
    prior number is computed over, and a measurement taken under a non-deployable opt level
    answers a question nothing asks: nothing trains on it (``Prior.add_rows``) and no deploy
    reads it. Kept, it would put half a card's pools in a lane no one runs, so half the headline
    number would describe a regime that does not exist. Of the other compiler flags only fast math
    is a regime (:func:`regime_of`); the rest a freeze neither stores nor gates on."""
    from emmy import gpu  # noqa: PLC0415

    if gpu.by_name(row.gpu) is None:
        return "unknown card (not in the GPU registry)"
    if row.opt != DEPLOYABLE_OPT:
        return f"non-deployable regime (H_opt={row.opt:g})"
    if row.status != "ok":
        return f"{row.status}: not a measurement"
    reason = implausible_value_reason(row)
    if reason is not None:
        return f"implausible value: {reason}"
    reason = impossible_kernel_reason(row)
    if reason is not None:
        return f"impossible kernel: {reason}"
    return None


def implausible_value_reason(row: PerfRow) -> str | None:
    """THE physical-plausibility predicate for a ``perf`` row's median — the reason it
    cannot be a real measurement, or ``None`` when it's plausible/ungateable. Part of
    :func:`freeze_reason`, so a freeze and every measured-pool reader drop the same rows.

    The bound is the arithmetic-intensity floor the golden A/B integrity gate uses: the
    throughput a latency implies from the row's stamped shape must stay below the card's
    recorded peak. ``2·free·reduce_max`` FLOPs is the true work ONLY when the reduce axes
    are **disjoint** from the output — the iteration space is then exactly
    ``free_prod × reduce`` (a contraction, or a pure output-shrinking reduce) — which the
    stamps certify as ``S_loop_depth == n_free + n_reduce + n_symbolic`` (every loop of
    the nest is either a counted free/symbolic output axis or a counted reduce axis). A
    norm/softmax kernel fails that equality — its reduced axis is part of the full-size
    output, so ``free_prod`` already contains it and the product overcounts by the reduce
    extent (a cooperative norm legitimately runs ~100x its serial sibling, so a latency
    floor there would flag honest rows); a fused multi-node kernel (attention) fails it
    too. Both stay ungated rather than falsely flagged — the identity was verified
    against every stamp combination in the 2026-07 sweep stores. ``reduce_max``, not
    ``reduce_prod``, keeps the bound a lower estimate of work even off the exact case. A
    symbolic axis is excluded from the stamped products, so the sizes the row was benched at
    (``bindings``) re-enter as one factor, their product; a static kernel binds nothing and a
    symbolic one with no sizes stored is read at the default hint. Ungateable rows also pass on:
    non-``ok`` status (a fail sentinel is not a measurement), no stamped shape, unknown card or
    unrecorded peak."""
    if row.status != "ok" or row.stats.median <= 0:
        return None
    f = row.knobs
    free = float(f.get("S_ext_free_prod") or 0.0)
    red = float(f.get("S_ext_reduce_max") or 0.0)
    if free <= 0 or red <= 0:
        return None
    # Work = free x red only when every loop multiplies the iteration space (disjoint axes).
    depth = float(f.get("S_loop_depth") or 0.0)
    n_sym = float(f.get("S_ext_n_symbolic_axis") or 0.0)
    n_axes = float(f.get("S_ext_n_free_axis") or 0.0) + float(f.get("S_ext_n_reduce_axis") or 0.0) + n_sym
    if depth <= 0 or depth != n_axes:
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(row.gpu) if row.gpu else None
    if spec is None:
        return None
    half = any(k.startswith("S_dtype_") and "f16" in k and v for k, v in f.items())  # f16 / bf16
    peak = spec.peak_tflops("fp16" if half else "fp32")
    if not peak:
        return None
    from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415

    hint = math.prod(row.bindings.values()) if row.bindings else (DEFAULT_SEQ_HINT if n_sym > 0 else 1)
    implied = 2.0 * free * red * hint / row.stats.median / 1e6  # FLOP / µs -> TFLOP/s
    if implied > peak:
        return f"implies {implied:.0f} TFLOP/s > {peak:.0f} device peak"
    return None


def impossible_kernel_reason(row: PerfRow) -> str | None:
    """The *validity* companion to :func:`implausible_value_reason` — the reason the row's
    stamped kernel could never have launched, or ``None``. A ``cp.async``-staged warp tile
    whose slab (``depth · (tile_m + tile_n) · bk_elems · elem_bytes``, the
    warp stage sizing) exceeds the card's dynamic-smem opt-in cap cannot
    materialize — pre-#330 code stamped such stages anyway, the materializer rejected the
    main kernel, and the bench recorded the surviving combine kernel's cached µs as an
    ``ok`` measurement of the whole op. On shapes too small for the latency floor to
    notice (square.512's combine implies a legal 133 TFLOP/s), THIS check is the only one
    that catches the class: the measurement is of a kernel set that provably didn't
    include the stamped kernel."""
    if row.status != "ok":
        return None
    f = row.knobs
    tile_spec = next((str(v) for k, v in f.items() if k.startswith("TILE") and v), "")
    stage_spec = next((str(v) for k, v in f.items() if k.startswith("STAGE") and v), "")
    if not tile_spec or not stage_spec.startswith("d"):
        return None
    from emmy.compiler.ir.schedule import Stage, Tile, Work  # noqa: PLC0415

    try:
        work = Work.parse(str(f.get("WORK") or ""))  # the row's unit widths live here, not in TILE
        tp, st = Tile.parse(tile_spec, work), Stage.parse(stage_spec)
    except ValueError:
        return None
    if not tp.is_warp:
        return None
    if st.transport != "smem-async":
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(row.gpu) if row.gpu else None
    if spec is None:
        return None
    atom = tp.atom
    tile_m = tp.units_m * tp.reg_m * atom.atom_m
    tile_n = tp.units_n * tp.reg_n * atom.atom_n
    slab = st.depth * (tile_m + tile_n) * tp.bk * atom.atom_k * atom.operand_dtype("a").nbytes
    if slab > spec.smem_optin:
        return f"staged slab {slab} B > {spec.smem_optin} B dynamic-smem cap (kernel cannot launch)"
    return None


def _gpu_filename(gpu_name: str, cap: tuple[int, int]) -> str:
    """The per-GPU golden file name, mirroring the ``golden/`` convention —
    e.g. ``nvidia_geforce_rtx_4090_sm89.json``."""
    slug = re.sub(r"[^a-z0-9]+", "_", gpu_name.lower()).strip("_") or "unknown_gpu"
    return f"{slug}_sm{cap[0]}{cap[1]}.json"


def schedule_row(row: PerfRow) -> dict[str, str]:
    """``row``'s schedule row alone: what the tuner recorded, without the kernel's stamps and identity."""
    return {str(k): str(v) for k, v in row.knobs.items() if not str(k).startswith(METADATA_PREFIXES)}


def freeze_documents(db: SearchDB) -> tuple[dict[str, object], Counter]:
    """The DB's admitted rows as one golden document per card, keyed by the card's file name, and the count of the
    rows left out, by reason. A document holds the kernels its rows measured, every decision that reaches one of
    them with the kernels it names, and a row per measurement, in content order."""
    from emmy.compiler.pipeline.search.golden import GoldenFile, Kernel, Measurements, Row  # noqa: PLC0415

    kernels = {k.exact_identity: k for k in db.iter_kernels()}
    routing = list(db.iter_routing())
    minted: dict[str, list] = defaultdict(list)
    for route in routing:
        for child in route.children:
            minted[child].append(route)
    dropped: Counter[str] = Counter()
    by_card: dict[tuple[str, tuple[int, int]], list[PerfRow]] = defaultdict(list)
    for row in db.iter_perf_rows(backend="cuda"):
        if row.source.startswith("golden:"):
            dropped["a golden file's row"] += 1
            continue
        reason = freeze_reason(row)
        if reason is not None:
            dropped[reason.split(":")[0]] += 1
            continue
        by_card[(row.gpu, divmod(row.cc, 10))].append(row)
    documents = {}
    for (gpu_name, cap), rows in sorted(by_card.items()):
        rows.sort(key=lambda r: (r.kernel, knobs_json(r.bindings), knobs_json(r.knobs), r.flags))
        # The decisions that reach a measured kernel, and every kernel they name: the routing closure upward.
        wanted = {row.kernel for row in rows}
        routes = []
        queue = sorted(wanted)
        while queue:
            identity = queue.pop()
            for route in minted.get(identity, ()):
                if route not in routes:
                    routes.append(route)
                    for named in (route.parent, *route.children):
                        if named not in wanted:
                            wanted.add(named)
                            queue.append(named)
        document = GoldenFile(
            gpu_name=gpu_name,
            compute_cap=cap,
            kernels=[Kernel(**{f.name: getattr(kernels[k], f.name) for f in fields(KernelDef)}) for k in sorted(wanted)],
            routing=[route for route in routing if route in routes],
            rows=[
                Row(
                    name=f"{kernels[row.kernel].name}.{row.kernel[:12]}.{n}",
                    kernel=row.kernel,
                    bindings=dict(row.bindings),
                    pins=dict(REGIME_PINS[regime_of(row.flags)]),
                    knobs=schedule_row(row),
                    measurements=Measurements(emmy_us=row.stats.median),
                )
                for n, row in enumerate(rows)
            ],
        )
        documents[_gpu_filename(gpu_name, cap)] = document
    return documents, dropped


def write_freeze(db_path: Path | str, out_dir: Path | str) -> dict[str, str]:
    """Read the DB instance at ``db_path`` read-only and atomically write the freeze DIRECTORY at
    ``out_dir``: one golden document per card. Returns each file's name and digest. Hard-errors when nothing
    survives the filter — a zero-row freeze means the wrong DB, not an empty dataset. An existing ``out_dir``
    is replaced only when it is itself a freeze (holds golden files and nothing else) — anything else is
    refused rather than deleted."""
    db = SearchDB.open_readonly(db_path)
    try:
        documents, dropped = freeze_documents(db)
    finally:
        db.close()
    for reason, n in dropped.most_common():
        logger.info("[freeze] left out %d row(s): %s", n, reason)
    if not documents:
        raise RuntimeError(f"no freezable rows in {db_path} — wrong DB, or its rows are in another regime")
    out = Path(out_dir)
    if out.exists() and not (out.is_dir() and all(p.suffix == ".json" for p in out.iterdir())):
        raise RuntimeError(f"{out} exists and is not a measurement freeze — refusing to replace it")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    digests = {}
    for name, document in sorted(documents.items()):
        digests[name] = hashlib.sha256(document.dump(tmp / name).read_bytes()).hexdigest()
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    return digests


def is_lfs_pointer(path: Path | str) -> bool:
    """Whether the payload at ``path`` is a git-LFS pointer rather than the data — what a clone or CI checkout
    without LFS leaves, three lines that fail to parse with an error that says nothing about the real problem."""
    with Path(path).open("r") as fh:
        return fh.read(len(_LFS_POINTER)) == _LFS_POINTER
