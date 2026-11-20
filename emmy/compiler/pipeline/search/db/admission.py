"""Admission — which measured rows a dataset reads.

A DB instance holds every row a bench wrote, the failed and the implausible among them. A measured pool and a golden
pool are built from the rows :func:`admission_reason` keeps: every ``ok`` CUDA row measured on a card the GPU registry
knows, of a kernel whose definition still lowers, that passes the physical-plausibility predicates. The effective
precision decisions and cache mode define a row's regime. One rule for every reader, so an analysis over the measured
pools and one over the golden pools drop the same rows for the same reasons.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping

from emmy.compiler.pipeline.search.db import PerfRow, SearchDB, knobs_json
from emmy.compiler.pipeline.search.features import stamps

logger = logging.getLogger(__name__)


def kernel_ops(db: SearchDB) -> Callable[[PerfRow], object]:
    """Each ``perf`` row's kernel as an op at the row's sizes (``KernelDef.op``) — what a row's features and its
    admission are computed from, since a row stores neither. A kernel is lifted once per set of sizes; ``None``
    for a kernel whose stored definition the compiler no longer takes back."""
    kernels = {kernel.exact_identity: kernel for kernel in db.iter_kernels()}
    ops: dict[tuple[str, str], object] = {}

    def op_of(row: PerfRow):
        key = (row.kernel, knobs_json(row.bindings))
        if key not in ops:
            try:
                ops[key] = kernels[row.kernel].op(row.bindings)
            except Exception:  # noqa: BLE001 — whatever the lift raises, the row measured a kernel no compile builds
                logger.debug("kernel %s no longer lowers", row.kernel[:12], exc_info=True)
                ops[key] = None
        return ops[key]

    return op_of


def shape_of(kernel) -> Mapping | None:
    """A kernel op's ``S_*`` stamps (``features.stamps``) — the shape :func:`admission_reason` reads; ``None`` for a
    kernel whose definition no longer lowers (:func:`kernel_ops` gives no op)."""
    return stamps(kernel) if kernel is not None else None


def admission_reason(row: PerfRow, shape: Mapping | None) -> str | None:
    """Why ``row``, a measurement of a kernel whose ``S_*`` stamps are ``shape`` (``None``: its definition no
    longer lowers), is left out of every pool a dataset reads, or ``None`` to keep it.

    THE admission filter, and nothing else — keep every ``ok`` row measured on a card the GPU registry knows that
    passes the shared plausibility predicates. Precision decisions and cache mode define the regime
    (``dataset.regime_of``)."""
    from emmy import gpu  # noqa: PLC0415

    if gpu.by_name(row.gpu) is None:
        return "unknown card (not in the GPU registry)"
    if row.status != "ok":
        return f"{row.status}: not a measurement"
    if shape is None:
        return "stale kernel: its definition no longer lowers"
    reason = implausible_value_reason(row, shape)
    if reason is not None:
        return f"implausible value: {reason}"
    reason = impossible_kernel_reason(row)
    if reason is not None:
        return f"impossible kernel: {reason}"
    return None


def implausible_value_reason(row: PerfRow, f: Mapping) -> str | None:
    """THE physical-plausibility predicate for a ``perf`` row's median — the reason it
    cannot be a real measurement, or ``None`` when it's plausible/ungateable. ``f`` is the ``S_*``
    row of the kernel it measured (``features.stamps``). Part of
    :func:`admission_reason`, so every pool reader drops the same rows.

    The bound is the arithmetic-intensity floor the golden A/B integrity gate uses: the
    throughput a latency implies from the kernel's shape must stay below the card's
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
    if row.status != "ok" or row.us <= 0:
        return None
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
    implied = 2.0 * free * red * hint / row.us / 1e6  # FLOP / µs -> TFLOP/s
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
