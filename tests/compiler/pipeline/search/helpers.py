"""Shared row builders for the search-package tests.

The dicts here are PHYSICS-CALIBRATED against the freeze's plausibility predicates, which
``freeze_reason`` composes: a row is kept or dropped by the same predicates every measured-pool
reader applies, so these spellings must track the featurizer vocabulary and the GPU registry
or the freeze suite silently stops exercising the real filter.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.db import KernelDef, PerfRow, PerfStats, SearchDB

GPU_5090 = "NVIDIA GeForce RTX 5090"  # registry records fp32/fp16 peaks -> the plausibility gate is active

# The poisoned class's shape: symbolic-M mlp_down (free excludes the symbolic axis,
# benched at the dynamic hint), fp16 operands. 2*4096*14336*512 FLOPs. The stamps
# certify every loop multiplies the iteration space (depth 3 = 1 free + 1 reduce +
# 1 symbolic — the dynM matmul spelling), which is what licenses the free x red work
# formula: 9.17 us implies ~6500 TFLOP/s (implausible) while 500 us is honest. A kernel's stamps are
# computed from its body (``features.stamps``); a test hands these to the predicates as that shape.
F16_MATMUL_STAMPS = {
    "S_ext_free_prod": 4096.0,
    "S_ext_reduce_max": 14336.0,
    "S_ext_n_free_axis": 1.0,
    "S_ext_n_reduce_axis": 1.0,
    "S_loop_depth": 3.0,
    "S_ext_n_symbolic_axis": 1.0,
    "S_dtype_f16": 2.0,
}
# The schedule row beside them: in-kernel choices only (a cross-CTA REDUCE half is a placement knob).
F16_MATMUL_ROW = {
    "TILE@map.1/inner": "mma_m16n8k16_f16_f32/f2x8/k8",
    "WORK": "w1x8",
    "REDUCE@map.1/inner": "coop",
}
#: The small shape of the square.512.dynM residue (:func:`impossible_staged_row`).
SQUARE_512_STAMPS = {**F16_MATMUL_STAMPS, "S_ext_free_prod": 512.0, "S_ext_reduce_max": 512.0}


def impossible_staged_row() -> dict:
    """The square.512.dynM residue: a cp.async-staged warp tile whose slab (139 KB for
    w1x8/f2x8/k8) exceeds the ~99 KB dynamic-smem cap could never launch — but the
    combine-only ~2 µs it left behind implies a LEGAL 133 TFLOP/s on that small shape
    (:data:`SQUARE_512_STAMPS`), so only the kernel-validity check catches it. The same config unstaged is real."""
    return {**F16_MATMUL_ROW, "STAGE@map.1/inner": "d1/smem-async"}


class StubKernel:
    """A stand-in for a kernel op whose ``S_*`` row is ``stamps``: ``features.stamps`` keeps a kernel's row on the
    kernel once computed, and this is a kernel whose row is already there. No tile stands behind it."""

    def __init__(self, stamps: dict) -> None:
        self._stamps = dict(stamps)

    def source_chain(self):
        return iter((self,))


def kernel_row(identity: str, *, name: str | None = None, symbolic: tuple[str, ...] = (), formed: bool = True) -> KernelDef:
    """A ``kernel`` row keyed ``identity`` with a minimal wire — one loop node whose output carries the
    ``symbolic`` dims. Every ``perf`` row names a kernel row, so tests seed one of these before recording
    measurements of it. The key is handed in (``KernelDef.keyed``), as the DB hands back the one it stores."""
    dims: list = [{"sym": var, "hint": 512} for var in symbolic] + [4]
    wire = {"inputs": [], "outputs": ["y"], "nodes": [{"id": "y", "op": "loop", "attrs": {"body": []}, "outputs": [["y", "f32", dims]]}]}
    return KernelDef(loop_ir=wire, name=name or f"k_{identity}", formed=formed).keyed(identity)


def perf_row(
    kernel: str,
    *,
    us: float,
    knobs: dict | None = None,
    bindings: dict | None = None,
    gpu: str = GPU_5090,
    cc: int = 120,
    opt: int = 3,
    flags: str = "",
    **over,
) -> PerfRow:
    """A measured CUDA ``perf`` row of ``kernel`` on a registry-known card, in the plain-flags regime
    of ``opt`` — the row a live bench there writes — with ``**over`` overriding any field. ``knobs``
    is the schedule row, the f16 matmul one by default."""
    kw = dict(
        gpu=gpu,
        cc=cc,
        opt=opt,
        flags=flags,
        kernel=kernel,
        bindings=dict(bindings or {}),
        knobs=dict(F16_MATMUL_ROW if knobs is None else knobs),
        backend="cuda",
        status="ok",
        stats=PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=30),
        measured_at="2026-07-09T00:00:00+00:00",
    )
    kw.update(over)
    return PerfRow(**kw)


#: A registry card per capability the realization corpus declares, so a case's rows are a card's.
CARDS = {(12, 0): GPU_5090, (7, 0): "NVIDIA Tesla V100 SXM2 16GB"}


def tuned_db(path, cases: tuple[str, ...], *, source: str = "measured", us: float = 1.0) -> SearchDB:
    """A DB as a tune of the named realization corpus cases leaves it: every row filed as a measured row (``us``
    microseconds where the case authors none) through the golden import, under the registry card of the case's
    capability and the case's own regime, ``source`` on every row."""
    from dataclasses import replace

    from emmy.compiler.context import FAST_MATH_FLAG, Context
    from emmy.compiler.pipeline.search.golden import Measurements, import_rows
    from tests.compiler.realization import helpers as corpus

    measured = Measurements(emmy_us=us, reference_us=us, reference_backend="corpus")
    db = SearchDB(path)
    for case_path in cases:
        case = corpus.load_case(corpus.CASES_DIR / case_path)
        rows = [replace(row, measurements=measured) if row.measurements is None else row for row in case.rows]
        regime = case.regime
        # Measured at the deployable opt level whatever lane the suite compiles at.
        ctx = Context.from_target(
            case.compute_cap, gpu_name=CARDS[case.compute_cap], compile_flags=FAST_MATH_FLAG if regime.get("FAST_MATH") else ""
        )
        import_rows(db, ctx, case.document, rows, source=source)
    return db
