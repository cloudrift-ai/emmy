"""Shared row builders for the search-package tests.

The dicts here are PHYSICS-CALIBRATED against the freeze's plausibility predicates, which
``freeze_reason`` composes: a row is kept or dropped by the same predicates every measured-pool
reader applies, so these spellings must track the featurizer vocabulary and the GPU registry
or the freeze suite silently stops exercising the real filter.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.db import KernelRow, PerfRow, PerfStats

GPU_5090 = "NVIDIA GeForce RTX 5090"  # registry records fp32/fp16 peaks -> the plausibility gate is active

# The poisoned class's shape: symbolic-M mlp_down (free excludes the symbolic axis,
# benched at the dynamic hint), fp16 operands. 2*4096*14336*512 FLOPs. The stamps
# certify every loop multiplies the iteration space (depth 3 = 1 free + 1 reduce +
# 1 symbolic — the dynM matmul spelling), which is what licenses the free x red work
# formula: 9.17 us implies ~6500 TFLOP/s (implausible) while 500 us is honest.
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
F16_MATMUL_FEATS = {**F16_MATMUL_STAMPS, **F16_MATMUL_ROW}


def impossible_staged_feats() -> dict:
    """The square.512.dynM residue: a cp.async-staged warp tile whose slab (139 KB for
    w1x8/f2x8/k8) exceeds the ~99 KB dynamic-smem cap could never launch — but the
    combine-only ~2 µs it left behind implies a LEGAL 133 TFLOP/s on that small shape,
    so only the kernel-validity check catches it. The same config unstaged is real."""
    return {
        **{k: v for k, v in F16_MATMUL_FEATS.items() if not k.startswith(("S_ext_free", "S_ext_reduce"))},
        "S_ext_free_prod": 512.0,
        "S_ext_reduce_max": 512.0,
        "STAGE@map.1/inner": "d1/smem-async",
    }


def kernel_row(identity: str, *, stamps: dict | None = None, name: str | None = None, symbolic: tuple[str, ...] = ()) -> KernelRow:
    """A ``kernel`` row named ``identity`` with a minimal wire — one loop node whose output carries the
    ``symbolic`` dims — and the f16 matmul stamps unless ``stamps`` says otherwise. Every ``perf`` row
    names a kernel row, so tests seed one of these before recording measurements of it."""
    dims: list = [{"sym": var, "hint": 512} for var in symbolic] + [4]
    wire = {"inputs": [], "outputs": ["y"], "nodes": [{"id": "y", "op": "loop", "attrs": {"body": []}, "outputs": [["y", "f32", dims]]}]}
    return KernelRow(
        exact_identity=identity,
        structural_identity=f"deploy:{identity}",
        loop_ir=wire,
        name=name or f"k_{identity}",
        stamps=dict(F16_MATMUL_STAMPS if stamps is None else stamps),
    )


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
    defaults to the f16 matmul stamps plus its schedule row; on write only the schedule row is the
    measurement's, the stamps are the kernel row's."""
    kw = dict(
        gpu=gpu,
        cc=cc,
        opt=opt,
        flags=flags,
        kernel=kernel,
        bindings=dict(bindings or {}),
        knobs=dict(F16_MATMUL_FEATS if knobs is None else knobs),
        backend="cuda",
        status="ok",
        stats=PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=30),
        measured_at="2026-07-09T00:00:00+00:00",
    )
    kw.update(over)
    return PerfRow(**kw)
