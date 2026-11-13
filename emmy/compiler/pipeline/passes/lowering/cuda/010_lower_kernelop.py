"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp`` body to a ``__global__`` source string and derives the
launch geometry. The body is a single :class:`Grid` (``007_decode_grid`` decoded the tile into it), so the
grid is sized from its element count over its block (``BLOCK_SIZE`` scalar tier, the cooperative ``block_threads``
otherwise). A **symbolic reduce / output-sweep axis** (a
dynamic ``seq_len`` inside the body's loops) becomes a runtime ``int`` arg: the kernel
signature gains one ``int <name>`` per symbolic ``Dim`` and the launch resolves it from the
input array shapes (``sym_values`` → ``runtime_args``). The free (grid) axes are static
here (``010_lift`` defers a symbolic free axis), so the grid stays a static int.
"""

from dataclasses import replace

from emmy.compiler.dim import spans_one_page
from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import is_f4_pair_table
from emmy.compiler.ir.cuda import CudaOp, TmaDescMeta
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.interface import buffer_dtypes, kernel_interface
from emmy.compiler.ir.kernel.ir import Grid, TmaDescriptor, tile_is_static, tile_n_dim, tile_n_elements
from emmy.compiler.ir.kernel.render import render_kernelop
from emmy.compiler.ir.pure.fold import EXTENTS
from emmy.compiler.ir.stmt import Load, Write, ZeroPrologue, check_input
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.cuda._helpers import atomic_outputs as _atomic_outputs
from emmy.compiler.pipeline.passes.lowering.cuda._helpers import launch_name, paged_buffers
from emmy.compiler.pipeline.search.space import FLUSH_TO_ZERO, precision_pin

PATTERN = [Pattern("root", KernelOp)]


def _tma_descriptors(kernel: KernelOp) -> tuple[TmaDescMeta, ...]:
    """Each distinct ``TmaDescriptor`` in the body → a ``TmaDescMeta`` the backend
    encodes at launch (``cuTensorMapEncodeTiled`` off the bound array's device
    pointer + the box). First-seen order; deduped by descriptor name so the param
    list matches the renderer's signature (``const CUtensorMap* <name>``)."""
    seen: dict[str, TmaDescMeta] = {}
    for s in kernel.body.iter_of_type(TmaDescriptor):
        seen.setdefault(s.name, TmaDescMeta(name=s.name, src_buf=s.src_buf, box_extents=s.box_extents, swizzle=s.swizzle))
    return tuple(seen.values())


def _symbolic_runtime_args(kernel: KernelOp) -> tuple[str, ...]:
    """Every symbolic ``Dim`` name the body references, in first-seen order — via an axis
    extent (a dynamic reduce / output-sweep ``seq_len``) or inside a stmt ``Expression`` (a
    ``Load``/``Write`` index carrying a normalized negative-slice offset like
    ``seq_len - 1``, which needs the arg even when every axis is static). The kernel takes
    one ``int`` arg per name, the launch resolves it from the input shapes. Statement-expression free
    vars are filtered to names carried by the kernel's tensor shapes so loop / SSA vars
    never match, beside the program's own symbolic sizes (``EXTENTS``: a term may read one no buffer of the kernel
    carries, the last index of a symbolic range). Dict-keyed for stable ordering."""
    shape_syms: set[str] = set(EXTENTS.get())
    for t in (*kernel.inputs.values(), *kernel.outputs.values()):
        for d in t.shape:
            if not d.is_static:
                shape_syms |= d.expression.free_vars()
    seen: dict[str, None] = {}
    for s in kernel.body.iter():
        axes = s.axes if isinstance(s, Grid) else ((s.axis,) if hasattr(s, "axis") else ())
        for ax in axes:
            if isinstance(ax, Axis) and not ax.extent.is_static:
                for nm in ax.extent.expression.free_vars():
                    seen.setdefault(nm, None)
        for e in s.expressions():
            for nm in e.free_vars():
                if nm in shape_syms:
                    seen.setdefault(nm, None)
    return tuple(seen)


def rewrite(match: Match, root: Node) -> CudaOp | None:
    kernel: KernelOp = root.op
    grids = [s for s in kernel.body if isinstance(s, Grid)]
    if len(grids) != 1:
        raise RuleSkipped("only single-grid KernelOps lower so far")
    (grid_stmt,) = grids

    # The kernel function name doubles as the CudaOp's launch name — keep them
    # identical. Loop naming always stamps a label, but a split piece is unnamed.
    name = launch_name(kernel, root.id)
    if name != kernel.name:
        kernel = replace(kernel, name=name)

    # Buffer shapes / dtypes for the renderer come from the op's I/O Tensors
    # (snapped to the graph by the matcher's populate_io). A symbolic reduce axis adds an
    # ``int <name>`` runtime arg to the signature, threaded through to the CudaOp.
    # A serial axis is a runtime coordinate: one ``int`` param, one launch per value.
    runtime_args = tuple(dict.fromkeys((*_symbolic_runtime_args(kernel), *(axis.name for axis in kernel.serial))))
    tensors = {**kernel.inputs, **kernel.outputs}
    tma_descs = _tma_descriptors(kernel)
    paged = paged_buffers(kernel, match.graph)
    if paged:
        # A paged buffer has no base pointer to take: only ``Load`` / ``Write`` resolve a page,
        # so any other stmt touching it (a TMA descriptor, a cp.async stage, a fragment store)
        # would need a base this ABI cannot give — unless one page spans the buffer, whose base
        # the renderer resolves once — and a zero-init would have to memset a slab that does not exist.
        names = {entry[0] for entry in paged}
        split = {n for n, axis, page in paged if not spans_one_page(tensors[n].shape, axis, page)}
        staging = [s for s in kernel.body.iter() if not isinstance(s, (Load, Write))]
        touched = {b for s in staging for b in (*s.external_reads(), *s.external_writes()) if b in split}
        if touched:
            raise NotImplementedError(
                f"paged buffer(s) {sorted(touched)} are staged by a stmt that takes their base address "
                f"(a TMA descriptor or a cp.async copy) — a paged build requires a schedule that reads them directly"
            )
        zeroed = sorted(names & {*_atomic_outputs(kernel), *(zp.dst for zp in kernel.body.iter_of_type(ZeroPrologue))})
        if zeroed:
            raise NotImplementedError(f"paged buffer(s) {zeroed} are zero-initialized per launch, which has no page table to clear")
    pair_tables = frozenset(n for n in kernel.inputs if is_f4_pair_table(match.graph.nodes.get(n)))
    # The input check of whatever generates this kernel's code: the body says every type and every conversion, every
    # access of a buffer that is one allocation holds one flat index, and no loop asks for unrolling.
    split_pages = [n for n, axis, page in paged if not spans_one_page(tensors[n].shape, axis, page)]
    check_input(kernel.body, {name: dtype.name for name, dtype in buffer_dtypes(kernel, tensors).items()}, split_pages)
    source = render_kernelop(
        kernel,
        tensors=tensors,
        runtime_args=runtime_args,
        paged_buffers=paged,
        pair_tables=pair_tables,
        flush_to_zero=precision_pin(FLUSH_TO_ZERO),
    )

    # A cooperative tile fixes the per-CTA thread count (``coop · ∏block-cells``): one CTA
    # per output-cell group, ``blockDim = block_threads``, ``gridDim = N / block_threads``
    # (the decode groups ``block_threads`` consecutive cells per CTA). The scalar tier is one
    # thread per cell over the fixed ``BLOCK_SIZE`` block. A warp-specialized producer band
    # (``aux_threads``) widens ``blockDim`` only — the grid still covers the cells.
    cells = grid_stmt.block_threads
    if tile_is_static(grid_stmt):
        n = tile_n_elements(grid_stmt)
        grid = (((n + cells - 1) // cells,), (1,), (1,))
    else:
        # Symbolic grid (a dynamic free axis): ``ceil(∏extents / blockDim)`` CTAs, the symbolic
        # factor resolved from ``sym_values`` at launch (the ``Expression`` grid factor).
        grid = ((tile_n_dim(grid_stmt).ceil_div(cells).expression,), (1,), (1,))
    block = ((cells + grid_stmt.aux_threads,), (1,), (1,))
    # The launch's argument order is the kernel's own parameter list (buffers, then the TMA descriptors bound to the
    # launch-encoded ``CUtensorMap`` device arrays); the symbolic ``int`` args are tail-appended in ``_launch``. A
    # paged operand binds its page TABLE, not the buffer: the name in ``arg_order`` is what the launcher looks up in
    # ``arrays``, so the rename is the whole runtime change.
    interface = kernel_interface(kernel, tensors, runtime_args, frozenset(entry[0] for entry in paged))
    return CudaOp(
        kernel_source=source,
        kernel_name=name,
        arg_order=interface.arg_order,
        grid=grid,
        block=block,
        smem_bytes=kernel.smem_bytes(),
        comment=name,
        runtime_args=runtime_args,
        serial=tuple((axis.name, axis.extent.as_static()) for axis in kernel.serial),
        # Buffers whose zero-init a predecessor carries (``005_delegate_zero_init``) drop off
        # the memset list; the prologue targets this kernel DOES carry are surfaced for the slab
        # planner (their live interval starts at THIS launch, not their own producer's).
        zero_outputs=tuple(b for b in _atomic_outputs(kernel) if b not in kernel.zero_delegated),
        zero_prologues=tuple(dict.fromkeys(zp.dst for zp in kernel.body.iter_of_type(ZeroPrologue))),
        tma_descriptors=tma_descs,
        arch_specific=interface.arch_specific,
        dependent_launch=True,
    )
