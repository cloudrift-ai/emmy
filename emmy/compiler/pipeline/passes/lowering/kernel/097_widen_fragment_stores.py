"""Store four adjacent warp-group fragment cells as one 16-byte row per lane.

A fragment cell's store writes two adjacent 16-bit columns per lane: four bytes, eight rows a
warp, half of every 32-byte sector. A warp-group GEMM stores whole rows of such cells at the end
of its K loop, all at once across the grid; on the H100 those stores cost the Qwen3-0.6B s512
layer about 3.5 of its 96 µs. Four cells along N share their rows, so the quad's lanes can trade column
pairs and each lane then writes one cell's eight columns as a single 16-byte store
(:attr:`~emmy.compiler.ir.kernel.ir.RegStore.run`): a quarter of the stores, every sector whole.

Legality, judged on four consecutive :class:`RegStore` stmts:

- one destination, the same row and column dims and row stride, the column dim the last, no
  guards, no atomic, no swizzle, the ``m16n8`` lane map (the ``wgmma`` accumulator's);
- cell ``k`` stores exactly ``8k`` columns past the first — the other dims equal;
- the first cell's column and the row stride are multiples of eight elements, so every lane's
  16-byte row is aligned (buffers themselves are allocated at least 256-byte aligned).

Only kernels that issue ``wgmma`` are rewritten: that is where it was measured. A destination the
render finds not 16-bit falls back to the four ordinary stores.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import RegStore, WgmmaMma
from emmy.compiler.ir.stmt import Body, Stmt
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

_RUN = 4  # cells per 16-byte row: the quad's four lanes
_COLS = 8  # columns per cell


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if not any(isinstance(s, WgmmaMma) for s in op.body.iter()):
        raise RuleSkipped("no warp-group MMA in the kernel")
    body, changed = _walk(op.body)
    if not changed:
        raise RuleSkipped("no four adjacent fragment stores")
    return replace(op, body=body)


def _walk(body: Body) -> tuple[Body, bool]:
    stmts: list[Stmt] = []
    changed = False
    for s in body:
        if nested := s.nested():
            walked = [_walk(b) for b in nested]
            if any(c for _, c in walked):
                s = s.with_bodies(tuple(b for b, _ in walked))
                changed = True
        stmts.append(s)
    out: list[Stmt] = []
    i = 0
    while i < len(stmts):
        group = stmts[i : i + _RUN]
        if len(group) == _RUN and _widens(group):
            out.append(replace(group[0], run=tuple(group[1:])))
            changed = True
            i += _RUN
        else:
            out.append(stmts[i])
            i += 1
    return (Body(tuple(out)) if changed else body), changed


def _plain(s: Stmt) -> bool:
    return (
        isinstance(s, RegStore)
        and not s.run
        and s.m_guard is None
        and s.n_guard is None
        and not s.atomic
        and s.swizzle == "NONE"
        and not s.volta_interleaved
        and s.fragment_layout != "m8n8k4"
        and s.ldn in (0, 1)
        and isinstance(s.ldm, int)
        and s.ldm > 0
        and s.ldm % _COLS == 0
        and s.col_dim == len(s.dst_index) - 1
    )


def _delta(a, b) -> int | None:
    """The constant ``b - a``, read off both affine forms, or ``None``."""
    free = a.free_vars() | b.free_vars()
    fa, fb = affine_form(a, free), affine_form(b, free)
    if fa is None or fb is None or fa[1] != fb[1]:
        return None
    diff = BinaryExpr("-", fb[0], fa[0]).simplify(SimplifyCtx.empty())
    return diff.value if isinstance(diff, Literal) and isinstance(diff.value, int) else None


def _aligned(col) -> bool:
    form = affine_form(col, col.free_vars())
    if form is None:
        return False
    anchor, coeffs = form
    anchor = anchor.simplify(SimplifyCtx.empty())
    return isinstance(anchor, Literal) and anchor.value % _COLS == 0 and all(c % _COLS == 0 for c in coeffs.values())


def _widens(group: list[Stmt]) -> bool:
    if not all(_plain(s) for s in group):
        return False
    first = group[0]
    if not _aligned(first.dst_index[-1]):
        return False
    for k, s in enumerate(group[1:], 1):
        if (s.dst_buffer, s.row_dim, s.col_dim, s.ldm, len(s.dst_index)) != (
            first.dst_buffer,
            first.row_dim,
            first.col_dim,
            first.ldm,
            len(first.dst_index),
        ):
            return False
        deltas = [_delta(a, b) for a, b in zip(first.dst_index, s.dst_index, strict=True)]
        if deltas[:-1] != [0] * (len(deltas) - 1) or deltas[-1] != _COLS * k:
            return False
    return True
