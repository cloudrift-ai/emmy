"""Pair adjacent staged fragment loads into one wide shared-memory instruction.

Every staged B drain costs one ``ldmatrix.x2``[``.trans``] per fragment, but ``ldmatrix.x4``
loads four 8×8 matrices — two adjacent B fragments — in one instruction, halving the drain's
LSU count. Volta has no ``ldmatrix``; its derived crosswise-A and B-congruous layouts pair two
logical fragments into one ordinary 128-bit shared load through the same pass. The staged mma
drain emits both forms, which is why pairing stays a policy peephole rather than moving into the
emitter.

Legality, judged structurally on the two loads:

- both staged role-``"b"``, same slab / ``ldm`` / ``b_trans`` / swizzle mode (the drain's
  swizzle XOR is per-lane address-based, so it commutes with the paired lane map — each lane
  un-permutes its own 16 B chunk), no masks (staged loads carry none);
- **transposed-B** (N-major slab): equal K col, N row exactly ``+8`` — the pair is one plain
  ``x4`` (lanes 16-31 address the ``+8`` N rows);
- **canonical-B** (K-major slab): equal K row, col exactly ``+8`` — the pair is one
  ``x4.trans`` (lanes 16-31 address the ``+8`` column). The ``+8``-half offsets are 16 B, so
  every paired lane address keeps ldmatrix's 16 B alignment;
- **Volta**: adjacent logical fragment indices under the matching derived layout, with equal
  logical K coordinates. A pairs register rows; B pairs register columns. Each becomes one
  LDS.128 and the coupled row/row MMA plus interleaved accumulator map preserve the output;
- no ordering constraint of the scope runs between the two loads, directly or through other statements
  (:class:`Siblings`): a body's statement order is its normal form's, which can put one fragment's whole chain
  ahead of the other's, so position says nothing. A barrier, a slab write or a consumer of the earlier value of
  either fragment between the two is such a path.

The fusion is a pure perf transform: same slab cells, same values, same fragments — the paired
kernel is bit-identical to its unpaired sibling. Idempotent: a paired load (``pair_frag`` set)
is never a candidate, so a re-run finds nothing and skips.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, SimplifyCtx, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import VOLTA_B_CONGRUOUS, VOLTA_CROSSWISE, LdmatrixLoad
from emmy.compiler.ir.stmt import Body, Statement
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._siblings import Siblings
from emmy.compiler.pipeline.search.space import PAIR_LDMATRIX

PATTERN = [Pattern("root", KernelOp)]

#: One 8×8 ldmatrix matrix — the fragment adjacency step, in elements (16 B of b16).
_ATOM8 = 8


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    # Idempotence + override via the recorded PAIR_LDMATRIX policy knob (the
    # 050_vectorize_loads pattern): only ``True`` is enumerated, so the autotuner
    # never forks on it; ``EMMY_PAIR_LDMATRIX=0`` still pins ``False``.
    if PAIR_LDMATRIX.name in op.knobs:
        raise RuleSkipped("PAIR_LDMATRIX already decided (idempotence via knob)")
    if not PAIR_LDMATRIX.narrow((True,))[0]:
        return replace(op, body=op.body, knobs={**op.knobs, PAIR_LDMATRIX.name: False})
    # Stamp the policy (True) even when nothing pairs — the realized config records that pairing
    # was enabled, keeping a uniform knob set (the 050 convention).
    new_body, _ = _walk(op.body)
    return replace(op, body=new_body, knobs={**op.knobs, PAIR_LDMATRIX.name: True})


def _walk(body: Body) -> tuple[Body, bool]:
    stmts: list[Statement] = []
    changed = False
    for s in body:
        nested = s.nested()
        if nested:
            new_bodies = []
            for b in nested:
                nb, c = _walk(b)
                new_bodies.append(nb)
                changed = changed or c
            if changed:
                s = s.with_bodies(tuple(new_bodies))
        stmts.append(s)
    paired = _pair(stmts)
    if paired is None:
        return (Body(tuple(stmts)) if changed else body), changed
    return Body(tuple(paired)), True


def _split(e: Expression) -> tuple[tuple[str, ...], int]:
    """``e`` as the terms of its ``+`` chain and the sum of their integer literals. The literal can
    sit anywhere in the chain: a ring drain's row is ``slot * rows + (warp_row + 8)``, which a
    top-level ``base + constant`` split never sees. Terms compare by their rendered form, order
    free."""
    if isinstance(e, BinaryExpression) and e.op == "+":
        (lt, lc), (rt, rc) = _split(e.left), _split(e.right)
        return tuple(sorted(lt + rt)), lc + rc
    if isinstance(e, Literal) and isinstance(e.value, int):
        return (), e.value
    return (repr(e),), 0


def _delta(a: Expression, b: Expression) -> int | None:
    """The structural constant difference ``b - a``, or ``None`` (not provably constant)."""
    if a == b:
        return 0
    free = a.free_vars() | b.free_vars()
    af, bf = affine_form(a, free), affine_form(b, free)
    if af is not None and bf is not None and af[1] == bf[1]:
        diff = BinaryExpression("-", bf[0], af[0]).simplify(SimplifyCtx.empty())
        if isinstance(diff, Literal) and isinstance(diff.value, int):
            return diff.value
    (ab, ac), (bb, bc) = _split(a), _split(b)
    return bc - ac if ab == bb else None


def _candidate(s: Statement) -> bool:
    # A staged byte-slab (fp8) drain is a cooperative gather, not an ldmatrix — nothing to pair.
    if not isinstance(s, LdmatrixLoad) or not s.staged or s.byte_slab or s.pair_frag is not None or len(s.src_index) != 2:
        return False
    if s.fragment_layout == "m8n8k4":
        # A and a transposed B both drain the crosswise layout; only a canonical B is congruous.
        return s.swizzle == VOLTA_CROSSWISE if s.role == "a" or s.b_trans else s.swizzle == VOLTA_B_CONGRUOUS
    return s.role == "b"


def _pairs_with(a: LdmatrixLoad, b: LdmatrixLoad) -> bool:
    """``b`` is ``a``'s slab-adjacent partner: the SECOND fragment of one x4 (the ``+8``
    matrices lanes 16-31 address)."""
    if (
        a.src_buffer != b.src_buffer
        or a.role != b.role
        or a.ldm != b.ldm
        or a.b_trans != b.b_trans
        or a.swizzle != b.swizzle
        or a.fragment_layout != b.fragment_layout
    ):
        return False
    row_d, col_d = _delta(a.src_index[0], b.src_index[0]), _delta(a.src_index[1], b.src_index[1])
    if a.fragment_layout == "m8n8k4":
        if a.fragment_index % 2 or b.fragment_index != a.fragment_index + 1:
            return False
        # Crosswise pairs along the slab's ROW axis (A's M, a transposed B's N) at one K
        # column; the congruous canonical B pairs along its columns at one row.
        return col_d == 0 if a.swizzle == VOLTA_CROSSWISE else row_d == 0
    if a.b_trans:  # N-major slab: N rows adjacent, same K col
        return row_d == _ATOM8 and col_d == 0
    return row_d == 0 and col_d == _ATOM8  # K-major slab: same K row, cols adjacent


def _pair(stmts: list[Statement]) -> list[Statement] | None:
    """Each low half fused with its high half wherever the scope's ordering constraints leave the two free of each
    other. A run of adjacent fragments pairs from its first: a load some open low half can still take waits for it."""
    candidates = [i for i, s in enumerate(stmts) if _candidate(s)]
    high = {a: [b for b in candidates if b != a and _pairs_with(stmts[a], stmts[b])] for a in candidates}
    if not any(high.values()):
        return None
    low = {b: [a for a in candidates if b in high[a]] for b in candidates}
    scope = Siblings(stmts)
    free = set(candidates)
    progress = True
    while progress:
        progress = False
        for a in candidates:
            if a not in free or any(c in free and scope.independent(c, a) for c in low[a]):
                continue
            b = next((b for b in high[a] if b in free and scope.independent(a, b)), None)
            if b is not None:
                scope.fuse([a, b], scope[a].replace(pair_frag=scope[b].frag))
                free -= {a, b}
                progress = True
    changed = len(free) < len(candidates)
    return scope.ordered() if changed else None
