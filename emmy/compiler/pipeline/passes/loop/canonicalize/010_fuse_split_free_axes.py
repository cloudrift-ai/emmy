"""Re-fuse adjacent free axes that a fused reshape split — the contraction-shape canonicalization.

A reshape fused into a producer splits one of the producer's output axes into a nest of two (an
attention projection's ``view(batch, seq, heads, head_dim)`` carves N = 12288 into 24 × 512), so the
kernel iterates the CONSUMER's post-view axes while its operand loads still address the producer's
single axis through a composite index (``wt[k, h*512 + d]``). Downstream that split is a lockout,
not a slowdown: contraction binding assigns ``(m, n)`` to the trailing free-axis pair, so the split
kernel binds the wrong row, the weight load carries a third grid axis, and the warp/mma tier is
never enumerated — the kernel stays a scalar reduce no amount of tuning can rescue.

This rule restores the canonical spelling: two free loops ``p`` (extent P) over ``q`` (extent
Q) — perfectly nested, both static — fuse into one axis of extent P·Q via the bijective
reindexing ``p → f / Q``, ``q → f % Q``, with every coordinate expression σ-substituted and
re-simplified. The pair need not be adjacent: free loops are parallel by definition, so a
perfectly-nested run of free loops between them (the ``transpose(1, 2)`` every attention
projection fuses after its view puts ``seq`` between ``heads`` and ``head_dim``) interchanges
outward and the fused axis takes ``q``'s place under it. The substitution is semantics-preserving
unconditionally; whether it lands is a PROFITABILITY gate checked after the fact: every rewritten
access must fold clean. A composite operand index collapses to the bare axis (``(f/Q)·Q + f%Q →
f``, the recomposition fold in ``Expr.simplify``); a store that indexes the pair as separate
buffer dims keeps the honest split-store spelling — ``[…, f/Q, f%Q]`` when the buffer's
row-major flatten folds it back to an affine address, or the permuted ``[…, f/Q, …, f%Q]`` of a
transposed output, whose address is per-element exact on every scalar tier and whose warp-tier
addressability is the scheduler's legality question (``ir.address.split_addressable``). Any access where a
div/mod residue would otherwise survive — an axis used alone, a predicate over the pair —
declines the pair, and the nest stands.

Runs to fixpoint inside one rewrite (a three-way split fuses pairwise), AFTER ``loop/fusion`` is
quiescent — canonicalizing a producer that still awaits a merge could re-spell the very indices the
splicer composes through — and before ``loop/stamp``, so kernel identity and everything downstream
(classification's trailing pair, shape keys, goldens) see one canonical spelling. Split and unsplit
spellings of the same contraction thereby converge to one kernel identity.

Before fusion, restore separate free coordinates when operands read a flattened axis through
its quotient and remainder. Neither operand owns the mixed coordinate; restoring its factors
makes each matrix role explicit. The fusion gate above cannot undo these separate operand reads.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.canonicalize._free_axes import canonical_free_axes

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None) -> LoopOp:
    op = root.op
    if not isinstance(op, LoopOp):
        raise RuleSkipped("root is no longer a LoopOp")
    body = canonical_free_axes(op.body, {name: t.shape for name, t in {**op.inputs, **op.outputs}.items()})
    if body is None:
        raise RuleSkipped("free coordinates already canonical")
    return replace(op, body=body)
