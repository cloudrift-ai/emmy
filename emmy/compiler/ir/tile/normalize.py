"""The tree-wide canonical forms of a Tile IR Fold tree.

Each :class:`Fold` owns every canonical form a node can state about itself at formation — its
lambda bodies' statement order, a bilinear term's A-first operand orientation. What remains here
is what only the WHOLE tree can decide: an identity projection dissolves into the one operand it
re-exposes (a closing rewrite can leave one behind, and it is what makes two occurrences of the
same computation compare unequal), a carrier the fusion left holding two independent contractions
becomes one term per state, and same-value cones become ONE shared object.

INVARIANT — normalization ends with no result component that no reader reads, the ROOT's own
included and the boundary stores counting as its readers, and with same-value cones (alpha-equal,
identical captures and interface names) as ONE shared object. Object identity is how the
placement machinery recognizes that two consumption sites read one value, so a rewrite that
copies a cone (the close rewrites do, by design) is only sound because this final pass restores
the sharing. Recompute elimination is a
Tile-level placement concern built on that identity: a duplicated value becomes one seam, and a
composed cut materializes it once for every reader. Do NOT patch recompute downstream — a Loop IR
fusion or emission workaround sees one kernel at a time and cannot know two kernels re-derive the
same value; fix the sharing or the seam offer here instead.
"""

from __future__ import annotations

from emmy.compiler.ir.pure import Fold


def normalize_fold_tree(root, stores: tuple = ()):
    """Normalize a complete Tile IR tree; ``None`` placeholders pass through.

    ``stores`` are the kernel's output specifications. A boundary store READS a term's exposed component — that is
    how a sweep's per-cell projection reaches its ``Write`` — so it is a reader like a consuming lift, and the prune
    keeps what it names: the values it writes, and of the names its coordinates and guard read, those some term of
    the tree defines or exposes (a scatter's row). The tree normalization is ``emmy-pure``'s (``Fold.normalized``),
    and it remembers the reached fixpoint on the result per stored set, so a reconstruction answers without walking
    again."""
    if not isinstance(root, Fold):
        return root
    values = tuple(name for spec in stores for name in spec.write.values)
    reads = tuple(name for spec in stores for name in spec.reads)
    return root.normalized(values, reads)


__all__ = [
    "normalize_fold_tree",
]
