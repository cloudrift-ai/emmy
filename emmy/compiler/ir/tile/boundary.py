"""The kernel's effect boundary: how the boundary stores an extraction took off a Tile kernel rejoin its pure
statement stream.

:func:`apply_output_specs` is the ONE reconstitution rule — the scheduler's tail gates, the materializer's
zero-axis ``Fold`` peel and ``030_cut`` all spell their effectful stmt stream through it, so the lowered kernels
stay byte-identical to the stored-``Write`` era. It reads only statements, which is what lets both the Tile IR and
the tile ops read it without reading each other.
"""

from __future__ import annotations

from emmy.compiler.ir.stmt import Body, Loop, Write
from emmy.compiler.ir.stmt.body import free_names


def _sweep_start(stmts, axis_name: str) -> int:
    """The first index of the trailing projection run a ``sweep`` store's output ``Loop``
    wraps — the earliest stmt reading the sweep axis. The trailing-RUN rule (everything from that
    stmt on is swept) is deliberately simple; the :func:`extract_output_specs` round-trip gate is
    what proves it reproduces the captured loop."""
    for i, s in enumerate(stmts):
        if axis_name in free_names(s):
            return i
    return len(stmts)


def _splice_streamed(stmts: list, write: Write) -> bool:
    """Splice a streamed store into the (deepest) reduce ``Loop`` whose body defines its observed
    values — at the body end, after the observer stmts the derived step placed last. ``False``
    when no such loop is present (a term-level stream, where the store stays at its post-node
    position)."""
    values = set(write.values)
    for i, stmt in enumerate(stmts):
        if not isinstance(stmt, Loop):
            continue
        inner = list(stmt.body)
        if _splice_streamed(inner, write):
            stmts[i] = stmt.with_bodies((Body(tuple(inner)),))
            return True
        if stmt.is_reduce and values <= set().union(*(Body((s,)).ssa_defs for s in inner)):
            stmts[i] = stmt.with_bodies((Body((*inner, write)),))
            return True
    return False


def apply_output_specs(stmts, specs, *, observed: frozenset = frozenset()) -> Body:
    """Reassemble the EFFECTFUL stmt stream from a pure statement STREAM + the kernel-boundary
    output specifications — the materializer's spelling, where the grid binds every free axis
    and nothing is a term: a store appends as the kernel tail, consecutive ``sweep`` stores on one
    axis path wrap the trailing run of stmts reading that path's axes (:func:`_sweep_start`) into
    a per-cell output ``Loop`` nest, and a store over OBSERVED values streams into its reduce loop
    when that loop is present. A group's loops open innermost-out along its path, stopping at the
    deepest PREFIX a later spec's path also carries — the last group carrying a shared prefix
    opens those loops, each wrapping every store and nest already built inside its trailing run —
    so one source loop is opened exactly once however many specs rode it. (Prefixes, not names:
    sibling nests legitimately reuse inner axis names, and only an identical prefix is the same
    source loop.) The ONE reconstitution rule the scheduler's tail gates, the materializer's
    zero-axis ``Fold`` peel and ``030_cut`` share, so the lowered kernels stay byte-identical to
    the stored-``Write`` era. A TERM places its stores itself (:meth:`Fold.lower`)."""
    out = list(stmts)
    stores = list(specs)
    index = 0
    while index < len(stores):
        st = stores[index]
        if not st.sweep:
            # At term level — the fold not yet a ``Loop`` — an observed store keeps its post-node
            # position, which is also where extraction's round-trip gate expects it.
            if not (observed and set(st.write.values) <= observed and _splice_streamed(out, st.write)):
                out.append(st.stmt())
            index += 1
            continue
        end = index + 1
        while end < len(stores) and stores[end].sweep == st.sweep:
            end += 1
        later = stores[end:]
        writes = tuple(store.stmt() for store in stores[index:end])
        if _shares_prefix(later, st.sweep, len(st.sweep)):
            out.extend(writes)  # the last group with this path opens the loop and collects these
        else:
            start = _sweep_start(out, st.sweep[-1].name)
            out = [*out[:start], Loop(axis=st.sweep[-1], body=Body((*out[start:], *writes)))]
            for depth in range(len(st.sweep) - 1, 0, -1):
                if _shares_prefix(later, st.sweep, depth):
                    break  # a shared prefix stays open for its last carrier (sharing is monotone)
                axis = st.sweep[depth - 1]
                # The loop just built rides this path too, whether or not its stores spell the axis (a scatter's
                # row is a value computed over it), so the run reaches at least to it.
                start = min(_sweep_start(out, axis.name), len(out) - 1)
                out = [*out[:start], Loop(axis=axis, body=Body(tuple(out[start:])))]
        index = end
    return Body(tuple(out))


def _shares_prefix(later, path: tuple, length: int) -> bool:
    """Whether a spec in ``later`` rides the same source loops as ``path``'s first ``length``."""
    return any(spec.sweep[:length] == path[:length] for spec in later)
