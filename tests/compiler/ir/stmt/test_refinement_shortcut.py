"""After an individualization, refinement skips the turns of cells nothing has split.

The canonical-labeling search refines once per individualized vertex. The partition it starts from
was equitable, so every untouched cell is a splitter that splits nothing; visiting them anyway made
each search node cost the whole graph again. The shortcut must not move a single cell: the ranks it
feeds are kernel identity.
"""

from __future__ import annotations

import random

from emmy.compiler.ir.stmt.order import _equitable_partition


def _copies(rng: random.Random) -> tuple[list[int], list[tuple[int, int, int]]]:
    """A few copies of one small random graph, sometimes chained: symmetric on purpose, like the
    rows of an unrolled loop that each read the rows before them."""
    unit, copies = rng.randint(2, 5), rng.randint(2, 6)
    unit_colors = [rng.randrange(2) for _ in range(unit)]
    unit_edges = [(rng.randrange(unit), rng.randrange(unit), rng.randrange(2)) for _ in range(rng.randint(1, 7))]
    edges = [(s + c * unit, t + c * unit, k) for c in range(copies) for s, t, k in unit_edges]
    if rng.random() < 0.5:
        edges += [(c * unit, (c + 1) * unit, 2) for c in range(copies - 1)]
    return unit_colors * copies, edges


def test_the_shortcut_refines_to_exactly_what_a_full_pass_does() -> None:
    rng = random.Random(11)
    individualized = 0
    for _ in range(300):
        colors, edges = _copies(rng)
        outgoing: list[list[tuple[int, int]]] = [[] for _ in colors]
        incoming: list[list[tuple[int, int]]] = [[] for _ in colors]
        for source, target, color in edges:
            outgoing[source].append((color, target))
            incoming[target].append((color, source))
        initial = tuple(tuple(v for v, c in enumerate(colors) if c == color) for color in sorted(set(colors)))
        partition = _equitable_partition(initial, incoming, outgoing)
        for index, cell in enumerate(partition):
            if len(cell) < 2:
                continue
            for vertex in cell:
                rest = tuple(member for member in cell if member != vertex)
                split = (*partition[:index], (vertex,), rest, *partition[index + 1 :])
                assert _equitable_partition(split, incoming, outgoing, index) == _equitable_partition(split, incoming, outgoing)
                individualized += 1
    assert individualized > 1000
