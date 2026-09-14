"""Finite move catalogs used by the classic schedule model, and the constrained integer spaces they are
generated from.

The candidate domain as a CONSTRAINED INTEGER SET — declare the dimensions, declare the
multiplicative bounds that couple them, enumerate the legal points.

This is the machinery for generating a schedule family's candidate values from its stated
constraints instead of curating them by hand. The
constraints that bound a schedule family are products of the unknowns — ``wm·wn·32 ≤ 1024`` (the
CTA thread budget), ``fm·fn ≤ 32`` (the C-fragment budget) — so the feasible set is not convex and
there is no coordinate change that makes both the products and the bounds affine at once: prime
exponents linearize the products but turn ``≤`` into divisibility (a partial order), and real logs
linearize both but leave the feasible points off any lattice. What survives is the honest thing:
keep integer coordinates, keep the products multiplicative, and ENUMERATE.

Brute force, but not blind. :meth:`Space.__iter__` walks the dimensions in declaration order and
drops a prefix the moment a bound's running product can no longer be satisfied — every value is
``≥ 1``, so a final product is a multiple of any partial one and never smaller. A budget like
``wm·wn·32 ≤ 1024`` therefore kills its subtree at the first factor that overruns it rather than at
the leaf.

Worked example — a warp tile's free geometry, generated rather than listed::

    Space(
        dims=(
            Dimension("wm", (1, 2, 4, 8, 16)),
            Dimension("wn", (1, 2, 4, 8, 16)),
            Dimension("fn", tuple(range(1, 33))),
        ),
        bounds=(Bound(("wm", "wn"), limit=1024, coeff=32),),  # the CTA thread budget
    )

A bound states ONE comparison, ``coeff · Πdims ≤ limit``, because that is the only one any live
domain needs. Equality and divisibility were spelled here too, for constraints the curated grids
still carry (a flash tile covering the head dimension exactly; a K-step dividing a static extent).
They had no caller, and a comparison nothing states is a comparison nobody has had to get right —
so they went. Both are a few lines to restore, and the pruning contract each would need is written
on :meth:`Bound.holds`: a partial product prunes only because the final one is a multiple of it.

Scope: this module knows integers and products only. Categorical legality (an operand dtype, a
transport's eligibility, a repack rule) and anything that reads the term is the scheduler's, exactly
as it is for the curated grids.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from emmy.compiler.ir.atom import ATOM_REGISTRY

from . import Reduce, Stage, Tile


@dataclass(frozen=True)
class Dimension:
    """One integer dimension: a name and its finite candidate values.

    Values must be ``≥ 1``. That is not a formality — the prefix pruning relies on a product being
    monotone in every factor, and a zero or negative value breaks it.
    """

    name: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"dimension {self.name!r} declares no values")
        bad = [v for v in self.values if v < 1]
        if bad:
            raise ValueError(f"dimension {self.name!r} declares non-positive value(s) {bad}; products must stay monotone to prune")


@dataclass(frozen=True)
class Bound:
    """A multiplicative budget over the dimensions: ``coeff · ∏ dims ≤ limit`` — a thread,
    fragment or box budget, the only comparison a live domain states (see the module docstring for
    the two that were spelled here and had no caller).

    A dimension may repeat in ``dims`` and then contributes its value once per occurrence.
    ``coeff`` folds in the constants the bound multiplies by (an atom's ``atom_n``, a warp's 32
    lanes), so the dimensions stay the only unknowns.
    """

    dims: tuple[str, ...]
    limit: int
    coeff: int = 1

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError(f"bound <= {self.limit} names no dimension")
        if self.limit < 1 or self.coeff < 1:
            raise ValueError(f"bound {self.spell()} needs a positive limit and coeff")

    def spell(self) -> str:
        """The bound as text — for error messages, not a stored codec."""
        lhs = "*".join(self.dims) if self.coeff == 1 else f"{self.coeff}*{'*'.join(self.dims)}"
        return f"{lhs} <= {self.limit}"

    def holds(self, product: int) -> bool:
        """Whether ``product`` — ``coeff`` times the dims bound SO FAR — can still satisfy this
        bound. A budget needs to know nothing else: over-budget stays over-budget, since every
        value is ``≥ 1`` and the final product is a multiple of any partial one. That is the whole
        pruning contract, and a comparison that does NOT have it (an equality, testable at a
        partial product only through divisibility) would have to say so here."""
        return product <= self.limit


@dataclass(frozen=True)
class Space:
    """A bounded integer set: the cartesian product of ``dims``, narrowed by ``bounds``.

    Iterating yields each legal point as ``{dimension name: value}`` in declaration order — the
    dimensions vary in declaration order too, last one fastest, so the first declared dimension's
    first value leads. Per-family option-0 ordering is therefore a property of how the caller
    declares the space, the same contract the curated grids carry.
    """

    dims: tuple[Dimension, ...]
    bounds: tuple[Bound, ...] = ()

    def __post_init__(self) -> None:
        names = [d.name for d in self.dims]
        if not names:
            raise ValueError("a space declares no dimensions")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dimension name among {names}")
        for b in self.bounds:
            unknown = sorted({d for d in b.dims if d not in set(names)})
            if unknown:
                raise ValueError(f"bound {b.spell()} names undeclared dimension(s) {unknown}")

    def contains(self, point: Mapping[str, int]) -> bool:
        """Whether ``point`` is one of the legal points: every dimension at a declared value and
        every bound satisfied — the membership question a parsed value asks of the space that
        :meth:`__iter__` walks."""
        if set(point) != {dim.name for dim in self.dims} or any(point[dim.name] not in dim.values for dim in self.dims):
            return False
        for bound in self.bounds:
            product = bound.coeff
            for name in bound.dims:
                product *= point[name]
            if not bound.holds(product):
                return False
        return True

    def __iter__(self) -> Iterator[dict[str, int]]:
        # Per bound: how many times each dimension occurs in it.
        reps = [tuple(b.dims.count(d.name) for d in self.dims) for b in self.bounds]

        def walk(i: int, point: dict[str, int], products: tuple[int, ...]) -> Iterator[dict[str, int]]:
            if i == len(self.dims):
                yield dict(point)
                return
            dim = self.dims[i]
            for v in dim.values:
                running = list(products)
                for bi, bound in enumerate(self.bounds):
                    if not reps[bi][i]:
                        continue
                    running[bi] *= v ** reps[bi][i]
                    if not bound.holds(running[bi]):
                        break
                else:
                    yield from walk(i + 1, {**point, dim.name: v}, tuple(running))

        yield from walk(0, {}, tuple(b.coeff for b in self.bounds))


MAX_BLOCK_THREADS = 1024
WARP_LANES = 32
MAX_FRAGMENT_CELLS = 32
MAX_FRAGMENT_REGISTERS = 128
MAX_REGISTERS_PER_THREAD = 255
MAX_REGISTERS_PER_CTA = 64 * 1024

_SCALAR_REGISTER_SPACE = Space(
    dims=(
        Dimension("reg_n", (1, 2, 3, 4)),
        Dimension("reg_m", (1, 2, 4)),
    )
)

_SCALAR_1D_TILE_SPACE = Space(dims=(Dimension("par_n", (32, 64, 128, 256, 512)),))

_SCALAR_PARALLEL_TILE_SPACE = Space(
    dims=(
        Dimension("par_n", (16, 32, 64)),
        Dimension("par_m", (8, 16)),
        Dimension("reg_n", (1, 2, 4, 26)),
        Dimension("reg_m", (1, 2, 4, 6, 8, 10, 12, 14, 26)),
    ),
    bounds=(Bound(("par_n", "par_m"), limit=MAX_BLOCK_THREADS),),
)


def scalar_tile_moves() -> list[Tile]:
    """Return the finite scalar-contraction tile domain."""
    moves = [Tile()]
    moves.extend(Tile(units=(1, point["par_n"])) for point in _SCALAR_1D_TILE_SPACE)
    moves.extend(
        Tile(regs=(point["reg_m"], point["reg_n"])) for point in _SCALAR_REGISTER_SPACE if (point["reg_m"], point["reg_n"]) != (1, 1)
    )
    moves.extend(
        Tile(
            units=(point["par_m"], point["par_n"]),
            regs=(point["reg_m"], point["reg_n"]),
        )
        for point in _SCALAR_PARALLEL_TILE_SPACE
    )
    return moves


# ``fn`` runs past ``fm``'s widest point because a register row is what covers an output axis a
# single warp column has to span whole: attention's expectation tiles the value's head dim on N, and
# at head_dim 256 the score's seam allows exactly one warp column there, so the 32 atoms are the only
# spelling of it. Both are still bounded by ``MAX_FRAGMENT_CELLS`` and, per atom, by
# ``MAX_FRAGMENT_REGISTERS`` below — the wide points are grid gaps under those limits, not a raise.
_WARP_TILE_SPACE = Space(
    dims=(
        Dimension("wm", (1, 2, 4, 8, 16)),
        Dimension("wn", (1, 2, 4, 8, 16)),
        Dimension("fm", (1, 2, 4, 8)),
        Dimension("fn", (1, 2, 4, 8, 16, 32)),
        Dimension("bk", (1, 2, 4, 8)),
    ),
    bounds=(
        Bound(("wm", "wn"), limit=MAX_BLOCK_THREADS, coeff=WARP_LANES),
        Bound(("fm", "fn"), limit=MAX_FRAGMENT_CELLS),
    ),
)


def warp_tile_moves(atom_names: tuple[str, ...]) -> list[Tile]:
    """Return the finite warp tile domain for the supplied atom families."""
    moves = []
    for name in atom_names:
        atom = ATOM_REGISTRY[name]
        moves.extend(
            Tile(
                atom=atom,
                units=(point["wm"], point["wn"]),
                regs=(point["fm"], point["fn"]),
                bk=point["bk"],
            )
            for point in _WARP_TILE_SPACE
            if point["fm"] * point["fn"] * atom.accumulator_registers_per_lane <= MAX_FRAGMENT_REGISTERS
        )
    return moves


def warp_tile_in_catalog(plan: Tile) -> bool:
    """Whether a parsed warp plan is a point of the warp tile domain — the same grid and budgets
    :func:`warp_tile_moves` enumerates, asked of one value instead of walked."""
    point = {"wm": plan.units[0], "wn": plan.units[1], "fm": plan.regs[0], "fn": plan.regs[1], "bk": plan.bk}
    return (
        _WARP_TILE_SPACE.contains(point)
        and plan.regs[0] * plan.regs[1] * plan.atom.accumulator_registers_per_lane <= MAX_FRAGMENT_REGISTERS
    )


#: The staging pipeline's parametrizations. ``transport`` is how gmem bytes reach the slab,
#: ``depth`` how many chunks that hop keeps in flight, ``reg_depth`` the smem→register
#: double-buffer beneath it. Independent knobs over one pipeline, so the domain is their PRODUCT.
STAGE_TRANSPORTS = ("smem", "smem-async", "smem-tma")
STAGE_DEPTHS = (1, 2, 3, 4)
STAGE_REG_DEPTHS = (1, 2)


def stage_moves(*, warp: bool, ctx=None) -> list[Stage]:
    """Return the finite staging domain — every combination the hardware allows.

    Nothing here picks which pairings are worth trying: :meth:`Stage.available_on` drops the
    transports the card cannot issue, the resolvers cap what a shape cannot size (the smem budget,
    and a split blocking copy's one-chunk register ring), and measured evidence ranks what survives.
    ``reg_depth >= 2`` is the fragment ping-pong under the mma drain, so it is warp-tier only."""
    reg_depths = STAGE_REG_DEPTHS if warp else (1,)
    moves = [
        Stage(depth=depth, transport=transport, reg_depth=reg_depth)
        for transport in STAGE_TRANSPORTS
        for depth in STAGE_DEPTHS
        for reg_depth in reg_depths
    ]
    return moves if ctx is None else [move for move in moves if move.available_on(ctx)]


def raster_moves() -> tuple[str, ...]:
    """Return the finite kernel raster domain."""
    return "", "gm8", "gn4", "gn8"


def producer_band_moves() -> tuple[int, ...]:
    """Return the finite producer-band domain, including uniform execution."""
    return 0, 1, 2


SPLITK_WIDTHS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


def splitk_moves() -> list[Reduce]:
    """Return cross-CTA split choices for both supported finalization modes."""
    return [Reduce.of(cta=width, finalize=finalize) for width in SPLITK_WIDTHS for finalize in ("kernel", "atomic")]


def coop_reduce_moves() -> list[Reduce]:
    """Return the finite cooperative and register reduction domain."""
    return [
        *(Reduce.of(coop=coop, reg=reg) for coop in (1, 4, 8, 16, 32, 64, 128, 256, 512) for reg in (1, 2, 4) if coop > 1 or reg > 1),
        *(Reduce.of(coop=width, coop_transposed=True) for width in (32, 64, 128, 256)),
    ]


__all__ = [
    "Bound",
    "Dimension",
    "MAX_BLOCK_THREADS",
    "MAX_FRAGMENT_CELLS",
    "MAX_FRAGMENT_REGISTERS",
    "MAX_REGISTERS_PER_CTA",
    "MAX_REGISTERS_PER_THREAD",
    "SPLITK_WIDTHS",
    "Space",
    "WARP_LANES",
    "coop_reduce_moves",
    "producer_band_moves",
    "raster_moves",
    "scalar_tile_moves",
    "splitk_moves",
    "stage_moves",
    "warp_tile_in_catalog",
    "warp_tile_moves",
]
