"""Schedule a lifted (UNMAPPED) ``TileOp``: map its free axes onto the grid and offer the
scheduling fork — the second half of the Loop-IR → Tile-IR boundary.

``010_lift`` is purely structural: it reads the algebra off a ``LoopOp`` and emits an UNMAPPED
:class:`~emmy.compiler.ir.tile.ir.TileOp` (its ``op`` set, ``place`` carrying just the free axes).
THIS rule picks that up and decides the schedule — the classic per-node choices, or register
storage across an ordered matrix loop. Both families use the generic schedule-fork adapter.

The fixed candidate-space contract is Algorithm 1(p, t, row): the problem and target, factored into sites, offer the
candidates — the row's value where the row names a site, the site's catalog where it does not — and one immutable
context composes them. The generic traversal never unpacks that context. Its composition may reject a prefix only
when the combined state proves there is no completion, and traversal order cannot change membership.

Splitting the two halves is what makes the fork ONE thing: a kernel reaches scheduling by
several routes — the ordinary lift and a cross-CTA split's partial and finalize — and all converge here. The engine restarts its
rule scan after every functional rewrite, so a ``TileOp`` this pass's ``010`` just emitted is
matched here on the next sweep, and so is every unmapped ``TileOp`` a structural rewrite minted.
That is exactly why none of them needs a special case: each arrives as a kernel with no schedule,
like any other, and this rule cannot tell them apart.

An unpinned enumeration is one lazy root; only a pool SAMPLE can come back empty, and an empty enumeration
remains a skip rather than a guessed schedule.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.schedule.classic import ClassicProblem, ClassicScheduleCodec, ClassicScheduleContext, materialize_classic
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import carries_partition, merges_partition
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import SCHEDULE_FORK_STAMPS, Fork, iter_leaves

# NOTE: no ``Knob`` objects (``TILE`` / ``REDUCE`` / ``STAGE``) may be imported here — ``Pass.load``
# scans rule modules for ``Knob`` attrs and OFF-fills any it finds bare onto every variant of the
# pass. Pin reads / knob-key spelling ride the enumerator's helpers instead; the family NAMES below
# are plain strings and a function, which that scan does not see.
from emmy.compiler.pipeline.knob import STRUCT_PREFIX, family_pins, kernel_pin, schedule_pin_fingerprint
from emmy.compiler.pipeline.schedule import fork_schedule
from emmy.compiler.structural import digest

PATTERN = [Pattern("root", TileOp)]

_FAMILIES = ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")


def pin_row(*kernel: str, split_consumed: bool, published: bool = True) -> dict[str, str]:
    """The environment's schedule pins for the kernel known by the names ``kernel`` as one knob row — the source
    every site reads, the same way it reads a golden row. A kernel pin that reaches this kernel is
    its family's bare pin here, in place of the one every kernel reads. A kernel that consumed a
    split (``split_consumed``) reads a ``REDUCE`` pin without the ``g<n>`` half the split already took.
    ``published=False`` leaves out the pins published to every kernel, keeping this kernel's own."""
    row: dict[str, str] = {}
    for family in _FAMILIES:
        pins = dict(family_pins(family)) if published else {}
        if (own := kernel_pin(family, *kernel)) is not None:
            pins[family] = own
        for key, value in pins.items():
            if split_consumed and family == "REDUCE":
                value = "/".join(part for part in value.split("/") if not part.startswith("g"))
            row[key] = value
    return row


def classic_forks(
    tile: TileOp, name: str, knobs: dict, ctx, *, kernel_set: bool = False, node: str = "", published: bool = True
) -> list[Fork]:
    """Adapt semantic enumerations to the lazy search tree, sourcing choices from the pins
    where they name a site. Ordered matrix loops may also offer register storage.

    ``kernel_set`` says the kernel is one piece of a cut kernel set. A hand pin is published to
    every piece at once, so each takes the values it can and keeps its catalog where it cannot —
    the reading a row published across peer kernels takes — instead of refusing a value that names
    a sibling piece; the post-compile pin check still asks that SOME kernel realized the pin. ``node``
    is the kernel's graph node id, which a kernel-scoped pin can name where the tile has no name."""
    from emmy.compiler.ir.schedule.register import RegisterCodec, RegisterContext, RegisterProblem, materialize_register  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, FP8_MMA, precision_pin  # noqa: PLC0415

    row = pin_row(tile.name, node, split_consumed=tile.split_consumed or carries_partition(tile), published=published)
    catalog = () if published else ("catalog",)  # a pool without the published pins is another pool
    register = []
    if tile.register_program is not None and not any(value for key, value in row.items() if key not in ("WORK", "TILE", "STAGE")):
        context = RegisterContext(
            RegisterProblem(
                tile,
                ctx,
                row={key: value for key, value in row.items() if key in ("WORK", "TILE", "STAGE")},
                allow_f16=precision_pin(F16_MMA_F32_ACC) is True,
            )
        )
        register = fork_schedule(
            context,
            codec=RegisterCodec(context),
            inherited_knobs=knobs,
            row_prefix={},
            materialize=lambda schedule, selected: materialize_register(tile, schedule, selected),
            pool_id=digest(
                tile.identity_key(with_io=True), ctx.structural_key(), "register", schedule_pin_fingerprint(tile.name, node), *catalog
            ),
            sample=getattr(ctx, "pool_sample", None),
        )
    if row.get("STAGE") == "d1/reg" and tile.place.serial:
        return register

    # A bare WORK / RASTER / REDUCE pin is published across the kernels a split minted and names the
    # partial (a warp ``WORK``, a ``coop`` band), not its finalize, which folds one partial per split
    # per cell serially: the finalize keeps its own domain instead of refusing every row and falling
    # unmapped. The partial, like every other kernel, keeps every verdict, and the post-compile pin
    # check still asks that SOME kernel realized the pin.
    peer = merges_partition(tile)
    # A kernel pin names this one kernel, so it is a hand pin to honour exactly, not a row published
    # across peers to take where it fits; a split's finalize still reads WORK / RASTER / REDUCE as
    # its partial's.
    named = frozenset(
        family
        for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")
        if kernel_pin(family, tile.name, node) is not None and not (peer and family in ("WORK", "RASTER", "REDUCE"))
    )
    problem = ClassicProblem(
        tile,
        ctx,
        row=row,
        allow_f16_accumulate=precision_pin(F16_MMA_F32_ACC) is True,
        allow_fp8=precision_pin(FP8_MMA) is True,
        validate_pins=ctx.validate_pins and not kernel_set,
        tolerate_kernel_pins=peer,
        _strict_row_keys=named,
    )
    context = ClassicScheduleContext(tile, ctx, problem)
    codec = ClassicScheduleCodec(context)
    pool_id = digest(
        tile.identity_key(with_io=True) or "",
        ctx.structural_key(),
        tuple((axis.name, repr(axis.extent)) for axis in tile.place.free),
        codec.keys(),
        schedule_pin_fingerprint(tile.name, node),
        tile.split_consumed,
        *catalog,
    )
    prefix = dict.fromkeys(SCHEDULE_FORK_STAMPS, 1.0) if problem.warp_eligible else {}
    forks = register + fork_schedule(
        context,
        codec=codec,
        inherited_knobs=knobs,
        row_prefix=prefix,
        materialize=lambda schedule, row: materialize_classic(
            tile,
            name=name,
            knobs=row,
            target=ctx,
            schedule=schedule,
        ),
        pool_id=pool_id,
        sample=getattr(ctx, "pool_sample", None),
    )
    if kernel_set and published and any(family_pins(family) for family in _FAMILIES) and next(iter_leaves(forks), None) is None:
        # Pins published to every piece of a cut take where they fit, but values that fit a site one at a time
        # can still leave a piece no complete row (an f32 GDN piece under a GEMM sweep's ``STAGE=d1/smem``,
        # which none of its scalar tiles' loads resolve). That piece keeps its catalog instead of running
        # unscheduled; its own kernel pins still hold.
        return classic_forks(tile, name, knobs, ctx, kernel_set=kernel_set, node=node, published=False)
    return forks


def rewrite(match: Match, root: Node, ctx=None) -> Fork | list[Fork]:
    del match  # the scheduled op replaces the matched node in place — no graph surgery here
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped:
        raise RuleSkipped("TileOp already scheduled / nothing to map")
    # This pass DECIDES, so it requires the kernel's identity. Every row it enumerates carries the
    # ``S_*`` stamp forward, and that is what the prior ranks on, what a recorded golden matches by,
    # and what the measurement is later filed under — decide without it and the fork's pick is made
    # against an empty signature that matches every kernel and identifies none. the ``IdentityStrategy`` stamps at birth
    # ahead of this rule for exactly that reason, so an unstamped kernel here is a pass-order
    # break, not a case to handle.
    assert any(k.startswith(STRUCT_PREFIX) for k in tile.knobs), (
        f"{tile.name!r}: scheduling a kernel with no structural identity — the IdentityStrategy stamps at birth"
    )
    # A cut's pieces carry the seam token in their name or read a workspace named by one.
    kernel_set = "__place_" in tile.name or any("__place_" in buffer for buffer in root.inputs)
    options = classic_forks(tile, tile.name, tile.knobs, ctx, kernel_set=kernel_set, node=root.id)
    # A pin that names THIS kernel and leaves it no row is refused here, with the pins that did it.
    # Left to the lazy fork, the empty enumeration was skipped: the kernel ran unscheduled and the
    # pin looked realized by nothing (a SiLU-prologue down projection under a mma TILE pin).
    scoped = {family: value for family in _FAMILIES if (value := kernel_pin(family, tile.name, root.id)) is not None}
    if scoped and next(iter_leaves(options), None) is None:
        pins = ", ".join(f"{family}={value}" for family, value in scoped.items())
        raise ValueError(f"{tile.name or root.id}: its kernel pins ({pins}) leave no schedule row this kernel offers")
    if not options:
        raise RuleSkipped("no enumerable schedule row for this term — leave it unmapped")
    return options if len(options) > 1 else options[0]
