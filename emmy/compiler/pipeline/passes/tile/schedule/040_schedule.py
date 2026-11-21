"""Schedule a lifted (UNMAPPED) ``TileOp``: map its free axes onto the grid and offer the
scheduling fork — the second half of the Loop-IR → Tile-IR boundary.

``010_lift`` is purely structural: it reads the algebra off a ``LoopOp`` and emits an UNMAPPED
:class:`~emmy.compiler.ir.tile.ir.TileOp` (its ``op`` set, ``place`` carrying just the free axes).
THIS rule picks that up and decides the schedule — the classic per-node choices, or register
storage across an ordered matrix loop; on the CPU backend, the CPU code generator's choices. Every family uses the
generic schedule-fork adapter.

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

from emmy.compiler.dim import paged_declarations
from emmy.compiler.graph import Node
from emmy.compiler.ir.schedule.choices import Stage, Tile, Work
from emmy.compiler.ir.schedule.classic import (
    ClassicProblem,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    classic_keys,
    materialize_classic,
)
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import merges_partition
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import Fork, exact_schedule_leaf, iter_leaves

# NOTE: no ``Knob`` objects (``TILE`` / ``REDUCE`` / ``STAGE``) may be imported here — ``Pass.load``
# scans rule modules for ``Knob`` attrs and OFF-fills any it finds bare onto every variant of the
# pass. Pin reads / knob-key spelling ride the enumerator's helpers instead; the family NAMES below
# are plain strings and a function, which that scan does not see.
from emmy.compiler.pipeline.knob import family_pins, kernel_pin, schedule_pin_fingerprint
from emmy.compiler.pipeline.passes.tile._cut import PIECE
from emmy.compiler.pipeline.passes.tile._fromloop import carried_program, serial_form, shared_state, shared_state_bytes
from emmy.compiler.pipeline.schedule import fork_schedule
from emmy.compiler.structural import digest

PATTERN = [Pattern("root", TileOp)]

_FAMILIES = ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")


def pin_row(kernel: str, *, split_consumed: bool, published: bool = True, families: tuple[str, ...] = _FAMILIES) -> dict[str, str]:
    """The environment's schedule pins for the kernel called ``kernel`` as one knob row — the source
    every site reads, the same way it reads a golden row. A kernel pin that names this kernel is
    its family's bare pin here, in place of the one every kernel reads. A kernel that consumed a
    split (``split_consumed``) reads a ``REDUCE`` pin without the ``g<n>`` half the split already took.
    ``published=False`` leaves out the pins published to every kernel, keeping this kernel's own. ``families`` are
    the knob families the reading family owns."""
    row: dict[str, str] = {}
    for family in families:
        pins = dict(family_pins(family)) if published else {}
        if (own := kernel_pin(family, kernel)) is not None:
            pins[family] = own
        for key, value in pins.items():
            if split_consumed and family == "REDUCE":
                value = "/".join(part for part in value.split("/") if not part.startswith("g"))
            row[key] = value
    return row


def cpu_forks(tile: TileOp, knobs: dict, ctx) -> list[Fork]:
    """The CPU family's lazy fork over ``tile``, its knobs read from the pins the way a CUDA family reads its own."""
    from emmy.compiler.ir.schedule.cpu import CpuCodec, CpuContext, CpuProblem, materialize_cpu  # noqa: PLC0415

    row = pin_row(tile.name, split_consumed=False, families=CpuCodec.keys())
    context = CpuContext(CpuProblem(tile, ctx, row=row))
    if row and not context.problem.bound:
        # A pin published to every kernel takes where it fits; a kernel it does not fit keeps its catalog.
        row = {}
        context = CpuContext(CpuProblem(tile, ctx))
    return fork_schedule(
        context,
        codec=CpuCodec(context),
        inherited_knobs=knobs,
        materialize=lambda schedule, row, tile=tile: materialize_cpu(tile, schedule, row),
        pool_id=digest(tile.identity_key(with_io=True), ctx.structural_key(), "cpu", tuple(sorted(row.items()))),
        sample=getattr(ctx, "pool_sample", None),
    )


def _shared_state_fits(tile: TileOp, ctx) -> bool:
    """Whether the one state the serial kernel ``tile`` carries can be kept in shared memory: one CTA owns each
    state whole (:func:`carried_program`), and the two padded buffers the steps alternate between fit beside
    nothing else. What only the lowered kernel shows is the kernel lowering's to decline."""
    program = carried_program(tile)
    state = shared_state(tile, program) if program is not None else None
    if state is None:
        return False
    port, shape = state
    return shared_state_bytes(shape, tile.outputs[port].dtype) <= ctx.max_dynamic_smem


def _resident_tiles(tile: TileOp, ctx) -> tuple[Tile, ...]:
    """The fragment geometries registers may hold ``tile``'s state in on this card: per 16-bit atom whose C
    fragments repack into its A, the geometry the state's columns give. Offered on the state's shape alone
    (:attr:`TileOp.resident_state`); a step the fragments cannot evaluate declines the row where it is lowered."""
    from emmy.compiler.dtype import BF16, F16, F32  # noqa: PLC0415
    from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for  # noqa: PLC0415

    if tile.resident_state is None:
        return ()
    columns = tile.axis_of(tile.resident_state[2]).extent.as_static()
    names = dict.fromkeys(name for dtype in (F16, BF16) for acc in (F32, dtype) for name in atoms_for(dtype, acc=acc, ctx=ctx))
    return tuple(
        Tile.parse(f"{atom.name}/f1x{columns // atom.atom_n}/k{period}", Work.parse("w1x1"))
        for atom in (ATOM_REGISTRY[name] for name in names)
        if atom.c_to_a_repack and columns % atom.atom_n == 0
        for period in (2, 4, 8)
    )


def classic_forks(
    tile: TileOp,
    knobs: dict,
    ctx,
    *,
    kernel_set: bool = False,
    node: str = "",
    published: bool = True,
    paged: dict | None = None,
) -> list[Fork]:
    """Adapt semantic enumerations to the lazy search tree, sourcing choices from the pins
    where they name a site.

    ``kernel_set`` says the kernel is one piece of a cut kernel set. A hand pin is published to
    every piece at once, so each takes the values it can and keeps its catalog where it cannot —
    the reading a row published across peer kernels takes — instead of refusing a value that names
    a sibling piece; the post-compile pin check still asks that SOME kernel realized the pin. ``node`` is the
    kernel's graph node id, which names the buffer a carried state's serial form keeps it in."""
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, FP8_MMA, TF32_MMA, precision_pin  # noqa: PLC0415

    row = pin_row(tile.name, split_consumed=tile.split_taken, published=published)
    catalog = () if published else ("catalog",)  # a pool without the published pins is another pool
    # A kernel that carries a state has a ``STAGE`` site for it. Direct is the serial form: a buffer in global
    # memory, one launch per step. ``d1/smem`` runs the steps of that form inside one launch; ``d1/reg`` keeps the
    # state in register fragments, and the carrying kernel itself is what the lowering then reads.
    state_stages, state_tiles = (), ()
    if tile.carries:
        state_tiles = _resident_tiles(tile, ctx)
        tile = serial_form(tile, node)
        if _shared_state_fits(tile, ctx):
            state_stages += (Stage(transport="smem"),)
        if state_tiles:
            state_stages += (Stage(transport="reg"),)
    # A block of a partition walks from zero, and each range is stated of the axis as a tier walks it.
    tile = tile.scheduled()
    # A value a kernel keeps lives in shared memory that one cell of the grid owns
    # (``lowering/kernel/012_keep_values``), so the kernel takes the layout with one thread to a cell: every site
    # at its direct choice, whatever a pin published to every kernel says.
    if tile.keeps:
        row = dict.fromkeys(classic_keys(tile), "")

    # A bare WORK / RASTER / REDUCE pin is published across the kernels a split minted and names the
    # partial (a warp ``WORK``, a ``coop`` band). A finalize that cannot take that row keeps its own
    # domain instead of refusing every row and falling unmapped. The partial, like every other
    # kernel, keeps every verdict, and the post-compile pin
    # check still asks that SOME kernel realized the pin.
    peer = merges_partition(tile)
    # A kernel pin names this one kernel, so it is a hand pin to honour exactly, not a row published
    # across peers to take where it fits; a split's finalize still reads WORK / RASTER / REDUCE as
    # its partial's.
    named = frozenset(
        family
        for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")
        if kernel_pin(family, tile.name) is not None and not (peer and family in ("WORK", "RASTER", "REDUCE"))
    )
    problem = ClassicProblem(
        tile,
        ctx,
        row=row,
        allow_f16_accumulate=precision_pin(F16_MMA_F32_ACC) is True,
        allow_fp8=precision_pin(FP8_MMA) is True,
        allow_tf32=precision_pin(TF32_MMA) is True,
        validate_pins=ctx.validate_pins and not kernel_set,
        tolerate_kernel_pins=peer,
        paged=paged or {},
        state_stages=state_stages,
        state_tiles=state_tiles,
        _strict_row_keys=named,
    )
    context = ClassicScheduleContext(tile, ctx, problem)
    codec = ClassicScheduleCodec(context)
    pool_id = digest(
        tile.identity_key(with_io=True) or "",
        ctx.structural_key(),
        tuple((axis.name, repr(axis.extent)) for axis in tile.place.free),
        codec.keys(),
        schedule_pin_fingerprint(tile.name),
        tile.split_consumed,
        *catalog,
    )
    classic = fork_schedule(
        context,
        codec=codec,
        inherited_knobs=knobs,
        materialize=lambda schedule, row: materialize_classic(tile, knobs=row, target=ctx, schedule=schedule),
        pool_id=pool_id,
        sample=getattr(ctx, "pool_sample", None),
    )
    if set(codec.keys()) <= set(row) and getattr(ctx, "pool_sample", None) is None:
        # A complete hand pin has one candidate; a refused pin keeps the usual peer fallback.
        exact = exact_schedule_leaf(classic, row, frozenset(codec.keys()))
        if exact is not None and exact[1] is not None:
            classic = [exact[1]]
    forks = classic
    if kernel_set and published and any(family_pins(family) for family in _FAMILIES) and next(iter_leaves(forks), None) is None:
        # Pins published to every piece of a cut take where they fit, but values that fit a site one at a time
        # can still leave a piece no complete row (an f32 GDN piece under a GEMM sweep's ``STAGE=d1/smem``,
        # which none of its scalar tiles' loads resolve). That piece keeps its catalog instead of running
        # unscheduled; its own kernel pins still hold.
        return classic_forks(tile, knobs, ctx, kernel_set=kernel_set, node=node, published=False, paged=paged)
    return forks


def rewrite(match: Match, root: Node, ctx=None) -> Fork | list[Fork]:
    # The scheduled op replaces the matched node in place — no graph surgery here; the graph is read for
    # its paged buffers, whose operands the stage catalog keeps TMA off (``ClassicProblem.paged``).
    declared = paged_declarations(match.graph.hints.get("cuda.paged_buffers", ()), match.graph)
    paged = {n: (axis, page) for n, (axis, page, _) in declared.items()}
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped:
        raise RuleSkipped("TileOp already scheduled / nothing to map")
    kernel_set = root.hints.has(PIECE)  # a kernel a cut left
    if getattr(ctx, "backend_name", "cuda") == "cpu":
        options = cpu_forks(tile, tile.knobs, ctx)
    else:
        options = classic_forks(tile, tile.knobs, ctx, kernel_set=kernel_set, node=root.id, paged=paged)
    # A pin that names THIS kernel and leaves it no row is refused here, with the pins that did it.
    # Left to the lazy fork, the empty enumeration was skipped: the kernel ran unscheduled and the
    # pin looked realized by nothing (a SiLU-prologue down projection under a mma TILE pin).
    scoped = {family: value for family in _FAMILIES if (value := kernel_pin(family, tile.name)) is not None}
    if scoped and next(iter_leaves(options), None) is None:
        pins = ", ".join(f"{family}={value}" for family, value in scoped.items())
        raise ValueError(f"{tile.name}: its kernel pins ({pins}) leave no schedule row this kernel offers")
    if not options:
        raise RuleSkipped("no enumerable schedule row for this term — leave it unmapped")
    return options if len(options) > 1 else options[0]
